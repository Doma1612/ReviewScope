import asyncio
import csv
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, update

from app.db.session import AsyncSessionLocal
from app.models import Cluster, Document, Embedding, PipelineJob, PipelineStepStatus, Project, ProjectStatus
from app.worker import celery_app


PIPELINE_STEPS = ["ingest", "preprocess", "embed", "reduce", "cluster", "sentiment", "label", "finalize"]


@celery_app.task(name="app.tasks.run_simulated_pipeline")
def run_simulated_pipeline(project_id: str) -> None:
    asyncio.run(_run_pipeline(uuid.UUID(project_id)))


async def _run_pipeline(project_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as db:
        project = await db.get(Project, project_id)
        if not project or not project.upload_path:
            return
        try:
            await db.execute(update(Project).where(Project.id == project_id).values(status=ProjectStatus.processing, last_error=None))
            await _ensure_jobs(db, project_id)
            await db.commit()

            rows = []
            text_column = None
            primary_key = None
            for step in PIPELINE_STEPS:
                await _mark_job(db, project_id, step, PipelineStepStatus.running, f"Running {step}")
                await asyncio.sleep(0.4)
                if step == "ingest":
                    rows = _read_rows(Path(project.upload_path))
                    text_column = _pick_text_column(rows)
                    primary_key = _pick_primary_key(rows)
                    if not rows or not text_column:
                        raise ValueError("Uploaded file must include at least one text-like column")
                if step == "finalize":
                    await _persist_mock_results(db, project_id, rows, text_column or "text", primary_key)
                await _mark_job(db, project_id, step, PipelineStepStatus.done, f"Completed {step}")

            await db.execute(update(Project).where(Project.id == project_id).values(status=ProjectStatus.ready, last_error=None))
            await db.commit()
        except Exception as exc:
            await db.rollback()
            await db.execute(update(Project).where(Project.id == project_id).values(status=ProjectStatus.failed, last_error=str(exc)))
            await _mark_running_jobs_failed(db, project_id, str(exc))
            await db.commit()


async def _ensure_jobs(db, project_id: uuid.UUID) -> None:
    result = await db.execute(select(PipelineJob.step).where(PipelineJob.project_id == project_id))
    existing = set(result.scalars().all())
    for step in PIPELINE_STEPS:
        if step not in existing:
            db.add(PipelineJob(project_id=project_id, step=step, status=PipelineStepStatus.pending))


async def _mark_job(db, project_id: uuid.UUID, step: str, status: PipelineStepStatus, message: str) -> None:
    values = {"status": status, "message": message}
    if status == PipelineStepStatus.running:
        values["started_at"] = datetime.now(UTC)
    if status in {PipelineStepStatus.done, PipelineStepStatus.failed}:
        values["finished_at"] = datetime.now(UTC)
    await db.execute(update(PipelineJob).where(PipelineJob.project_id == project_id, PipelineJob.step == step).values(**values))
    await db.commit()


async def _mark_running_jobs_failed(db, project_id: uuid.UUID, message: str) -> None:
    await db.execute(
        update(PipelineJob)
        .where(PipelineJob.project_id == project_id, PipelineJob.status == PipelineStepStatus.running)
        .values(status=PipelineStepStatus.failed, message=message, finished_at=datetime.now(UTC))
    )


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        text = path.read_text(errors="replace")
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        if rows:
            return rows
        value = json.loads(text)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for candidate in value.values():
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]
            return [value]
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _pick_text_column(rows: list[dict]) -> str | None:
    if not rows:
        return None
    names = list(rows[0].keys())
    for preferred in ("text", "review", "content", "comment", "body", "description"):
        for name in names:
            if preferred in name.lower():
                return name
    return max(names, key=lambda name: len(str(rows[0].get(name, ""))))


def _pick_primary_key(rows: list[dict]) -> str | None:
    if not rows:
        return None
    for preferred in ("id", "review_id", "document_id"):
        for name in rows[0].keys():
            if name.lower() == preferred:
                return name
    return None


async def _persist_mock_results(db, project_id: uuid.UUID, rows: list[dict], text_column: str, primary_key: str | None) -> None:
    await db.execute(delete(Embedding).where(Embedding.document_id.in_(select(Document.id).where(Document.project_id == project_id))))
    await db.execute(delete(Document).where(Document.project_id == project_id))
    await db.execute(delete(Cluster).where(Cluster.project_id == project_id))
    await db.flush()

    limited_rows = rows[:500]
    cluster_count = max(2, min(6, math.ceil(len(limited_rows) / 12))) if limited_rows else 0
    clusters = []
    for index in range(cluster_count):
        cluster = Cluster(
            project_id=project_id,
            label=f"Simulated Theme {index + 1}",
            summary=f"Synthetic summary for theme {index + 1}, generated for frontend and API testing.",
            top_terms=[],
            word_frequencies={},
            size=0,
            sentiment_avg=0,
        )
        db.add(cluster)
        clusters.append(cluster)
    await db.flush()

    grouped_terms: dict[uuid.UUID, Counter] = defaultdict(Counter)
    grouped_sentiment: dict[uuid.UUID, list[float]] = defaultdict(list)
    rng = random.Random(str(project_id))
    for idx, row in enumerate(limited_rows):
        cluster = clusters[idx % cluster_count]
        text = str(row.get(text_column) or "")[:5000]
        sentiment = round(rng.uniform(-0.8, 0.9), 3)
        document = Document(
            project_id=project_id,
            primary_key_value=str(row.get(primary_key) if primary_key else idx + 1),
            text=text,
            raw_data=row,
            cluster_id=cluster.id,
            sentiment_score=sentiment,
        )
        db.add(document)
        await db.flush()
        angle = (idx / max(len(limited_rows), 1)) * math.tau
        radius = 1 + (idx % cluster_count) * 0.45
        db.add(
            Embedding(
                document_id=document.id,
                vector=[round(rng.random(), 4) for _ in range(8)],
                umap_x=round(math.cos(angle) * radius + rng.uniform(-0.15, 0.15), 4),
                umap_y=round(math.sin(angle) * radius + rng.uniform(-0.15, 0.15), 4),
                umap_z=round(rng.uniform(-1, 1), 4),
            )
        )
        grouped_terms[cluster.id].update(_terms(text))
        grouped_sentiment[cluster.id].append(sentiment)

    for cluster in clusters:
        terms = grouped_terms[cluster.id].most_common(20)
        sentiments = grouped_sentiment[cluster.id]
        cluster.size = len(sentiments)
        cluster.sentiment_avg = round(sum(sentiments) / len(sentiments), 3) if sentiments else None
        cluster.top_terms = [{"term": term, "score": count} for term, count in terms[:10]]
        cluster.word_frequencies = dict(terms)

    await db.execute(update(Project).where(Project.id == project_id).values(doc_count=len(limited_rows)))


def _terms(text: str) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your", "but", "not"}
    words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
    return [word for word in words if len(word) > 3 and word not in stop]
