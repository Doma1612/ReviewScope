from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Callable

from sqlalchemy import delete, select, update

from app.models import Cluster, Document, Embedding, PipelineJob, PipelineStepStatus, Project, Segment


STEP_ORDER: tuple[str, ...] = (
    "ingest", "preprocess", "embed", "reduce",
    "cluster", "sentiment", "label", "finalize",
)

_TEXT_NAME_HINTS = ("text", "review", "content", "comment", "body", "description")
_RATING_NAME_HINTS = ("stars", "star", "rating", "rate", "score")
_NUMERIC_TYPES = ("integer", "float")


def derive_roles(columns: list[dict[str, Any]]) -> tuple[str, str | None]:
    text_cols = [
        c for c in columns
        if c.get("type") == "text" and not c.get("is_primary_key")
    ] or [c for c in columns if c.get("type") == "text"]
    text_column = (
        _first_named(text_cols, _TEXT_NAME_HINTS)
        or (text_cols[0]["name"] if text_cols else (columns[0]["name"] if columns else ""))
    )

    numeric_cols = [c for c in columns if c.get("type") in _NUMERIC_TYPES]
    rating_col = _first_named(numeric_cols, _RATING_NAME_HINTS)
    return text_column, rating_col


def _first_named(cols: list[dict[str, Any]], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        for col in cols:
            if hint in str(col.get("name", "")).lower():
                return col["name"]
    return None


def corpus_profile_for(source_filename: str | None) -> str | None:
    if not source_filename:
        return None
    from reviewscope_ml.core.corpus import PROFILES

    for slug in sorted((s for s in PROFILES if s), key=len, reverse=True):
        if slug in source_filename:
            return slug
    return None


def build_upload_schema(columns: list[dict[str, Any]]):
    from reviewscope_ml.app import ColumnSpec, UploadSchema

    text_column, rating_column = derive_roles(columns)
    specs = [
        ColumnSpec(c["name"], c.get("type", "text"), is_primary_key=bool(c.get("is_primary_key")))
        for c in columns
    ]
    schema = UploadSchema(columns=specs, text_column=text_column, rating_column=rating_column)
    schema.validate()
    return schema


class DbProgressSink:

    def __init__(self, session_factory: Callable[[], Any], project_id: uuid.UUID) -> None:
        self._session_factory = session_factory
        self._project_id = project_id

    def step(self, name: str, status: str, message: str = "", index: int = 0, total: int = 8) -> None:
        key = STEP_ORDER[index - 1] if 1 <= index <= len(STEP_ORDER) else name.lower()
        now = datetime.now(UTC)
        with self._session_factory() as session:
            if status == "running":
                earlier = [s for s in STEP_ORDER[: max(index - 1, 0)]]
                if earlier:
                    session.execute(
                        update(PipelineJob)
                        .where(
                            PipelineJob.project_id == self._project_id,
                            PipelineJob.step.in_(earlier),
                            PipelineJob.status != PipelineStepStatus.done,
                        )
                        .values(status=PipelineStepStatus.done, finished_at=now)
                    )
                self._set(session, key, PipelineStepStatus.running, message, started_at=now)
            elif status == "done":
                self._set(session, key, PipelineStepStatus.done, message, finished_at=now)
            elif status == "failed":
                self._set(session, key, PipelineStepStatus.failed, message, finished_at=now)
            session.commit()

    def _set(self, session, step: str, status: PipelineStepStatus, message: str, **extra) -> None:
        values: dict[str, Any] = {"status": status, "message": message or None, **extra}
        session.execute(
            update(PipelineJob)
            .where(PipelineJob.project_id == self._project_id, PipelineJob.step == step)
            .values(**values)
        )


def result_to_orm(result) -> tuple[list[Cluster], list[Document], list[Embedding], list[Segment]]:
    # Assign UUIDs before flush so cluster and document joins can be built eagerly.
    project_id = uuid.UUID(str(result.project_id))
    unit = getattr(result, "unit", "document")

    from app.services.metrics import cohesion_score

    vectors_by_cluster_int: dict[int, list] = {}
    if unit == "sentence":
        for rec in result.segments:
            if rec.cluster_id is not None and rec.vector:
                vectors_by_cluster_int.setdefault(rec.cluster_id, []).append(list(rec.vector))
    else:
        vector_by_pk = {rec.primary_key_value: list(rec.vector) for rec in result.embeddings}
        for rec in result.documents:
            if rec.cluster_id is None:
                continue
            vector = vector_by_pk.get(rec.primary_key_value)
            if vector:
                vectors_by_cluster_int.setdefault(rec.cluster_id, []).append(vector)

    clusters: list[Cluster] = []
    cluster_uuid_by_int: dict[int, uuid.UUID] = {}
    for rec in result.clusters:
        cid = uuid.uuid4()
        cluster_uuid_by_int[rec.cluster_id] = cid
        clusters.append(Cluster(
            id=cid,
            project_id=project_id,
            label=rec.label,
            summary=rec.summary,
            label_source=rec.label_source,
            top_terms=list(rec.top_terms),
            word_frequencies=dict(rec.word_frequencies),
            size=rec.size,
            n_mentions=getattr(rec, "n_mentions", rec.size),
            sentiment_avg=rec.sentiment_avg,
            mean_stars=rec.mean_stars,
            cohesion=cohesion_score(vectors_by_cluster_int.get(rec.cluster_id, [])),
        ))

    def to_uuid(int_cid) -> uuid.UUID | None:
        return cluster_uuid_by_int.get(int_cid) if int_cid is not None else None

    documents: list[Document] = []
    doc_uuid_by_pk: dict[str, uuid.UUID] = {}
    for rec in result.documents:
        did = uuid.uuid4()
        doc_uuid_by_pk[rec.primary_key_value] = did
        documents.append(Document(
            id=did,
            project_id=project_id,
            primary_key_value=rec.primary_key_value,
            text=rec.text,
            raw_data=dict(rec.raw_data),
            cluster_id=to_uuid(rec.cluster_id),
            sentiment_score=rec.sentiment_score,
        ))

    embeddings: list[Embedding] = []
    for rec in result.embeddings:
        did = doc_uuid_by_pk.get(rec.primary_key_value)
        if did is None:
            continue
        embeddings.append(Embedding(
            document_id=did,
            vector=list(rec.vector),
            vector_pg=list(rec.vector) if rec.vector else None,
            umap_x=rec.umap_x,
            umap_y=rec.umap_y,
            umap_z=rec.umap_z,
        ))

    segments: list[Segment] = []
    for rec in getattr(result, "segments", None) or []:
        did = doc_uuid_by_pk.get(rec.parent_key)
        if did is None:
            continue
        segments.append(Segment(
            id=uuid.uuid4(),
            project_id=project_id,
            document_id=did,
            segment_key=rec.segment_key,
            ordinal=rec.ordinal,
            text=rec.text,
            cluster_id=to_uuid(rec.cluster_id),
            sentiment_score=rec.sentiment_score,
            vector=list(rec.vector),
            vector_pg=list(rec.vector) if rec.vector else None,
            umap_x=rec.umap_x,
            umap_y=rec.umap_y,
            umap_z=rec.umap_z,
        ))

    return clusters, documents, embeddings, segments


def persist_run_result(session, result) -> int:
    # The caller owns the transaction and commit.
    project_id = uuid.UUID(str(result.project_id))
    unit = getattr(result, "unit", "document")
    session.execute(delete(Segment).where(Segment.project_id == project_id))
    session.execute(
        delete(Embedding).where(
            Embedding.document_id.in_(select(Document.id).where(Document.project_id == project_id))
        )
    )
    session.execute(delete(Document).where(Document.project_id == project_id))
    session.execute(delete(Cluster).where(Cluster.project_id == project_id))
    session.flush()

    clusters, documents, embeddings, segments = result_to_orm(result)
    session.add_all(clusters)
    session.flush()
    session.add_all(documents)
    session.flush()
    session.add_all(embeddings)
    session.add_all(segments)
    session.flush()

    spec = (getattr(result, "manifest", {}) or {}).get("spec", {})
    vectors = segments if unit == "sentence" else embeddings
    dimension = len(vectors[0].vector) if vectors else None
    # Retrieval requires complete, same-width vectors and the recorded query model.
    ready = bool(
        dimension
        and spec.get("embedding_model")
        and all(
            row.vector_pg is not None and len(row.vector_pg) == dimension
            for row in vectors
        )
    )
    session.execute(
        update(Project).where(Project.id == project_id).values(
            doc_count=len(documents),
            unit=unit,
            embedding_model=spec.get("embedding_model"),
            embedding_instruction=spec.get("instruction", "no_inst"),
            embedding_dimension=dimension,
            retrieval_ready=ready,
        )
    )
    return len(documents)
