"""
Pure adapters between the ``reviewscope_ml`` integration seam and the backend's
ORM — deliberately free of Celery, FastAPI and the heavy ML deps so the mapping
logic is unit-testable in isolation (see ``tests/test_ml_integration.py``).

Three concerns live here:

* :func:`derive_roles` / :func:`build_upload_schema` — turn the stored
  ``project_schema.columns`` into the seam's :class:`UploadSchema` (which needs a
  designated text column and an optional rating column the column table implies).
* :class:`DbProgressSink` — the seam's ``ProgressSink`` port, writing
  ``pipeline_jobs`` rows so ``GET /pipeline/status`` advances mid-run.
* :func:`result_to_orm` / :func:`persist_run_result` — map a finished
  ``RunResult`` (plain dataclasses) onto ``Cluster`` / ``Document`` / ``Embedding``
  rows, resolving the pipeline's integer cluster ids to row UUIDs.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Callable

from sqlalchemy import delete, select, update

from app.models import Cluster, Document, Embedding, PipelineJob, PipelineStepStatus, Project, Segment


# Canonical lowercase step order, matching the rows pre-created in
# ``create_project`` and the 1-based ``index`` the seam reports. Kept local so
# importing this module never drags in Celery.
STEP_ORDER: tuple[str, ...] = (
    "ingest", "preprocess", "embed", "reduce",
    "cluster", "sentiment", "label", "finalize",
)

_TEXT_NAME_HINTS = ("text", "review", "content", "comment", "body", "description")
_RATING_NAME_HINTS = ("stars", "star", "rating", "rate", "score")
_NUMERIC_TYPES = ("integer", "float")


# ── Schema derivation ─────────────────────────────────────────────────────────

def derive_roles(columns: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Pick the ``(text_column, rating_column)`` the seam needs from the columns.

    text:   a non-primary-key ``type == "text"`` column, preferring conventional
            names (a PK is identity, not content, so it is excluded unless it is
            the only text column); falls back to the first text column, or the
            first column of any type (the seam then rejects it with a clear
            SchemaError → failed status).
    rating: a numeric column with a rating-like name, else ``None``.
    """
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


def build_upload_schema(columns: list[dict[str, Any]]):
    """Build the seam's ``UploadSchema`` from stored ``project_schema.columns``.

    Imports ``reviewscope_ml`` lazily so this module (and the tests that import
    it) stay free of the heavy ML dependency tree.
    """
    from reviewscope_ml.app import ColumnSpec, UploadSchema

    text_column, rating_column = derive_roles(columns)
    specs = [
        ColumnSpec(c["name"], c.get("type", "text"), is_primary_key=bool(c.get("is_primary_key")))
        for c in columns
    ]
    schema = UploadSchema(columns=specs, text_column=text_column, rating_column=rating_column)
    schema.validate()
    return schema


# ── Progress sink (ProgressSink port) ─────────────────────────────────────────

class DbProgressSink:
    """Writes each pipeline step transition to ``pipeline_jobs``.

    The seam emits Capitalized canonical names (``Ingest``…``Finalize``) and only
    an explicit ``done`` for ``Finalize``; this sink lowercases to match the
    pre-created rows and, on each ``running`` event, marks every earlier step
    ``done`` so the dashboard progress bar fills monotonically.
    """

    def __init__(self, session_factory: Callable[[], Any], project_id: uuid.UUID) -> None:
        self._session_factory = session_factory
        self._project_id = project_id

    def step(self, name: str, status: str, message: str = "", index: int = 0, total: int = 8) -> None:
        key = STEP_ORDER[index - 1] if 1 <= index <= len(STEP_ORDER) else name.lower()
        now = datetime.now(UTC)
        with self._session_factory() as session:
            if status == "running":
                # Everything before this step is implicitly finished.
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


# ── Result → ORM ──────────────────────────────────────────────────────────────

def result_to_orm(result) -> tuple[list[Cluster], list[Document], list[Embedding], list[Segment]]:
    """Map a ``RunResult`` to (clusters, documents, embeddings, segments) ORM rows.

    UUIDs are assigned eagerly (not left to SQLAlchemy defaults) so the integer
    ``cluster_id`` → cluster-UUID map and the ``primary_key_value`` → document-UUID
    join can be built before any flush. Noise members keep a ``NULL`` FK.

    Document-unit runs populate ``embeddings`` (one per review) and leave
    ``segments`` empty; sentence-unit runs do the reverse — ``segments`` are the
    clustered mentions and each ``Document`` carries the review's derived primary
    cluster. Exactly one of the two collections is non-empty.
    """
    project_id = uuid.UUID(str(result.project_id))
    unit = getattr(result, "unit", "document")

    # Cohesion: mean cosine similarity of member vectors to their centroid, seeded
    # here from the clustered unit's vectors (embeddings for document runs, the
    # segments themselves for sentence runs) so `cohesion` matches later recomputes.
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
            continue  # embedding without a surviving document — skip defensively
        embeddings.append(Embedding(
            document_id=did,
            vector=list(rec.vector),
            umap_x=rec.umap_x,
            umap_y=rec.umap_y,
            umap_z=rec.umap_z,
        ))

    segments: list[Segment] = []
    for rec in getattr(result, "segments", None) or []:
        did = doc_uuid_by_pk.get(rec.parent_key)
        if did is None:
            continue  # segment whose parent review didn't survive — skip defensively
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
            umap_x=rec.umap_x,
            umap_y=rec.umap_y,
            umap_z=rec.umap_z,
        ))

    return clusters, documents, embeddings, segments


def persist_run_result(session, result) -> int:
    """Replace a project's analysis rows with a finished run. Returns doc count.

    The caller (the Celery task) owns the surrounding transaction/commit. Wipes in
    FK-safe order (segments + embeddings → documents → clusters) and re-inserts,
    then stamps the project's ``unit`` so the read path knows which shape to serve.
    """
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

    session.execute(
        update(Project).where(Project.id == project_id).values(doc_count=len(documents), unit=unit)
    )
    return len(documents)
