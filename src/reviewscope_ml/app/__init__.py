"""
Application integration layer — the seam the FastAPI/Celery backend imports.

The backend's responsibilities (auth, DB, web, Celery, frontend) stay in the
backend; this layer is framework-agnostic Python that turns an uploaded file
into DB-ready records by driving the existing ML pipeline.

Typical backend usage (inside a Celery task)::

    from reviewscope_ml.app import run_from_upload, UploadSchema, ColumnSpec

    result = run_from_upload(
        file_path=path,
        schema=UploadSchema(
            columns=[ColumnSpec("review_id", "text", is_primary_key=True),
                     ColumnSpec("text", "text"),
                     ColumnSpec("stars", "integer")],
            text_column="text",
            rating_column="stars",
        ),
        project_id=str(project.id),
        progress=DbProgressSink(job),     # backend-implemented ProgressSink
        device="cuda",
    )
    repository.save(result)               # backend-implemented ResultRepository

See ``docs/integration-guide.md`` for the full contract.
"""
from __future__ import annotations

from .defaults import APP_DEFAULT_VARIANT, app_default_spec
from .dto import ClusterRecord, DocumentRecord, EmbeddingRecord, RunResult, SegmentRecord
from .ingest_upload import IngestError, UploadedCorpus, reviewset_from_upload
from .persistence import to_records
from .ports import (
    PIPELINE_STEPS,
    STAGE_TO_STEP,
    TOTAL_STEPS,
    NullProgress,
    ProgressSink,
    ResultRepository,
)
from .schema import ColumnSpec, ColumnType, SchemaError, UploadSchema
from .service import (
    project_config,
    project_corpus_token,
    run_from_upload,
    run_project_pipeline,
)

__all__ = [
    # entry points
    "run_from_upload", "run_project_pipeline",
    # input
    "UploadSchema", "ColumnSpec", "ColumnType", "SchemaError",
    "reviewset_from_upload", "UploadedCorpus", "IngestError",
    # output
    "RunResult", "DocumentRecord", "EmbeddingRecord", "SegmentRecord", "ClusterRecord", "to_records",
    # ports
    "ProgressSink", "ResultRepository", "NullProgress",
    "PIPELINE_STEPS", "STAGE_TO_STEP", "TOTAL_STEPS",
    # config / defaults
    "app_default_spec", "APP_DEFAULT_VARIANT",
    "project_config", "project_corpus_token",
]
