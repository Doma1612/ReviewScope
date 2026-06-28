"""
Real ML pipeline Celery task — the counterpart to ``tasks.run_simulated_pipeline``.

Selected by ``create_project`` when ``settings.simulate_ml`` is false. It drives
the framework-agnostic ``reviewscope_ml`` seam (``run_from_upload``), reporting
progress through :class:`DbProgressSink` and persisting the resulting DTOs with
:func:`persist_run_result`. ``reviewscope_ml`` is imported lazily so the heavy ML
dependency tree only loads on this path, never for the simulated worker or tests.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.session import SyncSessionLocal
from app.ml_mapping import STEP_ORDER, DbProgressSink, build_upload_schema, persist_run_result
from app.models import PipelineJob, PipelineStepStatus, Project, ProjectSchema, ProjectStatus
from app.services.replay import replay_edits, snapshot_membership
from app.worker import celery_app


@celery_app.task(name="app.ml_pipeline.run_ml_pipeline")
def run_ml_pipeline(project_id: str) -> None:
    settings = get_settings()
    pid = uuid.UUID(project_id)

    with SyncSessionLocal() as session:
        project = session.get(Project, pid)
        if not project or not project.upload_path:
            return
        upload_path = project.upload_path
        schema_row = session.get(ProjectSchema, pid)
        columns = list(schema_row.columns) if schema_row else []
        _ensure_jobs(session, pid)
        session.execute(update(Project).where(Project.id == pid).values(status=ProjectStatus.processing, last_error=None))
        session.commit()

    try:
        # Lazy import: only this path needs torch / sentence-transformers / umap.
        from reviewscope_ml.app import IngestError, run_from_upload

        try:
            schema = build_upload_schema(columns)
            sink = DbProgressSink(SyncSessionLocal, pid)
            result = run_from_upload(
                file_path=upload_path,
                schema=schema,
                project_id=project_id,
                progress=sink,
                device=settings.ml_device,
                label_clusters=True,
            )
        except IngestError as exc:
            _fail(pid, "; ".join(exc.errors) if getattr(exc, "errors", None) else str(exc))
            return

        with SyncSessionLocal() as session:
            # Snapshot the human-curated membership *before* persist wipes it, then
            # replay the edit log over the fresh rows so manual work survives the
            # re-run. Replay runs after persist so re-applied human labels win.
            snapshot = snapshot_membership(session, pid)
            persist_run_result(session, result)
            replay_edits(session, pid, snapshot)
            # Capture the run-level quality report (silhouette/coherence/etc.) the ML
            # run already computed, so the UI can surface it (R17). metrics_run_at
            # marks when it was valid; edits after this make it stale.
            metrics = getattr(result, "metrics", None)
            session.execute(
                update(Project)
                .where(Project.id == pid)
                .values(
                    status=ProjectStatus.ready,
                    last_error=None,
                    metrics=dict(metrics) if metrics else None,
                    metrics_run_at=datetime.now(UTC),
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 — surface any failure as a failed run
        _fail(pid, str(exc))


def _ensure_jobs(session, project_id: uuid.UUID) -> None:
    existing = set(session.execute(select(PipelineJob.step).where(PipelineJob.project_id == project_id)).scalars().all())
    for step in STEP_ORDER:
        if step not in existing:
            session.add(PipelineJob(project_id=project_id, step=step, status=PipelineStepStatus.pending))


def _fail(project_id: uuid.UUID, message: str) -> None:
    with SyncSessionLocal() as session:
        session.execute(update(Project).where(Project.id == project_id).values(status=ProjectStatus.failed, last_error=message))
        session.execute(
            update(PipelineJob)
            .where(PipelineJob.project_id == project_id, PipelineJob.status == PipelineStepStatus.running)
            .values(status=PipelineStepStatus.failed, message=message, finished_at=datetime.now(UTC))
        )
        session.commit()
