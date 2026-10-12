from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.session import SyncSessionLocal
from app.ml_mapping import (
    STEP_ORDER,
    DbProgressSink,
    build_upload_schema,
    corpus_profile_for,
    persist_run_result,
)
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
        source_filename = project.source_filename
        schema_row = session.get(ProjectSchema, pid)
        columns = list(schema_row.columns) if schema_row else []
        _ensure_jobs(session, pid)
        session.execute(update(Project).where(Project.id == pid).values(status=ProjectStatus.processing, last_error=None))
        session.commit()

    claim = _claim_device(settings)
    try:
        # Keep heavy ML dependencies off the simulated worker and API import paths.
        from reviewscope_ml.app import IngestError, run_from_upload

        try:
            schema = build_upload_schema(columns)
            sink = DbProgressSink(SyncSessionLocal, pid)
            result = run_from_upload(
                file_path=upload_path,
                schema=schema,
                project_id=project_id,
                progress=sink,
                device=claim.device if claim else "cpu",
                label_clusters=True,
                corpus_profile=corpus_profile_for(source_filename),
                # Force reruns: the fixed project run name otherwise reloads stale parameters.
                force=True,
            )
        except IngestError as exc:
            _fail(pid, "; ".join(exc.errors) if getattr(exc, "errors", None) else str(exc))
            return

        with SyncSessionLocal() as session:
            # Snapshot before replacing rows; replay afterward so human edits survive reruns.
            snapshot = snapshot_membership(session, pid)
            persist_run_result(session, result)
            replay_edits(session, pid, snapshot)
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
    finally:
        if claim:
            claim.release()


def _claim_device(settings):
    """Pin this run to idle GPU(s), or None for a plain CPU run.

    The GPU box is shared and unscheduled, so ML_DEVICE=cuda only *requests* a
    GPU: claim_gpu() picks idle cards via nvidia-smi, sets CUDA_VISIBLE_DEVICES,
    and falls back to CPU when every card is busy or no driver is present. It
    must run before torch initialises CUDA, which is why the worker recycles
    its process after every task (see app.worker).
    """
    if settings.ml_device != "cuda":
        return None
    from reviewscope_ml.runtime.gpu import claim_gpu

    return claim_gpu(max_gpus=settings.ml_max_gpus or None)


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
