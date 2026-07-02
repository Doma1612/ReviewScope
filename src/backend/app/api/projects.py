import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import Float, case, cast, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_project_role
from app.core.config import get_settings
from app.db.session import get_db
from app.models import Cluster, ClusterEdit, Document, Embedding, PipelineJob, Project, ProjectMember, ProjectRole, ProjectSchema, ProjectStatus, Segment, User
from app.schemas import (
    BulkReassign,
    BulkReassignResult,
    BulkSegmentReassign,
    ClusterMembership,
    DocumentCount,
    ProjectMetricsRead,
    ClusterCreate,
    ClusterEditRead,
    ClusterFromSegments,
    ClusterMerge,
    ClusterRead,
    ClusterUpdate,
    DocumentRead,
    EmbeddingPoint,
    EmbeddingStats,
    MemberCreate,
    MemberRead,
    MemberUpdate,
    PipelineStatusRead,
    ProjectRead,
    ProjectSchemaRead,
    ProjectSchemaWrite,
    ProjectUpdate,
    ReviewReassign,
    SegmentReassign,
)
from app.ml_pipeline import run_ml_pipeline
from app.services.edits import record_edit
from app.services.recompute import recompute_clusters, recompute_document_primary
from app.tasks import PIPELINE_STEPS, run_simulated_pipeline


router = APIRouter()


@router.post("", response_model=ProjectRead, status_code=status.HTTP_202_ACCEPTED)
async def create_project(
    name: str = Form(...),
    schema_payload: str = Form("[]", alias="schema_json"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectRead:
    settings = get_settings()
    suffix = Path(file.filename or "upload.csv").suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl"}:
        raise HTTPException(status_code=400, detail="Only CSV, JSON, and JSONL uploads are supported")

    project = Project(name=name, owner_id=current_user.id, status=ProjectStatus.processing, source_filename=file.filename)
    db.add(project)
    await db.flush()
    db.add(ProjectMember(project_id=project.id, user_id=current_user.id, role=ProjectRole.owner))

    upload_path = settings.upload_dir / f"{project.id}{suffix}"
    with upload_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    project.upload_path = str(upload_path)

    try:
        columns = json.loads(schema_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="schema_json must be valid JSON") from exc
    db.add(ProjectSchema(project_id=project.id, columns=columns))
    for step in PIPELINE_STEPS:
        db.add(PipelineJob(project_id=project.id, step=step))
    await db.commit()
    await db.refresh(project)
    task = run_simulated_pipeline if settings.simulate_ml else run_ml_pipeline
    task.delay(str(project.id))
    return await _project_read(db, project, ProjectRole.owner)


@router.get("", response_model=list[ProjectRead])
async def list_projects(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[ProjectRead]:
    result = await db.execute(
        select(Project, ProjectMember.role, User.email)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .join(User, User.id == Project.owner_id)
        .where(ProjectMember.user_id == current_user.id)
        .order_by(Project.created_at.desc())
    )
    return [
        ProjectRead(
            id=project.id,
            name=project.name,
            owner_id=project.owner_id,
            owner_email=email,
            status=project.status,
            doc_count=project.doc_count,
            created_at=project.created_at,
            role=role,
            last_error=project.last_error,
            unit=project.unit,
        )
        for project, role, email in result.all()
    ]


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ProjectRead:
    member = await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    return await _project_read(db, project, member.role)


@router.patch("/{project_id}", response_model=ProjectRead)
async def update_project(project_id: uuid.UUID, payload: ProjectUpdate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ProjectRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    project = await _get_project_or_404(db, project_id)
    project.name = payload.name
    await db.commit()
    await db.refresh(project)
    return await _project_read(db, project, ProjectRole.owner)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> None:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    await db.execute(delete(Project).where(Project.id == project_id))
    await db.commit()


@router.get("/{project_id}/schema", response_model=ProjectSchemaRead)
async def get_schema(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ProjectSchemaRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    schema = await db.get(ProjectSchema, project_id)
    if not schema:
        raise HTTPException(status_code=404, detail="Project schema not found")
    return ProjectSchemaRead(columns=list(schema.columns))


@router.post("/{project_id}/schema", response_model=ProjectSchemaRead)
async def set_schema(project_id: uuid.UUID, payload: ProjectSchemaWrite, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ProjectSchemaRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    columns = [column.model_dump() for column in payload.columns]
    schema = await db.get(ProjectSchema, project_id)
    if schema:
        schema.columns = columns
    else:
        db.add(ProjectSchema(project_id=project_id, columns=columns))
    await db.commit()
    return ProjectSchemaRead(columns=columns)


@router.get("/{project_id}/pipeline/status", response_model=PipelineStatusRead)
async def pipeline_status(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> PipelineStatusRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    jobs = (await db.execute(select(PipelineJob).where(PipelineJob.project_id == project_id).order_by(PipelineJob.id))).scalars().all()
    return PipelineStatusRead(project_id=project_id, status=project.status, jobs=list(jobs))


@router.get("/{project_id}/metrics", response_model=ProjectMetricsRead)
async def project_metrics(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ProjectMetricsRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    # Stale = a manual edit happened after the run that produced these metrics.
    latest_edit = (
        await db.execute(select(func.max(ClusterEdit.created_at)).where(ClusterEdit.project_id == project_id))
    ).scalar_one_or_none()
    stale = bool(project.metrics_run_at and latest_edit and latest_edit > project.metrics_run_at)
    return ProjectMetricsRead(metrics=project.metrics, computed_at=project.metrics_run_at, stale=stale)


@router.get("/{project_id}/embeddings/stats", response_model=EmbeddingStats)
async def embeddings_stats(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> EmbeddingStats:
    """Total and noise (unclustered) point counts for the scatter.

    Cheap COUNT queries so the frontend can render an honest "N noise / M total"
    and decide whether the view is sampled, without downloading every point.
    """
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    model = Segment if project.unit == "sentence" else Document
    total = await db.scalar(select(func.count()).select_from(model).where(model.project_id == project_id))
    noise = await db.scalar(
        select(func.count()).select_from(model).where(model.project_id == project_id, model.cluster_id.is_(None))
    )
    return EmbeddingStats(total=int(total or 0), noise=int(noise or 0))


@router.get("/{project_id}/embeddings", response_model=list[EmbeddingPoint])
async def embeddings(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int | None = None, sample: int | None = None) -> list[EmbeddingPoint]:
    """Scatter points, one per clustered unit (segment for sentence, review for document).

    ``sample`` returns a random representative subset (``ORDER BY random() LIMIT``)
    so large projects — a sentence run is hundreds of thousands of segments — don't
    ship a ~180 MB payload the client only downsamples for display anyway. Use
    ``/embeddings/stats`` for the honest total/noise counts alongside a sampled fetch.
    """
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    if project.unit == "sentence":
        # One point per segment (mention); document_id is the parent review.
        query = (
            select(
                Segment.id,
                Segment.document_id,
                Segment.cluster_id,
                Segment.umap_x,
                Segment.umap_y,
                Segment.umap_z,
                Segment.text,
                Document.primary_key_value,
                Segment.sentiment_score,
                Cluster.label,
            )
            .join(Document, Document.id == Segment.document_id)
            .outerjoin(Cluster, Cluster.id == Segment.cluster_id)
            .where(Segment.project_id == project_id)
        )
        if sample is not None:
            query = query.order_by(func.random()).limit(sample)
        elif limit is not None:
            query = query.limit(limit)
        result = await db.execute(query)
        return [
            EmbeddingPoint(
                document_id=doc_id,
                segment_id=seg_id,
                cluster_id=cluster_id,
                x=x, y=y, z=z,
                snippet=text[:120] if text is not None else None,
                primary_key_value=primary_key_value,
                sentiment_score=sentiment_score,
                cluster_label=cluster_label,
            )
            for seg_id, doc_id, cluster_id, x, y, z, text, primary_key_value, sentiment_score, cluster_label in result.all()
        ]

    query = (
        select(
            Document.id,
            Document.cluster_id,
            Embedding.umap_x,
            Embedding.umap_y,
            Embedding.umap_z,
            Document.text,
            Document.primary_key_value,
            Document.sentiment_score,
            Cluster.label,
        )
        .join(Embedding, Embedding.document_id == Document.id)
        .outerjoin(Cluster, Cluster.id == Document.cluster_id)
        .where(Document.project_id == project_id)
    )
    if sample is not None:
        query = query.order_by(func.random()).limit(sample)
    elif limit is not None:
        query = query.limit(limit)
    result = await db.execute(query)
    return [
        EmbeddingPoint(
            document_id=doc_id,
            cluster_id=cluster_id,
            x=x,
            y=y,
            z=z,
            snippet=text[:120] if text is not None else None,
            primary_key_value=primary_key_value,
            sentiment_score=sentiment_score,
            cluster_label=cluster_label,
        )
        for doc_id, cluster_id, x, y, z, text, primary_key_value, sentiment_score, cluster_label in result.all()
    ]


@router.get("/{project_id}/clusters", response_model=list[ClusterRead])
async def clusters(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[ClusterRead]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    sentence = project.unit == "sentence"
    result = await db.execute(select(Cluster).where(Cluster.project_id == project_id).order_by(Cluster.label))
    # One grouped query for the per-cluster count of members that actually have a
    # sentiment score, so the UI can show coverage ("sentiment on n of N") instead
    # of implying the mean covers every member. Members are segments for sentence.
    if sentence:
        counts_q = (
            select(Segment.cluster_id, func.count())
            .where(Segment.project_id == project_id, Segment.cluster_id.is_not(None), Segment.sentiment_score.is_not(None))
            .group_by(Segment.cluster_id)
        )
    else:
        counts_q = (
            select(Document.cluster_id, func.count())
            .where(Document.project_id == project_id, Document.cluster_id.is_not(None), Document.sentiment_score.is_not(None))
            .group_by(Document.cluster_id)
        )
    counts = dict((await db.execute(counts_q)).all())
    items = []
    for cluster in result.scalars().all():
        sample_docs = await _sample_docs(db, project_id, cluster.id, sentence=sentence)
        items.append(
            ClusterRead.model_validate(cluster).model_copy(
                update={"sample_docs": sample_docs, "sentiment_count": counts.get(cluster.id, 0)}
            )
        )
    return items


@router.get("/{project_id}/clusters/{cluster_id}", response_model=ClusterRead)
async def cluster_detail(project_id: uuid.UUID, cluster_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    project = await _get_project_or_404(db, project_id)
    sentence = project.unit == "sentence"
    return ClusterRead.model_validate(cluster).model_copy(
        update={
            "sample_docs": await _sample_docs(db, project_id, cluster_id, 5, sentence=sentence),
            "sentiment_count": await _sentiment_count(db, project_id, cluster_id, sentence=sentence),
        }
    )


@router.get("/{project_id}/clusters/{cluster_id}/documents", response_model=list[DocumentRead])
async def cluster_documents(project_id: uuid.UUID, cluster_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int = 50, offset: int = 0) -> list[DocumentRead]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    if project.unit == "sentence":
        # Distinct reviews that have at least one mention in this cluster.
        doc_ids = (
            await db.execute(
                select(Segment.document_id)
                .where(Segment.project_id == project_id, Segment.cluster_id == cluster_id)
                .group_by(Segment.document_id)
                .limit(limit)
                .offset(offset)
            )
        ).scalars().all()
        if not doc_ids:
            return []
        docs = (await db.execute(select(Document).where(Document.id.in_(doc_ids)))).scalars().all()
        memberships = await _document_memberships(db, project_id, list(doc_ids))
        return [_document_read(doc, memberships.get(doc.id, [])) for doc in docs]
    result = await db.execute(select(Document).where(Document.project_id == project_id, Document.cluster_id == cluster_id).limit(limit).offset(offset))
    return [_document_read(doc, []) for doc in result.scalars().all()]


@router.post("/{project_id}/clusters", response_model=ClusterRead, status_code=status.HTTP_201_CREATED)
async def create_cluster(project_id: uuid.UUID, payload: ClusterCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    cluster = Cluster(
        id=uuid.uuid4(),
        project_id=project_id,
        label=payload.label,
        summary="",
        label_source="hitl_override",
        top_terms=[],
        word_frequencies={},
        size=0,
    )
    db.add(cluster)
    record_edit(db, project_id=project_id, actor_id=current_user.id, action="create_cluster", cluster_id=cluster.id, new_label=payload.label)
    await db.commit()
    await db.refresh(cluster)
    return ClusterRead.model_validate(cluster).model_copy(update={"sample_docs": []})


@router.post("/{project_id}/clusters/merge", response_model=ClusterRead)
async def merge_clusters(project_id: uuid.UUID, payload: ClusterMerge, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    project = await _get_project_or_404(db, project_id)
    _require_editable(project)
    if payload.target_id in payload.source_ids:
        raise HTTPException(status_code=400, detail="target_id cannot be among source_ids")
    target = await db.get(Cluster, payload.target_id)
    if not target or target.project_id != project_id:
        raise HTTPException(status_code=404, detail="Target cluster not found")
    sources = []
    for source_id in payload.source_ids:
        source = await db.get(Cluster, source_id)
        if not source or source.project_id != project_id:
            raise HTTPException(status_code=404, detail="Source cluster not found")
        sources.append(source)
    # Move every mention of the source clusters into the target, then refresh the
    # affected reviews' derived primary cluster.
    segs = (await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.cluster_id.in_(payload.source_ids))
    )).scalars().all()
    affected_docs = {seg.document_id for seg in segs}
    for seg in segs:
        seg.cluster_id = payload.target_id
    for source in sources:
        record_edit(db, project_id=project_id, actor_id=current_user.id, action="merge_clusters", cluster_id=source.id, target_cluster_id=payload.target_id)
        await db.delete(source)
    await recompute_clusters(db, project_id, [payload.target_id])
    await recompute_document_primary(db, project_id, list(affected_docs))
    await db.commit()
    await db.refresh(target)
    return ClusterRead.model_validate(target).model_copy(update={"sample_docs": await _sample_docs(db, project_id, target.id, 5, sentence=True)})


@router.post("/{project_id}/clusters/from-selection", response_model=ClusterRead, status_code=status.HTTP_201_CREATED)
async def cluster_from_selection(project_id: uuid.UUID, payload: ClusterFromSegments, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead:
    """Create a cluster from a lasso selection of *segment* mentions (sentence-unit)."""
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    cluster = Cluster(
        id=uuid.uuid4(),
        project_id=project_id,
        label=payload.label,
        summary="",
        label_source="hitl_override",
        top_terms=[],
        word_frequencies={},
        size=0,
    )
    db.add(cluster)
    segs = list((await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.id.in_(payload.segment_ids))
    )).scalars().all())
    affected: set[uuid.UUID] = {cluster.id}
    affected_docs: set[uuid.UUID] = set()
    for seg in segs:
        if seg.cluster_id is not None:
            affected.add(seg.cluster_id)
        seg.cluster_id = cluster.id
        affected_docs.add(seg.document_id)
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="create_from_selection",
        cluster_id=cluster.id,
        new_label=payload.label,
        payload={"segment_ids": [str(seg.id) for seg in segs]},
    )
    await recompute_clusters(db, project_id, list(affected))
    await recompute_document_primary(db, project_id, list(affected_docs))
    await db.commit()
    await db.refresh(cluster)
    return ClusterRead.model_validate(cluster).model_copy(update={"sample_docs": await _sample_docs(db, project_id, cluster.id, 5, sentence=True)})


@router.patch("/{project_id}/clusters/{cluster_id}", response_model=ClusterRead | None)
async def update_cluster(project_id: uuid.UUID, cluster_id: uuid.UUID, payload: ClusterUpdate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead | None:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    if payload.mark_junk:
        await _junk_cluster(db, project_id, cluster, current_user.id)
        await db.commit()
        return None
    changed = False
    if payload.label is not None:
        before_label = cluster.label
        cluster.label = payload.label
        cluster.label_source = "hitl_override"
        record_edit(db, project_id=project_id, actor_id=current_user.id, action="rename_label", cluster_id=cluster_id, new_label=payload.label, payload={"before": before_label})
        changed = True
    if payload.approve:
        cluster.label_source = "hitl_approved"
        record_edit(db, project_id=project_id, actor_id=current_user.id, action="approve_label", cluster_id=cluster_id)
        changed = True
    if not changed:
        raise HTTPException(status_code=400, detail="Provide one of label, approve, or mark_junk")
    await db.commit()
    await db.refresh(cluster)
    return ClusterRead.model_validate(cluster).model_copy(update={"sample_docs": await _sample_docs(db, project_id, cluster_id, 5, sentence=True)})


@router.delete("/{project_id}/clusters/{cluster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cluster(project_id: uuid.UUID, cluster_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> None:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    await _junk_cluster(db, project_id, cluster, current_user.id)
    await db.commit()


# Match a value safe to cast to a number, so a numeric range filter never errors on
# a non-numeric raw_data cell (the CASE returns NULL → the row simply doesn't match).
_NUMERIC_RE = r"^-?[0-9]+(\.[0-9]+)?$"


def _document_filter_conditions(filters_json: str | None) -> list:
    """Build WHERE conditions from a JSON facet spec over each doc's ``raw_data``.

    ``filters`` is a JSON list of ``{column, op, value, type}`` where ``op`` is
    ``eq``/``gte``/``lte``. Numeric ranges are cast through a regex-guarded CASE so a
    non-numeric cell can't raise; date ranges compare ISO text lexically (which is
    chronological); ``eq`` is an exact text match (booleans, exact values). The
    column comes from the typed schema, but we still treat it defensively. Invalid
    JSON / unknown ops are ignored rather than erroring the request."""
    if not filters_json:
        return []
    try:
        specs = json.loads(filters_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(specs, list):
        return []

    conditions = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        column = spec.get("column")
        op = spec.get("op")
        value = spec.get("value")
        col_type = spec.get("type", "text")
        if not column or value in (None, ""):
            continue
        text_expr = Document.raw_data[column].astext
        if op == "eq":
            conditions.append(text_expr == str(value))
        elif op in ("gte", "lte"):
            if col_type in ("integer", "float"):
                try:
                    bound = float(value)
                except (TypeError, ValueError):
                    continue
                numeric = case((text_expr.op("~")(_NUMERIC_RE), cast(text_expr, Float)), else_=None)
                conditions.append(numeric >= bound if op == "gte" else numeric <= bound)
            else:  # date / text — lexical comparison (ISO dates sort chronologically)
                conditions.append(text_expr >= str(value) if op == "gte" else text_expr <= str(value))
    return conditions


@router.get("/{project_id}/documents", response_model=list[DocumentRead])
async def documents(project_id: uuid.UUID, cluster_id: uuid.UUID | None = None, filters: str | None = None, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int = 50, offset: int = 0) -> list[Document]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    sentence = project.unit == "sentence"
    query = select(Document).where(Document.project_id == project_id)
    if cluster_id:
        # For sentence-unit, filter to reviews with a mention in the cluster (not
        # only those whose *primary* is the cluster) so the cluster page is complete.
        if sentence:
            member_ids = select(Segment.document_id).where(
                Segment.project_id == project_id, Segment.cluster_id == cluster_id
            )
            query = query.where(Document.id.in_(member_ids))
        else:
            query = query.where(Document.cluster_id == cluster_id)
    for condition in _document_filter_conditions(filters):
        query = query.where(condition)
    query = query.limit(limit).offset(offset)
    docs = list((await db.execute(query)).scalars().all())
    memberships = await _document_memberships(db, project_id, [d.id for d in docs]) if sentence else {}
    return [_document_read(doc, memberships.get(doc.id, [])) for doc in docs]


# Declared before /documents/{document_id} so "count" isn't matched as a doc id.
@router.get("/{project_id}/documents/count", response_model=DocumentCount)
async def documents_count(project_id: uuid.UUID, cluster_id: uuid.UUID | None = None, filters: str | None = None, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> DocumentCount:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    query = select(func.count()).select_from(Document).where(Document.project_id == project_id)
    if cluster_id:
        query = query.where(Document.cluster_id == cluster_id)
    for condition in _document_filter_conditions(filters):
        query = query.where(condition)
    total = (await db.execute(query)).scalar_one()
    return DocumentCount(total=int(total))


@router.get("/{project_id}/documents/{document_id}", response_model=DocumentRead)
async def document(project_id: uuid.UUID, document_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> DocumentRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="Document not found")
    project = await _get_project_or_404(db, project_id)
    memberships = (await _document_memberships(db, project_id, [doc.id])).get(doc.id, []) if project.unit == "sentence" else []
    return _document_read(doc, memberships)


@router.patch("/{project_id}/documents/{document_id}", response_model=DocumentRead)
async def reassign_document(project_id: uuid.UUID, document_id: uuid.UUID, payload: ReviewReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> DocumentRead:
    """Move *all* of a review's mentions to one cluster (sentence-unit "move all")."""
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    project = await _get_project_or_404(db, project_id)
    _require_editable(project)
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="Document not found")
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    old_primary = doc.cluster_id
    segs = list((await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.document_id == document_id)
    )).scalars().all())
    affected: set[uuid.UUID] = {target_cluster_id} if target_cluster_id is not None else set()
    for seg in segs:
        if seg.cluster_id is not None:
            affected.add(seg.cluster_id)
        seg.cluster_id = target_cluster_id
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="reassign_review",
        document_id=document_id,
        cluster_id=old_primary,
        target_cluster_id=target_cluster_id,
    )
    await recompute_clusters(db, project_id, list(affected))
    await recompute_document_primary(db, project_id, [document_id])
    await db.commit()
    await db.refresh(doc)
    memberships = (await _document_memberships(db, project_id, [doc.id])).get(doc.id, [])
    return _document_read(doc, memberships)


@router.post("/{project_id}/documents/reassign", response_model=BulkReassignResult)
async def bulk_reassign_documents(project_id: uuid.UUID, payload: BulkReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> BulkReassignResult:
    """Move all mentions of each listed review to one cluster (sentence-unit)."""
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    docs = list((await db.execute(
        select(Document).where(Document.project_id == project_id, Document.id.in_(payload.document_ids))
    )).scalars().all())
    doc_ids = [doc.id for doc in docs]
    # Capture each review's old primary before the move so undo can restore it.
    before = {str(doc.id): (str(doc.cluster_id) if doc.cluster_id else None) for doc in docs}
    segs = list((await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.document_id.in_(doc_ids))
    )).scalars().all())
    affected: set[uuid.UUID] = {target_cluster_id} if target_cluster_id is not None else set()
    for seg in segs:
        if seg.cluster_id is not None:
            affected.add(seg.cluster_id)
        seg.cluster_id = target_cluster_id
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="bulk_reassign",
        target_cluster_id=target_cluster_id,
        payload={"document_ids": [str(d) for d in doc_ids], "before": before},
    )
    await recompute_clusters(db, project_id, list(affected))
    await recompute_document_primary(db, project_id, doc_ids)
    await db.commit()
    return BulkReassignResult(moved=len(docs))


@router.patch("/{project_id}/segments/{segment_id}", response_model=EmbeddingPoint)
async def reassign_segment(project_id: uuid.UUID, segment_id: uuid.UUID, payload: SegmentReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> EmbeddingPoint:
    """Move a single mention (segment) to a cluster (or noise)."""
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    seg = await db.get(Segment, segment_id)
    if not seg or seg.project_id != project_id:
        raise HTTPException(status_code=404, detail="Segment not found")
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    old_cluster_id = seg.cluster_id
    seg.cluster_id = target_cluster_id
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="reassign_segment",
        segment_id=segment_id,
        cluster_id=old_cluster_id,
        target_cluster_id=target_cluster_id,
    )
    await recompute_clusters(db, project_id, [cid for cid in {old_cluster_id, target_cluster_id} if cid is not None])
    await recompute_document_primary(db, project_id, [seg.document_id])
    await db.commit()
    await db.refresh(seg)
    label = None
    if seg.cluster_id is not None:
        c = await db.get(Cluster, seg.cluster_id)
        label = c.label if c else None
    return EmbeddingPoint(
        document_id=seg.document_id, segment_id=seg.id, cluster_id=seg.cluster_id,
        x=seg.umap_x, y=seg.umap_y, z=seg.umap_z, snippet=seg.text[:120],
        sentiment_score=seg.sentiment_score, cluster_label=label,
    )


@router.post("/{project_id}/segments/reassign", response_model=BulkReassignResult)
async def bulk_reassign_segments(project_id: uuid.UUID, payload: BulkSegmentReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> BulkReassignResult:
    """Move a lasso selection of mentions (segments) to one cluster (or noise)."""
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    _require_editable(await _get_project_or_404(db, project_id))
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    segs = list((await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.id.in_(payload.segment_ids))
    )).scalars().all())
    before = {str(seg.id): (str(seg.cluster_id) if seg.cluster_id else None) for seg in segs}
    affected: set[uuid.UUID] = {target_cluster_id} if target_cluster_id is not None else set()
    affected_docs: set[uuid.UUID] = set()
    for seg in segs:
        if seg.cluster_id is not None:
            affected.add(seg.cluster_id)
        seg.cluster_id = target_cluster_id
        affected_docs.add(seg.document_id)
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="bulk_reassign_segments",
        target_cluster_id=target_cluster_id,
        payload={"segment_ids": [str(seg.id) for seg in segs], "before": before},
    )
    await recompute_clusters(db, project_id, list(affected))
    await recompute_document_primary(db, project_id, list(affected_docs))
    await db.commit()
    return BulkReassignResult(moved=len(segs))


@router.get("/{project_id}/members", response_model=list[MemberRead])
async def list_members(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[MemberRead]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    result = await db.execute(select(ProjectMember, User).join(User, User.id == ProjectMember.user_id).where(ProjectMember.project_id == project_id))
    return [MemberRead(user_id=user.id, email=user.email, role=member.role) for member, user in result.all()]


@router.post("/{project_id}/members", response_model=MemberRead, status_code=status.HTTP_201_CREATED)
async def add_member(project_id: uuid.UUID, payload: MemberCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> MemberRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    if payload.role == ProjectRole.owner:
        raise HTTPException(status_code=400, detail="Only viewer invitations are supported")
    user = (await db.execute(select(User).where(User.email == payload.email.lower()))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User must register before being invited")
    existing = await db.get(ProjectMember, {"project_id": project_id, "user_id": user.id})
    if existing:
        existing.role = payload.role
    else:
        db.add(ProjectMember(project_id=project_id, user_id=user.id, role=payload.role))
    await db.commit()
    return MemberRead(user_id=user.id, email=user.email, role=payload.role)


@router.patch("/{project_id}/members/{user_id}", response_model=MemberRead)
async def update_member(project_id: uuid.UUID, user_id: uuid.UUID, payload: MemberUpdate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> MemberRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    member = await db.get(ProjectMember, {"project_id": project_id, "user_id": user_id})
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    if member.role == ProjectRole.owner or payload.role == ProjectRole.owner:
        raise HTTPException(status_code=400, detail="Owner role cannot be changed here")
    member.role = payload.role
    user = await db.get(User, user_id)
    await db.commit()
    return MemberRead(user_id=user_id, email=user.email if user else "", role=member.role)


@router.delete("/{project_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(project_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> None:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    member = await db.get(ProjectMember, {"project_id": project_id, "user_id": user_id})
    if member and member.role != ProjectRole.owner:
        await db.delete(member)
        await db.commit()


@router.get("/{project_id}/edits", response_model=list[ClusterEditRead])
async def list_edits(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[ClusterEdit]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    result = await db.execute(select(ClusterEdit).where(ClusterEdit.project_id == project_id).order_by(ClusterEdit.created_at.desc()))
    return list(result.scalars().all())


async def _junk_cluster(db: AsyncSession, project_id: uuid.UUID, cluster: Cluster, actor_id: uuid.UUID) -> None:
    """Mark a cluster as junk: its mentions become noise and the cluster is removed.

    Sentence-unit only (guarded by callers via ``_require_editable``): moves the
    cluster's segments to noise, refreshes the affected reviews' derived primary,
    stages the audit row, and deletes the cluster. The caller owns the commit."""
    segs = list((await db.execute(
        select(Segment).where(Segment.project_id == project_id, Segment.cluster_id == cluster.id)
    )).scalars().all())
    affected_docs = {seg.document_id for seg in segs}
    for seg in segs:
        seg.cluster_id = None
    record_edit(
        db,
        project_id=project_id,
        actor_id=actor_id,
        action="mark_junk",
        cluster_id=cluster.id,
        payload={"segment_ids": [str(seg.id) for seg in segs]},
    )
    await db.delete(cluster)
    await recompute_document_primary(db, project_id, list(affected_docs))


async def _get_project_or_404(db: AsyncSession, project_id: uuid.UUID) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


async def _project_read(db: AsyncSession, project: Project, role: ProjectRole) -> ProjectRead:
    owner_email = (await db.execute(select(User.email).where(User.id == project.owner_id))).scalar_one_or_none()
    return ProjectRead(
        id=project.id,
        name=project.name,
        owner_id=project.owner_id,
        owner_email=owner_email,
        status=project.status,
        doc_count=project.doc_count,
        created_at=project.created_at,
        role=role,
        last_error=project.last_error,
        unit=project.unit,
    )


def _document_read(doc: Document, memberships: list[ClusterMembership]) -> DocumentRead:
    return DocumentRead(
        id=doc.id,
        primary_key_value=doc.primary_key_value,
        text=doc.text,
        raw_data=doc.raw_data,
        cluster_id=doc.cluster_id,
        sentiment_score=doc.sentiment_score,
        memberships=memberships,
        primary_cluster_id=doc.cluster_id,
    )


def _require_editable(project: Project) -> None:
    """Editing is only supported on sentence-unit projects. Document-unit projects
    are frozen (read-only) — re-run to upgrade them to sentence-level."""
    if project.unit != "sentence":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This project is document-level and read-only. Re-run it to enable segment-level editing.",
        )


async def _sample_docs(db: AsyncSession, project_id: uuid.UUID, cluster_id: uuid.UUID, limit: int = 3, *, sentence: bool = False) -> list[dict]:
    """Short member samples for a cluster card — segment texts for sentence-unit
    projects, whole-document texts otherwise."""
    if sentence:
        result = await db.execute(
            select(Segment.id, Segment.text).where(Segment.project_id == project_id, Segment.cluster_id == cluster_id).limit(limit)
        )
    else:
        result = await db.execute(
            select(Document.id, Document.text).where(Document.project_id == project_id, Document.cluster_id == cluster_id).limit(limit)
        )
    return [{"id": str(row_id), "text": text[:240]} for row_id, text in result.all()]


async def _sentiment_count(db: AsyncSession, project_id: uuid.UUID, cluster_id: uuid.UUID, *, sentence: bool = False) -> int:
    """Count of this cluster's members carrying a sentiment score (the n in the
    "sentiment on n of N" coverage). Members are segments for sentence-unit."""
    if sentence:
        query = select(func.count()).where(
            Segment.project_id == project_id,
            Segment.cluster_id == cluster_id,
            Segment.sentiment_score.is_not(None),
        )
    else:
        query = select(func.count()).where(
            Document.project_id == project_id,
            Document.cluster_id == cluster_id,
            Document.sentiment_score.is_not(None),
        )
    result = await db.execute(query)
    return int(result.scalar_one())


async def _document_memberships(db: AsyncSession, project_id: uuid.UUID, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[ClusterMembership]]:
    """Per-review cluster membership for sentence-unit projects: for each document,
    the clusters its mentions fall into with mention counts + share (of the review's
    segments). Noise mentions are excluded from the chips but counted in the share."""
    if not document_ids:
        return {}
    totals = dict(
        (
            await db.execute(
                select(Segment.document_id, func.count())
                .where(Segment.project_id == project_id, Segment.document_id.in_(document_ids))
                .group_by(Segment.document_id)
            )
        ).all()
    )
    rows = (
        await db.execute(
            select(Segment.document_id, Segment.cluster_id, Cluster.label, func.count())
            .join(Cluster, Cluster.id == Segment.cluster_id)
            .where(
                Segment.project_id == project_id,
                Segment.document_id.in_(document_ids),
                Segment.cluster_id.is_not(None),
            )
            .group_by(Segment.document_id, Segment.cluster_id, Cluster.label)
        )
    ).all()
    out: dict[uuid.UUID, list[ClusterMembership]] = {}
    for doc_id, cluster_id, label, count in rows:
        total = totals.get(doc_id, count) or count
        out.setdefault(doc_id, []).append(
            ClusterMembership(cluster_id=cluster_id, cluster_label=label, mention_count=int(count), share=round(count / total, 4))
        )
    for memberships in out.values():
        memberships.sort(key=lambda m: m.mention_count, reverse=True)
    return out
