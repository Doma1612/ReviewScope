import json
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
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
    ClusterChatRequest,
    ClusterChatResponse,
    ClusterChatResult,
    ClusterChatSource,
    ClusterEditRead,
    ClusterChatMessage,
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
from app.services.ollama_chat import OllamaChatError, chat_completion, chat_message, stream_chat_completion
from app.services.edits import record_edit
from app.services.recompute import _parse_rating, _rating_column, recompute_clusters, recompute_document_primary
from app.tasks import PIPELINE_STEPS, run_simulated_pipeline


router = APIRouter()

_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {"type": "string", "enum": ["rating", "sentiment", "useful", "funny", "cool", "date", "length", "text"]},
        "op": {"type": "string", "enum": ["eq", "gt", "gte", "lt", "lte", "contains"]},
        "value": {},
    },
    "required": ["field", "op", "value"],
}

_RESPONSE_MODE_SCHEMA = {"type": "string", "enum": ["direct_list", "analysis", "facts"]}

_CLUSTER_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_cluster",
            "description": "Return exact facts for the currently open cluster.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_current_cluster",
            "description": "Explain, interpret, summarize, or analyze the currently open cluster and its aggregate metrics. Use this for questions about what sentiment, average stars, cohesion, size, or mentions mean. Do not return review rows unless the user explicitly asks for examples or evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "enum": ["general", "sentiment", "rating", "cohesion", "size", "mentions"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reviews_by_rating",
            "description": "Return reviews in the current cluster having one exact star rating. Use for requests such as reviews with 1 star, 4 stars, or score 5.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rating": {"type": "number", "minimum": 0, "maximum": 5},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
                "required": ["rating"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_lowest_rated_reviews",
            "description": "Return the bottom or lowest-rated reviews in the current cluster, ordered by star rating ascending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_highest_rated_reviews",
            "description": "Return the top or highest-rated reviews in the current cluster, ordered by star rating descending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reviews",
            "description": "Return exact review records from the current cluster. Use only when the user explicitly asks to list, show, retrieve, filter, rank, or provide review examples/evidence. Do not use merely to explain a cluster-level metric. Top/highest scores means rating descending; bottom/lowest scores means rating ascending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "sort_by": {"type": "string", "enum": ["rating", "sentiment", "useful", "funny", "cool", "date", "length"]},
                    "sort_dir": {"type": "string", "enum": ["asc", "desc"]},
                    "filters": {"type": "array", "items": _FILTER_SCHEMA},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_current_cluster_reviews",
            "description": "Count all reviews associated with the currently open cluster. Use for unfiltered questions asking how many reviews the cluster contains. This tool takes no filters.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_reviews",
            "description": "Count reviews in the current cluster using an explicit rating, sentiment, metadata, date, length, or text filter. For the unfiltered total, use count_current_cluster_reviews instead.",
            "parameters": {
                "type": "object",
                "properties": {"filters": {"type": "array", "items": _FILTER_SCHEMA}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Find current-cluster reviews containing a literal text phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_review",
            "description": "Retrieve one exact review by ID, only if it belongs to the current cluster.",
            "parameters": {
                "type": "object",
                "properties": {"review_id": {"type": "string"}, "response_mode": _RESPONSE_MODE_SCHEMA},
                "required": ["review_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_reviews_by_opinion",
            "description": "Rank verified current-cluster reviews for a genuinely subjective goal such as most insightful, concerning, sarcastic, or best-written. Do not use for top/bottom numeric scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    "filters": {"type": "array", "items": _FILTER_SCHEMA},
                    "response_mode": _RESPONSE_MODE_SCHEMA,
                },
                "required": ["goal"],
            },
        },
    },
]

_RANKED_REVIEW_IDS_SCHEMA = {
    "type": "object",
    "properties": {
        "ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ids"],
    "additionalProperties": False,
}


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
            retrieval_ready=project.retrieval_ready,
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
    latest_edit = (
        await db.execute(select(func.max(ClusterEdit.created_at)).where(ClusterEdit.project_id == project_id))
    ).scalar_one_or_none()
    stale = bool(project.metrics_run_at and latest_edit and latest_edit > project.metrics_run_at)
    return ProjectMetricsRead(metrics=project.metrics, computed_at=project.metrics_run_at, stale=stale)


@router.get("/{project_id}/embeddings/stats", response_model=EmbeddingStats)
async def embeddings_stats(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> EmbeddingStats:
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
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    if project.unit == "sentence":
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


@router.post("/{project_id}/clusters/{cluster_id}/chat", response_model=ClusterChatResponse)
async def cluster_chat(project_id: uuid.UUID, cluster_id: uuid.UUID, payload: ClusterChatRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterChatResponse:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    question = _normalize_chat_question(payload.message)
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    settings = get_settings()
    context, sources, direct_answer, results, tool_state = await _cluster_chat_context(
        db,
        project_id,
        cluster,
        question=question,
        history=payload.history,
        active_source_ids=payload.active_source_ids,
        last_tool_state=payload.last_tool_state,
        sentence=project.unit == "sentence",
    )
    if direct_answer is not None:
        return ClusterChatResponse(message=ClusterChatMessage(role="assistant", content=direct_answer), model="deterministic", sources=sources, results=results, tool_state=tool_state)
    messages = _cluster_chat_messages(context, payload.history, question)
    try:
        answer = await run_in_threadpool(
            chat_completion,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=messages,
            timeout=settings.ollama_timeout_seconds,
        )
    except OllamaChatError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ClusterChatResponse(message=ClusterChatMessage(role="assistant", content=answer), model=settings.ollama_chat_model, sources=sources, results=results, tool_state=tool_state)


@router.post("/{project_id}/clusters/{cluster_id}/chat/stream")
async def cluster_chat_stream(project_id: uuid.UUID, cluster_id: uuid.UUID, payload: ClusterChatRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> StreamingResponse:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    question = _normalize_chat_question(payload.message)
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    settings = get_settings()
    context, sources, direct_answer, results, tool_state = await _cluster_chat_context(
        db,
        project_id,
        cluster,
        question=question,
        history=payload.history,
        active_source_ids=payload.active_source_ids,
        last_tool_state=payload.last_tool_state,
        sentence=project.unit == "sentence",
    )
    messages = _cluster_chat_messages(context, payload.history, question)

    def events():
        yield _ndjson({"type": "sources", "sources": [source.model_dump(mode="json") for source in sources]})
        if results:
            yield _ndjson({"type": "results", "results": [result.model_dump(mode="json") for result in results]})
        if tool_state:
            yield _ndjson({"type": "tool_state", "tool_state": tool_state})
        if direct_answer is not None:
            yield _ndjson({"type": "token", "content": direct_answer})
            yield _ndjson({"type": "done", "model": "deterministic"})
            return
        try:
            for token in stream_chat_completion(
                base_url=settings.ollama_base_url,
                model=settings.ollama_chat_model,
                messages=messages,
                timeout=settings.ollama_timeout_seconds,
            ):
                yield _ndjson({"type": "token", "content": token})
            yield _ndjson({"type": "done", "model": settings.ollama_chat_model})
        except OllamaChatError as exc:
            yield _ndjson({"type": "error", "message": str(exc)})

    return StreamingResponse(events(), media_type="application/x-ndjson")


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


# Guard numeric casts: uploaded cells may contain non-numeric text.
_NUMERIC_RE = r"^-?[0-9]+(\.[0-9]+)?$"


def _document_filter_conditions(filters_json: str | None) -> list:
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
            else:
                conditions.append(text_expr >= str(value) if op == "gte" else text_expr <= str(value))
    return conditions


@router.get("/{project_id}/documents", response_model=list[DocumentRead])
async def documents(project_id: uuid.UUID, cluster_id: uuid.UUID | None = None, filters: str | None = None, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int = 50, offset: int = 0) -> list[Document]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    sentence = project.unit == "sentence"
    query = select(Document).where(Document.project_id == project_id)
    if cluster_id:
        # Include every review with a cluster mention, not just its primary cluster.
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


# Register /count before /{document_id} so "count" is not parsed as a UUID.
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
        retrieval_ready=project.retrieval_ready,
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
    if project.unit != "sentence":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This project is document-level and read-only. Re-run it to enable segment-level editing.",
        )


async def _sample_docs(db: AsyncSession, project_id: uuid.UUID, cluster_id: uuid.UUID, limit: int = 3, *, sentence: bool = False) -> list[dict]:
    if sentence:
        result = await db.execute(
            select(Segment.id, Segment.text).where(Segment.project_id == project_id, Segment.cluster_id == cluster_id).limit(limit)
        )
    else:
        result = await db.execute(
            select(Document.id, Document.text).where(Document.project_id == project_id, Document.cluster_id == cluster_id).limit(limit)
        )
    return [{"id": str(row_id), "text": text[:240]} for row_id, text in result.all()]


def _ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _cluster_chat_messages(context: str, history: list[ClusterChatMessage], question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant for qualitative review analysis. "
                "Answer only from the provided current-cluster sources and context. If listed sources are provided, treat them as the only evidence. "
                "If a requested document, address, business id, metadata value, count, or review is not present in the current-cluster sources/context, say it was not found in this cluster. "
                "Never invent business IDs, addresses, reviews, statistics, counts, or evidence. Do not dump raw metadata or markdown code fences unless the user explicitly asks for raw metadata. "
                "When the user asks what you can tell them about a specific review, give a concise qualitative interpretation of the review's experience, tone, and notable signals. Be concise and analytical.\n\n"
                f"Cluster context:\n{context}"
            ),
        },
        *[m.model_dump() for m in history[-8:]],
        {"role": "user", "content": question},
    ]


async def _cluster_chat_context(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster: Cluster,
    *,
    question: str = "",
    history: list[ClusterChatMessage] | None = None,
    active_source_ids: list[uuid.UUID] | None = None,
    last_tool_state: dict | None = None,
    sentence: bool = False,
) -> tuple[str, list[ClusterChatSource], str | None, list[ClusterChatResult], dict | None]:
    structured = await _cluster_structured_answer(db, project_id, cluster.id, question, history=history or [], last_tool_state=last_tool_state, sentence=sentence)
    if structured is not None:
        direct_answer, sources, results, tool_state = structured
        return "", sources, direct_answer, results, tool_state

    samples = await _sample_docs(db, project_id, cluster.id, limit=8, sentence=sentence)
    terms = ", ".join(str(term.get("term", "")) for term in cluster.top_terms[:12] if term.get("term"))
    context = [
        f"Label: {cluster.label}",
        f"Size: {cluster.size} reviews" + (f"; {cluster.n_mentions} mentions" if sentence and cluster.n_mentions else ""),
    ]
    if cluster.summary:
        context.append(f"Summary: {cluster.summary}")
    if terms:
        context.append(f"Top terms: {terms}")
    if samples:
        quoted = "\n".join(f"- {sample['text']}" for sample in samples)
        context.append(f"Sample {'segments' if sentence else 'documents'}:\n{quoted}")
    return "\n".join(context), [], None, [], None


async def _cluster_structured_answer(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    question: str,
    *,
    history: list[ClusterChatMessage],
    last_tool_state: dict | None = None,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult], dict | None] | None:
    return await _planned_cluster_chat_answer(
        db,
        project_id,
        cluster_id,
        question,
        history=history,
        last_tool_state=last_tool_state,
        sentence=sentence,
    )


async def _planned_cluster_chat_answer(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    question: str,
    *,
    history: list[ClusterChatMessage],
    last_tool_state: dict | None,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult], dict | None] | None:
    if social_answer := _social_chat_answer(question):
        return social_answer, [], [], last_tool_state
    plan = await _plan_cluster_chat_tool(db, project_id, cluster_id, question, history, last_tool_state)
    if not plan:
        answer = await _answer_from_current_cluster_context(db, project_id, cluster_id, question)
        if fallback_plan := _tool_plan_from_message_content(answer):
            validated_fallback = _validate_cluster_chat_plan(fallback_plan)
            if validated_fallback:
                executed = await _execute_cluster_chat_plan(
                    db,
                    project_id,
                    cluster_id,
                    question,
                    validated_fallback,
                    sentence=sentence,
                )
                if executed is not None:
                    return executed
        if _looks_like_tool_call_content(answer):
            return "I couldn't safely process that data request. Please try again.", [], [], last_tool_state
        return answer, [], [], last_tool_state
    if direct_answer := plan.get("direct_answer"):
        return str(direct_answer), [], [], last_tool_state
    validated = _validate_cluster_chat_plan(plan)
    if validated is None:
        retry = await _plan_cluster_chat_tool(
            db,
            project_id,
            cluster_id,
            question,
            history,
            last_tool_state,
            correction="Your previous tool call had invalid arguments. Choose again using exactly the published tool schema.",
        )
        if retry and retry.get("direct_answer"):
            return str(retry["direct_answer"]), [], [], last_tool_state
        validated = _validate_cluster_chat_plan(retry) if retry else None
    if validated is None:
        return "I couldn't safely run that data request. Please clarify which review field or comparison you mean.", [], [], last_tool_state
    plan = validated
    executed = await _execute_cluster_chat_plan(db, project_id, cluster_id, question, plan, sentence=sentence)
    if executed is not None:
        return executed
    return "I could not execute that tool request safely. Please rephrase or clarify the field you want to use.", [], [], last_tool_state


async def _plan_cluster_chat_tool(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    question: str,
    history: list[ClusterChatMessage],
    last_tool_state: dict | None,
    correction: str | None = None,
) -> dict | None:
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        return {"direct_answer": "Current cluster was not found."}
    system = (
        "You are ReviewScope, a qualitative-analysis assistant operating inside one current cluster. "
        "Use a tool whenever the user requests exact reviews, IDs, counts, filters, sorting, ratings, metadata, or cluster facts. "
        "Use count_current_cluster_reviews with no arguments for an unfiltered total review count. Use count_reviews only when the user states a filter. "
        "Use analyze_current_cluster to explain aggregate cluster metrics such as sentiment, average stars, cohesion, size, and mentions. "
        "Do not call list_reviews for a metric explanation unless the user explicitly asks for review examples or evidence. "
        "Use list_reviews_by_rating for an exact star value, list_lowest_rated_reviews for bottom reviews, and list_highest_rated_reviews for top reviews. "
        "Use rank_reviews_by_opinion only for subjective judgments. "
        "Top or bottom scores are numeric: use list_reviews with rating descending or ascending. "
        "Examples: 'five reviews with 5 stars' means list_reviews filtered to rating=5; "
        "'bottom five reviews' means list_reviews sorted by rating ascending; "
        "'what can you tell me about this cluster' means analyze_current_cluster; "
        "after a count, 'show me those' means list_reviews with the previous filters. "
        "All scope is injected by the backend; never request or invent another cluster. "
        "You may answer directly only when no data lookup is needed. Never invent review content, IDs, ratings, counts, or statistics. "
        "For a greeting or social message such as hello or thanks, reply briefly and socially; do not volunteer cluster analysis. "
        f"Current cluster label: {cluster.label}; reviews: {cluster.size}; mentions: {cluster.n_mentions}; "
        f"average stars: {cluster.mean_stars}; sentiment: {cluster.sentiment_avg}; cohesion: {cluster.cohesion}. "
        "Sentiment is P(positive) - P(negative) on a -1 to 1 scale; below -0.2 is negative, -0.2 through 0.2 neutral, and above 0.2 positive. "
        "A fresh explicit request overrides previous tool state; use memory only when the user genuinely refers back to prior results. "
        f"Previous tool state: {json.dumps(last_tool_state, ensure_ascii=False)}"
        + (f" {correction}" if correction else "")
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        *[
            message.model_dump()
            for message in history[-8:]
            if message.content.strip().lower() not in {"none", "respond", "answer_from_context"}
        ],
        {"role": "user", "content": question},
    ]
    settings = get_settings()
    try:
        message = await run_in_threadpool(
            chat_message,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=messages,
            timeout=settings.ollama_timeout_seconds,
            tools=_CLUSTER_CHAT_TOOLS,
        )
    except OllamaChatError:
        return None

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        function = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
        name = str(function.get("name", ""))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _parse_json_object(arguments) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        response_mode = str(arguments.pop("response_mode", "direct_list"))
        return {"tool": name, "args": arguments, "response_mode": response_mode}

    content = message.get("content")
    if isinstance(content, str) and content.strip() and content.strip().lower() != "none":
        if content_plan := _tool_plan_from_message_content(content):
            return content_plan
        if _looks_like_tool_call_content(content):
            return {"tool": "__invalid_tool_call__", "args": {}, "response_mode": "direct_list"}
        return {"direct_answer": content.strip()}
    return None


async def _execute_cluster_chat_plan(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    question: str,
    plan: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult], dict | None] | None:
    tool = str(plan.get("tool", ""))
    args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
    mode = str(plan.get("response_mode", "direct_list"))
    args["original_question"] = question

    if tool in {"get_current_cluster", "analyze_current_cluster"}:
        cluster = await db.get(Cluster, cluster_id)
        if not cluster or cluster.project_id != project_id:
            return "Current cluster was not found.", [], []
        facts = _cluster_summary_answer(cluster, prefix="Current cluster")
        if tool == "analyze_current_cluster" or mode == "analysis":
            scored = await _sentiment_count(db, project_id, cluster_id, sentence=sentence)
            total = cluster.n_mentions if sentence else cluster.size
            unit = "sentence mentions" if sentence else "reviews"
            analysis = await _selected_clusters_analysis(
                question,
                [cluster],
                focus=str(args.get("focus", "general")),
                sentiment_coverage=(scored, total, unit),
            )
            return f"Current cluster {cluster.label}:\n\n{analysis}\n\nCluster facts:\n\n{facts}", [], [], _tool_state(tool, args, [])
        return facts, [], [], _tool_state(tool, args, [])

    if tool == "list_reviews":
        answer, sources, results = await _tool_list_reviews(db, project_id, cluster_id, args, sentence=sentence)
    elif tool == "search_reviews":
        answer, sources, results = await _tool_search_reviews(db, project_id, cluster_id, args, sentence=sentence)
    elif tool == "count_reviews":
        answer, sources, results = await _tool_count_reviews(db, project_id, cluster_id, args, sentence=sentence)
        return answer, sources, results, _tool_state(tool, args, results, summary=_count_summary_from_answer(answer))
    elif tool == "get_review":
        result = await _tool_get_review(db, project_id, cluster_id, args, sentence=sentence)
        if result is None:
            return None
        answer, sources, results = result
    elif tool == "rank_reviews_by_opinion":
        answer, sources, results = await _tool_rank_reviews_by_opinion(db, project_id, cluster_id, args, sentence=sentence)
    else:
        return None

    if results and not await _chat_results_belong_to_cluster(db, project_id, cluster_id, results, sentence=sentence):
        return "No reviews in this cluster matched that query.", [], [], _tool_state(tool, args, [])

    if mode == "analysis" and results:
        docs = await _documents_for_results(db, project_id, cluster_id, results, sentence=sentence)
        rating_col = await _rating_column(db, project_id)
        evidence = _chat_review_text_by_document(docs)
        title = answer.split("\n", 1)[0]
        analysis = await _selected_reviews_analysis(question, docs, rating_col, evidence)
        return f"{title}\n\n{analysis}", sources, results, _tool_state(tool, args, results)
    return answer, sources, results, _tool_state(tool, args, results)


async def _tool_list_reviews(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    args: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult]]:
    docs = await _cluster_documents_for_chat(db, project_id, cluster_id, sentence=sentence)
    evidence = _chat_review_text_by_document(docs)
    rating_col = await _rating_column(db, project_id)
    rows = _apply_tool_filters(docs, args.get("filters"), rating_col, evidence)
    sort_by = _normalize_tool_field(args.get("sort_by")) or "rating"
    sort_dir = "asc" if str(args.get("sort_dir", "desc")).lower() == "asc" else "desc"
    rows = [doc for doc in rows if _tool_value(doc, sort_by, rating_col, evidence[doc.id]) is not None]
    rows.sort(key=lambda doc: _tool_sort_value(doc, sort_by, rating_col, evidence[doc.id]), reverse=sort_dir != "asc")
    limit = _tool_limit(args.get("limit"), default=5, maximum=20)
    picked = rows[:limit]
    if not picked:
        return "No reviews in this cluster matched that query.", [], []
    sources = [_chat_source_from_doc(doc, [], evidence_text=evidence[doc.id], rating=_parse_rating(doc.raw_data, rating_col), matched_field=sort_by, matched_value=str(_tool_value(doc, sort_by, rating_col, evidence[doc.id]))) for doc in picked]
    results = [_chat_result_from_doc(doc, _parse_rating(doc.raw_data, rating_col), evidence[doc.id]) for doc in picked]
    return _review_list_answer("Matching reviews", results), sources, results


async def _tool_search_reviews(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    args: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult]]:
    query = str(args.get("query", "")).strip()
    if len(query) < 3:
        return "What text should I search for in this cluster's reviews?", [], []
    docs = await _cluster_documents_for_chat(db, project_id, cluster_id, sentence=sentence)
    evidence = _chat_review_text_by_document(docs)
    picked = [
        doc for doc in docs
        if query.lower() in evidence[doc.id].lower()
    ][:_tool_limit(args.get("limit"), default=5, maximum=20)]
    if not picked:
        return f"No reviews in this cluster mention `{query}`.", [], []
    rating_col = await _rating_column(db, project_id)
    sources = [_chat_source_from_doc(doc, [query], evidence_text=evidence[doc.id], rating=_parse_rating(doc.raw_data, rating_col)) for doc in picked]
    results = [_chat_result_from_doc(doc, _parse_rating(doc.raw_data, rating_col), evidence[doc.id]) for doc in picked]
    return _review_list_answer(f"Reviews mentioning `{query}`", results), sources, results


async def _tool_count_reviews(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    args: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult]]:
    if not args.get("filters"):
        count_query = select(func.count()).select_from(Document).where(
            Document.project_id == project_id,
            _chat_cluster_membership_clause(project_id, cluster_id, sentence=sentence),
        )
        count = int((await db.execute(count_query)).scalar_one())
        return f"This cluster contains {count} reviews.", [], []
    docs = await _cluster_documents_for_chat(db, project_id, cluster_id, sentence=sentence)
    evidence = _chat_review_text_by_document(docs)
    rating_col = await _rating_column(db, project_id)
    rows = _apply_tool_filters(docs, args.get("filters"), rating_col, evidence)
    return f"Found {len(rows)} reviews in this cluster matching that query.", [], []


async def _tool_get_review(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    args: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult]] | None:
    review_id = str(args.get("review_id", "")).strip()
    if not review_id:
        return None
    doc = await _find_cluster_review_by_id(db, project_id, cluster_id, review_id, sentence=sentence)
    if doc:
        rating_col = await _rating_column(db, project_id)
        rating = _parse_rating(doc.raw_data, rating_col)
        evidence = _chat_review_text_by_document([doc])
        source = _chat_source_from_doc(doc, [review_id], evidence_text=evidence[doc.id], include_full=True, rating=rating)
        result = _chat_result_from_doc(doc, rating, evidence[doc.id])
        return _review_list_answer("Matched review", [result]), [source], [result]

    doc = await _find_project_review_by_id(db, project_id, review_id)
    if not doc:
        return "No review with that ID was found in this project.", [], []
    membership = "does not have a mention in" if sentence else "is not assigned to"
    return f"That review exists, but it {membership} the currently open cluster.", [], []


async def _find_cluster_review_by_id(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    review_id: str,
    *,
    sentence: bool,
) -> Document | None:
    conditions = _review_id_conditions(review_id)
    query = select(Document).where(
        Document.project_id == project_id,
        _chat_cluster_membership_clause(project_id, cluster_id, sentence=sentence),
        or_(*conditions),
    )
    return (await db.execute(query.limit(1))).scalar_one_or_none()


async def _find_project_review_by_id(db: AsyncSession, project_id: uuid.UUID, review_id: str) -> Document | None:
    conditions = _review_id_conditions(review_id)
    return (await db.execute(
        select(Document).where(Document.project_id == project_id, or_(*conditions)).limit(1)
    )).scalar_one_or_none()


def _review_id_conditions(review_id: str) -> list:
    conditions = [Document.primary_key_value == review_id]
    for key in ("review_id", "id"):
        conditions.append(Document.raw_data[key].astext == review_id)
    try:
        conditions.append(Document.id == uuid.UUID(review_id))
    except ValueError:
        pass
    return conditions


async def _tool_rank_reviews_by_opinion(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    args: dict,
    *,
    sentence: bool,
) -> tuple[str, list[ClusterChatSource], list[ClusterChatResult]]:
    docs = await _cluster_documents_for_chat(db, project_id, cluster_id, sentence=sentence)
    rating_col = await _rating_column(db, project_id)
    evidence = _chat_review_text_by_document(docs)
    candidates = _apply_tool_filters(docs, args.get("filters"), rating_col, evidence)[:80]
    if not candidates:
        return "No reviews in this cluster matched the requested criteria.", [], []
    goal = str(args.get("goal") or "best matching reviews")[:200]
    limit = _tool_limit(args.get("limit"), default=5, maximum=10)
    ranked = await _rank_candidate_docs(goal, candidates, rating_col, limit, evidence)
    if not ranked:
        reverse = not re.search(r"\b(worst|least|negative|concerning|problematic|poor)\b", goal, re.IGNORECASE)
        ranked = sorted(candidates, key=lambda doc: _review_quality_score(doc, rating_col), reverse=reverse)[:limit]
    sources = [_chat_source_from_doc(doc, [], evidence_text=evidence[doc.id], rating=_parse_rating(doc.raw_data, rating_col)) for doc in ranked]
    results = [_chat_result_from_doc(doc, _parse_rating(doc.raw_data, rating_col), evidence[doc.id]) for doc in ranked]
    title = f"Reviews selected for: {goal}"
    return _review_list_answer(title, results), sources, results


def _normalize_chat_question(message: str) -> str:
    return re.sub(r"[=+*/\\|<>\s]+$", "", message.strip())


def _social_chat_answer(question: str) -> str | None:
    normalized = re.sub(r"[^a-z\s]", "", question.lower())
    normalized = " ".join(normalized.split())
    if normalized in {"hello", "hi", "hey", "hello there", "hi there", "good morning", "good afternoon", "good evening"}:
        return "Hello! What would you like to know about this cluster?"
    if normalized in {"thanks", "thank you", "thanks a lot", "thank you very much"}:
        return "You're welcome!"
    return None


def _tool_state(tool: str, args: dict, results: list[ClusterChatResult], *, summary: dict | None = None) -> dict:
    public_args = {key: value for key, value in args.items() if key != "original_question"}
    return {
        "tool": tool,
        "scope": {"type": "current_cluster"},
        "args": public_args,
        "result_ids": [result.primary_key_value for result in results],
        "result_summary": summary or {"count": len(results)},
    }


def _count_summary_from_answer(answer: str) -> dict:
    match = re.search(r"(?:Found|contains)\s+(\d+)", answer, re.IGNORECASE)
    return {"count": int(match.group(1))} if match else {}


def _review_quality_score(doc: Document, rating_col: str | None) -> float:
    rating = _parse_rating(doc.raw_data, rating_col)
    score = (rating or 0) * 10
    score += (_raw_number(doc.raw_data, "useful") or 0) * 0.8
    score += (_raw_number(doc.raw_data, "funny") or 0) * 0.4
    score += (_raw_number(doc.raw_data, "cool") or 0) * 0.3
    if doc.sentiment_score is not None:
        score += doc.sentiment_score * 2
    score += min(len(doc.text), 1200) / 1200
    return score


async def _selected_reviews_analysis(question: str, docs: list[Document], rating_col: str | None, evidence: dict[uuid.UUID, str]) -> str:
    snippets = []
    for doc in docs:
        rating = _parse_rating(doc.raw_data, rating_col)
        snippets.append(f"ID: {doc.primary_key_value}\nRating: {rating if rating is not None else 'n/a'}\nText: {_truncate(evidence[doc.id], 700)}")
    prompt = (
        "Analyze only the selected real reviews below. Do not add new review IDs or claims not supported by the text. "
        "Give 2-4 concise bullets explaining the shared pattern, tone, and why these reviews stand out.\n\n"
        f"User question: {question}\n\nSelected reviews:\n" + "\n\n".join(snippets)
    )
    settings = get_settings()
    try:
        return await run_in_threadpool(
            chat_completion,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.ollama_timeout_seconds,
            response_format=_RANKED_REVIEW_IDS_SCHEMA,
        )
    except OllamaChatError:
        ratings = [_parse_rating(doc.raw_data, rating_col) for doc in docs]
        known = [rating for rating in ratings if rating is not None]
        avg = sum(known) / len(known) if known else None
        descriptor = f" Their average rating is {avg:.1f} stars." if avg is not None else ""
        return f"These reviews stand out based on the selected ranking criteria and their review text.{descriptor} Read the cards for the concrete evidence."


def _cluster_summary_answer(cluster: Cluster, *, prefix: str, compact: bool = False) -> str:
    stats = [
        f"id: {cluster.id}",
        f"label: {cluster.label}",
        f"reviews: {cluster.size}",
    ]
    if cluster.n_mentions and cluster.n_mentions != cluster.size:
        stats.append(f"mentions: {cluster.n_mentions}")
    if cluster.mean_stars is not None:
        stats.append(f"avg stars: {cluster.mean_stars:.2f}")
    if cluster.sentiment_avg is not None:
        stats.append(f"sentiment: {cluster.sentiment_avg:.3f}")
    if cluster.cohesion is not None:
        stats.append(f"cohesion: {cluster.cohesion:.3f}")
    if compact:
        return f"{prefix} {cluster.label}\n" + "; ".join(stats)
    summary = f"\nSummary: {cluster.summary}" if cluster.summary else ""
    return f"{prefix} {cluster.label}\n" + "\n".join(stats) + summary


async def _selected_clusters_analysis(
    question: str,
    clusters: list[Cluster],
    *,
    focus: str = "general",
    sentiment_coverage: tuple[int, int, str] | None = None,
) -> str:
    rows = []
    for cluster in clusters:
        terms = ", ".join(str(term.get("term", "")) for term in cluster.top_terms[:8] if term.get("term"))
        rows.append(
            "\n".join(
                [
                    f"ID: {cluster.id}",
                    f"Label: {cluster.label}",
                    f"Reviews: {cluster.size}",
                    f"Mentions: {cluster.n_mentions}",
                    f"Average stars: {cluster.mean_stars:.2f}" if cluster.mean_stars is not None else "Average stars: n/a",
                    f"Sentiment: {cluster.sentiment_avg:.3f}" if cluster.sentiment_avg is not None else "Sentiment: n/a",
                    f"Cohesion: {cluster.cohesion:.3f}" if cluster.cohesion is not None else "Cohesion: n/a",
                    f"Top terms: {terms}" if terms else "Top terms: n/a",
                    f"Summary: {cluster.summary}" if cluster.summary else "Summary: n/a",
                ]
            )
        )
    methodology = ""
    if focus == "sentiment" or re.search(r"\bsentiment\b", question, re.IGNORECASE):
        coverage = ""
        if sentiment_coverage:
            scored, total, unit = sentiment_coverage
            coverage = f" Sentiment coverage is {scored} of {total} {unit}."
        methodology = (
            " Sentiment methodology: each text score is P(positive) - P(negative), ranging from -1 to 1. "
            "Scores below -0.2 are negative, -0.2 through 0.2 are neutral, and above 0.2 are positive. "
            "The cluster sentiment is the mean of available member scores. In sentence-level projects, members are sentence mentions, not whole reviews."
            + coverage
        )
    prompt = (
        "Analyze only the retrieved cluster facts below. Do not invent clusters, counts, ratings, sentiment, or examples. "
        "Give concise takeaways that directly answer the user's question. Explain the requested metric before interpreting its value. "
        "Mention tradeoffs when size, stars, sentiment, and cohesion disagree."
        f"{methodology}\n\n"
        f"User question: {question}\n\nRetrieved clusters:\n" + "\n\n".join(rows)
    )
    settings = get_settings()
    try:
        return await run_in_threadpool(
            chat_completion,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.ollama_timeout_seconds,
        )
    except OllamaChatError:
        if not clusters:
            return "No clusters were available to analyze."
        if focus == "sentiment" or re.search(r"\bsentiment\b", question, re.IGNORECASE):
            cluster = clusters[0]
            value = f"{cluster.sentiment_avg:.3f}" if cluster.sentiment_avg is not None else "unavailable"
            coverage = ""
            if sentiment_coverage:
                scored, total, unit = sentiment_coverage
                coverage = f" It covers {scored} of {total} {unit}."
            return (
                f"The cluster sentiment score is {value}. It is the mean of P(positive) - P(negative) scores, on a -1 to 1 scale. "
                "Values below -0.2 are negative, values from -0.2 to 0.2 are neutral, and values above 0.2 are positive."
                f"{coverage} It is model-derived and is not the same as the average star rating."
            )
        largest = max(clusters, key=lambda cluster: cluster.size)
        lowest = min((cluster for cluster in clusters if cluster.mean_stars is not None), key=lambda cluster: cluster.mean_stars, default=None)
        highest = max((cluster for cluster in clusters if cluster.mean_stars is not None), key=lambda cluster: cluster.mean_stars, default=None)
        lines = [f"The largest retrieved cluster is `{largest.label}` with {largest.size} reviews."]
        if lowest and highest and lowest.id != highest.id:
            lines.append(f"Ratings range from `{lowest.label}` at {lowest.mean_stars:.2f} stars to `{highest.label}` at {highest.mean_stars:.2f} stars.")
        lines.append("Use the cluster facts below for the exact metrics.")
        return " ".join(lines)


async def _answer_from_current_cluster_context(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    question: str,
) -> str:
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        return "Current cluster was not found."
    docs = list((await db.execute(
        select(Document)
        .where(Document.project_id == project_id, Document.cluster_id == cluster_id)
        .limit(8)
    )).scalars().all())
    rating_col = await _rating_column(db, project_id)
    evidence = []
    for doc in docs:
        rating = _parse_rating(doc.raw_data, rating_col)
        evidence.append(
            f"Review {doc.primary_key_value}; rating: {rating if rating is not None else 'n/a'}; text: {_truncate(doc.text, 600)}"
        )
    prompt = (
        "Answer the user's question about the currently open cluster. Use only the supplied cluster facts and review evidence. "
        "Do not mention routing, tools, or planner tokens. Never answer with only 'none'. If evidence is insufficient, say what cannot be concluded.\n\n"
        f"User question: {question}\n\n"
        f"Cluster facts:\n{_cluster_summary_answer(cluster, prefix='Current cluster')}\n\n"
        f"Primary-cluster review evidence:\n" + ("\n".join(evidence) if evidence else "No review samples available.")
    )
    settings = get_settings()
    try:
        return await run_in_threadpool(
            chat_completion,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.ollama_timeout_seconds,
        )
    except OllamaChatError:
        return "I couldn't generate a response because the chat model is currently unavailable."


def _raw_number(raw_data: dict, field: str) -> float | None:
    value = raw_data.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _cluster_documents_for_chat(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    *,
    sentence: bool,
) -> list[Document]:
    query = select(Document).where(
        Document.project_id == project_id,
        _chat_cluster_membership_clause(project_id, cluster_id, sentence=sentence),
    )
    return list((await db.execute(query)).scalars().all())


def _chat_cluster_membership_clause(
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    *,
    sentence: bool,
):
    if sentence:
        member_ids = select(Segment.document_id).where(
            Segment.project_id == project_id,
            Segment.cluster_id == cluster_id,
        )
        return Document.id.in_(member_ids)
    return Document.cluster_id == cluster_id


def _chat_review_text_by_document(docs: list[Document]) -> dict[uuid.UUID, str]:
    return {doc.id: doc.text for doc in docs}


async def _chat_results_belong_to_cluster(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    results: list[ClusterChatResult],
    *,
    sentence: bool,
) -> bool:
    result_ids = {result.document_id for result in results}
    if not result_ids:
        return True
    matched_ids = set((await db.execute(
        select(Document.id).where(
            Document.project_id == project_id,
            Document.id.in_(result_ids),
            _chat_cluster_membership_clause(project_id, cluster_id, sentence=sentence),
        )
    )).scalars().all())
    return matched_ids == result_ids


async def _documents_for_results(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    results: list[ClusterChatResult],
    *,
    sentence: bool,
) -> list[Document]:
    result_ids = [result.document_id for result in results]
    docs = list((await db.execute(
        select(Document).where(
            Document.project_id == project_id,
            Document.id.in_(result_ids),
            _chat_cluster_membership_clause(project_id, cluster_id, sentence=sentence),
        )
    )).scalars().all())
    by_id = {doc.id: doc for doc in docs}
    return [by_id[result.document_id] for result in results if result.document_id in by_id]


def _chat_result_from_doc(doc: Document, rating: float | None, evidence_text: str | None = None) -> ClusterChatResult:
    return ClusterChatResult(
        document_id=doc.id,
        primary_key_value=doc.primary_key_value,
        rating=rating,
        snippet=_truncate(evidence_text if evidence_text is not None else doc.text, 520),
    )


def _review_list_answer(title: str, results: list[ClusterChatResult]) -> str:
    lines = [f"{title} ({len(results)}):"]
    for index, result in enumerate(results, start=1):
        rating = f"Rating: {result.rating:g}" if result.rating is not None else "Rating: n/a"
        lines.append(f"{index}. Review {result.primary_key_value}\n{rating}\nText: {result.snippet}")
    return "\n\n".join(lines)


def _parse_ranked_ids(raw: str, allowed: set[str]) -> list[str]:
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None
    values = payload.get("ids") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        values = re.findall(r"[A-Za-z0-9_-]{3,}", raw)
    ranked = []
    seen = set()
    for value in values:
        primary_key = str(value)
        if primary_key in allowed and primary_key not in seen:
            ranked.append(primary_key)
            seen.add(primary_key)
    return ranked


def _parse_json_object(raw: str) -> dict | None:
    candidates = [raw.strip()]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match and match.group(0) not in candidates:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        for _ in range(2):
            if not isinstance(payload, str):
                break
            try:
                payload = json.loads(payload)
            except ValueError:
                break
        if isinstance(payload, dict):
            return payload
    return None


def _tool_plan_from_message_content(content: str) -> dict | None:
    payload = _parse_json_object(content)
    if not payload:
        return None
    function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
    name = function.get("name") or payload.get("tool")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = function.get("arguments", function.get("parameters", payload.get("args", {})))
    if isinstance(arguments, str):
        arguments = _parse_json_object(arguments) or {}
    if not isinstance(arguments, dict):
        arguments = {}
    arguments = dict(arguments)
    response_mode = str(arguments.pop("response_mode", payload.get("response_mode", "direct_list")))
    return {"tool": name.strip(), "args": arguments, "response_mode": response_mode}


def _looks_like_tool_call_content(content: str) -> bool:
    has_protocol_key = re.search(r"[\"']?(?:name|tool|function|parameters|arguments)[\"']?\s*:", content, re.IGNORECASE)
    tool_names = (
        "get_current_cluster",
        "analyze_current_cluster",
        "list_reviews",
        "count_current_cluster_reviews",
        "count_reviews",
        "search_reviews",
        "get_review",
        "rank_reviews_by_opinion",
    )
    return bool(has_protocol_key and any(name in content for name in tool_names))


def _validate_cluster_chat_plan(plan: dict) -> dict | None:
    tool = str(plan.get("tool", ""))
    unfiltered_count = tool in {"count_current_cluster_reviews", "count_cluster_reviews", "get_cluster_size"}
    tool = {
        "get_reviews": "list_reviews",
        "filter_reviews": "list_reviews",
        "get_reviews_by_rating": "list_reviews_by_rating",
        "reviews_by_rating": "list_reviews_by_rating",
        "bottom_reviews": "list_lowest_rated_reviews",
        "top_reviews": "list_highest_rated_reviews",
        "count_current_cluster_reviews": "count_reviews",
        "count_cluster_reviews": "count_reviews",
        "get_cluster_size": "count_reviews",
    }.get(tool, tool)
    allowed_tools = {
        "get_current_cluster",
        "analyze_current_cluster",
        "list_reviews",
        "list_reviews_by_rating",
        "list_lowest_rated_reviews",
        "list_highest_rated_reviews",
        "count_reviews",
        "search_reviews",
        "get_review",
        "rank_reviews_by_opinion",
    }
    if tool not in allowed_tools:
        return None
    mode = str(plan.get("response_mode", "direct_list"))
    if mode not in {"direct_list", "analysis", "count", "facts"}:
        mode = "direct_list"
    raw_args = plan.get("args")
    args = dict(raw_args) if isinstance(raw_args, dict) else {}
    for key in ("cluster_id", "project_id", "scope"):
        args.pop(key, None)
    if unfiltered_count:
        args = {"filters": []}

    if tool == "list_reviews_by_rating":
        rating = args.get("rating", args.get("stars", args.get("score")))
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return None
        tool = "list_reviews"
        args = {
            "limit": args.get("limit", 5),
            "sort_by": "rating",
            "sort_dir": "desc",
            "filters": [{"field": "rating", "op": "eq", "value": rating}],
        }
    elif tool in {"list_lowest_rated_reviews", "list_highest_rated_reviews"}:
        direction = "asc" if tool == "list_lowest_rated_reviews" else "desc"
        tool = "list_reviews"
        args = {"limit": args.get("limit", 5), "sort_by": "rating", "sort_dir": direction, "filters": []}
    elif tool == "rank_reviews_by_opinion":
        goal = str(args.get("goal", "")).lower()
        if re.search(r"\b(bottom|lowest)\b", goal):
            tool = "list_reviews"
            args = {"limit": args.get("limit", 5), "sort_by": "rating", "sort_dir": "asc", "filters": args.get("filters", [])}
        elif re.search(r"\b(top|highest)\b", goal):
            tool = "list_reviews"
            args = {"limit": args.get("limit", 5), "sort_by": "rating", "sort_dir": "desc", "filters": args.get("filters", [])}

    if tool == "get_current_cluster":
        args = {}
    elif tool == "analyze_current_cluster":
        focus = str(args.get("focus", "general")).lower()
        args = {"focus": focus if focus in {"general", "sentiment", "rating", "cohesion", "size", "mentions"} else "general"}
    elif tool in {"list_reviews", "count_reviews", "rank_reviews_by_opinion"}:
        raw_filters = args.get("filters", args.get("filter"))
        if raw_filters is None:
            rating = args.get("rating", args.get("stars", args.get("score")))
            if rating is not None:
                raw_filters = {"field": "rating", "op": "eq", "value": rating}
            elif "field" in args and "value" in args:
                raw_filters = {"field": args["field"], "op": args.get("op", "eq"), "value": args["value"]}
        filters = _validated_tool_filters(raw_filters)
        if filters is None:
            return None
        args["filters"] = filters
        if tool != "count_reviews":
            args["limit"] = _tool_limit(args.get("limit"), default=5, maximum=20 if tool == "list_reviews" else 10)
        if tool == "list_reviews":
            sort_by = _normalize_tool_field(args.get("sort_by")) or "rating"
            if sort_by == "text":
                return None
            args["sort_by"] = sort_by
            args["sort_dir"] = "asc" if str(args.get("sort_dir", "desc")).lower() == "asc" else "desc"
        elif tool == "rank_reviews_by_opinion":
            goal = str(args.get("goal", "")).strip()
            if not goal:
                return None
            args["goal"] = goal[:200]
    elif tool == "search_reviews":
        query = str(args.get("query", "")).strip()
        if len(query) < 2:
            return None
        args = {"query": query[:200], "limit": _tool_limit(args.get("limit"), default=5, maximum=20)}
    elif tool == "get_review":
        review_id = str(args.get("review_id", "")).strip()
        if not review_id:
            return None
        args = {"review_id": review_id}
    return {"tool": tool, "args": args, "response_mode": mode}


def _validated_tool_filters(filters) -> list[dict] | None:
    if filters is None or filters == {} or filters == []:
        return []
    if isinstance(filters, dict):
        if "field" not in filters and len(filters) == 1:
            field, value = next(iter(filters.items()))
            filters = [{"field": field, "op": "eq", "value": value}]
        else:
            filters = [filters]
    if not isinstance(filters, list) or len(filters) > 10:
        return None
    validated = []
    for item in filters:
        if not isinstance(item, dict):
            return None
        if not item:
            continue
        if item.get("value") is None and "value" in item:
            continue
        if item.get("field") is None and item.get("value") is None:
            continue
        raw_field = str(item.get("field", "")).lower()
        if raw_field in {"cluster", "cluster_id", "current_cluster", "project", "project_id", "scope"}:
            continue
        field = _normalize_tool_field(item.get("field"))
        op = str(item.get("op", "eq")).lower()
        if field is None or op not in {"eq", "gt", "gte", "lt", "lte", "contains"} or "value" not in item:
            return None
        if op == "contains" and field != "text":
            return None
        validated.append({"field": field, "op": op, "value": item["value"]})
    return validated


def _tool_limit(value, *, default: int, maximum: int) -> int:
    try:
        return max(1, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _normalize_tool_field(value) -> str | None:
    field = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "score": "rating",
        "stars": "rating",
        "star": "rating",
        "rated": "rating",
        "rating": "rating",
        "sentiment_score": "sentiment",
        "sentiment": "sentiment",
        "useful_votes": "useful",
        "useful": "useful",
        "funny_votes": "funny",
        "funniest": "funny",
        "funny": "funny",
        "cool_votes": "cool",
        "coolest": "cool",
        "cool": "cool",
        "created_at": "date",
        "date": "date",
        "time": "date",
        "text_length": "length",
        "length": "length",
        "longest": "length",
        "shortest": "length",
        "text": "text",
    }
    return aliases.get(field)


def _tool_value(doc: Document, field: str, rating_col: str | None, evidence_text: str | None = None):
    if field == "rating":
        return _parse_rating(doc.raw_data, rating_col)
    if field == "sentiment":
        return doc.sentiment_score
    if field in {"useful", "funny", "cool"}:
        return _raw_number(doc.raw_data, field)
    if field == "date":
        value = doc.raw_data.get("date")
        return str(value) if value is not None else None
    if field == "length":
        return len(evidence_text if evidence_text is not None else doc.text)
    if field == "text":
        return evidence_text if evidence_text is not None else doc.text
    return None


def _tool_sort_value(doc: Document, field: str, rating_col: str | None, evidence_text: str | None = None):
    value = _tool_value(doc, field, rating_col, evidence_text)
    if isinstance(value, str):
        return value.lower()
    return value if value is not None else -1


def _apply_tool_filters(
    docs: list[Document],
    filters,
    rating_col: str | None,
    evidence: dict[uuid.UUID, str] | None = None,
) -> list[Document]:
    if not isinstance(filters, list):
        return docs
    rows = docs
    for raw_filter in filters:
        if not isinstance(raw_filter, dict):
            continue
        field = _normalize_tool_field(raw_filter.get("field"))
        op = str(raw_filter.get("op", "eq")).lower()
        expected = raw_filter.get("value")
        if field is None:
            continue
        rows = [
            doc for doc in rows
            if _tool_filter_matches(_tool_value(doc, field, rating_col, evidence.get(doc.id) if evidence else None), op, expected)
        ]
    return rows


def _tool_filter_matches(actual, op: str, expected) -> bool:
    if actual is None:
        return False
    if op == "contains":
        return str(expected).lower() in str(actual).lower()
    if isinstance(actual, (int, float)):
        try:
            expected_num = float(expected)
        except (TypeError, ValueError):
            return False
        if op == "eq":
            return float(actual) == expected_num
        if op == "gt":
            return float(actual) > expected_num
        if op == "gte":
            return float(actual) >= expected_num
        if op == "lt":
            return float(actual) < expected_num
        if op == "lte":
            return float(actual) <= expected_num
    if op == "eq":
        return str(actual).lower() == str(expected).lower()
    if op in {"gt", "gte", "lt", "lte"}:
        actual_text = str(actual)
        expected_text = str(expected)
        if op == "gt":
            return actual_text > expected_text
        if op == "gte":
            return actual_text >= expected_text
        if op == "lt":
            return actual_text < expected_text
        if op == "lte":
            return actual_text <= expected_text
    return False


async def _rank_candidate_docs(
    goal: str,
    candidates: list[Document],
    rating_col: str | None,
    limit: int,
    evidence: dict[uuid.UUID, str],
) -> list[Document]:
    by_id = {doc.primary_key_value: doc for doc in candidates}
    candidate_text = []
    for doc in candidates:
        rating = _parse_rating(doc.raw_data, rating_col)
        extras = []
        for field in ("useful", "funny", "cool", "date"):
            if field in doc.raw_data:
                extras.append(f"{field}: {doc.raw_data[field]}")
        candidate_text.append(
            f"ID: {doc.primary_key_value}\nRating: {rating if rating is not None else 'n/a'}\n{'; '.join(extras)}\nText: {_truncate(evidence[doc.id], 700)}"
        )
    settings = get_settings()
    prompt = (
        "Rank review IDs for the requested goal using only the candidate IDs. "
        "Return strict JSON only: {\"ids\":[\"id1\",\"id2\"]}. Do not invent IDs.\n"
        f"Goal: {goal}\nLimit: {limit}\n\nCandidates:\n" + "\n\n".join(candidate_text)
    )
    try:
        raw = await run_in_threadpool(
            chat_completion,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.ollama_timeout_seconds,
        )
    except OllamaChatError:
        return []
    ranked_ids = _parse_ranked_ids(raw, set(by_id))[:limit]
    return [by_id[primary_key] for primary_key in ranked_ids]


def _chat_source_from_doc(
    doc: Document,
    terms: list[str],
    *,
    evidence_text: str | None = None,
    include_full: bool = False,
    rating: float | None = None,
    matched_field: str | None = None,
    matched_value: str | None = None,
) -> ClusterChatSource:
    text = evidence_text if evidence_text is not None else doc.text
    for term in terms:
        if matched_field is not None:
            break
        if doc.primary_key_value == term:
            matched_field = "primary_key_value"
            matched_value = doc.primary_key_value
            break
        found = _find_raw_match(doc.raw_data, term)
        if found:
            matched_field, matched_value = found
            break
        if term.lower() in text.lower():
            matched_field = "text"
            matched_value = term
            break
    return ClusterChatSource(
        document_id=doc.id,
        primary_key_value=doc.primary_key_value,
        matched_field=matched_field,
        matched_value=matched_value,
        rating=rating,
        snippet=_truncate(text, 360),
        full_text=text if include_full else None,
        raw_data=dict(doc.raw_data),
    )


def _find_raw_match(value, term: str, path: str = "raw_data") -> tuple[str, str] | None:
    if isinstance(value, dict):
        for key, child in value.items():
            found = _find_raw_match(child, term, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_raw_match(child, term, f"{path}[{index}]")
            if found:
                return found
    elif value is not None and term.lower() in str(value).lower():
        return path, str(value)
    return None


def _truncate(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


async def _sentiment_count(db: AsyncSession, project_id: uuid.UUID, cluster_id: uuid.UUID, *, sentence: bool = False) -> int:
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
