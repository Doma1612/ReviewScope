import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_project_role
from app.core.config import get_settings
from app.db.session import get_db
from app.models import Cluster, ClusterEdit, Document, Embedding, PipelineJob, Project, ProjectMember, ProjectRole, ProjectSchema, ProjectStatus, User
from app.schemas import (
    BulkReassign,
    BulkReassignResult,
    ClusterEditRead,
    ClusterRead,
    DocumentReassign,
    DocumentRead,
    EmbeddingPoint,
    MemberCreate,
    MemberRead,
    MemberUpdate,
    PipelineStatusRead,
    ProjectRead,
    ProjectUpdate,
)
from app.ml_pipeline import run_ml_pipeline
from app.services.edits import record_edit
from app.services.recompute import recompute_clusters
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


@router.get("/{project_id}/pipeline/status", response_model=PipelineStatusRead)
async def pipeline_status(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> PipelineStatusRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    project = await _get_project_or_404(db, project_id)
    jobs = (await db.execute(select(PipelineJob).where(PipelineJob.project_id == project_id).order_by(PipelineJob.id))).scalars().all()
    return PipelineStatusRead(project_id=project_id, status=project.status, jobs=list(jobs))


@router.get("/{project_id}/embeddings", response_model=list[EmbeddingPoint])
async def embeddings(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[EmbeddingPoint]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    result = await db.execute(
        select(Document.id, Document.cluster_id, Embedding.umap_x, Embedding.umap_y, Embedding.umap_z)
        .join(Embedding, Embedding.document_id == Document.id)
        .where(Document.project_id == project_id)
    )
    return [EmbeddingPoint(document_id=doc_id, cluster_id=cluster_id, x=x, y=y, z=z) for doc_id, cluster_id, x, y, z in result.all()]


@router.get("/{project_id}/clusters", response_model=list[ClusterRead])
async def clusters(project_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[ClusterRead]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    result = await db.execute(select(Cluster).where(Cluster.project_id == project_id).order_by(Cluster.label))
    items = []
    for cluster in result.scalars().all():
        sample_docs = await _sample_docs(db, project_id, cluster.id)
        items.append(ClusterRead.model_validate(cluster).model_copy(update={"sample_docs": sample_docs}))
    return items


@router.get("/{project_id}/clusters/{cluster_id}", response_model=ClusterRead)
async def cluster_detail(project_id: uuid.UUID, cluster_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> ClusterRead:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    cluster = await db.get(Cluster, cluster_id)
    if not cluster or cluster.project_id != project_id:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return ClusterRead.model_validate(cluster).model_copy(update={"sample_docs": await _sample_docs(db, project_id, cluster_id, 5)})


@router.get("/{project_id}/clusters/{cluster_id}/documents", response_model=list[DocumentRead])
async def cluster_documents(project_id: uuid.UUID, cluster_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int = 50, offset: int = 0) -> list[Document]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    result = await db.execute(select(Document).where(Document.project_id == project_id, Document.cluster_id == cluster_id).limit(limit).offset(offset))
    return list(result.scalars().all())


@router.get("/{project_id}/documents", response_model=list[DocumentRead])
async def documents(project_id: uuid.UUID, cluster_id: uuid.UUID | None = None, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user), limit: int = 50, offset: int = 0) -> list[Document]:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    query = select(Document).where(Document.project_id == project_id).limit(limit).offset(offset)
    if cluster_id:
        query = query.where(Document.cluster_id == cluster_id)
    result = await db.execute(query)
    return list(result.scalars().all())


@router.get("/{project_id}/documents/{document_id}", response_model=DocumentRead)
async def document(project_id: uuid.UUID, document_id: uuid.UUID, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> Document:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner, ProjectRole.viewer})
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/{project_id}/documents/{document_id}", response_model=DocumentRead)
async def reassign_document(project_id: uuid.UUID, document_id: uuid.UUID, payload: DocumentReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> Document:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="Document not found")
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    old_cluster_id = doc.cluster_id
    doc.cluster_id = target_cluster_id
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="reassign_doc",
        document_id=document_id,
        cluster_id=old_cluster_id,
        target_cluster_id=target_cluster_id,
    )
    affected = [cid for cid in {old_cluster_id, target_cluster_id} if cid is not None]
    await recompute_clusters(db, project_id, affected)
    await db.commit()
    await db.refresh(doc)
    return doc


@router.post("/{project_id}/documents/reassign", response_model=BulkReassignResult)
async def bulk_reassign_documents(project_id: uuid.UUID, payload: BulkReassign, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)) -> BulkReassignResult:
    await require_project_role(db, project_id, current_user.id, {ProjectRole.owner})
    target_cluster_id = payload.cluster_id
    if target_cluster_id is not None:
        cluster = await db.get(Cluster, target_cluster_id)
        if not cluster or cluster.project_id != project_id:
            raise HTTPException(status_code=404, detail="Cluster not found")
    result = await db.execute(
        select(Document).where(Document.project_id == project_id, Document.id.in_(payload.document_ids))
    )
    docs = list(result.scalars().all())
    affected: set[uuid.UUID | None] = {target_cluster_id}
    for doc in docs:
        affected.add(doc.cluster_id)
        doc.cluster_id = target_cluster_id
    record_edit(
        db,
        project_id=project_id,
        actor_id=current_user.id,
        action="bulk_reassign",
        target_cluster_id=target_cluster_id,
        payload={"document_ids": [str(doc.id) for doc in docs]},
    )
    await recompute_clusters(db, project_id, [cid for cid in affected if cid is not None])
    await db.commit()
    return BulkReassignResult(moved=len(docs))


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
    )


async def _sample_docs(db: AsyncSession, project_id: uuid.UUID, cluster_id: uuid.UUID, limit: int = 3) -> list[dict]:
    result = await db.execute(
        select(Document.id, Document.text).where(Document.project_id == project_id, Document.cluster_id == cluster_id).limit(limit)
    )
    return [{"id": str(doc_id), "text": text[:240]} for doc_id, text in result.all()]
