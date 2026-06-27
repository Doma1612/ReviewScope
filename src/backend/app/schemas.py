import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr

from app.models import PipelineStepStatus, ProjectRole, ProjectStatus


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    id: uuid.UUID
    email: EmailStr
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProjectRead(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    owner_email: str | None = None
    status: ProjectStatus
    doc_count: int
    created_at: datetime
    role: ProjectRole
    last_error: str | None = None


class ProjectUpdate(BaseModel):
    name: str


class PipelineJobRead(BaseModel):
    step: str
    status: PipelineStepStatus
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


class PipelineStatusRead(BaseModel):
    project_id: uuid.UUID
    status: ProjectStatus
    jobs: list[PipelineJobRead]


class EmbeddingPoint(BaseModel):
    document_id: uuid.UUID
    cluster_id: uuid.UUID | None
    x: float
    y: float
    z: float | None


class ClusterRead(BaseModel):
    id: uuid.UUID
    label: str
    summary: str
    top_terms: list[dict[str, Any]]
    word_frequencies: dict[str, Any]
    size: int
    sentiment_avg: float | None = None
    sample_docs: list[dict[str, Any]] = []

    model_config = {"from_attributes": True}


class DocumentRead(BaseModel):
    id: uuid.UUID
    primary_key_value: str
    text: str
    raw_data: dict[str, Any]
    cluster_id: uuid.UUID | None
    sentiment_score: float | None = None

    model_config = {"from_attributes": True}


class MemberCreate(BaseModel):
    email: EmailStr
    role: ProjectRole = ProjectRole.viewer


class MemberUpdate(BaseModel):
    role: ProjectRole


class MemberRead(BaseModel):
    user_id: uuid.UUID
    email: str
    role: ProjectRole
