import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

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
    unit: str = "document"
    retrieval_ready: bool = False


class ProjectUpdate(BaseModel):
    name: str


ALLOWED_COLUMN_TYPES = ("text", "integer", "float", "date", "boolean")


class SchemaColumn(BaseModel):
    name: str
    type: str
    is_primary_key: bool = False

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in ALLOWED_COLUMN_TYPES:
            raise ValueError(f"type must be one of {', '.join(ALLOWED_COLUMN_TYPES)}")
        return value


class ProjectSchemaRead(BaseModel):
    # Read legacy stored schemas without imposing current write-side validation.
    columns: list[dict[str, Any]]


class ProjectSchemaWrite(BaseModel):
    columns: list[SchemaColumn]

    @model_validator(mode="after")
    def _exactly_one_primary_key(self) -> "ProjectSchemaWrite":
        pk_count = sum(1 for c in self.columns if c.is_primary_key)
        if pk_count != 1:
            raise ValueError("exactly one column must have is_primary_key=true")
        return self


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
    # Sentence points identify a segment; document_id identifies its parent review.
    document_id: uuid.UUID
    segment_id: uuid.UUID | None = None
    cluster_id: uuid.UUID | None
    x: float
    y: float
    z: float | None
    snippet: str | None = None
    primary_key_value: str | None = None
    sentiment_score: float | None = None
    cluster_label: str | None = None


class EmbeddingStats(BaseModel):
    total: int
    noise: int


class ClusterRead(BaseModel):
    id: uuid.UUID
    label: str
    summary: str
    label_source: str = "terms_fallback"
    top_terms: list[dict[str, Any]]
    word_frequencies: dict[str, Any]
    size: int              # distinct parent reviews
    n_mentions: int = 0    # segment mentions (equal to size for document runs)
    sentiment_avg: float | None = None
    sentiment_count: int = 0
    mean_stars: float | None = None
    cohesion: float | None = None
    sample_docs: list[dict[str, Any]] = []

    model_config = {"from_attributes": True}


class ClusterChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ClusterChatSource(BaseModel):
    document_id: uuid.UUID
    primary_key_value: str
    matched_field: str | None = None
    matched_value: str | None = None
    rating: float | None = None
    snippet: str
    full_text: str | None = None
    raw_data: dict[str, Any]


class ClusterChatResult(BaseModel):
    document_id: uuid.UUID
    primary_key_value: str
    rating: float | None = None
    snippet: str


class ClusterChatRequest(BaseModel):
    message: str
    history: list[ClusterChatMessage] = Field(default_factory=list)
    active_source_ids: list[uuid.UUID] = Field(default_factory=list)
    last_tool_state: dict[str, Any] | None = None


class ClusterChatResponse(BaseModel):
    message: ClusterChatMessage
    model: str
    sources: list[ClusterChatSource] = Field(default_factory=list)
    results: list[ClusterChatResult] = Field(default_factory=list)
    tool_state: dict[str, Any] | None = None


class ModelsRead(BaseModel):
    embedding_model: str
    label_model: str
    variant: str
    simulated: bool


class ClusterMembership(BaseModel):
    cluster_id: uuid.UUID
    cluster_label: str
    mention_count: int
    share: float


class DocumentRead(BaseModel):
    id: uuid.UUID
    primary_key_value: str
    text: str
    raw_data: dict[str, Any]
    cluster_id: uuid.UUID | None
    sentiment_score: float | None = None
    memberships: list[ClusterMembership] = []
    primary_cluster_id: uuid.UUID | None = None

    model_config = {"from_attributes": True}


class ClusterEditRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    actor_id: uuid.UUID
    action: str
    created_at: datetime
    cluster_id: uuid.UUID | None = None
    target_cluster_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    segment_id: uuid.UUID | None = None
    new_label: str | None = None
    note: str | None = None
    payload: dict[str, Any] = {}

    model_config = {"from_attributes": True}


class BulkReassign(BaseModel):
    document_ids: list[uuid.UUID]
    cluster_id: uuid.UUID | None = None  # None means noise


class BulkReassignResult(BaseModel):
    moved: int


class DocumentCount(BaseModel):
    total: int


class ProjectMetricsRead(BaseModel):
    metrics: dict[str, Any] | None = None
    computed_at: datetime | None = None
    stale: bool = False


class ClusterCreate(BaseModel):
    label: str


class ClusterMerge(BaseModel):
    source_ids: list[uuid.UUID]
    target_id: uuid.UUID


class SegmentReassign(BaseModel):
    cluster_id: uuid.UUID | None = None  # None means noise


class BulkSegmentReassign(BaseModel):
    segment_ids: list[uuid.UUID]
    cluster_id: uuid.UUID | None = None


class ClusterFromSegments(BaseModel):
    segment_ids: list[uuid.UUID]
    label: str


class ReviewReassign(BaseModel):
    cluster_id: uuid.UUID | None = None


class ClusterUpdate(BaseModel):
    label: str | None = None
    approve: bool | None = None
    mark_junk: bool | None = None


class MemberCreate(BaseModel):
    email: EmailStr
    role: ProjectRole = ProjectRole.viewer


class MemberUpdate(BaseModel):
    role: ProjectRole


class MemberRead(BaseModel):
    user_id: uuid.UUID
    email: str
    role: ProjectRole
