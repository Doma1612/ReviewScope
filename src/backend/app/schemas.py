import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, field_validator, model_validator

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
    # "document" (one cluster per review; frozen/read-only) or "sentence"
    # (segment-level; a review maps to several clusters; editable).
    unit: str = "document"


class ProjectUpdate(BaseModel):
    name: str


# Column types the upload step and the editable schema both accept. Kept here
# (not in ml_mapping) so validation stays free of the heavy ML import.
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
    # Reflects the stored columns verbatim (which may predate this schema's
    # validation), so the read side stays tolerant rather than re-validating.
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
    # For sentence-unit projects a point is a segment (mention): `segment_id` is set
    # and `document_id` is the parent review. For document-unit projects `segment_id`
    # is None and `document_id` is the point.
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
    # Honest totals for a (possibly sampled) scatter fetch.
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
    n_mentions: int = 0    # segment mentions (== size for document unit)
    sentiment_avg: float | None = None
    sentiment_count: int = 0
    mean_stars: float | None = None
    cohesion: float | None = None
    sample_docs: list[dict[str, Any]] = []

    model_config = {"from_attributes": True}


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
    # Sentence-unit only: the clusters this review's mentions fall into (empty for
    # document-unit projects). primary_cluster_id == cluster_id (the plurality one).
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


class DocumentReassign(BaseModel):
    cluster_id: uuid.UUID | None = None  # None = move to noise


class BulkReassign(BaseModel):
    document_ids: list[uuid.UUID]
    cluster_id: uuid.UUID | None = None  # None = move to noise


class BulkReassignResult(BaseModel):
    moved: int


class DocumentCount(BaseModel):
    total: int


class ProjectMetricsRead(BaseModel):
    # Run-level clustering-quality report; None for simulated runs. `stale` is true
    # when manual edits postdate the run, so the figures reflect the original run.
    metrics: dict[str, Any] | None = None
    computed_at: datetime | None = None
    stale: bool = False


class ClusterCreate(BaseModel):
    label: str


class ClusterMerge(BaseModel):
    source_ids: list[uuid.UUID]
    target_id: uuid.UUID


class ClusterFromSelection(BaseModel):
    document_ids: list[uuid.UUID]
    label: str


class SegmentReassign(BaseModel):
    cluster_id: uuid.UUID | None = None  # None = move mention to noise


class BulkSegmentReassign(BaseModel):
    segment_ids: list[uuid.UUID]
    cluster_id: uuid.UUID | None = None


class ClusterFromSegments(BaseModel):
    segment_ids: list[uuid.UUID]
    label: str


class ReviewReassign(BaseModel):
    cluster_id: uuid.UUID | None = None  # move all of a review's mentions


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
