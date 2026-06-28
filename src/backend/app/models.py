import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ProjectStatus(str, enum.Enum):
    uploading = "uploading"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ProjectRole(str, enum.Enum):
    owner = "owner"
    viewer = "viewer"


class PipelineStepStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    status: Mapped[ProjectStatus] = mapped_column(Enum(ProjectStatus), default=ProjectStatus.uploading, nullable=False)
    doc_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    source_filename: Mapped[str | None] = mapped_column(Text)
    upload_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Run-level clustering-quality report (silhouette/coherence/rating-entropy/…)
    # captured from the real ML run (null in simulated mode). metrics_run_at lets the
    # UI flag the report as stale once manual edits change the membership.
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    metrics_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    owner: Mapped[User] = relationship()


class ProjectMember(Base):
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_project_members_project_user"),)

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[ProjectRole] = mapped_column(Enum(ProjectRole), nullable=False)

    user: Mapped[User] = relationship()


class ProjectSchema(Base):
    __tablename__ = "project_schema"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    columns: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # Label provenance: "ollama:<model>" | "terms_fallback" | "hitl_override".
    # Lets the UI distinguish LLM labels from term-fallbacks when Ollama is down.
    label_source: Mapped[str] = mapped_column(Text, default="terms_fallback", server_default="terms_fallback", nullable=False)
    top_terms: Mapped[list[dict]] = mapped_column(JSONB, default=list, nullable=False)
    word_frequencies: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sentiment_avg: Mapped[float | None] = mapped_column(Float)
    mean_stars: Mapped[float | None] = mapped_column(Float)
    # Cohesion = mean cosine similarity of member embeddings to the cluster centroid.
    # Higher = tighter/more trustworthy. None when undefined (singleton/empty).
    cohesion: Mapped[float | None] = mapped_column(Float)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    primary_key_value: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("clusters.id", ondelete="SET NULL"), index=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float)


class Embedding(Base):
    __tablename__ = "embeddings"

    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(JSONB, default=list, nullable=False)
    umap_x: Mapped[float] = mapped_column(Float, nullable=False)
    umap_y: Mapped[float] = mapped_column(Float, nullable=False)
    umap_z: Mapped[float | None] = mapped_column(Float)


# Audit vocabulary for ClusterEdit.action. Mirrors reviewscope_ml's
# hitl.feedback.ACTIONS, plus the app-only actions that have no notebook analogue
# (bulk reassign, create-from-selection, etc.). Keep in sync with the
# CheckConstraint below and the migration 0003_cluster_edits.
EDIT_ACTIONS = (
    "approve_label",
    "rename_label",
    "merge_clusters",
    "split_cluster",
    "reassign_doc",
    "bulk_reassign",
    "create_cluster",
    "create_from_selection",
    "mark_junk",
    "confirm_run",
)


class ClusterEdit(Base):
    """Append-only audit log of every cluster/document edit.

    Subject columns are plain UUIDs, not FKs: cluster ids are regenerated on each
    re-run and clusters get deleted, but the audit trail must outlive them so it
    can be replayed and shown in the undo/history UI.
    """

    __tablename__ = "cluster_edits"
    __table_args__ = (
        CheckConstraint(
            "action IN (" + ", ".join(f"'{a}'" for a in EDIT_ACTIONS) + ")",
            name="ck_cluster_edits_action",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    actor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    target_cluster_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    new_label: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}", nullable=False)


class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    step: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[PipelineStepStatus] = mapped_column(Enum(PipelineStepStatus), default=PipelineStepStatus.pending, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
