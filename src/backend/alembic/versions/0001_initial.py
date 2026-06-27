"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    project_status = postgresql.ENUM("uploading", "processing", "ready", "failed", name="projectstatus", create_type=False)
    project_role = postgresql.ENUM("owner", "viewer", name="projectrole", create_type=False)
    pipeline_status = postgresql.ENUM("pending", "running", "done", "failed", name="pipelinestepstatus", create_type=False)
    project_status.create(op.get_bind(), checkfirst=True)
    project_role.create(op.get_bind(), checkfirst=True)
    pipeline_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", project_status, nullable=False),
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("source_filename", sa.Text()),
        sa.Column("upload_path", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "project_members",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", project_role, nullable=False),
        sa.UniqueConstraint("project_id", "user_id", name="uq_project_members_project_user"),
    )

    op.create_table(
        "project_schema",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("columns", postgresql.JSONB(), nullable=False),
    )

    op.create_table(
        "clusters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("top_terms", postgresql.JSONB(), nullable=False),
        sa.Column("word_frequencies", postgresql.JSONB(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sentiment_avg", sa.Float()),
    )
    op.create_index("ix_clusters_project_id", "clusters", ["project_id"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("primary_key_value", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_data", postgresql.JSONB(), nullable=False),
        sa.Column("cluster_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clusters.id", ondelete="SET NULL")),
        sa.Column("sentiment_score", sa.Float()),
    )
    op.create_index("ix_documents_project_id", "documents", ["project_id"])
    op.create_index("ix_documents_cluster_id", "documents", ["cluster_id"])

    op.create_table(
        "embeddings",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("vector", postgresql.JSONB(), nullable=False),
        sa.Column("umap_x", sa.Float(), nullable=False),
        sa.Column("umap_y", sa.Float(), nullable=False),
        sa.Column("umap_z", sa.Float()),
    )

    op.create_table(
        "pipeline_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step", sa.String(length=50), nullable=False),
        sa.Column("status", pipeline_status, nullable=False),
        sa.Column("message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_pipeline_jobs_project_id", "pipeline_jobs", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_jobs_project_id", table_name="pipeline_jobs")
    op.drop_table("pipeline_jobs")
    op.drop_table("embeddings")
    op.drop_index("ix_documents_cluster_id", table_name="documents")
    op.drop_index("ix_documents_project_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_clusters_project_id", table_name="clusters")
    op.drop_table("clusters")
    op.drop_table("project_schema")
    op.drop_table("project_members")
    op.drop_table("projects")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    postgresql.ENUM(name="pipelinestepstatus").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="projectrole").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="projectstatus").drop(op.get_bind(), checkfirst=True)
