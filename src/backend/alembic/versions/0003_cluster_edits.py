"""cluster_edits audit table

Revision ID: 0003_cluster_edits
Revises: 0002_cluster_fields
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_cluster_edits"
down_revision = "0002_cluster_fields"
branch_labels = None
depends_on = None

# Keep in sync with app.models.EDIT_ACTIONS.
ACTIONS = (
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


def upgrade() -> None:
    op.create_table(
        "cluster_edits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("cluster_id", postgresql.UUID(as_uuid=True)),
        sa.Column("target_cluster_id", postgresql.UUID(as_uuid=True)),
        sa.Column("document_id", postgresql.UUID(as_uuid=True)),
        sa.Column("new_label", sa.Text()),
        sa.Column("note", sa.Text()),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.CheckConstraint(
            "action IN (" + ", ".join(f"'{a}'" for a in ACTIONS) + ")",
            name="ck_cluster_edits_action",
        ),
    )
    op.create_index("ix_cluster_edits_project_id", "cluster_edits", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_cluster_edits_project_id", table_name="cluster_edits")
    op.drop_table("cluster_edits")
