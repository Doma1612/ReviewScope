"""segment-level clustering (additive; preserves existing data)

Revision ID: 0006_segments
Revises: 0005_project_metrics
Create Date: 2026-07-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_segments"
down_revision = "0005_project_metrics"
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
    "reassign_segment",
    "bulk_reassign_segments",
    "reassign_review",
)


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("unit", sa.Text(), nullable=False, server_default="document"),
    )
    op.add_column(
        "clusters",
        sa.Column("n_mentions", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "cluster_edits",
        sa.Column("segment_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.create_table(
        "segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segment_key", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("cluster_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clusters.id", ondelete="SET NULL")),
        sa.Column("sentiment_score", sa.Float()),
        sa.Column("vector", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("umap_x", sa.Float(), nullable=False),
        sa.Column("umap_y", sa.Float(), nullable=False),
        sa.Column("umap_z", sa.Float()),
        sa.UniqueConstraint("project_id", "segment_key", name="uq_segments_project_key"),
    )
    op.create_index("ix_segments_project_id", "segments", ["project_id"])
    op.create_index("ix_segments_document_id", "segments", ["document_id"])
    op.create_index("ix_segments_cluster_id", "segments", ["cluster_id"])

    op.drop_constraint("ck_cluster_edits_action", "cluster_edits", type_="check")
    op.create_check_constraint(
        "ck_cluster_edits_action",
        "cluster_edits",
        "action IN (" + ", ".join(f"'{a}'" for a in ACTIONS) + ")",
    )


def downgrade() -> None:
    legacy = ACTIONS[:-3]
    op.drop_constraint("ck_cluster_edits_action", "cluster_edits", type_="check")
    op.create_check_constraint(
        "ck_cluster_edits_action",
        "cluster_edits",
        "action IN (" + ", ".join(f"'{a}'" for a in legacy) + ")",
    )

    op.drop_index("ix_segments_cluster_id", table_name="segments")
    op.drop_index("ix_segments_document_id", table_name="segments")
    op.drop_index("ix_segments_project_id", table_name="segments")
    op.drop_table("segments")

    op.drop_column("cluster_edits", "segment_id")
    op.drop_column("clusters", "n_mentions")
    op.drop_column("projects", "unit")
