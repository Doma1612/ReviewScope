"""segment-level clustering (additive; preserves existing data)

Revision ID: 0006_segments
Revises: 0005_project_metrics
Create Date: 2026-07-01

Additive only. Creates the ``segments`` table (the clustered unit for
sentence-unit projects), adds ``projects.unit`` (defaults every existing row to
"document"), ``clusters.n_mentions``, and ``cluster_edits.segment_id``, and
widens the edit-action CHECK constraint. Nothing is dropped and no existing rows
are rewritten — the ``embeddings`` table and all current documents/clusters stay
exactly as they are, so existing (document-unit) projects keep rendering.
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
    # Per-project clustering unit; existing rows backfill to "document".
    op.add_column(
        "projects",
        sa.Column("unit", sa.Text(), nullable=False, server_default="document"),
    )
    # Mention (segment) count alongside the distinct-review `size`.
    op.add_column(
        "clusters",
        sa.Column("n_mentions", sa.Integer(), nullable=False, server_default="0"),
    )
    # Subject column for segment-level edits (plain UUID, mirrors document_id).
    op.add_column(
        "cluster_edits",
        sa.Column("segment_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    # The clustered unit for sentence-unit projects.
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

    # Widen the edit-action vocabulary (drop + recreate the CHECK).
    op.drop_constraint("ck_cluster_edits_action", "cluster_edits", type_="check")
    op.create_check_constraint(
        "ck_cluster_edits_action",
        "cluster_edits",
        "action IN (" + ", ".join(f"'{a}'" for a in ACTIONS) + ")",
    )


def downgrade() -> None:
    # Restore the pre-0006 action list (drop the three sentence-unit actions).
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
