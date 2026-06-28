"""add label_source and mean_stars to clusters

Revision ID: 0002_cluster_fields
Revises: 0001_initial
Create Date: 2026-06-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_cluster_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "clusters",
        sa.Column("label_source", sa.Text(), nullable=False, server_default="terms_fallback"),
    )
    op.add_column("clusters", sa.Column("mean_stars", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("clusters", "mean_stars")
    op.drop_column("clusters", "label_source")
