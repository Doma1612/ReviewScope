"""add cohesion to clusters

Revision ID: 0004_cluster_cohesion
Revises: 0003_cluster_edits
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0004_cluster_cohesion"
down_revision = "0003_cluster_edits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("cohesion", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("clusters", "cohesion")
