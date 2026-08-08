"""add run-level metrics to projects

Revision ID: 0005_project_metrics
Revises: 0004_cluster_cohesion
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_project_metrics"
down_revision = "0004_cluster_cohesion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("metrics", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("metrics_run_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "metrics_run_at")
    op.drop_column("projects", "metrics")
