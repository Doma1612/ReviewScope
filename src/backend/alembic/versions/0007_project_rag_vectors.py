"""add pgvector columns for project RAG

Revision ID: 0007_project_rag_vectors
Revises: 0006_segments
Create Date: 2026-09-20
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0007_project_rag_vectors"
down_revision = "0006_segments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("projects", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.add_column("projects", sa.Column("embedding_instruction", sa.Text(), nullable=True))
    op.add_column("projects", sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column("projects", sa.Column("retrieval_ready", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("embeddings", sa.Column("vector_pg", Vector(768), nullable=True))
    op.add_column("segments", sa.Column("vector_pg", Vector(768), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "vector_pg")
    op.drop_column("embeddings", "vector_pg")
    op.drop_column("projects", "retrieval_ready")
    op.drop_column("projects", "embedding_dimension")
    op.drop_column("projects", "embedding_instruction")
    op.drop_column("projects", "embedding_model")
