"""widen the RAG vector columns to any embedding dimension

Revision ID: 0008_vector_any_dimension
Revises: 0007_project_rag_vectors
Create Date: 2026-09-22
"""
from alembic import op


revision = "0008_vector_any_dimension"
down_revision = "0007_project_rag_vectors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE embeddings ALTER COLUMN vector_pg TYPE vector USING vector_pg::vector")
    op.execute("ALTER TABLE segments ALTER COLUMN vector_pg TYPE vector USING vector_pg::vector")


def downgrade() -> None:
    # Clear incompatible vectors before restoring the 768-dimensional column type.
    op.execute("UPDATE embeddings SET vector_pg = NULL WHERE vector_dims(vector_pg) <> 768")
    op.execute("UPDATE segments SET vector_pg = NULL WHERE vector_dims(vector_pg) <> 768")
    op.execute("ALTER TABLE embeddings ALTER COLUMN vector_pg TYPE vector(768) USING vector_pg::vector(768)")
    op.execute("ALTER TABLE segments ALTER COLUMN vector_pg TYPE vector(768) USING vector_pg::vector(768)")
