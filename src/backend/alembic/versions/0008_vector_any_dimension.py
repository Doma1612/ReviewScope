"""widen the RAG vector columns to any embedding dimension

Revision ID: 0008_vector_any_dimension
Revises: 0007_project_rag_vectors
Create Date: 2026-09-22

0007 typed both ``vector_pg`` columns as ``vector(768)``, matching the
mpnet embeddings the pipeline produced at the time. The application now
defaults to all-MiniLM-L6-v2 (384), and a project records its own
``embedding_model`` / ``embedding_dimension``, so projects can legitimately
differ. A fixed column width contradicts that provenance and silently
disables retrieval for every project whose model is not 768-dimensional.

An unconstrained ``vector`` column accepts any dimension. The trade-off is
that ivfflat/hnsw indexes require a fixed dimension — 0007 created none, so
nothing is lost here, but adding one later means pinning a dimension (or
building partial indexes per model) in a further migration.
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
    # Narrowing back only succeeds if every stored vector is 768-dimensional;
    # rows from a differently sized model must be cleared first.
    op.execute("UPDATE embeddings SET vector_pg = NULL WHERE vector_dims(vector_pg) <> 768")
    op.execute("UPDATE segments SET vector_pg = NULL WHERE vector_dims(vector_pg) <> 768")
    op.execute("ALTER TABLE embeddings ALTER COLUMN vector_pg TYPE vector(768) USING vector_pg::vector(768)")
    op.execute("ALTER TABLE segments ALTER COLUMN vector_pg TYPE vector(768) USING vector_pg::vector(768)")
