"""Lazy, CPU-only query embedding for project semantic retrieval."""
from __future__ import annotations

from threading import Lock

from app.models import Project


_embedders: dict[tuple[str, str], object] = {}
_lock = Lock()


def embed_project_query(project: Project, text: str) -> list[float]:
    """Embed one question with the exact model recorded by the project run."""
    if not project.retrieval_ready or not project.embedding_model or not project.embedding_dimension:
        raise ValueError("Semantic search is unavailable until this project is reprocessed.")

    instruction = project.embedding_instruction or "no_inst"
    key = (project.embedding_model, instruction)
    with _lock:
        embedder = _embedders.get(key)
        if embedder is None:
            from reviewscope_ml.embed.models import encode_settings
            from reviewscope_ml.embed.sentence_transformer import SentenceTransformerEmbedder

            batch_size, max_seq = encode_settings(project.embedding_model, 32)
            embedder = SentenceTransformerEmbedder(
                project.embedding_model,
                instruction=instruction,
                device="cpu",
                batch_size=batch_size,
                show_progress=False,
                max_seq=max_seq,
            )
            _embedders[key] = embedder
        vector = embedder.encode([text])[0].tolist()

    if len(vector) != project.embedding_dimension:
        raise ValueError("The configured embedding model does not match this project's vectors.")
    return vector
