"""
sentence-transformers embedding backend.

Covers the model families benchmarked in notebook 04:
- plain models (all-MiniLM-L6-v2, all-mpnet-base-v2, bge-m3) — no instruction;
- prompt-API instruction models (multilingual-e5-large-instruct,
  Qwen3-Embedding) — instruction passed as ``prompt=``;
- INSTRUCTOR models (hkunlp/instructor-*) — instruction passed as
  [instruction, text] pairs via the InstructorEmbedding package.

Instruction texts are notebook 04's exact variants so cached embeddings and
logged results remain comparable between notebook and package runs.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.cache import embedding_path, load_array, save_array
from ..core.config import PipelineConfig
from ..runtime.gpu import release_cuda_memory

logger = logging.getLogger("reviewscope.embed")

# Instruction variants decided in notebook 04 — do not edit without re-running
# the embedding comparison, the slugs are baked into cache filenames.
INSTRUCTIONS: dict[str, Optional[str]] = {
    "no_inst": None,
    "generic": "Represent the hotel review for topic clustering:",
    "domain": (
        "Represent the hotel review for clustering by theme "
        "(room quality, staff, location, breakfast, cleanliness, value):"
    ),
    "sentiment": "Represent the hotel review to capture the main sentiment and opinion:",
}


def _is_cuda_oom(e: Exception) -> bool:
    return "CUDA out of memory" in str(e) or type(e).__name__ == "OutOfMemoryError"


class SentenceTransformerEmbedder:
    """Lazy-loading embedder; weights load on first encode, ``close`` frees them."""

    def __init__(
        self,
        model_name: str,
        instruction: str = "no_inst",
        device: str = "cpu",
        batch_size: int = 64,
        show_progress: bool = True,
    ):
        if instruction not in INSTRUCTIONS:
            raise ValueError(f"Unknown instruction slug {instruction!r}; known: {list(INSTRUCTIONS)}")
        self.model_name = model_name
        self.instruction = instruction
        self.device = device
        self.batch_size = batch_size
        # tqdm progress for encode: the embed stage is the longest single step
        # at 50k, and a silent half hour looks like a hang on the GPU server.
        self.show_progress = show_progress
        self._model = None

    @property
    def _is_instructor(self) -> bool:
        return "instructor" in self.model_name.lower() and self.model_name.startswith("hkunlp/")

    def _load(self):
        if self._model is not None:
            return self._model
        if self._is_instructor:
            from InstructorEmbedding import INSTRUCTOR

            self._model = INSTRUCTOR(self.model_name, device=self.device)
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode with CUDA-OOM protection: our VRAM share is hard-capped at
        cuda_mem_fraction (~6 GB on a TITAN X), so an over-ambitious batch
        size raises OutOfMemoryError instead of stealing a neighbour's memory.
        We catch it, halve the batch, and retry down to batch 8 rather than
        killing an hours-long run over a tunable.
        """
        model = self._load()
        batch = self.batch_size
        while True:
            logger.info(
                "encoding %d texts with %s (batch %d, device %s)",
                len(texts), self.model_name, batch, self.device,
            )
            try:
                return self._encode_once(model, texts, batch)
            except Exception as e:
                if not _is_cuda_oom(e) or batch <= 8:
                    raise
                batch //= 2
                release_cuda_memory()
                logger.warning("CUDA OOM at batch %d — retrying with %d", batch * 2, batch)

    def _encode_once(self, model, texts: list[str], batch: int) -> np.ndarray:
        instr_text = INSTRUCTIONS[self.instruction]
        if self._is_instructor and instr_text:
            pairs = [[instr_text, t] for t in texts]
            return np.asarray(
                model.encode(pairs, batch_size=batch,
                             show_progress_bar=self.show_progress)
            )
        kwargs = dict(batch_size=batch,
                      show_progress_bar=self.show_progress, convert_to_numpy=True)
        if instr_text and not self._is_instructor:
            kwargs["prompt"] = instr_text
        return np.asarray(model.encode(texts, **kwargs))

    def close(self) -> None:
        """Release weights immediately — no idle process may squat on VRAM."""
        self._model = None
        release_cuda_memory()
        logger.info("released embedding model %s", self.model_name)


def embed_with_cache(
    cfg: PipelineConfig,
    embedder: SentenceTransformerEmbedder,
    texts: list[str],
) -> tuple[np.ndarray, float]:
    """
    Encode *texts* through the on-disk embedding cache.

    Returns (embeddings, embed_seconds); embed_seconds is 0.0 on a cache hit.
    The cache key is (model, instruction, sample_size) — the same convention
    notebooks 03-08 use, so package runs reuse notebook embeddings and
    vice versa.
    """
    import time

    corpus = cfg.corpus_slug
    path = embedding_path(
        cfg.cache_dir,
        embedder.model_name,
        len(texts),
        instruction=embedder.instruction,
        # hotels keeps the historical bare filenames; other corpora must not
        # collide with them at the same sample size
        prefix="" if corpus == "hotels" else f"{corpus}__",
    )
    if path.exists():
        return load_array(path), 0.0
    t0 = time.time()
    embeddings = embedder.encode(texts)
    elapsed = time.time() - t0
    save_array(path, embeddings.astype(np.float32))
    return embeddings, elapsed
