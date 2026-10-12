from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.cache import embedding_path, load_array, save_array
from ..core.config import PipelineConfig
from ..core.corpus import HOTELS
from ..runtime.gpu import release_cuda_memory

logger = logging.getLogger("reviewscope.embed")

INSTRUCTIONS: dict[str, Optional[str]] = HOTELS.instructions


def _is_cuda_oom(e: Exception) -> bool:
    return "CUDA out of memory" in str(e) or type(e).__name__ == "OutOfMemoryError"


class SentenceTransformerEmbedder:
    # Use in-process replicas: spawned workers lose VRAM caps and can deadlock on OOM.

    def __init__(
        self,
        model_name: str,
        instruction: str = "no_inst",
        device: str = "cpu",
        batch_size: int = 64,
        show_progress: bool = True,
        max_seq: Optional[int] = None,
        instructions: Optional[dict[str, Optional[str]]] = None,
    ):
        self.instructions = INSTRUCTIONS if instructions is None else instructions
        if instruction not in self.instructions:
            raise ValueError(
                f"Unknown instruction slug {instruction!r}; "
                f"known: {list(self.instructions)}"
            )
        self.model_name = model_name
        self.instruction = instruction
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_seq = max_seq
        self._replicas: dict[str, object] = {}

    @property
    def _is_instructor(self) -> bool:
        return "instructor" in self.model_name.lower() and self.model_name.startswith("hkunlp/")

    def _target_devices(self) -> list[str]:
        if self.device != "cuda" or self._is_instructor:
            return [self.device]
        import torch

        n = max(1, torch.cuda.device_count())
        return [f"cuda:{i}" for i in range(n)]

    def _load(self, device: Optional[str] = None):
        device = device or self._target_devices()[0]
        if device in self._replicas:
            return self._replicas[device]
        if self._is_instructor:
            from InstructorEmbedding import INSTRUCTOR

            model = INSTRUCTOR(self.model_name, device=device)
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            # Pascal lacks bf16 support; force fp32 to avoid mixed-dtype matmul failures.
            model = SentenceTransformer(
                self.model_name,
                device=device,
                model_kwargs={"torch_dtype": torch.float32},
            )
            if self.max_seq is not None:
                model.max_seq_length = min(model.max_seq_length, self.max_seq)
        self._replicas[device] = model
        return model

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.batch_size
        while True:
            logger.info(
                "encoding %d texts with %s (batch %d, device %s%s)",
                len(texts), self.model_name, batch, self.device,
                f", seq<={self.max_seq}" if self.max_seq else "",
            )
            try:
                return self._encode_once(texts, batch)
            except Exception as e:
                if not _is_cuda_oom(e) or batch <= 8:
                    raise
                batch //= 2
                release_cuda_memory()
                logger.warning("CUDA OOM at batch %d — retrying with %d", batch * 2, batch)

    def _encode_once(self, texts: list[str], batch: int) -> np.ndarray:
        devices = self._target_devices()
        if len(devices) == 1 or len(texts) < 2 * batch:
            return self._encode_on(devices[0], texts, batch, progress=self.show_progress)

        from concurrent.futures import ThreadPoolExecutor

        logger.info("data-parallel encode across %s (in-process replicas)", devices)
        bounds = np.linspace(0, len(texts), len(devices) + 1, dtype=int)
        shards = [
            (dev, texts[a:b], i == 0)
            for i, (dev, a, b) in enumerate(zip(devices, bounds[:-1], bounds[1:]))
            if b > a
        ]
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            futures = [
                pool.submit(self._encode_on, dev, shard, batch,
                            progress=self.show_progress and first)
                for dev, shard, first in shards
            ]
            parts = [f.result() for f in futures]
        return np.vstack(parts)

    def _encode_on(self, device: str, texts: list[str], batch: int, progress: bool) -> np.ndarray:
        model = self._load(device)
        instr_text = self.instructions[self.instruction]
        if self._is_instructor and instr_text:
            pairs = [[instr_text, t] for t in texts]
            return np.asarray(
                model.encode(pairs, batch_size=batch, show_progress_bar=progress)
            )
        kwargs = dict(batch_size=batch, show_progress_bar=progress, convert_to_numpy=True)
        if instr_text and not self._is_instructor:
            kwargs["prompt"] = instr_text
        return np.asarray(model.encode(texts, **kwargs))

    def close(self) -> None:
        self._replicas.clear()
        release_cuda_memory()
        logger.info("released embedding model %s", self.model_name)


def embed_with_cache(
    cfg: PipelineConfig,
    embedder: SentenceTransformerEmbedder,
    texts: list[str],
    prefix_extra: str = "",
) -> tuple[np.ndarray, float]:
    import time

    corpus = cfg.corpus_slug
    path = embedding_path(
        cfg.cache_dir,
        embedder.model_name,
        len(texts),
        instruction=embedder.instruction,
        prefix=("" if corpus == "hotels" else f"{corpus}__") + prefix_extra,
    )
    if path.exists():
        return load_array(path), 0.0
    t0 = time.time()
    embeddings = embedder.encode(texts)
    elapsed = time.time() - t0
    save_array(path, embeddings.astype(np.float32))
    return embeddings, elapsed
