"""
Per-unit sentiment scoring (app-spec Celery step 6).

Model: ``cardiffnlp/twitter-roberta-base-sentiment-latest`` — RoBERTa-base
fine-tuned on ~124M tweets (TweetEval). Tweet-length training is exactly the
mention regime of the sentence-level pipeline; on whole reviews it still
works but mixes the aspects the review mixes.

Score and label:
- ``score = P(positive) - P(negative)`` in [-1, 1]
- label thresholds (team decision): negative < -0.2, neutral in [-0.2, 0.2],
  positive > 0.2. Stored as named constants so the report can cite them.

Architectural rule, enforced by placement: sentiment is metadata *about*
clusters, never an input *to* clustering. Tier 3 (rating entropy) exists to
punish sentiment-driven clusters — feeding sentiment into the embedding or
clustering stage would sabotage our own quality criterion. This stage
depends only on the texts and joins the artifacts at assembly time.

Compared to star ratings (Tier 3's source): stars are per *review*, this is
per *unit* — at sentence level that surfaces the negative breakfast mention
inside a 4-star review, which no star column can. On the benchmark both
exist, which doubles as a sanity check (score should correlate with stars).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ..core.cache import make_slug
from ..core.config import PipelineConfig
from ..runtime.gpu import release_cuda_memory

logger = logging.getLogger("reviewscope.sentiment")

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NEGATIVE_BELOW = -0.2
POSITIVE_ABOVE = 0.2
LABELS = ("negative", "neutral", "positive")


def score_to_label(score: float) -> str:
    if score < NEGATIVE_BELOW:
        return "negative"
    if score > POSITIVE_ABOVE:
        return "positive"
    return "neutral"


class SentimentScorer:
    """Lazy-loading, batched, device-aware; ``close`` frees the weights."""

    def __init__(
        self,
        model_name: str = SENTIMENT_MODEL,
        device: str = "cpu",
        batch_size: int = 128,
        max_length: int = 512,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.show_progress = show_progress
        # One (model, tokenizer) replica per device, loaded lazily. On CUDA the
        # replicas span every visible card so score() is data-parallel across GPUs
        # (CUDA_VISIBLE_DEVICES is already pinned to idle cards upstream).
        self._replicas: dict[str, tuple] = {}

    def _target_devices(self) -> list[str]:
        """Devices to shard across: every visible CUDA card, else the one device.

        Mirrors the embedder (embed/sentence_transformer.py::_target_devices):
        CUDA_VISIBLE_DEVICES is pinned to idle cards upstream, so using all of
        them is safe. RoBERTa-base is small (~0.5 GB), so a replica per card is cheap.
        """
        if self.device != "cuda":
            return [self.device]
        import torch

        if not torch.cuda.is_available():
            return [self.device]
        return [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]

    def _load(self, device: str):
        if device not in self._replicas:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, torch_dtype=torch.float32  # Pascal: no bf16
            ).to(device)
            model.eval()
            self._replicas[device] = (model, tokenizer)
        return self._replicas[device]

    def score(self, texts: list[str]) -> np.ndarray:
        """Signed sentiment score per text: P(positive) - P(negative).

        Sharded across every visible GPU — contiguous slices, one model replica
        per card, threads (torch releases the GIL inside CUDA kernels) — then
        reassembled in order. Single-device work (CPU, one GPU, or a tiny batch)
        stays on one card. Mirrors the embedder's in-process data-parallel encode.
        """
        devices = self._target_devices()
        logger.info(
            "scoring sentiment for %d texts (%s, batch %d, devices %s)",
            len(texts), self.model_name, self.batch_size, devices,
        )
        if len(devices) == 1 or len(texts) < 2 * self.batch_size:
            return self._score_on(devices[0], texts, progress=self.show_progress)

        from concurrent.futures import ThreadPoolExecutor

        bounds = np.linspace(0, len(texts), len(devices) + 1, dtype=int)
        shards = [(dev, a, b) for dev, a, b in zip(devices, bounds[:-1], bounds[1:]) if b > a]
        scores = np.empty(len(texts), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            # Only the first shard draws a bar — parallel bars would interleave;
            # shards are equal-sized, so one is representative (mirrors the embedder).
            futures = {
                pool.submit(self._score_on, dev, texts[a:b], progress=(self.show_progress and i == 0)): (a, b)
                for i, (dev, a, b) in enumerate(shards)
            }
            for fut, (a, b) in futures.items():
                scores[a:b] = fut.result()
        return scores

    def _score_on(self, device: str, texts: list[str], progress: bool = False) -> np.ndarray:
        """Score one contiguous shard on a single device."""
        import torch

        model, tokenizer = self._load(device)
        scores = np.empty(len(texts), dtype=np.float32)
        starts = range(0, len(texts), self.batch_size)
        if progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="Sentiment", unit="batch")
        with torch.inference_mode():
            for start in starts:
                chunk = texts[start:start + self.batch_size]
                enc = tokenizer(
                    chunk, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1)
                # TweetEval label order: 0=negative, 1=neutral, 2=positive
                scores[start:start + len(chunk)] = (
                    (probs[:, 2] - probs[:, 0]).float().cpu().numpy()
                )
        return scores

    def close(self) -> None:
        self._replicas.clear()
        release_cuda_memory()
        logger.info("released sentiment model %s", self.model_name)


def sentiment_with_cache(
    cfg: PipelineConfig,
    texts: list[str],
    device: str = "cpu",
    prefix_extra: str = "",
) -> tuple[np.ndarray, float]:
    """
    Score *texts* through the on-disk cache (same namespacing convention as
    embeddings: corpus + unit + size). Returns (scores, seconds).
    """
    corpus = cfg.corpus_slug
    prefix = ("" if corpus == "hotels" else f"{corpus}__") + prefix_extra
    k = f"{len(texts) // 1000}k" if len(texts) % 1000 == 0 else str(len(texts))
    path = cfg.cache_dir / "sentiment" / f"{prefix}{make_slug(SENTIMENT_MODEL)}__{k}.npy"

    if path.exists():
        return np.load(path), 0.0
    scorer = SentimentScorer(device=device)
    try:
        t0 = time.time()
        scores = scorer.score(texts)
        elapsed = time.time() - t0
    finally:
        scorer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    return scores, elapsed


def aggregate_cluster_sentiment(
    scores: np.ndarray, labels: np.ndarray, cluster_id: int
) -> tuple[Optional[float], Optional[dict[str, float]]]:
    """(mean score, label-share dict) for one cluster's units."""
    mask = labels == cluster_id
    if not mask.any():
        return None, None
    cluster_scores = scores[mask]
    unit_labels = [score_to_label(float(s)) for s in cluster_scores]
    n = len(unit_labels)
    dist = {lab: round(unit_labels.count(lab) / n, 4) for lab in LABELS}
    return round(float(cluster_scores.mean()), 4), dist
