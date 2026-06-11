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
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, torch_dtype=torch.float32  # Pascal: no bf16
            ).to(self.device)
            self._model.eval()
        return self._model, self._tokenizer

    def score(self, texts: list[str]) -> np.ndarray:
        """Signed sentiment score per text: P(positive) - P(negative)."""
        import torch

        model, tokenizer = self._load()
        logger.info(
            "scoring sentiment for %d texts (%s, batch %d, device %s)",
            len(texts), self.model_name, self.batch_size, self.device,
        )
        scores = np.empty(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start:start + self.batch_size]
                enc = tokenizer(
                    chunk, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(model(**enc).logits, dim=-1)
                # TweetEval label order: 0=negative, 1=neutral, 2=positive
                scores[start:start + len(chunk)] = (
                    (probs[:, 2] - probs[:, 0]).float().cpu().numpy()
                )
        return scores

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
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
