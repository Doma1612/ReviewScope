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
        self._replicas: dict[str, tuple] = {}

    def _target_devices(self) -> list[str]:
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
                self.model_name, torch_dtype=torch.float32  # Pascal has no bf16 support.
            ).to(device)
            model.eval()
            self._replicas[device] = (model, tokenizer)
        return self._replicas[device]

    def score(self, texts: list[str]) -> np.ndarray:
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
            # Show one progress bar to avoid interleaving bars from parallel shards.
            futures = {
                pool.submit(self._score_on, dev, texts[a:b], progress=(self.show_progress and i == 0)): (a, b)
                for i, (dev, a, b) in enumerate(shards)
            }
            for fut, (a, b) in futures.items():
                scores[a:b] = fut.result()
        return scores

    def _score_on(self, device: str, texts: list[str], progress: bool = False) -> np.ndarray:
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
                # TweetEval label order: negative, neutral, positive.
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
    mask = labels == cluster_id
    if not mask.any():
        return None, None
    cluster_scores = scores[mask]
    unit_labels = [score_to_label(float(s)) for s in cluster_scores]
    n = len(unit_labels)
    dist = {lab: round(unit_labels.count(lab) / n, 4) for lab in LABELS}
    return round(float(cluster_scores.mean()), 4), dist
