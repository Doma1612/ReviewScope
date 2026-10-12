from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import PipelineConfig, get_preprocessor
from .sources.yelp import build_benchmark_sample, subset_sample

logger = logging.getLogger("reviewscope.data")

DOC_ID_FIELDS = ("review_id", "speech_id", "doc_id", "id")
RATING_FIELDS = ("stars", "rating", "score")

__all__ = [
    "ReviewSet", "load_benchmark", "build_benchmark_sample", "subset_sample",
]


@dataclass
class ReviewSet:

    ids: list[str]
    texts: list[str]
    raw_texts: list[str]
    stars: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def has_ratings(self) -> bool:
        return bool(self.stars.size) and not bool(np.isnan(self.stars).all())


def _first_present(obj: dict, fields: tuple[str, ...]):
    for field in fields:
        if field in obj and obj[field] is not None:
            return obj[field]
    return None


def load_benchmark(cfg: PipelineConfig, preprocessor: str = "minimal") -> ReviewSet:
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Benchmark file missing: {cfg.data_path}\n"
            "Generate it with one of:\n"
            f"  python -m reviewscope_ml.data.sources.yelp --sample-size {cfg.sample_size}\n"
            f"  python -m reviewscope_ml.data.sources.europarl --sample-size {cfg.sample_size}"
        )
    preprocess = get_preprocessor(preprocessor)
    ids, texts, raw_texts, stars = [], [], [], []
    with open(cfg.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = _first_present(obj, DOC_ID_FIELDS)
            ids.append(str(doc_id) if doc_id is not None else str(len(ids)))
            raw_texts.append(obj["text"])
            texts.append(preprocess(obj["text"]))
            rating = _first_present(obj, RATING_FIELDS)
            stars.append(float(rating) if rating is not None else float("nan"))
            if len(texts) >= cfg.sample_size:
                break
    reviews = ReviewSet(
        ids=ids, texts=texts, raw_texts=raw_texts, stars=np.array(stars, dtype=float)
    )
    logger.info(
        "loaded %d documents from %s (ratings: %s)",
        len(reviews), cfg.data_path.name,
        "yes" if reviews.has_ratings else "none — rating metrics unavailable",
    )
    return reviews
