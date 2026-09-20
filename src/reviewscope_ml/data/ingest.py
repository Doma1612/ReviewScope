"""
Benchmark data loading.

``load_benchmark`` reads a benchmark JSONL — whatever corpus produced it — into
the in-memory shape every pipeline stage consumes. The files themselves are
written by the adapters in ``data.sources`` (Yelp hotel reviews, Europarl
English speeches); this module knows only the common record shape and nothing
about any vendor's archive format.

The Yelp builders are re-exported here so the notebooks and
``scripts/make_notebooks.py`` keep their existing import paths.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import PipelineConfig, get_preprocessor
from .sources.yelp import build_benchmark_sample, subset_sample

logger = logging.getLogger("reviewscope.data")

# Per-document key, in priority order. Corpora name their identifier after
# their own unit ("review_id", "speech_id"); the pipeline only needs *a*
# stable per-document key, and falls back to the row index if none is present.
DOC_ID_FIELDS = ("review_id", "speech_id", "doc_id", "id")
# Optional numeric rating. Absent on corpora that have none (Europarl), in
# which case `stars` is all-NaN and every rating-derived metric reports None
# rather than a fabricated number.
RATING_FIELDS = ("stars", "rating", "score")

__all__ = [
    "ReviewSet", "load_benchmark", "build_benchmark_sample", "subset_sample",
]


@dataclass
class ReviewSet:
    """The benchmark sample in the shape all pipeline stages consume."""

    ids: list[str]            # stable per-document key
    texts: list[str]          # preprocessed text (default: "minimal" variant)
    raw_texts: list[str]      # untouched text, for human inspection artifacts
    stars: np.ndarray         # numeric rating per document; NaN if the corpus has none

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def has_ratings(self) -> bool:
        """False for corpora without a rating column (Europarl, plain text dumps).

        Callers use this to skip rating-derived reporting entirely instead of
        printing a metric computed from an all-NaN column.
        """
        return bool(self.stars.size) and not bool(np.isnan(self.stars).all())


def _first_present(obj: dict, fields: tuple[str, ...]):
    for field in fields:
        if field in obj and obj[field] is not None:
            return obj[field]
    return None


def load_benchmark(cfg: PipelineConfig, preprocessor: str = "minimal") -> ReviewSet:
    """
    Load ``cfg.data_path`` (e.g. sample_hotels_5k.jsonl or
    sample_europarl-en_5k.jsonl), capped at ``cfg.sample_size`` rows and
    preprocessed with the variant notebook 02 chose.
    """
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
