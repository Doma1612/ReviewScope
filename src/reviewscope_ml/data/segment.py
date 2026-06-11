"""
Sentence segmentation for aspect-level clustering.

Why: hotel reviews average 8 sentences covering several aspects (room, staff,
breakfast, ...). One embedding per review averages those aspects away, and
~10% of reviews are silently truncated at the embedder's token window. At
sentence level the unit of clustering becomes the *mention*; the review is
its container. Counting semantics that follow from this (mentions vs.
distinct reviews) are implemented in the pipeline runner and documented in
docs/methodology.md.

Implementation notes:
- Regex splitter (., !, ? + whitespace), no NLP dependency. It mis-splits
  abbreviations ("St. Louis") occasionally; for clustering this is noise we
  accept — a benchmarked alternative (syntok/pysbd) can swap in behind the
  same function if inspection shows it matters.
- Segments shorter than ``min_chars`` are dropped: "Great!" carries sentiment
  but no topic, and millions of such fragments form one giant junk cluster.
- Very long sentences are hard-wrapped at ``max_chars`` so a single
  punctuation-free rant cannot blow past the token window again.
- Segment ids are ``{review_id}#{i}`` — the parent is always recoverable via
  ``parent_id()``, which is what the membership artifact and the deduplicated
  (per-review) star statistics are built from.
"""
from __future__ import annotations

import re

import numpy as np

from .ingest import ReviewSet

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

MIN_CHARS = 20
MAX_CHARS = 600


def split_sentences(
    text: str, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS
) -> list[str]:
    """Split one text into cleaned sentence segments."""
    out: list[str] = []
    for raw in _SENT_SPLIT.split(text.strip()):
        s = " ".join(raw.split())
        if len(s) < min_chars:
            continue
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > min_chars else max_chars
            out.append(s[:cut])
            s = s[cut:].strip()
        if len(s) >= min_chars:
            out.append(s)
    return out


def parent_id(segment_id: str) -> str:
    """``abc123#4`` -> ``abc123``."""
    return segment_id.rsplit("#", 1)[0]


def segment_reviews(
    reviews: ReviewSet, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS
) -> ReviewSet:
    """
    Deterministically explode a review set into a segment-level ReviewSet.

    Returned set quacks exactly like a document-level one — every downstream
    stage (embed, reduce, cluster, represent, label, metrics, inspection,
    HITL app) consumes it unchanged. Stars are inherited from the parent
    review; deduplication to per-review statistics happens later via
    ``parent_id``.

    Determinism matters: the run artifacts store segment ids, and the HITL
    app re-derives segment texts by re-running this function on the benchmark
    file — same input, same segmentation, always.
    """
    ids: list[str] = []
    texts: list[str] = []
    raw_texts: list[str] = []
    stars: list[float] = []
    for rid, text, raw, star in zip(
        reviews.ids, reviews.texts, reviews.raw_texts, reviews.stars
    ):
        for i, seg in enumerate(split_sentences(text, min_chars, max_chars)):
            ids.append(f"{rid}#{i}")
            texts.append(seg)
            raw_texts.append(seg)
            stars.append(float(star))
    return ReviewSet(ids=ids, texts=texts, raw_texts=raw_texts, stars=np.array(stars))
