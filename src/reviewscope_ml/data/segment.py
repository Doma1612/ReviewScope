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
    return segment_id.rsplit("#", 1)[0]


def segment_reviews(
    reviews: ReviewSet, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS
) -> ReviewSet:
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
