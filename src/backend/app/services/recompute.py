"""Recompute a cluster's aggregates from its current membership.

"Append + derive, don't patch": rather than incrementally
nudging a cluster's stored stats on every move, we recompute them from the
documents currently assigned to it, so the app and the notebook agree. The
``top_terms`` / ``word_frequencies`` reuse the very functions the pipeline uses
(:mod:`reviewscope_ml.represent.terms`); those are imported lazily so this module
stays free of the heavy ML deps at import time (the backend venv has no sklearn).

The simple numeric aggregates (``size`` / ``sentiment_avg`` / ``mean_stars``) are
factored into pure helpers below so they can be unit-tested without a database or
the ML stack — see ``tests/test_recompute.py``.
"""
from __future__ import annotations

import uuid
from collections import Counter
from statistics import fmean
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ml_mapping import derive_roles
from app.models import Cluster, Document, Embedding, Project, ProjectSchema, Segment
from app.services.metrics import cohesion_score


# ── Pure aggregate helpers (no DB, no ML deps) ─────────────────────────────────

def _parse_rating(raw_data: dict[str, Any] | None, rating_col: str | None) -> float | None:
    """Pull a numeric rating out of a document's ``raw_data``.

    Ratings live in the original upload columns, so the value may be a string
    ("4") or genuinely absent. Returns ``None`` for anything non-numeric (booleans
    included — ``True`` is not a 1-star rating)."""
    if not raw_data or not rating_col:
        return None
    value = raw_data.get(rating_col)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def numeric_aggregates(
    sentiment_scores: list[float | None],
    ratings: list[float | None],
) -> dict[str, Any]:
    """``size`` / ``sentiment_avg`` / ``mean_stars`` for one cluster's members.

    ``size`` is the member count (one entry per doc, ``None`` allowed); the means
    ignore ``None`` and are themselves ``None`` when nothing numeric remains."""
    sentiments = [s for s in sentiment_scores if s is not None]
    stars = [r for r in ratings if r is not None]
    return {
        "size": len(sentiment_scores),
        "sentiment_avg": _mean(sentiments),
        "mean_stars": _mean(stars),
    }


def segment_aggregates(
    document_ids: list[Any],
    sentiment_scores: list[float | None],
    ratings_by_document: dict[Any, float | None],
) -> dict[str, Any]:
    """Aggregates for a sentence-unit cluster from its member *segments*.

    ``size`` is the count of distinct parent reviews (the headline "customers"
    number); ``n_mentions`` is the raw segment count. ``sentiment_avg`` is over
    every segment, but ``mean_stars`` is deduped to one rating per distinct review
    so a rambling multi-segment review can't dominate the star profile (mirrors the
    runner's ``_dedup_parent_stats``)."""
    distinct = list(dict.fromkeys(document_ids))
    sentiments = [s for s in sentiment_scores if s is not None]
    stars = [ratings_by_document[d] for d in distinct if ratings_by_document.get(d) is not None]
    return {
        "size": len(distinct),
        "n_mentions": len(document_ids),
        "sentiment_avg": _mean(sentiments),
        "mean_stars": _mean(stars),
    }


_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your", "but", "not"}


def _simple_terms(texts: list[str]) -> tuple[list[dict], dict]:
    """Lightweight bag-of-words terms — the sim-mode fallback for the c-TF-IDF path.

    Mirrors the simulated pipeline's tokenizer (``tasks._terms``) so an edited
    cluster's ``top_terms`` / ``word_frequencies`` keep the same shape and feel as
    the ones the simulated run produced, without importing the heavy ML stack."""
    counter: Counter = Counter()
    for text in texts:
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        counter.update(word for word in words if len(word) > 3 and word not in _STOPWORDS)
    terms = counter.most_common(20)
    top_terms = [{"term": term, "score": count} for term, count in terms[:10]]
    return top_terms, dict(terms)


def _terms_and_frequencies(texts: list[str]) -> tuple[list[dict], dict]:
    """Top c-TF-IDF terms + within-cluster word counts for one cluster.

    All docs belong to this single cluster, so they are labelled ``0`` and the
    ``0`` entry is read back out. Heavy ML imports happen here, lazily. In
    simulated mode we skip them entirely and use :func:`_simple_terms`, so the
    cluster-edit features work in a lightweight env with no sklearn installed."""
    if not texts:
        return [], {}
    if get_settings().simulate_ml:
        return _simple_terms(texts)
    import numpy as np
    from reviewscope_ml.represent.terms import ctfidf_terms, word_frequencies

    labels = np.zeros(len(texts), dtype=int)
    terms = ctfidf_terms(texts, labels).get(0, [])
    freqs = word_frequencies(texts, labels).get(0, {})
    top_terms = [{"term": term, "score": score} for term, score in terms]
    return top_terms, freqs


# ── DB-bound recompute ─────────────────────────────────────────────────────────

async def recompute_cluster(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    *,
    delete_if_empty: bool = False,
) -> Cluster | None:
    """Recompute one cluster's aggregates from its current membership.

    Returns the updated :class:`Cluster`, or ``None`` if the cluster was missing,
    not part of ``project_id``, or empty and deleted (``delete_if_empty``). The
    caller owns the surrounding transaction/commit."""
    cluster = await db.get(Cluster, cluster_id)
    if cluster is None or cluster.project_id != project_id:
        return None

    rating_col = await _rating_column(db, project_id)
    if await _is_sentence_unit(db, project_id):
        # Sentence unit: aggregate over the cluster's member *segments*; size is
        # distinct parent reviews, mean_stars deduped per review.
        rows = (
            await db.execute(
                select(
                    Segment.text,
                    Segment.sentiment_score,
                    Segment.document_id,
                    Segment.vector,
                    Document.raw_data,
                )
                .join(Document, Document.id == Segment.document_id)
                .where(
                    Segment.project_id == project_id,
                    Segment.cluster_id == cluster_id,
                )
            )
        ).all()

        if not rows and delete_if_empty:
            await db.delete(cluster)
            return None

        texts = [text for text, _, _, _, _ in rows]
        sentiments = [s for _, s, _, _, _ in rows]
        document_ids = [doc_id for _, _, doc_id, _, _ in rows]
        vectors = [vec for _, _, _, vec, _ in rows if vec]
        ratings_by_document = {doc_id: _parse_rating(raw, rating_col) for _, _, doc_id, _, raw in rows}

        agg = segment_aggregates(document_ids, sentiments, ratings_by_document)
        cluster.n_mentions = agg["n_mentions"]
    else:
        # Document unit (frozen legacy projects): aggregate over documents.
        rows = (
            await db.execute(
                select(
                    Document.text,
                    Document.sentiment_score,
                    Document.raw_data,
                    Embedding.vector,
                )
                .outerjoin(Embedding, Embedding.document_id == Document.id)
                .where(
                    Document.project_id == project_id,
                    Document.cluster_id == cluster_id,
                )
            )
        ).all()

        if not rows and delete_if_empty:
            await db.delete(cluster)
            return None

        texts = [text for text, _, _, _ in rows]
        sentiments = [sentiment for _, sentiment, _, _ in rows]
        ratings = [_parse_rating(raw, rating_col) for _, _, raw, _ in rows]
        vectors = [vector for _, _, _, vector in rows if vector]
        agg = numeric_aggregates(sentiments, ratings)
        cluster.n_mentions = agg["size"]

    top_terms, freqs = _terms_and_frequencies(texts)

    cluster.size = agg["size"]
    cluster.sentiment_avg = agg["sentiment_avg"]
    cluster.mean_stars = agg["mean_stars"]
    cluster.cohesion = cohesion_score(vectors)
    cluster.top_terms = top_terms
    cluster.word_frequencies = freqs
    return cluster


async def recompute_document_primary(
    db: AsyncSession, project_id: uuid.UUID, document_ids: list[uuid.UUID]
) -> None:
    """Refresh each review's derived "primary" cluster from its segments.

    ``documents.cluster_id`` is a display convenience for sentence-unit projects
    (the plurality non-noise cluster among the review's mentions; ``None`` when the
    review has no clustered mention). Call after any segment membership change. The
    caller owns the transaction."""
    for document_id in dict.fromkeys(document_ids):
        rows = (
            await db.execute(
                select(Segment.cluster_id).where(
                    Segment.project_id == project_id,
                    Segment.document_id == document_id,
                    Segment.cluster_id.is_not(None),
                )
            )
        ).all()
        counts = Counter(cid for (cid,) in rows)
        primary = counts.most_common(1)[0][0] if counts else None
        doc = await db.get(Document, document_id)
        if doc is not None and doc.project_id == project_id:
            doc.cluster_id = primary


async def _is_sentence_unit(db: AsyncSession, project_id: uuid.UUID) -> bool:
    project = await db.get(Project, project_id)
    return bool(project is not None and project.unit == "sentence")


async def recompute_clusters(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_ids: list[uuid.UUID],
    *,
    delete_if_empty: bool = False,
) -> list[Cluster]:
    """Recompute several clusters; returns those still present afterwards."""
    updated: list[Cluster] = []
    for cluster_id in dict.fromkeys(cluster_ids):  # de-dup, keep order
        cluster = await recompute_cluster(
            db, project_id, cluster_id, delete_if_empty=delete_if_empty
        )
        if cluster is not None:
            updated.append(cluster)
    return updated


async def _rating_column(db: AsyncSession, project_id: uuid.UUID) -> str | None:
    schema = await db.get(ProjectSchema, project_id)
    if schema is None:
        return None
    _text_col, rating_col = derive_roles(schema.columns)
    return rating_col
