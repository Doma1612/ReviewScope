"""WP B2 — recompute a cluster's aggregates from its current membership.

"Append + derive, don't patch" (gap doc §3/§4a): rather than incrementally
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
from statistics import fmean
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_mapping import derive_roles
from app.models import Cluster, Document, ProjectSchema


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


def _terms_and_frequencies(texts: list[str]) -> tuple[list[dict], dict]:
    """Top c-TF-IDF terms + within-cluster word counts for one cluster.

    All docs belong to this single cluster, so they are labelled ``0`` and the
    ``0`` entry is read back out. Heavy ML imports happen here, lazily."""
    if not texts:
        return [], {}
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
    rows = (
        await db.execute(
            select(Document.text, Document.sentiment_score, Document.raw_data).where(
                Document.project_id == project_id,
                Document.cluster_id == cluster_id,
            )
        )
    ).all()

    if not rows and delete_if_empty:
        await db.delete(cluster)
        return None

    texts = [text for text, _, _ in rows]
    sentiments = [sentiment for _, sentiment, _ in rows]
    ratings = [_parse_rating(raw, rating_col) for _, _, raw in rows]

    agg = numeric_aggregates(sentiments, ratings)
    top_terms, freqs = _terms_and_frequencies(texts)

    cluster.size = agg["size"]
    cluster.sentiment_avg = agg["sentiment_avg"]
    cluster.mean_stars = agg["mean_stars"]
    cluster.top_terms = top_terms
    cluster.word_frequencies = freqs
    return cluster


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
