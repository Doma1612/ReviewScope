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


def _parse_rating(raw_data: dict[str, Any] | None, rating_col: str | None) -> float | None:
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
    # Deduplicate review ratings so multi-segment reviews cannot dominate mean_stars.
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
    counter: Counter = Counter()
    for text in texts:
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        counter.update(word for word in words if len(word) > 3 and word not in _STOPWORDS)
    terms = counter.most_common(20)
    top_terms = [{"term": term, "score": count} for term, count in terms[:10]]
    return top_terms, dict(terms)


def _terms_and_frequencies(texts: list[str]) -> tuple[list[dict], dict]:
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


async def recompute_cluster(
    db: AsyncSession,
    project_id: uuid.UUID,
    cluster_id: uuid.UUID,
    *,
    delete_if_empty: bool = False,
) -> Cluster | None:
    # The caller owns the transaction and commit.
    cluster = await db.get(Cluster, cluster_id)
    if cluster is None or cluster.project_id != project_id:
        return None

    rating_col = await _rating_column(db, project_id)
    if await _is_sentence_unit(db, project_id):
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
    # Refresh derived primary clusters after segment membership changes; the caller commits.
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
    updated: list[Cluster] = []
    for cluster_id in dict.fromkeys(cluster_ids):
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
