"""Pure-logic tests for the cluster recompute service (app/services/recompute.py).

No DB and no ML stack: the size/sentiment/rating aggregation is factored into
pure helpers, so seeding "a couple of docs" and moving one between clusters is
just list manipulation. The c-TF-IDF / word-frequency path needs sklearn (absent
from the backend venv) and is exercised by the ML package's own tests, so it is
deliberately not imported here. Mirrors the no-DB approach of
test_ml_integration.py / test_cluster_edits.py.

Run from the backend dir:  python -m pytest tests/test_recompute.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.services.recompute import _parse_rating, numeric_aggregates, segment_aggregates  # noqa: E402


def test_segment_aggregates_counts_reviews_and_dedups_stars():
    # Cluster with 4 mentions across 2 reviews (r1 has 3, r2 has 1).
    document_ids = ["r1", "r1", "r1", "r2"]
    sentiments = [0.4, 0.6, None, -0.2]
    # One rating per distinct review (a rambling review must not dominate stars).
    ratings_by_document = {"r1": 5.0, "r2": 1.0}
    agg = segment_aggregates(document_ids, sentiments, ratings_by_document)
    assert agg["size"] == 2            # distinct reviews (the headline count)
    assert agg["n_mentions"] == 4      # segment mentions
    assert agg["sentiment_avg"] == pytest.approx((0.4 + 0.6 - 0.2) / 3)  # non-null segments
    assert agg["mean_stars"] == pytest.approx(3.0)  # (5 + 1) / 2, deduped per review


def test_segment_aggregates_empty():
    agg = segment_aggregates([], [], {})
    assert agg == {"size": 0, "n_mentions": 0, "sentiment_avg": None, "mean_stars": None}


def test_numeric_aggregates_basic():
    agg = numeric_aggregates([0.5, 0.1, None], [5.0, 4.0, 1.0])
    assert agg["size"] == 3                              # member count, Nones included
    assert agg["sentiment_avg"] == pytest.approx(0.3)   # mean of non-null sentiments
    assert agg["mean_stars"] == pytest.approx(10 / 3)


def test_aggregates_update_after_moving_a_doc():
    # Cluster A: 3 docs, Cluster B: 1 doc.
    a_sent, a_rate = [0.4, 0.6, 0.8], [5.0, 4.0, 3.0]
    b_sent, b_rate = [-0.2], [1.0]

    a_before = numeric_aggregates(a_sent, a_rate)
    assert a_before["size"] == 3
    assert a_before["sentiment_avg"] == pytest.approx(0.6)

    # Move A's last doc (sentiment 0.8 / 3 stars) into B, then recompute both.
    a_sent.pop(); a_rate.pop()
    b_sent.append(0.8); b_rate.append(3.0)

    a_after = numeric_aggregates(a_sent, a_rate)
    b_after = numeric_aggregates(b_sent, b_rate)

    assert a_after["size"] == 2 and b_after["size"] == 2
    assert a_after["sentiment_avg"] == pytest.approx(0.5)   # (0.4 + 0.6) / 2
    assert b_after["sentiment_avg"] == pytest.approx(0.3)   # (-0.2 + 0.8) / 2
    assert b_after["mean_stars"] == pytest.approx(2.0)      # (1 + 3) / 2


def test_means_are_none_when_no_numeric_values():
    agg = numeric_aggregates([None, None], [None, None])
    assert agg["size"] == 2
    assert agg["sentiment_avg"] is None
    assert agg["mean_stars"] is None


def test_empty_cluster_aggregates():
    agg = numeric_aggregates([], [])
    assert agg == {"size": 0, "sentiment_avg": None, "mean_stars": None}


def test_parse_rating_coerces_and_guards():
    assert _parse_rating({"stars": 5}, "stars") == 5.0
    assert _parse_rating({"stars": "4"}, "stars") == 4.0   # raw_data may carry strings
    assert _parse_rating({"stars": "n/a"}, "stars") is None
    assert _parse_rating({"stars": True}, "stars") is None  # bool is not a rating
    assert _parse_rating({"stars": 5}, None) is None        # no rating column
    assert _parse_rating(None, "stars") is None
    assert _parse_rating({}, "stars") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
