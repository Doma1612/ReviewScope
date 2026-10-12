from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.recompute import _parse_rating, numeric_aggregates, segment_aggregates  # noqa: E402


def test_segment_aggregates_counts_reviews_and_dedups_stars():
    document_ids = ["r1", "r1", "r1", "r2"]
    sentiments = [0.4, 0.6, None, -0.2]
    ratings_by_document = {"r1": 5.0, "r2": 1.0}
    agg = segment_aggregates(document_ids, sentiments, ratings_by_document)
    assert agg["size"] == 2
    assert agg["n_mentions"] == 4
    assert agg["sentiment_avg"] == pytest.approx((0.4 + 0.6 - 0.2) / 3)
    assert agg["mean_stars"] == pytest.approx(3.0)


def test_segment_aggregates_empty():
    agg = segment_aggregates([], [], {})
    assert agg == {"size": 0, "n_mentions": 0, "sentiment_avg": None, "mean_stars": None}


def test_numeric_aggregates_basic():
    agg = numeric_aggregates([0.5, 0.1, None], [5.0, 4.0, 1.0])
    assert agg["size"] == 3
    assert agg["sentiment_avg"] == pytest.approx(0.3)
    assert agg["mean_stars"] == pytest.approx(10 / 3)


def test_aggregates_update_after_moving_a_doc():
    a_sent, a_rate = [0.4, 0.6, 0.8], [5.0, 4.0, 3.0]
    b_sent, b_rate = [-0.2], [1.0]

    a_before = numeric_aggregates(a_sent, a_rate)
    assert a_before["size"] == 3
    assert a_before["sentiment_avg"] == pytest.approx(0.6)

    a_sent.pop(); a_rate.pop()
    b_sent.append(0.8); b_rate.append(3.0)

    a_after = numeric_aggregates(a_sent, a_rate)
    b_after = numeric_aggregates(b_sent, b_rate)

    assert a_after["size"] == 2 and b_after["size"] == 2
    assert a_after["sentiment_avg"] == pytest.approx(0.5)
    assert b_after["sentiment_avg"] == pytest.approx(0.3)
    assert b_after["mean_stars"] == pytest.approx(2.0)


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
    assert _parse_rating({"stars": "4"}, "stars") == 4.0
    assert _parse_rating({"stars": "n/a"}, "stars") is None
    assert _parse_rating({"stars": True}, "stars") is None
    assert _parse_rating({"stars": 5}, None) is None
    assert _parse_rating(None, "stars") is None
    assert _parse_rating({}, "stars") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
