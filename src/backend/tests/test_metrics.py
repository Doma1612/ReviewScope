"""Pure-logic tests for the cohesion metric (app/services/metrics.py).

No DB / no ML stack — cohesion_score is plain Python over lists of floats.

Run from the backend dir:  python -m pytest tests/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.services.metrics import cohesion_score  # noqa: E402


def test_identical_vectors_are_perfectly_cohesive():
    vecs = [[1.0, 0.0, 0.0]] * 4
    assert cohesion_score(vecs) == pytest.approx(1.0)


def test_tight_cluster_beats_loose_cluster():
    tight = [[1.0, 0.1], [1.0, 0.0], [0.9, 0.1]]
    loose = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    assert cohesion_score(tight) > cohesion_score(loose)


def test_none_for_singletons_and_empty():
    assert cohesion_score([]) is None
    assert cohesion_score([[1.0, 2.0]]) is None


def test_zero_norm_vectors_are_skipped():
    # A zero vector can't be cosine-compared; it's dropped, leaving one usable
    # vector → undefined cohesion rather than a crash.
    assert cohesion_score([[0.0, 0.0], [1.0, 1.0]]) is None


def test_orthogonal_pair_is_zero_ish():
    # Centroid of two orthogonal unit vectors sits at 45°; each is cos(45°) from it.
    score = cohesion_score([[1.0, 0.0], [0.0, 1.0]])
    assert score == pytest.approx(0.70710678, rel=1e-3)
