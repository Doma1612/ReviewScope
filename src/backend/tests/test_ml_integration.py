"""
Pure-logic tests for the reviewscope_ml ↔ backend mapping (app/ml_mapping.py).

No DB, no GPU, no model downloads, and no real reviewscope_ml run — per AGENTS.md
the ML integration is exercised against lightweight stand-ins, not by invoking the
ML source. We only need SQLAlchemy (to build ORM instances in memory) and the
backend app package on the path.

Run from the backend dir:  python -m pytest tests/test_ml_integration.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

# app.ml_mapping imports app.models -> app.core.config is NOT touched, but other
# backend modules require these env vars at import; set safe values up front.
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.ml_mapping import DbProgressSink, STEP_ORDER, derive_roles, result_to_orm  # noqa: E402
from app.models import PipelineStepStatus  # noqa: E402


# ── Lightweight stand-ins for the reviewscope_ml DTOs (duck-typed) ─────────────

@dataclass
class FakeDoc:
    primary_key_value: str
    text: str
    raw_data: dict
    cluster_id: Optional[int]
    sentiment_score: Optional[float] = None


@dataclass
class FakeEmb:
    primary_key_value: str
    vector: list
    umap_x: float
    umap_y: float
    umap_z: Optional[float] = None


@dataclass
class FakeCluster:
    cluster_id: int
    label: str
    summary: str
    label_source: str
    top_terms: list
    word_frequencies: dict
    size: int
    sentiment_avg: Optional[float] = None
    mean_stars: Optional[float] = None
    sample_doc_ids: list = field(default_factory=list)


@dataclass
class FakeResult:
    project_id: str
    documents: list
    embeddings: list
    clusters: list


# ── derive_roles ──────────────────────────────────────────────────────────────

def test_derive_roles_prefers_named_text_and_numeric_rating():
    cols = [
        {"name": "review_id", "type": "text", "is_primary_key": True},
        {"name": "body", "type": "text"},
        {"name": "stars", "type": "integer"},
    ]
    text, rating = derive_roles(cols)
    assert text == "body"        # name hint beats the PK text column
    assert rating == "stars"


def test_derive_roles_rating_none_when_no_numeric():
    cols = [{"name": "comment", "type": "text"}, {"name": "title", "type": "text"}]
    text, rating = derive_roles(cols)
    assert text == "comment"
    assert rating is None


def test_derive_roles_falls_back_to_first_text_column():
    cols = [{"name": "id", "type": "integer"}, {"name": "freeform", "type": "text"}]
    text, _ = derive_roles(cols)
    assert text == "freeform"


# ── DbProgressSink ─────────────────────────────────────────────────────────────

class _FakeSession:
    """Captures the .step() effects without a database."""

    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt):
        self._log.append(stmt)

    def commit(self):
        self._log.append("commit")


def test_progress_sink_marks_earlier_steps_done(monkeypatch):
    calls: list = []

    # Patch the SQLAlchemy update() the sink uses to a recorder so we can assert
    # on the (steps, values) without a live engine.
    import app.ml_mapping as mapping

    recorded: list[dict] = []

    class _Stmt:
        def __init__(self):
            self._where = []
            self._values = {}

        def where(self, *args):
            self._where.extend(args)
            return self

        def values(self, **kw):
            self._values = kw
            recorded.append(self._values)
            return self

    monkeypatch.setattr(mapping, "update", lambda model: _Stmt())

    sink = DbProgressSink(lambda: _FakeSession(calls), project_id="pid")
    sink.step("Cluster", "running", index=5, total=8)

    # One update marks earlier steps done, one sets Cluster running.
    statuses = [v.get("status") for v in recorded]
    assert PipelineStepStatus.done in statuses
    assert PipelineStepStatus.running in statuses
    assert STEP_ORDER[4] == "cluster"


# ── result_to_orm ──────────────────────────────────────────────────────────────

def test_result_to_orm_resolves_clusters_and_joins_embeddings():
    pid = "11111111-1111-1111-1111-111111111111"
    result = FakeResult(
        project_id=pid,
        clusters=[
            FakeCluster(0, "Rooms", "Summary A", "ollama:llama3.2", [{"term": "bed", "score": 1.0}], {"bed": 3}, 2, -0.1, 4.2),
            FakeCluster(1, "Food", "Summary B", "terms_fallback", [], {}, 1, None, None),
        ],
        documents=[
            FakeDoc("a", "great bed", {"stars": 5}, 0, 0.5),
            FakeDoc("b", "comfy room", {"stars": 4}, 0, 0.2),
            FakeDoc("c", "tasty meal", {"stars": 5}, 1, 0.9),
            FakeDoc("d", "noise", {"stars": 1}, None, None),  # noise → NULL FK
        ],
        embeddings=[
            FakeEmb("a", [0.1, 0.2], 1.0, 2.0, 3.0),
            FakeEmb("b", [0.3, 0.4], 1.1, 2.1, 3.1),
            FakeEmb("c", [0.5, 0.6], 1.2, 2.2, 3.2),
            FakeEmb("d", [0.7, 0.8], 1.3, 2.3, 3.3),
        ],
    )

    clusters, documents, embeddings = result_to_orm(result)

    assert len(clusters) == 2 and len(documents) == 4 and len(embeddings) == 4
    by_int = {c.label: c.id for c in clusters}

    docs_by_pk = {d.primary_key_value: d for d in documents}
    assert docs_by_pk["a"].cluster_id == by_int["Rooms"]
    assert docs_by_pk["c"].cluster_id == by_int["Food"]
    assert docs_by_pk["d"].cluster_id is None          # noise stays unassigned

    # label_source / mean_stars carried through.
    rooms = next(c for c in clusters if c.label == "Rooms")
    assert rooms.label_source == "ollama:llama3.2"
    assert rooms.mean_stars == 4.2

    # Embeddings joined to documents by primary_key_value.
    emb_doc_ids = {e.document_id for e in embeddings}
    assert emb_doc_ids == {d.id for d in documents}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
