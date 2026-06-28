"""Unit tests for GET /{project_id}/embeddings (B7 rich hover payload).

Mirrors the lightweight fake-session approach used by test_cluster_crud.py: the
route function is driven directly with an in-memory session so we can assert the
row→EmbeddingPoint transformation (snippet capping, noise→null label) without a DB.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("SIMULATE_ML", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.api import projects  # noqa: E402
from app.models import Document, ProjectMember, ProjectRole  # noqa: E402


def asyncio_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class _FakeResult:
    def __init__(self, *, scalar=None, items=None):
        self._scalar = scalar
        self._items = items or []

    def scalar_one_or_none(self):
        return self._scalar

    def all(self):
        return self._items


class _FakeSession:
    """Answers the role check, then serves the embeddings join rows."""

    def __init__(self, *, member=None, rows=None):
        self.member = member
        self.rows = rows or []

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        if entity is ProjectMember:
            return _FakeResult(scalar=self.member)
        if entity is Document:
            return _FakeResult(items=self.rows)
        raise AssertionError(f"unexpected query for {entity!r}")


def _row(*, cluster_id=None, text="hello", pk="1", sentiment=0.5, label=None):
    # Tuple order matches the select() in the embeddings route.
    return (uuid.uuid4(), cluster_id, 1.0, 2.0, 3.0, text, pk, sentiment, label)


@asyncio_test
async def test_embeddings_populates_rich_fields():
    pid = uuid.uuid4()
    cid = uuid.uuid4()
    long_text = "x" * 200
    session = _FakeSession(
        member=ProjectMember(role=ProjectRole.viewer),
        rows=[
            _row(cluster_id=cid, text=long_text, pk="A1", sentiment=0.9, label="Praise"),
            _row(cluster_id=None, text="noisy", pk="B2", sentiment=-0.1, label=None),
        ],
    )
    points = await projects.embeddings(pid, db=session, current_user=SimpleNamespace(id=uuid.uuid4()))

    clustered, noise = points
    assert len(clustered.snippet) == 120  # capped server-side
    assert clustered.primary_key_value == "A1"
    assert clustered.sentiment_score == 0.9
    assert clustered.cluster_label == "Praise"

    assert noise.cluster_id is None
    assert noise.cluster_label is None  # noise points carry no label
    assert noise.snippet == "noisy"


@asyncio_test
async def test_embeddings_snippet_none_when_text_missing():
    session = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        rows=[_row(text=None)],
    )
    (point,) = await projects.embeddings(uuid.uuid4(), db=session, current_user=SimpleNamespace(id=uuid.uuid4()))
    assert point.snippet is None
