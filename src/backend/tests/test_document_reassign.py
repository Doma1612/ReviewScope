"""Endpoint-logic tests for the document reassignment routes (api/projects.py).

No real DB: the routes are plain async functions, so a fake ``AsyncSession`` that
serves ``get``/``execute`` from in-memory objects and records ``add``/``commit``
exercises the wiring directly. ``recompute_clusters`` is DB- and ML-heavy (sklearn,
absent from the backend venv), so it is monkeypatched to just capture the affected
cluster ids — B2 already tests its aggregation. Mirrors the no-DB approach of
test_cluster_edits.py / test_recompute.py.

Run from the backend dir:  python -m pytest tests/test_document_reassign.py
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("SIMULATE_ML", "true")
# Importing app.api.projects pulls in get_settings(), which mkdir()s upload_dir
# (default /workspace/...). Point it at a writable temp dir for the test run.
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="reviewscope-test-uploads-"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from fastapi import HTTPException  # noqa: E402

from app.api import projects  # noqa: E402
from app.models import Cluster, ClusterEdit, Document, Project, ProjectMember, ProjectRole, Segment  # noqa: E402
from app.schemas import BulkReassign, ReviewReassign, SegmentReassign  # noqa: E402


# The backend venv has no pytest-asyncio; run coroutine tests on a fresh loop
# while keeping fixture injection (functools.wraps exposes the original signature).
def asyncio_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, *, scalar=None, items=None):
        self._scalar = scalar
        self._items = items or []

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._items


class _FakeSession:
    """Serves get()/execute() from in-memory state; records mutations.

    Sentence-unit editing moves *segments*: the review-level endpoints select the
    review's segments (``query_segs``) and, for the bulk path, the reviews
    (``query_docs``)."""

    def __init__(self, *, member=None, objects=None, query_docs=None, query_segs=None):
        self.member = member                  # what require_project_role finds
        self.objects = objects or {}          # (Model, id) -> instance for .get
        self.query_docs = query_docs or []    # rows for the bulk select(Document)
        self.query_segs = query_segs or []    # rows for select(Segment)
        self.added: list = []
        self.committed = False
        self.refreshed: list = []

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        if entity is ProjectMember:
            return _FakeResult(scalar=self.member)
        if entity is Segment:
            return _FakeResult(items=self.query_segs)
        if entity is Document:
            return _FakeResult(items=self.query_docs)
        raise AssertionError(f"unexpected query for {entity!r}")

    async def get(self, model, pk):
        return self.objects.get((model, pk))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        self.refreshed.append(obj)


def _sentence_project(pid):
    return Project(id=pid, name="p", owner_id=uuid.uuid4(), unit="sentence")


@pytest.fixture
def capture_recompute(monkeypatch):
    """Record recompute_clusters calls as (project_id, {cluster_ids}); stub the
    document-primary refresh and membership read so the endpoint's own segment
    moves + edit-log wiring are what's under test."""
    calls: list = []

    async def _fake(db, project_id, cluster_ids, **kwargs):
        calls.append((project_id, set(cluster_ids)))
        return []

    async def _fake_primary(db, project_id, document_ids, **kwargs):
        return None

    async def _fake_memberships(db, project_id, document_ids):
        return {}

    monkeypatch.setattr(projects, "recompute_clusters", _fake)
    monkeypatch.setattr(projects, "recompute_document_primary", _fake_primary)
    monkeypatch.setattr(projects, "_document_memberships", _fake_memberships)
    return calls


def _owner():
    return SimpleNamespace(id=uuid.uuid4())


def _edits(session):
    return [o for o in session.added if isinstance(o, ClusterEdit)]


def _seg(pid, doc_id, key, cluster_id):
    return Segment(id=uuid.uuid4(), project_id=pid, document_id=doc_id, segment_key=key, ordinal=0, text="t", cluster_id=cluster_id)


# ── Review-level reassign (PATCH /documents/{id}: move all a review's mentions) ──

@asyncio_test
async def test_reassign_review_moves_all_mentions_and_records_edit(capture_recompute):
    pid = uuid.uuid4()
    old, other, new = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    doc = Document(id=uuid.uuid4(), project_id=pid, cluster_id=old, primary_key_value="1", text="t", raw_data={})
    seg1 = _seg(pid, doc.id, "1#0", old)
    seg2 = _seg(pid, doc.id, "1#1", other)   # a review can span clusters
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={
            (Project, pid): _sentence_project(pid),
            (Document, doc.id): doc,
            (Cluster, new): Cluster(id=new, project_id=pid, label="x", summary=""),
        },
        query_segs=[seg1, seg2],
    )

    result = await projects.reassign_document(pid, doc.id, ReviewReassign(cluster_id=new), db, _owner())

    assert result.id == doc.id
    assert seg1.cluster_id == new and seg2.cluster_id == new  # every mention moved
    assert db.committed
    edit = _edits(db)[0]
    assert edit.action == "reassign_review"
    assert edit.document_id == doc.id and edit.cluster_id == old and edit.target_cluster_id == new
    # Both source clusters + the target get recomputed.
    assert capture_recompute == [(pid, {old, other, new})]


@asyncio_test
async def test_reassign_review_to_noise_recomputes_only_old(capture_recompute):
    pid = uuid.uuid4()
    old = uuid.uuid4()
    doc = Document(id=uuid.uuid4(), project_id=pid, cluster_id=old, primary_key_value="1", text="t", raw_data={})
    seg = _seg(pid, doc.id, "1#0", old)
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={(Project, pid): _sentence_project(pid), (Document, doc.id): doc},
        query_segs=[seg],
    )

    await projects.reassign_document(pid, doc.id, ReviewReassign(cluster_id=None), db, _owner())

    assert seg.cluster_id is None
    assert _edits(db)[0].target_cluster_id is None
    assert capture_recompute == [(pid, {old})]  # None target is skipped


@asyncio_test
async def test_reassign_viewer_forbidden(capture_recompute):
    pid = uuid.uuid4()
    db = _FakeSession(member=ProjectMember(role=ProjectRole.viewer))

    with pytest.raises(HTTPException) as exc:
        await projects.reassign_document(pid, uuid.uuid4(), ReviewReassign(cluster_id=None), db, _owner())

    assert exc.value.status_code == 403
    assert not db.committed and capture_recompute == []


@asyncio_test
async def test_reassign_document_unit_project_frozen_409(capture_recompute):
    pid = uuid.uuid4()
    doc = Document(id=uuid.uuid4(), project_id=pid, cluster_id=None, primary_key_value="1", text="t", raw_data={})
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={(Project, pid): Project(id=pid, name="p", owner_id=uuid.uuid4(), unit="document"), (Document, doc.id): doc},
    )

    with pytest.raises(HTTPException) as exc:
        await projects.reassign_document(pid, doc.id, ReviewReassign(cluster_id=None), db, _owner())

    assert exc.value.status_code == 409  # document-unit projects are read-only
    assert not db.committed and capture_recompute == []


@asyncio_test
async def test_reassign_unknown_document_404(capture_recompute):
    pid = uuid.uuid4()
    db = _FakeSession(member=ProjectMember(role=ProjectRole.owner), objects={(Project, pid): _sentence_project(pid)})

    with pytest.raises(HTTPException) as exc:
        await projects.reassign_document(pid, uuid.uuid4(), ReviewReassign(cluster_id=None), db, _owner())

    assert exc.value.status_code == 404


@asyncio_test
async def test_reassign_target_cluster_in_other_project_404(capture_recompute):
    pid = uuid.uuid4()
    new = uuid.uuid4()
    doc = Document(id=uuid.uuid4(), project_id=pid, cluster_id=None, primary_key_value="1", text="t", raw_data={})
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={
            (Project, pid): _sentence_project(pid),
            (Document, doc.id): doc,
            (Cluster, new): Cluster(id=new, project_id=uuid.uuid4(), label="x", summary=""),  # other project
        },
    )

    with pytest.raises(HTTPException) as exc:
        await projects.reassign_document(pid, doc.id, ReviewReassign(cluster_id=new), db, _owner())

    assert exc.value.status_code == 404


# ── Bulk review reassign (POST: move every mention of each listed review) ───────

@asyncio_test
async def test_bulk_reassign_moves_all_mentions_and_unions_affected(capture_recompute):
    pid = uuid.uuid4()
    a, b, target = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    d1 = Document(id=uuid.uuid4(), project_id=pid, cluster_id=a, primary_key_value="1", text="t", raw_data={})
    d2 = Document(id=uuid.uuid4(), project_id=pid, cluster_id=b, primary_key_value="2", text="t", raw_data={})
    seg1 = _seg(pid, d1.id, "1#0", a)
    seg2 = _seg(pid, d2.id, "2#0", b)
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={(Project, pid): _sentence_project(pid), (Cluster, target): Cluster(id=target, project_id=pid, label="x", summary="")},
        query_docs=[d1, d2],
        query_segs=[seg1, seg2],
    )

    result = await projects.bulk_reassign_documents(
        pid, BulkReassign(document_ids=[d1.id, d2.id], cluster_id=target), db, _owner()
    )

    assert result.moved == 2
    assert seg1.cluster_id == target and seg2.cluster_id == target
    edit = _edits(db)[0]
    assert edit.action == "bulk_reassign" and edit.target_cluster_id == target
    assert edit.payload == {
        "document_ids": [str(d1.id), str(d2.id)],
        "before": {str(d1.id): str(a), str(d2.id): str(b)},
    }
    # Union of the reviews' old clusters (a, b) + the target.
    assert capture_recompute == [(pid, {a, b, target})]


@asyncio_test
async def test_bulk_reassign_viewer_forbidden(capture_recompute):
    db = _FakeSession(member=ProjectMember(role=ProjectRole.viewer))

    with pytest.raises(HTTPException) as exc:
        await projects.bulk_reassign_documents(
            uuid.uuid4(), BulkReassign(document_ids=[uuid.uuid4()], cluster_id=None), db, _owner()
        )

    assert exc.value.status_code == 403
    assert not db.committed and capture_recompute == []


@asyncio_test
async def test_bulk_reassign_document_unit_project_frozen_409(capture_recompute):
    pid = uuid.uuid4()
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={(Project, pid): Project(id=pid, name="p", owner_id=uuid.uuid4(), unit="document")},
    )

    with pytest.raises(HTTPException) as exc:
        await projects.bulk_reassign_documents(
            pid, BulkReassign(document_ids=[uuid.uuid4()], cluster_id=None), db, _owner()
        )

    assert exc.value.status_code == 409
    assert not db.committed and capture_recompute == []


# ── Segment-level reassign (PATCH /segments/{id}: move a single mention) ─────────

@asyncio_test
async def test_reassign_segment_moves_one_mention_and_records_edit(capture_recompute):
    pid = uuid.uuid4()
    old, new = uuid.uuid4(), uuid.uuid4()
    doc_id = uuid.uuid4()
    seg = Segment(id=uuid.uuid4(), project_id=pid, document_id=doc_id, segment_key="1#0", ordinal=0, text="t", cluster_id=old, umap_x=0.0, umap_y=0.0)
    db = _FakeSession(
        member=ProjectMember(role=ProjectRole.owner),
        objects={
            (Project, pid): _sentence_project(pid),
            (Segment, seg.id): seg,
            (Cluster, new): Cluster(id=new, project_id=pid, label="x", summary=""),
        },
    )

    result = await projects.reassign_segment(pid, seg.id, SegmentReassign(cluster_id=new), db, _owner())

    assert seg.cluster_id == new
    assert result.segment_id == seg.id and result.cluster_id == new
    edit = _edits(db)[0]
    assert edit.action == "reassign_segment"
    assert edit.segment_id == seg.id and edit.cluster_id == old and edit.target_cluster_id == new
    assert capture_recompute == [(pid, {old, new})]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
