"""Endpoint-logic tests for the cluster CRUD/merge/from-selection routes (api/projects.py, WP B4).

No real DB: the routes are plain async functions, so a fake ``AsyncSession`` that
serves ``get``/``execute`` from in-memory objects and records ``add``/``delete``/
``commit`` exercises the wiring directly. ``recompute_clusters`` is DB- and ML-heavy
(sklearn, absent from the backend venv) and ``_sample_docs`` issues a real tuple
query, so both are monkeypatched — B2 already covers aggregation. Mirrors the no-DB
approach of test_document_reassign.py.

Run from the backend dir:  python -m pytest tests/test_cluster_crud.py
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
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="reviewscope-test-uploads-"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from fastapi import HTTPException  # noqa: E402

from app.api import projects  # noqa: E402
from app.models import Cluster, ClusterEdit, Document, ProjectMember, ProjectRole  # noqa: E402
from app.schemas import ClusterCreate, ClusterFromSelection, ClusterMerge, ClusterUpdate  # noqa: E402


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
    """Serves get()/execute() from in-memory state; records mutations."""

    def __init__(self, *, member=None, objects=None, query_docs=None):
        self.member = member                  # what require_project_role finds
        self.objects = objects or {}          # (Model, id) -> instance for .get
        self.query_docs = query_docs or []    # rows for the select(Document)
        self.added: list = []
        self.deleted: list = []
        self.committed = False
        self.refreshed: list = []

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        if entity is ProjectMember:
            return _FakeResult(scalar=self.member)
        if entity is Document:
            return _FakeResult(items=self.query_docs)
        raise AssertionError(f"unexpected query for {entity!r}")

    async def get(self, model, pk):
        return self.objects.get((model, pk))

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        self.refreshed.append(obj)


@pytest.fixture(autouse=True)
def _stub_helpers(monkeypatch):
    """Replace recompute_clusters (records affected ids) and _sample_docs (heavy query)."""
    calls: list = []

    async def _fake_recompute(db, project_id, cluster_ids, **kwargs):
        calls.append((project_id, set(cluster_ids)))
        return []

    async def _fake_sample(db, project_id, cluster_id, limit=3):
        return []

    monkeypatch.setattr(projects, "recompute_clusters", _fake_recompute)
    monkeypatch.setattr(projects, "_sample_docs", _fake_sample)
    return calls


def _owner():
    return SimpleNamespace(id=uuid.uuid4())


def _member(role=ProjectRole.owner):
    return ProjectMember(role=role)


def _edits(session):
    return [o for o in session.added if isinstance(o, ClusterEdit)]


def _doc(pid, cluster_id):
    return Document(id=uuid.uuid4(), project_id=pid, cluster_id=cluster_id, primary_key_value="1", text="t", raw_data={})


def _cluster(cid, pid, label="x", label_source="terms_fallback"):
    # SQLAlchemy column defaults only apply at flush, so set the JSON/int defaults
    # explicitly — the route serializes the returned cluster to ClusterRead.
    return Cluster(id=cid, project_id=pid, label=label, summary="", label_source=label_source, top_terms=[], word_frequencies={}, size=0)


# ── Create ───────────────────────────────────────────────────────────────────

@asyncio_test
async def test_create_cluster_records_edit(_stub_helpers):
    pid = uuid.uuid4()
    db = _FakeSession(member=_member())

    result = await projects.create_cluster(pid, ClusterCreate(label="Pricing"), db, _owner())

    assert result.label == "Pricing" and result.label_source == "hitl_override" and result.size == 0
    created = [o for o in db.added if isinstance(o, Cluster)][0]
    assert created.summary == "" and created.top_terms == [] and created.word_frequencies == {}
    edit = _edits(db)[0]
    assert edit.action == "create_cluster" and edit.cluster_id == created.id and edit.new_label == "Pricing"
    assert db.committed


@asyncio_test
async def test_create_cluster_viewer_forbidden(_stub_helpers):
    db = _FakeSession(member=_member(ProjectRole.viewer))
    with pytest.raises(HTTPException) as exc:
        await projects.create_cluster(uuid.uuid4(), ClusterCreate(label="x"), db, _owner())
    assert exc.value.status_code == 403
    assert not db.committed


# ── Merge ────────────────────────────────────────────────────────────────────

@asyncio_test
async def test_merge_moves_docs_deletes_sources_records_edits(_stub_helpers):
    pid = uuid.uuid4()
    s1, s2, target = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    d1, d2 = _doc(pid, s1), _doc(pid, s2)
    src1 = _cluster(s1, pid, label="a")
    src2 = _cluster(s2, pid, label="b")
    tgt = _cluster(target, pid, label="t")
    db = _FakeSession(
        member=_member(),
        objects={(Cluster, s1): src1, (Cluster, s2): src2, (Cluster, target): tgt},
        query_docs=[d1, d2],
    )

    result = await projects.merge_clusters(pid, ClusterMerge(source_ids=[s1, s2], target_id=target), db, _owner())

    assert result.id == target
    assert d1.cluster_id == target and d2.cluster_id == target
    assert src1 in db.deleted and src2 in db.deleted
    actions = {(e.action, e.cluster_id, e.target_cluster_id) for e in _edits(db)}
    assert actions == {("merge_clusters", s1, target), ("merge_clusters", s2, target)}
    assert _stub_helpers == [(pid, {target})]
    assert db.committed


@asyncio_test
async def test_merge_target_in_sources_400(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    db = _FakeSession(member=_member())
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[target], target_id=target), db, _owner())
    assert exc.value.status_code == 400
    assert not db.committed


@asyncio_test
async def test_merge_unknown_source_404(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    tgt = Cluster(id=target, project_id=pid, label="t", summary="")
    db = _FakeSession(member=_member(), objects={(Cluster, target): tgt})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[uuid.uuid4()], target_id=target), db, _owner())
    assert exc.value.status_code == 404
    assert not db.committed


@asyncio_test
async def test_merge_target_other_project_404(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    tgt = Cluster(id=target, project_id=uuid.uuid4(), label="t", summary="")
    db = _FakeSession(member=_member(), objects={(Cluster, target): tgt})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[uuid.uuid4()], target_id=target), db, _owner())
    assert exc.value.status_code == 404


# ── From selection ─────────────────────────────────────────────────────────────

@asyncio_test
async def test_from_selection_creates_cluster_and_recomputes_previous_owners(_stub_helpers):
    pid = uuid.uuid4()
    old_a, old_b = uuid.uuid4(), uuid.uuid4()
    d1, d2, d3 = _doc(pid, old_a), _doc(pid, old_b), _doc(pid, None)
    db = _FakeSession(member=_member(), query_docs=[d1, d2, d3])

    result = await projects.cluster_from_selection(
        pid, ClusterFromSelection(document_ids=[d1.id, d2.id, d3.id], label="New theme"), db, _owner()
    )

    created = [o for o in db.added if isinstance(o, Cluster)][0]
    assert result.id == created.id and result.label == "New theme" and result.label_source == "hitl_override"
    assert d1.cluster_id == created.id and d2.cluster_id == created.id and d3.cluster_id == created.id
    edit = _edits(db)[0]
    assert edit.action == "create_from_selection" and edit.cluster_id == created.id
    assert edit.payload == {"document_ids": [str(d1.id), str(d2.id), str(d3.id)]}
    # New cluster + both previous owners recomputed; the None (noise) owner is skipped.
    assert _stub_helpers == [(pid, {created.id, old_a, old_b})]
    assert db.committed


# ── Patch (rename / approve / junk) ────────────────────────────────────────────

@asyncio_test
async def test_patch_rename_sets_override(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid, label="old")
    db = _FakeSession(member=_member(), objects={(Cluster, cid): cluster})

    result = await projects.update_cluster(pid, cid, ClusterUpdate(label="new"), db, _owner())

    assert result.label == "new" and result.label_source == "hitl_override"
    edit = _edits(db)[0]
    assert edit.action == "rename_label" and edit.cluster_id == cid and edit.new_label == "new"
    assert edit.payload == {"before": "old"}  # old label captured for F7 undo
    assert cluster not in db.deleted and db.committed


@asyncio_test
async def test_patch_approve_sets_approved(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid)
    db = _FakeSession(member=_member(), objects={(Cluster, cid): cluster})

    result = await projects.update_cluster(pid, cid, ClusterUpdate(approve=True), db, _owner())

    assert result.label_source == "hitl_approved"
    assert _edits(db)[0].action == "approve_label"


@asyncio_test
async def test_patch_mark_junk_nulls_docs_and_deletes(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = Cluster(id=cid, project_id=pid, label="x", summary="")
    d1, d2 = _doc(pid, cid), _doc(pid, cid)
    db = _FakeSession(member=_member(), objects={(Cluster, cid): cluster}, query_docs=[d1, d2])

    result = await projects.update_cluster(pid, cid, ClusterUpdate(mark_junk=True), db, _owner())

    assert result is None
    assert d1.cluster_id is None and d2.cluster_id is None
    assert cluster in db.deleted
    assert _edits(db)[0].action == "mark_junk"
    assert db.committed


@asyncio_test
async def test_patch_no_fields_400(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = Cluster(id=cid, project_id=pid, label="x", summary="")
    db = _FakeSession(member=_member(), objects={(Cluster, cid): cluster})
    with pytest.raises(HTTPException) as exc:
        await projects.update_cluster(pid, cid, ClusterUpdate(), db, _owner())
    assert exc.value.status_code == 400
    assert not db.committed


@asyncio_test
async def test_patch_unknown_cluster_404(_stub_helpers):
    pid = uuid.uuid4()
    db = _FakeSession(member=_member())
    with pytest.raises(HTTPException) as exc:
        await projects.update_cluster(pid, uuid.uuid4(), ClusterUpdate(label="x"), db, _owner())
    assert exc.value.status_code == 404


@asyncio_test
async def test_patch_viewer_forbidden(_stub_helpers):
    db = _FakeSession(member=_member(ProjectRole.viewer))
    with pytest.raises(HTTPException) as exc:
        await projects.update_cluster(uuid.uuid4(), uuid.uuid4(), ClusterUpdate(label="x"), db, _owner())
    assert exc.value.status_code == 403


# ── Delete ──────────────────────────────────────────────────────────────────

@asyncio_test
async def test_delete_cluster_nulls_docs_and_deletes(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = Cluster(id=cid, project_id=pid, label="x", summary="")
    d1 = _doc(pid, cid)
    db = _FakeSession(member=_member(), objects={(Cluster, cid): cluster}, query_docs=[d1])

    await projects.delete_cluster(pid, cid, db, _owner())

    assert d1.cluster_id is None and cluster in db.deleted
    assert _edits(db)[0].action == "mark_junk" and db.committed


@asyncio_test
async def test_delete_viewer_forbidden(_stub_helpers):
    db = _FakeSession(member=_member(ProjectRole.viewer))
    with pytest.raises(HTTPException) as exc:
        await projects.delete_cluster(uuid.uuid4(), uuid.uuid4(), db, _owner())
    assert exc.value.status_code == 403
    assert not db.committed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
