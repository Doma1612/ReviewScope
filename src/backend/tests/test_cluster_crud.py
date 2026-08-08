"""Endpoint-logic tests for the cluster CRUD/merge/from-selection routes (api/projects.py).

No real DB: the routes are plain async functions, so a fake ``AsyncSession`` that
serves ``get``/``execute`` from in-memory objects and records ``add``/``delete``/
``commit`` exercises the wiring directly. Editing is sentence-unit only, so every
route first loads the project and requires ``unit == "sentence"``; membership lives
on *segments*, so merge/junk/from-selection move ``Segment`` rows. ``recompute_clusters``
/ ``recompute_document_primary`` / ``_sample_docs`` are DB- and ML-heavy and are
monkeypatched — B2 already covers aggregation.

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
from app.models import Cluster, ClusterEdit, Project, ProjectMember, ProjectRole, Segment  # noqa: E402
from app.schemas import ClusterCreate, ClusterFromSegments, ClusterMerge, ClusterUpdate  # noqa: E402


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

    def __init__(self, *, member=None, objects=None, query_segs=None):
        self.member = member                  # what require_project_role finds
        self.objects = objects or {}          # (Model, id) -> instance for .get
        self.query_segs = query_segs or []    # rows for the select(Segment)
        self.added: list = []
        self.deleted: list = []
        self.committed = False
        self.refreshed: list = []

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        if entity is ProjectMember:
            return _FakeResult(scalar=self.member)
        if entity is Segment:
            return _FakeResult(items=self.query_segs)
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
        # Mirror the DB server_default so ClusterRead serialization sees an int.
        if isinstance(obj, Cluster) and getattr(obj, "n_mentions", None) is None:
            obj.n_mentions = 0


@pytest.fixture(autouse=True)
def _stub_helpers(monkeypatch):
    """Record recompute_clusters as (project_id, {cluster_ids}); stub the
    document-primary refresh and _sample_docs (heavy tuple query)."""
    calls: list = []

    async def _fake_recompute(db, project_id, cluster_ids, **kwargs):
        calls.append((project_id, set(cluster_ids)))
        return []

    async def _fake_primary(db, project_id, document_ids, **kwargs):
        return None

    async def _fake_sample(db, project_id, cluster_id, limit=3, *, sentence=False):
        return []

    monkeypatch.setattr(projects, "recompute_clusters", _fake_recompute)
    monkeypatch.setattr(projects, "recompute_document_primary", _fake_primary)
    monkeypatch.setattr(projects, "_sample_docs", _fake_sample)
    return calls


def _owner():
    return SimpleNamespace(id=uuid.uuid4())


def _member(role=ProjectRole.owner):
    return ProjectMember(role=role)


def _project(pid, unit="sentence"):
    return Project(id=pid, name="p", owner_id=uuid.uuid4(), unit=unit)


def _edits(session):
    return [o for o in session.added if isinstance(o, ClusterEdit)]


def _seg(pid, cluster_id, doc_id=None):
    return Segment(
        id=uuid.uuid4(), project_id=pid, document_id=doc_id or uuid.uuid4(),
        segment_key=f"{uuid.uuid4()}#0", ordinal=0, text="t", cluster_id=cluster_id,
        umap_x=0.0, umap_y=0.0,
    )


def _cluster(cid, pid, label="x", label_source="terms_fallback"):
    # SQLAlchemy column defaults only apply at flush, so set the JSON/int defaults
    # explicitly — the route serializes the returned cluster to ClusterRead.
    return Cluster(id=cid, project_id=pid, label=label, summary="", label_source=label_source, top_terms=[], word_frequencies={}, size=0, n_mentions=0)


# ── Create ───────────────────────────────────────────────────────────────────

@asyncio_test
async def test_create_cluster_records_edit(_stub_helpers):
    pid = uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid)})

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


@asyncio_test
async def test_create_cluster_document_unit_frozen_409(_stub_helpers):
    pid = uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid, unit="document")})
    with pytest.raises(HTTPException) as exc:
        await projects.create_cluster(pid, ClusterCreate(label="x"), db, _owner())
    assert exc.value.status_code == 409
    assert not db.committed


# ── Merge ────────────────────────────────────────────────────────────────────

@asyncio_test
async def test_merge_moves_mentions_deletes_sources_records_edits(_stub_helpers):
    pid = uuid.uuid4()
    s1, s2, target = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    seg1, seg2 = _seg(pid, s1), _seg(pid, s2)
    src1 = _cluster(s1, pid, label="a")
    src2 = _cluster(s2, pid, label="b")
    tgt = _cluster(target, pid, label="t")
    db = _FakeSession(
        member=_member(),
        objects={(Project, pid): _project(pid), (Cluster, s1): src1, (Cluster, s2): src2, (Cluster, target): tgt},
        query_segs=[seg1, seg2],
    )

    result = await projects.merge_clusters(pid, ClusterMerge(source_ids=[s1, s2], target_id=target), db, _owner())

    assert result.id == target
    assert seg1.cluster_id == target and seg2.cluster_id == target
    assert src1 in db.deleted and src2 in db.deleted
    actions = {(e.action, e.cluster_id, e.target_cluster_id) for e in _edits(db)}
    assert actions == {("merge_clusters", s1, target), ("merge_clusters", s2, target)}
    assert _stub_helpers == [(pid, {target})]
    assert db.committed


@asyncio_test
async def test_merge_target_in_sources_400(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid)})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[target], target_id=target), db, _owner())
    assert exc.value.status_code == 400
    assert not db.committed


@asyncio_test
async def test_merge_unknown_source_404(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, target): _cluster(target, pid, label="t")})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[uuid.uuid4()], target_id=target), db, _owner())
    assert exc.value.status_code == 404
    assert not db.committed


@asyncio_test
async def test_merge_target_other_project_404(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    tgt = _cluster(target, uuid.uuid4(), label="t")  # other project
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, target): tgt})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[uuid.uuid4()], target_id=target), db, _owner())
    assert exc.value.status_code == 404


@asyncio_test
async def test_merge_document_unit_frozen_409(_stub_helpers):
    pid, target = uuid.uuid4(), uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid, unit="document")})
    with pytest.raises(HTTPException) as exc:
        await projects.merge_clusters(pid, ClusterMerge(source_ids=[uuid.uuid4()], target_id=target), db, _owner())
    assert exc.value.status_code == 409
    assert not db.committed


# ── From selection (of segment mentions) ────────────────────────────────────────

@asyncio_test
async def test_from_selection_creates_cluster_and_recomputes_previous_owners(_stub_helpers):
    pid = uuid.uuid4()
    old_a, old_b = uuid.uuid4(), uuid.uuid4()
    seg1, seg2, seg3 = _seg(pid, old_a), _seg(pid, old_b), _seg(pid, None)  # seg3 = noise mention
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid)}, query_segs=[seg1, seg2, seg3])

    result = await projects.cluster_from_selection(
        pid, ClusterFromSegments(segment_ids=[seg1.id, seg2.id, seg3.id], label="New theme"), db, _owner()
    )

    created = [o for o in db.added if isinstance(o, Cluster)][0]
    assert result.id == created.id and result.label == "New theme" and result.label_source == "hitl_override"
    assert seg1.cluster_id == created.id and seg2.cluster_id == created.id and seg3.cluster_id == created.id
    edit = _edits(db)[0]
    assert edit.action == "create_from_selection" and edit.cluster_id == created.id
    assert edit.payload == {"segment_ids": [str(seg1.id), str(seg2.id), str(seg3.id)]}
    # New cluster + both previous owners recomputed; the None (noise) owner is skipped.
    assert _stub_helpers == [(pid, {created.id, old_a, old_b})]
    assert db.committed


# ── Patch (rename / approve / junk) ────────────────────────────────────────────

@asyncio_test
async def test_patch_rename_sets_override(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid, label="old")
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, cid): cluster})

    result = await projects.update_cluster(pid, cid, ClusterUpdate(label="new"), db, _owner())

    assert result.label == "new" and result.label_source == "hitl_override"
    edit = _edits(db)[0]
    assert edit.action == "rename_label" and edit.cluster_id == cid and edit.new_label == "new"
    assert edit.payload == {"before": "old"}  # old label captured for undo
    assert cluster not in db.deleted and db.committed


@asyncio_test
async def test_patch_approve_sets_approved(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid)
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, cid): cluster})

    result = await projects.update_cluster(pid, cid, ClusterUpdate(approve=True), db, _owner())

    assert result.label_source == "hitl_approved"
    assert _edits(db)[0].action == "approve_label"


@asyncio_test
async def test_patch_mark_junk_nulls_mentions_and_deletes(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid)
    seg1, seg2 = _seg(pid, cid), _seg(pid, cid)
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, cid): cluster}, query_segs=[seg1, seg2])

    result = await projects.update_cluster(pid, cid, ClusterUpdate(mark_junk=True), db, _owner())

    assert result is None
    assert seg1.cluster_id is None and seg2.cluster_id is None
    assert cluster in db.deleted
    assert _edits(db)[0].action == "mark_junk"
    assert db.committed


@asyncio_test
async def test_patch_no_fields_400(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid)
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, cid): cluster})
    with pytest.raises(HTTPException) as exc:
        await projects.update_cluster(pid, cid, ClusterUpdate(), db, _owner())
    assert exc.value.status_code == 400
    assert not db.committed


@asyncio_test
async def test_patch_unknown_cluster_404(_stub_helpers):
    pid = uuid.uuid4()
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid)})
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
async def test_delete_cluster_nulls_mentions_and_deletes(_stub_helpers):
    pid, cid = uuid.uuid4(), uuid.uuid4()
    cluster = _cluster(cid, pid)
    seg = _seg(pid, cid)
    db = _FakeSession(member=_member(), objects={(Project, pid): _project(pid), (Cluster, cid): cluster}, query_segs=[seg])

    await projects.delete_cluster(pid, cid, db, _owner())

    assert seg.cluster_id is None and cluster in db.deleted
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
