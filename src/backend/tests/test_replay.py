"""Replay-logic tests for the re-run survival service (services/replay.py, WP B6).

No real DB / ML stack (sklearn is absent from the backend venv): replay is plain
sync code, so a fake ``Session`` that serves the three entity ``select``s from
in-memory lists and records ``add``/``delete`` exercises it directly.
``_recompute_clusters_sync`` is monkeypatched to record affected ids (B2 covers
aggregation), mirroring tests/test_cluster_crud.py.

Run from the backend dir:  python -m pytest tests/test_replay.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("SIMULATE_ML", "true")
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="reviewscope-test-uploads-"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.models import Cluster, ClusterEdit, Document  # noqa: E402
from app.services import replay as replay_mod  # noqa: E402
from app.services.replay import MembershipSnapshot, replay_edits, snapshot_membership  # noqa: E402


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return self._items


class _FakeSession:
    """Serves entity selects from in-memory lists; records add/delete/flush."""

    def __init__(self, *, docs=None, clusters=None, edits=None):
        self._docs = docs or []
        self._clusters = clusters or []
        self._edits = edits or []
        self.added: list = []
        self.deleted: list = []
        self.flushed = 0

    def execute(self, stmt):
        descs = stmt.column_descriptions
        if len(descs) >= 2:  # snapshot tuple query: (id, primary_key_value, cluster_id)
            return _FakeResult([(d.id, d.primary_key_value, d.cluster_id) for d in self._docs])
        entity = descs[0]["entity"]
        if entity is Document:
            return _FakeResult(list(self._docs))
        if entity is Cluster:
            return _FakeResult(list(self._clusters))
        if entity is ClusterEdit:
            return _FakeResult(list(self._edits))
        raise AssertionError(f"unexpected query for {entity!r}")

    def get(self, model, pk):
        if model is Cluster:
            return next((c for c in self._clusters if c.id == pk), None)
        return None

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, Cluster):
            self._clusters.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        if obj in self._clusters:
            self._clusters.remove(obj)

    def flush(self):
        self.flushed += 1


@pytest.fixture(autouse=True)
def _stub_recompute(monkeypatch):
    """Record the ids replay asks to recompute instead of touching the ML stack."""
    calls: list = []

    def _fake(session, project_id, cluster_ids):
        calls.append((project_id, set(cluster_ids)))

    monkeypatch.setattr(replay_mod, "_recompute_clusters_sync", _fake)
    return calls


# ── Builders ────────────────────────────────────────────────────────────────────

def _doc(pid, pk, cluster_id):
    return Document(id=uuid.uuid4(), project_id=pid, primary_key_value=pk, text="t", raw_data={}, cluster_id=cluster_id)


def _cluster(cid, pid, label="auto", label_source="terms_fallback"):
    return Cluster(id=cid, project_id=pid, label=label, summary="", label_source=label_source, top_terms=[], word_frequencies={}, size=0)


def _edit(pid, action, **fields):
    return ClusterEdit(project_id=pid, actor_id=uuid.uuid4(), action=action, **fields)


def _new_cluster(session):
    return next(c for c in session.added if isinstance(c, Cluster))


# ── Acceptance scenario: reassign + rename survive a re-run ──────────────────────

def test_reassign_and_rename_survive_rerun(_stub_recompute):
    """The B6 acceptance test: reassign a doc + rename a cluster, then re-run."""
    pid = uuid.uuid4()
    a_old, b_old = uuid.uuid4(), uuid.uuid4()
    # Old doc UUIDs, keyed to stable primary keys "1".."5".
    old_doc_ids = {pk: uuid.uuid4() for pk in ("1", "2", "3", "4", "5")}
    # Membership as the human left it: doc "1" was moved out of A into B.
    snapshot = MembershipSnapshot(
        doc_pk_by_id={str(did): pk for pk, did in old_doc_ids.items()},
        cluster_members={a_old: ["2", "3"], b_old: ["4", "5", "1"]},
    )
    edits = [
        _edit(pid, "reassign_doc", document_id=old_doc_ids["1"], cluster_id=a_old, target_cluster_id=b_old),
        _edit(pid, "rename_label", cluster_id=a_old, new_label="My Theme"),
    ]

    # Fresh run re-clusters from scratch, ignoring the human edits: "1" back in A.
    a_new, b_new = uuid.uuid4(), uuid.uuid4()
    fresh_docs = {
        "1": _doc(pid, "1", a_new), "2": _doc(pid, "2", a_new), "3": _doc(pid, "3", a_new),
        "4": _doc(pid, "4", b_new), "5": _doc(pid, "5", b_new),
    }
    a, b = _cluster(a_new, pid, "auto A"), _cluster(b_new, pid, "auto B")
    session = _FakeSession(docs=list(fresh_docs.values()), clusters=[a, b], edits=edits)

    replay_edits(session, pid, snapshot)

    # The human label landed on the new cluster that holds A's old members.
    assert a.label == "My Theme" and a.label_source == "hitl_override"
    # The reassigned doc lands back in B (plurality of B's old members).
    assert fresh_docs["1"].cluster_id == b_new
    # Both touched clusters were recomputed.
    assert _stub_recompute[0][1] == {a_new, b_new}


# ── Human-created clusters are recreated ─────────────────────────────────────────

def test_create_from_selection_recreated(_stub_recompute):
    pid = uuid.uuid4()
    h_old = uuid.uuid4()
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    snapshot = MembershipSnapshot(doc_pk_by_id={str(d1): "1", str(d2): "2"}, cluster_members={})
    edits = [_edit(pid, "create_from_selection", cluster_id=h_old, new_label="Human",
                   payload={"document_ids": [str(d1), str(d2)]})]

    x = uuid.uuid4()
    fresh = {"1": _doc(pid, "1", x), "2": _doc(pid, "2", x)}
    session = _FakeSession(docs=list(fresh.values()), clusters=[_cluster(x, pid)], edits=edits)

    replay_edits(session, pid, snapshot)

    created = _new_cluster(session)
    assert created.label == "Human" and created.label_source == "hitl_override"
    assert fresh["1"].cluster_id == created.id and fresh["2"].cluster_id == created.id
    # New cluster + the docs' previous owner are recomputed.
    assert _stub_recompute[0][1] == {created.id, x}


# ── Merge survives ───────────────────────────────────────────────────────────────

def test_merge_survives(_stub_recompute):
    pid = uuid.uuid4()
    m_src, m_dst = uuid.uuid4(), uuid.uuid4()
    snapshot = MembershipSnapshot(doc_pk_by_id={}, cluster_members={m_src: ["1"], m_dst: ["2"]})
    edits = [_edit(pid, "merge_clusters", cluster_id=m_src, target_cluster_id=m_dst)]

    a_new, b_new = uuid.uuid4(), uuid.uuid4()
    fresh = {"1": _doc(pid, "1", a_new), "2": _doc(pid, "2", b_new)}
    a, b = _cluster(a_new, pid), _cluster(b_new, pid)
    session = _FakeSession(docs=list(fresh.values()), clusters=[a, b], edits=edits)

    replay_edits(session, pid, snapshot)

    assert fresh["1"].cluster_id == b_new  # moved into the merge target
    assert a in session.deleted  # source cluster removed
    assert _stub_recompute[0][1] == {b_new}


# ── Junk survives (docs → noise, cluster removed) ────────────────────────────────

def test_mark_junk_survives(_stub_recompute):
    pid = uuid.uuid4()
    j_old = uuid.uuid4()
    snapshot = MembershipSnapshot(doc_pk_by_id={}, cluster_members={j_old: ["1", "2"]})
    edits = [_edit(pid, "mark_junk", cluster_id=j_old, payload={"document_ids": []})]

    a_new = uuid.uuid4()
    fresh = {"1": _doc(pid, "1", a_new), "2": _doc(pid, "2", a_new)}
    a = _cluster(a_new, pid)
    session = _FakeSession(docs=list(fresh.values()), clusters=[a], edits=edits)

    replay_edits(session, pid, snapshot)

    assert fresh["1"].cluster_id is None and fresh["2"].cluster_id is None
    assert a in session.deleted


# ── approve_label stamps the resolved cluster ────────────────────────────────────

def test_approve_label_survives(_stub_recompute):
    pid = uuid.uuid4()
    c_old = uuid.uuid4()
    snapshot = MembershipSnapshot(doc_pk_by_id={}, cluster_members={c_old: ["1"]})
    edits = [_edit(pid, "approve_label", cluster_id=c_old)]

    a_new = uuid.uuid4()
    fresh = {"1": _doc(pid, "1", a_new)}
    a = _cluster(a_new, pid, "auto", "ollama:llama3")
    session = _FakeSession(docs=list(fresh.values()), clusters=[a], edits=edits)

    replay_edits(session, pid, snapshot)

    assert a.label_source == "hitl_approved"


# ── Unresolvable edits are skipped without error ─────────────────────────────────

def test_unresolvable_edits_skipped(_stub_recompute):
    pid = uuid.uuid4()
    gone = uuid.uuid4()  # old cluster whose members are all noise / absent now
    snapshot = MembershipSnapshot(doc_pk_by_id={}, cluster_members={gone: ["99"]})
    edits = [
        _edit(pid, "rename_label", cluster_id=gone, new_label="orphan"),
        _edit(pid, "reassign_doc", document_id=uuid.uuid4(), cluster_id=gone, target_cluster_id=gone),
    ]

    a_new = uuid.uuid4()
    fresh = {"1": _doc(pid, "1", a_new)}
    a = _cluster(a_new, pid, "auto")
    session = _FakeSession(docs=list(fresh.values()), clusters=[a], edits=edits)

    replay_edits(session, pid, snapshot)  # must not raise

    assert a.label == "auto"  # untouched
    assert fresh["1"].cluster_id == a_new


# ── snapshot_membership ──────────────────────────────────────────────────────────

def test_snapshot_membership_builds_maps(_stub_recompute):
    pid = uuid.uuid4()
    c1 = uuid.uuid4()
    d1, d2, d3 = _doc(pid, "1", c1), _doc(pid, "2", c1), _doc(pid, "3", None)
    session = _FakeSession(docs=[d1, d2, d3])

    snap = snapshot_membership(session, pid)

    assert snap.doc_pk_by_id == {str(d1.id): "1", str(d2.id): "2", str(d3.id): "3"}
    assert snap.cluster_members == {c1: ["1", "2"]}  # noise (None) excluded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
