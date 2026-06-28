"""
Pure-logic tests for the cluster-edit audit helper (app/services/edits.py).

No DB: ``record_edit`` only stages a row via ``db.add`` (the caller commits), so a
tiny fake session that captures added objects is enough. Mirrors the no-DB
approach used in test_ml_integration.py.

Run from the backend dir:  python -m pytest tests/test_cluster_edits.py
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.models import EDIT_ACTIONS, ClusterEdit  # noqa: E402
from app.services.edits import record_edit  # noqa: E402

# The reviewscope_ml feedback vocabulary the app vocabulary must be a superset of.
import importlib.util  # noqa: E402

_FEEDBACK = Path(__file__).resolve().parents[2] / "reviewscope_ml" / "hitl" / "feedback.py"


class _FakeSession:
    """Captures objects handed to .add() without a database."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)


def test_edit_actions_superset_of_feedback_actions():
    spec = importlib.util.spec_from_file_location("_fb", _FEEDBACK)
    fb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fb  # @dataclass resolves cls.__module__ via sys.modules
    spec.loader.exec_module(fb)
    assert set(fb.ACTIONS).issubset(set(EDIT_ACTIONS))


def test_record_edit_stages_row_with_fields():
    db = _FakeSession()
    pid, actor, doc, src, dst = (uuid.uuid4() for _ in range(5))

    edit = record_edit(
        db,
        project_id=pid,
        actor_id=actor,
        action="reassign_doc",
        document_id=doc,
        cluster_id=src,
        target_cluster_id=dst,
    )

    assert db.added == [edit]
    assert isinstance(edit, ClusterEdit)
    assert edit.action == "reassign_doc"
    assert edit.project_id == pid and edit.actor_id == actor
    assert edit.document_id == doc and edit.cluster_id == src and edit.target_cluster_id == dst


def test_record_edit_accepts_payload_for_bulk():
    db = _FakeSession()
    ids = [str(uuid.uuid4()) for _ in range(3)]
    edit = record_edit(
        db,
        project_id=uuid.uuid4(),
        actor_id=uuid.uuid4(),
        action="bulk_reassign",
        payload={"document_ids": ids},
    )
    assert edit.payload == {"document_ids": ids}


def test_record_edit_rejects_unknown_action():
    db = _FakeSession()
    with pytest.raises(ValueError):
        record_edit(db, project_id=uuid.uuid4(), actor_id=uuid.uuid4(), action="frobnicate")
    assert db.added == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
