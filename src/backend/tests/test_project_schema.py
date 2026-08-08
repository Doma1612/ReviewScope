"""Endpoint-logic tests for GET/POST /api/projects/{id}/schema (api/projects.py).

No real DB: the routes are plain async functions, so a fake ``AsyncSession`` that
serves ``get`` from in-memory objects and records ``add``/``commit`` exercises the
wiring directly. The "exactly one PK" / "known type" rules live on the Pydantic
request models, so FastAPI surfaces them as 422 before the route runs — they are
asserted here as ``ValidationError`` on the schema, which is what produces that 422.
Mirrors the no-DB approach of test_cluster_crud.py.

Run from the backend dir:  python -m pytest tests/test_project_schema.py
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
from pydantic import ValidationError  # noqa: E402

from app.api import projects  # noqa: E402
from app.models import ProjectMember, ProjectRole, ProjectSchema  # noqa: E402
from app.schemas import ProjectSchemaWrite  # noqa: E402


def asyncio_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class _FakeResult:
    def __init__(self, scalar=None):
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    """Serves require_project_role's ProjectMember query and get(); records mutations."""

    def __init__(self, *, member=None, objects=None):
        self.member = member
        self.objects = objects or {}
        self.added: list = []
        self.committed = False

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        if entity is ProjectMember:
            return _FakeResult(scalar=self.member)
        raise AssertionError(f"unexpected query for {entity!r}")

    async def get(self, model, pk):
        return self.objects.get((model, pk))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


def _owner():
    return SimpleNamespace(id=uuid.uuid4())


def _member(role=ProjectRole.owner):
    return ProjectMember(role=role)


def _cols(*, two_pks=False):
    return [
        {"name": "id", "type": "integer", "is_primary_key": True},
        {"name": "review", "type": "text", "is_primary_key": two_pks},
    ]


# ── GET ─────────────────────────────────────────────────────────────────────

@asyncio_test
async def test_get_schema_returns_stored_columns():
    pid = uuid.uuid4()
    cols = _cols()
    db = _FakeSession(member=_member(ProjectRole.viewer), objects={(ProjectSchema, pid): ProjectSchema(project_id=pid, columns=cols)})

    result = await projects.get_schema(pid, db, _owner())

    assert result.columns == cols


@asyncio_test
async def test_get_schema_404_when_absent():
    pid = uuid.uuid4()
    db = _FakeSession(member=_member(ProjectRole.viewer))
    with pytest.raises(HTTPException) as exc:
        await projects.get_schema(pid, db, _owner())
    assert exc.value.status_code == 404


@asyncio_test
async def test_get_schema_requires_membership():
    db = _FakeSession(member=None)
    with pytest.raises(HTTPException) as exc:
        await projects.get_schema(uuid.uuid4(), db, _owner())
    assert exc.value.status_code == 403


# ── POST ────────────────────────────────────────────────────────────────────

@asyncio_test
async def test_post_schema_inserts_when_absent():
    pid = uuid.uuid4()
    db = _FakeSession(member=_member())

    result = await projects.set_schema(pid, ProjectSchemaWrite(columns=_cols()), db, _owner())

    created = [o for o in db.added if isinstance(o, ProjectSchema)][0]
    assert created.project_id == pid
    assert [c["name"] for c in created.columns] == ["id", "review"]
    assert result.columns == created.columns
    assert db.committed


@asyncio_test
async def test_post_schema_upserts_existing():
    pid = uuid.uuid4()
    existing = ProjectSchema(project_id=pid, columns=[{"name": "old", "type": "text", "is_primary_key": True}])
    db = _FakeSession(member=_member(), objects={(ProjectSchema, pid): existing})

    result = await projects.set_schema(pid, ProjectSchemaWrite(columns=_cols()), db, _owner())

    assert [c["name"] for c in existing.columns] == ["id", "review"]
    assert result.columns == existing.columns
    assert not [o for o in db.added if isinstance(o, ProjectSchema)]  # updated in place, not re-added
    assert db.committed


@asyncio_test
async def test_post_schema_viewer_forbidden():
    db = _FakeSession(member=_member(ProjectRole.viewer))
    with pytest.raises(HTTPException) as exc:
        await projects.set_schema(uuid.uuid4(), ProjectSchemaWrite(columns=_cols()), db, _owner())
    assert exc.value.status_code == 403
    assert not db.committed


# ── Validation (surfaced by FastAPI as 422) ─────────────────────────────────

def test_two_primary_keys_rejected():
    with pytest.raises(ValidationError):
        ProjectSchemaWrite(columns=_cols(two_pks=True))


def test_zero_primary_keys_rejected():
    with pytest.raises(ValidationError):
        ProjectSchemaWrite(columns=[{"name": "review", "type": "text", "is_primary_key": False}])


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        ProjectSchemaWrite(columns=[{"name": "id", "type": "uuid", "is_primary_key": True}])


def test_valid_types_accepted():
    cols = [
        {"name": "id", "type": "integer", "is_primary_key": True},
        {"name": "body", "type": "text"},
        {"name": "score", "type": "float"},
        {"name": "day", "type": "date"},
        {"name": "flag", "type": "boolean"},
    ]
    model = ProjectSchemaWrite(columns=cols)
    assert len(model.columns) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
