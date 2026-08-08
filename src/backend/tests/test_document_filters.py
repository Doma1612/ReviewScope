"""Parsing tests for the document facet-filter helper (api/projects.py).

No DB: ``_document_filter_conditions`` turns a JSON facet spec into a list of
SQLAlchemy WHERE clauses, so we assert how many clauses it builds (and that bad
input is ignored) without executing them.

Run from the backend dir:  python -m pytest tests/test_document_filters.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("SIMULATE_ML", "true")
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp(prefix="reviewscope-test-uploads-"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/backend on path

from app.api.projects import _document_filter_conditions  # noqa: E402


def test_none_and_blank_yield_no_conditions():
    assert _document_filter_conditions(None) == []
    assert _document_filter_conditions("") == []


def test_invalid_json_is_ignored():
    assert _document_filter_conditions("not json") == []
    assert _document_filter_conditions(json.dumps({"not": "a list"})) == []


def test_valid_specs_build_one_condition_each():
    specs = [
        {"column": "rating", "op": "gte", "value": "3", "type": "integer"},
        {"column": "rating", "op": "lte", "value": "5", "type": "integer"},
        {"column": "verified", "op": "eq", "value": "true", "type": "boolean"},
        {"column": "created", "op": "gte", "value": "2024-01-01", "type": "date"},
    ]
    assert len(_document_filter_conditions(json.dumps(specs))) == 4


def test_empty_value_and_missing_column_skipped():
    specs = [
        {"column": "rating", "op": "gte", "value": "", "type": "integer"},  # blank value
        {"op": "eq", "value": "x"},                                          # no column
        {"column": "rating", "op": "gte", "value": "not-a-number", "type": "float"},  # bad numeric
    ]
    assert _document_filter_conditions(json.dumps(specs)) == []
