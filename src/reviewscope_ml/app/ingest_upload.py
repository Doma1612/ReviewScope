"""
Input adapter: an uploaded CSV/JSONL + a confirmed :class:`UploadSchema`
-> the in-memory :class:`~reviewscope_ml.data.ingest.ReviewSet` every pipeline
stage consumes, plus the per-document raw rows the app stores in
``documents.raw_data``.

This is the production counterpart to ``data.ingest.load_benchmark`` (which
only knows the fixed Yelp benchmark). It implements app-spec Celery steps
1 (Ingest: parse against schema, validate types + PK uniqueness, reject on
error) and 2 (Preprocess: clean text, dedup by primary key, drop very short
documents) — the same "minimal" preprocessing notebook 02 selected.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..core.config import get_preprocessor
from ..data.ingest import ReviewSet
from .schema import NUMERIC_TYPES, ColumnSpec, UploadSchema

# Match the benchmark's 50-char floor (methodology §1): drop documents too
# short to embed/cluster meaningfully. Configurable per call.
DEFAULT_MIN_TEXT_LEN = 50


class IngestError(ValueError):
    """Raised when an uploaded file fails validation against its schema.

    ``errors`` is a list of human-readable issues (row + column + reason), so
    the API can surface them inline exactly as the upload modal expects
    ("validation runs on submit — errors block submit, no auto-fixing").
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        n = len(errors)
        shown = "; ".join(errors[:10]) + (" …" if n > 10 else "")
        super().__init__(f"{n} validation error(s): {shown}")


@dataclass
class UploadedCorpus:
    """An ingested upload: the pipeline input plus the rows for ``raw_data``."""

    reviews: ReviewSet                 # ids = primary-key values, texts preprocessed
    raw_rows: list[dict[str, Any]]     # all original columns, aligned with reviews.ids
    schema: UploadSchema
    n_dropped_short: int               # documents removed by the min-length filter
    n_dropped_duplicate: int           # documents removed by primary-key dedup

    @property
    def n_documents(self) -> int:
        return len(self.reviews)


# ── Public entry point ────────────────────────────────────────────────────────

def reviewset_from_upload(
    file_path: str | Path,
    schema: UploadSchema,
    *,
    preprocessor: str = "minimal",
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
) -> UploadedCorpus:
    """
    Parse and validate *file_path* against *schema*, returning an
    :class:`UploadedCorpus`. Raises :class:`IngestError` (with a per-row error
    list) if any value violates its declared type or the primary key is not
    unique. No auto-fixing — invalid input is rejected, per the app spec.
    """
    schema.validate()
    path = Path(file_path)
    rows = list(_read_rows(path))

    pk_col = schema.primary_key()
    text_col = schema.text_column
    rating_col = schema.rating_column
    preprocess = get_preprocessor(preprocessor)

    errors: list[str] = []
    # Validate every typed column up front so the user sees all problems at once
    # (bounded — the API may cap how many it renders).
    for i, row in enumerate(rows):
        if text_col not in row or _is_blank(row.get(text_col)):
            errors.append(f"row {i + 1}: text column {text_col!r} is empty")
        for col in schema.columns:
            if col.name not in row:
                continue
            ok, _ = _coerce(row[col.name], col.type)
            if not ok and not _is_blank(row[col.name]):
                errors.append(
                    f"row {i + 1}: column {col.name!r} value "
                    f"{row[col.name]!r} is not a valid {col.type}"
                )

    pk_seen: dict[str, int] = {}
    if pk_col is not None:
        for i, row in enumerate(rows):
            val = row.get(pk_col)
            if _is_blank(val):
                errors.append(f"row {i + 1}: primary key {pk_col!r} is empty")

    if errors:
        raise IngestError(errors)

    ids: list[str] = []
    texts: list[str] = []
    raw_texts: list[str] = []
    stars: list[float] = []
    raw_rows: list[dict[str, Any]] = []

    n_short = 0
    n_dup = 0
    for i, row in enumerate(rows):
        raw_text = str(row.get(text_col, "") or "")
        if len(raw_text.strip()) < min_text_len:
            n_short += 1
            continue

        pk_val = str(row[pk_col]) if pk_col is not None else str(i)
        if pk_col is not None:
            if pk_val in pk_seen:   # dedup by primary key (exact re-uploads)
                n_dup += 1
                continue
            pk_seen[pk_val] = i

        ids.append(pk_val)
        raw_texts.append(raw_text)
        texts.append(preprocess(raw_text))
        stars.append(_rating_value(row.get(rating_col)) if rating_col else float("nan"))
        raw_rows.append(_typed_row(row, schema))

    reviews = ReviewSet(
        ids=ids, texts=texts, raw_texts=raw_texts, stars=np.array(stars, dtype=float)
    )
    return UploadedCorpus(
        reviews=reviews,
        raw_rows=raw_rows,
        schema=schema,
        n_dropped_short=n_short,
        n_dropped_duplicate=n_dup,
    )


# ── File reading ──────────────────────────────────────────────────────────────

def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        yield from _read_jsonl(path)
    elif suffix == ".csv":
        yield from _read_csv(path)
    else:
        raise IngestError([f"unsupported file type {path.suffix!r}; expected .csv or .jsonl"])


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise IngestError([f"line {lineno}: invalid JSON ({e.msg})"]) from e
            if not isinstance(obj, dict):
                raise IngestError([f"line {lineno}: expected a JSON object, got {type(obj).__name__}"])
            yield obj


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


# ── Type coercion ─────────────────────────────────────────────────────────────

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _coerce(value: Any, type_: str) -> tuple[bool, Any]:
    """Return (ok, coerced_value). Blank values are considered valid (-> None)."""
    if _is_blank(value):
        return True, None
    try:
        if type_ == "text":
            return True, str(value)
        if type_ == "integer":
            if isinstance(value, bool):       # bool is an int subclass — reject for a number column
                return False, None
            if isinstance(value, int):
                return True, value
            if isinstance(value, float):
                return (True, int(value)) if value.is_integer() else (False, None)
            return True, int(str(value).strip())   # ValueError on "3.5" -> caught below
        if type_ == "float":
            if isinstance(value, bool):
                return False, None
            return True, float(value)
        if type_ == "boolean":
            if isinstance(value, bool):
                return True, value
            s = str(value).strip().lower()
            if s in _TRUE:
                return True, True
            if s in _FALSE:
                return True, False
            return False, None
        if type_ == "date":
            return True, _parse_date(value)
    except (ValueError, TypeError):
        return False, None
    return False, None


def _parse_date(value: Any) -> str:
    """Validate a date and return it normalised to ISO (yyyy-mm-dd...) as a string."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    s = str(value).strip()
    # ISO first (covers date and datetime); then a couple of common formats.
    try:
        return datetime.fromisoformat(s).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {s!r}")


def _rating_value(value: Any) -> float:
    ok, coerced = _coerce(value, "float")
    return float(coerced) if ok and coerced is not None else float("nan")


def _typed_row(row: dict[str, Any], schema: UploadSchema) -> dict[str, Any]:
    """Coerce a raw row to declared types for clean, JSON-serialisable raw_data.

    Columns absent from the schema are passed through untouched; declared
    columns are normalised (ints as ints, dates as ISO strings, …).
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        col: ColumnSpec | None = schema.column(key)
        if col is None:
            out[key] = value
            continue
        ok, coerced = _coerce(value, col.type)
        out[key] = coerced if ok else value
    return out
