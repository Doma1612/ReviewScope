"""
Upload schema — the application's description of a user-uploaded corpus.

This is the package-side mirror of the app spec's "Step 2: Schema Confirmation"
table (one row per column: name, type, primary-key flag) plus two role
designations the pipeline needs and the table implies:

* ``text_column``   — which column carries the natural-language text the NLP
                      pipeline embeds and clusters (app spec: ``documents.text``).
* ``rating_column`` — optional numeric column used for the star-rating display,
                      Tier-3 entropy and sentiment context (app spec:
                      ``documents`` star rating / ``clusters.sentiment``).

The backend builds an ``UploadSchema`` from the confirmed column types and
hands it to :func:`reviewscope_ml.app.ingest_upload.reviewset_from_upload`.
``validate`` performs the structural checks; per-row type validation happens
during ingest (it needs the file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColumnType = Literal["text", "integer", "float", "date", "boolean"]
NUMERIC_TYPES = ("integer", "float")
_VALID_TYPES = ("text", "integer", "float", "date", "boolean")


class SchemaError(ValueError):
    """Raised when the declared schema is structurally invalid (before ingest)."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: ColumnType
    is_primary_key: bool = False


@dataclass
class UploadSchema:
    """Confirmed column types plus the text / rating role designations."""

    columns: list[ColumnSpec]
    text_column: str
    rating_column: str | None = None

    # ── Derived ───────────────────────────────────────────────────────────
    _by_name: dict[str, ColumnSpec] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._by_name = {c.name: c for c in self.columns}

    def primary_key(self) -> str | None:
        """Name of the primary-key column, or None (then ingest uses row index)."""
        pks = [c.name for c in self.columns if c.is_primary_key]
        return pks[0] if pks else None

    def column(self, name: str) -> ColumnSpec | None:
        return self._by_name.get(name)

    def validate(self) -> None:
        """Structural validation. Raises :class:`SchemaError` on the first problem.

        Per-value type checking is deferred to ingest, which has the file.
        """
        if not self.columns:
            raise SchemaError("schema has no columns")

        names = [c.name for c in self.columns]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise SchemaError(f"duplicate column name(s): {sorted(dupes)}")

        for c in self.columns:
            if c.type not in _VALID_TYPES:
                raise SchemaError(
                    f"column {c.name!r}: unknown type {c.type!r}; "
                    f"choose one of {_VALID_TYPES}"
                )

        if self.text_column not in self._by_name:
            raise SchemaError(f"text_column {self.text_column!r} is not a declared column")
        if self.column(self.text_column).type != "text":
            raise SchemaError(
                f"text_column {self.text_column!r} must have type 'text', "
                f"got {self.column(self.text_column).type!r}"
            )

        if self.rating_column is not None:
            rc = self.column(self.rating_column)
            if rc is None:
                raise SchemaError(
                    f"rating_column {self.rating_column!r} is not a declared column"
                )
            if rc.type not in NUMERIC_TYPES:
                raise SchemaError(
                    f"rating_column {self.rating_column!r} must be numeric "
                    f"(integer/float), got {rc.type!r}"
                )

        pks = [c.name for c in self.columns if c.is_primary_key]
        if len(pks) > 1:
            raise SchemaError(
                f"multiple primary-key columns declared: {pks}; the document "
                "identity must be a single column (or none → row index)"
            )
