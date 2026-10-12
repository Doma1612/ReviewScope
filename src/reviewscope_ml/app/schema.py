from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColumnType = Literal["text", "integer", "float", "date", "boolean"]
NUMERIC_TYPES = ("integer", "float")
_VALID_TYPES = ("text", "integer", "float", "date", "boolean")


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: ColumnType
    is_primary_key: bool = False


@dataclass
class UploadSchema:

    columns: list[ColumnSpec]
    text_column: str
    rating_column: str | None = None

    _by_name: dict[str, ColumnSpec] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._by_name = {c.name: c for c in self.columns}

    def primary_key(self) -> str | None:
        pks = [c.name for c in self.columns if c.is_primary_key]
        return pks[0] if pks else None

    def column(self, name: str) -> ColumnSpec | None:
        return self._by_name.get(name)

    def validate(self) -> None:
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
