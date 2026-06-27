import json
import math

import pytest

from reviewscope_ml.app import ColumnSpec, IngestError, UploadSchema, reviewset_from_upload


def schema() -> UploadSchema:
    return UploadSchema(
        columns=[
            ColumnSpec("review_id", "text", is_primary_key=True),
            ColumnSpec("text", "text"),
            ColumnSpec("stars", "integer"),
        ],
        text_column="text",
        rating_column="stars",
    )


LONG = "This hotel was genuinely lovely and the staff were helpful."  # > 10 chars


def write_csv(path, rows, header=("review_id", "text", "stars")):
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(c) for c in r))
    path.write_text("\n".join(lines) + "\n")
    return path


class TestCsvIngest:
    def test_basic_ingest(self, tmp_path):
        f = write_csv(tmp_path / "r.csv", [
            ("a", LONG, 5),
            ("b", LONG, 3),
        ])
        corpus = reviewset_from_upload(f, schema(), min_text_len=10)
        assert corpus.n_documents == 2
        assert corpus.reviews.ids == ["a", "b"]
        assert list(corpus.reviews.stars) == [5.0, 3.0]
        # raw_data is type-coerced: stars stored as int, not "5"
        assert corpus.raw_rows[0]["stars"] == 5
        assert isinstance(corpus.raw_rows[0]["stars"], int)

    def test_short_documents_dropped(self, tmp_path):
        f = write_csv(tmp_path / "r.csv", [
            ("a", LONG, 5),
            ("b", "short", 3),       # < 10 chars after strip
        ])
        corpus = reviewset_from_upload(f, schema(), min_text_len=10)
        assert corpus.n_documents == 1
        assert corpus.n_dropped_short == 1

    def test_duplicate_primary_key_deduped(self, tmp_path):
        f = write_csv(tmp_path / "r.csv", [
            ("a", LONG, 5),
            ("a", LONG + " extra", 4),   # same PK -> dropped (keep first)
        ])
        corpus = reviewset_from_upload(f, schema(), min_text_len=10)
        assert corpus.n_documents == 1
        assert corpus.n_dropped_duplicate == 1
        assert corpus.reviews.stars[0] == 5.0

    def test_invalid_integer_rejected(self, tmp_path):
        f = write_csv(tmp_path / "r.csv", [("a", LONG, "not-a-number")])
        with pytest.raises(IngestError) as exc:
            reviewset_from_upload(f, schema(), min_text_len=10)
        assert any("stars" in e for e in exc.value.errors)

    def test_float_for_integer_column_rejected(self, tmp_path):
        f = write_csv(tmp_path / "r.csv", [("a", LONG, "4.5")])
        with pytest.raises(IngestError):
            reviewset_from_upload(f, schema(), min_text_len=10)

    def test_no_primary_key_uses_row_index(self, tmp_path):
        s = UploadSchema(
            columns=[ColumnSpec("text", "text"), ColumnSpec("stars", "integer")],
            text_column="text", rating_column="stars",
        )
        f = write_csv(tmp_path / "r.csv", [("", LONG, 5), ("", LONG, 4)],
                      header=("review_id", "text", "stars"))
        corpus = reviewset_from_upload(f, s, min_text_len=10)
        assert corpus.reviews.ids == ["0", "1"]


class TestJsonlIngest:
    def test_jsonl_ingest(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text("\n".join(json.dumps(o) for o in [
            {"review_id": "a", "text": LONG, "stars": 5},
            {"review_id": "b", "text": LONG, "stars": 2},
        ]) + "\n")
        corpus = reviewset_from_upload(f, schema(), min_text_len=10)
        assert corpus.n_documents == 2
        assert corpus.raw_rows[1]["stars"] == 2

    def test_missing_rating_is_nan(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text(json.dumps({"review_id": "a", "text": LONG}) + "\n")
        corpus = reviewset_from_upload(f, schema(), min_text_len=10)
        assert math.isnan(corpus.reviews.stars[0])

    def test_invalid_json_line_raises(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text('{"review_id": "a", "text": "' + LONG + '"}\nnot json\n')
        with pytest.raises(IngestError):
            reviewset_from_upload(f, schema(), min_text_len=10)
