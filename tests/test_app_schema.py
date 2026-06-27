import pytest

from reviewscope_ml.app import ColumnSpec, SchemaError, UploadSchema


def make_schema(**over) -> UploadSchema:
    kw = dict(
        columns=[
            ColumnSpec("review_id", "text", is_primary_key=True),
            ColumnSpec("text", "text"),
            ColumnSpec("stars", "integer"),
        ],
        text_column="text",
        rating_column="stars",
    )
    kw.update(over)
    return UploadSchema(**kw)


class TestSchemaValidation:
    def test_valid_schema_passes(self):
        make_schema().validate()  # no raise

    def test_primary_key_detected(self):
        assert make_schema().primary_key() == "review_id"

    def test_no_primary_key_is_allowed(self):
        s = make_schema(columns=[ColumnSpec("text", "text")], rating_column=None)
        s.validate()
        assert s.primary_key() is None

    def test_text_column_must_exist(self):
        with pytest.raises(SchemaError, match="not a declared column"):
            make_schema(text_column="missing").validate()

    def test_text_column_must_be_text_type(self):
        with pytest.raises(SchemaError, match="must have type 'text'"):
            make_schema(text_column="stars").validate()

    def test_rating_column_must_be_numeric(self):
        with pytest.raises(SchemaError, match="must be numeric"):
            make_schema(rating_column="text").validate()

    def test_duplicate_columns_rejected(self):
        with pytest.raises(SchemaError, match="duplicate"):
            UploadSchema(
                columns=[ColumnSpec("text", "text"), ColumnSpec("text", "text")],
                text_column="text",
            ).validate()

    def test_multiple_primary_keys_rejected(self):
        with pytest.raises(SchemaError, match="multiple primary-key"):
            UploadSchema(
                columns=[
                    ColumnSpec("a", "text", is_primary_key=True),
                    ColumnSpec("text", "text", is_primary_key=True),
                ],
                text_column="text",
            ).validate()
