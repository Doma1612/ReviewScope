import numpy as np

from reviewscope_ml.app import ColumnSpec, UploadSchema, to_records
from reviewscope_ml.app.ingest_upload import UploadedCorpus
from reviewscope_ml.data.ingest import ReviewSet


def corpus_for(doc_ids: list[str]) -> UploadedCorpus:
    n = len(doc_ids)
    reviews = ReviewSet(
        ids=list(doc_ids),
        texts=[f"text {i}" for i in range(n)],
        raw_texts=[f"text {i}" for i in range(n)],
        stars=np.array([4.0] * n),
    )
    raw_rows = [{"review_id": d, "text": f"text {i}", "stars": 4}
                for i, d in enumerate(doc_ids)]
    schema = UploadSchema(
        columns=[ColumnSpec("review_id", "text", is_primary_key=True),
                 ColumnSpec("text", "text"), ColumnSpec("stars", "integer")],
        text_column="text", rating_column="stars",
    )
    return UploadedCorpus(reviews=reviews, raw_rows=raw_rows, schema=schema,
                          n_dropped_short=0, n_dropped_duplicate=0)


class TestToRecords:
    def test_document_mapping(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        emb = np.arange(len(small_run.doc_ids) * 4, dtype=float).reshape(-1, 4)
        result = to_records("proj1", corpus, small_run, emb)

        assert result.project_id == "proj1"
        assert result.n_documents == 12
        # first 5 docs -> cluster 0, next 4 -> cluster 1, last 3 -> noise (None)
        assert result.documents[0].cluster_id == 0
        assert result.documents[5].cluster_id == 1
        assert result.documents[-1].cluster_id is None
        # raw_data carried through
        assert result.documents[0].raw_data["stars"] == 4

    def test_embedding_mapping(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        emb = np.arange(len(small_run.doc_ids) * 4, dtype=float).reshape(-1, 4)
        result = to_records("proj1", corpus, small_run, emb)

        rec = result.embeddings[2]
        assert rec.primary_key_value == small_run.doc_ids[2]
        assert rec.vector == [8.0, 9.0, 10.0, 11.0]
        # 3-D UMAP coords drive x/y/z (coords_3d row 2 = [6,7,8])
        assert (rec.umap_x, rec.umap_y, rec.umap_z) == (6.0, 7.0, 8.0)

    def test_cluster_mapping(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        emb = np.zeros((12, 4))
        result = to_records("proj1", corpus, small_run, emb)

        assert result.n_clusters == 2
        c0 = next(c for c in result.clusters if c.cluster_id == 0)
        assert c0.size == 5
        assert c0.label == "label 0"
        assert c0.label_source == "terms_fallback"
        assert c0.top_terms == [{"term": "alpha", "score": 1.0},
                                {"term": "beta", "score": 0.5}]
        assert c0.word_frequencies == {"alpha": 3}

    def test_no_sentiment_yields_none(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        result = to_records("p", corpus, small_run, np.zeros((12, 4)))
        assert all(d.sentiment_score is None for d in result.documents)

    def test_embedding_count_mismatch_raises(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        import pytest
        with pytest.raises(ValueError, match="same order"):
            to_records("p", corpus, small_run, np.zeros((5, 4)))
