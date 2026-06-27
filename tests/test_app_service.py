import numpy as np
import pytest

from reviewscope_ml.app import (
    ColumnSpec,
    UploadSchema,
    app_default_spec,
    project_config,
    project_corpus_token,
)
from reviewscope_ml.app import service as service_mod
from reviewscope_ml.app.ingest_upload import UploadedCorpus
from reviewscope_ml.data.ingest import ReviewSet
from reviewscope_ml.pipelines.spec import PipelineSpec


class RecordingProgress:
    def __init__(self):
        self.events = []

    def step(self, name, status, message="", index=0, total=0):
        self.events.append((index, name, status))


def corpus_for(doc_ids):
    n = len(doc_ids)
    reviews = ReviewSet(ids=list(doc_ids), texts=[f"t{i}" for i in range(n)],
                        raw_texts=[f"t{i}" for i in range(n)], stars=np.array([4.0] * n))
    raw_rows = [{"review_id": d} for d in doc_ids]
    schema = UploadSchema(columns=[ColumnSpec("review_id", "text", is_primary_key=True),
                                   ColumnSpec("text", "text")],
                          text_column="text")
    return UploadedCorpus(reviews=reviews, raw_rows=raw_rows, schema=schema,
                          n_dropped_short=0, n_dropped_duplicate=0)


class TestProjectConfig:
    def test_corpus_token_is_underscore_free_and_namespaced(self):
        tok = project_corpus_token("My Project! 42")
        assert tok == "proj-my-project-42"
        assert "_" not in tok and tok != "hotels"

    def test_corpus_slug_round_trips_through_config(self):
        # The synthetic data_file must parse back to exactly the token, so every
        # cached artifact is keyed to this project (and never collides with hotels).
        cfg = project_config("abc-123", n_docs=1234)
        assert cfg.corpus_slug == project_corpus_token("abc-123")
        assert cfg.sample_size == 1234


class TestRunProjectPipeline:
    def test_orchestrates_and_maps(self, monkeypatch, small_run):
        corpus = corpus_for(small_run.doc_ids)
        emb = np.arange(len(small_run.doc_ids) * 4, dtype=float).reshape(-1, 4)

        seen_kwargs = {}

        def fake_run_pipeline(cfg, spec, **kwargs):
            seen_kwargs.update(kwargs)
            # exercise the stage -> step progress translation
            for stage in ("embed", "reduce", "cluster", "sentiment", "label", "evaluate"):
                kwargs["on_stage"](stage)
            return small_run

        monkeypatch.setattr(service_mod, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(service_mod, "load_run_embeddings", lambda *a, **k: emb)

        progress = RecordingProgress()
        result = service_mod.run_project_pipeline(
            corpus=corpus, project_id="p1", progress=progress, label_clusters=False,
        )

        assert result.n_documents == 12
        assert result.n_clusters == 2
        assert seen_kwargs["on_stage"] is not None
        # the canonical 8-step vocabulary surfaced, ending with Finalize done
        steps = [name for _, name, _ in progress.events]
        assert "Embed" in steps and "Cluster" in steps and "Finalize" in steps
        assert progress.events[-1] == (8, "Finalize", "done")
        # steps are monotonically non-decreasing (no backwards progress)
        indices = [i for i, _, _ in progress.events]
        assert indices == sorted(indices)

    def test_sentence_level_is_rejected(self, small_run):
        corpus = corpus_for(small_run.doc_ids)
        with pytest.raises(ValueError, match="sentence_level"):
            service_mod.run_project_pipeline(
                corpus=corpus, project_id="p1",
                spec=PipelineSpec(variant="sentence_level"),
            )

    def test_empty_corpus_rejected(self):
        corpus = corpus_for([])
        with pytest.raises(ValueError, match="no documents"):
            service_mod.run_project_pipeline(corpus=corpus, project_id="p1")


def test_default_spec_is_document_level():
    spec = app_default_spec()
    assert spec.variant == "custom_hdbscan"
