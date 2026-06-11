import numpy as np
import pytest

from reviewscope_ml.core.cache import clustering_path, embedding_path, make_slug, umap_path
from reviewscope_ml.core.config import get_preprocessor, load_config


class TestConfig:
    def test_data_file_derived_from_sample_size(self):
        assert load_config(sample_size=1_000).data_file == "sample_hotels_1k.jsonl"
        assert load_config(sample_size=50_000).data_file == "sample_hotels_50k.jsonl"
        # non-round sizes must not silently truncate to a misleading name
        assert load_config(sample_size=1_500).data_file == "sample_hotels_1500.jsonl"

    def test_data_file_override(self):
        cfg = load_config(sample_size=1_000, data_file="custom.jsonl")
        assert cfg.data_file == "custom.jsonl"

    def test_project_root_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REVIEWSCOPE_ROOT", str(tmp_path))
        cfg = load_config()
        assert cfg.project_root == tmp_path
        assert cfg.cache_dir == tmp_path / "data" / "cache"

    def test_cpu_thread_env_pinning(self, monkeypatch):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)
        load_config(cpu_threads=2)
        import os

        assert os.environ["OMP_NUM_THREADS"] == "2"


class TestPreprocessors:
    def test_minimal_collapses_whitespace_only(self):
        f = get_preprocessor("minimal")
        assert f("  Great\n\nroom!  ") == "Great room!"

    def test_aggressive_strips_punctuation_and_case(self):
        f = get_preprocessor("aggressive")
        assert f("GREAT room!!!") == "great room"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            get_preprocessor("nope")


class TestCacheKeys:
    def test_make_slug_strips_org_and_specials(self):
        assert make_slug("intfloat/multilingual-e5-large-instruct") == "multilingual-e5-large-instruct"
        assert make_slug("all-MiniLM-L6-v2") == "all-minilm-l6-v2"

    def test_embedding_path_is_deterministic(self, tmp_path):
        a = embedding_path(tmp_path, "org/Model-X", 5_000, instruction="domain")
        b = embedding_path(tmp_path, "org/Model-X", 5_000, instruction="domain")
        assert a == b
        assert a.name == "model-x__domain__5k.npy"

    def test_paths_distinguish_all_axes(self, tmp_path):
        base = dict(min_dist=0.0, metric="cosine", sample_size=5_000)
        p1 = umap_path(tmp_path, "m", 10, 15, **base)
        p2 = umap_path(tmp_path, "m", 10, 30, **base)
        p3 = umap_path(tmp_path, "m", 10, 15, **base, prefix="s43_")
        assert len({p1, p2, p3}) == 3

    def test_clustering_path_includes_params(self, tmp_path):
        p = clustering_path(tmp_path, "hdbscan", "mcs15__ms5", "slugA", 1_000)
        assert "mcs15__ms5" in p.name and "slugA" in p.name


class TestParamsSlug:
    def test_sorted_and_filesystem_safe(self):
        from reviewscope_ml.cluster import params_slug

        assert params_slug({"ms": 5, "mcs": 15}) == "mcs15__ms5"
        assert "." not in params_slug({"eps": 0.5})
