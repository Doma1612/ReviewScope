import numpy as np

from reviewscope_ml.cluster import TwoStageBackend, scaled_min_cluster_size
from reviewscope_ml.pipelines.runner import _make_backend
from reviewscope_ml.pipelines.spec import PipelineSpec


class TestScaledMinClusterSize:
    def test_anchored_to_notebook_decision(self):
        # 0.3% of 5k = 15 — exactly the notebook 06 choice
        assert scaled_min_cluster_size(5_000) == 15

    def test_scales_to_50k(self):
        assert scaled_min_cluster_size(50_000) == 150

    def test_floor_protects_smoke_runs(self):
        assert scaled_min_cluster_size(1_000) == 15
        assert scaled_min_cluster_size(0) == 15

    def test_custom_fraction_and_floor(self):
        assert scaled_min_cluster_size(50_000, fraction=0.001, floor=5) == 50


class TestAutoBackendResolution:
    def test_auto_resolves_against_unit_count(self):
        spec = PipelineSpec(
            variant="custom_hdbscan",
            cluster={"min_cluster_size": "auto", "min_samples": "auto"},
        )
        b = _make_backend(spec, seed=42, n_units=50_000)
        assert b.min_cluster_size == 150
        assert b.min_samples == 50

    def test_explicit_values_respected(self):
        spec = PipelineSpec(
            variant="custom_hdbscan",
            cluster={"min_cluster_size": 30, "min_samples": 7},
        )
        b = _make_backend(spec, seed=42, n_units=50_000)
        assert b.min_cluster_size == 30
        assert b.min_samples == 7

    def test_two_stage_micro_scaling(self):
        spec = PipelineSpec(
            variant="two_stage",
            cluster={"micro_min_cluster_size": "auto",
                     "micro_min_samples": "auto", "n_macro": None},
        )
        b = _make_backend(spec, seed=42, n_units=50_000)
        assert b.micro_min_cluster_size == 50
        assert b.micro_min_samples == 30

    def test_benchmark_scale_matches_notebook_defaults(self):
        spec = PipelineSpec(
            variant="custom_hdbscan",
            cluster={"min_cluster_size": "auto", "min_samples": "auto"},
        )
        b = _make_backend(spec, seed=42, n_units=5_000)
        assert (b.min_cluster_size, b.min_samples) == (15, 5)


def three_groups(seed=0):
    """Six tight micro-blobs arranged as three well-separated pairs."""
    rng = np.random.default_rng(seed)
    centers = [(0, 0), (1, 0), (10, 10), (11, 10), (20, 0), (21, 0)]
    return np.vstack([rng.normal(c, 0.15, (25, 2)) for c in centers])


class TestTwoStage:
    def test_micro_macro_hierarchy_is_consistent(self):
        X = three_groups()
        backend = TwoStageBackend(micro_min_cluster_size=5, n_macro=3)
        labels = backend.fit_predict(X)

        micro = backend.micro_labels_
        mapping = backend.micro_to_macro_
        # every non-noise document's macro label equals its micro's mapping
        for doc_micro, doc_macro in zip(micro, labels):
            if doc_micro == -1:
                assert doc_macro == -1
            else:
                assert mapping[int(doc_micro)] == int(doc_macro)

    def test_macro_count_respects_n_macro(self):
        X = three_groups()
        backend = TwoStageBackend(micro_min_cluster_size=5, n_macro=3)
        labels = backend.fit_predict(X)
        assert len(set(labels) - {-1}) == 3

    def test_macro_merges_adjacent_micro_blobs(self):
        X = three_groups()
        backend = TwoStageBackend(micro_min_cluster_size=5, n_macro=3)
        labels = backend.fit_predict(X)
        # the two micro-blobs of each pair must land in the same macro topic
        for pair_start in (0, 50, 100):
            a = labels[pair_start:pair_start + 25]
            b = labels[pair_start + 25:pair_start + 50]
            a, b = a[a != -1], b[b != -1]
            assert len(set(a) | set(b)) == 1

    def test_auto_macro_heuristic_bounded(self):
        X = three_groups()
        backend = TwoStageBackend(micro_min_cluster_size=5, n_macro=None)
        labels = backend.fit_predict(X)
        n = len(set(labels) - {-1})
        assert 2 <= n <= 30

    def test_degenerate_single_micro_cluster(self):
        X = np.random.default_rng(0).normal(0, 0.1, (30, 2))
        backend = TwoStageBackend(micro_min_cluster_size=5)
        labels = backend.fit_predict(X)  # must not crash
        assert set(labels) <= {-1, 0}
