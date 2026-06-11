import numpy as np

from reviewscope_ml.core.metrics import compute_metrics, compute_rating_entropy
from reviewscope_ml.eval.harness import evaluate_labels, failure_flags, stability_ari


def blobs(n_per=30, centers=((0, 0), (5, 5)), seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.3, (n_per, 2)) for c in centers])
    labels = np.repeat(np.arange(len(centers)), n_per)
    return X, labels


class TestComputeMetricsEdgeCases:
    def test_single_cluster_yields_no_geometric_scores(self):
        X = np.random.default_rng(0).normal(0, 1, (50, 3))
        m = compute_metrics(X, np.zeros(50, dtype=int), runtime_s=0.0)
        assert m["n_clusters"] == 1
        assert m["silhouette"] is None
        assert m["davies_bouldin"] is None

    def test_all_noise(self):
        X = np.random.default_rng(0).normal(0, 1, (20, 3))
        m = compute_metrics(X, np.full(20, -1), runtime_s=0.0)
        assert m["n_clusters"] == 0
        assert m["noise_ratio"] == 1.0
        assert m["silhouette"] is None

    def test_noise_excluded_from_silhouette(self):
        X, labels = blobs()
        labels_noisy = labels.copy()
        labels_noisy[:5] = -1
        m = compute_metrics(X, labels_noisy, runtime_s=0.0)
        assert m["noise_count"] == 5
        assert m["silhouette"] > 0.8  # computed on the clean remainder


class TestRatingEntropy:
    def test_thematic_high_sentiment_low(self):
        labels = np.array([0] * 50 + [1] * 50)
        mixed = np.tile([1, 2, 3, 4, 5], 20)          # both clusters mixed
        polar = np.array([5.0] * 50 + [1.0] * 50)     # clusters = star levels
        assert compute_rating_entropy(mixed, labels) > 0.9
        assert compute_rating_entropy(polar, labels) < 0.1

    def test_single_cluster_returns_none(self):
        assert compute_rating_entropy(np.ones(10), np.zeros(10, dtype=int)) is None


class TestEvaluateLabels:
    def test_incl_noise_silhouette_differs_for_noisy_runs(self):
        X, labels = blobs(n_per=40)
        noisy = labels.copy()
        noisy[:10] = -1
        m = evaluate_labels(X, noisy, ["a b"] * len(noisy), None, compute_coh=False)
        assert m["silhouette"] is not None
        assert m["silhouette_incl_noise"] is not None
        # the pseudo-cluster of scattered noise must hurt the score
        assert m["silhouette_incl_noise"] < m["silhouette"]

    def test_partitioner_silhouettes_match(self):
        X, labels = blobs(n_per=40)
        m = evaluate_labels(X, labels, ["a b"] * len(labels), None, compute_coh=False)
        assert abs(m["silhouette"] - m["silhouette_incl_noise"]) < 1e-6


class TestStabilityARI:
    def test_identical_runs_score_one(self):
        labels = np.array([0, 0, 1, 1, -1])
        assert stability_ari([labels, labels.copy()])["ari_mean"] == 1.0

    def test_permuted_labels_still_score_one(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([2, 2, 0, 0, 1, 1])  # same partition, renamed
        assert stability_ari([a, b])["ari_mean"] == 1.0

    def test_single_run_returns_none(self):
        assert stability_ari([np.array([0, 1])])["ari_mean"] is None


class TestFailureFlags:
    def test_giant_cluster_flagged(self):
        flags = failure_flags({"max_cluster_share": 0.8, "n_clusters": 5})
        assert any("giant" in f for f in flags)

    def test_sentiment_blob_flagged(self):
        flags = failure_flags({"rating_entropy": 0.3, "n_clusters": 5})
        assert any("sentiment" in f for f in flags)

    def test_duplicate_clusters_flagged_by_term_overlap(self):
        terms = {
            0: [("room", 1), ("bed", 1), ("clean", 1)],
            1: [("room", 1), ("bed", 1), ("dirty", 1)],
            2: [("breakfast", 1), ("coffee", 1), ("eggs", 1)],
        }
        flags = failure_flags({"n_clusters": 3}, cluster_terms=terms)
        assert any("near-duplicate clusters 0 and 1" in f for f in flags)
        assert not any("2" in f and "near-duplicate" in f for f in flags)

    def test_clean_run_no_flags(self):
        m = {"max_cluster_share": 0.2, "noise_ratio": 0.1,
             "rating_entropy": 0.9, "n_clusters": 12}
        assert failure_flags(m) == []
