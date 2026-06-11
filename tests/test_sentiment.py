import numpy as np

from reviewscope_ml.sentiment import (
    LABELS,
    aggregate_cluster_sentiment,
    score_to_label,
)


class TestScoreToLabel:
    def test_thresholds(self):
        # team decision: negative < -0.2, neutral in [-0.2, 0.2], positive > 0.2
        assert score_to_label(-0.9) == "negative"
        assert score_to_label(-0.21) == "negative"
        assert score_to_label(-0.2) == "neutral"
        assert score_to_label(0.0) == "neutral"
        assert score_to_label(0.2) == "neutral"
        assert score_to_label(0.21) == "positive"
        assert score_to_label(0.9) == "positive"


class TestAggregation:
    def test_mean_and_distribution(self):
        scores = np.array([0.8, 0.6, -0.5, 0.0])
        labels = np.array([1, 1, 1, 1])
        avg, dist = aggregate_cluster_sentiment(scores, labels, 1)
        assert avg == round((0.8 + 0.6 - 0.5 + 0.0) / 4, 4)
        assert dist == {"negative": 0.25, "neutral": 0.25, "positive": 0.5}
        assert set(dist) == set(LABELS)

    def test_only_targets_requested_cluster(self):
        scores = np.array([1.0, -1.0])
        labels = np.array([0, 1])
        avg, _ = aggregate_cluster_sentiment(scores, labels, 0)
        assert avg == 1.0

    def test_empty_cluster_returns_none(self):
        avg, dist = aggregate_cluster_sentiment(np.array([0.5]), np.array([0]), 99)
        assert avg is None and dist is None


class TestArtifactRoundtripWithSentiment:
    def test_sentiment_columns_roundtrip(self, small_run, tmp_path):
        from reviewscope_ml.pipelines.artifacts import load_run, save_run
        from reviewscope_ml.sentiment import score_to_label

        n = len(small_run.doc_ids)
        small_run.sentiment_scores = np.linspace(-1, 1, n).astype(np.float32)
        small_run.sentiment_labels = [
            score_to_label(float(s)) for s in small_run.sentiment_scores
        ]
        save_run(tmp_path / "r", small_run)
        loaded = load_run(tmp_path / "r")
        assert np.allclose(loaded.sentiment_scores, small_run.sentiment_scores, atol=1e-4)
        assert loaded.sentiment_labels == small_run.sentiment_labels

    def test_runs_without_sentiment_still_load(self, small_run, tmp_path):
        from reviewscope_ml.pipelines.artifacts import load_run, save_run

        save_run(tmp_path / "r", small_run)  # no sentiment set
        loaded = load_run(tmp_path / "r")
        assert loaded.sentiment_scores is None
        assert loaded.sentiment_labels is None
