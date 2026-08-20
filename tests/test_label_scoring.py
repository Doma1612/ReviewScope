import numpy as np
import pytest

from reviewscope_ml.eval.label_scoring import (
    build_scoring_items,
    collect_scores,
    coverage,
    krippendorff_alpha,
    label_key,
    pairwise_agreement,
    score_by_labeler,
    select_exemplars,
)
from reviewscope_ml.hitl.feedback import FeedbackRecord


def rec(labeler, variant, cid, label):
    return {
        "labeler": labeler, "prompt_variant": variant,
        "cluster_id": cid, "label": label,
    }


RECORDS = [
    rec("qwen3:4b", "v1", 3, "Casino in Reno"),
    rec("gemma3:4b", "v1", 3, "casino in reno"),   # same words, different case
    rec("phi4-mini", "v1", 3, "Food"),
    rec("qwen3:4b", "v1", 7, "Breakfast Quality"),
    rec("gemma3:4b", "v1", 7, "Room Cleanliness"),
]


class TestBuildScoringItems:
    def test_identical_labels_collapse_to_one_judgement(self):
        items = build_scoring_items(RECORDS)
        casino = [i for i in items[3] if i.key == "casino in reno"]
        assert len(casino) == 1, "case-variant labels must not be scored twice"
        assert casino[0].produced_by == [("gemma3:4b", "v1"), ("qwen3:4b", "v1")]

    def test_distinct_labels_stay_separate(self):
        items = build_scoring_items(RECORDS)
        assert len(items[3]) == 2  # "Casino in Reno" + "Food"
        assert len(items[7]) == 2

    def test_order_is_stable_across_calls(self):
        # A reviewer reloading the page must not be shown a reshuffled list.
        a = [i.label for i in build_scoring_items(RECORDS)[3]]
        b = [i.label for i in build_scoring_items(RECORDS)[3]]
        assert a == b

    def test_order_does_not_follow_input_order(self):
        # Blind means no positional advantage; with enough labels the shuffle
        # must actually move something.
        many = [rec("m", "v1", 1, f"label {i}") for i in range(12)]
        shown = [i.label for i in build_scoring_items(many)[1]]
        assert shown != [r["label"] for r in many]
        assert sorted(shown) == sorted(r["label"] for r in many)

    def test_empty_and_placeholder_labels_are_dropped(self):
        items = build_scoring_items(
            [rec("m", "v1", 1, "—"), rec("m", "v2", 1, "  "), rec("m", "v3", 1, "Ok")]
        )
        assert [i.label for i in items[1]] == ["Ok"]

    def test_label_key_normalises_whitespace_and_case(self):
        assert label_key("  Casino   In Reno ") == label_key("casino in reno")


class TestCollectScores:
    def test_ignores_non_scoring_actions(self):
        records = [
            FeedbackRecord("r", "ann", "confirm_model", note="x"),
            FeedbackRecord("r", "ann", "score_label", cluster_id=3,
                           label="Food", score=2),
        ]
        assert collect_scores(records) == {(3, "food"): {"ann": 2}}

    def test_last_score_from_a_reviewer_wins(self):
        records = [
            FeedbackRecord("r", "ann", "score_label", cluster_id=3,
                           label="Food", score=2),
            FeedbackRecord("r", "ann", "score_label", cluster_id=3,
                           label="Food", score=5),
        ]
        assert collect_scores(records)[(3, "food")]["ann"] == 5

    def test_out_of_range_score_is_rejected_at_write_time(self):
        with pytest.raises(ValueError):
            FeedbackRecord("r", "ann", "score_label", cluster_id=3,
                           label="Food", score=6)
        with pytest.raises(ValueError):
            FeedbackRecord("r", "ann", "score_label", cluster_id=3, label="Food")


class TestScoreByLabeler:
    def test_shared_label_credits_every_producer(self):
        items = build_scoring_items(RECORDS)
        scores = {(3, "casino in reno"): {"ann": 5}, (3, "food"): {"ann": 1}}
        rows = {(r["labeler"], r["prompt"]): r for r in score_by_labeler(items, scores)}
        assert rows[("qwen3:4b", "v1")]["human_score"] == 5
        assert rows[("gemma3:4b", "v1")]["human_score"] == 5
        assert rows[("phi4-mini", "v1")]["human_score"] == 1

    def test_averages_across_reviewers_then_across_clusters(self):
        items = build_scoring_items(RECORDS)
        scores = {
            (3, "casino in reno"): {"ann": 4, "bob": 2},   # mean 3
            (7, "breakfast quality"): {"ann": 5},          # mean 5
        }
        row = next(
            r for r in score_by_labeler(items, scores) if r["labeler"] == "qwen3:4b"
        )
        assert row["human_score"] == 4.0
        assert row["n_clusters"] == 2

    def test_unscored_clusters_are_excluded_not_counted_as_zero(self):
        items = build_scoring_items(RECORDS)
        scores = {(3, "casino in reno"): {"ann": 4}}
        row = next(
            r for r in score_by_labeler(items, scores) if r["labeler"] == "qwen3:4b"
        )
        assert row["human_score"] == 4.0
        assert row["n_clusters"] == 1  # cluster 7 unscored, not averaged in as 0


class TestCoverage:
    def test_per_reviewer_and_overall(self):
        items = build_scoring_items(RECORDS)
        scores = {(3, "food"): {"ann": 3}, (7, "room cleanliness"): {"bob": 4}}
        assert coverage(items, scores) == (2, 4)
        assert coverage(items, scores, reviewer="ann") == (1, 4)
        assert coverage(items, scores, reviewer="zoe") == (0, 4)


class TestAgreement:
    def test_alpha_is_none_without_double_scoring(self):
        assert krippendorff_alpha({(1, "a"): {"ann": 3}}) is None

    def test_alpha_is_one_for_identical_ratings(self):
        scores = {
            (1, "a"): {"ann": 1, "bob": 1},
            (2, "b"): {"ann": 5, "bob": 5},
            (3, "c"): {"ann": 3, "bob": 3},
        }
        assert krippendorff_alpha(scores) == pytest.approx(1.0)

    def test_alpha_is_negative_for_systematic_disagreement(self):
        scores = {
            (1, "a"): {"ann": 1, "bob": 5},
            (2, "b"): {"ann": 5, "bob": 1},
            (3, "c"): {"ann": 1, "bob": 5},
            (4, "d"): {"ann": 5, "bob": 1},
        }
        assert krippendorff_alpha(scores) < 0

    def test_alpha_is_none_when_everyone_used_one_value(self):
        # No variance: perfectly consistent but the statistic is undefined,
        # and reporting 1.0 there would be a lie about reliability.
        scores = {(1, "a"): {"ann": 4, "bob": 4}, (2, "b"): {"ann": 4, "bob": 4}}
        assert krippendorff_alpha(scores) is None

    def test_ordinal_scale_treats_near_misses_as_near_agreement(self):
        close = {
            (1, "a"): {"ann": 3, "bob": 4}, (2, "b"): {"ann": 4, "bob": 3},
            (3, "c"): {"ann": 1, "bob": 2}, (4, "d"): {"ann": 5, "bob": 4},
        }
        far = {
            (1, "a"): {"ann": 1, "bob": 5}, (2, "b"): {"ann": 5, "bob": 1},
            (3, "c"): {"ann": 1, "bob": 4}, (4, "d"): {"ann": 5, "bob": 2},
        }
        assert krippendorff_alpha(close) > krippendorff_alpha(far)

    def test_pairwise_reports_overlap_only(self):
        scores = {
            (1, "a"): {"ann": 3, "bob": 3},
            (2, "b"): {"ann": 5, "bob": 4},
            (3, "c"): {"ann": 2},            # bob never scored this one
        }
        (row,) = pairwise_agreement(scores)
        assert (row["a"], row["b"], row["n"]) == ("ann", "bob", 2)
        assert row["exact"] == 0.5
        assert row["within_1"] == 1.0
        assert row["mean_abs_diff"] == 0.5


class TestSelectExemplars:
    def test_draws_from_the_requested_cluster_only(self):
        labels = np.array([0, 0, 0, 1, 1, 1, -1])
        texts = [f"t{i}" for i in range(7)]
        out = select_exemplars(labels, texts, [0, 1], seed=42, k=2)
        assert set(out[0]) <= {"t0", "t1", "t2"}
        assert set(out[1]) <= {"t3", "t4", "t5"}

    def test_is_deterministic_and_order_dependent(self):
        # The generator advances once per cluster, so the sheet and the GUI
        # must iterate `chosen` identically — this pins that contract.
        labels = np.array([0] * 10 + [1] * 10)
        texts = [f"t{i}" for i in range(20)]
        a = select_exemplars(labels, texts, [0, 1], seed=42, k=3)
        b = select_exemplars(labels, texts, [0, 1], seed=42, k=3)
        c = select_exemplars(labels, texts, [1, 0], seed=42, k=3)
        assert a == b
        assert a[1] != c[1]

    def test_truncates_long_mentions(self):
        labels = np.array([0])
        out = select_exemplars(labels, ["x" * 500], [0], seed=1, k=1)
        assert len(out[0][0]) == 300
