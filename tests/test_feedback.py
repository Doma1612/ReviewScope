import numpy as np
import pytest

from reviewscope_ml.hitl.apply_feedback import apply_feedback
from reviewscope_ml.hitl.feedback import (
    FeedbackRecord,
    append_record,
    load_feedback,
    session_file,
)


def rec(action, run="testrun", **kw):
    return FeedbackRecord(run_name=run, reviewer="dominik", action=action, **kw)


class TestFeedbackIO:
    def test_unknown_action_rejected(self):
        with pytest.raises(ValueError):
            rec("delete_everything")

    def test_append_and_load_roundtrip(self, tmp_path):
        path = session_file(tmp_path, "testrun")
        append_record(path, rec("rename_label", cluster_id=0, new_label="Breakfast"))
        append_record(path, rec("mark_junk", cluster_id=3))

        records = load_feedback(tmp_path, "testrun")
        assert [r.action for r in records] == ["rename_label", "mark_junk"]
        assert records[0].new_label == "Breakfast"

    def test_load_only_matching_run(self, tmp_path):
        append_record(session_file(tmp_path, "runA"), rec("approve_label", run="runA", cluster_id=0))
        append_record(session_file(tmp_path, "runB"), rec("mark_junk", run="runB", cluster_id=0))
        assert [r.run_name for r in load_feedback(tmp_path, "runA")] == ["runA"]


class TestApplyFeedback:
    def test_rename_overrides_label(self, small_run):
        out = apply_feedback(small_run, [rec("rename_label", cluster_id=0, new_label="Pool area")])
        assert out.clusters[0].label == "Pool area"
        assert out.clusters[0].label_source == "hitl_override"
        # original untouched
        assert small_run.clusters[0].label == "label 0"

    def test_approve_stamps_source(self, small_run):
        out = apply_feedback(small_run, [rec("approve_label", cluster_id=1)])
        assert out.clusters[1].label_source == "hitl_approved"

    def test_merge_reassigns_documents_and_sums_sizes(self, small_run):
        out = apply_feedback(small_run, [rec("merge_clusters", cluster_id=1, merge_into=0)])
        assert 1 not in out.clusters
        assert out.clusters[0].size == 9
        assert (out.labels == 1).sum() == 0
        assert (out.labels == 0).sum() == 9

    def test_mark_junk_moves_docs_to_noise(self, small_run):
        out = apply_feedback(small_run, [rec("mark_junk", cluster_id=0)])
        assert 0 not in out.clusters
        assert (out.labels == -1).sum() == 8  # 3 original noise + 5 junked

    def test_reassign_doc(self, small_run):
        out = apply_feedback(
            small_run, [rec("reassign_doc", doc_id="doc0", target_cluster_id=1)]
        )
        assert out.labels[0] == 1
        assert out.clusters[1].size == 5
        assert out.clusters[0].size == 4

    def test_split_flat_run_flags_for_recluster(self, small_run):
        out = apply_feedback(small_run, [rec("split_cluster", cluster_id=0)])
        assert out.manifest["needs_recluster"] == [0]
        assert 0 in out.clusters  # not destroyed, only flagged

    def test_split_two_stage_promotes_micro_clusters(self, two_stage_run):
        out = apply_feedback(two_stage_run, [rec("split_cluster", cluster_id=0, run="twostage")])
        assert 0 not in out.clusters
        new_ids = [c for c in out.clusters if c != 1]
        assert len(new_ids) == 2  # micro 10 and 11 promoted
        sizes = sorted(out.clusters[c].size for c in new_ids)
        assert sizes == [2, 2]
        # macro cluster 1 untouched
        assert out.clusters[1].size == 2
        assert out.manifest.get("needs_recluster", []) == []

    def test_confirm_run_recorded_in_manifest(self, small_run):
        out = apply_feedback(small_run, [rec("confirm_run", note="looks coherent")])
        confirmed = out.manifest["human_confirmed"]
        assert confirmed["reviewer"] == "dominik"
        assert confirmed["note"] == "looks coherent"

    def test_order_merge_before_rename(self, small_run):
        # rename targets the merge survivor; both must apply
        out = apply_feedback(small_run, [
            rec("rename_label", cluster_id=0, new_label="Everything"),
            rec("merge_clusters", cluster_id=1, merge_into=0),
        ])
        assert out.clusters[0].label == "Everything"
        assert out.clusters[0].size == 9

    def test_result_is_new_run_name(self, small_run):
        out = apply_feedback(small_run, [rec("approve_label", cluster_id=0)])
        assert out.run_name == "testrun__reviewed"
