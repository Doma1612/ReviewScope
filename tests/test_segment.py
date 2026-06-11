import json

import numpy as np

from reviewscope_ml.data.ingest import ReviewSet
from reviewscope_ml.data.segment import parent_id, segment_reviews, split_sentences
from reviewscope_ml.pipelines.runner import _dedup_parent_stats, _write_doc_membership


class TestSplitSentences:
    def test_splits_on_terminators(self):
        out = split_sentences(
            "The room was spotless and quiet. Breakfast was cold every day! Would I return?  Maybe."
        )
        assert out == [
            "The room was spotless and quiet.",
            "Breakfast was cold every day!",
        ]  # "Would I return?" and "Maybe." are under min_chars

    def test_short_fragments_dropped(self):
        assert split_sentences("Great! Loved it. Wow.") == []

    def test_long_sentence_hard_wrapped(self):
        text = "word " * 300  # 1500 chars, no terminator
        out = split_sentences(text)
        assert len(out) > 1
        assert all(len(s) <= 600 for s in out)

    def test_deterministic(self):
        t = "The pool area was crowded but clean. Staff handled it well enough."
        assert split_sentences(t) == split_sentences(t)


class TestSegmentReviews:
    def make_reviews(self):
        texts = [
            "The room was spotless and quiet. Breakfast was cold every single day.",
            "Check-in took forever at the front desk.",
        ]
        return ReviewSet(
            ids=["rA", "rB"],
            texts=texts,
            raw_texts=texts,
            stars=np.array([4.0, 2.0]),
        )

    def test_ids_carry_parent(self):
        seg = segment_reviews(self.make_reviews())
        assert seg.ids == ["rA#0", "rA#1", "rB#0"]
        assert [parent_id(i) for i in seg.ids] == ["rA", "rA", "rB"]

    def test_stars_inherited_from_parent(self):
        seg = segment_reviews(self.make_reviews())
        assert list(seg.stars) == [4.0, 4.0, 2.0]


class TestDedupParentStats:
    def test_one_pair_per_review_and_cluster(self):
        ids = ["rA#0", "rA#1", "rA#2", "rB#0"]
        stars = np.array([5.0, 5.0, 5.0, 1.0])
        labels = np.array([0, 0, 1, 0])  # rA mentions cluster 0 twice
        dstars, dlabels = _dedup_parent_stats(ids, stars, labels)
        # (rA,0), (rA,1), (rB,0) — the duplicate (rA,0) collapsed
        assert len(dstars) == 3
        assert (dlabels == 0).sum() == 2

    def test_noise_excluded(self):
        ids = ["rA#0", "rB#0"]
        dstars, dlabels = _dedup_parent_stats(
            ids, np.array([3.0, 4.0]), np.array([-1, 2])
        )
        assert list(dlabels) == [2]


class TestDocMembership:
    def test_primary_and_shares(self, tmp_path):
        ids = ["rA#0", "rA#1", "rA#2", "rA#3", "rB#0"]
        labels = np.array([0, 0, 1, -1, 1])
        _write_doc_membership(tmp_path, ids, labels)
        m = json.loads((tmp_path / "doc_membership.json").read_text())
        assert m["rA"]["primary"] == 0
        assert m["rA"]["n_segments"] == 4
        assert m["rA"]["clusters"]["0"] == 0.5
        assert m["rA"]["clusters"]["-1"] == 0.25
        assert m["rB"]["primary"] == 1

    def test_all_noise_review_primary_is_noise(self, tmp_path):
        _write_doc_membership(tmp_path, ["rC#0"], np.array([-1]))
        m = json.loads((tmp_path / "doc_membership.json").read_text())
        assert m["rC"]["primary"] == -1


class TestModelRegistry:
    def test_registry_sane(self):
        from reviewscope_ml.embed.models import CANDIDATES, candidates

        names = [c.model for c in CANDIDATES]
        assert len(names) == len(set(names))
        # everything in the registry must fit the 6 GB Pascal VRAM slice
        assert all(c.params_m <= 700 for c in CANDIDATES)
        assert all(not c.gated for c in candidates(include_gated=False))
