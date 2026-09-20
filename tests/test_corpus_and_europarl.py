"""Corpus profiles, prompt resolution, and the Europarl source adapter."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reviewscope_ml.core.config import load_config
from reviewscope_ml.core.corpus import EUROPARL_EN, HOTELS, NEUTRAL, profile_for
from reviewscope_ml.data.ingest import load_benchmark
from reviewscope_ml.data.sources.europarl import (
    Speech, _file_date, _normalise_name, parse_transcript,
)
from reviewscope_ml.eval.model_sweep import _mean_ranks
from reviewscope_ml.label.ollama import PROMPT_VARIANTS, prompt_hash, resolve_prompts


class TestCorpusProfiles:
    def test_registered_lookup(self):
        assert profile_for("hotels") is HOTELS
        assert profile_for("europarl-en") is EUROPARL_EN

    def test_unregistered_falls_back_to_neutral(self):
        # Every application upload gets its own proj-<id> slug, so this is the
        # common case and must never raise.
        assert profile_for("proj-deadbeef") is NEUTRAL
        assert profile_for("") is NEUTRAL

    def test_every_profile_has_no_inst(self):
        for p in (HOTELS, EUROPARL_EN, NEUTRAL):
            assert p.instructions["no_inst"] is None

    def test_hotel_instructions_are_frozen(self):
        # docs/technology-selection.md's numbers were produced under exactly
        # these strings; changing one silently invalidates that table.
        assert HOTELS.instructions["generic"] == (
            "Represent the hotel review for topic clustering:"
        )

    def test_europarl_reuses_slug_names_with_different_text(self):
        assert set(EUROPARL_EN.instructions) == set(HOTELS.instructions)
        assert EUROPARL_EN.instructions["generic"] != HOTELS.instructions["generic"]
        assert "hotel" not in EUROPARL_EN.instructions["domain"].lower()

    def test_europarl_declares_no_ratings(self):
        assert EUROPARL_EN.has_ratings is False
        assert HOTELS.has_ratings is True


class TestPromptResolution:
    def test_named_variant(self):
        label, summary = resolve_prompts("v2_mention")
        assert label == PROMPT_VARIANTS["v2_mention"][0]

    def test_domain_substituted_and_docs_placeholder_survives(self):
        label, summary = resolve_prompts("neutral", "speeches in the European Parliament")
        assert "speeches in the European Parliament" in label
        assert "{domain}" not in label and "{domain}" not in summary
        # The caller still has to fill these in, so they must not be consumed.
        assert "{docs}" in label
        assert "{docs}" in summary and "{label}" in summary

    def test_override_wins_over_variant(self):
        label, summary = resolve_prompts(
            "neutral", "anything", label_prompt="L {docs}", summary_prompt="S {docs}"
        )
        assert (label, summary) == ("L {docs}", "S {docs}")

    def test_half_an_override_is_ignored(self):
        # Only one of the pair supplied -> fall back to the named variant.
        label, _ = resolve_prompts("v1", label_prompt="L {docs}")
        assert label == PROMPT_VARIANTS["v1"][0]

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt_variant"):
            resolve_prompts("does_not_exist")

    def test_different_domains_hash_differently(self):
        a = prompt_hash(*resolve_prompts("neutral", "hotel reviews"))
        b = prompt_hash(*resolve_prompts("neutral", "parliamentary speeches"))
        assert a != b, "provenance hash must distinguish the actual prompt text"


class TestEuroparlParsing:
    TRANSCRIPT = (
        '<CHAPTER ID=1>\n'
        'Resumption of the session\n'
        '<SPEAKER ID=1 NAME="President">\n'
        'I declare resumed the session.\n'
        '<P>\n'
        'Please rise for a minute of silence.\n'
        '\n'
        '<SPEAKER ID=2 NAME="Evans, Robert J">\n'
        'Madam President, on a point of order about Sri Lanka.\n'
        '\n'
        '<SPEAKER ID=3 LANGUAGE="DE" NAME="Schroedter">\n'
        'This was translated from German.\n'
    )

    def parse(self, tmp_path):
        p = tmp_path / "ep-00-01-17.txt"
        p.write_text(self.TRANSCRIPT, encoding="utf-8")
        return list(parse_transcript(p))

    def test_speech_boundaries_and_paragraph_joining(self, tmp_path):
        speeches = self.parse(tmp_path)
        assert [s.speech_id for s in speeches] == [
            "ep-00-01-17#1", "ep-00-01-17#2", "ep-00-01-17#3",
        ]
        # <P> is a paragraph mark, not content, and the parts are joined.
        assert speeches[0].text == (
            "I declare resumed the session. Please rise for a minute of silence."
        )

    def test_language_attribute_semantics(self, tmp_path):
        speeches = self.parse(tmp_path)
        # Untagged = originally English (the inverse of the obvious reading).
        assert speeches[0].language is None
        assert speeches[1].language is None
        # Tagged = translated FROM that language, so not an English original.
        assert speeches[2].language == "DE"

    def test_chapter_title_attached_not_treated_as_speech(self, tmp_path):
        speeches = self.parse(tmp_path)
        assert all(s.chapter == "Resumption of the session" for s in speeches)
        assert "Resumption of the session" not in speeches[0].text

    def test_chair_detected(self, tmp_path):
        speeches = self.parse(tmp_path)
        assert speeches[0].is_chair is True
        assert speeches[1].is_chair is False

    @pytest.mark.parametrize("name", ["President", "President.", "President. –", " President "])
    def test_chair_name_noise_variants(self, name):
        assert _normalise_name(name) == "president"

    def test_president_in_office_is_not_the_chair(self):
        # A Council representative is a substantive speaker, not the chair.
        assert _normalise_name("President-in-Office of the Council") != "president"

    @pytest.mark.parametrize("stem,expected", [
        ("ep-96-04-15", "1996-04-15"),      # two-digit years straddle 2000
        ("ep-00-01-17", "2000-01-17"),
        ("ep-11-03-09-010-03", "2011-03-09"),
    ])
    def test_file_date(self, stem, expected):
        assert _file_date(stem) == expected


class TestRatinglessCorpus:
    def write(self, tmp_path, rows):
        cache = tmp_path / "data" / "cache"
        cache.mkdir(parents=True)
        path = cache / "sample_europarl-en_1k.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return load_config(
            sample_size=1_000, project_root=tmp_path,
            data_file="sample_europarl-en_1k.jsonl",
        )

    def test_speech_id_used_as_key_and_no_ratings(self, tmp_path):
        cfg = self.write(tmp_path, [
            {"speech_id": "ep-00-01-17#2", "text": "A speech about fisheries policy."},
            {"speech_id": "ep-00-01-17#4", "text": "A speech about the budget."},
        ])
        rs = load_benchmark(cfg)
        assert rs.ids == ["ep-00-01-17#2", "ep-00-01-17#4"]
        assert rs.has_ratings is False
        assert all(math.isnan(s) for s in rs.stars)

    def test_corpus_slug_round_trips_through_the_hyphen(self, tmp_path):
        # corpus_slug splits the stem on its LAST underscore, so the corpus
        # token must not contain one — "europarl-en", never "europarl_en".
        cfg = self.write(tmp_path, [{"speech_id": "x#1", "text": "t"}])
        assert cfg.corpus_slug == "europarl-en"

    def test_yelp_records_still_load(self, tmp_path):
        cache = tmp_path / "data" / "cache"
        cache.mkdir(parents=True)
        (cache / "sample_hotels_1k.jsonl").write_text(
            json.dumps({"review_id": "abc", "text": "Great room.", "stars": 5}),
            encoding="utf-8",
        )
        cfg = load_config(sample_size=1_000, project_root=tmp_path)
        rs = load_benchmark(cfg)
        assert rs.ids == ["abc"] and rs.stars[0] == 5.0
        assert rs.has_ratings is True


class TestSweepRankingWithoutRatings:
    def test_metric_no_one_has_is_excluded(self):
        rows = [
            {"silhouette": 0.5, "coherence_cv": 0.4, "rating_entropy": None},
            {"silhouette": 0.3, "coherence_cv": 0.6, "rating_entropy": None},
        ]
        used = _mean_ranks(rows)
        assert "rating_entropy" not in used
        # Each row is 1st on one metric and 2nd on the other -> mean 1.5.
        # Had rating_entropy been scored as a tie at last place it would be 1.67.
        assert rows[0]["_mean_rank"] == 1.5
        assert rows[1]["_mean_rank"] == 1.5

    def test_metric_someone_has_is_kept(self):
        rows = [
            {"silhouette": 0.5, "rating_entropy": 0.9},
            {"silhouette": 0.3, "rating_entropy": 0.7},
        ]
        assert "rating_entropy" in _mean_ranks(rows)


class TestBenchmarkPicker:
    """The HITL app's corpus selector.

    Its whole reason to exist is that ``load_config(sample_size=...)`` resolves
    to the *hotels* file, so inspecting a Europarl ranking without choosing the
    corpus silently rebuilds hotel clusters. These cover the discovery and the
    default that prevents that pairing.
    """

    @staticmethod
    def _app():
        pytest.importorskip("streamlit")  # hitl is an optional extra
        from reviewscope_ml.hitl import app

        return app

    @pytest.fixture
    def cfg(self, tmp_path):
        cache = tmp_path / "data" / "cache"
        cache.mkdir(parents=True)
        for name in (
            "sample_hotels_1k.jsonl",
            "sample_hotels_5k.jsonl",
            "sample_hotels_50k.jsonl",
            "sample_europarl-en_1k.jsonl",
            "sample_europarl-en_5k.jsonl",
            "sample_europarl-en_26391.jsonl",
            "europarl-en_verified_full.jsonl",  # not a sample — must be ignored
            "sample_broken.jsonl",              # no size token — must be ignored
        ):
            (cache / name).write_text("")
        return load_config(project_root=tmp_path)

    def test_discovers_samples_only(self, cfg):
        found = self._app()._benchmarks(cfg)
        assert [(slug, size) for _, slug, size in found] == [
            ("europarl-en", 1_000),
            ("europarl-en", 5_000),
            ("europarl-en", 26_391),
            ("hotels", 1_000),
            ("hotels", 5_000),
            ("hotels", 50_000),
        ]

    def test_slug_matches_config(self, cfg):
        # The slug shown in the picker must be the one the cache keys use, or
        # the reviewer picks a corpus and reads another's embeddings.
        for data_file, slug, _ in self._app()._benchmarks(cfg):
            assert load_config(data_file=data_file).corpus_slug == slug

    @pytest.mark.parametrize(
        "report, expected",
        [
            ("model_sweep_5000_europarl-en_sent.md", ("europarl-en", 5_000)),
            ("model_sweep_1000_europarl-en_sent.md", ("europarl-en", 1_000)),
            # No corpus tag means hotels — the historical default the sweep
            # omits from the filename.
            ("model_sweep_5000_sent.md", ("hotels", 5_000)),
            ("model_sweep_50000_sent_finalists.md", ("hotels", 50_000)),
        ],
    )
    def test_report_selects_its_own_corpus(self, cfg, report, expected):
        app = self._app()
        found = app._benchmarks(cfg)
        idx = app._preferred_benchmark(found, Path(report))
        assert found[idx][1:] == expected

    def test_sizeless_report_uses_the_default(self, cfg):
        # label_quality_*.json carries neither corpus nor size; scoring
        # campaigns aggregate across reviewers, so the fallback has to be the
        # size the labeler sweep ran at rather than the first file found.
        app = self._app()
        found = app._benchmarks(cfg)
        idx = app._preferred_benchmark(found, Path("label_quality_sent.json"), 5_000)
        assert found[idx][1:] == ("hotels", 5_000)
