# EuroParl Phase 2 — corpus, dataset-agnostic refactor, model re-selection

**Status:** corpus built and refactor landed; embedding sweep running.
**Date:** 2026-09-18

---

## 1. The corpus

Source: Europarl v7 source release (`https://www.statmt.org/europarl/v7/europarl.tgz`,
1.5 GB) → `data/raw/europarl.tgz`. Scope decision: **English only**, no
multilingual work (see §5).

### The `LANGUAGE` attribute works backwards

This is the finding that shapes everything else, and getting it wrong yields a
corpus of translationese that looks perfectly fine.

Each `<SPEAKER>` block may carry a `LANGUAGE="XX"` attribute. The attribute is
**omitted when the speech was originally delivered in that file's own
language**, and present — naming the *source* language — when the text is a
translation.

Evidence, measured on this dump:

| Observation | Count |
|---|---|
| `<SPEAKER>` blocks in `txt/en/` | 211,359 |
| …tagged `LANGUAGE="EN"` | **95** |
| …carrying no `LANGUAGE` attribute | 103,650 |

Cross-checked against `txt/de/` on `ep-00-01-17`: speakers 2 and 4 are untagged
in `txt/en/` and tagged `LANGUAGE="EN"` in `txt/de/`; speaker 8 is tagged
`LANGUAGE="DE"` in `txt/en/` and untagged in `txt/de/`. The convention holds in
both directions.

So "English originals" are the untagged blocks, **not** the 95 tagged `EN`.

### Untagged is necessary but not sufficient

Cross-checking 18,371 untagged blocks (600 files) against `txt/de/`:

| Verdict | Share |
|---|---|
| Confirmed English by the pivot | 34.9% |
| Unannotated in both (ambiguous) | 58.0% |
| **Contradicted** — pivot names another source language | **6.3%** |

The contradictions are translations whose English tag was simply dropped. And
**44% of untagged blocks are the sitting President** ("President",
"President.", "President. –"); 71% of the everywhere-unannotated group is the
chair. Those are procedural formulae — "the debate is closed", "the vote will
take place tomorrow" — repeated thousands of times. They are unannotated
because nobody recorded the chair's language, not because they are English, and
at clustering time they are exactly the near-duplicate mass that
`eval.harness.failure_flags` exists to catch.

### Selection modes

`data/sources/europarl.py` therefore offers two, with the chair excluded by
default in both:

- **`verified`** (default) — untagged in `txt/en/` **and** explicitly tagged
  `LANGUAGE="EN"` by at least one pivot language (`de, fr, es, it, nl`).
  Near-zero translationese. **This is what was built.**
- **`untagged`** — untagged in `txt/en/`. Larger, ~11% contaminated among
  non-chair speakers. For when size matters more than purity.

### What was built

```
data/cache/europarl-en_verified_full.jsonl   26,391 speeches   (51 MB)
data/cache/sample_europarl-en_5k.jsonl        5,000 speeches   (8.3 MB)
data/cache/sample_europarl-en_1k.jsonl        1,000 speeches   (1.7 MB)
```

Rebuild: `python -m reviewscope_ml.data.sources.europarl --sample-size 5000`

| Property | Value |
|---|---|
| Verified English originals available | 26,391 |
| Date range | 1996-04-15 → 2011-06-23 |
| Distinct speakers | 2,428 |
| Median speech length | 1,336 chars (hotel reviews: ~500–700) |
| 5,000 speeches → segments | **58,230** (11.6 per speech) |
| Ratings | **none** |

Dropped when building from 207,946 blocks: 92,116 translated, 27,722 chair,
61,520 unconfirmed, 197 too short.

Smaller files are strict prefixes of larger ones (`subset_sample`), and files
are visited in **chronological** order — a plain filename sort puts `ep-00-…`
(2000) ahead of `ep-96-…` (1996).

> The verified corpus caps at 26,391 speeches, so a 50k benchmark is not
> available in this mode. Use `--selection untagged` if 50k is needed, and
> state the ~11% contamination wherever the result is reported.

---

## 2. Dataset-agnostic refactor

### `data/sources/` — one module per raw distribution format

```
data/sources/yelp.py       build_benchmark_sample, subset_sample   (moved, unchanged)
data/sources/europarl.py   build_europarl_sample, parse_transcript (new)
```

Input is the vendor archive under `data/raw/`; output is a JSONL under
`data/cache/` whose lines carry at least `text`. `data.ingest.load_benchmark`
reads that and is the only thing the rest of the package talks to, so adding a
corpus never touches a pipeline stage. Old import paths still work.

### `load_benchmark` no longer assumes Yelp

Document keys are resolved from `DOC_ID_FIELDS`
(`review_id, speech_id, doc_id, id`) and ratings from `RATING_FIELDS`
(`stars, rating, score`). New `ReviewSet.has_ratings` lets callers skip
rating-derived reporting instead of printing a number computed from an all-NaN
column.

### `core/corpus.py` — corpus profiles

The pipeline stages are corpus-agnostic; a handful of *texts* were not. One
registry entry per corpus now carries: `unit_noun`, `domain_description`,
`has_ratings`, the embedding-instruction table, the label prompt variant, and
`boilerplate_terms`. Registered: `hotels`, `europarl-en`, plus a `NEUTRAL`
fallback that every application upload (`proj-<id>`) lands on — deliberately
saying nothing about a domain, because a wrong domain hint is worse than none.

**Cache-safety rule:** instruction slugs are baked into embedding cache
filenames, but the corpus is already part of that key
(`runner._corpus_prefix`), so two corpora may reuse a slug name with different
text. Never change the text behind an existing *(corpus, slug)* pair — the
cache would silently serve vectors built from the old instruction.

Hotel instruction texts are frozen verbatim: `docs/technology-selection.md`'s
numbers were produced under exactly those strings.

### Ratings are now properly optional

`eval/model_sweep._mean_ranks` previously ranked every row **last** on a metric
no row had, adding a constant to every mean. Harmless to ordering, but it
inflated `mean rank` and made the column incomparable with a sweep that had all
four metrics. It now skips such metrics entirely and returns which ones
counted; the report states them, drops the empty column, and prints an explicit
warning that Tier 3 is unavailable and the sentiment-blob failure mode is
therefore **not** screened for on this corpus.

---

## 3. Configurable label prompts

Layered, registry-first:

1. **Named variants** in `PROMPT_VARIANTS` (`label/ollama.py`), selected by the
   corpus profile. Keeps `prompt_hash` provenance meaningful and keeps
   `eval/label_sweep.py` able to A/B variants — machinery that only works on a
   closed, named set.
2. **`neutral`** — a new domain-*parameterised* pair keeping v2's proven
   structure (name the unit, ask for the topic, forbid stance-only labels) with
   `{domain}` filled from the profile. `v1` and `v2_mention` are untouched, so
   existing artifacts keep their hashes.
3. **Per-project free-text override** — `run_from_upload(label_prompt=…,
   summary_prompt=…)` → `PipelineSpec.label_prompt/summary_prompt` →
   `OllamaLabeler`. Both must be supplied together; a half-filled override is
   ignored. The text is hashed into every label, so provenance survives.

### Bug fixed on the way

`runner.py` built `OllamaLabeler(model=spec.label_model)` with no
`prompt_variant`, so it always fell back to `"v1"` — the **document** prompt —
even for `sentence_level` runs. `v2_mention` was unreachable from the pipeline,
meaning every sentence-level app run was labelling mentions with the document
prompt. The unit and the prompt now agree.

---

## 4. Embedding model re-selection (running)

Same experiment as WP5, on the new corpus: every registry candidate under the
fixed downstream pipeline (UMAP 10d/nn15 + HDBSCAN with size-scaled
parameters), so the embedding model stays the only manipulated variable.

```
tmux new-session -d -s revscope-europarl "bash scripts/run_europarl_sweep.sh"
tail -f data/runs/europarl_sweep.log
```

5,000 speeches → 58,230 segments (WP5 was 43,012, so the tables are comparable
in scale). 13 contestants. `--gpus 2` claims at most two *idle* devices.

**Instruction texts are corpus-specific now** — running
`"Represent the hotel review…"` over parliamentary speech would have been a
confound in the two instruction-tuned contestants.

Outputs:
- `data/runs/model_sweep_5000_europarl-en_sent.md` — ranking report
- `data/runs/europarl_sweep_status.txt` — completion status
- `data/runs/europarl_sweep.log` — full log

Cleanup runs from an `EXIT` trap (success, failure and Ctrl-C alike): GPUs
released, the 2.3 GB of extracted language directories deleted (re-derivable
from the tarball), status file written. Kept: `europarl.tgz`, the sample
JSONLs, and the embedding/UMAP/clustering caches — those are the expensive
artifacts and make a re-run nearly free.

**The ranking is a shortlist, not a verdict.** WP5's own conclusion was that
metrics shortlist and human inspection decides; with Tier 3 unavailable here
that is more true, not less. Eyeball the finalists before picking:

```
streamlit run src/reviewscope_ml/hitl/app.py   # sidebar -> "Model selection"
```

Pick the Europarl report in the browser and the **Benchmark** selector below it
follows the report's corpus; "Load clusters" then rebuilds any candidate from
the sweep's own caches (no model loads, nothing re-embeds), so what you read is
what the table scored. `eval/inspection.py` renders the same thing from a
*finished pipeline run*, which Europarl does not have yet — the sweep produced
caches and a ranking, not a run directory.

---

## 5. Explicitly out of scope / still open

- **Multilingual.** Dropped by decision. The registered multilingual candidates
  (`bge-m3`, `arctic-embed-l-v2.0`, `multilingual-e5`) still have **zero**
  evidence either way — they compete here on English only.
- **Sentence segmentation is still the English regex splitter** with
  review-tuned bounds (`MIN_CHARS=20`, `MAX_CHARS=600`). Parliamentary text is
  full of "Mr.", "Art. 5", "No. 12" — each a false split — and speeches have
  genuinely long sentences, so the 600-char hard wrap fires on real ones. pysbd
  is the known-worthwhile swap and **is not installed**; it needs a new optional
  extra in `pyproject.toml`. Held constant across contestants, so it does not
  bias *this* comparison — but it caps absolute quality.
- **`boilerplate_terms` are defined but not yet wired** into `represent/terms.py`.
  Until then, "president", "commission" and "parliament" will top the keyword
  list of most clusters.
- **Backend prompt override has no DB column or UI.** The ML seam is done; the
  API still needs a `projects.label_prompt` / `summary_prompt` migration and a
  pass-through in `ml_pipeline.run_ml_pipeline`.
- **Sentiment stage** still runs `cardiffnlp/twitter-roberta-base-sentiment-latest`,
  English tweet-domain, on parliamentary rhetoric. It will not fail — it will
  return confident noise. Consider gating it off for this corpus.
- **Tier-3 substitute.** Europarl has no ratings but does carry `speaker`,
  `date` and `chapter`. Generalising `compute_rating_entropy` to a categorical
  `compute_metadata_entropy(values, labels)` would restore the "is this cluster
  a metadata artifact or a real topic" test rather than losing it.
