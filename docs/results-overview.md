# Results overview — what exists, what it says, what's next

Snapshot of all experiment outputs as of **2026-06-11**, with an evaluation
of each. Companion docs: `pipeline-guide.md` (how everything works and why),
`methodology.md` (threats to validity). This file is the index you open
first when you want to know "where are we?".

---

## 1. Output inventory

| Artifact | Path | What it is |
|---|---|---|
| Run directories | `data/runs/<variant>__[model__][corpus__]<size>__s<seed>/` | one finished pipeline run: `manifest.json` (provenance, per-stage cost, human confirmation), `assignments.csv` (doc → cluster, 2D/3D coords, sentiment score+label), `clusters.json` (label, terms, samples, sentiment avg+distribution), `metrics.json`; sentence runs add `doc_membership.json` |
| Comparison reports | `data/runs/comparison_<size>[_<tag>]/report.md` + charts | multi-variant comparison: metric tables, noise fairness, ARI stability, failure flags, inspection sheets + intruder tests, human sign-off block |
| Model sweep reports | `data/runs/model_sweep_<size>[_sent].md` | embedding-model ranking under the fixed downstream pipeline |
| Progress logs | `data/runs/*.log` | full logs of every CLI run — `tail -f` these |
| Reviewer feedback | `data/feedback/<run>__<timestamp>.jsonl` | append-only HITL actions, one file per session |
| Shared results log | `data/cache/results.csv` | every experiment row (notebooks 03–08 + package runs), dedup by config hash |
| Stage caches | `data/cache/{embeddings,umap,clustering,sentiment}/` | resumability; keyed on model × instruction × corpus × unit × seed × params |

## 2. Evaluation of everything that has run

### 2.1 Embedding model sweep — sentence level (`model_sweep_5000_sent.md`)

6 of 7 registry models (EmbeddingGemma pending HF login). **MiniLM-L6-v2
(22M) ranks #1** by mean rank — short sentences are its regime, and entropy
0.918 says strongly thematic clusters. **Qwen3-0.6B** has the *honesty
profile*: lowest noise (33%) and the best incl.-noise silhouette (0.148).
bge-m3 illustrates the noise-fairness problem perfectly: best C_v, but 46%
of mentions discarded and incl.-noise silhouette of −0.03. The
instruction-tuned e5-large brought nothing for 25× MiniLM's size.
**Document-level sweep: not yet run** — required before the remaining 50k
variants get a final embedding choice.

### 2.2 Sentence-level finalists head-to-head (5k, full pipeline)

| | MiniLM | Qwen3-0.6B |
|---|---|---|
| clusters | 61 | 57 |
| noise | 34.7% | **32.8%** |
| silhouette (excl/incl) | **0.589** / 0.117 | 0.544 / **0.148** |
| C_v | 0.572 | 0.572 |
| rating entropy (dedup) | **0.918** | 0.854 |
| ARI stability (mean/min) | 0.682 / 0.661 | **0.744 / 0.729** |

Genuine trade-off: MiniLM = more thematic + better classic geometry;
Qwen3 = more stable across seeds + keeps more mentions. **Decision belongs
to the inspection sheets** (`comparison_5000_<model>/report.md`, Step 2) and
the HITL sign-off — not to this table. Note Qwen3's stability edge (0.744)
matters for WP9b.

### 2.3 Sentiment stage (new — backfilled into all runs)

Aspect-level sentiment on the MiniLM 5k run behaves exactly as intended:

| Cluster (terms) | mentions / reviews | avg score | neg/neu/pos |
|---|---|---|---|
| staff / friendly / helpful | 915 / 822 | **+0.70** | 11/3/86% |
| location / beach / walking | 1040 / 829 | +0.49 | 9/23/69% |
| shower / bathroom / water | 1041 / 724 | **−0.08** | 49/20/32% |
| clean / rooms / room | 1755 / 1371 | +0.38 | 22/16/62% |

Reading: clusters are *thematic* (mixed sentiment within each — the Tier-3
story confirmed at aspect level), and the per-cluster polarity is plausible
(bathrooms complain, staff delights). Per-unit score+label are in
`assignments.csv`; this is the data the app's sentiment badges will consume.
Caveat: ±0.2 thresholds are a team convention; "okay, nothing special"
scores +0.24 = barely positive.

### 2.4 Five-variant comparison at 1k (`comparison_1000/report.md`)

Smoke-scale only — read for *shape*, not for verdicts: custom_hdbscan shows
the giant-cluster pattern (63% — the very motivation for the now-active
size-scaled mcs), BERTopic's silhouette collapses 0.63→0.22 when its 36%
noise counts, flat_agglomerative is most stable but near-duplicate-prone,
sentence_level finds 74 fine-grained clusters at 42% noise. All 1k runs
now carry sentiment fields.

### 2.5 50k comparison — **incomplete**

`bertopic__50000__s42` finished (now with sentiment). The remaining three
document-level variants were interrupted mid-embed (log ends 15:53). The
caches keep everything done so far; resume with:

```bash
python -m reviewscope_ml.eval.report --sample-size 50000 --device cuda \
    --variants custom_hdbscan flat_agglomerative two_stage
```

(Consider waiting for the document-level model sweep first — if a model
beats mpnet, the 50k run should use it from the start.)

### 2.6 HITL round trip

Demonstrated end to end on `two_stage__1000__s42`: 4 feedback records
(rename / approve / merge / micro-split) → `__reviewed` artifact with
7→13 clusters. The reviewed derivative predates the sentiment stage; it
regenerates via `apply_feedback` whenever needed. **No run has a human
sign-off yet** — by protocol there is therefore no winner yet, anywhere.

## 3. The workflow (current, condensed)

1. **Sweeps** → `model_sweep` (doc-level **still to do**; sentence-level done).
2. **Human picks embedding model(s)** from the three-tier ranking.
3. **Comparisons** → 50k doc-level (resume above), sentence-level finalists
   at 5k (done: MiniLM vs Qwen3, awaiting inspection).
4. **Human pass** → read inspection sheets + intruder tests in the report;
   correct + sign off in Streamlit (`streamlit run
   src/reviewscope_ml/hitl/app.py`; hover shows cluster/sentiment, the
   multiselect focuses clusters; large runs display a 12k-point sample).
5. **Apply feedback** → `python -m reviewscope_ml.hitl.apply_feedback <run>`.
6. Optional contrast corpus (Automotive) to test ranking transfer.

GPU runs claim every *idle* device (`--gpus auto`), cap VRAM at 50% each,
and auto-shrink batches on OOM. `--force` rebuilds artifacts of complete
runs from caches (used to backfill sentiment).

## 4. Open items (the honest to-do list)

Updated 2026-08-08 after the WP5 technology-selection re-run
(`docs/technology-selection.md`).

- [x] **Sentence-level embedding selection re-run** — 12 contestants,
      verified constraints, measured cost, multi-seed stability on the
      finalists. Winner: `all-MiniLM-L6-v2` (no instruction).
- [x] **GPU tier run** — closes one of methodology §3's three weaknesses.
- [x] **Ollama up + labeler scored** — nb08's never-completed experiment;
      5 models x 2 prompts, with a mention-level prompt variant.
- [x] **Confirm the finalists at 50k** — done 2026-08-08, 454,493 segments,
      ~48 min on an idle box. **The margin did not reproduce**: MiniLM and
      arctic tie on mean rank at 50k and the decision now rests on cost, not
      quality. Some head-to-head geometry margins flipped sign with scale.
      See technology-selection §6b.
- [ ] **Human inspection of the sentence finalists** → sign-off in the app
      (sidebar → "Model selection"; records a `confirm_model` action).
- [ ] **Human label scoring** — the oldest open item in the project, inherited
      from nb08. Blind multi-reviewer scoring now lives in the app (sidebar →
      "Label scoring"); nothing has been scored yet, so **every labeler claim
      is an automatic-metric claim**. Needs ≥2 reviewers for the agreement
      number to exist at all.
- [ ] **Multi-seed stability at 50k** — `--stability-seeds` was not passed on
      the 50k run, so every ARI figure remains a 5k figure.
- [ ] **Document-level model sweep** — still not run. Only needed if the
      document-unit variants are revived; the application runs sentence unit.
- [ ] EmbeddingGemma: still **blocked**. Logged in as `Dmnk-Mrtns`, but the
      repo is `gated: manual` and access is not granted (403, was 401).
      Accept the licence and wait for approval, then re-run the sweep.
- [ ] Category-leak decision (hotel-only filter?) — **now due**: the 50k
      numbers exist. Assessed as *not* changing the model ranking (all
      candidates see the same corpus), but cluster 3 of the labeled output is
      a Reno casino cluster, so any *inventory* of themes is wrong until the
      corpus is filtered — see technology-selection §8.
- [ ] Per-scale validation of the mcs auto-scaling (currently an anchored
      heuristic, see guide §6)
- [ ] The DR referee is still calibrated on mpnet, the model it displaced —
      the last of methodology §3's three weaknesses, and the only one the
      re-run did not close. Needs a per-model DR sweep.

Beyond these operational items, the gaps between "working pipeline" and
"really good results" — ground truth, noise rescue, language/dup hygiene,
domain adaptation, consensus stability — are analysed and prioritised in
**`quality-roadmap.md`**.
