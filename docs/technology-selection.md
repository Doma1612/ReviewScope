# WP5 technology selection — embedding and labeling models

**Date:** 2026-08-08 · **Corpus:** Hotels 5k benchmark (`sample_hotels_5k.jsonl`)
· **Unit:** sentence segments (mentions) · **Commit:** `d411728`

This document records which models the pipeline uses, what evidence chose them,
and — the part that matters most for a defensible claim — **what the evidence
does not support**. It supersedes the embedding decision in
`methodology.md` §3, which was made at document level in notebook 04.

Companion artifacts, all regenerable (commands in §9):

| Artifact | What it holds |
|---|---|
| `data/runs/admissibility_sent.md` | hard-constraint verification + cost per candidate |
| `data/runs/model_sweep_5000_sent.md` | the full 12-contestant ranking |
| `data/runs/model_sweep_5000_sent_finalists.md` | finalists with multi-seed ARI |
| `data/runs/model_sweep_50000_sent_finalists.md` | finalists confirmed at 454,493 segments |
| `data/runs/label_quality_sent.md` | labeler comparison |
| `data/runs/label_quality_sent_nothink.md` | qwen3 re-scored with reasoning disabled |
| `data/runs/label_sheet_sent.md` | human scoring sheet, static export |
| `data/runs/label_quality_sent.json` | every label with model + prompt hash; also the input to the app's scoring view |
| `data/feedback/label_scoring__*.jsonl` | human 1-5 scores, one file per reviewer session |

---

## 0. Decisions at a glance

| Decision | Choice | Chosen over | On what |
|---|---|---|---|
| Embedding (segment unit) | **`all-MiniLM-L6-v2`**, no instruction | arctic-embed-l-v2.0 (2nd), Qwen3-0.6B (3rd), mpnet (incumbent, 4th) | best mean rank across all three tiers at 5k; **tied** with arctic at 50k, decided there by ~10× throughput at ~⅕ the VRAM (§6b) |
| Instruction | **none** | `generic` / `domain` variants | instructions improved geometry and stability but lowered coherence and entropy |
| Unit | **mention (sentence segment)** | whole review | ~8 aspects/review mean-pooled into one vector; ~10% of reviews truncated at 384 tokens |
| Labeler | see §7 | — | format compliance, faithfulness, hallucination + human sheet |

**Status of the evidence:** the ranking is established at 5k (43,012 segments,
12 contestants) and the two finalists were confirmed at 50k (454,493 segments)
on 2026-08-08 — §6b. The 50k run **did not reproduce MiniLM's margin**: the two
finalists tie on mean rank there and the decision rests on cost, which is a
weaker claim than §5 makes and is stated as such below. Multi-seed stability
(§6) was measured at 5k only.

## 1. Scope, and why it is drawn here

**Sentence/mention unit only.** The document-unit variants (`bertopic`,
`custom_hdbscan`, `flat_agglomerative`, `two_stage`) are deliberately absent
from every table below, and this is a claim about the corpus, not a
convenience:

- The median hotel review runs **~8 sentences across several aspects** (room,
  staff, location, breakfast, value). One vector per review mean-pools those
  aspects into a centroid that represents none of them — a review praising the
  staff and condemning the bathroom lands between the "staff" and "bathroom"
  regions, in a place that is about neither.
- Roughly **10% of hotel reviews exceed mpnet's 384-token window** and are
  silently truncated. The reviews that overflow are the long ones, which are
  precisely the most multi-aspect ones — so the document path degrades hardest
  on exactly the documents whose structure matters most.

At mention level the clustered unit is the aspect mention and both failure
modes disappear. This also means **no cross-unit comparison tables**: segment
and document metrics are computed over different populations and are not
commensurable (methodology §5b).

**Hotels only.** The existing benchmark stays, for continuity — every prior
result is calibrated on it. No Automotive contrast run. The known category
co-occurrence leak is carried forward as a stated threat to validity (§8).

**Counting semantics** (segment-unit throughout): cluster `size` counts
mentions; `n_documents` counts distinct parent reviews. Anything
customer-facing — per-cluster mean stars, Tier-3 rating entropy — is computed
on **deduplicated (review, cluster) pairs**, so one verbose reviewer cannot
dominate a cluster's star profile.

## 2. Hard constraints (verified, not assumed)

A candidate failing any of these is inadmissible regardless of leaderboard
position. Each was checked on this box rather than read off a model card:

| Constraint | Why | How verified |
|---|---|---|
| plain `sentence-transformers`, **no `trust_remote_code`** | the Celery worker must stay free of custom code paths | `AutoConfig.from_pretrained(..., trust_remote_code=False)` per candidate, then a real load |
| fp32 weights + activations within the ~6 GB VRAM slice | shared box, 12 GB cards, courtesy-only fairness | `torch.cuda.max_memory_allocated()` around a real encode |
| Pascal (sm_61): no flash-attention | FA2-preferring models (ModernBERT class) must run eager | they load and run; throughput measured, not estimated |
| torch pinned at **2.7.1+cu126** | wheels ≥ 2.8 dropped Pascal cubins; `cuda.is_available()` then lies and kernels crash | left pinned; **do not "fix"** |

**Result: VRAM was never the binding constraint.** Every admissible candidate
peaked between 498 MB and 3,152 MB against a ~6 GB budget. The constraint that
actually separates candidates is **throughput**, which spans 20× (167–3,413
segments/s).

## 3. Admissibility findings

**Rejected on `trust_remote_code`** (verified against `transformers 5.10.1` /
`sentence-transformers 5.5.1`):

- `Alibaba-NLP/gte-large-en-v1.5` — requires `Alibaba-NLP/new-impl`
- `jinaai/jina-embeddings-v3` — requires `jinaai/xlm-roberta-flash-implementation`

**Re-admitted — a previously registered objection that has expired:**

- `nomic-ai/nomic-embed-text-v1.5` — transformers now ships `NomicBertModel`
  natively, so the `trust_remote_code` objection no longer holds. It was
  admitted and evaluated (rank 9). Caveat: nomic expects task prefixes
  (`clustering: `) that our instruction mechanism only approximates, so its
  result is a floor, not a ceiling.

**Blocked, and recorded as such rather than silently skipped:**

- `google/embeddinggemma-300m` — repo is `gated: manual`. Being logged in is
  not sufficient; access must be granted per account. The June sweep failed
  with 401 (unauthenticated); it now fails with **403** (authenticated, not
  authorised). **This candidate remains unevaluated.** Any claim that the
  chosen model beats the best sub-500M multilingual model is therefore not
  supported by this evidence.

**Added for this review**, selected on the MTEB **Clustering** and **STS** task
groups rather than the overall average — the downstream task is density-based
clustering of short single-aspect spans, and holding up on sentence-length
input is not the same competence as document retrieval:

- `mixedbread-ai/mxbai-embed-large-v1` — best admissible Clustering (46.71)
  *and* best STS (85.00)
- `BAAI/bge-large-en-v1.5` — size-matched contrast to mxbai (Clustering 46.08,
  STS 83.11): same architecture and parameter count, different training recipe,
  which isolates recipe from capacity
- `Snowflake/snowflake-arctic-embed-l-v2.0` — admissible, multilingual,
  forward-looking for EuroParl Phase 2

**Re-weighted, not dropped.** The long-context argument that justified `bge-m3`
and `gte-modernbert` at document level (8k windows fixing the 10% truncation)
**buys nothing when the unit is one sentence**. Both were kept in the sweep but
judged on clustering quality per unit of compute. Multilingual capacity remains
a Phase-2 criterion only; it earns no credit on this corpus.

## 4. Method

Every contestant saw an identical corpus, an identical segmentation (regex
splitter, segments < 20 chars dropped, > 600 chars hard-wrapped), identical
DR and identical clustering configuration — UMAP(10d, nn=15, min_dist=0.0,
cosine) → HDBSCAN(mcs = 0.3% of units, floor 15; ms = mcs/3). **The embedding
model is the only manipulated variable.**

Instruction and no-instruction runs are **separate contestants**, never
averaged: an instruction mechanically reshapes the space, and collapsing the
two would hide exactly the effect under test.

Two measurement faults were fixed before any number here was believed:

1. **Runtime was fictitious.** `embed_with_cache` returns 0.0 s on a cache hit,
   so the June report printed `0.000` in its runtime column for every
   previously-run model. Cost now comes from a fixed re-encoded probe on a
   single pinned device, comparable regardless of cache state.
2. **The first model probed absorbed CUDA context creation.** Without a warm-up
   pass MiniLM measured 221 segments/s; its true figure is **3,413** — a 15×
   error that inverted the cost ranking. Every probe now warms up first.

## 5. Embedding results (5k Hotels, 43,012 segments)

Ranked by mean rank across silhouette (excl./incl. noise), C_v and deduplicated
rating entropy. Full table in `data/runs/model_sweep_5000_sent.md`.

| # | model | M | instr | mean rank | clusters | noise | sil | sil+noise | C_v | entropy | seg/s | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **all-MiniLM-L6-v2** | 22 | no_inst | **4.00** | 61 | .347 | **.589** | .117 | .572 | **.918** | **3413** | **498** |
| 2 | arctic-embed-l-v2.0 | 568 | no_inst | 4.50 | 56 | **.331** | .544 | .127 | .577 | .911 | 341 | 2337 |
| 3 | Qwen3-Embedding-0.6B | 600 | no_inst | 5.50 | 61 | .394 | .553 | .036 | .590 | .863 | 235 | 3092 |
| 4 | all-mpnet-base-v2 *(incumbent)* | 110 | no_inst | 5.75 | 50 | .337 | .498 | .091 | .577 | .887 | 887 | 886 |
| 5 | bge-m3 | 570 | no_inst | 6.25 | 55 | .460 | .589 | −.031 | .594 | .828 | 345 | 2337 |
| 6 | Qwen3-Embedding-0.6B | 600 | generic | 6.25 | 57 | .328 | .544 | **.148** | .572 | .854 | 190 | 3152 |
| 7 | gte-modernbert-base | 149 | no_inst | 6.50 | 59 | .430 | .555 | −.008 | .587 | .838 | 836 | 774 |
| 8 | bge-large-en-v1.5 | 335 | no_inst | 6.50 | 49 | .385 | .508 | .051 | **.596** | .838 | 340 | 1562 |
| 9 | nomic-embed-text-v1.5 | 137 | no_inst | 7.00 | 64 | .383 | .543 | .042 | .577 | .880 | 887 | 822 |
| 10 | multilingual-e5-large-instruct | 560 | domain | 7.50 | 51 | .381 | .546 | .058 | .573 | .827 | 166 | 2533 |
| 11 | mxbai-embed-large-v1 | 335 | no_inst | 9.00 | 55 | .349 | .445 | .045 | .569 | .863 | 343 | 1562 |
| 12 | multilingual-e5-large-instruct | 560 | no_inst | 9.25 | **4** | .010 | .030 | −.068 | .439 | .930 | 325 | 2461 |

### Decision: `all-MiniLM-L6-v2`, no instruction

It wins the mean rank, and it wins the cost argument outright — **10× the
throughput and 4.7× less VRAM than the model directly behind it**. At segment
granularity the quality gap that would justify a larger model does not exist.

Note honestly that this **confirms rather than overturns** the June sentence-level
result: MiniLM ranked first there too. What is new is that the ranking now rests
on 12 contestants including the current leaderboard leaders, with verified
constraints and real cost numbers, instead of 6 contestants with a fabricated
runtime column.

### Findings that carry more weight than the winner

**MTEB rank did not transfer.** `mxbai-embed-large-v1` — the best admissible
model on *both* the MTEB Clustering (46.71) and STS (85.00) task groups, chosen
specifically because those are the competences this task needs — finished
**11th of 12**, below the 22M baseline on every Tier-1 metric. `bge-large-en-v1.5`
(8th) and `gte-modernbert-base` (7th) tell the same story. Leaderboard position
on general benchmarks predicted very little about density-based clustering of
hotel aspect mentions. This is the single strongest argument in this document
for running the sweep at all rather than picking off a leaderboard.

**Long context bought nothing, as predicted.** `bge-m3` (8k) discards **46% of
mentions as noise** and posts a *negative* incl.-noise silhouette; `gte-modernbert`
(8k) discards 43%. Both rank mid-table. The re-weighting in §3 was correct: an
8k window is irrelevant when the unit is one sentence.

**The instruction effect reproduced exactly.** Qwen3 `generic` posts the best
incl.-noise silhouette in the entire table (.148) and the lowest noise (.328) —
better *geometry* — while its C_v **falls** (.590 → .572) and its entropy falls
(.863 → .854). The instruction reshapes the space without improving the topics,
which is precisely notebook 04's finding, now confirmed at mention level. Had
the two variants been averaged into one row, this would have been invisible.

**A cautionary result on Tier 3.** `multilingual-e5-large-instruct` without its
instruction collapses to **4 clusters** covering 99% of mentions — a degenerate
clustering by any reading — yet posts the **highest rating entropy in the table
(.930)**. Four buckets containing everything necessarily contain every star
level. Tier 3 is a tripwire, not a score, and this row is the proof; it is also
why the mean rank is a shortlisting convention rather than a verdict.

**Instructions are not optional for every model.** The same row shows the
mirror image of the instruction story: for e5-large-instruct the instruction is
load-bearing (51 clusters with it, 4 without). "Instruction-tuned models inflate
silhouette" is a real effect but not a universal law, and running both variants
is what distinguishes the two cases.

### Limits of this ranking

- **Noise is high for everyone.** The winner discards 34.7% of mentions; nobody
  is below 32.8%. Incl.-noise silhouette is ≤ .148 across the whole table, so
  *no* configuration here looks good once discarded mentions are counted. The
  ranking says which model is best on this corpus, not that the result is good.
- **The referee is still calibrated on the incumbent.** UMAP(10d, nn=15) +
  HDBSCAN(mcs/ms) was chosen in notebooks 05/06 using mpnet embeddings
  (methodology §3). Holding it fixed is what makes the comparison controlled,
  but a model that only shines under different DR parameters would still be
  missed. Fixing this needs a per-model DR sweep, which has not been run.
- **Single seed for the ranking.** The table is seed 42. Stability is measured
  separately for the finalists (§6).

## 6. Stability (multi-seed ARI, finalists)

Adjusted Rand Index of cluster assignments across seeds 42/43/44, noise treated
as its own label. UMAP is deterministic per seed but not across seeds, so this
measures whether a model has found structure that survives re-projection —
WP9b's "same corpus → same clusters" goal depends on it.

| model | instr | mean rank | ARI mean | ARI min |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | no_inst | **2.25** | 0.682 | 0.661 |
| arctic-embed-l-v2.0 | no_inst | 2.75 | 0.622 | 0.572 |
| Qwen3-Embedding-0.6B | no_inst | 3.00 | 0.615 | 0.566 |
| all-mpnet-base-v2 | no_inst | 3.50 | 0.661 | 0.626 |
| Qwen3-Embedding-0.6B | generic | 3.50 | **0.744** | **0.729** |

**The instruction buys stability, and that is the strongest thing it buys.**
Qwen3 `generic` is the most seed-stable configuration measured (0.744) while
Qwen3 `no_inst` is the *least* (0.615) — the same model, differing only by an
instruction prefix, moves 0.13 ARI. Combined with §5, the full picture of what
the instruction does to Qwen3 is:

| | no_inst | generic | effect |
|---|---|---|---|
| silhouette incl. noise | .036 | .148 | geometry **much better** |
| noise fraction | .394 | .328 | keeps more mentions |
| ARI (stability) | .615 | .744 | **much more stable** |
| C_v (coherence) | .590 | .572 | topics **worse** |
| rating entropy | .863 | .854 | slightly worse |

This refines rather than contradicts notebook 04. The instruction *does*
reshape the geometry without improving the topics — the C_v and entropy both
fall, exactly as documented. What notebook 04 could not see, because it never
measured stability, is that the reshaping also makes the projection markedly
more reproducible. So "instruction = cosmetic silhouette inflation" is too
strong a summary: it is a real trade of topic quality for geometric regularity
and reproducibility.

**MiniLM wins on stability among the no-instruction candidates** (0.682), which
matters because it is not the usual pattern — the smallest model is normally
the most seed-sensitive. It is not the most stable configuration overall.

### Decision on the stability trade-off

**The default stays `all-MiniLM-L6-v2`, and the trade-off is documented rather
than silently resolved.**

The case for switching to Qwen3 `generic` is real: +0.062 ARI mean (0.744 vs
0.682) and +0.068 ARI min, which speaks directly to WP9b's "same corpus → same
clusters" goal. The case against, which decided it:

- It costs **~18× the compute** (190 vs 3,413 segments/s) and 6.3× the VRAM.
- It is **worse on the topics themselves** — C_v 0.572 vs 0.572 (equal) and
  entropy 0.854 vs 0.918, the latter a large gap on the metric that detects
  sentiment blobs.
- Its stability advantage comes from an instruction that *lowers* its own
  coherence (§6 table). We would be buying reproducibility with topic quality.
- 0.68 and 0.74 are both well short of "reproducible". Neither delivers WP9b's
  goal; pinning the seed is what delivers it, and the seed is pinned. The
  choice between them is therefore not the difference between reproducible and
  not — it is a small movement inside the "unstable across seeds" regime.

**Revisit this if** cross-seed identity becomes a hard requirement rather than a
goal, in which case Qwen3 `generic` is the documented alternative and the
inspection sheets should be re-read for it before switching.

**Caveat on the cost columns in this report.** `model_sweep_5000_sent_finalists.md`
was generated while the labeler sweep and Ollama were using the same box, and
its throughput column is depressed accordingly (arctic 341 → 151 seg/s, Qwen3
234 → 118). Peak VRAM is unaffected. **The authoritative cost numbers are the
ones in §5**, from the sweep that ran alone. The report now carries this
warning in its own provenance block.

## 6b. Confirmation at scale (50,000 reviews)

`data/runs/model_sweep_50000_sent_finalists.md` · run 2026-08-08, 13:31–14:19,
the box otherwise idle. 50,000 reviews → **454,493 segments** (9.09 per review),
10.6× the segments the ranking was decided on. Identical pipeline, identical
seed; the only change is corpus size, and `min_cluster_size` is 0.3% of units so
the clustering scales with it rather than being re-tuned.

| | MiniLM 5k | MiniLM 50k | arctic 5k | arctic 50k |
|---|---|---|---|---|
| clusters | 61 | 68 | 56 | 59 |
| noise | .347 | .374 | .331 | .393 |
| silhouette (excl. noise) | **.589** | .560 | .544 | **.578** |
| silhouette (incl. noise) | .117 | .066 | .127 | .075 |
| C_v coherence | .572 | **.643** | .577 | .634 |
| rating entropy | .918 | **.935** | .911 | .923 |
| mean rank † | 2.25 | 1.50 | 2.75 | 1.50 |
| texts/s ‡ | 3,413 | 3,593 | 341 | 347 |
| peak VRAM | 498 MB | 498 MB | 2,337 MB | 2,337 MB |

† Mean rank is **only comparable down a column pair, never across the 5k/50k
divide**: it is a rank within the contestant set of its own run (5 finalists at
5k, 2 at 50k), so 2.25 → 1.50 is a change of denominator, not an improvement.
The raw metrics above it are the comparable rows.
‡ 5k throughput is quoted from `model_sweep_5000_sent.md`, which ran alone; the
5k finalists report was contended (§6, end).

**The headline: the two finalists tie.** Mean rank 1.50 each. MiniLM takes
coherence and rating entropy; arctic takes both silhouettes. Two metrics each,
so the "1" and "2" printed in the report are an artefact of tie-breaking and
carry no information. **At 50k there is no quality argument for MiniLM over
arctic** — there is a cost argument, and it is decisive: 10.4× the throughput
(3,593 vs 347 seg/s) at 21% of the VRAM (498 vs 2,337 MB), for quality that
measurement cannot separate. That is a sound basis for the decision and a
different one from §5's, which claimed a quality win. Both are recorded.

**What changed with scale, and what it means for the 5k evidence:**

- **The geometry ordering flipped.** At 5k MiniLM won silhouette-excluding-noise
  (.589 vs .544); at 50k arctic wins it (.578 vs .560). Arctic also went from
  the *lower*-noise model (.331 vs .347) to the *higher*-noise one (.393 vs
  .374). A head-to-head margin at 5k therefore does not predict its own sign at
  50k. **Treat the 5k table as a shortlisting instrument, not as a measurement
  of how much better one model is** — that is what §4 claimed for it, and this
  run is the evidence that the caution was warranted.
- **Coherence improved for both** (MiniLM .572 → .643, arctic .577 → .634) and
  rating entropy rose. More data per theme yields more lexically coherent
  clusters, which is the expected direction and mild evidence that the pipeline
  is not overfitting small-sample structure.
- **Silhouette-including-noise fell for both** (.117 → .066, .127 → .075) as
  noise rose. The §5 conclusion that *no* configuration scores well once
  discarded mentions are counted holds at scale, and gets worse. 37–39% of
  mentions are still thrown away.
- **Cluster count barely moved** — 61 → 68 for MiniLM against 10.6× the data.
  Because `min_cluster_size` scales with corpus size, extra data makes existing
  clusters bigger rather than finding new themes. Whether that is the pipeline
  finding a stable topic inventory or the parameterisation *imposing* one is not
  answerable from this run; it needs a sweep over `min_cluster_size` at fixed
  corpus size, which is not done.
- **Throughput was reproducible** (3,413 → 3,593 and 341 → 347 seg/s) because
  this run had the box to itself. Compare with §6's contended figures — this is
  what the "quote cost from a sweep that ran alone" rule is for.

**What this run does not establish.** No multi-seed ARI was computed at 50k —
`--stability-seeds` was not passed, and re-running the seeded UMAP fit twice
more at 454k segments was not affordable in the window. **Every stability claim
in §6 remains a 5k claim**, and the possibility that the two models' stability
ordering also flips with scale is untested. Nor were any of the other ten
contestants re-run at 50k: this confirms the finalists against each other, not
the shortlisting that produced them.

## 7. LLM labeler

This closes the oldest open item in the project: notebook 08 chose the centroid
context strategy and the v1 prompt on informal reading, its human scoring sheet
was never filled in, and Ollama was down for long enough that **every label in
every artifact is still `terms_fallback`**. No language model had ever been
scored on this corpus.

### Setup

Ollama 0.30.7, rootless, pinned to an idle GPU with `OLLAMA_CONTEXT_LENGTH=8192`
(the default context is what previously caused a 2 GB model to spread ~26 GB
across all four devices). Note there is also a **root-owned system `ollama`** on
this box that is not listening on 11434; it is not ours and is left alone.

Clustering is held fixed (MiniLM, seed 42, the 20 largest mention clusters);
only the labeler and the prompt vary. Five candidates:
`llama3.2:1b`, `llama3.2:3b`, `qwen3:4b`, `gemma3:4b`, `phi4-mini`.

### A mention-level prompt (v2)

Notebook 08's v1 prompt says "reviews" and asks what they "have in common" —
sensible for document clusters, wrong for mention clusters. Mention clusters are
tighter and already single-topic, so asking what members "have in common"
invites the model to hedge toward a generic theme. **v2_mention** names the unit
("sentences ... from customer reviews"), asks for the aspect rather than the
commonality, and explicitly forbids the sentiment-only labels that short
evaluative mentions tempt a model into ("Breakfast Buffet Quality", never
"Guests Were Happy"). Both prompts are scored; hashes `a300b8f5` (v1) and
`68ece919` (v2_mention) are recorded with every label.

### Scoring

Three axes, per the tech-selection requirement:

- **Format compliance** — deterministic. 3–6 words, no preamble, no quotes, no
  trailing punctuation, single line. The one axis with a ground truth.
- **Faithfulness** — `discrimination@1`: embed the label and check whether its
  own cluster's centroid is the nearest of all centroids. An automatic analogue
  of the methodology's intruder test.
- **Hallucination** — content words appearing in *no* member mention.

Read `hallucination_rate` with care: it is lexical, so a correct paraphrase
("Breakfast" for mentions saying "morning buffet") counts against it. It is a
tripwire for invention, not a quality score, which is why the human sheet
(`label_sheet_sent.md`) exists alongside it.

### Results

`data/runs/label_quality_sent.md` — with `data/runs/label_quality_sent_nothink.md`
for the qwen3 re-run described below, and every individual label plus its model
and prompt hash in `label_quality_sent.json`.

Ten (labeler × prompt) configurations, 20 clusters each. Top of the table:
`qwen3:4b`+v1 (composite 0.900), `qwen3:4b`+v2_mention (0.883), `gemma3:4b`
+v2_mention (0.850). Bottom: `llama3.2:1b` (0.533 / 0.467).

**The reasoning re-run, and what it actually showed.** `qwen3:4b` reasons by
default, and for a 3–6 word label that reasoning looked like pure latency with a
risk of reasoning tokens leaking into the answer. Scoring a model only in its
thinking configuration would have reported a configuration artefact as a model
verdict, so it was re-run with `--no-think` (`data/runs/label_quality_sent_nothink.md`).

**The suspicion was wrong, and the re-run is what establishes that.** Disabling
reasoning made qwen3 *worse*, not faster:

| | v1 | v1+nothink | v2_mention | v2_mention+nothink |
|---|---|---|---|---|
| composite | **0.900** | 0.850 | 0.883 | 0.883 |
| discrim@1 | **0.950** | 0.850 | 0.850 | 0.850 |
| mean margin | 0.168 | 0.135 | 0.092 | **0.115** |
| s/label | 44.7 | 42.0 | 52.8 | 42.2 |

v1 loses 0.10 discrimination@1 with thinking off; v2_mention holds its composite
and gains margin. And the latency saving is **6%** on v1 (44.7 → 42.0 s/label),
not the order of magnitude the reasoning hypothesis predicted — qwen3's cost is
the model, not the thinking. So: **quote the default rows.** Thinking was not
inflating qwen3's score, and the `+nothink` rows exist to demonstrate that
rather than to replace anything. The deployability objection to qwen3 stands
untouched and is unrelated to reasoning: at 44.7 s/label against gemma3's 1.9 it
is **23× the cost of a model 0.05 composite behind it**.

**The small-model result, stated in full.** `llama3.2:1b` behaves exactly as the
"a smaller model may well suffice" hypothesis fails: the v2_mention prompt
*improved* its faithfulness (discrimination@1 0.40 → 0.60) and reduced
hallucination (0.40 → 0.30) while *collapsing* format compliance (0.60 → 0.10),
because a 1B model answers a more specific prompt with a list of candidate
labels rather than one label. It understands the task better and follows the
output contract worse.

This is not a flaw in v2_mention, but neither is it cured by size alone: only
`qwen3:4b` and `gemma3:4b` hold format compliance at 1.000 under **both**
prompts. `llama3.2:3b` manages 0.800/0.700 and `phi4-mini` collapses from 0.950
to 0.450 on v2_mention — the same failure as the 1B model, at 4B. Following a
stricter output contract tracks the individual model, not the parameter count,
which is the argument for keeping format compliance as the first gate in the
reading order below rather than assuming a mid-size model will pass it.

### How to decide from the table

The composite column weights format compliance, discrimination@1 and
(1 − hallucination) equally — a stated convention, not a truth, with the raw
columns present so it can be re-weighted. Recommended reading order:

1. **Format compliance first.** A label that needs salvaging by a regex is
   unusable in the UI regardless of aptness; below ~0.9 the model is a liability.
2. **Then discrimination@1**, which is the closest automatic proxy for "this
   label identifies this cluster".
3. **Then the human scores.** Neither metric sees usefulness, and label approval
   is a mandatory HITL step (methodology §9) precisely because of this. Scoring
   happens in the app (`Label scoring`), blind and multi-reviewer;
   `label_sheet_sent.md` remains as a static export for anyone who wants to read
   the task offline. **This is still outstanding — every labeler claim above is
   an automatic-metric claim.**
4. **Latency last**, but not never: labeling runs once per cluster per run, so
   seconds-per-label matters far less than quality — except where it signals a
   misconfiguration, as with qwen3's reasoning.

## 8. Threats to validity

- **Category co-occurrence leak (carried forward, not fixed).** "Hotels"-tagged
  Yelp businesses are frequently co-tagged restaurants/nightlife/venues, so the
  benchmark contains substantial non-hotel content — the 1k smoke run's largest
  cluster was about oysters on Bourbon Street. This was carried forward
  deliberately for continuity with prior results rather than re-filtered
  mid-stream. **Assessment for this decision: it does not appear to change the
  model ranking.** The leak is a property of the corpus, and every candidate saw
  exactly the same corpus, so it shifts all contestants together rather than
  favouring one. It would matter for an absolute claim about hotel review
  topics; it does not undermine a relative claim between embedding models.
  **The 50k numbers now exist (§6b), so this is due.** It is visible in the
  labeled output — cluster 3, one of the 20 largest mention clusters scored in
  §7, is a Reno casino cluster that **9 of the 10 labeler configurations named
  correctly** ("Best Casino in Reno", "Reno Casino Experience", …). The labels
  are right; the cluster simply is not about hotels. Any slide showing
  cluster inventories, and any claim of the form "the pipeline finds N themes in
  hotel reviews", is wrong until the corpus is filtered — while the model
  *ranking* stands unaffected. See `docs/quality-roadmap.md` for the fix.
- **"First N matching" is not a random sample** — dataset order correlates with
  business ID and therefore geography (methodology §1). Fine for comparing
  candidates against each other, not for claims about Yelp hotels in general.
- **English-only evidence.** `arctic-embed-l-v2.0`, `bge-m3` and
  `multilingual-e5` were evaluated only on English hotel mentions. Their
  multilingual capacity — the reason two of them are registered — is
  **completely untested here**. The EuroParl Phase-2 question remains open, and
  nothing in this document should be read as answering it.
- **`embeddinggemma-300m` was never evaluated** (§3). The claim "MiniLM beats
  the admissible field" excludes it.
- **Regex sentence splitting** mis-handles abbreviations ("St. Louis"). Held
  constant across candidates, so it does not bias the comparison, but it does
  put a floor under everyone's noise fraction.

## What this evidence does *not* support

Stated plainly, because a selection document that only lists conclusions is not
auditable:

1. That MiniLM-L6-v2 is the best embedding model for this pipeline **in
   general** — the evidence covers English hotel reviews, mention units, one
   fixed DR/clustering referee, and one seed at 50k.
2. That MiniLM-L6-v2 produces **better clusters than arctic-embed-l-v2.0**. At
   50k the two are indistinguishable on the measured axes (§6b); the choice
   between them is a cost decision, and a deployment with spare VRAM and no
   throughput constraint has no evidence-based reason to prefer either.
3. That it would win on a **different corpus**. EuroParl, support tickets and
   Automotive are all untested; WP9 must re-run this sweep per corpus.
4. That the resulting clusters are **good** — only that they are better than the
   alternatives measured. 37% of mentions are discarded as noise at 50k, and no
   configuration scores well once that noise is counted.
5. That the **MTEB leaderboard is wrong** — only that its ranking did not
   transfer to this task, corpus and unit. That is a statement about transfer,
   not about the benchmark.
6. That any model beats **`embeddinggemma-300m`**, which remains unevaluated
   behind a gated repo.
7. That the multilingual candidates are or are not suitable for Phase 2 — that
   question was not asked here.
8. That any labeler is **useful**, as opposed to well-formed and faithful. No
   human has scored a label yet (§7).

---

## 9. Reproducing this

```bash
# 1. admissibility + cost (verifies the hard constraints on this box)
bash scripts/run_admissibility.sh

# 2. the ranking (12 contestants, ~20 min on 2 idle GPUs)
bash scripts/run_sentence_sweep.sh

# 3. finalists with multi-seed stability
python -m reviewscope_ml.eval.model_sweep --sample-size 5000 --device cuda \
    --gpus 2 --sentence-level --stability-seeds 3 --tag finalists \
    --models all-MiniLM arctic-embed Qwen3 all-mpnet

# 4. labeler comparison (needs Ollama up, pinned to an idle GPU)
CUDA_VISIBLE_DEVICES=<idle-gpu-uuid> OLLAMA_CONTEXT_LENGTH=8192 \
    nohup ~/ollama/bin/ollama serve > ~/ollama/serve.log 2>&1 &
python -m reviewscope_ml.eval.label_sweep --sample-size 5000 --device cuda \
    --models llama3.2:1b llama3.2:latest qwen3:4b gemma3:4b phi4-mini \
    --variants v1 v2_mention \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 --n-clusters 20

# 5. confirm the finalists at scale (454,493 segments; ~50 min on an idle box)
python -m reviewscope_ml.eval.model_sweep --sample-size 50000 --device cuda \
    --gpus 2 --sentence-level --tag finalists --models all-MiniLM arctic-embed

# 6. qualitative inspection, paired with the ranking
streamlit run src/reviewscope_ml/hitl/app.py   # sidebar → "Model selection"

# 7. human label scoring — blind, multi-reviewer, writes to data/feedback/
streamlit run src/reviewscope_ml/hitl/app.py   # sidebar → "Label scoring"
```

Steps 1–5 all ran on 2026-08-08; `scripts/run_unattended.sh` chains 4–5 with
cleanup that releases the shared GPUs afterwards. Step 7 is the open one — see
§7 and `docs/quality-roadmap.md`.
