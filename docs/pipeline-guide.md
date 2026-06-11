# ReviewScope ML pipeline — the full story

A narrative walkthrough for the team: what runs, in what order, why each
decision was made, where the bodies are buried, and what we deliberately did
NOT do. The rigorous per-stage threat analysis lives in `methodology.md`;
this document is the one you read first (and the one you explain from).

---

## 0. The one rule that overrides everything

**Clusters must make sense to a human.** A pipeline that scores brilliantly
on silhouette but produces clusters you can't name in one phrase has failed.
Every automated number in this project is a *proxy* used to rank candidates
cheaply — the winner is always picked by a person reading actual documents.
This is not a disclaimer, it's the architecture: the comparison report
physically ends in a human sign-off block, not a computed winner.

Why so strict? Because every classic failure mode of review clustering
*scores well*:

| Failure mode | What it looks like | Why metrics love it | Our tripwire |
|---|---|---|---|
| Sentiment blobs | "angry reviews" vs "happy reviews" | polarity is geometrically crisp → high silhouette | Tier 3 rating entropy |
| Blob + crumbs | one 60% mega-cluster + tiny shards | the shards are tight → decent averages | max_cluster_share flag |
| Noise dumping | 40% of docs discarded as noise | silhouette computed on the easy remainder | incl.-noise silhouette + noise ratio |
| Near-duplicate clusters | two "room cleanliness" clusters | both individually coherent | top-term overlap flag |
| Artifact clusters | grouped by length/boilerplate, not content | lexically self-similar → high C_v | only human inspection |

---

## 1. The data (and its two honest problems)

Fixed benchmark: Yelp reviews of businesses tagged **Hotels**, ≥50 chars,
first N in dataset order (notebook 02's decision, reproduced by
`data/ingest.py`). Sizes: 1k (smoke), 5k (decisions), 50k (production
scale). Median review: 101 words, **8 sentences**, covering several aspects.

Problem 1 — **category leak**: "Hotels"-tagged businesses are co-tagged
restaurants/bars; the inspection sheets show oyster-bar clusters. No metric
caught this; reading 5 random samples did. Decision pending: filter to
hotel-only businesses or accept and document.

Problem 2 — **hotels are small**: only ~185k hotel reviews exist in the dump,
so the 50k benchmark is ~27% of everything, geographically skewed by
"first N" order. For cross-domain validation, the same tooling runs any
category (`--category Automotive` → 228k reviews, disjoint vocabulary).

## 2. The pipeline, stage by stage

```
ingest → preprocess → [segment] → embed → reduce → cluster → represent → label
                                                       ↓
            feedback ← HITL review ← report ← evaluate ┘
```

| Stage | What it does | Decided how | Module |
|---|---|---|---|
| Ingest | benchmark sample from the Yelp dump | nb 02 | `data/ingest.py` |
| Preprocess | whitespace normalisation only ("minimal") | nb 02 sweep: aggressive cleaning hurts transformer embeddings | `core/config.py` |
| Segment *(sentence variant only)* | reviews → sentence mentions | regex splitter, <20 chars dropped | `data/segment.py` |
| Embed | text → vector; cached per (model, instruction, corpus, unit, size) | nb 04 + model sweep | `embed/` |
| Reduce | UMAP 768d→10d for clustering; separate 2D/3D for plots | nb 05 sweep | `reduce/` |
| Cluster | five interchangeable backends (below) | nb 06 + comparison harness | `cluster/` |
| Represent | c-TF-IDF top terms, word-cloud frequencies | BERTopic-style | `represent/` |
| Label | Ollama LLM label+summary per cluster; term fallback when LLM down | nb 08 prompts, centroid context | `label/` |
| Evaluate | three tiers + fairness + stability + failure flags | this project | `eval/` |
| HITL | human approves/renames/merges/splits/junks; recorded as JSONL | this project | `hitl/` |

Everything heavy is cached on disk and keyed on *all* of its inputs
(model, instruction, corpus, unit, seed, parameters) — a crashed run resumes,
a repeated run is free, and two corpora can never silently share artifacts.

## 3. The five competing pipelines

| Variant | Idea | Why it's in the race |
|---|---|---|
| `bertopic` | the off-the-shelf standard (MiniLM + UMAP-5d + HDBSCAN + c-TF-IDF) | the "what you get without thinking" baseline; only deviation: a seed, or stability would be unmeasurable |
| `custom_hdbscan` | our tuned embed→UMAP→HDBSCAN | the notebook-decided default; honest noise handling |
| `flat_agglomerative` | same front-end, ward tree cut at k=15 | no-noise contrast; most stable across seeds so far (ARI 0.70) |
| `two_stage` | many small HDBSCAN micro-clusters → ward-merged macro topics | macro topics stay human-sized; micro hierarchy answers HITL splits and (later) incremental updates |
| `sentence_level` | cluster sentence *mentions*, not reviews | reviews are multi-aspect — one vector per review averages the aspects away; ~10% of reviews are also truncated at the token window |

All five emit the **same artifact schema** (assignments + 2D/3D coords +
per-cluster terms/label/summary + manifest), so the app backend consumes any
of them interchangeably and the HITL app reviews them identically.

Sentence-level counting (the part people get wrong): cluster `size` counts
**mentions**, `n_documents` counts **distinct reviews**; star statistics and
Tier-3 entropy are deduplicated per (review, cluster) so a 10-sentence rant
counts once; `doc_membership.json` maps each review to cluster shares + one
*primary* cluster for the app's one-cluster-per-document field. Caveat:
segment-unit metrics are not directly comparable to document-unit metrics in
the same table — the dedup entropy and human inspection are the fair bridge.

## 4. How a winner gets chosen (two-step protocol)

1. **Automated shortlist.** Per variant: Tier 1 (silhouette/DB/CH in reduced
   space — geometry, biased), Tier 2 (C_v coherence from raw text —
   independent of the embedding), Tier 3 (rating entropy — sentiment-blob
   tripwire), noise-fair silhouette variants, multi-seed ARI stability, and
   structural failure flags. Mean rank shortlists 3 finalists.
2. **Human verdict.** Per finalist, the report renders an inspection sheet —
   label, top terms, **5 random member documents** (random, never
   centroid-nearest: centroid samples flatter the cluster) — and an intruder
   test (4 members + 1 outsider, shuffled; if you can't spot the intruder,
   the boundary isn't real). The reviewer then confirms in the HITL app,
   which records reviewer + timestamp. Only then does the sentence
   *"a human reviewed the clusters of the winning pipeline and confirmed
   they are thematically coherent"* become claimable.

## 5. Operational order (the runbook)

1. `model_sweep` at 5k (document-level), and `--sentence-level` if the
   sentence pipeline is a serious candidate — short texts are a different
   embedding regime, the winner may differ.
2. Human reads both sweep reports, picks the embedding model(s) on the
   three-tier picture (silhouette-only winners are geometry reshaping).
3. 50k comparison of the four document-level variants with that model;
   sentence_level first at 5k, scaled up only if inspection justifies it.
4. Optional: same comparison on a contrast corpus (Automotive) to test
   whether the ranking transfers — the parameters won't, the *ordering*
   should.
5. Human pass: inspection, intruder tests, HITL corrections, sign-off;
   feedback applies to the next run via documented semantics
   (`hitl/apply_feedback.py`).

GPU etiquette is enforced in code, not in memos: idle devices only, VRAM
capped at 50% per device, claim/release logged, per-model batch/sequence
limits, OOM auto-backoff, CPU capped at 4 threads. torch is pinned to
2.7.1+cu126 (newer wheels dropped Pascal kernels and crash at the first op).

## 6. Hyperparameters — what we actually did, honestly

**What happened:** notebooks 05/06 ran *sequential* one-dimensional sweeps —
tune UMAP dims with clustering fixed, then clustering with UMAP fixed. That
is coordinate descent with one iteration: cheap, defensible, but it never
saw parameter *interactions*, and it optimised mostly on the 5k hotel corpus
with mpnet embeddings.

**The scale problem (fixed, heuristically):** HDBSCAN's `min_cluster_size`
is an absolute count whose meaning is relative. mcs=15 was decided at 5k
(0.3% of the corpus); reused at 50k it calls 15 documents a topic and
produces blob-and-crumbs. Default specs now use `"auto"`:
`mcs = max(15, 0.3% × units)`, `ms = mcs/3` (micro pass: 0.1%, floor 5) —
**anchored to the notebook decision**, so behaviour at 5k is unchanged and
extrapolated linearly elsewhere. Know what this is: an anchor + a linearity
assumption, not a validated optimum. k for KMeans/agglomerative deliberately
does *not* scale — more reviews mean bigger topics, not more topics.

**What proper tuning would look like (proposed, not built):** Optuna
multi-objective TPE over the joint space (UMAP dims/neighbours × HDBSCAN
sizes), objectives = incl.-noise silhouette + C_v + rating entropy, noise
fraction as a constraint, averaged over 2–3 seeds (or you tune UMAP's
randomness). Output is a **Pareto front, not a winner** — which slots
exactly into our two-step protocol: front automated, human picks from the
front by inspection. Cost ≈ 50 trials × 1–2 min at 5k with cached
embeddings. Repeat per corpus. The danger to respect: any scalarised
objective is Goodhart bait — our three failure-mode tables above are the
list of things a naive optimiser will happily produce.

**Deliberately not tuned:** embedding model (that's the sweep's job, a
discrete choice), preprocessing variant (decided, low sensitivity), metric
thresholds (they direct the human eye, they don't decide).

## 7. Ordering critique — where our process is circular

**Decision circularity.** The embedding comparison (nb 04) judged models
under a fixed UMAP+HDBSCAN config — which had been tuned *using mpnet*
(nb 05/06). The referee was calibrated on one contestant. Mitigation
applied: parameters are mid-range and the ranking was stable across the
grid. Remaining risk: a model that only shines under very different DR
parameters was structurally invisible. Cheap fix on the roadmap: after the
sweep picks a winner, re-sweep DR×clustering *for that winner* instead of
inheriting mpnet's parameters.

**Stage-order alternatives worth one experiment each:**

- **Skip UMAP for partitioners** (KMeans/ward directly on normalised
  embeddings or PCA-50): costs some geometric quality, but removes UMAP —
  the single biggest source of cross-seed instability (our ARI: 0.45–0.70).
  If raw-space clusters read "good enough" in inspection, that trade may be
  worth more to an app that promises reproducibility (WP9b) than a few
  silhouette points.
- **densMAP instead of UMAP** (`densmap=True`, built into umap-learn):
  directly addresses the documented "UMAP distorts density, then HDBSCAN
  clusters the distortion" critique. One-line experiment.

**A known cosmetic mismatch:** scatter plots are projected from raw
embeddings, clustering happens in the 10d space — clusters can look
intermingled in 2D while being clean in 10d. It can erode reviewer trust;
we document it rather than fix it (a 2D projection *of* the 10d space would
distort differently, not less).

## 8. Algorithm landscape — evaluated, with verdicts

Considered and **worth evaluating** (in order):

1. **GMM on the reduced space** — soft assignments; a document can be 60%
   "location" and 40% "value", which matches both multi-aspect reality and
   our `doc_membership` format. Trivial to add (sklearn).
2. **Leiden community detection on a kNN graph** — no k, robust to variable
   density, the de-facto standard in single-cell genomics for exactly this
   shape of problem; a `resolution` knob instead of mcs. The most serious
   HDBSCAN challenger.
3. **MiniBatchKMeans / BIRCH** — not as quality contenders but as the only
   straightforward path to WP9b goal 2 (incremental updates without full
   re-runs). Evaluate for identity stability, not for silhouette.
4. **NMF on TF-IDF** — fully deterministic, dirt cheap, and an honest
   "pre-embedding era" baseline that keeps us humble in the report.

Considered and **skipped, with reasons:** DBSCAN (subsumed by HDBSCAN),
OPTICS (slower, same family), spectral clustering (O(n²)+ memory at 50k,
unclear gain over ward), Top2Vec/CTM (overlaps BERTopic's slot), deep
clustering à la DEC (engineering cost far beyond course scope). KeyBERT is
parked as a labeling middle ground if Ollama labels disappoint.

## 9. Weakness index (the things to say *before* someone asks)

1. Every parameter was tuned on English hotel reviews; the app promises any
   CSV. Architecture transfers, values don't — rerun the harness per corpus.
2. The benchmark leaks restaurant content (category co-tagging) and covers
   27% of a small category, non-randomly.
3. Tier-1 metrics live in UMAP space and inherit its density distortion;
   Tier 2 (C_v) is noisy on short texts; Tier 3 needs a rating column and
   cannot tell a sentiment blob from a legitimately polarised topic
   ("bed bugs" is ~all 1-star AND a real topic). That's three flawed judges
   — which is exactly why they vote together and a human presides.
4. Determinism is achieved per seed (pinned), not across seeds (ARI
   0.45–0.70) — WP9b's "same corpus → same clusters" currently holds only
   with frozen seeds; cross-seed stability is open research.
5. LLM labels see 5 sampled documents, not the cluster; hallucination is
   expected, which is why label approval is a mandatory HITL step and
   `label_source` always distinguishes machine text from human-approved text.
6. The mcs auto-scaling is an anchored heuristic (see §6); sentence-level
   parameters inherit document-level UMAP settings unswept.
7. Sentence segmentation is regex-based (abbreviations mis-split) and
   produces a generic-praise cluster by construction — `mark_junk` exists
   for it.

If you can explain this table and §0 to someone, you understand the project.
