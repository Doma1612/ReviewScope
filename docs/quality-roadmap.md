# Quality roadmap — what separates this from *really good* results

A self-critique of the current pipeline, written after the machinery was
built. The uncomfortable summary: we have a very good **comparison
machine**, but no proof yet that any result is *good* — every decision so
far rests on proxy metrics whose correlation with human judgment we assume
rather than know. This document lists the gaps in priority order, with
effort estimates, so the team can decide deliberately what to invest in.

Companions: `pipeline-guide.md` (how it works), `methodology.md` (per-stage
threats), `results-overview.md` (current state).

---

## A. Validity — the foundation gap (most important)

### A1. No ground truth anywhere
All selection decisions hang on silhouette/C_v/entropy — proxies whose
agreement with human judgment on *this* data was never tested.

**Fix:** a hand-labeled evaluation set: 300–500 sentences annotated with
aspect labels (room, staff, breakfast, location, value, cleanliness, …),
2 annotators + agreement. Yields external metrics (NMI/ARI vs. humans) and —
more importantly — tells us *which of our three tiers actually correlates
with human judgment*, i.e. validates the referee itself. SemEval ABSA
datasets (hotels/restaurants) are a usable head start.
**Effort:** 1–2 person-days annotation. **Highest value item on this list.**

### A2. The human evaluation is anecdotal
Intruder tests without multiple raters and agreement statistics (Cohen's κ)
are anecdotes; notebook 08's label-quality sheet is still empty.
**Fix:** structured mini-eval — 3 raters × 20 clusters × (intruder test +
label score 1–5), report κ. **Effort:** half a day of team time.

### A3. Tuning and confirmation share data
The 1k/5k benchmarks are *prefixes* of the 50k sample — the winner is
confirmed on data that influenced its selection.
**Fix:** a disjoint hold-out slice (e.g. reviews 50,001–60,000) used only
for the final confirmation. **Effort:** trivial.

### A4. No uncertainty estimates
Is MiniLM's silhouette 0.589 vs. Qwen3's 0.544 a real difference? Unknown
without error bars.
**Fix:** bootstrap over documents for the headline metrics; report
intervals in the comparison table. **Effort:** small.

## B. Data quality — unglamorous, high leverage

### B5. No language detection
Yelp contains non-English reviews; they will form "language clusters" — the
textbook artifact cluster, currently unfiltered.
**Fix:** langdetect/fasttext-lid pass in preprocess, non-English share
reported per corpus. **Effort:** ~10 lines + a dependency.

### B6. No near-duplicate detection
Copy-paste reviews and multi-posts artificially densify regions and create
phantom topics. PK-dedup (app spec) only catches exact re-uploads.
**Fix:** MinHash/SimHash near-dup pass at ingest. **Effort:** small.

### B7. Category leak (measured 2026-08-09, decision due)
Yelp businesses carry several categories, so "tagged `Hotels`" is not "is a
hotel". Measured against `business.json` (2,977 Hotels-tagged businesses):

| co-tag | businesses | share |
|---|---|---|
| Restaurants | 287 | 9.6% |
| Nightlife | 144 | 4.8% |
| Bars | 118 | 4.0% |
| Arts & Entertainment | 73 | 2.5% |
| Casinos | 38 | **1.3%** |

Business counts understate it badly, because the co-tagged venues are the big
ones. By **review** volume: **40.2% of the 5k benchmark and 34.2% of the 50k
benchmark** come from a business co-tagged with one of the above, and casinos
alone — 1.3% of businesses — contribute **14.3% / 10.1% of reviews**. It shows
in the output: cluster 3 of the labeled 5k run is a Reno casino cluster that 9
of 10 labeler configurations named correctly.

**Do not simply filter these businesses out.** A review of a casino hotel is
usually still a hotel review — it talks about the room *and* the casino floor.
Business-level exclusion would discard a third of the corpus along with a lot of
legitimate lodging content, and it would silently change the corpus the whole
selection was run on. The leak is at **mention** level; the filter available is
at **business** level, and they do not line up.

**Recommended, in order:**
1. **Fix the claim, not the corpus** (free). The benchmark is "reviews of Yelp
   businesses tagged Hotels", not "hotel reviews" — say so on every slide.
   Nothing needs re-running and nothing measured becomes wrong.
2. **Triage off-topic clusters in HITL** (already built). A casino cluster is a
   correctly-found theme that is out of scope; `mark_junk` in the review app
   records exactly that and `apply_feedback` drops it on the next run. This is
   the mention-level fix, done by the people who are reviewing anyway.
3. **Sensitivity check, then stop** (~20 min at 5k). Re-run the ranking on a
   corpus with the leak categories excluded and confirm the model ordering is
   unchanged. That converts technology-selection §8's *argument* that the leak
   cannot bias a comparison every candidate faces identically into a
   *measurement*. If the ordering does move, that is a finding worth more than
   the filter was.

Only build a filtered corpus as the default if step 3 shows the ranking moves.

## C. Method upgrades — where real quality jumps live

### C8. Zero-shot embeddings only — no domain adaptation
Likely the biggest model-side jump available: unsupervised adaptation
(TSDAE or SimCSE) on the ~185k hotel reviews, then re-run the sweep with
the adapted model as an extra candidate. **Effort:** ~1 GPU-night +
evaluation; research-flavoured, well-scoped.

### C9. Sentence splitting ignores clause structure
"Room was great **but** breakfast was awful" is one mention today — the
multi-aspect failure one level down. Splitting at contrastive conjunctions
(but/however/although) sharpens mentions noticeably. **Effort:** small,
slots into `data/segment.py` behind the same function.

### C10. Noise is discarded, never rescued
33–46% of mentions land in noise and are never looked at again; every one
is user feedback the app will never show.
**Fix:** second-pass assignment via HDBSCAN `membership_vector` /
`approximate_predict` with a confidence threshold — "uncertain" as a state
instead of "discarded". **Effort:** moderate. **Best product-value per
hour on this list.**

### C11. The LLM labeling path — generated and auto-scored, not yet judged
**Half closed (2026-08-08).** Ollama ran, 5 models × 2 prompts were scored on
format compliance, discrimination@1 and hallucination over the 20 largest
mention clusters (technology-selection §7); `terms_fallback` is no longer the
only label the project has produced.

**What remains is the half that matters:** all three metrics measure form, none
measures usefulness, so no labeler has yet been shown to produce a label a
*reader* benefits from. Blind multi-reviewer scoring is built (app → "Label
scoring", `score_label` records, Krippendorff α over reviewers) and **nothing
has been scored**. Until at least two people complete a pass, every labeler
claim in the project is an automatic-metric claim.
Still open from the original entry: watch for near-identical labels on
near-duplicate clusters (known LLM failure) — B6 would reduce the cause.

## D. Stability — WP9b, honestly

### D12. Cross-seed stability is unsolved
Best ARI is 0.74: even the most stable candidate reshuffles a quarter of
its pair relations on a seed change. Frozen seeds are a workaround, not a
solution. Two serious paths, both unbuilt:
- **Consensus clustering**: k seeded runs → co-assignment matrix → cluster
  that. More stable *and* yields per-document confidence.
- **Deterministic raw-space path**: partitioners without UMAP (experiment
  already proposed in guide §7).

### D13. No cluster identity across runs
Prerequisite for time series (WP10) and incremental updates (WP9b goal 2):
nothing exists. Start: Hungarian matching on cluster centroids between
runs; the two-stage micro-centroids are the natural anchor.

---

## The top 5, if time is scarce

1. **A1 — labeled eval set + metric validation:** turns every later
   decision from belief into knowledge.
2. **C10 — noise rescue:** largest product value per hour.
3. **B5/B6 — language detection + dedup:** cheap, prevents embarrassing
   artifact clusters in the demo.
4. **C11 + A2 — generate LLM labels and score them properly:** without
   named clusters everything else is theory.
5. **C8 — domain adaptation:** the one research-grade item that can lift
   the result level itself.

Perspective: A1–A4 are what separate coursework from defensible work — and
the fact that we can name them precisely is a product of the machinery
standing. None of them requires a rebuild; each docks onto an existing
seam (eval harness, segment module, label port, cluster backends).
