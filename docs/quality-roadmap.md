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

### B7. Category leak (known, undecided)
Hotels-tagged businesses include restaurants/bars. Decide hotel-only filter
vs. documented acceptance **before** 50k numbers reach a presentation.

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

### C11. The LLM labeling path is factually untested
Every label so far is `terms_fallback`; not a single Ollama label has been
generated, let alone evaluated. Named clusters are the product promise —
this part is entirely outstanding. **Fix:** run Ollama labeling, then A2's
structured scoring. Watch for near-identical labels on near-duplicate
clusters (known LLM failure).

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
