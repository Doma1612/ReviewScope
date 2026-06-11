# ReviewScope ML pipeline — methodology and threats to validity

This document records, per pipeline stage: what was decided, on what evidence,
and what the known weaknesses are. It is written to be argued with. Where the
evidence is thin or ambiguous, it says so rather than overstating a
conclusion. The companion artifacts are the comparison report
(`data/runs/comparison_<N>/report.md`) and notebooks 02–08, which hold the raw
sweeps behind each decision.

---

## 1. Data and sampling

**Decision.** The fixed WP5 benchmark is the first N Yelp reviews (in dataset
order) whose business is tagged `Hotels` and whose text is ≥ 50 characters
(notebook 02). The 1k smoke sample is a strict prefix of the 5k benchmark so
results stay comparable across scales.

**Evidence.** Hotels were chosen over restaurants because hotel reviews are
naturally multi-aspect (room, staff, location, breakfast, value), giving
clusters that are distinguishable *by topic* rather than by cuisine.

**Weaknesses / threats to validity.**

- **Hotel-only sampling bias vs. the app's "any CSV/JSONL" promise.** Every
  empirical decision in this pipeline — preprocessing depth, embedding model,
  UMAP parameters, HDBSCAN minimum cluster sizes, even the Tier-3 metric —
  was tuned on English hotel reviews of 50–5,000 characters. The application
  accepts arbitrary corpora. There is **no evidence** the chosen
  configuration transfers to, say, German parliamentary speeches (the
  EuroParl Phase-2 target) or support tickets. The honest claim is: *the
  pipeline architecture is domain-independent; the parameter values are
  hotel-tuned defaults.* WP9 must re-run the comparison harness on each new
  corpus rather than trusting these defaults.
- **"First N matching" is not a random sample.** Dataset order correlates
  with business ID, which correlates with geography. The benchmark
  over-represents whichever cities come first in the dump. Acceptable for
  comparing pipelines against each other (all candidates see the same bias),
  not acceptable for claims about Yelp hotels in general.
- **Category co-occurrence leakage.** "Hotels"-tagged Yelp businesses are
  frequently co-tagged (restaurants, nightlife, venues), so the benchmark
  contains substantial restaurant/bar content — visible in the 1k smoke run,
  where a finalist's largest cluster is about oysters on Bourbon Street, not
  rooms. The qualitative inspection caught this; no metric did. Either accept
  it (it is what "hotel business reviews" really are) or tighten the filter
  to businesses tagged *only* Hotels — a decision to revisit before the 50k
  GPU benchmark.
- **Minimum length filter.** Dropping < 50-char reviews removes exactly the
  documents that are hardest to embed and cluster. Production data will
  contain them; the preprocess Celery step replicates the filter, but that
  means the app silently discards user data — this is documented behaviour,
  surfaced in the pipeline status, but a user may still be surprised.

## 2. Preprocessing

**Decision.** "Minimal" — whitespace normalisation only (notebook 02).

**Evidence.** Notebook 02 compared raw / minimal / aggressive variants;
sentence-transformer models are trained on natural text, and aggressive
lowercasing/punctuation-stripping consistently hurt downstream clustering
metrics slightly while never helping.

**Weaknesses.** Deduplication is not part of the benchmark preprocessing
(Yelp review IDs are unique), but real uploads will contain duplicates; the
app spec assigns dedup to the Preprocess step by primary key, which catches
exact re-uploads only, not near-duplicates (copy-paste reviews differing by
one word). Near-duplicate documents inflate cluster densities and can create
phantom "topics" — a known, unmitigated gap.

## 3. Embeddings

**Decision.** `all-mpnet-base-v2`, no instruction (notebook 04's comparison
on the 5k benchmark; instruction variants and larger models cached and
re-checkable).

**Evidence.** Notebook 04 evaluated 7 CPU-tier models × instruction variants
under a *fixed* UMAP+HDBSCAN configuration, so metric differences are
attributable to the embedding. Decisive was not raw silhouette but the
three-tier picture: instruction-tuned models *mechanically* raise silhouette
by reshaping the space toward the instruction, without raising coherence —
the geometry improves, the topics don't.

**Weaknesses.**

- The "fixed evaluation pipeline" (UMAP nc=10/nn=15, HDBSCAN mcs=15/ms=5) was
  itself chosen using mpnet embeddings (notebooks 05/06), creating a mild
  circularity: the referee was calibrated on one contestant. Mitigation:
  the parameters are deliberately mid-range, and the model ranking was stable
  across the sweep grid; but a model that only shines under very different
  DR parameters would have been missed.
- English-only model on an English-only benchmark; the multilingual question
  (bge-m3, multilingual-e5) is deferred to WP9 with no evidence either way.
- GPU-tier models (instructor-xl, e5-mistral-7b) were never run — the
  comparison is honest only within the CPU tier until the GPU sweep happens.

## 4. Dimensionality reduction

**Decision.** UMAP to 10 dimensions, `n_neighbors=15`, `min_dist=0.0`,
cosine metric, fixed seed; separate 2-D/3-D projections (`min_dist=0.1`) for
visualisation only (notebook 05). PCA→UMAP and PCA-only were tried and not
selected at this embedding dimensionality.

**Weaknesses — these are the big ones.**

- **UMAP distorts density, and HDBSCAN then clusters density.** UMAP
  preserves neighbourhood topology, not distances or densities; `min_dist=0`
  actively compacts regions. HDBSCAN's notion of "variable density clusters"
  is therefore applied to an artifact of the projection, not to the
  embedding space. Cluster boundaries in UMAP space can be projection
  artifacts. We accept this (the field broadly does — BERTopic is built on
  the same stack) because clustering in 768-d raw space is itself unreliable
  (distance concentration), but Tier-1 metrics measured *in UMAP space* must
  be read as "quality of the projection's clusters", never "quality of the
  semantic clusters". This is exactly why Tier 2 (coherence, computed from
  raw text) exists.
- **Nondeterminism.** UMAP is deterministic only with a fixed seed (which
  forces single-threaded layout, a real runtime cost at 50k+), and unstable
  *across* seeds. The harness quantifies this with multi-seed ARI; early smoke
  results show the instability is significant — which makes WP9b's
  "same corpus → same clusters" goal achievable only by *pinning* seeds, and
  "meaningful cluster identity across runs" still open.
- Tier-1 metrics in 10-d UMAP space systematically look better than the same
  clustering would in raw space. All variants share the bias except BERTopic
  (5-d internal space) — cross-variant Tier-1 comparisons therefore carry a
  small systematic advantage we cannot fully remove; flagged in the report.

## 5. Clustering

**Decision.** Four candidates compared as full pipelines (this is WP5's core
question, answered in the comparison report, finalised by a human):
BERTopic off-the-shelf, UMAP→HDBSCAN, UMAP→agglomerative(ward), and the
two-stage micro→macro clusterer.

**Evidence.** Notebook 06's sweeps set the per-algorithm defaults; the
package harness compares the four assembled pipelines under identical
artifacts and adds stability + noise-fairness, which the notebooks lacked.

**Weaknesses.**

- **Silhouette's bias toward noise-discarding algorithms.** HDBSCAN may label
  30–60% of a noisy corpus as `-1`; classic silhouette is then computed on
  the easy remainder. An algorithm can "win" Tier 1 by refusing to cluster
  the hard half of the data. Mitigation: the harness reports silhouette
  including noise as a pseudo-cluster, plus the noise fraction, and the
  report instructs readers to use them together. There is no single fair
  number; we publish the disagreement instead of hiding it.
- **The two-stage clusterer's macro count heuristic** (≈√n_micro, clamped to
  [5, 30]) is a readability heuristic, not an optimum; it has no validation
  beyond "produces human-reviewable counts". Treated as a reviewable
  parameter, with HITL splits as the correction mechanism.
- **Agglomerative ward on UMAP output** inherits the density distortion
  issue and additionally assumes roughly isotropic clusters; its k is fixed
  by hand. It earns its slot as the no-noise contrast candidate.

## 5b. Sentence-level (aspect) clustering — the fifth variant

**Decision.** ``sentence_level`` splits each review into sentence segments
(regex splitter, segments < 20 chars dropped, > 600 chars hard-wrapped) and
clusters the segments. Motivation: the median review has 8 sentences covering
several aspects; one embedding per review averages those aspects into one
vector, and ~10% of reviews are silently truncated at the embedder's token
window. At sentence level the clustered unit is the *mention* and both
problems disappear.

**Counting semantics** (the part that is easy to get wrong):

- Cluster ``size`` counts **mentions**; ``n_documents`` counts **distinct
  parent reviews** — both are stored and displayed together ("3,180 mentions
  in 2,410 reviews"). Anything that should reflect customers rather than
  verbosity (per-cluster mean stars, Tier-3 rating entropy) is computed on
  deduplicated (review, cluster) pairs, so one rambling reviewer cannot
  dominate a cluster's star profile.
- ``doc_membership.json`` maps every review to its cluster shares and a
  *primary* cluster (most mentions; noise never outranks a real cluster) —
  this is what fills the app's one-cluster-per-document field.

**Weaknesses.**

- The regex splitter mis-handles abbreviations; accepted as noise, swap in a
  proper sentence splitter behind ``data/segment.py`` if inspection shows it
  matters.
- Short evaluative sentences carry sentiment but no topic; the minimum-length
  filter removes the worst, but expect one or two **generic praise/complaint
  clusters** — that is normal at sentence level and exactly what the HITL
  ``mark_junk`` action is for. Watch Tier 3 on these.
- ~6x more points: the seeded (single-threaded) UMAP fit dominates runtime at
  50k documents (~300k segments). Run it deliberately, not casually.
- Metrics are computed on segments, so Tier-1/Tier-2 numbers are **not
  directly comparable** to the document-level variants — same corpus,
  different unit. Only the entropy (deduplicated) and the human inspection
  compare fairly across the unit boundary; the report ranks them together
  regardless, which is a known limitation to keep in mind when reading it.

## 6. Sentiment/topic entanglement (why Tier 3 exists)

Review embeddings encode sentiment as strongly as topic — angry reviews
resemble other angry reviews. A clustering can therefore score excellently on
geometry while having separated *ratings*, not *themes* ("all 1-star rants"
is a textbook failure cluster). Tier 3 (normalised star-rating entropy per
cluster) detects this: a genuinely thematic cluster (e.g. "breakfast")
attracts both praise and complaints → mixed ratings → high entropy; a
sentiment blob collapses to one rating level → low entropy. Thresholds
(>0.85 thematic, <0.60 sentiment-dominated) are calibrated only informally on
this corpus.

**Limitations.** Tier 3 requires a rating column (absent → metric silently
unavailable, app falls back to Tiers 1–2); it cannot distinguish "sentiment
blob" from a legitimately polarised topic (e.g. "bed bugs" will be ~all
1-star and is still a real topic — entropy alone would wrongly flag it; only
a human reading samples catches this). Tier 3 is a tripwire, not a verdict.

## 7. Representation and LLM labeling

**Decision.** c-TF-IDF top terms (label input, app word clouds via raw
within-cluster frequencies) + Ollama labeling with notebook 08's centroid
context strategy and v1 prompts; prompt hash and model recorded per label.

**Weaknesses.**

- **LLM label hallucination risk.** The model sees 5 sampled documents, not
  the cluster. Labels can be plausible-but-wrong (over-specific from a
  non-representative sample, or generic to the point of useless). Notebook
  08's human scoring sheet was never completed — the centroid/v1 choice rests
  on informal reading, which is why **label approval is a mandatory HITL
  step**, not a nicety. Every label carries `label_source` so unreviewed LLM
  text is always distinguishable from human-approved text.
- Centroid-sampled context makes clusters look more homogeneous to the LLM
  than they are (the same bias the inspection artifact avoids by using
  random samples). A label can be accurate for the core and wrong for the
  fringe.
- When Ollama is down the pipeline falls back to term-join labels and says
  so; it never fails the run and never fakes an LLM.

## 8. Evaluation protocol

Metrics shortlist; humans decide. Encoded in the report structure itself:
Step 1 ranks by mean rank over silhouette (excl. and incl. noise), C_v and
rating entropy; Step 2 is a per-finalist inspection sheet (label + top terms
+ **5 random member documents** — random, not centroid, because centroid
samples flatter the cluster) and an intruder test (4 members + 1 outsider,
shuffled; if the intruder isn't obvious the boundary isn't meaningful). The
report ends with a sign-off block; the Streamlit app records the
confirmation as a `confirm_run` feedback record. The final recommendation
must be able to state: *"a human reviewed the clusters of the winning
pipeline and confirmed they are thematically coherent."*

**Weaknesses.**

- **C_v coherence on short review texts.** C_v was designed for
  document-level topic models on longer texts; on short reviews the sliding
  window co-occurrence statistics are noisy, and C_v is known to reward
  generic high-frequency word sets. We use it as a *relative* signal between
  configurations on the same corpus, never as an absolute quality claim, and
  it is one vote among four in the shortlist ranking.
- The mean-rank shortlist weights all four metrics equally — defensible
  only as a tie-breaking convention; the report makes the raw table visible
  so a reader can re-rank under different priorities.
- Inspection and intruder tests are rendered for the *first seed's* run;
  a config could look better or worse under another seed (the ARI column
  says how much that risk exists per variant).

## 9. Where a human must stay in the loop (and why)

| Decision | Why automation is unreliable | Mechanism |
|---|---|---|
| Winning pipeline | metrics are proxies with documented biases (§4–§8) | report sign-off + `confirm_run` record |
| LLM label approval | hallucination risk, no ground truth (§7) | `approve_label` / `rename_label` |
| Cluster merges | near-duplicate detection is lexical; "same theme" is semantic | term-overlap flag proposes, human disposes (`merge_clusters`) |
| Cluster splits | no metric sees "two themes in one cluster" reliably | `split_cluster` → micro-cluster promotion or targeted re-cluster |
| Junk calls | length/boilerplate artifacts look like topics to every metric | `mark_junk` |
| Noise triage | HDBSCAN noise contains both garbage and misfits | `reassign_doc` |
| Sentiment-blob vs. polarised-topic | Tier 3 cannot tell them apart (§6) | inspection sheet reading |

Feedback is append-only versioned JSONL (`data/feedback/`), with reviewer and
timestamp; application semantics are specified in
`src/reviewscope_ml/hitl/apply_feedback.py` and tested in
`tests/test_feedback.py`.

## 10. Reproducibility status (WP9b)

- Deterministic re-runs: achieved *per seed* — config, seed, and parameter
  set fully determine every artifact (caches are keyed on all of them).
- Cross-seed stability: measured (ARI), not yet achieved; this is open
  research within WP9b, not an engineering gap.
- Incremental updates: not implemented; the two-stage design anticipates it
  (assign new documents to nearest micro-centroid) but no evidence yet.
