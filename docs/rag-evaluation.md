# RAG evaluation: project chat retrieval and answer grounding

**Date:** 2026-09-23 · **Script:** `scripts/rag_eval.py` · **Raw output:**
`data/runs/rag_eval/rag_eval_{hotels,europarl}.json` (every query, ranking,
grade, answer and per-sentence verdict)

A first measurement of the project chat (`api/project_chat.py` →
`services/project_rag.py`), which had not been evaluated before this. It
evaluates what ships: MiniLM-L6-v2 query embedding, exact cosine search over
the project's segments, the app's candidate policy (30 candidates, at most two
segments per review, 10 results), the app's own prompt (`rag_messages`,
rendered inside the api container), and the app's chat model (`llama3.2`,
same Ollama digest). Only the tool-planning step is bypassed: every question
goes straight to `search_project_evidence`, the path the planner falls back to.

---

## 1. Setup

| | Hotels 5k | EuroParl EN 26k (verified) |
|---|---|---|
| project | `1115f858…` | `08e2f047…` |
| segments | 43,012 | 352,924 |
| clustered (retrievable by the app) | 65.7% | 57.4% |
| vectors | re-encoded with MiniLM (the project predates migration 0007) | stored `vector_pg`; re-encoding 200 of them gives cosine ≥ 0.99999 |

Four retrievers, all under the app's candidate policy so that only the scoring
differs:

- **app**: cosine over *clustered* segments only (`Segment.cluster_id IS NOT NULL`), i.e. what ships
- **dense_all**: the same, with HDBSCAN noise segments included
- **bm25**: Okapi BM25 (k1 = 1.2, b = 0.75, English stop words) over all segments
- **hybrid**: reciprocal-rank fusion (k = 60) of dense_all and bm25

Three measurements:

- **A. Known-item retrieval.** 200 random segments (≥ 60 chars) per corpus.
  `gemma3:4b` writes one natural question that the segment answers, paraphrased
  rather than copied (mean content-word overlap with the source: 0.28 hotels,
  0.30 EuroParl). Metric: is *that* segment retrieved, Hit@k, MRR@10, nDCG@10;
  plus Doc-Hit@10 (any segment of the same review/speech).
- **B. Graded topical relevance.** 12 hand-written analyst queries per corpus
  plus 2 deliberately niche ones. The top 10 of all four retrievers are pooled
  and each (query, segment) pair is graded 0/1/2 by `gemma3:4b`. P@5, P@10
  (grade ≥ 1), nDCG@10 against the judged pool, MRR.
  Niche queries are excluded from the averages (see §4).
- **C. Answer grounding.** For each of the 14 queries, the app's top 10 go
  through the app's prompt to `llama3.2`. Citation validity (cited IDs exist),
  citation coverage (share of answer sentences carrying an `[S#]` citation),
  faithfulness (share of cited sentences that `gemma3:4b` judges fully
  supported by the cited segments; RAGAS-style, per sentence rather than per
  decomposed claim).

## 2. Results

### A. Known-item retrieval (200 queries per corpus)

| | Hotels Hit@10 | Hotels MRR@10 | EuroParl Hit@10 | EuroParl MRR@10 | EuroParl nDCG@10 |
|---|---|---|---|---|---|
| **app** | **0.225** | **0.155** | **0.325** | **0.214** | **0.240** |
| dense_all | 0.335 | 0.244 | **0.540** | **0.372** | **0.412** |
| bm25 | 0.300 | 0.210 | 0.290 | 0.189 | 0.213 |
| hybrid | **0.370** | 0.240 | 0.445 | 0.300 | 0.335 |

Split by whether the source segment is clustered or noise:

| Hit@10 | Hotels clustered (n=123) | Hotels noise (n=77) | EuroParl clustered (n=120) | EuroParl noise (n=80) |
|---|---|---|---|---|
| app | 0.366 | **0.000** | 0.542 | **0.000** |
| dense_all | 0.333 | 0.338 | 0.525 | 0.562 |

### B. Topical precision (12 queries per corpus, LLM-judged)

| | Hotels P@10 | Hotels nDCG@10 | EuroParl P@10 | EuroParl nDCG@10 |
|---|---|---|---|---|
| **app** | 0.942 | 0.942 | 0.983 | 0.972 |
| dense_all | 0.942 | 0.949 | 1.000 | 0.996 |
| bm25 | 0.958 | 0.962 | 0.967 | 0.962 |
| hybrid | **0.983** | **0.985** | 1.000 | 0.996 |

MRR is 1.0 for every retriever on every query: the first result is always
relevant. The only hotel query with real misses is "rude or unhelpful front
desk staff" (app P@10 = 0.4), where dense retrieval returns *friendly* front
desk mentions, which match the topic but not the stance.

### C. Answer grounding (12 queries per corpus)

| | Hotels | EuroParl run 1 | EuroParl run 2 |
|---|---|---|---|
| citation validity (no invented IDs) | 1.00 | 1.00 | 1.00 |
| answers using the `[S#]` format at all | 10 / 12 | 11 / 12 | 9 / 12 |
| citation coverage | 0.64 | 0.60 | 0.46 |
| faithfulness of cited sentences | 0.89 | 0.79 | 0.89 |
| median latency (GPU Ollama) | 2.6 s | 3.1 s | 2.6 s |

EuroParl was run twice. Generation with a fixed seed is not reproducible under
parallel requests, and with 12 queries the difference between the two runs
(coverage 0.60 vs 0.46) is the error bar. The hotel run reproduced exactly.

## 3. Findings

1. **The noise filter is the largest retrieval defect, and it is a one-line
   choice.** The app searches only clustered segments, so the 34% (hotels) and
   43% (EuroParl) of mentions that HDBSCAN labels as noise are *unretrievable*:
   Hit@10 on noise-sourced queries is exactly 0. On clustered sources the app
   and dense_all perform the same (0.54 vs 0.53 EuroParl). Removing the
   filter alone lifts EuroParl Hit@10 from 0.33 to 0.54 (+66%) and hotels
   from 0.23 to 0.34. Noise is a clustering judgment ("this mention does not
   belong to a theme"), not a relevance judgment. The price: noise segments
   have no cluster label to show beside the citation, so the UI needs a
   "no cluster" state.
2. **Whether hybrid helps depends on the corpus, and the difference is the
   same one that shows up in embedding selection.** On hotels, BM25 is nearly
   as good as dense (0.30 vs 0.34) and fusion is best (0.37). On EuroParl,
   BM25 is much weaker (0.29 vs 0.54) and fusion *hurts* (0.45). Hotel
   mentions are short and lexically repetitive; parliamentary sentences are
   long and paraphrase-heavy. A hybrid retriever should be switched on per
   corpus profile, not globally.
3. **On topics that the corpus covers well, precision is at ceiling for every
   retriever.** P@10 is 0.94–1.00 for all four systems. At this corpus density
   *any* reasonable retriever finds ten relevant sentences, so B does not
   separate the systems; A does. This is a statement about the queries (broad
   topics), not proof that retrieval is solved.
4. **Out-of-vocabulary queries fail in a characteristic way.** "cryptocurrencies
   like Bitcoin" against a 1996–2011 corpus retrieves "We must compare like
   with like" and "I do not like this system": with nothing matching
   "Bitcoin", the stop word dominates MiniLM's pooled query vector. Hotels'
   "paying with cryptocurrency" gets P@10 = 0.0, and the answer then
   **hallucinated support** ("'They accept all forms of payment', which
   includes cryptocurrencies like Bitcoin"). The judge caught it
   (faithfulness 0.5). A retrieval score threshold, or an explicit
   "no evidence" path when the best cosine is low, would stop the model from
   being handed irrelevant evidence and told to answer from it.
5. **The generator never invents citation IDs, but often doesn't use the
   format.** Validity is 1.00 everywhere. However, 2–3 of 12 answers per run
   write "Review S2", "Review #S1", "(S4, S8)", "[$S1]" or cite the speech's
   primary key (`ep-04-12-15#4`), and about half of all answer sentences carry
   no citation. The frontend only lists sources and does not link inline
   citations, so nothing breaks, but traceability suffers. Cheap fixes:
   normalise `S\d+` variants server-side, and stop printing
   `Review: {primary_key}` in the evidence block, which invites citing the key.
6. **The prompt is hotel-specific.** `rag_messages` hardcodes "Review:" and
   "reviews", so llama3.2 calls parliamentary speeches "reviews" throughout.
   The unit noun belongs in the corpus profile (`core/corpus.py` already has
   `unit_noun`).
7. **Faithfulness failures are overgeneralisation, not invention.** Typical
   unsupported sentences: "[S3], [S5], [S8] and [S10] all mention difficulty
   sleeping" (two of them don't), "[S1] … BSE is also known as mad cow
   disease" (true, but not in S1). The model summarises a group of citations
   more confidently than the individual segments allow.

## 4. Validity of the measurements

- **LLM judge agreement.** The author regraded 40 random hotel judgments:
  relevant/not relevant agreed on 38/40 (Cohen's κ ≈ 0.87); both
  disagreements were borderline 2-vs-1 calls. A random 25 of the EuroParl
  topical judgments agreed 25/25. **On the niche queries the judge is lenient**:
  it gave grade 2 to "Firstly, there is the scope of the mechanism" for REACH,
  to "It has broad support from social platforms and civil society" for
  social-media regulation, and to "However, I would like to give you …" for
  Bitcoin. By the author's reading, roughly 20–30% of niche-query judgments
  are false positives. That is why niche queries are excluded from B's
  averages, and why B should be read as an upper bound. The judge is a 4B
  model; the one-person validation is a sanity check, not an agreement study.
- **Known-item recall is a lower bound.** Only the source segment counts as
  relevant. Short review sentences have many near-duplicates ("The front desk
  was not helpful"), so a retriever that returns an equivalent sentence from
  another review is scored as a miss. This hurts hotels more than EuroParl, and
  it affects all retrievers equally, so the *comparison* holds even though the
  absolute numbers are pessimistic.
- **Synthetic queries share a model family with nothing in the stack except
  the judge.** Queries were generated and judged by `gemma3:4b`, while
  retrieval uses MiniLM and generation uses llama3.2. The queries are
  paraphrased (≈ 30% content-word overlap). A retriever-query leak is
  therefore unlikely, but BM25 still benefits from whatever overlap remains.
- **Small n.** 12 topical queries per corpus; 200 known-item queries. Treat
  differences below ~0.05 in B and ~0.07 in A (binomial SE at n = 200 is
  ~0.03) as noise.
- **Not evaluated:** the llama3.2 tool planner (known to be unstable, see
  commit `1502acd`), the cluster-level chat, multi-turn history, and anything
  about the answer's *usefulness* to an analyst.

## 5. Recommendations, in order of value per effort

1. Drop the `cluster_id IS NOT NULL` filter in `retrieve_project_sources` (show
   "unclustered" in the source list). Largest measured gain; one line plus a UI
   state.
2. Normalise citation variants (`Review S2`, `#S2`, `(S2)`) to `[S2]`
   server-side, and remove the `Review: {primary_key}` line from the evidence
   block.
3. Take the unit noun ("review" / "speech") from the corpus profile in
   `rag_messages`.
4. Add a minimum-similarity gate: below it, answer "no evidence in this
   project" without calling the LLM.
5. Hybrid retrieval, enabled per corpus profile (hotels yes, EuroParl no), once
   an HNSW index exists; BM25 would need Postgres full-text search or a
   sidecar index.

## Reproduce

Needs the docker stack up; the Ollama instance can be the stack's own (CPU,
slow) or a GPU one with the same model digests.

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu-uuid> OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=8192 \
    nohup ~/ollama/bin/ollama serve > ~/ollama/serve.log 2>&1 &
export CUDA_VISIBLE_DEVICES=<idle-gpu-uuid>
.venv/bin/python scripts/rag_eval.py --corpus hotels   --ollama http://127.0.0.1:11435
.venv/bin/python scripts/rag_eval.py --corpus europarl --ollama http://127.0.0.1:11435
```

~6 minutes per corpus on one TITAN X. The segment export is cached in
`data/cache/rag_eval/`.
