# Mission: Build, evaluate, and harden the ReviewScope ML pipeline

You are acting as a senior ML engineer **and** a methodically rigorous reviewer (think: a professor grading this work). You build production-quality, modular pipeline code — and you also critique the methodology, document weaknesses honestly, and identify where a human must stay in the loop.

## North star: the clusters must make sense to a human

This overrides everything else. A pipeline that scores well on silhouette/coherence but produces clusters a reader can't name in one phrase has **failed**. Metrics are proxies for ranking candidates cheaply — they never declare the winner alone. Concretely:

- Every evaluated configuration gets a **qualitative inspection artifact**: per cluster, the top terms, the LLM label, and 5 randomly sampled documents (random, not nearest-to-centroid — centroids flatter the cluster), rendered side by side so a human can judge "is this one theme?" in seconds.
- Apply an **intruder test** to the finalists: for each cluster, show 4 of its documents plus 1 from another cluster; if the intruder isn't obvious, the cluster boundary is not meaningful. Automate the rendering, let the human judge (this is a core HITL point).
- Watch for the classic failure modes and call them out per run: sentiment blobs ("all angry reviews") instead of topics — that's what Tier 3 rating entropy detects; one giant junk cluster plus crumbs; near-duplicate clusters that should be one; clusters defined by length or language artifacts rather than content.
- The final recommendation must include the sentence "a human reviewed the clusters of the winning pipeline and confirmed they are thematically coherent" — and the HITL GUI is where that confirmation happens and gets recorded.

## Read first (in this order, before writing any code)

1. `docs/application-spec.md` — the app this pipeline must plug into (Celery steps: Ingest → Preprocess → Embed → Reduce → Cluster → Sentiment → Label).
2. `docs/project-plan.md` — WP5 (pipeline eval + impl) and WP9b (deterministic re-runs, incremental clustering) define the scope.
3. `docs/architecture/tech-selection.md` — decided stack: FastAPI, Celery + Redis, Postgres + pgvector, Ollama for labeling. The NLP/ML choices are explicitly to be decided **empirically** — that is your job.
4. `notebooks/00_config.ipynb` through `08_llm_labeling.ipynb` — the existing (unfinished) experiment pipeline. Reuse its decisions and utilities; do not silently re-decide what notebooks already decided (preprocessing variant, embedding candidates, metric tiers).
5. `notebooks/utils/` — `config.py` (PipelineConfig with GPU pinning), `metrics.py` (three-tier evaluation), `cache.py`, `results_tracker.py`. **Extend these, don't duplicate them.**

## Hard constraints — shared university GPU server etiquette (non-negotiable)

The server is a shared bare box: 4× TITAN X Pascal (12 GB each), 32 CPU cores, **no scheduler** — fairness is courtesy only. Other users are usually on GPUs 0 and 1.

- **Before any GPU work**, run `nvidia-smi`, pick the *emptiest* GPU, and pin to that ONE device via `load_config(device="cuda", gpu_id=<free>, cuda_mem_fraction=0.5, cpu_threads=4)`. Never claim more than one GPU. Never raise `cuda_mem_fraction` above 0.5 without checking the GPU is otherwise idle.
- **Release resources immediately when a stage finishes**: `del model`, `torch.cuda.empty_cache()`, and prefer running each pipeline stage as a separate short-lived process over one long-lived notebook kernel holding VRAM. No idle kernels squatting on a GPU.
- Write a small `src/reviewscope_ml/runtime/gpu.py` helper that (a) queries `nvidia-smi` programmatically, (b) selects the freest GPU, (c) refuses to start if every GPU is busy (fall back to CPU or a smaller sample instead of squeezing in), and (d) logs what it claimed and when it released it. Every entry point must go through it.
- Long runs must be **checkpointed and resumable** (per-stage caching already exists in `utils/cache.py` — build on it), so a crash never forces a full re-run.
- Smoke-test everything on CPU with `sample_size=1_000` before any GPU run.

## What to build

### 1. Modular package: `src/reviewscope_ml/`

Pure-Python package, importable by the FastAPI/Celery backend later — **no notebook or Jupyter dependencies in src**. Structure it as pluggable stages behind small interfaces (protocol classes or ABCs), each configurable and individually cacheable:

- `data/` — ingest + preprocess (reuse `get_preprocessor` decisions from notebook 02).
- `embed/` — sentence-transformers backends incl. instruction-tuned models from notebook 04; batched, device-aware, cached to disk.
- `reduce/` — UMAP (and PCA→UMAP) with the parameters notebook 05 selected; deterministic via seed handling (WP9b goal 1).
- `cluster/` — interchangeable backends: HDBSCAN, KMeans, Agglomerative, **and a two-stage micro→macro clusterer** (fine-grained HDBSCAN micro-clusters, then agglomerative merging of micro-cluster centroids into macro topics, preserving the micro→macro hierarchy).
- `represent/` — cluster keyword extraction: c-TF-IDF (BERTopic-style) and plain TF-IDF top terms; this also feeds the word clouds in the app spec.
- `label/` — LLM labeling via Ollama (notebook 08 strategies), with label/summary per cluster.
- `pipelines/` — four end-to-end assemblies to compare: (a) BERTopic off-the-shelf, (b) custom embed→UMAP→HDBSCAN, (c) flat hierarchical (agglomerative), (d) custom two-stage micro→macro. All driven by one config object, all emitting the same artifact schema (per-doc cluster_id, 2D/3D coords, per-cluster terms/label/summary) so the app can consume any of them interchangeably.
- `eval/` — see below.
- `hitl/` — see below.

### 2. Evaluation harness — decide empirically which pipeline wins

Extend the existing three-tier evaluation (geometric: silhouette/Davies-Bouldin/Calinski-Harabasz; topic coherence C_v; star-rating entropy) and add:

- **Stability**: Adjusted Rand Index across ≥3 seeds per configuration (directly serves WP9b determinism).
- **Noise handling fairness**: HDBSCAN discards noise points, which inflates silhouette — report metrics both including and excluding noise, and report noise fraction itself.
- **Runtime + peak VRAM** per stage (matters for the app's Celery workers).
- A comparison report (markdown + charts) ranking the four pipelines, with an explicit reasoned recommendation: not just "X scored highest" but *why*, on which tiers, and with which caveats.
- The **qualitative inspection artifact and intruder test from the north-star section are part of this harness**, not an afterthought: metrics shortlist 2–3 finalists, human inspection picks the winner. Encode that two-step protocol in the report structure itself.

Log every run through `results_tracker.py` so notebook 07's comparison keeps working.

### 3. Human-in-the-loop — identify, then build

First **identify and document** where automated steps are unreliable and need human review (expected candidates: cluster merge/split decisions, LLM label approval, outlier/noise triage, micro→macro merge boundaries). Then build a small review GUI:

- **Streamlit app** in `src/reviewscope_ml/hitl/app.py` (Streamlit so it stays decoupled from the React frontend; the *feedback format* is the contract, not the GUI).
- Loads a finished run's artifacts; shows clusters with terms, samples, label, 2D scatter.
- Reviewer actions: approve/rename label, merge clusters, split cluster (flag for re-clustering), reassign individual misfiled documents, mark cluster as junk.
- Persists feedback as versioned JSONL in `data/feedback/` with reviewer + timestamp.
- **Feedback must flow back**: a `hitl/apply_feedback.py` step that consumes the JSONL and applies it on the next run — label overrides directly; merges as post-hoc cluster mapping; splits/reassignments as must-link/cannot-link-style constraints or targeted re-clustering of the flagged cluster. Document precisely what each feedback type does on re-run.

### 4. Thin notebooks

New notebooks that only orchestrate src code and visualize results (no logic in cells): `10_pipeline_comparison.ipynb` (run all four pipelines on the benchmark sample, render the comparison report) and `11_hitl_roundtrip.ipynb` (demonstrate feedback → re-run → improved result). Keep them runnable top-to-bottom on CPU at small sample size.

### 5. Methodology review — the professor pass

Write `docs/methodology.md`. For each pipeline stage: what was decided, on what evidence, what the known weaknesses and threats to validity are. Be honest and specific. At minimum address: hotel-only sampling bias vs. the app's "any CSV/JSONL" promise; silhouette's bias toward noise-discarding algorithms; C_v coherence reliability on short review texts; UMAP's nondeterminism and distortion of density (which HDBSCAN then clusters); sentiment/topic entanglement (Tier 3 exists for this — explain it); LLM label hallucination risk; and exactly which decisions you concluded require a human in the loop and why.

## Working style

- Plan before coding; keep a todo list; work in small, reviewable commits on this branch with clear messages.
- Every module gets docstrings explaining **why**, not just what. Reasoning must be understandable to teammates (Daniel, Shokoufeh) who didn't watch you work.
- Add lightweight pytest tests for the pure logic (config, caching keys, feedback application, metric edge cases like single-cluster/all-noise) — no GPU needed in tests.
- Don't break existing notebooks 00–08 or change their recorded results.
- When evidence is ambiguous, say so in the docs rather than overstating a conclusion.

## Definition of done

1. `pytest` passes; `pip install -e .` (or path import) works; src has zero Jupyter deps.
2. CPU smoke run of all four pipelines at 1k samples completes end-to-end and writes comparison artifacts.
3. GPU run instructions documented (one command per stage, with the nvidia-smi check built in).
4. Streamlit HITL app launches, round-trips feedback, and a re-run demonstrably incorporates it.
5. `docs/methodology.md` and the comparison report exist and are readable by a non-author.
6. The comparison report contains the qualitative inspection (random samples + intruder test) for the finalists, and the winner was chosen by human-readable cluster quality, with metrics as supporting evidence — not the other way around.
7. A short `src/reviewscope_ml/README.md` maps each module to the app-spec Celery step it will plug into.
