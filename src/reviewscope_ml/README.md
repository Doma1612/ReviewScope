# reviewscope_ml

Pure-Python ML pipeline package — no Jupyter dependencies. The experiment
notebooks (00–11) orchestrate it; the FastAPI/Celery backend will import it.

```bash
pip install -e .            # from the repo root
pytest                      # pure-logic tests, no GPU, no model downloads
```

## Module → app-spec Celery step mapping

| Module | Celery step (app spec) | Notes |
|---|---|---|
| `data/` (ingest, preprocess, segment) | 1 Ingest, 2 Preprocess | benchmark builder (any Yelp category), "minimal" preprocessing (nb 02), sentence segmentation for aspect-level clustering |
| `embed/` | 3 Embed | sentence-transformers (+instruction prompts), disk-cached, device-aware, CUDA-OOM batch backoff; `models.py` = curated candidate registry |
| `reduce/` | 4 Reduce | UMAP / PCA→UMAP, seeded; separate 2-D/3-D viz projections |
| `cluster/` | 5 Cluster | HDBSCAN, KMeans, agglomerative, two-stage micro→macro |
| (sentiment) | 6 Sentiment | not in this package yet; Tier-3 rating entropy covers the benchmark's needs |
| `represent/` | feeds 7 + word clouds | c-TF-IDF & TF-IDF terms, word frequencies |
| `label/` | 7 Label | Ollama label+summary, prompt-hash recorded, term fallback when LLM is down |
| `pipelines/` | 8 Finalize | five end-to-end variants (incl. sentence-level mentions), one artifact schema (assignments, coords, clusters, manifest, doc membership) |
| `eval/` | — (WP5 harness) | three-tier metrics, noise fairness, multi-seed ARI, inspection + intruder test, comparison report |
| `hitl/` | — (review loop) | Streamlit review app, versioned JSONL feedback, apply-on-rerun |
| `runtime/` | — (ops) | shared-GPU claim/release etiquette, per-stage wall/RSS/VRAM monitor |

## Common commands

```bash
# Build the benchmark sample from the raw Yelp dump (idempotent)
python -m reviewscope_ml.data.ingest --sample-size 5000

# CPU smoke test — REQUIRED before any GPU run
python -m reviewscope_ml.eval.report --sample-size 1000 --device cpu

# Embedding model sweep (curated registry in embed/models.py; nb 04 protocol)
python -m reviewscope_ml.eval.model_sweep --sample-size 5000 --device cuda
#   subset: --models qwen3 bge-m3      gated EmbeddingGemma needs `hf auth login`

# Subset of pipeline variants (sentence_level is the expensive one at 50k)
python -m reviewscope_ml.eval.report --sample-size 50000 --device cuda \
    --variants custom_hdbscan two_stage sentence_level

# Review a finished run, record feedback
streamlit run src/reviewscope_ml/hitl/app.py

# Apply recorded feedback -> <run>__reviewed
python -m reviewscope_ml.hitl.apply_feedback two_stage__1000__s42 --sample-size 1000
```

## Running on the shared GPU server (etiquette is enforced, not optional)

The box has 4× TITAN X (12 GB), no scheduler; other groups are usually on
GPUs 0–1. **torch must be 2.7.1+cu126** — newer wheels dropped Pascal
(sm_61) kernels and fail at the first CUDA op (see requirements.txt).
Every GPU entry point goes through `runtime.gpu.claim_gpu()`, which

1. queries `nvidia-smi` and claims **idle devices only** — by default every
   idle GPU (`--gpus auto`); the embed stage runs data-parallel across them
   for a near-linear speedup. Busy devices are never touched,
2. pins the process to the claim via `CUDA_VISIBLE_DEVICES`,
3. **refuses to start** if no device has ≥ 6 GB free (falls back to CPU or
   tells you to come back later — never squeeze in),
4. logs the claim and the release.

`--gpus 1` restores the conservative single-device claim; only embedding
parallelises (UMAP/HDBSCAN are CPU), so multi-GPU shortens the embed stage,
not the whole run.

```bash
# 0) look before you leap (claim_gpu does this too, but look anyway)
nvidia-smi

# 1) CPU smoke first, always
python -m reviewscope_ml.eval.report --sample-size 1000 --device cpu

# 2) full comparison on GPU — one short-lived process; the embed stage
#    releases the model + CUDA cache the moment embeddings are cached
python -m reviewscope_ml.eval.report --sample-size 5000 --device cuda
```

In Python, the same flow per stage:

```python
from reviewscope_ml.runtime import claim_gpu
from reviewscope_ml import load_config

with claim_gpu() as claim:                       # picks + pins freest GPU
    cfg = load_config(sample_size=50_000, device=claim.device,
                      gpu_id=claim.gpu_id, cuda_mem_fraction=0.5, cpu_threads=4)
    ...                                          # run ONE stage, then exit
```

Rules baked into the code (see `runtime/gpu.py` for the rationale):
one GPU only, ≤ 50% of its VRAM, ≤ 4 CPU threads, prefer one short-lived
process per stage over a long-lived kernel, and long runs are
checkpoint/resume via the stage caches — a crash never forces a full re-run.

## Artifact schema (the contract with the app)

Every pipeline variant writes the same run directory under `data/runs/`:
`manifest.json` (spec, seed, per-stage cost, label sources, human
confirmation), `assignments.csv` (doc → cluster + 2-D/3-D coords),
`clusters.json` (label, summary, label_source, terms, word frequencies,
random sample doc ids), `metrics.json`. The backend can consume any variant
interchangeably; the HITL feedback JSONL in `data/feedback/` is the
GUI-independent review contract.
