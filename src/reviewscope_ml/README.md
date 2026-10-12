# reviewscope_ml

Python package with the ML pipeline. The backend worker calls it through one function, `reviewscope_ml.app.run_from_upload`; the notebooks and evaluation scripts use the same modules. It has no web framework or database dependencies.

```
pip install -e .        # from the repository root
pytest tests            # no GPU and no model downloads required
```

## Pipeline

```
upload -> ingest -> preprocess -> segment -> embed -> reduce -> cluster -> sentiment -> represent -> label -> result
```

The app runs the `sentence_level` variant: every document is split into sentences, and the sentences are clustered. A document belongs to the cluster that holds most of its sentences.

| Step | Default in the app |
|---|---|
| Preprocess | Whitespace normalisation, texts shorter than 50 characters removed |
| Segment | Regex sentence splitter, sentences shorter than 20 characters removed |
| Embed | `all-MiniLM-L6-v2` |
| Reduce | UMAP to 10 dimensions for clustering; separate 2D and 3D projections for the map |
| Cluster | HDBSCAN, minimum cluster size 0.3% of the sentences (at least 15) |
| Sentiment | `twitter-roberta-base-sentiment-latest`, score per sentence, averaged per cluster |
| Represent | c-TF-IDF top terms and word frequencies |
| Label | `gemma3:4b` via Ollama; falls back to joined top terms when Ollama is unavailable |

The reasons behind these choices are documented in `docs/methodology.md` and `docs/technology-selection.md`.

## Structure

| Module | Responsibility |
|---|---|
| `app/` | Interface for the backend: upload schema, file ingest, default pipeline spec, progress and persistence ports, `run_from_upload` |
| `data/` | Benchmark loading, sentence segmentation, dataset builders in `data/sources/` (Yelp, Europarl) |
| `core/` | Configuration, disk caches, corpus profiles, metrics, experiment tracking |
| `embed/` | sentence-transformers wrapper with caching and multi-GPU encoding; `models.py` lists the evaluated models |
| `reduce/` | UMAP and PCA reduction |
| `cluster/` | HDBSCAN, KMeans, agglomerative and two-stage clustering backends |
| `sentiment/` | Sentiment scoring and per-cluster aggregation |
| `represent/` | Cluster terms and word frequencies |
| `label/` | Ollama labeler and the term fallback |
| `pipelines/` | Pipeline specs, the runner and the run artifact format |
| `eval/` | Metrics, comparison reports, model and labeler sweeps, label scoring |
| `hitl/` | Streamlit review app and feedback that is applied on the next run |
| `runtime/` | GPU selection on shared servers, resource monitoring |

## Pipeline variants

`pipelines/spec.py` defines five variants for comparison: `bertopic`, `custom_hdbscan`, `flat_agglomerative`, `two_stage` and `sentence_level`. All of them write the same run directory under `data/runs/`: `manifest.json`, `assignments.csv`, `clusters.json`, `metrics.json`, and for `sentence_level` also `doc_membership.json`. Every expensive step is cached under `data/cache/`, keyed on all of its inputs, so interrupted runs resume.

## Command line

| Command | Purpose |
|---|---|
| `python -m reviewscope_ml.data.sources.yelp --sample-size 5000` | Build a hotel review sample from the Yelp download |
| `python -m reviewscope_ml.data.sources.europarl --sample-size 5000` | Build a Europarl sample |
| `python -m reviewscope_ml.eval.report --sample-size 1000 --device cpu` | Run and compare all variants, write a report |
| `python -m reviewscope_ml.eval.model_sweep --sample-size 5000 --device cuda --sentence-level` | Compare embedding models |
| `python -m reviewscope_ml.eval.label_sweep --sample-size 5000 --device cuda` | Compare LLM labelers |
| `python -m reviewscope_ml.eval.cost_probe` | Measure runtime and memory per model |
| `streamlit run src/reviewscope_ml/hitl/app.py` | Review runs, approve labels, score labels |
| `python -m reviewscope_ml.hitl.apply_feedback <run_name>` | Apply recorded feedback to a run |

The scripts in `scripts/` wrap the runs used for the technology selection.

## GPU use on shared servers

Every GPU entry point calls `runtime.gpu.claim_gpu()`. It reads `nvidia-smi`, claims only idle GPUs with at least 6 GB free, pins the process with `CUDA_VISIBLE_DEVICES`, and falls back to CPU (or stops, for sweeps) when none is idle. Each process uses at most 50% of a GPU's memory and 4 CPU threads. torch is pinned to 2.7.1 with CUDA 12.6; newer builds lack kernels for Pascal GPUs such as the TITAN X.
