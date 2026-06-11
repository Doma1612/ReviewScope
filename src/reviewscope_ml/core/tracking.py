"""
Append-only results log.

Every (model × instruction × reduction × clustering) combination appends one
row to results.csv.  Rows are deduplicated by run_id so re-running a notebook
is safe — already-logged experiments are silently skipped.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ── Column order (canonical schema) ──────────────────────────────────────────

RESULTS_COLUMNS: list[str] = [
    # ── Provenance ──────────────────────────────────────────────────────────
    "run_id",               # sha256[:8] of identity params — dedup key
    "pipeline",             # "custom" | "bertopic"
    "sample_size",
    "device",
    "timestamp",            # ISO 8601 UTC

    # ── Embedding ───────────────────────────────────────────────────────────
    "embedding_model",
    "embedding_instruction",  # "no_inst" | "generic" | "domain" | "sentiment"
    "embed_dim",
    "embed_time_s",

    # ── Dimensionality reduction ─────────────────────────────────────────────
    "reduction_method",     # "umap" | "pca_umap" | "pca"
    "umap_n_components",    # null for pca-only
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "pca_components",       # null unless pca_umap or pca

    # ── Clustering ───────────────────────────────────────────────────────────
    "clustering_algo",      # "hdbscan" | "kmeans" | "agglomerative" | "bertopic_internal"
    "cluster_params",       # JSON string of algo-specific params

    # ── Quality metrics ──────────────────────────────────────────────────────
    "n_clusters",
    "noise_count",
    "noise_ratio",
    # Tier 1 — geometric (measured in UMAP/PCA space)
    "silhouette",           # null if < 2 clusters; computed on non-noise docs only
    "davies_bouldin",       # lower is better; null if < 2 clusters
    "calinski_harabasz",    # higher is better; null if < 2 clusters
    # Tier 2 — topic coherence (independent of embedding geometry)
    "coherence_cv",         # Gensim C_v NPMI; null if gensim not installed or < 2 clusters
    # Tier 3 — rating distribution (thematic vs sentiment-blob check)
    "rating_entropy",       # normalised star-rating entropy [0,1]; null if no star data

    # ── Timing ──────────────────────────────────────────────────────────────
    "runtime_s",            # wall-clock: embed + reduce + cluster

    # ── Free text ───────────────────────────────────────────────────────────
    "notes",
]

# Fields that together uniquely identify an experiment (used for run_id hash)
_IDENTITY_FIELDS: list[str] = [
    "pipeline",
    "sample_size",
    "embedding_model",
    "embedding_instruction",
    "reduction_method",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "pca_components",
    "clustering_algo",
    "cluster_params",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_run_id(params: dict[str, Any]) -> str:
    """Return an 8-char deterministic hash of the experiment identity."""
    stable = json.dumps(
        {k: params.get(k) for k in _IDENTITY_FIELDS},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:8]


# ── Public API ────────────────────────────────────────────────────────────────

def log_result(results_csv: Path, row: dict[str, Any]) -> None:
    """
    Append one experiment row to *results_csv*.

    - Creates the file with a header row if it does not exist yet.
    - Skips silently if a row with the same ``run_id`` already exists
      (idempotent: safe to re-run notebooks).
    - Extra keys in *row* are ignored; missing keys are filled with ``None``.

    Parameters
    ----------
    results_csv : Path
        ``cfg.results_csv`` from PipelineConfig.
    row : dict
        At minimum, fill the identity fields plus the metric fields returned
        by ``compute_metrics()``.  Example::

            log_result(cfg.results_csv, {
                "pipeline": "custom",
                "sample_size": cfg.sample_size,
                "device": cfg.device,
                "embedding_model": "all-mpnet-base-v2",
                "embedding_instruction": "no_inst",
                "embed_dim": 768,
                "embed_time_s": 12.4,
                "reduction_method": "umap",
                "umap_n_components": 10,
                "umap_n_neighbors": 15,
                "umap_min_dist": 0.0,
                "umap_metric": "cosine",
                "clustering_algo": "hdbscan",
                "cluster_params": json.dumps({"min_cluster_size": 15, "min_samples": 5}),
                **metrics,   # dict from compute_metrics()
            })
    """
    run_id = _make_run_id(row)
    row.setdefault("run_id", run_id)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Dedup check
    if results_csv.exists():
        try:
            existing_ids = pd.read_csv(results_csv, usecols=["run_id"])["run_id"].values
            if run_id in existing_ids:
                print(f"  [skip]   run_id={run_id}  (already logged)")
                return
        except Exception:
            pass  # malformed CSV: just append

    # Normalise to schema column order, fill missing with None
    normalised = {col: row.get(col) for col in RESULTS_COLUMNS}

    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(normalised)

    sil  = row.get("silhouette")
    coh  = row.get("coherence_cv")
    entr = row.get("rating_entropy")
    n_c  = row.get("n_clusters")
    rt   = row.get("runtime_s")
    print(
        f"  [logged] run_id={run_id}  "
        f"clusters={n_c}  silhouette={sil}  "
        f"coherence={coh}  entropy={entr}  runtime={rt}s"
    )


def load_results(results_csv: Path) -> pd.DataFrame:
    """
    Load the full results CSV as a DataFrame.

    Returns an empty DataFrame with the correct columns if the file does not
    exist yet — safe to call before any experiments have been run.
    """
    if not results_csv.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    df = pd.read_csv(results_csv)
    # Ensure all canonical columns are present (forward-compatible)
    for col in RESULTS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[RESULTS_COLUMNS]
