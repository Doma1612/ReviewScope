from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_COLUMNS: list[str] = [
    "run_id",
    "pipeline",
    "sample_size",
    "device",
    "timestamp",

    "embedding_model",
    "embedding_instruction",
    "embed_dim",
    "embed_time_s",

    "reduction_method",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "pca_components",

    "clustering_algo",
    "cluster_params",

    "n_clusters",
    "noise_count",
    "noise_ratio",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "coherence_cv",
    "rating_entropy",

    "runtime_s",

    "notes",
]

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


def _make_run_id(params: dict[str, Any]) -> str:
    stable = json.dumps(
        {k: params.get(k) for k in _IDENTITY_FIELDS},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:8]


def log_result(results_csv: Path, row: dict[str, Any]) -> None:
    run_id = _make_run_id(row)
    row.setdefault("run_id", run_id)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    if results_csv.exists():
        try:
            existing_ids = pd.read_csv(results_csv, usecols=["run_id"])["run_id"].values
            if run_id in existing_ids:
                print(f"  [skip]   run_id={run_id}  (already logged)")
                return
        except Exception:
            pass

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
    if not results_csv.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    df = pd.read_csv(results_csv)
    for col in RESULTS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[RESULTS_COLUMNS]
