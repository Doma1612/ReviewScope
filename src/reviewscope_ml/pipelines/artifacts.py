"""
Common artifact schema every pipeline variant emits.

The point of this module is the contract in the mission brief and app spec:
whichever pipeline wins, the application consumes the SAME files —
per-document cluster assignment + 2D/3D coords, per-cluster terms/label/
summary, and a manifest with provenance. The four pipeline variants differ
only in how they fill these files in.

Layout of one run directory (``data/runs/<run_name>/``)::

    manifest.json      provenance: spec, seed, sample size, stage timings,
                       peak memory, label source, applied feedback files
    assignments.csv    doc_id, cluster_id, micro_cluster_id, x2/y2, x3/y3/z3
    clusters.json      per cluster: label, summary, label_source, terms,
                       word_frequencies, size, mean_stars, sample_doc_ids
    metrics.json       three-tier metrics incl. noise-incl/excl variants

Plain CSV/JSON on purpose: artifacts must be diffable, inspectable without
pandas/pyarrow, and trivially loadable by the FastAPI backend.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    label: str
    summary: str
    label_source: str                       # "ollama:<model>" | "terms_fallback" | "hitl_override"
    top_terms: list[list]                   # [[term, score], ...] c-TF-IDF
    tfidf_terms: list[list]                 # [[term, score], ...] plain TF-IDF
    word_frequencies: dict[str, int]        # word-cloud counts
    sample_doc_ids: list[str]               # RANDOM sample, not nearest-centroid
    mean_stars: Optional[float] = None
    prompt_hash: Optional[str] = None
    micro_cluster_ids: list[int] = field(default_factory=list)  # two-stage only


@dataclass
class RunArtifacts:
    run_name: str
    manifest: dict[str, Any]
    doc_ids: list[str]
    labels: np.ndarray                      # cluster id per doc, -1 noise
    coords_2d: np.ndarray                   # (n, 2)
    coords_3d: np.ndarray                   # (n, 3)
    clusters: dict[int, ClusterInfo]
    metrics: dict[str, Any]
    micro_labels: Optional[np.ndarray] = None  # two-stage only

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(self.clusters)


def save_run(run_dir: Path, art: RunArtifacts) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = dict(art.manifest)
    manifest.setdefault("run_name", art.run_name)
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    with open(run_dir / "assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "cluster_id", "micro_cluster_id", "x2", "y2", "x3", "y3", "z3"])
        micro = art.micro_labels if art.micro_labels is not None else [None] * len(art.doc_ids)
        for i, doc_id in enumerate(art.doc_ids):
            writer.writerow([
                doc_id,
                int(art.labels[i]),
                "" if micro[i] is None else int(micro[i]),
                *(round(float(v), 5) for v in art.coords_2d[i]),
                *(round(float(v), 5) for v in art.coords_3d[i]),
            ])

    clusters_payload = {str(cid): asdict(info) for cid, info in art.clusters.items()}
    (run_dir / "clusters.json").write_text(json.dumps(clusters_payload, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(art.metrics, indent=2, default=str))
    return run_dir


def load_run(run_dir: Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    doc_ids, labels, micro, c2, c3 = [], [], [], [], []
    has_micro = False
    with open(run_dir / "assignments.csv") as f:
        for row in csv.DictReader(f):
            doc_ids.append(row["doc_id"])
            labels.append(int(row["cluster_id"]))
            if row["micro_cluster_id"] != "":
                has_micro = True
                micro.append(int(row["micro_cluster_id"]))
            else:
                micro.append(-1)
            c2.append([float(row["x2"]), float(row["y2"])])
            c3.append([float(row["x3"]), float(row["y3"]), float(row["z3"])])

    clusters_raw = json.loads((run_dir / "clusters.json").read_text())
    clusters = {int(cid): ClusterInfo(**info) for cid, info in clusters_raw.items()}
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    return RunArtifacts(
        run_name=manifest.get("run_name", run_dir.name),
        manifest=manifest,
        doc_ids=doc_ids,
        labels=np.array(labels, dtype=int),
        coords_2d=np.array(c2),
        coords_3d=np.array(c3),
        clusters=clusters,
        metrics=metrics,
        micro_labels=np.array(micro, dtype=int) if has_micro else None,
    )


def run_is_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).exists()
        for name in ("manifest.json", "assignments.csv", "clusters.json", "metrics.json")
    )
