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
    label_source: str
    top_terms: list[list]
    tfidf_terms: list[list]
    word_frequencies: dict[str, int]
    sample_doc_ids: list[str]               # Random members, not centroid-nearest exemplars.
    mean_stars: Optional[float] = None
    prompt_hash: Optional[str] = None
    micro_cluster_ids: list[int] = field(default_factory=list)
    # size counts mentions; n_documents counts distinct parent reviews.
    n_documents: Optional[int] = None
    sentiment_avg: Optional[float] = None
    sentiment_dist: Optional[dict[str, float]] = None


@dataclass
class RunArtifacts:
    run_name: str
    manifest: dict[str, Any]
    doc_ids: list[str]
    labels: np.ndarray
    coords_2d: np.ndarray
    coords_3d: np.ndarray
    clusters: dict[int, ClusterInfo]
    metrics: dict[str, Any]
    micro_labels: Optional[np.ndarray] = None
    sentiment_scores: Optional[np.ndarray] = None
    sentiment_labels: Optional[list[str]] = None

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
        writer.writerow([
            "doc_id", "cluster_id", "micro_cluster_id",
            "x2", "y2", "x3", "y3", "z3",
            "sentiment_score", "sentiment_label",
        ])
        micro = art.micro_labels if art.micro_labels is not None else [None] * len(art.doc_ids)
        has_sentiment = art.sentiment_scores is not None
        for i, doc_id in enumerate(art.doc_ids):
            writer.writerow([
                doc_id,
                int(art.labels[i]),
                "" if micro[i] is None else int(micro[i]),
                *(round(float(v), 5) for v in art.coords_2d[i]),
                *(round(float(v), 5) for v in art.coords_3d[i]),
                round(float(art.sentiment_scores[i]), 4) if has_sentiment else "",
                art.sentiment_labels[i] if has_sentiment else "",
            ])

    clusters_payload = {str(cid): asdict(info) for cid, info in art.clusters.items()}
    (run_dir / "clusters.json").write_text(
        json.dumps(clusters_payload, indent=2, default=_json_default)
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(art.metrics, indent=2, default=_json_default)
    )
    return run_dir


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return str(obj)


def load_run(run_dir: Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    doc_ids, labels, micro, c2, c3 = [], [], [], [], []
    sent_scores, sent_labels = [], []
    has_micro = False
    has_sentiment = False
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
            if row.get("sentiment_score"):
                has_sentiment = True
                sent_scores.append(float(row["sentiment_score"]))
                sent_labels.append(row.get("sentiment_label", ""))
            else:
                sent_scores.append(float("nan"))
                sent_labels.append("")

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
        sentiment_scores=np.array(sent_scores, dtype=np.float32) if has_sentiment else None,
        sentiment_labels=sent_labels if has_sentiment else None,
    )


def run_is_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).exists()
        for name in ("manifest.json", "assignments.csv", "clusters.json", "metrics.json")
    )
