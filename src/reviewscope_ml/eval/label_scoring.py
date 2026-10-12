from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def select_exemplars(
    labels: Any, texts: list[str], chosen: list[int], seed: int, k: int = 5
) -> dict[int, list[str]]:
    # Both sheet and GUI must use identical cluster order and seed for exemplars.
    import numpy as np

    rng = np.random.default_rng(seed)
    out: dict[int, list[str]] = {}
    for cid in chosen:
        idx = np.flatnonzero(labels == cid)
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        out[cid] = [texts[i][:300] for i in pick]
    return out


@dataclass
class ScoringItem:

    cluster_id: int
    label: str
    produced_by: list[tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return label_key(self.label)


def load_label_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def build_scoring_items(
    records: Iterable[dict[str, Any]], seed: int = 42
) -> dict[int, list[ScoringItem]]:
    by_cluster: dict[int, dict[str, ScoringItem]] = defaultdict(dict)
    for rec in records:
        label = str(rec.get("label") or "").strip()
        if not label or label == "—":
            continue
        cid = int(rec["cluster_id"])
        key = label_key(label)
        item = by_cluster[cid].get(key)
        if item is None:
            item = ScoringItem(cluster_id=cid, label=label)
            by_cluster[cid][key] = item
        item.produced_by.append(
            (str(rec.get("labeler", "?")), str(rec.get("prompt_variant", "?")))
        )

    out: dict[int, list[ScoringItem]] = {}
    for cid, items in by_cluster.items():
        ordered = sorted(items.values(), key=lambda i: i.key)
        random.Random(f"{seed}:{cid}").shuffle(ordered)
        for item in ordered:
            item.produced_by.sort()
        out[cid] = ordered
    return out


def collect_scores(records: Iterable[Any]) -> dict[tuple[int, str], dict[str, int]]:
    # Feedback is append-only; the latest score per reviewer wins.
    out: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    for rec in records:
        if getattr(rec, "action", None) != "score_label":
            continue
        if rec.cluster_id is None or not rec.label or rec.score is None:
            continue
        out[(int(rec.cluster_id), label_key(rec.label))][rec.reviewer] = int(rec.score)
    return dict(out)


def score_by_labeler(
    items: dict[int, list[ScoringItem]],
    scores: dict[tuple[int, str], dict[str, int]],
) -> list[dict[str, Any]]:
    # Exclude unscored clusters rather than treating missing judgements as zero.
    per_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cid, cluster_items in items.items():
        for item in cluster_items:
            reviewer_scores = scores.get((cid, item.key))
            if not reviewer_scores:
                continue
            mean = sum(reviewer_scores.values()) / len(reviewer_scores)
            for model in item.produced_by:
                per_model[model].append(mean)

    rows = [
        {
            "labeler": labeler,
            "prompt": prompt,
            "human_score": sum(vals) / len(vals),
            "n_clusters": len(vals),
        }
        for (labeler, prompt), vals in per_model.items()
        if vals
    ]
    rows.sort(key=lambda r: (-r["human_score"], r["labeler"]))
    return rows


def coverage(
    items: dict[int, list[ScoringItem]],
    scores: dict[tuple[int, str], dict[str, int]],
    reviewer: Optional[str] = None,
) -> tuple[int, int]:
    total = sum(len(v) for v in items.values())
    done = 0
    for cid, cluster_items in items.items():
        for item in cluster_items:
            got = scores.get((cid, item.key), {})
            if (reviewer in got) if reviewer is not None else bool(got):
                done += 1
    return done, total


def pairwise_agreement(
    scores: dict[tuple[int, str], dict[str, int]],
) -> list[dict[str, Any]]:
    reviewers = sorted({r for per in scores.values() for r in per})
    rows = []
    for i, a in enumerate(reviewers):
        for b in reviewers[i + 1:]:
            diffs = [
                abs(per[a] - per[b])
                for per in scores.values()
                if a in per and b in per
            ]
            if not diffs:
                continue
            rows.append({
                "a": a,
                "b": b,
                "n": len(diffs),
                "exact": sum(d == 0 for d in diffs) / len(diffs),
                "within_1": sum(d <= 1 for d in diffs) / len(diffs),
                "mean_abs_diff": sum(diffs) / len(diffs),
            })
    return rows


def krippendorff_alpha(
    scores: dict[tuple[int, str], dict[str, int]],
) -> Optional[float]:
    units = [list(per.values()) for per in scores.values() if len(per) >= 2]
    if not units:
        return None

    # Weight by 1/(m-1) so heavily reviewed items do not dominate agreement.
    coincidence: dict[tuple[int, int], float] = defaultdict(float)
    for ratings in units:
        m = len(ratings)
        for idx, c in enumerate(ratings):
            for jdx, k in enumerate(ratings):
                if idx != jdx:
                    coincidence[(c, k)] += 1.0 / (m - 1)

    values = sorted({v for pair in coincidence for v in pair})
    n_v = {v: sum(coincidence[(v, k)] for k in values) for v in values}
    n_total = sum(n_v.values())
    if n_total <= 1:
        return None

    def delta(c: int, k: int) -> float:
        lo, hi = (c, k) if c <= k else (k, c)
        between = sum(n_v[v] for v in values if lo <= v <= hi)
        return (between - (n_v[lo] + n_v[hi]) / 2.0) ** 2

    d_obs = sum(coincidence[(c, k)] * delta(c, k) for c in values for k in values)
    d_exp = sum(
        (n_v[c] * n_v[k] / (n_total - 1) if c != k
         else n_v[c] * (n_v[c] - 1) / (n_total - 1)) * delta(c, k)
        for c in values
        for k in values
    )
    if math.isclose(d_exp, 0.0):
        # Constant ratings make reliability undefined, not perfect agreement.
        return None
    return 1.0 - d_obs / d_exp
