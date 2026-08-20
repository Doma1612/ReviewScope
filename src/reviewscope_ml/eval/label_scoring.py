"""
Human scoring of LLM cluster labels: task construction and agreement stats.

The automatic labeler metrics in ``label_quality.py`` measure *form* — does the
label parse, does it sit nearest its own centroid, are its content words present
in member mentions. None of them measures whether a label tells a reader what
the cluster is about, which is the only thing the label exists to do. A model
can post a perfect format-ok rate while emitting "Guest Experience Reviews" for
every cluster.

That judgement lives in a person, so this module builds the task a person
actually performs and then reports how much the people agreed. Everything here
is pure: the Streamlit view in ``hitl/app.py`` renders it, the tests exercise it
without a browser, and the numbers can be regenerated from the feedback JSONL
alone.

Two design decisions are load-bearing:

**Labels are scored blind.** The markdown sheet this replaces put the labeler's
name in the same row as its label, which invites scoring the reputation instead
of the words. Here the labeler is hidden until scoring is finished and the
presentation order is shuffled per cluster, so a model cannot benefit from
being read first or from being recognised.

**Identical labels are one judgement.** Several labelers frequently emit the
same string for a cluster. Scoring that string once and attributing the score to
every model that produced it removes ~9% of the work and, more importantly,
makes it impossible to award the same words different scores.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def label_key(label: str) -> str:
    """Identity of a label for scoring: case- and whitespace-insensitive.

    'Casino in Reno' and 'casino in reno' are the same judgement; treating them
    as two would let a reviewer score them differently and would split one
    model's credit across two rows.
    """
    return " ".join(label.split()).casefold()


def select_exemplars(
    labels: Any, texts: list[str], chosen: list[int], seed: int, k: int = 5
) -> dict[int, list[str]]:
    """Random member mentions per cluster — the sheet's exemplars, exactly.

    RANDOM members, not centroid-nearest: centroid samples flatter the cluster
    (methodology §8) and hide the fringe that reveals a bad label. The reviewer
    must see what the labeler did not.

    Both the generated markdown sheet and the scoring GUI call this, because a
    reviewer scoring a label against different text than the sheet shows would
    make the two records silently incomparable. `chosen` must be in the same
    order in both callers — the generator is advanced once per cluster.
    """
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
    """One label to be scored, and the models that produced it."""

    cluster_id: int
    label: str
    produced_by: list[tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return label_key(self.label)


def load_label_records(path: Path) -> list[dict[str, Any]]:
    """The per-label rows written by ``label_quality.py`` (``*.json``)."""
    return json.loads(Path(path).read_text())


def build_scoring_items(
    records: Iterable[dict[str, Any]], seed: int = 42
) -> dict[int, list[ScoringItem]]:
    """Cluster id -> blind-shuffled, deduplicated labels to score.

    The shuffle is seeded so a reviewer who reloads the page sees the same order
    and does not re-score a differently-ordered list; two reviewers on the same
    report also see the same order, which keeps any order effect common to both
    rather than a difference between them.
    """
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
        ordered = sorted(items.values(), key=lambda i: i.key)  # stable pre-shuffle
        random.Random(f"{seed}:{cid}").shuffle(ordered)
        for item in ordered:
            item.produced_by.sort()
        out[cid] = ordered
    return out


def collect_scores(records: Iterable[Any]) -> dict[tuple[int, str], dict[str, int]]:
    """Feedback records -> {(cluster_id, label_key): {reviewer: score}}.

    Records are append-only and a reviewer may score the same label twice after
    changing their mind, so the *last* score from each reviewer wins — matching
    ``load_feedback``'s "later records win" rule.
    """
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
    """Per (labeler, prompt) human results, best mean first.

    A model's score on a cluster is the mean across reviewers of the label it
    produced there; its overall score is the mean across the clusters that have
    been scored at all. Unscored clusters are excluded rather than counted as
    zero, so a half-finished sheet reports a smaller-n number instead of a
    wrong one — ``n_clusters`` carries that caveat with the number.
    """
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
    """(scored, total) label judgements — for one reviewer, or for anyone."""
    total = sum(len(v) for v in items.values())
    done = 0
    for cid, cluster_items in items.items():
        for item in cluster_items:
            got = scores.get((cid, item.key), {})
            if (reviewer in got) if reviewer is not None else bool(got):
                done += 1
    return done, total


# ── agreement ────────────────────────────────────────────────────────────────
#
# Reported because a single reviewer's sheet is one person's opinion, and a
# group that never measures its disagreement cannot tell a shared standard from
# a shared blind spot.


def pairwise_agreement(
    scores: dict[tuple[int, str], dict[str, int]],
) -> list[dict[str, Any]]:
    """Per reviewer pair: overlap, exact and within-1 agreement, mean |diff|."""
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
    """Ordinal Krippendorff's alpha, or None if nothing is doubly scored.

    Chosen over Cohen's kappa because this is exactly the awkward case kappa
    cannot handle: more than two reviewers, each scoring an arbitrary and
    incomplete subset. Alpha handles missing judgements natively and treats the
    1-5 scale as ordered, so scoring a label 4 when a colleague said 5 counts as
    near-agreement rather than as being simply wrong.

    Read it as: 1.0 identical, ~0.8 conventionally "reliable", 0.0 no better
    than chance, negative systematic disagreement.
    """
    units = [list(per.values()) for per in scores.values() if len(per) >= 2]
    if not units:
        return None

    # Coincidence matrix: each unit contributes its ordered rating pairs,
    # weighted 1/(m-1) so units rated by many people do not dominate.
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
        """Ordinal distance: squared gap in cumulative rank mass."""
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
        # Every reviewer used a single value throughout: perfectly consistent,
        # but the statistic is undefined rather than 1.0 — say so.
        return None
    return 1.0 - d_obs / d_exp
