"""Cluster cohesion metric (pure, no DB / no ML deps).

Cohesion answers "how tight / trustworthy is this cluster?" — the higher the value
the more its members agree. We use the mean cosine similarity of each member's
embedding vector to the cluster centroid, which lands in ``[-1, 1]`` (in practice
``[0, 1]`` for sane embeddings).

It is deliberately a *per-cluster* metric and pure Python — not the ML package's
silhouette. That's a design choice, not a dependency limitation: silhouette is a
*whole-partition* score (each point relative to its nearest other cluster), so it
can't be recomputed for a single edited cluster the way our incremental recompute
works, and it would need every project's points loaded at once on every edit.
Cohesion-to-centroid is cheap, incremental, and runs identically in simulated and
real mode (real mode has the full sklearn/numpy/gensim stack; simulated mode does
not — so keeping this free of heavy deps means it never branches on the mode). The
run-level silhouette/coherence/rating-entropy the pipeline already computes
(``RunResult.metrics``) is a separate, project-level concern — see the plan's
follow-up for surfacing those.
"""
from __future__ import annotations

import math
from statistics import fmean


def cohesion_score(vectors: list[list[float]] | list) -> float | None:
    """Mean cosine similarity of member vectors to their centroid.

    Returns ``None`` when cohesion is undefined: fewer than two usable vectors
    (a singleton or empty cluster), or a degenerate (zero-norm) centroid. Empty or
    zero-norm member vectors are skipped rather than poisoning the mean.
    """
    vecs = [list(v) for v in vectors if v]
    if len(vecs) < 2:
        return None
    dim = len(vecs[0])
    # Keep only well-formed, non-zero vectors: a zero vector has no direction to be
    # cosine-compared, so it's dropped before it can skew the centroid or the mean.
    vecs = [v for v in vecs if len(v) == dim and any(x != 0 for x in v)]
    if len(vecs) < 2:
        return None

    centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    centroid_norm = math.sqrt(sum(c * c for c in centroid))
    if centroid_norm == 0:
        return None

    sims: list[float] = []
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v))
        dot = sum(v[i] * centroid[i] for i in range(dim))
        sims.append(dot / (norm * centroid_norm))
    return fmean(sims) if sims else None
