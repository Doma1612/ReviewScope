from __future__ import annotations

import math
from statistics import fmean


def cohesion_score(vectors: list[list[float]] | list) -> float | None:
    vecs = [list(v) for v in vectors if v]
    if len(vecs) < 2:
        return None
    dim = len(vecs[0])
    # Zero vectors have no cosine direction and would skew the centroid.
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
