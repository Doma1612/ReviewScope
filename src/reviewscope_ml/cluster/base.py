"""
Clustering stage interface.

Every backend maps a reduced (n, d) array to integer labels, -1 meaning noise
(HDBSCAN convention; partitioning algorithms simply never emit -1). ``params``
feeds both the cache filename slug and the results log, so two backends with
the same params hash to the same experiment identity.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClusterBackend(Protocol):
    #: short algorithm id used in cache paths and the results CSV
    algorithm: str

    @property
    def params(self) -> dict[str, Any]: ...

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray: ...


def scaled_min_cluster_size(
    n_units: int, fraction: float = 0.003, floor: int = 15
) -> int:
    """
    Scale HDBSCAN's ``min_cluster_size`` with corpus size.

    Why: ``min_cluster_size`` is an *absolute* count, but its meaning is
    relative — "a theme smaller than this share of the corpus is noise".
    Notebook 06 chose mcs=15 on the 5k benchmark; kept absolute at 50k that
    would call a 15-document group a topic (0.03% of the corpus) and is the
    direct cause of the blob-and-crumbs failure observed in smoke runs.

    The default fraction 0.003 is *anchored to the notebook decision*
    (15/5000), not independently validated — it preserves the decided
    behaviour at the benchmark scale and extrapolates it. The floor keeps
    tiny smoke samples meaningful. Treat the fraction as a tunable, not a
    truth; the Pareto-tuning proposal in docs/pipeline-guide.md is the
    proper follow-up.
    """
    if n_units <= 0:
        return floor
    return max(floor, round(fraction * n_units))


def scaled_min_samples(
    min_cluster_size: int, ratio: int = 3, cap: int = 50, floor: int = 5
) -> int:
    """HDBSCAN's ``min_samples`` from ``min_cluster_size`` — ratio, then capped.

    Why the cap: ``min_samples`` is the *smoothing* parameter, not a size
    parameter. It sets each point's core distance as the distance to its
    ``min_samples``-th neighbour, so raising it flattens exactly the local
    density variation HDBSCAN needs to separate modes. Notebook 06's 15:5
    choice fixed the 3:1 ratio at benchmark scale, where the absolute value
    stays small; extrapolating the ratio alongside a corpus-scaled
    ``min_cluster_size`` eventually smooths the space into one component.

    Measured on the EuroParl 26k sentence run (352,924 segments, mcs=1059):
    ms=25 -> 62 clusters, ms=50 -> 62, ms=100 -> 55, and the uncapped
    ms=353 -> **2 clusters holding 98% of all segments**. The cap of 50 is
    the largest value that leaves every previously validated configuration
    untouched (5k docs -> ms=5, 50k docs -> ms=50) while clamping the
    extrapolation beyond them. Between 100 and 353 the behaviour is a cliff
    we have not mapped; 50 sits well clear of it.
    """
    return max(floor, min(cap, min_cluster_size // ratio))


def params_slug(params: dict[str, Any]) -> str:
    """Deterministic filesystem-safe slug, e.g. {'mcs':15,'ms':5} -> 'mcs15__ms5'."""
    parts = []
    for k in sorted(params):
        v = params[k]
        v = str(v).replace(".", "p") if isinstance(v, float) else str(v)
        parts.append(f"{k}{v}")
    return "__".join(parts)
