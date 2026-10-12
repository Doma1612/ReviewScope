from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClusterBackend(Protocol):
    algorithm: str

    @property
    def params(self) -> dict[str, Any]: ...

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray: ...


def scaled_min_cluster_size(
    n_units: int, fraction: float = 0.003, floor: int = 15
) -> int:
    # The 0.3% size rule extrapolates the 5k benchmark; it remains a tunable heuristic.
    if n_units <= 0:
        return floor
    return max(floor, round(fraction * n_units))


def scaled_min_samples(
    min_cluster_size: int, ratio: int = 3, cap: int = 50, floor: int = 5
) -> int:
    # Cap smoothing: scaling min_samples with corpus size collapsed the EuroParl run.
    return max(floor, min(cap, min_cluster_size // ratio))


def params_slug(params: dict[str, Any]) -> str:
    parts = []
    for k in sorted(params):
        v = params[k]
        v = str(v).replace(".", "p") if isinstance(v, float) else str(v)
        parts.append(f"{k}{v}")
    return "__".join(parts)
