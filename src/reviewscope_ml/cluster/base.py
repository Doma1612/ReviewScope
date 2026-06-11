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


def params_slug(params: dict[str, Any]) -> str:
    """Deterministic filesystem-safe slug, e.g. {'mcs':15,'ms':5} -> 'mcs15__ms5'."""
    parts = []
    for k in sorted(params):
        v = params[k]
        v = str(v).replace(".", "p") if isinstance(v, float) else str(v)
        parts.append(f"{k}{v}")
    return "__".join(parts)
