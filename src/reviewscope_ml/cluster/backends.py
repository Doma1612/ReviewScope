"""
Flat clustering backends: HDBSCAN, KMeans, Agglomerative.

Default parameters are the winners of notebook 06's sweeps on the 5k hotel
benchmark (HDBSCAN mcs=15/ms=5, KMeans k=15, Agglomerative k=15/ward); they
remain constructor arguments because the eval harness re-sweeps them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HDBSCANBackend:
    """Primary candidate: no fixed k, variable density, explicit noise label.

    The noise label is honest (forcing outliers into clusters pollutes them)
    but also flatters geometric metrics — the eval harness therefore reports
    metrics with and without the noise points.
    """

    min_cluster_size: int = 15
    min_samples: int = 5
    algorithm = "hdbscan"

    @property
    def params(self) -> dict[str, Any]:
        return {"mcs": self.min_cluster_size, "ms": self.min_samples}

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
        ).fit_predict(reduced)


@dataclass
class KMeansBackend:
    """Baseline partitioner: fixed k, no noise concept, fully deterministic per seed."""

    k: int = 15
    seed: int = 42
    algorithm = "kmeans"

    @property
    def params(self) -> dict[str, Any]:
        return {"k": self.k}

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=self.k, random_state=self.seed, n_init="auto").fit_predict(
            reduced
        )


@dataclass
class AgglomerativeBackend:
    """Flat cut of a hierarchical (ward) tree — gives topic-tree potential later."""

    k: int = 15
    linkage: str = "ward"
    algorithm = "agglomerative"

    @property
    def params(self) -> dict[str, Any]:
        return {"k": self.k, "linkage": self.linkage}

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=self.k, linkage=self.linkage).fit_predict(
            reduced
        )
