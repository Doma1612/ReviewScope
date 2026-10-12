from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HDBSCANBackend:

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
