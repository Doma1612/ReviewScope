from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("reviewscope.cluster")


@dataclass
class TwoStageBackend:
    micro_min_cluster_size: int = 5
    micro_min_samples: int = 3
    n_macro: Optional[int] = None
    linkage: str = "ward"
    algorithm = "two_stage"

    micro_labels_: Optional[np.ndarray] = field(default=None, repr=False)
    micro_to_macro_: Optional[dict[int, int]] = field(default=None, repr=False)
    micro_centroids_: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "mmcs": self.micro_min_cluster_size,
            "mms": self.micro_min_samples,
            "nmac": self.n_macro if self.n_macro is not None else "auto",
            "linkage": self.linkage,
        }

    def fit_predict(self, reduced: np.ndarray) -> np.ndarray:
        import hdbscan
        from sklearn.cluster import AgglomerativeClustering

        micro = hdbscan.HDBSCAN(
            min_cluster_size=self.micro_min_cluster_size,
            min_samples=self.micro_min_samples,
        ).fit_predict(reduced)
        micro_ids = sorted(int(c) for c in set(micro) if c != -1)

        if len(micro_ids) < 2:
            logger.warning("two-stage: only %d micro-clusters; skipping macro pass", len(micro_ids))
            self.micro_labels_ = micro
            self.micro_to_macro_ = {m: m for m in micro_ids}
            return micro

        centroids = np.vstack([reduced[micro == m].mean(axis=0) for m in micro_ids])

        n_macro = self.n_macro
        if n_macro is None:
            n_macro = int(np.clip(round(np.sqrt(len(micro_ids))), 5, 30))
        n_macro = min(n_macro, len(micro_ids))

        macro_of_centroid = AgglomerativeClustering(
            n_clusters=n_macro, linkage=self.linkage
        ).fit_predict(centroids)

        micro_to_macro = {m: int(g) for m, g in zip(micro_ids, macro_of_centroid)}
        labels = np.array([micro_to_macro.get(m, -1) for m in micro], dtype=int)

        self.micro_labels_ = micro
        self.micro_to_macro_ = micro_to_macro
        self.micro_centroids_ = centroids
        logger.info(
            "two-stage: %d micro-clusters -> %d macro topics (noise %.1f%%)",
            len(micro_ids), n_macro, 100 * float(np.mean(micro == -1)),
        )
        return labels
