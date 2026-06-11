"""
Two-stage micro->macro clustering.

Motivation: a single HDBSCAN pass at "topic" granularity tends to produce one
giant blob plus crumbs on review data, because review topics differ wildly in
density. Splitting the problem helps:

1. **Micro pass** — HDBSCAN with a small ``min_cluster_size`` finds many
   fine-grained, high-purity micro-clusters (specific complaints, specific
   praises). Purity is cheap at this scale; interpretability is not required
   yet.
2. **Macro pass** — agglomerative (ward) clustering on the micro-cluster
   centroids merges micro-clusters into human-sized macro topics. Merging
   centroids instead of documents means the macro step is tiny (hundreds of
   points, not thousands) and respects the micro structure.

The micro->macro mapping is preserved on the instance (``micro_labels_``,
``micro_to_macro_``) because it is exactly the lever the HITL review needs:
a reviewer's "split this macro cluster" can be answered by promoting its
micro-clusters instead of re-clustering from scratch, and WP9b's incremental
updates can assign new documents to the nearest micro centroid.
"""
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
    n_macro: Optional[int] = None     # None -> ~sqrt(n_micro), bounded to [5, 30]
    linkage: str = "ward"
    algorithm = "two_stage"

    # Fitted state (populated by fit_predict)
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
            # Degenerate corpus (e.g. tiny smoke sample): nothing to merge.
            logger.warning("two-stage: only %d micro-clusters; skipping macro pass", len(micro_ids))
            self.micro_labels_ = micro
            self.micro_to_macro_ = {m: m for m in micro_ids}
            return micro

        centroids = np.vstack([reduced[micro == m].mean(axis=0) for m in micro_ids])

        n_macro = self.n_macro
        if n_macro is None:
            # sqrt heuristic: ~100 micro-clusters -> ~10 macro topics, the
            # granularity a human can actually review. Bounded so tiny or huge
            # micro counts still land in a readable range.
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
