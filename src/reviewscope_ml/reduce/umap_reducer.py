"""
Dimensionality reduction stage.

Parameters default to what notebook 05 selected on the 5k hotel benchmark:
UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine") for the
clustering input, and a separate 2-D/3-D projection (min_dist=0.1) for the
scatter plots — visual spread and clustering input have different needs, so
they are different projections on purpose.

Determinism (WP9b goal 1): every UMAP gets ``random_state=seed``. That forces
single-threaded layout optimisation — slower, but the same corpus + seed +
parameters then reproduce the same projection bit-for-bit, which is a
prerequisite for "same corpus -> same clusters". The residual caveat (UMAP is
deterministic per seed but unstable ACROSS seeds) is what the eval harness's
multi-seed ARI stability check quantifies.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("reviewscope.reduce")


@dataclass
class UMAPReducer:
    """UMAP, optionally with a PCA pre-reduction step (notebook 05 exp. 4).

    PCA->UMAP exists because UMAP's nearest-neighbour search degrades and
    slows down in very high dimensional spaces (1024-d+ embeddings); PCA(50)
    keeps ~90% of the variance and makes the UMAP step cheaper. On 768-d
    embeddings notebook 05 found plain UMAP sufficient.
    """

    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"
    seed: int = 42
    pca_components: Optional[int] = None  # e.g. 50 for PCA->UMAP

    @property
    def method(self) -> str:
        return "pca_umap" if self.pca_components else "umap"

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        import umap

        x = embeddings
        t0 = time.time()
        if self.pca_components:
            from sklearn.decomposition import PCA

            x = PCA(n_components=self.pca_components, random_state=self.seed).fit_transform(x)
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            # after PCA the space is dense + decorrelated; euclidean matches nb 05
            metric="euclidean" if self.pca_components else self.metric,
            random_state=self.seed,
        )
        reduced = reducer.fit_transform(x)
        logger.info(
            "%s -> %dd in %.1fs (n=%d)",
            self.method, self.n_components, time.time() - t0, len(reduced),
        )
        return np.asarray(reduced)


def viz_projection(
    embeddings: np.ndarray, n_components: int, n_neighbors: int = 15, seed: int = 42
) -> np.ndarray:
    """2-D/3-D scatter coordinates (min_dist=0.1 spreads points for reading)."""
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(embeddings))
