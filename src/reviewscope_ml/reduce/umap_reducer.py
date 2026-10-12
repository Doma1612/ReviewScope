from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("reviewscope.reduce")


@dataclass
class UMAPReducer:

    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"
    seed: int = 42
    pca_components: Optional[int] = None

    @property
    def method(self) -> str:
        return "pca_umap" if self.pca_components else "umap"

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        import umap

        x = embeddings
        logger.info(
            "fitting %s: %d docs %dd -> %dd (nn=%d, seeded/single-threaded)",
            self.method, len(x), x.shape[1], self.n_components, self.n_neighbors,
        )
        t0 = time.time()
        if self.pca_components:
            from sklearn.decomposition import PCA

            x = PCA(n_components=self.pca_components, random_state=self.seed).fit_transform(x)
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            # PCA decorrelates the space, so use Euclidean distance afterward.
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
    import umap

    logger.info("fitting viz projection: %d docs -> %dd", len(embeddings), n_components)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(embeddings))
