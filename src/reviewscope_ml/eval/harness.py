"""
Quantitative evaluation harness.

Extends the notebooks' three-tier framework (geometric / coherence / rating
entropy — see ``core.metrics`` for the rationale) with what the notebooks
could not measure:

- **Noise-handling fairness.** HDBSCAN discards noise points, and silhouette
  is computed on the survivors — which structurally flatters noise-discarding
  algorithms over partitioners that must own every point. We therefore report
  silhouette both excluding noise (the classic number) and *including* noise
  as its own pseudo-cluster, plus the noise fraction itself, and the report
  must always read them together.
- **Stability** (WP9b): Adjusted Rand Index of cluster assignments across
  seeds. UMAP is deterministic per seed but not across seeds; if a config's
  clusters reshuffle whenever the seed changes, "same corpus -> same
  clusters" is unattainable for it.
- **Failure-mode flags**: cheap structural detectors for the classic ways
  review clustering goes wrong, surfaced per run so a human reads them next
  to the metrics.

None of this declares a winner. Metrics shortlist finalists; the qualitative
inspection (``eval.inspection``) and a human decide.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Optional

import numpy as np

from ..core.metrics import compute_coherence, compute_metrics, compute_rating_entropy


def evaluate_labels(
    reduced: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    stars: Optional[np.ndarray],
    runtime_s: float = 0.0,
    compute_coh: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Full three-tier metrics dict for one labeling, noise-fair variants included."""
    out = compute_metrics(reduced, labels, runtime_s=runtime_s, seed=seed)

    # Noise-handling fairness: score the clustering as if noise were a cluster
    # of its own. For partitioners (no -1) the two silhouettes are identical.
    out["silhouette_incl_noise"] = _silhouette_incl_noise(reduced, labels, seed=seed)

    out["coherence_cv"] = compute_coherence(texts, labels) if compute_coh else None
    out["rating_entropy"] = (
        compute_rating_entropy(stars, labels) if stars is not None else None
    )

    # Structural stats for the failure-mode flags.
    valid = labels[labels != -1]
    if len(valid):
        sizes = np.bincount(valid)
        sizes = sizes[sizes > 0]
        out["max_cluster_share"] = round(float(sizes.max()) / len(labels), 4)
        out["median_cluster_size"] = int(np.median(sizes))
    else:
        out["max_cluster_share"] = None
        out["median_cluster_size"] = None
    return out


def _silhouette_incl_noise(
    reduced: np.ndarray, labels: np.ndarray, seed: int, sample_cap: int = 5_000
) -> Optional[float]:
    from sklearn.metrics import silhouette_score

    if len(set(labels)) < 2:
        return None
    n = len(labels)
    if n > sample_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_cap, replace=False)
        reduced, labels = reduced[idx], labels[idx]
    try:
        return round(float(silhouette_score(reduced, labels)), 4)
    except Exception:
        return None


def stability_ari(label_runs: list[np.ndarray]) -> dict[str, Any]:
    """
    Pairwise Adjusted Rand Index across >=2 label arrays from different seeds.

    ARI compares partitions while ignoring label permutations; 1.0 = identical
    clusterings, ~0 = random agreement. Noise (-1) is treated as a label of
    its own, so unstable noise assignment lowers the score too — intentional,
    because to the app a document flapping between "noise" and "topic 3"
    across runs IS instability.
    """
    from sklearn.metrics import adjusted_rand_score

    if len(label_runs) < 2:
        return {"ari_mean": None, "ari_min": None, "n_runs": len(label_runs)}
    scores = [
        adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)
    ]
    return {
        "ari_mean": round(float(np.mean(scores)), 4),
        "ari_min": round(float(np.min(scores)), 4),
        "ari_pairwise": [round(float(s), 4) for s in scores],
        "n_runs": len(label_runs),
    }


# Thresholds for the structural failure-mode flags. Heuristics, not truths —
# they exist to direct the human eye, and the inspection artifact has the
# final say.
GIANT_CLUSTER_SHARE = 0.50
HIGH_NOISE_RATIO = 0.40
SENTIMENT_ENTROPY_FLOOR = 0.60
DUPLICATE_TERM_OVERLAP = 0.6


def failure_flags(
    metrics: dict[str, Any],
    cluster_terms: Optional[dict[int, list[tuple[str, float]]]] = None,
) -> list[str]:
    """Human-readable warnings for the classic review-clustering failure modes."""
    flags: list[str] = []

    share = metrics.get("max_cluster_share")
    if share is not None and share > GIANT_CLUSTER_SHARE:
        flags.append(
            f"giant cluster: one cluster holds {share:.0%} of all documents "
            "(blob + crumbs pattern)"
        )
    noise = metrics.get("noise_ratio")
    if noise is not None and noise > HIGH_NOISE_RATIO:
        flags.append(
            f"high noise: {noise:.0%} of documents discarded as noise — "
            "silhouette is computed on the easy remainder"
        )
    entropy = metrics.get("rating_entropy")
    if entropy is not None and entropy < SENTIMENT_ENTROPY_FLOOR:
        flags.append(
            f"sentiment blobs: rating entropy {entropy:.2f} < {SENTIMENT_ENTROPY_FLOOR} — "
            "clusters separate star ratings, not topics"
        )
    n_clusters = metrics.get("n_clusters")
    if n_clusters is not None and n_clusters < 3:
        flags.append(f"only {n_clusters} clusters — no usable topic structure")

    if cluster_terms:
        for a, b in combinations(sorted(cluster_terms), 2):
            terms_a = {w for w, _ in cluster_terms[a][:10]}
            terms_b = {w for w, _ in cluster_terms[b][:10]}
            if not terms_a or not terms_b:
                continue
            overlap = len(terms_a & terms_b) / min(len(terms_a), len(terms_b))
            if overlap >= DUPLICATE_TERM_OVERLAP:
                flags.append(
                    f"near-duplicate clusters {a} and {b}: "
                    f"{overlap:.0%} top-term overlap — candidates for merging"
                )
    return flags
