"""
Three-tier clustering evaluation metrics.

Tier 1 — Geometric (silhouette, Davies-Bouldin, Calinski-Harabasz):
  Measured in UMAP/PCA space.  Fast, always available, but geometry-circular:
  instruction models mechanically improve silhouette by reshaping the space,
  and silhouette rewards fewer large blobs (often sentiment polarity rather
  than thematic distinction).

Tier 2 — Topic coherence (C_v via Gensim):
  Measures whether the top words of each cluster co-occur in the corpus.
  Computed from raw text, independent of embedding geometry, so it can
  validate or contradict silhouette.  Requires gensim (optional dep).

Tier 3 — Rating entropy:
  Hotel reviews carry 1–5 star ratings.  A thematic cluster (e.g. "breakfast")
  should attract reviews of all rating levels → high entropy.  A sentiment
  cluster (e.g. "5-star enthusiasts") collapses to one rating level → low
  entropy.  This distinguishes meaningful topic separation from polarity
  separation, which is not useful for our NLP analysis goal.

All notebooks call compute_metrics() so results are always comparable.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_metrics(
    reduced: np.ndarray,
    labels: np.ndarray,
    runtime_s: float,
    silhouette_sample: int = 5_000,
    seed: int = 42,
) -> dict:
    """
    Compute a standard set of clustering quality metrics.

    Parameters
    ----------
    reduced : np.ndarray, shape (n_docs, n_dims)
        The dimensionality-reduced representation used for clustering
        (UMAP or PCA output — NOT the raw embedding space).
    labels : np.ndarray, shape (n_docs,)
        Cluster label per document.  -1 means noise (HDBSCAN convention).
        KMeans / Agglomerative produce no -1 labels.
    runtime_s : float
        Wall-clock time in seconds for the step that produced these labels
        (embed + reduce + cluster, or whichever stages were timed).
    silhouette_sample : int
        Sub-sample cap for silhouette_score (O(n²) memory).
    seed : int
        Random seed for sub-sampling.

    Returns
    -------
    dict with keys:
        n_docs, n_clusters, noise_count, noise_ratio,
        silhouette, davies_bouldin, calinski_harabasz, runtime_s
    """
    n_docs = len(labels)
    noise_mask = labels == -1
    noise_count = int(noise_mask.sum())
    noise_ratio = round(noise_count / n_docs, 4)

    valid_mask = ~noise_mask
    valid_labels = labels[valid_mask]
    n_clusters = int(len(set(valid_labels))) if valid_mask.any() else 0

    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None

    if n_clusters >= 2 and valid_mask.sum() > n_clusters:
        X = reduced[valid_mask]
        y = valid_labels

        # Sub-sample for silhouette (expensive at large n)
        n_sample = min(silhouette_sample, len(X))
        if n_sample < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=n_sample, replace=False)
            X_s, y_s = X[idx], y[idx]
        else:
            X_s, y_s = X, y

        try:
            silhouette = round(float(silhouette_score(X_s, y_s)), 4)
        except Exception:
            silhouette = None

        try:
            davies_bouldin = round(float(davies_bouldin_score(X, y)), 4)
        except Exception:
            davies_bouldin = None

        try:
            calinski_harabasz = round(float(calinski_harabasz_score(X, y)), 1)
        except Exception:
            calinski_harabasz = None

    return {
        "n_docs": n_docs,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "runtime_s": round(runtime_s, 2),
    }


def compute_coherence(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    coherence_type: str = "c_v",
) -> float | None:
    """
    Compute mean C_v topic coherence across all non-noise clusters.

    C_v uses normalised pointwise mutual information (NPMI) between the top
    words of each cluster and their co-occurrence in a sliding window over the
    corpus.  Unlike silhouette, it is computed entirely from raw text — not
    from the embedding geometry — so it provides an independent signal.

    A high silhouette with low coherence indicates geometrically separated but
    lexically fuzzy clusters (often sentiment blobs).  Both scores high is the
    target for a meaningful topic model.

    Parameters
    ----------
    texts : list[str]
        The preprocessed review texts (same order as labels).
    labels : np.ndarray
        Cluster label per document; -1 = noise.
    top_n : int
        Number of top TF-IDF words to use as the topic representation per cluster.
    coherence_type : str
        Gensim coherence measure.  "c_v" (default) is the most commonly
        reported in the topic modelling literature.

    Returns
    -------
    float | None
        Weighted-average C_v score (0–1, higher is better), or None if gensim
        is not installed or fewer than 2 valid clusters exist.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        return None  # gensim is an optional dependency

    valid_mask = labels != -1
    cluster_ids = sorted(set(labels[valid_mask]))
    if len(cluster_ids) < 2:
        return None

    # Tokenise once for the whole corpus (gensim expects list-of-tokens)
    tokenized = [re.findall(r"[a-z]+", t.lower()) for t in texts]

    # Extract top-N TF-IDF words per cluster as topic representation
    tfidf = TfidfVectorizer(max_features=20_000, stop_words="english")
    try:
        tfidf.fit(texts)
    except Exception:
        return None

    feature_names = np.array(tfidf.get_feature_names_out())
    topic_words: list[list[str]] = []
    cluster_sizes: list[int] = []

    for cid in cluster_ids:
        mask = labels == cid
        cluster_texts = [t for t, m in zip(texts, mask) if m]
        if not cluster_texts:
            continue
        try:
            cluster_matrix = tfidf.transform(cluster_texts)
            scores = np.asarray(cluster_matrix.sum(axis=0)).ravel()
            top_idx = scores.argsort()[::-1][:top_n]
            words = feature_names[top_idx].tolist()
            if len(words) >= 2:
                topic_words.append(words)
                cluster_sizes.append(len(cluster_texts))
        except Exception:
            continue

    if len(topic_words) < 2:
        return None

    try:
        dictionary = Dictionary(tokenized)
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence=coherence_type,
        )
        # Weighted average by cluster size so large clusters count more
        per_topic = cm.get_coherence_per_topic()
        total = sum(s * c for s, c in zip(per_topic, cluster_sizes))
        score = total / sum(cluster_sizes)
        return round(float(score), 4)
    except Exception:
        return None


def compute_rating_entropy(
    stars: np.ndarray | list,
    labels: np.ndarray,
) -> float | None:
    """
    Compute mean normalised star-rating entropy across all non-noise clusters.

    Why this matters
    ----------------
    A **thematic** cluster (e.g. "hotel breakfast quality") attracts reviews
    from guests who loved or hated breakfast — so the 1–5 star distribution is
    spread out → high entropy.

    A **sentiment** cluster (e.g. "5-star fans") collapses almost entirely to
    one rating level → low entropy.  Silhouette cannot distinguish these two
    cases; rating entropy can.

    Interpretation
    --------------
    Normalised entropy is bounded [0, 1]:
    - > 0.85  → strongly thematic (star-rating-independent topic)
    - 0.60–0.85 → mixed
    - < 0.60  → dominated by a single rating level (sentiment blob)

    Parameters
    ----------
    stars : array-like of float/int
        Star rating (1–5) for each document (same order as labels).
    labels : np.ndarray
        Cluster label per document; -1 = noise.

    Returns
    -------
    float | None
        Weighted-average normalised entropy (0–1), or None if fewer than 2
        valid clusters exist or star data is unavailable.
    """
    from scipy.stats import entropy as scipy_entropy

    stars = np.asarray(stars, dtype=float)
    valid_mask = labels != -1
    cluster_ids = sorted(set(labels[valid_mask]))
    if len(cluster_ids) < 2:
        return None

    MAX_ENTROPY = np.log(5)  # uniform distribution over 5 star values
    weighted_sum = 0.0
    total_docs = 0

    for cid in cluster_ids:
        mask = labels == cid
        cluster_stars = stars[mask]
        cluster_stars = cluster_stars[~np.isnan(cluster_stars)]
        if len(cluster_stars) == 0:
            continue

        # Count how many reviews per star value (1–5)
        counts = np.array([np.sum(cluster_stars == s) for s in range(1, 6)], dtype=float)
        if counts.sum() == 0:
            continue

        probs = counts / counts.sum()
        norm_entropy = float(scipy_entropy(probs)) / MAX_ENTROPY
        weighted_sum += norm_entropy * len(cluster_stars)
        total_docs += len(cluster_stars)

    if total_docs == 0:
        return None

    return round(weighted_sum / total_docs, 4)
