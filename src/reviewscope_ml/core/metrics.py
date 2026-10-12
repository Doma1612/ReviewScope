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
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        return None

    valid_mask = labels != -1
    cluster_ids = sorted(set(labels[valid_mask]))
    if len(cluster_ids) < 2:
        return None

    tokenized = [re.findall(r"[a-z]+", t.lower()) for t in texts]

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
    from scipy.stats import entropy as scipy_entropy

    stars = np.asarray(stars, dtype=float)
    valid_mask = labels != -1
    cluster_ids = sorted(set(labels[valid_mask]))
    if len(cluster_ids) < 2:
        return None

    MAX_ENTROPY = np.log(5)
    weighted_sum = 0.0
    total_docs = 0

    for cid in cluster_ids:
        mask = labels == cid
        cluster_stars = stars[mask]
        cluster_stars = cluster_stars[~np.isnan(cluster_stars)]
        if len(cluster_stars) == 0:
            continue

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
