"""
Cluster keyword extraction.

Two flavours, both kept because they answer different questions:

- **c-TF-IDF** (BERTopic-style): treats each cluster as one pseudo-document
  and asks "which words are characteristic of this cluster *relative to the
  other clusters*". Best for distinguishing labels — feeds the LLM labeler
  and the per-cluster top-terms list.
- **plain TF-IDF top terms**: per-document TF-IDF summed within the cluster.
  Matches what the Tier-2 coherence metric uses, so a cluster's displayed
  terms and its coherence score talk about the same vocabulary.

``word_frequencies`` provides raw within-cluster counts for the app-spec word
clouds (term size proportional to frequency *within* the cluster).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _cluster_ids(labels: np.ndarray) -> list[int]:
    return sorted(int(c) for c in set(labels) if c != -1)


def ctfidf_terms(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    max_features: int = 20_000,
) -> dict[int, list[tuple[str, float]]]:
    """
    BERTopic-style class-based TF-IDF: tf of term t in class c, scaled by
    log(1 + A / tf_t) where A is the average word count per class and tf_t the
    corpus frequency of t. Noise documents (-1) are excluded from classes but
    still shape nothing — they are simply ignored.
    """
    ids = _cluster_ids(labels)
    if not ids:
        return {}

    pseudo_docs = [" ".join(t for t, l in zip(texts, labels) if l == cid) for cid in ids]
    cv = CountVectorizer(stop_words="english", max_features=max_features)
    tf = cv.fit_transform(pseudo_docs)  # (n_classes, vocab)
    vocab = np.array(cv.get_feature_names_out())

    tf = np.asarray(tf.todense(), dtype=float)
    corpus_freq = tf.sum(axis=0)
    avg_words_per_class = tf.sum() / len(ids)
    idf = np.log(1.0 + avg_words_per_class / np.maximum(corpus_freq, 1.0))
    scores = tf * idf

    out: dict[int, list[tuple[str, float]]] = {}
    for row, cid in enumerate(ids):
        top_idx = scores[row].argsort()[::-1][:top_n]
        out[cid] = [(str(vocab[i]), float(scores[row, i])) for i in top_idx if scores[row, i] > 0]
    return out


def tfidf_top_terms(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    max_features: int = 20_000,
) -> dict[int, list[tuple[str, float]]]:
    """Per-document TF-IDF summed per cluster — same scheme as Tier-2 coherence."""
    ids = _cluster_ids(labels)
    if not ids:
        return {}
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())

    out: dict[int, list[tuple[str, float]]] = {}
    for cid in ids:
        mask = labels == cid
        scores = np.asarray(matrix[mask].sum(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:top_n]
        out[cid] = [(str(vocab[i]), float(scores[i])) for i in top_idx if scores[i] > 0]
    return out


def word_frequencies(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 50,
    stop_words: Optional[set[str]] = None,
) -> dict[int, dict[str, int]]:
    """Within-cluster word counts for the app's word clouds."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stops = stop_words if stop_words is not None else set(ENGLISH_STOP_WORDS)
    out: dict[int, dict[str, int]] = {}
    for cid in _cluster_ids(labels):
        counter: Counter[str] = Counter()
        for text, label in zip(texts, labels):
            if label != cid:
                continue
            counter.update(
                w for w in _tokenize(text) if w not in stops and len(w) > 2
            )
        out[cid] = dict(counter.most_common(top_n))
    return out


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", text.lower())
