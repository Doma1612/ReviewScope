"""
LLM cluster labeling via Ollama (notebook 08's strategies).

Context strategy: centroid-nearest documents (notebook 08 strategy A) — the
most *average* members describe what the cluster is mostly about, which suits
a 3-6 word label. Prompt templates are notebook 08's verbatim; the prompt
hash and model name are stored with every label (tech-selection requirement:
reproducibility of generated text).

Honesty over availability: if Ollama is not reachable, we do NOT fail the
pipeline and do NOT silently fake an LLM. Each cluster falls back to a
term-based label (top c-TF-IDF words) and the artifact records
``label_source="terms_fallback"`` so the HITL reviewer and the report can see
that no language model stood behind these labels.

LLM labels are also a known hallucination risk (the model sees 5 documents,
not the cluster) — they are *proposals*, and approving/renaming them is one
of the documented human-in-the-loop steps.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("reviewscope.label")

# Notebook 08 templates (v1 label prompt + summary prompt), verbatim.
LABEL_PROMPT = """You are analyzing customer reviews. The following reviews all belong to the same topic cluster.

Reviews:
{docs}

Give a short, specific topic label (3-6 words) that describes what these reviews have in common.
Label only, no explanation."""

SUMMARY_PROMPT = """You are analyzing a cluster of customer reviews with the theme: "{label}".

Representative reviews:
{docs}

Write a 2-3 sentence summary of what customers in this cluster are saying.
Be specific and factual. Do not repeat the label."""


def prompt_hash() -> str:
    """Identifies the exact prompt pair used — stored next to every label."""
    return hashlib.sha256((LABEL_PROMPT + SUMMARY_PROMPT).encode()).hexdigest()[:8]


@dataclass
class ClusterLabel:
    cluster_id: int
    label: str
    summary: str
    source: str          # "ollama:<model>" | "terms_fallback"
    prompt_hash: Optional[str] = None


def centroid_docs(
    cluster_id: int,
    labels: np.ndarray,
    texts: list[str],
    embeddings: np.ndarray,
    n: int = 5,
    max_chars: int = 300,
) -> list[str]:
    """The n documents nearest the cluster centroid in embedding space."""
    mask = labels == cluster_id
    cluster_embs = embeddings[mask]
    cluster_texts = [t for t, m in zip(texts, mask) if m]
    centroid = cluster_embs.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid)
    sims = (cluster_embs @ centroid.T).ravel() / np.maximum(norm, 1e-12)
    top_idx = sims.argsort()[::-1][:n]
    return [cluster_texts[i][:max_chars] for i in top_idx]


@dataclass
class OllamaLabeler:
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    n_docs: int = 5
    timeout_s: int = 120
    _available: Optional[bool] = field(default=None, repr=False)

    def available(self) -> bool:
        if self._available is None:
            import requests

            try:
                r = requests.get(f"{self.base_url}/api/tags", timeout=3)
                names = [m["name"] for m in r.json().get("models", [])]
                self._available = any(self.model in n for n in names)
                if not self._available:
                    logger.warning(
                        "Ollama reachable but model %r not pulled (have: %s)",
                        self.model, names,
                    )
            except Exception as e:
                logger.warning("Ollama not reachable at %s: %s", self.base_url, e)
                self._available = False
        return self._available

    def _generate(self, prompt: str) -> str:
        import requests

        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()["response"].strip().strip('"')

    def label_clusters(
        self,
        texts: list[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        terms: dict[int, list[tuple[str, float]]],
    ) -> dict[int, ClusterLabel]:
        """
        Label + summary per cluster. Falls back to term labels per-cluster on
        request failure and globally when Ollama is down; the ``source`` field
        always says which path produced the text.
        """
        cluster_ids = sorted(int(c) for c in set(labels) if c != -1)
        use_llm = self.available()
        if not use_llm:
            logger.warning("labeling %d clusters with term fallback (no LLM)", len(cluster_ids))

        out: dict[int, ClusterLabel] = {}
        for cid in cluster_ids:
            if use_llm:
                try:
                    docs = centroid_docs(cid, labels, texts, embeddings, n=self.n_docs)
                    doc_block = "\n\n".join(f"- {d}" for d in docs)
                    label = self._generate(LABEL_PROMPT.format(docs=doc_block))
                    summary = self._generate(
                        SUMMARY_PROMPT.format(label=label, docs=doc_block)
                    )
                    out[cid] = ClusterLabel(
                        cluster_id=cid,
                        label=label,
                        summary=summary,
                        source=f"ollama:{self.model}",
                        prompt_hash=prompt_hash(),
                    )
                    continue
                except Exception as e:
                    logger.warning("LLM labeling failed for cluster %d: %s", cid, e)
            out[cid] = term_fallback_label(cid, terms)
        return out


def term_fallback_label(
    cid: int, terms: dict[int, list[tuple[str, float]]]
) -> ClusterLabel:
    """Top-3 c-TF-IDF terms joined — crude but honest, and clearly flagged."""
    words = [w for w, _ in terms.get(cid, [])][:3]
    label = " / ".join(words) if words else f"cluster {cid}"
    return ClusterLabel(
        cluster_id=cid,
        label=label,
        summary="(no LLM available — label derived from top cluster terms)",
        source="terms_fallback",
    )
