from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("reviewscope.label")

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

LABEL_PROMPT_MENTION = """The following sentences are all taken from customer reviews of hotels. They were grouped together because they discuss the same specific aspect.

Sentences:
{docs}

Name the aspect these sentences are about, as a short noun phrase of 3-6 words.
Describe the topic, not the opinion: write "Breakfast Buffet Quality", never "Guests Were Happy".
Output the label only, with no explanation, quotes or trailing punctuation."""

SUMMARY_PROMPT_MENTION = """The following sentences are customer review excerpts about: "{label}".

Sentences:
{docs}

Write 2-3 sentences summarising what customers say about this aspect, including
where they disagree. Be specific and factual. Do not repeat the label."""

LABEL_PROMPT_NEUTRAL = """The following excerpts are all taken from {domain}. They were grouped together because they discuss the same specific topic.

Excerpts:
{docs}

Name the topic these excerpts are about, as a short noun phrase of 3-6 words.
Describe the subject matter, not the stance: write "Fisheries Quota Enforcement", never "Speakers Were Critical".
Output the label only, with no explanation, quotes or trailing punctuation."""

SUMMARY_PROMPT_NEUTRAL = """The following excerpts are taken from {domain}, on the topic: "{label}".

Excerpts:
{docs}

Write 2-3 sentences summarising what is said about this topic, including where
the excerpts disagree. Be specific and factual. Do not repeat the label."""

PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "v1": (LABEL_PROMPT, SUMMARY_PROMPT),
    "v2_mention": (LABEL_PROMPT_MENTION, SUMMARY_PROMPT_MENTION),
    "neutral": (LABEL_PROMPT_NEUTRAL, SUMMARY_PROMPT_NEUTRAL),
}


def resolve_prompts(
    variant: str = "v1",
    domain_description: str = "documents",
    label_prompt: Optional[str] = None,
    summary_prompt: Optional[str] = None,
) -> tuple[str, str]:
    if label_prompt is not None and summary_prompt is not None:
        pair = (label_prompt, summary_prompt)
    else:
        try:
            pair = PROMPT_VARIANTS[variant]
        except KeyError:
            raise ValueError(
                f"Unknown prompt_variant {variant!r}; "
                f"known: {list(PROMPT_VARIANTS)}"
            ) from None
    # Replace only {domain}; {docs} and {label} must survive for the caller.
    return tuple(t.replace("{domain}", domain_description) for t in pair)


def prompt_hash(
    label_prompt: Optional[str] = None, summary_prompt: Optional[str] = None
) -> str:
    label_prompt = LABEL_PROMPT if label_prompt is None else label_prompt
    summary_prompt = SUMMARY_PROMPT if summary_prompt is None else summary_prompt
    return hashlib.sha256((label_prompt + summary_prompt).encode()).hexdigest()[:8]


@dataclass
class ClusterLabel:
    cluster_id: int
    label: str
    summary: str
    source: str
    prompt_hash: Optional[str] = None


def centroid_docs(
    cluster_id: int,
    labels: np.ndarray,
    texts: list[str],
    embeddings: np.ndarray,
    n: int = 5,
    max_chars: int = 300,
) -> list[str]:
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
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    n_docs: int = 5
    timeout_s: int = 120
    prompt_variant: str = "v1"
    domain_description: str = "documents"
    label_prompt_override: Optional[str] = None
    summary_prompt_override: Optional[str] = None
    think: Optional[bool] = None
    _available: Optional[bool] = field(default=None, repr=False)

    @property
    def prompts(self) -> tuple[str, str]:
        return resolve_prompts(
            variant=self.prompt_variant,
            domain_description=self.domain_description,
            label_prompt=self.label_prompt_override,
            summary_prompt=self.summary_prompt_override,
        )

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

        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if self.think is not None:
            payload["think"] = self.think
        r = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s,
        )
        r.raise_for_status()
        body = r.json()
        # Some Ollama builds inline reasoning tags instead of using a separate field.
        text = body.get("response", "")
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return text.strip().strip('"')

    def label_clusters(
        self,
        texts: list[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        terms: dict[int, list[tuple[str, float]]],
    ) -> dict[int, ClusterLabel]:
        cluster_ids = sorted(int(c) for c in set(labels) if c != -1)
        use_llm = self.available()
        if not use_llm:
            logger.warning("labeling %d clusters with term fallback (no LLM)", len(cluster_ids))

        label_prompt, summary_prompt = self.prompts
        phash = prompt_hash(label_prompt, summary_prompt)
        out: dict[int, ClusterLabel] = {}
        for cid in cluster_ids:
            if use_llm:
                try:
                    docs = centroid_docs(cid, labels, texts, embeddings, n=self.n_docs)
                    doc_block = "\n\n".join(f"- {d}" for d in docs)
                    label = self._generate(label_prompt.format(docs=doc_block))
                    summary = self._generate(
                        summary_prompt.format(label=label, docs=doc_block)
                    )
                    out[cid] = ClusterLabel(
                        cluster_id=cid,
                        label=label,
                        summary=summary,
                        source=f"ollama:{self.model}",
                        prompt_hash=phash,
                    )
                    continue
                except Exception as e:
                    logger.warning("LLM labeling failed for cluster %d: %s", cid, e)
            out[cid] = term_fallback_label(cid, terms)
        return out


def term_fallback_label(
    cid: int, terms: dict[int, list[tuple[str, float]]]
) -> ClusterLabel:
    words = [w for w, _ in terms.get(cid, [])][:3]
    label = " / ".join(words) if words else f"cluster {cid}"
    return ClusterLabel(
        cluster_id=cid,
        label=label,
        summary="(no LLM available — label derived from top cluster terms)",
        source="terms_fallback",
    )
