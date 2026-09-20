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
import os
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

# ── Mention-level variants (v2) ───────────────────────────────────────────────
# The sentence_level pipeline hands the labeler *mentions*, not whole reviews:
# single sentences about one aspect, so a cluster is far tighter and more
# single-topic than a document cluster. v1 says "reviews" and asks what they
# "have in common", which invites the model to hedge toward a generic theme that
# covers a diverse document cluster — the wrong instinct when every member is
# already one aspect. v2 names the unit, asks for the aspect itself, and forbids
# the sentiment-only labels ("Great Experience") that short evaluative mentions
# tempt a model into.
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

# ── Domain-neutral variant ────────────────────────────────────────────────────
# v1 and v2_mention hard-code "customer reviews" and "customers"; applied to a
# parliamentary debate they ask the model to frame legislation as consumer
# feedback. This pair keeps v2's proven structure — name the unit, ask for the
# aspect, forbid sentiment-only labels — and takes the domain as a parameter
# instead of baking one in. ``{domain}`` is substituted from the corpus
# profile's ``domain_description`` BEFORE the ``{docs}`` substitution; see
# ``resolve_prompts``.
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
    """Return the (label, summary) prompt pair actually to be used.

    Resolution order, narrowest first:

    1. ``label_prompt`` / ``summary_prompt`` — a free-text override supplied per
       project by the application. Always wins; it is what "configurable from
       the backend" means. Passed through ``prompt_hash`` like any other pair,
       so an artifact still records exactly which text produced it.
    2. ``variant`` — a named entry in :data:`PROMPT_VARIANTS`. This is the path
       the research harness sweeps over (``eval.label_sweep``), which only works
       because the variants are a closed, named set.

    ``{domain}`` in a template is filled from *domain_description* here, so the
    caller's later ``.format(docs=...)`` sees a template with only the
    placeholders it supplies.
    """
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
    # str.replace, not str.format: the templates also contain {docs} and
    # {label}, which must survive untouched for the caller to substitute.
    return tuple(t.replace("{domain}", domain_description) for t in pair)


def prompt_hash(
    label_prompt: Optional[str] = None, summary_prompt: Optional[str] = None
) -> str:
    """Identifies the exact prompt pair used — stored next to every label.

    Defaults to the v1 pair so existing artifacts keep their historical hash;
    pass the prompts explicitly when using a variant, or the hash would claim a
    provenance the text does not have.
    """
    label_prompt = LABEL_PROMPT if label_prompt is None else label_prompt
    summary_prompt = SUMMARY_PROMPT if summary_prompt is None else summary_prompt
    return hashlib.sha256((label_prompt + summary_prompt).encode()).hexdigest()[:8]


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
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    n_docs: int = 5
    timeout_s: int = 120
    # Prompt pair to use. "v1" is notebook 08's document-level pair (the
    # historical default); "v2_mention" is the hotel sentence/mention pair;
    # "neutral" is the domain-parameterised pair used by every other corpus.
    prompt_variant: str = "v1"
    # Corpus description substituted into a {domain} placeholder, from
    # CorpusProfile.domain_description. Ignored by variants without one.
    domain_description: str = "documents"
    # Free-text override supplied per project by the application layer. Both
    # must be set together; they win over prompt_variant entirely. This is the
    # "configurable from the backend" path — provenance is preserved because
    # the resulting text is hashed into every label like any other pair.
    label_prompt_override: Optional[str] = None
    summary_prompt_override: Optional[str] = None
    # None = leave the model's default (thinking models think); False = ask the
    # server to disable reasoning. Labeling is a short structured-output task,
    # so reasoning buys little and costs a lot of latency.
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
            # Reasoning models (qwen3 and friends) think by default. For a
            # 3-6 word label the reasoning is pure cost — it multiplies latency
            # and its tokens can leak into the answer — so the labeler must be
            # able to turn it off. Servers that do not know the field ignore it.
            payload["think"] = self.think
        r = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s,
        )
        r.raise_for_status()
        body = r.json()
        # When thinking is on, ollama returns reasoning in a separate field —
        # but some builds inline it in <think> tags, which would otherwise be
        # scored as part of the label.
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
        """
        Label + summary per cluster. Falls back to term labels per-cluster on
        request failure and globally when Ollama is down; the ``source`` field
        always says which path produced the text.
        """
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
    """Top-3 c-TF-IDF terms joined — crude but honest, and clearly flagged."""
    words = [w for w, _ in terms.get(cid, [])][:3]
    label = " / ".join(words) if words else f"cluster {cid}"
    return ClusterLabel(
        cluster_id=cid,
        label=label,
        summary="(no LLM available — label derived from top cluster terms)",
        source="terms_fallback",
    )
