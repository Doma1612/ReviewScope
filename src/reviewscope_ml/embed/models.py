"""
Curated embedding-model candidates for the clustering benchmark.

Selection criteria (June 2026):
- strong MTEB clustering/overall scores for their size class — Qwen3-Embedding
  0.6B is the best sub-1B model overall, EmbeddingGemma-300m the best under
  500M, gte-modernbert-base the strongest ~150M with an 8k context;
- loadable with plain sentence-transformers (no trust_remote_code), so the
  Celery worker needs no custom code paths;
- fp32 weights + activations fit the 6 GB VRAM slice we may claim on a shared
  TITAN X Pascal (rules out the 4B/7B instruction embedders);
- Pascal has no flash-attention — models that *prefer* FA2 (ModernBERT) must
  run in eager mode, which works but is slower.

Long-context models (bge-m3, Qwen3, gte-modernbert: 8k-32k tokens) matter
beyond leaderboard points here: ~10% of hotel reviews exceed mpnet's
384-token window and are silently truncated (see docs/methodology.md §3).

This list feeds ``python -m reviewscope_ml.eval.model_sweep``; notebook 04's
three-tier verdict (not raw silhouette) decides, and the comparison harness
re-validates the winner end to end.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingCandidate:
    model: str
    params_m: int            # parameters, millions
    dim: int
    max_seq: int             # tokens; mind the ~10% of reviews > 384 tokens
    instruction: str = "no_inst"   # instruction slug to use for this model
    gated: bool = False      # needs HF license acceptance + token
    notes: str = ""


CANDIDATES: list[EmbeddingCandidate] = [
    EmbeddingCandidate(
        "sentence-transformers/all-MiniLM-L6-v2", 22, 384, 256,
        notes="speed baseline; BERTopic default",
    ),
    EmbeddingCandidate(
        "sentence-transformers/all-mpnet-base-v2", 110, 768, 384,
        notes="current default (notebook 04 winner on the 5k benchmark)",
    ),
    EmbeddingCandidate(
        "Alibaba-NLP/gte-modernbert-base", 149, 768, 8192,
        notes="strongest ~150M on MTEB clustering; 8k ctx fixes truncation; "
              "eager attention on Pascal (no flash-attn)",
    ),
    EmbeddingCandidate(
        "google/embeddinggemma-300m", 308, 768, 2048, gated=True,
        notes="best <500M multilingual on MTEB; requires HF license "
              "acceptance + `hf auth login`",
    ),
    EmbeddingCandidate(
        "BAAI/bge-m3", 570, 1024, 8192,
        notes="multilingual + 8k ctx; relevant for the EuroParl phase",
    ),
    EmbeddingCandidate(
        "intfloat/multilingual-e5-large-instruct", 560, 1024, 512,
        instruction="domain",
        notes="instruction-tuned; compare against no_inst — watch for "
              "silhouette inflation without coherence gain",
    ),
    EmbeddingCandidate(
        "Qwen/Qwen3-Embedding-0.6B", 600, 1024, 32768, instruction="generic",
        notes="best sub-1B on MTEB overall; 32k ctx; largest candidate that "
              "comfortably fits the 6 GB VRAM slice in fp32",
    ),
]


def candidates(
    max_params_m: int = 700, include_gated: bool = True
) -> list[EmbeddingCandidate]:
    return [
        c for c in CANDIDATES
        if c.params_m <= max_params_m and (include_gated or not c.gated)
    ]
