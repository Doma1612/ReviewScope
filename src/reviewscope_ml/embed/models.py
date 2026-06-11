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
from typing import Optional


@dataclass(frozen=True)
class EmbeddingCandidate:
    model: str
    params_m: int            # parameters, millions
    dim: int
    max_seq: int             # tokens the model *supports*
    instruction: str = "no_inst"   # instruction slug to use for this model
    gated: bool = False      # needs HF license acceptance + token
    notes: str = ""
    # VRAM-safe encode settings for a 12 GB TITAN X. batch_hint caps the
    # requested batch size; encode_seq caps the sequence length at encode time
    # (None = model native). Long-context models NEED the cap: bge-m3 padding
    # a batch to 8k tokens allocates activations no 12 GB card can hold —
    # 2048 tokens still covers >99.9% of reviews un-truncated.
    batch_hint: int = 64
    encode_seq: Optional[int] = None


CANDIDATES: list[EmbeddingCandidate] = [
    EmbeddingCandidate(
        "sentence-transformers/all-MiniLM-L6-v2", 22, 384, 256,
        batch_hint=256,
        notes="speed baseline; BERTopic default",
    ),
    EmbeddingCandidate(
        "sentence-transformers/all-mpnet-base-v2", 110, 768, 384,
        batch_hint=128,
        notes="current default (notebook 04 winner on the 5k benchmark)",
    ),
    EmbeddingCandidate(
        "Alibaba-NLP/gte-modernbert-base", 149, 768, 8192,
        batch_hint=64, encode_seq=2048,
        notes="strongest ~150M on MTEB clustering; 8k ctx fixes truncation; "
              "eager attention on Pascal (no flash-attn)",
    ),
    EmbeddingCandidate(
        "google/embeddinggemma-300m", 308, 768, 2048, gated=True,
        batch_hint=64,
        notes="best <500M multilingual on MTEB; requires HF license "
              "acceptance + `hf auth login`",
    ),
    EmbeddingCandidate(
        "BAAI/bge-m3", 570, 1024, 8192,
        batch_hint=32, encode_seq=2048,
        notes="multilingual + 8k ctx; relevant for the EuroParl phase",
    ),
    EmbeddingCandidate(
        "intfloat/multilingual-e5-large-instruct", 560, 1024, 512,
        instruction="domain", batch_hint=64,
        notes="instruction-tuned; compare against no_inst — watch for "
              "silhouette inflation without coherence gain",
    ),
    EmbeddingCandidate(
        "Qwen/Qwen3-Embedding-0.6B", 600, 1024, 32768, instruction="generic",
        batch_hint=32, encode_seq=2048,
        notes="best sub-1B on MTEB overall; 32k ctx; largest candidate that "
              "comfortably fits the 6 GB VRAM slice in fp32",
    ),
]


def encode_settings(model_name: str, requested_batch: int) -> tuple[int, Optional[int]]:
    """(safe batch size, sequence cap) for a model — registry hint wins when
    smaller than the requested batch; unknown models pass through unchanged."""
    for c in CANDIDATES:
        if c.model == model_name:
            return min(requested_batch, c.batch_hint), c.encode_seq
    return requested_batch, None


def candidates(
    max_params_m: int = 700, include_gated: bool = True
) -> list[EmbeddingCandidate]:
    return [
        c for c in CANDIDATES
        if c.params_m <= max_params_m and (include_gated or not c.gated)
    ]
