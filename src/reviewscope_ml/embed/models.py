from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EmbeddingCandidate:
    model: str
    params_m: int
    dim: int
    max_seq: int
    instruction: str = "no_inst"
    gated: bool = False      # needs HF license acceptance + token
    notes: str = ""
    batch_hint: int = 64
    encode_seq: Optional[int] = None
    extra_instructions: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: str = ""

    def instruction_variants(self) -> tuple[str, ...]:
        return (self.instruction, *self.extra_instructions)


CANDIDATES: list[EmbeddingCandidate] = [
    EmbeddingCandidate(
        "sentence-transformers/all-MiniLM-L6-v2", 22, 384, 256,
        batch_hint=256,
        notes="speed baseline; BERTopic default. At segment granularity this is "
              "a serious contender, not just a floor — the 5k sentence sweep "
              "ranked it first on mean rank.",
    ),
    EmbeddingCandidate(
        "sentence-transformers/all-mpnet-base-v2", 110, 768, 384,
        batch_hint=128,
        notes="incumbent default (notebook 04 winner on the 5k document "
              "benchmark); the model the fixed UMAP/HDBSCAN referee was "
              "calibrated on — see the circularity note in methodology §3",
    ),
    EmbeddingCandidate(
        "Alibaba-NLP/gte-modernbert-base", 149, 768, 8192,
        batch_hint=64, encode_seq=2048,
        notes="MTEB Clustering 46.47 / STS 81.57; strongest ~150M at document "
              "level, but its 8k context — the main reason it was registered — "
              "is worth little at segment granularity; eager attention on Pascal",
    ),
    EmbeddingCandidate(
        "nomic-ai/nomic-embed-text-v1.5", 137, 768, 2048,
        batch_hint=64,
        notes="re-admitted August 2026: transformers now ships NomicBertModel "
              "natively, so it no longer needs trust_remote_code. Exploratory — "
              "it expects task prefixes ('clustering: ') that our instruction "
              "mechanism only approximates, so read a weak result with that caveat",
    ),
    EmbeddingCandidate(
        "mixedbread-ai/mxbai-embed-large-v1", 335, 1024, 512,
        batch_hint=64,
        notes="added August 2026: best admissible MTEB Clustering (46.71) AND "
              "best STS (85.00) — STS is the competence that matters when the "
              "unit is one sentence. Plain BERT-large, no remote code, ungated",
    ),
    EmbeddingCandidate(
        "BAAI/bge-large-en-v1.5", 335, 1024, 512,
        batch_hint=64,
        notes="added August 2026 as the size-matched contrast to mxbai "
              "(Clustering 46.08 / STS 83.11): same architecture and parameter "
              "count, different training recipe — isolates recipe from capacity",
    ),
    EmbeddingCandidate(
        "google/embeddinggemma-300m", 308, 768, 2048, gated=True,
        batch_hint=64,
        notes="best <500M multilingual on MTEB; repo is gated:manual — being "
              "logged in is not enough, access must be granted per account, "
              "which is why the June sweep skipped it",
    ),
    EmbeddingCandidate(
        "BAAI/bge-m3", 570, 1024, 8192,
        batch_hint=32, encode_seq=2048,
        notes="multilingual + 8k ctx; registered for the EuroParl phase. Its "
              "long-context rationale does not transfer to segment units — "
              "judge it here on clustering quality per unit of compute",
    ),
    EmbeddingCandidate(
        "Snowflake/snowflake-arctic-embed-l-v2.0", 568, 1024, 8192,
        batch_hint=32, encode_seq=2048,
        notes="added August 2026: admissible (XLMRoberta, no remote code), "
              "multilingual, strong retrieval/clustering — the forward-looking "
              "EuroParl candidate that competes on this corpus too",
    ),
    EmbeddingCandidate(
        "intfloat/multilingual-e5-large-instruct", 560, 1024, 512,
        instruction="domain", extra_instructions=("no_inst",), batch_hint=64,
        notes="instruction-tuned; both variants run so the instruction effect is "
              "measurable — watch for silhouette inflation without coherence gain",
    ),
    EmbeddingCandidate(
        "Qwen/Qwen3-Embedding-0.6B", 600, 1024, 32768, instruction="generic",
        extra_instructions=("no_inst",),
        batch_hint=32, encode_seq=2048,
        notes="best sub-1B on MTEB overall; largest candidate that comfortably "
              "fits the 6 GB fp32 slice. 32k ctx is irrelevant at segment level, "
              "so it must justify its cost on quality alone",
    ),
]


def encode_settings(model_name: str, requested_batch: int) -> tuple[int, Optional[int]]:
    for c in CANDIDATES:
        if c.model == model_name:
            return min(requested_batch, c.batch_hint), c.encode_seq
    return requested_batch, None


def candidates(
    max_params_m: int = 700,
    include_gated: bool = True,
    include_superseded: bool = True,
) -> list[EmbeddingCandidate]:
    return [
        c for c in CANDIDATES
        if c.params_m <= max_params_m
        and (include_gated or not c.gated)
        and (include_superseded or not c.superseded_by)
    ]
