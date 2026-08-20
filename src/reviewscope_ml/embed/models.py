"""
Curated embedding-model candidates for the clustering benchmark.

Hard constraints (a candidate that fails any one is inadmissible regardless of
leaderboard position — recorded in the admissibility report, not silently
dropped; see ``eval/cost_probe.py``, which *verifies* these on the box):

- loadable with plain sentence-transformers, **no trust_remote_code**, so the
  Celery worker needs no custom code paths;
- fp32 weights + activations fit the ~6 GB VRAM slice we may claim on a shared
  TITAN X Pascal (rules out the 4B/7B instruction embedders);
- Pascal (sm_61) has no flash-attention — models that *prefer* FA2 (ModernBERT
  class) must run eager, which works but costs throughput. torch stays pinned at
  2.7.1+cu126: wheels >= 2.8 dropped Pascal cubins.

Selection criteria (August 2026 re-review, sentence/segment unit)
-----------------------------------------------------------------
The downstream task is density-based clustering of short, single-aspect text
spans, so candidates are weighted on the MTEB **Clustering** and **STS** task
groups rather than the overall average — holding up on sentence-length input is
not the same competence as document retrieval, and the overall average is
dominated by retrieval datasets.

The long-context argument, which justified bge-m3 / gte-modernbert at document
level (~10% of hotel reviews exceed mpnet's 384-token window and are silently
truncated, docs/methodology.md §3), **weakens considerably at segment
granularity**: an 8k window buys nothing when the unit is one sentence. Those
models are therefore re-weighted here on clustering quality per unit of compute.
Multilingual capacity stays a forward-looking criterion for EuroParl Phase 2
only — it does not earn a candidate a slot on the Hotels benchmark.

Considered and rejected on the trust_remote_code constraint (verified against
transformers 5.10.1 / sentence-transformers 5.5.1, not assumed):
``Alibaba-NLP/gte-large-en-v1.5`` (needs Alibaba-NLP/new-impl) and
``jinaai/jina-embeddings-v3`` (needs jinaai/xlm-roberta-flash-implementation).
``nomic-ai/nomic-embed-text-v1.5`` *was* in that group and no longer is:
transformers now ships NomicBertModel natively, so it is admissible again.

This list feeds ``python -m reviewscope_ml.eval.model_sweep``; the three-tier
verdict (not raw silhouette) decides, and the comparison harness re-validates
the winner end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EmbeddingCandidate:
    model: str
    params_m: int            # parameters, millions
    dim: int
    max_seq: int             # tokens the model *supports*
    instruction: str = "no_inst"   # primary instruction slug for this model
    gated: bool = False      # needs HF license acceptance + token
    notes: str = ""
    # VRAM-safe encode settings for a 12 GB TITAN X. batch_hint caps the
    # requested batch size; encode_seq caps the sequence length at encode time
    # (None = model native). Long-context models NEED the cap: bge-m3 padding
    # a batch to 8k tokens allocates activations no 12 GB card can hold —
    # 2048 tokens still covers >99.9% of reviews un-truncated.
    batch_hint: int = 64
    encode_seq: Optional[int] = None
    # Additional instruction slugs to evaluate this model under. Instruction and
    # no-instruction runs stay SEPARATE rows in the sweep: an instruction
    # mechanically reshapes the space and inflates silhouette without moving
    # coherence (notebook 04), so averaging them would hide exactly the effect
    # we are testing for.
    extra_instructions: tuple[str, ...] = field(default_factory=tuple)
    # Set when a later candidate makes this one redundant; kept in the registry
    # so the selection history stays auditable rather than being edited away.
    superseded_by: str = ""

    def instruction_variants(self) -> tuple[str, ...]:
        """Every instruction slug this candidate is evaluated under."""
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
    """(safe batch size, sequence cap) for a model — registry hint wins when
    smaller than the requested batch; unknown models pass through unchanged."""
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
