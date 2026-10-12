from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

VARIANTS = (
    "bertopic", "custom_hdbscan", "flat_agglomerative", "two_stage", "sentence_level",
)

# Document-unit and sentence-unit models were selected in separate experiments.
DEFAULT_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"

SENTENCE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REDUCER: dict[str, Any] = {
    "n_components": 10,
    "n_neighbors": 15,
    "min_dist": 0.0,
    "metric": "cosine",
}


@dataclass
class PipelineSpec:
    variant: str
    embedding_model: str = DEFAULT_EMBEDDING
    instruction: str = "no_inst"
    reducer: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_REDUCER))
    cluster: dict[str, Any] = field(default_factory=dict)
    label_model: str = "llama3.2"
    prompt_variant: Optional[str] = None
    label_prompt: Optional[str] = None
    summary_prompt: Optional[str] = None
    # Uploads need an explicit corpus profile; their project cache slugs resolve to neutral.
    corpus_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose one of {VARIANTS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_specs() -> dict[str, PipelineSpec]:
    return {
        "bertopic": PipelineSpec(
            variant="bertopic",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster={"min_topic_size": 10},
        ),
        "custom_hdbscan": PipelineSpec(
            variant="custom_hdbscan",
            cluster={"min_cluster_size": "auto", "min_samples": "auto"},
        ),
        "flat_agglomerative": PipelineSpec(
            variant="flat_agglomerative",
            cluster={"k": 15, "linkage": "ward"},
        ),
        "two_stage": PipelineSpec(
            variant="two_stage",
            cluster={"micro_min_cluster_size": "auto", "micro_min_samples": "auto",
                     "n_macro": None},
        ),
        "sentence_level": PipelineSpec(
            variant="sentence_level",
            embedding_model=SENTENCE_EMBEDDING,
            cluster={"min_cluster_size": "auto", "min_samples": "auto"},
        ),
    }
