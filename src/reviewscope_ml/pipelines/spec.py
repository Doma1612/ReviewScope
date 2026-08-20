"""
Pipeline specification: one config object drives all four variants.

The four candidates under comparison (mission brief / WP5):

a. ``bertopic``           — BERTopic off-the-shelf (its default UMAP+HDBSCAN+
                            c-TF-IDF, its default embedding model MiniLM).
                            The only deviation from stock is that we seed its
                            UMAP — without a controllable seed the multi-seed
                            stability comparison would be meaningless.
b. ``custom_hdbscan``     — our embed -> UMAP -> HDBSCAN with the parameters
                            notebooks 04-06 selected.
c. ``flat_agglomerative`` — same embed/reduce, agglomerative (ward) cut.
d. ``two_stage``          — fine HDBSCAN micro-clusters, agglomerative merge
                            of micro centroids into macro topics.
e. ``sentence_level``     — reviews are split into sentences before embedding;
                            the unit of clustering becomes the *mention*, so
                            multi-aspect reviews stop averaging their aspects
                            into one vector and clusters become aspect themes.
                            Cluster size counts mentions; distinct-review
                            counts and the per-review membership map are in
                            the artifacts (``n_documents``,
                            ``doc_membership.json``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VARIANTS = (
    "bertopic", "custom_hdbscan", "flat_agglomerative", "two_stage", "sentence_level",
)

# Notebook 04 decision (5k hotel benchmark, DOCUMENT unit): mpnet without
# instruction beat the instruction-tuned candidates once coherence/entropy were
# taken into account. This stays the document-unit default — the 2026-08
# re-selection was run on segments and does not license re-deciding the
# document variants, which are out of its scope.
DEFAULT_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"

# 2026-08 sentence/mention-unit decision (docs/technology-selection.md): across
# 12 contestants on 43,012 segments, MiniLM-L6-v2 won the mean rank across all
# three tiers AND cost ~10x less than the runner-up (3,413 vs 341 segments/s,
# 498 vs 2,337 MB peak VRAM). Short single-aspect spans are its regime; the
# larger models' advantages (long context, general MTEB rank) did not transfer.
SENTENCE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
# Notebook 05 decision: UMAP 10d, nn=15, min_dist=0.0, cosine.
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

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose one of {VARIANTS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_specs() -> dict[str, PipelineSpec]:
    """The four comparison candidates with their notebook-decided defaults."""
    return {
        "bertopic": PipelineSpec(
            variant="bertopic",
            # BERTopic's own default embedding model — that is what
            # "off-the-shelf" means; not our mpnet choice.
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            cluster={"min_topic_size": 10},
        ),
        # "auto" size parameters resolve against the actual unit count at run
        # time (runner._make_backend): mcs = 0.3% of units (floor 15), the
        # ratio anchored to notebook 06's mcs=15 at 5k. k for the
        # partitioners stays fixed — topic count does not grow with corpus
        # size, topic *size* does. BERTopic stays stock (min_topic_size=10):
        # it is the "what you get without thinking" baseline by definition,
        # including its scale problems.
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
