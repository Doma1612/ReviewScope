"""
Persistence DTOs — the contract between the ML pipeline and the backend's
database. Each dataclass mirrors one table in the app spec's PostgreSQL schema,
field-for-field, so the backend can map a :class:`RunResult` to ORM rows
without reading any pipeline internals.

Deliberately plain dataclasses (no pydantic / SQLAlchemy): the library stays
framework-agnostic. The backend converts these to its own models.

Mapping to the app-spec tables
-------------------------------
``DocumentRecord``  -> ``documents``   (one per kept document)
``EmbeddingRecord`` -> ``embeddings``  (one per document; vector + UMAP coords)
``ClusterRecord``   -> ``clusters``    (one per non-noise cluster)

Notes for the backend
----------------------
* ``DocumentRecord.cluster_id`` is the pipeline's **integer** cluster id (or
  None for noise / unassigned). The backend resolves it to the cluster row's
  UUID after inserting the ``ClusterRecord``s.
* ``EmbeddingRecord`` carries the **3-D** UMAP projection in ``umap_x/y/z``;
  the 2-D scatter uses ``(x, y)``. The pipeline also computes a dedicated 2-D
  projection (``coords_2d`` in the run artifact) — wire it only if you add
  ``umap_x2/y2`` columns for a truer 2-D layout (see integration-guide.md).
* ``vector`` is a plain ``list[float]``; cast to pgvector on insert.

TODO(integration, with backend owner): single source of truth for this shape.
Today the shape is defined twice — here (DTO) and in the backend's ORM models —
which can drift. Resolve to one of:
  (a) keep these DTOs as the canonical contract; backend maps DTO -> ORM (status quo);
  (b) extract a tiny shared ``reviewscope_contracts`` package both ML and backend
      import (removes the duplication).
NOT an option: a base class owned by the backend that these inherit from — the ML
package must not depend on the backend (ports & adapters: the core has no outward
dependencies). So if we deduplicate, the shared definition lives in a neutral
package, never in the backend. Decision belongs with whoever owns persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentRecord:
    primary_key_value: str           # documents.primary_key_value
    text: str                        # documents.text (preprocessed NLP text)
    raw_data: dict[str, Any]         # documents.raw_data (all original columns)
    cluster_id: Optional[int]        # documents.cluster_id (None = noise/unassigned)
    sentiment_score: Optional[float] = None  # documents.sentiment_score


@dataclass
class EmbeddingRecord:
    primary_key_value: str           # join key to the document
    vector: list[float]              # embeddings.vector (pgvector on insert)
    umap_x: float                    # embeddings.umap_x  (3-D projection)
    umap_y: float                    # embeddings.umap_y
    umap_z: Optional[float] = None   # embeddings.umap_z


@dataclass
class ClusterRecord:
    cluster_id: int                  # pipeline integer id (backend assigns the UUID)
    label: str                       # clusters.label
    summary: str                     # clusters.summary
    label_source: str                # provenance: ollama:<model> | terms_fallback | hitl_override
    top_terms: list[dict[str, Any]]  # clusters.top_terms  [{term, score}, ...]
    word_frequencies: dict[str, int] # clusters.word_frequencies (drives the word cloud)
    size: int                        # clusters.size
    sentiment_avg: Optional[float] = None  # clusters.sentiment_avg
    mean_stars: Optional[float] = None     # avg star rating (if a rating column exists)
    sample_doc_ids: list[str] = field(default_factory=list)  # random member samples


@dataclass
class RunResult:
    """Everything the backend persists for one finished pipeline run."""

    project_id: str
    documents: list[DocumentRecord]
    embeddings: list[EmbeddingRecord]
    clusters: list[ClusterRecord]
    manifest: dict[str, Any]         # provenance: spec, seed, per-stage cost, label sources
    metrics: dict[str, Any]          # three-tier metrics + failure flags

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)
