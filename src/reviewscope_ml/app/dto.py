from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentRecord:
    primary_key_value: str
    text: str
    raw_data: dict[str, Any]
    cluster_id: Optional[int]
    sentiment_score: Optional[float] = None
    primary_cluster_id: Optional[int] = None
    n_segments: int = 1


@dataclass
class SegmentRecord:

    segment_key: str
    parent_key: str
    ordinal: int
    text: str
    cluster_id: Optional[int]
    sentiment_score: Optional[float] = None
    vector: list[float] = field(default_factory=list)
    umap_x: float = 0.0
    umap_y: float = 0.0
    umap_z: Optional[float] = None


@dataclass
class EmbeddingRecord:
    primary_key_value: str
    vector: list[float]
    umap_x: float
    umap_y: float
    umap_z: Optional[float] = None


@dataclass
class ClusterRecord:
    cluster_id: int
    label: str
    summary: str
    label_source: str
    top_terms: list[dict[str, Any]]
    word_frequencies: dict[str, int]
    size: int
    sentiment_avg: Optional[float] = None
    mean_stars: Optional[float] = None
    sample_doc_ids: list[str] = field(default_factory=list)
    n_mentions: int = 0


@dataclass
class RunResult:

    project_id: str
    documents: list[DocumentRecord]
    embeddings: list[EmbeddingRecord]
    clusters: list[ClusterRecord]
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    unit: str = "document"
    segments: list[SegmentRecord] = field(default_factory=list)

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)
