"""
Persistence mapper: a finished :class:`~reviewscope_ml.pipelines.artifacts.RunArtifacts`
(+ the embedding matrix + the originating :class:`UploadedCorpus`) ->
a :class:`~reviewscope_ml.app.dto.RunResult` of DB-ready records.

This is pure translation — no compute. It is the one place that knows how the
pipeline's arrays line up with the spec's ``documents`` / ``embeddings`` /
``clusters`` tables, so the backend never has to.

Alignment contract (document-level variants only)
--------------------------------------------------
``RunArtifacts.doc_ids``, ``.labels``, ``.coords_2d/3d``, ``.sentiment_scores``
and the embedding matrix are all in the same order — the order of the
``ReviewSet`` passed to the runner. Document text / raw_data / stars are looked
up by primary-key value, so the mapping is robust even if that ever changes.
"""
from __future__ import annotations

import numpy as np

from ..pipelines.artifacts import RunArtifacts
from .dto import ClusterRecord, DocumentRecord, EmbeddingRecord, RunResult
from .ingest_upload import UploadedCorpus


def to_records(
    project_id: str,
    corpus: UploadedCorpus,
    artifacts: RunArtifacts,
    embeddings: np.ndarray,
) -> RunResult:
    """Map one finished run to the backend's persistence DTOs."""
    reviews = corpus.reviews
    n = len(artifacts.doc_ids)
    if embeddings.shape[0] != n:
        raise ValueError(
            f"embedding matrix has {embeddings.shape[0]} rows but the run has "
            f"{n} documents — they must be in the same order"
        )

    text_by_pk = dict(zip(reviews.ids, reviews.texts))
    raw_by_pk = dict(zip(reviews.ids, corpus.raw_rows))

    has_sentiment = artifacts.sentiment_scores is not None

    documents: list[DocumentRecord] = []
    embedding_records: list[EmbeddingRecord] = []
    for i, pk in enumerate(artifacts.doc_ids):
        cid = int(artifacts.labels[i])
        sentiment = None
        if has_sentiment:
            s = float(artifacts.sentiment_scores[i])
            sentiment = None if np.isnan(s) else round(s, 4)

        documents.append(DocumentRecord(
            primary_key_value=pk,
            text=text_by_pk.get(pk, ""),
            raw_data=raw_by_pk.get(pk, {}),
            cluster_id=None if cid == -1 else cid,   # noise -> unassigned
            sentiment_score=sentiment,
        ))

        # 3-D UMAP projection drives x/y/z; the 2-D scatter uses (x, y).
        x3, y3, z3 = (float(v) for v in artifacts.coords_3d[i])
        embedding_records.append(EmbeddingRecord(
            primary_key_value=pk,
            vector=[float(v) for v in embeddings[i]],
            umap_x=x3,
            umap_y=y3,
            umap_z=z3,
        ))

    clusters: list[ClusterRecord] = []
    for cid in artifacts.cluster_ids:
        info = artifacts.clusters[cid]
        clusters.append(ClusterRecord(
            cluster_id=cid,
            label=info.label,
            summary=info.summary,
            label_source=info.label_source,
            top_terms=[{"term": t, "score": s} for t, s in info.top_terms],
            word_frequencies=dict(info.word_frequencies),
            size=info.size,
            sentiment_avg=info.sentiment_avg,
            mean_stars=info.mean_stars,
            sample_doc_ids=list(info.sample_doc_ids),
        ))

    return RunResult(
        project_id=project_id,
        documents=documents,
        embeddings=embedding_records,
        clusters=clusters,
        manifest=dict(artifacts.manifest),
        metrics=dict(artifacts.metrics),
    )
