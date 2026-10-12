# Artifact arrays and embeddings share the input ReviewSet order.
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict

import numpy as np

from ..data.segment import parent_id, segment_reviews
from ..pipelines.artifacts import RunArtifacts
from .dto import ClusterRecord, DocumentRecord, EmbeddingRecord, RunResult, SegmentRecord
from .ingest_upload import UploadedCorpus


def to_records(
    project_id: str,
    corpus: UploadedCorpus,
    artifacts: RunArtifacts,
    embeddings: np.ndarray,
) -> RunResult:
    n = len(artifacts.doc_ids)
    if embeddings.shape[0] != n:
        raise ValueError(
            f"embedding matrix has {embeddings.shape[0]} rows but the run has "
            f"{n} units — they must be in the same order"
        )

    unit = str(artifacts.manifest.get("unit", "document"))
    if unit == "sentence":
        return _sentence_records(project_id, corpus, artifacts, embeddings)
    return _document_records(project_id, corpus, artifacts, embeddings)


def _sentiment_at(artifacts: RunArtifacts, i: int) -> float | None:
    if artifacts.sentiment_scores is None:
        return None
    s = float(artifacts.sentiment_scores[i])
    return None if np.isnan(s) else round(s, 4)


def _cluster_records(artifacts: RunArtifacts, *, sentence: bool) -> list[ClusterRecord]:
    out: list[ClusterRecord] = []
    for cid in artifacts.cluster_ids:
        info = artifacts.clusters[cid]
        size = info.n_documents if (sentence and info.n_documents is not None) else info.size
        out.append(ClusterRecord(
            cluster_id=cid,
            label=info.label,
            summary=info.summary,
            label_source=info.label_source,
            top_terms=[{"term": t, "score": s} for t, s in info.top_terms],
            word_frequencies=dict(info.word_frequencies),
            size=size,
            sentiment_avg=info.sentiment_avg,
            mean_stars=info.mean_stars,
            sample_doc_ids=list(info.sample_doc_ids),
            n_mentions=info.size,
        ))
    return out


def _document_records(
    project_id: str, corpus: UploadedCorpus, artifacts: RunArtifacts, embeddings: np.ndarray,
) -> RunResult:
    reviews = corpus.reviews
    text_by_pk = dict(zip(reviews.ids, reviews.texts))
    raw_by_pk = dict(zip(reviews.ids, corpus.raw_rows))

    documents: list[DocumentRecord] = []
    embedding_records: list[EmbeddingRecord] = []
    for i, pk in enumerate(artifacts.doc_ids):
        cid = int(artifacts.labels[i])
        cluster_id = None if cid == -1 else cid
        documents.append(DocumentRecord(
            primary_key_value=pk,
            text=text_by_pk.get(pk, ""),
            raw_data=raw_by_pk.get(pk, {}),
            cluster_id=cluster_id,
            sentiment_score=_sentiment_at(artifacts, i),
            primary_cluster_id=cluster_id,
            n_segments=1,
        ))
        # x/y/z use the 3-D projection; the 2-D scatter displays its x/y axes.
        x3, y3, z3 = (float(v) for v in artifacts.coords_3d[i])
        embedding_records.append(EmbeddingRecord(
            primary_key_value=pk,
            vector=[float(v) for v in embeddings[i]],
            umap_x=x3, umap_y=y3, umap_z=z3,
        ))

    return RunResult(
        project_id=project_id,
        documents=documents,
        embeddings=embedding_records,
        clusters=_cluster_records(artifacts, sentence=False),
        manifest=dict(artifacts.manifest),
        metrics=dict(artifacts.metrics),
        unit="document",
        segments=[],
    )


def _sentence_records(
    project_id: str, corpus: UploadedCorpus, artifacts: RunArtifacts, embeddings: np.ndarray,
) -> RunResult:
    reviews = corpus.reviews
    text_by_pk = dict(zip(reviews.ids, reviews.texts))
    raw_by_pk = dict(zip(reviews.ids, corpus.raw_rows))
    seg_units = segment_reviews(reviews)
    text_by_segid = dict(zip(seg_units.ids, seg_units.texts))

    segments: list[SegmentRecord] = []
    label_counts: dict[str, Counter] = OrderedDict()
    seg_sentiments: dict[str, list[float]] = defaultdict(list)
    for i, segid in enumerate(artifacts.doc_ids):
        cid = int(artifacts.labels[i])
        cluster_id = None if cid == -1 else cid
        parent = parent_id(segid)
        sentiment = _sentiment_at(artifacts, i)
        x3, y3, z3 = (float(v) for v in artifacts.coords_3d[i])
        segments.append(SegmentRecord(
            segment_key=segid,
            parent_key=parent,
            ordinal=int(segid.rsplit("#", 1)[1]) if "#" in segid else 0,
            text=text_by_segid.get(segid, ""),
            cluster_id=cluster_id,
            sentiment_score=sentiment,
            vector=[float(v) for v in embeddings[i]],
            umap_x=x3, umap_y=y3, umap_z=z3,
        ))
        label_counts.setdefault(parent, Counter())
        if cluster_id is not None:
            label_counts[parent][cluster_id] += 1
        if sentiment is not None:
            seg_sentiments[parent].append(sentiment)

    seg_totals: Counter = Counter(parent_id(s) for s in artifacts.doc_ids)
    documents: list[DocumentRecord] = []
    for parent, counts in label_counts.items():
        primary = counts.most_common(1)[0][0] if counts else None
        sentiments = seg_sentiments.get(parent, [])
        documents.append(DocumentRecord(
            primary_key_value=parent,
            text=text_by_pk.get(parent, ""),
            raw_data=raw_by_pk.get(parent, {}),
            cluster_id=primary,
            sentiment_score=round(sum(sentiments) / len(sentiments), 4) if sentiments else None,
            primary_cluster_id=primary,
            n_segments=int(seg_totals[parent]),
        ))

    return RunResult(
        project_id=project_id,
        documents=documents,
        embeddings=[],
        clusters=_cluster_records(artifacts, sentence=True),
        manifest=dict(artifacts.manifest),
        metrics=dict(artifacts.metrics),
        unit="sentence",
        segments=segments,
    )
