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
    """Map one finished run to the backend's persistence DTOs.

    Two shapes, keyed off the run manifest's ``unit``:
    - ``document`` — one document + one embedding per review (the legacy path).
    - ``sentence`` — ``artifacts.doc_ids`` are segment ids; each is a clustered
      mention (a SegmentRecord), and reviews are grouped back into one
      DocumentRecord carrying a derived "primary" cluster.
    """
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
    """ClusterRecords shared by both paths.

    ``size`` is the distinct-review count (``n_documents`` for sentence runs,
    else the member count); ``n_mentions`` is always the raw member/segment
    count (equal to ``size`` for document runs).
    """
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
        # 3-D UMAP projection drives x/y/z; the 2-D scatter uses (x, y).
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
    # Re-derive segment texts deterministically; segment_reviews produced the
    # exact ids the runner clustered, so this maps id -> text with no drift.
    seg_units = segment_reviews(reviews)
    text_by_segid = dict(zip(seg_units.ids, seg_units.texts))

    segments: list[SegmentRecord] = []
    # Per-review accumulators (insertion order = first appearance of the review).
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

    # Total segments per review (incl. noise) — n_segments; label_counts holds
    # only the non-noise labels used to pick the plurality "primary" cluster.
    seg_totals: Counter = Counter(parent_id(s) for s in artifacts.doc_ids)
    documents: list[DocumentRecord] = []
    for parent, counts in label_counts.items():
        primary = counts.most_common(1)[0][0] if counts else None  # noise never wins
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
