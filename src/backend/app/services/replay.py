from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.ml_mapping import derive_roles
from app.models import Cluster, ClusterEdit, Document, Embedding, Project, ProjectSchema, Segment
from app.services.metrics import cohesion_score
from app.services.recompute import (
    _parse_rating,
    _terms_and_frequencies,
    numeric_aggregates,
    segment_aggregates,
)

logger = logging.getLogger("reviewscope.replay")


@dataclass(frozen=True)
class MembershipSnapshot:

    doc_pk_by_id: dict[str, str] = field(default_factory=dict)
    cluster_members: dict[uuid.UUID, list[str]] = field(default_factory=dict)
    seg_key_by_id: dict[str, str] = field(default_factory=dict)
    seg_cluster_members: dict[uuid.UUID, list[str]] = field(default_factory=dict)


def snapshot_membership(session, project_id: uuid.UUID) -> MembershipSnapshot:
    # Snapshot in the same session before persist_run_result deletes the old rows.
    doc_pk_by_id: dict[str, str] = {}
    cluster_members: dict[uuid.UUID, list[str]] = {}
    rows = session.execute(
        select(Document.id, Document.primary_key_value, Document.cluster_id).where(
            Document.project_id == project_id
        )
    ).all()
    for doc_id, pk, cluster_id in rows:
        doc_pk_by_id[str(doc_id)] = pk
        if cluster_id is not None:
            cluster_members.setdefault(cluster_id, []).append(pk)

    seg_key_by_id: dict[str, str] = {}
    seg_cluster_members: dict[uuid.UUID, list[str]] = {}
    seg_rows = session.execute(
        select(Segment.id, Segment.segment_key, Segment.cluster_id).where(
            Segment.project_id == project_id
        )
    ).all()
    for seg_id, seg_key, cluster_id in seg_rows:
        seg_key_by_id[str(seg_id)] = seg_key
        if cluster_id is not None:
            seg_cluster_members.setdefault(cluster_id, []).append(seg_key)

    return MembershipSnapshot(
        doc_pk_by_id=doc_pk_by_id,
        cluster_members=cluster_members,
        seg_key_by_id=seg_key_by_id,
        seg_cluster_members=seg_cluster_members,
    )


def replay_edits(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    project = session.get(Project, project_id)
    if project is not None and project.unit == "sentence":
        _replay_sentence(session, project_id, snapshot)
    else:
        _replay_document(session, project_id, snapshot)


def _replay_document(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    docs = list(session.execute(select(Document).where(Document.project_id == project_id)).scalars().all())
    doc_by_pk: dict[str, Document] = {d.primary_key_value: d for d in docs}
    # Resolve old clusters against the untouched base assignment, before replay mutates it.
    base_cluster_by_pk: dict[str, uuid.UUID | None] = {d.primary_key_value: d.cluster_id for d in docs}

    clusters = list(session.execute(select(Cluster).where(Cluster.project_id == project_id)).scalars().all())
    cluster_by_id: dict[uuid.UUID, Cluster] = {c.id: c for c in clusters}

    edits = list(
        session.execute(
            select(ClusterEdit).where(ClusterEdit.project_id == project_id).order_by(ClusterEdit.created_at)
        ).scalars().all()
    )

    remap: dict[uuid.UUID, uuid.UUID] = {}
    ml_cache: dict[uuid.UUID, uuid.UUID | None] = {}
    affected: set[uuid.UUID] = set()

    def resolve_doc(old_doc_id: Any) -> Document | None:
        pk = snapshot.doc_pk_by_id.get(str(old_doc_id))
        return doc_by_pk.get(pk) if pk is not None else None

    def resolve_ml(old_cluster_id: uuid.UUID) -> uuid.UUID | None:
        if old_cluster_id in ml_cache:
            return ml_cache[old_cluster_id]
        counts: Counter[uuid.UUID] = Counter()
        for pk in snapshot.cluster_members.get(old_cluster_id, []):
            new_cid = base_cluster_by_pk.get(pk)
            if new_cid is not None:
                counts[new_cid] += 1
        result = counts.most_common(1)[0][0] if counts else None
        ml_cache[old_cluster_id] = result
        return result

    def resolve(old_cluster_id: Any) -> uuid.UUID | None:
        if old_cluster_id is None:
            return None
        new_cid = remap.get(old_cluster_id)
        if new_cid is None:
            new_cid = resolve_ml(old_cluster_id)
        return new_cid if new_cid in cluster_by_id else None

    def by_action(*actions: str) -> list[ClusterEdit]:
        return [e for e in edits if e.action in actions]

    def new_cluster(label: str | None) -> Cluster:
        c = Cluster(
            id=uuid.uuid4(),
            project_id=project_id,
            label=label or "",
            summary="",
            label_source="hitl_override",
            top_terms=[],
            word_frequencies={},
            size=0,
        )
        session.add(c)
        cluster_by_id[c.id] = c
        affected.add(c.id)
        return c

    def move_doc(doc: Document, target: uuid.UUID | None) -> None:
        if doc.cluster_id is not None:
            affected.add(doc.cluster_id)
        doc.cluster_id = target
        if target is not None:
            affected.add(target)

    # Create targets before replaying merges and reassignments.
    for e in by_action("create_cluster", "create_from_selection"):
        c = new_cluster(e.new_label)
        remap[e.cluster_id] = c.id
        for old_doc_id in (e.payload or {}).get("document_ids", []):
            doc = resolve_doc(old_doc_id)
            if doc is not None:
                move_doc(doc, c.id)

    for e in by_action("merge_clusters"):
        src, dst = resolve(e.cluster_id), resolve(e.target_cluster_id)
        if src is None or dst is None or src == dst:
            logger.info("replay: skipping unresolvable merge %s -> %s", e.cluster_id, e.target_cluster_id)
            continue
        for doc in docs:
            if doc.cluster_id == src:
                move_doc(doc, dst)
        session.delete(cluster_by_id.pop(src))
        affected.discard(src)

    for e in by_action("mark_junk"):
        cid = resolve(e.cluster_id)
        if cid is None:
            continue
        for doc in docs:
            if doc.cluster_id == cid:
                move_doc(doc, None)
        session.delete(cluster_by_id.pop(cid))
        affected.discard(cid)

    for e in by_action("split_cluster"):
        logger.info("replay: split_cluster %s not replayable yet — skipping", e.cluster_id)

    for e in by_action("approve_label"):
        cid = resolve(e.cluster_id)
        if cid is not None:
            cluster_by_id[cid].label_source = "hitl_approved"
            affected.add(cid)
    for e in by_action("rename_label"):
        cid = resolve(e.cluster_id)
        if cid is not None and e.new_label:
            cluster_by_id[cid].label = e.new_label
            cluster_by_id[cid].label_source = "hitl_override"
            affected.add(cid)

    for e in by_action("reassign_doc", "bulk_reassign"):
        target = resolve(e.target_cluster_id)
        if e.action == "reassign_doc":
            doc = resolve_doc(e.document_id)
            if doc is not None:
                move_doc(doc, target)
        else:
            for old_doc_id in (e.payload or {}).get("document_ids", []):
                doc = resolve_doc(old_doc_id)
                if doc is not None:
                    move_doc(doc, target)

    for e in by_action("confirm_run"):
        logger.info("replay: confirm_run by %s has no app artifact yet — skipping", e.actor_id)

    session.flush()
    _recompute_clusters_sync(session, project_id, [cid for cid in affected if cid in cluster_by_id])


def _replay_sentence(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    segs = list(session.execute(select(Segment).where(Segment.project_id == project_id)).scalars().all())
    seg_by_key: dict[str, Segment] = {s.segment_key: s for s in segs}
    base_cluster_by_segkey: dict[str, uuid.UUID | None] = {s.segment_key: s.cluster_id for s in segs}
    segs_by_doc: dict[uuid.UUID, list[Segment]] = defaultdict(list)
    for s in segs:
        segs_by_doc[s.document_id].append(s)

    docs = list(session.execute(select(Document).where(Document.project_id == project_id)).scalars().all())
    doc_by_pk: dict[str, Document] = {d.primary_key_value: d for d in docs}

    clusters = list(session.execute(select(Cluster).where(Cluster.project_id == project_id)).scalars().all())
    cluster_by_id: dict[uuid.UUID, Cluster] = {c.id: c for c in clusters}

    edits = list(
        session.execute(
            select(ClusterEdit).where(ClusterEdit.project_id == project_id).order_by(ClusterEdit.created_at)
        ).scalars().all()
    )

    remap: dict[uuid.UUID, uuid.UUID] = {}
    ml_cache: dict[uuid.UUID, uuid.UUID | None] = {}
    affected: set[uuid.UUID] = set()
    affected_docs: set[uuid.UUID] = set()

    def resolve_segment(old_seg_id: Any) -> Segment | None:
        key = snapshot.seg_key_by_id.get(str(old_seg_id))
        return seg_by_key.get(key) if key is not None else None

    def resolve_doc(old_doc_id: Any) -> Document | None:
        pk = snapshot.doc_pk_by_id.get(str(old_doc_id))
        return doc_by_pk.get(pk) if pk is not None else None

    def resolve_ml(old_cluster_id: uuid.UUID) -> uuid.UUID | None:
        if old_cluster_id in ml_cache:
            return ml_cache[old_cluster_id]
        counts: Counter[uuid.UUID] = Counter()
        for seg_key in snapshot.seg_cluster_members.get(old_cluster_id, []):
            new_cid = base_cluster_by_segkey.get(seg_key)
            if new_cid is not None:
                counts[new_cid] += 1
        result = counts.most_common(1)[0][0] if counts else None
        ml_cache[old_cluster_id] = result
        return result

    def resolve(old_cluster_id: Any) -> uuid.UUID | None:
        if old_cluster_id is None:
            return None
        new_cid = remap.get(old_cluster_id)
        if new_cid is None:
            new_cid = resolve_ml(old_cluster_id)
        return new_cid if new_cid in cluster_by_id else None

    def by_action(*actions: str) -> list[ClusterEdit]:
        return [e for e in edits if e.action in actions]

    def new_cluster(label: str | None) -> Cluster:
        c = Cluster(
            id=uuid.uuid4(), project_id=project_id, label=label or "", summary="",
            label_source="hitl_override", top_terms=[], word_frequencies={}, size=0,
        )
        session.add(c)
        cluster_by_id[c.id] = c
        affected.add(c.id)
        return c

    def move_seg(seg: Segment, target: uuid.UUID | None) -> None:
        if seg.cluster_id is not None:
            affected.add(seg.cluster_id)
        seg.cluster_id = target
        if target is not None:
            affected.add(target)
        affected_docs.add(seg.document_id)

    def move_review(old_doc_id: Any, target: uuid.UUID | None) -> None:
        doc = resolve_doc(old_doc_id)
        if doc is None:
            return
        for seg in segs_by_doc.get(doc.id, []):
            move_seg(seg, target)

    # Create targets before replaying merges and reassignments.
    for e in by_action("create_cluster", "create_from_selection"):
        c = new_cluster(e.new_label)
        remap[e.cluster_id] = c.id
        for old_seg_id in (e.payload or {}).get("segment_ids", []):
            seg = resolve_segment(old_seg_id)
            if seg is not None:
                move_seg(seg, c.id)

    for e in by_action("merge_clusters"):
        src, dst = resolve(e.cluster_id), resolve(e.target_cluster_id)
        if src is None or dst is None or src == dst:
            logger.info("replay: skipping unresolvable merge %s -> %s", e.cluster_id, e.target_cluster_id)
            continue
        for seg in segs:
            if seg.cluster_id == src:
                move_seg(seg, dst)
        session.delete(cluster_by_id.pop(src))
        affected.discard(src)

    for e in by_action("mark_junk"):
        cid = resolve(e.cluster_id)
        if cid is None:
            continue
        for seg in segs:
            if seg.cluster_id == cid:
                move_seg(seg, None)
        session.delete(cluster_by_id.pop(cid))
        affected.discard(cid)

    for e in by_action("split_cluster"):
        logger.info("replay: split_cluster %s not replayable yet — skipping", e.cluster_id)

    for e in by_action("approve_label"):
        cid = resolve(e.cluster_id)
        if cid is not None:
            cluster_by_id[cid].label_source = "hitl_approved"
            affected.add(cid)
    for e in by_action("rename_label"):
        cid = resolve(e.cluster_id)
        if cid is not None and e.new_label:
            cluster_by_id[cid].label = e.new_label
            cluster_by_id[cid].label_source = "hitl_override"
            affected.add(cid)

    # Interleave segment/review overrides in timestamp order; legacy moves affect whole reviews.
    for e in by_action(
        "reassign_segment", "bulk_reassign_segments", "reassign_review",
        "reassign_doc", "bulk_reassign",
    ):
        target = resolve(e.target_cluster_id)
        if e.action == "reassign_segment":
            seg = resolve_segment(e.segment_id)
            if seg is not None:
                move_seg(seg, target)
        elif e.action == "bulk_reassign_segments":
            for old_seg_id in (e.payload or {}).get("segment_ids", []):
                seg = resolve_segment(old_seg_id)
                if seg is not None:
                    move_seg(seg, target)
        elif e.action == "bulk_reassign":
            for old_doc_id in (e.payload or {}).get("document_ids", []):
                move_review(old_doc_id, target)
        else:
            move_review(e.document_id, target)

    for e in by_action("confirm_run"):
        logger.info("replay: confirm_run by %s has no app artifact yet — skipping", e.actor_id)

    session.flush()
    _recompute_clusters_sync(session, project_id, [cid for cid in affected if cid in cluster_by_id])
    _recompute_document_primary_sync(session, project_id, list(affected_docs))


def _recompute_clusters_sync(session, project_id: uuid.UUID, cluster_ids: list[uuid.UUID]) -> None:
    rating_col = _rating_column_sync(session, project_id)
    project = session.get(Project, project_id)
    sentence = bool(project is not None and project.unit == "sentence")
    for cluster_id in dict.fromkeys(cluster_ids):
        cluster = session.get(Cluster, cluster_id)
        if cluster is None or cluster.project_id != project_id:
            continue
        if sentence:
            rows = session.execute(
                select(
                    Segment.text,
                    Segment.sentiment_score,
                    Segment.document_id,
                    Segment.vector,
                    Document.raw_data,
                )
                .join(Document, Document.id == Segment.document_id)
                .where(Segment.project_id == project_id, Segment.cluster_id == cluster_id)
            ).all()
            texts = [text for text, _, _, _, _ in rows]
            sentiments = [s for _, s, _, _, _ in rows]
            document_ids = [doc_id for _, _, doc_id, _, _ in rows]
            vectors = [vec for _, _, _, vec, _ in rows if vec]
            ratings_by_document = {doc_id: _parse_rating(raw, rating_col) for _, _, doc_id, _, raw in rows}
            agg = segment_aggregates(document_ids, sentiments, ratings_by_document)
            cluster.n_mentions = agg["n_mentions"]
        else:
            rows = session.execute(
                select(
                    Document.text,
                    Document.sentiment_score,
                    Document.raw_data,
                    Embedding.vector,
                )
                .outerjoin(Embedding, Embedding.document_id == Document.id)
                .where(Document.project_id == project_id, Document.cluster_id == cluster_id)
            ).all()
            texts = [text for text, _, _, _ in rows]
            sentiments = [sentiment for _, sentiment, _, _ in rows]
            ratings = [_parse_rating(raw, rating_col) for _, _, raw, _ in rows]
            vectors = [vector for _, _, _, vector in rows if vector]
            agg = numeric_aggregates(sentiments, ratings)
            cluster.n_mentions = agg["size"]

        top_terms, freqs = _terms_and_frequencies(texts)
        cluster.size = agg["size"]
        cluster.sentiment_avg = agg["sentiment_avg"]
        cluster.mean_stars = agg["mean_stars"]
        cluster.cohesion = cohesion_score(vectors)
        cluster.top_terms = top_terms
        cluster.word_frequencies = freqs


def _recompute_document_primary_sync(session, project_id: uuid.UUID, document_ids: list[uuid.UUID]) -> None:
    for document_id in dict.fromkeys(document_ids):
        rows = session.execute(
            select(Segment.cluster_id).where(
                Segment.project_id == project_id,
                Segment.document_id == document_id,
                Segment.cluster_id.is_not(None),
            )
        ).all()
        counts = Counter(cid for (cid,) in rows)
        primary = counts.most_common(1)[0][0] if counts else None
        doc = session.get(Document, document_id)
        if doc is not None and doc.project_id == project_id:
            doc.cluster_id = primary


def _rating_column_sync(session, project_id: uuid.UUID) -> str | None:
    schema = session.get(ProjectSchema, project_id)
    if schema is None:
        return None
    _text_col, rating_col = derive_roles(schema.columns)
    return rating_col
