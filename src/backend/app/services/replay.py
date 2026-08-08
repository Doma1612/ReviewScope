"""Replay manual edits over a fresh run so re-processing never destroys
human work.

``persist_run_result`` (``ml_mapping.py``) wipes and rewrites a project's clusters
and documents from scratch on every run. Cluster ids are regenerated and any
human-only cluster (created from a selection, or by hand) is gone. This module is
the app analogue of ``reviewscope_ml.hitl.apply_feedback.apply_run_feedback``: it
takes the project's :class:`ClusterEdit` audit log and re-applies the human
*decisions* on top of the new run — keeping the better base clustering while
restoring renames, reassignments and human-made clusters.

Two stable identities bridge the runs:

* **Documents** by ``primary_key_value`` — cluster ids change, doc ids change, but
  the upload's primary key is stable. Because the edit log stores *old* document
  UUIDs, a pre-persist snapshot (:func:`snapshot_membership`, taken before
  ``persist_run_result`` deletes the old rows) maps each old doc UUID to its
  ``primary_key_value``, which then maps to the freshly inserted document.
* **Clusters** have no stable id, so an old cluster UUID is resolved to a new one
  by membership: the new cluster holding the plurality of the old cluster's
  members (from the same snapshot, against the fresh run's base assignment).
  Human-created clusters are tracked in ``remap`` as they are recreated.

Application order mirrors ``apply_feedback``: creates → merges → junk → splits →
label actions → doc reassignments → confirm, each group in ``created_at`` order
(creates run first because later merges/reassignments may target them; splits and
confirm have no app-side artifact yet and are logged + skipped). Affected clusters
are recomputed afterwards.

``label_source == "hitl_override"`` protection: replay runs *after* persist, so a
human rename re-applied here always wins over the run's machine label. Any future
LLM relabel pass must likewise skip ``hitl_override`` clusters.
"""
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
    """Pre-persist state needed to bridge old ids onto the fresh run."""

    doc_pk_by_id: dict[str, str] = field(default_factory=dict)
    """old document UUID (str) -> primary_key_value"""
    cluster_members: dict[uuid.UUID, list[str]] = field(default_factory=dict)
    """old cluster UUID -> primary_key_values that belonged to it (document unit)"""
    seg_key_by_id: dict[str, str] = field(default_factory=dict)
    """old segment UUID (str) -> segment_key (sentence unit)"""
    seg_cluster_members: dict[uuid.UUID, list[str]] = field(default_factory=dict)
    """old cluster UUID -> segment_keys that belonged to it (sentence unit)"""


def snapshot_membership(session, project_id: uuid.UUID) -> MembershipSnapshot:
    """Capture document/segment/cluster membership **before**
    ``persist_run_result`` wipes the old rows. Must be called inside the same
    session, prior to persist. Snapshots both units; the empty maps are harmless
    for whichever unit the project isn't."""
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
    """Re-apply the project's :class:`ClusterEdit` log onto the freshly persisted
    rows. The caller owns the surrounding transaction/commit (``run_ml_pipeline``
    calls this after persist, before the ``status=ready`` commit).

    Dispatches on the just-persisted project ``unit``: sentence-unit runs replay
    edits over segments, document-unit runs over documents (legacy)."""
    project = session.get(Project, project_id)
    if project is not None and project.unit == "sentence":
        _replay_sentence(session, project_id, snapshot)
    else:
        _replay_document(session, project_id, snapshot)


def _replay_document(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    docs = list(session.execute(select(Document).where(Document.project_id == project_id)).scalars().all())
    doc_by_pk: dict[str, Document] = {d.primary_key_value: d for d in docs}
    # The fresh run's base assignment — the anchor for resolving old clusters by
    # membership. Captured before any edit mutates it so resolution stays stable.
    base_cluster_by_pk: dict[str, uuid.UUID | None] = {d.primary_key_value: d.cluster_id for d in docs}

    clusters = list(session.execute(select(Cluster).where(Cluster.project_id == project_id)).scalars().all())
    cluster_by_id: dict[uuid.UUID, Cluster] = {c.id: c for c in clusters}

    edits = list(
        session.execute(
            select(ClusterEdit).where(ClusterEdit.project_id == project_id).order_by(ClusterEdit.created_at)
        ).scalars().all()
    )

    remap: dict[uuid.UUID, uuid.UUID] = {}  # old human-cluster UUID -> recreated UUID
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
        # Could resolve onto a cluster deleted earlier in this replay (e.g. a merge
        # source); treat that as unresolvable.
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

    # 1) creates (app-only; must precede merges/reassignments that target them)
    for e in by_action("create_cluster", "create_from_selection"):
        c = new_cluster(e.new_label)
        remap[e.cluster_id] = c.id
        for old_doc_id in (e.payload or {}).get("document_ids", []):
            doc = resolve_doc(old_doc_id)
            if doc is not None:
                move_doc(doc, c.id)

    # 2) merges
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

    # 3) junk — the cluster's documents become noise and the cluster is removed
    for e in by_action("mark_junk"):
        cid = resolve(e.cluster_id)
        if cid is None:
            continue
        for doc in docs:
            if doc.cluster_id == cid:
                move_doc(doc, None)
        session.delete(cluster_by_id.pop(cid))
        affected.discard(cid)

    # 4) splits — no app-side artifact yet (would need micro-labels / re-cluster)
    for e in by_action("split_cluster"):
        logger.info("replay: split_cluster %s not replayable yet — skipping", e.cluster_id)

    # 5) label actions (approve then rename, matching apply_feedback)
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

    # 6) doc reassignments (hard overrides; interleaved in timestamp order)
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

    # 7) confirm_run — no app-side manifest yet; skipped (logged for traceability)
    for e in by_action("confirm_run"):
        logger.info("replay: confirm_run by %s has no app artifact yet — skipping", e.actor_id)

    session.flush()
    _recompute_clusters_sync(session, project_id, [cid for cid in affected if cid in cluster_by_id])


def _replay_sentence(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    """Sentence-unit replay: re-apply edits over the fresh run's *segments*.

    Mirrors :func:`_replay_document` but the moved unit is the segment. Old cluster
    UUIDs resolve to new ones by *segment-key* plurality; segment-level actions
    resolve by ``segment_key``; review-level actions (``reassign_review`` and the
    legacy ``reassign_doc``/``bulk_reassign``) move *all* of a review's current
    segments to the target, which survives re-segmentation."""
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

    # 1) creates (app-only; must precede merges/reassignments that target them)
    for e in by_action("create_cluster", "create_from_selection"):
        c = new_cluster(e.new_label)
        remap[e.cluster_id] = c.id
        for old_seg_id in (e.payload or {}).get("segment_ids", []):
            seg = resolve_segment(old_seg_id)
            if seg is not None:
                move_seg(seg, c.id)

    # 2) merges — move every segment of the source cluster into the target
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

    # 3) junk — the cluster's segments become noise and the cluster is removed
    for e in by_action("mark_junk"):
        cid = resolve(e.cluster_id)
        if cid is None:
            continue
        for seg in segs:
            if seg.cluster_id == cid:
                move_seg(seg, None)
        session.delete(cluster_by_id.pop(cid))
        affected.discard(cid)

    # 4) splits — no app-side artifact yet
    for e in by_action("split_cluster"):
        logger.info("replay: split_cluster %s not replayable yet — skipping", e.cluster_id)

    # 5) label actions (approve then rename, matching apply_feedback)
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

    # 6) reassignments (hard overrides; segment- and review-level interleaved in
    #    timestamp order). Legacy document actions are replayed as review-level.
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
        else:  # reassign_review / legacy reassign_doc
            move_review(e.document_id, target)

    # 7) confirm_run — skipped
    for e in by_action("confirm_run"):
        logger.info("replay: confirm_run by %s has no app artifact yet — skipping", e.actor_id)

    session.flush()
    _recompute_clusters_sync(session, project_id, [cid for cid in affected if cid in cluster_by_id])
    _recompute_document_primary_sync(session, project_id, list(affected_docs))


def _recompute_clusters_sync(session, project_id: uuid.UUID, cluster_ids: list[uuid.UUID]) -> None:
    """Sync counterpart of ``recompute.recompute_clusters`` for the Celery worker.

    Reuses the pure aggregate/term helpers from :mod:`app.services.recompute`;
    branches on the project ``unit`` (segments for sentence, documents for
    document). The term recompute imports the heavy ML stack lazily."""
    rating_col = _rating_column_sync(session, project_id)
    project = session.get(Project, project_id)
    sentence = bool(project is not None and project.unit == "sentence")
    for cluster_id in dict.fromkeys(cluster_ids):  # de-dup, keep order
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
    """Sync counterpart of ``recompute.recompute_document_primary``: refresh each
    touched review's derived primary cluster from its segments' plurality."""
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
