"""WP B6 — replay manual edits over a fresh run so re-processing never destroys
human work.

``persist_run_result`` (``ml_mapping.py``) wipes and rewrites a project's clusters
and documents from scratch on every run. Cluster ids are regenerated and any
human-only cluster (created from a selection, or by hand) is gone. This module is
the app analogue of ``reviewscope_ml.hitl.apply_feedback.apply_run_feedback``: it
takes the project's :class:`ClusterEdit` audit log (B1) and re-applies the human
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
are recomputed (B2) afterwards.

``label_source == "hitl_override"`` protection: replay runs *after* persist, so a
human rename re-applied here always wins over the run's machine label. Any future
LLM relabel pass must likewise skip ``hitl_override`` clusters.
"""
from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.ml_mapping import derive_roles
from app.models import Cluster, ClusterEdit, Document, ProjectSchema
from app.services.recompute import _parse_rating, _terms_and_frequencies, numeric_aggregates

logger = logging.getLogger("reviewscope.replay")


@dataclass(frozen=True)
class MembershipSnapshot:
    """Pre-persist state needed to bridge old ids onto the fresh run."""

    doc_pk_by_id: dict[str, str] = field(default_factory=dict)
    """old document UUID (str) -> primary_key_value"""
    cluster_members: dict[uuid.UUID, list[str]] = field(default_factory=dict)
    """old cluster UUID -> primary_key_values that belonged to it"""


def snapshot_membership(session, project_id: uuid.UUID) -> MembershipSnapshot:
    """Capture document/cluster membership **before** ``persist_run_result`` wipes
    the old rows. Must be called inside the same session, prior to persist."""
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
    return MembershipSnapshot(doc_pk_by_id=doc_pk_by_id, cluster_members=cluster_members)


def replay_edits(session, project_id: uuid.UUID, snapshot: MembershipSnapshot) -> None:
    """Re-apply the project's :class:`ClusterEdit` log onto the freshly persisted
    rows. The caller owns the surrounding transaction/commit (B6 calls this from
    ``run_ml_pipeline`` after persist, before the ``status=ready`` commit)."""
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


def _recompute_clusters_sync(session, project_id: uuid.UUID, cluster_ids: list[uuid.UUID]) -> None:
    """Sync counterpart of ``recompute.recompute_clusters`` for the Celery worker.

    Reuses the pure aggregate/term helpers from :mod:`app.services.recompute`; the
    term recompute imports the heavy ML stack lazily (only when a cluster has
    texts), matching that module's behaviour."""
    rating_col = _rating_column_sync(session, project_id)
    for cluster_id in dict.fromkeys(cluster_ids):  # de-dup, keep order
        cluster = session.get(Cluster, cluster_id)
        if cluster is None or cluster.project_id != project_id:
            continue
        rows = session.execute(
            select(Document.text, Document.sentiment_score, Document.raw_data).where(
                Document.project_id == project_id,
                Document.cluster_id == cluster_id,
            )
        ).all()
        texts = [text for text, _, _ in rows]
        sentiments = [sentiment for _, sentiment, _ in rows]
        ratings = [_parse_rating(raw, rating_col) for _, _, raw in rows]

        agg = numeric_aggregates(sentiments, ratings)
        top_terms, freqs = _terms_and_frequencies(texts)
        cluster.size = agg["size"]
        cluster.sentiment_avg = agg["sentiment_avg"]
        cluster.mean_stars = agg["mean_stars"]
        cluster.top_terms = top_terms
        cluster.word_frequencies = freqs


def _rating_column_sync(session, project_id: uuid.UUID) -> str | None:
    schema = session.get(ProjectSchema, project_id)
    if schema is None:
        return None
    _text_col, rating_col = derive_roles(schema.columns)
    return rating_col
