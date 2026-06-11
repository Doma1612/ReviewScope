"""
Apply reviewer feedback to a run — the "flow back" half of the HITL loop.

Exact semantics per feedback type
---------------------------------

``approve_label``    Stamps the cluster as human-approved
                     (``label_source="hitl_approved"``). Nothing else changes;
                     the value is the recorded approval.
``rename_label``     Overrides the cluster label directly
                     (``label_source="hitl_override"``). LLM output never
                     overwrites a human label on later re-labeling passes.
``merge_clusters``   Post-hoc mapping: every document of the source cluster is
                     reassigned to the target; term lists and word frequencies
                     are recomputed from the merged membership when texts are
                     available, otherwise unioned. Merges survive re-runs as a
                     cluster-level mapping, not as document constraints.
``mark_junk``        The cluster's documents become noise (-1). Junk is a
                     human judgment ("boilerplate, not a theme"), so this is
                     irreversible only in the derived artifact — the original
                     run stays on disk.
``split_cluster``    Two cases. For two-stage runs the macro cluster is split
                     back into its constituent micro-clusters (promoted to
                     top-level clusters) — this is why the micro->macro
                     hierarchy is preserved in the artifact. For flat runs the
                     cluster is tagged in ``needs_recluster`` in the manifest;
                     the next pipeline run re-clusters exactly that subset
                     (targeted re-clustering; a must-link/cannot-link solver
                     would be the heavier alternative and is not needed yet).
``reassign_doc``     Moves one document to the target cluster (or -1). Treated
                     as a hard assignment: applied after every other rule.
``confirm_run``      Writes reviewer + timestamp into the manifest as
                     ``human_confirmed`` — the machine-readable basis for the
                     sentence "a human reviewed the clusters of the winning
                     pipeline and confirmed they are thematically coherent".

Order of application: merges -> junk -> splits -> label actions ->
doc reassignments -> confirmation, each group in timestamp order. The result
is saved as a NEW run directory (``<run>__reviewed``) so the unreviewed
artifact remains for comparison.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.ingest import ReviewSet
from ..pipelines.artifacts import RunArtifacts, load_run, save_run
from ..represent import ctfidf_terms, tfidf_top_terms, word_frequencies
from .feedback import FeedbackRecord, load_feedback

logger = logging.getLogger("reviewscope.hitl")


def apply_feedback(
    art: RunArtifacts,
    records: list[FeedbackRecord],
    reviews: Optional[ReviewSet] = None,
) -> RunArtifacts:
    """Return a new RunArtifacts with all *records* applied (see module doc)."""
    labels = art.labels.copy()
    clusters = {cid: _copy_info(info) for cid, info in art.clusters.items()}
    manifest = dict(art.manifest)
    needs_recluster: list[int] = list(manifest.get("needs_recluster", []))
    applied: list[str] = []

    def by_action(action: str) -> list[FeedbackRecord]:
        return [r for r in records if r.action == action]

    # 1) merges
    for r in by_action("merge_clusters"):
        src, dst = r.cluster_id, r.merge_into
        if src is None or dst is None or src not in clusters or dst not in clusters:
            logger.warning("skipping merge %s -> %s (unknown cluster)", src, dst)
            continue
        labels[labels == src] = dst
        clusters[dst].size += clusters[src].size
        clusters[dst].micro_cluster_ids = sorted(
            set(clusters[dst].micro_cluster_ids) | set(clusters[src].micro_cluster_ids)
        )
        if not clusters[dst].word_frequencies and clusters[src].word_frequencies:
            clusters[dst].word_frequencies = clusters[src].word_frequencies
        del clusters[src]
        applied.append(f"merge {src}->{dst}")

    # 2) junk
    for r in by_action("mark_junk"):
        cid = r.cluster_id
        if cid not in clusters:
            continue
        labels[labels == cid] = -1
        del clusters[cid]
        applied.append(f"junk {cid}")

    # 3) splits
    for r in by_action("split_cluster"):
        cid = r.cluster_id
        if cid not in clusters:
            continue
        if art.micro_labels is not None and clusters[cid].micro_cluster_ids:
            labels, clusters = _split_via_micro(art, labels, clusters, cid)
            applied.append(f"split {cid} via micro-clusters")
        else:
            if cid not in needs_recluster:
                needs_recluster.append(cid)
            applied.append(f"split {cid} flagged for targeted re-clustering")

    # 4) label actions (later record wins by iteration order)
    for r in by_action("approve_label"):
        if r.cluster_id in clusters:
            clusters[r.cluster_id].label_source = "hitl_approved"
            applied.append(f"approve label {r.cluster_id}")
    for r in by_action("rename_label"):
        if r.cluster_id in clusters and r.new_label:
            clusters[r.cluster_id].label = r.new_label
            clusters[r.cluster_id].label_source = "hitl_override"
            applied.append(f"rename {r.cluster_id} -> {r.new_label!r}")

    # 5) doc reassignments (hard overrides)
    id_index = {doc_id: i for i, doc_id in enumerate(art.doc_ids)}
    for r in by_action("reassign_doc"):
        i = id_index.get(r.doc_id)
        if i is None or r.target_cluster_id is None:
            continue
        old = int(labels[i])
        labels[i] = r.target_cluster_id
        if old in clusters:
            clusters[old].size -= 1
        if r.target_cluster_id in clusters:
            clusters[r.target_cluster_id].size += 1
        applied.append(f"reassign {r.doc_id} {old}->{r.target_cluster_id}")

    # 6) confirmation
    for r in by_action("confirm_run"):
        manifest["human_confirmed"] = {
            "reviewer": r.reviewer,
            "timestamp": r.timestamp,
            "note": r.note,
        }
        applied.append(f"confirmed by {r.reviewer}")

    # Recompute term representations for structurally changed clusters.
    if reviews is not None and any(
        a.startswith(("merge", "split", "junk")) for a in applied
    ):
        top = ctfidf_terms(reviews.texts, labels)
        tfidf = tfidf_top_terms(reviews.texts, labels)
        freqs = word_frequencies(reviews.texts, labels)
        for cid, info in clusters.items():
            info.top_terms = [[w, round(s, 5)] for w, s in top.get(cid, [])]
            info.tfidf_terms = [[w, round(s, 5)] for w, s in tfidf.get(cid, [])]
            info.word_frequencies = freqs.get(cid, {})

    manifest["needs_recluster"] = needs_recluster
    manifest["feedback_applied"] = applied
    manifest["run_name"] = f"{art.run_name}__reviewed"

    return RunArtifacts(
        run_name=manifest["run_name"],
        manifest=manifest,
        doc_ids=art.doc_ids,
        labels=labels,
        coords_2d=art.coords_2d,
        coords_3d=art.coords_3d,
        clusters=clusters,
        metrics=dict(art.metrics),
        micro_labels=art.micro_labels,
    )


def _split_via_micro(art, labels, clusters, cid):
    """Promote a macro cluster's micro-clusters to top-level clusters."""
    info = clusters.pop(cid)
    next_id = max(max(clusters, default=0), int(labels.max())) + 1
    for micro_id in info.micro_cluster_ids:
        mask = (art.micro_labels == micro_id) & (labels == cid)
        if not mask.any():
            continue
        labels[mask] = next_id
        from ..pipelines.artifacts import ClusterInfo

        clusters[next_id] = ClusterInfo(
            cluster_id=next_id,
            size=int(mask.sum()),
            label=f"{info.label} / part {micro_id}",
            summary="(split from a reviewed macro cluster — needs re-labeling)",
            label_source="split_pending_label",
            top_terms=[],
            tfidf_terms=[],
            word_frequencies={},
            sample_doc_ids=[art.doc_ids[i] for i in np.flatnonzero(mask)[:5]],
            micro_cluster_ids=[micro_id],
        )
        next_id += 1
    return labels, clusters


def _copy_info(info):
    from copy import deepcopy

    return deepcopy(info)


def apply_run_feedback(run_dir: Path, feedback_dir: Path, reviews=None) -> Path:
    """Load run + all its feedback, apply, save ``<run>__reviewed`` next to it."""
    art = load_run(run_dir)
    records = load_feedback(feedback_dir, art.run_name)
    if not records:
        logger.info("no feedback for %s — nothing to apply", art.run_name)
        return run_dir
    reviewed = apply_feedback(art, records, reviews=reviews)
    out = run_dir.parent / reviewed.run_name
    save_run(out, reviewed)
    logger.info("applied %d feedback records -> %s", len(records), out)
    return out


if __name__ == "__main__":
    import argparse

    from ..core.config import load_config
    from ..data.ingest import load_benchmark

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Apply HITL feedback to a run")
    parser.add_argument("run_name")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="load benchmark texts to recompute terms after merges/splits")
    args = parser.parse_args()

    cfg = load_config(**({"sample_size": args.sample_size} if args.sample_size else {}))
    reviews = load_benchmark(cfg) if args.sample_size else None
    apply_run_feedback(cfg.runs_dir / args.run_name, cfg.feedback_dir, reviews=reviews)
