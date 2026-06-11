"""
Qualitative inspection artifacts — the part of the harness a human actually
reads, and the part with the final say.

Design rules (mission north star):

- Sample documents are **random** members, not nearest-to-centroid. Centroid
  documents are the most average members and make every cluster look tidier
  than it is; random members show the cluster a reviewer will actually get.
- The rendering must let a human judge "is this one theme?" in seconds:
  per cluster, the label, the top terms, and the sampled documents sit side
  by side.
- The intruder test makes boundary quality testable by a human: 4 documents
  from the cluster plus 1 from elsewhere, shuffled. If the reader cannot spot
  the intruder, the boundary between those clusters is not meaningful —
  regardless of silhouette. Rendering is automated; judging is deliberately
  not.
"""
from __future__ import annotations

import numpy as np

from ..data.ingest import ReviewSet
from ..pipelines.artifacts import RunArtifacts

SNIPPET_CHARS = 280


def _doc_lookup(reviews: ReviewSet) -> dict[str, str]:
    return dict(zip(reviews.ids, reviews.raw_texts))


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")


def render_inspection(art: RunArtifacts, reviews: ReviewSet) -> str:
    """
    Markdown inspection sheet for one run: every cluster with its label, top
    terms and the random samples stored in the artifact.
    """
    docs = _doc_lookup(reviews)
    lines = [
        f"## Qualitative inspection — `{art.run_name}`",
        "",
        "_Samples are random cluster members (not centroid-nearest). "
        "Ask per cluster: can you name this in one phrase, and do the samples fit that phrase?_",
        "",
    ]
    noise = int((art.labels == -1).sum())
    if noise:
        lines += [f"**Noise:** {noise} documents ({noise / len(art.labels):.0%}) unassigned.", ""]

    for cid in art.cluster_ids:
        info = art.clusters[cid]
        terms = ", ".join(w for w, _ in (tuple(t) for t in info.top_terms[:8]))
        stars = f" · avg {info.mean_stars}★" if info.mean_stars is not None else ""
        lines += [
            f"### Cluster {cid} — “{info.label}” ({info.size} docs{stars}, label: {info.label_source})",
            f"**Top terms:** {terms}",
            "",
        ]
        for doc_id in info.sample_doc_ids:
            lines.append(f"- {_snippet(docs.get(doc_id, '(document not in sample)'))}")
        lines.append("")
    return "\n".join(lines)


def render_intruder_test(
    art: RunArtifacts, reviews: ReviewSet, seed: int = 42, n_members: int = 4
) -> str:
    """
    Markdown intruder test: per cluster, ``n_members`` random members plus one
    document from a different cluster, shuffled. The answer key is collected
    at the bottom so the reader can self-test honestly first.
    """
    rng = np.random.default_rng(seed)
    docs = _doc_lookup(reviews)
    ids_by_cluster = {
        cid: [reviews.ids[i] for i in np.flatnonzero(art.labels == cid)]
        for cid in art.cluster_ids
    }
    eligible = [cid for cid in art.cluster_ids if len(ids_by_cluster[cid]) >= n_members]
    if len(eligible) < 2:
        return "_Intruder test skipped: fewer than two clusters with enough documents._"

    lines = [
        f"## Intruder test — `{art.run_name}`",
        "",
        f"_Each block shows {n_members} documents from one cluster and 1 from another, shuffled. "
        "If the intruder is not obvious, the boundary between the two clusters is not meaningful. "
        "Answer key at the bottom._",
        "",
    ]
    answers = []
    for cid in eligible:
        members = rng.choice(ids_by_cluster[cid], size=n_members, replace=False)
        other = eligible[int(rng.integers(len(eligible) - 1))]
        if other == cid:  # pick the next cluster instead of re-rolling
            other = eligible[(eligible.index(cid) + 1) % len(eligible)]
        intruder = rng.choice(ids_by_cluster[other])

        block = [(doc_id, False) for doc_id in members] + [(intruder, True)]
        rng.shuffle(block)

        label = art.clusters[cid].label
        lines.append(f"### Block {cid} — cluster “{label}”")
        for pos, (doc_id, is_intruder) in enumerate(block, start=1):
            lines.append(f"{pos}. {_snippet(docs.get(doc_id, '?'))}")
            if is_intruder:
                answers.append(
                    f"Block {cid}: intruder is #{pos} "
                    f"(from cluster {other}, “{art.clusters[other].label}”)"
                )
        lines.append("")

    lines += ["---", "", "<details><summary>Answer key</summary>", ""]
    lines += [f"- {a}" for a in answers]
    lines += ["", "</details>"]
    return "\n".join(lines)
