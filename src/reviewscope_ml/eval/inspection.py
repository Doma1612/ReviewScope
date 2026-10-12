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
        if info.n_documents is not None:
            count = f"{info.size} mentions in {info.n_documents} reviews"
        else:
            count = f"{info.size} docs"
        lines += [
            f"### Cluster {cid} — “{info.label}” ({count}{stars}, label: {info.label_source})",
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
        if other == cid:
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
