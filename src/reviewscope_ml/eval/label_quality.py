"""
LLM cluster-label quality scoring — notebook 08's unfinished experiment.

Notebook 08 chose the centroid context strategy and the v1 prompt on informal
reading; its human scoring sheet was never filled in, so docs/methodology.md §7
has carried "the centroid/v1 choice rests on informal reading" as an open
weakness ever since. Ollama being down then made it worse: every label in every
artifact is ``terms_fallback``, so no language model has ever actually been
judged on this corpus.

This module scores labelers on the three axes the tech-selection requirement
names, and writes the human sheet alongside so the qualitative call can finally
be made:

1. **Format compliance** — deterministic. The prompt asks for a 3-6 word label,
   nothing else; a model that answers "Sure! Here's a label: \"Clean Rooms\"."
   is unusable in a UI regardless of how apt the words are. Measured, not
   eyeballed, because this is the one axis with a ground truth.

2. **Faithfulness to the cluster** — a label is faithful if it describes *this*
   cluster and not a neighbouring one. We embed the label with the same model
   that produced the space and check whether its own cluster's centroid is the
   nearest one (``discrimination@1``) and by what margin. This is an automatic
   analogue of the methodology's intruder test: a label that cannot pick its own
   cluster out of the lineup does not identify it.

3. **Hallucination** — content words in the label that occur in *no* member
   mention. Deliberately a lexical check: the failure mode is the model
   inventing specifics it never saw ("Rooftop Infinity Pool" for a cluster about
   bathrooms), and that shows up as vocabulary with no support in the text.

Honest limits of these proxies, which the report repeats so no one over-reads it:

- Lexical grounding penalises *correct paraphrase*. "Breakfast" as a label for
  mentions that all say "morning buffet" is a good label and scores as
  ungrounded. The metric is a tripwire for invention, not a quality score, and
  it must be read next to the human sheet.
- Discrimination@1 rewards distinctive labels, which usually means faithful
  ones — but a cluster that genuinely overlaps its neighbour caps the score
  through no fault of the labeler.
- None of the three measures whether a label is *useful* to a human. That is
  what the scoring sheet is for, and why label approval stays a mandatory HITL
  step (methodology §9).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("reviewscope.label_quality")

# Preambles and wrappers a chat-tuned model adds when it ignores "label only".
_META_PREFIX = re.compile(
    r"^\s*(sure|certainly|here(?:'s| is)|okay|ok|label|topic|answer|aspect)\b[:,!]?\s*",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z']+")

# Words that carry no topical content, so their presence or absence says
# nothing about grounding.
_STOPWORDS = frozenset("""
a an the and or of for to in on at by with from about into over after is are was
were be been being this that these those it its their there here very quite
""".split())

LABEL_MIN_WORDS = 3
LABEL_MAX_WORDS = 6


@dataclass
class LabelScore:
    cluster_id: int
    label: str
    raw_label: str
    n_words: int
    format_ok: bool
    format_issues: list[str] = field(default_factory=list)
    own_sim: Optional[float] = None
    best_other_sim: Optional[float] = None
    margin: Optional[float] = None
    discriminates: Optional[bool] = None
    grounded_words: int = 0
    content_words: int = 0
    grounding_rate: Optional[float] = None
    hallucinated: Optional[bool] = None
    gen_s: Optional[float] = None


def check_format(raw: str) -> tuple[str, bool, list[str]]:
    """(cleaned label, compliant?, issues). Compliance is judged on the RAW
    string; the cleaned label is what a UI would have to salvage.

    Cleaning is iterative because the wrappers nest: "Sure! Here is a label:
    \"Clean Rooms\"." carries a preamble, a nested second preamble, quotes and a
    full stop, and a single pass would leave most of it in place. That matters
    beyond tidiness — the cleaned label is what gets embedded for the
    faithfulness metric, so leftover boilerplate would drag a model's
    discrimination score down for a formatting sin it is already charged for.
    """
    issues: list[str] = []
    label = raw.strip()
    if not label:
        return "", False, ["empty"]

    if "\n" in label:
        issues.append("multi-line")
        label = label.split("\n")[0].strip()

    # Peel preambles and quoting until the string stops shrinking.
    for _ in range(5):
        before = label
        if _META_PREFIX.match(label):
            if "meta-preamble" not in issues:
                issues.append("meta-preamble")
            label = _META_PREFIX.sub("", label, count=1).strip()
        stripped = label.strip('"\'“”‘’').strip()
        if stripped != label:
            if "quoted" not in issues:
                issues.append("quoted")
            label = stripped
        if label == before:
            break

    if label.endswith((".", "!", ";", ":", ",")):
        issues.append("trailing-punctuation")
        label = label.rstrip(".!;:,").strip()

    n_words = len(_WORD.findall(label))
    if n_words < LABEL_MIN_WORDS:
        issues.append(f"too-short ({n_words}w)")
    elif n_words > LABEL_MAX_WORDS:
        issues.append(f"too-long ({n_words}w)")

    return label, not issues, issues


def _content_words(label: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(label)
            if w.lower() not in _STOPWORDS and len(w) > 2]


def score_label(
    cluster_id: int,
    raw_label: str,
    member_texts: list[str],
    label_vec: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    centroid_ids: Optional[list[int]] = None,
    gen_s: Optional[float] = None,
) -> LabelScore:
    """Score one label on format, faithfulness and grounding."""
    label, ok, issues = check_format(raw_label)
    score = LabelScore(
        cluster_id=cluster_id, label=label, raw_label=raw_label,
        n_words=len(_WORD.findall(label)), format_ok=ok,
        format_issues=issues, gen_s=gen_s,
    )

    # Lexical grounding: does each content word occur in any member mention?
    haystack = " ".join(member_texts).lower()
    words = _content_words(label)
    score.content_words = len(words)
    if words:
        grounded = sum(1 for w in words if w in haystack or w.rstrip("s") in haystack)
        score.grounded_words = grounded
        score.grounding_rate = round(grounded / len(words), 3)
        score.hallucinated = grounded < len(words)

    # Faithfulness: is this cluster's centroid the nearest one to the label?
    if label_vec is not None and centroids is not None and centroid_ids:
        norms = np.linalg.norm(centroids, axis=1) * np.linalg.norm(label_vec)
        sims = (centroids @ label_vec) / np.maximum(norms, 1e-12)
        own_idx = centroid_ids.index(cluster_id)
        own = float(sims[own_idx])
        others = np.delete(sims, own_idx)
        best_other = float(others.max()) if others.size else float("-inf")
        score.own_sim = round(own, 4)
        score.best_other_sim = round(best_other, 4) if others.size else None
        score.margin = round(own - best_other, 4) if others.size else None
        score.discriminates = own > best_other
    return score


def aggregate(scores: list[LabelScore]) -> dict[str, Any]:
    """Per-model summary over its labels."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    n = len(scores)
    return {
        "n_labels": n,
        "format_compliance": round(sum(s.format_ok for s in scores) / n, 3) if n else None,
        "mean_words": mean([s.n_words for s in scores]),
        "discrimination_at_1": (
            round(sum(bool(s.discriminates) for s in scores) / n, 3) if n else None
        ),
        "mean_own_sim": mean([s.own_sim for s in scores]),
        "mean_margin": mean([s.margin for s in scores]),
        "grounding_rate": mean([s.grounding_rate for s in scores]),
        "hallucination_rate": (
            round(sum(bool(s.hallucinated) for s in scores) / n, 3) if n else None
        ),
        "mean_gen_s": mean([s.gen_s for s in scores]),
    }


def cluster_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Mean embedding per non-noise cluster, plus the matching cluster ids."""
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    mat = np.vstack([embeddings[labels == cid].mean(axis=0) for cid in ids])
    return mat, ids


def render_report(
    per_model: dict[str, dict[str, Any]],
    prompt_hashes: dict[str, str],
    header: str,
    notes: list[str],
) -> str:
    """Markdown comparison table, ranked by a compliance-weighted composite."""
    def fmt(v):
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    # Composite: a label must be well-formed to be usable at all, must identify
    # its own cluster, and must not invent specifics. Equal weights, stated
    # openly as a convention — the raw columns are right there to re-weight.
    def composite(m: dict[str, Any]) -> float:
        parts = [
            m.get("format_compliance") or 0.0,
            m.get("discrimination_at_1") or 0.0,
            1.0 - (m.get("hallucination_rate") or 0.0),
        ]
        return sum(parts) / len(parts)

    ranked = sorted(per_model.items(), key=lambda kv: -composite(kv[1]))
    lines = [
        header, "",
        *notes, "",
        "| # | labeler | prompt | composite | format ok | discrim@1 | mean margin | "
        "grounding | halluc. rate | mean words | s/label |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (key, m) in enumerate(ranked, 1):
        model, variant = key.rsplit("|", 1)
        lines.append(
            f"| {i} | `{model}` | {variant} | {composite(m):.3f} "
            f"| {fmt(m.get('format_compliance'))} | {fmt(m.get('discrimination_at_1'))} "
            f"| {fmt(m.get('mean_margin'))} | {fmt(m.get('grounding_rate'))} "
            f"| {fmt(m.get('hallucination_rate'))} | {fmt(m.get('mean_words'))} "
            f"| {fmt(m.get('mean_gen_s'))} |"
        )
    lines += ["", "Prompt hashes (recorded with every generated label):", ""]
    lines += [f"- `{k}` → `{v}`" for k, v in sorted(prompt_hashes.items())]
    return "\n".join(lines)


def render_human_sheet(
    cluster_ids: list[int],
    sizes: dict[int, int],
    terms: dict[int, list[tuple[str, float]]],
    exemplars: dict[int, list[str]],
    labels_by_model: dict[str, dict[int, str]],
) -> str:
    """Notebook 08's scoring sheet, finally generated.

    Blind-ish by construction: the models are columns, so a reviewer scores the
    labels side by side against the same exemplars rather than reading one
    model's output in isolation and grading on vibes.
    """
    lines = [
        "# LLM label quality — human scoring sheet",
        "",
        "Score each label 1-5 against the exemplar mentions:",
        "**1** = wrong or uselessly generic · **3** = correct but vague · "
        "**5** = specific and correct.",
        "",
        "The automatic metrics cannot see usefulness; this sheet is the "
        "qualitative half of the decision and the record that a human read the "
        "clusters (methodology §8/§9).",
        "",
    ]
    for cid in cluster_ids:
        lines += [
            f"## Cluster {cid} — {sizes.get(cid, 0):,} mentions", "",
            "Top terms: " + ", ".join(w for w, _ in terms.get(cid, [])[:8]),
            "",
            "Exemplar mentions:", "",
        ]
        lines += [f"> {t}" for t in exemplars.get(cid, [])]
        lines += ["", "| labeler | prompt | label | score 1-5 | note |",
                  "|---|---|---|---|---|"]
        for key, labels in labels_by_model.items():
            # The key is "model|variant"; a literal pipe would split the cell.
            model, _, variant = key.rpartition("|")
            label = str(labels.get(cid, "—")).replace("|", "\\|")
            lines.append(f"| `{model}` | {variant} | {label} |  |  |")
        lines.append("")
    return "\n".join(lines)
