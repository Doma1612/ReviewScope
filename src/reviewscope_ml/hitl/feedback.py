"""
HITL feedback records: the contract between the review GUI and the pipeline.

The GUI is replaceable (today Streamlit, later the React app); this JSONL
format is not. One record per reviewer action, append-only, one file per
review session in ``data/feedback/`` — versioned by filename timestamp, never
rewritten, so the full review history stays auditable.

Where humans are needed (and the action that captures each decision):

- **LLM label approval** (``approve_label`` / ``rename_label``): labels are
  generated from 5 sampled docs; hallucinated or too-generic labels are
  expected and must be caught by a person.
- **Cluster merge decisions** (``merge_clusters``): near-duplicate clusters
  (flagged by term overlap) need a human to say "same theme".
- **Cluster split decisions** (``split_cluster``): "this cluster mixes two
  themes" is exactly what the metrics cannot see.
- **Outlier/noise triage** (``reassign_doc``): misfiled documents, or noise
  documents that clearly belong somewhere.
- **Junk calls** (``mark_junk``): clusters defined by artifacts (length,
  boilerplate) rather than content.
- **Run confirmation** (``confirm_run``): the recorded human sign-off that
  the winning pipeline's clusters are thematically coherent.
- **Model confirmation** (``confirm_model``): the recorded human sign-off on a
  *technology-selection* candidate — that a reviewer read the clusters a given
  embedding model produces and not merely its metrics. Kept in the same
  append-only trail as everything else so the selection rationale can cite a
  reviewer and a timestamp rather than an assertion. Unlike the other actions
  this one describes a candidate rather than a cluster, so ``apply_feedback``
  ignores it — it is evidence, not a mutation.
- **Label scoring** (``score_label``): one reviewer's 1-5 usefulness rating of
  one candidate label for one cluster. The automatic labeler metrics measure
  form (parses, discriminates, grounded in member text); none of them measures
  whether the label *tells a reader what the cluster is about*. That judgement
  only exists in a person, so it has to be recorded like any other human
  decision. Like ``confirm_model`` this is evidence rather than a mutation and
  ``apply_feedback`` ignores it — but unlike it, several reviewers are expected
  to score the same labels independently, which is what makes disagreement
  measurable instead of invisible.

Application semantics on re-run live in ``apply_feedback.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ACTIONS = (
    "approve_label",
    "rename_label",
    "merge_clusters",
    "split_cluster",
    "reassign_doc",
    "mark_junk",
    "confirm_run",
    "confirm_model",
    "score_label",
)


@dataclass
class FeedbackRecord:
    run_name: str
    reviewer: str
    action: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    cluster_id: Optional[int] = None          # subject of the action
    new_label: Optional[str] = None           # rename_label
    merge_into: Optional[int] = None          # merge_clusters: target cluster
    doc_id: Optional[str] = None              # reassign_doc
    target_cluster_id: Optional[int] = None   # reassign_doc destination (-1 = noise)
    note: Optional[str] = None
    # score_label. `label` is the scored string itself rather than a labeler
    # name because identical labels from different models are one judgement:
    # scoring the string keeps a reviewer from grading the same words twice and
    # (more importantly) from grading them differently.
    label: Optional[str] = None
    score: Optional[int] = None               # 1-5 usefulness

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown action {self.action!r}; known: {ACTIONS}")
        if self.action == "score_label":
            # An out-of-range score would average into the per-labeler means
            # silently, so it is rejected at the point of writing rather than
            # discovered later in an aggregate nobody can reproduce.
            if self.score is None or not 1 <= int(self.score) <= 5:
                raise ValueError(
                    f"score_label needs a score in 1-5, got {self.score!r}"
                )
            if not self.label:
                raise ValueError("score_label needs the label it scored")

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


def session_file(feedback_dir: Path, run_name: str) -> Path:
    """One JSONL file per (run, session-start) — the filename is the version."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return feedback_dir / f"{run_name}__{stamp}.jsonl"


def append_record(path: Path, record: FeedbackRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.to_json() + "\n")


def load_feedback(feedback_dir: Path, run_name: str) -> list[FeedbackRecord]:
    """
    All records for a run across all session files, in timestamp order.
    Later records win when actions conflict (e.g. two renames of one cluster).
    """
    records: list[FeedbackRecord] = []
    if not feedback_dir.exists():
        return records
    for path in sorted(feedback_dir.glob(f"{run_name}__*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            data: dict[str, Any] = json.loads(line)
            records.append(FeedbackRecord(**data))
    records.sort(key=lambda r: r.timestamp)
    return records
