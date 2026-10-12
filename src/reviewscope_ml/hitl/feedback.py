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
    cluster_id: Optional[int] = None
    new_label: Optional[str] = None
    merge_into: Optional[int] = None
    doc_id: Optional[str] = None
    target_cluster_id: Optional[int] = None
    note: Optional[str] = None
    label: Optional[str] = None
    score: Optional[int] = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown action {self.action!r}; known: {ACTIONS}")
        if self.action == "score_label":
            if self.score is None or not 1 <= int(self.score) <= 5:
                raise ValueError(
                    f"score_label needs a score in 1-5, got {self.score!r}"
                )
            if not self.label:
                raise ValueError("score_label needs the label it scored")

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


def session_file(feedback_dir: Path, run_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return feedback_dir / f"{run_name}__{stamp}.jsonl"


def append_record(path: Path, record: FeedbackRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.to_json() + "\n")


def load_feedback(feedback_dir: Path, run_name: str) -> list[FeedbackRecord]:
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
