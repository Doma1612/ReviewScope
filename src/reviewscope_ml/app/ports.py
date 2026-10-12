from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dto import RunResult

PIPELINE_STEPS: tuple[str, ...] = (
    "Ingest", "Preprocess", "Embed", "Reduce",
    "Cluster", "Sentiment", "Label", "Finalize",
)
TOTAL_STEPS = len(PIPELINE_STEPS)

STAGE_TO_STEP: dict[str, tuple[int, str]] = {
    "embed": (3, "Embed"),
    "reduce": (4, "Reduce"),
    "reduce_cluster": (5, "Cluster"),
    "viz_coords": (4, "Reduce"),
    "cluster": (5, "Cluster"),
    "sentiment": (6, "Sentiment"),
    "represent": (7, "Label"),
    "label": (7, "Label"),
    "evaluate": (8, "Finalize"),
}


@runtime_checkable
class ProgressSink(Protocol):

    def step(self, name: str, status: str, message: str = "",
             index: int = 0, total: int = TOTAL_STEPS) -> None:
        ...


@runtime_checkable
class ResultRepository(Protocol):

    def save(self, result: RunResult) -> None: ...


class NullProgress:

    def step(self, name: str, status: str, message: str = "",
             index: int = 0, total: int = TOTAL_STEPS) -> None:
        pass
