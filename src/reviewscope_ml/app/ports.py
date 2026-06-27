"""
Ports the backend implements — the small interfaces that keep the ML package
framework-agnostic. The backend provides a progress sink (writes
``pipeline_jobs`` rows) and, optionally, a repository (persists a
:class:`~reviewscope_ml.app.dto.RunResult`).

The eight canonical pipeline steps (app spec "ML Pipeline (Celery Tasks)") and
the mapping from the runner's internal stage names to them live here so both
the service and the backend agree on the vocabulary the status endpoint reports.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dto import RunResult

# Canonical app-spec steps, in order. ``index`` in ProgressSink.step is 1-8.
PIPELINE_STEPS: tuple[str, ...] = (
    "Ingest", "Preprocess", "Embed", "Reduce",
    "Cluster", "Sentiment", "Label", "Finalize",
)
TOTAL_STEPS = len(PIPELINE_STEPS)

# Runner-internal StageMonitor names -> (step index, step name). Several
# internal stages fold into one canonical step (viz_coords is part of Reduce,
# represent precedes Label, evaluate is part of Finalize).
STAGE_TO_STEP: dict[str, tuple[int, str]] = {
    "embed": (3, "Embed"),
    "reduce": (4, "Reduce"),
    "reduce_cluster": (5, "Cluster"),  # bertopic does reduce+cluster together
    "viz_coords": (4, "Reduce"),
    "cluster": (5, "Cluster"),
    "sentiment": (6, "Sentiment"),
    "represent": (7, "Label"),
    "label": (7, "Label"),
    "evaluate": (8, "Finalize"),
}


@runtime_checkable
class ProgressSink(Protocol):
    """Receives per-step progress. The backend writes each call to ``pipeline_jobs``."""

    def step(self, name: str, status: str, message: str = "",
             index: int = 0, total: int = TOTAL_STEPS) -> None:
        """One progress update.

        Parameters
        ----------
        name   : canonical step name (one of :data:`PIPELINE_STEPS`)
        status : "running" | "done" | "failed"
        index  : 1-based step number (drives "Clustering… step 5/8")
        total  : total step count (:data:`TOTAL_STEPS`)
        """
        ...


@runtime_checkable
class ResultRepository(Protocol):
    """Persists a finished run. Optional — the backend may insert the DTOs directly."""

    def save(self, result: RunResult) -> None: ...


class NullProgress:
    """No-op progress sink — the default when the caller does not supply one."""

    def step(self, name: str, status: str, message: str = "",
             index: int = 0, total: int = TOTAL_STEPS) -> None:  # noqa: D401
        pass
