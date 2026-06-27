"""
Per-stage runtime and memory accounting.

The app's Celery workers need to know what each stage costs (wall time, RAM,
VRAM) before we size worker pools, so every pipeline run records this per
stage in its manifest.

Caveat worth knowing: ``ru_maxrss`` is a process-lifetime high-water mark, so
a stage's "peak RSS" can be inherited from an earlier, hungrier stage. We
report both the global mark and the delta; only a *rising* mark proves the
stage itself was the peak. VRAM peaks are exact per stage (torch resets the
counter between stages).
"""
from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional


def _rss_mb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class StageMonitor:
    """Collects one record per stage; attach ``.records`` to the run manifest.

    ``on_stage`` (optional) is called with the stage name the moment it starts.
    The application's Celery worker passes a callback here to translate stage
    transitions into ``pipeline_jobs`` progress rows (see
    ``reviewscope_ml.app.service``); the experiment notebooks leave it None and
    are unaffected.
    """

    def __init__(self, on_stage: Optional[Callable[[str], None]] = None) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.on_stage = on_stage

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.on_stage is not None:
            self.on_stage(name)
        cuda = _cuda_available()
        if cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        rss_before = _rss_mb()
        t0 = time.time()
        try:
            yield
        finally:
            record: dict[str, Any] = {
                "wall_s": round(time.time() - t0, 2),
                "rss_peak_mb": round(_rss_mb(), 1),
                "rss_delta_mb": round(_rss_mb() - rss_before, 1),
            }
            if cuda:
                import torch

                record["vram_peak_mb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                )
            self.records[name] = record
