# ru_maxrss is a process-lifetime high-water mark, not an independent stage peak.
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
