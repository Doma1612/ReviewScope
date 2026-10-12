from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("reviewscope.gpu")


@dataclass(frozen=True)
class GpuStatus:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int
    uuid: str = ""

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def is_busy(self) -> bool:
        return self.memory_used_mb > 1_024 or self.utilization_pct > 10


def query_gpus(timeout_s: float = 10.0) -> list[GpuStatus]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, used, util, uuid = [p.strip() for p in line.split(",")]
        gpus.append(
            GpuStatus(
                index=int(idx),
                name=name,
                memory_total_mb=int(total),
                memory_used_mb=int(used),
                utilization_pct=int(util),
                uuid=uuid,
            )
        )
    return gpus


def pick_idle_gpus(
    gpus: Optional[list[GpuStatus]] = None,
    min_free_mb: int = 6_000,
    max_gpus: Optional[int] = None,
) -> list[int]:
    if gpus is None:
        gpus = query_gpus()
    candidates = sorted(
        (g for g in gpus if not g.is_busy and g.memory_free_mb >= min_free_mb),
        key=lambda g: g.memory_free_mb,
        reverse=True,
    )
    if max_gpus is not None:
        candidates = candidates[:max_gpus]
    return [g.index for g in candidates]


def pick_freest_gpu(
    gpus: Optional[list[GpuStatus]] = None,
    min_free_mb: int = 6_000,
) -> Optional[int]:
    idle = pick_idle_gpus(gpus, min_free_mb=min_free_mb, max_gpus=1)
    return idle[0] if idle else None


class GpuClaim:

    def __init__(self, gpu_ids: list[int], reason: str):
        self.gpu_ids = gpu_ids
        self.reason = reason
        self.claimed_at = datetime.now(timezone.utc)
        self.released_at: Optional[datetime] = None

    @property
    def gpu_id(self) -> Optional[int]:
        return self.gpu_ids[0] if self.gpu_ids else None

    @property
    def device(self) -> str:
        return "cpu" if not self.gpu_ids else "cuda"

    def __enter__(self) -> "GpuClaim":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self.released_at is not None:
            return
        self.released_at = datetime.now(timezone.utc)
        if self.gpu_ids:
            release_cuda_memory()
            held = (self.released_at - self.claimed_at).total_seconds()
            logger.info(
                "released GPU(s) %s at %s (held %.0fs)",
                self.gpu_ids, self.released_at.isoformat(timespec="seconds"), held,
            )


def claim_gpu(
    require_gpu: bool = False,
    min_free_mb: int = 6_000,
    max_gpus: Optional[int] = 1,
) -> GpuClaim:
    # Call before importing torch or initializing CUDA; pass claimed IDs to load_config.
    gpus = query_gpus()
    if not gpus:
        if require_gpu:
            raise RuntimeError("No NVIDIA GPU visible (nvidia-smi unavailable).")
        claim = GpuClaim([], "no NVIDIA driver — CPU fallback")
        logger.info("GPU claim: %s", claim.reason)
        return claim

    for g in gpus:
        logger.info(
            "GPU %d %s: %d/%d MB used, %d%% util%s",
            g.index, g.name, g.memory_used_mb, g.memory_total_mb,
            g.utilization_pct, "  [busy]" if g.is_busy else "",
        )

    chosen_idx = pick_idle_gpus(gpus, min_free_mb=min_free_mb, max_gpus=max_gpus)
    chosen = chosen_idx
    if not chosen:
        msg = (
            "all GPUs busy or below the free-memory floor "
            f"({min_free_mb} MB) — refusing to squeeze in next to other users"
        )
        if require_gpu:
            raise RuntimeError(f"GPU required but {msg}. Retry later or reduce sample_size.")
        claim = GpuClaim([], f"{msg}; CPU fallback")
        logger.warning("GPU claim: %s", claim.reason)
        return claim

    # Pin by UUID: CUDA and nvidia-smi may enumerate device indices differently.
    by_index = {g.index: g for g in gpus}
    uuids = [by_index[i].uuid for i in chosen if by_index[i].uuid]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        ",".join(uuids) if len(uuids) == len(chosen)
        else ",".join(str(i) for i in chosen)
    )
    claim = GpuClaim(chosen, "idle devices")
    logger.info(
        "claimed GPU(s) %s at %s (CUDA_VISIBLE_DEVICES=%s)",
        chosen, claim.claimed_at.isoformat(timespec="seconds"),
        os.environ["CUDA_VISIBLE_DEVICES"],
    )
    return claim


def release_cuda_memory() -> None:
    # Drop model references first; empty_cache only releases unreferenced allocations.
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
