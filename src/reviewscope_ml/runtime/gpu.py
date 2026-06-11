"""
Shared-GPU-server etiquette, enforced in code.

The university box has 4x TITAN X Pascal (12 GB) and NO scheduler — fairness
is courtesy only, and other groups are usually active on GPUs 0 and 1. Every
entry point that may touch CUDA must go through :func:`claim_gpu` so that:

1. we query ``nvidia-smi`` and pick the *emptiest* device, never a fixed one;
2. we pin the process to that ONE device via ``CUDA_VISIBLE_DEVICES``;
3. we refuse to start if every GPU is busy — falling back to CPU (or telling
   the caller to use a smaller sample) instead of squeezing in next to a
   neighbour's job;
4. the claim and the release are logged, so anyone reading our logs (or `who`
   is on the box) can see what we held and when we let it go.

Release is equally important: a finished stage must drop its model references
and empty the CUDA cache (:func:`release_gpu`) instead of squatting on VRAM
from a long-lived process.
"""
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
        """Heuristic for "someone is actively using this device".

        >1 GB resident memory or any sustained utilisation counts as busy:
        idle CUDA contexts hold ~100-200 MB, so a 1 GB threshold separates
        parked sessions from real jobs without being trigger-happy.
        """
        return self.memory_used_mb > 1_024 or self.utilization_pct > 10


def query_gpus(timeout_s: float = 10.0) -> list[GpuStatus]:
    """
    Query ``nvidia-smi`` programmatically. Returns [] when no NVIDIA driver
    is present (laptops, CI) so callers can fall back to CPU cleanly.
    """
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
    """
    Return the indices of claimable GPUs, emptiest first.

    A device qualifies only if it is not busy (see :meth:`GpuStatus.is_busy`)
    AND has at least *min_free_mb* free — we scale onto *idle* devices, never
    next to a neighbour's job. ``max_gpus=None`` means "all idle devices"
    (the data-parallel embed stage speeds up near-linearly); ``max_gpus=1``
    is the conservative single-device claim.
    """
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
    """Single emptiest claimable GPU, or None (see :func:`pick_idle_gpus`)."""
    idle = pick_idle_gpus(gpus, min_free_mb=min_free_mb, max_gpus=1)
    return idle[0] if idle else None


class GpuClaim:
    """
    A logged, released claim on one or more idle GPUs (or a CPU fallback).

    Use as a context manager so release happens even on crashes::

        with claim_gpu(max_gpus=None) as claim:   # all idle devices
            cfg = load_config(device=claim.device, gpu_ids=claim.gpu_ids, ...)
            ...
    """

    def __init__(self, gpu_ids: list[int], reason: str):
        self.gpu_ids = gpu_ids
        self.reason = reason
        self.claimed_at = datetime.now(timezone.utc)
        self.released_at: Optional[datetime] = None

    @property
    def gpu_id(self) -> Optional[int]:
        """First claimed device — kept for single-GPU call sites."""
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
    """
    Select and pin idle GPU(s); fall back to CPU if all are busy.

    Sets ``CUDA_VISIBLE_DEVICES`` immediately, so call this BEFORE importing
    torch or anything that initialises CUDA. The selected ids must still be
    passed to ``load_config(gpu_ids=...)`` so the VRAM fraction cap applies.

    Parameters
    ----------
    require_gpu : if True, raise RuntimeError instead of falling back to CPU.
                  Use for stages where a CPU run would silently take hours;
                  the right reaction to a full box is to come back later or
                  run a smaller sample — not to squeeze in.
    max_gpus :    how many idle devices to claim. 1 = conservative default;
                  None = every idle device (the embed stage parallelises
                  across them). Busy devices are never claimed regardless.
    """
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

    # Pin by UUID, not index: nvidia-smi indices are PCI-bus order while CUDA
    # runtimes may enumerate "fastest first" — observed live: a process told
    # CUDA_VISIBLE_DEVICES=3 loaded onto physical GPU 2. UUIDs are immune to
    # any ordering and are accepted by CUDA_VISIBLE_DEVICES everywhere.
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
    """Free cached CUDA memory after a stage finishes.

    Callers must additionally drop their own references (``del model``) first;
    this only empties the allocator cache for whatever is no longer referenced.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
