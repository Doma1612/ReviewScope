"""
Centralised pipeline configuration.

Usage:
    from utils import load_config
    cfg = load_config(sample_size=1_000, device="cpu")
    cfg.ensure_dirs()
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    sample_size: int = 5_000
    seed: int = 42
    # Benchmark file is normally derived from sample_size (see the `data_file`
    # property), so the two can never disagree. Pass `data_file=...` to
    # load_config only to force a specific filename.
    _data_file: Optional[str] = field(default=None, repr=False, compare=False)

    # ── Compute ───────────────────────────────────────────────────────────
    device: str = "cpu"       # "cpu" | "cuda"
    batch_size: int = 64

    # ── Shared-server etiquette ───────────────────────────────────────────
    # On the shared university box (no SLURM/PBS scheduler) we must pin
    # ourselves to ONE GPU and cap our footprint so other users keep their
    # share. Check `nvidia-smi` first and set gpu_id to the emptiest device.
    gpu_id: Optional[int] = None     # physical GPU index to claim (cuda only)
    cuda_mem_fraction: float = 0.5   # hard cap on our share of that GPU's VRAM
    cpu_threads: int = 4             # cap CPU threads (UMAP/HDBSCAN/torch on 32-core box)

    # ── Internal: project root resolved once at construction ──────────────
    _project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """
        Pin compute resources as early as possible.

        These env vars must be set *before* CUDA initialises and before numba
        (pulled in by `import umap`) reads its thread count, so we do it at
        config-construction time — i.e. the first thing the setup cell runs.
        """
        if self.device == "cuda" and self.gpu_id is not None:
            # Make every other GPU invisible to this process: we physically
            # cannot touch our neighbours' devices, and ours becomes "cuda:0".
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Be a good citizen on the shared 32-core CPU (UMAP/HDBSCAN/BLAS/numba).
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.setdefault(var, str(self.cpu_threads))

    def apply_runtime_limits(self) -> str:
        """
        Apply torch-level caps and return the device string to hand to models.

        Call this once in the setup cell *after* torch is importable. It caps
        our slice of GPU VRAM and torch's CPU thread pool, then returns
        "cuda" / "cpu" for use as `SentenceTransformer(device=...)`.
        """
        try:
            import torch
        except ImportError:
            return self.device

        torch.set_num_threads(self.cpu_threads)

        if self.device == "cuda" and torch.cuda.is_available():
            # After CUDA_VISIBLE_DEVICES pinning, our GPU is index 0.
            torch.cuda.set_per_process_memory_fraction(self.cuda_mem_fraction, 0)
            return "cuda"
        return "cpu"

    # ── Derived paths ─────────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def cache_dir(self) -> Path:
        return self._project_root / "data" / "cache"

    @property
    def embeddings_dir(self) -> Path:
        return self.cache_dir / "embeddings"

    @property
    def umap_dir(self) -> Path:
        return self.cache_dir / "umap"

    @property
    def clustering_dir(self) -> Path:
        return self.cache_dir / "clustering"

    @property
    def bertopic_dir(self) -> Path:
        return self.cache_dir / "bertopic"

    @property
    def results_csv(self) -> Path:
        return self.cache_dir / "results.csv"

    @property
    def data_file(self) -> str:
        """Benchmark filename for this sample_size (e.g. 'sample_hotels_5k.jsonl').

        Derived from sample_size so the displayed name always matches the rows
        actually loaded; pass `data_file=...` to load_config to override.
        """
        if self._data_file is not None:
            return self._data_file
        return f"sample_hotels_{self.sample_size // 1000}k.jsonl"

    @property
    def data_path(self) -> Path:
        return self.cache_dir / self.data_file

    def ensure_dirs(self) -> None:
        """Create all cache subdirectories if they don't exist."""
        for d in [
            self.embeddings_dir,
            self.umap_dir,
            self.clustering_dir,
            self.bertopic_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        gpu = "—" if self.gpu_id is None else f"{self.gpu_id} (mem≤{self.cuda_mem_fraction:.0%})"
        return (
            f"PipelineConfig(\n"
            f"  sample_size  = {self.sample_size:,}\n"
            f"  data_file    = {self.data_file}\n"
            f"  device       = {self.device}\n"
            f"  gpu_id       = {gpu}\n"
            f"  batch_size   = {self.batch_size}\n"
            f"  cpu_threads  = {self.cpu_threads}\n"
            f"  seed         = {self.seed}\n"
            f"  cache_dir    = {self.cache_dir}\n"
            f")"
        )


def load_config(**overrides) -> PipelineConfig:
    """
    Return a PipelineConfig, optionally overriding any field.

    Examples
    --------
    # CPU small-scale (local dev)
    cfg = load_config(sample_size=1_000, device="cpu", batch_size=32)

    # GPU large-scale (shared university server) — pin to ONE free GPU.
    # data_file derives from sample_size → "sample_hotels_50k.jsonl".
    cfg = load_config(sample_size=50_000, device="cuda", gpu_id=2,
                      batch_size=256, cuda_mem_fraction=0.5)
    """
    if "data_file" in overrides:
        overrides["_data_file"] = overrides.pop("data_file")
    return PipelineConfig(**overrides)


# ── Preprocessors ─────────────────────────────────────────────────────────────

def get_preprocessor(name: str = "minimal") -> Callable[[str], str]:
    """
    Return the preprocessing function decided in 02_preprocessing.ipynb.

    Parameters
    ----------
    name : "raw" | "minimal" | "aggressive"
    """
    if name == "raw":
        return lambda t: t.strip()
    if name == "minimal":
        return lambda t: re.sub(r"\s+", " ", t.strip())
    if name == "aggressive":
        return lambda t: re.sub(
            r"\s+", " ",
            re.sub(r"[^a-z0-9\s]", " ", t.lower()),
        ).strip()
    raise ValueError(f"Unknown preprocessor '{name}'. Choose: raw, minimal, aggressive.")
