"""
Centralised pipeline configuration.

This is the single source of truth for compute/data configuration; the
notebook-side ``notebooks/utils/config.py`` re-exports from here so the
experiment notebooks (00-08) and the production-bound package share one
implementation.

Usage:
    from reviewscope_ml import load_config
    cfg = load_config(sample_size=1_000, device="cpu")
    cfg.ensure_dirs()
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


def find_project_root() -> Path:
    """
    Resolve the repository root in a cwd-independent way.

    Order of precedence:
    1. ``REVIEWSCOPE_ROOT`` environment variable (explicit override — used by
       tests and by deployments where the package is installed site-wide).
    2. Upward search from the current working directory for a ``pyproject.toml``
       or ``.git`` marker (covers notebooks/ and repo-root invocations alike).
    3. Upward search from this file (covers the editable install in-repo).
    4. The current working directory as a last resort.
    """
    env = os.environ.get("REVIEWSCOPE_ROOT")
    if env:
        return Path(env).resolve()

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                return candidate
    return Path.cwd()


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
    # share. Check `nvidia-smi` first and set gpu_id to the emptiest device,
    # or use reviewscope_ml.runtime.gpu.claim_gpu() to do it programmatically.
    gpu_id: Optional[int] = None     # physical GPU index to claim (cuda only)
    cuda_mem_fraction: float = 0.5   # hard cap on our share of that GPU's VRAM
    cpu_threads: int = 4             # cap CPU threads (UMAP/HDBSCAN/torch on 32-core box)

    # ── Internal: project root resolved once at construction ──────────────
    _project_root: Path = field(
        default_factory=find_project_root,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """
        Pin compute resources as early as possible.

        These env vars must be set *before* CUDA initialises and before numba
        (pulled in by `import umap`) reads its thread count, so we do it at
        config-construction time — i.e. the first thing any entry point runs.
        """
        self._project_root = Path(self._project_root)

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

        Call this once per entry point *after* torch is importable. It caps
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
    def runs_dir(self) -> Path:
        """End-to-end pipeline run artifacts (one subdirectory per run)."""
        return self._project_root / "data" / "runs"

    @property
    def feedback_dir(self) -> Path:
        """Versioned HITL reviewer feedback (JSONL, one file per review session)."""
        return self._project_root / "data" / "feedback"

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
        if self.sample_size % 1_000 == 0:
            return f"sample_hotels_{self.sample_size // 1000}k.jsonl"
        return f"sample_hotels_{self.sample_size}.jsonl"

    @property
    def data_path(self) -> Path:
        return self.cache_dir / self.data_file

    @property
    def corpus_slug(self) -> str:
        """Corpus identity for cache keys, e.g. 'hotels' or 'automotive'.

        Derived from the benchmark filename (sample_<corpus>_<size>.jsonl).
        Cached artifacts from different corpora at the same sample size must
        never collide — 'hotels' is the historical default and keeps the
        original unprefixed cache filenames.
        """
        stem = Path(self.data_file).stem
        s = stem[len("sample_"):] if stem.startswith("sample_") else stem
        return s.rsplit("_", 1)[0] if "_" in s else s

    def ensure_dirs(self) -> None:
        """Create all cache subdirectories if they don't exist."""
        for d in [
            self.embeddings_dir,
            self.umap_dir,
            self.clustering_dir,
            self.bertopic_dir,
            self.runs_dir,
            self.feedback_dir,
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
    # CPU small-scale (local dev / smoke test)
    cfg = load_config(sample_size=1_000, device="cpu", batch_size=32)

    # GPU large-scale (shared university server) — pin to ONE free GPU.
    # data_file derives from sample_size → "sample_hotels_50k.jsonl".
    cfg = load_config(sample_size=50_000, device="cuda", gpu_id=2,
                      batch_size=256, cuda_mem_fraction=0.5)
    """
    if "data_file" in overrides:
        overrides["_data_file"] = overrides.pop("data_file")
    if "project_root" in overrides:
        overrides["_project_root"] = Path(overrides.pop("project_root"))
    return PipelineConfig(**overrides)


# ── Preprocessors ─────────────────────────────────────────────────────────────

def get_preprocessor(name: str = "minimal") -> Callable[[str], str]:
    """
    Return the preprocessing function decided in 02_preprocessing.ipynb.

    "minimal" (whitespace normalisation only) won the notebook comparison:
    sentence-transformer models are trained on natural text, so aggressive
    lowercasing/punctuation-stripping removes signal they can use.

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
