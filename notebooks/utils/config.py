"""
Centralised pipeline configuration.

Usage:
    from utils import load_config
    cfg = load_config(sample_size=1_000, device="cpu")
    cfg.ensure_dirs()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    sample_size: int = 5_000
    data_file: str = "sample_hotels_10k.jsonl"
    seed: int = 42

    # ── Compute ───────────────────────────────────────────────────────────
    device: str = "cpu"       # "cpu" | "cuda"
    batch_size: int = 64

    # ── Internal: project root resolved once at construction ──────────────
    _project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        repr=False,
        compare=False,
    )

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
        return (
            f"PipelineConfig(\n"
            f"  sample_size  = {self.sample_size:,}\n"
            f"  data_file    = {self.data_file}\n"
            f"  device       = {self.device}\n"
            f"  batch_size   = {self.batch_size}\n"
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

    # GPU large-scale (university server)
    cfg = load_config(sample_size=50_000, device="cuda", batch_size=512,
                      data_file="sample_hotels_50k.jsonl")
    """
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
