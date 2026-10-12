from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


def find_project_root() -> Path:
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
    sample_size: int = 5_000
    seed: int = 42
    _data_file: Optional[str] = field(default=None, repr=False, compare=False)

    device: str = "cpu"
    batch_size: int = 64

    gpu_id: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
    cuda_mem_fraction: float = 0.5
    cpu_threads: int = 4

    _project_root: Path = field(
        default_factory=find_project_root,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._project_root = Path(self._project_root)

        if self.device == "cuda":
            # Preserve an existing UUID pin; CUDA and nvidia-smi index ordering can differ.
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            if self.gpu_ids:
                os.environ.setdefault(
                    "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in self.gpu_ids)
                )
            elif self.gpu_id is not None:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))

        # Set before CUDA initialization to reduce fragmentation from variable-length batches.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.setdefault(var, str(self.cpu_threads))

    def apply_runtime_limits(self) -> str:
        try:
            import torch
        except ImportError:
            return self.device

        torch.set_num_threads(self.cpu_threads)

        if self.device == "cuda" and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(self.cuda_mem_fraction, i)
            return "cuda"
        return "cpu"

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
        return self._project_root / "data" / "runs"

    @property
    def feedback_dir(self) -> Path:
        return self._project_root / "data" / "feedback"

    @property
    def results_csv(self) -> Path:
        return self.cache_dir / "results.csv"

    @property
    def data_file(self) -> str:
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
        stem = Path(self.data_file).stem
        s = stem[len("sample_"):] if stem.startswith("sample_") else stem
        return s.rsplit("_", 1)[0] if "_" in s else s

    def ensure_dirs(self) -> None:
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
    if "data_file" in overrides:
        overrides["_data_file"] = overrides.pop("data_file")
    if "project_root" in overrides:
        overrides["_project_root"] = Path(overrides.pop("project_root"))
    return PipelineConfig(**overrides)


def get_preprocessor(name: str = "minimal") -> Callable[[str], str]:
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
