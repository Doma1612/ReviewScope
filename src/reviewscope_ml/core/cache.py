from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def make_slug(s: str) -> str:
    s = s.split("/")[-1]
    s = re.sub(r"[^a-zA-Z0-9_\-]", "_", s)
    return s.lower()


def _k(sample_size: int) -> str:
    if sample_size % 1_000 == 0:
        return f"{sample_size // 1_000}k"
    return str(sample_size)


def embedding_path(
    base_dir: Path,
    model_name: str,
    sample_size: int,
    instruction: str = "no_inst",
    prefix: str = "",
) -> Path:
    name = f"{prefix}{make_slug(model_name)}__{make_slug(instruction)}__{_k(sample_size)}.npy"
    return base_dir / "embeddings" / name


def umap_path(
    base_dir: Path,
    model_name: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    sample_size: int,
    instruction: str = "no_inst",
    prefix: str = "",
) -> Path:
    md_str = f"{min_dist:.2f}".replace(".", "")
    name = (
        f"{prefix}{make_slug(model_name)}__{make_slug(instruction)}"
        f"__nc{n_components}__nn{n_neighbors}__md{md_str}__{metric}"
        f"__{_k(sample_size)}.npy"
    )
    return base_dir / "umap" / name


def clustering_path(
    base_dir: Path,
    algorithm: str,
    params_slug: str,
    umap_slug: str,
    sample_size: int,
) -> Path:
    name = f"{algorithm}__{params_slug}__{umap_slug}__{_k(sample_size)}.npy"
    return base_dir / "clustering" / name


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"  [saved]  {path.name}  shape={array.shape}  dtype={array.dtype}")


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"\n  [cache miss] {path}\n"
            f"  Run the upstream notebook to generate this file first.\n"
        )
    arr = np.load(path)
    print(f"  [loaded] {path.name}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def array_exists(path: Path) -> bool:
    return path.exists()
