"""
Cache helpers: deterministic path generation + save/load for numpy arrays.

Naming convention
-----------------
  embeddings/{model_slug}__{instr_slug}__{k}k.npy
  umap/{prefix}{model_slug}__{instr_slug}__nc{n}__nn{n}__md{f}__{metric}__{k}k.npy
  clustering/{algo}__{params_slug}__{umap_slug}__{k}k.npy

All slugs are filesystem-safe lowercase strings.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ── Slug helpers ──────────────────────────────────────────────────────────────

def make_slug(s: str) -> str:
    """
    Convert an arbitrary string to a lowercase, filesystem-safe slug.

    Examples
    --------
    make_slug("intfloat/multilingual-e5-large-instruct") -> "multilingual-e5-large-instruct"
    make_slug("hkunlp/instructor-large")                 -> "instructor-large"
    make_slug("all-MiniLM-L6-v2")                        -> "all-minilm-l6-v2"
    """
    s = s.split("/")[-1]                        # strip HF org prefix
    s = re.sub(r"[^a-zA-Z0-9_\-]", "_", s)     # replace special chars
    return s.lower()


def _k(sample_size: int) -> str:
    """Format sample size as compact string: 5000 -> '5k', 500 -> '500'."""
    if sample_size % 1_000 == 0:
        return f"{sample_size // 1_000}k"
    return str(sample_size)


# ── Path builders ─────────────────────────────────────────────────────────────

def embedding_path(
    base_dir: Path,
    model_name: str,
    sample_size: int,
    instruction: str = "no_inst",
    prefix: str = "",
) -> Path:
    """
    Path for a cached embedding array.

    Parameters
    ----------
    instruction : slug identifying which instruction was used, e.g.
                  "no_inst", "generic", "domain", "sentiment"
    prefix : corpus namespace for non-hotel benchmarks (e.g. "automotive__");
             empty for the original hotel benchmark so existing caches and
             notebook conventions stay valid.
    """
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
    """
    Path for a cached UMAP projection.

    Parameters
    ----------
    prefix : prepended to the filename for variants, e.g.
             "pca50_" for PCA→UMAP, "viz_" for 2-D visualisation projection
    """
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
    """
    Path for a cached cluster-label array.

    Parameters
    ----------
    params_slug : algo-specific parameter string, e.g.
                  "mcs15__ms5" for HDBSCAN, "k15" for KMeans,
                  "k15__ward" for Agglomerative
    umap_slug   : identifies the UMAP config used as input, e.g.
                  "all-mpnet-base-v2__no_inst__nc10__nn15"
    """
    name = f"{algorithm}__{params_slug}__{umap_slug}__{_k(sample_size)}.npy"
    return base_dir / "clustering" / name


# ── IO helpers ────────────────────────────────────────────────────────────────

def save_array(path: Path, array: np.ndarray) -> None:
    """Save a numpy array; creates parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"  [saved]  {path.name}  shape={array.shape}  dtype={array.dtype}")


def load_array(path: Path) -> np.ndarray:
    """
    Load a numpy array from *path*.

    Raises
    ------
    FileNotFoundError
        With a helpful message pointing to the upstream notebook.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n  [cache miss] {path}\n"
            f"  Run the upstream notebook to generate this file first.\n"
        )
    arr = np.load(path)
    print(f"  [loaded] {path.name}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def array_exists(path: Path) -> bool:
    """Return True if a cached array file exists on disk."""
    return path.exists()
