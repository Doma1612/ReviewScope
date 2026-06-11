"""
Shim — the implementation lives in ``src/reviewscope_ml/core/cache.py``.

Install the package once from the repo root: ``pip install -e .``
"""
try:
    from reviewscope_ml.core.cache import (  # noqa: F401
        array_exists,
        clustering_path,
        embedding_path,
        load_array,
        make_slug,
        save_array,
        umap_path,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "notebooks/utils now wraps the reviewscope_ml package. "
        "Install it from the repo root first: pip install -e ."
    ) from e
