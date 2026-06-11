"""
ReviewScope ML pipeline package.

Pure-Python, notebook-free implementation of the WP5 pipeline:
ingest -> preprocess -> embed -> reduce -> cluster -> represent -> label,
plus the evaluation harness (WP5) and HITL review loop, structured so the
FastAPI/Celery backend can import each stage directly.

Keep this package free of Jupyter dependencies: the experiment notebooks
orchestrate it, they are not part of it.
"""
from .core import (
    RESULTS_COLUMNS,
    PipelineConfig,
    array_exists,
    clustering_path,
    compute_coherence,
    compute_metrics,
    compute_rating_entropy,
    embedding_path,
    find_project_root,
    get_preprocessor,
    load_array,
    load_config,
    load_results,
    log_result,
    make_slug,
    save_array,
    umap_path,
)

__all__ = [
    "PipelineConfig", "find_project_root", "get_preprocessor", "load_config",
    "make_slug", "embedding_path", "umap_path", "clustering_path",
    "save_array", "load_array", "array_exists",
    "compute_metrics", "compute_coherence", "compute_rating_entropy",
    "log_result", "load_results", "RESULTS_COLUMNS",
]
