"""
Shim — the implementation lives in ``src/reviewscope_ml/core/tracking.py``.

Install the package once from the repo root: ``pip install -e .``
"""
try:
    from reviewscope_ml.core.tracking import (  # noqa: F401
        RESULTS_COLUMNS,
        load_results,
        log_result,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "notebooks/utils now wraps the reviewscope_ml package. "
        "Install it from the repo root first: pip install -e ."
    ) from e
