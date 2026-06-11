"""
Shim — the implementation lives in ``src/reviewscope_ml/core/config.py``.

The notebooks and the production-bound package share one config so a decision
recorded here is automatically the decision the backend uses. Install the
package once from the repo root before running notebooks:

    pip install -e .
"""
try:
    from reviewscope_ml.core.config import (  # noqa: F401
        PipelineConfig,
        find_project_root,
        get_preprocessor,
        load_config,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "notebooks/utils now wraps the reviewscope_ml package. "
        "Install it from the repo root first: pip install -e ."
    ) from e
