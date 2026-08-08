"""
The frozen pipeline configuration the application runs.

The experiment harness *compares* five variants; the application *runs one*.
This module is the single source of truth for which one — change it here and
every uploaded project picks up the new default.

Why the sentence-level variant
------------------------------
``sentence_level`` clusters *mentions* (sentence segments), so a review maps to
several clusters and multi-aspect reviews stop averaging their aspects into one
muddy vector — materially better cluster quality. The backend now persists this
shape: segments land in the ``segments`` table, each review keeps a derived
"primary" cluster, and the scatter plots one point per mention (Phase-2, see
docs/integration-guide.md).

Set this back to ``custom_hdbscan`` for the legacy one-cluster-per-document
shape (embed → UMAP → HDBSCAN, mpnet). Existing document-unit projects are
preserved and served read-only; only new runs pick up this default.
"""
from __future__ import annotations

from ..pipelines.spec import PipelineSpec, default_specs

# Sentence-unit variant: reviews are split into mentions, so a review maps to
# several clusters. The backend persists these to the ``segments`` table and the
# per-review membership fan-out (see docs/integration-guide.md Phase-2). Set back
# to ``custom_hdbscan`` for the legacy one-cluster-per-document shape.
APP_DEFAULT_VARIANT = "sentence_level"


def app_default_spec() -> PipelineSpec:
    """Return the frozen pipeline spec the application runs for every project."""
    return default_specs()[APP_DEFAULT_VARIANT]
