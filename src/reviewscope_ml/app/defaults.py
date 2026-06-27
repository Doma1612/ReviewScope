"""
The frozen pipeline configuration the application runs.

The experiment harness *compares* five variants; the application *runs one*.
This module is the single source of truth for which one — change it here and
every uploaded project picks up the new default.

Why a document-level variant (not the better-scoring sentence_level)
--------------------------------------------------------------------
The app spec's database is one-cluster-per-document and one-point-per-document.
``sentence_level`` clusters *mentions* (sentence segments), so a document maps
to several clusters via ``doc_membership.json`` and the scatter shows segments,
not documents — that needs a ``segments`` table the spec does not have. For a
clean 1:1 mapping to the spec's tables the application defaults to
``custom_hdbscan`` (embed → UMAP → HDBSCAN, mpnet, notebook-decided params).

Sentence-level support is a documented Phase-2 extension (add a segments table
+ per-segment scatter); :func:`reviewscope_ml.app.service.run_project_pipeline`
rejects it explicitly until then.

Swapping the winner later (after the doc-level sweep / human sign-off, see
docs/quality-roadmap.md) is a one-line change to ``APP_DEFAULT_VARIANT``.
"""
from __future__ import annotations

from ..pipelines.spec import PipelineSpec, default_specs

# Document-unit variant whose artifacts map 1:1 onto the spec's DB tables.
APP_DEFAULT_VARIANT = "custom_hdbscan"


def app_default_spec() -> PipelineSpec:
    """Return the frozen pipeline spec the application runs for every project."""
    return default_specs()[APP_DEFAULT_VARIANT]
