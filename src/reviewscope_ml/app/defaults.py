from __future__ import annotations

from dataclasses import replace

from ..pipelines.spec import PipelineSpec, default_specs

APP_DEFAULT_VARIANT = "sentence_level"

APP_LABEL_MODEL = "gemma3:4b"


def app_default_spec() -> PipelineSpec:
    return replace(default_specs()[APP_DEFAULT_VARIANT], label_model=APP_LABEL_MODEL)
