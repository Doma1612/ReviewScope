from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.cache import embedding_path, load_array
from ..core.config import PipelineConfig, load_config
from ..pipelines.runner import run_pipeline
from ..pipelines.spec import PipelineSpec
from .defaults import app_default_spec
from .dto import RunResult
from .ingest_upload import UploadedCorpus, reviewset_from_upload
from .persistence import to_records
from .ports import STAGE_TO_STEP, TOTAL_STEPS, NullProgress, ProgressSink
from .schema import UploadSchema


def run_from_upload(
    *,
    file_path: str | Path,
    schema: UploadSchema,
    project_id: str,
    progress: ProgressSink | None = None,
    spec: PipelineSpec | None = None,
    device: str = "cpu",
    seed: int = 42,
    label_clusters: bool = True,
    min_text_len: int = 50,
    label_prompt: str | None = None,
    summary_prompt: str | None = None,
    corpus_profile: str | None = None,
    force: bool = False,
) -> RunResult:
    progress = progress or NullProgress()
    progress.step("Ingest", "running", index=1, total=TOTAL_STEPS)
    corpus = reviewset_from_upload(file_path, schema, min_text_len=min_text_len)
    progress.step(
        "Preprocess", "running", index=2, total=TOTAL_STEPS,
        message=(f"{corpus.n_documents} documents kept "
                 f"({corpus.n_dropped_short} short, "
                 f"{corpus.n_dropped_duplicate} duplicate dropped)"),
    )
    return run_project_pipeline(
        corpus=corpus, project_id=project_id, progress=progress, spec=spec,
        device=device, seed=seed, label_clusters=label_clusters,
        label_prompt=label_prompt, summary_prompt=summary_prompt,
        corpus_profile=corpus_profile, force=force,
    )


def run_project_pipeline(
    *,
    corpus: UploadedCorpus,
    project_id: str,
    progress: ProgressSink | None = None,
    spec: PipelineSpec | None = None,
    device: str = "cpu",
    seed: int = 42,
    label_clusters: bool = True,
    label_prompt: str | None = None,
    summary_prompt: str | None = None,
    corpus_profile: str | None = None,
    force: bool = False,
) -> RunResult:
    progress = progress or NullProgress()
    spec = spec or app_default_spec()
    # Apply prompt overrides as a pair so label and summary contexts cannot disagree.
    if label_prompt and summary_prompt:
        spec = replace(spec, label_prompt=label_prompt, summary_prompt=summary_prompt)
    # Project cache slugs are not corpus identities; pass the profile explicitly.
    if corpus_profile:
        spec = replace(spec, corpus_profile=corpus_profile)
    if corpus.n_documents == 0:
        raise ValueError("no documents to cluster after preprocessing")

    cfg = project_config(project_id, n_docs=corpus.n_documents, device=device, seed=seed)
    cfg.ensure_dirs()

    last_step = {"index": 2}

    def on_stage(stage_name: str) -> None:
        mapped = STAGE_TO_STEP.get(stage_name)
        if mapped is None or mapped[0] == last_step["index"]:
            return
        last_step["index"] = mapped[0]
        progress.step(mapped[1], "running", index=mapped[0], total=TOTAL_STEPS)

    artifacts = run_pipeline(
        cfg, spec,
        reviews=corpus.reviews,
        seed=seed,
        run_name=f"{project_corpus_token(project_id)}__{spec.variant}",
        label_clusters=label_clusters,
        on_stage=on_stage,
        force=force,
    )

    # Match the sentence-unit cache namespace and segment count, not the review count.
    unit = str(artifacts.manifest.get("unit", "document"))
    embeddings = load_run_embeddings(
        cfg, spec, len(artifacts.doc_ids), sentence=(unit == "sentence")
    )

    progress.step("Finalize", "running", index=TOTAL_STEPS, total=TOTAL_STEPS)
    result = to_records(project_id, corpus, artifacts, embeddings)
    progress.step("Finalize", "done", index=TOTAL_STEPS, total=TOTAL_STEPS)
    return result


def project_corpus_token(project_id: str) -> str:
    # Project tokens must be underscore-free for corpus_slug to round-trip.
    safe = re.sub(r"[^a-z0-9]+", "-", project_id.lower()).strip("-") or "x"
    return f"proj-{safe}"


def project_config(
    project_id: str, *, n_docs: int, device: str = "cpu", seed: int = 42
) -> PipelineConfig:
    token = project_corpus_token(project_id)
    return load_config(
        sample_size=n_docs,
        device=device,
        seed=seed,
        data_file=f"sample_{token}_{n_docs}.jsonl",
    )


def load_run_embeddings(
    cfg: PipelineConfig, spec: PipelineSpec, n_units: int, *, sentence: bool = False
) -> np.ndarray:
    corpus = cfg.corpus_slug
    prefix = ("" if corpus == "hotels" else f"{corpus}__") + ("sent__" if sentence else "")
    path = embedding_path(
        cfg.cache_dir, spec.embedding_model, n_units,
        instruction=spec.instruction, prefix=prefix,
    )
    return load_array(path)
