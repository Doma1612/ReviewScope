"""
Application service — the one function the backend's Celery worker calls.

``run_from_upload`` is the full path: ingest an uploaded file (steps 1-2),
run the frozen pipeline (steps 3-7), map the result to DB records (step 8),
reporting progress to the supplied :class:`ProgressSink` throughout.
``run_project_pipeline`` is the same minus ingest, for callers that already
hold an :class:`UploadedCorpus`.

Everything heavy is delegated to the existing
:func:`reviewscope_ml.pipelines.runner.run_pipeline`; this module only adapts
its input (arbitrary upload instead of the Yelp benchmark), surfaces the
embedding vectors for pgvector, and translates stage transitions into the
app's eight-step progress vocabulary.
"""
from __future__ import annotations

import re
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
) -> RunResult:
    """Ingest an uploaded file and run the full pipeline -> persistence DTOs.

    Raises :class:`~reviewscope_ml.app.ingest_upload.IngestError` if the file
    fails schema validation (the backend maps that to a 4xx + failed status).
    """
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
) -> RunResult:
    """Run the frozen pipeline on an already-ingested corpus -> persistence DTOs."""
    progress = progress or NullProgress()
    spec = spec or app_default_spec()
    if spec.variant == "sentence_level":
        raise ValueError(
            "sentence_level clusters mentions, not documents, and does not map "
            "onto the spec's one-cluster-per-document schema; see "
            "docs/integration-guide.md (Phase-2 extension)."
        )
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
    )

    embeddings = load_run_embeddings(cfg, spec, corpus.n_documents)

    progress.step("Finalize", "running", index=TOTAL_STEPS, total=TOTAL_STEPS)
    result = to_records(project_id, corpus, artifacts, embeddings)
    progress.step("Finalize", "done", index=TOTAL_STEPS, total=TOTAL_STEPS)
    return result


# ── Per-project config / cache plumbing ───────────────────────────────────────

def project_corpus_token(project_id: str) -> str:
    """Filesystem- and cache-safe corpus token for a project.

    Underscore-free (only ``[a-z0-9-]``) so ``PipelineConfig.corpus_slug`` —
    which splits the benchmark filename on the last underscore — round-trips it
    cleanly. Never equals 'hotels', so project caches never collide with the
    research benchmark's.
    """
    safe = re.sub(r"[^a-z0-9]+", "-", project_id.lower()).strip("-") or "x"
    return f"proj-{safe}"


def project_config(
    project_id: str, *, n_docs: int, device: str = "cpu", seed: int = 42
) -> PipelineConfig:
    """A PipelineConfig whose caches and run directory are namespaced per project.

    ``data_file`` is synthetic — the pipeline receives its data in memory, so
    the file is never read; it exists only to give ``corpus_slug`` a unique,
    stable value that keys every cached artifact to this project.
    """
    token = project_corpus_token(project_id)
    return load_config(
        sample_size=n_docs,
        device=device,
        seed=seed,
        data_file=f"sample_{token}_{n_docs}.jsonl",
    )


def load_run_embeddings(
    cfg: PipelineConfig, spec: PipelineSpec, n_docs: int
) -> np.ndarray:
    """Reload the embedding matrix the run just cached (no recompute, no model load).

    Mirrors ``embed.embed_with_cache``'s key for document-level units, so this
    hits the file ``run_pipeline`` wrote during its embed stage.
    """
    corpus = cfg.corpus_slug
    prefix = "" if corpus == "hotels" else f"{corpus}__"
    path = embedding_path(
        cfg.cache_dir, spec.embedding_model, n_docs,
        instruction=spec.instruction, prefix=prefix,
    )
    return load_array(path)
