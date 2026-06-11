"""
End-to-end pipeline runner.

One entry point, four variants (see ``spec.py``), one artifact schema
(see ``artifacts.py``). Heavy intermediates (embeddings, projections, label
arrays) go through the same on-disk cache the notebooks use, which makes runs
checkpointed and resumable: a crash after the embed stage costs only the
stages after embed, and a re-run with identical parameters is almost free.

Shared-GPU etiquette is handled at the edges: the embed stage is the only
CUDA consumer; its model is dropped and the CUDA cache emptied the moment the
embeddings exist (or come from cache). UMAP/HDBSCAN run on CPU with the
thread cap from PipelineConfig.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.cache import clustering_path, load_array, make_slug, save_array, umap_path
from ..core.config import PipelineConfig
from ..core.tracking import log_result
from ..cluster import (
    AgglomerativeBackend,
    HDBSCANBackend,
    TwoStageBackend,
    params_slug,
)
from ..data.ingest import ReviewSet, load_benchmark
from ..embed import SentenceTransformerEmbedder, embed_with_cache
from ..eval.harness import evaluate_labels, failure_flags
from ..label import OllamaLabeler
from ..reduce import UMAPReducer, viz_projection
from ..represent import ctfidf_terms, tfidf_top_terms, word_frequencies
from ..runtime.monitor import StageMonitor
from .artifacts import ClusterInfo, RunArtifacts, run_is_complete, save_run
from .spec import PipelineSpec

logger = logging.getLogger("reviewscope.pipeline")

N_SAMPLE_DOCS = 5  # random per-cluster samples stored in the artifact


def run_pipeline(
    cfg: PipelineConfig,
    spec: PipelineSpec,
    reviews: Optional[ReviewSet] = None,
    seed: Optional[int] = None,
    run_name: Optional[str] = None,
    force: bool = False,
    compute_coherence: bool = True,
    label_clusters: bool = True,
) -> RunArtifacts:
    """
    Run one pipeline variant end to end and persist its artifacts.

    ``seed`` overrides ``cfg.seed`` (used by the multi-seed stability check);
    seeded intermediates get a distinct cache key so notebook caches (seed 42)
    are never clobbered.
    """
    seed = cfg.seed if seed is None else seed
    if run_name is None:
        run_name = f"{spec.variant}__{cfg.sample_size}__s{seed}"
        # Embedding-model sweeps: non-default models get their own run dirs,
        # otherwise a sweep would silently overwrite the default-model run.
        from .spec import default_specs

        if spec.embedding_model != default_specs()[spec.variant].embedding_model:
            run_name = (
                f"{spec.variant}__{make_slug(spec.embedding_model)}"
                f"__{cfg.sample_size}__s{seed}"
            )
    run_dir = cfg.runs_dir / run_name

    from ..pipelines.artifacts import load_run

    if run_is_complete(run_dir) and not force:
        logger.info("run %s already complete — loading artifacts", run_name)
        return load_run(run_dir)

    cfg.ensure_dirs()
    if reviews is None:
        reviews = load_benchmark(cfg)

    monitor = StageMonitor()

    # ── Embed (cached; model released immediately after) ──────────────────
    with monitor.stage("embed"):
        embedder = SentenceTransformerEmbedder(
            spec.embedding_model,
            instruction=spec.instruction,
            device=cfg.apply_runtime_limits(),
            batch_size=cfg.batch_size,
        )
        try:
            embeddings, embed_s = embed_with_cache(cfg, embedder, reviews.texts)
        finally:
            embedder.close()

    # ── Reduce + Cluster (variant-specific) ───────────────────────────────
    micro_labels = None
    micro_to_macro = None
    if spec.variant == "bertopic":
        with monitor.stage("reduce_cluster"):
            labels, reduced = _bertopic_fit(cfg, spec, reviews, embeddings, seed)
    else:
        with monitor.stage("reduce"):
            reduced = _reduce_cached(cfg, spec, embeddings, seed)
        with monitor.stage("cluster"):
            labels, micro_labels, micro_to_macro = _cluster_cached(
                cfg, spec, reduced, seed
            )

    # ── Visualisation coords (cached) ─────────────────────────────────────
    with monitor.stage("viz_coords"):
        coords_2d = _viz_cached(cfg, spec, embeddings, seed, n_components=2)
        coords_3d = _viz_cached(cfg, spec, embeddings, seed, n_components=3)

    # ── Represent ─────────────────────────────────────────────────────────
    with monitor.stage("represent"):
        top_terms = ctfidf_terms(reviews.texts, labels)
        tfidf_terms = tfidf_top_terms(reviews.texts, labels)
        word_freqs = word_frequencies(reviews.texts, labels)

    # ── Label ─────────────────────────────────────────────────────────────
    with monitor.stage("label"):
        if label_clusters:
            labeler = OllamaLabeler(model=spec.label_model)
            cluster_labels = labeler.label_clusters(
                reviews.texts, labels, embeddings, top_terms
            )
        else:
            from ..label import term_fallback_label

            cluster_labels = {
                cid: term_fallback_label(cid, top_terms)
                for cid in sorted(int(c) for c in set(labels) if c != -1)
            }

    # ── Evaluate ──────────────────────────────────────────────────────────
    with monitor.stage("evaluate"):
        pipeline_s = sum(
            r["wall_s"] for n, r in monitor.records.items() if n != "label"
        )
        metrics = evaluate_labels(
            reduced,
            labels,
            reviews.texts,
            reviews.stars,
            runtime_s=pipeline_s,
            compute_coh=compute_coherence,
            seed=seed,
        )
        metrics["failure_flags"] = failure_flags(metrics, top_terms)
        metrics["seed"] = seed

    # ── Assemble artifacts ────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    clusters: dict[int, ClusterInfo] = {}
    for cid in sorted(int(c) for c in set(labels) if c != -1):
        mask = labels == cid
        member_ids = [reviews.ids[i] for i in np.flatnonzero(mask)]
        sample = list(
            rng.choice(member_ids, size=min(N_SAMPLE_DOCS, len(member_ids)), replace=False)
        )
        cl = cluster_labels[cid]
        member_stars = reviews.stars[mask]
        member_stars = member_stars[~np.isnan(member_stars)]
        clusters[cid] = ClusterInfo(
            cluster_id=cid,
            size=int(mask.sum()),
            label=cl.label,
            summary=cl.summary,
            label_source=cl.source,
            prompt_hash=cl.prompt_hash,
            top_terms=[[w, round(s, 5)] for w, s in top_terms.get(cid, [])],
            tfidf_terms=[[w, round(s, 5)] for w, s in tfidf_terms.get(cid, [])],
            word_frequencies=word_freqs.get(cid, {}),
            sample_doc_ids=sample,
            mean_stars=round(float(member_stars.mean()), 2) if len(member_stars) else None,
            micro_cluster_ids=sorted(
                m for m, g in (micro_to_macro or {}).items() if g == cid
            ),
        )

    manifest = {
        "run_name": run_name,
        "variant": spec.variant,
        "spec": spec.to_dict(),
        "sample_size": cfg.sample_size,
        "data_file": cfg.data_file,
        "seed": seed,
        "device": cfg.device,
        "stages": monitor.records,
        "label_sources": sorted({c.label_source for c in clusters.values()}),
    }

    art = RunArtifacts(
        run_name=run_name,
        manifest=manifest,
        doc_ids=reviews.ids,
        labels=labels,
        coords_2d=coords_2d,
        coords_3d=coords_3d,
        clusters=clusters,
        metrics=metrics,
        micro_labels=micro_labels,
    )
    save_run(run_dir, art)
    _log_to_results_csv(cfg, spec, embeddings.shape[1], embed_s, metrics)
    logger.info("run %s complete: %d clusters, flags: %s",
                run_name, metrics["n_clusters"], metrics["failure_flags"] or "none")
    return art


def cluster_labels_only(
    cfg: PipelineConfig,
    spec: PipelineSpec,
    reviews: Optional[ReviewSet] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Embed (cached) + reduce + cluster for one seed, skipping viz projections,
    representation, labeling and artifact writes. This is what the multi-seed
    stability check needs — at 50k each skipped UMAP fit is minutes, so the
    repeats must not pay for artifacts nobody reads.
    """
    seed = cfg.seed if seed is None else seed
    if reviews is None:
        reviews = load_benchmark(cfg)

    embedder = SentenceTransformerEmbedder(
        spec.embedding_model,
        instruction=spec.instruction,
        device=cfg.apply_runtime_limits(),
        batch_size=cfg.batch_size,
    )
    try:
        embeddings, _ = embed_with_cache(cfg, embedder, reviews.texts)
    finally:
        embedder.close()

    if spec.variant == "bertopic":
        path = clustering_path(
            cfg.cache_dir, "bertopic",
            f"mts{spec.cluster.get('min_topic_size', 10)}",
            f"{_seed_prefix(cfg, seed)}{make_slug(spec.embedding_model)}",
            cfg.sample_size,
        )
        if path.exists():
            return load_array(path).astype(int)
        labels, _ = _bertopic_fit(cfg, spec, reviews, embeddings, seed)
        save_array(path, labels.astype(np.int32))
        return labels

    reduced = _reduce_cached(cfg, spec, embeddings, seed)
    labels, _, _ = _cluster_cached(cfg, spec, reduced, seed)
    return labels


# ── Variant internals ─────────────────────────────────────────────────────────

def _seed_prefix(cfg: PipelineConfig, seed: int, base: str = "") -> str:
    """Distinct cache namespace for non-default seeds (stability runs)."""
    return base if seed == cfg.seed else f"s{seed}_{base}"


def _reduce_cached(
    cfg: PipelineConfig, spec: PipelineSpec, embeddings: np.ndarray, seed: int
) -> np.ndarray:
    r = spec.reducer
    reducer = UMAPReducer(seed=seed, **r)
    path = umap_path(
        cfg.cache_dir,
        spec.embedding_model,
        r["n_components"],
        r["n_neighbors"],
        r["min_dist"],
        r["metric"],
        cfg.sample_size,
        instruction=spec.instruction,
        prefix=_seed_prefix(cfg, seed, "pca50_" if r.get("pca_components") else ""),
    )
    if path.exists():
        return load_array(path)
    reduced = reducer.fit_transform(embeddings)
    save_array(path, reduced.astype(np.float32))
    return reduced


def _make_backend(spec: PipelineSpec, seed: int):
    if spec.variant == "custom_hdbscan":
        return HDBSCANBackend(**spec.cluster)
    if spec.variant == "flat_agglomerative":
        return AgglomerativeBackend(**spec.cluster)
    if spec.variant == "two_stage":
        return TwoStageBackend(**spec.cluster)
    raise ValueError(f"no clustering backend for variant {spec.variant!r}")


def _cluster_cached(
    cfg: PipelineConfig, spec: PipelineSpec, reduced: np.ndarray, seed: int
):
    backend = _make_backend(spec, seed)
    umap_slug = (
        f"{_seed_prefix(cfg, seed)}{make_slug(spec.embedding_model)}"
        f"__{make_slug(spec.instruction)}"
        f"__nc{spec.reducer['n_components']}__nn{spec.reducer['n_neighbors']}"
    )
    path = clustering_path(
        cfg.cache_dir, backend.algorithm, params_slug(backend.params),
        umap_slug, cfg.sample_size,
    )
    micro_path = path.with_name("micro_" + path.name)

    if path.exists() and (backend.algorithm != "two_stage" or micro_path.exists()):
        labels = load_array(path).astype(int)
        if backend.algorithm == "two_stage":
            micro = load_array(micro_path).astype(int)
            mapping = _micro_to_macro_from_arrays(micro, labels)
            return labels, micro, mapping
        return labels, None, None

    labels = backend.fit_predict(reduced)
    save_array(path, labels.astype(np.int32))
    if isinstance(backend, TwoStageBackend):
        save_array(micro_path, backend.micro_labels_.astype(np.int32))
        return labels, backend.micro_labels_, backend.micro_to_macro_
    return labels, None, None


def _micro_to_macro_from_arrays(micro: np.ndarray, macro: np.ndarray) -> dict[int, int]:
    return {
        int(m): int(g)
        for m, g in {(m, g) for m, g in zip(micro, macro) if m != -1}
    }


def _viz_cached(
    cfg: PipelineConfig, spec: PipelineSpec, embeddings: np.ndarray, seed: int,
    n_components: int,
) -> np.ndarray:
    path = umap_path(
        cfg.cache_dir,
        spec.embedding_model,
        n_components,
        spec.reducer.get("n_neighbors", 15),
        0.1,
        "cosine",
        cfg.sample_size,
        instruction=spec.instruction,
        prefix=_seed_prefix(cfg, seed, "viz_" if n_components == 2 else "viz3d_"),
    )
    if path.exists():
        return load_array(path)
    coords = viz_projection(
        embeddings, n_components, spec.reducer.get("n_neighbors", 15), seed=seed
    )
    save_array(path, coords.astype(np.float32))
    return coords


def _bertopic_fit(
    cfg: PipelineConfig,
    spec: PipelineSpec,
    reviews: ReviewSet,
    embeddings: np.ndarray,
    seed: int,
):
    """
    BERTopic with stock components, except a seeded UMAP (same defaults:
    5d/nn15/min_dist 0.0/cosine). Embeddings are passed in pre-computed —
    they ARE BERTopic's default model (MiniLM), just routed through our cache.
    Labels/terms/metrics are then computed by the same downstream code as
    every other variant, so the comparison measures the clustering, not
    cosmetic differences in artifact assembly.
    """
    from bertopic import BERTopic
    from umap import UMAP

    umap_model = UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine",
        random_state=seed,
    )
    model = BERTopic(
        umap_model=umap_model,
        min_topic_size=spec.cluster.get("min_topic_size", 10),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(reviews.texts, embeddings=embeddings)
    labels = np.asarray(topics, dtype=int)
    reduced = np.asarray(model.umap_model.embedding_)
    return labels, reduced


def _log_to_results_csv(
    cfg: PipelineConfig, spec: PipelineSpec, embed_dim: int, embed_s: float, metrics: dict
) -> None:
    """Append to the shared results.csv so notebook 07's comparison still works."""
    if spec.variant == "bertopic":
        algo, params = "bertopic_internal", spec.cluster
        reduction = {"reduction_method": "umap", "umap_n_components": 5,
                     "umap_n_neighbors": 15, "umap_min_dist": 0.0, "umap_metric": "cosine"}
        pipeline = "bertopic"
    else:
        backend = _make_backend(spec, metrics.get("seed", cfg.seed))
        algo, params = backend.algorithm, backend.params
        r = spec.reducer
        reduction = {
            "reduction_method": "pca_umap" if r.get("pca_components") else "umap",
            "umap_n_components": r["n_components"],
            "umap_n_neighbors": r["n_neighbors"],
            "umap_min_dist": r["min_dist"],
            "umap_metric": r["metric"],
            "pca_components": r.get("pca_components"),
        }
        pipeline = "custom"

    log_result(cfg.results_csv, {
        "pipeline": pipeline,
        "sample_size": cfg.sample_size,
        "device": cfg.device,
        "embedding_model": spec.embedding_model,
        "embedding_instruction": spec.instruction,
        "embed_dim": embed_dim,
        "embed_time_s": round(embed_s, 2),
        **reduction,
        "clustering_algo": algo,
        "cluster_params": json.dumps(params),
        **{k: v for k, v in metrics.items()
           if k in ("n_docs", "n_clusters", "noise_count", "noise_ratio",
                    "silhouette", "davies_bouldin", "calinski_harabasz",
                    "coherence_cv", "rating_entropy", "runtime_s")},
        "notes": f"pipeline-comparison variant={spec.variant} seed={metrics.get('seed')}",
    })
