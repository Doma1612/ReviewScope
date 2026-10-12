from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

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
from ..embed.models import encode_settings
from ..eval.harness import evaluate_labels, failure_flags
from ..label import OllamaLabeler
from ..reduce import UMAPReducer, viz_projection
from ..represent import ctfidf_terms, tfidf_top_terms, word_frequencies
from ..runtime.monitor import StageMonitor
from .artifacts import ClusterInfo, RunArtifacts, run_is_complete, save_run
from .spec import PipelineSpec

logger = logging.getLogger("reviewscope.pipeline")

N_SAMPLE_DOCS = 5


def run_pipeline(
    cfg: PipelineConfig,
    spec: PipelineSpec,
    reviews: Optional[ReviewSet] = None,
    seed: Optional[int] = None,
    run_name: Optional[str] = None,
    force: bool = False,
    compute_coherence: bool = True,
    label_clusters: bool = True,
    on_stage: Optional[Callable[[str], None]] = None,
) -> RunArtifacts:
    seed = cfg.seed if seed is None else seed
    if run_name is None:
        from .spec import default_specs

        parts = [spec.variant]
        if cfg.corpus_slug != "hotels":
            parts.append(cfg.corpus_slug)
        if spec.embedding_model != default_specs()[spec.variant].embedding_model:
            parts.append(make_slug(spec.embedding_model))
        parts += [str(cfg.sample_size), f"s{seed}"]
        run_name = "__".join(parts)
    run_dir = cfg.runs_dir / run_name

    from ..pipelines.artifacts import load_run

    if run_is_complete(run_dir) and not force:
        logger.info("run %s already complete — loading artifacts", run_name)
        return load_run(run_dir)

    cfg.ensure_dirs()
    if reviews is None:
        reviews = load_benchmark(cfg)

    is_sentence = spec.variant == "sentence_level"
    if is_sentence:
        from ..data.segment import segment_reviews

        units = segment_reviews(reviews)
        logger.info(
            "sentence segmentation: %d reviews -> %d segments",
            len(reviews), len(units),
        )
    else:
        units = reviews

    monitor = StageMonitor(on_stage=on_stage)
    profile = run_profile(cfg, spec)

    with monitor.stage("embed"):
        batch, seq_cap = encode_settings(spec.embedding_model, cfg.batch_size)
        embedder = SentenceTransformerEmbedder(
            spec.embedding_model,
            instruction=spec.instruction,
            device=cfg.apply_runtime_limits(),
            batch_size=batch,
            max_seq=seq_cap,
            instructions=profile.instructions,
        )
        try:
            embeddings, embed_s = embed_with_cache(
                cfg, embedder, units.texts,
                prefix_extra="sent__" if is_sentence else "",
            )
        finally:
            embedder.close()

    micro_labels = None
    micro_to_macro = None
    if spec.variant == "bertopic":
        with monitor.stage("reduce_cluster"):
            labels, reduced = _bertopic_fit(cfg, spec, units, embeddings, seed)
    else:
        with monitor.stage("reduce"):
            reduced = _reduce_cached(cfg, spec, embeddings, seed)
        with monitor.stage("cluster"):
            labels, micro_labels, micro_to_macro = _cluster_cached(
                cfg, spec, reduced, seed
            )

    with monitor.stage("viz_coords"):
        coords_2d = _viz_cached(cfg, spec, embeddings, seed, n_components=2)
        coords_3d = _viz_cached(cfg, spec, embeddings, seed, n_components=3)

    # Sentiment is cluster metadata, never input to embeddings or clustering.
    with monitor.stage("sentiment"):
        sentiment_scores = None
        sentiment_labels = None
        try:
            from ..sentiment import score_to_label, sentiment_with_cache

            sentiment_scores, _sent_s = sentiment_with_cache(
                cfg, units.texts,
                device=cfg.apply_runtime_limits(),
                prefix_extra="sent__" if is_sentence else "",
            )
            sentiment_labels = [score_to_label(float(s)) for s in sentiment_scores]
        except Exception as e:
            logger.warning("sentiment stage failed (continuing without): %s", e)

    with monitor.stage("represent"):
        top_terms = ctfidf_terms(units.texts, labels)
        tfidf_terms = tfidf_top_terms(units.texts, labels)
        word_freqs = word_frequencies(units.texts, labels)

    with monitor.stage("label"):
        if label_clusters:
            labeler = OllamaLabeler(
                model=spec.label_model,
                prompt_variant=spec.prompt_variant or profile.label_prompt_variant,
                domain_description=profile.domain_description,
                label_prompt_override=spec.label_prompt,
                summary_prompt_override=spec.summary_prompt,
            )
            cluster_labels = labeler.label_clusters(
                units.texts, labels, embeddings, top_terms
            )
        else:
            from ..label import term_fallback_label

            cluster_labels = {
                cid: term_fallback_label(cid, top_terms)
                for cid in sorted(int(c) for c in set(labels) if c != -1)
            }

    with monitor.stage("evaluate"):
        pipeline_s = sum(
            r["wall_s"] for n, r in monitor.records.items() if n != "label"
        )
        metrics = evaluate_labels(
            reduced,
            labels,
            units.texts,
            units.stars,
            runtime_s=pipeline_s,
            compute_coh=compute_coherence,
            seed=seed,
        )
        if is_sentence:
            # Count each review once per cluster so verbosity cannot skew rating entropy.
            from ..core.metrics import compute_rating_entropy

            dstars, dlabels = _dedup_parent_stats(units.ids, units.stars, labels)
            metrics["rating_entropy"] = compute_rating_entropy(dstars, dlabels)
        metrics["failure_flags"] = failure_flags(metrics, top_terms)
        metrics["seed"] = seed
        metrics["unit"] = "sentence" if is_sentence else "document"

    from ..data.segment import parent_id

    rng = np.random.default_rng(seed)
    clusters: dict[int, ClusterInfo] = {}
    for cid in sorted(int(c) for c in set(labels) if c != -1):
        mask = labels == cid
        member_ids = [units.ids[i] for i in np.flatnonzero(mask)]
        sample = list(
            rng.choice(member_ids, size=min(N_SAMPLE_DOCS, len(member_ids)), replace=False)
        )
        cl = cluster_labels[cid]
        if is_sentence:
            by_parent: dict[str, float] = {}
            for i in np.flatnonzero(mask):
                by_parent.setdefault(parent_id(units.ids[i]), float(units.stars[i]))
            member_stars = np.array([s for s in by_parent.values() if not np.isnan(s)])
            n_documents = len(by_parent)
        else:
            member_stars = units.stars[mask]
            member_stars = member_stars[~np.isnan(member_stars)]
            n_documents = None
        if sentiment_scores is not None:
            from ..sentiment import aggregate_cluster_sentiment

            sent_avg, sent_dist = aggregate_cluster_sentiment(
                sentiment_scores, labels, cid
            )
        else:
            sent_avg, sent_dist = None, None
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
            n_documents=n_documents,
            sentiment_avg=sent_avg,
            sentiment_dist=sent_dist,
        )

    manifest = {
        "run_name": run_name,
        "variant": spec.variant,
        "spec": spec.to_dict(),
        "sample_size": cfg.sample_size,
        "data_file": cfg.data_file,
        "unit": "sentence" if is_sentence else "document",
        "n_units": len(units),
        "seed": seed,
        "device": cfg.device,
        "stages": monitor.records,
        "label_sources": sorted({c.label_source for c in clusters.values()}),
    }

    art = RunArtifacts(
        run_name=run_name,
        manifest=manifest,
        doc_ids=units.ids,
        labels=labels,
        coords_2d=coords_2d,
        coords_3d=coords_3d,
        clusters=clusters,
        metrics=metrics,
        micro_labels=micro_labels,
        sentiment_scores=sentiment_scores,
        sentiment_labels=sentiment_labels,
    )
    save_run(run_dir, art)
    if is_sentence:
        _write_doc_membership(run_dir, units.ids, labels)
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
    seed = cfg.seed if seed is None else seed
    if reviews is None:
        reviews = load_benchmark(cfg)

    is_sentence = spec.variant == "sentence_level"
    if is_sentence:
        from ..data.segment import segment_reviews

        reviews = segment_reviews(reviews)

    batch, seq_cap = encode_settings(spec.embedding_model, cfg.batch_size)
    embedder = SentenceTransformerEmbedder(
        spec.embedding_model,
        instruction=spec.instruction,
        device=cfg.apply_runtime_limits(),
        batch_size=batch,
        max_seq=seq_cap,
        instructions=run_profile(cfg, spec).instructions,
    )
    try:
        embeddings, _ = embed_with_cache(
            cfg, embedder, reviews.texts,
            prefix_extra="sent__" if is_sentence else "",
        )
    finally:
        embedder.close()

    if spec.variant == "bertopic":
        path = clustering_path(
            cfg.cache_dir, "bertopic",
            f"mts{spec.cluster.get('min_topic_size', 10)}",
            f"{_seed_prefix(cfg, spec, seed)}{make_slug(spec.embedding_model)}",
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


def run_profile(cfg: PipelineConfig, spec: PipelineSpec):
    from ..core.corpus import profile_for

    return profile_for(spec.corpus_profile or cfg.corpus_slug)


def _corpus_prefix(cfg: PipelineConfig) -> str:
    return "" if cfg.corpus_slug == "hotels" else f"{cfg.corpus_slug}__"


def _unit_prefix(spec: PipelineSpec) -> str:
    return "sent__" if spec.variant == "sentence_level" else ""


def _seed_prefix(cfg: PipelineConfig, spec: PipelineSpec, seed: int, base: str = "") -> str:
    ns = f"{_corpus_prefix(cfg)}{_unit_prefix(spec)}"
    return f"{ns}{base}" if seed == cfg.seed else f"{ns}s{seed}_{base}"


def _dedup_parent_stats(
    ids: list[str], stars: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from ..data.segment import parent_id

    seen: set[tuple[str, int]] = set()
    out_stars: list[float] = []
    out_labels: list[int] = []
    for i, seg_id in enumerate(ids):
        label = int(labels[i])
        if label == -1:
            continue
        key = (parent_id(seg_id), label)
        if key in seen:
            continue
        seen.add(key)
        out_stars.append(float(stars[i]))
        out_labels.append(label)
    return np.array(out_stars), np.array(out_labels)


def _write_doc_membership(run_dir, segment_ids: list[str], labels: np.ndarray) -> None:
    from collections import Counter, defaultdict

    from ..data.segment import parent_id

    per_doc: dict[str, Counter] = defaultdict(Counter)
    for seg_id, label in zip(segment_ids, labels):
        per_doc[parent_id(seg_id)][int(label)] += 1

    membership = {}
    for doc, counts in per_doc.items():
        n = sum(counts.values())
        real = {c: k for c, k in counts.items() if c != -1}
        primary = max(real, key=real.get) if real else -1
        membership[doc] = {
            "primary": primary,
            "clusters": {str(c): round(k / n, 4) for c, k in counts.items()},
            "n_segments": n,
        }
    (run_dir / "doc_membership.json").write_text(json.dumps(membership, indent=1))


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
        prefix=_seed_prefix(cfg, spec, seed, "pca50_" if r.get("pca_components") else ""),
    )
    if path.exists():
        return load_array(path)
    reduced = reducer.fit_transform(embeddings)
    save_array(path, reduced.astype(np.float32))
    return reduced


def _make_backend(spec: PipelineSpec, seed: int, n_units: int):
    from ..cluster import scaled_min_cluster_size, scaled_min_samples

    c = dict(spec.cluster)
    if spec.variant in ("custom_hdbscan", "sentence_level"):
        if c.get("min_cluster_size") == "auto":
            c["min_cluster_size"] = scaled_min_cluster_size(n_units)
        if c.get("min_samples") == "auto":
            c["min_samples"] = scaled_min_samples(c["min_cluster_size"])
        return HDBSCANBackend(**c)
    if spec.variant == "flat_agglomerative":
        return AgglomerativeBackend(**c)
    if spec.variant == "two_stage":
        if c.get("micro_min_cluster_size") == "auto":
            c["micro_min_cluster_size"] = scaled_min_cluster_size(
                n_units, fraction=0.001, floor=5
            )
        if c.get("micro_min_samples") == "auto":
            c["micro_min_samples"] = scaled_min_samples(
                round(c["micro_min_cluster_size"] * 0.6), ratio=1, floor=3
            )
        return TwoStageBackend(**c)
    raise ValueError(f"no clustering backend for variant {spec.variant!r}")


def _cluster_cached(
    cfg: PipelineConfig, spec: PipelineSpec, reduced: np.ndarray, seed: int
):
    backend = _make_backend(spec, seed, n_units=len(reduced))
    umap_slug = (
        f"{_seed_prefix(cfg, spec, seed)}{make_slug(spec.embedding_model)}"
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
        prefix=_seed_prefix(cfg, spec, seed, "viz_" if n_components == 2 else "viz3d_"),
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
    if spec.variant == "bertopic":
        algo, params = "bertopic_internal", spec.cluster
        reduction = {"reduction_method": "umap", "umap_n_components": 5,
                     "umap_n_neighbors": 15, "umap_min_dist": 0.0, "umap_metric": "cosine"}
        pipeline = "bertopic"
    else:
        backend = _make_backend(
            spec, metrics.get("seed", cfg.seed), n_units=metrics.get("n_docs", 0)
        )
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
