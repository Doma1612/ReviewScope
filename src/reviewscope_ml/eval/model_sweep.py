"""
Embedding-model sweep: notebook 04's experiment, GPU-capable and unattended.

Each registry candidate (``embed/models.py``) is evaluated under the FIXED
downstream pipeline (UMAP 10d/nn15 + HDBSCAN mcs15/ms5 — the notebook 05/06
decisions), so metric differences are attributable to the embedding model and
to nothing else. Three-tier metrics + noise fairness per model, logged to
results.csv (notebook 07 keeps working) and ranked in a markdown report.

The ranking is a shortlist, not a verdict: as notebook 04 documents,
instruction-tuned models inflate silhouette mechanically — read coherence and
rating entropy before believing Tier 1, and confirm the finalist with the
full pipeline comparison + human inspection.

CLI::

    python -m reviewscope_ml.eval.model_sweep --sample-size 5000 --device cuda
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.config import PipelineConfig
from ..data.ingest import load_benchmark
from ..embed import SentenceTransformerEmbedder, embed_with_cache
from ..embed.models import CANDIDATES, EmbeddingCandidate, candidates
from ..pipelines.runner import _cluster_cached, _reduce_cached
from ..pipelines.spec import PipelineSpec
from .harness import evaluate_labels

logger = logging.getLogger("reviewscope.model_sweep")


def sweep(
    cfg: PipelineConfig,
    models: Optional[list[EmbeddingCandidate]] = None,
    compute_coherence: bool = True,
    sentence_level: bool = False,
) -> Path:
    """Run the sweep; returns the path of the ranking report.

    ``sentence_level=True`` evaluates the candidates on sentence segments
    instead of whole reviews — short texts are a different embedding regime
    (long-context advantages vanish, small models close the gap), so if the
    sentence_level pipeline is a serious finalist, pick its model here, not
    from the document-level ranking.
    """
    models = models if models is not None else candidates()
    reviews = load_benchmark(cfg)
    if sentence_level:
        from ..data.segment import segment_reviews

        units = segment_reviews(reviews)
        logger.info("sentence sweep: %d reviews -> %d segments", len(reviews), len(units))
    else:
        units = reviews
    device = cfg.apply_runtime_limits()

    rows: list[dict] = []
    for cand in models:
        logger.info("=== %s (%dM, instr=%s) ===", cand.model, cand.params_m, cand.instruction)
        spec = PipelineSpec(
            variant="sentence_level" if sentence_level else "custom_hdbscan",
            embedding_model=cand.model,
            instruction=cand.instruction,
            cluster=(
                {"min_cluster_size": 25, "min_samples": 10}
                if sentence_level
                else {"min_cluster_size": 15, "min_samples": 5}
            ),
        )
        embedder = SentenceTransformerEmbedder(
            cand.model, instruction=cand.instruction,
            device=device,
            batch_size=min(cfg.batch_size, cand.batch_hint),
            max_seq=cand.encode_seq,
        )
        try:
            embeddings, embed_s = embed_with_cache(
                cfg, embedder, units.texts,
                prefix_extra="sent__" if sentence_level else "",
            )
        except Exception as e:
            # Gated models without HF login, network failures: skip, don't die.
            logger.warning("SKIPPING %s: %s", cand.model, e)
            rows.append({"model": cand.model, "error": str(e)[:200]})
            continue
        finally:
            embedder.close()  # one model on the GPU at a time, never two

        reduced = _reduce_cached(cfg, spec, embeddings, cfg.seed)
        labels, _, _ = _cluster_cached(cfg, spec, reduced, cfg.seed)
        metrics = evaluate_labels(
            reduced, labels, units.texts, units.stars,
            runtime_s=embed_s, compute_coh=compute_coherence, seed=cfg.seed,
        )
        if sentence_level:
            # Customers, not mentions (same dedup as the pipeline runner).
            from ..core.metrics import compute_rating_entropy
            from ..pipelines.runner import _dedup_parent_stats

            dstars, dlabels = _dedup_parent_stats(units.ids, units.stars, labels)
            metrics["rating_entropy"] = compute_rating_entropy(dstars, dlabels)
        rows.append({
            "model": cand.model,
            "params_m": cand.params_m,
            "instruction": cand.instruction,
            "max_seq": cand.max_seq,
            "embed_s": round(embed_s, 1),
            **metrics,
        })
        _log_row(cfg, cand, embeddings.shape[1], embed_s, metrics)
        logger.info(
            "%s: clusters=%s sil=%s coh=%s entropy=%s",
            cand.model, metrics.get("n_clusters"), metrics.get("silhouette"),
            metrics.get("coherence_cv"), metrics.get("rating_entropy"),
        )

    corpus = "" if cfg.corpus_slug == "hotels" else f"_{cfg.corpus_slug}"
    unit = "_sent" if sentence_level else ""
    out = cfg.runs_dir / f"model_sweep_{cfg.sample_size}{corpus}{unit}.md"
    out.write_text(_render(cfg, rows, sentence_level=sentence_level))
    logger.info("model sweep report -> %s", out)
    return out


def _log_row(cfg, cand, dim, embed_s, metrics) -> None:
    import json

    from ..core.tracking import log_result

    log_result(cfg.results_csv, {
        "pipeline": "custom",
        "sample_size": cfg.sample_size,
        "device": cfg.device,
        "embedding_model": cand.model,
        "embedding_instruction": cand.instruction,
        "embed_dim": dim,
        "embed_time_s": round(embed_s, 2),
        "reduction_method": "umap",
        "umap_n_components": 10, "umap_n_neighbors": 15,
        "umap_min_dist": 0.0, "umap_metric": "cosine",
        "clustering_algo": "hdbscan",
        "cluster_params": json.dumps({"min_cluster_size": 15, "min_samples": 5}),
        **{k: v for k, v in metrics.items()
           if k in ("n_docs", "n_clusters", "noise_count", "noise_ratio",
                    "silhouette", "davies_bouldin", "calinski_harabasz",
                    "coherence_cv", "rating_entropy", "runtime_s")},
        "notes": f"model_sweep corpus={cfg.corpus_slug}",
    })


def _render(cfg, rows, sentence_level: bool = False) -> str:
    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    # Rank by mean rank over the three tiers + noise-fair silhouette,
    # same convention as the pipeline comparison report.
    rank_metrics = ("silhouette", "silhouette_incl_noise", "coherence_cv", "rating_entropy")
    scored = [r for r in rows if "error" not in r]
    for metric in rank_metrics:
        ordered = sorted(
            (r for r in scored if r.get(metric) is not None),
            key=lambda r: r[metric], reverse=True,
        )
        for r in scored:
            r.setdefault("_ranks", []).append(
                ordered.index(r) + 1 if r in ordered else len(scored)
            )
    scored.sort(key=lambda r: sum(r["_ranks"]) / len(r["_ranks"]))

    unit_note = (
        " · unit: sentence segments (mention-level; entropy deduplicated per review)"
        if sentence_level else ""
    )
    lines = [
        f"# Embedding model sweep — {cfg.sample_size:,} reviews (`{cfg.data_file}`){unit_note}",
        "",
        "Fixed downstream pipeline: UMAP(10d, nn=15, cosine) + "
        + ("HDBSCAN(mcs=25, ms=10)." if sentence_level else "HDBSCAN(mcs=15, ms=5)."),
        "Ranked by mean rank across silhouette (excl./incl. noise), C_v, rating entropy.",
        "Shortlist only — confirm the winner with the full pipeline comparison and",
        "human inspection; instruction-tuned silhouette gains without coherence gains",
        "are geometry reshaping, not better topics.",
        "",
        "| # | model | params M | instr | clusters | noise | sil (excl) | sil (incl) | C_v | entropy | embed s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(scored, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['params_m']} | {r['instruction']} "
            f"| {fmt(r.get('n_clusters'))} | {fmt(r.get('noise_ratio'))} "
            f"| {fmt(r.get('silhouette'))} | {fmt(r.get('silhouette_incl_noise'))} "
            f"| {fmt(r.get('coherence_cv'))} | {fmt(r.get('rating_entropy'))} "
            f"| {fmt(r.get('embed_s'))} |"
        )
    failed = [r for r in rows if "error" in r]
    if failed:
        lines += ["", "## Skipped", ""]
        lines += [f"- `{r['model']}`: {r['error']}" for r in failed]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from ..core.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Embedding model sweep")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpus", default="auto",
                        help="'auto' = claim every idle GPU; or a number, e.g. 2")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of registry models (substring match)")
    parser.add_argument("--max-params-m", type=int, default=700)
    parser.add_argument("--no-coherence", action="store_true")
    parser.add_argument("--sentence-level", action="store_true",
                        help="evaluate candidates on sentence segments instead "
                             "of whole reviews (~6x more texts; pick the model "
                             "for the sentence_level pipeline here)")
    args = parser.parse_args()

    overrides = {"sample_size": args.sample_size, "device": args.device}
    if args.data_file:
        overrides["data_file"] = args.data_file
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    elif args.device == "cuda":
        overrides["batch_size"] = 128

    if args.device == "cuda":
        from ..runtime.gpu import claim_gpu

        max_gpus = None if args.gpus == "auto" else int(args.gpus)
        claim = claim_gpu(require_gpu=True, max_gpus=max_gpus)
        overrides.update(device=claim.device, gpu_ids=claim.gpu_ids)
    cfg = load_config(**overrides)
    cfg.ensure_dirs()

    selected = candidates(max_params_m=args.max_params_m)
    if args.models:
        selected = [c for c in selected
                    if any(m.lower() in c.model.lower() for m in args.models)]
        if not selected:
            raise SystemExit(f"no registry model matches {args.models}; "
                             f"registry: {[c.model for c in CANDIDATES]}")

    log_path = cfg.runs_dir / f"model_sweep_{cfg.sample_size}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)

    sweep(
        cfg,
        models=selected,
        compute_coherence=not args.no_coherence,
        sentence_level=args.sentence_level,
    )
