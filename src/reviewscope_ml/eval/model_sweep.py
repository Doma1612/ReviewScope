"""
Embedding-model sweep: notebook 04's experiment, GPU-capable and unattended.

Each registry candidate (``embed/models.py``) is evaluated under the FIXED
downstream pipeline (UMAP 10d/nn15 + HDBSCAN with size-scaled parameters — the
notebook 05/06 decisions) against identical corpus artifacts, so metric
differences are attributable to the embedding model and to nothing else. The
segmentation (regex splitter, <20 chars dropped, >600 hard-wrapped) is held
constant across candidates for the same reason.

Three-tier metrics + noise fairness per model, logged to results.csv (notebook
07 keeps working) and ranked in a markdown report.

What this module measures that the June version did not
-------------------------------------------------------
- **Honest cost.** ``embed_with_cache`` returns 0.0 seconds on a cache hit, so
  the old report printed "0.000" in its runtime column for every model that had
  ever been run — unusable as evidence. Runtime and peak VRAM now come from
  ``eval.cost_probe``: a fixed re-encoded probe on a single pinned device,
  comparable across candidates whatever the cache holds.
- **Instruction variants kept separate.** A candidate declaring
  ``extra_instructions`` produces one row per slug. Instructions mechanically
  reshape the space and inflate silhouette without moving coherence, so an
  instructed and an uninstructed run are different contestants, never averaged.
- **Stability.** Optional multi-seed ARI (``--stability-seeds``): a model whose
  clusters reshuffle when the seed changes has not found stable structure.
- **Provenance.** Every report carries the commit, config, unit and segmentation
  parameters it was produced under, so a table can be traced back to a run.

The ranking is a shortlist, not a verdict: instruction-tuned models inflate
silhouette mechanically — read coherence and rating entropy before believing
Tier 1, and confirm the finalist with the full pipeline comparison + human
inspection.

CLI::

    python -m reviewscope_ml.eval.model_sweep --sample-size 5000 --device cuda \
        --sentence-level
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.config import PipelineConfig
from ..data.ingest import load_benchmark
from ..embed import SentenceTransformerEmbedder, embed_with_cache
from ..embed.models import CANDIDATES, EmbeddingCandidate, candidates
from ..pipelines.runner import _cluster_cached, _reduce_cached
from ..pipelines.spec import PipelineSpec
from .harness import evaluate_labels, stability_ari

logger = logging.getLogger("reviewscope.model_sweep")

RANK_METRICS = ("silhouette", "silhouette_incl_noise", "coherence_cv", "rating_entropy")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def sweep(
    cfg: PipelineConfig,
    models: Optional[list[EmbeddingCandidate]] = None,
    compute_coherence: bool = True,
    sentence_level: bool = False,
    cost_probe: bool = True,
    stability_seeds: int = 1,
    tag: str = "",
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

    probe_set: list[str] = []
    if cost_probe and device == "cuda":
        from .cost_probe import PROBE_TEXTS

        probe_set = list(units.texts[:PROBE_TEXTS])

    rows: list[dict] = []
    for cand in models:
        for instr in cand.instruction_variants():
            row = _evaluate_one(
                cfg, cand, instr, units, device, sentence_level,
                compute_coherence, probe_set, stability_seeds,
            )
            rows.append(row)

    corpus = "" if cfg.corpus_slug == "hotels" else f"_{cfg.corpus_slug}"
    unit = "_sent" if sentence_level else ""
    # A tagged run (e.g. a finalists-only stability pass) must not overwrite the
    # full ranking table it was derived from.
    suffix = f"_{tag}" if tag else ""
    out = cfg.runs_dir / f"model_sweep_{cfg.sample_size}{corpus}{unit}{suffix}.md"
    out.write_text(_render(
        cfg, rows, sentence_level=sentence_level,
        n_units=len(units.texts), n_reviews=len(reviews.texts),
        stability_seeds=stability_seeds,
    ))
    logger.info("model sweep report -> %s", out)
    return out


def _evaluate_one(
    cfg, cand, instr, units, device, sentence_level,
    compute_coherence, probe_set, stability_seeds,
) -> dict:
    """One (candidate, instruction) contestant end to end."""
    logger.info("=== %s (%dM, instr=%s) ===", cand.model, cand.params_m, instr)
    spec = PipelineSpec(
        variant="sentence_level" if sentence_level else "custom_hdbscan",
        embedding_model=cand.model,
        instruction=instr,
        cluster={"min_cluster_size": "auto", "min_samples": "auto"},
    )

    # Cost probe first: on a cache hit the full encode never loads the model,
    # so this is the only place a runtime/VRAM number can come from.
    probe = None
    if probe_set:
        from .cost_probe import probe_model

        probe = probe_model(
            cand.model, probe_set, instruction=instr, device=device,
            batch_size=cand.batch_hint, max_seq=cand.encode_seq,
        )
        if not probe.admissible:
            logger.warning("SKIPPING %s (%s): failed cost probe", cand.model, instr)
            return {"model": cand.model, "instruction": instr,
                    "params_m": cand.params_m, "error": probe.error}

    embedder = SentenceTransformerEmbedder(
        cand.model, instruction=instr, device=device,
        batch_size=min(cfg.batch_size, cand.batch_hint),
        max_seq=cand.encode_seq,
    )
    try:
        embeddings, embed_s = embed_with_cache(
            cfg, embedder, units.texts,
            prefix_extra="sent__" if sentence_level else "",
        )
    except Exception as e:
        # Gated models without granted access, network failures: skip, don't die.
        logger.warning("SKIPPING %s (%s): %s", cand.model, instr, e)
        return {"model": cand.model, "instruction": instr,
                "params_m": cand.params_m, "error": str(e)[:200]}
    finally:
        embedder.close()  # one model on the GPU at a time, never two

    reduced = _reduce_cached(cfg, spec, embeddings, cfg.seed)
    labels, _, _ = _cluster_cached(cfg, spec, reduced, cfg.seed)
    metrics = evaluate_labels(
        reduced, labels, units.texts, units.stars,
        runtime_s=embed_s, compute_coh=compute_coherence, seed=cfg.seed,
    )
    if sentence_level:
        # Customers, not mentions (same dedup as the pipeline runner): one
        # verbose reviewer must not dominate a cluster's star profile.
        from ..core.metrics import compute_rating_entropy
        from ..pipelines.runner import _dedup_parent_stats

        dstars, dlabels = _dedup_parent_stats(units.ids, units.stars, labels)
        metrics["rating_entropy"] = compute_rating_entropy(dstars, dlabels)

    ari = None
    if stability_seeds > 1:
        runs = [labels]
        for extra_seed in range(cfg.seed + 1, cfg.seed + stability_seeds):
            r = _reduce_cached(cfg, spec, embeddings, extra_seed)
            lab, _, _ = _cluster_cached(cfg, spec, r, extra_seed)
            runs.append(lab)
        ari = stability_ari(runs)
        logger.info("%s stability: ARI mean=%s min=%s",
                    cand.model, ari["ari_mean"], ari["ari_min"])

    row = {
        "model": cand.model,
        "params_m": cand.params_m,
        "instruction": instr,
        "max_seq": cand.max_seq,
        "embed_s": round(embed_s, 1),
        "embed_cached": embed_s == 0.0,
        **metrics,
    }
    if probe is not None:
        row.update(
            texts_per_s=probe.texts_per_s,
            vram_peak_mb=probe.vram_peak_alloc_mb,
            params_m_measured=probe.params_m,
        )
    if ari is not None:
        row.update(ari_mean=ari["ari_mean"], ari_min=ari["ari_min"])
    _log_row(cfg, cand, instr, embeddings.shape[1], embed_s, metrics)
    logger.info(
        "%s (%s): clusters=%s sil=%s coh=%s entropy=%s",
        cand.model, instr, metrics.get("n_clusters"), metrics.get("silhouette"),
        metrics.get("coherence_cv"), metrics.get("rating_entropy"),
    )
    return row


def _log_row(cfg, cand, instr, dim, embed_s, metrics) -> None:
    import json

    from ..core.tracking import log_result

    log_result(cfg.results_csv, {
        "pipeline": "custom",
        "sample_size": cfg.sample_size,
        "device": cfg.device,
        "embedding_model": cand.model,
        "embedding_instruction": instr,
        "embed_dim": dim,
        "embed_time_s": round(embed_s, 2),
        "reduction_method": "umap",
        "umap_n_components": 10, "umap_n_neighbors": 15,
        "umap_min_dist": 0.0, "umap_metric": "cosine",
        "clustering_algo": "hdbscan",
        "cluster_params": json.dumps({"min_cluster_size": "auto_0.003", "min_samples": "auto"}),
        **{k: v for k, v in metrics.items()
           if k in ("n_docs", "n_clusters", "noise_count", "noise_ratio",
                    "silhouette", "davies_bouldin", "calinski_harabasz",
                    "coherence_cv", "rating_entropy", "runtime_s")},
        "notes": f"model_sweep corpus={cfg.corpus_slug}",
    })


def _mean_ranks(scored: list[dict]) -> None:
    """Attach ``_mean_rank`` to each row: mean position across RANK_METRICS.

    Ranked by identity, not equality — two contestants can produce identical
    metric dicts, and ``list.index`` would then score both as the first one.
    """
    for metric in RANK_METRICS:
        have = [r for r in scored if r.get(metric) is not None]
        order = sorted(have, key=lambda r: r[metric], reverse=True)
        position = {id(r): i + 1 for i, r in enumerate(order)}
        for r in scored:
            r.setdefault("_ranks", []).append(position.get(id(r), len(scored)))
    for r in scored:
        r["_mean_rank"] = round(sum(r["_ranks"]) / len(r["_ranks"]), 2)


def _render(
    cfg, rows, sentence_level: bool = False,
    n_units: int = 0, n_reviews: int = 0, stability_seeds: int = 1,
) -> str:
    def fmt(v, nd=3):
        if v is None:
            return "—"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    scored = [r for r in rows if "error" not in r]
    _mean_ranks(scored)
    scored.sort(key=lambda r: r["_mean_rank"])

    from ..data.segment import MAX_CHARS, MIN_CHARS

    unit_note = (
        " · unit: sentence segments (mention-level; entropy deduplicated per review)"
        if sentence_level else " · unit: whole reviews"
    )
    lines = [
        f"# Embedding model sweep — {cfg.sample_size:,} reviews (`{cfg.data_file}`){unit_note}",
        "",
        "## Provenance",
        "",
        f"- commit `{_git_commit()}` · generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- corpus `{cfg.corpus_slug}` · {n_reviews:,} reviews"
        + (f" → {n_units:,} segments" if sentence_level else "")
        + f" · seed {cfg.seed} · device {cfg.device}",
    ]
    if sentence_level:
        lines.append(
            f"- segmentation held constant: regex splitter, segments < {MIN_CHARS} chars "
            f"dropped, > {MAX_CHARS} chars hard-wrapped"
        )
    lines += [
        "- fixed downstream pipeline: UMAP(10d, nn=15, min_dist=0.0, cosine) + HDBSCAN "
        "with size-scaled parameters (mcs = 0.3% of units, floor 15; ms = mcs/3)",
        "- the embedding model is the only manipulated variable; every candidate sees "
        "identical corpus artifacts, segmentation, DR and clustering configuration",
        "- throughput and peak VRAM come from a fixed re-encoded probe on a single "
        "pinned device, so they are comparable regardless of what the cache held",
        "- **throughput is comparable within this table, not across tables**: the box "
        "is shared and unscheduled, so a concurrent job depresses seg/s (observed: "
        "large models roughly halved when a labeler sweep ran alongside). Peak VRAM "
        "is unaffected. Quote cost numbers from a sweep that ran alone.",
        "",
        "## Ranking",
        "",
        "Mean rank across silhouette (excl./incl. noise), C_v, rating entropy.",
        "Shortlist only — confirm the winner with the full pipeline comparison and",
        "human inspection; instruction-tuned silhouette gains without coherence gains",
        "are geometry reshaping, not better topics.",
        "",
    ]

    columns = ["#", "model", "params M", "instr", "mean rank", "clusters", "noise",
               "sil (excl)", "sil (incl)", "C_v", "entropy", "texts/s",
               "peak VRAM MB"]
    if stability_seeds > 1:
        columns.append(f"ARI mean ({stability_seeds} seeds)")
    lines += [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]

    for i, r in enumerate(scored, 1):
        cells = (
            f"| {i} | `{r['model']}` | {r['params_m']} | {r['instruction']} "
            f"| {fmt(r.get('_mean_rank'), 2)} "
            f"| {fmt(r.get('n_clusters'))} | {fmt(r.get('noise_ratio'))} "
            f"| {fmt(r.get('silhouette'))} | {fmt(r.get('silhouette_incl_noise'))} "
            f"| {fmt(r.get('coherence_cv'))} | {fmt(r.get('rating_entropy'))} "
            f"| {fmt(r.get('texts_per_s'), 1)} | {fmt(r.get('vram_peak_mb'), 0)} |"
        )
        if stability_seeds > 1:
            cells += f" {fmt(r.get('ari_mean'))} |"
        lines.append(cells)

    failed = [r for r in rows if "error" in r]
    if failed:
        lines += ["", "## Skipped", "",
                  "Recorded rather than silently dropped — an unusable candidate is a finding.", ""]
        lines += [f"- `{r['model']}` ({r.get('instruction', '?')}): {r['error']}"
                  for r in failed]
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
    parser.add_argument("--no-cost-probe", action="store_true",
                        help="skip the runtime/VRAM probe (metrics only, faster)")
    parser.add_argument("--stability-seeds", type=int, default=1,
                        help="re-reduce and re-cluster under N seeds and report "
                             "multi-seed ARI (1 = off; the UMAP fit is repeated "
                             "per seed, so this multiplies wall-clock)")
    parser.add_argument("--tag", default="",
                        help="suffix for the report/log filename, so a subset "
                             "run (e.g. finalists-only stability) does not "
                             "overwrite the full ranking table")
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

    # Unit and corpus in the log name too: a sentence sweep must not overwrite
    # the document sweep's log for the same sample size.
    corpus_tag = "" if cfg.corpus_slug == "hotels" else f"_{cfg.corpus_slug}"
    unit_tag = "_sent" if args.sentence_level else ""
    name_tag = f"_{args.tag}" if args.tag else ""
    log_path = (
        cfg.runs_dir / f"model_sweep_{cfg.sample_size}{corpus_tag}{unit_tag}{name_tag}.log"
    )
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)

    sweep(
        cfg,
        models=selected,
        compute_coherence=not args.no_coherence,
        sentence_level=args.sentence_level,
        cost_probe=not args.no_cost_probe,
        stability_seeds=args.stability_seeds,
        tag=args.tag,
    )
