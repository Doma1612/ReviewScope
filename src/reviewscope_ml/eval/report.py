"""
Pipeline comparison: run all four variants, evaluate, and write a report that
encodes the two-step decision protocol:

    Step 1 (automated)  — metrics + stability shortlist 2-3 finalists.
    Step 2 (human)      — qualitative inspection + intruder test pick the
                          winner; the HITL app records the confirmation.

The report deliberately does NOT contain a "winner" computed from metrics.
Metrics are proxies for ranking candidates cheaply; a configuration that
scores well but produces clusters a reader cannot name in one phrase has
failed. The final section is a sign-off block the human reviewer completes.

CLI::

    python -m reviewscope_ml.eval.report --sample-size 1000 --device cpu
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core.config import PipelineConfig
from ..data.ingest import load_benchmark
from ..pipelines.artifacts import RunArtifacts
from ..pipelines.runner import cluster_labels_only, run_pipeline
from ..pipelines.spec import PipelineSpec, default_specs
from .harness import stability_ari
from .inspection import render_inspection, render_intruder_test

logger = logging.getLogger("reviewscope.report")

# Metrics used for the shortlist ranking. Higher is better for all of these
# (silhouette twice: once classic, once with noise as a pseudo-cluster, so a
# noise-discarding algorithm cannot win on selective grading alone).
RANK_METRICS = ("silhouette", "silhouette_incl_noise", "coherence_cv", "rating_entropy")
N_FINALISTS = 3
STABILITY_SEEDS = (42, 43, 44)


def run_comparison(
    cfg: PipelineConfig,
    specs: Optional[dict[str, PipelineSpec]] = None,
    seeds: Sequence[int] = STABILITY_SEEDS,
    label_clusters: bool = True,
    embedding_model: Optional[str] = None,
    instruction: Optional[str] = None,
    tag: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Run every variant once with full artifacts (first seed) plus cheap
    label-free repeats for the remaining seeds (stability), then write
    ``data/runs/comparison_{size}[_{tag}]/report.md`` + charts.

    ``embedding_model``/``instruction`` override the three custom variants
    (BERTopic stays stock — that is what "off-the-shelf baseline" means), so
    a model sweep is one CLI call per candidate; ``tag`` defaults to the
    model slug to keep each sweep's report separate.
    """
    specs = specs or default_specs()
    if embedding_model:
        from ..core.cache import make_slug

        for name, spec in specs.items():
            if spec.variant != "bertopic":
                spec.embedding_model = embedding_model
                if instruction:
                    spec.instruction = instruction
        tag = tag or make_slug(embedding_model)
    reviews = load_benchmark(cfg)
    base_seed, *extra_seeds = list(seeds)

    arts: dict[str, RunArtifacts] = {}
    stability: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        logger.info("=== variant %s (seed %d) ===", name, base_seed)
        art = run_pipeline(
            cfg, spec, reviews=reviews, seed=base_seed,
            label_clusters=label_clusters, force=force,
        )
        arts[name] = art

        label_runs = [art.labels]
        for s in extra_seeds:
            logger.info("--- stability repeat: %s seed %d ---", name, s)
            label_runs.append(
                cluster_labels_only(cfg, spec, reviews=reviews, seed=s)
            )
        stability[name] = stability_ari(label_runs)

    suffix = f"_{tag}" if tag else ""
    out_dir = cfg.runs_dir / f"comparison_{cfg.sample_size}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    finalists = _shortlist(arts)
    _write_charts(arts, out_dir)
    report = _render_report(cfg, arts, stability, finalists, reviews, seeds)
    report_path = out_dir / "report.md"
    report_path.write_text(report)
    logger.info("comparison report -> %s", report_path)
    return report_path


def _shortlist(arts: dict[str, RunArtifacts]) -> list[str]:
    """Mean rank across RANK_METRICS; missing metric = worst rank for it."""
    names = list(arts)
    mean_ranks: dict[str, list[float]] = {n: [] for n in names}
    for metric in RANK_METRICS:
        values = {n: arts[n].metrics.get(metric) for n in names}
        present = sorted(
            (n for n in names if values[n] is not None),
            key=lambda n: values[n],
            reverse=True,
        )
        worst = len(names)
        for n in names:
            mean_ranks[n].append(present.index(n) + 1 if n in present else worst)
    ordered = sorted(names, key=lambda n: float(np.mean(mean_ranks[n])))
    return ordered[:N_FINALISTS]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_report(
    cfg: PipelineConfig,
    arts: dict[str, RunArtifacts],
    stability: dict[str, dict[str, Any]],
    finalists: list[str],
    reviews,
    seeds: Sequence[int],
) -> str:
    lines = [
        f"# Pipeline comparison — {cfg.sample_size:,} reviews",
        "",
        f"Benchmark: `{cfg.data_file}` · device: {cfg.device} · seeds: {list(seeds)}",
        "",
        "## How to read this report",
        "",
        "This comparison follows a two-step protocol:",
        "",
        "1. **Metrics shortlist** (this file, automated): three-tier metrics, "
        "stability, and structural failure flags narrow the field to "
        f"{len(finalists)} finalists. Metrics are cheap proxies — they rank, "
        "they never declare a winner.",
        "2. **Human inspection picks the winner**: read the qualitative "
        "inspection and take the intruder test below for each finalist; then "
        "confirm (or veto) in the HITL review app, which records the decision. "
        "A pipeline whose clusters you cannot name in one phrase loses, "
        "whatever its scores.",
        "",
        "## Tier overview",
        "",
        "| Metric | What it measures | Known bias |",
        "|---|---|---|",
        "| Silhouette (excl. noise) | geometric separation in reduced space | rewards discarding hard points as noise; rewards sentiment polarity |",
        "| Silhouette (incl. noise) | same, noise as own pseudo-cluster | fairer to partitioners; punishes honest noise labeling |",
        "| C_v coherence | top-term co-occurrence in raw text | unreliable on very short texts; insensitive to boundary quality |",
        "| Rating entropy | star-rating mix within clusters (high = thematic) | needs a rating column; hotel-specific calibration |",
        "| ARI stability | agreement of clusterings across seeds | says nothing about quality, only repeatability |",
        "",
        "## Results",
        "",
    ]

    header = (
        "| Variant | clusters | noise | sil (excl) | sil (incl) | C_v | entropy "
        "| max share | ARI mean | ARI min | runtime s |"
    )
    lines += [header, "|" + "---|" * 11]
    for name, art in arts.items():
        m = art.metrics
        s = stability[name]
        lines.append(
            f"| {name} | {_fmt(m.get('n_clusters'))} | {_fmt(m.get('noise_ratio'))} "
            f"| {_fmt(m.get('silhouette'))} | {_fmt(m.get('silhouette_incl_noise'))} "
            f"| {_fmt(m.get('coherence_cv'))} | {_fmt(m.get('rating_entropy'))} "
            f"| {_fmt(m.get('max_cluster_share'))} | {_fmt(s.get('ari_mean'))} "
            f"| {_fmt(s.get('ari_min'))} | {_fmt(m.get('runtime_s'))} |"
        )

    lines += ["", "### Per-stage cost", ""]
    lines += ["| Variant | stage | wall s | RSS peak MB | VRAM peak MB |", "|---|---|---|---|---|"]
    for name, art in arts.items():
        for stage, rec in art.manifest.get("stages", {}).items():
            lines.append(
                f"| {name} | {stage} | {rec.get('wall_s')} | {rec.get('rss_peak_mb')} "
                f"| {rec.get('vram_peak_mb', '—')} |"
            )

    lines += ["", "### Failure flags", ""]
    for name, art in arts.items():
        flags = art.metrics.get("failure_flags") or []
        if flags:
            lines.append(f"- **{name}**:")
            lines += [f"  - {f}" for f in flags]
        else:
            lines.append(f"- **{name}**: no structural flags")

    lines += [
        "",
        "![metrics](charts_metrics.png)",
        "",
        "## Step 1 result — shortlisted finalists",
        "",
        f"Mean rank across {', '.join(RANK_METRICS)} shortlists: "
        + ", ".join(f"**{n}**" for n in finalists),
        "",
        "Noise-fairness reminder: HDBSCAN-family variants discard noise, which "
        "inflates the classic silhouette — compare the *incl.-noise* column and "
        "the noise fraction before trusting Tier-1 differences.",
        "",
        "## Step 2 — human inspection of the finalists",
        "",
    ]

    # Sentence-level artifacts store segment ids; rebuild the (deterministic)
    # segment set so inspection shows the actual clustered sentences.
    seg_reviews = None
    for name in finalists:
        art = arts[name]
        if art.manifest.get("unit") == "sentence":
            if seg_reviews is None:
                from ..data.segment import segment_reviews

                seg_reviews = segment_reviews(reviews)
            unit_set = seg_reviews
        else:
            unit_set = reviews
        lines += [render_inspection(art, unit_set), ""]
        lines += [render_intruder_test(art, unit_set), ""]

    lines += [
        "---",
        "",
        "## Decision (to be completed by a human reviewer)",
        "",
        "- [ ] I read the inspection sheets above for every finalist.",
        "- [ ] I took the intruder tests; clusters whose intruder was not obvious are noted below.",
        "- [ ] Winner: `____________` — because: ____________",
        "- [ ] The sentence below is true and may be quoted in the final recommendation:",
        "",
        "> a human reviewed the clusters of the winning pipeline and confirmed "
        "they are thematically coherent",
        "",
        "Record the confirmation in the HITL app "
        "(`streamlit run src/reviewscope_ml/hitl/app.py`), which persists it to "
        "`data/feedback/` with reviewer name and timestamp.",
        "",
    ]
    return "\n".join(lines)


def _write_charts(arts: dict[str, RunArtifacts], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping charts")
        return

    names = list(arts)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Silhouette vs coherence (the "real winners are top-right" view)
    for n in names:
        m = arts[n].metrics
        if m.get("silhouette") is not None and m.get("coherence_cv") is not None:
            axes[0].scatter(m["silhouette"], m["coherence_cv"], s=90)
            axes[0].annotate(n, (m["silhouette"], m["coherence_cv"]),
                             fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("silhouette (excl. noise)")
    axes[0].set_ylabel("C_v coherence")
    axes[0].set_title("Tier 1 × Tier 2 — top-right is good")
    axes[0].grid(alpha=0.3)

    entropies = [arts[n].metrics.get("rating_entropy") for n in names]
    axes[1].bar(names, [e if e is not None else 0 for e in entropies], color="seagreen", alpha=0.8)
    axes[1].axhline(0.85, color="seagreen", ls="--", alpha=0.6)
    axes[1].axhline(0.60, color="tomato", ls="--", alpha=0.6)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Tier 3 — rating entropy (thematic > 0.85)")
    axes[1].tick_params(axis="x", rotation=20)

    runtimes = [arts[n].metrics.get("runtime_s") or 0 for n in names]
    noise = [arts[n].metrics.get("noise_ratio") or 0 for n in names]
    axes[2].bar(names, runtimes, color="steelblue", alpha=0.8)
    ax2 = axes[2].twinx()
    ax2.plot(names, noise, "o", color="tomato")
    ax2.set_ylabel("noise ratio", color="tomato")
    ax2.set_ylim(0, 1)
    axes[2].set_title("Runtime (bars) and noise ratio (dots)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_dir / "charts_metrics.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    from ..core.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the four-pipeline comparison")
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpus", default="auto",
                        help="'auto' = claim every idle GPU (embed stage runs "
                             "data-parallel across them); or a number, e.g. 2. "
                             "Busy GPUs are never claimed either way")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="embedding batch size PER GPU (suggested: 64 cpu, 256 cuda)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STABILITY_SEEDS))
    parser.add_argument("--no-llm", action="store_true",
                        help="skip Ollama labeling (term-fallback labels)")
    parser.add_argument("--embedding-model", default=None,
                        help="override the custom variants' embedding model "
                             "(e.g. BAAI/bge-m3); BERTopic stays stock")
    parser.add_argument("--instruction", default=None,
                        choices=["no_inst", "generic", "domain", "sentiment"],
                        help="instruction slug for instruction-tuned models")
    parser.add_argument("--data-file", default=None,
                        help="benchmark JSONL in data/cache for non-hotel "
                             "categories (e.g. sample_automotive_50k.jsonl)")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=list(default_specs()),
                        help="subset of pipeline variants to run (default: all). "
                             "Note: sentence_level multiplies points ~6x — its "
                             "UMAP fit dominates runtime at 50k")
    parser.add_argument("--force", action="store_true",
                        help="rebuild artifacts even for complete runs (heavy "
                             "intermediates still come from cache — use to add "
                             "newly introduced fields like sentiment to old runs)")
    args = parser.parse_args()

    overrides = {"sample_size": args.sample_size, "device": args.device}
    if args.data_file:
        overrides["data_file"] = args.data_file
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    elif args.device == "cuda":
        overrides["batch_size"] = 64

    if args.device == "cuda":
        # Shared-box etiquette: claim idle devices only, programmatically.
        from ..runtime.gpu import claim_gpu

        max_gpus = None if args.gpus == "auto" else int(args.gpus)
        claim = claim_gpu(require_gpu=True, max_gpus=max_gpus)
        overrides.update(device=claim.device, gpu_ids=claim.gpu_ids)
    cfg = load_config(**overrides)

    # Persist the full progress log next to the artifacts: long runs are
    # usually followed via `tail -f` from another shell or after a reconnect.
    cfg.ensure_dirs()
    from ..core.cache import make_slug

    # Separate report dirs per (category, embedding model) so sweeps and
    # cross-domain runs never overwrite each other's results.
    tag_parts = []
    if args.data_file:
        tag_parts.append(Path(args.data_file).stem.removeprefix("sample_").rsplit("_", 1)[0])
    if args.embedding_model:
        tag_parts.append(make_slug(args.embedding_model))
    tag = "_".join(tag_parts)
    log_path = cfg.runs_dir / f"comparison_{cfg.sample_size}{'_' + tag if tag else ''}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("logging to %s", log_path)

    specs = default_specs()
    if args.variants:
        specs = {k: v for k, v in specs.items() if k in args.variants}

    run_comparison(
        cfg,
        specs=specs,
        seeds=args.seeds,
        label_clusters=not args.no_llm,
        embedding_model=args.embedding_model,
        instruction=args.instruction,
        tag=tag or None,
        force=args.force,
    )
