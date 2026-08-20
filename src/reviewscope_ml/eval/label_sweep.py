"""
Labeler comparison driver: score every candidate LLM on the same mention clusters.

Holds the clustering fixed (one embedding model, one seed, one set of clusters)
and varies only the labeler and the prompt variant, so differences are
attributable to the language model and the prompt — the same controlled design
the embedding sweep uses.

Produces three artifacts:

- ``label_quality_<unit>.md`` — the quantitative comparison table;
- ``label_sheet_<unit>.md``  — notebook 08's human scoring sheet, populated;
- ``label_quality_<unit>.json`` — every label with its model and prompt hash,
  which is the tech-selection requirement ("reproducibility of generated text").

CLI::

    python -m reviewscope_ml.eval.label_sweep --sample-size 5000 \
        --models llama3.2:1b llama3.2:latest qwen3:4b --n-clusters 20
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np

from ..core.cache import load_array
from ..label.ollama import PROMPT_VARIANTS, OllamaLabeler, centroid_docs, prompt_hash
from .label_quality import (
    aggregate,
    cluster_centroids,
    render_human_sheet,
    render_report,
    score_label,
)

logger = logging.getLogger("reviewscope.label_sweep")


def build_clusters(cfg, embedding_model: str, instruction: str, sentence_level: bool):
    """Rebuild the fixed clustering the labelers are judged on."""
    from ..core.cache import embedding_path
    from ..data.ingest import load_benchmark
    from ..pipelines.runner import _cluster_cached, _reduce_cached
    from ..pipelines.spec import PipelineSpec

    reviews = load_benchmark(cfg)
    if sentence_level:
        from ..data.segment import segment_reviews

        units = segment_reviews(reviews)
    else:
        units = reviews

    prefix = ("" if cfg.corpus_slug == "hotels" else f"{cfg.corpus_slug}__") + (
        "sent__" if sentence_level else ""
    )
    path = embedding_path(
        cfg.cache_dir, embedding_model, len(units.texts),
        instruction=instruction, prefix=prefix,
    )
    if not path.exists():
        raise SystemExit(
            f"no cached embeddings at {path}. Run the model sweep first so the "
            "labelers are scored on the same space the selection ranked."
        )
    embeddings = load_array(path)

    spec = PipelineSpec(
        variant="sentence_level" if sentence_level else "custom_hdbscan",
        embedding_model=embedding_model, instruction=instruction,
        cluster={"min_cluster_size": "auto", "min_samples": "auto"},
    )
    reduced = _reduce_cached(cfg, spec, embeddings, cfg.seed)
    labels, _, _ = _cluster_cached(cfg, spec, reduced, cfg.seed)
    return units, embeddings, labels


def run(
    cfg,
    labeler_models: list[str],
    variants: list[str],
    embedding_model: str,
    instruction: str = "no_inst",
    sentence_level: bool = True,
    n_clusters: int = 20,
    n_docs: int = 5,
    base_url: Optional[str] = None,
    think: Optional[bool] = None,
    tag: str = "",
) -> None:
    from ..represent.terms import ctfidf_terms

    units, embeddings, labels = build_clusters(
        cfg, embedding_model, instruction, sentence_level
    )
    centroids, centroid_ids = cluster_centroids(embeddings, labels)
    logger.info("%d units, %d clusters", len(units.texts), len(centroid_ids))

    # Largest clusters first: they carry the most of the corpus, so a bad label
    # there costs the most, and they are the ones a reviewer sees first.
    sizes = {cid: int((labels == cid).sum()) for cid in centroid_ids}
    chosen = sorted(centroid_ids, key=lambda c: -sizes[c])[:n_clusters]
    terms = ctfidf_terms(units.texts, labels, top_n=10)

    # Exemplars for the human sheet. Shared with the scoring GUI so a reviewer
    # in the app judges a label against the same mentions the sheet prints.
    from .label_scoring import select_exemplars

    exemplars = select_exemplars(labels, units.texts, chosen, cfg.seed, k=5)

    per_model: dict[str, dict] = {}
    prompt_hashes: dict[str, str] = {}
    labels_by_model: dict[str, dict[int, str]] = {}
    records: list[dict] = []

    # Embed labels in the same space as the clusters, so "nearest centroid" is
    # a meaningful question. One embedder, loaded once, reused for all models.
    from ..embed import SentenceTransformerEmbedder
    from ..embed.models import encode_settings

    device = cfg.apply_runtime_limits()
    batch, seq = encode_settings(embedding_model, cfg.batch_size)
    label_embedder = SentenceTransformerEmbedder(
        embedding_model, instruction=instruction, device=device,
        batch_size=batch, show_progress=False, max_seq=seq,
    )

    try:
        for model in labeler_models:
            for variant in variants:
                # Key stays exactly two "|"-separated parts — both renderers
                # split on it — so the nothink marker rides in the variant.
                variant_name = variant + ("+nothink" if think is False else "")
                key = f"{model}|{variant_name}"
                labeler = OllamaLabeler(
                    model=model, n_docs=n_docs, prompt_variant=variant, think=think
                )
                if base_url:
                    labeler.base_url = base_url
                if not labeler.available():
                    logger.warning("SKIP %s — not available on the Ollama server", key)
                    continue
                label_prompt, summary_prompt = labeler.prompts
                phash = prompt_hash(label_prompt, summary_prompt)
                prompt_hashes[key] = phash
                logger.info("=== %s (prompt %s / hash %s) ===", model, variant, phash)

                raw_labels: dict[int, str] = {}
                gen_times: dict[int, float] = {}
                for cid in chosen:
                    docs = centroid_docs(cid, labels, units.texts, embeddings, n=n_docs)
                    doc_block = "\n\n".join(f"- {d}" for d in docs)
                    t0 = time.time()
                    try:
                        raw = labeler._generate(label_prompt.format(docs=doc_block))
                    except Exception as e:  # noqa: BLE001
                        logger.warning("%s cluster %d failed: %s", key, cid, e)
                        raw = ""
                    gen_times[cid] = time.time() - t0
                    raw_labels[cid] = raw

                cleaned = {}
                vecs = label_embedder.encode(
                    [raw_labels[cid] or " " for cid in chosen]
                )
                scores = []
                for i, cid in enumerate(chosen):
                    member_texts = [units.texts[j] for j in np.flatnonzero(labels == cid)]
                    s = score_label(
                        cid, raw_labels[cid], member_texts,
                        label_vec=vecs[i], centroids=centroids,
                        centroid_ids=centroid_ids, gen_s=gen_times[cid],
                    )
                    scores.append(s)
                    cleaned[cid] = s.label
                    records.append({
                        "labeler": model, "prompt_variant": variant_name,
                        "prompt_hash": phash, "cluster_id": cid,
                        "raw_label": s.raw_label, "label": s.label,
                        "format_ok": s.format_ok, "format_issues": s.format_issues,
                        "discriminates": s.discriminates, "margin": s.margin,
                        "grounding_rate": s.grounding_rate,
                        "hallucinated": s.hallucinated, "gen_s": round(s.gen_s or 0, 2),
                    })
                per_model[key] = aggregate(scores)
                labels_by_model[key] = cleaned
                logger.info("%s: %s", key, per_model[key])
    finally:
        label_embedder.close()

    if not per_model:
        raise SystemExit("no labeler produced results — is the Ollama server up?")

    unit = "sent" if sentence_level else "doc"
    unit = f"{unit}_{tag}" if tag else unit
    notes = [
        f"Fixed clustering: `{embedding_model}` ({instruction}), seed {cfg.seed}, "
        f"{len(centroid_ids)} clusters over {len(units.texts):,} "
        f"{'mentions' if sentence_level else 'reviews'}; "
        f"the {len(chosen)} largest clusters are scored.",
        "Only the labeler and the prompt vary. Context strategy is notebook 08's "
        f"centroid-nearest, {n_docs} docs.",
        "",
        "`discrim@1` = the label's own cluster centroid is its nearest centroid "
        "(an automatic analogue of the intruder test). `halluc. rate` = share of "
        "labels containing a content word found in no member mention — a "
        "tripwire for invention that also penalises correct paraphrase, so read "
        "it with the human sheet, not instead of it.",
    ]
    out = cfg.runs_dir / f"label_quality_{unit}.md"
    out.write_text(render_report(
        per_model, prompt_hashes,
        f"# LLM labeler comparison — {'mention' if sentence_level else 'document'} clusters",
        notes,
    ))
    (cfg.runs_dir / f"label_sheet_{unit}.md").write_text(render_human_sheet(
        chosen, sizes, terms, exemplars, labels_by_model
    ))
    (cfg.runs_dir / f"label_quality_{unit}.json").write_text(json.dumps(records, indent=2))
    logger.info("label quality report -> %s", out)


if __name__ == "__main__":
    import argparse

    from ..core.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="LLM labeler comparison")
    p.add_argument("--sample-size", type=int, default=5_000)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--models", nargs="+", required=True, help="Ollama model tags")
    p.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS),
                   choices=list(PROMPT_VARIANTS))
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2",
                   help="model whose space the clusters and labels live in")
    p.add_argument("--instruction", default="no_inst")
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--n-docs", type=int, default=5)
    p.add_argument("--base-url", default=None)
    p.add_argument("--no-think", action="store_true",
                   help="ask the server to disable reasoning (qwen3 and other "
                        "thinking models); labeling is short structured output, "
                        "so reasoning is mostly latency")
    p.add_argument("--tag", default="",
                   help="suffix for output filenames so a follow-up run does "
                        "not overwrite the main comparison")
    p.add_argument("--document-level", action="store_true")
    args = p.parse_args()

    overrides = {"sample_size": args.sample_size, "device": args.device}
    if args.device == "cuda":
        from ..runtime.gpu import claim_gpu

        claim = claim_gpu(require_gpu=False, max_gpus=1)
        overrides.update(device=claim.device, gpu_ids=claim.gpu_ids)
    cfg = load_config(**overrides)
    cfg.ensure_dirs()

    run(
        cfg,
        labeler_models=args.models,
        variants=args.variants,
        embedding_model=args.embedding_model,
        instruction=args.instruction,
        sentence_level=not args.document_level,
        n_clusters=args.n_clusters,
        n_docs=args.n_docs,
        base_url=args.base_url,
        think=False if args.no_think else None,
        tag=args.tag,
    )
