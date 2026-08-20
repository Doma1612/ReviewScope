"""
Empirical admissibility + cost probe for embedding candidates.

The registry's hard constraints (``embed/models.py``) are claims about models;
this module *verifies* them on the actual box instead of trusting model cards:

- **loadable under our constraints** — plain ``sentence-transformers``, no
  ``trust_remote_code``, fp32 weights. A model that needs custom code paths is
  inadmissible however well it scores, because the Celery worker must stay free
  of them; a gated repo without accepted licence terms fails here too, visibly.
- **fits the VRAM slice** — peak allocated/reserved bytes during a real encode,
  measured with torch's per-device counters, not estimated from parameter count
  (activations, not weights, are what actually blow the 6 GB slice).
- **cost per unit of throughput** — segments/second on a *fixed* probe batch.

Why a separate probe rather than timing the sweep's own encode: the sweep reads
through an on-disk embedding cache, so ``embed_with_cache`` returns 0.0 seconds
on a hit. A ranking table whose runtime column is "0.000" for every previously
run model is not evidence. The probe always re-encodes the same fixed texts on
a single pinned device, so its numbers are comparable across candidates and
across invocations, independent of what happens to be cached.

Single device is deliberate: the sweep's embed stage may fan out across every
idle GPU, which makes wall-clock a function of how many neighbours were away
that afternoon rather than of the model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("reviewscope.cost_probe")

# Fixed probe size: large enough that per-batch overhead does not dominate,
# small enough that probing the whole registry costs minutes, not hours.
PROBE_TEXTS = 2_048


@dataclass
class ProbeResult:
    model: str
    instruction: str
    admissible: bool
    error: str = ""
    dim: Optional[int] = None
    params_m: Optional[float] = None
    max_seq_native: Optional[int] = None
    max_seq_used: Optional[int] = None
    load_s: Optional[float] = None
    encode_s: Optional[float] = None
    texts_per_s: Optional[float] = None
    vram_peak_alloc_mb: Optional[float] = None
    vram_peak_reserved_mb: Optional[float] = None
    weights_fp32_mb: Optional[float] = None
    batch_size: Optional[int] = None
    device: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def probe_model(
    model_name: str,
    texts: list[str],
    instruction: str = "no_inst",
    device: str = "cuda",
    batch_size: int = 64,
    max_seq: Optional[int] = None,
    vram_fraction: Optional[float] = None,
) -> ProbeResult:
    """
    Load *model_name* under the hard constraints and measure one encode.

    Never raises: an inadmissible model is a *result* (``admissible=False`` plus
    the error), because "this candidate cannot be used here" is exactly the
    finding the selection write-up needs to record.
    """
    from ..embed.sentence_transformer import SentenceTransformerEmbedder

    res = ProbeResult(
        model=model_name, instruction=instruction, admissible=False,
        device=device, batch_size=batch_size, max_seq_used=max_seq,
    )

    torch = None
    if device == "cuda":
        try:
            import torch as _torch

            torch = _torch
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except ImportError:
            torch = None

    embedder = SentenceTransformerEmbedder(
        model_name, instruction=instruction, device=device,
        batch_size=batch_size, show_progress=False, max_seq=max_seq,
    )
    # Pin to one device: the embedder fans out across every visible CUDA device,
    # which would make throughput depend on how many GPUs were idle, not on the
    # model. _target_devices is overridden rather than patched globally.
    embedder._target_devices = lambda: [device]  # type: ignore[method-assign]

    try:
        t0 = time.time()
        model = embedder._load(device)
        res.load_s = round(time.time() - t0, 2)

        res.dim = int(model.get_embedding_dimension())
        res.max_seq_native = int(getattr(model, "max_seq_length", 0)) or None
        n_params = sum(p.numel() for p in model.parameters())
        res.params_m = round(n_params / 1e6, 1)
        res.weights_fp32_mb = round(n_params * 4 / 1024**2, 1)

        # Warm-up, excluded from the timing. CUDA context creation, kernel
        # autotuning and cuBLAS handle setup all land on the first encode of
        # the process — without this the FIRST candidate probed is charged for
        # them and looks slower than models several times its size (observed:
        # MiniLM-L6 "slower" than mpnet). Peak-memory counters are reset after
        # it so VRAM reflects steady-state encoding, not the warm-up path.
        embedder.encode(texts[: min(len(texts), 2 * batch_size)])
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        embs = embedder.encode(texts)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        res.encode_s = round(time.time() - t0, 2)
        res.texts_per_s = round(len(texts) / max(res.encode_s, 1e-9), 1)
        if res.dim is None:
            res.dim = int(embs.shape[1])

        if torch is not None and torch.cuda.is_available():
            res.vram_peak_alloc_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            res.vram_peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
        res.admissible = True
        logger.info(
            "probe OK %s (%s): %.1f texts/s, %.0f MB peak VRAM",
            model_name, instruction, res.texts_per_s or 0.0,
            res.vram_peak_alloc_mb or 0.0,
        )
    except Exception as e:  # noqa: BLE001 — an unusable candidate is a finding
        res.error = f"{type(e).__name__}: {e}"[:300]
        logger.warning("probe FAILED %s (%s): %s", model_name, instruction, res.error)
    finally:
        embedder.close()

    return res


def probe_texts(cfg, sentence_level: bool = True, n: int = PROBE_TEXTS) -> list[str]:
    """A deterministic slice of the real corpus — same texts for every candidate.

    Real segments, not synthetic strings: throughput and peak activation memory
    both depend on the actual length distribution, and hotel mentions are short.
    """
    from ..data.ingest import load_benchmark

    reviews = load_benchmark(cfg)
    if sentence_level:
        from ..data.segment import segment_reviews

        units = segment_reviews(reviews)
    else:
        units = reviews
    return list(units.texts[:n])


def render_report(rows: list[ProbeResult], header: str, n_texts: int = PROBE_TEXTS) -> str:
    """Markdown admissibility table — admissible first, then the rejections."""
    ok = [r for r in rows if r.admissible]
    bad = [r for r in rows if not r.admissible]

    def fmt(v, spec="") -> str:
        if v is None:
            return "—"
        return f"{v:{spec}}" if spec else str(v)

    lines = [
        header,
        "",
        f"Measured on a fixed probe of {n_texts:,} real segments, single pinned "
        "device, fp32, plain sentence-transformers (no `trust_remote_code`). "
        "Each model is warmed up first, so no candidate is charged for CUDA "
        "context creation.",
        "Peak VRAM is torch's per-device counter around the encode — activations "
        "included, which is what actually decides whether a model fits the slice.",
        "",
        "| model | instr | params M | dim | native seq | seq cap | texts/s | peak VRAM MB | reserved MB | fp32 weights MB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(ok, key=lambda r: -(r.texts_per_s or 0)):
        lines.append(
            f"| `{r.model}` | {r.instruction} | {fmt(r.params_m)} | {fmt(r.dim)} "
            f"| {fmt(r.max_seq_native)} | {fmt(r.max_seq_used)} "
            f"| {fmt(r.texts_per_s)} | {fmt(r.vram_peak_alloc_mb)} "
            f"| {fmt(r.vram_peak_reserved_mb)} | {fmt(r.weights_fp32_mb)} |"
        )
    if bad:
        lines += ["", "## Inadmissible", "",
                  "Rejected by the hard constraints — recorded, not silently dropped.", ""]
        lines += [f"- `{r.model}` ({r.instruction}): {r.error}" for r in bad]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    from ..core.config import load_config
    from ..embed.models import CANDIDATES, candidates

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Embedding candidate admissibility + cost probe")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of registry models (substring match)")
    parser.add_argument("--max-params-m", type=int, default=700)
    parser.add_argument("--document-level", action="store_true",
                        help="probe on whole reviews instead of sentence segments")
    parser.add_argument("--n-texts", type=int, default=PROBE_TEXTS)
    args = parser.parse_args()

    overrides = {"sample_size": args.sample_size, "device": args.device}
    if args.device == "cuda":
        from ..runtime.gpu import claim_gpu

        # One device only — the probe measures per-device cost by design.
        claim = claim_gpu(require_gpu=True, max_gpus=1)
        overrides.update(device=claim.device, gpu_ids=claim.gpu_ids)
    cfg = load_config(**overrides)
    cfg.ensure_dirs()
    device = cfg.apply_runtime_limits()

    selected = candidates(max_params_m=args.max_params_m)
    if args.models:
        selected = [c for c in selected
                    if any(m.lower() in c.model.lower() for m in args.models)]
        if not selected:
            raise SystemExit(f"no registry model matches {args.models}; "
                             f"registry: {[c.model for c in CANDIDATES]}")

    sentence_level = not args.document_level
    texts = probe_texts(cfg, sentence_level=sentence_level, n=args.n_texts)
    logger.info("probing %d candidates on %d texts (%s unit)",
                len(selected), len(texts),
                "segment" if sentence_level else "document")

    results: list[ProbeResult] = []
    for cand in selected:
        for instr in cand.instruction_variants():
            results.append(probe_model(
                cand.model, texts, instruction=instr, device=device,
                batch_size=cand.batch_hint, max_seq=cand.encode_seq,
            ))

    unit = "sent" if sentence_level else "doc"
    out = cfg.runs_dir / f"admissibility_{unit}.md"
    out.write_text(render_report(
        results,
        f"# Embedding candidate admissibility & cost — {'segment' if sentence_level else 'document'} unit",
        n_texts=len(texts),
    ))
    (cfg.runs_dir / f"admissibility_{unit}.json").write_text(
        json.dumps([r.as_row() for r in results], indent=2)
    )
    logger.info("admissibility report -> %s", out)
