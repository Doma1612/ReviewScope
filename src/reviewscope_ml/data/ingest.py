"""
Benchmark data ingestion.

Two jobs:

1. ``build_benchmark_sample`` — reproduce the fixed WP5 benchmark file that
   notebook 02 defined (Yelp reviews joined to businesses, filtered to the
   "Hotels" category, minimum 50 chars, first N matches in dataset order).
   The sampling decisions are notebook 02's; this is a faithful port so the
   file can be regenerated on a fresh checkout without Jupyter.

2. ``load_benchmark`` — load that JSONL into the in-memory shape every
   pipeline stage consumes (preprocessed texts + star ratings + ids).

The raw Yelp dump ships as ``data/raw/Yelp-JSON.zip`` containing
``yelp_dataset.tar``. We stream tar members directly out of the zip instead
of extracting ~9 GB to disk; building the sample needs two sequential passes
(businesses first, then reviews) because tar members can only be read in
archive order when streaming.
"""
from __future__ import annotations

import json
import logging
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Optional

import numpy as np

from ..core.config import PipelineConfig, get_preprocessor

logger = logging.getLogger("reviewscope.data")

RAW_ZIP_NAME = "Yelp-JSON.zip"
# Path used by notebook 02 when the archive was extracted manually.
RAW_EXTRACTED_TAR = Path("Yelp-JSON") / "Yelp JSON" / "yelp_dataset.tar"


@dataclass
class ReviewSet:
    """The benchmark sample in the shape all pipeline stages consume."""

    ids: list[str]            # Yelp review_id — stable per-document key
    texts: list[str]          # preprocessed text (default: "minimal" variant)
    raw_texts: list[str]      # untouched text, for human inspection artifacts
    stars: np.ndarray         # 1-5 rating per review; NaN if missing

    def __len__(self) -> int:
        return len(self.texts)


# ── Loading the benchmark file ────────────────────────────────────────────────

def load_benchmark(cfg: PipelineConfig, preprocessor: str = "minimal") -> ReviewSet:
    """
    Load ``cfg.data_path`` (e.g. sample_hotels_5k.jsonl), capped at
    ``cfg.sample_size`` rows, preprocessed with the variant notebook 02 chose.
    """
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Benchmark file missing: {cfg.data_path}\n"
            "Generate it with: python -m reviewscope_ml.data.ingest "
            f"--sample-size {cfg.sample_size}"
        )
    preprocess = get_preprocessor(preprocessor)
    ids, texts, raw_texts, stars = [], [], [], []
    with open(cfg.data_path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj.get("review_id", str(len(ids))))
            raw_texts.append(obj["text"])
            texts.append(preprocess(obj["text"]))
            stars.append(float(obj.get("stars", float("nan"))))
            if len(texts) >= cfg.sample_size:
                break
    return ReviewSet(ids=ids, texts=texts, raw_texts=raw_texts, stars=np.array(stars))


# ── Building the benchmark file from the raw Yelp dump ───────────────────────

def _open_tar_member(raw_dir: Path, name_contains: str) -> tuple[IO[bytes], object]:
    """
    Return a binary stream for the first tar member whose name contains
    *name_contains*, plus an opaque handle tuple to keep underlying files alive.

    Prefers an extracted tar (seekable, fast); falls back to streaming the tar
    straight out of the zip (slower, but avoids a 9 GB extraction).
    """
    extracted = raw_dir / RAW_EXTRACTED_TAR
    if extracted.exists():
        tar = tarfile.open(extracted)
        member = next(m for m in tar.getmembers() if name_contains in m.name.lower())
        return tar.extractfile(member), (tar,)

    zip_path = raw_dir / RAW_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Raw Yelp dataset not found: neither {extracted} nor {zip_path}"
        )
    zf = zipfile.ZipFile(zip_path)
    tar_entry = next(n for n in zf.namelist() if n.endswith(".tar"))
    # "r|*" = forward-only streaming with compression auto-detect: the shipped
    # yelp_dataset.tar is actually gzip-compressed despite its name, and a
    # streaming mode avoids seeks on the (decompressing) zip stream.
    tar = tarfile.open(fileobj=zf.open(tar_entry), mode="r|*")
    for member in tar:
        if name_contains in member.name.lower():
            return tar.extractfile(member), (tar, zf)
    raise FileNotFoundError(f"No tar member matching {name_contains!r} in {zip_path}")


def _collect_category_business_ids(raw_dir: Path, category: str) -> set[str]:
    """Stream business.json and collect ids whose categories include *category*."""
    stream, _handles = _open_tar_member(raw_dir, "business")
    ids: set[str] = set()
    total = 0
    for line in stream:
        b = json.loads(line)
        total += 1
        cats = b.get("categories") or ""
        if category in (c.strip() for c in cats.split(",")):
            ids.add(b["business_id"])
    logger.info("%d/%d businesses tagged %r", len(ids), total, category)
    return ids


def build_benchmark_sample(
    project_root: Path,
    sample_size: int,
    category: str = "Hotels",
    min_text_len: int = 50,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build ``data/cache/sample_hotels_{N}k.jsonl`` exactly as notebook 02 did:
    stream reviews in dataset order, keep those whose business carries the
    target category and whose text is >= *min_text_len* chars, stop at
    *sample_size*. Idempotent: returns immediately if the file exists.
    """
    raw_dir = project_root / "data" / "raw"
    if output_path is None:
        suffix = f"{sample_size // 1000}k" if sample_size % 1000 == 0 else str(sample_size)
        output_path = project_root / "data" / "cache" / f"sample_hotels_{suffix}.jsonl"
    if output_path.exists():
        logger.info("benchmark sample already exists: %s", output_path)
        return output_path

    business_ids = _collect_category_business_ids(raw_dir, category)

    stream, _handles = _open_tar_member(raw_dir, "review")
    kept: list[str] = []
    scanned = 0
    for line in stream:
        scanned += 1
        r = json.loads(line)
        if r["business_id"] not in business_ids:
            continue
        if len(r.get("text", "")) < min_text_len:
            continue
        kept.append(json.dumps(r))
        if len(kept) >= sample_size:
            break
        if len(kept) % 500 == 0:
            logger.info("collected %d/%d (scanned %d)", len(kept), sample_size, scanned)

    if len(kept) < sample_size:
        logger.warning(
            "dataset exhausted at %d/%d matching reviews", len(kept), sample_size
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(kept) + "\n")
    logger.info("wrote %d reviews -> %s (scanned %d)", len(kept), output_path, scanned)
    return output_path


def subset_sample(source: Path, target: Path, n: int) -> Path:
    """
    Derive a smaller benchmark file as the first *n* rows of a larger one,
    so e.g. the 1k smoke sample is a strict prefix of the 5k benchmark and
    results stay comparable across scales.
    """
    if target.exists():
        return target
    with open(source) as fin, open(target, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
    return target


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the hotel benchmark sample")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--category", default="Hotels")
    args = parser.parse_args()

    from ..core.config import find_project_root

    build_benchmark_sample(find_project_root(), args.sample_size, category=args.category)
