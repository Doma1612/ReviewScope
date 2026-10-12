from __future__ import annotations

import json
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger("reviewscope.data.yelp")

RAW_ZIP_NAME = "Yelp-JSON.zip"
RAW_EXTRACTED_TAR = Path("Yelp-JSON") / "Yelp JSON" / "yelp_dataset.tar"


def _open_tar_member(raw_dir: Path, name_contains: str) -> tuple[IO[bytes], object]:
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
    # The .tar is gzip-compressed; streaming auto-detection avoids decompressor seeks.
    tar = tarfile.open(fileobj=zf.open(tar_entry), mode="r|*")
    for member in tar:
        if name_contains in member.name.lower():
            return tar.extractfile(member), (tar, zf)
    raise FileNotFoundError(f"No tar member matching {name_contains!r} in {zip_path}")


def _collect_category_business_ids(raw_dir: Path, category: str) -> set[str]:
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
    raw_dir = project_root / "data" / "raw"
    if output_path is None:
        from ...core.cache import make_slug

        suffix = f"{sample_size // 1000}k" if sample_size % 1000 == 0 else str(sample_size)
        output_path = (
            project_root / "data" / "cache"
            / f"sample_{make_slug(category)}_{suffix}.jsonl"
        )
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

    from ...core.config import find_project_root

    build_benchmark_sample(find_project_root(), args.sample_size, category=args.category)
