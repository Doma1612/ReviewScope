from __future__ import annotations

import json
import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("reviewscope.data.europarl")

RAW_TARBALL = "europarl.tgz"
TXT_ROOT = "txt"
DEFAULT_PIVOTS = ("de", "fr", "es", "it", "nl")

_SPEAKER_RE = re.compile(r"^<SPEAKER\b([^>]*)>")
_CHAPTER_RE = re.compile(r"^<CHAPTER\b[^>]*>")
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_ID_RE = re.compile(r'\bID\s*=\s*"?(\d+)"?')
_DATE_RE = re.compile(r"^ep-(\d{2})-(\d{2})-(\d{2})")

_CHAIR_NAMES = {"president", "the president", "presidente"}


@dataclass
class Speech:

    speech_id: str
    text: str
    speaker: str
    date: str
    chapter: Optional[str]
    language: Optional[str]
    source_file: str

    @property
    def is_chair(self) -> bool:
        return _normalise_name(self.speaker) in _CHAIR_NAMES


def _normalise_name(name: str) -> str:
    return name.strip().rstrip(".-–—− ").strip().lower()


def _file_date(stem: str) -> str:
    m = _DATE_RE.match(stem)
    if not m:
        return ""
    yy, mm, dd = (int(g) for g in m.groups())
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def parse_transcript(path: Path) -> Iterator[Speech]:
    stem = path.stem
    date = _file_date(stem)
    chapter: Optional[str] = None
    expecting_chapter_title = False
    attrs: dict[str, str] = {}
    speaker_id: Optional[str] = None
    parts: list[str] = []

    def flush() -> Optional[Speech]:
        if speaker_id is None:
            return None
        text = " ".join(parts).strip()
        if not text:
            return None
        return Speech(
            speech_id=f"{stem}#{speaker_id}",
            text=text,
            speaker=attrs.get("NAME", ""),
            date=date,
            chapter=chapter,
            # An empty LANGUAGE annotation is unannotated, not a source language.
            language=attrs.get("LANGUAGE") or None,
            source_file=path.name,
        )

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if _CHAPTER_RE.match(line):
                done = flush()
                if done:
                    yield done
                speaker_id, parts, attrs = None, [], {}
                expecting_chapter_title = True
                continue
            m = _SPEAKER_RE.match(line)
            if m:
                done = flush()
                if done:
                    yield done
                raw = m.group(1)
                attrs = dict(_ATTR_RE.findall(raw))
                id_match = _ID_RE.search(raw)
                speaker_id = id_match.group(1) if id_match else "0"
                parts = []
                expecting_chapter_title = False
                continue
            stripped = line.strip()
            if not stripped or stripped == "<P>":
                continue
            if expecting_chapter_title:
                chapter = stripped
                expecting_chapter_title = False
                continue
            if speaker_id is not None:
                parts.append(stripped)
    done = flush()
    if done:
        yield done


def _english_ids_from_pivot(pivot_dir: Path) -> set[tuple[str, str]]:
    confirmed: set[tuple[str, str]] = set()
    for path in pivot_dir.glob("*.txt"):
        stem = path.stem
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("<SPEAKER"):
                    continue
                m = _SPEAKER_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                raw = m.group(1)
                if 'LANGUAGE="EN"' not in raw.upper().replace("'", '"'):
                    continue
                id_match = _ID_RE.search(raw)
                if id_match:
                    confirmed.add((stem, id_match.group(1)))
    return confirmed


def ensure_extracted(raw_dir: Path, languages: tuple[str, ...]) -> Path:
    txt_root = raw_dir / TXT_ROOT
    missing = [l for l in languages if not (txt_root / l).is_dir()]
    if not missing:
        return txt_root

    tarball = raw_dir / RAW_TARBALL
    if not tarball.exists():
        raise FileNotFoundError(
            f"Europarl source release not found: {tarball}\n"
            "Download it with:\n"
            f"  curl -L -o {tarball} https://www.statmt.org/europarl/v7/europarl.tgz"
        )
    logger.info("extracting %s from %s", missing, tarball.name)
    wanted = tuple(f"{TXT_ROOT}/{l}" for l in missing)
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if member.name.startswith(wanted):
                tar.extract(member, path=raw_dir)
    return txt_root


def build_europarl_sample(
    project_root: Path,
    sample_size: int,
    *,
    selection: str = "verified",
    pivots: tuple[str, ...] = DEFAULT_PIVOTS,
    exclude_chair: bool = True,
    min_text_len: int = 50,
    output_path: Optional[Path] = None,
) -> Path:
    if selection not in ("verified", "untagged"):
        raise ValueError(
            f"Unknown selection {selection!r}; choose 'verified' or 'untagged'"
        )
    raw_dir = project_root / "data" / "raw"
    if output_path is None:
        suffix = f"{sample_size // 1000}k" if sample_size % 1000 == 0 else str(sample_size)
        # Use a hyphen: corpus_slug splits on the final underscore.
        output_path = project_root / "data" / "cache" / f"sample_europarl-en_{suffix}.jsonl"
    if output_path.exists():
        logger.info("europarl sample already exists: %s", output_path)
        return output_path

    languages = ("en", *pivots) if selection == "verified" else ("en",)
    txt_root = ensure_extracted(raw_dir, languages)
    en_dir = txt_root / "en"

    confirmed: set[tuple[str, str]] = set()
    if selection == "verified":
        for lang in pivots:
            found = _english_ids_from_pivot(txt_root / lang)
            confirmed |= found
            logger.info("pivot %s confirms %d English speeches", lang, len(found))
        logger.info("%d distinct speeches confirmed English by %d pivots",
                    len(confirmed), len(pivots))

    kept: list[str] = []
    scanned = n_translated = n_chair = n_short = n_unconfirmed = 0
    # Sort by date: two-digit filename years put 2000 before 1996 lexically.
    for path in sorted(en_dir.glob("*.txt"), key=lambda p: (_file_date(p.stem), p.name)):
        for speech in parse_transcript(path):
            scanned += 1
            if speech.language is not None:
                n_translated += 1          # A language tag means translated into English, not originally English.
                continue
            if exclude_chair and speech.is_chair:
                n_chair += 1
                continue
            if len(speech.text) < min_text_len:
                n_short += 1
                continue
            if selection == "verified":
                stem, _, sid = speech.speech_id.partition("#")
                if (stem, sid) not in confirmed:
                    n_unconfirmed += 1
                    continue
            kept.append(json.dumps({
                "speech_id": speech.speech_id,
                "text": speech.text,
                "speaker": speech.speaker,
                "date": speech.date,
                "chapter": speech.chapter,
                "source_file": speech.source_file,
            }, ensure_ascii=False))
            if len(kept) >= sample_size:
                break
            if len(kept) % 1000 == 0:
                logger.info("collected %d/%d (scanned %d blocks)",
                            len(kept), sample_size, scanned)
        if len(kept) >= sample_size:
            break

    if len(kept) < sample_size:
        logger.warning("corpus exhausted at %d/%d speeches", len(kept), sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    logger.info(
        "wrote %d speeches -> %s (scanned %d; dropped %d translated, %d chair, "
        "%d short, %d unconfirmed)",
        len(kept), output_path, scanned, n_translated, n_chair, n_short, n_unconfirmed,
    )
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the Europarl English sample")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--selection", choices=("verified", "untagged"), default="verified")
    parser.add_argument("--include-chair", action="store_true",
                        help="keep the sitting President's procedural speeches")
    args = parser.parse_args()

    from ...core.config import find_project_root

    build_europarl_sample(
        find_project_root(), args.sample_size,
        selection=args.selection, exclude_chair=not args.include_chair,
    )
