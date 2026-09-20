"""
Europarl v7 source adapter — English-original parliamentary speeches.

Raw input is the full source release (``data/raw/europarl.tgz`` from
statmt.org/europarl/v7), which ships one directory of debate transcripts per
language: ``txt/<lang>/ep-YY-MM-DD[-NNN].txt``. Each file is a sequence of
``<CHAPTER>`` headings and ``<SPEAKER>`` blocks, with ``<P>`` paragraph marks.

Identifying the English speeches
--------------------------------
The ``LANGUAGE`` attribute on ``<SPEAKER>`` is **omitted when the speech was
originally delivered in that file's own language**, and present (naming the
source language) when the text is a translation. This is the inverse of the
obvious reading, and getting it backwards silently yields a corpus of
translationese. Measured on this dump:

* ``txt/en/`` carries only **95** blocks tagged ``LANGUAGE="EN"`` out of
  211,359 — annotation noise, not the English set.
* 103,650 blocks in ``txt/en/`` carry **no** LANGUAGE attribute.
* Cross-checking 600 files against ``txt/de/``: speakers 2 and 4 of
  ``ep-00-01-17`` are untagged in ``txt/en/`` and tagged ``LANGUAGE="EN"`` in
  ``txt/de/``; speaker 8 is tagged ``LANGUAGE="DE"`` in ``txt/en/`` and
  untagged in ``txt/de/``. The convention holds in both directions.

"Untagged in ``txt/en/``" is therefore necessary but not sufficient. Of 18,371
untagged blocks in that 600-file sample, 34.9% were confirmed English by a
pivot file, 58.0% were unannotated everywhere, and **6.3% were contradicted**
— a pivot named a different source language, so they are translations whose
English tag was simply dropped. Hence two selection modes:

* ``"verified"`` (default) — untagged in ``txt/en/`` **and** explicitly tagged
  ``LANGUAGE="EN"`` in at least one pivot language. Near-zero translationese
  at the cost of recall, and the only mode whose output supports a claim about
  English-original parliamentary language.
* ``"untagged"`` — untagged in ``txt/en/``. Larger, ~11% contaminated among
  non-chair speakers. Use when corpus size matters more than purity.

Excluding the chair
-------------------
44% of untagged blocks in the sample are the sitting President ("President",
"President.", "President. -"), and 71% of the everywhere-unannotated group is
the chair. These are procedural formulae — "The debate is closed", "The vote
will take place tomorrow" — repeated thousands of times. They are unannotated
because nobody recorded the chair's language, not because they are English,
and at clustering time they are precisely the near-duplicate mass that forms
the giant junk cluster ``eval.harness.failure_flags`` exists to catch. Dropped
by default; ``exclude_chair=False`` keeps them.

No ratings
----------
Parliamentary speech carries no star rating, so records have no rating field
and ``load_benchmark`` yields ``stars = NaN``. Tier-3 rating entropy is
therefore unavailable on this corpus — see ``core.metrics.category_entropy``
for the metadata-based substitute this corpus can support instead.
"""
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
# Pivot languages used to confirm an English original. More pivots = more
# confirmations (a speech may be annotated in one file and not another); these
# five are the largest directories, which is where the annotations are densest.
DEFAULT_PIVOTS = ("de", "fr", "es", "it", "nl")

_SPEAKER_RE = re.compile(r"^<SPEAKER\b([^>]*)>")
_CHAPTER_RE = re.compile(r"^<CHAPTER\b[^>]*>")
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_ID_RE = re.compile(r'\bID\s*=\s*"?(\d+)"?')
# Filename -> sitting date: ep-00-01-17.txt, ep-11-03-09-010-03.txt
_DATE_RE = re.compile(r"^ep-(\d{2})-(\d{2})-(\d{2})")

# The chair, with the punctuation noise the dump actually contains
# ("President", "President.", "President. –", "President. -", " President").
# Matched on the normalised name, so "President-in-Office of the Council" — a
# substantive Council speaker, not the chair — is deliberately NOT caught.
_CHAIR_NAMES = {"president", "the president", "presidente"}


@dataclass
class Speech:
    """One ``<SPEAKER>`` block."""

    speech_id: str          # "ep-00-01-17#2" — file stem + speaker id
    text: str
    speaker: str
    date: str               # ISO sitting date derived from the filename
    chapter: Optional[str]  # enclosing agenda item, when one was seen
    language: Optional[str] # source-language attribute; None = original here
    source_file: str

    @property
    def is_chair(self) -> bool:
        return _normalise_name(self.speaker) in _CHAIR_NAMES


def _normalise_name(name: str) -> str:
    """'President. –' -> 'president'; strips trailing punctuation and spaces."""
    return name.strip().rstrip(".-–—− ").strip().lower()


def _file_date(stem: str) -> str:
    """'ep-00-01-17' -> '2000-01-17'. Europarl v7 spans 1996-2011."""
    m = _DATE_RE.match(stem)
    if not m:
        return ""
    yy, mm, dd = (int(g) for g in m.groups())
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return f"{year:04d}-{mm:02d}-{dd:02d}"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_transcript(path: Path) -> Iterator[Speech]:
    """Parse one ``ep-*.txt`` transcript into :class:`Speech` records."""
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
            # LANGUAGE="" (15,593 blocks) is an empty annotation, not a source
            # language; treat it as unannotated rather than as a language.
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
                # The line after <CHAPTER> is the agenda item's title.
                chapter = stripped
                expecting_chapter_title = False
                continue
            if speaker_id is not None:
                parts.append(stripped)
    done = flush()
    if done:
        yield done


def _english_ids_from_pivot(pivot_dir: Path) -> set[tuple[str, str]]:
    """``{(file stem, speaker id)}`` this pivot language tags as English.

    Only ``<SPEAKER>`` lines are inspected — the speech bodies are irrelevant
    here and skipping them keeps a full-directory scan to a few seconds.
    """
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


# ── Building the benchmark sample ─────────────────────────────────────────────

def ensure_extracted(raw_dir: Path, languages: tuple[str, ...]) -> Path:
    """Extract the ``txt/<lang>/`` directories we need from the tarball.

    Idempotent and incremental: a language already on disk is never re-read out
    of the 1.5 GB archive, and only the members actually wanted are extracted.
    """
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
    """
    Write ``data/cache/sample_europarl-en_{N}k.jsonl``: English-original
    European Parliament speeches in sitting-date order.

    Parameters
    ----------
    selection : "verified" | "untagged"
        How an English original is identified — see the module docstring.
        "verified" additionally requires a pivot language to tag the speech
        ``LANGUAGE="EN"``.
    exclude_chair : bool
        Drop the sitting President's procedural boilerplate (default True).

    Deterministic and idempotent: files are visited in sorted (chronological)
    order and the function returns immediately if the output already exists,
    so the sample is reproducible on a fresh checkout and a 1k subset is a
    strict prefix of the 5k one.
    """
    if selection not in ("verified", "untagged"):
        raise ValueError(
            f"Unknown selection {selection!r}; choose 'verified' or 'untagged'"
        )
    raw_dir = project_root / "data" / "raw"
    if output_path is None:
        suffix = f"{sample_size // 1000}k" if sample_size % 1000 == 0 else str(sample_size)
        # Hyphen, not underscore: PipelineConfig.corpus_slug splits the stem on
        # its LAST underscore, so "europarl-en" round-trips as one token.
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
    # Chronological, not alphabetical: the corpus spans 1996-2011 and a plain
    # filename sort puts "ep-00-..." (2000) ahead of "ep-96-..." (1996), which
    # would make "the first N speeches" a confusing slice of the timeline.
    for path in sorted(en_dir.glob("*.txt"), key=lambda p: (_file_date(p.stem), p.name)):
        for speech in parse_transcript(path):
            scanned += 1
            if speech.language is not None:
                n_translated += 1          # tagged = translated into English
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
