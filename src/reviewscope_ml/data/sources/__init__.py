"""
Corpus source adapters: raw third-party dumps -> the benchmark JSONL shape.

Each module here knows exactly one raw distribution format (Yelp's tarred
JSON, Europarl's SGML-ish speech transcripts) and nothing about the pipeline.
The contract in both directions is narrow:

* input  — the vendor's archive under ``data/raw/``
* output — a JSONL file under ``data/cache/`` where every line is one document
  with at least ``text``, plus whatever metadata that corpus carries.

``data.ingest.load_benchmark`` reads that JSONL and is the only thing the rest
of the package talks to, so adding a corpus never touches a pipeline stage.
"""
from .europarl import build_europarl_sample
from .yelp import build_benchmark_sample, subset_sample

__all__ = ["build_benchmark_sample", "subset_sample", "build_europarl_sample"]
