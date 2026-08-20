"""
Regenerate the human scoring sheet from a completed label sweep's JSON.

Useful when the sheet needs re-rendering without paying for LLM generation
again — the labels, their model and their prompt hash are all in
``label_quality_<unit>.json``; only the surrounding cluster context (sizes,
c-TF-IDF terms, random exemplar mentions) has to be rebuilt, and that comes
from caches.

    python scripts/regen_label_sheet.py --sample-size 5000 \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reviewscope_ml.core.config import load_config
from reviewscope_ml.eval.label_quality import render_human_sheet
from reviewscope_ml.eval.label_sweep import build_clusters
from reviewscope_ml.represent.terms import ctfidf_terms

p = argparse.ArgumentParser()
p.add_argument("--sample-size", type=int, default=5_000)
p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
p.add_argument("--instruction", default="no_inst")
p.add_argument("--document-level", action="store_true")
args = p.parse_args()

sentence_level = not args.document_level
unit = "doc" if args.document_level else "sent"
cfg = load_config(sample_size=args.sample_size)
records = json.loads((cfg.runs_dir / f"label_quality_{unit}.json").read_text())

units, _emb, labels = build_clusters(
    cfg, args.embedding_model, args.instruction, sentence_level
)
terms = ctfidf_terms(units.texts, labels, top_n=10)

cluster_ids = sorted({r["cluster_id"] for r in records},
                     key=lambda c: -int((labels == c).sum()))
sizes = {cid: int((labels == cid).sum()) for cid in cluster_ids}

# Same seed and same draw order as the sweep, so the sheet shows the same
# exemplars a reviewer would have seen there.
rng = np.random.default_rng(cfg.seed)
exemplars = {}
for cid in cluster_ids:
    idx = np.flatnonzero(labels == cid)
    pick = rng.choice(idx, size=min(5, len(idx)), replace=False)
    exemplars[cid] = [units.texts[i][:300] for i in pick]

labels_by_model: dict[str, dict[int, str]] = {}
for r in records:
    key = f"{r['labeler']}|{r['prompt_variant']}"
    labels_by_model.setdefault(key, {})[r["cluster_id"]] = r["label"]

out = cfg.runs_dir / f"label_sheet_{unit}.md"
out.write_text(render_human_sheet(cluster_ids, sizes, terms, exemplars, labels_by_model))
print(f"regenerated {out} ({len(labels_by_model)} labelers x {len(cluster_ids)} clusters)")
