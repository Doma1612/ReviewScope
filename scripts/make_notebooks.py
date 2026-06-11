"""Generate the thin orchestration notebooks 10 and 11.

Notebooks must contain no logic — only src calls and rendering — so they are
written programmatically from this script to keep them reproducible.
Run: .venv/bin/python scripts/make_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


def save(nb, name):
    nbf.validate(nb)
    path = NB_DIR / name
    nbf.write(nb, path)
    print(f"wrote {path}")


# ── 10_pipeline_comparison ────────────────────────────────────────────────────
nb10 = nbf.v4.new_notebook()
nb10.cells = [
    md(
        "# 10 — Four-pipeline comparison\n\n"
        "Thin orchestration only — all logic lives in `src/reviewscope_ml/`.\n\n"
        "Runs the four end-to-end candidates on the benchmark sample and renders the\n"
        "comparison report:\n\n"
        "| Variant | Description |\n"
        "|---|---|\n"
        "| `bertopic` | BERTopic off-the-shelf (MiniLM, its own UMAP+HDBSCAN; seeded) |\n"
        "| `custom_hdbscan` | mpnet → UMAP(10d) → HDBSCAN (notebooks 04–06 decisions) |\n"
        "| `flat_agglomerative` | mpnet → UMAP(10d) → agglomerative (ward) |\n"
        "| `two_stage` | fine HDBSCAN micro-clusters → agglomerative macro merge |\n\n"
        "**Decision protocol** (also embedded in the report): metrics shortlist 2–3\n"
        "finalists; a human picks the winner from the qualitative inspection and the\n"
        "intruder test. Metrics never declare the winner alone.\n\n"
        "Runs top-to-bottom on CPU at 1k in a few minutes (embeddings are cached after\n"
        "the first run). For the full 5k/50k GPU runs see `docs/methodology.md`."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')  # notebooks/utils shims; package itself is pip-installed\n"
        "\n"
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')\n"
        "\n"
        "from reviewscope_ml import load_config\n"
        "\n"
        "# CPU smoke configuration — keep this runnable on any laptop.\n"
        "# GPU: use reviewscope_ml.runtime.claim_gpu() and pass device/gpu_id (see docs).\n"
        "cfg = load_config(sample_size=1_000, device='cpu')\n"
        "cfg.ensure_dirs()\n"
        "print(cfg)"
    ),
    code(
        "# Benchmark sample (built from the raw Yelp dump if missing; idempotent)\n"
        "from reviewscope_ml.data import build_benchmark_sample, subset_sample\n"
        "\n"
        "five_k = build_benchmark_sample(cfg.project_root, 5_000)\n"
        "subset_sample(five_k, cfg.data_path, cfg.sample_size)\n"
        "print('benchmark:', cfg.data_path)"
    ),
    code(
        "# Run all four variants + multi-seed stability repeats, write the report.\n"
        "# label_clusters=True uses Ollama when reachable and falls back to term\n"
        "# labels (recorded as label_source='terms_fallback') when not.\n"
        "from reviewscope_ml.eval.report import run_comparison\n"
        "\n"
        "report_path = run_comparison(cfg, seeds=(42, 43, 44), label_clusters=True)\n"
        "report_path"
    ),
    code(
        "# Render the report inline (same file a teammate reads on disk)\n"
        "from IPython.display import Markdown\n"
        "\n"
        "Markdown(report_path.read_text())"
    ),
    code(
        "# Side-by-side 2-D scatter of the four runs\n"
        "import matplotlib.pyplot as plt\n"
        "from reviewscope_ml.pipelines import load_run\n"
        "from reviewscope_ml.pipelines.spec import VARIANTS\n"
        "\n"
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n"
        "for ax, variant in zip(axes, VARIANTS):\n"
        "    art = load_run(cfg.runs_dir / f'{variant}__{cfg.sample_size}__s42')\n"
        "    noise = art.labels == -1\n"
        "    ax.scatter(art.coords_2d[noise, 0], art.coords_2d[noise, 1], s=2, c='lightgrey', alpha=0.3)\n"
        "    ax.scatter(art.coords_2d[~noise, 0], art.coords_2d[~noise, 1], s=2,\n"
        "               c=art.labels[~noise], cmap='tab20', alpha=0.6)\n"
        "    ax.set_title(f\"{variant}\\n{art.metrics['n_clusters']} clusters, \"\n"
        "                 f\"noise {art.metrics['noise_ratio']:.0%}\")\n"
        "    ax.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Next step — the human part\n\n"
        "1. Read the **qualitative inspection** and take the **intruder test** in the\n"
        "   report above for each shortlisted finalist.\n"
        "2. Open the HITL app and record your verdict:\n"
        "   ```bash\n"
        "   streamlit run src/reviewscope_ml/hitl/app.py\n"
        "   ```\n"
        "3. Only after the sign-off is recorded does the comparison have a winner.\n"
        "   See `11_hitl_roundtrip.ipynb` for how recorded feedback changes the next run."
    ),
]
save(nb10, "10_pipeline_comparison.ipynb")

# ── 11_hitl_roundtrip ─────────────────────────────────────────────────────────
nb11 = nbf.v4.new_notebook()
nb11.cells = [
    md(
        "# 11 — HITL feedback round-trip\n\n"
        "Demonstrates the full loop: pipeline run → human feedback → feedback applied →\n"
        "measurably different artifacts. Thin orchestration only; semantics live in\n"
        "`src/reviewscope_ml/hitl/apply_feedback.py` (its module docstring is the\n"
        "authoritative description of what each action does on re-run).\n\n"
        "Normally feedback comes from the Streamlit app\n"
        "(`streamlit run src/reviewscope_ml/hitl/app.py`); here we write the same\n"
        "records programmatically so the notebook runs unattended."
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import pandas as pd\n"
        "from reviewscope_ml import load_config\n"
        "from reviewscope_ml.data import load_benchmark\n"
        "from reviewscope_ml.pipelines import load_run\n"
        "\n"
        "cfg = load_config(sample_size=1_000, device='cpu')\n"
        "reviews = load_benchmark(cfg)\n"
        "\n"
        "# Two-stage run from notebook 10 — its micro→macro hierarchy lets a split\n"
        "# be answered without re-clustering.\n"
        "RUN = f'two_stage__{cfg.sample_size}__s42'\n"
        "art = load_run(cfg.runs_dir / RUN)\n"
        "print(f'{RUN}: {len(art.clusters)} clusters')"
    ),
    code(
        "# Before: cluster overview\n"
        "def overview(a):\n"
        "    return pd.DataFrame([\n"
        "        {'cluster': cid, 'label': i.label, 'size': i.size,\n"
        "         'source': i.label_source,\n"
        "         'micro_ids': i.micro_cluster_ids,\n"
        "         'top_terms': ', '.join(w for w, _ in (tuple(t) for t in i.top_terms[:5]))}\n"
        "        for cid, i in sorted(a.clusters.items())\n"
        "    ])\n"
        "\n"
        "before = overview(art)\n"
        "before"
    ),
    code(
        "# A reviewer session, written as records (the Streamlit app produces exactly these)\n"
        "from reviewscope_ml.hitl import FeedbackRecord, append_record, session_file\n"
        "\n"
        "cids = art.cluster_ids\n"
        "session = session_file(cfg.feedback_dir, RUN)\n"
        "actions = [\n"
        "    FeedbackRecord(RUN, 'demo-reviewer', 'rename_label', cluster_id=cids[0],\n"
        "                   new_label='Front desk & check-in experience'),\n"
        "    FeedbackRecord(RUN, 'demo-reviewer', 'approve_label', cluster_id=cids[1]),\n"
        "    FeedbackRecord(RUN, 'demo-reviewer', 'merge_clusters', cluster_id=cids[3],\n"
        "                   merge_into=cids[2], note='same theme, split by phrasing'),\n"
        "    FeedbackRecord(RUN, 'demo-reviewer', 'split_cluster', cluster_id=cids[4],\n"
        "                   note='mixes two distinct complaints'),\n"
        "]\n"
        "for r in actions:\n"
        "    append_record(session, r)\n"
        "print(f'{len(actions)} records -> {session.name}')"
    ),
    code(
        "# Apply: produces <run>__reviewed next to the original (original untouched)\n"
        "from reviewscope_ml.hitl import apply_run_feedback\n"
        "\n"
        "reviewed_dir = apply_run_feedback(cfg.runs_dir / RUN, cfg.feedback_dir, reviews=reviews)\n"
        "reviewed = load_run(reviewed_dir)\n"
        "after = overview(reviewed)\n"
        "after"
    ),
    code(
        "# What changed?\n"
        "print('applied:', *reviewed.manifest['feedback_applied'], sep='\\n  - ')\n"
        "print()\n"
        "print(f'clusters before: {len(art.clusters)}  after: {len(reviewed.clusters)}')\n"
        "print('needs_recluster:', reviewed.manifest.get('needs_recluster'))"
    ),
    md(
        "## What each demonstrated action did\n\n"
        "- **rename_label** — label overridden, `label_source='hitl_override'`; later\n"
        "  LLM passes will not overwrite it.\n"
        "- **approve_label** — recorded as `hitl_approved`; the report can count\n"
        "  human-verified labels.\n"
        "- **merge_clusters** — documents reassigned to the target cluster; term lists\n"
        "  and word frequencies recomputed from the merged membership.\n"
        "- **split_cluster** — the macro cluster was decomposed into its micro-clusters\n"
        "  (promoted to top level). On a flat run the same record flags the cluster in\n"
        "  `needs_recluster` for targeted re-clustering instead.\n\n"
        "The reviewed artifact is a normal run directory — the HITL app, the report\n"
        "renderer and the future backend consume it exactly like an unreviewed one."
    ),
]
save(nb11, "11_hitl_roundtrip.ipynb")
