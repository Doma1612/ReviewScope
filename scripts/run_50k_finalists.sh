#!/usr/bin/env bash
# Confirm the sentence-level finalists at scale: 50k Hotels reviews = 454,493
# segments.
#
# This is HOURS of work, not minutes, and the reason is structural: UMAP is
# deterministic only with a fixed seed, a fixed seed forces the single-threaded
# layout, and the layout is O(n) in points with a large constant. At 454k
# segments the UMAP fit dominates everything else — the MiniLM embed stage is
# ~2 minutes of it. Run it in tmux, expect to come back later, and do not
# casually re-run it.
#
# Only the finalists are confirmed here. The RANKING was established at 5k
# (model_sweep_5000_sent.md); this run answers the narrower question of whether
# that ranking survives a 10x larger corpus, where min_cluster_size scaling and
# density both change.
#
# GPU etiquette: claims at most 2 idle devices, never a busy one.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

python -W ignore -m reviewscope_ml.eval.model_sweep \
    --sample-size 50000 \
    --device cuda \
    --gpus 2 \
    --sentence-level \
    --tag finalists \
    --models all-MiniLM arctic-embed \
    "$@" 2>&1 | grep -viE "^(Batches|Loading weights|Fetching)"

echo "=== 50k finalists done ==="
