#!/usr/bin/env bash
# Sentence-level embedding sweep on the 5k Hotels benchmark.
#
# Establishes the ranking at 5k (~43k segments); finalists are confirmed at 50k
# separately, because the seeded single-threaded UMAP fit dominates wall-clock
# at ~300k segments and must be spent deliberately.
#
# GPU etiquette: --gpus 2 claims at most two *idle* devices, deliberately
# leaving headroom on the shared box — one device is reserved for the Ollama
# labeler and one for whoever else turns up.
#
# Long-running — meant for tmux.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

python -W ignore -m reviewscope_ml.eval.model_sweep \
    --sample-size 5000 \
    --device cuda \
    --gpus 2 \
    --sentence-level \
    "$@" 2>&1 | grep -viE "^(Batches|Loading weights|Fetching|Computing)"

echo "=== sentence sweep done ==="
