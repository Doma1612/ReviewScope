#!/usr/bin/env bash
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
