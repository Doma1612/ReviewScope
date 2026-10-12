#!/usr/bin/env bash
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
