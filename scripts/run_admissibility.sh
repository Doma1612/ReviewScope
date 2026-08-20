#!/usr/bin/env bash
# Admissibility + cost probe over the whole embedding registry.
#
# Verifies the hard constraints on the actual box (plain sentence-transformers,
# no trust_remote_code, fp32 inside the VRAM slice) and measures comparable
# throughput / peak VRAM per candidate on one pinned idle GPU.
#
# Long-running (downloads several GB of weights on first run) — meant for tmux.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

python -W ignore -m reviewscope_ml.eval.cost_probe \
    --sample-size 5000 \
    --device cuda \
    --n-texts 2048 \
    "$@" 2>&1 | grep -viE "^(Batches|Loading weights|Fetching)"

echo "=== admissibility probe done ==="
