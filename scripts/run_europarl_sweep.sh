#!/usr/bin/env bash
# Europarl (English-original) sentence-level embedding sweep — unattended.
#
# The WP5 experiment repeated on the Phase-2 corpus: every registry candidate
# under the same fixed downstream pipeline, so the embedding model is again the
# only manipulated variable. 5,000 verified English-original speeches ->
# 58,230 segments (the 5k Hotels benchmark was 43,012, so the two rankings are
# comparable in scale).
#
# Written to be started and walked away from. Cleanup runs from an EXIT trap,
# so it happens on success, on failure and on Ctrl-C alike:
#   - every GPU is handed back (process exit frees VRAM; we verify and report)
#   - the 2.3 GB of extracted Europarl language directories are deleted; they
#     are pure intermediates, re-derivable from data/raw/europarl.tgz, and the
#     benchmark JSONL they produced is already on disk
#   - a status file says what finished and where the report is
#
# Kept deliberately: europarl.tgz (the source of truth), the sample JSONLs, and
# the embedding/UMAP/clustering caches — those ARE the expensive artifacts and
# make any re-run nearly free.
#
# GPU etiquette: --gpus 2 claims at most two *idle* devices (busy ones are
# never taken), leaving half the shared box for whoever else turns up.
#
# Usage:  tmux new-session -d -s revscope-europarl "bash scripts/run_europarl_sweep.sh"
#         tail -f data/runs/europarl_sweep.log
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/data/runs/europarl_sweep.log"
STATUS="$REPO/data/runs/europarl_sweep_status.txt"
EXTRACTS="$REPO/data/raw/txt"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
    local rc=$?
    say "─── cleanup (exit code $rc) ───"

    # Re-derivable intermediates: the sample JSONL is built, so the 2.3 GB of
    # extracted transcripts have no further use. europarl.tgz regenerates them.
    if [ -d "$EXTRACTS" ]; then
        local freed
        freed=$(du -sh "$EXTRACTS" 2>/dev/null | cut -f1)
        rm -rf "$EXTRACTS" && say "removed extracted transcripts ($freed reclaimed)"
    else
        say "no extracted transcripts to remove"
    fi

    sleep 3
    say "final GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    say "our remaining GPU processes (should be none):"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

    {
        echo "finished:  $(date '+%F %T')"
        echo "exit_code: $rc"
        echo
        echo "report:"
        ls -1 "$REPO/data/runs"/model_sweep_5000_europarl-en_sent*.md 2>/dev/null \
            || echo "  (no report produced — check $LOG)"
        echo
        echo "corpus:"
        ls -1 "$REPO/data/cache"/sample_europarl-en_*.jsonl 2>/dev/null || true
        echo
        echo "disk free:"
        df -h "$REPO" | tail -1
    } > "$STATUS"
    say "status written to $STATUS"
    say "─── done ───"
}
trap cleanup EXIT

source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

say "starting Europarl sentence-level sweep (5k speeches -> 58,230 segments)"
say "initial GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

python -W ignore -m reviewscope_ml.eval.model_sweep \
    --sample-size 5000 \
    --data-file sample_europarl-en_5k.jsonl \
    --device cuda \
    --gpus 2 \
    --sentence-level \
    "$@" 2>&1 | grep -viE "^(Batches|Loading weights|Fetching|Computing)" \
    || say "WARN: sweep exited non-zero"

say "sweep finished; cleanup follows"
