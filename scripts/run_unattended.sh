#!/usr/bin/env bash

cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/data/runs/unattended.log"
STATUS="$REPO/data/runs/unattended_status.txt"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

# Match executable paths so cleanup cannot kill a shell mentioning Ollama or another install.
our_ollama_pids() {
    local pid exe
    for pid in $(pgrep -f "ollama" 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
        case "$exe" in
            "$HOME"/ollama/*) echo "$pid" ;;
        esac
    done
}

stop_our_ollama() {
    local pids
    pids=$(our_ollama_pids | tr '\n' ' ')
    if [ -z "${pids// /}" ]; then
        say "no rootless ollama running"
        return
    fi
    say "stopping our ollama (pids: $pids)"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 5
    pids=$(our_ollama_pids | tr '\n' ' ')
    if [ -n "${pids// /}" ]; then
        say "still up, forcing: $pids"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 2
    fi
    say "ollama stopped; remaining: $(our_ollama_pids | tr '\n' ' ' || true)"
}

cleanup() {
    local rc=$?
    say "─── cleanup (exit code $rc) ───"
    stop_our_ollama
    sleep 3
    say "final GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    say "our remaining GPU processes (should be none):"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
    {
        echo "finished: $(date '+%F %T')"
        echo "exit_code: $rc"
        echo "artifacts:"
        ls -1 "$REPO/data/runs"/model_sweep_50000*sent*.md 2>/dev/null || echo "  (no 50k report)"
        ls -1 "$REPO/data/runs"/label_quality_*.md 2>/dev/null || true
    } > "$STATUS"
    say "status written to $STATUS"
    for s in revscope-label revscope-stab revscope-sweep revscope-probe; do
        tmux kill-session -t "$s" 2>/dev/null && say "closed tmux session $s" || true
    done
    say "─── done ───"
}
trap cleanup EXIT

source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false


say "step 0: waiting for the in-flight labeler sweep"
# Bound the wait so slow reasoning requests cannot stall the entire chain.
STEP0_MAX_S=7200
waited=0
if tmux has-session -t revscope-label 2>/dev/null; then
    while tmux has-session -t revscope-label 2>/dev/null; do
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge "$STEP0_MAX_S" ]; then
            say "WARN: labeler still running after ${STEP0_MAX_S}s — proceeding anyway"
            break
        fi
    done
    [ "$waited" -lt "$STEP0_MAX_S" ] && say "labeler sweep session ended after ${waited}s"
else
    say "no labeler session in flight"
fi

say "step 1: regenerating the human scoring sheet"
python -W ignore scripts/regen_label_sheet.py \
    --sample-size 5000 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 || say "WARN: sheet regen failed"

say "step 2: qwen3 with reasoning disabled"
timeout 3600 python -W ignore -m reviewscope_ml.eval.label_sweep \
    --sample-size 5000 --device cuda \
    --models qwen3:4b --variants v1 v2_mention --no-think --tag nothink \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-clusters 20 2>&1 | grep -viE "^(Batches|Loading weights)" \
    || say "WARN: qwen3 nothink run failed or timed out"

say "step 3: stopping ollama before the 50k run (nothing left needs it)"
stop_our_ollama
sleep 5

say "step 4: 50k finalists (MiniLM + arctic) — expect hours"
python -W ignore -m reviewscope_ml.eval.model_sweep \
    --sample-size 50000 --device cuda --gpus 2 --sentence-level \
    --tag finalists --models all-MiniLM arctic-embed \
    2>&1 | grep -viE "^(Batches|Loading weights|Fetching)" \
    || say "WARN: 50k sweep failed"

say "all steps attempted; cleanup follows"
