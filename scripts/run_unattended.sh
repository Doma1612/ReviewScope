#!/usr/bin/env bash
# Unattended completion of the WP5 technology-selection run, then full cleanup.
#
# Written to be started and walked away from: it waits for work already in
# flight, runs what is left, and then RELEASES THE BOX — our Ollama server is
# stopped and every GPU is handed back. The shared box must not be left with a
# model resident in VRAM just because nobody was watching.
#
# Sequence:
#   0. wait for the in-flight labeler sweep (started separately) to finish
#   1. regenerate the human scoring sheet (pipe-escaping fix)
#   2. fair re-run of qwen3 with reasoning disabled
#   3. confirm the finalists at 50k (MiniLM + arctic) — the long one, hours
#   4. cleanup: stop OUR ollama, end tmux sessions, report GPU state
#
# Cleanup runs from an EXIT trap, so it happens on success, on failure, and on
# Ctrl-C alike. It never touches the root-owned system ollama (pid from a
# different user) — only processes under $HOME/ollama.
#
# Usage:  tmux new-session -d -s revscope-night "bash scripts/run_unattended.sh"
#         tail -f data/runs/unattended.log

cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/data/runs/unattended.log"
STATUS="$REPO/data/runs/unattended_status.txt"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

# PIDs whose *executable* lives under our rootless ollama install.
#
# Matching on the command STRING is not safe here: any shell whose command line
# happens to contain the pattern matches itself, which is how `pkill -f "ollama
# serve"` famously kills the shell that ran it. Resolving /proc/<pid>/exe asks
# what the process actually IS, so a shell mentioning the path can never
# qualify — and root's /bin/ollama never qualifies either.
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
    # End the other sessions last, so their logs are already flushed.
    for s in revscope-label revscope-stab revscope-sweep revscope-probe; do
        tmux kill-session -t "$s" 2>/dev/null && say "closed tmux session $s" || true
    done
    say "─── done ───"
}
trap cleanup EXIT

source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false


# ── 0. wait for the in-flight labeler sweep ──────────────────────────────────
say "step 0: waiting for the in-flight labeler sweep"
# Bounded: qwen3 in thinking mode can hit the labeler's 120s per-request
# timeout on every cluster, so an unbounded wait risks the whole chain
# stalling behind it and the 50k run never starting. If it overruns, carry on
# and let the two share the box — the 50k job's numbers are about cluster
# structure, not throughput.
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

# ── 1. regenerate the human scoring sheet ────────────────────────────────────
# The sweep that produced it imported the pre-fix renderer, which emitted an
# unescaped '|' inside table cells.
say "step 1: regenerating the human scoring sheet"
python -W ignore scripts/regen_label_sheet.py \
    --sample-size 5000 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 || say "WARN: sheet regen failed"

# ── 2. fair qwen3 comparison, reasoning disabled ─────────────────────────────
# qwen3 thinks by default; for a 3-6 word label that is pure latency and its
# reasoning can leak into the answer. Scoring it only with thinking on would
# report a configuration artifact as a model verdict.
say "step 2: qwen3 with reasoning disabled"
timeout 3600 python -W ignore -m reviewscope_ml.eval.label_sweep \
    --sample-size 5000 --device cuda \
    --models qwen3:4b --variants v1 v2_mention --no-think --tag nothink \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --n-clusters 20 2>&1 | grep -viE "^(Batches|Loading weights)" \
    || say "WARN: qwen3 nothink run failed or timed out"

# ── 3. free the labeler GPU before the long job ──────────────────────────────
say "step 3: stopping ollama before the 50k run (nothing left needs it)"
stop_our_ollama
sleep 5

# ── 4. the long one: confirm finalists at 50k ────────────────────────────────
# 50,000 reviews = 454,493 segments. The seeded UMAP fit is single-threaded by
# construction and dominates; this is hours, not minutes.
say "step 4: 50k finalists (MiniLM + arctic) — expect hours"
python -W ignore -m reviewscope_ml.eval.model_sweep \
    --sample-size 50000 --device cuda --gpus 2 --sentence-level \
    --tag finalists --models all-MiniLM arctic-embed \
    2>&1 | grep -viE "^(Batches|Loading weights|Fetching)" \
    || say "WARN: 50k sweep failed"

say "all steps attempted; cleanup follows"
