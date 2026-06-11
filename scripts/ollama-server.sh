#!/usr/bin/env bash
# Manage the rootless Ollama server on the shared GPU box.
#
#   scripts/ollama-server.sh start    # pinned + context-capped, detached
#   scripts/ollama-server.sh stop     # stop server AND model runner (frees all VRAM)
#   scripts/ollama-server.sh status   # process + VRAM view
#
# Facts worth knowing:
# - The server holds GPU memory ONLY while a model is loaded; it unloads
#   10 min after the last request (OLLAMA_KEEP_ALIVE) -> 0 VRAM when idle.
# - Without a CUDA pin ollama spreads one small model across ALL visible
#   GPUs (~26 GB observed) — never start it unpinned on this box.
# - "[o]llama" pgrep patterns so this script never kills its own shell.
set -euo pipefail

OLLAMA=${OLLAMA_BIN:-$HOME/ollama/bin/ollama}
LOG=$HOME/ollama/serve.log

freest_gpu_uuid() {
    nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits \
        | sort -n | head -1 | awk -F', ' '{print $2}'
}

case "${1:-status}" in
  start)
    if pgrep -f "[o]llama serve" > /dev/null; then
        echo "already running (PID $(pgrep -f '[o]llama serve'))"; exit 0
    fi
    UUID=$(freest_gpu_uuid)
    echo "starting ollama pinned to $UUID (note: the runner may still pick"
    echo "another FREE gpu — harmless, the pipeline routes around it)"
    CUDA_VISIBLE_DEVICES="$UUID" OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=10m \
        setsid nohup "$OLLAMA" serve > "$LOG" 2>&1 < /dev/null &
    sleep 3
    curl -s --max-time 5 http://localhost:11434/api/version && echo " — up"
    ;;
  stop)
    pkill -f "[o]llama serve" 2>/dev/null && echo "server stopped" || echo "no server running"
    sleep 1
    pkill -f "lib/[o]llama/llama-server" 2>/dev/null && echo "runner stopped (VRAM freed)" || true
    ;;
  status)
    pgrep -af "[o]llama" || echo "no ollama processes"
    echo "---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ;;
  *)
    echo "usage: $0 start|stop|status"; exit 1
    ;;
esac
