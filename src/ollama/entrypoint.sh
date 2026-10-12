#!/bin/sh
set -e

MODEL="${OLLAMA_LABEL_MODEL:-llama3.2}"

ollama serve &
server_pid=$!

trap 'kill "$server_pid" 2>/dev/null' TERM INT

until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "entrypoint: ensuring Ollama model '$MODEL' is present..."
ollama pull "$MODEL"
echo "entrypoint: model '$MODEL' ready."

wait "$server_pid"
