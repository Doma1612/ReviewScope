#!/bin/sh
# Start the Ollama server, ensure the label model is pulled, then keep the
# server in the foreground as the container's main process.
#
# The pull is idempotent: once the model is in the mounted ollama_data volume it
# is a no-op, so this only downloads on first boot. The model name mirrors the
# reviewscope_ml default (PipelineSpec.label_model = "llama3.2") and can be
# overridden with OLLAMA_LABEL_MODEL.
set -e

MODEL="${OLLAMA_LABEL_MODEL:-llama3.2}"

ollama serve &
server_pid=$!

# Forward termination to the server so `docker compose down` shuts down cleanly.
trap 'kill "$server_pid" 2>/dev/null' TERM INT

# Wait for the daemon to answer before pulling.
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "entrypoint: ensuring Ollama model '$MODEL' is present..."
ollama pull "$MODEL"
echo "entrypoint: model '$MODEL' ready."

wait "$server_pid"
