#!/bin/sh
set -e

# Models the app needs: the cluster labeler (reviewscope_ml APP_LABEL_MODEL)
# and the chat model (OLLAMA_CHAT_MODEL). Pulls are no-ops once the models are
# in the volume.
MODELS="${OLLAMA_MODELS:-gemma3:4b llama3.2}"
READY=/tmp/models-ready

rm -f "$READY"
ollama serve &
server_pid=$!

trap 'kill "$server_pid" 2>/dev/null' TERM INT

until ollama list >/dev/null 2>&1; do
  sleep 1
done

for model in $MODELS; do
  echo "entrypoint: ensuring Ollama model '$model' is present..."
  ollama pull "$model"
done
touch "$READY"
echo "entrypoint: models ready: $MODELS"

wait "$server_pid"
