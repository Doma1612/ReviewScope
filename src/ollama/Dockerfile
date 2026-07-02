# Ollama, wrapped so the label model is present without a manual
# `docker compose exec ollama ollama pull ...` step. The entrypoint starts the
# server, pulls the model (idempotent — a no-op once it is in the ollama_data
# volume), then hands the server the foreground.
FROM ollama/ollama:latest

# Invoked via `sh` (below) so no `RUN chmod` is needed — this keeps the image
# build to FROM + COPY with zero sub-process spawns, which matters on hosts whose
# old Docker seccomp profile blocks clone3 (see src/backend/Dockerfile).
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]
