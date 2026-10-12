FROM ollama/ollama:latest

# Invoke via sh to avoid RUN/chmod subprocesses blocked by the host's old seccomp profile.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]
