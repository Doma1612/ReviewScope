from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaChatError(RuntimeError):
    pass


def chat_message(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    tools: list[dict] | None = None,
    response_format: str | dict | None = None,
) -> dict:
    """Call Ollama and return its complete assistant message, including tool calls."""
    url = f"{base_url.rstrip('/')}/api/chat"
    request_payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        request_payload["tools"] = tools
    if response_format is not None:
        request_payload["format"] = response_format
    body = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaChatError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaChatError(f"Could not reach Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaChatError("Ollama returned an invalid JSON response") from exc

    message = payload.get("message")
    if not isinstance(message, dict):
        raise OllamaChatError("Ollama response did not include a message")
    return message


def chat_completion(*, base_url: str, model: str, messages: list[dict[str, str]], timeout: int, response_format: str | dict | None = None) -> str:
    """Call Ollama's native non-streaming chat API.

    This deliberately has no API-key handling: local Docker, localhost, and LAN
    Ollama deployments are the first supported target for ReviewScope chat.
    """
    message = chat_message(
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        response_format=response_format,
    )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaChatError("Ollama response did not include message.content")
    return content.strip()


def stream_chat_completion(*, base_url: str, model: str, messages: list[dict[str, str]], timeout: int):
    """Yield content chunks from Ollama's native streaming chat API."""
    url = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OllamaChatError("Ollama returned an invalid streaming JSON response") from exc
                if payload.get("done"):
                    break
                content = payload.get("message", {}).get("content")
                if isinstance(content, str) and content:
                    yield content
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaChatError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaChatError(f"Could not reach Ollama at {url}: {exc}") from exc
