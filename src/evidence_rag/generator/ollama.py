"""Ollama backend — fast local inference via GGUF-quantized models.

Uses Ollama's REST API (localhost:11434). Streaming via /api/generate
with newline-delimited JSON.
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"


class OllamaTextGenerator:
    """Non-streaming Ollama backend — implements TextGenerator protocol."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def generate(self, prompt: str) -> str:
        import urllib.request

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()).get("response", "").strip()


class OllamaStreamingGenerator:
    """Streaming Ollama backend — for the SSE endpoint."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    async def generate_sse_events(
        self, prompt: str
    ) -> AsyncGenerator[tuple[str, dict | str], None]:
        """Yield ('chunk', {'text': ...}) then ('done', full_text)."""
        import httpx

        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            full_response += token
                            yield ("chunk", {"text": token})
                        if chunk.get("done"):
                            break
        except httpx.ConnectError:
            yield ("error", {"message": f"Cannot connect to Ollama at {OLLAMA_HOST}. Is it running?"})
            return
        except Exception as exc:
            yield ("error", {"message": f"Ollama error: {exc}"})
            return

        yield ("done", full_response)
