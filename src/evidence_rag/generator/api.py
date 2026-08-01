"""Anthropic / OpenAI API backends for the Generator module.

Drop-in replacements for GraniteLLMClient — same TextGenerator protocol,
plus native streaming support for the SSE endpoint.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Protocol


class StreamingTokenGenerator(Protocol):
    """Yields (event_type, payload) pairs for the SSE event stream."""

    async def generate_sse_events(
        self, prompt: str
    ) -> AsyncGenerator[tuple[str, dict | str], None]:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicTextGenerator:
    """Anthropic Messages API — non-streaming, implements TextGenerator."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL_ID", "claude-haiku-4-5-20251001")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: object | None = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-untyped]
            except ImportError:
                raise RuntimeError(
                    "pip install anthropic, or uv add anthropic"
                ) from None
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="You are a precise scientific assistant. Answer only using the provided evidence.",
            messages=[{"role": "user", "content": prompt}],
        )
        # content is a list of blocks; take the first text block
        for block in message.content:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        return ""


class AnthropicStreamingGenerator:
    """Anthropic Messages API — streaming, for the SSE endpoint."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: object | None = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-untyped]
            except ImportError:
                raise RuntimeError(
                    "pip install anthropic, or uv add anthropic"
                ) from None
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client

    async def generate_sse_events(
        self, prompt: str
    ) -> AsyncGenerator[tuple[str, dict | str], None]:
        """Yield SSE event tuples: ('chunk', {'text': ...}) then ('done', full_text)."""
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            yield ("error", {"message": "pip install anthropic"})
            return

        full_response = ""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a precise scientific assistant. Answer only using the provided evidence.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield ("chunk", {"text": text})
        except anthropic.APIError as exc:
            yield ("error", {"message": f"Anthropic API error: {exc}"})
            return

        yield ("done", full_response)


