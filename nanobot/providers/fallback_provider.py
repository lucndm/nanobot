"""Fallback LLM provider — wraps primary + fallback providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse

_FALLBACK_WARNING = "⚠️ Primary model unavailable, using fallback model.\n\n"


class FallbackProvider(LLMProvider):
    """Wraps primary and fallback providers.

    On ``chat_with_retry()`` / ``chat_stream_with_retry()``, calls primary first.
    If primary returns ``finish_reason == "error"``, calls fallback instead.
    Prepends a warning to fallback response content when fallback succeeds.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        fallback_model: str,
    ) -> None:
        # Do NOT call super().__init__() — we delegate all attributes.
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model

    # -- Property delegation to primary --

    @property
    def api_key(self) -> str | None:  # type: ignore[override]
        return self.primary.api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        self.primary.api_key = value

    @property
    def api_base(self) -> str | None:  # type: ignore[override]
        return self.primary.api_base

    @api_base.setter
    def api_base(self, value: str | None) -> None:
        self.primary.api_base = value

    @property
    def generation(self) -> GenerationSettings:  # type: ignore[override]
        return self.primary.generation

    @generation.setter
    def generation(self, value: GenerationSettings) -> None:
        self.primary.generation = value

    def get_default_model(self) -> str:
        return self.primary.get_default_model()

    # -- chat() / chat_stream() — stubs, not called directly --

    async def chat(self, **kwargs: Any) -> LLMResponse:
        """Not used — FallbackProvider overrides chat_with_retry instead."""
        return await self.primary.chat(**kwargs)

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        """Not used — FallbackProvider overrides chat_stream_with_retry instead."""
        return await self.primary.chat_stream(**kwargs)

    # -- Fallback logic --

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        response = await self.primary.chat_with_retry(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if response.finish_reason != "error":
            return response

        logger.warning(
            "Primary provider failed, falling back to %s: %s",
            self.fallback_model,
            (response.content or "")[:120],
        )
        fb_response = await self.fallback.chat_with_retry(
            messages=messages,
            tools=tools,
            model=self.fallback_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if fb_response.finish_reason != "error" and fb_response.content:
            fb_response.content = _FALLBACK_WARNING + fb_response.content
        return fb_response

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        response = await self.primary.chat_stream_with_retry(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )
        if response.finish_reason != "error":
            return response

        logger.warning(
            "Primary provider failed (stream), falling back to %s: %s",
            self.fallback_model,
            (response.content or "")[:120],
        )
        fb_response = await self.fallback.chat_stream_with_retry(
            messages=messages,
            tools=tools,
            model=self.fallback_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )
        if fb_response.finish_reason != "error" and fb_response.content:
            fb_response.content = _FALLBACK_WARNING + fb_response.content
        return fb_response
