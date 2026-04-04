"""LiteLLMProvider -- unified LLM provider backed by litellm.

Supports two modes:
- proxy: calls go through a litellm proxy server (primary)
- direct: calls go through an in-process litellm Router (fallback)
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import litellm
from litellm import Router
from loguru import logger

from nanobot.config.schema import LiteLLMConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.litellm_otel import OTelCallback


class LiteLLMProvider(LLMProvider):
    """Unified LLM provider using litellm.

    Proxy mode is primary. Direct Router mode is used when proxy is unavailable.
    """

    _PROXY_CHECK_INTERVAL = 30.0

    def __init__(self, config: LiteLLMConfig, default_model: str) -> None:
        super().__init__(api_key=config.api_key, api_base=config.api_base)
        self._default_model = default_model
        self._mode = config.mode
        self._proxy_base = config.api_base
        self._proxy_key = config.api_key
        self._proxy_available: bool = config.mode == "proxy"
        self._last_proxy_check: float = 0.0
        self._proxy_lock = asyncio.Lock()

        # Direct mode: initialize Router
        self._router: Router | None = None
        if config.models:
            model_list = [m.model_dump() for m in config.models]
            self._router = Router(
                model_list=model_list,
                fallbacks=config.fallbacks or None,
            )

        # Register OTel callback
        try:
            otel_cb = OTelCallback()
            callbacks = list(litellm.callbacks or [])
            callbacks.append(otel_cb)
            litellm.callbacks = callbacks
        except Exception:
            logger.debug("LiteLLMProvider: failed to register OTel callback")

        if config.default_headers:
            litellm.default_headers = config.default_headers

        # Drop unsupported params (e.g. reasoning_effort for non-OpenAI models)
        litellm.drop_params = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self._default_model
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return await self._do_chat(kwargs)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        model = model or self._default_model
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            stream_options={"include_usage": True},
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return await self._do_chat_stream(kwargs, on_content_delta)

    async def _do_chat(self, kwargs: dict[str, Any]) -> LLMResponse:
        """Execute a chat call with proxy->direct fallback."""
        try:
            if self._mode == "proxy" and self._proxy_available:
                response = await litellm.acompletion(
                    **kwargs,
                    api_base=self._proxy_base,
                    api_key=self._proxy_key,
                    custom_llm_provider="openai",
                )
                return self._parse_response(response)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("LiteLLM proxy unavailable: {} -- falling back to direct", exc)
            self._proxy_available = False
        except Exception:
            raise

        if self._router:
            response = await self._router.acompletion(**kwargs)
            return self._parse_response(response)

        return LLMResponse(content="Error: no LLM backend available", finish_reason="error")

    async def _do_chat_stream(
        self, kwargs: dict[str, Any], on_content_delta: Callable | None
    ) -> LLMResponse:
        """Execute a streaming chat call with proxy->direct fallback."""
        try:
            if self._mode == "proxy" and self._proxy_available:
                response_stream = await litellm.acompletion(
                    **kwargs,
                    api_base=self._proxy_base,
                    api_key=self._proxy_key,
                    custom_llm_provider="openai",
                )
                return await self._consume_stream(response_stream, on_content_delta)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("LiteLLM proxy unavailable: {} -- falling back to direct", exc)
            self._proxy_available = False
        except Exception:
            raise

        if self._router:
            response_stream = await self._router.acompletion(**kwargs)
            return await self._consume_stream(response_stream, on_content_delta)

        return LLMResponse(content="Error: no LLM backend available", finish_reason="error")

    async def _consume_stream(self, stream, on_content_delta) -> LLMResponse:
        """Consume a litellm streaming response."""
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            choice = chunk.choices[0]

            if delta.content:
                content_parts.append(delta.content)
                if on_content_delta:
                    await on_content_delta(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if hasattr(tc, "index") else 0
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                    entry = tool_calls_map[idx]
                    if hasattr(tc, "id") and tc.id:
                        entry["id"] = tc.id
                    if hasattr(tc, "function") and tc.function:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
                }

        content = "".join(content_parts) or None
        tool_calls: list[ToolCallRequest] = []
        if tool_calls_map:
            tool_calls = [
                ToolCallRequest(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=json.loads(tc["arguments"]) if tc["arguments"] else {},
                )
                for tc in tool_calls_map.values()
            ]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_response(self, response) -> LLMResponse:
        """Convert litellm ModelResponse -> nanobot LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCallRequest] = []
        if message.tool_calls:
            tool_calls = [
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        thinking_blocks = None
        if hasattr(message, "thinking") and message.thinking:
            thinking_blocks = [{"type": "thinking", "thinking": message.thinking}]

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=getattr(message, "reasoning_content", None),
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        return self._default_model

    async def check_proxy_health(self) -> None:
        """Background health check: re-enables proxy if it recovers."""
        if not self._proxy_base or self._mode != "proxy":
            return

        now = time.monotonic()
        if now - self._last_proxy_check < self._PROXY_CHECK_INTERVAL:
            return

        async with self._proxy_lock:
            if now - self._last_proxy_check < self._PROXY_CHECK_INTERVAL:
                return
            self._last_proxy_check = now

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._proxy_base}/health")
                if resp.status_code == 200:
                    if not self._proxy_available:
                        logger.info("LiteLLM proxy recovered -- switching back to proxy mode")
                    self._proxy_available = True
                else:
                    self._proxy_available = False
        except Exception:
            self._proxy_available = False
