"""Tests for FallbackProvider — primary/fallback LLM provider wrapper."""

import pytest

from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.providers.fallback_provider import FallbackProvider


class _FakeProvider:
    """Minimal async provider for testing. Implements enough of LLMProvider interface."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs: dict = {}
        self.api_key = "fake-key"
        self.api_base = "https://fake.api"
        self.generation = GenerationSettings()

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_kwargs = kwargs
        return self._responses.pop(0)

    async def chat_stream_with_retry(self, **kwargs) -> LLMResponse:
        return await self.chat_with_retry(**kwargs)

    def get_default_model(self) -> str:
        return "fake-model"


_OK = LLMResponse(content="hello from primary")
_ERR = LLMResponse(content="Error: connection refused", finish_reason="error")
_OK_FALLBACK = LLMResponse(content="hello from fallback")
_ERR_FALLBACK = LLMResponse(content="Error: fallback also down", finish_reason="error")
_TOOLS = [{"type": "function", "function": {"name": "test", "parameters": {}}}]


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback_call():
    primary = _FakeProvider([_OK])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    result = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert result.content == "hello from primary"
    assert result.finish_reason == "stop"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_primary_fails_fallback_succeeds_with_warning():
    primary = _FakeProvider([_ERR])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    result = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert "⚠️" in result.content
    assert "hello from fallback" in result.content
    assert result.finish_reason == "stop"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_primary_fails_fallback_also_fails():
    primary = _FakeProvider([_ERR])
    fallback = _FakeProvider([_ERR_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    result = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert result.finish_reason == "error"
    assert "fallback also down" in result.content
    assert "⚠️" not in result.content  # no warning on double failure


@pytest.mark.asyncio
async def test_fallback_passes_tools():
    primary = _FakeProvider([_ERR])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
    )

    assert fallback.last_kwargs.get("tools") == _TOOLS


@pytest.mark.asyncio
async def test_fallback_uses_fallback_model():
    primary = _FakeProvider([_ERR])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="openrouter/free")

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        model="primary-model",
    )

    assert primary.last_kwargs.get("model") == "primary-model"
    assert fallback.last_kwargs.get("model") == "openrouter/free"


@pytest.mark.asyncio
async def test_streaming_primary_succeeds():
    primary = _FakeProvider([_OK])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    result = await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert result.content == "hello from primary"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_streaming_primary_fails_fallback_succeeds():
    primary = _FakeProvider([_ERR])
    fallback = _FakeProvider([_OK_FALLBACK])
    provider = FallbackProvider(primary, fallback, fallback_model="fallback-model")

    result = await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert "⚠️" in result.content
    assert "hello from fallback" in result.content
    assert primary.calls == 1
    assert fallback.calls == 1


def test_delegates_properties_to_primary():
    primary = _FakeProvider([])
    fallback = _FakeProvider([])
    primary.api_key = "primary-key"
    primary.api_base = "https://primary.api"
    primary.generation = GenerationSettings(temperature=0.5, max_tokens=1234)

    provider = FallbackProvider(primary, fallback, fallback_model="fb")

    assert provider.api_key == "primary-key"
    assert provider.api_base == "https://primary.api"
    assert provider.generation.temperature == 0.5
    assert provider.generation.max_tokens == 1234
    assert provider.get_default_model() == "fake-model"
