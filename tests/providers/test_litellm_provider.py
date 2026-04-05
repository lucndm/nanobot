"""Tests for LiteLLMProvider -- proxy + direct modes, response parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import LiteLLMConfig, LiteLLMModelConfig


def _proxy_config(**overrides):
    cfg = dict(
        api_base="http://localhost:4000",
        api_key="sk-proxy-key",
    )
    cfg.update(overrides)
    return LiteLLMConfig(**cfg)




def _direct_config():
    return LiteLLMConfig(
        models=[
            LiteLLMModelConfig(
                model_name="gpt-4o",
                litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test"},
            )
        ],
        fallbacks=[{"gpt-4o": ["claude-sonnet"]}],
    )


def _fake_response(content="ok", tool_calls=None, finish_reason="stop"):
    """Build a litellm-style ModelResponse."""
    tc_objs = None
    if tool_calls:
        tc_objs = []
        for tc in tool_calls:
            func = MagicMock()
            func.name = tc["name"]
            func.arguments = json.dumps(tc["arguments"])
            tc_obj = MagicMock()
            tc_obj.id = tc["id"]
            tc_obj.function = func
            tc_objs.append(tc_obj)

    message = MagicMock()
    message.content = content
    message.tool_calls = tc_objs
    message.reasoning_content = None
    message.thinking = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# --- Response Parsing Tests ---


@pytest.mark.asyncio
async def test_parse_simple_response():
    with patch("nanobot.providers.litellm_provider.litellm"):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        parsed = provider._parse_response(_fake_response("Hello world"))

    assert parsed.content == "Hello world"
    assert parsed.finish_reason == "stop"
    assert not parsed.has_tool_calls
    assert parsed.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_parse_tool_calls():
    with patch("nanobot.providers.litellm_provider.litellm"):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        parsed = provider._parse_response(
            _fake_response(
                content=None,
                tool_calls=[{"id": "tc_1", "name": "exec", "arguments": {"cmd": "ls"}}],
                finish_reason="tool_calls",
            )
        )

    assert parsed.has_tool_calls
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "tc_1"
    assert parsed.tool_calls[0].name == "exec"
    assert parsed.tool_calls[0].arguments == {"cmd": "ls"}
    assert parsed.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_reasoning_content():
    resp = _fake_response()
    resp.choices[0].message.reasoning_content = "Let me think..."

    with patch("nanobot.providers.litellm_provider.litellm"):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        parsed = provider._parse_response(resp)

    assert parsed.reasoning_content == "Let me think..."


@pytest.mark.asyncio
async def test_parse_thinking_blocks():
    resp = _fake_response()
    resp.choices[0].message.thinking = "Internal reasoning..."

    with patch("nanobot.providers.litellm_provider.litellm"):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        parsed = provider._parse_response(resp)

    assert parsed.thinking_blocks is not None
    assert len(parsed.thinking_blocks) == 1
    assert parsed.thinking_blocks[0]["thinking"] == "Internal reasoning..."


# --- Proxy Mode Tests ---


@pytest.mark.asyncio
async def test_proxy_mode_calls_acompletion_with_api_base():
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["api_base"] == "http://localhost:4000"
    assert call_kwargs["api_key"] == "sk-proxy-key"
    assert call_kwargs["model"] == "gpt-4o"


# --- Direct Mode Tests ---


@pytest.mark.asyncio
async def test_direct_mode_calls_router():
    mock_router_resp = AsyncMock(return_value=_fake_response())
    mock_router = MagicMock()
    mock_router.acompletion = mock_router_resp

    with (
        patch("nanobot.providers.litellm_provider.litellm"),
        patch("nanobot.providers.litellm_provider.Router", return_value=mock_router),
    ):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_direct_config(), "gpt-4o")
        result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    mock_router_resp.assert_called_once()


# --- Proxy -> Direct Fallback Tests ---


@pytest.mark.asyncio
async def test_proxy_fallback_to_direct_on_connection_error():
    import httpx

    mock_router_resp = AsyncMock(return_value=_fake_response("from-direct"))
    mock_router = MagicMock()
    mock_router.acompletion = mock_router_resp

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(side_effect=httpx.ConnectError("refused"))

        config = LiteLLMConfig(
            api_base="http://localhost:4000",
            api_key="sk-proxy",
            models=[
                LiteLLMModelConfig(
                    model_name="gpt-4o",
                    litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test"},
                )
            ],
        )

        with patch("nanobot.providers.litellm_provider.Router", return_value=mock_router):
            from nanobot.providers.litellm_provider import LiteLLMProvider

            provider = LiteLLMProvider(config, "gpt-4o")
            result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "from-direct"
    assert provider._proxy_available is False


@pytest.mark.asyncio
async def test_get_default_model():
    with patch("nanobot.providers.litellm_provider.litellm"):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
    assert provider.get_default_model() == "gpt-4o"


# --- Proxy Mode: custom_llm_provider and drop_params ---


@pytest.mark.asyncio
async def test_proxy_mode_sets_custom_llm_provider_to_openai():
    """Proxy mode must use custom_llm_provider='openai' to pass model name as-is."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "minimax/MiniMax-M2.7")
        result = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["custom_llm_provider"] == "openai"
    assert call_kwargs["model"] == "minimax/MiniMax-M2.7"


@pytest.mark.asyncio
async def test_proxy_stream_mode_sets_custom_llm_provider_to_openai():
    """Proxy streaming mode must also use custom_llm_provider='openai'."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="hi", tool_calls=None))]
    mock_chunk.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)

    async def _aiter():
        yield mock_chunk

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=_aiter())
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "minimax/MiniMax-M2.7")
        await provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            on_content_delta=AsyncMock(),
        )

    call_kwargs = mock_litellm.acompletion.call_args.kwargs
    assert call_kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_init_sets_drop_params():
    """LiteLLMProvider.__init__ must set litellm.drop_params=True."""
    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        LiteLLMProvider(_proxy_config(), "gpt-4o")

    assert mock_litellm.drop_params is True


@pytest.mark.asyncio
async def test_proxy_mode_passes_slashed_model_name_unchanged():
    """Model names with slashes (e.g. 'minimax/MiniMax-M2.7') must not be parsed by litellm."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "minimax/MiniMax-M2.7")
        await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            reasoning_effort="medium",
        )

    call_kwargs = mock_acompletion.call_args.kwargs
    # Model name passed as-is to proxy
    assert call_kwargs["model"] == "minimax/MiniMax-M2.7"
    # custom_llm_provider prevents litellm from parsing provider prefix
    assert call_kwargs["custom_llm_provider"] == "openai"


# --- System Prompt Extraction Tests ---


@pytest.mark.asyncio
async def test_extract_system_message_passed_as_system_param():
    """System message must be extracted and passed via litellm's 'system' param."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful bot."},
                {"role": "user", "content": "hi"},
            ],
        )

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["system"] == "You are a helpful bot."
    # System message must NOT be in messages list
    assert all(m.get("role") != "system" for m in call_kwargs["messages"])


@pytest.mark.asyncio
async def test_no_system_message_no_system_param():
    """When no system message exists, 'system' param must not be set."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        await provider.chat(messages=[{"role": "user", "content": "hi"}])

    call_kwargs = mock_acompletion.call_args.kwargs
    assert "system" not in call_kwargs


# --- Metadata Pass-Through Tests ---


@pytest.mark.asyncio
async def test_metadata_passed_to_litellm():
    """metadata dict must be forwarded to litellm.acompletion."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"channel": "telegram", "chat_id": "123", "topic_name": "finance"},
        )

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["metadata"]["channel"] == "telegram"
    assert call_kwargs["metadata"]["chat_id"] == "123"
    assert call_kwargs["metadata"]["topic_name"] == "finance"


# --- Timeout Tests ---


@pytest.mark.asyncio
async def test_timeout_passed_to_litellm():
    """Configured timeout must be passed to litellm.acompletion."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(timeout=30), "gpt-4o")
        await provider.chat(messages=[{"role": "user", "content": "hi"}])

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["timeout"] == 30


@pytest.mark.asyncio
async def test_no_timeout_by_default():
    """Timeout must not be passed when not configured."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        await provider.chat(messages=[{"role": "user", "content": "hi"}])

    call_kwargs = mock_acompletion.call_args.kwargs
    assert "timeout" not in call_kwargs


# --- num_retries Tests ---


@pytest.mark.asyncio
async def test_init_sets_num_retries():
    """LiteLLMProvider.__init__ must set litellm.num_retries from config."""
    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        LiteLLMProvider(_proxy_config(num_retries=3), "gpt-4o")

    assert mock_litellm.num_retries == 3


# --- Prompt Caching Markers Tests ---


@pytest.mark.asyncio
async def test_prompt_caching_disabled_by_default():
    """When enable_prompt_caching is False, system param is a plain string."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(), "gpt-4o")
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "hi"},
            ],
        )

    call_kwargs = mock_acompletion.call_args.kwargs
    assert isinstance(call_kwargs["system"], str)


@pytest.mark.asyncio
async def test_prompt_caching_enabled_adds_cache_control():
    """When enable_prompt_caching is True, system param must be a list with cache_control."""
    mock_acompletion = AsyncMock(return_value=_fake_response())

    with patch("nanobot.providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.acompletion = mock_acompletion
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(_proxy_config(enable_prompt_caching=True), "gpt-4o")
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "hi"},
            ],
        )

    call_kwargs = mock_acompletion.call_args.kwargs
    assert isinstance(call_kwargs["system"], list)
    assert len(call_kwargs["system"]) == 1
    block = call_kwargs["system"][0]
    assert block["type"] == "text"
    assert block["text"] == "You are a bot."
    assert block["cache_control"] == {"type": "ephemeral"}


# --- Router Circuit Breaker Tests ---


@pytest.mark.asyncio
async def test_router_receives_circuit_breaker_params():
    """Router must be initialized with allowed_fails and cooldown_time from config."""
    mock_router = MagicMock()

    with (
        patch("nanobot.providers.litellm_provider.litellm"),
        patch("nanobot.providers.litellm_provider.Router", return_value=mock_router) as mock_router_cls,
    ):
        from nanobot.providers.litellm_provider import LiteLLMProvider

        LiteLLMProvider(
            LiteLLMConfig(
                models=[
                    LiteLLMModelConfig(
                        model_name="gpt-4o",
                        litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test"},
                    )
                ],
                allowed_fails=5,
                cooldown_time=120,
            ),
            "gpt-4o",
        )

    mock_router_cls.assert_called_once()
    call_kwargs = mock_router_cls.call_args.kwargs
    assert call_kwargs["allowed_fails"] == 5
    assert call_kwargs["cooldown_time"] == 120
