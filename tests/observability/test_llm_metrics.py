"""Tests for LLM metrics — now handled by OTelCallback, not AgentRunner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.providers.base import LLMResponse


def _make_spec(**kwargs):
    defaults = dict(
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=MagicMock(),
        model="gpt-4o-mini",
        max_iterations=1,
        channel="telegram",
    )
    defaults.update(kwargs)
    return AgentRunSpec(**defaults)


def _make_response(content="hi", prompt_tokens=10, completion_tokens=5):
    return LLMResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


@pytest.mark.asyncio
async def test_runner_does_not_record_llm_metrics():
    """AgentRunner no longer records LLM metrics — that's handled by OTelCallback."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=_make_response())
    runner = AgentRunner(provider)
    spec = _make_spec()

    # AgentRunner should NOT have metrics methods (removed in migration)
    assert not hasattr(runner, "_init_llm_metrics")
    assert not hasattr(runner, "_record_llm_metrics")
    assert not hasattr(runner, "_record_llm_error")

    result = await runner.run(spec)

    assert result.final_content == "hi"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.asyncio
async def test_runner_still_propagates_llm_exception():
    """AgentRunner still re-raises LLM exceptions (no longer catches for metrics)."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API overload"))
    runner = AgentRunner(provider)
    spec = _make_spec()

    with pytest.raises(RuntimeError, match="API overload"):
        await runner.run(spec)
