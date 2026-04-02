"""Tests for LLM metrics in AgentRunner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.providers.base import LLMResponse


@pytest.fixture
def mock_meter():
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram
    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_counter, mock_histogram


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
async def test_llm_metrics_recorded_on_success(mock_meter):
    """On successful LLM call, duration histogram and token counters are recorded."""
    _, mock_counter, mock_histogram = mock_meter

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=_make_response())
    runner = AgentRunner(provider)
    spec = _make_spec()

    result = await runner.run(spec)

    assert result.final_content == "hi"

    # Duration histogram should be recorded
    mock_histogram.record.assert_called_once()
    record_args, record_kwargs = mock_histogram.record.call_args
    assert record_args[0] >= 0
    assert record_kwargs["attributes"]["model"] == "gpt-4o-mini"
    assert record_kwargs["attributes"]["channel"] == "telegram"

    # Token counters: prompt and completion
    assert mock_counter.add.call_count == 2

    add_calls = mock_counter.add.call_args_list
    # Find prompt_tokens counter call
    prompt_call = [c for c in add_calls if c.kwargs["attributes"].get("type") == "prompt"][0]
    assert prompt_call.args[0] == 10
    assert prompt_call.kwargs["attributes"]["model"] == "gpt-4o-mini"
    assert prompt_call.kwargs["attributes"]["channel"] == "telegram"

    # Find completion_tokens counter call
    completion_call = [c for c in add_calls if c.kwargs["attributes"].get("type") == "completion"][0]
    assert completion_call.args[0] == 5


@pytest.mark.asyncio
async def test_llm_error_counter_on_exception(mock_meter):
    """On LLM call exception, error counter is incremented and exception re-raised."""
    _, mock_counter, mock_histogram = mock_meter

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API overload"))
    runner = AgentRunner(provider)
    spec = _make_spec()

    with pytest.raises(RuntimeError, match="API overload"):
        await runner.run(spec)

    # Error counter should be incremented
    error_calls = [c for c in mock_counter.add.call_args_list if "error_type" in (c.kwargs.get("attributes") or {})]
    assert len(error_calls) == 1
    assert error_calls[0].kwargs["attributes"]["model"] == "gpt-4o-mini"
    assert error_calls[0].kwargs["attributes"]["channel"] == "telegram"
    assert error_calls[0].kwargs["attributes"]["error_type"] == "RuntimeError"
