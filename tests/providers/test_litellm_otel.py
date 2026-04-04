"""Tests for OTelCallback -- bridges litellm events to nanobot OTel metrics."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from nanobot.providers.litellm_otel import OTelCallback


def _make_kwargs(model="gpt-4o", channel="telegram"):
    """Simulate litellm callback kwargs."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": channel},
    }


def _make_response(prompt_tokens=10, completion_tokens=5, total_tokens=15):
    """Simulate litellm ModelResponse."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    resp = MagicMock()
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_callback_records_success_metrics():
    mock_duration = MagicMock()
    mock_prompt = MagicMock()
    mock_completion = MagicMock()
    mock_errors = MagicMock()

    with patch("nanobot.observability.otel.get_meter") as mock_get_meter:
        meter = MagicMock()
        meter.create_histogram.return_value = mock_duration
        meter.create_counter.side_effect = [mock_prompt, mock_completion, mock_errors]
        mock_get_meter.return_value = meter

        cb = OTelCallback()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        await cb.async_log_success_event(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=start,
            end_time=end,
        )

    mock_duration.record.assert_called_once()
    call_attrs = mock_duration.record.call_args
    assert call_attrs[0][0] == 1000.0

    mock_prompt.add.assert_called_once_with(10, attributes={"model": "gpt-4o", "type": "prompt"})
    mock_completion.add.assert_called_once_with(5, attributes={"model": "gpt-4o", "type": "completion"})


@pytest.mark.asyncio
async def test_callback_records_failure_metrics():
    mock_errors = MagicMock()

    with patch("nanobot.observability.otel.get_meter") as mock_get_meter:
        meter = MagicMock()
        meter.create_histogram.return_value = MagicMock()
        meter.create_counter.side_effect = [MagicMock(), MagicMock(), mock_errors]
        mock_get_meter.return_value = meter

        cb = OTelCallback()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        await cb.async_log_failure_event(
            kwargs=_make_kwargs(),
            response_obj=None,
            start_time=start,
            end_time=end,
        )

    mock_errors.add.assert_called_once()
    call_attrs = mock_errors.add.call_args
    assert call_attrs[0][0] == 1
    assert "error_type" in call_attrs[1]["attributes"]


@pytest.mark.asyncio
async def test_callback_handles_no_meter():
    """OTelCallback gracefully handles missing meter (OTel disabled)."""
    with patch("nanobot.observability.otel.get_meter", return_value=None):
        cb = OTelCallback()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        await cb.async_log_success_event(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=start,
            end_time=end,
        )
