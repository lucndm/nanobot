"""Tests for bus metrics (queue depth gauge + latency histogram)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage


@pytest.fixture
def mock_meter():
    mock_histogram = MagicMock()
    mock_gauge = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_histogram.return_value = mock_histogram
    mock_meter.create_observable_gauge.return_value = mock_gauge
    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_histogram, mock_gauge


def _make_inbound(**kwargs):
    defaults = dict(
        channel="telegram",
        sender_id="user1",
        chat_id="chat1",
        content="hello",
    )
    defaults.update(kwargs)
    return InboundMessage(**defaults)


def _make_outbound(**kwargs):
    defaults = dict(
        channel="telegram",
        chat_id="chat1",
        content="hi there",
    )
    defaults.update(kwargs)
    return OutboundMessage(**defaults)


@pytest.mark.asyncio
async def test_bus_sets_queued_at_on_publish(mock_meter):
    """publish_inbound and publish_outbound set msg.queued_at > 0."""
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()

    in_msg = _make_inbound()
    assert in_msg.queued_at == 0.0

    await bus.publish_inbound(in_msg)
    assert in_msg.queued_at > 0.0

    out_msg = _make_outbound()
    assert out_msg.queued_at == 0.0

    await bus.publish_outbound(out_msg)
    assert out_msg.queued_at > 0.0


@pytest.mark.asyncio
async def test_bus_records_latency_on_consume(mock_meter):
    """consume_inbound records latency histogram with channel attribute."""
    from nanobot.bus.queue import MessageBus

    _, mock_histogram, _ = mock_meter
    bus = MessageBus()

    in_msg = _make_inbound(channel="discord")
    await bus.publish_inbound(in_msg)

    # Small sleep to ensure measurable latency
    time.sleep(0.01)

    result = await bus.consume_inbound()
    assert result is in_msg

    mock_histogram.record.assert_called()
    record_args, record_kwargs = mock_histogram.record.call_args
    assert record_args[0] >= 0  # latency in ms
    assert record_kwargs["attributes"]["channel"] == "discord"


@pytest.mark.asyncio
async def test_bus_creates_observable_gauge(mock_meter):
    """MessageBus constructor creates an observable gauge for queue depth."""
    mock_meter, _, mock_gauge = mock_meter

    from nanobot.bus.queue import MessageBus

    MessageBus()

    mock_meter.create_observable_gauge.assert_called_once()
    call_args = mock_meter.create_observable_gauge.call_args
    assert call_args[0][0] == "nanobot.bus.queue.depth"
    assert "callbacks" in call_args[1]
