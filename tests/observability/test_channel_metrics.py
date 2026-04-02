"""Tests for channel metrics (messages counter, send errors, send duration)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.channels.base import BaseChannel


@pytest.fixture
def mock_meter():
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram
    with (
        patch("nanobot.observability.otel.get_meter", return_value=mock_meter),
        patch("nanobot.channels.base.get_meter", return_value=mock_meter),
        patch("nanobot.channels.telegram.get_meter", return_value=mock_meter),
    ):
        yield mock_meter, mock_counter, mock_histogram


# ---------------------------------------------------------------------------
# BaseChannel: inbound message counter
# ---------------------------------------------------------------------------


class _TestChannel(BaseChannel):
    """Minimal concrete subclass of BaseChannel for testing."""

    name = "test"

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, msg):
        pass


def _make_allowed_config():
    cfg = MagicMock()
    cfg.allow_from = ["*"]
    return cfg


@pytest.mark.asyncio
async def test_inbound_message_increments_counter(mock_meter):
    """_handle_message increments the message counter with direction=inbound."""
    _, mock_counter, _ = mock_meter

    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    config = _make_allowed_config()
    ch = _TestChannel(config, bus)

    await ch._handle_message(sender_id="u1", chat_id="c1", content="hello")

    mock_counter.add.assert_called_once()
    call_args, call_kwargs = mock_counter.add.call_args
    assert call_args[0] == 1
    assert call_kwargs["attributes"]["channel"] == "test"
    assert call_kwargs["attributes"]["direction"] == "inbound"


@pytest.mark.asyncio
async def test_inbound_denied_does_not_increment_counter(mock_meter):
    """_handle_message for denied sender does NOT increment counter."""
    _, mock_counter, _ = mock_meter

    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    config = MagicMock()
    config.allow_from = []  # deny all
    ch = _TestChannel(config, bus)

    await ch._handle_message(sender_id="u1", chat_id="c1", content="hello")

    mock_counter.add.assert_not_called()


# ---------------------------------------------------------------------------
# TelegramChannel: outbound counter, send duration, send errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_send_records_outbound_counter(mock_meter):
    """TelegramChannel.send() increments counter with direction=outbound."""
    _, mock_counter, mock_histogram = mock_meter

    from nanobot.bus.events import OutboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.telegram import TelegramChannel

    bus = MessageBus()
    config = MagicMock()
    config.allow_from = ["*"]
    config.token = "fake"
    config.reply_to_message = False

    ch = TelegramChannel(config, bus)
    ch._app = MagicMock()
    ch._app.bot.send_message = AsyncMock()
    ch._stop_typing = MagicMock()

    msg = OutboundMessage(channel="telegram", chat_id="123", content="hi")
    await ch.send(msg)

    # Find the outbound counter call
    outbound_calls = [
        c
        for c in mock_counter.add.call_args_list
        if c.kwargs.get("attributes", {}).get("direction") == "outbound"
    ]
    assert len(outbound_calls) == 1
    assert outbound_calls[0].kwargs["attributes"]["channel"] == "telegram"


@pytest.mark.asyncio
async def test_telegram_send_records_duration(mock_meter):
    """TelegramChannel.send() records send duration histogram."""
    _, _, mock_histogram = mock_meter

    from nanobot.bus.events import OutboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.telegram import TelegramChannel

    bus = MessageBus()
    config = MagicMock()
    config.allow_from = ["*"]
    config.token = "fake"
    config.reply_to_message = False

    ch = TelegramChannel(config, bus)
    ch._app = MagicMock()
    ch._app.bot.send_message = AsyncMock()
    ch._stop_typing = MagicMock()

    msg = OutboundMessage(channel="telegram", chat_id="123", content="hi")
    await ch.send(msg)

    mock_histogram.record.assert_called()
    call_args, call_kwargs = mock_histogram.record.call_args
    assert call_args[0] >= 0  # duration in ms
    assert call_kwargs["attributes"]["channel"] == "telegram"


@pytest.mark.asyncio
async def test_telegram_send_error_increments_error_counter(mock_meter):
    """TelegramChannel._send_text() increments error counter on exception."""
    _, mock_counter, _ = mock_meter

    from nanobot.bus.queue import MessageBus
    from nanobot.channels.telegram import TelegramChannel

    bus = MessageBus()
    config = MagicMock()
    config.allow_from = ["*"]
    config.token = "fake"
    config.reply_to_message = False

    ch = TelegramChannel(config, bus)
    ch._app = MagicMock()
    ch._app.bot.send_message = AsyncMock(side_effect=RuntimeError("send failed"))

    with pytest.raises(RuntimeError, match="send failed"):
        await ch._send_text(123, "hello", None, None)

    # The send_errors counter should be incremented
    error_calls = [
        c for c in mock_counter.add.call_args_list if "error_type" in (c.kwargs.get("attributes") or {})
    ]
    assert len(error_calls) == 1
    assert error_calls[0].kwargs["attributes"]["channel"] == "telegram"
    assert error_calls[0].kwargs["attributes"]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_telegram_send_records_outbound_per_chunk(mock_meter):
    """Each text chunk in send() records one outbound counter increment."""
    _, mock_counter, _ = mock_meter

    from nanobot.bus.events import OutboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.telegram import TelegramChannel

    bus = MessageBus()
    config = MagicMock()
    config.allow_from = ["*"]
    config.token = "fake"
    config.reply_to_message = False

    ch = TelegramChannel(config, bus)
    ch._app = MagicMock()
    ch._app.bot.send_message = AsyncMock()
    ch._stop_typing = MagicMock()

    # Content that will be split into 2 chunks (>4000 chars)
    long_content = "x" * 4500
    msg = OutboundMessage(channel="telegram", chat_id="123", content=long_content)
    await ch.send(msg)

    outbound_calls = [
        c
        for c in mock_counter.add.call_args_list
        if c.kwargs.get("attributes", {}).get("direction") == "outbound"
    ]
    assert len(outbound_calls) == 2
