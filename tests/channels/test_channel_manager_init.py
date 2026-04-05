"""Tests for ChannelManager._init_channels — verifies direct Telegram init."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def test_channel_manager_init_disabled():
    """When channel.enabled=False, no channels are created."""
    config = Config(channel={"enabled": False})
    manager = ChannelManager(config, MessageBus())
    assert manager.channels == {}


def test_channel_manager_init_enabled():
    """When channel.enabled=True, Telegram channel is created."""
    config = Config(channel={"enabled": True, "token": "test-token"})

    with patch("nanobot.channels.manager.TelegramChannel") as MockTG:
        instance = MagicMock()
        MockTG.return_value = instance
        manager = ChannelManager(config, MessageBus())

        MockTG.assert_called_once()
        assert "telegram" in manager.channels
