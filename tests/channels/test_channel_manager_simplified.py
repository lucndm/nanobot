"""Tests for simplified ChannelManager — direct Telegram init."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def test_init_disabled_channel():
    """When channel.enabled is False, no channels are initialized."""
    config = Config(channel={"enabled": False})
    manager = ChannelManager(config, MessageBus())
    assert manager.channels == {}


def test_init_enabled_telegram():
    """When channel.enabled is True, Telegram is initialized."""
    config = Config(
        channel={"enabled": True, "token": "test-token"},
    )
    with patch("nanobot.channels.manager.TelegramChannel") as MockTG:
        instance = MagicMock()
        MockTG.return_value = instance
        manager = ChannelManager(config, MessageBus())
        MockTG.assert_called_once()
        assert "telegram" in manager.channels


def test_init_passes_workspace():
    """ChannelManager passes workspace_path to TelegramChannel."""
    tmp = Path("/tmp/test-workspace")
    config = Config(
        agents={"defaults": {"workspace": str(tmp)}},
        channel={"enabled": True, "token": "test-token"},
    )
    with patch("nanobot.channels.manager.TelegramChannel") as MockTG:
        instance = MagicMock()
        MockTG.return_value = instance
        manager = ChannelManager(config, MessageBus())
        call_kwargs = MockTG.call_args
        assert call_kwargs[1].get("workspace") == tmp or call_kwargs[0][2] == tmp
