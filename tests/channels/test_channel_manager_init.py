"""Tests for ChannelManager._init_channels — verifies workspace parameter is passed to channels."""

from __future__ import annotations

from unittest.mock import patch

from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class _RecordingChannel(BaseChannel):
    """Fake channel that records all __init__ keyword arguments."""

    name = "testchan"
    display_name = "Test Channel"
    init_kwargs: dict = {}

    def __init__(self, config, bus, **kwargs):
        super().__init__(config, bus)
        _RecordingChannel.init_kwargs = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, msg) -> None:
        pass


def test_channel_manager_passes_workspace_to_channel(tmp_path):
    """ChannelManager._init_channels must pass workspace=workspace_path to each channel.

    Regression test: TelegramChannel previously did not accept the workspace kwarg,
    causing 'got an unexpected keyword argument workspace' at runtime when
    ChannelManager initialized the Telegram channel.
    """
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.schema import Config

    _RecordingChannel.init_kwargs.clear()

    # Minimal config with a known workspace_path
    config = Config(
        agents={"defaults": {"workspace": str(tmp_path)}},
        channels={"testchan": {"enabled": True}},
    )

    # Patch discover_all to return our recording channel
    with patch(
        "nanobot.channels.registry.discover_all",
        return_value={"testchan": _RecordingChannel},
    ):
        ChannelManager(config, MessageBus())

    # The channel must have been instantiated with workspace set
    assert "workspace" in _RecordingChannel.init_kwargs
    assert _RecordingChannel.init_kwargs["workspace"] == tmp_path


def test_channel_manager_handles_missing_workspace_kwarg():
    """Channels that don't accept workspace kwarg should not crash the manager.

    When a channel's __init__ signature does NOT include 'workspace',
    the TypeError should be caught and logged as a warning, not crash the process.
    """
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.schema import Config

    class _NoWorkspaceChannel(BaseChannel):
        name = "noworkspace"
        display_name = "No Workspace"

        def __init__(self, config, bus):
            # Does NOT accept workspace kwarg — old bug
            super().__init__(config, bus)

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def send(self, msg) -> None:
            pass

    config = Config(
        channels={"noworkspace": {"enabled": True}},
    )

    with patch(
        "nanobot.channels.registry.discover_all",
        return_value={"noworkspace": _NoWorkspaceChannel},
    ):
        # Should not raise — warning is logged instead
        manager = ChannelManager(config, MessageBus())

    # Channel is skipped but manager remains functional
    assert "noworkspace" not in manager.channels
