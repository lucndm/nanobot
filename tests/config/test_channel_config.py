"""Tests for ChannelConfig (replacing ChannelsConfig)."""
import pytest
from pydantic import ValidationError
from nanobot.config.schema import ChannelConfig, Config


def test_channel_config_defaults():
    """ChannelConfig has sensible defaults."""
    cfg = ChannelConfig()
    assert cfg.enabled is False
    assert cfg.token == ""
    assert cfg.allow_from == ["*"]
    assert cfg.proxy is None
    assert cfg.reply_to_message is False
    assert cfg.react_emoji == ""
    assert cfg.group_policy == "mention"
    assert cfg.connection_pool_size == 32
    assert cfg.pool_timeout == 5.0
    assert cfg.streaming is True
    assert cfg.send_progress is True
    assert cfg.send_tool_hints is False
    assert cfg.send_max_retries == 3


def test_channel_config_from_camel_case():
    """ChannelConfig accepts camelCase aliases."""
    cfg = ChannelConfig(allowFrom=["123"], groupPolicy="all")
    assert cfg.allow_from == ["123"]
    assert cfg.group_policy == "all"


def test_config_root_has_channel_field():
    """Config root uses 'channel' (singular) field."""
    cfg = Config()
    assert hasattr(cfg, "channel")
    assert isinstance(cfg.channel, ChannelConfig)


def test_config_root_no_channels_field():
    """Config root no longer has 'channels' (plural) field."""
    cfg = Config()
    assert not hasattr(cfg, "channels")


def test_channel_config_send_max_retries_bounds():
    """send_max_retries must be 0-10."""
    ChannelConfig(send_max_retries=0)
    ChannelConfig(send_max_retries=10)
    with pytest.raises(ValidationError):
        ChannelConfig(send_max_retries=-1)
    with pytest.raises(ValidationError):
        ChannelConfig(send_max_retries=11)
