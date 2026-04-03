"""Tests for Telegram reaction handler."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.channels.telegram import TelegramChannel


@pytest.fixture
def channel(tmp_path):
    ch = TelegramChannel.__new__(TelegramChannel)
    ch._bot_user_id = 12345
    ch._bot_username = "testbot"
    ch._app = MagicMock()
    ch._reaction_store = None  # will be set per test
    return ch


def test_classify_known_emoji(channel):
    assert channel._classify_emoji("\U0001f44d") == "positive"
    assert channel._classify_emoji("\u2764\ufe0f") == "positive"
    assert channel._classify_emoji("\U0001f44e") == "negative"
    assert channel._classify_emoji("\U0001f621") == "negative"


def test_classify_unknown_emoji(channel):
    assert channel._classify_emoji("\U0001f916") is None


def test_resolve_topic_from_thread_id(channel):
    assert channel._resolve_reaction_topic("-1003738155502", 935) == "telegram:-1003738155502:topic:935"


def test_resolve_topic_no_thread(channel):
    assert channel._resolve_reaction_topic("164211708", None) == "telegram:164211708"
