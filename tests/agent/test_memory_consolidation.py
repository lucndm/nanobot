"""Tests for MemoryStore consolidation logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nanobot.agent.store import MemoryStore


@pytest.fixture
def mock_store():
    """MemoryStore with mocked connection pool."""
    with patch("nanobot.agent.store.ConnectionPool"):
        store = MemoryStore("postgresql://test:test@localhost/test")
        return store


def test_format_messages_for_consolidation_empty(mock_store):
    result = mock_store.format_messages_for_consolidation([], None)
    assert result == ""


def test_format_messages_for_consolidation_basic(mock_store):
    messages = [{"role": "user", "content": "hello", "timestamp": "2026-01-01T10:00"}]
    result = mock_store.format_messages_for_consolidation(messages, None)
    assert "[2026-01-01T10:00] USER: hello" in result


def test_format_messages_high_value(mock_store):
    mock_store.get_high_value_messages = MagicMock(return_value=[42])
    messages = [
        {"role": "user", "content": "important", "timestamp": "2026-01-01T10:00", "telegram_message_id": 42},
        {"role": "user", "content": "normal", "timestamp": "2026-01-01T10:01", "telegram_message_id": 43},
    ]
    result = mock_store.format_messages_for_consolidation(messages, "test_topic")
    assert "[HIGH VALUE] important" in result
    assert "normal" in result
    assert "[HIGH VALUE] normal" not in result


@pytest.mark.asyncio
async def test_consolidate_no_messages(mock_store):
    result = await mock_store.consolidate([], None, "model")
    assert result is True


def test_failure_key_global(mock_store):
    assert mock_store._failure_key(None) == "global"


def test_failure_key_topic(mock_store):
    assert mock_store._failure_key("my_topic") == "my_topic"


def test_fail_or_raw_archive_under_threshold(mock_store):
    result = mock_store._fail_or_raw_archive(None, [{"role": "user", "content": "x"}])
    assert result is False
    assert mock_store._failures["global"] == 1


def test_fail_or_raw_archive_at_threshold(mock_store):
    mock_store._failures["global"] = 2  # one below max
    result = mock_store._fail_or_raw_archive(None, [{"role": "user", "content": "x"}])
    assert result is True
    assert mock_store._failures["global"] == 0  # reset after archive


def test_format_raw_messages_empty(mock_store):
    result = MemoryStore._format_raw_messages([])
    assert result == ""


def test_format_raw_messages(mock_store):
    messages = [
        {"role": "user", "content": "hello", "timestamp": "2026-01-01T10:00"},
        {"role": "assistant", "content": "hi", "timestamp": "2026-01-01T10:01"},
    ]
    result = MemoryStore._format_raw_messages(messages)
    assert "[2026-01-01T10:00] USER: hello" in result
    assert "[2026-01-01T10:01] ASSISTANT: hi" in result
