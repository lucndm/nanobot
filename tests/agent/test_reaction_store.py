"""Tests for reaction storage in SqliteMemoryStore."""

import pytest
from pathlib import Path

from nanobot.agent.memory import SqliteMemoryStore


@pytest.fixture
def store(tmp_path):
    return SqliteMemoryStore(tmp_path)


class TestReactionRecording:
    def test_record_positive_reaction(self, store):
        store.record_reaction(
            chat_id="-1003738155502",
            message_id=123,
            emoji="👍",
            sentiment="positive",
            topic="telegram:-1003738155502:topic:935",
        )
        counts = store.get_message_sentiment("-1003738155502", 123)
        assert counts == {"positive_count": 1, "negative_count": 0, "neutral_count": 0}

    def test_record_negative_reaction(self, store):
        store.record_reaction(
            chat_id="-1003738155502",
            message_id=456,
            emoji="👎",
            sentiment="negative",
            topic="telegram:-1003738155502:topic:935",
        )
        counts = store.get_message_sentiment("-1003738155502", 456)
        assert counts["negative_count"] == 1

    def test_record_neutral_reaction(self, store):
        store.record_reaction(
            chat_id="-1003738155502",
            message_id=789,
            emoji="😀",
            sentiment="neutral",
            topic="telegram:-1003738155502:topic:935",
        )
        counts = store.get_message_sentiment("-1003738155502", 789)
        assert counts["neutral_count"] == 1

    def test_duplicate_emoji_same_message_idempotent(self, store):
        store.record_reaction("-1003738155502", 123, "👍", "positive", "telegram:-1003738155502:topic:935")
        store.record_reaction("-1003738155502", 123, "👍", "positive", "telegram:-1003738155502:topic:935")
        counts = store.get_message_sentiment("-1003738155502", 123)
        assert counts["positive_count"] == 1  # UNIQUE constraint prevents double

    def test_multiple_emojis_same_message(self, store):
        store.record_reaction("-1003738155502", 123, "👍", "positive", "telegram:-1003738155502:topic:935")
        store.record_reaction("-1003738155502", 123, "❤️", "positive", "telegram:-1003738155502:topic:935")
        counts = store.get_message_sentiment("-1003738155502", 123)
        assert counts["positive_count"] == 2

    def test_remove_reaction(self, store):
        store.record_reaction("-1003738155502", 123, "👍", "positive", "telegram:-1003738155502:topic:935")
        store.remove_reaction("-1003738155502", 123, "👍")
        counts = store.get_message_sentiment("-1003738155502", 123)
        assert counts["positive_count"] == 0

    def test_remove_nonexistent_reaction_noop(self, store):
        store.remove_reaction("-1003738155502", 999, "👍")  # should not raise


class TestTopicScoping:
    def test_get_high_value_messages_per_topic(self, store):
        topic_a = "telegram:-1003738155502:topic:935"
        topic_b = "telegram:-1003738155502:topic:123"
        store.record_reaction("-1003738155502", 10, "👍", "positive", topic_a)
        store.record_reaction("-1003738155502", 20, "👍", "positive", topic_b)
        store.record_reaction("-1003738155502", 30, "👍", "positive", topic_a)

        high_a = store.get_high_value_messages(topic_a)
        assert len(high_a) == 2
        assert 10 in high_a
        assert 30 in high_a

    def test_different_topic_no_cross_contamination(self, store):
        topic_a = "telegram:-1003738155502:topic:935"
        topic_b = "telegram:-1003738155502:topic:123"
        store.record_reaction("-1003738155502", 10, "👍", "positive", topic_a)
        high_b = store.get_high_value_messages(topic_b)
        assert len(high_b) == 0


class TestEmojiLearning:
    def test_learn_unknown_emoji(self, store):
        store.learn_emoji("🤖", "positive")
        assert store.resolve_emoji_sentiment("🤖") == "positive"

    def test_known_emoji_not_in_table(self, store):
        # Hardcoded emojis should work without DB lookup
        assert store.resolve_emoji_sentiment("👍") == "positive"
        assert store.resolve_emoji_sentiment("👎") == "negative"
        assert store.resolve_emoji_sentiment("😀") is None  # unknown

    def test_learned_emoji_persists(self, store, tmp_path):
        store.learn_emoji("🤖", "positive")
        # Re-read from DB using the same workspace path
        store2 = SqliteMemoryStore(tmp_path)
        assert store2.resolve_emoji_sentiment("🤖") == "positive"

    def test_is_emoji_known(self, store):
        assert store.is_emoji_known("👍") is True
        assert store.is_emoji_known("🤖") is False
        store.learn_emoji("🤖", "positive")
        assert store.is_emoji_known("🤖") is True


class TestCleanup:
    def test_cleanup_old_reactions(self, store):
        import time
        store.record_reaction("-1003738155502", 10, "👍", "positive", "telegram:-1003738155502:topic:935")
        # Manually age the record
        with store._conn() as conn:
            conn.execute(
                "UPDATE message_reactions SET created_at = datetime('now', '-31 days') WHERE message_id = 10"
            )
        removed = store.cleanup_old_reactions(max_age_days=30)
        assert removed == 1
