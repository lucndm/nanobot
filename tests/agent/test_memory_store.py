"""Tests for MemoryStore.

Requires a running Postgres instance. Set NANOBOT_TEST_PG_URL env var to enable.
Example: NANOBOT_TEST_PG_URL=postgresql://test:test@localhost:5432/test_nanobot
"""

from __future__ import annotations

import os

import pytest

pg_url = os.environ.get("NANOBOT_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not pg_url, reason="NANOBOT_TEST_PG_URL not set")


@pytest.fixture
def store():
    from nanobot.agent.store import MemoryStore

    s = MemoryStore(pg_url, pool_size=2)
    yield s
    # Cleanup: truncate all tables
    with s._pool.connection() as conn:
        for table in [
            "topic_mapping", "emoji_sentiment", "message_sentiment",
            "message_reactions", "topic_history", "topic_memory",
            "global_history", "global_memory",
        ]:
            conn.execute(f"DELETE FROM {table}")
    s.close()


class TestGlobalMemory:
    def test_read_long_term_empty(self, store):
        assert store.read_long_term() == ""

    def test_write_and_read(self, store):
        store.write_long_term("# Memory\nUser likes coffee.")
        assert store.read_long_term() == "# Memory\nUser likes coffee."

    def test_overwrite(self, store):
        store.write_long_term("old")
        store.write_long_term("new")
        assert store.read_long_term() == "new"

    def test_append_history(self, store):
        store.append_history("[2026-01-01] First.")
        store.append_history("[2026-01-02] Second.")
        h = store.read_history()
        assert "First." in h and "Second." in h

    def test_get_memory_context(self, store):
        store.write_long_term("Hello")
        assert "## Global Memory" in store.get_memory_context()

    def test_get_memory_context_empty(self, store):
        assert store.get_memory_context() == ""


class TestTopicMemory:
    def test_read_missing(self, store):
        assert store.read_topic_memory("nope") is None

    def test_write_and_read(self, store):
        store.write_topic_memory("finance", "Finance facts.")
        assert store.read_topic_memory("finance") == "Finance facts."

    def test_overwrite(self, store):
        store.write_topic_memory("t", "v1")
        store.write_topic_memory("t", "v2")
        assert store.read_topic_memory("t") == "v2"

    def test_list_topics(self, store):
        store.write_topic_memory("alpha", "a")
        store.write_topic_memory("beta", "b")
        assert store.list_topics() == ["alpha", "beta"]

    def test_get_topic_memory_context(self, store):
        store.write_topic_memory("finance", "Facts")
        ctx = store.get_topic_memory_context("finance")
        assert "## Topic Memory (finance)" in ctx

    def test_get_topic_memory_context_missing(self, store):
        assert store.get_topic_memory_context("nope") is None


class TestTopicMapping:
    def test_get_missing(self, store):
        assert store.get_topic_mapping(-100, 4) is None

    def test_set_and_get(self, store):
        store.set_topic_mapping(-100, 4, "Finance")
        assert store.get_topic_mapping(-100, 4) == "Finance"

    def test_overwrite(self, store):
        store.set_topic_mapping(-100, 4, "Finance")
        store.set_topic_mapping(-100, 4, "Finance Tracker")
        assert store.get_topic_mapping(-100, 4) == "Finance Tracker"

    def test_delete(self, store):
        store.set_topic_mapping(-100, 4, "Finance")
        store.delete_topic_mapping(-100, 4)
        assert store.get_topic_mapping(-100, 4) is None

    def test_load_all(self, store):
        store.set_topic_mapping(-100, 4, "Finance")
        store.set_topic_mapping(-100, 6, "General")
        m = store.load_all_topic_mappings()
        assert m[(-100, 4)] == "Finance"
        assert m[(-100, 6)] == "General"


class TestReactions:
    def test_record_and_get(self, store):
        store.record_reaction("-100", 123, "👍", "positive", "telegram:-100:topic:4")
        counts = store.get_message_sentiment("-100", 123)
        assert counts["positive_count"] == 1

    def test_remove_reaction(self, store):
        store.record_reaction("-100", 123, "👍", "positive", "t")
        store.remove_reaction("-100", 123, "👍")
        counts = store.get_message_sentiment("-100", 123)
        assert counts["positive_count"] == 0

    def test_emoji_learning(self, store):
        store.learn_emoji("🚀", "positive")
        assert store.resolve_emoji_sentiment("🚀") == "positive"
        assert store.is_emoji_known("🚀")

    def test_high_value_messages(self, store):
        store.record_reaction("-100", 10, "👍", "positive", "t")
        store.record_reaction("-100", 20, "👍", "positive", "t")
        store.record_reaction("-100", 30, "👎", "negative", "t")
        hv = store.get_high_value_messages("t")
        assert 10 in hv and 20 in hv and 30 not in hv


class TestTopicLitellm:
    def test_get_missing(self, store):
        assert store.get_topic_litellm("nope") is None

    def test_set_and_get(self, store):
        store.set_topic_litellm("finance", "gpt-4", 0.7, 1000)
        cfg = store.get_topic_litellm("finance")
        assert cfg == ("gpt-4", 0.7, 1000)

    def test_delete(self, store):
        store.set_topic_litellm("finance", "gpt-4", 0.7, 1000)
        store.delete_topic_litellm("finance")
        assert store.get_topic_litellm("finance") is None
