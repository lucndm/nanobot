"""Tests for SqliteMemoryStore."""
import pytest
from pathlib import Path
from nanobot.agent.memory import SqliteMemoryStore


class TestSqliteMemoryStoreGlobalCRUD:
    def test_creates_db_on_init(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        db_path = tmp_path / "data" / "memories.db"
        assert db_path.exists()

    def test_read_long_term_returns_empty_when_no_data(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        assert store.read_long_term() == ""

    def test_write_and_read_long_term(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_long_term("# Memory\nUser likes coffee.")
        assert store.read_long_term() == "# Memory\nUser likes coffee."

    def test_write_long_term_overwrites(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_long_term("old")
        store.write_long_term("new")
        assert store.read_long_term() == "new"

    def test_append_history(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.append_history("[2026-01-01] First event.")
        store.append_history("[2026-01-02] Second event.")
        history = store.read_history()
        assert "First event." in history
        assert "Second event." in history

    def test_get_memory_context_returns_formatted_string(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_long_term("User likes Python.")
        ctx = store.get_memory_context()
        assert "User likes Python." in ctx
        assert "## Global Memory" in ctx

    def test_get_memory_context_returns_empty_when_no_data(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        assert store.get_memory_context() == ""


class TestSqliteMemoryStoreTopicCRUD:
    def test_read_topic_memory_returns_none_when_missing(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        assert store.read_topic_memory("558") is None

    def test_write_and_read_topic_memory(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("558", "Finance facts here.")
        assert store.read_topic_memory("558") == "Finance facts here."

    def test_write_topic_memory_overwrites(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("558", "old")
        store.write_topic_memory("558", "new")
        assert store.read_topic_memory("558") == "new"

    def test_append_topic_history(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.append_topic_history("558", "[2026-01-01] Topic event.")
        store.append_topic_history("558", "[2026-01-02] Another event.")
        history = store.read_topic_history("558")
        assert "Topic event." in history
        assert "Another event." in history

    def test_get_topic_memory_context_returns_formatted(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("558", "User banks at VPBank.")
        ctx = store.get_topic_memory_context("558")
        assert "User banks at VPBank." in ctx
        assert "## Topic Memory (558)" in ctx

    def test_get_topic_memory_context_returns_none_when_missing(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        assert store.get_topic_memory_context("999") is None

    def test_multiple_topics_isolated(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("558", "Finance data")
        store.write_topic_memory("677", "Dev data")
        assert store.read_topic_memory("558") == "Finance data"
        assert store.read_topic_memory("677") == "Dev data"
        store.append_topic_history("558", "Finance event")
        store.append_topic_history("677", "Dev event")
        h558 = store.read_topic_history("558")
        h677 = store.read_topic_history("677")
        assert "Finance event" in h558
        assert "Finance event" not in h677
        assert "Dev event" in h677

    def test_list_topics(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("558", "a")
        store.write_topic_memory("677", "b")
        topics = store.list_topics()
        assert set(topics) == {"558", "677"}
