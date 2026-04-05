"""Tests for SqliteMemoryStore."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from nanobot.agent.memory import SqliteMemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


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


class TestSqliteMemoryStoreConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_global_writes_to_db(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="c1", name="save_memory",
                    arguments={"history_entry": "[2026-01-01] Event.", "memory_update": "# Mem\nFact."},
                )],
            )
        )
        result = await store.consolidate(
            [{"role": "user", "content": "hello", "timestamp": "2026-01-01"}],
            provider, "model",
        )
        assert result is True
        assert store.read_long_term() == "# Mem\nFact."
        assert "Event." in store.read_history()

    @pytest.mark.asyncio
    async def test_consolidate_topic_writes_to_topic_tables(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="c1", name="save_memory",
                    arguments={"history_entry": "[2026-01-01] Topic event.", "memory_update": "# Topic\nFact."},
                )],
            )
        )
        result = await store.consolidate_topic(
            "558",
            [{"role": "user", "content": "hello", "timestamp": "2026-01-01"}],
            provider, "model",
        )
        assert result is True
        assert store.read_topic_memory("558") == "# Topic\nFact."
        assert "Topic event." in store.read_topic_history("558")
        assert store.read_long_term() == ""

    @pytest.mark.asyncio
    async def test_consolidate_topic_does_not_affect_other_topics(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.write_topic_memory("677", "Dev data")
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(
                    id="c1", name="save_memory",
                    arguments={"history_entry": "[2026-01-01] Finance.", "memory_update": "# Finance"},
                )],
            )
        )
        await store.consolidate_topic("558", [{"role": "user", "content": "x", "timestamp": "t"}], provider, "m")
        assert store.read_topic_memory("677") == "Dev data"


class TestTopicMapping:
    def test_get_mapping_returns_none_when_missing(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        assert store.get_topic_mapping(-1003738155502, 4) is None

    def test_set_and_get_mapping(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_mapping(-1003738155502, 4, "Finance")
        assert store.get_topic_mapping(-1003738155502, 4) == "Finance"

    def test_set_mapping_overwrites(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_mapping(-1003738155502, 4, "Finance")
        store.set_topic_mapping(-1003738155502, 4, "Finance Tracker")
        assert store.get_topic_mapping(-1003738155502, 4) == "Finance Tracker"

    def test_delete_mapping(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_mapping(-1003738155502, 4, "Finance")
        store.delete_topic_mapping(-1003738155502, 4)
        assert store.get_topic_mapping(-1003738155502, 4) is None

    def test_delete_nonexistent_is_noop(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.delete_topic_mapping(-1003738155502, 999)  # should not raise

    def test_load_all_mappings(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_mapping(-1003738155502, 4, "Finance")
        store.set_topic_mapping(-1003738155502, 6, "General")
        store.set_topic_mapping(-1003738155502, 558, "Skills")
        mappings = store.load_all_topic_mappings()
        assert mappings[(-1003738155502, 4)] == "Finance"
        assert mappings[(-1003738155502, 6)] == "General"
        assert mappings[(-1003738155502, 558)] == "Skills"
        assert len(mappings) == 3

    def test_different_chats_same_thread_id(self, tmp_path: Path):
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_mapping(-1003738155502, 4, "Finance")
        store.set_topic_mapping(-9999999999, 4, "Other Chat Topic")
        assert store.get_topic_mapping(-1003738155502, 4) == "Finance"
        assert store.get_topic_mapping(-9999999999, 4) == "Other Chat Topic"


class TestTopicLitellm:
    def test_set_and_get_topic_litellm(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_litellm("my-topic", "test/model", 0.7, 4096)
        result = store.get_topic_litellm("my-topic")
        assert result is not None
        assert result == ("test/model", 0.7, 4096)

    def test_get_topic_litellm_missing_returns_none(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)
        result = store.get_topic_litellm("nonexistent")
        assert result is None

    def test_set_topic_litellm_upserts(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_litellm("my-topic", "old/model", 0.5, 2048)
        store.set_topic_litellm("my-topic", "new/model", 0.9, 8192)
        result = store.get_topic_litellm("my-topic")
        assert result == ("new/model", 0.9, 8192)

    def test_delete_topic_litellm(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_litellm("my-topic", "test/model", 0.7, 4096)
        store.delete_topic_litellm("my-topic")
        result = store.get_topic_litellm("my-topic")
        assert result is None


class TestTopicLitellmSync:
    def test_sync_creates_topic_md_from_store(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)
        store.set_topic_litellm("my-topic", "test/model", 0.7, 4096)

        store.sync_topic_files(tmp_path)

        topic_file = tmp_path / "topics" / "my-topic" / "TOPIC.md"
        assert topic_file.exists()
        content = topic_file.read_text()
        assert "test/model" in content

    def test_sync_imports_orphan_topic_files(self, tmp_path: Path):
        from nanobot.agent.store_sqlite import SqliteMemoryStore
        store = SqliteMemoryStore(tmp_path)

        topic_dir = tmp_path / "topics" / "orphan-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: orphan-topic\n\n## litellm\nmodel: orphan/model\ntemperature: 0.5\nmax_tokens: 2048\n"
        )

        store.sync_topic_files(tmp_path)

        result = store.get_topic_litellm("orphan-topic")
        assert result is not None
        assert result[0] == "orphan/model"
