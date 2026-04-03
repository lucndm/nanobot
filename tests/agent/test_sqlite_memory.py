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
