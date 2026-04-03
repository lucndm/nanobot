"""Test one-time migration from file-based memory to SQLite."""

from pathlib import Path

from nanobot.agent.memory import SqliteMemoryStore
from nanobot.agent.memory_migrate import migrate_files_to_sqlite


class TestMigration:
    def test_migrates_memory_md_to_sqlite(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Memory\nUser likes Python.", encoding="utf-8")
        (memory_dir / "HISTORY.md").write_text("[2026-01-01] Event.\n", encoding="utf-8")

        migrate_files_to_sqlite(tmp_path)

        store = SqliteMemoryStore(tmp_path)
        assert store.read_long_term() == "# Memory\nUser likes Python."
        assert "Event." in store.read_history()

    def test_skips_if_no_files(self, tmp_path: Path):
        migrate_files_to_sqlite(tmp_path)
        store = SqliteMemoryStore(tmp_path)
        assert store.read_long_term() == ""

    def test_skips_if_db_already_exists(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("old data", encoding="utf-8")
        # Create DB first
        store = SqliteMemoryStore(tmp_path)
        store.write_long_term("new data")

        migrate_files_to_sqlite(tmp_path)

        assert store.read_long_term() == "new data"
