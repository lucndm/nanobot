"""One-time migration: memory/MEMORY.md + HISTORY.md → data/memories.db."""

from pathlib import Path

from loguru import logger


def migrate_files_to_sqlite(workspace: Path) -> None:
    """Migrate file-based memory to SQLite if DB doesn't exist yet."""
    db_path = workspace / "data" / "memories.db"
    if db_path.exists():
        return

    memory_file = workspace / "memory" / "MEMORY.md"
    history_file = workspace / "memory" / "HISTORY.md"

    if not memory_file.exists() and not history_file.exists():
        return

    from nanobot.agent.memory import SqliteMemoryStore

    store = SqliteMemoryStore(workspace)

    if memory_file.exists():
        content = memory_file.read_text(encoding="utf-8")
        if content.strip():
            store.write_long_term(content)
            logger.info("Migrated MEMORY.md → SQLite ({} chars)", len(content))

    if history_file.exists():
        content = history_file.read_text(encoding="utf-8")
        for entry in content.split("\n\n"):
            entry = entry.strip()
            if entry:
                store.append_history(entry)
        logger.info("Migrated HISTORY.md → SQLite")

    logger.info("Migration complete. Old files can be deleted manually.")
