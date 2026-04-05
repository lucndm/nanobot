"""One-time migration: SQLite + JSONL + config.json → PostgreSQL."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from loguru import logger


def migrate_sqlite(sqlite_path: Path, pg_dsn: str) -> dict[str, int]:
    """Migrate all tables from SQLite to PostgreSQL. Returns {table: row_count}."""
    from psycopg_pool import ConnectionPool

    if not sqlite_path.exists():
        logger.warning("SQLite DB not found: {}", sqlite_path)
        return {}

    src = sqlite3.connect(str(sqlite_path))
    pool = ConnectionPool(pg_dsn, min_size=1, max_size=1, open=True)
    counts = {}

    tables = [
        "global_memory", "global_history", "topic_memory", "topic_history",
        "message_reactions", "message_sentiment", "emoji_sentiment",
        "topic_mapping", "topic_litellm",
    ]

    with pool.connection() as conn:
        for table in tables:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                counts[table] = 0
                continue

            cols = [d[0] for d in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
            placeholders = ", ".join(["%s"] * len(cols))
            col_names = ", ".join(cols)

            for row in rows:
                conn.execute(
                    f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
                    f"ON CONFLICT DO NOTHING",
                    row,
                )
            counts[table] = len(rows)
            logger.info("Migrated {} rows from {}", len(rows), table)
        conn.commit()

    src.close()
    pool.close()
    return counts


def migrate_sessions(sessions_dir: Path, pg_dsn: str) -> int:
    """Migrate JSONL session files to PostgreSQL turn_log. Returns message count."""
    from psycopg_pool import ConnectionPool

    if not sessions_dir.exists():
        logger.warning("Sessions dir not found: {}", sessions_dir)
        return 0

    pool = ConnectionPool(pg_dsn, min_size=1, max_size=1, open=True)
    total = 0

    with pool.connection() as conn:
        for jsonl_file in sessions_dir.rglob("*.jsonl"):
            session_key = jsonl_file.stem
            with open(jsonl_file) as f:
                for seq, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    conn.execute(
                        "INSERT INTO turn_log "
                        "(session_key, seq, role, content, tool_calls, tool_call_id, "
                        "model, prompt_tokens, completion_tokens, extra) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            session_key, seq, msg.get("role"), msg.get("content"),
                            json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None,
                            msg.get("tool_call_id"),
                            msg.get("model", ""),
                            msg.get("prompt_tokens", 0), msg.get("completion_tokens", 0),
                            json.dumps({"migrated": True}),
                        ),
                    )
                    total += 1
            logger.info("Migrated {} messages from {}", seq + 1, jsonl_file.name)
        conn.commit()

    pool.close()
    return total


def migrate_config(config_path: Path, pg_dsn: str) -> bool:
    """Import config.json into PostgreSQL config table."""
    from psycopg_pool import ConnectionPool

    if not config_path.exists():
        logger.warning("Config not found: {}", config_path)
        return False

    config_json = config_path.read_text()
    pool = ConnectionPool(pg_dsn, min_size=1, max_size=1, open=True)

    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO config (key, value, updated_at) VALUES ('main', %s, now()) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=now()",
            (config_json,),
        )
        conn.commit()

    pool.close()
    logger.info("Migrated config from {}", config_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate nanobot data to PostgreSQL")
    parser.add_argument("--sqlite-path", required=True, help="Path to memories.db")
    parser.add_argument("--postgres-url", required=True, help="PostgreSQL DSN")
    parser.add_argument("--sessions-dir", help="Path to sessions/ directory")
    parser.add_argument("--config-file", help="Path to config.json")
    args = parser.parse_args()

    logger.info("Starting migration to PostgreSQL...")

    # Create tables first (importing stores triggers table creation)
    from nanobot.agent.store import MemoryStore
    from nanobot.session.store import SessionStore

    MemoryStore(args.postgres_url)
    SessionStore(args.postgres_url)

    # Migrate data
    counts = migrate_sqlite(Path(args.sqlite_path), args.postgres_url)
    for table, count in counts.items():
        logger.info("  {}: {} rows", table, count)

    if args.sessions_dir:
        msg_count = migrate_sessions(Path(args.sessions_dir), args.postgres_url)
        logger.info("  sessions: {} messages", msg_count)

    if args.config_file:
        migrate_config(Path(args.config_file), args.postgres_url)

    # Backup
    sqlite_path = Path(args.sqlite_path)
    if sqlite_path.exists():
        backup = sqlite_path.with_suffix(".db.bak")
        sqlite_path.rename(backup)
        logger.info("Backed up {} → {}", sqlite_path, backup)

    if args.sessions_dir:
        sessions = Path(args.sessions_dir)
        if sessions.exists():
            backup = sessions.with_name(sessions.name + ".bak")
            sessions.rename(backup)
            logger.info("Backed up {} → {}", sessions, backup)

    logger.info("Migration complete!")


if __name__ == "__main__":
    main()
