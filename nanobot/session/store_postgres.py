"""PostgreSQL-backed session store using psycopg."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger
from psycopg_pool import ConnectionPool


class PostgresSessionStore:
    """PostgreSQL implementation of SessionStoreProtocol."""

    def __init__(self, dsn: str, *, pool_size: int = 5) -> None:
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=pool_size,
            open=False,
        )
        self._pool.open()
        self._init_tables()
        logger.info(
            "PostgresSessionStore connected to {}",
            dsn.split("@")[-1] if "@" in dsn else dsn,
        )

    def _init_tables(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    last_consolidated INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_key TEXT NOT NULL REFERENCES sessions(key) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    message JSONB NOT NULL,
                    UNIQUE(session_key, seq)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_key "
                "ON session_messages(session_key)"
            )

    def close(self) -> None:
        self._pool.close()

    def get_or_create(self, key: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT created_at, updated_at, metadata, last_consolidated "
                "FROM sessions WHERE key = %s",
                (key,),
            ).fetchone()

            if row is None:
                now = datetime.now().isoformat()
                conn.execute(
                    "INSERT INTO sessions (key, created_at, updated_at, metadata, last_consolidated) "
                    "VALUES (%s, %s, %s, '{}', 0)",
                    (key, now, now),
                )
                return {
                    "key": key,
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                    "metadata": {},
                    "last_consolidated": 0,
                }

            created_at, updated_at, metadata, last_consolidated = row
            msg_rows = conn.execute(
                "SELECT message FROM session_messages WHERE session_key = %s ORDER BY seq",
                (key,),
            ).fetchall()
            messages = [r[0] for r in msg_rows]

            return {
                "key": key,
                "messages": messages,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata if isinstance(metadata, dict) else json.loads(metadata),
                "last_consolidated": last_consolidated,
            }

    def save(self, session_data: dict[str, Any]) -> None:
        key = session_data["key"]
        now = session_data.get("updated_at", datetime.now().isoformat())
        messages = session_data.get("messages", [])
        metadata = session_data.get("metadata", {})
        last_consolidated = session_data.get("last_consolidated", 0)
        created_at = session_data.get("created_at", now)

        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO sessions (key, created_at, updated_at, metadata, last_consolidated) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT(key) DO UPDATE SET "
                "updated_at=excluded.updated_at, metadata=excluded.metadata, "
                "last_consolidated=excluded.last_consolidated",
                (key, created_at, now, json.dumps(metadata), last_consolidated),
            )
            # Replace all messages for this session
            conn.execute("DELETE FROM session_messages WHERE session_key = %s", (key,))
            if messages:
                # Use executemany for batch insert
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO session_messages (session_key, seq, message) "
                        "VALUES (%s, %s, %s)",
                        [
                            (key, i, json.dumps(msg, ensure_ascii=False))
                            for i, msg in enumerate(messages)
                        ],
                    )

    def invalidate(self, key: str) -> None:
        # Postgres has no in-memory cache — no-op
        pass

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [{"key": r[0], "created_at": r[1], "updated_at": r[2]} for r in rows]
