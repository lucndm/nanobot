"""PostgreSQL-backed session store with append-only turn_log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from psycopg_pool import ConnectionPool

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS system_prompts (
    hash       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS turn_log (
    id                    BIGSERIAL PRIMARY KEY,
    session_key           TEXT NOT NULL,
    role                  TEXT NOT NULL,
    seq                   INTEGER NOT NULL,
    content               TEXT,
    tool_calls            JSONB,
    tool_call_id          TEXT,
    tool_name             TEXT,
    model                 TEXT,
    system_prompt_hash    TEXT REFERENCES system_prompts(hash),
    prompt_tokens         INTEGER DEFAULT 0,
    completion_tokens     INTEGER DEFAULT 0,
    cache_read_tokens     INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    stop_reason           TEXT,
    topic_id              TEXT,
    channel_message_id    TEXT,
    extra                 JSONB DEFAULT '{}',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_session_seq ON turn_log(session_key, seq);
CREATE INDEX IF NOT EXISTS idx_turn_created ON turn_log(created_at);

CREATE TABLE IF NOT EXISTS turn_summaries (
    session_key   TEXT NOT NULL,
    topic_id      TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL,
    last_seq      INTEGER NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_key, topic_id)
);
"""

_DROP_LEGACY_SQL = """
DROP TABLE IF EXISTS session_messages;
DROP TABLE IF EXISTS sessions;
"""


class PostgresSessionStore:
    """PostgreSQL session store with append-only turn_log."""

    def __init__(self, dsn: str, *, pool_size: int = 5) -> None:
        self._pool = ConnectionPool(dsn, min_size=1, max_size=pool_size, open=True)
        self._init_tables()
        logger.info(
            "PostgresSessionStore connected to {}",
            dsn.split("@")[-1] if "@" in dsn else dsn,
        )

    def _init_tables(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(_DROP_LEGACY_SQL)
            conn.execute(_SCHEMA_SQL)
            conn.commit()
        logger.info("Postgres session tables initialized (turn_log schema)")

    def close(self) -> None:
        self._pool.close()

    def get_or_create(self, key: str) -> dict[str, Any]:
        """Load session or return empty template."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, tool_name, "
                "model, system_prompt_hash, prompt_tokens, completion_tokens, "
                "cache_read_tokens, cache_creation_tokens, stop_reason, "
                "topic_id, channel_message_id, extra, seq "
                "FROM turn_log WHERE session_key = %s ORDER BY seq",
                (key,),
            ).fetchall()

        if not rows:
            return _empty_session(key)

        now = datetime.now(timezone.utc)
        messages = []
        for row in rows:
            msg: dict[str, Any] = {"role": row[0], "content": row[1], "timestamp": now.isoformat()}
            if row[2] is not None:
                msg["tool_calls"] = row[2]
            if row[3] is not None:
                msg["tool_call_id"] = row[3]
            if row[4] is not None:
                msg["tool_name"] = row[4]
            if row[5] is not None:
                msg["model"] = row[5]
            if row[6] is not None:
                msg["system_prompt_hash"] = row[6]
            if row[7]:
                msg["prompt_tokens"] = row[7]
            if row[8]:
                msg["completion_tokens"] = row[8]
            if row[9]:
                msg["cache_read_tokens"] = row[9]
            if row[10]:
                msg["cache_creation_tokens"] = row[10]
            if row[11] is not None:
                msg["stop_reason"] = row[11]
            if row[12] is not None:
                msg["topic_id"] = row[12]
            if row[13] is not None:
                msg["channel_message_id"] = row[13]
                # Backward compat: also expose as telegram_message_id
                msg["telegram_message_id"] = row[13]
            if row[14] and row[14] != {}:
                msg.update(row[14])
            messages.append(msg)

        return {
            "key": key,
            "messages": messages,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "metadata": {},
            "last_consolidated": 0,
        }

    def save(self, session_data: dict[str, Any]) -> None:
        """Append new messages to turn_log."""
        key = session_data["key"]
        messages = session_data.get("messages", [])

        with self._pool.connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM turn_log WHERE session_key = %s", (key,)
            ).fetchone()[0]

            new_msgs = messages[existing:]
            if not new_msgs:
                return

            for i, msg in enumerate(new_msgs):
                seq = existing + i
                extra = _extract_extra(msg)
                # Accept both telegram_message_id (legacy) and channel_message_id
                ch_msg_id = msg.get("channel_message_id") or msg.get("telegram_message_id")
                # JSONB columns need JSON serialization
                tool_calls_val = _json_sanitize(msg.get("tool_calls"))
                extra_val = _json_sanitize(extra)
                conn.execute(
                    "INSERT INTO turn_log "
                    "(session_key, seq, role, content, tool_calls, tool_call_id, tool_name, "
                    "model, system_prompt_hash, prompt_tokens, completion_tokens, "
                    "cache_read_tokens, cache_creation_tokens, stop_reason, topic_id, "
                    "channel_message_id, extra) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        key, seq,
                        msg.get("role"), msg.get("content"),
                        tool_calls_val,
                        msg.get("tool_call_id"), msg.get("tool_name"),
                        msg.get("model"), msg.get("system_prompt_hash"),
                        msg.get("prompt_tokens", 0), msg.get("completion_tokens", 0),
                        msg.get("cache_read_tokens", 0), msg.get("cache_creation_tokens", 0),
                        msg.get("stop_reason"), msg.get("topic_id"),
                        ch_msg_id,
                        extra_val,
                    ),
                )
            conn.commit()

    def consolidate(self, session_key: str, topic_id: str, summary: str, last_seq: int) -> None:
        """Upsert consolidation summary. turn_log rows are kept."""
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO turn_summaries (session_key, topic_id, summary, last_seq, updated_at) "
                "VALUES (%s, %s, %s, %s, now()) "
                "ON CONFLICT (session_key, topic_id) DO UPDATE "
                "SET summary = EXCLUDED.summary, last_seq = EXCLUDED.last_seq, updated_at = now()",
                (session_key, topic_id or "", summary, last_seq),
            )
            conn.commit()

    def get_summary(self, session_key: str, topic_id: str) -> dict | None:
        """Get consolidation summary for a session+topic."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT summary, last_seq, updated_at FROM turn_summaries "
                "WHERE session_key = %s AND topic_id = %s",
                (session_key, topic_id or ""),
            ).fetchone()
        if not row:
            return None
        return {"summary": row[0], "last_seq": row[1], "updated_at": row[2]}

    def get_usage(
        self,
        *,
        session_key: str | None = None,
        topic_id: str | None = None,
        model: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, int]:
        """Aggregate token usage with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if session_key:
            conditions.append("session_key = %s")
            params.append(session_key)
        if topic_id:
            conditions.append("topic_id = %s")
            params.append(topic_id)
        if model:
            conditions.append("model = %s")
            params.append(model)
        if since:
            conditions.append("created_at >= %s")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT COALESCE(SUM(prompt_tokens), 0), "
                f"COALESCE(SUM(completion_tokens), 0), "
                f"COALESCE(SUM(cache_read_tokens), 0), "
                f"COALESCE(SUM(cache_creation_tokens), 0), "
                f"COUNT(*) "
                f"FROM turn_log {where}",
                params,
            ).fetchone()
        return {
            "prompt_tokens": row[0],
            "completion_tokens": row[1],
            "cache_read_tokens": row[2],
            "cache_creation_tokens": row[3],
            "turns": row[4],
        }

    def invalidate(self, key: str) -> None:
        pass  # No in-memory cache

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT session_key, MIN(created_at) as created_at, MAX(created_at) as updated_at "
                "FROM turn_log GROUP BY session_key ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {"key": r[0], "created_at": r[1].isoformat() if r[1] else "", "updated_at": r[2].isoformat() if r[2] else ""}
            for r in rows
        ]


def _empty_session(key: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "key": key,
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "metadata": {},
        "last_consolidated": 0,
    }


# Fields that go into the `extra` JSONB column instead of dedicated columns
_EXTRA_KEYS = frozenset([
    "reasoning_content", "thinking_blocks",
    "timestamp", "total_tokens",
])


def _extract_extra(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k in _EXTRA_KEYS and v is not None}


def _json_sanitize(obj: Any) -> str | None:
    """Strip null bytes and serialize to JSON for PostgreSQL JSONB columns."""
    if obj is None:
        return None
    cleaned = _strip_null_bytes(obj)
    return json.dumps(cleaned, ensure_ascii=False)


def _strip_null_bytes(obj: Any) -> Any:
    """Recursively strip null bytes for PostgreSQL JSONB compatibility."""
    if isinstance(obj, str):
        return obj.replace("\u0000", "")
    if isinstance(obj, dict):
        return {k: _strip_null_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_null_bytes(v) for v in obj]
    return obj
