"""PostgreSQL-backed memory store using psycopg."""

from __future__ import annotations

from datetime import datetime

from loguru import logger
from psycopg_pool import ConnectionPool

_POSITIVE_EMOJI = frozenset({"👍", "❤️", "🔥", "💯", "👏", "😊", "🎉", "⭐"})
_NEGATIVE_EMOJI = frozenset({"👎", "😡", "😢", "🤔", "😤"})


class PostgresMemoryStore:
    """PostgreSQL implementation of MemoryStoreProtocol."""

    def __init__(self, dsn: str, *, pool_size: int = 5) -> None:
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=pool_size,
            open=False,
        )
        self._pool.open()
        self._init_tables()
        logger.info("PostgresMemoryStore connected to {}", dsn.split("@")[-1] if "@" in dsn else dsn)

    def _init_tables(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_history (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    entry TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_memory (
                    topic TEXT PRIMARY KEY,
                    memory TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_history (
                    id BIGSERIAL PRIMARY KEY,
                    topic TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    entry TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_reactions (
                    id BIGSERIAL PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    message_id BIGINT NOT NULL,
                    emoji TEXT NOT NULL,
                    sentiment TEXT NOT NULL DEFAULT 'neutral',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(chat_id, message_id, emoji)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reactions_chat ON message_reactions(chat_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_sentiment (
                    chat_id TEXT NOT NULL,
                    message_id BIGINT NOT NULL,
                    topic TEXT NOT NULL,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    neutral_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (chat_id, message_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_topic ON message_sentiment(topic)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emoji_sentiment (
                    emoji TEXT PRIMARY KEY,
                    sentiment TEXT NOT NULL CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                    learned_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_mapping (
                    chat_id BIGINT NOT NULL,
                    thread_id BIGINT NOT NULL,
                    topic_name TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (chat_id, thread_id)
                )
            """)

    def close(self) -> None:
        self._pool.close()

    # ── Global Memory ────────────────────────────────────────────

    def read_long_term(self) -> str:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT value FROM global_memory WHERE key = 'long_term'").fetchone()
        return row[0] if row else ""

    def write_long_term(self, content: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO global_memory (key, value, updated_at) VALUES ('long_term', %s, now()) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (content,),
            )

    def read_history(self) -> str:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT entry FROM global_history ORDER BY id").fetchall()
        return "\n\n".join(r[0] for r in rows)

    def append_history(self, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO global_history (timestamp, entry) VALUES (%s, %s)",
                (ts, entry.rstrip()),
            )

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        if not long_term:
            return ""
        return f"## Global Memory\n{long_term}"

    # ── Topic Memory ─────────────────────────────────────────────

    def read_topic_memory(self, topic: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT memory FROM topic_memory WHERE topic = %s", (topic,)
            ).fetchone()
        return row[0] if row else None

    def write_topic_memory(self, topic: str, content: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO topic_memory (topic, memory, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT(topic) DO UPDATE SET memory=excluded.memory, updated_at=excluded.updated_at",
                (topic, content),
            )

    def read_topic_history(self, topic: str) -> str:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT entry FROM topic_history WHERE topic = %s ORDER BY id", (topic,)
            ).fetchall()
        return "\n\n".join(r[0] for r in rows)

    def append_topic_history(self, topic: str, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO topic_history (topic, timestamp, entry) VALUES (%s, %s, %s)",
                (topic, ts, entry.rstrip()),
            )

    def get_topic_memory_context(self, topic: str) -> str | None:
        memory = self.read_topic_memory(topic)
        if not memory:
            return None
        return f"## Topic Memory ({topic})\n{memory}"

    def list_topics(self) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT topic FROM topic_memory ORDER BY topic").fetchall()
        return [r[0] for r in rows]

    # ── Topic Mapping ────────────────────────────────────────────

    def get_topic_mapping(self, chat_id: int, thread_id: int) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT topic_name FROM topic_mapping WHERE chat_id = %s AND thread_id = %s",
                (chat_id, thread_id),
            ).fetchone()
        return row[0] if row else None

    def set_topic_mapping(self, chat_id: int, thread_id: int, topic_name: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO topic_mapping (chat_id, thread_id, topic_name, updated_at) "
                "VALUES (%s, %s, %s, now()) "
                "ON CONFLICT(chat_id, thread_id) DO UPDATE "
                "SET topic_name=excluded.topic_name, updated_at=excluded.updated_at",
                (chat_id, thread_id, topic_name),
            )

    def delete_topic_mapping(self, chat_id: int, thread_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM topic_mapping WHERE chat_id = %s AND thread_id = %s",
                (chat_id, thread_id),
            )

    def load_all_topic_mappings(self) -> dict[tuple[int, int], str]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT chat_id, thread_id, topic_name FROM topic_mapping").fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    # ── Reactions & Sentiment ────────────────────────────────────

    def record_reaction(
        self, chat_id: str, message_id: int, emoji: str, sentiment: str, topic: str
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO message_reactions (chat_id, message_id, emoji, sentiment) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (chat_id, message_id, emoji, sentiment),
            )
            count_col = f"{sentiment}_count"
            conn.execute(
                f"INSERT INTO message_sentiment (chat_id, message_id, topic, {count_col}) "
                f"VALUES (%s, %s, %s, 1) "
                f"ON CONFLICT(chat_id, message_id) DO UPDATE "
                f"SET {count_col} = message_sentiment.{count_col} + 1, "
                f"updated_at = now()",
                (chat_id, message_id, topic),
            )

    def remove_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT sentiment FROM message_reactions "
                "WHERE chat_id = %s AND message_id = %s AND emoji = %s",
                (chat_id, message_id, emoji),
            ).fetchone()
            if row:
                sentiment = row[0]
                count_col = f"{sentiment}_count"
                conn.execute(
                    "DELETE FROM message_reactions "
                    "WHERE chat_id = %s AND message_id = %s AND emoji = %s",
                    (chat_id, message_id, emoji),
                )
                conn.execute(
                    f"UPDATE message_sentiment SET {count_col} = GREATEST({count_col} - 1, 0) "
                    f"WHERE chat_id = %s AND message_id = %s",
                    (chat_id, message_id),
                )

    def get_message_sentiment(self, chat_id: str, message_id: int) -> dict[str, int]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT positive_count, negative_count, neutral_count "
                "FROM message_sentiment WHERE chat_id = %s AND message_id = %s",
                (chat_id, message_id),
            ).fetchone()
        if row:
            return {"positive_count": row[0], "negative_count": row[1], "neutral_count": row[2]}
        return {"positive_count": 0, "negative_count": 0, "neutral_count": 0}

    def get_high_value_messages(self, topic: str) -> list[int]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT message_id FROM message_sentiment WHERE topic = %s AND positive_count >= 1",
                (topic,),
            ).fetchall()
        return [r[0] for r in rows]

    def resolve_emoji_sentiment(self, emoji: str) -> str | None:
        if emoji in _POSITIVE_EMOJI:
            return "positive"
        if emoji in _NEGATIVE_EMOJI:
            return "negative"
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT sentiment FROM emoji_sentiment WHERE emoji = %s", (emoji,)
            ).fetchone()
        return row[0] if row else None

    def is_emoji_known(self, emoji: str) -> bool:
        if emoji in _POSITIVE_EMOJI or emoji in _NEGATIVE_EMOJI:
            return True
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM emoji_sentiment WHERE emoji = %s", (emoji,)
            ).fetchone()
        return row is not None

    def learn_emoji(self, emoji: str, sentiment: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO emoji_sentiment (emoji, sentiment) VALUES (%s, %s) "
                "ON CONFLICT(emoji) DO UPDATE SET sentiment=excluded.sentiment, learned_at=now()",
                (emoji, sentiment),
            )

    def cleanup_old_reactions(self, max_age_days: int = 30) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM message_reactions WHERE created_at < now() - interval '%s days'",
                    (max_age_days,),
                )
                return cur.rowcount
