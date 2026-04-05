"""SQLite-backed memory store."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger

_POSITIVE_EMOJI = frozenset({"👍", "❤️", "🔥", "💯", "👏", "😊", "🎉", "⭐"})
_NEGATIVE_EMOJI = frozenset({"👎", "😡", "😢", "🤔", "😤"})

_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value):
    """Normalize tool-call payload values to text for file storage."""
    import json

    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args):
    """Normalize provider tool-call arguments to the expected dict shape."""
    import json

    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class SqliteMemoryStore:
    """SQLite-backed memory: global + per-topic. Replaces file-based MemoryStore."""

    def __init__(self, workspace: Path):
        self._db_path = workspace / "data" / "memories.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self._failures: dict[str, int] = {}

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS global_memory (
                key TEXT PRIMARY KEY, value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS global_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                entry TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS topic_memory (
                topic TEXT PRIMARY KEY, memory TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS topic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL,
                timestamp TEXT NOT NULL, entry TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS message_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL, message_id INTEGER NOT NULL,
                emoji TEXT NOT NULL, sentiment TEXT NOT NULL DEFAULT 'neutral',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(chat_id, message_id, emoji))""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reactions_chat ON message_reactions(chat_id)"
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS message_sentiment (
                chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, topic TEXT NOT NULL,
                positive_count INTEGER DEFAULT 0, negative_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chat_id, message_id))""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentiment_topic ON message_sentiment(topic)"
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS emoji_sentiment (
                emoji TEXT PRIMARY KEY,
                sentiment TEXT NOT NULL CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                learned_at TEXT NOT NULL DEFAULT (datetime('now')))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS topic_mapping (
                chat_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                topic_name TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chat_id, thread_id))""")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS topic_litellm ("
                "  topic_name TEXT PRIMARY KEY,"
                "  model TEXT NOT NULL,"
                "  temperature REAL NOT NULL,"
                "  max_tokens INTEGER NOT NULL,"
                "  updated_at TEXT"
                ")"
            )

    def read_long_term(self) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM global_memory WHERE key = 'long_term'").fetchone()
        return row[0] if row else ""

    def write_long_term(self, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO global_memory (key, value, updated_at) VALUES ('long_term', ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (content,),
            )

    def read_history(self) -> str:
        with self._conn() as conn:
            rows = conn.execute("SELECT entry FROM global_history ORDER BY id").fetchall()
        return "\n\n".join(r[0] for r in rows)

    def append_history(self, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO global_history (timestamp, entry) VALUES (?, ?)", (ts, entry.rstrip())
            )

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        if not long_term:
            return ""
        return f"## Global Memory\n{long_term}"

    # ── Topic CRUD ───────────────────────────────────────────────────

    def read_topic_memory(self, topic: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT memory FROM topic_memory WHERE topic = ?", (topic,)
            ).fetchone()
        return row[0] if row else None

    def write_topic_memory(self, topic: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO topic_memory (topic, memory, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(topic) DO UPDATE SET memory=excluded.memory, updated_at=excluded.updated_at",
                (topic, content),
            )

    def read_topic_history(self, topic: str) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT entry FROM topic_history WHERE topic = ? ORDER BY id", (topic,)
            ).fetchall()
        return "\n\n".join(r[0] for r in rows)

    def append_topic_history(self, topic: str, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO topic_history (topic, timestamp, entry) VALUES (?, ?, ?)",
                (topic, ts, entry.rstrip()),
            )

    def get_topic_memory_context(self, topic: str) -> str | None:
        memory = self.read_topic_memory(topic)
        if not memory:
            return None
        return f"## Topic Memory ({topic})\n{memory}"

    def list_topics(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT topic FROM topic_memory ORDER BY topic").fetchall()
        return [r[0] for r in rows]

    # ── Topic Mapping ────────────────────────────────────────────────

    def get_topic_mapping(self, chat_id: int, thread_id: int) -> str | None:
        """Get topic name for a (chat_id, thread_id) pair."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT topic_name FROM topic_mapping WHERE chat_id = ? AND thread_id = ?",
                (chat_id, thread_id),
            ).fetchone()
        return row[0] if row else None

    def set_topic_mapping(self, chat_id: int, thread_id: int, topic_name: str) -> None:
        """Insert or update topic name mapping."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO topic_mapping (chat_id, thread_id, topic_name, updated_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(chat_id, thread_id) DO UPDATE "
                "SET topic_name=excluded.topic_name, updated_at=excluded.updated_at",
                (chat_id, thread_id, topic_name),
            )

    def delete_topic_mapping(self, chat_id: int, thread_id: int) -> None:
        """Remove a topic mapping."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM topic_mapping WHERE chat_id = ? AND thread_id = ?",
                (chat_id, thread_id),
            )

    def load_all_topic_mappings(self) -> dict[tuple[int, int], str]:
        """Load all topic mappings. Returns {(chat_id, thread_id): topic_name}."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chat_id, thread_id, topic_name FROM topic_mapping"
            ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    # ── Topic Litellm Config ────────────────────────────────────

    def get_topic_litellm(self, topic_name: str) -> tuple[str, float, int] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT model, temperature, max_tokens FROM topic_litellm WHERE topic_name = ?",
                (topic_name,),
            ).fetchone()
        return tuple(row) if row else None

    def set_topic_litellm(
        self, topic_name: str, model: str, temperature: float, max_tokens: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO topic_litellm (topic_name, model, temperature, max_tokens, updated_at) "
                "VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(topic_name) DO UPDATE "
                "SET model=excluded.model, temperature=excluded.temperature, "
                "max_tokens=excluded.max_tokens, updated_at=excluded.updated_at",
                (topic_name, model, temperature, max_tokens),
            )

    def delete_topic_litellm(self, topic_name: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM topic_litellm WHERE topic_name = ?", (topic_name,))

    def _list_topic_litellm(self) -> list[tuple[str, str, float, int]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT topic_name, model, temperature, max_tokens FROM topic_litellm"
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def sync_topic_files(self, workspace: Path) -> None:
        """Regenerate all TOPIC.md files from PostgreSQL (DB is source of truth).

        This is a one-way sync: DB -> File. The file is always regenerated from DB,
        never read back to update DB.
        """
        topics_dir = workspace / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)

        for topic_name, model, temp, tokens in self._list_topic_litellm():
            topic_dir = topics_dir / topic_name
            topic_dir.mkdir(parents=True, exist_ok=True)
            topic_file = topic_dir / "TOPIC.md"

            # Read purpose from topic_memory (if any)
            purpose = self.read_topic_memory(topic_name) or ""
            purpose_section = f"## purpose\n{purpose.strip()}\n\n" if purpose.strip() else ""

            topic_file.write_text(
                f"# Topic: {topic_name}\n\n{purpose_section}## litellm\n"
                f"model: {model}\ntemperature: {temp}\nmax_tokens: {tokens}\n",
                encoding="utf-8",
            )

    # ── Reactions ──────────────────────────────────────────────────

    def record_reaction(
        self, chat_id: str, message_id: int, emoji: str, sentiment: str, topic: str
    ) -> None:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO message_reactions "
                "(chat_id, message_id, emoji, sentiment) VALUES (?, ?, ?, ?)",
                (chat_id, message_id, emoji, sentiment),
            )
            if cursor.rowcount == 0:
                return  # Duplicate reaction, nothing to do
            # Upsert sentiment counts
            count_col = f"{sentiment}_count"
            conn.execute(
                f"INSERT INTO message_sentiment (chat_id, message_id, topic, {count_col}) "
                f"VALUES (?, ?, ?, 1) "
                f"ON CONFLICT(chat_id, message_id) DO UPDATE SET "
                f"{count_col} = {count_col} + 1, "
                f"updated_at = datetime('now')",
                (chat_id, message_id, topic),
            )

    def remove_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT sentiment FROM message_reactions "
                "WHERE chat_id = ? AND message_id = ? AND emoji = ?",
                (chat_id, message_id, emoji),
            ).fetchone()
            if row is None:
                return
            sentiment = row[0]
            conn.execute(
                "DELETE FROM message_reactions WHERE chat_id = ? AND message_id = ? AND emoji = ?",
                (chat_id, message_id, emoji),
            )
            count_col = f"{sentiment}_count"
            conn.execute(
                f"UPDATE message_sentiment SET {count_col} = MAX({count_col} - 1, 0), "
                f"updated_at = datetime('now') "
                f"WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            )

    def get_message_sentiment(self, chat_id: str, message_id: int) -> dict[str, int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT positive_count, negative_count, neutral_count "
                "FROM message_sentiment WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ).fetchone()
        if row is None:
            return {"positive_count": 0, "negative_count": 0, "neutral_count": 0}
        return {
            "positive_count": row[0] or 0,
            "negative_count": row[1] or 0,
            "neutral_count": row[2] or 0,
        }

    def get_high_value_messages(self, topic: str) -> list[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT message_id FROM message_sentiment WHERE topic = ? AND positive_count >= 1",
                (topic,),
            ).fetchall()
        return [r[0] for r in rows]

    def resolve_emoji_sentiment(self, emoji: str) -> str | None:
        if emoji in _POSITIVE_EMOJI:
            return "positive"
        if emoji in _NEGATIVE_EMOJI:
            return "negative"
        with self._conn() as conn:
            row = conn.execute(
                "SELECT sentiment FROM emoji_sentiment WHERE emoji = ?", (emoji,)
            ).fetchone()
        return row[0] if row else None

    def is_emoji_known(self, emoji: str) -> bool:
        if emoji in _POSITIVE_EMOJI or emoji in _NEGATIVE_EMOJI:
            return True
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM emoji_sentiment WHERE emoji = ?", (emoji,)).fetchone()
        return row is not None

    def learn_emoji(self, emoji: str, sentiment: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO emoji_sentiment (emoji, sentiment) VALUES (?, ?)",
                (emoji, sentiment),
            )

    def cleanup_old_reactions(self, max_age_days: int = 30) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM message_reactions WHERE created_at < datetime('now', ?)",
                (f"-{max_age_days} days",),
            )
            return cursor.rowcount

    # ── Consolidation ──────────────────────────────────────────────

    _MAX_FAILURES = 3

    def _failure_key(self, topic: str | None) -> str:
        return topic if topic else "global"

    def format_messages_for_consolidation(self, messages: list[dict], topic: str | None) -> str:
        """Format messages, prefixing high-value ones with [HIGH VALUE]."""
        high_value_ids: set[int] = set()
        if topic:
            high_value_ids = set(self.get_high_value_messages(topic))

        lines = []
        for msg in messages:
            if not msg.get("content"):
                continue
            tools = f" [tools: {', '.join(msg['tools_used'])}]" if msg.get("tools_used") else ""
            prefix = "[HIGH VALUE] " if msg.get("telegram_message_id") in high_value_ids else ""
            lines.append(
                f"[{msg.get('timestamp', '?')[:16]}] {msg['role'].upper()}{tools}: {prefix}{msg['content']}"
            )
        return "\n".join(lines)

    async def consolidate(self, messages: list[dict], provider, model: str) -> bool:
        return await self._do_consolidate(None, messages, provider, model)

    async def consolidate_topic(
        self, topic: str, messages: list[dict], provider, model: str
    ) -> bool:
        return await self._do_consolidate(topic, messages, provider, model)

    async def _do_consolidate(
        self, topic: str | None, messages: list[dict], provider, model: str
    ) -> bool:
        if not messages:
            return True

        high_value_ids = set(self.get_high_value_messages(topic)) if topic else set()
        high_count = sum(1 for m in messages if m.get("telegram_message_id") in high_value_ids)
        logger.debug(
            "CONSOLIDATION: topic={} high_value={} total={} messages",
            topic,
            high_count,
            len(messages),
        )

        current = self.read_topic_memory(topic) if topic else self.read_long_term()
        formatted = self.format_messages_for_consolidation(messages, topic)
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current {"Topic" if topic else "Long-term"} Memory
{current or "(empty)"}

## Conversation to Process
{formatted}

Prioritize preserving insights from [HIGH VALUE] messages — the user confirmed these are valuable."""

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory tool.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                return self._fail_or_raw_archive(topic, messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if not args or "history_entry" not in args or "memory_update" not in args:
                return self._fail_or_raw_archive(topic, messages)

            entry = _ensure_text(args["history_entry"]).strip()
            update = _ensure_text(args["memory_update"])

            if not entry or args["history_entry"] is None or args["memory_update"] is None:
                return self._fail_or_raw_archive(topic, messages)

            if topic:
                self.append_topic_history(topic, entry)
                if update != current:
                    self.write_topic_memory(topic, update)
            else:
                self.append_history(entry)
                if update != current:
                    self.write_long_term(update)

            self._failures[self._failure_key(topic)] = 0
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(topic, messages)

    def _fail_or_raw_archive(self, topic: str | None, messages: list[dict]) -> bool:
        key = self._failure_key(topic)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] < self._MAX_FAILURES:
            return False
        self._raw_archive(topic, messages)
        self._failures[key] = 0
        return True

    @staticmethod
    def _format_raw_messages(messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            if not msg.get("content"):
                continue
            lines.append(
                f"[{msg.get('timestamp', '?')[:16]}] {msg['role'].upper()}: {msg['content']}"
            )
        return "\n".join(lines)

    def _raw_archive(self, topic: str | None, messages: list[dict]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = f"[{ts}] [RAW] {len(messages)} messages\n{self._format_raw_messages(messages)}"
        if topic:
            self.append_topic_history(topic, raw)
        else:
            self.append_history(raw)
        logger.warning("Raw-archived {} messages for {}", len(messages), topic or "global")
