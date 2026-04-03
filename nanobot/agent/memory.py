"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


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


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reactions_chat ON message_reactions(chat_id)")
            conn.execute("""CREATE TABLE IF NOT EXISTS message_sentiment (
                chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, topic TEXT NOT NULL,
                positive_count INTEGER DEFAULT 0, negative_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chat_id, message_id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_topic ON message_sentiment(topic)")
            conn.execute("""CREATE TABLE IF NOT EXISTS emoji_sentiment (
                emoji TEXT PRIMARY KEY,
                sentiment TEXT NOT NULL CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                learned_at TEXT NOT NULL DEFAULT (datetime('now')))""")

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
        from datetime import datetime

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
        from datetime import datetime

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
                "DELETE FROM message_reactions "
                "WHERE chat_id = ? AND message_id = ? AND emoji = ?",
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
                "SELECT message_id FROM message_sentiment "
                "WHERE topic = ? AND positive_count >= 1",
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
            row = conn.execute(
                "SELECT 1 FROM emoji_sentiment WHERE emoji = ?", (emoji,)
            ).fetchone()
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
                "DELETE FROM message_reactions "
                "WHERE created_at < datetime('now', ?)",
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
        logger.debug("CONSOLIDATION: topic={} high_value={} total={} messages", topic, high_count, len(messages))

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

    def _raw_archive(self, topic: str | None, messages: list[dict]) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw = f"[{ts}] [RAW] {len(messages)} messages\n{MemoryStore._format_messages(messages)}"
        if topic:
            self.append_topic_history(topic, raw)
        else:
            self.append_history(raw)
        logger.warning("Raw-archived {} messages for {}", len(messages), topic or "global")


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = (
                f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            )
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
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
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning(
                    "Memory consolidation: save_memory payload contains null required fields"
                )
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            self.append_history(entry)
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n{self._format_messages(messages)}"
        )
        logger.warning("Memory consolidation degraded: raw-archived {} messages", len(messages))


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = SqliteMemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self, messages: list[dict[str, object]], topic_name: str | None = None
    ) -> bool:
        """Archive a selected message chunk into persistent memory."""
        if topic_name is not None:
            return await self.store.consolidate_topic(
                topic_name, messages, self.provider, self.model
            )
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
        *,
        topic_name: str | None = None,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens.

        High-value messages (positively reacted) are skipped during boundary
        selection, so they are preserved longer in context.
        """
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        high_value_ids: set[int] = set()
        if topic_name:
            high_value_ids = set(self.store.get_high_value_messages(topic_name))

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            # Skip high-value messages — don't count them for removal
            msg_id = message.get("telegram_message_id")
            if msg_id is not None and msg_id in high_value_ids:
                continue
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = session.key.split(":", 1) if ":" in session.key else (None, None)
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES):
            if await self.consolidate_messages(messages):
                return True
        return True

    async def maybe_consolidate_by_tokens(
        self, session: Session, topic_name: str | None = None
    ) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(
                    session, max(1, estimated - target), topic_name=topic_name
                )
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated : end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk, topic_name=topic_name):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
