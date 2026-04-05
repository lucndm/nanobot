"""Session management for conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.metrics import Observation

from nanobot.observability.otel import get_meter
from nanobot.session.store import SessionStore


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start : i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(
        self, max_messages: int, *, high_value_ids: set[int] | None = None
    ) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules.

        When high_value_ids is provided, prefer keeping messages whose
        telegram_message_id is in the set.
        """
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        high_value_ids = high_value_ids or set()

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        # Prefer keeping high-value messages: if any dropped messages are high-value
        # and any retained messages are not, swap them.
        if high_value_ids:
            dropped = self.messages[: len(self.messages) - len(retained)]
            retained_non_hv = [
                m for m in retained if m.get("telegram_message_id") not in high_value_ids
            ]
            dropped_hv = [m for m in dropped if m.get("telegram_message_id") in high_value_ids]
            # Swap up to min(count) messages from dropped high-value with retained non-high-value
            swap_count = min(len(dropped_hv), len(retained_non_hv))
            if swap_count > 0:
                # Build new retained list: keep existing order, swap the oldest non-HV with oldest HV from dropped
                new_retained = []
                hv_idx = 0
                for m in retained:
                    if (
                        swap_count > 0
                        and m.get("telegram_message_id") not in high_value_ids
                        and hv_idx < len(dropped_hv)
                    ):
                        new_retained.append(dropped_hv[hv_idx])
                        hv_idx += 1
                        swap_count -= 1
                    else:
                        new_retained.append(m)
                retained = new_retained

        dropped_count = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped_count)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Delegates persistence to SessionStore backend.
    """

    def __init__(self, store: SessionStore):
        self._store: SessionStore = store
        self._cache: dict[str, Session] = {}

        meter = get_meter()
        if meter is not None:
            meter.create_observable_gauge(
                "nanobot.session.active",
                callbacks=[self._observe_active_sessions],
                description="Number of active sessions in cache",
            )

    def _observe_active_sessions(self, options):
        yield Observation(len(self._cache), attributes={})

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        data = self._store.get_or_create(key)
        session = self._data_to_session(data)

        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        """Save a session via the store backend."""
        self._store.save(self._session_to_data(session))
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
        self._store.invalidate(key)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        return self._store.list_sessions()

    @staticmethod
    def _data_to_session(data: dict[str, Any]) -> Session:
        """Convert store data dict to Session object."""
        created_at = data.get("created_at", "")
        try:
            created_at_dt = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            created_at_dt = datetime.now()

        return Session(
            key=data["key"],
            messages=data.get("messages", []),
            created_at=created_at_dt,
            metadata=data.get("metadata", {}),
            last_consolidated=data.get("last_consolidated", 0),
        )

    @staticmethod
    def _session_to_data(session: Session) -> dict[str, Any]:
        """Convert Session object to store data dict."""
        return {
            "key": session.key,
            "messages": session.messages,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated,
        }
