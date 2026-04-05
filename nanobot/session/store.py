"""Session store protocol and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Interface for session persistence backends."""

    def get_or_create(self, key: str) -> dict[str, Any]:
        """Load session data dict or return empty template.

        Returns dict with keys: key, messages, created_at, updated_at, metadata, last_consolidated.
        """
        ...

    def save(self, session_data: dict[str, Any]) -> None:
        """Persist session data dict."""
        ...

    def invalidate(self, key: str) -> None:
        """Remove session from cache (keep persisted data)."""
        ...

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with metadata.

        Returns list of dicts with keys: key, created_at, updated_at.
        """
        ...

    def consolidate(self, session_key: str, topic_id: str, summary: str, last_seq: int) -> None:
        """Store consolidation summary for a session+topic."""
        ...

    def get_summary(self, session_key: str, topic_id: str) -> dict | None:
        """Get consolidation summary. Returns None if no summary exists."""
        ...

    def get_usage(
        self,
        *,
        session_key: str | None = None,
        topic_id: str | None = None,
        model: str | None = None,
        since: Any | None = None,
    ) -> dict[str, int]:
        """Aggregate token usage with optional filters."""
        ...


def create_session_store(config: object, workspace: Path) -> SessionStoreProtocol:
    """Create a session store based on config.database.backend."""
    db_config = getattr(config, "database", None)
    backend = getattr(db_config, "backend", "sqlite") if db_config else "sqlite"

    if backend == "postgres":
        from nanobot.session.store_postgres import PostgresSessionStore

        url = getattr(db_config, "url", "")
        pool_size = getattr(db_config, "pool_size", 5)
        if not url:
            raise ValueError("database.url is required when backend is 'postgres'")
        return PostgresSessionStore(url, pool_size=pool_size)

    from nanobot.session.store_jsonl import JsonlSessionStore

    return JsonlSessionStore(workspace)
