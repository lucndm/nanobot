"""Tests for PostgresSessionStore CRUD operations."""

from __future__ import annotations

import os

import pytest

pg_url = os.environ.get("NANOBOT_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not pg_url,
    reason="Set NANOBOT_PG_URL to run PostgresSessionStore tests",
)


@pytest.fixture()
def store():
    from nanobot.session.store_postgres import PostgresSessionStore

    s = PostgresSessionStore(pg_url, pool_size=2)
    yield s
    # Cleanup: drop test tables
    with s._pool.connection() as conn:
        conn.execute("DELETE FROM session_messages")
        conn.execute("DELETE FROM sessions")
    s.close()


def test_get_or_create_returns_empty(store):
    data = store.get_or_create("test:new_session")
    assert data["key"] == "test:new_session"
    assert data["messages"] == []
    assert data["last_consolidated"] == 0


def test_save_and_reload(store):
    data = store.get_or_create("test:save_reload")
    data["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    data["metadata"] = {"topic": "test"}
    data["last_consolidated"] = 1
    store.save(data)

    # Reload from DB
    loaded = store.get_or_create("test:save_reload")
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["role"] == "user"
    assert loaded["metadata"]["topic"] == "test"
    assert loaded["last_consolidated"] == 1


def test_save_replaces_messages(store):
    data = store.get_or_create("test:replace")
    data["messages"] = [{"role": "user", "content": "old"}]
    store.save(data)

    data["messages"] = [{"role": "user", "content": "new1"}, {"role": "assistant", "content": "new2"}]
    store.save(data)

    loaded = store.get_or_create("test:replace")
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["content"] == "new1"


def test_list_sessions(store):
    store.get_or_create("test:list_a")
    store.get_or_create("test:list_b")

    sessions = store.list_sessions()
    keys = [s["key"] for s in sessions]
    assert "test:list_a" in keys
    assert "test:list_b" in keys


def test_invalidate_is_noop(store):
    # Postgres has no in-memory cache
    store.invalidate("test:whatever")
    # Should not raise


def test_session_manager_with_postgres_store():
    """SessionManager works with PostgresSessionStore via protocol."""
    from nanobot.session.manager import SessionManager
    from nanobot.session.store_postgres import PostgresSessionStore

    store = PostgresSessionStore(pg_url, pool_size=2)
    mgr = SessionManager(store)

    session = mgr.get_or_create("test:mgr_pg")
    from nanobot.session.manager import Session

    assert isinstance(session, Session)
    assert session.key == "test:mgr_pg"

    session.add_message("user", "hello pg")
    mgr.save(session)

    mgr.invalidate("test:mgr_pg")
    loaded = mgr.get_or_create("test:mgr_pg")
    assert len(loaded.messages) == 1

    # Cleanup
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM session_messages WHERE session_key = %s", ("test:mgr_pg",))
        conn.execute("DELETE FROM sessions WHERE key = %s", ("test:mgr_pg",))
    store.close()
