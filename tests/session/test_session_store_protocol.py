"""Tests for SessionStoreProtocol compliance and deploy regression cases."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.session.store import SessionStoreProtocol, create_session_store
from nanobot.session.store_jsonl import JsonlSessionStore


def test_jsonl_store_satisfies_protocol():
    """JsonlSessionStore implements SessionStoreProtocol."""
    assert isinstance(JsonlSessionStore, SessionStoreProtocol)


def test_jsonl_store_get_or_create(tmp_path: Path):
    store = JsonlSessionStore(tmp_path)
    data = store.get_or_create("test:proto")
    assert data["key"] == "test:proto"
    assert data["messages"] == []


def test_jsonl_store_save_and_reload(tmp_path: Path):
    store = JsonlSessionStore(tmp_path)
    data = store.get_or_create("test:save")
    data["messages"] = [{"role": "user", "content": "hi"}]
    store.save(data)

    # New store instance to verify persistence
    store2 = JsonlSessionStore(tmp_path)
    loaded = store2.get_or_create("test:save")
    assert len(loaded["messages"]) == 1
    assert loaded["messages"][0]["content"] == "hi"


def test_session_manager_backward_compat(tmp_path: Path):
    """SessionManager still works with old Path(workspace) signature."""
    from nanobot.session.manager import SessionManager

    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("test:compat")
    session.add_message("user", "hello")
    mgr.save(session)

    # Reload
    mgr2 = SessionManager(tmp_path)
    loaded = mgr2.get_or_create("test:compat")
    assert len(loaded.messages) == 1


def test_session_manager_with_store_instance(tmp_path: Path):
    """SessionManager works with explicit store instance."""
    from nanobot.session.manager import SessionManager

    store = JsonlSessionStore(tmp_path)
    mgr = SessionManager(store)
    session = mgr.get_or_create("test:store_arg")
    session.add_message("user", "via store")
    mgr.save(session)

    loaded = mgr.get_or_create("test:store_arg")
    assert loaded.messages[0]["content"] == "via store"


def test_create_session_store_sqlite(tmp_path: Path):
    """create_session_store returns JsonlSessionStore for sqlite backend."""
    config = SimpleNamespace(database=SimpleNamespace(backend="sqlite", url="", pool_size=5))
    store = create_session_store(config, tmp_path)
    assert isinstance(store, JsonlSessionStore)


# ── Deploy regression tests ────────────────────────────────────────────


def test_null_bytes_in_messages_sanitized(tmp_path: Path):
    """Regression: PostgreSQL JSONB rejects \\u0000.

    Messages containing null bytes (from exec tool output, binary data)
    must be sanitized before serialization.
    """
    try:
        from nanobot.session.store_postgres import PostgresSessionStore
    except ImportError:
        pytest.skip("psycopg not installed")

    # Verify the sanitization method strips null bytes recursively
    assert PostgresSessionStore._sanitize_for_pg("hello\u0000world") == "helloworld"
    assert PostgresSessionStore._sanitize_for_pg({"content": "a\u0000b"}) == {"content": "ab"}
    assert PostgresSessionStore._sanitize_for_pg([{"x": "1\u00002"}]) == [{"x": "12"}]
    # Non-string types pass through
    assert PostgresSessionStore._sanitize_for_pg(42) == 42
    assert PostgresSessionStore._sanitize_for_pg(None) is None
    # Nested structures
    nested = {"messages": [{"role": "user", "content": "run\ncode\u0000output"}]}
    result = PostgresSessionStore._sanitize_for_pg(nested)
    assert result["messages"][0]["content"] == "run\ncodeoutput"


def test_create_session_store_requires_url_for_postgres(tmp_path: Path):
    """Regression: missing database.url when backend=postgres must raise ValueError.

    Deploy failed silently when config had backend=postgres but no URL.
    """
    config = SimpleNamespace(database=SimpleNamespace(backend="postgres", url="", pool_size=5))

    try:
        create_session_store(config, tmp_path)
        assert False, "Should have raised ValueError"
    except (ValueError, ImportError):
        pass  # ImportError if psycopg not installed — acceptable


def test_jsonl_store_handles_corrupted_file(tmp_path: Path):
    """Regression: corrupted JSONL file should not crash, just skip the session."""
    store = JsonlSessionStore(tmp_path)
    data = store.get_or_create("test:good")
    data["messages"] = [{"role": "user", "content": "ok"}]
    store.save(data)

    # Corrupt a file by writing invalid JSON
    sessions_dir = tmp_path / "sessions"
    bad_path = sessions_dir / "test_bad.jsonl"
    bad_path.write_text("NOT VALID JSON {{{\n")

    # Loading corrupted session returns None (graceful fallback)
    result = store.get_or_create("test_bad")
    assert result["key"] == "test_bad"
    assert result["messages"] == []  # Fresh session, corrupted data ignored

    # Good session still works
    good = store.get_or_create("test_good")
    assert good["messages"][0]["content"] == "ok"


def test_session_manager_handles_empty_metadata(tmp_path: Path):
    """Regression: sessions with missing metadata fields should load correctly."""
    from nanobot.session.manager import SessionManager

    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("test:empty_meta")
    session.add_message("user", "hello")
    session.metadata = {}
    session.last_consolidated = 0
    mgr.save(session)

    # Reload via fresh manager
    mgr2 = SessionManager(tmp_path)
    loaded = mgr2.get_or_create("test:empty_meta")
    assert loaded.messages[0]["content"] == "hello"
    assert loaded.metadata == {}
    assert loaded.last_consolidated == 0


def test_large_message_list_save_and_load(tmp_path: Path):
    """Verify session with many messages persists and loads correctly."""
    from nanobot.session.manager import SessionManager

    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("test:large")

    for i in range(100):
        session.add_message("user", f"message {i}" * 50)  # ~400 chars each

    mgr.save(session)

    mgr2 = SessionManager(tmp_path)
    loaded = mgr2.get_or_create("test:large")
    assert len(loaded.messages) == 100
    assert loaded.messages[0]["content"].startswith("message 0")
    assert loaded.messages[99]["content"].startswith("message 99")
