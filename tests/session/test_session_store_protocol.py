"""Tests for SessionStoreProtocol compliance."""

from __future__ import annotations

from pathlib import Path

from nanobot.session.store_jsonl import JsonlSessionStore
from nanobot.session.store import SessionStoreProtocol


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
    from types import SimpleNamespace

    from nanobot.session.store import create_session_store

    config = SimpleNamespace(database=SimpleNamespace(backend="sqlite", url="", pool_size=5))
    store = create_session_store(config, tmp_path)
    assert isinstance(store, JsonlSessionStore)
