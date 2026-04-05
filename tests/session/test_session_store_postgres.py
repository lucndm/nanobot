"""Tests for PostgresSessionStore with append-only turn_log schema."""

from __future__ import annotations

import os
from datetime import datetime, timezone

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
    # Clean slate
    with s._pool.connection() as conn:
        conn.execute("DELETE FROM turn_log")
        conn.execute("DELETE FROM turn_summaries")
        conn.execute("DELETE FROM system_prompts")
        conn.commit()
    yield s
    with s._pool.connection() as conn:
        conn.execute("DELETE FROM turn_log")
        conn.execute("DELETE FROM turn_summaries")
        conn.execute("DELETE FROM system_prompts")
        conn.commit()
    s.close()


def test_get_or_create_returns_empty(store):
    data = store.get_or_create("test:new_session")
    assert data["key"] == "test:new_session"
    assert data["messages"] == []
    assert data["last_consolidated"] == 0


def test_save_appends_new_messages(store):
    """save() must INSERT new messages, not delete-all + re-insert."""
    key = "test:append"
    data = {
        "key": key,
        "messages": [
            {"role": "user", "content": "hello", "timestamp": datetime.now(timezone.utc).isoformat()},
            {"role": "assistant", "content": "hi there",
             "timestamp": datetime.now(timezone.utc).isoformat(),
             "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5,
             "stop_reason": "stop"},
        ],
    }
    store.save(data)

    loaded = store.get_or_create(key)
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][1]["model"] == "gpt-4o"


def test_save_incremental_append(store):
    """Second save() must only append new messages, not rewrite all."""
    key = "test:incremental"
    ts1 = datetime.now(timezone.utc).isoformat()
    data1 = {
        "key": key,
        "messages": [
            {"role": "user", "content": "first", "timestamp": ts1},
        ],
    }
    store.save(data1)

    ts2 = datetime.now(timezone.utc).isoformat()
    data2 = {
        "key": key,
        "messages": [
            {"role": "user", "content": "first", "timestamp": ts1},
            {"role": "assistant", "content": "reply", "timestamp": ts2,
             "model": "gpt-4o", "prompt_tokens": 8, "completion_tokens": 3},
        ],
    }
    store.save(data2)

    loaded = store.get_or_create(key)
    assert len(loaded["messages"]) == 2

    # Verify only 2 rows in DB (not 3 from delete+reinsert)
    with store._pool.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM turn_log WHERE session_key = %s", (key,)
        ).fetchone()[0]
    assert count == 2


def test_consolidate_stores_summary(store):
    """consolidate() must upsert summary without deleting turn_log rows."""
    key = "test:consolidate"
    data = {
        "key": key,
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "model": "gpt-4o",
             "prompt_tokens": 5, "completion_tokens": 2},
        ],
    }
    store.save(data)

    store.consolidate(key, "", "Summary of conversation so far", 2)

    summary = store.get_summary(key, "")
    assert summary is not None
    assert summary["last_seq"] == 2

    # turn_log rows still exist
    loaded = store.get_or_create(key)
    assert len(loaded["messages"]) == 2


def test_get_usage_returns_token_sums(store):
    """get_usage() must return aggregated token counts."""
    key = "test:usage"
    data = {
        "key": key,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "model": "gpt-4o",
             "prompt_tokens": 100, "completion_tokens": 50},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "model": "gpt-4o",
             "prompt_tokens": 200, "completion_tokens": 80},
        ],
    }
    store.save(data)

    usage = store.get_usage(session_key=key)
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 130


def test_system_prompt_hash_stored(store):
    """system_prompt_hash must be stored on messages that have it."""
    key = "test:sysprompt"
    data = {
        "key": key,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "system_prompt_hash": "abc123",
             "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5},
        ],
    }
    store.save(data)
    loaded = store.get_or_create(key)
    assert loaded["messages"][1]["system_prompt_hash"] == "abc123"


def test_list_sessions(store):
    """list_sessions must return all sessions with timestamps."""
    data1 = {"key": "test:list_a", "messages": [{"role": "user", "content": "a"}]}
    data2 = {"key": "test:list_b", "messages": [{"role": "user", "content": "b"}]}
    store.save(data1)
    store.save(data2)

    sessions = store.list_sessions()
    keys = [s["key"] for s in sessions]
    assert "test:list_a" in keys
    assert "test:list_b" in keys


def test_invalidate_is_noop(store):
    """Postgres has no in-memory cache."""
    store.invalidate("test:whatever")


def test_save_tool_message_with_call_id(store):
    """Tool messages with tool_call_id must be persisted correctly."""
    key = "test:tool_msg"
    data = {
        "key": key,
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "tc_1", "name": "exec", "content": "result"},
        ],
    }
    store.save(data)

    loaded = store.get_or_create(key)
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][1]["tool_call_id"] == "tc_1"
    assert loaded["messages"][1]["tool_name"] == "exec"


def test_sanitize_null_bytes(store):
    """Null bytes in content must be stripped for PostgreSQL JSONB."""
    key = "test:null_bytes"
    data = {
        "key": key,
        "messages": [
            {"role": "tool", "tool_call_id": "tc_1", "name": "exec",
             "content": "output with \u0000 null byte"},
        ],
    }
    store.save(data)  # Must not raise

    loaded = store.get_or_create(key)
    assert "\u0000" not in loaded["messages"][0]["content"]


def test_session_manager_with_postgres_store():
    """SessionManager works with PostgresSessionStore via protocol."""
    from nanobot.session.manager import Session, SessionManager
    from nanobot.session.store_postgres import PostgresSessionStore

    store = PostgresSessionStore(pg_url, pool_size=2)
    # Clean slate
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM turn_log")
        conn.execute("DELETE FROM turn_summaries")
        conn.execute("DELETE FROM system_prompts")
        conn.commit()

    mgr = SessionManager(store)
    session = mgr.get_or_create("test:mgr_pg")
    assert isinstance(session, Session)
    assert session.key == "test:mgr_pg"

    session.add_message("user", "hello pg")
    mgr.save(session)

    mgr.invalidate("test:mgr_pg")
    loaded = mgr.get_or_create("test:mgr_pg")
    assert len(loaded.messages) == 1

    # Cleanup
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM turn_log WHERE session_key = %s", ("test:mgr_pg",))
        conn.commit()
    store.close()
