"""Integration test: full flow from message -> store -> verify metadata."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pg_url = os.environ.get("NANOBOT_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not pg_url,
    reason="Set NANOBOT_PG_URL to run integration tests",
)


def test_full_turn_lifecycle():
    """Verify: save messages with metadata -> get_or_create -> get_usage -> consolidate."""
    from nanobot.session.store import SessionStore

    store = SessionStore(pg_url, pool_size=2)
    key = "test:lifecycle"

    # Clean
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM turn_log WHERE session_key = %s", (key,))
        conn.execute("DELETE FROM turn_summaries WHERE session_key = %s", (key,))
        conn.commit()

    # Save turn 1 — topic_id is the stable thread_id (not the display name)
    store.save({
        "key": key,
        "messages": [
            {"role": "user", "content": "hello", "topic_id": "42",
             "timestamp": datetime.now(timezone.utc).isoformat()},
            {"role": "assistant", "content": "hi", "model": "gpt-4o",
             "system_prompt_hash": "abc", "prompt_tokens": 50, "completion_tokens": 10,
             "stop_reason": "stop", "topic_id": "42"},
        ],
    })

    # Verify load
    loaded = store.get_or_create(key)
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][1]["model"] == "gpt-4o"
    assert loaded["messages"][1]["topic_id"] == "42"

    # Verify usage
    usage = store.get_usage(session_key=key)
    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 10

    # Consolidate by topic_id
    store.consolidate(key, "42", "User greeted bot", 2)

    # Verify summary
    summary = store.get_summary(key, "42")
    assert summary["last_seq"] == 2
    assert "greeted" in summary["summary"]

    # turn_log rows still intact
    loaded2 = store.get_or_create(key)
    assert len(loaded2["messages"]) == 2

    store.close()
