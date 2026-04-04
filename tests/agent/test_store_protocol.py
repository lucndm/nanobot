"""Verify SqliteMemoryStore satisfies MemoryStoreProtocol."""

from pathlib import Path

from nanobot.agent.memory import SqliteMemoryStore
from nanobot.agent.store import MemoryStoreProtocol


def test_sqlite_store_satisfies_protocol(tmp_path: Path):
    store = SqliteMemoryStore(tmp_path)
    assert isinstance(store, MemoryStoreProtocol)
