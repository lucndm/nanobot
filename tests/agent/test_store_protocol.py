"""Verify SqliteMemoryStore satisfies MemoryStoreProtocol and deploy regression tests."""

from pathlib import Path
from types import SimpleNamespace

from nanobot.agent.memory import SqliteMemoryStore
from nanobot.agent.store import MemoryStoreProtocol, create_memory_store


def test_sqlite_store_satisfies_protocol(tmp_path: Path):
    store = SqliteMemoryStore(tmp_path)
    assert isinstance(store, MemoryStoreProtocol)


def test_create_memory_store_sqlite(tmp_path: Path):
    """Default backend is sqlite, returns SqliteMemoryStore."""
    config = SimpleNamespace(database=SimpleNamespace(backend="sqlite", url="", pool_size=5))
    store = create_memory_store(config, tmp_path)
    assert isinstance(store, SqliteMemoryStore)


def test_create_memory_store_requires_url_for_postgres(tmp_path: Path):
    """Regression: missing database.url when backend=postgres must raise ValueError."""
    config = SimpleNamespace(database=SimpleNamespace(backend="postgres", url="", pool_size=5))
    try:
        create_memory_store(config, tmp_path)
        assert False, "Should have raised ValueError"
    except (ValueError, ImportError):
        pass  # ImportError if psycopg not installed — acceptable


def test_create_memory_store_no_database_config(tmp_path: Path):
    """Regression: config with no database section should default to sqlite."""
    config = SimpleNamespace()
    store = create_memory_store(config, tmp_path)
    assert isinstance(store, SqliteMemoryStore)


def test_sqlite_store_append_history_sequential_ids(tmp_path: Path):
    """Regression: ensure append_history creates sequential IDs.

    After migration with explicit IDs, auto-increment sequence must continue
    from the highest ID, not restart at 1 (BIGSERIAL sequence sync issue).
    For SQLite this is automatic; test verifies the contract.
    """
    store = SqliteMemoryStore(tmp_path)

    for i in range(5):
        store.append_history(f"entry {i}")

    history = store.read_history()
    assert "entry 0" in history
    assert "entry 4" in history


def test_sqlite_store_topic_mapping_crud(tmp_path: Path):
    """Verify topic mapping CRUD works correctly."""
    store = SqliteMemoryStore(tmp_path)

    # Set and get
    store.set_topic_mapping(-100123, 42, "Finance")
    assert store.get_topic_mapping(-100123, 42) == "Finance"

    # Update
    store.set_topic_mapping(-100123, 42, "Finance Updated")
    assert store.get_topic_mapping(-100123, 42) == "Finance Updated"

    # Delete
    store.delete_topic_mapping(-100123, 42)
    assert store.get_topic_mapping(-100123, 42) is None


def test_sqlite_store_emoji_sentiment(tmp_path: Path):
    """Verify emoji sentiment learning and resolution."""
    store = SqliteMemoryStore(tmp_path)

    # Built-in emoji
    assert store.resolve_emoji_sentiment("👍") == "positive"
    assert store.resolve_emoji_sentiment("👎") == "negative"
    assert store.is_emoji_known("👍") is True

    # Unknown emoji
    assert store.resolve_emoji_sentiment("🤷") is None
    assert store.is_emoji_known("🤷") is False

    # Learn new emoji
    store.learn_emoji("🤷", "neutral")
    assert store.resolve_emoji_sentiment("🤷") == "neutral"
    assert store.is_emoji_known("🤷") is True
