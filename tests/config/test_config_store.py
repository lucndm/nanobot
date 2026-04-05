"""Tests for ConfigStore."""
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_config_store():
    """ConfigStore with mocked pool."""
    with patch("nanobot.config.store.ConnectionPool") as MockPool:
        mock_conn = MagicMock()
        MockPool.return_value.connection.return_value.__enter__ = lambda s: mock_conn
        MockPool.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

        from nanobot.config.store import ConfigStore
        store = ConfigStore("postgresql://test:test@localhost/test")
        return store, mock_conn


def test_resolve_dsn_from_env():
    with patch.dict("os.environ", {"NANOBOT_DATABASE_URL": "postgresql://x@host/db"}):
        from nanobot.config.store import _resolve_dsn
        assert _resolve_dsn() == "postgresql://x@host/db"


def test_resolve_dsn_missing_raises():
    with patch.dict("os.environ", {"NANOBOT_DATABASE_URL": ""}, clear=False):
        with patch.object(Path, "exists", return_value=False):
            from nanobot.config.store import _resolve_dsn
            with pytest.raises(ValueError, match="NANOBOT_DATABASE_URL"):
                _resolve_dsn()


def test_load_config_from_db(mock_config_store):
    store, mock_conn = mock_config_store
    mock_conn.execute.return_value.fetchone.return_value = (
        {"agents": [], "channels": {"enabled": False}},
    )
    config = store.load_config()
    assert config is not None


def test_save_config(mock_config_store):
    store, mock_conn = mock_config_store
    mock_config = MagicMock()
    mock_config.model_dump.return_value = {"agents": []}
    store.save_config(mock_config)
    mock_conn.execute.assert_called()
    mock_conn.commit.assert_called()
