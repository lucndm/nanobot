"""Tests for session metrics (active sessions gauge)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_meter():
    mock_gauge = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_observable_gauge.return_value = mock_gauge
    with patch("nanobot.session.manager.get_meter", return_value=mock_meter):
        yield mock_meter, mock_gauge


def test_session_manager_creates_active_gauge(tmp_path, mock_meter):
    """SessionManager constructor creates an observable gauge for active sessions."""
    mock_meter, mock_gauge = mock_meter

    from nanobot.session.manager import SessionManager

    SessionManager(tmp_path)

    mock_meter.create_observable_gauge.assert_called_once()
    call_args = mock_meter.create_observable_gauge.call_args
    assert call_args[0][0] == "nanobot.session.active"
    assert "callbacks" in call_args[1]
    assert call_args[1]["description"] == "Number of active sessions in cache"


def test_active_sessions_gauge_observes_cache_size(tmp_path, mock_meter):
    """The gauge callback yields Observation(len(cache))."""
    from nanobot.session.manager import SessionManager

    mock_meter_obj, _ = mock_meter
    sm = SessionManager(tmp_path)

    # Retrieve the callback registered with the gauge
    call_args = mock_meter_obj.create_observable_gauge.call_args
    callback = call_args[1]["callbacks"][0]

    # Empty cache -> 0
    observations = list(callback(MagicMock()))
    assert len(observations) == 1
    assert observations[0].value == 0

    # Create sessions to populate cache
    sm.get_or_create("telegram:123")
    sm.get_or_create("discord:456")
    observations = list(callback(MagicMock()))
    assert len(observations) == 1
    assert observations[0].value == 2
