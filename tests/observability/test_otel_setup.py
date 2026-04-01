"""Tests for OTEL setup and shutdown."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_otel_state():
    """Reset OTEL state before each test."""
    from nanobot.observability.otel import _reset_otel_state

    _reset_otel_state()
    yield


def test_setup_otel_creates_providers():
    from nanobot.observability.otel import setup_otel, get_meter, get_tracer

    with (
        patch("nanobot.observability.otel.OTLPMetricExporter") as mock_metric_exp,
        patch("nanobot.observability.otel.OTLPSpanExporter") as mock_span_exp,
        patch("nanobot.observability.otel.MeterProvider") as mock_meter_prov,
        patch("nanobot.observability.otel.TracerProvider") as mock_tracer_prov,
    ):
        setup_otel("http://localhost:4317", "test-service")

        mock_metric_exp.assert_called_once_with(endpoint="http://localhost:4317")
        mock_span_exp.assert_called_once_with(endpoint="http://localhost:4317")
        mock_meter_prov.assert_called_once()
        mock_tracer_prov.assert_called_once()

    meter = get_meter()
    assert meter is not None

    tracer = get_tracer()
    assert tracer is not None


def test_setup_otel_is_idempotent():
    """Calling setup_otel twice should not create duplicate providers."""
    from nanobot.observability.otel import setup_otel

    with (
        patch("nanobot.observability.otel.MeterProvider") as mock_mp,
        patch("nanobot.observability.otel.TracerProvider") as mock_tp,
        patch("nanobot.observability.otel.OTLPMetricExporter"),
        patch("nanobot.observability.otel.OTLPSpanExporter"),
    ):
        setup_otel("http://localhost:4317", "test")
        setup_otel("http://localhost:4317", "test")

        assert mock_mp.call_count == 1
        assert mock_tp.call_count == 1


def test_shutdown_otel():
    from nanobot.observability.otel import setup_otel, shutdown_otel

    with (
        patch("nanobot.observability.otel.OTLPMetricExporter"),
        patch("nanobot.observability.otel.OTLPSpanExporter"),
        patch("nanobot.observability.otel.MeterProvider") as mock_mp,
        patch("nanobot.observability.otel.TracerProvider") as mock_tp,
    ):
        mock_meter_prov = MagicMock()
        mock_tracer_prov = MagicMock()
        mock_mp.return_value = mock_meter_prov
        mock_tp.return_value = mock_tracer_prov

        setup_otel("http://localhost:4317", "test")
        shutdown_otel()

        mock_meter_prov.shutdown.assert_called_once()
        mock_tracer_prov.shutdown.assert_called_once()
