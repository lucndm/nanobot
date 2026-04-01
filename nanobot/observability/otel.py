"""OpenTelemetry SDK setup and global accessor functions."""

from __future__ import annotations

from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_meter: metrics.Meter | None = None
_tracer: trace.Tracer | None = None
_meter_provider: MeterProvider | None = None
_tracer_provider: TracerProvider | None = None


def setup_otel(endpoint: str, service_name: str) -> None:
    """Initialize OTEL MeterProvider and TracerProvider with OTLP gRPC exporter.

    Idempotent — calling again after the first time is a no-op.
    """
    global _meter, _tracer, _meter_provider, _tracer_provider

    if _meter_provider is not None:
        return

    try:
        metric_exporter = OTLPMetricExporter(endpoint=endpoint)
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        _meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(_meter_provider)
        _meter = metrics.get_meter(service_name)

        span_exporter = OTLPSpanExporter(endpoint=endpoint)
        _tracer_provider = TracerProvider()
        _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(_tracer_provider)
        _tracer = trace.get_tracer(service_name)

        logger.info("OTEL initialized: endpoint={}, service={}", endpoint, service_name)
    except Exception:
        logger.exception("Failed to initialize OTEL")
        _meter_provider = _tracer_provider = None
        _meter = _tracer = None


def get_meter() -> metrics.Meter | None:
    return _meter


def get_tracer() -> trace.Tracer | None:
    return _tracer


def shutdown_otel() -> None:
    """Gracefully flush and shut down OTEL providers."""
    global _meter, _tracer, _meter_provider, _tracer_provider

    for provider in (_meter_provider, _tracer_provider):
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                logger.warning("Error shutting down OTEL provider")

    _meter = _tracer = _meter_provider = _tracer_provider = None


def _reset_otel_state() -> None:
    """Reset OTEL state to initial values. For testing purposes only."""
    global _meter, _tracer, _meter_provider, _tracer_provider
    _meter = _tracer = _meter_provider = _tracer_provider = None
