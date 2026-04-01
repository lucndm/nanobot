"""Tests for OTEL configuration."""

from nanobot.config.schema import Config, OtelConfig


def test_otel_config_defaults():
    cfg = OtelConfig()
    assert cfg.enabled is False
    assert cfg.endpoint == "http://100.68.251.84:4317"
    assert cfg.service_name == "nanobot"


def test_otel_config_from_dict():
    cfg = OtelConfig.model_validate(
        {"enabled": True, "endpoint": "http://localhost:4317", "service_name": "test"}
    )
    assert cfg.enabled is True
    assert cfg.endpoint == "http://localhost:4317"
    assert cfg.service_name == "test"


def test_root_config_has_otel():
    cfg = Config()
    assert hasattr(cfg, "otel")
    assert isinstance(cfg.otel, OtelConfig)
    assert cfg.otel.enabled is False
