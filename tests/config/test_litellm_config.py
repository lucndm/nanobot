"""Tests for LiteLLMConfig schema."""

import os

import pytest

from nanobot.config.schema import AgentDefaults, Config, LiteLLMConfig, LiteLLMModelConfig


def test_litellm_config_defaults():
    cfg = LiteLLMConfig()
    assert cfg.mode == "proxy"
    assert cfg.api_base is None
    assert cfg.api_key is None
    assert cfg.models == []
    assert cfg.fallbacks == []


def test_litellm_config_proxy_mode():
    cfg = LiteLLMConfig(api_base="http://unraid:4000", api_key="sk-test")
    assert cfg.mode == "proxy"
    assert cfg.api_base == "http://unraid:4000"


def test_litellm_config_direct_mode():
    models = [
        LiteLLMModelConfig(
            model_name="gpt-4o",
            litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test"},
        )
    ]
    cfg = LiteLLMConfig(models=models)
    assert cfg.mode == "direct"


def test_litellm_config_camel_case():
    """Config accepts camelCase keys from JSON."""
    cfg = Config.model_validate(
        {
            "litellm": {
                "apiBase": "http://localhost:4000",
                "apiKey": "sk-test",
                "models": [
                    {
                        "modelName": "gpt-4o",
                        "litellmParams": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                    }
                ],
            }
        }
    )
    assert cfg.litellm.api_base == "http://localhost:4000"
    assert len(cfg.litellm.models) == 1
    assert cfg.litellm.models[0].model_name == "gpt-4o"


def test_litellm_config_env_var_resolution():
    """${ENV_VAR} in api_key resolves to os.environ."""
    os.environ["TEST_LITELLM_KEY"] = "resolved-key"
    try:
        cfg = LiteLLMConfig(api_key="${TEST_LITELLM_KEY}")
        assert cfg.api_key == "resolved-key"
    finally:
        del os.environ["TEST_LITELLM_KEY"]


def test_config_has_litellm_field():
    """Root Config has litellm field with sensible defaults."""
    cfg = Config()
    assert isinstance(cfg.litellm, LiteLLMConfig)


def test_agent_defaults_no_fallback():
    """AgentDefaults still works without fallback_model/fallback_provider."""
    cfg = Config()
    assert cfg.agents.defaults.model is not None
