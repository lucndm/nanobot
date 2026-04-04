"""Test _make_provider creates LiteLLMProvider with correct settings."""

from nanobot.config.schema import Config


def _config_with_model(model="gpt-4o-mini"):
    data = {
        "agents": {
            "defaults": {
                "model": model,
            },
            "sessions": {},
        },
        "channels": {},
        "gateway": {},
    }
    return Config(**data)


def test_make_provider_creates_litellm_provider():
    from nanobot.cli.commands import _make_provider
    from nanobot.providers.litellm_provider import LiteLLMProvider

    config = _config_with_model()
    provider = _make_provider(config)

    assert isinstance(provider, LiteLLMProvider)


def test_make_provider_sets_generation_settings():
    from nanobot.cli.commands import _make_provider

    config = _config_with_model()
    config.agents.defaults.temperature = 0.7
    config.agents.defaults.max_tokens = 4096

    provider = _make_provider(config)

    assert provider.generation.temperature == 0.7
    assert provider.generation.max_tokens == 4096
