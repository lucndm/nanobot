"""Test _make_provider creates FallbackProvider when fallback is configured."""

from unittest.mock import patch

from nanobot.config.schema import Config


def _config_with_fallback(fallback_model=None, fallback_provider=None):
    data = {
        "agents": {
            "defaults": {
                "model": "gpt-4o-mini",
                "provider": "custom",
            },
            "sessions": {},
        },
        "providers": {
            "custom": {"apiKey": "sk-test", "apiBase": "http://localhost:4000/v1"},
            "openrouter": {"apiKey": "sk-or-test"},
        },
        "channels": {},
        "gateway": {},
    }
    if fallback_model:
        data["agents"]["defaults"]["fallbackModel"] = fallback_model
    if fallback_provider:
        data["agents"]["defaults"]["fallbackProvider"] = fallback_provider
    return Config(**data)


def test_no_fallback_when_not_configured():
    from nanobot.cli.commands import _make_provider
    from nanobot.providers.fallback_provider import FallbackProvider

    config = _config_with_fallback()

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = _make_provider(config)

    assert not isinstance(provider, FallbackProvider)


def test_fallback_provider_created_when_configured():
    from nanobot.cli.commands import _make_provider
    from nanobot.providers.fallback_provider import FallbackProvider

    config = _config_with_fallback(
        fallback_model="openrouter/free",
        fallback_provider="openrouter",
    )

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = _make_provider(config)

    assert isinstance(provider, FallbackProvider)
    assert provider.fallback_model == "openrouter/free"
