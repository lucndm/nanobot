"""Test fallback_model and fallback_provider config fields."""

from nanobot.config.schema import AgentDefaults


def test_fallback_defaults_are_none():
    defaults = AgentDefaults()
    assert defaults.fallback_model is None
    assert defaults.fallback_provider is None


def test_fallback_config_from_dict():
    defaults = AgentDefaults(
        model="glm-5-turbo",
        fallback_model="openrouter/free",
        fallback_provider="openrouter",
    )
    assert defaults.fallback_model == "openrouter/free"
    assert defaults.fallback_provider == "openrouter"


def test_fallback_config_camel_case():
    """Config JSON uses camelCase, Pydantic aliases handle it."""
    defaults = AgentDefaults(
        **{"fallbackModel": "openrouter/free", "fallbackProvider": "openrouter"}
    )
    assert defaults.fallback_model == "openrouter/free"
    assert defaults.fallback_provider == "openrouter"
