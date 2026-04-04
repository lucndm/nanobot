"""LLM provider interface and implementations."""

from nanobot.providers.base import LLMProvider, LLMResponse

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LiteLLMProvider",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.litellm_provider import LiteLLMProvider

_LAZY_IMPORTS = {
    "LiteLLMProvider": "nanobot.providers.litellm_provider",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
