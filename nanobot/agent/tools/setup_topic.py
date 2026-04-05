"""SetupTopicTool — create a new topic's TOPIC.md and persist its mapping."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.store import MemoryStoreProtocol
from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class SetupTopicTool(Tool):
    """Tool to set up a new topic: creates TOPIC.md and persists the mapping.

    Use this when the user tells you what a topic is for. It creates the
    TOPIC.md file and records the chat_id/thread_id -> topic_name mapping
    so future messages in this topic are correctly routed.
    """

    def __init__(
        self,
        workspace: Path,
        topic_store: MemoryStoreProtocol,
        provider: LLMProvider,
    ):
        self._workspace = workspace
        self._topic_store = topic_store
        self._provider = provider
        # In-memory cache of thread_id -> topic_name, for quick lookup
        self._topic_names: dict[int, str] = {}
        # Current message context — set via set_context() before each agent turn
        self._chat_id: int | None = None
        self._thread_id: int | None = None
        self._topic_name: str | None = None

    def set_context(
        self,
        chat_id: int | None = None,
        thread_id: int | None = None,
        topic_name: str | None = None,
    ) -> None:
        """Set current message context from the agent loop."""
        self._chat_id = chat_id
        self._thread_id = thread_id
        self._topic_name = topic_name

    @property
    def name(self) -> str:
        return "setup_topic"

    @property
    def description(self) -> str:
        return (
            "Create a new topic's TOPIC.md file and persist its chat_id/thread_id "
            "mapping. Call this when the user tells you what a topic is for. "
            "This replaces the need to use write_file separately for topic setup."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "purpose": {
                    "type": "string",
                    "description": (
                        "What this topic is for — a brief description "
                        "that will be written to the ## purpose section."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "The LLM model to use for this topic "
                        "(e.g. minimax/MiniMax-M2.7, gpt-4o). "
                        "Required if not calling setup_topic for the first time."
                    ),
                },
            },
            "required": ["purpose"],
        }

    def _get_available_models(self) -> list[str]:
        """Get sorted list of available model names from the provider."""
        if hasattr(self._provider, "get_available_models"):
            return sorted(self._provider.get_available_models())
        # Fallback for providers without this method
        return []

    def _write_topic_file(self, purpose: str, model: str | None) -> None:
        """Write the TOPIC.md file."""
        key = self._topic_name.lower().replace(" ", "-").replace("_", "-").strip("-")
        topic_dir = self._workspace / "topics" / key
        topic_dir.mkdir(parents=True, exist_ok=True)
        topic_file = topic_dir / "TOPIC.md"

        model_line = f"model: {model}" if model else "model:"
        content = f"""# Topic: {self._topic_name}

## purpose
{purpose}

## litellm
{model_line}
temperature:
max_tokens:
"""
        topic_file.write_text(content, encoding="utf-8")
        logger.info("setup_topic: wrote TOPIC.md for '{}' at {}", self._topic_name, topic_file)

    async def execute(self, purpose: str, model: str | None = None) -> str:
        if not self._topic_name:
            return "Error: No topic context set. Cannot set up topic without knowing its name."

        if self._chat_id is None or self._thread_id is None:
            return (
                f"Error: Missing context (chat_id={self._chat_id}, "
                f"thread_id={self._thread_id}). Cannot set up topic."
            )

        available_models = self._get_available_models()

        # No model specified: list models and ask user to pick one
        if not model:
            # Persist purpose to topic_memory in PostgreSQL (source of truth)
            # Store just the purpose text — sync_topic_files will add the ## purpose header
            self._topic_store.write_topic_memory(self._topic_name, purpose)
            self._topic_store.set_topic_mapping(self._chat_id, self._thread_id, self._topic_name)
            models_str = (
                "\n".join(f"- {m}" for m in available_models)
                if available_models
                else "(none detected -- proxy may be unreachable)"
            )
            return (
                f"Topic '{self._topic_name}' purpose saved. "
                f"Available models:\n{models_str}\n\n"
                "Please tell me which model you want to use, then I'll finalize the TOPIC.md."
            )

        # Model specified: validate and write TOPIC.md
        if available_models and model not in available_models:
            return (
                f"Model '{model}' is not available on your proxy. "
                f"Available models:\n" + "\n".join(f"- {m}" for m in available_models)
            )

        self._write_topic_file(purpose, model)

        # Persist mapping and litellm config to PostgreSQL (source of truth)
        self._topic_names[self._thread_id] = self._topic_name
        self._topic_store.set_topic_mapping(self._chat_id, self._thread_id, self._topic_name)
        self._topic_store.set_topic_litellm(
            self._topic_name, model, temperature=None, max_tokens=None
        )
        logger.info(
            "setup_topic: persisted mapping thread_id={} -> '{}', litellm model='{}'",
            self._thread_id,
            self._topic_name,
            model,
        )

        return (
            f"Topic '{self._topic_name}' set up successfully with model={model}. TOPIC.md created."
        )
