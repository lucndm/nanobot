"""SetupTopicTool — create a new topic's TOPIC.md and persist its mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.store import MemoryStoreProtocol
from nanobot.agent.tools.base import Tool


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
    ):
        self._workspace = workspace
        self._topic_store = topic_store
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
            },
            "required": ["purpose"],
        }

    async def execute(self, purpose: str) -> str:
        if not self._topic_name:
            return "Error: No topic context set. Cannot set up topic without knowing its name."

        if self._chat_id is None or self._thread_id is None:
            return (
                f"Error: Missing context (chat_id={self._chat_id}, "
                f"thread_id={self._thread_id}). Cannot set up topic."
            )

        key = self._topic_name.lower().replace(" ", "-").replace("_", "-").strip("-")
        topic_dir = self._workspace / "topics" / key
        topic_dir.mkdir(parents=True, exist_ok=True)
        topic_file = topic_dir / "TOPIC.md"

        content = f"# Topic: {self._topic_name}\n\n## purpose\n{purpose}\n"
        topic_file.write_text(content, encoding="utf-8")
        logger.info("setup_topic: created TOPIC.md for '{}' at {}", self._topic_name, topic_file)

        # Persist mapping
        self._topic_names[self._thread_id] = self._topic_name
        self._topic_store.set_topic_mapping(self._chat_id, self._thread_id, self._topic_name)
        logger.info(
            "setup_topic: persisted mapping thread_id={} -> '{}'",
            self._thread_id,
            self._topic_name,
        )

        return (
            f"Topic '{self._topic_name}' set up successfully. "
            f"TOPIC.md created at {topic_file} and mapping persisted."
        )
