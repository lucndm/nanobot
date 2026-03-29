"""RAG search tool for querying vector memory."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class RagSearchTool(Tool):
    """Search RAG memory for relevant facts."""

    name = "rag_search"
    description = "Search RAG memory vector store for relevant facts from past conversations. Returns matched memories with relevance scores."

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - what fact or information to find from memory",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        """Execute RAG search."""
        # Access rag via agent_loop reference set by Tool base class
        if not hasattr(self, "_agent_loop") or self._agent_loop is None:
            return "RAG memory is not available (no agent_loop)"
        if not hasattr(self._agent_loop, "rag") or self._agent_loop.rag is None:
            return "RAG memory is not enabled"

        try:
            results = await self.agent_loop.rag.retrieve(query, top_k=top_k)
            if not results:
                return "No relevant memories found for this query."

            formatted = []
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                text = r.get("text", "")
                formatted.append(f"[Score: {score:.2f}] {text}")

            return "\n".join(formatted)

        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return f"RAG search failed: {str(e)}"
