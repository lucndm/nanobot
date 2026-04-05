"""Tests for ContextBuilder.build_messages return shape."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_build_messages_returns_system_prompt_hash():
    """build_messages must return dict with 'messages' list and '_system_prompt_hash'."""
    with (
        patch("nanobot.agent.store.MemoryStore") as mock_store,
        patch("nanobot.agent.context.SkillsLoader") as mock_skills,
    ):
        mock_store.return_value.get_memory_context.return_value = ""
        mock_skills.return_value.get_always_skills.return_value = []
        mock_skills.return_value.build_skills_summary.return_value = ""

        from nanobot.agent.context import ContextBuilder

        ctx = ContextBuilder(workspace=Path("/tmp/test"))
        result = ctx.build_messages(
            history=[],
            current_message="hello",
        )

    assert isinstance(result, dict)
    assert "messages" in result
    assert "_system_prompt_hash" in result
    assert isinstance(result["messages"], list)
    assert len(result["_system_prompt_hash"]) == 64  # SHA-256 hex digest


@pytest.mark.asyncio
async def test_build_messages_system_prompt_hash_deterministic():
    """Same system prompt must produce the same hash."""
    with (
        patch("nanobot.agent.store.MemoryStore") as mock_store,
        patch("nanobot.agent.context.SkillsLoader") as mock_skills,
    ):
        mock_store.return_value.get_memory_context.return_value = ""
        mock_skills.return_value.get_always_skills.return_value = []
        mock_skills.return_value.build_skills_summary.return_value = ""

        from nanobot.agent.context import ContextBuilder

        ctx = ContextBuilder(workspace=Path("/tmp/test"))
        r1 = ctx.build_messages(history=[], current_message="hello")
        r2 = ctx.build_messages(history=[], current_message="world")

    # Same system prompt -> same hash, different user messages
    assert r1["_system_prompt_hash"] == r2["_system_prompt_hash"]
    assert r1["messages"][-1]["content"] != r2["messages"][-1]["content"]
