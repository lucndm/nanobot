"""Tests for topic_configured detection and setup prompt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(tmp_path):
    with (
        patch("nanobot.agent.context.SqliteMemoryStore") as mock_store,
        patch("nanobot.agent.context.SkillsLoader") as mock_skills,
    ):
        mock_store.return_value.get_memory_context.return_value = ""
        mock_store.return_value.get_topic_memory_context.return_value = ""
        mock_skills.return_value.get_always_skills.return_value = []
        mock_skills.return_value.build_skills_summary.return_value = ""
        from nanobot.agent.context import ContextBuilder
        return ContextBuilder(workspace=tmp_path)


class TestBuildSystemPromptWithTopicConfigured:
    def test_unconfigured_topic_prompts_setup(self, tmp_path):
        """When topic is resolved but not configured, system prompt asks user to set it up."""
        ctx = _make_context(tmp_path)
        prompt = ctx.build_system_prompt(
            skill_names=[],
            user_mood="neutral",
            topic_name="finance",
            topic_resolved=True,
            topic_configured=False,
        )
        # Should contain setup prompt for unconfigured topic
        assert "topic" in prompt.lower()
        # Should not load rules (no TOPIC.md)
        assert "TOPIC.md" in prompt or "set up" in prompt.lower() or "workspace" in prompt.lower()

    def test_configured_topic_injects_rules(self, tmp_path):
        """When topic is configured, system prompt injects TOPIC.md rules."""
        topic_dir = tmp_path / "topics" / "finance"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: finance\n\n## purpose\nBudget tracking\n\n## rules\nBe precise.\n"
        )
        ctx = _make_context(tmp_path)
        prompt = ctx.build_system_prompt(
            skill_names=[],
            user_mood="neutral",
            topic_name="finance",
            topic_resolved=True,
            topic_configured=True,
        )
        assert "finance" in prompt.lower()
        assert "Budget tracking" in prompt

    def test_no_topic_name_skips_check(self, tmp_path):
        """When no topic_name, no setup prompt needed."""
        ctx = _make_context(tmp_path)
        prompt = ctx.build_system_prompt(
            skill_names=[],
            user_mood="neutral",
            topic_name=None,
            topic_resolved=False,
            topic_configured=True,
        )
        # Should not crash or contain topic setup prompt
        assert "# Topic Rules" not in prompt

    def test_topic_resolved_false_uses_ask_prompt(self, tmp_path):
        """When topic_resolved is False, the 'ask user' prompt is used."""
        ctx = _make_context(tmp_path)
        prompt = ctx.build_system_prompt(
            skill_names=[],
            user_mood="neutral",
            topic_name=None,
            topic_resolved=False,
            topic_configured=False,
        )
        assert "Which topic" in prompt
