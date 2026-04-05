"""Integration tests for per-topic TOPIC.md rules end-to-end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTopicEndToEnd:
    def test_topic_md_to_config_to_run_spec(self, tmp_path):
        """Full flow: TOPIC.md → parse → TopicConfig → AgentRunSpec overrides."""
        from nanobot.agent.context import ContextBuilder

        # Folder structure: topics/<chat_id>/<thread_id>/TOPIC.md
        topic_dir = tmp_path / "topics" / "123" / "456"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: finance\n\n"
            "## purpose\nQuan ly chi tieu\n\n"
            "## litellm\n"
            "model: anthropic/claude-3-5-haiku-20241007\n"
            "temperature: 0.7\n"
            "max_tokens: 4096\n\n"
            "## rules\nAlways respond in Vietnamese\n"
        )

        with (
            patch("nanobot.agent.store.MemoryStore") as mock_store,
            patch("nanobot.agent.context.SkillsLoader") as mock_skills,
        ):
            mock_store.return_value.get_memory_context.return_value = ""
            mock_skills.return_value.get_always_skills.return_value = []
            mock_skills.return_value.build_skills_summary.return_value = ""

            ctx = ContextBuilder(workspace=tmp_path)

        rules = ctx.load_topic_rules("finance", chat_id=123, thread_id=456)
        assert rules is not None
        assert "## purpose" in rules
        assert "## litellm" in rules

        config = ctx.get_topic_config("finance", chat_id=123, thread_id=456)
        assert config is not None
        assert config.model == "anthropic/claude-3-5-haiku-20241007"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_topic_without_litellm_uses_defaults(self, tmp_path):
        """Topic with no litellm section should return None config."""
        from nanobot.agent.context import ContextBuilder

        topic_dir = tmp_path / "topics" / "123" / "456"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: plain\n\n## purpose\nGeneral chat\n\n## rules\nBe helpful\n"
        )

        with (
            patch("nanobot.agent.store.MemoryStore") as mock_store,
            patch("nanobot.agent.context.SkillsLoader") as mock_skills,
        ):
            mock_store.return_value.get_memory_context.return_value = ""
            mock_skills.return_value.get_always_skills.return_value = []
            mock_skills.return_value.build_skills_summary.return_value = ""

            ctx = ContextBuilder(workspace=tmp_path)

        rules = ctx.load_topic_rules("plain", chat_id=123, thread_id=456)
        assert rules is not None
        config = ctx.get_topic_config("plain", chat_id=123, thread_id=456)
        assert config is None

    def test_cache_invalidation_after_update(self, tmp_path):
        """Updating TOPIC.md and invalidating cache should reflect new config."""
        from nanobot.agent.context import ContextBuilder

        topic_dir = tmp_path / "topics" / "123" / "456"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text("## litellm\nmodel: old/model\n")

        with (
            patch("nanobot.agent.store.MemoryStore") as mock_store,
            patch("nanobot.agent.context.SkillsLoader") as mock_skills,
        ):
            mock_store.return_value.get_memory_context.return_value = ""
            mock_skills.return_value.get_always_skills.return_value = []
            mock_skills.return_value.build_skills_summary.return_value = ""

            ctx = ContextBuilder(workspace=tmp_path)

        config1 = ctx.get_topic_config("dynamic", chat_id=123, thread_id=456)
        assert config1.model == "old/model"

        (topic_dir / "TOPIC.md").write_text("## litellm\nmodel: new/model\n")
        ctx.invalidate_topic_cache(chat_id=123, thread_id=456)

        config2 = ctx.get_topic_config("dynamic", chat_id=123, thread_id=456)
        assert config2.model == "new/model"
