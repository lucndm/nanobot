"""Tests for topic config extraction from TOPIC.md via ContextBuilder."""

from __future__ import annotations

from unittest.mock import patch


def _make_context(tmp_path):
    """Create a ContextBuilder with mocked dependencies."""
    with (
        patch("nanobot.agent.store.MemoryStore") as mock_store,
        patch("nanobot.agent.context.SkillsLoader") as mock_skills,
    ):
        mock_store.return_value.get_memory_context.return_value = ""
        mock_skills.return_value.get_always_skills.return_value = []
        mock_skills.return_value.build_skills_summary.return_value = ""

        from nanobot.agent.context import ContextBuilder

        return ContextBuilder(workspace=tmp_path)


class TestLoadTopicRulesWithConfig:
    def test_returns_rules_text(self, tmp_path):
        # Folder structure: topics/<chat_id>/<thread_id>/TOPIC.md
        topic_dir = tmp_path / "topics" / "111" / "222"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: my-topic\n\n## purpose\nTest\n\n## litellm\nmodel: test/m\n"
        )
        ctx = _make_context(tmp_path)
        result = ctx.load_topic_rules("my-topic", chat_id=111, thread_id=222)
        assert result is not None
        assert "## purpose" in result

    def test_get_topic_config_returns_config(self, tmp_path):
        topic_dir = tmp_path / "topics" / "111" / "222"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: my-topic\n\n## purpose\nTest\n\n## litellm\nmodel: test/m\ntemperature: 0.5\nmax_tokens: 2048\n"
        )
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("my-topic", chat_id=111, thread_id=222)
        assert config is not None
        assert config.model == "test/m"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_get_topic_config_no_litellm_returns_none(self, tmp_path):
        topic_dir = tmp_path / "topics" / "111" / "222"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text("# Topic: plain\n\n## purpose\nNo config\n")
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("plain", chat_id=111, thread_id=222)
        assert config is None

    def test_get_topic_config_no_topic_file_returns_none(self, tmp_path):
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("nonexistent", chat_id=111, thread_id=222)
        assert config is None

    def test_load_topic_rules_returns_none_without_chat_id(self, tmp_path):
        ctx = _make_context(tmp_path)
        result = ctx.load_topic_rules("my-topic", chat_id=None, thread_id=None)
        assert result is None
