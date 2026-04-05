"""Tests for topic config extraction from TOPIC.md via ContextBuilder."""

from __future__ import annotations

from unittest.mock import patch


def _make_context(tmp_path):
    """Create a ContextBuilder with mocked dependencies."""
    with (
        patch("nanobot.agent.context.SqliteMemoryStore") as mock_store,
        patch("nanobot.agent.context.SkillsLoader") as mock_skills,
    ):
        mock_store.return_value.get_memory_context.return_value = ""
        mock_skills.return_value.get_always_skills.return_value = []
        mock_skills.return_value.build_skills_summary.return_value = ""

        from nanobot.agent.context import ContextBuilder

        return ContextBuilder(workspace=tmp_path)


class TestLoadTopicRulesWithConfig:
    def test_returns_rules_text(self, tmp_path):
        topic_dir = tmp_path / "topics" / "my-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: my-topic\n\n## purpose\nTest\n\n## litellm\nmodel: test/m\n"
        )
        ctx = _make_context(tmp_path)
        result = ctx.load_topic_rules("my-topic")
        assert result is not None
        assert "## purpose" in result

    def test_get_topic_config_returns_config(self, tmp_path):
        topic_dir = tmp_path / "topics" / "my-topic"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "# Topic: my-topic\n\n## purpose\nTest\n\n## litellm\nmodel: test/m\ntemperature: 0.5\nmax_tokens: 2048\n"
        )
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("my-topic")
        assert config is not None
        assert config.model == "test/m"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_get_topic_config_no_litellm_returns_none(self, tmp_path):
        topic_dir = tmp_path / "topics" / "plain"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text("# Topic: plain\n\n## purpose\nNo config\n")
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("plain")
        assert config is None

    def test_get_topic_config_no_topic_file_returns_none(self, tmp_path):
        ctx = _make_context(tmp_path)
        config = ctx.get_topic_config("nonexistent")
        assert config is None

    def test_get_topic_config_uses_cache(self, tmp_path):
        topic_dir = tmp_path / "topics" / "cached"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text(
            "## litellm\nmodel: test/m\n"
        )
        ctx = _make_context(tmp_path)
        config1 = ctx.get_topic_config("cached")
        # Delete file to prove second call uses cache
        (topic_dir / "TOPIC.md").unlink()
        config2 = ctx.get_topic_config("cached")
        assert config1 is not None
        assert config2 is not None
        assert config1.model == config2.model

    def test_invalidate_clears_config_cache(self, tmp_path):
        topic_dir = tmp_path / "topics" / "cached"
        topic_dir.mkdir(parents=True)
        (topic_dir / "TOPIC.md").write_text("## litellm\nmodel: old/model\n")
        ctx = _make_context(tmp_path)
        config1 = ctx.get_topic_config("cached")
        assert config1.model == "old/model"

        # Update file and invalidate
        (topic_dir / "TOPIC.md").write_text("## litellm\nmodel: new/model\n")
        ctx.invalidate_topic_cache("cached")
        config2 = ctx.get_topic_config("cached")
        assert config2.model == "new/model"
