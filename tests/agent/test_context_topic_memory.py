"""Tests for ContextBuilder topic memory injection."""
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.context import ContextBuilder


def _make_builder(tmp_path: Path) -> ContextBuilder:
    return ContextBuilder(tmp_path, timezone="UTC")


class TestContextBuilderTopicMemory:
    def test_global_memory_injected_when_present(self, tmp_path: Path):
        builder = _make_builder(tmp_path)
        builder.memory = MagicMock()
        builder.memory.get_memory_context.return_value = "## Global Memory\nUser likes Python."
        builder.memory.get_topic_memory_context = MagicMock(return_value=None)
        prompt = builder.build_system_prompt()
        assert "User likes Python." in prompt

    def test_topic_memory_injected_when_topic_name_set(self, tmp_path: Path):
        builder = _make_builder(tmp_path)
        builder.memory = MagicMock()
        builder.memory.get_memory_context.return_value = "## Global Memory\nGlobal fact."
        builder.memory.get_topic_memory_context = MagicMock(return_value="## Topic Memory (558)\nTopic fact.")
        prompt = builder.build_system_prompt(topic_name="558")
        assert "Global fact." in prompt
        assert "Topic fact." in prompt

    def test_topic_memory_not_injected_when_no_topic(self, tmp_path: Path):
        builder = _make_builder(tmp_path)
        builder.memory = MagicMock()
        builder.memory.get_memory_context.return_value = "## Global Memory\nGlobal."
        builder.memory.get_topic_memory_context = MagicMock(return_value=None)
        prompt = builder.build_system_prompt()
        assert "Global." in prompt
        assert "Topic Memory" not in prompt

    def test_topic_memory_called_with_correct_topic_name(self, tmp_path: Path):
        builder = _make_builder(tmp_path)
        builder.memory = MagicMock()
        builder.memory.get_memory_context.return_value = ""
        builder.memory.get_topic_memory_context = MagicMock(return_value=None)
        builder.build_system_prompt(topic_name="558")
        builder.memory.get_topic_memory_context.assert_called_with("558")

    def test_prompt_order_memory_before_skills(self, tmp_path: Path):
        builder = _make_builder(tmp_path)
        builder.memory = MagicMock()
        builder.memory.get_memory_context.return_value = "## Global Memory\nM"
        builder.memory.get_topic_memory_context = MagicMock(return_value="## Topic Memory (558)\nT")
        prompt = builder.build_system_prompt(topic_name="558")
        mem_pos = prompt.find("## Global Memory")
        topic_pos = prompt.find("## Topic Memory")
        assert mem_pos < topic_pos
