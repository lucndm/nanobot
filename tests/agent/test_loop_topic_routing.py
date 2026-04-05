"""Tests for per-topic model routing in AgentLoop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.topic_config import TopicConfig


def _make_loop():
    """Create an AgentLoop with minimal mocked dependencies."""
    from nanobot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop.model = "default/model"
    loop.max_iterations = 40
    loop.context = MagicMock()
    loop.runner = MagicMock()
    loop.runner.run = AsyncMock()
    loop._session_locks = {}
    loop._last_usage = {}
    loop._last_run_metadata = {}
    loop.tools = MagicMock()
    loop.tools.get_definitions.return_value = []
    loop._otel_config = None  # Needed to avoid AttributeError in _run_agent_loop
    # Mock provider with is_model_available (returns True to allow all models in tests)
    loop.provider = MagicMock()
    loop.provider.is_model_available = MagicMock(return_value=True)
    return loop


class TestTopicModelRouting:
    @pytest.mark.asyncio
    async def test_topic_config_overrides_model_in_run_spec(self):
        loop = _make_loop()
        topic_config = TopicConfig(model="topic/model", temperature=0.9, max_tokens=2048)
        loop.context.get_topic_config.return_value = topic_config
        loop.runner.run.return_value = MagicMock(usage={}, stop_reason="stop")

        await loop._run_agent_loop(
            initial_messages=[],
            on_progress=MagicMock(),
            on_stream=MagicMock(),
            on_stream_end=MagicMock(),
            channel="telegram",
            chat_id="123",
            message_id=1,
            message_thread_id=42,
            topic_name="test-topic",
        )

        assert loop.runner.run.called
        spec = loop.runner.run.call_args[0][0]
        assert spec.model == "topic/model"
        assert spec.temperature == 0.9
        assert spec.max_tokens == 2048

    @pytest.mark.asyncio
    async def test_no_topic_config_uses_defaults(self):
        loop = _make_loop()
        loop.context.get_topic_config.return_value = None
        loop.runner.run.return_value = MagicMock(usage={}, stop_reason="stop")

        await loop._run_agent_loop(
            initial_messages=[],
            on_progress=MagicMock(),
            on_stream=MagicMock(),
            on_stream_end=MagicMock(),
            channel="telegram",
            chat_id="123",
            message_id=1,
            message_thread_id=42,
            topic_name="plain-topic",
        )

        spec = loop.runner.run.call_args[0][0]
        assert spec.model == "default/model"
        assert spec.temperature is None
        assert spec.max_tokens is None

    @pytest.mark.asyncio
    async def test_no_topic_name_skips_config_lookup(self):
        loop = _make_loop()
        loop.runner.run.return_value = MagicMock(usage={}, stop_reason="stop")

        await loop._run_agent_loop(
            initial_messages=[],
            on_progress=MagicMock(),
            on_stream=MagicMock(),
            on_stream_end=MagicMock(),
            channel="telegram",
            chat_id="123",
            message_id=1,
            message_thread_id=None,
            topic_name=None,
        )

        loop.context.get_topic_config.assert_not_called()
        spec = loop.runner.run.call_args[0][0]
        assert spec.model == "default/model"

    @pytest.mark.asyncio
    async def test_partial_topic_config_only_overrides_model(self):
        loop = _make_loop()
        topic_config = TopicConfig(model="topic/model")
        loop.context.get_topic_config.return_value = topic_config
        loop.runner.run.return_value = MagicMock(usage={}, stop_reason="stop")

        await loop._run_agent_loop(
            initial_messages=[],
            on_progress=MagicMock(),
            on_stream=MagicMock(),
            on_stream_end=MagicMock(),
            channel="telegram",
            chat_id="123",
            message_id=1,
            message_thread_id=42,
            topic_name="partial-topic",
        )

        spec = loop.runner.run.call_args[0][0]
        assert spec.model == "topic/model"
        assert spec.temperature is None
        assert spec.max_tokens is None

    @pytest.mark.asyncio
    async def test_invalid_topic_model_falls_back_to_default(self):
        loop = _make_loop()
        loop.provider.is_model_available = MagicMock(return_value=False)
        topic_config = TopicConfig(model="invalid/model", temperature=0.9, max_tokens=2048)
        loop.context.get_topic_config.return_value = topic_config
        loop.runner.run.return_value = MagicMock(usage={}, stop_reason="stop")

        await loop._run_agent_loop(
            initial_messages=[],
            on_progress=MagicMock(),
            on_stream=MagicMock(),
            on_stream_end=MagicMock(),
            channel="telegram",
            chat_id="123",
            message_id=1,
            message_thread_id=42,
            topic_name="bad-model-topic",
        )

        spec = loop.runner.run.call_args[0][0]
        # Should fall back to default model when is_model_available returns False
        assert spec.model == "default/model"
        assert spec.temperature is None  # Also cleared on fallback
        assert spec.max_tokens is None
