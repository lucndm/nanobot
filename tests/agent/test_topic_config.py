"""Tests for TopicConfig parsing from TOPIC.md litellm sections."""

from __future__ import annotations

from nanobot.agent.topic_config import TopicConfig, parse_topic_config


class TestParseTopicConfig:
    def test_parse_full_litellm_section(self):
        content = """\
# Topic: my-topic

## purpose
Test topic

## litellm
model: anthropic/claude-3-5-haiku-20241007
temperature: 0.7
max_tokens: 4096

## rules
Be concise.
"""
        config = parse_topic_config(content)
        assert config is not None
        assert config.model == "anthropic/claude-3-5-haiku-20241007"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_parse_no_litellm_section(self):
        content = """\
# Topic: plain-topic

## purpose
No model config here
"""
        config = parse_topic_config(content)
        assert config is None

    def test_parse_partial_litellm_model_only(self):
        content = """\
## litellm
model: openai/gpt-4o
"""
        config = parse_topic_config(content)
        assert config is not None
        assert config.model == "openai/gpt-4o"
        assert config.temperature is None
        assert config.max_tokens is None

    def test_parse_empty_litellm_section(self):
        content = """\
## litellm

## rules
Something
"""
        config = parse_topic_config(content)
        assert config is None

    def test_parse_invalid_temperature_uses_none(self):
        content = """\
## litellm
model: test/model
temperature: hot
max_tokens: 4096
"""
        config = parse_topic_config(content)
        assert config is not None
        assert config.model == "test/model"
        assert config.temperature is None
        assert config.max_tokens == 4096

    def test_parse_invalid_max_tokens_uses_none(self):
        content = """\
## litellm
model: test/model
max_tokens: unlimited
"""
        config = parse_topic_config(content)
        assert config is not None
        assert config.model == "test/model"
        assert config.max_tokens is None


class TestTopicConfig:
    def test_defaults(self):
        config = TopicConfig(model="test/model")
        assert config.model == "test/model"
        assert config.temperature is None
        assert config.max_tokens is None

    def test_all_fields(self):
        config = TopicConfig(model="m", temperature=0.5, max_tokens=2048)
        assert config.model == "m"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
