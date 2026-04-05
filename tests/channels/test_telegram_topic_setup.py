"""Tests for new topic setup flow in TelegramChannel."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestNewTopicSetup:
    @pytest.mark.asyncio
    async def test_handle_new_topic_creates_topic_md_and_mapping(self, tmp_path):
        from nanobot.channels.telegram import TelegramChannel

        channel = TelegramChannel.__new__(TelegramChannel)
        channel.config = MagicMock()
        channel._topic_names = {}
        channel._topic_store = MagicMock()
        channel.workspace = tmp_path

        await channel._handle_new_topic(
            chat_id=-100,
            thread_id=42,
            topic_name="finance",
            purpose="Quản lý chi tiêu hàng tháng",
        )

        topic_file = tmp_path / "topics" / "finance" / "TOPIC.md"
        assert topic_file.exists()
        content = topic_file.read_text()
        assert "finance" in content.lower()

        channel._topic_store.set_topic_mapping.assert_called_once_with(-100, 42, "finance")
