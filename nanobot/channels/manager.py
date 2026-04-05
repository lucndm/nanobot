"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import Config

if TYPE_CHECKING:
    from nanobot.channels.base import BaseChannel

# Retry delays for message sending (exponential backoff: 1s, 2s, 4s)
_SEND_RETRY_DELAYS = (1, 2, 4)


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(self, config: Config, bus: MessageBus, topic_store=None):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._topic_store = topic_store

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize Telegram channel directly."""
        cfg = self.config.channel
        if not cfg.enabled:
            return
        try:
            workspace = getattr(self.config, "workspace_path", None)
            channel = TelegramChannel(cfg, self.bus, workspace=workspace)
            channel.transcription_api_key = self.config.litellm.groq_api_key
            # Share the memory store with channels for topic mapping
            if self._topic_store is None and workspace is not None:
                from nanobot.agent.store import create_memory_store

                self._topic_store = create_memory_store(self.config, workspace)
            if self._topic_store is not None:
                channel.topic_store = self._topic_store
            self.channels["telegram"] = channel
            logger.info("Telegram channel enabled")
        except Exception as e:
            logger.warning("Telegram channel not available: {}", e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            if getattr(ch.config, "allow_from", None) == []:
                raise SystemExit(
                    f'Error: "{name}" has empty allowFrom (denies all). '
                    f'Set ["*"] to allow everyone, or add specific user IDs.'
                )

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        # Buffer for messages that couldn't be processed during delta coalescing
        # (since asyncio.Queue doesn't support push_front)
        pending: list[OutboundMessage] = []

        while True:
            try:
                # First check pending buffer before waiting on queue
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channel.send_tool_hints:
                        continue
                    if not msg.metadata.get("_tool_hint") and not self.config.channel.send_progress:
                        continue

                # Coalesce consecutive _stream_delta messages for the same (channel, chat_id)
                # to reduce API calls and improve streaming latency
                if msg.metadata.get("_stream_delta") and not msg.metadata.get("_stream_end"):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)

                channel = self.channels.get(msg.channel)
                if channel:
                    await self._send_with_retry(channel, msg)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send one outbound message without retry policy."""
        if msg.metadata.get("_stream_delta") or msg.metadata.get("_stream_end"):
            await channel.send_delta(msg.chat_id, msg.content, msg.metadata)
        elif not msg.metadata.get("_streamed"):
            await channel.send(msg)

    def _coalesce_stream_deltas(
        self, first_msg: OutboundMessage
    ) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """Merge consecutive _stream_delta messages for the same (channel, chat_id).

        This reduces the number of API calls when the queue has accumulated multiple
        deltas, which happens when LLM generates faster than the channel can process.

        Returns:
            tuple of (merged_message, list_of_non_matching_messages)
        """
        target_key = (first_msg.channel, first_msg.chat_id)
        combined_content = first_msg.content
        final_metadata = dict(first_msg.metadata or {})
        non_matching: list[OutboundMessage] = []

        # Only merge consecutive deltas. As soon as we hit any other message,
        # stop and hand that boundary back to the dispatcher via `pending`.
        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Check if this message belongs to the same stream
            same_target = (next_msg.channel, next_msg.chat_id) == target_key
            is_delta = next_msg.metadata and next_msg.metadata.get("_stream_delta")
            is_end = next_msg.metadata and next_msg.metadata.get("_stream_end")

            if same_target and is_delta and not final_metadata.get("_stream_end"):
                # Accumulate content
                combined_content += next_msg.content
                # If we see _stream_end, remember it and stop coalescing this stream
                if is_end:
                    final_metadata["_stream_end"] = True
                    # Stream ended - stop coalescing this stream
                    break
            else:
                # First non-matching message defines the coalescing boundary.
                non_matching.append(next_msg)
                break

        merged = OutboundMessage(
            channel=first_msg.channel,
            chat_id=first_msg.chat_id,
            content=combined_content,
            metadata=final_metadata,
        )
        return merged, non_matching

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send a message with retry on failure.

        - RetryAfter: uses the server-specified delay
        - TimedOut / transient: exponential backoff (1, 2, 4s)
        - BadRequest: non-retryable (e.g. message too long)
        - CancelledError: re-raised for graceful shutdown
        """
        max_attempts = max(self.config.channel.send_max_retries, 1)

        for attempt in range(max_attempts):
            try:
                await self._send_once(channel, msg)
                return  # Send succeeded
            except asyncio.CancelledError:
                raise  # Propagate cancellation for graceful shutdown
            except Exception as e:
                retry_delay = self._retry_delay_for(e, attempt)
                if retry_delay is None:
                    # Non-retryable error (e.g. BadRequest)
                    logger.error(
                        "Non-retryable send to {} (attempt {}/{}): {} - {}",
                        msg.channel,
                        attempt + 1,
                        max_attempts,
                        type(e).__name__,
                        e,
                    )
                    return
                if attempt == max_attempts - 1:
                    logger.error(
                        "Failed to send to {} after {} attempts: {} - {}",
                        msg.channel,
                        max_attempts,
                        type(e).__name__,
                        e,
                    )
                    return
                logger.warning(
                    "Send to {} failed (attempt {}/{}): {}, retrying in {:.1f}s",
                    msg.channel,
                    attempt + 1,
                    max_attempts,
                    type(e).__name__,
                    retry_delay,
                )
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    raise  # Propagate cancellation during sleep

    @staticmethod
    def _retry_delay_for(exc: Exception, attempt: int) -> float | None:
        """Return retry delay in seconds, or None if non-retryable.

        Uses duck-typing on exception name to keep ChannelManager channel-agnostic.
        """
        exc_name = type(exc).__name__

        # Non-retryable: BadRequest (message too long, bad format, etc.)
        if exc_name == "BadRequest":
            return None

        # RetryAfter: use server-specified delay
        if exc_name == "RetryAfter":
            try:
                retry_after = getattr(exc, "_retry_after", None)
                if retry_after is not None:
                    return float(retry_after.total_seconds()) + 0.5
            except Exception:
                pass
            # Fallback if we can't read the delay
            return _SEND_RETRY_DELAYS[min(attempt, len(_SEND_RETRY_DELAYS) - 1)]

        # Transient errors: exponential backoff
        return _SEND_RETRY_DELAYS[min(attempt, len(_SEND_RETRY_DELAYS) - 1)]

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
