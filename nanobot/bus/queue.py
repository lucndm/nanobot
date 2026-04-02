"""Async message queue for decoupled channel-agent communication."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from nanobot.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from opentelemetry.metrics import Histogram, Meter, Observation


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

        self._latency_histogram: Histogram | None = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize OTel metrics (no-op if OTel is not set up)."""
        from opentelemetry.metrics import Observation

        from nanobot.observability.otel import get_meter

        meter: Meter | None = get_meter()
        if meter is None:
            return

        self._latency_histogram = meter.create_histogram(
            "nanobot.bus.latency",
            unit="ms",
            description="Time messages spend waiting in the queue",
        )

        def _queue_depth_callback(_options: object) -> list[Observation]:
            return [
                Observation(value=self.inbound.qsize(), attributes={"queue": "inbound"}),
                Observation(value=self.outbound.qsize(), attributes={"queue": "outbound"}),
            ]

        meter.create_observable_gauge(
            "nanobot.bus.queue.depth",
            callbacks=[_queue_depth_callback],
            description="Current depth of inbound and outbound message queues",
        )

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        msg.queued_at = time.monotonic()
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        msg = await self.inbound.get()
        if self._latency_histogram is not None and msg.queued_at > 0:
            latency_ms = (time.monotonic() - msg.queued_at) * 1000
            self._latency_histogram.record(latency_ms, attributes={"channel": msg.channel})
        return msg

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        msg.queued_at = time.monotonic()
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        msg = await self.outbound.get()
        if self._latency_histogram is not None and msg.queued_at > 0:
            latency_ms = (time.monotonic() - msg.queued_at) * 1000
            self._latency_histogram.record(latency_ms, attributes={"channel": msg.channel})
        return msg

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
