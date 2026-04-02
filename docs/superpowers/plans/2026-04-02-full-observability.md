# Full Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 13 new OTel metrics across 5 layers (LLM, Channel, Bus, Session, Skill) so Grafana dashboards show the full request lifecycle.

**Architecture:** Per-layer instrumentation — each module calls `get_meter()` singleton from `nanobot.observability.otel`. No hook references passed between layers. All OTel calls guarded with `if instrument` and `try/except` to never crash the agent.

**Tech Stack:** Python 3.11+, OpenTelemetry SDK (already installed), VictoriaMetrics, Grafana

**Spec:** `docs/superpowers/specs/2026-04-02-full-observability-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `nanobot/agent/runner.py` | Add 4 LLM metrics (duration, prompt tokens, completion tokens, errors) |
| Modify | `nanobot/agent/runner.py` | Add `channel: str` to `AgentRunSpec` |
| Modify | `nanobot/agent/loop.py` | Pass `channel=channel` to `AgentRunSpec` |
| Modify | `nanobot/bus/events.py` | Add `queued_at: float` to `InboundMessage` and `OutboundMessage` |
| Modify | `nanobot/bus/queue.py` | Add 2 bus metrics (queue depth gauge, latency histogram) |
| Modify | `nanobot/channels/base.py` | Add channel inbound counter + bus latency recording |
| Modify | `nanobot/channels/telegram.py` | Add 3 channel metrics (messages, send errors, send duration) |
| Modify | `nanobot/session/manager.py` | Add 1 session gauge (active sessions) |
| Modify | `nanobot/agent/skills.py` | Add 2 skill metrics (duration, errors) |
| Modify | `nanobot/observability/hook.py` | Add `topic_name` to `record_skill()` |
| Create | `tests/observability/test_llm_metrics.py` | Tests for LLM metrics in runner |
| Create | `tests/observability/test_channel_metrics.py` | Tests for channel metrics |
| Create | `tests/observability/test_bus_metrics.py` | Tests for bus metrics |
| Create | `tests/observability/test_session_metrics.py` | Tests for session gauge |
| Create | `tests/observability/test_skill_metrics.py` | Tests for skill metrics |
| Modify | `ws_nanobot/dashboards/nanobot-overview.json` | Add 10 new panels |

---

## Task 1: LLM Metrics in AgentRunner

**Files:**
- Modify: `nanobot/agent/runner.py:22-36` (AgentRunSpec), `nanobot/agent/runner.py:52-56` (AgentRunner.__init__), `nanobot/agent/runner.py:68-98` (run method)
- Modify: `nanobot/agent/loop.py:307-317` (AgentRunSpec construction)
- Create: `tests/observability/test_llm_metrics.py`

- [ ] **Step 1: Add `channel` field to `AgentRunSpec`**

In `nanobot/agent/runner.py`, add to the `AgentRunSpec` dataclass (after `reasoning_effort`):

```python
channel: str = ""
```

- [ ] **Step 2: Pass `channel` in `loop.py`**

In `nanobot/agent/loop.py:308`, add `channel=channel` to the `AgentRunSpec(...)` call:

```python
AgentRunSpec(
    initial_messages=initial_messages,
    tools=self.tools,
    model=self.model,
    max_iterations=self.max_iterations,
    hook=hook,
    error_message="Sorry, I encountered an error calling the AI model.",
    concurrent_tools=True,
    channel=channel,
)
```

- [ ] **Step 3: Write failing test for LLM metrics**

Create `tests/observability/test_llm_metrics.py`:

```python
"""Tests for LLM metrics emitted by AgentRunner."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.base import LLMResponse


@pytest.fixture
def mock_meter():
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_counter, mock_histogram


@pytest.mark.asyncio
async def test_llm_metrics_recorded_on_success(mock_meter):
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    mock_meter, mock_counter, mock_histogram = mock_meter

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[], usage={"prompt_tokens": 50, "completion_tokens": 20},
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="gpt-4o",
        max_iterations=1,
        channel="telegram",
    ))

    # Verify duration histogram was recorded
    assert mock_histogram.record.call_count >= 1
    duration_call = mock_histogram.record.call_args
    assert duration_call.kwargs["attributes"]["model"] == "gpt-4o"
    assert duration_call.kwargs["attributes"]["channel"] == "telegram"

    # Verify token counters were incremented
    token_adds = [c for c in mock_counter.add.call_args_list
                  if "model" in (c.kwargs.get("attributes") or {})]
    assert len(token_adds) >= 2  # prompt + completion


@pytest.mark.asyncio
async def test_llm_error_counter_on_exception(mock_meter):
    from nanobot.agent.runner import AgentRunSpec, AgentRunner

    mock_meter, mock_counter, _ = mock_meter

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("API down"))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    try:
        await runner.run(AgentRunSpec(
            initial_messages=[],
            tools=tools,
            model="gpt-4o",
            max_iterations=1,
            channel="telegram",
        ))
    except RuntimeError:
        pass

    # Verify error counter was incremented
    error_adds = [c for c in mock_counter.add.call_args_list
                  if c.kwargs.get("attributes", {}).get("model") == "gpt-4o"
                  and "error" in str(c)]
    # At least one counter.add with model attribute
    assert any(c.kwargs.get("attributes", {}).get("model") == "gpt-4o"
               for c in mock_counter.add.call_args_list)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd nanobot && pytest tests/observability/test_llm_metrics.py -v`
Expected: FAIL — runner doesn't emit LLM metrics yet

- [ ] **Step 5: Implement LLM metrics in AgentRunner**

In `nanobot/agent/runner.py`:

Add imports at top:
```python
import time

from loguru import logger
```

In `AgentRunner.__init__`, add instrument creation:
```python
from nanobot.observability.otel import get_meter

def __init__(self, provider: LLMProvider):
    self.provider = provider
    meter = get_meter()
    self._llm_duration = (
        meter.create_histogram("nanobot.llm.request.duration", description="LLM request duration in ms", unit="ms")
        if meter else None
    )
    self._llm_prompt_tokens = (
        meter.create_counter("nanobot.llm.tokens.prompt", description="Prompt tokens sent to LLM")
        if meter else None
    )
    self._llm_completion_tokens = (
        meter.create_counter("nanobot.llm.tokens.completion", description="Completion tokens from LLM")
        if meter else None
    )
    self._llm_errors = (
        meter.create_counter("nanobot.llm.errors", description="LLM request errors")
        if meter else None
    )
```

In `run()`, wrap the LLM call (around line 83-92) with timing and recording. Before the `if hook.wants_streaming():` block, add `llm_start = time.monotonic()`. After the response is received and usage is extracted (after line 98), add:

```python
# Record LLM metrics
try:
    attrs = {"model": spec.model, "channel": spec.channel}
    if self._llm_duration:
        self._llm_duration.record((time.monotonic() - llm_start) * 1000, attributes=attrs)
    if self._llm_prompt_tokens and usage.get("prompt_tokens"):
        self._llm_prompt_tokens.add(usage["prompt_tokens"], attributes=attrs)
    if self._llm_completion_tokens and usage.get("completion_tokens"):
        self._llm_completion_tokens.add(usage["completion_tokens"], attributes=attrs)
except Exception:
    logger.debug("OTEL: failed to record LLM metrics")
```

For error recording, wrap the LLM call in try/except:
```python
try:
    if hook.wants_streaming():
        ...existing streaming code...
    else:
        response = await self.provider.chat_with_retry(**kwargs)
except Exception:
    try:
        if self._llm_errors:
            self._llm_errors.add(1, attributes={"model": spec.model, "channel": spec.channel})
    except Exception:
        logger.debug("OTEL: failed to record LLM error")
    raise
```

- [ ] **Step 6: Run tests**

Run: `cd nanobot && pytest tests/observability/test_llm_metrics.py tests/agent/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
cd nanobot && git add nanobot/agent/runner.py nanobot/agent/loop.py tests/observability/test_llm_metrics.py && git commit -m "feat(otel): add LLM request metrics (duration, tokens, errors)"
```

---

## Task 2: Bus Metrics (Queue Depth + Latency)

**Files:**
- Modify: `nanobot/bus/events.py` (add `queued_at`)
- Modify: `nanobot/bus/queue.py` (add metrics)
- Create: `tests/observability/test_bus_metrics.py`

- [ ] **Step 1: Add `queued_at` to bus events**

In `nanobot/bus/events.py`, add `queued_at: float = 0.0` to both `InboundMessage` and `OutboundMessage`:

```python
@dataclass
class InboundMessage:
    ...
    queued_at: float = 0.0  # monotonic timestamp set by MessageBus
```

```python
@dataclass
class OutboundMessage:
    ...
    queued_at: float = 0.0  # monotonic timestamp set by MessageBus
```

- [ ] **Step 2: Write failing test**

Create `tests/observability/test_bus_metrics.py`:

```python
"""Tests for bus metrics (queue depth, latency)."""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage


@pytest.fixture
def mock_meter():
    mock_histogram = MagicMock()
    mock_observable = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_histogram.return_value = mock_histogram
    mock_meter.create_observable_gauge.return_value = mock_observable

    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_histogram, mock_observable


@pytest.mark.asyncio
async def test_bus_sets_queued_at_on_publish(mock_meter):
    mock_meter, _, _ = mock_meter
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hi")
    await bus.publish_inbound(msg)

    assert msg.queued_at > 0


@pytest.mark.asyncio
async def test_bus_records_latency_on_consume(mock_meter):
    _, mock_histogram, _ = mock_meter
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hi")
    await bus.publish_inbound(msg)
    await bus.consume_inbound()

    assert mock_histogram.record.call_count >= 1


@pytest.mark.asyncio
async def test_bus_creates_observable_gauge(mock_meter):
    _, _, mock_observable = mock_meter
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    mock_meter.create_observable_gauge.assert_called()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd nanobot && pytest tests/observability/test_bus_metrics.py -v`
Expected: FAIL — MessageBus doesn't set `queued_at` or record metrics

- [ ] **Step 4: Implement bus metrics**

In `nanobot/bus/queue.py`:

```python
"""Async message queue for decoupled channel-agent communication."""
import time

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._setup_metrics()

    def _setup_metrics(self):
        from nanobot.observability.otel import get_meter
        meter = get_meter()
        if not meter:
            self._latency_hist = None
            return

        self._latency_hist = meter.create_histogram(
            "nanobot.bus.latency",
            description="Message publish-to-consume latency in ms",
            unit="ms",
        )
        meter.create_observable_gauge(
            "nanobot.bus.queue.depth",
            callbacks=[self._observe_queue_depth],
            description="Current queue depth",
        )

    def _observe_queue_depth(self, options):
        from opentelemetry.metrics import Observation
        yield Observation(self.inbound.qsize(), attributes={"queue": "inbound"})
        yield Observation(self.outbound.qsize(), attributes={"queue": "outbound"})

    async def publish_inbound(self, msg: InboundMessage) -> None:
        msg.queued_at = time.monotonic()
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        msg = await self.inbound.get()
        self._record_latency(msg.queued_at, msg.channel)
        return msg

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        msg.queued_at = time.monotonic()
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        msg = await self.outbound.get()
        self._record_latency(msg.queued_at, msg.channel)
        return msg

    def _record_latency(self, queued_at: float, channel: str) -> None:
        try:
            if self._latency_hist and queued_at > 0:
                latency_ms = (time.monotonic() - queued_at) * 1000
                self._latency_hist.record(latency_ms, attributes={"channel": channel})
        except Exception:
            logger.debug("OTEL: failed to record bus latency")

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
```

Note: The OTel Python SDK (>=1.28) observable gauge callback signature is `Callable[[CallbackOptions], Iterable[Observation]]`. Import `Observation` from `opentelemetry.metrics`. Each callback yields `Observation(value, attributes)` instances.

- [ ] **Step 5: Run tests**

Run: `cd nanobot && pytest tests/observability/test_bus_metrics.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd nanobot && git add nanobot/bus/events.py nanobot/bus/queue.py tests/observability/test_bus_metrics.py && git commit -m "feat(otel): add bus metrics (queue depth gauge, latency histogram)"
```

---

## Task 3: Channel Metrics (Messages, Send Errors, Send Duration)

**Files:**
- Modify: `nanobot/channels/base.py:145-167` (publish_inbound)
- Modify: `nanobot/channels/telegram.py` (add metrics)
- Create: `tests/observability/test_channel_metrics.py`

- [ ] **Step 1: Write failing test**

Create `tests/observability/test_channel_metrics.py`:

```python
"""Tests for channel metrics."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_meter():
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_counter, mock_histogram


@pytest.mark.asyncio
async def test_inbound_message_increments_counter(mock_meter):
    """base.py publish_inbound should record nanobot.channel.messages."""
    _, mock_counter, _ = mock_meter

    from nanobot.channels.base import BaseChannel

    # Use a minimal concrete subclass
    class TestChannel(BaseChannel):
        name = "test"
        def __init__(self):
            self._config = MagicMock()
            self._config.allowFrom = []
            self.bus = MagicMock()
            self.bus.publish_inbound = AsyncMock()
            self._msg_counter = None  # will be set by _setup_metrics

        async def start(self): pass
        async def stop(self): pass
        async def send(self, msg): pass

    ch = TestChannel()
    await ch.publish_inbound(
        sender_id="u1", chat_id="c1", content="hello",
    )

    # Verify counter was incremented
    assert mock_counter.add.call_count >= 1
    attrs = mock_counter.add.call_args.kwargs.get("attributes", {})
    assert attrs.get("direction") == "inbound"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd nanobot && pytest tests/observability/test_channel_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement channel metrics in `base.py`**

In `nanobot/channels/base.py`, add to `BaseChannel.__init__` pattern — since `BaseChannel` is ABC and subclasses call `super().__init__()`, add metric creation there. Check the actual `__init__` method signature first and add:

```python
from nanobot.observability.otel import get_meter

# In __init__:
meter = get_meter()
self._msg_counter = (
    meter.create_counter("nanobot.channel.messages", description="Channel messages received/sent")
    if meter else None
)
```

In `publish_inbound()` (line ~167), before `await self.bus.publish_inbound(msg)`, add:

```python
try:
    if self._msg_counter:
        self._msg_counter.add(1, attributes={"channel": self.name, "direction": "inbound"})
except Exception:
    logger.debug("OTEL: failed to record inbound message metric")
```

- [ ] **Step 4: Add Telegram send metrics**

In `nanobot/channels/telegram.py`, add instruments in `__init__` (check actual `__init__` location):

```python
meter = get_meter()
self._send_duration = (
    meter.create_histogram("nanobot.channel.send.duration", description="Channel send duration in ms", unit="ms")
    if meter else None
)
self._send_errors = (
    meter.create_counter("nanobot.channel.send.errors", description="Channel send errors")
    if meter else None
)
```

In `send()` method, wrap the text sending portion with timing:

```python
send_start = time.monotonic()
# ... existing send logic ...
try:
    if self._send_duration:
        self._send_duration.record((time.monotonic() - send_start) * 1000, attributes={"channel": "telegram"})
except Exception:
    logger.debug("OTEL: failed to record send duration")
```

On error in `_send_text`, increment error counter:

```python
try:
    if self._send_errors:
        self._send_errors.add(1, attributes={"channel": "telegram"})
except Exception:
    logger.debug("OTEL: failed to record send error")
```

Also record outbound direction in `send()`:

```python
try:
    if self._msg_counter:
        self._msg_counter.add(1, attributes={"channel": "telegram", "direction": "outbound"})
except Exception:
    logger.debug("OTEL: failed to record outbound message metric")
```

- [ ] **Step 5: Run tests**

Run: `cd nanobot && pytest tests/observability/test_channel_metrics.py tests/channels/test_telegram_channel.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd nanobot && git add nanobot/channels/base.py nanobot/channels/telegram.py tests/observability/test_channel_metrics.py && git commit -m "feat(otel): add channel metrics (messages, send errors, send duration)"
```

---

## Task 4: Session Active Gauge

**Files:**
- Modify: `nanobot/session/manager.py:135-139` (__init__)
- Create: `tests/observability/test_session_metrics.py`

- [ ] **Step 1: Write failing test**

Create `tests/observability/test_session_metrics.py`:

```python
"""Tests for session metrics."""
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path


@pytest.fixture
def mock_meter():
    mock_observable = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_observable_gauge.return_value = mock_observable

    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_observable


def test_session_manager_creates_active_gauge(mock_meter, tmp_path):
    mock_meter, mock_observable = mock_meter
    from nanobot.session.manager import SessionManager

    mgr = SessionManager(tmp_path)

    mock_meter.create_observable_gauge.assert_called_once_with(
        "nanobot.session.active",
        callbacks=[mgr._observe_active_sessions],
        description="Number of active sessions in cache",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd nanobot && pytest tests/observability/test_session_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement session gauge**

In `nanobot/session/manager.py`, add to `SessionManager.__init__`:

```python
from nanobot.observability.otel import get_meter
from loguru import logger

# In __init__, after self._cache:
meter = get_meter()
if meter:
    meter.create_observable_gauge(
        "nanobot.session.active",
        callbacks=[self._observe_active_sessions],
        description="Number of active sessions in cache",
    )
```

Add the callback method:

```python
def _observe_active_sessions(self, options):
    from opentelemetry.metrics import Observation
    yield Observation(len(self._cache), attributes={})
```

- [ ] **Step 4: Run tests**

Run: `cd nanobot && pytest tests/observability/test_session_metrics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd nanobot && git add nanobot/session/manager.py tests/observability/test_session_metrics.py && git commit -m "feat(otel): add active sessions gauge"
```

---

## Task 5: Skill Metrics (Duration + Errors)

**Files:**
- Modify: `nanobot/agent/skills.py:22-25` (__init__), `nanobot/agent/skills.py:64-85` (load_skill)
- Modify: `nanobot/observability/hook.py:164-170` (record_skill — add topic_name)
- Create: `tests/observability/test_skill_metrics.py`

- [ ] **Step 1: Write failing test**

Create `tests/observability/test_skill_metrics.py`:

```python
"""Tests for skill duration/error metrics."""
import time
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path


@pytest.fixture
def mock_meter():
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with patch("nanobot.observability.otel.get_meter", return_value=mock_meter):
        yield mock_meter, mock_counter, mock_histogram


def test_skill_load_records_duration(mock_meter, tmp_path):
    mock_meter, _, mock_histogram = mock_meter
    from nanobot.agent.skills import SkillsLoader

    # Create a minimal skill
    skill_dir = tmp_path / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Test Skill\nHello")

    loader = SkillsLoader(tmp_path)
    content = loader.load_skill("test_skill")

    assert content is not None
    assert mock_histogram.record.call_count >= 1
    attrs = mock_histogram.record.call_args.kwargs.get("attributes", {})
    assert attrs.get("skill_name") == "test_skill"


def test_skill_load_nonexistent_records_no_error(mock_meter, tmp_path):
    _, mock_counter, _ = mock_meter
    from nanobot.agent.skills import SkillsLoader

    loader = SkillsLoader(tmp_path)
    content = loader.load_skill("nonexistent")

    assert content is None
    # No error should be recorded — skill not found is not an error
    assert mock_counter.add.call_count == 0


def test_record_skill_includes_topic_name():
    from nanobot.observability.hook import OTelHook

    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with (
        patch("nanobot.observability.otel.get_meter", return_value=mock_meter),
        patch("nanobot.observability.otel.get_tracer", return_value=None),
    ):
        hook = OTelHook(channel="telegram", chat_id="123", topic_name="finance")
        hook.record_skill("firefly_tools")

    attrs = mock_counter.add.call_args.kwargs.get("attributes", {})
    assert attrs.get("topic_name") == "finance"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd nanobot && pytest tests/observability/test_skill_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement skill metrics**

In `nanobot/agent/skills.py`, add to `SkillsLoader.__init__`:

```python
import time
from loguru import logger

# In __init__, after self.builtin_skills:
from nanobot.observability.otel import get_meter
meter = get_meter()
self._skill_duration = (
    meter.create_histogram("nanobot.skill.duration", description="Skill load duration in ms", unit="ms")
    if meter else None
)
self._skill_errors = (
    meter.create_counter("nanobot.skill.errors", description="Skill load errors")
    if meter else None
)
```

In `load_skill()`, wrap file reads with timing:

```python
def load_skill(self, name: str) -> str | None:
    start = time.monotonic()
    # Check workspace first
    workspace_skill = self.workspace_skills / name / "SKILL.md"
    if workspace_skill.exists():
        content = workspace_skill.read_text(encoding="utf-8")
        self._record_duration(name, start)
        return content

    # Check built-in
    if self.builtin_skills:
        builtin_skill = self.builtin_skills / name / "SKILL.md"
        if builtin_skill.exists():
            content = builtin_skill.read_text(encoding="utf-8")
            self._record_duration(name, start)
            return content

    return None

def _record_duration(self, skill_name: str, start: float) -> None:
    try:
        if self._skill_duration:
            self._skill_duration.record(
                (time.monotonic() - start) * 1000,
                attributes={"skill_name": skill_name},
            )
    except Exception:
        logger.debug("OTEL: failed to record skill duration")
```

- [ ] **Step 4: Enhance `record_skill()` in hook.py with topic_name**

In `nanobot/observability/hook.py:168-170`, change:

```python
self._skill_counter.add(
    1,
    attributes={"skill_name": skill_name, "channel": self._channel, "topic_name": self._topic_name},
)
```

- [ ] **Step 5: Run tests**

Run: `cd nanobot && pytest tests/observability/test_skill_metrics.py tests/observability/test_otel_hook.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd nanobot && git add nanobot/agent/skills.py nanobot/observability/hook.py tests/observability/test_skill_metrics.py && git commit -m "feat(otel): add skill duration/error metrics + topic_name on skill counter"
```

---

## Task 6: Grafana Dashboard Panels

**Files:**
- Modify: `ws_nanobot/dashboards/nanobot-overview.json`

This task adds 10 new panels to the existing dashboard. Panels continue IDs from 13 (existing ends at 12). Layout starts at y=36 (after existing panels end at y=36).

- [ ] **Step 1: Update the dashboard JSON**

Add these panels to the `"panels"` array in `ws_nanobot/dashboards/nanobot-overview.json`. Use the existing `${ds}` variable and `topic_name=~"$topic_name"` filter pattern.

New panels (IDs 13-22):

1. **Panel 13: LLM Request Duration p50/p95/p99** — timeseries, y=36, h=8, w=12
   - Query A: `histogram_quantile(0.50, sum(rate(nanobot.llm.request.duration_bucket{topic_name=~"$topic_name"}[$__rate_interval])) by (le, model))`
   - Query B: p95, Query C: p99
   - Unit: ms

2. **Panel 14: Token Usage** — timeseries, y=36, h=8, w=12
   - Query A: `sum(rate(nanobot.llm.tokens.prompt{topic_name=~"$topic_name"}[$__rate_interval])) by (model)`
   - Query B: `sum(rate(nanobot.llm.tokens.completion{topic_name=~"$topic_name"}[$__rate_interval])) by (model)`
   - Unit: tokens/min

3. **Panel 15: LLM Error Rate** — stat, y=44, h=4, w=6
   - Query: `sum(rate(nanobot.llm.errors{topic_name=~"$topic_name"}[$__rate_interval])) / sum(rate(nanobot.llm.request.duration_count{topic_name=~"$topic_name"}[$__rate_interval])) * 100`
   - Unit: percent, thresholds: green/yellow/red

4. **Panel 16: Channel Throughput** — timeseries, y=44, h=8, w=12
   - Query A: `sum(rate(nanobot.channel.messages{direction="inbound"}[$__rate_interval])) by (channel)`
   - Query B: `sum(rate(nanobot.channel.messages{direction="outbound"}[$__rate_interval])) by (channel)`

5. **Panel 17: Channel Send Errors** — timeseries, y=44, h=8, w=12
   - Query: `sum(rate(nanobot.channel.send.errors{topic_name=~"$topic_name"}[$__rate_interval])) by (channel)`

6. **Panel 18: Bus Queue Depth** — timeseries, y=52, h=4, w=6
   - Query: `nanobot.bus.queue.depth`

7. **Panel 19: Active Sessions** — stat, y=52, h=4, w=6
   - Query: `nanobot.session.active`

8. **Panel 20: Skill Usage / min** — timeseries, y=56, h=8, w=12
   - Query: `sum(rate(nanobot.skill.loaded{topic_name=~"$topic_name"}[$__rate_interval])) by (skill_name, topic_name)`

9. **Panel 21: Skill Duration p95** — timeseries, y=56, h=8, w=12
   - Query: `histogram_quantile(0.95, sum(rate(nanobot.skill.duration_bucket{topic_name=~"$topic_name"}[$__rate_interval])) by (le, skill_name))`

10. **Panel 22: Top Skills** — barchart, y=64, h=8, w=12
    - Query: `topk(10, sum(increase(nanobot.skill.loaded{topic_name=~"$topic_name"}[$__range])) by (skill_name))`

Write the complete panel JSON objects following the same structure as existing panels (same datasource reference, legend config, field config pattern).

- [ ] **Step 2: Validate JSON**

Run: `python -c "import json; json.load(open('ws_nanobot/dashboards/nanobot-overview.json'))"`
Expected: no error

- [ ] **Step 3: Push to Grafana via MCP tool**

Use `mcp__grafana__update_dashboard` with the updated JSON to push the dashboard to Grafana. The dashboard UID is `nanobot-overview`.

- [ ] **Step 4: Commit**

```bash
cd ws_nanobot && git add dashboards/nanobot-overview.json && git commit -m "feat(dashboard): add 10 new panels for LLM, channel, bus, session, skill metrics"
```

---

## Task 7: Integration Test + Cleanup

**Files:**
- Run full test suite
- Verify lint

- [ ] **Step 1: Run all tests**

Run: `cd nanobot && pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Run linter**

Run: `cd nanobot && ruff check nanobot/`
Expected: no errors

- [ ] **Step 3: Run formatter**

Run: `cd nanobot && ruff format --check nanobot/`
Expected: no changes needed

- [ ] **Step 4: Final commit if any formatting fixes needed**

```bash
cd nanobot && ruff format nanobot/ && git add -A && git commit -m "style: ruff format"
```

---

## Summary

| Task | New Metrics | Files Modified | Tests Created |
|------|------------|----------------|---------------|
| 1 | 4 LLM (duration, prompt tokens, completion tokens, errors) | `runner.py`, `loop.py` | `test_llm_metrics.py` |
| 2 | 2 Bus (queue depth, latency) | `events.py`, `queue.py` | `test_bus_metrics.py` |
| 3 | 3 Channel (messages, send errors, send duration) | `base.py`, `telegram.py` | `test_channel_metrics.py` |
| 4 | 1 Session (active gauge) | `manager.py` | `test_session_metrics.py` |
| 5 | 2 Skill (duration, errors) + enhance existing | `skills.py`, `hook.py` | `test_skill_metrics.py` |
| 6 | Dashboard panels | `nanobot-overview.json` | — |
| 7 | Integration test + cleanup | — | — |

**Total: 12 new metrics + 1 enhanced metric + 10 dashboard panels across 7 tasks.**

**Deferred:** `nanobot.session.message.count` — requires instrumenting `loop.py` where `session.add_message()` is called. Low value since message count can be derived from `nanobot.agent.iterations` counter. Add later if needed.
