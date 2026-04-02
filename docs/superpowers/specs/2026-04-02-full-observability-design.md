# Full Observability for Nanobot

## Context

Nanobot has 5 OTel metrics (tool calls, tool duration, skill loaded, active tools, agent iterations) emitted through `OTelHook`. This covers the agent loop layer only. Critical gaps: LLM cost/performance, channel reliability, bus health, session lifecycle, skill duration/errors.

## Goal

Add metrics for every layer of the system so dashboards can show the full request lifecycle: message received -> context built -> LLM call -> tool execution -> response sent.

## Existing Metrics (unchanged)

| Metric | Type | Description |
|--------|------|-------------|
| `nanobot.tool.calls` | counter | Tool invocations with tool_name, status, channel |
| `nanobot.tool.duration` | histogram | Tool execution duration in ms |
| `nanobot.skill.loaded` | counter | Skill activations with skill_name, channel |
| `nanobot.session.active_tools` | histogram | Distinct tools per session |
| `nanobot.agent.iterations` | counter | Agent loop iterations with stop_reason, channel |

## New Metrics (13)

### LLM Layer

| Metric | Type | Unit | Attributes | Location |
|--------|------|------|------------|----------|
| `nanobot.llm.request.duration` | histogram | ms | model, channel | `runner.py` |
| `nanobot.llm.tokens.prompt` | counter | tokens | model, channel | `runner.py` |
| `nanobot.llm.tokens.completion` | counter | tokens | model, channel | `runner.py` |
| `nanobot.llm.errors` | counter | - | model, channel | `runner.py` |

**Note**: LLM retries metric deferred — retry logic lives inside `provider.chat_with_retry()` and exposing it would require changing the provider interface.

### Channel Layer

| Metric | Type | Unit | Attributes | Location |
|--------|------|------|------------|----------|
| `nanobot.channel.messages` | counter | - | channel, direction (inbound/outbound) | `telegram.py` |
| `nanobot.channel.send.errors` | counter | - | channel, error_type | `telegram.py` |
| `nanobot.channel.send.duration` | histogram | ms | channel | `telegram.py` |

### Bus Layer

| Metric | Type | Unit | Attributes | Location |
|--------|------|------|------------|----------|
| `nanobot.bus.queue.depth` | gauge | - | - | `bus/queue.py` |
| `nanobot.bus.latency` | histogram | ms | channel | `bus/queue.py` |

### Session Layer

| Metric | Type | Unit | Attributes | Location |
|--------|------|------|------------|----------|
| `nanobot.session.active` | gauge | - | channel | `session/manager.py` |
| `nanobot.session.message.count` | counter | - | channel, session_id | `session/manager.py` |

### Skill Layer (new)

| Metric | Type | Unit | Attributes | Location |
|--------|------|------|------------|----------|
| `nanobot.skill.duration` | histogram | ms | skill_name, channel | `skills.py` |
| `nanobot.skill.errors` | counter | - | skill_name, channel, error_type | `skills.py` |

## Enhanced Existing Metrics

### `nanobot.skill.loaded`

Add `topic_name` attribute (already available in hook context):

```python
attributes={"skill_name": skill_name, "channel": self._channel, "topic_name": self._topic_name}
```

## Instrumentation Approach

**Per-layer `get_meter()` singleton** — each module imports `get_meter()` from `nanobot.observability.otel` and creates its own instruments at module level or class init. No hook references passed between layers.

### Pattern

```python
from nanobot.observability.otel import get_meter

class MyClass:
    def __init__(self):
        meter = get_meter()
        self._counter = (
            meter.create_counter("nanobot.my.metric", description="...")
            if meter
            else None
        )

    def do_work(self):
        try:
            if self._counter:
                self._counter.add(1, attributes={...})
        except Exception:
            logger.debug("OTEL: failed to record metric")
```

Key principles:
- Always guard with `if meter` (meter is None when OTel not configured)
- Always wrap in try/except (metrics must never crash the agent)
- Use `logger.debug` for OTel failures (not warnings, to avoid noise)

## Code Changes

### File: `nanobot/agent/runner.py`

**Problem**: `AgentRunner` has no channel context. `AgentRunSpec` carries `model` but not `channel`.

**Solution**: Add optional `channel: str` field to `AgentRunSpec` dataclass. `loop.py` already has channel info and passes spec to runner.

Add LLM metrics around the `chat_with_retry()` / `chat_stream_with_retry()` calls (lines 87-92):

1. Add `channel: str = ""` field to `AgentRunSpec`
2. Import `get_meter` from `nanobot.observability.otel`, `time` from stdlib
3. Create instruments in `AgentRunner.__init__`:
   - `nanobot.llm.request.duration` histogram
   - `nanobot.llm.tokens.prompt` counter
   - `nanobot.llm.tokens.completion` counter
   - `nanobot.llm.errors` counter
4. In `run()`, wrap LLM call with `time.monotonic()`:
   - Before line 83: `llm_start = time.monotonic()`
   - After line 92: record `request.duration`, `tokens.prompt`, `tokens.completion`
   - On exception from LLM: increment `errors` counter
5. Attributes: `model=spec.model`, `channel=spec.channel`
6. **Skip `nanobot.llm.retries`** — retry logic is inside `chat_with_retry()` which is provider-level. Exposing retry count would require changing provider interface. Defer to future work.

### File: `nanobot/channels/telegram.py`

Add channel metrics:

1. Import `get_meter`, `time`
2. Create instruments in `__init__`:
   - `nanobot.channel.messages` counter
   - `nanobot.channel.send.errors` counter
   - `nanobot.channel.send.duration` histogram
3. In `_on_message` / `_on_edited_message`: increment `messages` with `direction=inbound`
4. In `send()`: record `send.duration`, increment `messages` with `direction=outbound`, on error increment `send.errors`

### File: `nanobot/bus/queue.py`

**Problem**: Messages don't carry timestamps, so publish-to-consume latency can't be measured directly.

**Solution**: Add `queued_at: float` field to `InboundMessage` and `OutboundMessage` dataclasses. Set in `publish_*()`, read in `consume_*()`.

1. Add `queued_at: float = 0.0` to `InboundMessage` and `OutboundMessage` in `bus/events.py`
2. Import `get_meter`, `time`
3. Create instruments in `MessageBus.__init__`:
   - `nanobot.bus.queue.depth` observable gauge (callback reads `inbound.qsize()`)
   - `nanobot.bus.latency` histogram
4. In `publish_inbound()`: set `msg.queued_at = time.monotonic()` before put
5. In `consume_inbound()`: compute latency from `msg.queued_at` to now
6. Same for outbound publish/consume

### File: `nanobot/session/manager.py`

Add session metrics:

1. Import `get_meter`
2. Create instruments in `SessionManager.__init__`:
   - `nanobot.session.active` observable gauge (callback reads `len(self._cache)`)
   - `nanobot.session.message.count` counter
3. In `get_or_create()` — after adding to cache: increment message count on existing sessions when `add_message()` is called
4. **Simplification**: Move counter emission to `loop.py` instead (loop already calls `session.add_message()` and has meter access). Session manager only provides the gauge.
5. Observable gauge callback: `lambda: len(self._cache)` registered with `meter.create_observable_gauge()`

### File: `nanobot/agent/skills.py`

Add skill duration/error metrics:

1. Import `get_meter`, `time`
2. Create instruments in `SkillsLoader.__init__`:
   - `nanobot.skill.duration` histogram
   - `nanobot.skill.errors` counter
3. In `load_skill()`: wrap in `time.monotonic()`, record duration on success, increment errors on exception
4. Attributes: `skill_name`

### File: `nanobot/observability/hook.py`

Enhance existing `record_skill()`:

1. Add `topic_name` attribute to skill counter

### File: `nanobot/agent/loop.py`

Pass `channel` to `AgentRunSpec`:

1. When creating `AgentRunSpec`, add `channel=channel` (variable already available)

## Dashboard Changes

### File: `ws_nanobot/dashboards/nanobot-overview.json`

Add new panels:

1. **LLM Request Duration p50/p95/p99** — timeseries, `nanobot.llm.request.duration`
2. **Token Usage** — stacked timeseries, `nanobot.llm.tokens.prompt` + `nanobot.llm.tokens.completion` by model
3. **LLM Error Rate** — stat panel, errors / total requests
4. **Channel Throughput** — timeseries, `nanobot.channel.messages` by direction
5. **Channel Send Errors** — timeseries, `nanobot.channel.send.errors`
6. **Bus Queue Depth** — timeseries gauge, `nanobot.bus.queue.depth`
7. **Active Sessions** — stat panel, `nanobot.session.active`
8. **Skill Usage / min** — timeseries, `nanobot.skill.loaded` by skill_name (enhanced with topic_name)
9. **Skill Duration p95** — timeseries, `nanobot.skill.duration`
10. **Top Skills** — barchart, sum by skill_name

Layout: reorganize into rows:
- Row 1: Agent metrics (existing tool calls, iterations, stop reason)
- Row 2: LLM metrics (duration, tokens, errors)
- Row 3: Channel + Bus (throughput, errors, queue depth)
- Row 4: Session + Skill (active sessions, skill usage, skill duration)
- Row 5: Summary stats (existing stat panels)

## Verification

1. `pytest tests/` — all existing tests pass
2. `ruff check nanobot/` — no lint errors
3. Start bot, send message, verify metrics appear in VictoriaMetrics
4. Check Grafana dashboard shows data for all panels
5. Verify no performance regression (metrics add < 1ms overhead per call)

## Out of Scope

- Alerting rules (dashboard only per user preference)
- Distributed tracing (spans) beyond existing iteration spans
- Log correlation with trace IDs
- MCP tool connection metrics (deferred)
- Memory/context build duration metrics (deferred)
- Subagent lifecycle metrics (deferred)
