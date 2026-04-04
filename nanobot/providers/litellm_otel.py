"""OTelCallback -- bridges litellm CustomLogger events to nanobot OTel metrics."""

from datetime import datetime

from loguru import logger


class OTelCallback:
    """Bridge litellm CustomLogger callbacks to nanobot OTel metrics.

    Emits the same metric names as the previous AgentRunner-based recording:
    - nanobot.llm.request.duration (histogram, ms)
    - nanobot.llm.tokens.prompt (counter)
    - nanobot.llm.tokens.completion (counter)
    - nanobot.llm.errors (counter)
    """

    def __init__(self) -> None:
        self._duration = None
        self._prompt_tokens = None
        self._completion_tokens = None
        self._errors = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        try:
            from nanobot.observability.otel import get_meter

            meter = get_meter()
            if not meter:
                return
            self._duration = meter.create_histogram(
                "nanobot.llm.request.duration",
                description="LLM request duration in milliseconds",
                unit="ms",
            )
            self._prompt_tokens = meter.create_counter(
                "nanobot.llm.tokens.prompt",
                description="LLM prompt tokens consumed",
                unit="tokens",
            )
            self._completion_tokens = meter.create_counter(
                "nanobot.llm.tokens.completion",
                description="LLM completion tokens consumed",
                unit="tokens",
            )
            self._errors = meter.create_counter(
                "nanobot.llm.errors",
                description="LLM request errors",
            )
        except Exception:
            logger.debug("OTelCallback: failed to initialize metrics")

    async def async_log_success_event(
        self, kwargs: dict, response_obj, start_time: datetime, end_time: datetime
    ) -> None:
        model = kwargs.get("model", "unknown")
        duration_ms = (end_time - start_time).total_seconds() * 1000

        try:
            if self._duration:
                self._duration.record(duration_ms, attributes={"model": model})
        except Exception:
            logger.debug("OTelCallback: failed to record duration")

        usage = getattr(response_obj, "usage", None) if response_obj else None
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        try:
            if self._prompt_tokens:
                self._prompt_tokens.add(prompt_tokens, attributes={"model": model, "type": "prompt"})
        except Exception:
            logger.debug("OTelCallback: failed to record prompt tokens")

        try:
            if self._completion_tokens:
                self._completion_tokens.add(
                    completion_tokens, attributes={"model": model, "type": "completion"}
                )
        except Exception:
            logger.debug("OTelCallback: failed to record completion tokens")

    async def async_log_failure_event(
        self, kwargs: dict, response_obj, start_time: datetime, end_time: datetime
    ) -> None:
        model = kwargs.get("model", "unknown")
        error_type = type(response_obj).__name__ if response_obj else "unknown"

        try:
            if self._errors:
                self._errors.add(
                    1, attributes={"model": model, "error_type": error_type}
                )
        except Exception:
            logger.debug("OTelCallback: failed to record error")
