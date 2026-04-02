"""Tests for skill duration/error metrics in SkillsLoader + topic_name on skill counter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _setup_otel_mocks():
    """Create mock meter/tracer and patch get_meter/get_tracer.

    Returns (mock_meter, mock_counter, mock_histogram, mock_tracer) where
    create_counter and create_histogram are ordered by call order.
    """
    mock_meter = MagicMock()
    mock_tracer = MagicMock()

    # Return distinct mocks per instrument so we can assert on specific ones.
    mock_histogram = MagicMock()
    mock_counter = MagicMock()

    def _create_counter(*a, **kw):
        return mock_counter

    def _create_histogram(*a, **kw):
        return mock_histogram

    mock_meter.create_counter.side_effect = _create_counter
    mock_meter.create_histogram.side_effect = _create_histogram

    return mock_meter, mock_counter, mock_histogram, mock_tracer


# ---------------------------------------------------------------------------
# Test 1: load_skill records duration histogram
# ---------------------------------------------------------------------------

def test_skill_load_records_duration(tmp_path):
    """load_skill should record nanobot.skill.duration histogram with skill_name attr."""
    from nanobot.agent.skills import SkillsLoader

    # Create a workspace skill
    skill_dir = tmp_path / "skills" / "my_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# My Skill\nSome content.\n")

    mock_meter, mock_counter, mock_histogram, mock_tracer = _setup_otel_mocks()

    with (
        patch("nanobot.observability.otel.get_meter", return_value=mock_meter),
        patch("nanobot.observability.otel.get_tracer", return_value=mock_tracer),
    ):
        loader = SkillsLoader(tmp_path)
        result = loader.load_skill("my_skill")

    assert result is not None
    assert "# My Skill" in result

    # The histogram should have been recorded with skill_name attr
    mock_histogram.record.assert_called()
    call_kwargs = mock_histogram.record.call_args.kwargs
    assert call_kwargs["attributes"]["skill_name"] == "my_skill"
    # Duration value should be a positive float (ms)
    duration_val = mock_histogram.record.call_args.args[0] if mock_histogram.record.call_args.args else call_kwargs.get("value", 0)
    assert duration_val >= 0


# ---------------------------------------------------------------------------
# Test 2: load_skill nonexistent does NOT increment error counter
# ---------------------------------------------------------------------------

def test_skill_load_nonexistent_no_error(tmp_path):
    """Loading a nonexistent skill should return None and NOT increment the error counter."""
    from nanobot.agent.skills import SkillsLoader

    mock_meter, mock_counter, mock_histogram, mock_tracer = _setup_otel_mocks()

    with (
        patch("nanobot.observability.otel.get_meter", return_value=mock_meter),
        patch("nanobot.observability.otel.get_tracer", return_value=mock_tracer),
    ):
        loader = SkillsLoader(tmp_path)
        result = loader.load_skill("does_not_exist")

    assert result is None

    # The error counter should NOT have been incremented
    # We need to check the specific counter for nanobot.skill.errors
    # Since our mock reuses the same counter for all create_counter calls,
    # we check that add was NOT called with error-related attributes.
    for call in mock_counter.add.call_args_list:
        attrs = call.kwargs.get("attributes", call[1].get("attributes", {})) if call.kwargs else {}
        if call.args:
            # positional arg is the value
            pass
        # If any call has attributes that look like skill errors, fail
        assert not (isinstance(attrs, dict) and attrs.get("skill_name") == "does_not_exist" and attrs.get("error") == "true"), \
            "Error counter should NOT be incremented for nonexistent skill"


# ---------------------------------------------------------------------------
# Test 3: record_skill includes topic_name attribute
# ---------------------------------------------------------------------------

def test_record_skill_includes_topic_name():
    """OTelHook.record_skill should include topic_name in counter attributes."""
    from nanobot.observability.hook import OTelHook

    mock_meter = MagicMock()
    mock_counter = MagicMock()
    mock_histogram = MagicMock()
    mock_tracer = MagicMock()

    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    with (
        patch("nanobot.observability.otel.get_meter", return_value=mock_meter),
        patch("nanobot.observability.otel.get_tracer", return_value=mock_tracer),
    ):
        hook = OTelHook(channel="telegram", chat_id="123", topic_name="finance")
        hook.record_skill("firefly_tools")

    assert mock_counter.add.call_count >= 1
    call_kwargs = mock_counter.add.call_args.kwargs
    attrs = call_kwargs["attributes"]
    assert attrs["skill_name"] == "firefly_tools"
    assert attrs["channel"] == "telegram"
    assert attrs["topic_name"] == "finance"
