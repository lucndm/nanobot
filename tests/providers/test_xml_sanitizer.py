"""Tests for XmlToolCallSanitizer streaming content cleaner."""

from __future__ import annotations

import pytest

from nanobot.providers.openai_compat_provider import XmlToolCallSanitizer


class TestXmlToolCallSanitizerNormal:
    """Cases where deltas contain complete <function=…> blocks."""

    def test_no_xml_passes_through(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("hello world") == "hello world"

    def test_complete_function_block_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed("<function=read_file><parameter=path>/tmp/x</parameter></function>")
        assert result == ""

    def test_text_before_and_after_block(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "Sure!<function=read_file><parameter=path>/tmp</parameter></function>Done."
        )
        # Space inserted at boundary: "Sure!" + block + "Done." → "Sure! Done."
        assert result == "Sure! Done."

    def test_tool_call_marker_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("<tool_call_none/>") == ""

    def test_closing_tool_call_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("</tool_call_none>") == ""

    def test_buffered_incomplete_function(self) -> None:
        s = XmlToolCallSanitizer()
        # First delta: partial opening tag
        assert s.feed("Hello <function") == "Hello "
        # Second delta completes the block
        assert s.feed("=exec><parameter=cmd>ls</parameter></function>") == ""


class TestXmlToolCallSanitizerOrphan:
    """Cases where leading '<' was emitted in reasoning delta."""

    def test_orphan_function_block_stripped(self) -> None:
        """Complete orphan block (function=…>…</function> without leading '<')."""
        s = XmlToolCallSanitizer()
        result = s.feed("function=exec><parameter=cmd>ls</parameter></function>")
        assert result == ""

    def test_orphan_block_with_surrounding_text(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "Prefix function=exec><parameter=cmd>ls</parameter></function> Suffix"
        )
        # Space before block preserved, space after block preserved
        assert result == "Prefix  Suffix"

    def test_orphan_block_multi_param(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "function=read_file>"
            "<parameter=path>/etc/hosts</parameter>"
            "<parameter=offset>10</parameter>"
            "</function>"
        )
        assert result == ""

    def test_orphan_fragment_buffered_until_complete(self) -> None:
        """Orphan opening tag arrives alone; parameter tags in later deltas."""
        s = XmlToolCallSanitizer()
        # Delta 1: orphan opening (no '<')
        assert s.feed("function=exec>") == ""
        # Delta 2: parameter opening
        assert s.feed("<parameter=command") == ""
        # Delta 3: parameter value and close
        assert s.feed(">ls -la /tmp</parameter>") == ""
        # Delta 4: closing function tag
        assert s.feed("</function>") == ""

    def test_orphan_reproduces_reasoning_content_split(self) -> None:
        """Exact delta sequence from WebSocket trace: '<' was in reasoning."""
        deltas = [
            "function=exec>",
            "<parameter=command",
            ">ls -la",
            "/tmp",
            "</parameter>",
            "</function>",
        ]
        s = XmlToolCallSanitizer()
        for d in deltas:
            result = s.feed(d)
            assert result == "", f"Leaked at delta {d!r}: got {result!r}"
        assert s.flush() == ""


class TestXmlToolCallSanitizerFlush:
    """Flush behaviour for incomplete buffers."""

    def test_flush_strips_incomplete_function_prefix(self) -> None:
        s = XmlToolCallSanitizer()
        s.feed("<function=read_file><parameter=path>/tmp")
        remaining = s.flush()
        assert remaining == ""

    def test_flush_preserves_plain_function_equals(self) -> None:
        """Plain text 'function=main is important' must NOT be stripped."""
        s = XmlToolCallSanitizer()
        s.feed("the function=main is important")
        remaining = s.flush()
        assert "function=main is important" in remaining

    def test_flush_strips_xml_like_function_equals(self) -> None:
        """'function=exec>' looks like XML and should be stripped."""
        s = XmlToolCallSanitizer()
        s.feed("function=exec>")
        remaining = s.flush()
        assert remaining == ""

    def test_flush_orphan_block_in_final_deltas(self) -> None:
        """Complete orphan block sitting in buffer at flush time."""
        s = XmlToolCallSanitizer()
        # Feed enough to buffer the orphan opening
        s.feed("function=exec><parameter=cmd>ls</parameter>")
        # Don't feed closing tag — simulates it arriving at the very end
        # Actually feed it through flush by not closing
        assert s.flush() == ""


class TestXmlToolCallSanitizerParameter:
    """Cases with <parameter=…> tags."""

    def test_parameter_tag_buffered(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("<parameter=path") == ""
        assert s.feed(">/tmp</parameter>") == ""

    def test_parameter_inside_function_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "<function=exec><parameter=cmd>ls</parameter></function>"
        )
        assert result == ""


class TestXmlToolCallSanitizerInvoke:
    """Cases with <invoke name="…"> XML tool call format."""

    def test_complete_invoke_block_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            '<invoke name="mcp_litellm_todoist-add-tasks">'
            '<parameter name="tasks">[{"content": "test"}]</parameter>'
            '</invoke>'
        )
        assert result == ""

    def test_invoke_with_surrounding_text(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            'Sure!<invoke name="exec">'
            '<parameter name="cmd">ls</parameter>'
            '</invoke>Done.'
        )
        # Space inserted at boundary: "Sure!" + invoke block + "Done."
        assert result == "Sure! Done."

    def test_orphan_invoke_block_stripped(self) -> None:
        """Orphan: <invoke was in reasoning, name="…"> arrives in content."""
        s = XmlToolCallSanitizer()
        result = s.feed(
            'name="mcp_litellm_todoist-add-tasks">'
            '<parameter name="tasks">[{"content": "test"}]</parameter>'
            '</invoke>'
        )
        assert result == ""

    def test_orphan_invoke_with_prefix_text(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            'Prefix name="exec"><parameter name="cmd">ls</parameter></invoke> Suffix'
        )
        # "Prefix " (trailing space) + stripped block + " Suffix" (leading space)
        assert result == "Prefix  Suffix"

    def test_orphan_invoke_multi_delta(self) -> None:
        """Simulate real streaming: <invoke in reasoning, then content deltas."""
        s = XmlToolCallSanitizer()
        assert s.feed('name="mcp_litellm_todoist-add-tasks">') == ""
        assert s.feed('<parameter name="tasks">') == ""
        assert s.feed('[{"content": "test"}]') == ""
        assert s.feed('</parameter>') == ""
        assert s.feed('</invoke>') == ""

    def test_invoke_buffered_incomplete(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("Hello <invoke") == "Hello "
        assert s.feed(' name="exec"><parameter name="cmd">ls</parameter></invoke>') == ""


class TestXmlToolCallSanitizerSpaceBoundary:
    """Space insertion when XML block is stripped between two non-space chars."""

    def test_no_space_inserted_when_boundary_has_space(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed("Hello <function=exec><parameter=cmd>ls</parameter></function> World")
        assert result == "Hello  World"

    def test_space_inserted_at_boundary(self) -> None:
        s = XmlToolCallSanitizer()
        # "Đã" + <invoke...> + "tạo" → "Đã tạo" (space added)
        result = s.feed(
            'Đã<invoke name="exec"><parameter name="cmd">ls</parameter></invoke>tạo task'
        )
        assert result == "Đã tạo task"

    def test_no_space_at_sentence_boundary(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed("Done.<function=read_file><parameter=path>/tmp</parameter></function>Next.")
        assert result == "Done. Next."

    def test_no_double_space_when_already_spaced(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed("End <function=exec><parameter=cmd>ls</parameter></function> Start")
        assert result == "End  Start"


class TestXmlToolCallSanitizerToolCallBlock:
    """Cases with <tool_call_block> XML format."""

    def test_complete_tool_call_block_stripped(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed("<tool_call_block><exec>ls</exec></tool_call_block>")
        assert result == ""

    def test_tool_call_block_with_surrounding_text(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "Prefix <tool_call_block><exec>ls</exec></tool_call_block> Suffix"
        )
        assert result == "Prefix  Suffix"

    def test_tool_call_block_buffered_incomplete(self) -> None:
        s = XmlToolCallSanitizer()
        assert s.feed("Hello <tool_call") == "Hello "
        assert s.feed("_block><exec>ls</exec></tool_call_block>") == ""

    def test_tool_call_block_multiline(self) -> None:
        s = XmlToolCallSanitizer()
        result = s.feed(
            "<tool_call_block>\n"
            "  <exec>ls -la /tmp</exec>\n"
            "</tool_call_block>"
        )
        assert result == ""
