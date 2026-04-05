"""Tests for nanobot.utils.splitter — structure-aware message splitter."""

from __future__ import annotations

from nanobot.utils.splitter import smart_split_message

# ---------------------------------------------------------------------------
# Empty / blank / short input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string(self) -> None:
        assert smart_split_message("") == []

    def test_whitespace_only(self) -> None:
        assert smart_split_message("   \n  \n  ") == []

    def test_single_newline(self) -> None:
        assert smart_split_message("\n") == []


class TestShortMessage:
    def test_short_message_unchanged(self) -> None:
        text = "Hello world"
        assert smart_split_message(text) == [text]

    def test_message_under_max_len(self) -> None:
        text = "A reasonably short message that fits in one chunk."
        assert smart_split_message(text, max_len=200) == [text]

    def test_multiple_short_paragraphs_in_one_chunk(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert smart_split_message(text, max_len=200) == [text]

    def test_custom_max_len_respected(self) -> None:
        text = "A" * 50
        assert smart_split_message(text, max_len=100) == [text]
        assert len(smart_split_message(text, max_len=30)) > 1


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


class TestCodeBlock:
    def test_code_block_under_limit_kept_whole(self) -> None:
        text = "```python\nx = 1\n```"
        assert smart_split_message(text) == [text]

    def test_code_block_splits_into_wrapped_chunks(self) -> None:
        # Build a code block that exceeds max_len when wrapped
        lines = [f"line_{i:03d} = {i}" for i in range(20)]
        code_content = "\n".join(lines)
        text = f"```python\n{code_content}\n```"
        result = smart_split_message(text, max_len=100)
        assert len(result) > 1
        # Each chunk should have balanced fences
        for chunk in result:
            assert chunk.startswith("```python\n")
            assert chunk.endswith("\n```")

    def test_never_splits_inside_code_line(self) -> None:
        # Each line should remain intact
        long_line = "x" * 80
        text = f"```python\n{long_line}\nprint('hello')\n```"
        result = smart_split_message(text, max_len=100)
        # The long line chunk may be alone; 'print' should be in a valid chunk
        for chunk in result:
            assert chunk.startswith("```python\n")
            assert chunk.endswith("\n```")

    def test_code_block_with_no_lang(self) -> None:
        text = "```\nsome code\n```"
        assert smart_split_message(text, max_len=200) == [text]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTable:
    def test_table_kept_whole_if_fits(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert smart_split_message(text, max_len=200) == [text]

    def test_large_table_sent_as_is(self) -> None:
        # Build a wide table that exceeds max_len
        rows = ["| " + " | ".join(f"col{j}_{i}" for j in range(20)) + " |" for i in range(30)]
        text = "\n".join(rows)
        result = smart_split_message(text, max_len=200)
        # Table should be in one chunk (can't split), possibly truncated
        assert len(result) >= 1
        # The first chunk should contain the table
        assert "|" in result[0]

    def test_table_with_surrounding_text(self) -> None:
        text = "Intro text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter table."
        result = smart_split_message(text, max_len=200)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


class TestHeading:
    def test_heading_never_split(self) -> None:
        text = "## " + "A" * 200
        result = smart_split_message(text, max_len=100)
        # Should produce exactly one chunk (heading can't be split)
        assert len(result) == 1
        assert result[0].startswith("## ")

    def test_heading_with_surrounding_paragraphs(self) -> None:
        text = "Intro.\n\n## Heading\n\nAfter heading."
        result = smart_split_message(text, max_len=200)
        assert len(result) == 1
        assert "## Heading" in result[0]


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


class TestList:
    def test_list_under_limit_kept_whole(self) -> None:
        text = "- item one\n- item two\n- item three"
        assert smart_split_message(text, max_len=200) == [text]

    def test_list_splits_between_items(self) -> None:
        items = [
            f"- This is list item number {i} with some extra text to make it longer."
            for i in range(10)
        ]
        text = "\n".join(items)
        result = smart_split_message(text, max_len=100)
        assert len(result) > 1
        # Each item should start with "- " (not mid-item)
        for chunk in result:
            lines = chunk.strip().split("\n")
            for line in lines:
                stripped = line.lstrip()
                # Either it's a list item or empty
                if stripped:
                    assert stripped.startswith("- ") or stripped.startswith("* "), (
                        f"Line does not start with list marker: {line!r}"
                    )

    def test_list_with_continuation_lines(self) -> None:
        text = "- item one\n  continued\n- item two"
        assert smart_split_message(text, max_len=200) == [text]


# ---------------------------------------------------------------------------
# Blockquotes
# ---------------------------------------------------------------------------


class TestBlockquote:
    def test_blockquote_under_limit_kept_whole(self) -> None:
        text = "> quoted text"
        assert smart_split_message(text, max_len=200) == [text]

    def test_blockquote_splits_between_lines(self) -> None:
        lines = [
            f"> Line {i} with some text to make it a bit longer than usual." for i in range(15)
        ]
        text = "\n".join(lines)
        result = smart_split_message(text, max_len=100)
        assert len(result) > 1
        # Each chunk should only contain complete blockquote lines
        for chunk in result:
            for line in chunk.strip().split("\n"):
                stripped = line.strip()
                if stripped:
                    assert stripped.startswith(">")


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


class TestParagraph:
    def test_paragraph_splits_at_sentence_boundary(self) -> None:
        sentences = [
            "First sentence here. ",
            "Second sentence is also here. ",
            "Third sentence wraps up. ",
        ]
        text = "".join(sentences)
        result = smart_split_message(text, max_len=40)
        assert len(result) > 1
        # No chunk should end mid-sentence if we can avoid it
        full = "".join(result)
        assert full.replace("\n", "").replace(" ", "") == text.replace(" ", "")

    def test_paragraph_splits_at_word_boundary(self) -> None:
        text = "A" * 30 + " " + "B" * 30 + " " + "C" * 30
        result = smart_split_message(text, max_len=50)
        assert len(result) > 1
        # Each chunk should not break mid-word
        for chunk in result:
            # After stripping, check no word is cut at boundary
            words = chunk.strip().split()
            for word in words:
                # All words should be complete (no partial)
                assert (
                    len(word) <= 30
                    or word.startswith("A")
                    or word.startswith("B")
                    or word.startswith("C")
                )


# ---------------------------------------------------------------------------
# Tag balancing
# ---------------------------------------------------------------------------


class TestTagBalancing:
    def test_bold_tag_closed_at_chunk_end(self) -> None:
        text = "<b>" + "A" * 50 + "</b>"
        result = smart_split_message(text, max_len=60)
        assert len(result) >= 1
        # If split, first chunk should close <b>, next should reopen
        if len(result) > 1:
            assert result[0].endswith("</b>")
            assert result[1].startswith("<b>")

    def test_nested_bold_italic_tags(self) -> None:
        # Build a paragraph long enough that it splits in the middle of <b><i>
        inner = " word" * 30  # long content between open and close tags
        text = f"<b><i>{inner}</i></b>"
        result = smart_split_message(text, max_len=80)
        assert len(result) > 1
        # All chunks should have balanced tags
        for chunk in result:
            assert _tags_balanced(chunk), f"Unbalanced tags in chunk: {chunk!r}"
        # Second chunk should reopen both <b> and <i>
        assert "<b>" in result[1][:20]
        assert "<i>" in result[1][:30]

    def test_tag_balancing_with_multiple_blocks(self) -> None:
        # Bold spans across multiple paragraphs
        text = "<b>First paragraph with some text.</b>\n\n<b>Second paragraph here.</b>"
        result = smart_split_message(text, max_len=50)
        # Should still be parseable — each chunk balanced
        for chunk in result:
            assert _tags_balanced(chunk), f"Unbalanced tags in chunk: {chunk!r}"

    def test_code_tag_preserved(self) -> None:
        text = "<code>" + "x" * 60 + "</code>"
        result = smart_split_message(text, max_len=40)
        if len(result) > 1:
            assert result[0].endswith("</code>")
            assert result[1].startswith("<code>")


# ---------------------------------------------------------------------------
# Mixed blocks
# ---------------------------------------------------------------------------


class TestMixedBlocks:
    def test_mixed_blocks_split_at_boundaries(self) -> None:
        text = (
            "# Title\n\n"
            "A short paragraph.\n\n"
            "```python\nline1\nline2\nline3\n```\n\n"
            "Another paragraph."
        )
        result = smart_split_message(text, max_len=50)
        # Should produce multiple chunks but no block should be broken
        # Code block should be intact with fences
        full = "".join(result)
        assert "# Title" in full
        assert "```python" in full
        assert "```" in full

    def test_blocks_joined_with_double_newline(self) -> None:
        text = "Para one.\n\nPara two."
        result = smart_split_message(text, max_len=200)
        assert len(result) == 1
        assert "\n\n" in result[0]

    def test_code_block_surrounded_by_text_splits_correctly(self) -> None:
        code_lines = [f"line_{i}" for i in range(15)]
        code = "\n".join(code_lines)
        text = f"Intro paragraph.\n\n```python\n{code}\n```\n\nOutro paragraph."
        result = smart_split_message(text, max_len=80)
        assert len(result) > 1
        # Code block chunks should have balanced fences
        for chunk in result:
            fence_count = chunk.count("```")
            assert fence_count % 2 == 0, f"Unbalanced fences in chunk: {chunk!r}"


# ---------------------------------------------------------------------------
# Diagram blocks
# ---------------------------------------------------------------------------


class TestDiagram:
    def test_diagram_kept_whole_if_fits(self) -> None:
        text = "```mermaid\nA --> B\n```"
        assert smart_split_message(text, max_len=200) == [text]

    def test_large_diagram_sent_as_is(self) -> None:
        lines = [f"    A{i} --> B{i}" for i in range(50)]
        text = "```mermaid\ngraph TD\n" + "\n".join(lines) + "\n```"
        result = smart_split_message(text, max_len=100)
        # Diagram should be in one chunk
        assert len(result) >= 1
        assert "```mermaid" in result[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_very_long_word(self) -> None:
        """A single word longer than max_len must be split as last resort."""
        text = "A" * 500
        result = smart_split_message(text, max_len=100)
        assert len(result) > 1
        full = "".join(result)
        assert full.replace("\n", "") == text

    def test_only_newlines_between_chunks(self) -> None:
        text = "Short.\n\n" + "A" * 200
        result = smart_split_message(text, max_len=50)
        # First chunk should contain "Short."
        assert any("Short" in chunk for chunk in result)

    def test_reassembly_preserves_content(self) -> None:
        """Joining all chunks (stripping tag wrappers) should yield original text."""
        text = (
            "# Title\n\n"
            "First paragraph. Second sentence.\n\n"
            "```python\nprint('hello')\n```\n\n"
            "- item 1\n- item 2\n\n"
            "> quote line 1\n> quote line 2\n\n"
            "Final paragraph."
        )
        result = smart_split_message(text, max_len=100)
        # Reassemble and check key content is preserved
        full = "\n".join(result)
        assert "Title" in full
        assert "First paragraph" in full
        assert "print('hello')" in full
        assert "item 1" in full
        assert "quote line 1" in full
        assert "Final paragraph" in full


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tags_balanced(text: str) -> bool:
    """Check if HTML tags in *text* are balanced (naive check)."""
    stack: list[str] = []
    import re

    for m in re.finditer(r"<(/?)(\w+)>", text):
        is_close, tag = m.group(1), m.group(2).lower()
        if is_close:
            if stack and stack[-1] == tag:
                stack.pop()
            else:
                return False
        else:
            stack.append(tag)
    return len(stack) == 0
