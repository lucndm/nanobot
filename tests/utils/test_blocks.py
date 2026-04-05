from __future__ import annotations

import pytest

from nanobot.utils.blocks import Block, BlockType, parse_blocks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(
    btype: BlockType,
    content: str,
    *,
    lang: str = "",
    diagram_type: str = "",
) -> Block:
    return Block(type=btype, content=content, lang=lang, diagram_type=diagram_type)


# ---------------------------------------------------------------------------
# Empty / blank input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string(self) -> None:
        assert parse_blocks("") == []

    def test_whitespace_only(self) -> None:
        assert parse_blocks("   \n  \n  ") == []

    def test_newlines_only(self) -> None:
        assert parse_blocks("\n\n\n") == []


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


class TestParagraph:
    def test_single_paragraph(self) -> None:
        text = "Hello world"
        result = parse_blocks(text)
        assert result == [_block(BlockType.PARAGRAPH, "Hello world")]

    def test_multiline_paragraph(self) -> None:
        text = "Line one\nLine two\nLine three"
        result = parse_blocks(text)
        assert result == [_block(BlockType.PARAGRAPH, "Line one\nLine two\nLine three")]

    def test_multiple_paragraphs(self) -> None:
        text = "First paragraph\n\nSecond paragraph"
        result = parse_blocks(text)
        assert result == [
            _block(BlockType.PARAGRAPH, "First paragraph"),
            _block(BlockType.PARAGRAPH, "Second paragraph"),
        ]

    def test_paragraph_strips_trailing_newlines(self) -> None:
        text = "Hello\n\n"
        result = parse_blocks(text)
        assert result == [_block(BlockType.PARAGRAPH, "Hello")]


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


class TestCodeBlock:
    def test_code_block_with_language(self) -> None:
        text = '```python\nprint("hello")\n```'
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.CODE
        assert result[0].content == 'print("hello")'
        assert result[0].lang == "python"

    def test_code_block_without_language(self) -> None:
        text = "```\nsome code\n```"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.CODE
        assert result[0].content == "some code"
        assert result[0].lang == ""

    def test_code_block_multiline(self) -> None:
        text = "```python\ndef foo():\n    return 42\n```"
        result = parse_blocks(text)
        assert result[0].type == BlockType.CODE
        assert result[0].content == "def foo():\n    return 42"
        assert result[0].lang == "python"

    def test_code_block_empty(self) -> None:
        text = "```\n```"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.CODE
        assert result[0].content == ""

    def test_consecutive_code_blocks(self) -> None:
        text = '```python\nprint("a")\n```\n\n```bash\necho b\n```'
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.CODE
        assert result[0].lang == "python"
        assert result[1].type == BlockType.CODE
        assert result[1].lang == "bash"


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------


class TestDiagram:
    @pytest.mark.parametrize("lang", ["mermaid", "graphviz", "plantuml", "d2"])
    def test_diagram_type_detected(self, lang: str) -> None:
        text = f"```{lang}\ngraph A --> B\n```"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.DIAGRAM
        assert result[0].diagram_type == lang
        assert result[0].lang == ""

    def test_mermaid_diagram(self) -> None:
        text = "```mermaid\nsequenceDiagram\n    A->>B: Hello\n```"
        result = parse_blocks(text)
        assert result[0].type == BlockType.DIAGRAM
        assert result[0].diagram_type == "mermaid"
        assert "sequenceDiagram" in result[0].content


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTable:
    def test_simple_table(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.TABLE
        assert result[0].content == "| A | B |\n|---|---|\n| 1 | 2 |"

    def test_table_separated_by_blank_line(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n\nSome text"
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.TABLE
        assert result[1].type == BlockType.PARAGRAPH

    def test_table_must_have_pipe_at_start_and_end(self) -> None:
        text = "not a | table |\n| real | table |"
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.PARAGRAPH
        assert result[1].type == BlockType.TABLE


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


class TestHeading:
    @pytest.mark.parametrize(
        ("marker", "level_text"),
        [("#", "Title"), ("##", "Subtitle"), ("###", "Section"), ("######", "Deep")],
    )
    def test_heading_levels(self, marker: str, level_text: str) -> None:
        text = f"{marker} {level_text}"
        result = parse_blocks(text)
        assert result == [_block(BlockType.HEADING, level_text)]

    def test_heading_content_no_hash(self) -> None:
        text = "## My Heading"
        result = parse_blocks(text)
        assert result[0].content == "My Heading"

    def test_heading_multiple(self) -> None:
        text = "# Title\n## Subtitle"
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.HEADING
        assert result[0].content == "Title"
        assert result[1].type == BlockType.HEADING
        assert result[1].content == "Subtitle"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


class TestList:
    def test_unordered_list(self) -> None:
        text = "- item one\n- item two\n- item three"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.LIST
        assert result[0].content == "- item one\n- item two\n- item three"

    def test_star_list(self) -> None:
        text = "* item one\n* item two"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.LIST

    def test_list_with_continuation(self) -> None:
        text = "- item one\n  continued\n- item two"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.LIST
        assert "continued" in result[0].content

    def test_list_separated_by_paragraph(self) -> None:
        text = "- item\n\nParagraph text"
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.LIST
        assert result[1].type == BlockType.PARAGRAPH


# ---------------------------------------------------------------------------
# Blockquotes
# ---------------------------------------------------------------------------


class TestBlockquote:
    def test_simple_blockquote(self) -> None:
        text = "> quoted text"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.BLOCKQUOTE
        assert result[0].content == "> quoted text"

    def test_multiline_blockquote(self) -> None:
        text = "> line one\n> line two"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.BLOCKQUOTE
        assert result[0].content == "> line one\n> line two"

    def test_blockquote_without_space(self) -> None:
        text = ">no space"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.BLOCKQUOTE

    def test_blockquote_separated_by_paragraph(self) -> None:
        text = "> quoted\n\nnot quoted"
        result = parse_blocks(text)
        assert len(result) == 2
        assert result[0].type == BlockType.BLOCKQUOTE
        assert result[1].type == BlockType.PARAGRAPH


# ---------------------------------------------------------------------------
# Mixed content
# ---------------------------------------------------------------------------


class TestMixedContent:
    def test_all_block_types(self) -> None:
        text = (
            "# Title\n"
            "\n"
            "Some intro paragraph.\n"
            "\n"
            "```python\nx = 1\n```\n"
            "\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| 1     | 2     |\n"
            "\n"
            "- list item 1\n"
            "- list item 2\n"
            "\n"
            "> A blockquote\n"
            "\n"
            "```mermaid\nA --> B\n```\n"
            "\n"
            "Final paragraph."
        )
        result = parse_blocks(text)
        assert len(result) == 8
        assert result[0].type == BlockType.HEADING
        assert result[0].content == "Title"
        assert result[1].type == BlockType.PARAGRAPH
        assert result[2].type == BlockType.CODE
        assert result[2].lang == "python"
        assert result[3].type == BlockType.TABLE
        assert result[4].type == BlockType.LIST
        assert result[5].type == BlockType.BLOCKQUOTE
        assert result[6].type == BlockType.DIAGRAM
        assert result[6].diagram_type == "mermaid"
        assert result[7].type == BlockType.PARAGRAPH

    def test_code_block_surrounded_by_paragraphs(self) -> None:
        text = "Before code\n\n```\ncode\n```\n\nAfter code"
        result = parse_blocks(text)
        assert len(result) == 3
        assert result[0].type == BlockType.PARAGRAPH
        assert result[1].type == BlockType.CODE
        assert result[2].type == BlockType.PARAGRAPH

    def test_inline_code_not_parsed_as_block(self) -> None:
        text = "Use `inline code` here"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.PARAGRAPH


# ---------------------------------------------------------------------------
# Consecutive same-type blocks
# ---------------------------------------------------------------------------


class TestConsecutiveBlocks:
    def test_consecutive_paragraphs_no_blank_line(self) -> None:
        """Lines without blank lines are one paragraph."""
        text = "Line one\nLine two\nLine three"
        result = parse_blocks(text)
        assert len(result) == 1
        assert result[0].type == BlockType.PARAGRAPH

    def test_consecutive_lists(self) -> None:
        text = "- first list\n- item\n\n- second list\n- item"
        result = parse_blocks(text)
        assert len(result) == 2
        assert all(b.type == BlockType.LIST for b in result)

    def test_consecutive_headings(self) -> None:
        text = "# One\n# Two"
        result = parse_blocks(text)
        assert len(result) == 2
        assert all(b.type == BlockType.HEADING for b in result)
