"""Markdown block parser for smart message splitting.

Parses markdown text into typed structural blocks (code, table, list, etc.)
so that the smart message splitter can avoid breaking formatting boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class BlockType(Enum):
    """Type of a structural markdown block."""

    PARAGRAPH = "paragraph"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    BLOCKQUOTE = "blockquote"
    HEADING = "heading"
    DIAGRAM = "diagram"


@dataclass(frozen=True)
class Block:
    """A contiguous region of the same markdown block type."""

    type: BlockType
    content: str
    lang: str = ""
    diagram_type: str = ""


# Languages recognized as diagram types
_DIAGRAM_LANGS = frozenset({"mermaid", "graphviz", "plantuml", "d2"})

# Box-drawing characters used to detect ASCII art in paragraphs
_BOX_DRAWING_CHARS = frozenset("╔╗╚╝║═╠╣╦╩╬┌┐└┘│─├┤┬┴┼▶◀►◄▸◂▴▾")

# Compiled patterns
_RE_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_RE_LIST_MARKER = re.compile(r"^[-*]\s+")
_RE_LIST_CONTINUATION = re.compile(r"^\s+\S")
_RE_BLOCKQUOTE = re.compile(r"^>\s?")
_RE_TABLE_ROW = re.compile(r"^\|.+\|$")
_RE_FENCED_OPEN = re.compile(r"^```(\S*)")
_RE_FENCED_CLOSE = re.compile(r"^```")


def parse_blocks(text: str) -> list[Block]:
    """Parse markdown *text* into a list of typed structural blocks.

    Returns blocks in document order.  Adjacent blank lines are ignored.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Skip blank lines
        if not line.strip():
            i += 1
            continue

        # --- Fenced code block / diagram ---
        m = _RE_FENCED_OPEN.match(line)
        if m:
            lang = m.group(1).lower()
            i += 1
            code_lines: list[str] = []
            while i < n and not _RE_FENCED_CLOSE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            # consume closing fence
            if i < n:
                i += 1

            content = "\n".join(code_lines)

            if lang in _DIAGRAM_LANGS:
                blocks.append(Block(type=BlockType.DIAGRAM, content=content, diagram_type=lang))
            else:
                blocks.append(Block(type=BlockType.CODE, content=content, lang=lang))
            continue

        # --- Heading ---
        m = _RE_HEADING.match(line)
        if m:
            blocks.append(Block(type=BlockType.HEADING, content=m.group(1)))
            i += 1
            continue

        # --- Table ---
        if _RE_TABLE_ROW.match(line):
            table_lines: list[str] = [line]
            i += 1
            while i < n and _RE_TABLE_ROW.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.TABLE, content="\n".join(table_lines)))
            continue

        # --- Blockquote ---
        if _RE_BLOCKQUOTE.match(line):
            bq_lines: list[str] = [line]
            i += 1
            while i < n and _RE_BLOCKQUOTE.match(lines[i]):
                bq_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.BLOCKQUOTE, content="\n".join(bq_lines)))
            continue

        # --- List ---
        if _RE_LIST_MARKER.match(line):
            list_lines: list[str] = [line]
            i += 1
            while i < n and (_RE_LIST_MARKER.match(lines[i]) or _RE_LIST_CONTINUATION.match(lines[i])):
                list_lines.append(lines[i])
                i += 1
            blocks.append(Block(type=BlockType.LIST, content="\n".join(list_lines)))
            continue

        # --- Paragraph ---
        para_lines: list[str] = [line]
        i += 1
        while i < n:
            peek = lines[i]
            # Blank line ends paragraph
            if not peek.strip():
                break
            # Any structural element ends the paragraph
            if (
                _RE_FENCED_OPEN.match(peek)
                or _RE_HEADING.match(peek)
                or _RE_TABLE_ROW.match(peek)
                or _RE_BLOCKQUOTE.match(peek)
                or _RE_LIST_MARKER.match(peek)
            ):
                break
            para_lines.append(peek)
            i += 1

        content = "\n".join(para_lines)
        # Detect box-drawing ASCII art: if >= 3 lines contain box-drawing
        # characters, treat the paragraph as a DIAGRAM block.
        box_lines = sum(1 for line in para_lines if any(c in _BOX_DRAWING_CHARS for c in line))
        if box_lines >= 3:
            blocks.append(Block(type=BlockType.DIAGRAM, content=content, diagram_type="ascii"))
        else:
            blocks.append(Block(type=BlockType.PARAGRAPH, content=content))

    return blocks
