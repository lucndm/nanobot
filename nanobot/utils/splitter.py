"""Structure-aware message splitter with HTML tag balancing.

Splits markdown text into chunks that respect block boundaries (code fences,
list items, headings, tables, etc.) and keeps HTML tags balanced across chunk
boundaries.
"""

from __future__ import annotations

import re

from nanobot.utils.blocks import Block, BlockType, parse_blocks

# HTML tags we track for balancing
_TRACKED_TAGS = frozenset({"b", "i", "s", "code", "pre", "a"})
_RE_OPENTAG = re.compile(r"<(b|i|s|code|pre|a)(?:\s[^>]*)?\s*>", re.IGNORECASE)
_RE_CLOSETAG = re.compile(r"</(b|i|s|code|pre|a)>", re.IGNORECASE)

# Sentence boundaries for paragraph splitting
_RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Word boundary fallback
_RE_WORD_BOUNDARY = re.compile(r"(\s+)")


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def _render_block(block: Block) -> str:
    """Convert a *Block* back to its original markdown representation."""
    match block.type:
        case BlockType.CODE:
            lang = block.lang
            return f"```{lang}\n{block.content}\n```" if lang else f"```\n{block.content}\n```"
        case BlockType.DIAGRAM:
            return f"```{block.diagram_type}\n{block.content}\n```"
        case BlockType.HEADING:
            return f"## {block.content}"
        case _:
            return block.content


# ---------------------------------------------------------------------------
# Block-level splitting
# ---------------------------------------------------------------------------


def _split_code_block(block: Block, max_len: int) -> list[str]:
    """Split a code block at newline boundaries, wrapping each chunk in fences."""
    lang = block.lang
    prefix = f"```{lang}\n" if lang else "```\n"
    fence_overhead = len(prefix) + len("\n```")
    budget = max_len - fence_overhead

    if budget < 1:
        budget = 1

    lines = block.content.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > budget and current_lines:
            chunks.append(prefix + "\n".join(current_lines) + "\n```")
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append(prefix + "\n".join(current_lines) + "\n```")

    return chunks


def _split_list_block(block: Block, max_len: int) -> list[str]:
    """Split a list block between items (lines starting with ``- `` or ``* ``)."""
    lines = block.content.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line starts a new item
        is_new_item = bool(re.match(r"^[-*]\s+", line))

        if is_new_item and current_lines and current_len + len(line) + 1 > max_len:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += len(line) + 1  # +1 for newline

        i += 1

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _split_blockquote_block(block: Block, max_len: int) -> list[str]:
    """Split a blockquote between lines."""
    lines = block.content.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        if current_lines and current_len + len(line) + 1 > max_len:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += len(line) + 1

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _split_paragraph_block(block: Block, max_len: int) -> list[str]:
    """Split a paragraph at sentence boundaries, then word boundaries."""
    content = block.content

    if len(content) <= max_len:
        return [content]

    # Try splitting at sentence boundaries first
    parts = _RE_SENTENCE_END.split(content)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= max_len:
            current = candidate
        else:
            # Current chunk is full; try word-level splitting for the remainder
            if current:
                chunks.append(current)
            # The part itself might be too long — split by words
            current = _split_by_words(part, max_len, chunks)

    if current:
        chunks.append(current)

    return chunks


def _split_by_words(text: str, max_len: int, chunks: list[str]) -> str:
    """Split *text* by word boundaries, appending full chunks to *chunks*.

    Returns the remainder that didn't fill a chunk.
    """
    words = text.split(" ")
    current = ""

    for word in words:
        candidate = (current + " " + word).strip() if current else word
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single word exceeds max_len, it goes as-is (last resort)
            if len(word) > max_len:
                # Split the oversized word character by character
                while word:
                    fit = word[:max_len]
                    chunks.append(fit)
                    word = word[max_len:]
                current = ""
            else:
                current = word

    return current


def _split_block(block: Block, max_len: int) -> list[str]:
    """Split a single block into pieces that each fit within *max_len*.

    Unsplittable block types (table, heading, diagram) are returned as a
    single piece, possibly exceeding *max_len* (last resort).
    """
    rendered = _render_block(block)

    if len(rendered) <= max_len:
        return [rendered]

    match block.type:
        case BlockType.CODE:
            return _split_code_block(block, max_len)
        case BlockType.LIST:
            return _split_list_block(block, max_len)
        case BlockType.BLOCKQUOTE:
            return _split_blockquote_block(block, max_len)
        case BlockType.PARAGRAPH:
            return _split_paragraph_block(block, max_len)
        case _:
            # Table, Heading, Diagram — can't split further
            return [rendered]


# ---------------------------------------------------------------------------
# HTML tag tracking
# ---------------------------------------------------------------------------


def _extract_open_tags(text: str) -> list[str]:
    """Return a list of currently open HTML tags in *text*.

    Parses opening and closing tags in order and returns the stack of
    unclosed tag names (lowercase).
    """
    opens = list(_RE_OPENTAG.finditer(text))
    closes = list(_RE_CLOSETAG.finditer(text))

    # Merge events by position
    events: list[tuple[int, bool, str]] = []
    for m in opens:
        events.append((m.start(), False, m.group(1).lower()))
    for m in closes:
        events.append((m.start(), True, m.group(1).lower()))
    events.sort(key=lambda e: e[0])

    stack: list[str] = []
    for _, is_close, tag in events:
        if tag not in _TRACKED_TAGS:
            continue
        if is_close:
            if stack and stack[-1] == tag:
                stack.pop()
        else:
            stack.append(tag)

    return stack


def _close_tags(tags: list[str]) -> str:
    """Return closing tags for *tags* in reverse order."""
    return "".join(f"</{t}>" for t in reversed(tags))


def _reopen_tags(tags: list[str]) -> str:
    """Return opening tags for *tags* in original order."""
    return "".join(f"<{t}>" for t in tags)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smart_split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split *text* into chunks that respect markdown structure and HTML tags.

    Algorithm:
    1. Parse text into structural blocks.
    2. Greedily pack blocks into chunks; split oversized blocks by type.
    3. At each chunk boundary, close open HTML tags and re-open them in the
       next chunk.

    Args:
        text: The markdown text to split.
        max_len: Maximum length per chunk.

    Returns:
        A list of string chunks, each ideally within *max_len* characters.
    """
    if not text or not text.strip():
        return []

    blocks = parse_blocks(text)
    if not blocks:
        # Text has no parseable blocks but isn't empty — treat as paragraph
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= max_len:
            return [stripped]
        # Fall back to word-level splitting
        chunks: list[str] = []
        remainder = _split_by_words(stripped, max_len, chunks)
        if remainder:
            chunks.append(remainder)
        return chunks

    # Expand blocks into rendered pieces (some blocks produce multiple pieces)
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(_split_block(block, max_len))

    # Pack pieces into chunks
    result: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    separator = "\n\n"

    for piece in pieces:
        piece_len = len(piece)
        # Budget for separator between parts
        needed = piece_len + (len(separator) if current_parts else 0)

        if current_parts and current_len + needed > max_len:
            # Flush current chunk
            chunk_text = separator.join(current_parts)
            result.append(chunk_text)
            current_parts = [piece]
            current_len = piece_len
        else:
            current_parts.append(piece)
            current_len += needed

    if current_parts:
        result.append(separator.join(current_parts))

    # Apply tag balancing across chunks
    if len(result) <= 1:
        return result

    balanced: list[str] = []
    for i, chunk in enumerate(result):
        open_tags = _extract_open_tags(chunk)

        if i < len(result) - 1 and open_tags:
            # Close open tags at end of this chunk
            close_suffix = _close_tags(open_tags)
            chunk = chunk + close_suffix
            # Reopen at start of next chunk
            result[i + 1] = _reopen_tags(open_tags) + result[i + 1]

        balanced.append(chunk)

    return balanced
