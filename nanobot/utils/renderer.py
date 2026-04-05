from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

_SUPPORTED_DIAGRAM_TYPES = frozenset({"mermaid", "graphviz", "plantuml", "d2"})

_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSansMono.ttf"),
]

_CELL_PADDING = 8
_HEADER_BG = "#f0f0f0"
_LINE_COLOR = "#d0d0d0"


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in _FONT_PATHS:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=14)
            except OSError:
                continue
    return ImageFont.load_default()


def _parse_markdown_table(text: str) -> list[list[str]] | None:
    """Parse a markdown pipe-table into a list of rows (each a list of cell strings).

    Returns None if the input does not contain a valid table (header + separator
    + at least one data row).
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if len(lines) < 3:
        return None

    # First line must look like a header row
    if not lines[0].startswith("|"):
        return None

    # Second line must be a separator (|---|---| style)
    if not re.match(r"^\|[\s\-:|]+\|$", lines[1]):
        return None

    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)

    # Need at least header + 1 data row (separator already skipped)
    if len(rows) < 3:
        return None

    # Return header + data rows (skip separator row at index 1)
    return [rows[0]] + rows[2:]


class KrokiRenderer:
    """Renders diagrams via the Kroki API (https://kroki.io)."""

    def __init__(self, base_url: str = "https://kroki.io", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_supported(self, diagram_type: str) -> bool:
        """Check whether a diagram type is supported by Kroki."""
        return diagram_type in _SUPPORTED_DIAGRAM_TYPES

    async def render(
        self, text: str, diagram_type: str, output_format: str = "png"
    ) -> bytes | None:
        """Render a diagram to image bytes.

        Returns PNG/SVG bytes on success, None on any failure.
        """
        if not self.is_supported(diagram_type):
            return None

        url = f"{self.base_url}/{diagram_type}/{output_format}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    content=text,
                    headers={"Content-Type": "text/plain"},
                )
                response.raise_for_status()
                return response.content
        except httpx.TimeoutException:
            logger.warning("Kroki request timed out for {} diagram", diagram_type)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Kroki returned HTTP {} for {} diagram: {}",
                exc.response.status_code,
                diagram_type,
                exc,
            )
        except httpx.HTTPError as exc:
            logger.warning("Kroki request failed for {} diagram: {}", diagram_type, exc)

        return None


def render_table_pillow(table_text: str) -> bytes | None:
    """Render a markdown pipe-table to PNG bytes using Pillow.

    Returns None if the input is not a valid table (fewer than 2 content rows
    after parsing header and separator).
    """
    rows = _parse_markdown_table(table_text)
    if rows is None:
        return None

    font = _load_font()
    draw_buffer = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(draw_buffer)

    # Measure column widths
    num_cols = len(rows[0])
    col_widths = [0] * num_cols

    for row in rows:
        for col_idx, cell in enumerate(row):
            if col_idx >= num_cols:
                break
            try:
                text_width = int(font.getlength(cell))
            except AttributeError:
                bbox = draw.textbbox((0, 0), cell, font=font)
                text_width = bbox[2] - bbox[0]
            col_widths[col_idx] = max(col_widths[col_idx], text_width)

    row_height = 14 + _CELL_PADDING * 2  # font size + padding
    img_width = sum(col_widths) + _CELL_PADDING * 2 * num_cols + num_cols + 1
    img_height = row_height * len(rows) + len(rows) + 1

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    y = 0
    for row_idx, row in enumerate(rows):
        if row_idx == 0:
            draw.rectangle([0, y, img_width, y + row_height], fill=_HEADER_BG)

        x = 0
        for col_idx, cell in enumerate(row):
            cell_width = col_widths[col_idx] + _CELL_PADDING * 2
            text_x = x + _CELL_PADDING
            text_y = y + _CELL_PADDING
            draw.text((text_x, text_y), cell, fill="black", font=font)
            x += cell_width

            # Vertical line between columns (skip after last column)
            if col_idx < num_cols - 1:
                draw.line([(x, y), (x, y + row_height)], fill=_LINE_COLOR, width=1)
                x += 1

        y += row_height

        # Horizontal line between rows
        if row_idx < len(rows) - 1:
            draw.line([(0, y), (img_width, y)], fill=_LINE_COLOR, width=1)
            y += 1

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
