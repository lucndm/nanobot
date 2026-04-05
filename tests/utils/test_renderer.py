from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from nanobot.utils.renderer import KrokiRenderer, render_ascii_art_pillow, render_table_pillow

# ---------------------------------------------------------------------------
# KrokiRenderer.is_supported
# ---------------------------------------------------------------------------


class TestKrokiRendererIsSupported:
    def test_mermaid(self) -> None:
        assert KrokiRenderer().is_supported("mermaid") is True

    def test_graphviz(self) -> None:
        assert KrokiRenderer().is_supported("graphviz") is True

    def test_plantuml(self) -> None:
        assert KrokiRenderer().is_supported("plantuml") is True

    def test_d2(self) -> None:
        assert KrokiRenderer().is_supported("d2") is True

    def test_unsupported(self) -> None:
        assert KrokiRenderer().is_supported("bogus") is False

    def test_case_sensitive(self) -> None:
        assert KrokiRenderer().is_supported("Mermaid") is False


# ---------------------------------------------------------------------------
# KrokiRenderer.render
# ---------------------------------------------------------------------------


class TestKrokiRendererRender:
    async def test_render_success(self) -> None:
        renderer = KrokiRenderer()
        png_bytes = b"\x89PNG\r\n\x1a\ntest"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("graph LR\n  A-->B", "mermaid")

        assert result == png_bytes
        mock_client.post.assert_called_once_with(
            "https://kroki.io/mermaid/png",
            content="graph LR\n  A-->B",
            headers={"Content-Type": "text/plain"},
        )

    async def test_render_custom_base_url(self) -> None:
        renderer = KrokiRenderer(base_url="http://localhost:8000")
        png_bytes = b"\x89PNG\r\n\x1a\n"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("digraph { a -> b }", "graphviz")

        mock_client.post.assert_called_once_with(
            "http://localhost:8000/graphviz/png",
            content="digraph { a -> b }",
            headers={"Content-Type": "text/plain"},
        )
        assert result == png_bytes

    async def test_render_svg_format(self) -> None:
        renderer = KrokiRenderer()
        svg_bytes = b"<svg>test</svg>"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = svg_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("graph LR\n  A-->B", "mermaid", output_format="svg")

        mock_client.post.assert_called_once_with(
            "https://kroki.io/mermaid/svg",
            content="graph LR\n  A-->B",
            headers={"Content-Type": "text/plain"},
        )
        assert result == svg_bytes

    async def test_render_timeout_returns_none(self) -> None:
        renderer = KrokiRenderer(timeout=0.001)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("graph LR\n  A-->B", "mermaid")

        assert result is None

    async def test_render_http_error_returns_none(self) -> None:
        renderer = KrokiRenderer()

        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "bad request", request=MagicMock(), response=mock_response
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("invalid", "mermaid")

        assert result is None

    async def test_render_network_error_returns_none(self) -> None:
        renderer = KrokiRenderer()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("nanobot.utils.renderer.httpx.AsyncClient", return_value=mock_client):
            result = await renderer.render("graph LR\n  A-->B", "mermaid")

        assert result is None

    async def test_render_unsupported_type_returns_none(self) -> None:
        renderer = KrokiRenderer()
        result = await renderer.render("test", "bogus")
        assert result is None


# ---------------------------------------------------------------------------
# render_table_pillow
# ---------------------------------------------------------------------------


class TestRenderTablePillow:
    def test_basic_table_returns_png_bytes(self) -> None:
        table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        result = render_table_pillow(table)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_wide_table_returns_png(self) -> None:
        table = (
            "| Col1 | Col2 | Col3 | Col4 | Col5 | Col6 | Col7 | Col8 |\n"
            "|------|------|------|------|------|------|------|------|\n"
            "| a | b | c | d | e | f | g | h |"
        )
        result = render_table_pillow(table)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_empty_string_returns_none(self) -> None:
        assert render_table_pillow("") is None

    def test_single_row_returns_none(self) -> None:
        assert render_table_pillow("| Name | Age |") is None

    def test_only_header_returns_none(self) -> None:
        assert render_table_pillow("| Name | Age |\n|------|-----|") is None

    def test_non_table_text_returns_none(self) -> None:
        assert render_table_pillow("just some text") is None

    def test_two_rows_minimum(self) -> None:
        """Header + separator + one data row is the minimum valid table."""
        table = "| A |\n|---|\n| 1 |"
        result = render_table_pillow(table)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_header_only_plus_separator_returns_none(self) -> None:
        """Header + separator but no data rows should return None."""
        table = "| A |\n|---|"
        assert render_table_pillow(table) is None

    def test_multiline_values(self) -> None:
        table = "| Key | Value |\n|-----|-------|\n| foo | bar baz |"
        result = render_table_pillow(table)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# render_ascii_art_pillow
# ---------------------------------------------------------------------------


class TestAsciiArtRenderer:
    def test_basic_box_art_renders(self) -> None:
        text = "╔═══╗\n║ A ║\n╚═══╝"
        result = render_ascii_art_pillow(text)
        assert result.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(result) > 100

    def test_empty_returns_empty(self) -> None:
        assert render_ascii_art_pillow("") == b""
        assert render_ascii_art_pillow("   ") == b""

    def test_wide_art_renders(self) -> None:
        line = "║" + "═" * 200 + "║"
        text = f"╔{'═' * 200}╗\n{line}\n╚{'═' * 200}╝"
        result = render_ascii_art_pillow(text)
        assert result.startswith(b"\x89PNG\r\n\x1a\n")

    def test_single_line_renders(self) -> None:
        """Even a single non-empty line produces a valid PNG."""
        result = render_ascii_art_pillow("Hello")
        assert result.startswith(b"\x89PNG\r\n\x1a\n")

    def test_many_lines_renders(self) -> None:
        """Many lines of box art should still produce a valid PNG."""
        lines = ["║ row " + str(i).zfill(3) + " ║" for i in range(50)]
        text = "╔═══════════╗\n" + "\n".join(lines) + "\n╚═══════════╝"
        result = render_ascii_art_pillow(text)
        assert result.startswith(b"\x89PNG\r\n\x1a\n")
