"""Lightweight HTTP webhook listener for receiving external callbacks."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from aiohttp import web
from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


class WebhookServer:
    """Receives HTTP POST callbacks and injects them as system messages into the agent."""

    def __init__(self, agent: AgentLoop, port: int = 8080, secret: str = ""):
        self._agent = agent
        self._port = port
        self._secret = secret
        self._app = web.Application()
        self._app.router.add_post("/webhook/research", self._handle_research)
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("Webhook server listening on port {}", self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            logger.info("Webhook server stopped")

    async def _handle_research(self, request: web.Request) -> web.Response:
        """Handle POST /webhook/research from ResearchClaw MCP server."""
        if self._secret:
            auth = request.headers.get("X-Webhook-Secret", "")
            if auth != self._secret:
                return web.Response(status=403, text="Forbidden")

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        run_id = payload.get("run_id", "unknown")
        status = payload.get("status", "unknown")
        logger.info("Research webhook received: run_id={}, status={}", run_id, status)

        if status == "completed":
            commit = payload.get("commit", "unknown")
            message = (
                f"[Research Complete] Research run '{run_id}' has completed.\n"
                f"Commit: {commit}\n"
                f"Use the get_paper MCP tool with run_id '{run_id}' to retrieve the paper "
                f"and summarize it for the user."
            )
        else:
            exit_code = payload.get("exit_code", "?")
            message = (
                f"[Research Failed] Research run '{run_id}' has failed (exit code: {exit_code}).\n"
                f"Notify the user about the failure."
            )

        asyncio.create_task(
            self._agent.process_direct(
                message,
                session_key=f"webhook:research:{run_id}",
                channel="system",
                chat_id=f"webhook:research:{run_id}",
            )
        )

        return web.Response(status=200, text="OK")
