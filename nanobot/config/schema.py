"""Configuration schema using Pydantic."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChannelConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default=["*"], alias="allowFrom")
    group_policy: str = Field(default="mention", alias="groupPolicy")
    streaming: bool = True
    send_progress: bool = True
    send_tool_hints: bool = False
    send_max_retries: int = Field(
        default=3, ge=0, le=10
    )


class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = (
        "auto"  # Provider name (e.g. "anthropic", "openrouter") or "auto" for auto-detection
    )
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    temperature: float = 0.1
    max_tool_iterations: int = 40
    reasoning_effort: str | None = None  # low / medium / high - enables LLM thinking mode
    timezone: str = "UTC"  # IANA timezone, e.g. "Asia/Shanghai", "America/New_York"


class AgentsConfig(Base):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60  # 30 minutes
    keep_recent_messages: int = 8


class WebhookConfig(Base):
    """Webhook HTTP listener configuration."""

    enabled: bool = False
    port: int = 8080
    secret: str = ""  # Optional shared secret for request validation


class GatewayConfig(Base):
    """Gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "brave"  # brave, tavily, duckduckgo, searxng, jina
    api_key: str = ""
    base_url: str = ""  # SearXNG base URL
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    proxy: str | None = (
        None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # auto-detected if omitted
    command: str = ""  # Stdio: command to run (e.g. "npx")
    args: list[str] = Field(default_factory=list)  # Stdio: command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Stdio: extra env vars
    url: str = ""  # HTTP/SSE: endpoint URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE: custom headers
    tool_timeout: int = 30  # seconds before a tool call is cancelled
    enabled_tools: list[str] = Field(
        default_factory=lambda: ["*"]
    )  # Only register these tools; accepts raw MCP names or wrapped mcp_<server>_<tool> names; ["*"] = all tools; [] = no tools


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class OtelConfig(Base):
    """OpenTelemetry configuration."""

    enabled: bool = False
    endpoint: str = "http://100.68.251.84:4317"
    service_name: str = "nanobot"


class DatabaseConfig(Base):
    """Database backend configuration."""

    backend: Literal["sqlite", "postgres"] = "sqlite"
    url: str = ""
    pool_size: int = 5
    sqlite_path: str = "data/memories.db"


class LiteLLMModelConfig(Base):
    """A single model entry for litellm Router."""

    model_name: str
    litellm_params: dict[str, str]


class LiteLLMConfig(Base):
    """LiteLLM client configuration.

    Supports two modes:
    - proxy: calls go through a litellm proxy server (api_base required)
    - direct: calls go through an in-process litellm Router (models required)
    Proxy is primary; direct is used when proxy is unavailable.
    """

    api_base: str | None = None
    api_key: str | None = None
    groq_api_key: str = ""  # Groq API key for transcription
    models: list[LiteLLMModelConfig] = Field(default_factory=list)
    fallbacks: list[dict[str, list[str]]] = Field(default_factory=list)
    default_headers: dict[str, str] = Field(default_factory=dict)
    success_callback: list[str] = Field(default_factory=list)
    failure_callback: list[str] = Field(default_factory=list)
    mode: Literal["proxy", "direct"] | None = None
    # litellm SDK features
    num_retries: int = Field(default=2, ge=0, le=5)  # built-in retry on transient errors
    timeout: int | None = Field(default=None, ge=1)  # per-request timeout in seconds
    enable_prompt_caching: bool = False  # add cache_control markers for Anthropic/Gemini
    allowed_fails: int = Field(default=3, ge=0)  # Router circuit breaker: fails before cooldown
    cooldown_time: int = Field(default=60, ge=1)  # Router circuit breaker: cooldown seconds

    def model_post_init(self, __context: object) -> None:
        # Auto-detect mode: proxy if api_base set, direct if models configured, else proxy
        if self.mode is None:
            if self.api_base:
                self.mode = "proxy"
            elif self.models:
                self.mode = "direct"
            else:
                self.mode = "proxy"

        # Resolve ${ENV_VAR} in api_key
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")

        # Resolve ${ENV_VAR} in groq_api_key
        if (
            self.groq_api_key
            and self.groq_api_key.startswith("${")
            and self.groq_api_key.endswith("}")
        ):
            env_var = self.groq_api_key[2:-1]
            self.groq_api_key = os.environ.get(env_var, "")

        # Resolve ${ENV_VAR} in model litellm_params api_key
        for model in self.models:
            params = model.litellm_params
            if "api_key" in params:
                val = params["api_key"]
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    env_var = val[2:-1]
                    params["api_key"] = os.environ.get(env_var, "")


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    otel: OtelConfig = Field(default_factory=OtelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    litellm: LiteLLMConfig = Field(default_factory=LiteLLMConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
