"""Config store — load/save config from PostgreSQL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from nanobot.config.schema import Config


def _resolve_dsn() -> str:
    """Resolve database DSN from env or bootstrap file."""
    dsn = os.environ.get("NANOBOT_DATABASE_URL", "")
    if dsn:
        return dsn
    db_json = Path.home() / ".nanobot" / "db.json"
    if db_json.exists():
        data = json.loads(db_json.read_text())
        dsn = data.get("url", "")
    if not dsn:
        raise ValueError(
            "NANOBOT_DATABASE_URL not set and ~/.nanobot/db.json not found. "
            "Set NANOBOT_DATABASE_URL=postgresql://user:pass@host:port/dbname"
        )
    return dsn


class ConfigStore:
    """Load and persist config from PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        self._pool = ConnectionPool(dsn, min_size=1, max_size=1, open=True)
        self._init_table()
        logger.info(
            "ConfigStore connected to {}", dsn.split("@")[-1] if "@" in dsn else dsn
        )

    def _init_table(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.commit()

    def load_config(self) -> Config:
        """Load config from DB. First run imports from config.json."""
        from nanobot.config.schema import Config

        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'main'"
            ).fetchone()
            if row:
                return Config.model_validate(row[0])

        # First run: import from config.json
        config_path = Path.home() / ".nanobot" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config in DB and {config_path} not found. "
                "Create config.json or run migration script first."
            )
        config = Config.model_validate_json(config_path.read_text())
        self.save_config(config)
        logger.info("Imported config from {} → DB", config_path)
        return config

    def save_config(self, config: Config) -> None:
        """Persist config to DB."""
        value = json.dumps(config.model_dump(mode="json"))
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO config (key, value, updated_at) VALUES ('main', %s, now()) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=now()",
                (value,),
            )
            conn.commit()

    def close(self) -> None:
        self._pool.close()
