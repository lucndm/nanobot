"""JSONL file-backed session store — extracted from SessionManager."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


class JsonlSessionStore:
    """File-based session store using JSONL format (one JSON object per line)."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, dict[str, Any]] = {}

    def _session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _legacy_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> dict[str, Any]:
        if key in self._cache:
            return self._cache[key]

        data = self._load(key)
        if data is None:
            data = {
                "key": key,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": {},
                "last_consolidated": 0,
            }

        self._cache[key] = data
        return data

    def _load(self, key: str) -> dict[str, Any] | None:
        path = self._session_path(key)
        if not path.exists():
            legacy_path = self._legacy_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = data.get("created_at")
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return {
                "key": key,
                "messages": messages,
                "created_at": created_at or datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": metadata,
                "last_consolidated": last_consolidated,
            }
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session_data: dict[str, Any]) -> None:
        key = session_data["key"]
        path = self._session_path(key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": key,
                "created_at": session_data.get("created_at", ""),
                "updated_at": session_data.get("updated_at", datetime.now().isoformat()),
                "metadata": session_data.get("metadata", {}),
                "last_consolidated": session_data.get("last_consolidated", 0),
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session_data.get("messages", []):
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[key] = session_data

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                }
                            )
            except Exception:
                continue
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def consolidate(self, session_key: str, topic_name: str, summary: str, last_seq: int) -> None:
        pass  # JSONL store does not support consolidation

    def get_summary(self, session_key: str, topic_name: str) -> dict | None:
        return None

    def get_usage(
        self,
        *,
        session_key: str | None = None,
        topic_name: str | None = None,
        model: str | None = None,
        since: Any | None = None,
    ) -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "turns": 0,
        }
