"""Parse litellm config from TOPIC.md ## litellm sections."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class TopicConfig:
    """Per-topic LLM generation overrides."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


def parse_topic_config(content: str) -> TopicConfig | None:
    """Extract TopicConfig from TOPIC.md content.

    Looks for a ``## litellm`` section and parses key: value pairs.
    Returns None if no litellm section found or no model specified.
    """
    match = re.search(r"^## litellm\s*\n(.*?)(?=\n## |\Z)", content, re.DOTALL | re.MULTILINE)
    if not match:
        return None

    section = match.group(1).strip()
    if not section:
        return None

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    for line in section.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "model":
            model = value
        elif key == "temperature":
            try:
                temperature = float(value)
            except ValueError:
                pass
        elif key == "max_tokens":
            try:
                max_tokens = int(value)
            except ValueError:
                pass

    if model is None:
        return None

    return TopicConfig(model=model, temperature=temperature, max_tokens=max_tokens)
