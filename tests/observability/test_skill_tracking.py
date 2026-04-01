"""Tests for skill tracking via SkillsLoader callback."""

from pathlib import Path

import pytest


def test_skills_loader_fires_callback_on_load(tmp_path):
    from nanobot.agent.skills import SkillsLoader

    skill_dir = tmp_path / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test_skill\ndescription: A test skill\n---\nDo something.\n"
    )

    loader = SkillsLoader(tmp_path)
    loaded_skills: list[str] = []

    def on_loaded(names: list[str]):
        loaded_skills.extend(names)

    result = loader.load_skills_for_context(["test_skill"], on_skills_loaded=on_loaded)

    assert result != ""
    assert loaded_skills == ["test_skill"]


def test_skills_loader_callback_not_called_when_no_skills():
    from nanobot.agent.skills import SkillsLoader

    loader = SkillsLoader(Path("/tmp"))
    loaded_skills: list[str] = []

    def on_loaded(names: list[str]):
        loaded_skills.extend(names)

    result = loader.load_skills_for_context(["nonexistent"], on_skills_loaded=on_loaded)

    assert result == ""
    assert loaded_skills == []
