"""Prompt templates in prompts/*.txt, filled with the playbook, tool list and date."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from support_triage_agents.tools import ROLE_TOOLS, TOOL_SPECS, WRITE_TOOLS, support_playbook

PROMPT_DIR = Path(__file__).parent / "prompts"


@cache
def _template(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _list(names: frozenset[str] | set[str]) -> str:
    return "\n".join(f"- {TOOL_SPECS[n]}" for n in sorted(names))


def render(name: str, facts: dict[str, Any], as_of: str) -> str:
    """The system instructions for one role."""
    tools = ROLE_TOOLS.get(name, frozenset())
    return _template(name).format(
        playbook=support_playbook(facts),
        tools=_list(tools),
        actions=_list(WRITE_TOOLS),
        as_of=as_of,
    )
