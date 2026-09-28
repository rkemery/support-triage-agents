"""Loaders for the vendored Tallowbrook files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from support_triage_agents.vendor import DATA_DIR

TALLOWBROOK_DIR = DATA_DIR / "tallowbrook"
SNAPSHOT_DIR = DATA_DIR / "rag_snapshot"


@dataclass(frozen=True)
class Task:
    """One support ticket and its gold labels. Only `ticket_text` and `customer_id` reach agents.

    `hidden_facts` go to the customer simulator only. The gold fields go to the
    scorer and the review oracle only.
    """

    task_id: str
    customer_id: str
    ticket_text: str
    hidden_facts: tuple[str, ...]
    needs_clarification: bool
    should_escalate: bool
    gold_actions: tuple[dict[str, Any], ...]
    gold_final_state: dict[str, Any]
    policy_refs: tuple[str, ...]
    rationale: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Task:
        return cls(
            task_id=row["task_id"],
            customer_id=row["customer_id"],
            ticket_text=row["ticket_text"],
            hidden_facts=tuple(row["hidden_facts"]),
            needs_clarification=bool(row["needs_clarification"]),
            should_escalate=bool(row["should_escalate"]),
            gold_actions=tuple(row["gold_actions"]),
            gold_final_state=row["gold_final_state"],
            policy_refs=tuple(row["policy_refs"]),
            rationale=row["rationale"],
        )


def load_tasks(path: Path | None = None) -> list[Task]:
    path = path or TALLOWBROOK_DIR / "tasks.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [Task.from_dict(json.loads(line)) for line in fh if line.strip()]


def task_by_id(task_id: str) -> Task:
    for task in load_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"no task {task_id!r}")


@lru_cache(maxsize=1)
def _seed_text() -> str:
    return (TALLOWBROOK_DIR / "bank_seed.json").read_text(encoding="utf-8")


def load_seed() -> dict[str, Any]:
    """A fresh copy of the bank seed each call, so callers can't mutate a shared one."""
    return json.loads(_seed_text())


@lru_cache(maxsize=1)
def load_facts() -> dict[str, Any]:
    with (TALLOWBROOK_DIR / "policies.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
