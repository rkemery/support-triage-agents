"""A scripted stand-in for the model, for tests, the pipeline demo and the cost estimate.

It reads the gold answer and walks the happy path, so it proves the plumbing
works end to end. It is not a model, and its numbers are never reported as
results. Every reply is a function of the request, so it drops into the
harness `FakeClient`.

`wrong=True` makes it propose an action the bank refuses (for testing the
oracle and the violation scoring).
"""

from __future__ import annotations

import json
from typing import Any

from llm_eval_harness import FakeClient, ModelRequest

from support_triage_agents.data import Task, load_tasks


def role_of(instructions: str) -> str:
    for role, marker in (
        ("intake", "You are the intake agent"),
        ("researcher", "You are the research agent"),
        ("resolver", "You are the resolver agent"),
        ("compliance", "You are the compliance reviewer"),
        ("single_agent", "You are a support agent"),
        ("crosscheck", "You are auditing the answer key"),
    ):
        if instructions.startswith(marker):
            return role
    raise ValueError("unrecognized prompt")


def _task_for(request: ModelRequest, tasks: dict[str, Task]) -> Task:
    contents = (
        [request.input]
        if isinstance(request.input, str)
        else [str(m.get("content", "")) for m in request.input]
    )
    for text, task in tasks.items():
        escaped = json.dumps(text, ensure_ascii=False)[1:-1]
        if any(text in c or escaped in c for c in contents):
            return task
    raise ValueError("no task's ticket text appears in the request")


def _actions(task: Task, wrong: bool) -> list[dict[str, Any]]:
    actions = [
        {"action": a["action"], "args": {k: str(v) for k, v in a["args"].items()}}
        for a in task.gold_actions
    ]
    if wrong:
        # Freezing someone else's card is refused under a support rule for every customer.
        other = "card_002p" if task.customer_id != "cus_002" else "card_001p"
        actions.append({"action": "freeze_card", "args": {"card_id": other}})
    return actions


def scripted_reply(request: ModelRequest, tasks: dict[str, Task], wrong: bool = False) -> str:
    role = role_of(request.instructions or "")
    task = _task_for(request, tasks)
    messages = request.input if isinstance(request.input, list) else []
    turns = len(messages) - 1  # messages after the opening one
    if role == "intake":
        return json.dumps(
            {
                "category": "other",
                "summary": task.ticket_text[:200],
                "needs_clarification": task.needs_clarification,
                "clarifying_question": "Can you tell me a bit more?"
                if task.needs_clarification
                else None,
                "risk_flags": [],
            }
        )
    if role == "researcher":
        return json.dumps({"queries": [task.ticket_text]})
    if role == "compliance":
        return json.dumps({"approve": True, "issues": []})
    if role == "crosscheck":
        return json.dumps(
            {"actions": task.gold_actions, "should_escalate": task.should_escalate, "reason": "-"}
        )
    actions = _actions(task, wrong)
    if role == "resolver":
        if turns == 0:
            return json.dumps({"type": "tool", "tool": "list_cards", "args": {}})
        plan = {
            "actions": [{**a, "why": "scripted"} for a in actions],
            "reply_to_customer": "Done.",
            "rationale": "scripted",
            "policy_refs": list(task.policy_refs),
        }
        return json.dumps({"type": "plan", "plan": plan})
    # single agent: search, maybe ask, then one write per turn, then reply
    steps: list[dict[str, Any]] = [
        {"type": "tool", "tool": "search_help_center", "args": {"query": task.ticket_text}}
    ]
    if task.needs_clarification:
        steps.append({"type": "tool", "tool": "ask_customer", "args": {"question": "Which one?"}})
    steps += [{"type": "tool", "tool": a["action"], "args": a["args"]} for a in actions]
    steps.append({"type": "final", "reply_to_customer": "Done."})
    return json.dumps(steps[min(turns // 2, len(steps) - 1)])


def scripted_client(wrong: bool = False, latency_ms: float = 0.0) -> FakeClient:
    tasks = {t.ticket_text: t for t in load_tasks()}
    return FakeClient(lambda r: scripted_reply(r, tasks, wrong), latency_ms=latency_ms)
