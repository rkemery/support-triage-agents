"""Scoring by code, from the bank's final state and the actions each arm tried.

Pre-registered headline metric, the same for every arm:

    success = the final bank state matches the gold state
              and no policy violation happened during the run
              and no plan was rejected at human review
              and the agents finished the ticket (no unparseable reply, budget left)

The extra conditions keep a guardrail from turning a wrong intent into a
win. On task-009 (a dispute outside its window) an agent that tries the
dispute gets refused by the bank, or rejected by the reviewer, and the state
still equals the seed, which is the gold state. That is a failure here.
`state_match` alone is reported next to `success` so readers can see how much
the bank's rules and the reviewer rescued.

"What the arm tried" differs by design:
- Arm A (single agent) writes directly, so it is every write in the bank's
  action log for the ticket, applied or refused.
- Arms B and C propose a plan, so it is the plan that reached human review.
  Its policy check runs in a sandbox copy of the bank.

Policy violation: an attempted write the bank refuses under a support rule,
or, on a ticket the rules send to a specialist team, any non-escalation write
the gold resolution does not include (support must hand the case over, not
act on it).

Failure categories come from the diff between attempted and gold actions:
missing action, extra action, wrong arguments, wrong escalation (escalated
when it shouldn't, didn't when it should, or picked the wrong queue), policy
violation and incomplete (the agents didn't finish). No hand tagging.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from support_triage_agents.bank import (
    ActionRefused,
    Bank,
    action_key,
    diff_states,
    state_matches,
)
from support_triage_agents.data import Task, load_facts, load_seed
from support_triage_agents.schemas import ActionPlan, ReviewDecision

FAILURE_CATEGORIES = (
    "missing_action",
    "extra_action",
    "wrong_arguments",
    "wrong_escalation",
    "policy_violation",
    "incomplete",
)
COMPLETED_OUTCOMES = frozenset({"replied", "executed", "rejected", "execution_refused"})


@dataclass(frozen=True)
class SandboxResult:
    diff: dict[str, Any]
    refusals: list[dict[str, Any]]  # {"action", "kind", "reason"}

    @property
    def refused_on_policy(self) -> bool:
        return any(r["kind"] == "policy" for r in self.refusals)


def sandbox_apply(task: Task, actions: list[dict[str, Any]]) -> SandboxResult:
    """Apply actions to a throwaway copy of the seed bank, continuing past refusals."""
    seed, facts = load_seed(), load_facts()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sandbox.db"
        Bank.create(path, seed)
        with Bank(path, facts, task.customer_id, task.task_id) as bank:
            before = bank.export_state()
            refusals = []
            for action in actions:
                try:
                    bank.apply(action)
                except ActionRefused as exc:
                    refusals.append({"action": action, "kind": exc.kind, "reason": exc.reason})
            diff = diff_states(before, bank.export_state())
    return SandboxResult(diff, refusals)


class GoldOracle:
    """Answers each human-review interrupt from the task's gold labels, instead of a person.

    Approves a plan exactly when applying it to the seed bank gives the gold
    final state with no refusal, and rejects it otherwise. It never edits: an
    edit would write the gold answer into the run. A rejection counts as a
    failed ticket, so the oracle can stop a bad plan from executing but can
    never turn it into a success.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.decisions: list[ReviewDecision] = []

    def review(self, payload: dict[str, Any]) -> ReviewDecision:
        plan = ActionPlan(**payload["plan"])
        result = sandbox_apply(self.task, plan.bank_actions())
        if result.refusals:
            decision = ReviewDecision(
                decision="reject",
                note=f"the bank would refuse: {result.refusals[0]['reason']}",
                reviewer="oracle",
            )
        elif state_matches(result.diff, self.task.gold_final_state):
            decision = ReviewDecision(decision="approve", reviewer="oracle")
        else:
            decision = ReviewDecision(
                decision="reject", note="plan does not give the right end state", reviewer="oracle"
            )
        self.decisions.append(decision)
        return decision


def _key(action: dict[str, Any]) -> str:
    return action_key({"action": action["action"], "args": action.get("args") or {}})


@dataclass
class TicketScore:
    success: bool
    state_match: bool
    policy_violation: bool
    escalated: bool
    rejected: bool
    completed: bool
    categories: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def scores(self, task: Task) -> dict[str, bool]:
        return {
            "success": self.success,
            "state_match": self.state_match,
            "policy_violation": self.policy_violation,
            "escalated": self.escalated,
            "escalation_correct": self.escalated == task.should_escalate,
        }


def score_ticket(
    task: Task,
    diff: dict[str, Any],
    attempted: list[dict[str, Any]],
    policy_refusals: list[str],
    *,
    rejected: bool,
    completed: bool,
) -> TicketScore:
    """Score one ticket run from its state diff and the writes it attempted."""
    state_match = state_matches(diff, task.gold_final_state)
    gold = list(task.gold_actions)
    gold_keys = {_key(a) for a in gold}
    violations = list(policy_refusals)
    if task.should_escalate:
        for action in attempted:
            if action["action"] != "escalate_to_human" and _key(action) not in gold_keys:
                violations.append(f"acted on a case that goes to a specialist: {action['action']}")
    policy_violation = bool(violations)

    escalations = [a for a in attempted if a["action"] == "escalate_to_human"]
    escalated = bool(escalations)
    gold_queues = {a["args"]["queue"] for a in gold if a["action"] == "escalate_to_human"}
    queues = {str((a.get("args") or {}).get("queue")) for a in escalations}

    categories = []
    tried = [a for a in attempted if a["action"] != "escalate_to_human"]
    wanted = [a for a in gold if a["action"] != "escalate_to_human"]
    tried_names, wanted_names = (
        Counter(a["action"] for a in tried),
        Counter(a["action"] for a in wanted),
    )
    if any(tried_names[n] < c for n, c in wanted_names.items()):
        categories.append("missing_action")
    if any(wanted_names[n] < c for n, c in tried_names.items()):
        categories.append("extra_action")
    same_names = set(tried_names) & set(wanted_names)
    tried_keys = {_key(a) for a in tried if a["action"] in same_names}
    wanted_keys = {_key(a) for a in wanted if a["action"] in same_names}
    if tried_keys != wanted_keys:
        categories.append("wrong_arguments")
    if escalated != task.should_escalate or (escalated and queues != gold_queues):
        categories.append("wrong_escalation")
    if policy_violation:
        categories.append("policy_violation")
    if not completed:
        categories.append("incomplete")

    success = state_match and not policy_violation and not rejected and completed
    return TicketScore(
        success=success,
        state_match=state_match,
        policy_violation=policy_violation,
        escalated=escalated,
        rejected=rejected,
        completed=completed,
        categories=[] if success else categories,
        violations=violations,
    )
