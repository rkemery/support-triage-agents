"""Typed handoffs between agents. Every model reply is parsed into one of these or rejected.

Each agent sees only what its input model carries. The compliance reviewer's
input (`ComplianceInput`) has no field for the resolver's tool transcript or
for anything the customer has not said, so it can't lean on them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from support_triage_agents.bank import WRITE_ACTIONS

ActionName = Literal[
    "freeze_card",
    "unfreeze_card",
    "report_card_lost_stolen",
    "order_replacement_card",
    "open_dispute",
    "refund_fee",
    "change_plan",
    "close_account",
    "escalate_to_human",
]
assert set(ActionName.__args__) == set(WRITE_ACTIONS)  # type: ignore[attr-defined]

Category = Literal[
    "card", "dispute", "fee", "plan", "account", "fraud", "verification", "complaint", "other"
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntakeResult(Strict):
    """Intake's classification. It sees the ticket and the customer's profile, and no tools."""

    category: Category
    summary: str = Field(max_length=600)
    needs_clarification: bool
    clarifying_question: str | None = Field(default=None, max_length=400)
    risk_flags: list[str] = Field(default_factory=list, max_length=8)


class Clarification(Strict):
    question: str
    answer: str


class Excerpt(Strict):
    chunk_id: str
    article_id: str
    title: str
    effective_date: str
    text: str


class ResearchQueries(Strict):
    """What the Researcher model writes. Code runs the searches on its behalf."""

    queries: list[Annotated[str, Field(min_length=3, max_length=300)]] = Field(
        min_length=1, max_length=3
    )


class ResearchNotes(Strict):
    queries: list[str]
    excerpts: list[Excerpt]


class ProposedAction(Strict):
    action: ActionName
    args: dict[str, str] = Field(default_factory=dict)
    why: str = Field(default="", max_length=400)

    def as_bank_action(self) -> dict[str, Any]:
        return {"action": self.action, "args": dict(self.args)}


class ActionPlan(Strict):
    """The Resolver's output. It describes writes, it can't perform them."""

    actions: list[ProposedAction] = Field(default_factory=list, max_length=6)
    reply_to_customer: str = Field(max_length=2000)
    rationale: str = Field(default="", max_length=1200)
    policy_refs: list[str] = Field(default_factory=list, max_length=10)

    def bank_actions(self) -> list[dict[str, Any]]:
        return [a.as_bank_action() for a in self.actions]

    def fingerprint(self) -> str:
        """sha256 of the actions only. The executor refuses a plan whose actions changed."""
        blob = json.dumps(self.bank_actions(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ToolCall(Strict):
    type: Literal["tool"]
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class PlanReply(Strict):
    type: Literal["plan"]
    plan: ActionPlan


class FinalReply(Strict):
    type: Literal["final"]
    reply_to_customer: str = Field(max_length=2000)


ResolverStep = TypeAdapter(Annotated[ToolCall | PlanReply, Field(discriminator="type")])
AgentStep = TypeAdapter(Annotated[ToolCall | FinalReply, Field(discriminator="type")])


class ComplianceInput(Strict):
    """Everything the compliance reviewer sees: ticket facts, the draft, policy excerpts."""

    channel: str
    ticket_text: str
    clarification: Clarification | None
    customer_profile: dict[str, Any]
    referenced_records: list[dict[str, Any]]
    draft: ActionPlan
    policy_excerpts: list[Excerpt]


class ComplianceReview(Strict):
    approve: bool
    issues: list[Annotated[str, Field(max_length=400)]] = Field(default_factory=list, max_length=8)


class ReviewDecision(Strict):
    """A human reviewer's answer to the approval interrupt."""

    decision: Literal["approve", "edit", "reject"]
    edited_plan: ActionPlan | None = None
    note: str = ""
    reviewer: str = "human"

    def model_post_init(self, _context: Any) -> None:
        if (self.decision == "edit") != (self.edited_plan is not None):
            raise ValueError("edited_plan is required for edit and only for edit")
