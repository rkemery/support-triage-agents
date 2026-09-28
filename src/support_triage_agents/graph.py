"""The triage graph (arm B, and arm C without the compliance reviewer), as an explicit StateGraph.

    intake -> [clarify] -> researcher -> resolver <-> read tools
           -> compliance (B only, up to 2 bounce-backs to the resolver)
           -> human_review (interrupt: approve, edit or reject)
           -> executor (writes, only after approval) -> finish

The agents differ in what they may call and what they see, not only in their
prompts (see `tools.ROLE_TOOLS` and `schemas.ComplianceInput`). State is
checkpointed to SQLite after every node, so a killed process resumes from the
last finished node. `human_review` does nothing before `interrupt()`, because
a node re-runs from its top on resume. The executor's writes are idempotent
per ticket and action, so re-running it after a crash applies each write once.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from llm_eval_harness import ModelClient

from support_triage_agents.bank import Bank
from support_triage_agents.data import Task
from support_triage_agents.llm import (
    AgentLLM,
    AgentOutputError,
    StepBudgetExhausted,
    Usage,
    dump,
    merge_usage,
)
from support_triage_agents.prompts import render
from support_triage_agents.retrieval import HybridRetriever
from support_triage_agents.schemas import (
    ActionPlan,
    Clarification,
    ComplianceInput,
    ComplianceReview,
    Excerpt,
    IntakeResult,
    PlanReply,
    ResearchNotes,
    ResearchQueries,
    ResolverStep,
    ReviewDecision,
    ToolCall,
)
from support_triage_agents.tools import (
    ASK_TOOL,
    ROLE_TOOLS,
    SEARCH_TOOL,
    CustomerSimulator,
    Toolbox,
    channel_line,
    excerpt_text,
    render_result,
)

MAX_BOUNCES = 2
MAX_RESOLVER_TOOL_CALLS = 6
MAX_EXCERPTS = 8
# Exit code of a process killed on purpose by the crash hook (128 + SIGKILL).
CRASH_EXIT_CODE = 137


class TicketState(TypedDict, total=False):
    task_id: str
    customer_id: str
    ticket_text: str
    profile: dict[str, Any]
    intake: dict[str, Any]
    clarification: dict[str, Any] | None
    research: dict[str, Any]
    resolver_turns: list[dict[str, str]]
    plan: dict[str, Any] | None
    proposed_plan: dict[str, Any] | None
    reviews: list[dict[str, Any]]
    bounces: int
    bounce_back: bool
    decision: dict[str, Any] | None
    approved_fingerprint: str | None
    executed: list[dict[str, Any]]
    outcome: str
    failure: str | None
    usage: Annotated[dict[str, Any], merge_usage]


@dataclass
class Runtime:
    """What the nodes need that doesn't belong in checkpointed state (clients, connections)."""

    client: ModelClient
    model: str
    trial: int
    task: Task
    bank_path: Path
    facts: dict[str, Any]
    retriever: HybridRetriever
    customer: CustomerSimulator
    crash_after_writes: int | None = None
    _bank: Bank | None = field(default=None, repr=False)
    _toolbox: Toolbox | None = field(default=None, repr=False)

    @property
    def bank(self) -> Bank:
        if self._bank is None:
            self._bank = Bank(self.bank_path, self.facts, self.task.customer_id, self.task.task_id)
        return self._bank

    @property
    def toolbox(self) -> Toolbox:
        if self._toolbox is None:
            self._toolbox = Toolbox(self.bank, self.retriever, self.customer)
        return self._toolbox

    @property
    def as_of(self) -> str:
        return self.bank.as_of.isoformat()

    def llm(self, state: TicketState) -> AgentLLM:
        used = int((state.get("usage") or {}).get("calls", 0))
        return AgentLLM(self.client, self.model, self.trial, calls_used=used)

    def close(self) -> None:
        if self._bank is not None:
            self._bank.close()
            self._bank = None
            self._toolbox = None


def initial_state(task: Task) -> TicketState:
    """Only the ticket and the customer id enter the graph. Gold labels never do."""
    return {
        "task_id": task.task_id,
        "customer_id": task.customer_id,
        "ticket_text": task.ticket_text,
        "bounces": 0,
        "reviews": [],
        "resolver_turns": [],
        "executed": [],
        "usage": {},
    }


Node = Callable[[TicketState], dict[str, Any]]


def guarded(fn: Callable[[TicketState, Usage], dict[str, Any]]) -> Node:
    """Run a model-calling node. An unusable reply or an exhausted budget ends the ticket."""

    # No annotation on `state`: LangGraph would read one as this node's input schema and
    # drop every key the annotated type lacks (the single agent's state is a different type).
    def node(state):  # type: ignore[no-untyped-def]
        usage = Usage()
        try:
            update = fn(state, usage)
        except AgentOutputError as exc:
            update = {"outcome": "agent_error", "failure": str(exc)}
        except StepBudgetExhausted as exc:
            update = {"outcome": "budget_exhausted", "failure": str(exc)}
        return {**update, "usage": usage.as_dict()}

    node.__name__ = fn.__name__
    return node


def _ticket_block(state: TicketState) -> str:
    parts = [
        channel_line(state["customer_id"]),
        f"Ticket from customer {state['customer_id']}:\n{state['ticket_text']}",
    ]
    clar = state.get("clarification")
    if clar:
        parts.append(f"We asked: {clar['question']}\nThe customer answered: {clar['answer']}")
    return "\n\n".join(parts)


def build_graph(rt: Runtime, *, with_compliance: bool) -> StateGraph:
    def intake(state: TicketState, usage: Usage) -> dict[str, Any]:
        profile = rt.bank.get_customer_profile()
        result = rt.llm(state).json_call(
            "intake",
            render("intake", rt.facts, rt.as_of),
            [
                {
                    "role": "user",
                    "content": f"Customer profile: {dump(profile)}\n\n{_ticket_block(state)}",
                }
            ],
            IntakeResult,
            usage,
        )
        return {"profile": profile, "intake": result.model_dump(mode="json")}

    def clarify(state: TicketState) -> dict[str, Any]:
        question = state["intake"]["clarifying_question"]
        answer = rt.toolbox.customer.answer(question)
        return {"clarification": Clarification(question=question, answer=answer).model_dump()}

    def researcher(state: TicketState, usage: Usage) -> dict[str, Any]:
        intake_summary = state["intake"]["summary"]
        result = rt.llm(state).json_call(
            "researcher",
            render("researcher", rt.facts, rt.as_of),
            [
                {
                    "role": "user",
                    "content": f"{_ticket_block(state)}\n\nIntake summary "
                    f"({state['intake']['category']}): {intake_summary}",
                }
            ],
            ResearchQueries,
            usage,
        )
        excerpts: dict[str, Excerpt] = {}
        for query in result.queries:
            found = rt.toolbox.call("researcher", SEARCH_TOOL, {"query": query})
            for hit in found["result"]:
                excerpts.setdefault(hit["chunk_id"], Excerpt(**hit))
        notes = ResearchNotes(
            queries=result.queries, excerpts=list(excerpts.values())[:MAX_EXCERPTS]
        )
        return {"research": notes.model_dump()}

    def resolver_context(state: TicketState) -> list[dict[str, str]]:
        research = ResearchNotes(**state["research"])
        lines = [
            _ticket_block(state),
            f"Customer profile: {dump(state['profile'])}",
            f"Intake ({state['intake']['category']}): {state['intake']['summary']}",
            f"Intake risk flags: {state['intake']['risk_flags']}",
            "Policy excerpts from the researcher:",
        ]
        lines += [excerpt_text(e) for e in research.excerpts]
        reviews = state.get("reviews") or []
        if reviews and not reviews[-1]["approve"]:
            lines.append(
                "Your previous plan was sent back by compliance.\n"
                f"Previous plan: {dump(state['plan'])}\nIssues: {dump(reviews[-1]['issues'])}\n"
                "Write a corrected plan."
            )
        messages = [{"role": "user", "content": "\n\n".join(lines)}]
        return messages + list(state.get("resolver_turns") or [])

    def resolver(state: TicketState, usage: Usage) -> dict[str, Any]:
        turns = list(state.get("resolver_turns") or [])
        step = rt.llm(state).json_call(
            "resolver",
            render("resolver", rt.facts, rt.as_of),
            resolver_context(state),
            ResolverStep,
            usage,
        )
        if isinstance(step, PlanReply):
            return {"plan": step.plan.model_dump(mode="json"), "resolver_turns": []}
        assert isinstance(step, ToolCall)
        n_calls = len(turns) // 2
        if step.tool not in ROLE_TOOLS["resolver"]:
            result: dict[str, Any] = {
                "ok": False,
                "error": f"you can't call {step.tool}. Put write actions in your plan.",
            }
        elif n_calls >= MAX_RESOLVER_TOOL_CALLS:
            result = {"ok": False, "error": "tool call limit reached, write your plan now"}
        elif step.tool == ASK_TOOL:
            return _resolver_ask(state, step, turns)
        else:
            result = rt.toolbox.call("resolver", step.tool, step.args)
        turns += [
            {"role": "assistant", "content": dump(step)},
            {"role": "user", "content": f"Tool result: {render_result(result)}"},
        ]
        return {"resolver_turns": turns}

    def _resolver_ask(state: TicketState, step: ToolCall, turns: list) -> dict[str, Any]:
        """One question per ticket, shared with intake (the single agent gets one too)."""
        update: dict[str, Any] = {}
        if state.get("clarification"):
            result: dict[str, Any] = {"ok": False, "error": "the customer was already asked"}
        else:
            result = rt.toolbox.call("resolver", ASK_TOOL, step.args)
            question = str(step.args.get("question", ""))
            if result["ok"]:
                update["clarification"] = Clarification(
                    question=question, answer=str(result["result"])
                ).model_dump()
        turns = [
            *turns,
            {"role": "assistant", "content": dump(step)},
            {"role": "user", "content": f"Tool result: {render_result(result)}"},
        ]
        return {**update, "resolver_turns": turns}

    def compliance(state: TicketState, usage: Usage) -> dict[str, Any]:
        plan = ActionPlan(**state["plan"])
        review_input = ComplianceInput(
            channel=channel_line(state["customer_id"]),
            ticket_text=state["ticket_text"],
            clarification=Clarification(**state["clarification"])
            if state.get("clarification")
            else None,
            customer_profile=state["profile"],
            referenced_records=referenced_records(rt.bank, plan),
            draft=plan,
            policy_excerpts=ResearchNotes(**state["research"]).excerpts,
        )
        review = rt.llm(state).json_call(
            "compliance",
            render("compliance", rt.facts, rt.as_of),
            [{"role": "user", "content": dump(review_input)}],
            ComplianceReview,
            usage,
        )
        reviews = [*(state.get("reviews") or []), review.model_dump()]
        bounce = not review.approve and state.get("bounces", 0) < MAX_BOUNCES
        return {
            "reviews": reviews,
            "bounces": state.get("bounces", 0) + int(bounce),
            "bounce_back": bounce,
        }

    def human_review(state: TicketState) -> dict[str, Any]:
        plan = ActionPlan(**state["plan"])
        if not plan.actions:
            return {"proposed_plan": plan.model_dump(mode="json"), "decision": None}
        reviews = state.get("reviews") or []
        raw = interrupt(
            {
                "task_id": state["task_id"],
                "plan": plan.model_dump(mode="json"),
                "fingerprint": plan.fingerprint(),
                "compliance": reviews[-1] if reviews else None,
            }
        )
        decision = ReviewDecision.model_validate(raw)
        final = decision.edited_plan if decision.decision == "edit" else plan
        assert final is not None
        return {
            "proposed_plan": plan.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "plan": final.model_dump(mode="json"),
            "approved_fingerprint": final.fingerprint()
            if decision.decision in ("approve", "edit")
            else None,
        }

    def executor(state: TicketState) -> dict[str, Any]:
        decision = state.get("decision") or {}
        plan = ActionPlan(**state["plan"])
        if decision.get("decision") not in ("approve", "edit"):
            raise RuntimeError("executor reached without an approval")
        if plan.fingerprint() != state.get("approved_fingerprint"):
            raise RuntimeError("plan changed after approval")
        results, applied = [], 0
        for action in plan.bank_actions():
            result = rt.toolbox.call("executor", action["action"], action["args"])
            results.append({"action": action, "result": result})
            if not result["ok"]:
                break
            if result["outcome"] == "applied":
                applied += 1
                if rt.crash_after_writes is not None and applied >= rt.crash_after_writes:
                    rt.close()
                    os._exit(CRASH_EXIT_CODE)  # simulate a kill -9 mid-execution
        refused = any(not r["result"]["ok"] for r in results)
        return {"executed": results, "outcome": "execution_refused" if refused else "executed"}

    def finish(state: TicketState) -> dict[str, Any]:
        if state.get("outcome"):
            return {}
        decision = state.get("decision")
        if decision and decision["decision"] == "reject":
            return {"outcome": "rejected"}
        return {"outcome": "replied"}

    def after_intake(state: TicketState) -> str:
        if state.get("outcome"):
            return "finish"
        intake_result = state["intake"]
        if (
            intake_result["needs_clarification"]
            and intake_result.get("clarifying_question")
            and not state.get("clarification")
        ):
            return "clarify"
        return "researcher"

    def after_researcher(state: TicketState) -> str:
        return "finish" if state.get("outcome") else "resolver"

    def after_resolver(state: TicketState) -> str:
        if state.get("outcome"):
            return "finish"
        if state.get("resolver_turns"):
            return "resolver"
        return "compliance" if with_compliance else "human_review"

    def after_compliance(state: TicketState) -> str:
        if state.get("outcome"):
            return "finish"
        return "resolver" if state.get("bounce_back") else "human_review"

    def after_review(state: TicketState) -> str:
        decision = state.get("decision")
        if decision and decision["decision"] in ("approve", "edit"):
            return "executor"
        return "finish"

    g = StateGraph(TicketState)
    g.add_node("intake", guarded(intake))
    g.add_node("clarify", clarify)
    g.add_node("researcher", guarded(researcher))
    g.add_node("resolver", guarded(resolver))
    g.add_node("human_review", human_review)
    g.add_node("executor", executor)
    g.add_node("finish", finish)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["clarify", "researcher", "finish"])
    g.add_edge("clarify", "researcher")
    g.add_conditional_edges("researcher", after_researcher, ["resolver", "finish"])
    targets = ["resolver", "finish", "compliance" if with_compliance else "human_review"]
    g.add_conditional_edges("resolver", after_resolver, targets)
    if with_compliance:
        g.add_node("compliance", guarded(compliance))
        g.add_conditional_edges(
            "compliance", after_compliance, ["resolver", "human_review", "finish"]
        )
    g.add_conditional_edges("human_review", after_review, ["executor", "finish"])
    g.add_edge("executor", "finish")
    g.add_edge("finish", END)
    return g


def referenced_records(bank: Bank, plan: ActionPlan) -> list[dict[str, Any]]:
    """The account records a plan's actions point at, read by code for the compliance reviewer."""
    cards = {c["card_id"]: c for c in bank.list_cards()}
    txns = {t["txn_id"]: t for t in bank.list_transactions(limit=100)}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in plan.actions:
        for key, table in (("card_id", cards), ("txn_id", txns)):
            rid = action.args.get(key)
            if rid and rid not in seen:
                seen.add(rid)
                out.append(table.get(rid, {key: rid, "note": "no such record for this customer"}))
        if action.action in ("close_account", "change_plan", "refund_fee"):
            for account in bank.list_accounts():
                if account["account_id"] not in seen:
                    seen.add(account["account_id"])
                    out.append(account)
        if action.action in ("close_account", "open_dispute"):
            for dispute in bank.list_disputes():
                if dispute["dispute_id"] not in seen:
                    seen.add(dispute["dispute_id"])
                    out.append(dispute)
    return out
