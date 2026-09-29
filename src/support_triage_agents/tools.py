"""The tool layer every arm uses, and who may call what.

All arms call tools through one `Toolbox`, so the single agent and the graph
act on the same bank, search the same snapshot and talk to the same simulated
customer. What differs is the permission set of the caller:

| Role | Tools |
|---|---|
| intake | none (gets the customer profile in its prompt) |
| researcher | search_help_center |
| resolver | read-only account tools and ask_customer |
| compliance | none |
| executor | write tools (code, no model, only after approval) |
| single_agent | everything above plus ask_customer |

A call outside the caller's set raises `ToolPermissionError`. That is a bug
in the graph, not an agent mistake, so it is never caught.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from support_triage_agents.bank import WRITE_ACTIONS, ActionRefused, Bank
from support_triage_agents.data import Task
from support_triage_agents.retrieval import HybridRetriever

READ_TOOLS = frozenset(
    {"get_customer_profile", "list_accounts", "list_cards", "list_transactions", "list_disputes"}
)
WRITE_TOOLS = frozenset(WRITE_ACTIONS)
SEARCH_TOOL = "search_help_center"
ASK_TOOL = "ask_customer"

ROLE_TOOLS: dict[str, frozenset[str]] = {
    "intake": frozenset(),
    "researcher": frozenset({SEARCH_TOOL}),
    "resolver": READ_TOOLS | {ASK_TOOL},
    "compliance": frozenset(),
    "executor": WRITE_TOOLS,
    "single_agent": READ_TOOLS | WRITE_TOOLS | {SEARCH_TOOL, ASK_TOOL},
}

TOOL_SPECS: dict[str, str] = {
    "get_customer_profile": "get_customer_profile(): plan, status, billing dates of the customer",
    "list_accounts": "list_accounts(): the customer's accounts and balances",
    "list_cards": "list_cards(): the customer's cards (id, kind, status, last4)",
    "list_transactions": "list_transactions(limit=40): recent transactions, newest first",
    "list_disputes": "list_disputes(): the customer's disputes",
    "search_help_center": "search_help_center(query): top help-center excerpts for a query",
    "ask_customer": "ask_customer(question): ask the customer one clarifying question",
    "freeze_card": "freeze_card(card_id)",
    "unfreeze_card": "unfreeze_card(card_id)",
    "report_card_lost_stolen": "report_card_lost_stolen(card_id, reason: lost|stolen)",
    "order_replacement_card": (
        "order_replacement_card(card_id, reason: lost|stolen|damaged, shipping: standard|expedited)"
    ),
    "open_dispute": (
        "open_dispute(txn_id, reason: unauthorized|merchant_not_received|"
        "merchant_not_as_described|duplicate|cancelled_recurring|atm_cash_not_dispensed)"
    ),
    "refund_fee": "refund_fee(txn_id, basis: error|goodwill|cooling_off)",
    "change_plan": (
        "change_plan(new_plan: basic|plus|premium, effective: immediate|next_billing_date)"
    ),
    "close_account": "close_account()",
    "escalate_to_human": (
        "escalate_to_human(queue: fraud|disputes|verification|account_services|complaints, summary)"
    ),
}

SEARCH_TOP_K = 4
MAX_ASKS = 1
TOOL_RESULT_CHARS = 6000


class ToolPermissionError(PermissionError):
    """A role called a tool outside its permission set."""


class CustomerSimulator:
    """A scripted customer. Any clarifying question gets every hidden fact, once.

    Deterministic, so replays match. It is generous on purpose: a real
    customer might answer only what was asked.
    """

    def __init__(self, task: Task) -> None:
        self._facts = task.hidden_facts
        self.questions: list[str] = []

    def answer(self, question: str) -> str:
        self.questions.append(question)
        if len(self.questions) > MAX_ASKS:
            return "(The customer has already answered and has nothing to add.)"
        if not self._facts:
            return "I don't have anything else to add."
        return " ".join(self._facts)


@dataclass
class Toolbox:
    bank: Bank
    retriever: HybridRetriever
    customer: CustomerSimulator

    def call(self, role: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool for a role. Returns a JSON-safe result, or an error the agent can read."""
        if name not in ROLE_TOOLS[role]:
            raise ToolPermissionError(f"{role} may not call {name}")
        handler = self._handlers().get(name)
        if handler is not None:
            result = self._run(handler, args)
        elif name in WRITE_TOOLS:
            result = self._write(name, args)
        else:
            raise ToolPermissionError(f"unknown tool {name}")
        return result

    def _handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_customer_profile": self.bank.get_customer_profile,
            "list_accounts": self.bank.list_accounts,
            "list_cards": self.bank.list_cards,
            "list_transactions": self.bank.list_transactions,
            "list_disputes": self.bank.list_disputes,
            SEARCH_TOOL: self.search,
            ASK_TOOL: self.ask,
        }

    @staticmethod
    def _run(handler: Callable[..., Any], args: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "result": handler(**args)}
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments: {exc}"}
        except ActionRefused as exc:
            return {"ok": False, "error": exc.reason}

    def _write(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.bank.apply({"action": name, "args": args})
        except ActionRefused as exc:
            return {"ok": False, "refused": exc.kind, "error": exc.reason}
        return {"ok": True, "outcome": result.outcome, "result": result.detail}

    def search(self, query: str) -> list[dict[str, str]]:
        hits = self.retriever.search(str(query), top_k=SEARCH_TOP_K)
        return [
            {
                "chunk_id": h.chunk.chunk_id,
                "article_id": h.chunk.article_id,
                "title": h.chunk.title,
                "effective_date": h.chunk.effective_date,
                "text": h.chunk.text,
            }
            for h in hits
        ]

    def ask(self, question: str) -> str:
        return self.customer.answer(str(question))


def channel_line(customer_id: str) -> str:
    """How every ticket arrives, as the scenario defines it. The same line for every agent.

    It describes the session, not the writer: a relative typing in the account
    holder's app is still someone who is not the account holder.
    """
    return (
        f"Channel: in-app chat, sent from the signed-in, identity-verified app session of "
        f"customer {customer_id}."
    )


def excerpt_text(excerpt: Any) -> str:
    """A help-center excerpt as agents see it, with its effective date, as the RAG repo shows it."""
    return (
        f"[{excerpt.article_id}] {excerpt.title} (effective {excerpt.effective_date})\n"
        f"{excerpt.text}"
    )


def render_result(result: dict[str, Any]) -> str:
    text = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
    if len(text) > TOOL_RESULT_CHARS:
        text = text[:TOOL_RESULT_CHARS] + " ...(truncated)"
    return text


def support_playbook(facts: dict[str, Any]) -> str:
    """The support team's rules, rendered from the facts file (no hand-written policy)."""
    s = facts["support"]
    lines = ["Support may:"]
    lines += [f"- {item}" for item in s["may_do"]]
    lines.append("Support must escalate (queue: cases):")
    lines += [f"- {queue}: {', '.join(cases)}" for queue, cases in s["must_escalate"].items()]
    lines.append("Support must refuse:")
    lines += [f"- {item}" for item in s["must_refuse"]]
    lines.append(f"Identity: {s['identity']}")
    lines.append(f"Account takeover: {facts['security']['account_takeover_rule']}")
    return "\n".join(lines)
