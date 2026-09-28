"""Second-model cross-check of the gold labels, instead of a human audit.

For each task, gpt-5-mini (a different model from the one under test) gets
the ticket, every hidden fact, the customer's full account records, the
support rules and the help-center excerpts for the ticket, and proposes the
write actions and whether to escalate. It agrees with the gold labels when
its actions, run through the bank, give the gold final state and its
escalation call matches `should_escalate`. Disagreements are listed, not
adjudicated by a person.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from llm_eval_harness import EvalRecord, ModelClient, write_records
from pydantic import Field

from support_triage_agents.bank import Bank, state_matches
from support_triage_agents.clients import CROSSCHECK_MODEL
from support_triage_agents.data import Task, load_facts, load_seed
from support_triage_agents.llm import AgentLLM, AgentOutputError, Usage, dump
from support_triage_agents.prompts import render
from support_triage_agents.retrieval import HybridRetriever
from support_triage_agents.schemas import ProposedAction, Strict
from support_triage_agents.scoring import sandbox_apply
from support_triage_agents.vendor import REPO_ROOT

CROSSCHECK_PATH = REPO_ROOT / "results" / "crosscheck.jsonl"
EXCERPTS = 6


class CrosscheckReply(Strict):
    actions: list[ProposedAction] = Field(default_factory=list, max_length=6)
    should_escalate: bool
    reason: str = Field(default="", max_length=1000)


def _context(task: Task, retriever: HybridRetriever) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.db"
        Bank.create(path, load_seed())
        with Bank(path, load_facts(), task.customer_id, task.task_id) as bank:
            records = {
                "profile": bank.get_customer_profile(),
                "accounts": bank.list_accounts(),
                "cards": bank.list_cards(),
                "transactions": bank.list_transactions(limit=100),
                "disputes": bank.list_disputes(),
            }
    excerpts = [
        f"[{h.chunk.article_id}] {h.chunk.title}\n{h.chunk.text}"
        for h in retriever.search(task.ticket_text, top_k=EXCERPTS)
    ]
    return "\n\n".join(
        [
            f"Ticket from customer {task.customer_id}:\n{task.ticket_text}",
            "What the customer would say if asked: " + " ".join(task.hidden_facts),
            f"Account records: {dump(records)}",
            "Policy excerpts:\n" + "\n\n".join(excerpts),
        ]
    )


def run_crosscheck(
    tasks: Sequence[Task],
    client: ModelClient,
    retriever: HybridRetriever,
    *,
    model: str = CROSSCHECK_MODEL,
    out_path: Path = CROSSCHECK_PATH,
) -> list[EvalRecord]:
    facts = load_facts()
    instructions = render("crosscheck", facts, str(load_seed()["as_of"]))
    records = []
    for task in tasks:
        usage = Usage()
        llm = AgentLLM(client, model, trial=0, reasoning_effort="low")
        context = _context(task, retriever)
        try:
            reply = llm.json_call(
                "crosscheck",
                instructions,
                [{"role": "user", "content": context}],
                CrosscheckReply,
                usage,
            )
        except AgentOutputError as exc:
            records.append(_record(task, model, usage, None, error=str(exc)))
            continue
        records.append(_record(task, model, usage, reply))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(out_path, records)
    return records


def _record(
    task: Task, model: str, usage: Usage, reply: CrosscheckReply | None, error: str | None = None
) -> EvalRecord:
    common = {
        "run_id": "crosscheck",
        "item_id": task.task_id,
        "config": "gold-crosscheck",
        "model": model,
        "cluster": task.task_id,
        "tokens_in": usage.tokens_in,
        "tokens_out": usage.tokens_out,
        "reasoning_tokens": usage.reasoning_tokens,
        "cost_usd": usage.cost_usd,
        "latency_ms": usage.latency_ms,
    }
    if reply is None:
        return EvalRecord(**common, scores={}, score_error=error)
    actions = [a.as_bank_action() for a in reply.actions]
    sandbox = sandbox_apply(task, actions)
    state_agrees = not sandbox.refusals and state_matches(sandbox.diff, task.gold_final_state)
    escalation_agrees = reply.should_escalate == task.should_escalate
    return EvalRecord(
        **common,
        scores={
            "agrees": state_agrees and escalation_agrees,
            "state_agrees": state_agrees,
            "escalation_agrees": escalation_agrees,
        },
        meta={
            "proposed_actions": actions,
            "refusals": sandbox.refusals,
            "should_escalate": reply.should_escalate,
            "reason": reply.reason,
            "gold_actions": list(task.gold_actions),
        },
    )
