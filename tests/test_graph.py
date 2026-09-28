"""The graph and the single agent, driven by scripted clients (no model, no network)."""

from __future__ import annotations

import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from llm_eval_harness import FakeClient, ModelRequest

from support_triage_agents.bank import Bank, state_matches
from support_triage_agents.data import load_facts, load_seed, load_tasks, task_by_id
from support_triage_agents.graph import Runtime, build_graph, initial_state
from support_triage_agents.llm import STEP_BUDGET
from support_triage_agents.retrieval import cached_retriever
from support_triage_agents.runner import ARMS, run_ticket
from support_triage_agents.schemas import ComplianceInput
from support_triage_agents.scoring import GoldOracle, sandbox_apply, score_ticket
from support_triage_agents.scripted import scripted_client, scripted_reply
from support_triage_agents.tools import (
    ROLE_TOOLS,
    CustomerSimulator,
    Toolbox,
    ToolPermissionError,
    channel_line,
)

TASKS = load_tasks()


@pytest.fixture(scope="module")
def retriever():
    return cached_retriever(embed_new=False)


@pytest.fixture
def saver():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    yield SqliteSaver(conn)
    conn.close()


def _run(arm, task, client, retriever, tmp_path, saver, trial=0):
    return run_ticket(ARMS[arm], task, trial, client, "gpt-6-luna", retriever, tmp_path, saver)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_gold_path_succeeds_on_every_task(arm, retriever, tmp_path, saver):
    client = scripted_client()
    for task in TASKS:
        rec = _run(arm, task, client, retriever, tmp_path, saver)
        assert rec.scores["success"], (task.task_id, rec.meta)
        assert rec.meta["llm_calls"] <= STEP_BUDGET


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_a_refused_extra_write_fails_the_ticket(arm, retriever, tmp_path, saver):
    rec = _run(arm, task_by_id("task-004"), scripted_client(wrong=True), retriever, tmp_path, saver)
    assert not rec.scores["success"]
    assert rec.scores["policy_violation"]
    assert "policy_violation" in rec.meta["failure_categories"]
    if arm == "A":
        # The bank refused the bad write and kept the good one, so the state still matches.
        assert rec.scores["state_match"]
    else:
        assert rec.meta["oracle_decisions"] == ["reject"]
        assert rec.meta["applied_writes"] == []


def _dispute_anyway(request: ModelRequest) -> str:
    """Task-009: the dispute is outside its window. This agent tries it anyway."""
    tasks = {t.ticket_text: t for t in TASKS}
    reply = json.loads(scripted_reply(request, tasks))
    bad = {"action": "open_dispute", "args": {"txn_id": "txn_009_volta", "reason": "unauthorized"}}
    if reply.get("type") == "plan":
        reply["plan"]["actions"] = [{**bad, "why": "customer asked"}]
    if reply.get("type") == "final" and "open_dispute" not in json.dumps(request.input):
        reply = {"type": "tool", "tool": bad["action"], "args": bad["args"]}
    return json.dumps(reply)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_guardrail_rescue_is_not_a_success(arm, retriever, tmp_path, saver):
    """The refused (or rejected) dispute leaves the seed state, which is gold. Still a failure."""
    rec = _run(arm, task_by_id("task-009"), FakeClient(_dispute_anyway), retriever, tmp_path, saver)
    assert rec.scores["state_match"]
    assert not rec.scores["success"]
    assert rec.scores["policy_violation"]


def test_oracle_only_approves_or_rejects(retriever, tmp_path, saver):
    decisions = set()
    for wrong in (False, True):
        for task in TASKS[:12]:
            rec = _run("B", task, scripted_client(wrong=wrong), retriever, tmp_path, saver)
            decisions |= set(rec.meta["oracle_decisions"])
    assert decisions == {"approve", "reject"}


def _runtime(task, client, retriever, tmp_path) -> Runtime:
    path = tmp_path / f"{task.task_id}.db"
    path.unlink(missing_ok=True)
    Bank.create(path, load_seed())
    return Runtime(
        client, "gpt-6-luna", 0, task, path, load_facts(), retriever, CustomerSimulator(task)
    )


def test_nothing_is_written_before_approval_and_executor_calls_no_model(retriever, tmp_path, saver):
    task = task_by_id("task-001")
    client = scripted_client()
    rt = _runtime(task, client, retriever, tmp_path)
    app = build_graph(rt, with_compliance=True).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t"}}
    state = app.invoke(initial_state(task), config, durability="sync")
    assert "__interrupt__" in state
    assert rt.bank.applied_actions() == []
    calls_before = len(client.calls)
    app.invoke(Command(resume={"decision": "approve"}), config, durability="sync")
    assert len(client.calls) == calls_before  # human_review and executor made no model call
    assert len(rt.bank.applied_actions()) == 2
    rt.close()


def test_reject_executes_nothing(retriever, tmp_path, saver):
    task = task_by_id("task-001")
    rt = _runtime(task, scripted_client(), retriever, tmp_path)
    app = build_graph(rt, with_compliance=False).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t"}}
    app.invoke(initial_state(task), config, durability="sync")
    final = app.invoke(Command(resume={"decision": "reject"}), config, durability="sync")
    assert final["outcome"] == "rejected"
    assert rt.bank.applied_actions() == []
    rt.close()


def test_edit_executes_the_edited_plan(retriever, tmp_path, saver):
    task = task_by_id("task-002")  # gold: report lost, expedited replacement
    rt = _runtime(task, scripted_client(), retriever, tmp_path)
    app = build_graph(rt, with_compliance=True).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t"}}
    state = app.invoke(initial_state(task), config, durability="sync")
    plan = state["__interrupt__"][0].value["plan"]
    plan["actions"] = plan["actions"][:1]
    final = app.invoke(
        Command(resume={"decision": "edit", "edited_plan": plan}), config, durability="sync"
    )
    assert final["outcome"] == "executed"
    assert [a["action"] for a in rt.bank.applied_actions()] == ["report_card_lost_stolen"]
    rt.close()


def test_role_permissions():
    assert ROLE_TOOLS["compliance"] == frozenset()
    assert ROLE_TOOLS["intake"] == frozenset()
    assert ROLE_TOOLS["researcher"] == {"search_help_center"}
    assert not ROLE_TOOLS["resolver"] & ROLE_TOOLS["executor"]
    assert ROLE_TOOLS["resolver"] - {"ask_customer"} <= ROLE_TOOLS["single_agent"]
    assert not ROLE_TOOLS["resolver"] & ROLE_TOOLS["executor"]


def test_toolbox_enforces_permissions(retriever, tmp_path):
    task = task_by_id("task-004")
    Bank.create(tmp_path / "b.db", load_seed())
    with Bank(tmp_path / "b.db", load_facts(), task.customer_id, task.task_id) as bank:
        box = Toolbox(bank, retriever, CustomerSimulator(task))
        with pytest.raises(ToolPermissionError):
            box.call("resolver", "freeze_card", {"card_id": "card_004p"})
        with pytest.raises(ToolPermissionError):
            box.call("compliance", "list_cards", {})
        assert bank.applied_actions() == []


def test_resolver_asking_for_a_write_tool_gets_an_error_not_a_write(retriever, tmp_path, saver):
    task = task_by_id("task-004")
    tasks = {t.ticket_text: t for t in TASKS}

    def sneaky(request: ModelRequest) -> str:
        if (request.instructions or "").startswith("You are the resolver agent") and len(
            request.input
        ) == 1:
            return json.dumps(
                {"type": "tool", "tool": "freeze_card", "args": {"card_id": "card_004p"}}
            )
        return scripted_reply(request, tasks)

    rt = _runtime(task, FakeClient(sneaky), retriever, tmp_path)
    app = build_graph(rt, with_compliance=True).compile(checkpointer=saver)
    state = app.invoke(initial_state(task), {"configurable": {"thread_id": "t"}})
    assert "__interrupt__" in state
    assert rt.bank.applied_actions() == []
    rt.close()


def test_compliance_input_has_no_transcript_or_hidden_facts():
    fields = set(ComplianceInput.model_fields)
    assert fields == {
        "channel",
        "ticket_text",
        "clarification",
        "customer_profile",
        "referenced_records",
        "draft",
        "policy_excerpts",
    }


def test_hidden_facts_reach_no_agent_unless_the_customer_is_asked(retriever, tmp_path, saver):
    for task_id in ("task-001", "task-005"):
        task = task_by_id(task_id)
        client = scripted_client()
        _run("B", task, client, retriever, tmp_path, saver)
        blobs = [json.dumps(r.input) for r in client.calls]
        seen = any(task.hidden_facts[0] in b for b in blobs)
        assert seen == task.needs_clarification, task_id


def test_every_request_carries_the_trial_index(retriever, tmp_path, saver):
    task = task_by_id("task-004")
    for trial in (0, 3):
        client = scripted_client()
        _run("B", task, client, retriever, tmp_path, saver, trial=trial)
        assert {r.trial for r in client.calls} == {trial}


def test_step_budget_ends_a_looping_agent(retriever, tmp_path, saver):
    loop = FakeClient(lambda r: json.dumps({"type": "tool", "tool": "list_cards", "args": {}}))
    rec = _run("A", task_by_id("task-009"), loop, retriever, tmp_path, saver)
    assert rec.meta["outcome"] == "budget_exhausted"
    assert rec.meta["llm_calls"] == STEP_BUDGET
    assert not rec.scores["success"]  # task-009's gold is "do nothing", but it never finished


def test_unparseable_reply_gets_one_repair_then_fails(retriever, tmp_path, saver):
    tasks = {t.ticket_text: t for t in TASKS}
    seen = {"bad": 0}

    def flaky(request: ModelRequest) -> str:
        if (request.instructions or "").startswith("You are the intake agent") and seen["bad"] < 1:
            seen["bad"] += 1
            return "sure! here you go"
        return scripted_reply(request, tasks)

    rec = _run("B", task_by_id("task-004"), FakeClient(flaky), retriever, tmp_path, saver)
    assert rec.scores["success"]
    rec = _run(
        "B", task_by_id("task-004"), FakeClient(lambda r: "nope"), retriever, tmp_path, saver
    )
    assert rec.meta["outcome"] == "agent_error"
    assert rec.meta["llm_calls"] == 2
    assert "incomplete" in rec.meta["failure_categories"]


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_every_agent_sees_the_same_channel_line(arm, retriever, tmp_path, saver):
    task = task_by_id("task-035")  # someone typing in the account holder's app for her
    client = scripted_client()
    _run(arm, task, client, retriever, tmp_path, saver)
    line = channel_line(task.customer_id)
    assert client.calls
    for request in client.calls:
        assert line in json.dumps(request.input, ensure_ascii=False), request.instructions[:40]
    assert "Delphine's husband Marc" in json.dumps(client.calls[0].input)


def test_freezing_before_reporting_lost_is_the_same_end_state():
    """A reported card is cancelled for good, so a freeze before the report leaves no trace.

    The dataset's reference model and this bank both overwrite the card status on report."""
    task = task_by_id("task-001")
    plan = [{"action": "freeze_card", "args": {"card_id": "card_001p"}}, *task.gold_actions]
    result = sandbox_apply(task, plan)
    assert not result.refusals
    assert state_matches(result.diff, task.gold_final_state)
    score = score_ticket(task, result.diff, plan, [], rejected=False, completed=True)
    assert score.success
    oracle = GoldOracle(task)
    actions = [{**a, "why": ""} for a in plan]
    payload = {"plan": {"actions": actions, "reply_to_customer": "ok"}}
    assert oracle.review(payload).decision == "approve"


def test_resolver_can_ask_once_and_its_answer_reaches_the_reviewer(retriever, tmp_path, saver):
    task = task_by_id("task-005")  # "Please freeze my card." with two cards
    tasks = {t.ticket_text: t for t in TASKS}
    asks = {"n": 0}

    def resolver_asks(request: ModelRequest) -> str:
        role = request.instructions or ""
        if role.startswith("You are the intake agent"):
            reply = json.loads(scripted_reply(request, tasks))
            reply.update(needs_clarification=False, clarifying_question=None)
            return json.dumps(reply)
        if role.startswith("You are the resolver agent") and asks["n"] < 2:
            asks["n"] += 1
            return json.dumps(
                {"type": "tool", "tool": "ask_customer", "args": {"question": "Which card?"}}
            )
        return scripted_reply(request, tasks)

    client = FakeClient(resolver_asks)
    rec = _run("B", task, client, retriever, tmp_path, saver)
    assert rec.scores["success"]
    assert rec.meta["asked_customer"]
    compliance = [r for r in client.calls if r.instructions.startswith("You are the compliance")]
    assert task.hidden_facts[0] in json.dumps(compliance[0].input, ensure_ascii=False)
    blob = json.dumps([r.input for r in client.calls], ensure_ascii=False)
    assert "the customer was already asked" in blob


def test_excerpts_carry_their_effective_date(retriever, tmp_path, saver):
    client = scripted_client()
    _run("B", task_by_id("task-001"), client, retriever, tmp_path, saver)
    resolver = next(r for r in client.calls if r.instructions.startswith("You are the resolver"))
    assert "(effective 20" in resolver.input[0]["content"]
