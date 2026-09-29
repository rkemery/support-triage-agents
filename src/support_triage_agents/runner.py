"""Run an arm over the tasks and write one harness `EvalRecord` per ticket per trial.

Each ticket run gets a fresh copy of the bank and its own checkpoint thread.
Human-review interrupts are answered by `GoldOracle`. Records go to
results/runs/<arm>/trial-<t>.jsonl, one run_id per trial, which is the shape
pass^k needs.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from llm_eval_harness import EvalRecord, ModelClient, write_records

from support_triage_agents.bank import Bank, diff_states
from support_triage_agents.data import Task, load_facts, load_seed
from support_triage_agents.graph import Runtime, build_graph, initial_state
from support_triage_agents.retrieval import HybridRetriever
from support_triage_agents.scoring import (
    COMPLETED_OUTCOMES,
    GoldOracle,
    sandbox_apply,
    score_ticket,
)
from support_triage_agents.single_agent import build_single_agent, initial_agent_state
from support_triage_agents.tools import CustomerSimulator
from support_triage_agents.vendor import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
WORK_DIR = REPO_ROOT / "runs"


@dataclass(frozen=True)
class Arm:
    key: str
    name: str
    label: str
    k: int

    @property
    def uses_graph(self) -> bool:
        return self.key in ("B", "C")

    @property
    def with_compliance(self) -> bool:
        return self.key == "B"


ARMS = {
    "A": Arm("A", "single-agent", "A: single agent", 4),
    "B": Arm("B", "graph", "B: full graph", 4),
    "C": Arm("C", "graph-no-review", "C: graph without compliance reviewer", 2),
}


def run_dir(arm: Arm, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "runs" / arm.name


_SEED_STATE: dict[str, Any] | None = None


def seed_state(workdir: Path) -> dict[str, Any]:
    """The seed as the SQLite bank exports it, the baseline every diff starts from."""
    global _SEED_STATE
    if _SEED_STATE is None:
        path = workdir / "seed-export.db"
        path.unlink(missing_ok=True)
        Bank.create(path, load_seed())
        with Bank(path, load_facts(), "cus_001", "seed") as bank:
            _SEED_STATE = bank.export_state()
        path.unlink()
    return _SEED_STATE


def run_ticket(
    arm: Arm,
    task: Task,
    trial: int,
    client: ModelClient,
    model: str,
    retriever: HybridRetriever,
    workdir: Path,
    checkpointer: SqliteSaver,
    *,
    item_errors: tuple[type[BaseException], ...] = (),
) -> EvalRecord:
    workdir.mkdir(parents=True, exist_ok=True)
    bank_path = workdir / f"{arm.name}-{task.task_id}-t{trial}.db"
    bank_path.unlink(missing_ok=True)
    Bank.create(bank_path, load_seed())
    rt = Runtime(
        client=client,
        model=model,
        trial=trial,
        task=task,
        bank_path=bank_path,
        facts=load_facts(),
        retriever=retriever,
        customer=CustomerSimulator(task),
    )
    oracle = GoldOracle(task)
    config = {"configurable": {"thread_id": f"{arm.name}:{task.task_id}:t{trial}:{time.time_ns()}"}}
    if arm.uses_graph:
        app = build_graph(rt, with_compliance=arm.with_compliance).compile(
            checkpointer=checkpointer
        )
        start_state: Any = initial_state(task)
    else:
        app = build_single_agent(rt).compile(checkpointer=checkpointer)
        start_state = initial_agent_state(task)
    model_error = None
    state: dict[str, Any] = {}
    try:
        state = app.invoke(start_state, config, durability="sync")
        while "__interrupt__" in state:
            decision = oracle.review(state["__interrupt__"][0].value)
            state = app.invoke(
                Command(resume=decision.model_dump(mode="json")), config, durability="sync"
            )
    except item_errors as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        state = dict(app.get_state(config).values)
    try:
        return _record(arm, task, trial, model, rt, state, oracle, model_error, workdir)
    finally:
        rt.close()
        bank_path.unlink(missing_ok=True)


def _record(
    arm: Arm,
    task: Task,
    trial: int,
    model: str,
    rt: Runtime,
    state: dict[str, Any],
    oracle: GoldOracle,
    model_error: str | None,
    workdir: Path,
) -> EvalRecord:
    diff = diff_states(seed_state(workdir), rt.bank.export_state())
    log = rt.bank.action_log()
    if arm.uses_graph:
        proposal = (state.get("proposed_plan") or {}).get("actions") or []
        attempted = [{"action": a["action"], "args": a.get("args") or {}} for a in proposal]
        sandbox = sandbox_apply(task, attempted)
        refusals = [r["reason"] for r in sandbox.refusals if r["kind"] == "policy"]
        invalid = [r["reason"] for r in sandbox.refusals if r["kind"] == "invalid"]
    else:
        tried = [e for e in log if e["outcome"] in ("applied", "refused")]
        attempted = [e["action"] for e in tried]
        refusals = [e["reason"] for e in tried if e["refusal_kind"] == "policy"]
        invalid = [e["reason"] for e in tried if e["refusal_kind"] == "invalid"]
    decision = state.get("decision") or {}
    rejected = decision.get("decision") == "reject"
    outcome = state.get("outcome") or ("model_error" if model_error else "unfinished")
    completed = outcome in COMPLETED_OUTCOMES and model_error is None
    score = score_ticket(task, diff, attempted, refusals, rejected=rejected, completed=completed)
    usage = state.get("usage") or {}
    clarification = state.get("clarification")
    asked = bool(clarification) if arm.uses_graph else int(state.get("asks", 0)) > 0
    meta = {
        "arm": arm.key,
        "trial": trial,
        "outcome": outcome,
        "failure": state.get("failure"),
        "model_error": model_error,
        "failure_categories": score.categories,
        "violations": score.violations,
        "invalid_writes": invalid,
        "attempted_writes": attempted,
        "applied_writes": rt.bank.applied_actions(),
        "rejected": rejected,
        "oracle_decisions": [d.decision for d in oracle.decisions],
        "should_escalate": task.should_escalate,
        "needs_clarification": task.needs_clarification,
        "asked_customer": asked,
        "llm_calls": int(usage.get("calls", 0)),
        "replayed_calls": int(usage.get("replayed_calls", 0)),
        "cached_tokens_in": int(usage.get("cached_tokens_in", 0)),
        "bounces": int(state.get("bounces", 0) or 0),
    }
    return EvalRecord(
        run_id=f"{arm.name}/trial-{trial}",
        item_id=task.task_id,
        config=arm.name,
        model=model,
        scores=score.scores(task),
        cluster=task.task_id,
        tokens_in=int(usage.get("tokens_in", 0)),
        tokens_out=int(usage.get("tokens_out", 0)),
        reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
        cost_usd=float(usage.get("cost_usd", 0.0)),
        latency_ms=float(usage.get("latency_ms", 0.0)),
        meta=meta,
    )


def run_arm(
    arm: Arm,
    tasks: Sequence[Task],
    client: ModelClient,
    model: str,
    retriever: HybridRetriever,
    *,
    trials: int | None = None,
    results_dir: Path = RESULTS_DIR,
    workdir: Path = WORK_DIR,
    item_errors: tuple[type[BaseException], ...] = (),
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Run k trials of an arm and write one JSONL file per trial. Returns the paths."""
    out_dir = run_dir(arm, results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_work = workdir / arm.name
    arm_work.mkdir(parents=True, exist_ok=True)
    paths = []
    ckpt_path = arm_work / "checkpoints.sqlite"
    ckpt_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(ckpt_path, check_same_thread=False)) as conn:
        saver = SqliteSaver(conn)
        for trial in range(trials if trials is not None else arm.k):
            records = []
            for task in tasks:
                rec = run_ticket(
                    arm, task, trial, client, model, retriever, arm_work, saver,
                    item_errors=item_errors,
                )  # fmt: skip
                records.append(rec)
                if progress is not None:
                    progress(
                        f"{arm.key} t{trial} {task.task_id}: success={rec.scores['success']} "
                        f"calls={rec.meta['llm_calls']} outcome={rec.meta['outcome']}"
                    )
            path = out_dir / f"trial-{trial}.jsonl"
            write_records(path, records)
            paths.append(path)
    return paths
