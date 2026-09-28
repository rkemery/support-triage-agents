"""`triage`: every step of the project from one command.

Offline, no keys: verify-data, demo, estimate, resume-demo, ticket (scripted).
Model calls: run --live (needs AZURE_OPENAI_BASE_URL plus a key or Entra ID),
or run --replay to reproduce a live run from the committed cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from llm_eval_harness import BudgetExceeded, CacheMiss

from support_triage_agents import readme
from support_triage_agents.bank import Bank, diff_states, state_matches
from support_triage_agents.clients import (
    AGENT_MODEL,
    ClientStack,
    item_level_errors,
    live_stack,
    replay_stack,
)
from support_triage_agents.crosscheck import run_crosscheck
from support_triage_agents.data import Task, load_facts, load_seed, load_tasks, task_by_id
from support_triage_agents.graph import CRASH_EXIT_CODE, Runtime, build_graph, initial_state
from support_triage_agents.retrieval import QueryEmbeddingMiss, cached_retriever
from support_triage_agents.runner import ARMS, RESULTS_DIR, WORK_DIR, run_arm, seed_state
from support_triage_agents.scripted import scripted_client
from support_triage_agents.tools import CustomerSimulator
from support_triage_agents.vendor import REPO_ROOT, VendorError, verify_all

DEFAULT_CAP_USD = 3.00
MODEL_CACHE = REPO_ROOT / "cache" / "model"


def cmd_verify_data(_: argparse.Namespace) -> int:
    for name, files in verify_all().items():
        print(f"verify-data: {name}: {len(files)} files match MANIFEST.json")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cmd_verify_data(args)
    changed = readme.render(cap_usd=args.cap)
    print(f"demo: README results and cost sections {'rewritten' if changed else 'unchanged'}")
    print(readme.results_section())
    return 0


def cmd_estimate(_: argparse.Namespace) -> int:
    from support_triage_agents.estimate import estimate

    for s in estimate():
        print(
            f"{s.stage}: {s.calls} calls, expected ${s.expected_usd:.2f}, heavier path "
            f"${s.heavier_usd:.2f}, worst case ${s.worst_usd:.2f}, {s.minutes_at_quota:.0f} min"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.live == args.replay:
        print("run: pass exactly one of --live or --replay", file=sys.stderr)
        return 2
    stack = live_stack(MODEL_CACHE, args.cap) if args.live else replay_stack(MODEL_CACHE)
    retriever = cached_retriever(embed_new=args.live)
    tasks = load_tasks()
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.task_id in wanted]
    keys = ["A", "B", "C", "crosscheck"] if args.arm == "all" else [args.arm]
    try:
        for key in keys:
            if key == "crosscheck":
                records = run_crosscheck(
                    tasks, stack.client, retriever, out_path=args.results_dir / "crosscheck.jsonl"
                )
                agree = sum(bool(r.scores.get("agrees")) for r in records)
                print(f"crosscheck: {agree} of {len(records)} agree with the gold labels")
                continue
            run_arm(
                ARMS[key],
                tasks,
                stack.client,
                AGENT_MODEL,
                retriever,
                trials=args.trials,
                results_dir=args.results_dir,
                item_errors=item_level_errors(),
                progress=print if args.verbose else None,
            )
            print(f"run: arm {key} done")
    except (BudgetExceeded, CacheMiss, QueryEmbeddingMiss) as exc:
        print(f"run: stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        _log_live(stack, args)
        return 1
    _log_live(stack, args)
    print(json.dumps(stack.summary()))
    return 0


def _log_live(stack: ClientStack, args: argparse.Namespace) -> None:
    if stack.mode != "live":
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {**stack.summary(), "arm": args.arm, "tasks": args.tasks or "all"}
    with (RESULTS_DIR / "live_runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------- one ticket, resumable


def cmd_ticket(args: argparse.Namespace) -> int:
    """Run one ticket through the graph with a scripted client, in resumable steps."""
    task = task_by_id(args.task)
    workdir = Path(args.workdir)
    if args.step == "start":
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        Bank.create(workdir / "bank.db", load_seed())
    rt = Runtime(
        client=scripted_client(),
        model=AGENT_MODEL,
        trial=0,
        task=task,
        bank_path=workdir / "bank.db",
        facts=load_facts(),
        retriever=cached_retriever(embed_new=False),
        customer=CustomerSimulator(task),
        crash_after_writes=args.crash_after_writes,
    )
    config = {"configurable": {"thread_id": f"{args.arm}:{task.task_id}"}}
    with closing(sqlite3.connect(workdir / "checkpoints.sqlite", check_same_thread=False)) as conn:
        graph = build_graph(rt, with_compliance=ARMS[args.arm].with_compliance)
        app = graph.compile(checkpointer=SqliteSaver(conn))
        if args.step == "start":
            state = app.invoke(initial_state(task), config, durability="sync")
        elif args.step == "resume":
            state = app.invoke(
                Command(resume={"decision": args.decision}), config, durability="sync"
            )
        else:  # continue after a crash, from the last checkpoint
            state = app.invoke(None, config, durability="sync")
        if "__interrupt__" in state:
            pending = state["__interrupt__"][0].value
            print(
                f"ticket: paused for human review. Plan: {json.dumps(pending['plan']['actions'])}"
            )
        else:
            print(f"ticket: finished with outcome {state.get('outcome')}")
    _print_bank(rt, task)
    rt.close()
    return 0


def _print_bank(rt: Runtime, task: Task) -> None:
    log = rt.bank.action_log()
    print("ticket: bank action log:", [(e["action"]["action"], e["outcome"]) for e in log])
    diff = diff_states(seed_state(WORK_DIR), rt.bank.export_state())
    print(f"ticket: bank state matches gold: {state_matches(diff, task.gold_final_state)}")


def cmd_resume_demo(args: argparse.Namespace) -> int:
    """Pause at approval, kill the process mid-execution, resume, and check nothing ran twice."""
    workdir = str(args.workdir)
    base = [sys.executable, "-m", "support_triage_agents.cli", "ticket", "--task", args.task]
    base += ["--arm", "B", "--workdir", workdir]
    steps = [
        ("1. Run until the graph pauses for human approval", ["--step", "start"], 0),
        (
            "2. Approve, and kill the process right after the first write",
            ["--step", "resume", "--decision", "approve", "--crash-after-writes", "1"],
            CRASH_EXIT_CODE,
        ),
        ("3. Start a new process and continue from the last checkpoint", ["--step", "continue"], 0),
    ]
    for title, extra, expected in steps:
        print(f"\n{title}")
        proc = subprocess.run([*base, *extra], capture_output=True, text=True, check=False)
        print(proc.stdout.strip())
        if proc.returncode != expected:
            print(proc.stderr, file=sys.stderr)
            print(f"resume-demo: exit code {proc.returncode}, expected {expected}", file=sys.stderr)
            return 1
        if expected == CRASH_EXIT_CODE:
            print(f"(process exited with code {proc.returncode}, as a kill -9 would)")
    task = task_by_id(args.task)
    with Bank(Path(workdir) / "bank.db", load_facts(), task.customer_id, task.task_id) as bank:
        applied = bank.applied_actions()
        outcomes = [e["outcome"] for e in bank.action_log()]
        diff = diff_states(seed_state(WORK_DIR), bank.export_state())
    ok = state_matches(diff, task.gold_final_state) and len(applied) == len(task.gold_actions)
    print(
        f"\nresume-demo: {len(applied)} writes applied once each, log {outcomes}, "
        f"final state matches gold: {ok}"
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-data", help="check vendored data hashes").set_defaults(
        fn=cmd_verify_data
    )
    demo = sub.add_parser("demo", help="offline: rewrite the README results and cost sections")
    demo.add_argument("--cap", type=float, default=DEFAULT_CAP_USD)
    demo.set_defaults(fn=cmd_demo)
    sub.add_parser("estimate", help="price the live run").set_defaults(fn=cmd_estimate)
    run = sub.add_parser("run", help="run arms (and the gold cross-check) live or from the cache")
    run.add_argument("--arm", choices=["A", "B", "C", "crosscheck", "all"], default="all")
    run.add_argument("--live", action="store_true")
    run.add_argument("--replay", action="store_true")
    run.add_argument("--cap", type=float, default=DEFAULT_CAP_USD, help="dollar cap (live)")
    run.add_argument("--trials", type=int, default=None, help="override k (for smoke runs)")
    run.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    run.add_argument("--verbose", action="store_true")
    run.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="where records go (default results/)"
    )
    run.set_defaults(fn=cmd_run)
    ticket = sub.add_parser("ticket", help="one ticket, scripted client, in resumable steps")
    ticket.add_argument("--task", required=True)
    ticket.add_argument("--arm", choices=["B", "C"], default="B")
    ticket.add_argument("--workdir", required=True)
    ticket.add_argument("--step", choices=["start", "resume", "continue"], required=True)
    ticket.add_argument("--decision", choices=["approve", "reject"], default="approve")
    ticket.add_argument("--crash-after-writes", type=int, default=None)
    ticket.set_defaults(fn=cmd_ticket)
    resume = sub.add_parser("resume-demo", help="kill mid-execution, resume, check idempotency")
    resume.add_argument("--task", default="task-001")
    resume.add_argument("--workdir", type=Path, default=WORK_DIR / "resume-demo")
    resume.set_defaults(fn=cmd_resume_demo)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except VendorError as exc:
        print(f"triage: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
