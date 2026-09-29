"""Analysis, README rendering, the client stack pieces, the estimate and the cross-check."""

from __future__ import annotations

import json
import random
import re

import pytest
from llm_eval_harness import EvalRecord, FakeClient, ModelRequest, write_records

from support_triage_agents import analysis, readme
from support_triage_agents.cli import main
from support_triage_agents.clients import RateLimitedClient, deployment_tpm
from support_triage_agents.crosscheck import run_crosscheck
from support_triage_agents.data import load_tasks
from support_triage_agents.estimate import _stage
from support_triage_agents.retrieval import cached_retriever
from support_triage_agents.runner import ARMS, run_dir
from support_triage_agents.scripted import scripted_client, scripted_reply

TASKS = load_tasks()


def _synthetic_trials(arm_key: str, p_success: float, p_violation: float, seed: int):
    """Made-up records in the runner's shape, for exercising the statistics only."""
    rng = random.Random(seed)
    arm = ARMS[arm_key]
    trials = []
    for t in range(arm.k):
        records = []
        for task in TASKS:
            violation = rng.random() < p_violation
            success = (not violation) and rng.random() < p_success
            escalated = task.should_escalate if rng.random() < 0.9 else not task.should_escalate
            records.append(
                EvalRecord(
                    run_id=f"{arm.name}/trial-{t}",
                    item_id=task.task_id,
                    config=arm.name,
                    model="gpt-6-luna",
                    scores={
                        "success": success,
                        "state_match": success or violation,
                        "policy_violation": violation,
                        "escalated": escalated,
                        "escalation_correct": escalated == task.should_escalate,
                    },
                    cluster=task.task_id,
                    tokens_in=3000,
                    tokens_out=300,
                    cost_usd=0.0005,
                    latency_ms=rng.uniform(2000, 9000),
                    meta={
                        "should_escalate": task.should_escalate,
                        "failure_categories": [] if success else ["missing_action"],
                    },
                )
            )
        trials.append(records)
    return trials


def test_pending_rows_and_offline_checks_render_without_live_results(tmp_path):
    text = readme.results_section(tmp_path)
    assert "pending live run" in text
    assert "Verdict: pending live run." in text
    detail = readme.results_detail_section(tmp_path)
    for section in (text, detail):
        assert "| Do nothing (reference) | 26.0% (" in section
        assert "| Gold actions (reference) | 100.0% (" in section
    assert "for 50 of 50 tasks" in detail
    order = re.search(r"same order for (\d+) of 200 questions, and as a set for 200 of 200", detail)
    assert order is not None
    assert int(order.group(1)) >= 198


def test_live_rows_render_from_records(tmp_path):
    for key, (ps, pv, seed) in {
        "A": (0.6, 0.3, 1),
        "B": (0.6, 0.05, 2),
        "C": (0.6, 0.2, 3),
    }.items():
        for t, records in enumerate(_synthetic_trials(key, ps, pv, seed)):
            write_records(run_dir(ARMS[key], tmp_path) / f"trial-{t}.jsonl", records)
    text = readme.results_section(tmp_path)
    arm_rows = [line for line in text.splitlines() if line.startswith(("| A:", "| B:", "| C:"))]
    assert len(arm_rows) == 3
    assert all("pending" not in row for row in arm_rows)
    assert "Verdict: supported." in text  # far fewer violations in B, same success rate
    assert "k=4" in arm_rows[0]
    assert "k=2" in arm_rows[2]


def test_compare_arms_pairs_trials_by_task():
    a = _synthetic_trials("A", 0.5, 0.3, 1)
    b = _synthetic_trials("B", 0.5, 0.05, 2)
    cmp = analysis.compare_arms(a, b, "policy_violation")
    assert cmp.clustered.comparison.n == 4 * len(TASKS)
    assert cmp.clustered.comparison.n_clusters == len(TASKS)
    assert cmp.clustered.comparison.diff < 0
    c = _synthetic_trials("C", 0.5, 0.2, 3)
    assert analysis.compare_arms(c, b, "success").n_trials == 2


def test_escalation_precision_and_recall():
    trials = _synthetic_trials("A", 0.5, 0.1, 4)
    records = analysis.flatten(trials)
    precision = analysis.escalation_precision(records)
    recall = analysis.escalation_recall(records)
    escalated = [r for r in records if r.scores["escalated"]]
    tp = sum(r.meta["should_escalate"] for r in escalated)
    assert precision.estimate == pytest.approx(tp / len(escalated))
    should = [r for r in records if r.meta["should_escalate"]]
    assert recall.estimate == pytest.approx(
        sum(r.scores["escalated"] for r in should) / len(should)
    )


def test_dollars_per_resolved_ticket():
    records = analysis.flatten(_synthetic_trials("B", 0.5, 0.0, 5))
    s = analysis.spend(records)
    wins = sum(r.scores["success"] for r in records)
    assert s.per_resolved.estimate == pytest.approx(0.0005 * len(records) / wins)
    assert s.per_resolved.low <= s.per_resolved.estimate <= s.per_resolved.high


def test_pass_k_needs_k_trials():
    trials = _synthetic_trials("A", 0.7, 0.0, 6)
    result = analysis.pass_hat_k(trials, 4)
    assert (
        result.pass_hat_k.estimate
        <= analysis.pooled_rate(analysis.flatten(trials), "success").estimate
    )


def test_rate_limiter_waits_instead_of_exceeding_the_window():
    now = [0.0]
    slept = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    inner = FakeClient(lambda r: "ok")
    client = RateLimitedClient(inner, {"m": 1000}, headroom=1.0, clock=lambda: now[0], sleep=sleep)
    request = ModelRequest(model="m", input="x" * 100, max_output_tokens=400)
    for _ in range(3):
        client.complete(request)
    assert slept  # the third request had to wait for the window to roll
    assert len(inner.calls) == 3


def test_tpm_override_parses_and_rejects_garbage():
    assert deployment_tpm({"TRIAGE_TPM": "gpt-6-luna=100000"})["gpt-6-luna"] == 100_000
    with pytest.raises(ValueError, match="TRIAGE_TPM"):
        deployment_tpm({"TRIAGE_TPM": "gpt-6-luna=lots"})


def test_estimate_stage_prices_recorded_requests():
    client = scripted_client()
    tasks = {t.ticket_text: t for t in TASKS}
    request = ModelRequest(
        model="gpt-6-luna",
        input=[{"role": "user", "content": TASKS[0].ticket_text}],
        instructions="You are the intake agent. " + "x" * 400,
        max_output_tokens=400,
    )
    client.complete(request)
    assert scripted_reply(request, tasks)
    stage = _stage("intake", "gpt-6-luna", client, repeats=4)
    assert stage.calls == 4
    assert 0 < stage.expected_usd < stage.worst_usd


def test_crosscheck_agrees_with_gold_and_flags_a_wrong_answer(tmp_path):
    retriever = cached_retriever(embed_new=False)
    records = run_crosscheck(
        TASKS[:5], scripted_client(), retriever, out_path=tmp_path / "cc.jsonl"
    )
    assert all(r.scores["agrees"] for r in records)
    tasks = {t.ticket_text: t for t in TASKS}

    def disagree(request: ModelRequest) -> str:
        reply = json.loads(scripted_reply(request, tasks))
        reply["should_escalate"] = not reply["should_escalate"]
        return json.dumps(reply)

    records = run_crosscheck(
        TASKS[:2], FakeClient(disagree), retriever, out_path=tmp_path / "cc.jsonl"
    )
    assert not any(r.scores["agrees"] for r in records)
    assert all(r.scores["state_agrees"] for r in records)


def test_kill_and_resume_demo(tmp_path, capsys):
    assert main(["resume-demo", "--workdir", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "process exited with code 137" in out
    assert "log ['applied', 'duplicate', 'applied'], final state matches gold: True" in out
