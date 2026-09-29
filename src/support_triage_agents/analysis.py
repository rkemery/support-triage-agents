"""Turn per-ticket records into the numbers the README reports, with harness statistics.

Shapes:
- pass^k: one run per trial, item_id = task (what `pass_k_from_records` wants).
- Pooled rates over all trials: one value per ticket run, clustered by task,
  with the harness's clustered Wilson interval. Trials of the same task are
  not independent, and clustering is what keeps the interval honest.
- Arm vs arm: trial t of one arm paired with trial t of the other on the
  same task (item_id = task#t, cluster = task), through the harness's
  clustered paired t-test, plus an exact McNemar test on trial 0 alone
  (one pair per task, so the pairs are independent).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from llm_eval_harness import EvalRecord, read_records
from llm_eval_harness.analysis import RunComparison, compare_runs, pass_k_from_records
from llm_eval_harness.stats import (
    Interval,
    McNemarResult,
    PassKResult,
    mcnemar_exact,
    percentile_interval,
    wilson_interval_clustered,
)

from support_triage_agents.data import Task
from support_triage_agents.runner import RESULTS_DIR, Arm, run_dir
from support_triage_agents.scoring import FAILURE_CATEGORIES, sandbox_apply, score_ticket


def load_trials(arm: Arm, n_tasks: int, results_dir: Path = RESULTS_DIR) -> list[list[EvalRecord]]:
    """Every complete trial file of an arm, in trial order. Empty when the arm has not run."""
    trials = []
    for t in range(arm.k):
        path = run_dir(arm, results_dir) / f"trial-{t}.jsonl"
        if not path.exists():
            break
        records = read_records(path)
        if len(records) != n_tasks:
            raise ValueError(f"{path} has {len(records)} records, expected {n_tasks}")
        trials.append(records)
    return trials


def flatten(trials: Sequence[Sequence[EvalRecord]]) -> list[EvalRecord]:
    return [r for trial in trials for r in trial]


def pooled_rate(records: Sequence[EvalRecord], metric: str) -> Interval:
    values = np.array([float(r.scores[metric]) for r in records])
    return wilson_interval_clustered(values, [r.item_id for r in records])


def _subset_rate(records: Sequence[EvalRecord], metric: str, keep: str) -> Interval | None:
    subset = [r for r in records if r.scores[keep]]
    if len({r.item_id for r in subset}) < 2:
        return None
    return pooled_rate(subset, metric)


def escalation_precision(records: Sequence[EvalRecord]) -> Interval | None:
    """Of the tickets an arm escalated, the share that should have been."""
    relabeled = [
        replace(r, scores={**r.scores, "tp": bool(r.meta["should_escalate"])}) for r in records
    ]
    return _subset_rate(relabeled, "tp", "escalated")


def escalation_recall(records: Sequence[EvalRecord]) -> Interval | None:
    relabeled = [
        replace(r, scores={**r.scores, "should": bool(r.meta["should_escalate"])}) for r in records
    ]
    return _subset_rate(relabeled, "escalated", "should")


def pass_hat_k(trials: Sequence[Sequence[EvalRecord]], k: int) -> PassKResult:
    return pass_k_from_records(flatten(trials), "success", k)


@dataclass(frozen=True)
class Spend:
    per_resolved: Interval | None
    tokens_per_ticket: float
    p50_s: float
    p95_s: float


def spend(records: Sequence[EvalRecord], seed: int = 0) -> Spend:
    """Dollars per resolved ticket (total cost / successes), bootstrapped over tasks."""
    by_task: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_task.setdefault(r.item_id, []).append(r)
    cost = np.array([sum(r.cost_usd for r in rs) for rs in by_task.values()])
    wins = np.array([sum(bool(r.scores["success"]) for r in rs) for rs in by_task.values()])
    per_resolved = None
    if wins.sum() > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, cost.size, size=(10_000, cost.size))
        num, den = cost[idx].sum(axis=1), wins[idx].sum(axis=1)
        reps = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        per_resolved = percentile_interval(
            reps, float(cost.sum() / wins.sum()), cost.size, method="bootstrap over tasks"
        )
    latencies = np.array([r.latency_ms for r in records]) / 1000.0
    return Spend(
        per_resolved=per_resolved,
        tokens_per_ticket=float(np.mean([r.tokens_in + r.tokens_out for r in records])),
        p50_s=float(np.percentile(latencies, 50)),
        p95_s=float(np.percentile(latencies, 95)),
    )


def failure_counts(records: Sequence[EvalRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for r in records:
        counts.update(r.meta["failure_categories"])
    return Counter({c: counts[c] for c in FAILURE_CATEGORIES})


@dataclass(frozen=True)
class ArmComparison:
    metric: str
    clustered: RunComparison
    mcnemar_trial0: McNemarResult
    n_trials: int


def _paired(trials: Sequence[Sequence[EvalRecord]], n: int, name: str) -> list[EvalRecord]:
    out = []
    for t, trial in enumerate(trials[:n]):
        for r in trial:
            out.append(replace(r, run_id=f"{name}/paired", item_id=f"{r.item_id}#t{t}"))
    return out


def compare_arms(
    baseline: Sequence[Sequence[EvalRecord]],
    candidate: Sequence[Sequence[EvalRecord]],
    metric: str,
) -> ArmComparison:
    """Candidate minus baseline on trial-matched pairs, clustered by task."""
    n = min(len(baseline), len(candidate))
    if n == 0:
        raise ValueError("both arms need at least one trial")
    base = _paired(baseline, n, "baseline")
    cand = _paired(candidate, n, "candidate")
    clustered = compare_runs(base, cand, metric, use_clusters=True)
    b0 = {r.item_id: bool(r.scores[metric]) for r in baseline[0]}
    c0 = {r.item_id: bool(r.scores[metric]) for r in candidate[0]}
    ids = sorted(b0)
    mcnemar = mcnemar_exact([b0[i] for i in ids], [c0[i] for i in ids])
    return ArmComparison(metric, clustered, mcnemar, n)


# ---------------------------------------------------------------------- reference policies


@dataclass(frozen=True)
class ReferenceRow:
    name: str
    success: Interval
    state_match: Interval
    policy_violation: Interval
    precision: Interval | None
    recall: Interval | None


def _reference_records(name: str, tasks: Sequence[Task], use_gold: bool) -> list[EvalRecord]:
    records = []
    for task in tasks:
        attempted = [dict(a) for a in task.gold_actions] if use_gold else []
        sandbox = sandbox_apply(task, attempted)
        refusals = [r["reason"] for r in sandbox.refusals if r["kind"] == "policy"]
        score = score_ticket(
            task, sandbox.diff, attempted, refusals, rejected=False, completed=True
        )
        records.append(
            EvalRecord(
                run_id=name,
                item_id=task.task_id,
                config=name,
                model="none",
                scores=score.scores(task),
                cluster=task.task_id,
                meta={"should_escalate": task.should_escalate, "failure_categories": []},
            )
        )
    return records


def reference_rows(tasks: Sequence[Task]) -> list[ReferenceRow]:
    """Policies computed by code through the same scorer: they bracket what an arm can score."""
    rows = []
    for name, use_gold in (("Do nothing (reference)", False), ("Gold actions (reference)", True)):
        recs = _reference_records(name, tasks, use_gold)
        rows.append(
            ReferenceRow(
                name=name,
                success=pooled_rate(recs, "success"),
                state_match=pooled_rate(recs, "state_match"),
                policy_violation=pooled_rate(recs, "policy_violation"),
                precision=escalation_precision(recs),
                recall=escalation_recall(recs),
            )
        )
    return rows
