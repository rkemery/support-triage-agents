"""Render the README's results, results detail and cost sections from committed files.

No model calls.

Rows whose results don't exist yet print "pending live run" instead of a number.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from llm_eval_harness import EvalRecord, read_records
from llm_eval_harness.report import write_section
from llm_eval_harness.stats import Interval

from support_triage_agents import analysis
from support_triage_agents.clients import AGENT_MODEL, CROSSCHECK_MODEL, DEPLOYMENT_TPM
from support_triage_agents.crosscheck import CROSSCHECK_PATH
from support_triage_agents.data import Task, load_tasks
from support_triage_agents.estimate import load_estimate
from support_triage_agents.llm import STEP_BUDGET
from support_triage_agents.runner import ARMS, RESULTS_DIR, Arm
from support_triage_agents.scoring import FAILURE_CATEGORIES, sandbox_apply
from support_triage_agents.vendor import REPO_ROOT

PENDING = "pending live run"
README = REPO_ROOT / "README.md"
EQUIVALENCE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "rag_equivalence"


def _pct(interval: Interval | None) -> str:
    if interval is None:
        return "n/a"
    return f"{interval.estimate * 100:.1f}% ({interval.low * 100:.1f} to {interval.high * 100:.1f})"


def _p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_eq(p: float | None) -> str:
    text = _p(p)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def _usd(interval: Interval | None) -> str:
    if interval is None:
        return "n/a (none resolved)"
    return f"${interval.estimate:.4f} (${interval.low:.4f} to ${interval.high:.4f})"


# Columns of the full arms table, and the subset the headline table shows.
FULL_COLUMNS = (
    "success",
    "pass_k_with_2",
    "state_match",
    "policy_violation",
    "precision",
    "recall",
    "usd",
    "tokens",
    "latency",
)
HEADLINE_COLUMNS = ("success", "pass_k", "policy_violation", "usd")


def _arm_cells(arm: Arm, trials: Sequence[Sequence[EvalRecord]]) -> dict[str, str] | None:
    """Every table cell for one arm, or None while its trials are pending."""
    if len(trials) < arm.k:
        return None
    records = analysis.flatten(trials)
    pk = analysis.pass_hat_k(trials, arm.k)
    s = analysis.spend(records)
    pass_k = f"{_pct(pk.pass_hat_k)}, k={arm.k}"
    pass_2 = f", pass^2 {_pct(analysis.pass_hat_k(trials, 2).pass_hat_k)}" if arm.k > 2 else ""
    return {
        "success": _pct(analysis.pooled_rate(records, "success")),
        "pass_k": pass_k,
        "pass_k_with_2": pass_k + pass_2,
        "state_match": _pct(analysis.pooled_rate(records, "state_match")),
        "policy_violation": _pct(analysis.pooled_rate(records, "policy_violation")),
        "precision": _pct(analysis.escalation_precision(records)),
        "recall": _pct(analysis.escalation_recall(records)),
        "usd": _usd(s.per_resolved),
        "tokens": f"{s.tokens_per_ticket:,.0f}",
        "latency": f"{s.p50_s:.1f} / {s.p95_s:.1f}",
        "runs": str(len(records)),
    }


def _reference_cells(ref: analysis.ReferenceRow, n_tasks: int) -> dict[str, str]:
    return {
        "success": _pct(ref.success),
        "pass_k": "n/a (deterministic)",
        "pass_k_with_2": "n/a (deterministic)",
        "state_match": _pct(ref.state_match),
        "policy_violation": _pct(ref.policy_violation),
        "precision": _pct(ref.precision),
        "recall": _pct(ref.recall),
        "usd": "n/a (no model)",
        "tokens": "0",
        "latency": "n/a",
        "runs": str(n_tasks),
    }


def _row(label: str, cells: dict[str, str] | None, columns: Sequence[str]) -> str:
    if cells is None:
        return f"| {label} | " + " | ".join([PENDING] * len(columns)) + " | |"
    return f"| {label} | " + " | ".join(cells[c] for c in columns) + f" | {cells['runs']} |"


def _arm_rows(
    tasks: Sequence[Task], trials: dict[str, list[list[EvalRecord]]], columns: Sequence[str]
) -> list[str]:
    """One row per arm, then the two reference rows."""
    rows = [_row(ARMS[k].label, _arm_cells(ARMS[k], trials[k]), columns) for k in "ABC"]
    for ref in analysis.reference_rows(tasks):
        rows.append(_row(ref.name, _reference_cells(ref, len(tasks)), columns))
    return rows


def _comparisons(trials: dict[str, list[list[EvalRecord]]]) -> list[str]:
    out = [
        "| Comparison | Metric | Difference, pts (95% CI) | p (clustered) | McNemar p, first trial "
        "| MDE, pts | Paired runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for base_key, cand_key in (("A", "B"), ("C", "B"), ("A", "C")):
        base, cand = trials[base_key], trials[cand_key]
        ready = len(base) == ARMS[base_key].k and len(cand) == ARMS[cand_key].k
        label = f"{ARMS[cand_key].label.split(':')[0]} minus {ARMS[base_key].label.split(':')[0]}"
        for metric in ("success", "policy_violation"):
            if not ready:
                out.append(f"| {label} | {metric} | {PENDING} | {PENDING} | {PENDING} | | |")
                continue
            cmp = analysis.compare_arms(base, cand, metric)
            c = cmp.clustered.comparison
            mde = "n/a" if cmp.clustered.mde is None else f"{cmp.clustered.mde * 100:.1f}"
            out.append(
                f"| {label} | {metric} | {c.diff * 100:+.1f} ({c.low * 100:+.1f} to "
                f"{c.high * 100:+.1f}) | {_p(c.pvalue)} | {_p(cmp.mcnemar_trial0.pvalue)} | {mde} "
                f"| {c.n} |"
            )
    return out


def _verdict(trials: dict[str, list[list[EvalRecord]]]) -> str:
    a, b = trials["A"], trials["B"]
    if len(a) < ARMS["A"].k or len(b) < ARMS["B"].k:
        return f"Verdict: {PENDING}."
    viol = analysis.compare_arms(a, b, "policy_violation").clustered.comparison
    succ = analysis.compare_arms(a, b, "success").clustered.comparison
    fewer_violations = viol.pvalue is not None and viol.pvalue < 0.05 and viol.diff < 0
    better_resolution = succ.pvalue is not None and succ.pvalue < 0.05 and succ.diff > 0
    v = f"{viol.diff * 100:+.1f} pts, {_p_eq(viol.pvalue)}"
    s = f"{succ.diff * 100:+.1f} pts, {_p_eq(succ.pvalue)}"
    if fewer_violations and not better_resolution:
        verdict = f"supported. Violations fell ({v}) and success did not rise significantly ({s})"
    elif fewer_violations:
        verdict = f"half right. Violations fell ({v}) but success also rose ({s})"
    elif better_resolution:
        verdict = (
            f"not supported. Success rose ({s}) but violations did not fall significantly ({v})"
        )
    else:
        verdict = (
            f"not supported. Neither difference is significant, violations ({v}) and success ({s})"
        )
    return f"Verdict: {verdict}."


def _failures(trials: dict[str, list[list[EvalRecord]]]) -> list[str]:
    header = "| Category | " + " | ".join(ARMS[k].label for k in "ABC") + " |"
    out = [header, "|---|---|---|---|"]
    counts = {}
    for key in "ABC":
        if len(trials[key]) == ARMS[key].k:
            records = analysis.flatten(trials[key])
            counts[key] = (analysis.failure_counts(records), len(records))
    for category in FAILURE_CATEGORIES:
        cells = []
        for key in "ABC":
            if key not in counts:
                cells.append(PENDING)
            else:
                c, n = counts[key]
                cells.append(f"{c[category]} of {n}")
        out.append(f"| {category.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return out


def bank_agreement(tasks: Sequence[Task]) -> int:
    from support_triage_agents.bank import state_matches

    agree = 0
    for task in tasks:
        result = sandbox_apply(task, [dict(a) for a in task.gold_actions])
        agree += not result.refusals and state_matches(result.diff, task.gold_final_state)
    return agree


def retrieval_agreement() -> tuple[int, int, int]:
    """(exact top-8 order, same top-8 set, questions) against the RAG repo's frozen contexts."""
    from support_triage_agents.retrieval import build_retriever

    rows = [json.loads(line) for line in (EQUIVALENCE_FIXTURE / "questions.jsonl").open()]
    vectors = np.load(EQUIVALENCE_FIXTURE / "query_vectors.npy")
    by_text = {row["question"]: vectors[i] for i, row in enumerate(rows)}

    class _Fixture:
        def encode(self, query: str) -> np.ndarray:
            return by_text[query]

    retriever = build_retriever(_Fixture())
    exact = same = 0
    for row in rows:
        top8 = [h.chunk.chunk_id for h in retriever.search(row["question"], top_k=8)]
        exact += top8 == row["top8_chunks"]
        same += set(top8) == set(row["top8_chunks"])
    return exact, same, len(rows)


def _crosscheck_line(n_tasks: int) -> str:
    if not CROSSCHECK_PATH.exists():
        return f"- Second-model cross-check of the gold labels ({CROSSCHECK_MODEL}): {PENDING}."
    records = read_records(CROSSCHECK_PATH)
    scored = [r for r in records if "agrees" in r.scores]
    agree = [r.item_id for r in scored if r.scores["agrees"]]
    disagree = sorted(r.item_id for r in scored if not r.scores["agrees"])
    unparsed = len(records) - len(scored)
    line = (
        f"- Second-model cross-check of the gold labels ({CROSSCHECK_MODEL}): {len(agree)} of "
        f"{n_tasks} agree"
    )
    if disagree:
        line += f". Disagreements, listed and not adjudicated: {', '.join(disagree)}"
    if unparsed:
        line += f". Replies that did not parse: {unparsed}"
    return line + "."


def _load_trials(results_dir: Path) -> tuple[list[Task], dict[str, list[list[EvalRecord]]]]:
    tasks = load_tasks()
    trials = {key: analysis.load_trials(arm, len(tasks), results_dir) for key, arm in ARMS.items()}
    return tasks, trials


def results_section(results_dir: Path = RESULTS_DIR) -> str:
    """The visible results: the headline table and the pre-registered verdict."""
    tasks, trials = _load_trials(results_dir)
    lines = []
    if not any(trials.values()):
        lines += [
            '> Rows marked "pending live run" need Azure model calls, which have not been made '
            "yet. Every number shown was produced offline by `make demo`, with no keys and no "
            "model calls.",
            "",
        ]
    lines += [
        f"Every arm uses {AGENT_MODEL} with the same tools, docs and step budget, on "
        f"{len(tasks)} tasks. Intervals are 95% and clustered by task.",
        "",
        "| Arm | Success | pass^k | Policy violations | $ per resolved ticket | Ticket runs |",
        "|---|---|---|---|---|---|",
    ]
    lines += _arm_rows(tasks, trials, HEADLINE_COLUMNS)
    lines += [
        "",
        "**Pre-registered hypothesis: the graph wins on policy violations, not on raw "
        "resolution.** " + _verdict(trials),
    ]
    return "\n".join(lines)


def results_detail_section(results_dir: Path = RESULTS_DIR) -> str:
    """Every metric, the paired comparisons, the failure categories and the offline checks."""
    tasks, trials = _load_trials(results_dir)
    exact, same, n_q = retrieval_agreement()
    do_nothing_right = sum(not t.gold_actions for t in tasks)
    lines = [
        f"**Arms on the {len(tasks)} tasks.** {AGENT_MODEL}, reasoning effort none, the same "
        f"tools, help-center snapshot and step budget ({STEP_BUDGET} model calls) in every arm. "
        "Rates pool every trial and carry a 95% clustered Wilson interval with tasks as "
        "clusters. pass^k is the chance all k trials of a task succeed, averaged over tasks, "
        "with a bootstrap interval over tasks. Model seconds are the sum of a ticket's "
        "model-call latencies.",
        "",
        "| Arm | Success (pass^1) | pass^k | State match | Policy violations | Escalation "
        "precision | Escalation recall | $ per resolved ticket | Tokens per ticket | p50 / p95 "
        "model s | Ticket runs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += _arm_rows(tasks, trials, FULL_COLUMNS)
    lines += [
        "",
        "Success needs the gold end state, no policy violation, no plan rejected at review and a "
        "finished ticket. State match alone counts a ticket where a bank rule or the reviewer "
        "stopped a wrong action and the state stayed right. The two reference rows are policies "
        "computed by code through the same scorer: doing nothing is right on the "
        f"{do_nothing_right} tasks whose gold resolution is to explain or decline, and replaying "
        "the gold actions is the ceiling.",
        "",
        "**Paired comparisons**, trial t of one arm against trial t of the other on the same "
        "task, clustered by task (harness clustered paired t-test, with the minimum detectable "
        "effect at 80% power). The McNemar column uses the first trial only, one independent "
        "pair per task. C minus A is not pre-registered and pairs on C's two trials.",
        "",
    ]
    lines += _comparisons(trials)
    lines += [
        "",
        "**Failure categories**, derived by code from the diff between the actions an arm tried "
        "and the gold actions. A ticket run can fall in more than one.",
        "",
    ]
    lines += _failures(trials)
    lines += [
        "",
        "**Checks that need no model.**",
        "",
        f"- The SQLite bank reproduces the dataset's gold final state from the gold actions for "
        f"{bank_agreement(tasks)} of {len(tasks)} tasks. The dataset computed those states with "
        "its own reference model, so two implementations agree.",
        f"- The Researcher's hybrid search matches the RAG repo's frozen top 8 chunks in the same "
        f"order for {exact} of {n_q} questions, and as a set for {same} of {n_q}.",
        _crosscheck_line(len(tasks)),
    ]
    return "\n".join(lines)


def cost_section(results_dir: Path = RESULTS_DIR, cap_usd: float | None = None) -> str:
    stages = load_estimate(results_dir / "estimate.json")
    if stages is None:
        return "Run `make estimate` to price the live run."
    lines = [
        "Estimated before any live call by `make estimate`, which runs every arm over all tasks "
        "with a scripted client that builds the real prompts and takes the shortest correct "
        "path. Expected cost counts input as bytes / 4 tokens and a typical output length per "
        "agent. Real agents take more steps, so the heavier-path column doubles it. The worst "
        "case is what `DollarCap` reserves per call (one token per input byte plus "
        "`max_output_tokens`) on the scripted path. The prompt cache is ignored, which errs high.",
        "",
        "| Stage | Model | Calls | Input tokens | Output tokens | Expected $ | Heavier path $ "
        "| DollarCap worst case $ | Minutes at default quota |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stages:
        lines.append(
            f"| {s.stage} | {s.model} | {s.calls:,} | {s.input_tokens:,} | {s.output_tokens:,} | "
            f"{s.expected_usd:.2f} | {s.heavier_usd:.2f} | {s.worst_usd:.2f} | "
            f"{s.minutes_at_quota:.0f} |"
        )
    total = [sum(getattr(s, f) for s in stages) for f in ("calls", "input_tokens", "output_tokens")]
    money = [
        sum(getattr(s, f) for s in stages) for f in ("expected_usd", "heavier_usd", "worst_usd")
    ]
    minutes = sum(s.minutes_at_quota for s in stages)
    lines.append(
        f"| Total | | {total[0]:,} | {total[1]:,} | {total[2]:,} | {money[0]:.2f} | "
        f"{money[1]:.2f} | {money[2]:.2f} | {minutes:.0f} |"
    )
    lines += [
        "",
        f"At the day-1 capacities ({DEPLOYMENT_TPM[AGENT_MODEL] // 1000}K tokens per minute on "
        f"{AGENT_MODEL} and {CROSSCHECK_MODEL}) the run "
        f"needs about {minutes / 60:.1f} hours of wall time. Raise the capacities and set "
        "`TRIAGE_TPM` to match to go faster.",
    ]
    if cap_usd is not None:
        lines += [
            "",
            f"`make eval-live` runs everything in one process under one hard cap of "
            f"${cap_usd:.2f} (`make eval-live CAP=...` to change it), about three times the "
            "heavier-path estimate. A refused call stops the run, and cached calls cost nothing "
            "when it is started again.",
        ]
    lines += ["", f"Actual spend: {_actual_spend(results_dir)}"]
    return "\n".join(lines)


def _actual_spend(results_dir: Path) -> str:
    path = results_dir / "live_runs.jsonl"
    if not path.exists():
        return f"{PENDING}."
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    spent = sum(float(r.get("spent_usd", 0.0)) for r in runs)
    calls = sum(int(r.get("live_calls", 0)) for r in runs)
    runs_word = "run" if len(runs) == 1 else "runs"
    return f"${spent:.2f} over {calls:,} live calls in {len(runs)} {runs_word}, from `DollarCap`."


def render(
    readme: Path = README, results_dir: Path = RESULTS_DIR, cap_usd: float | None = None
) -> bool:
    sections = {
        "results": results_section(results_dir),
        "results-detail": results_detail_section(results_dir),
        "cost": cost_section(results_dir, cap_usd),
    }
    changed = [write_section(readme, name, body) for name, body in sections.items()]
    return any(changed)
