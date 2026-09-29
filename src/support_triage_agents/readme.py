"""Render the README's results and cost sections from committed files. No model calls.

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
from support_triage_agents.crosscheck import CROSSCHECK_PATH
from support_triage_agents.data import Task, load_tasks
from support_triage_agents.estimate import load_estimate
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


def _arm_row(arm: Arm, trials: Sequence[Sequence[EvalRecord]]) -> str:
    if len(trials) < arm.k:
        return f"| {arm.label} | " + " | ".join([PENDING] * 9) + " | |"
    records = analysis.flatten(trials)
    pk = analysis.pass_hat_k(trials, arm.k)
    s = analysis.spend(records)
    cells = [
        _pct(analysis.pooled_rate(records, "success")),
        f"{_pct(pk.pass_hat_k)}, k={arm.k}"
        + (f", pass^2 {_pct(analysis.pass_hat_k(trials, 2).pass_hat_k)}" if arm.k > 2 else ""),
        _pct(analysis.pooled_rate(records, "state_match")),
        _pct(analysis.pooled_rate(records, "policy_violation")),
        _pct(analysis.escalation_precision(records)),
        _pct(analysis.escalation_recall(records)),
        _usd(s.per_resolved),
        f"{s.tokens_per_ticket:,.0f}",
        f"{s.p50_s:.1f} / {s.p95_s:.1f}",
    ]
    return f"| {arm.label} | " + " | ".join(cells) + f" | {len(records)} |"


def _reference_table(tasks: Sequence[Task]) -> list[str]:
    rows = []
    for ref in analysis.reference_rows(tasks):
        cells = [
            _pct(ref.success),
            "n/a (deterministic)",
            _pct(ref.state_match),
            _pct(ref.policy_violation),
            _pct(ref.precision),
            _pct(ref.recall),
            "n/a (no model)",
            "0",
            "n/a",
        ]
        rows.append(f"| {ref.name} | " + " | ".join(cells) + f" | {len(tasks)} |")
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
    if fewer_violations and not better_resolution:
        verdict = "supported"
    elif fewer_violations:
        verdict = (
            "half right. The graph did cut policy violations, but it also resolved more "
            'tickets, so the "not on raw resolution" half was wrong'
        )
    elif better_resolution:
        verdict = (
            "not supported. The graph resolved more tickets but did not cut policy "
            "violations significantly"
        )
    else:
        verdict = "not supported. Neither difference is significant"
    return (
        f"Verdict: {verdict}. "
        f"Policy violations, B minus A: {viol.diff * 100:+.1f} pts ({_p_eq(viol.pvalue)}). "
        f"Success, B minus A: {succ.diff * 100:+.1f} pts ({_p_eq(succ.pvalue)})."
    )


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
        return f"- Second-model cross-check of the gold labels (gpt-5-mini): {PENDING}."
    records = read_records(CROSSCHECK_PATH)
    scored = [r for r in records if "agrees" in r.scores]
    agree = [r.item_id for r in scored if r.scores["agrees"]]
    disagree = sorted(r.item_id for r in scored if not r.scores["agrees"])
    unparsed = len(records) - len(scored)
    line = (
        f"- Second-model cross-check of the gold labels (gpt-5-mini): {len(agree)} of "
        f"{n_tasks} agree"
    )
    if disagree:
        line += f". Disagreements, listed and not adjudicated: {', '.join(disagree)}"
    if unparsed:
        line += f". Replies that did not parse: {unparsed}"
    return line + "."


def results_section(results_dir: Path = RESULTS_DIR) -> str:
    tasks = load_tasks()
    trials = {key: analysis.load_trials(arm, len(tasks), results_dir) for key, arm in ARMS.items()}
    any_live = any(trials.values())
    exact, same, n_q = retrieval_agreement()
    lines = []
    if not any_live:
        lines += [
            '> Rows marked "pending live run" need Azure model calls, which have not been made '
            "yet. Every number shown was produced offline by `make demo`, with no keys and no "
            "model calls.",
            "",
        ]
    lines += [
        f"**Arms on the {len(tasks)} tasks.** gpt-6-luna, reasoning effort none, the same tools, "
        "help-center snapshot and step budget (16 model calls) in every arm. Rates pool every "
        "trial and carry a 95% clustered Wilson interval with tasks as clusters. pass^k is the "
        "chance all k trials of a task succeed, averaged over tasks, with a bootstrap interval "
        "over tasks. Model seconds are the sum of a ticket's model-call latencies.",
        "",
        "| Arm | Success (pass^1) | pass^k | State match | Policy violations | Escalation "
        "precision | Escalation recall | $ per resolved ticket | Tokens per ticket | p50 / p95 "
        "model s | Ticket runs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_arm_row(ARMS[k], trials[k]) for k in "ABC"]
    lines += _reference_table(tasks)
    lines += [
        "",
        "Success needs the gold end state, no policy violation, no plan rejected at review and a "
        "finished ticket. State match alone counts a ticket where a bank rule or the reviewer "
        "stopped a wrong action and the state stayed right. The two reference rows are policies "
        "computed by code through the same scorer: doing nothing is right on the 13 tasks whose "
        "gold resolution is to explain or decline, and replaying the gold actions is the ceiling.",
        "",
        "**Pre-registered hypothesis: the graph wins on policy violations, not on raw "
        "resolution.** " + _verdict(trials),
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
        f"At the day-1 capacities (20K tokens per minute on gpt-6-luna and gpt-5-mini) the run "
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
    changed_results = write_section(readme, "results", results_section(results_dir))
    changed_cost = write_section(readme, "cost", cost_section(results_dir, cap_usd))
    return changed_results or changed_cost
