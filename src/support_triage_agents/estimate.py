"""Estimate the live run's calls, tokens, dollars and wall time before making any call.

Runs every arm once over all tasks with the scripted client, which builds the
real prompts (the same instructions, tool results and excerpts a live run
sends) and walks the shortest correct path. Then:

- Expected cost: input tokens as bytes / 4, output tokens at a typical
  length per role (below), times k trials. Real agents take more steps than
  the scripted path, so the "heavier path" column assumes 2x the calls.
- Worst case: what `DollarCap` reserves per call (one token per input byte
  plus max_output_tokens), times k, on the scripted path.
- Minutes at quota: the least wall time the rate limiter allows at the
  deployment's default tokens per minute, using 80% of it.

The prompt cache is ignored (all input at the full rate), which errs high:
every agent call repeats its long instructions, which Azure may serve from
the prompt cache at a tenth of the price.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_eval_harness import FakeClient
from llm_eval_harness.client import DEFAULT_PRICES, input_token_bound, max_cost_usd

from support_triage_agents.clients import (
    AGENT_MODEL,
    CROSSCHECK_MODEL,
    TPM_HEADROOM,
    deployment_tpm,
    estimated_tokens,
)
from support_triage_agents.crosscheck import run_crosscheck
from support_triage_agents.data import load_tasks
from support_triage_agents.retrieval import cached_retriever
from support_triage_agents.runner import ARMS, RESULTS_DIR, run_arm
from support_triage_agents.scripted import role_of, scripted_client

# Typical output tokens per call, by role, for the expected-cost column.
TYPICAL_OUTPUT_TOKENS = {
    "intake": 90,
    "researcher": 40,
    "resolver": 160,
    "compliance": 80,
    "single_agent": 70,
    "crosscheck": 900,  # gpt-5-mini at effort low spends most of this reasoning
}
HEAVIER_PATH_FACTOR = 2.0


@dataclass(frozen=True)
class StageEstimate:
    stage: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    expected_usd: float
    heavier_usd: float
    worst_usd: float
    minutes_at_quota: float


def _stage(name: str, model: str, client: FakeClient, repeats: int) -> StageEstimate:
    price = DEFAULT_PRICES[model]
    tpm = deployment_tpm()[model] * TPM_HEADROOM
    in_tok = sum(input_token_bound(r) for r in client.calls) / 4
    out_tok = sum(TYPICAL_OUTPUT_TOKENS[role_of(r.instructions or "")] for r in client.calls)
    expected = (in_tok * price.input_per_m + out_tok * price.output_per_m) / 1e6
    worst = sum(max_cost_usd(price, r) for r in client.calls)
    minutes = sum(estimated_tokens(r) for r in client.calls) / tpm
    return StageEstimate(
        stage=name,
        model=model,
        calls=len(client.calls) * repeats,
        input_tokens=round(in_tok * repeats),
        output_tokens=round(out_tok * repeats),
        expected_usd=round(expected * repeats, 4),
        heavier_usd=round(expected * repeats * HEAVIER_PATH_FACTOR, 4),
        worst_usd=round(worst * repeats, 4),
        minutes_at_quota=round(minutes * repeats, 1),
    )


def estimate(out_path: Path = RESULTS_DIR / "estimate.json") -> list[StageEstimate]:
    tasks = load_tasks()
    retriever = cached_retriever(embed_new=False)
    stages = []
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ARMS.values():
            client = scripted_client()
            run_arm(
                arm,
                tasks,
                client,
                AGENT_MODEL,
                retriever,
                trials=1,
                results_dir=Path(tmp) / "results",
                workdir=Path(tmp) / "work",
            )
            stages.append(_stage(f"{arm.label} (k={arm.k})", AGENT_MODEL, client, arm.k))
        client = scripted_client()
        run_crosscheck(tasks, client, retriever, out_path=Path(tmp) / "crosscheck.jsonl")
        stages.append(_stage("Gold cross-check (second model)", CROSSCHECK_MODEL, client, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(s) for s in stages], indent=2) + "\n", encoding="utf-8")
    return stages


def load_estimate(path: Path = RESULTS_DIR / "estimate.json") -> list[StageEstimate] | None:
    if not path.exists():
        return None
    return [StageEstimate(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
