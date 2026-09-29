"""One way for every agent to call a model: JSON out, validated, counted, within a step budget.

The harness client returns text, so each agent is asked for a single JSON
object and the reply is parsed into a Pydantic model. A reply that doesn't
parse gets one repair turn (the parse error is shown to the model). A second
failure raises `AgentOutputError`, which ends the ticket as an agent failure.

Every request carries the trial index, so k trials of the same ticket get k
independent replies and a replay returns the same k.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from llm_eval_harness import ModelClient, ModelRequest, ModelResponse
from llm_eval_harness.client import DEFAULT_PRICES, cost_usd
from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T")

# Output caps per role. gpt-6-luna runs with reasoning effort none, so these are all text.
MAX_OUTPUT_TOKENS = {
    "intake": 400,
    "researcher": 200,
    "resolver": 900,
    "compliance": 500,
    "single_agent": 900,
    "crosscheck": 2000,
}
# Model calls allowed per ticket, the same for every arm.
STEP_BUDGET = 16


class AgentOutputError(ValueError):
    """A model reply did not parse into the expected shape, even after one repair turn."""


class StepBudgetExhausted(RuntimeError):
    """The ticket used its whole step budget."""


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cached_tokens_in: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    replayed_calls: int = 0

    def add(self, response: ModelResponse, cost: float) -> None:
        self.calls += 1
        self.tokens_in += response.input_tokens
        self.tokens_out += response.output_tokens
        self.reasoning_tokens += response.reasoning_tokens
        self.cached_tokens_in += response.cached_input_tokens
        self.cost_usd += cost
        self.latency_ms += response.latency_ms
        self.replayed_calls += int(response.from_cache)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def merge_usage(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """LangGraph reducer: usage from each node adds up across the ticket."""
    out = dict(left or {})
    for key, value in (right or {}).items():
        out[key] = out.get(key, 0) + value
    return out


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """The first JSON object in a reply, tolerating code fences and text around it."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return obj


@dataclass
class AgentLLM:
    """Model access for one ticket. Counts calls against the step budget and adds up usage."""

    client: ModelClient
    model: str
    trial: int
    calls_used: int = 0
    budget: int = STEP_BUDGET
    reasoning_effort: str | None = "none"

    def complete(
        self, role: str, instructions: str, messages: list[dict[str, str]], usage: Usage
    ) -> str:
        if self.calls_used >= self.budget:
            raise StepBudgetExhausted(f"step budget of {self.budget} model calls used up")
        self.calls_used += 1
        request = ModelRequest(
            model=self.model,
            input=messages,
            instructions=instructions,
            max_output_tokens=MAX_OUTPUT_TOKENS[role],
            reasoning_effort=self.reasoning_effort,
            trial=self.trial,
        )
        response = self.client.complete(request)
        price = DEFAULT_PRICES.get(self.model)
        usage.add(response, cost_usd(price, response) if price else 0.0)
        return response.text

    def json_call(
        self,
        role: str,
        instructions: str,
        messages: list[dict[str, str]],
        schema: type[T] | TypeAdapter[T],
        usage: Usage,
    ) -> T:
        adapter = schema if isinstance(schema, TypeAdapter) else TypeAdapter(schema)
        text = self.complete(role, instructions, messages, usage)
        try:
            return _parse(adapter, text)
        except (ValueError, ValidationError) as first:
            repair = [
                *messages,
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "Your reply was not valid. Error: "
                    + _short_error(first)
                    + "\nReply again with one JSON object only, in the required shape.",
                },
            ]
            text = self.complete(role, instructions, repair, usage)
            try:
                return _parse(adapter, text)
            except (ValueError, ValidationError) as second:
                raise AgentOutputError(f"{role}: {_short_error(second)}") from second


def _parse(adapter: TypeAdapter[T], text: str) -> T:
    return adapter.validate_python(extract_json(text))


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4]]
        return "; ".join(parts)
    return str(exc)[:300]


def dump(obj: BaseModel | dict[str, Any] | list[Any]) -> str:
    """Compact, deterministic JSON for prompts, so replays send byte-identical requests."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
