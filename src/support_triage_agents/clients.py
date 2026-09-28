"""The client stack for live and replay runs, built from harness pieces.

Live, outermost first:

    CachedClient(RetryingClient(DollarCap(RateLimitedClient(FoundryClient())), retryable), cache/)

- `CachedClient` answers repeats from disk for free. The cache is committed
  after a live run, so a replay (and CI) reproduces it with no keys.
- `RetryingClient` retries rate limits, timeouts and 5xx with jittered backoff.
- `DollarCap` refuses any attempt whose worst case could take spend past the
  cap, and charges a failed attempt its worst case.
- `RateLimitedClient` (this repo) waits so estimated tokens per minute stay
  under each deployment's quota, since Azure counts max_output_tokens against
  TPM on arrival. Staying under quota avoids 429s, which DollarCap would
  charge at worst case.

Replay: `CachedClient(None, cache/, replay_only=True)`. A miss raises
`CacheMiss` and nothing reaches the network.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_eval_harness import (
    CachedClient,
    DollarCap,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RetryingClient,
)
from llm_eval_harness.client import input_token_bound

# Tokens per minute per deployment (Global Standard capacity set on day 1).
DEPLOYMENT_TPM: dict[str, int] = {
    "gpt-6-luna": 20_000,
    "gpt-6-sol": 10_000,
    "gpt-5-mini": 20_000,
    "Llama-3.3-70B-Instruct": 10_000,
}
TPM_HEADROOM = 0.8
# English runs near 4 bytes per token. The limiter assumes 3.5, which only makes it wait longer.
BYTES_PER_TOKEN = 3.5
AGENT_MODEL = "gpt-6-luna"
CROSSCHECK_MODEL = "gpt-5-mini"


def deployment_tpm(env: Mapping[str, str] | None = None) -> dict[str, int]:
    """DEPLOYMENT_TPM with overrides from TRIAGE_TPM, e.g. "gpt-6-luna=100000,gpt-5-mini=50000"."""
    env = os.environ if env is None else env
    out = dict(DEPLOYMENT_TPM)
    for part in filter(None, (p.strip() for p in env.get("TRIAGE_TPM", "").split(","))):
        name, _, value = part.partition("=")
        if not value.strip().isdigit():
            raise ValueError(
                f"TRIAGE_TPM entry {part!r} must look like deployment=tokens_per_minute"
            )
        out[name.strip()] = int(value)
    return out


def estimated_tokens(request: ModelRequest) -> int:
    """Rough count Azure charges against TPM on arrival: prompt estimate plus max_output_tokens."""
    return int(input_token_bound(request) / BYTES_PER_TOKEN) + (request.max_output_tokens or 0)


class RateLimitedClient:
    """Sliding one-minute window of estimated tokens per model. Waits instead of drawing 429s."""

    def __init__(
        self,
        inner: ModelClient,
        tpm: Mapping[str, int] | None = None,
        *,
        headroom: float = TPM_HEADROOM,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < headroom <= 1:
            raise ValueError(f"headroom must be in (0, 1], got {headroom}")
        self._inner = inner
        self._budget = {m: int(v * headroom) for m, v in (tpm or deployment_tpm()).items()}
        self._clock = clock
        self._sleep = sleep
        self._sent: dict[str, deque[tuple[float, int]]] = {}
        self.waited_s = 0.0

    def complete(self, request: ModelRequest) -> ModelResponse:
        budget = self._budget.get(request.model)
        if budget is not None:
            # A request bigger than the whole budget goes alone, once the window is empty.
            self._wait_for(request.model, min(estimated_tokens(request), budget), budget)
        return self._inner.complete(request)

    def _wait_for(self, model: str, tokens: int, budget: int) -> None:
        window = self._sent.setdefault(model, deque())
        while True:
            now = self._clock()
            while window and now - window[0][0] >= 60.0:
                window.popleft()
            if sum(t for _, t in window) + tokens <= budget:
                window.append((now, tokens))
                return
            delay = max(0.05, 60.0 - (now - window[0][0]))
            self.waited_s += delay
            self._sleep(delay)


@dataclass
class ClientStack:
    client: ModelClient
    cache: CachedClient
    cap: DollarCap | None
    limiter: RateLimitedClient | None
    mode: str

    def summary(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "mode": self.mode,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }
        if self.cap is not None:
            out.update(
                spent_usd=round(self.cap.spent_usd, 6),
                cap_usd=self.cap.cap_usd,
                live_calls=self.cap.calls,
                failed_calls=self.cap.failed_calls,
            )
        if self.limiter is not None:
            out["rate_limit_wait_s"] = round(self.limiter.waited_s, 1)
        return out


def live_stack(cache_dir: Path, cap_usd: float) -> ClientStack:
    """Needs AZURE_OPENAI_BASE_URL plus AZURE_OPENAI_API_KEY or Entra ID."""
    from llm_eval_harness.azure import FoundryClient, retryable_errors

    limiter = RateLimitedClient(FoundryClient())
    cap = DollarCap(limiter, cap_usd=cap_usd)
    cache = CachedClient(RetryingClient(cap, retryable_errors()), cache_dir)
    return ClientStack(cache, cache, cap, limiter, "live")


def replay_stack(cache_dir: Path) -> ClientStack:
    cache = CachedClient(None, cache_dir, replay_only=True)
    return ClientStack(cache, cache, None, None, "replay")


def item_level_errors() -> tuple[type[BaseException], ...]:
    """Provider errors that belong to one request, not to the run.

    A 400 (an invalid request, or input the content filter refused) is
    recorded on that ticket and the run goes on. Budget, cache-miss, auth and
    network errors are not in this tuple, so they stop the run.
    """
    try:
        import openai
    except ImportError:
        return ()
    return (openai.BadRequestError,)
