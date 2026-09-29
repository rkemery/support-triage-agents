# Tallowbrook data (vendored)

A pinned copy of the three files this repo needs from the synthetic Tallowbrook dataset (source repo `neobank-support-data`, commit in `MANIFEST.json`). The dataset's canonical home will be a Hugging Face dataset.

- `tasks.jsonl`: 50 support tickets with gold actions, a gold final state, a `should_escalate` label, a `needs_clarification` flag and hidden facts for the customer simulator.
- `bank_seed.json`: the fake bank as of 2026-09-15 (30 customers, 33 accounts, 33 cards, 196 transactions, 2 disputes).
- `policies.yaml`: the facts file the bank's rules read (fees per plan, dispute windows, refund rules).

Everything is synthetic and was written by an AI. Card numbers are fake Luhn-valid test numbers. License: CC-BY-4.0.

`uv run python scripts/sync_data.py` checks every file against its sha256.
