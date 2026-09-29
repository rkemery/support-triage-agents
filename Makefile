.PHONY: install lint format test test-download demo data estimate resume-demo eval-live eval-replay smoke-live

# Hard dollar cap for one live run of every arm plus the gold cross-check. DollarCap refuses any
# call that could take spend past it. About three times the heavier-path estimate (see README).
CAP ?= 3.00

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

# Downloads bge-small (33M) once and checks the query encoder against the committed fixture.
test-download:
	uv sync --extra embed
	uv run pytest -q -m download

# Offline, no keys: checks data hashes and rewrites the README results and cost sections
# from committed files. Arm rows read "pending live run" until a live run exists.
demo:
	uv run triage demo --cap $(CAP)

# Verify the vendored data against data/*/MANIFEST.json.
data:
	uv run python scripts/sync_data.py

# Price the live run from the real prompts (scripted client, no model calls).
estimate:
	uv run triage estimate

# Pause at human approval, kill the process mid-execution, resume, check no write ran twice.
resume-demo:
	uv run triage resume-demo

# ---- model calls (need AZURE_OPENAI_BASE_URL plus AZURE_OPENAI_API_KEY or Entra ID)

# Two tickets, one trial each, every arm: a cheap check that the live path works.
smoke-live:
	uv sync --extra embed
	uv run triage run --arm all --live --cap 0.25 --trials 1 --tasks task-001,task-009 \
		--results-dir results/smoke --verbose

# The full live run: arms A (k=4), B (k=4), C (k=2) and the gold cross-check, one process, one cap.
eval-live:
	uv sync --extra embed
	uv run triage run --arm all --live --cap $(CAP) --verbose
	uv run triage demo --cap $(CAP)

# Replay the committed cache. Sends nothing, fails on any cache miss.
eval-replay:
	uv run triage run --arm all --replay
	uv run triage demo --cap $(CAP)
