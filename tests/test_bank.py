from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from support_triage_agents.bank import (
    ActionRefused,
    Bank,
    action_key,
    diff_states,
    new_bank_for_task,
    state_matches,
)
from support_triage_agents.data import load_facts, load_seed, load_tasks

TASKS = load_tasks()
FACTS = load_facts()


def _bank(tmp_path: Path, customer_id: str, ticket_id: str = "t") -> Bank:
    return new_bank_for_task(tmp_path / "bank.db", load_seed(), FACTS, customer_id, ticket_id)


def test_seed_round_trips_through_sqlite(tmp_path):
    seed = load_seed()
    with _bank(tmp_path, "cus_001") as bank:
        state = bank.export_state()
    for table in ("customers", "accounts", "cards", "transactions", "disputes", "escalations"):
        expected = [{k: v for k, v in row.items() if v is not None} for row in seed[table]]
        assert state[table] == expected, table


@pytest.mark.parametrize("task", TASKS, ids=[t.task_id for t in TASKS])
def test_gold_actions_reproduce_gold_final_state(tmp_path, task):
    """The dataset computed each gold state with its own model. This SQLite bank must agree."""
    with _bank(tmp_path, task.customer_id, task.task_id) as bank:
        before = bank.export_state()
        for action in task.gold_actions:
            bank.apply(action)
        diff = diff_states(before, bank.export_state())
    assert state_matches(diff, task.gold_final_state)


def test_doing_nothing_matches_only_empty_gold(tmp_path):
    empty = {"updated": [], "inserted": []}
    matched = [t.task_id for t in TASKS if state_matches(empty, t.gold_final_state)]
    assert len(matched) == sum(1 for t in TASKS if not t.gold_actions) == 13


def test_state_match_ignores_order_and_minted_ids(tmp_path):
    task = next(t for t in TASKS if t.task_id == "task-033")  # two freezes, then escalate
    with _bank(tmp_path, task.customer_id, task.task_id) as bank:
        before = bank.export_state()
        for action in reversed(task.gold_actions):
            bank.apply(action)
        diff = diff_states(before, bank.export_state())
    assert state_matches(diff, task.gold_final_state)


def test_state_match_rejects_an_extra_insert():
    task = next(t for t in TASKS if t.task_id == "task-007")
    gold = task.gold_final_state
    doubled = {"updated": gold["updated"], "inserted": gold["inserted"] * 2}
    assert not state_matches(doubled, gold)


# Wrong moves from the dataset's own tests: its reference model rejects each sequence.
WRONG_MOVES = [
    (
        "cus_009",
        [
            ("freeze_card", {"card_id": "card_009p"}),
            ("open_dispute", {"txn_id": "txn_009_volta", "reason": "unauthorized"}),
        ],
    ),
    (
        "cus_012",
        [("open_dispute", {"txn_id": "txn_012_orbital", "reason": "merchant_not_as_described"})],
    ),
    (
        "cus_013",
        [
            ("freeze_card", {"card_id": "card_013p"}),
            ("open_dispute", {"txn_id": "txn_013_aurora", "reason": "unauthorized"}),
            ("open_dispute", {"txn_id": "txn_013_skyline", "reason": "unauthorized"}),
        ],
    ),
    (
        "cus_014",
        [("open_dispute", {"txn_id": "txn_014_ember", "reason": "merchant_not_as_described"})],
    ),
    (
        "cus_015",
        [("open_dispute", {"txn_id": "txn_015_bright", "reason": "merchant_not_received"})],
    ),
    ("cus_015", [("close_account", {})]),
    ("cus_020", [("refund_fee", {"txn_id": "txn_020_fxfee", "basis": "goodwill"})]),
    ("cus_023", [("refund_fee", {"txn_id": "txn_023_wirefee", "basis": "goodwill"})]),
    ("cus_024", [("close_account", {})]),
    ("cus_007", [("unfreeze_card", {"card_id": "card_007p"})]),
    ("cus_029", [("freeze_card", {"card_id": "card_029p"})]),
    ("cus_008", [("open_dispute", {"txn_id": "txn_008_vntx", "reason": "unauthorized"})]),
    ("cus_019", [("refund_fee", {"txn_id": "txn_019_atmfee", "basis": "error"})]),
    ("cus_019", [("open_dispute", {"txn_id": "txn_019_cof2", "reason": "duplicate"})]),
    ("cus_018", [("open_dispute", {"txn_id": "txn_018_sep", "reason": "cancelled_recurring"})]),
    ("cus_022", [("change_plan", {"new_plan": "plus", "effective": "immediate"})]),
    ("cus_001", [("freeze_card", {"card_id": "card_002p"})]),
]


@pytest.mark.parametrize(("customer", "moves"), WRONG_MOVES)
def test_bank_refuses_the_reference_models_wrong_moves(tmp_path, customer, moves):
    with _bank(tmp_path, customer) as bank:
        actions = [{"action": name, "args": args} for name, args in moves]
        for action in actions[:-1]:
            bank.apply(action)
        with pytest.raises(ActionRefused) as info:
            bank.apply(actions[-1])
        assert info.value.kind == "policy"
        assert bank.action_log()[-1]["outcome"] == "refused"


def test_refused_write_changes_nothing(tmp_path):
    with _bank(tmp_path, "cus_012") as bank:
        before = bank.export_state()
        with pytest.raises(ActionRefused):
            bank.apply(
                {
                    "action": "open_dispute",
                    "args": {"txn_id": "txn_012_orbital", "reason": "merchant_not_as_described"},
                }
            )
        assert bank.export_state() == before


def test_same_action_twice_for_a_ticket_applies_once(tmp_path):
    action = {"action": "freeze_card", "args": {"card_id": "card_004p"}}
    with _bank(tmp_path, "cus_004", "task-004") as bank:
        first = bank.apply(action)
        second = bank.apply(action)
        assert (first.outcome, second.outcome) == ("applied", "duplicate")
        assert second.detail == first.detail
        assert [e["outcome"] for e in bank.action_log()] == ["applied", "duplicate"]
        assert bank.applied_actions() == [action]


def test_idempotency_survives_reopening_the_database(tmp_path):
    task = next(t for t in TASKS if t.task_id == "task-001")
    path = tmp_path / "bank.db"
    Bank.create(path, load_seed())
    with Bank(path, FACTS, task.customer_id, task.task_id) as bank:
        before = bank.export_state()
        bank.apply(task.gold_actions[0])
    with Bank(path, FACTS, task.customer_id, task.task_id) as bank:  # a new process, say
        for action in task.gold_actions:
            bank.apply(action)
        diff = diff_states(before, bank.export_state())
        assert [e["outcome"] for e in bank.action_log()] == ["applied", "duplicate", "applied"]
    assert state_matches(diff, task.gold_final_state)


def test_escalation_summary_is_not_part_of_the_key():
    a = {"action": "escalate_to_human", "args": {"queue": "fraud", "summary": "one wording"}}
    b = {"action": "escalate_to_human", "args": {"queue": "fraud", "summary": "another"}}
    assert action_key(a) == action_key(b)


def test_another_customers_card_is_a_policy_refusal(tmp_path):
    with _bank(tmp_path, "cus_001") as bank, pytest.raises(ActionRefused) as info:
        bank.apply({"action": "freeze_card", "args": {"card_id": "card_002p"}})
    assert (info.value.kind, info.value.code) == ("policy", "not_owner")


@pytest.mark.parametrize(
    "action",
    [
        {"action": "freeze_card", "args": {"card_id": "card_404"}},
        {"action": "freeze_card", "args": {}},
        {"action": "freeze_card", "args": {"card_id": "card_001p", "extra": 1}},
        {"action": "wire_money", "args": {}},
        {"action": "escalate_to_human", "args": {"queue": "nowhere"}},
    ],
)
def test_malformed_actions_are_invalid_not_policy(tmp_path, action):
    with _bank(tmp_path, "cus_001") as bank, pytest.raises(ActionRefused) as info:
        bank.apply(action)
    assert info.value.kind == "invalid"


def test_dispute_limit_counts_a_refused_attempt_like_the_reference_model(tmp_path):
    # cus_013: $4,200 then $2,600. The first fits the $5,000 case limit, the second does not.
    with _bank(tmp_path, "cus_013") as bank:
        bank.apply({"action": "freeze_card", "args": {"card_id": "card_013p"}})
        bank.apply(
            {
                "action": "open_dispute",
                "args": {"txn_id": "txn_013_aurora", "reason": "unauthorized"},
            }
        )
        with pytest.raises(ActionRefused) as info:
            bank.apply(
                {
                    "action": "open_dispute",
                    "args": {"txn_id": "txn_013_skyline", "reason": "unauthorized"},
                }
            )
    assert info.value.code == "dispute_limit"


def test_read_tools_never_return_the_card_number(tmp_path):
    with _bank(tmp_path, "cus_001") as bank:
        cards = bank.list_cards()
    assert cards
    assert all("test_pan" not in c for c in cards)


def test_create_refuses_to_overwrite(tmp_path):
    path = tmp_path / "bank.db"
    Bank.create(path, load_seed())
    with pytest.raises(FileExistsError):
        Bank.create(path, load_seed())


def test_money_is_stored_in_cents(tmp_path):
    path = tmp_path / "bank.db"
    Bank.create(path, load_seed())
    conn = sqlite3.connect(path)
    (cents,) = conn.execute(
        "SELECT balance_cents FROM accounts WHERE account_id = 'acc_001m'"
    ).fetchone()
    conn.close()
    assert cents == 84216
