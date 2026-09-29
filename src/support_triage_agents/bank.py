"""The fake Tallowbrook bank in SQLite, and the support tools that act on it.

The dataset ships a reference model of what each support action does and
which rules it enforces (`scripts/bank_sim.py` in the Tallowbrook dataset, tag
tallowbrook-v0.1 of rkemery/rag-support-assistant). This module is an independent
SQLite implementation of the same semantics, so the two can check each other:
`tests/test_bank.py` runs every task's gold actions through these tools and
requires the resulting change to equal the task's `gold_final_state`, which the
dataset computed with its own model.

Design points:

- Money is stored as integer cents and exported as dollars.
- Every write runs in one SQLite transaction together with its idempotency
  record, keyed on (ticket_id, sha256 of the canonical action). Applying the
  same action twice for the same ticket is a no-op that returns the first
  result. So an executor that crashes after a write and is re-run from its
  last checkpoint cannot apply that write twice.
- Every write attempt, applied, refused or duplicate, is logged in
  `action_log`. Refusals carry a kind: `policy` (a support rule said no) or
  `invalid` (unknown record, missing or malformed arguments).
- Tools act only for the signed-in customer of the ticket. A record that
  belongs to someone else is refused as a policy violation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from calendar import monthrange
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PK = {
    "customers": "customer_id",
    "accounts": "account_id",
    "cards": "card_id",
    "transactions": "txn_id",
    "disputes": "dispute_id",
    "escalations": "escalation_id",
}
# Fields an implementation is free to choose, so they are not part of an insert's match.
# Same list as the dataset's reference model.
UNMATCHED_FIELDS = frozenset(
    {"date", "opened_date", "created_date", "summary", "description", "test_pan", "last4", "exp"}
)
# Columns stored as integer cents and exported as dollars under the seed's field name.
MONEY_COLUMNS = {
    "balance_cents": "balance",
    "amount_cents": "amount",
    "direct_deposit_last_30_days_cents": "direct_deposit_last_30_days_usd",
}
JSON_COLUMNS = {"scheduled_plan_change"}
BOOL_COLUMNS = {"cushion_enabled"}

WRITE_ACTIONS: dict[str, tuple[str, ...]] = {
    "freeze_card": ("card_id",),
    "unfreeze_card": ("card_id",),
    "report_card_lost_stolen": ("card_id", "reason"),
    "order_replacement_card": ("card_id", "reason", "shipping"),
    "open_dispute": ("txn_id", "reason"),
    "refund_fee": ("txn_id", "basis"),
    "change_plan": ("new_plan", "effective"),
    "close_account": (),
    "escalate_to_human": ("queue", "summary"),
}
OPTIONAL_ARGS = {"escalate_to_human": {"summary"}}
QUEUES = ("fraud", "disputes", "verification", "account_services", "complaints")
PLAN_ORDER = ("basic", "plus", "premium")
FEE_TYPE_TO_KEY = {
    "out_of_network_atm": "out_of_network_atm_fee_usd",
    "international_atm": "international_atm_fee_usd",
    "card_replacement": "card_replacement_fee_usd",
    "expedited_shipping": "expedited_shipping_fee_usd",
    "domestic_wire_out": "domestic_wire_out_fee_usd",
    "international_wire_out": "international_wire_out_fee_usd",
    "retail_cash_deposit": "retail_cash_deposit_fee_usd",
    "plan_fee": "monthly_fee_usd",
}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL,
    phone TEXT NOT NULL, date_of_birth TEXT NOT NULL, address TEXT NOT NULL,
    joined_date TEXT NOT NULL, plan TEXT NOT NULL, status TEXT NOT NULL,
    kyc_status TEXT NOT NULL, next_billing_date TEXT, first_paid_plan_charge_date TEXT,
    scheduled_plan_change TEXT, cushion_enabled INTEGER NOT NULL,
    direct_deposit_last_30_days_cents INTEGER NOT NULL
);
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers,
    type TEXT NOT NULL, name TEXT NOT NULL, balance_cents INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE cards (
    card_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers,
    account_id TEXT NOT NULL REFERENCES accounts, kind TEXT NOT NULL, material TEXT,
    status TEXT NOT NULL, test_pan TEXT, last4 TEXT, exp TEXT,
    replaces_card_id TEXT REFERENCES cards, shipping TEXT
);
CREATE TABLE transactions (
    txn_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers,
    account_id TEXT NOT NULL REFERENCES accounts, date TEXT NOT NULL, status TEXT NOT NULL,
    type TEXT NOT NULL, amount_cents INTEGER NOT NULL, description TEXT NOT NULL,
    card_id TEXT REFERENCES cards, fee_type TEXT, related_txn_id TEXT REFERENCES transactions,
    merchant_country TEXT, basis TEXT
);
CREATE TABLE disputes (
    dispute_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers,
    txn_id TEXT NOT NULL REFERENCES transactions, reason TEXT NOT NULL,
    amount_cents INTEGER NOT NULL, status TEXT NOT NULL, opened_date TEXT NOT NULL
);
CREATE TABLE escalations (
    escalation_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers,
    queue TEXT NOT NULL, status TEXT NOT NULL, created_date TEXT NOT NULL, summary TEXT
);
CREATE TABLE applied_actions (
    ticket_id TEXT NOT NULL, action_key TEXT NOT NULL, action TEXT NOT NULL,
    result TEXT NOT NULL, PRIMARY KEY (ticket_id, action_key)
);
CREATE TABLE action_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL, action TEXT NOT NULL,
    outcome TEXT NOT NULL, refusal_kind TEXT, reason TEXT, counted_dispute_cents INTEGER
);
"""


class ActionRefused(Exception):
    """The bank refused a write. `kind` is 'policy' (a support rule) or 'invalid' (bad input)."""

    def __init__(self, kind: Literal["policy", "invalid"], reason: str, code: str = "") -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.code = code


def _policy(reason: str, code: str = "") -> ActionRefused:
    return ActionRefused("policy", reason, code)


def _invalid(reason: str) -> ActionRefused:
    return ActionRefused("invalid", reason)


def action_key(action: dict[str, Any]) -> str:
    """sha256 of the canonical action (name plus arguments, summary excluded).

    The escalation summary is free text a model may word differently on a
    retry, so it is not part of what makes two escalations the same action.
    """
    args = {k: v for k, v in (action.get("args") or {}).items() if k != "summary"}
    blob = json.dumps({"action": action["action"], "args": args}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cents(value: float) -> int:
    return round(value * 100)


def _d(value: str | dt.date) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


def statement_date(txn_date: dt.date) -> dt.date:
    """Statements close on the last calendar day of each month."""
    return txn_date.replace(day=monthrange(txn_date.year, txn_date.month)[1])


def add_month(d: dt.date) -> dt.date:
    y, m = (d.year + (d.month // 12), d.month % 12 + 1)
    return d.replace(year=y, month=m, day=min(d.day, monthrange(y, m)[1]))


@dataclass(frozen=True)
class WriteResult:
    action: dict[str, Any]
    outcome: Literal["applied", "duplicate"]
    detail: dict[str, Any]


class Bank:
    """One SQLite database of bank state plus the support tools, scoped to one customer.

    `Bank.create(path, seed)` writes a fresh database from the seed.
    `Bank(path, facts, customer_id, ticket_id)` opens an existing one, so a
    resumed process can pick up where a killed one stopped.
    """

    def __init__(
        self, path: str | Path, facts: dict[str, Any], customer_id: str, ticket_id: str
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no bank database at {self.path}")
        self.facts = facts
        self.customer_id = customer_id
        self.ticket_id = ticket_id
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.as_of = _d(self._meta("as_of"))

    @classmethod
    def create(cls, path: str | Path, seed: dict[str, Any]) -> Path:
        """Write a new database from the seed. Refuses to overwrite an existing file."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.executescript(SCHEMA)
            conn.execute("INSERT INTO meta VALUES ('as_of', ?)", (seed["as_of"],))
            conn.execute("INSERT INTO meta VALUES ('next_id', '0')")
            # Insert order respects foreign keys (replacement cards and refunds point
            # at earlier rows of their own table, which the seed lists first).
            for table in PK:
                for row in seed[table]:
                    _insert(conn, table, row)
            conn.commit()
        finally:
            conn.close()
        return path

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Bank:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _meta(self, key: str) -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"meta key {key!r} missing from {self.path}")
        return row["value"]

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _row(self, table: str, rid: str) -> dict[str, Any]:
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE {PK[table]} = ?",
            (rid,),
        ).fetchone()
        if row is None:
            raise _invalid(f"{table} {rid} not found")
        return dict(row)

    def _own(self, row: dict[str, Any], rid: str) -> None:
        if row.get("customer_id") != self.customer_id:
            raise _policy(f"{rid} does not belong to the signed-in customer", "not_owner")

    def _customer(self) -> dict[str, Any]:
        return self._row("customers", self.customer_id)

    def _main_account(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM accounts WHERE customer_id = ? AND type = 'main'", (self.customer_id,)
        ).fetchone()
        if row is None:
            raise _invalid("no main account")
        return dict(row)

    def _plan_fees(self, plan: str) -> dict[str, Any]:
        fees = self.facts["fees"]
        return fees["versions"][f"v{fees['current_version']}"][plan]

    def _new_id(self, prefix: str) -> str:
        n = int(self._meta("next_id")) + 1
        self._conn.execute("UPDATE meta SET value = ? WHERE key = 'next_id'", (str(n),))
        return f"{prefix}_new{n:02d}"

    def _set(self, table: str, rid: str, **values: Any) -> None:
        cols = ", ".join(f"{c} = ?" for c in values)
        self._conn.execute(
            f"UPDATE {table} SET {cols} WHERE {PK[table]} = ?",
            (*values.values(), rid),
        )

    def _charge(self, cents: int, fee_type: str, description: str) -> None:
        if cents <= 0:
            return
        acc = self._main_account()
        _insert_raw(
            self._conn,
            "transactions",
            {
                "txn_id": self._new_id("txn"),
                "account_id": acc["account_id"],
                "customer_id": self.customer_id,
                "date": self.as_of.isoformat(),
                "status": "posted",
                "type": "fee",
                "fee_type": fee_type,
                "amount_cents": -cents,
                "description": description,
            },
        )
        self._set("accounts", acc["account_id"], balance_cents=acc["balance_cents"] - cents)

    def _require_active_customer(self) -> None:
        if self._customer()["status"] != "active":
            raise _policy(
                "customer is not active, support can't act, escalate instead", "not_active"
            )

    # ------------------------------------------------------------------ read tools

    def get_customer_profile(self) -> dict[str, Any]:
        """Plan, status and billing fields. No date of birth or address."""
        c = _export_row("customers", self._customer())
        keep = (
            "customer_id",
            "full_name",
            "plan",
            "status",
            "kyc_status",
            "joined_date",
            "next_billing_date",
            "first_paid_plan_charge_date",
            "scheduled_plan_change",
            "cushion_enabled",
            "direct_deposit_last_30_days_usd",
        )
        return {k: c.get(k) for k in keep}

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM accounts WHERE customer_id = ? ORDER BY account_id",
            (self.customer_id,),
        ).fetchall()
        return [_export_row("accounts", dict(r)) for r in rows]

    def list_cards(self) -> list[dict[str, Any]]:
        """Cards without the full card number (support never reveals it)."""
        rows = self._conn.execute(
            "SELECT * FROM cards WHERE customer_id = ? ORDER BY card_id", (self.customer_id,)
        ).fetchall()
        out = []
        for r in rows:
            card = _export_row("cards", dict(r))
            card.pop("test_pan", None)
            out.append(card)
        return out

    def list_transactions(self, limit: int = 40) -> list[dict[str, Any]]:
        """Most recent first."""
        if not isinstance(limit, int) or limit <= 0:
            raise _invalid("limit must be a positive int")
        rows = self._conn.execute(
            "SELECT * FROM transactions WHERE customer_id = ? ORDER BY date DESC, txn_id DESC "
            "LIMIT ?",
            (self.customer_id, min(limit, 100)),
        ).fetchall()
        return [_export_row("transactions", dict(r)) for r in rows]

    def list_disputes(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM disputes WHERE customer_id = ? ORDER BY dispute_id",
            (self.customer_id,),
        ).fetchall()
        return [_export_row("disputes", dict(r)) for r in rows]

    # ------------------------------------------------------------------ write tools

    def apply(self, action: dict[str, Any]) -> WriteResult:
        """Validate and apply one write action, idempotently for this ticket.

        Raises `ActionRefused` when the bank says no. Every attempt is logged.
        """
        name = action.get("action")
        args = dict(action.get("args") or {})
        if name not in WRITE_ACTIONS:
            self._log(action, "refused", "invalid", f"unknown action {name}")
            raise _invalid(f"unknown action {name}")
        allowed = set(WRITE_ACTIONS[name])
        missing = [a for a in WRITE_ACTIONS[name] if a not in args and a not in _optional(name)]
        unknown = sorted(set(args) - allowed)
        if missing or unknown:
            reason = f"{name}: missing args {missing}, unknown args {unknown}"
            self._log(action, "refused", "invalid", reason)
            raise _invalid(reason)
        clean = {"action": name, "args": args}
        key = action_key(clean)
        done = self._conn.execute(
            "SELECT result FROM applied_actions WHERE ticket_id = ? AND action_key = ?",
            (self.ticket_id, key),
        ).fetchone()
        if done is not None:
            self._log(clean, "duplicate", None, None)
            return WriteResult(clean, "duplicate", json.loads(done["result"]))
        self._dispute_counted = 0
        try:
            with self._transaction() as conn:
                detail = getattr(self, f"_do_{name}")(**args) or {}
                conn.execute(
                    "INSERT INTO applied_actions VALUES (?, ?, ?, ?)",
                    (self.ticket_id, key, json.dumps(clean, sort_keys=True), json.dumps(detail)),
                )
                self._log(clean, "applied", None, None, self._dispute_counted)
        except ActionRefused as exc:
            self._log(clean, "refused", exc.kind, exc.reason, self._dispute_counted)
            raise
        except TypeError as exc:  # wrong argument types reach the handlers as TypeError
            self._log(clean, "refused", "invalid", str(exc))
            raise _invalid(str(exc)) from exc
        return WriteResult(clean, "applied", detail)

    def _log(
        self,
        action: dict[str, Any],
        outcome: str,
        kind: str | None,
        reason: str | None,
        counted_dispute_cents: int = 0,
    ) -> None:
        self._conn.execute(
            "INSERT INTO action_log (ticket_id, action, outcome, refusal_kind, reason, "
            "counted_dispute_cents) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.ticket_id,
                json.dumps(action, sort_keys=True, default=str),
                outcome,
                kind,
                reason,
                counted_dispute_cents,
            ),
        )

    def action_log(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM action_log WHERE ticket_id = ? ORDER BY seq", (self.ticket_id,)
        ).fetchall()
        return [{**dict(r), "action": json.loads(r["action"])} for r in rows]

    def applied_actions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT action FROM applied_actions WHERE ticket_id = ? ORDER BY rowid",
            (self.ticket_id,),
        ).fetchall()
        return [json.loads(r["action"]) for r in rows]

    def _card(self, card_id: Any) -> dict[str, Any]:
        if not isinstance(card_id, str):
            raise _invalid("card_id must be a string")
        card = self._row("cards", card_id)
        self._own(card, card_id)
        return card

    def _do_freeze_card(self, card_id: str) -> dict[str, Any]:
        self._require_active_customer()
        card = self._card(card_id)
        if card["status"] != "active":
            raise _policy(f"can only freeze an active card, {card_id} is {card['status']}")
        self._set("cards", card_id, status="frozen")
        return {"card_id": card_id, "status": "frozen"}

    def _do_unfreeze_card(self, card_id: str) -> dict[str, Any]:
        self._require_active_customer()
        card = self._card(card_id)
        if card["status"] != "frozen":
            raise _policy(f"can only unfreeze a frozen card, {card_id} is {card['status']}")
        self._set("cards", card_id, status="active")
        return {"card_id": card_id, "status": "active"}

    def _do_report_card_lost_stolen(self, card_id: str, reason: str) -> dict[str, Any]:
        self._require_active_customer()
        if reason not in ("lost", "stolen"):
            raise _invalid("reason must be lost or stolen")
        card = self._card(card_id)
        if card["status"] not in ("active", "frozen"):
            raise _policy(f"can't report a card that is {card['status']}")
        self._set("cards", card_id, status=reason)
        return {"card_id": card_id, "status": reason}

    def _do_order_replacement_card(self, card_id: str, reason: str, shipping: str) -> dict:
        self._require_active_customer()
        card = self._card(card_id)
        if card["kind"] != "physical":
            raise _policy("only physical cards are replaced by mail")
        if reason in ("lost", "stolen") and card["status"] != reason:
            raise _policy("report the card lost or stolen before ordering a replacement")
        if reason == "damaged" and card["status"] not in ("active", "frozen"):
            raise _policy("damaged replacement needs a working card")
        if shipping not in ("standard", "expedited"):
            raise _invalid("shipping must be standard or expedited")
        plan = self._customer()["plan"]
        fees = self._plan_fees(plan)
        new_id = self._new_id("card")
        _insert_raw(
            self._conn,
            "cards",
            {
                "card_id": new_id,
                "customer_id": self.customer_id,
                "account_id": card["account_id"],
                "kind": "physical",
                "material": "metal" if plan == "premium" else "plastic",
                "status": "shipping",
                "replaces_card_id": card_id,
                "shipping": shipping,
            },
        )
        self._charge(
            _cents(float(fees["card_replacement_fee_usd"])),
            "card_replacement",
            "Card replacement fee",
        )
        if shipping == "expedited":
            self._charge(
                _cents(float(fees["expedited_shipping_fee_usd"])),
                "expedited_shipping",
                "Expedited shipping fee",
            )
        return {"new_card_id": new_id}

    def _do_open_dispute(self, txn_id: str, reason: str) -> dict[str, Any]:
        self._require_active_customer()
        d = self.facts["disputes"]
        if reason not in d["reasons"]:
            raise _invalid(f"unknown dispute reason {reason}")
        if not isinstance(txn_id, str):
            raise _invalid("txn_id must be a string")
        txn = self._row("transactions", txn_id)
        self._own(txn, txn_id)
        if txn["status"] != "posted":
            raise _policy("pending transactions can't be disputed")
        open_dispute = self._conn.execute(
            "SELECT 1 FROM disputes WHERE txn_id = ? AND status = 'open'", (txn_id,)
        ).fetchone()
        if open_dispute is not None:
            raise _policy("transaction already has an open dispute")
        tdate = _d(txn["date"])
        if reason == "unauthorized":
            window = (
                d["ach_unauthorized_window_days"]
                if txn["type"].startswith("ach")
                else d["unauthorized_window_days"]
            )
            if (self.as_of - statement_date(tdate)).days > window:
                raise _policy("outside the unauthorized dispute window")
            if txn["card_id"]:
                status = self._row("cards", txn["card_id"])["status"]
                if status not in ("frozen", "lost", "stolen", "fraud_lock"):
                    raise _policy("freeze or report the card before an unauthorized dispute")
        elif reason == "atm_cash_not_dispensed":
            if (self.as_of - tdate).days > self.facts["atm"][
                "cash_not_dispensed_dispute_window_days"
            ]:
                raise _policy("outside the ATM dispute window")
        else:
            window = d["versions"][f"v{d['current_version']}"]["merchant_dispute_window_days"]
            if (self.as_of - tdate).days > window:
                raise _policy("outside the merchant dispute window")
        amount = abs(txn["amount_cents"])
        # The support limit is per case. As in the reference model, an attempt that
        # reaches this check counts toward the case total even if it is refused here.
        self._dispute_counted = amount
        case_total = self._case_dispute_cents() + amount
        if case_total > _cents(float(d["support_max_dispute_usd"])):
            raise _policy("above the support dispute limit, escalate to disputes", "dispute_limit")
        new_id = self._new_id("dsp")
        _insert_raw(
            self._conn,
            "disputes",
            {
                "dispute_id": new_id,
                "customer_id": self.customer_id,
                "txn_id": txn_id,
                "reason": reason,
                "amount_cents": amount,
                "status": "open",
                "opened_date": self.as_of.isoformat(),
            },
        )
        return {"dispute_id": new_id}

    def _case_dispute_cents(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(counted_dispute_cents), 0) AS total FROM action_log "
            "WHERE ticket_id = ?",
            (self.ticket_id,),
        ).fetchone()
        return int(row["total"])

    def _fee_should_be_zero(self, txn: dict[str, Any], plan: str) -> bool:
        key = FEE_TYPE_TO_KEY.get(txn["fee_type"])
        if key is None:
            return False
        value = self._plan_fees(plan).get(key)
        return value is not None and float(value) == 0.0

    def _is_duplicate_fee(self, txn: dict[str, Any]) -> bool:
        if not txn["related_txn_id"]:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM transactions WHERE txn_id != ? AND type = 'fee' AND fee_type = ? "
            "AND related_txn_id = ?",
            (txn["txn_id"], txn["fee_type"], txn["related_txn_id"]),
        ).fetchone()
        return row is not None

    def _do_refund_fee(self, txn_id: str, basis: str) -> dict[str, Any]:
        self._require_active_customer()
        r = self.facts["refunds"]
        if not isinstance(txn_id, str):
            raise _invalid("txn_id must be a string")
        txn = self._row("transactions", txn_id)
        self._own(txn, txn_id)
        if txn["type"] != "fee":
            raise _policy("not a fee")
        refunded = self._conn.execute(
            "SELECT 1 FROM transactions WHERE type = 'fee_refund' AND related_txn_id = ?",
            (txn_id,),
        ).fetchone()
        if refunded is not None:
            raise _policy("fee already refunded")
        cust = self._customer()
        amount = abs(txn["amount_cents"])
        if basis == "error":
            if not (self._fee_should_be_zero(txn, cust["plan"]) or self._is_duplicate_fee(txn)):
                raise _policy("fee was not charged in error")
        elif basis == "goodwill":
            if txn["fee_type"] not in r["goodwill_eligible_fees"]:
                raise _policy("fee type not eligible for goodwill")
            if amount > _cents(float(r["goodwill_max_fee_usd"])):
                raise _policy("fee above the goodwill maximum")
            rows = self._conn.execute(
                "SELECT date FROM transactions WHERE type = 'fee_refund' AND basis = 'goodwill' "
                "AND customer_id = ?",
                (self.customer_id,),
            ).fetchall()
            if any(
                (self.as_of - _d(t["date"])).days < r["goodwill_per_rolling_days"] for t in rows
            ):
                raise _policy("goodwill refund already used in the last 12 months")
        elif basis == "cooling_off":
            if txn["fee_type"] != "plan_fee":
                raise _policy("cooling-off only applies to plan fees")
            if cust["first_paid_plan_charge_date"] != txn["date"]:
                raise _policy("not the first paid plan charge")
            if (self.as_of - _d(txn["date"])).days > self.facts["plan_billing"]["cooling_off_days"]:
                raise _policy("outside the cooling-off period")
        else:
            raise _invalid(f"unknown refund basis {basis}")
        acc = self._row("accounts", txn["account_id"])
        new_id = self._new_id("txn")
        _insert_raw(
            self._conn,
            "transactions",
            {
                "txn_id": new_id,
                "account_id": acc["account_id"],
                "customer_id": self.customer_id,
                "date": self.as_of.isoformat(),
                "status": "posted",
                "type": "fee_refund",
                "amount_cents": amount,
                "related_txn_id": txn_id,
                "basis": basis,
                "description": "Fee refund",
            },
        )
        self._set("accounts", acc["account_id"], balance_cents=acc["balance_cents"] + amount)
        return {"refund_txn_id": new_id, "amount": amount / 100}

    def _do_change_plan(self, new_plan: str, effective: str) -> dict[str, Any]:
        self._require_active_customer()
        cust = self._customer()
        old = cust["plan"]
        if new_plan not in PLAN_ORDER or new_plan == old:
            raise _invalid("invalid plan change")
        if PLAN_ORDER.index(new_plan) > PLAN_ORDER.index(old):
            if effective != "immediate":
                raise _policy("upgrades take effect immediately")
            fee = _cents(float(self._plan_fees(new_plan)["monthly_fee_usd"]))
            if self._main_account()["balance_cents"] < fee:
                raise _policy("not enough money for the new plan fee")
            updates: dict[str, Any] = {
                "plan": new_plan,
                "scheduled_plan_change": None,
                "next_billing_date": add_month(self.as_of).isoformat(),
            }
            if cust["first_paid_plan_charge_date"] is None:
                updates["first_paid_plan_charge_date"] = self.as_of.isoformat()
            self._set("customers", self.customer_id, **updates)
            self._charge(fee, "plan_fee", f"Plan fee: {new_plan.capitalize()}")
            return {"plan": new_plan}
        if effective == "next_billing_date":
            change = {"plan": new_plan, "effective_date": cust["next_billing_date"]}
            self._set("customers", self.customer_id, scheduled_plan_change=json.dumps(change))
            return {"scheduled_plan_change": change}
        if effective == "immediate":
            refunded = self._conn.execute(
                "SELECT 1 FROM transactions WHERE type = 'fee_refund' AND basis = 'cooling_off' "
                "AND customer_id = ?",
                (self.customer_id,),
            ).fetchone()
            if refunded is None or new_plan != "basic":
                raise _policy(
                    "immediate downgrades only happen with a cooling-off refund, to Basic"
                )
            self._set(
                "customers",
                self.customer_id,
                plan=new_plan,
                next_billing_date=None,
                scheduled_plan_change=None,
            )
            return {"plan": new_plan}
        raise _invalid("effective must be immediate or next_billing_date")

    def _do_close_account(self) -> dict[str, Any]:
        self._require_active_customer()
        cid = self.customer_id
        accounts = self._conn.execute(
            "SELECT * FROM accounts WHERE customer_id = ?", (cid,)
        ).fetchall()
        if any(a["balance_cents"] != 0 for a in accounts):
            raise _policy("support can only close an account with a $0.00 balance")
        pending = self._conn.execute(
            "SELECT 1 FROM transactions WHERE customer_id = ? AND status = 'pending'", (cid,)
        ).fetchone()
        if pending is not None:
            raise _policy("pending transactions")
        disputes = self._conn.execute(
            "SELECT 1 FROM disputes WHERE customer_id = ? AND status = 'open'", (cid,)
        ).fetchone()
        if disputes is not None:
            raise _policy("open dispute")
        self._set("customers", cid, status="closed")
        self._conn.execute("UPDATE accounts SET status = 'closed' WHERE customer_id = ?", (cid,))
        self._conn.execute(
            "UPDATE cards SET status = 'cancelled' WHERE customer_id = ? "
            "AND status IN ('active', 'frozen', 'shipping', 'inactive')",
            (cid,),
        )
        return {"status": "closed"}

    def _do_escalate_to_human(self, queue: str, summary: str = "") -> dict[str, Any]:
        if queue not in QUEUES:
            raise _invalid(f"unknown queue {queue}")
        new_id = self._new_id("esc")
        _insert_raw(
            self._conn,
            "escalations",
            {
                "escalation_id": new_id,
                "customer_id": self.customer_id,
                "queue": queue,
                "status": "open",
                "created_date": self.as_of.isoformat(),
                "summary": str(summary),
            },
        )
        return {"escalation_id": new_id, "queue": queue}

    # ------------------------------------------------------------------ export

    def export_state(self) -> dict[str, list[dict[str, Any]]]:
        """Every table as the seed's JSON shape (dollars, parsed JSON, bools)."""
        return export_state(self._conn)


def _optional(name: str) -> set[str]:
    return OPTIONAL_ARGS.get(name, set())


def _to_columns(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        money = next((col for col, field in MONEY_COLUMNS.items() if field == key), None)
        if money is not None:
            out[money] = _cents(value)
        elif key in JSON_COLUMNS:
            out[key] = None if value is None else json.dumps(value)
        elif key in BOOL_COLUMNS:
            out[key] = int(bool(value))
        else:
            out[key] = value
    return out


def _insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    _insert_raw(conn, table, _to_columns(row))


def _insert_raw(conn: sqlite3.Connection, table: str, columns: dict[str, Any]) -> None:
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {table} ({names}) VALUES ({marks})",
        tuple(columns.values()),
    )


def _export_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """Seed-shaped dict. NULL columns are dropped, since the seed omits absent fields."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in MONEY_COLUMNS:
            out[MONEY_COLUMNS[key]] = value / 100
        elif key in JSON_COLUMNS:
            out[key] = None if value is None else json.loads(value)
        elif key in BOOL_COLUMNS:
            out[key] = bool(value)
        else:
            out[key] = value
    return {k: v for k, v in out.items() if v is not None}


def export_state(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    conn.row_factory = sqlite3.Row
    state: dict[str, list[dict[str, Any]]] = {}
    for table, pk in PK.items():
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        state[table] = [_export_row(table, dict(r)) for r in rows]
        if len({r[pk] for r in state[table]}) != len(state[table]):
            raise ValueError(f"duplicate ids in {table}")
    return state


# ---------------------------------------------------------------------- diffs


def _drop_none(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if v is not None}


def diff_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[dict]]:
    """The change from `before` to `after` in the dataset's gold_final_state format.

    `updated`: one entry per changed existing record with {field: {from, to}}.
    `inserted`: one entry per new record with every field except the primary
    key and the implementation-chosen fields. A missing field and a null are
    the same thing, as in the reference model.
    """
    updated, inserted = [], []
    for table, pk in PK.items():
        old = {r[pk]: r for r in before[table]}
        for rec in after[table]:
            rid = rec[pk]
            if rid in old:
                changes = {}
                for key in sorted(set(old[rid]) | set(rec)):
                    if old[rid].get(key) != rec.get(key):
                        changes[key] = {"from": old[rid].get(key), "to": rec.get(key)}
                if changes:
                    updated.append({"table": table, "id": rid, "changes": changes})
            else:
                match = {
                    k: v
                    for k, v in rec.items()
                    if k != pk and k not in UNMATCHED_FIELDS and v is not None
                }
                inserted.append({"table": table, "match": match})
    return {"updated": updated, "inserted": inserted}


def _updated_key(entry: dict[str, Any]) -> tuple:
    changes = tuple(
        sorted(
            (
                field,
                json.dumps(ch.get("from"), sort_keys=True),
                json.dumps(ch.get("to"), sort_keys=True),
            )
            for field, ch in entry["changes"].items()
        )
    )
    return (entry["table"], entry["id"], changes)


def _inserted_key(entry: dict[str, Any]) -> tuple:
    match = _drop_none({k: v for k, v in entry["match"].items() if k not in UNMATCHED_FIELDS})
    return (entry["table"], json.dumps(match, sort_keys=True))


def state_matches(diff: dict[str, Any], gold: dict[str, Any]) -> bool:
    """True when a diff equals the gold final state.

    Updates compare as a set of (table, id, changes). Inserts compare as a
    multiset of (table, fields), ignoring the implementation-chosen fields, so
    the order actions ran in and the IDs they minted don't matter.
    """
    return (
        {_updated_key(u) for u in diff["updated"]} == {_updated_key(u) for u in gold["updated"]}
        and len(diff["updated"]) == len(gold["updated"])
        and Counter(_inserted_key(i) for i in diff["inserted"])
        == Counter(_inserted_key(i) for i in gold["inserted"])
    )


def new_bank_for_task(
    path: str | Path, seed: dict[str, Any], facts: dict[str, Any], customer_id: str, ticket_id: str
) -> Bank:
    """Create a fresh database from the seed at `path` and open it for one ticket."""
    Bank.create(path, seed)
    return Bank(path, facts, customer_id, ticket_id)
