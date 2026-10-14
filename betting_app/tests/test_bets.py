"""Tests for /api/bets and /api/wallets endpoints."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_UP
import json
from threading import Barrier
from time import monotonic, sleep
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from betting_app.core.db import get_session


def _seed_wallet(balance: float = 1000.0) -> int:
    """Create a test wallet, return its ID."""
    session = get_session()
    wid = session.execute(
        text("""
        INSERT INTO bookmaker_accounts (bookmaker_id, account_name, currency, opening_balance, current_balance, is_active, created_at, updated_at)
        VALUES (1, 'test-wallet', 'PLN', :bal, :bal, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        RETURNING id
        """),
        {"bal": balance},
    ).scalar_one()
    session.commit()
    session.close()
    return int(wid)


CENT = Decimal("0.01")


def _cents(value: str | Decimal) -> Decimal:
    """Independent public-money oracle; never use production rounding helpers."""
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _wallet_state(wallet_id: int) -> dict:
    """Read committed Numeric values on a separate PostgreSQL transaction."""
    with get_session() as session:
        assert session.bind.dialect.name == "postgresql"
        wallet = dict(session.execute(
            text("SELECT opening_balance, current_balance FROM bookmaker_accounts WHERE id=:id"),
            {"id": wallet_id},
        ).mappings().one())
        bets = [dict(row) for row in session.execute(
            text("SELECT * FROM bets WHERE bookmaker_account_id=:id ORDER BY id"),
            {"id": wallet_id},
        ).mappings()]
        ledger = [dict(row) for row in session.execute(
            text(
                "SELECT bet_id, transaction_type, amount, balance_after "
                "FROM bookmaker_wallet_transactions "
                "WHERE bookmaker_account_id=:id ORDER BY id"
            ),
            {"id": wallet_id},
        ).mappings()]
    return {"wallet": wallet, "bets": bets, "ledger": ledger}


def _assert_money_reconciles(state: dict) -> None:
    balance = state["wallet"]["opening_balance"]
    assert isinstance(balance, Decimal)
    for entry in state["ledger"]:
        assert isinstance(entry["amount"], Decimal)
        balance += entry["amount"]
        assert entry["balance_after"] == balance
    assert state["wallet"]["current_balance"] == balance
    for bet in state["bets"]:
        entries = [entry for entry in state["ledger"] if entry["bet_id"] == bet["id"]]
        assert entries[0]["transaction_type"] == "bet_placed"
        assert entries[0]["amount"] == -bet["stake"]
        if bet["status"] != "open":
            assert sum((entry["amount"] for entry in entries), Decimal("0")) == bet["profit"]


@contextmanager
def _reject_ledger_insert(wallet_id: int, kind: str):
    """Inject a real PostgreSQL write failure after earlier writes have executed."""
    name = f"plan11_wallet_failure_{uuid4().hex}"
    with get_session() as session:
        session.execute(text(
            f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'plan11 injected ledger persistence failure'; END; $$"
        ))
        session.execute(text(
            f"CREATE TRIGGER {name} BEFORE INSERT ON bookmaker_wallet_transactions "
            f"FOR EACH ROW WHEN (NEW.bookmaker_account_id = {int(wallet_id)} "
            f"AND NEW.transaction_type = '{kind}') EXECUTE FUNCTION {name}()"
        ))
        session.commit()
    try:
        yield
    finally:
        with get_session() as session:
            session.execute(text(f"DROP TRIGGER {name} ON bookmaker_wallet_transactions"))
            session.execute(text(f"DROP FUNCTION {name}()"))
            session.commit()


def _concurrent_posts(client: TestClient, table: str, row_id: int, path: str, payload: dict):
    """Hold a real row lock until both independent HTTP transactions contend."""
    assert table in {"bookmaker_accounts", "bets"}
    start = Barrier(2)

    def post():
        start.wait(timeout=10)
        return client.post(path, json=payload)

    with get_session() as locker:
        locker.execute(text(f"SELECT id FROM {table} WHERE id=:id FOR UPDATE"), {"id": row_id})
        lock_pid = locker.execute(text("SELECT pg_backend_pid()")).scalar_one()
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(post) for _ in range(2)]
            try:
                deadline = monotonic() + 10
                with get_session() as observer:
                    while True:
                        # PostgreSQL can cache activity information within a transaction.
                        observer.execute(text("SELECT pg_stat_clear_snapshot()"))
                        blocked = observer.execute(text(
                            "SELECT count(*) FROM pg_stat_activity a "
                            "WHERE a.datname = current_database() AND EXISTS ("
                            "SELECT 1 FROM unnest(pg_blocking_pids(a.pid)) AS blocking(pid) "
                            "WHERE blocking.pid=:pid OR :pid=ANY(pg_blocking_pids(blocking.pid)))"
                        ), {"pid": lock_pid}).scalar_one()
                        if blocked >= 2:
                            break
                        assert monotonic() < deadline, "Both HTTP transactions must reach PostgreSQL contention"
                        sleep(0.02)
            finally:
                # Release even on assertion failure before waiting for worker shutdown.
                locker.rollback()
            return [future.result(timeout=15) for future in futures]


class TestWallets:
    def test_list_empty(self, client: TestClient):
        resp = client.get("/wallets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_list(self, client: TestClient):
        resp = client.post(
            "/wallets",
            params={"bookmaker_id": 1, "account_name": "test", "opening_balance": 500},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["current_balance"] == 500.0
        assert data["account_name"] == "test"
        assert data["id"] > 0

        resp = client.get("/wallets")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_create_wallet_negative_opening_balance(self, client: TestClient):
        resp = client.post(
            "/wallets",
            params={"bookmaker_id": 1, "account_name": "neg", "opening_balance": -50},
        )
        assert resp.status_code == 400

    def test_wallet_non_negative_balance_constraint(self, client: TestClient):
        from sqlalchemy.exc import IntegrityError
        session = get_session()
        try:
            import pytest
            with pytest.raises(IntegrityError):
                session.execute(
                    text("""
                    INSERT INTO bookmaker_accounts (bookmaker_id, account_name, currency, opening_balance, current_balance, is_active, created_at, updated_at)
                    VALUES (1, 'bad-wallet', 'PLN', -10.0, -10.0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """)
                )
                session.commit()
        finally:
            session.rollback()
            session.close()

class TestBets:
    def test_list_empty(self, client: TestClient):
        resp = client.get("/bets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_place_and_settle_win(self, client: TestClient):
        wid = _seed_wallet()
        resp = client.post(
            "/bets",
            json={
                "bookmaker_account_id": wid,
                "team_a": "TeamA",
                "team_b": "TeamB",
                "side": "a",
                "stake": 100,
                "odds": 2.0,
            },
        )
        assert resp.status_code == 201, resp.text
        bet = resp.json()
        assert bet["status"] == "open"
        assert bet["stake"] == 100.0

        resp = client.post(f"/bets/{bet['id']}/settle", json={"result": "won"})
        assert resp.status_code == 200
        settled = resp.json()
        assert settled["status"] == "won"
        assert settled["profit"] == 76.0

        resp = client.get("/wallets")
        wallets = resp.json()
        assert wallets[0]["current_balance"] == 1076.0

    def test_settlement_is_idempotent(self, client: TestClient):
        wid = _seed_wallet()
        placed = client.post(
            "/bets",
            json={
                "bookmaker_account_id": wid,
                "side": "a",
                "stake": 100,
                "odds": 2.0,
            },
        ).json()
        assert client.post(
            f"/bets/{placed['id']}/settle",
            json={"result": "won"},
        ).status_code == 200
        assert client.post(
            f"/bets/{placed['id']}/settle",
            json={"result": "won"},
        ).status_code == 400
        assert client.get("/wallets").json()[0]["current_balance"] == 1076.0
        session = get_session()
        try:
            assert session.execute(
                text(
                    "SELECT COUNT(*) FROM bookmaker_wallet_transactions "
                    "WHERE bet_id = :bet_id"
                ),
                {"bet_id": placed["id"]},
            ).scalar_one() == 2
        finally:
            session.close()

    def test_place_and_settle_loss(self, client: TestClient):
        wid = _seed_wallet()
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "b", "stake": 50, "odds": 3.0},
        )
        assert resp.status_code == 201, resp.text
        bet = resp.json()
        resp = client.post(f"/bets/{bet['id']}/settle", json={"result": "lost"})
        assert resp.status_code == 200
        settled = resp.json()
        assert settled["status"] == "lost"
        assert settled["profit"] == -50.0

    def test_insufficient_balance(self, client: TestClient):
        wid = _seed_wallet()
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "a", "stake": 9999, "odds": 2.0},
        )
        assert resp.status_code == 400
        session = get_session()
        try:
            assert session.execute(text("SELECT COUNT(*) FROM bets")).scalar_one() == 0
            assert session.execute(
                text("SELECT COUNT(*) FROM bookmaker_wallet_transactions")
            ).scalar_one() == 0
            assert session.execute(
                text("SELECT current_balance FROM bookmaker_accounts WHERE id = :id"),
                {"id": wid},
            ).scalar_one() == 1000.0
        finally:
            session.close()

    def test_wallet_not_found(self, client: TestClient):
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": 999, "side": "a", "stake": 10, "odds": 2.0},
        )
        assert resp.status_code == 404

    def test_settle_loss_is_idempotent(self, client: TestClient):
        wid = _seed_wallet()
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "b", "stake": 50, "odds": 3.0},
        )
        assert resp.status_code == 201
        bet = resp.json()
        # First settle as lost
        resp1 = client.post(f"/bets/{bet['id']}/settle", json={"result": "lost"})
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "lost"
        assert resp1.json()["profit"] == -50.0

        # Second settle must fail
        resp2 = client.post(f"/bets/{bet['id']}/settle", json={"result": "lost"})
        assert resp2.status_code == 400
        assert "already settled" in resp2.json()["detail"].lower()

        # Third settle as won must also fail
        resp3 = client.post(f"/bets/{bet['id']}/settle", json={"result": "won"})
        assert resp3.status_code == 400

        # Balance remains deducted only once
        wallets = client.get("/wallets").json()
        assert wallets[0]["current_balance"] == 950.0

    def test_settle_void_refunds_stake(self, client: TestClient):
        wid = _seed_wallet()
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "a", "stake": 150, "odds": 2.5},
        )
        assert resp.status_code == 201
        bet = resp.json()
        assert client.get("/wallets").json()[0]["current_balance"] == 850.0

        # Settle void
        resp_void = client.post(f"/bets/{bet['id']}/settle", json={"result": "void"})
        assert resp_void.status_code == 200
        settled = resp_void.json()
        assert settled["status"] == "void"
        assert settled["profit"] == 0.0

        # Wallet must be refunded
        assert client.get("/wallets").json()[0]["current_balance"] == 1000.0

        # Idempotency check: settling again must be rejected
        assert client.post(f"/bets/{bet['id']}/settle", json={"result": "void"}).status_code == 400
        assert client.get("/wallets").json()[0]["current_balance"] == 1000.0

    def test_atomicity_on_bet_placement_error(self, client: TestClient):
        wid = _seed_wallet()
        # Invalid side ('c' not allowed by schema)
        resp = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "invalid", "stake": 100, "odds": 2.0},
        )
        assert resp.status_code == 422
        # Verify wallet balance unchanged, no bet, no transaction
        assert client.get("/wallets").json()[0]["current_balance"] == 1000.0
        session = get_session()
        try:
            assert session.execute(text("SELECT COUNT(*) FROM bets")).scalar_one() == 0
            assert session.execute(
                text("SELECT COUNT(*) FROM bookmaker_wallet_transactions")
            ).scalar_one() == 0
        finally:
            session.close()

    def test_cannot_overdraw_wallet(self, client: TestClient):
        wid = _seed_wallet(balance=100.0)
        # Place bet 1: 60 PLN -> balance becomes 40 PLN
        resp1 = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "a", "stake": 60, "odds": 2.0},
        )
        assert resp1.status_code == 201
        assert client.get("/wallets").json()[0]["current_balance"] == 40.0

        # Place bet 2: 60 PLN -> insufficient balance (needs 60, only 40 left)
        resp2 = client.post(
            "/bets",
            json={"bookmaker_account_id": wid, "side": "b", "stake": 60, "odds": 2.0},
        )
        assert resp2.status_code == 400
        # Verify wallet remains at 40 PLN, only 1 bet placed
        assert client.get("/wallets").json()[0]["current_balance"] == 40.0
        session = get_session()
        try:
            assert session.execute(text("SELECT COUNT(*) FROM bets")).scalar_one() == 1
            assert session.execute(
                text("SELECT COUNT(*) FROM bookmaker_wallet_transactions")
            ).scalar_one() == 1
        finally:
            session.close()

    def test_settle_nonexistent_bet(self, client: TestClient):
        resp = client.post("/bets/99999/settle", json={"result": "won"})
        assert resp.status_code == 404


class TestWalletMoneyContracts:
    @pytest.mark.parametrize(
        "raw_stake,raw_odds,raw_tax,result",
        [
            pytest.param("10.005", "2", "0.12", "won", id="fractional-win"),
            pytest.param("10.005", "2", "0.12", "lost", id="fractional-loss"),
            pytest.param("10.005", "2", "0.12", "void", id="fractional-void"),
            pytest.param("10.005", "2", "0.12", "cancelled", id="fractional-cancelled"),
            pytest.param("0.01", "1.5", "0", "won", id="half-cent-win"),
            pytest.param("0.05", "1.25", "0.12", "won", id="taxed-half-cent-win"),
            pytest.param("10.01", "1.23456", "0.125", "won", id="persisted-odds-and-tax"),
        ],
    )
    def test_http_money_uses_persisted_decimal_values(
        self, client: TestClient, raw_stake: str, raw_odds: str, raw_tax: str, result: str
    ):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid,
            "side": "a",
            "stake": raw_stake,
            "odds": raw_odds,
            "tax_rate": raw_tax,
            "note": "Manual money contract; no qualified recommendation",
        })
        assert response.status_code == 201, response.text
        placed = response.json()
        assert isinstance(placed["stake"], (int, float))
        assert isinstance(placed["odds"], (int, float))
        before = _wallet_state(wid)
        bet = before["bets"][0]
        expected_stake = _cents(raw_stake)
        assert bet["source"] == "manual"
        assert bet["model_ev_signal_id"] is None
        assert bet["signal_id"] is None
        assert bet["stake"] == expected_stake
        assert Decimal(str(placed["stake"])) == expected_stake
        assert before["wallet"]["current_balance"] == Decimal("100.00") - expected_stake
        _assert_money_reconciles(before)

        if result == "won":
            payout = _cents(bet["stake"] * bet["taken_odds"] * (Decimal("1") - bet["tax_rate"]))
        elif result in {"void", "cancelled"}:
            payout = bet["stake"]
        else:
            payout = Decimal("0.00")
        expected_profit = payout - bet["stake"]
        response = client.post(f"/bets/{placed['id']}/settle", json={"result": result})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == result
        assert Decimal(str(response.json()["profit"])) == expected_profit
        after = _wallet_state(wid)
        assert after["bets"][0]["profit"] == expected_profit
        assert after["wallet"]["current_balance"] == Decimal("100.00") - expected_stake + payout
        assert len(after["ledger"]) == (1 if result == "lost" else 2)
        if payout:
            assert after["ledger"][1]["transaction_type"] == f"settled_{result}"
            assert after["ledger"][1]["amount"] == payout
        _assert_money_reconciles(after)
        wallets = client.get("/wallets").json()
        assert Decimal(str(wallets[0]["current_balance"])) == after["wallet"]["current_balance"]

    def test_http_settlement_override_rounds_payout_before_profit(self, client: TestClient):
        wid = _seed_wallet(Decimal("1.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "0.01",
            "odds": "2", "tax_rate": "0",
        })
        assert response.status_code == 201, response.text
        bet_id = response.json()["id"]
        payout = _cents(Decimal("0.01") * Decimal("1.5"))
        response = client.post(f"/bets/{bet_id}/settle", json={
            "result": "won", "settlement_odds": "1.5",
        })
        assert response.status_code == 200, response.text
        assert Decimal(str(response.json()["profit"])) == payout - Decimal("0.01")
        state = _wallet_state(wid)
        assert state["bets"][0]["taken_odds"] == Decimal("2")
        assert state["ledger"][1]["amount"] == payout
        assert state["wallet"]["current_balance"] == Decimal("1.01")
        _assert_money_reconciles(state)

    @pytest.mark.parametrize("field,value", [
        pytest.param("odds", "1.00001", id="odds-rounds-to-one"),
        pytest.param("odds", "1.000049999999999999999999999", id="odds-below-half-up"),
        pytest.param("odds", "9999.99995", id="odds-rounds-beyond-numeric-eight-four"),
        pytest.param("odds", "10000", id="odds-outside-numeric-eight-four"),
        pytest.param("tax_rate", "0.995", id="tax-rounds-to-one"),
        pytest.param("tax_rate", "0.99999", id="tax-rounds-to-one-from-below"),
    ])
    def test_http_nonstorable_rounded_terms_leave_no_writes(
        self, client: TestClient, field: str, value: str
    ):
        wid = _seed_wallet(Decimal("100.00"))
        before = _wallet_state(wid)
        payload = {
            "bookmaker_account_id": wid, "side": "a", "stake": "10",
            "odds": "2", "tax_rate": "0",
        }
        payload[field] = value
        response = client.post("/bets", json=payload)
        assert response.status_code == 400, response.text
        assert _wallet_state(wid) == before

    @pytest.mark.parametrize("override", [False, True], ids=["taken", "won-override"])
    def test_http_odds_half_up_precision_is_used_before_payout(
        self, client: TestClient, override: bool
    ):
        wid = _seed_wallet(Decimal("2000.00"))
        raw_odds = Decimal("1.00005")
        stored_odds = raw_odds.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "1000",
            "odds": "2" if override else str(raw_odds), "tax_rate": "0",
        })
        assert response.status_code == 201, response.text
        bet_id = response.json()["id"]
        before = _wallet_state(wid)
        assert before["bets"][0]["taken_odds"] == (Decimal("2") if override else stored_odds)
        payload = {"result": "won"}
        if override:
            payload["settlement_odds"] = str(raw_odds)
        response = client.post(f"/bets/{bet_id}/settle", json=payload)
        assert response.status_code == 200, response.text
        payout = _cents(Decimal("1000") * stored_odds)
        assert payout == Decimal("1000.10")
        after = _wallet_state(wid)
        assert after["bets"][0]["taken_odds"] == before["bets"][0]["taken_odds"]
        assert after["bets"][0]["profit"] == Decimal("0.10")
        assert after["ledger"][1]["amount"] == payout
        assert after["wallet"]["current_balance"] == Decimal("2000.10")
        _assert_money_reconciles(after)

    @pytest.mark.parametrize("odds", ["1.00001", "9999.99995", "10000"])
    def test_http_nonstorable_won_override_leaves_every_write_unchanged(
        self, client: TestClient, odds: str
    ):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "10", "odds": "2",
        })
        assert response.status_code == 201, response.text
        before = _wallet_state(wid)
        response = client.post(f"/bets/{response.json()['id']}/settle", json={
            "result": "won", "settlement_odds": odds,
        })
        assert response.status_code == 400, response.text
        assert _wallet_state(wid) == before

    @pytest.mark.parametrize("result", ["lost", "void", "cancelled"])
    @pytest.mark.parametrize("override", [False, True], ids=["stored-only", "unused-override"])
    def test_http_legacy_odds_one_do_not_block_nonwinning_settlement(
        self, client: TestClient, result: str, override: bool
    ):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "10", "odds": "2",
        })
        assert response.status_code == 201, response.text
        bet_id = response.json()["id"]
        # Recreate an already-persisted legacy bet without relaxing placement rules.
        with get_session() as session:
            session.execute(
                text("UPDATE bets SET taken_odds=:odds WHERE id=:id"),
                {"odds": Decimal("1.0000"), "id": bet_id},
            )
            session.commit()
        payload = {"result": result}
        if override:
            payload["settlement_odds"] = "1.00001"
        response = client.post(f"/bets/{bet_id}/settle", json=payload)
        assert response.status_code == 200, response.text
        after = _wallet_state(wid)
        payout = Decimal("0.00") if result == "lost" else Decimal("10.00")
        assert after["bets"][0]["taken_odds"] == Decimal("1.0000")
        assert after["bets"][0]["status"] == result
        assert after["bets"][0]["profit"] == payout - Decimal("10.00")
        assert after["wallet"]["current_balance"] == Decimal("90.00") + payout
        assert len(after["ledger"]) == (1 if result == "lost" else 2)
        _assert_money_reconciles(after)
        repeated = client.post(f"/bets/{bet_id}/settle", json=payload)
        assert repeated.status_code == 400, repeated.text
        assert _wallet_state(wid) == after

    def test_http_tax_below_rounding_boundary_preserves_stored_policy(self, client: TestClient):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "10",
            "odds": "2", "tax_rate": "0.994999999999999999999999999",
        })
        assert response.status_code == 201, response.text
        assert _wallet_state(wid)["bets"][0]["tax_rate"] == Decimal("0.99")
        response = client.post(f"/bets/{response.json()['id']}/settle", json={"result": "won"})
        assert response.status_code == 200, response.text
        after = _wallet_state(wid)
        assert after["ledger"][1]["amount"] == Decimal("0.20")
        assert after["wallet"]["current_balance"] == Decimal("90.20")
        _assert_money_reconciles(after)

    @pytest.mark.parametrize(
        "field,value",
        [
            pytest.param(field, value, id=f"{field}-{label}")
            for field, cases in [
                ("stake", [("zero", 0), ("negative", -1), ("nan", "NaN"),
                           ("positive-infinity", "Infinity"), ("negative-infinity", "-Infinity")]),
                ("odds", [("one", 1), ("below-one", "0.99"), ("nan", "NaN"),
                          ("positive-infinity", "Infinity"), ("negative-infinity", "-Infinity")]),
                ("tax_rate", [("negative", "-0.01"), ("one", 1), ("above-one", "1.01"),
                              ("nan", "NaN"), ("positive-infinity", "Infinity"),
                              ("negative-infinity", "-Infinity")]),
            ]
            for label, value in cases
        ],
    )
    def test_http_invalid_money_is_rejected_without_writes(self, client: TestClient, field, value):
        wid = _seed_wallet(Decimal("100.00"))
        before = _wallet_state(wid)
        payload = {"bookmaker_account_id": wid, "side": "a", "stake": 1, "odds": 2, "tax_rate": 0.12}
        payload[field] = value
        # Strings encode nonfinite values legally in JSON and exercise HTTP coercion.
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            response = failing_client.post("/bets", content=json.dumps(payload), headers={"Content-Type": "application/json"})
        finally:
            failing_client.close()
        assert _wallet_state(wid) == before
        assert response.status_code == 422, response.text

    @pytest.mark.parametrize("odds", [
        pytest.param("1", id="one"),
        pytest.param("0.99", id="below-one"),
        pytest.param("NaN", id="nan"),
        pytest.param("Infinity", id="positive-infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ])
    def test_http_invalid_settlement_odds_leave_bet_open(self, client: TestClient, odds: str):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": 1, "odds": 2,
        })
        assert response.status_code == 201, response.text
        before = _wallet_state(wid)
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            response = failing_client.post(f"/bets/{response.json()['id']}/settle", json={
                "result": "won", "settlement_odds": odds,
            })
        finally:
            failing_client.close()
        assert _wallet_state(wid) == before
        assert response.status_code == 422, response.text

    @pytest.mark.parametrize("tax", [
        pytest.param("0", id="zero-tax"), pytest.param("0.99", id="below-one-tax"),
    ])
    def test_http_finite_tax_boundaries_remain_valid(self, client: TestClient, tax: str):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": 1,
            "odds": 2, "tax_rate": tax,
        })
        assert response.status_code == 201, response.text
        state = _wallet_state(wid)
        assert state["bets"][0]["tax_rate"] == Decimal(tax)
        _assert_money_reconciles(state)

    @pytest.mark.parametrize("opening", [
        pytest.param("NaN", id="nan"), pytest.param("Infinity", id="positive-infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ])
    def test_http_nonfinite_opening_balance_creates_no_wallet(self, client: TestClient, opening: str):
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            response = failing_client.post("/wallets", params={
                "bookmaker_id": 1, "account_name": "nonfinite", "opening_balance": opening,
            })
        finally:
            failing_client.close()
        with get_session() as session:
            assert session.execute(text("SELECT count(*) FROM bookmaker_accounts")).scalar_one() == 0
        assert response.status_code in {400, 422}, response.text

    @pytest.mark.parametrize("opening", ["0", "10.005", "10.004999999999999999999999999"])
    def test_http_opening_balance_uses_exact_half_up_cents(self, client: TestClient, opening: str):
        response = client.post("/wallets", params={
            "bookmaker_id": 1, "account_name": "cent-opening", "opening_balance": opening,
        })
        assert response.status_code == 201, response.text
        expected = _cents(opening)
        assert isinstance(response.json()["current_balance"], (int, float))
        assert Decimal(str(response.json()["current_balance"])) == expected
        state = _wallet_state(response.json()["id"])
        assert state["wallet"]["opening_balance"] == expected
        assert state["wallet"]["current_balance"] == expected
        assert state["bets"] == state["ledger"] == []

    @pytest.mark.parametrize("stake", ["0.004", "0.004999999999999999999999999"])
    def test_http_stake_rounding_to_zero_leaves_no_writes(self, client: TestClient, stake: str):
        wid = _seed_wallet(Decimal("1.00"))
        before = _wallet_state(wid)
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": stake, "odds": "2",
        })
        assert response.status_code == 400, response.text
        assert _wallet_state(wid) == before

    @pytest.mark.parametrize("field", ["model_prob", "ev"])
    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    def test_http_nonfinite_manual_metadata_leaves_no_writes(
        self, client: TestClient, field: str, value: str
    ):
        wid = _seed_wallet(Decimal("1.00"))
        before = _wallet_state(wid)
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": "0.01", "odds": "2",
            field: value,
        })
        assert response.status_code == 422, response.text
        assert _wallet_state(wid) == before

    @pytest.mark.parametrize("field", ["stake", "odds", "tax_rate", "ev"])
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_http_nonfinite_numeric_literals_return_validation_error_without_writes(
        self, client: TestClient, field: str, literal: str
    ):
        wid = _seed_wallet(Decimal("1.00"))
        before = _wallet_state(wid)
        payload = {
            "bookmaker_account_id": wid, "side": "a", "stake": 1, "odds": 2,
            "tax_rate": 0.12, field: float(literal),
        }
        # Python's JSON decoder accepts these nonstandard numeric literals. Their
        # validation error must itself be serializable rather than raising HTTP500.
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            response = failing_client.post(
                "/bets", content=json.dumps(payload),
                headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
            )
        finally:
            failing_client.close()
        assert _wallet_state(wid) == before
        assert response.status_code == 422, response.text
        detail = json.loads(
            response.text,
            parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"),
        )["detail"]
        error = next(error for error in detail if error["loc"] == ["body", field])
        assert error["type"] == "finite_number"
        assert error["msg"] == "Input should be a finite number"
        assert error["input"] == literal
        assert response.headers["content-type"] == "application/json"
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert response.headers["access-control-allow-credentials"] == "true"

    def test_simultaneous_http_placements_cannot_overdraw(self, client: TestClient):
        wid = _seed_wallet(Decimal("100.00"))
        responses = _concurrent_posts(client, "bookmaker_accounts", wid, "/bets", {
            "bookmaker_account_id": wid, "side": "a", "stake": 60, "odds": 2,
        })
        assert sorted(response.status_code for response in responses) == [201, 400]
        state = _wallet_state(wid)
        assert state["wallet"]["current_balance"] == Decimal("40.00")
        assert len(state["bets"]) == 1
        assert len(state["ledger"]) == 1
        _assert_money_reconciles(state)

    def test_simultaneous_http_settlements_credit_once(self, client: TestClient):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": 10, "odds": 2,
        })
        assert response.status_code == 201, response.text
        bet_id = response.json()["id"]
        responses = _concurrent_posts(client, "bets", bet_id, f"/bets/{bet_id}/settle", {"result": "won"})
        assert sorted(response.status_code for response in responses) == [200, 400]
        state = _wallet_state(wid)
        assert state["wallet"]["current_balance"] == Decimal("107.60")
        assert state["bets"][0]["profit"] == Decimal("7.60")
        assert state["bets"][0]["status"] == "won"
        assert len(state["ledger"]) == 2
        assert state["ledger"][1]["amount"] == Decimal("17.60")
        _assert_money_reconciles(state)

    def test_http_placement_ledger_failure_rolls_back_every_write(self, client: TestClient):
        wid = _seed_wallet(Decimal("100.00"))
        before = _wallet_state(wid)
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            with _reject_ledger_insert(wid, "bet_placed"):
                response = failing_client.post("/bets", json={
                    "bookmaker_account_id": wid, "side": "a", "stake": 10, "odds": 2,
                })
                assert response.status_code == 500, response.text
                assert _wallet_state(wid) == before
        finally:
            failing_client.close()
        # The same real interface must recover once the failure injection is removed.
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": 10, "odds": 2,
        })
        assert response.status_code == 201, response.text
        state = _wallet_state(wid)
        assert len(state["bets"]) == len(state["ledger"]) == 1
        _assert_money_reconciles(state)

    def test_http_settlement_ledger_failure_rolls_back_every_write(self, client: TestClient):
        wid = _seed_wallet(Decimal("100.00"))
        response = client.post("/bets", json={
            "bookmaker_account_id": wid, "side": "a", "stake": 10, "odds": 2,
        })
        assert response.status_code == 201, response.text
        bet_id = response.json()["id"]
        before = _wallet_state(wid)
        failing_client = TestClient(client.app, raise_server_exceptions=False)
        try:
            with _reject_ledger_insert(wid, "settled_won"):
                response = failing_client.post(f"/bets/{bet_id}/settle", json={"result": "won"})
                assert response.status_code == 500, response.text
                assert _wallet_state(wid) == before
        finally:
            failing_client.close()
        response = client.post(f"/bets/{bet_id}/settle", json={"result": "won"})
        assert response.status_code == 200, response.text
        state = _wallet_state(wid)
        assert state["wallet"]["current_balance"] == Decimal("107.60")
        assert len(state["ledger"]) == 2
        _assert_money_reconciles(state)
