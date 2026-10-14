"""Unit tests for the automated shadow forward-ledger (paper-trading) daemon."""
import json
import pytest
from datetime import UTC, datetime, timedelta
import pandas as pd
from scripts.run_shadow_forward_ledger import (
    record_shadow_bets,
    settle_shadow_bets,
    LEDGER_PATH,
)


def test_shadow_record_and_settle(tmp_path, monkeypatch):
    test_ledger = tmp_path / "test_shadow_ledger.jsonl"
    monkeypatch.setattr("scripts.run_shadow_forward_ledger.LEDGER_PATH", test_ledger)

    # Mock engine / database
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            class FakeResult:
                def fetchall(self):
                    return [(101, "a", "finished")]
            return FakeResult()

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)

    # 1. Create a dummy record
    record = {
        "bet_key": "shadow_101_a_sts",
        "recorded_at": now.isoformat(),
        "canonical_match_id": 101,
        "match": "T1 vs Gen.G",
        "league": "LCK",
        "start_time": (now + timedelta(hours=2)).isoformat(),
        "bookmaker": "sts",
        "side": "a",
        "odds_taken": 2.50,
        "p_model": 0.52,
        "p_market_shin": 0.42,
        "logit_gap": 0.40,
        "ev_net": 0.144,
        "stake": 100.0,
        "status": "open",
        "settled_at": None,
        "won": None,
        "payout": None,
        "profit": None,
        "clv": None,
    }

    test_ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(test_ledger, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # 2. Settle the bet
    settled = settle_shadow_bets(engine=FakeEngine(), tax_rate=0.12)
    assert settled == 1

    # 3. Read back
    with open(test_ledger, "r", encoding="utf-8") as f:
        resolved = json.loads(f.read().strip())

    assert resolved["status"] == "settled"
    assert resolved["won"] is True
    # Payout: 100 * 2.50 * 0.88 = 220.0
    assert resolved["payout"] == 220.0
    assert resolved["profit"] == 120.0
