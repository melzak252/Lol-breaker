"""Unit tests for the realistic chronological betting operations simulator and wallet."""
from datetime import UTC, datetime, timedelta
import pandas as pd
import pytest

from src.analysis.historical_production_simulator import (
    HistoricalProductionSimulator,
    OperationalWallet,
)


class TestOperationalWallet:
    def test_initial_state(self):
        wallet = OperationalWallet(initial_capital=5000.0)
        assert wallet.initial_capital == 5000.0
        assert wallet.free_capital == 5000.0
        assert wallet.reserved_capital == 0.0
        assert wallet.total_equity == 5000.0

    def test_reserve_deducts_free_capital(self):
        wallet = OperationalWallet(initial_capital=1000.0)
        now = datetime.now(UTC)
        wallet.reserve("bet_1", amount=200.0, match_id=1, side="a", odds=2.0, placed_at=now, settle_at=now + timedelta(hours=3))
        assert wallet.free_capital == 800.0
        assert wallet.reserved_capital == 200.0
        assert wallet.total_equity == 1000.0

    def test_no_overdraft_constraint(self):
        wallet = OperationalWallet(initial_capital=100.0)
        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="Insufficient free capital"):
            wallet.reserve("bet_fail", amount=150.0, match_id=1, side="a", odds=2.0, placed_at=now, settle_at=now + timedelta(hours=3))

    def test_settle_win_with_tax(self):
        wallet = OperationalWallet(initial_capital=1000.0)
        now = datetime.now(UTC)
        wallet.reserve("b1", amount=100.0, match_id=1, side="a", odds=3.0, placed_at=now, settle_at=now + timedelta(hours=3))
        # Payout after 12% tax: 100 * 3.0 * 0.88 = 264.0
        # Net profit: 264.0 - 100.0 = +164.0
        rec = wallet.settle("b1", won=True, settled_at=now + timedelta(hours=3), tax_rate=0.12)
        assert rec["won"] is True
        assert rec["payout"] == pytest.approx(264.0, abs=1e-6)
        assert rec["profit"] == pytest.approx(164.0, abs=1e-6)
        assert wallet.reserved_capital == 0.0
        assert wallet.free_capital == pytest.approx(1164.0, abs=1e-6)
        assert wallet.total_equity == pytest.approx(1164.0, abs=1e-6)

    def test_settle_loss(self):
        wallet = OperationalWallet(initial_capital=1000.0)
        now = datetime.now(UTC)
        wallet.reserve("b2", amount=100.0, match_id=2, side="b", odds=2.0, placed_at=now, settle_at=now + timedelta(hours=3))
        rec = wallet.settle("b2", won=False, settled_at=now + timedelta(hours=3), tax_rate=0.12)
        assert rec["won"] is False
        assert rec["payout"] == 0.0
        assert rec["profit"] == -100.0
        assert wallet.reserved_capital == 0.0
        assert wallet.free_capital == 900.0
        assert wallet.total_equity == 900.0


class TestHistoricalProductionSimulator:
    def test_simulator_basic_run(self):
        sim = HistoricalProductionSimulator(
            initial_bankroll=1000.0,
            policy_kind="unconstrained",
            min_ev=0.05,
            tier1_only=False,
            flat_stake=50.0,
        )
        now = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
        df = pd.DataFrame([
            {
                "snapshot_id": 1,
                "canonical_match_id": 101,
                "scraped_at": now - timedelta(hours=4),
                "start_time_normalized": now.isoformat(),
                "league": "LCK",
                "odds_a": 2.50,
                "odds_b": 1.60,
                "p_shin_a": 0.38,
                "p_model_a": 0.55,  # EV_a = 0.55 * (2.5 * 0.88) - 1.0 = +21%
                "winner_side": "a",
            },
            {
                "snapshot_id": 2,
                "canonical_match_id": 102,
                "scraped_at": now - timedelta(hours=4),
                "start_time_normalized": now.isoformat(),
                "league": "LCK",
                "odds_a": 1.50,
                "odds_b": 2.80,
                "p_shin_a": 0.65,
                "p_model_a": 0.30,  # Side B: 0.70 * (2.8 * 0.88) - 1.0 = +72% -> filtered if max_ev active
                "winner_side": "a",
            }
        ])
        res = sim.run(df)
        assert res["bets"] >= 1
        assert "net_profit" in res
        assert "final_equity" in res

    def test_guard_b_filters_divergence_in_simulator(self):
        sim_strict = HistoricalProductionSimulator(
            initial_bankroll=1000.0,
            guard_b_logit=0.10,  # extremely strict logit gap
            tier1_only=False,
            min_ev=0.05,
        )
        now = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
        df = pd.DataFrame([{
            "snapshot_id": 1,
            "canonical_match_id": 101,
            "scraped_at": now - timedelta(hours=4),
            "start_time_normalized": now.isoformat(),
            "league": "LCK",
            "odds_a": 2.50,
            "odds_b": 1.60,
            "p_shin_a": 0.38,
            "p_model_a": 0.55,  # logit gap: |logit(0.55) - logit(0.38)| = |0.20 - (-0.49)| = 0.69 > 0.10
            "winner_side": "a",
        }])
        res = sim_strict.run(df)
        assert res["bets"] == 0
