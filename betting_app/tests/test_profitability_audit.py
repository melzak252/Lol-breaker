"""Tests for model profitability audit and calibration anomaly detection."""

from datetime import UTC, datetime, timedelta
import json
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from betting_app.api.routers.timing import model_profitability_audit


@pytest.fixture
def test_db():
    """Create a temporary SQLite database with minimal schema and sample matches/odds/predictions."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE canonical_matches (
                id INTEGER PRIMARY KEY,
                team_a_name TEXT,
                team_b_name TEXT,
                start_time_normalized TEXT,
                winner_side TEXT,
                status TEXT,
                best_of INTEGER,
                league TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE canonical_predictions (
                id INTEGER PRIMARY KEY,
                canonical_match_id INTEGER,
                model_name TEXT,
                model_version TEXT,
                prob_a REAL,
                prob_b REAL,
                predicted_at TEXT,
                prediction_status TEXT,
                diagnostics_json TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE bookmakers (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE odds_snapshots (
                id INTEGER PRIMARY KEY,
                canonical_match_id INTEGER,
                bookmaker_id INTEGER,
                odds_a REAL,
                odds_b REAL,
                scraped_at TEXT,
                market_type TEXT,
                is_live INTEGER
            )
        """))
        conn.commit()

        # Insert bookmaker
        conn.execute(text("INSERT INTO bookmakers (id, name) VALUES (1, 'Superbet')"))

        # Match 1: Golden underdog (odds 3.00, winner team_a)
        t_start1 = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        t_quote_entry1 = (datetime.now(UTC) - timedelta(days=2, hours=10)).isoformat()
        t_quote_close1 = (datetime.now(UTC) - timedelta(days=2, hours=1)).isoformat()

        conn.execute(text("""
            INSERT INTO canonical_matches (id, team_a_name, team_b_name, start_time_normalized, winner_side, status, best_of, league)
            VALUES (1, 'Karmine Corp', 'G2 Esports', :st, 'team_a', 'finished', 1, 'LEC')
        """), {"st": t_start1})

        conn.execute(text("""
            INSERT INTO canonical_predictions (id, canonical_match_id, model_name, model_version, prob_a, prob_b, predicted_at, prediction_status, diagnostics_json)
            VALUES (1, 1, 'Operational-PlayerTeamRatings-W20', 'v0.4-binom-series-chronological-v1', 0.48, 0.52, :pa, 'active', :dj)
        """), {"pa": t_quote_entry1, "dj": json.dumps({"player_rating_consensus": 0.47, "team_rating_consensus": 0.49})})

        conn.execute(text("""
            INSERT INTO odds_snapshots (id, canonical_match_id, bookmaker_id, odds_a, odds_b, scraped_at, market_type, is_live)
            VALUES (1, 1, 1, 3.00, 1.40, :sa, 'match_winner', 0)
        """), {"sa": t_quote_entry1})
        conn.execute(text("""
            INSERT INTO odds_snapshots (id, canonical_match_id, bookmaker_id, odds_a, odds_b, scraped_at, market_type, is_live)
            VALUES (2, 1, 1, 2.80, 1.45, :sa, 'match_winner', 0)
        """), {"sa": t_quote_close1})

        # Match 2: Quarantine trap (odds 4.00, winner team_b)
        t_start2 = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        t_quote_entry2 = (datetime.now(UTC) - timedelta(days=1, hours=8)).isoformat()
        t_quote_close2 = (datetime.now(UTC) - timedelta(days=1, hours=1)).isoformat()

        conn.execute(text("""
            INSERT INTO canonical_matches (id, team_a_name, team_b_name, start_time_normalized, winner_side, status, best_of, league)
            VALUES (2, 'Team Heretics', 'Fnatic', :st, 'team_b', 'finished', 3, 'LEC')
        """), {"st": t_start2})

        conn.execute(text("""
            INSERT INTO canonical_predictions (id, canonical_match_id, model_name, model_version, prob_a, prob_b, predicted_at, prediction_status, diagnostics_json)
            VALUES (2, 2, 'Operational-PlayerTeamRatings-W20', 'v0.4-binom-series-chronological-v1', 0.42, 0.58, :pa, 'active', :dj)
        """), {"pa": t_quote_entry2, "dj": json.dumps({"player_rating_consensus": 0.55, "team_rating_consensus": 0.35})})

        conn.execute(text("""
            INSERT INTO odds_snapshots (id, canonical_match_id, bookmaker_id, odds_a, odds_b, scraped_at, market_type, is_live)
            VALUES (3, 2, 1, 4.00, 1.25, :sa, 'match_winner', 0)
        """), {"sa": t_quote_entry2})
        conn.execute(text("""
            INSERT INTO odds_snapshots (id, canonical_match_id, bookmaker_id, odds_a, odds_b, scraped_at, market_type, is_live)
            VALUES (4, 2, 1, 4.20, 1.22, :sa, 'match_winner', 0)
        """), {"sa": t_quote_close2})

        conn.commit()

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_model_profitability_audit_calculation(test_db):
    """Test that model profitability audit correctly categorizes brackets, flags quarantine, and computes impact."""
    res = model_profitability_audit(
        model_key="operational",
        tax_rate=0.12,
        min_ev=0.05,
        max_days_back=30,
        db=test_db,
    )

    assert res["model_key"] == "operational"
    assert "overall" in res
    assert res["overall"]["total_bets"] == 2

    # Check filter impact
    impact = res["filter_impact"]
    assert impact is not None
    assert impact["raw_bets"] == 2
    assert impact["quarantined_bets"] == 1
    assert impact["filtered_bets"] == 1
    # Match 1 won: payout = 3.00 * 0.88 - 1 = +1.64 units
    # Match 2 lost: payout = -1.00 unit
    # Quarantined pnl = -1.00
    # Filtered pnl = +1.64
    assert impact["quarantined_pnl_units"] == -1.00
    assert impact["filtered_pnl_units"] == 1.64
    assert impact["pnl_improvement_units"] == 1.00

    # Check odds brackets
    brackets = {b["bracket_label"]: b for b in res["odds_brackets"]}
    assert any("2.80 - 3.50" in k for k in brackets)
    assert any("3.50 - 5.00" in k for k in brackets)

    golden_bracket = next(v for k, v in brackets.items() if "2.80 - 3.50" in k)
    assert golden_bracket["status"] == "RECOMMENDED"
    assert golden_bracket["n_bets"] == 1
    assert golden_bracket["n_wins"] == 1

    quarantine_bracket = next(v for k, v in brackets.items() if "3.50 - 5.00" in k)
    assert quarantine_bracket["status"] == "NORMAL"
    assert quarantine_bracket["n_bets"] == 1
    assert quarantine_bracket["n_wins"] == 0

    # Check anomalies
    assert len(res["anomalies"]) > 0
    # Match 2 had IDI = |0.55 - 0.35| = 0.20 (20%)
    anomaly_m2 = next((a for a in res["anomalies"] if a["match_id"] == 2), None)
    assert anomaly_m2 is not None
    assert anomaly_m2["idi_pct"] == 20.0
    assert any("ISSUE-001" in r for r in anomaly_m2["reasons"])

    # Check operational rules
    assert len(res["operational_rules"]) >= 4
