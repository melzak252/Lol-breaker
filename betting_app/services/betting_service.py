"""Services for EV signals, bets and bankroll analytics."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from betting_app.core.config import load_config
from betting_app.core.db import query_df, transaction
from betting_app.core.ev import best_ev_side, fair_market_probabilities
from betting_app.core.staking import fractional_kelly_stake


def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def current_bankroll(default: float | None = None) -> float:
    """Return current bankroll from events or configured default."""

    cfg = load_config()
    fallback = cfg.default_bankroll if default is None else default
    rows = query_df("SELECT bankroll_after FROM bankroll_events ORDER BY event_time DESC, id DESC LIMIT 1")
    if rows.empty:
        return float(fallback)
    return float(rows.iloc[0]["bankroll_after"])


def initialize_bankroll(amount: float, note: str = "Initial bankroll") -> None:
    """Initialize bankroll if there are no events yet."""

    if not query_df("SELECT id FROM bankroll_events LIMIT 1").empty:
        return
    with transaction() as connection:
        connection.execute(
            "INSERT INTO bankroll_events(event_type, amount, bankroll_after, note) VALUES ('deposit', ?, ?, ?)",
            (float(amount), float(amount), note),
        )


def generate_signals(min_ev: float | None = None, tax_rate: float | None = None) -> int:
    """Generate EV+ bet signals for latest odds/predictions."""

    cfg = load_config()
    min_ev = cfg.min_ev if min_ev is None else min_ev
    tax_rate = cfg.tax_rate if tax_rate is None else tax_rate
    rows = query_df(
        """
        WITH latest_odds AS (
            SELECT os.*
            FROM odds_snapshots os
            JOIN (
                SELECT match_id, MAX(scraped_at) AS scraped_at
                FROM odds_snapshots
                GROUP BY match_id
            ) latest ON latest.match_id = os.match_id AND latest.scraped_at = os.scraped_at
        ), latest_predictions AS (
            SELECT p.*
            FROM predictions p
            JOIN (
                SELECT match_id, MAX(predicted_at) AS predicted_at
                FROM predictions
                WHERE status = 'active'
                GROUP BY match_id
            ) latest ON latest.match_id = p.match_id AND latest.predicted_at = p.predicted_at
        )
        SELECT
            lo.id AS odds_snapshot_id,
            lo.match_id,
            lo.odds_a,
            lo.odds_b,
            lp.id AS prediction_id,
            lp.prob_a
        FROM latest_odds lo
        JOIN latest_predictions lp ON lp.match_id = lo.match_id
        """
    )
    created = 0
    bankroll = current_bankroll()
    with transaction() as connection:
        for _, row in rows.iterrows():
            selected = best_ev_side(float(row.prob_a), float(row.odds_a), float(row.odds_b), tax_rate, min_ev)
            if selected is None:
                continue
            existing = connection.execute(
                "SELECT id FROM bet_signals WHERE prediction_id = ? AND odds_snapshot_id = ? AND side = ?",
                (int(row.prediction_id), int(row.odds_snapshot_id), selected["side"]),
            ).fetchone()
            if existing:
                continue
            suggested = fractional_kelly_stake(
                bankroll=bankroll,
                probability=float(selected["model_prob"]),
                decimal_odds=float(selected["odds"]),
                fraction=0.05,
                tax_rate=tax_rate,
            )
            connection.execute(
                """
                INSERT INTO bet_signals(
                    match_id, prediction_id, odds_snapshot_id, side, odds,
                    model_prob, market_prob, ev, tax_rate, suggested_stake
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row.match_id),
                    int(row.prediction_id),
                    int(row.odds_snapshot_id),
                    selected["side"],
                    float(selected["odds"]),
                    float(selected["model_prob"]),
                    float(selected["market_prob"]),
                    float(selected["ev"]),
                    float(tax_rate),
                    suggested,
                ),
            )
            created += 1
    return created


def signals(status: str | None = None) -> pd.DataFrame:
    """Return bet signals with match and bookmaker context."""

    where = "WHERE bs.status = ?" if status else ""
    params = (status,) if status else ()
    return query_df(
        f"""
        SELECT
            bs.*,
            um.raw_team_a, um.raw_team_b, um.canonical_team_a, um.canonical_team_b,
            um.match_start_time, um.league,
            b.name AS bookmaker,
            os.scraped_at, os.odds_a, os.odds_b,
            p.prob_a, p.prob_b, p.model_name, p.model_version
        FROM bet_signals bs
        JOIN upcoming_matches um ON um.id = bs.match_id
        JOIN odds_snapshots os ON os.id = bs.odds_snapshot_id
        JOIN bookmakers b ON b.id = os.bookmaker_id
        JOIN predictions p ON p.id = bs.prediction_id
        {where}
        ORDER BY bs.ev DESC, bs.created_at DESC
        """,
        params,
    )


def place_bet(signal_id: int, stake: float, taken_odds: float | None = None, note: str | None = None) -> int:
    """Record a manually placed bet from a signal."""

    row = query_df(
        """
        SELECT bs.*, os.bookmaker_id
        FROM bet_signals bs
        JOIN odds_snapshots os ON os.id = bs.odds_snapshot_id
        WHERE bs.id = ?
        """,
        (signal_id,),
    )
    if row.empty:
        raise ValueError(f"Signal {signal_id} does not exist")
    signal = row.iloc[0]
    odds = float(taken_odds or signal["odds"])
    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO bets(signal_id, bookmaker_id, side, stake, taken_odds, note)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (signal_id, int(signal["bookmaker_id"]), signal["side"], float(stake), odds, note),
        )
        connection.execute("UPDATE bet_signals SET status = 'placed' WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        if row and "id" in row:
            return int(row["id"])
        return int(cursor.lastrowid or 0)


def settle_bet(bet_id: int, result: str, tax_rate: float | None = None) -> float:
    """Settle a bet as won/lost/void and update bankroll."""

    if result not in {"won", "lost", "void", "cancelled"}:
        raise ValueError("result must be won/lost/void/cancelled")
    cfg = load_config()
    tax_rate = cfg.tax_rate if tax_rate is None else tax_rate
    bet_df = query_df("SELECT * FROM bets WHERE id = ?", (bet_id,))
    if bet_df.empty:
        raise ValueError(f"Bet {bet_id} does not exist")
    bet = bet_df.iloc[0]
    if bet["status"] != "open":
        raise ValueError(f"Bet {bet_id} is already settled ({bet['status']})")
    stake = float(bet["stake"])
    if result == "won":
        profit = stake * (float(bet["taken_odds"]) * (1.0 - tax_rate) - 1.0)
    elif result == "lost":
        profit = -stake
    else:
        profit = 0.0
    bankroll_after = current_bankroll() + profit
    with transaction() as connection:
        updated = connection.execute(
            "UPDATE bets SET status = ?, result = ?, profit = ?, settled_at = ? WHERE id = ? AND status = 'open'",
            (result, result, profit, utc_now_iso(), bet_id),
        )
        if updated.rowcount != 1:
            raise ValueError(f"Bet {bet_id} already settled")
        connection.execute(
            "INSERT INTO bankroll_events(event_type, amount, bankroll_after, bet_id, note) VALUES (?, ?, ?, ?, ?)",
            (f"bet_{result}", profit, bankroll_after, bet_id, f"Settled bet {bet_id}"),
        )
    return profit


def bets() -> pd.DataFrame:
    """Return all tracked bets with signal context."""

    return query_df(
        """
        SELECT bets.*, b.name AS bookmaker, bs.ev, bs.model_prob, um.raw_team_a, um.raw_team_b, um.match_start_time
        FROM bets
        LEFT JOIN bookmakers b ON b.id = bets.bookmaker_id
        LEFT JOIN bet_signals bs ON bs.id = bets.signal_id
        LEFT JOIN upcoming_matches um ON um.id = bs.match_id
        ORDER BY bets.placed_at DESC, bets.id DESC
        """
    )


def bankroll_history() -> pd.DataFrame:
    """Return bankroll event history."""

    return query_df("SELECT * FROM bankroll_events ORDER BY event_time, id")
