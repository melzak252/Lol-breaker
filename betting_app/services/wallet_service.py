"""Bookmaker account wallets and manual bet history."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session

from betting_app.core.db import query_df


class WalletNotFoundError(ValueError):
    """The requested wallet does not exist."""


class InsufficientWalletBalanceError(ValueError):
    """The wallet cannot cover the requested stake."""


class BetNotFoundError(ValueError):
    """The requested bet does not exist."""


class BetAlreadySettledError(ValueError):
    """A settlement was already committed for this bet."""


def _finite_decimal(value: Decimal | float | str, field: str) -> Decimal:
    """Keep persisted Numeric values exact; convert other inputs without float math."""
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not amount.is_finite():
        raise ValueError(f"{field} must be a finite number")
    return amount


def money_cents(value: Decimal | float | str, field: str) -> Decimal:
    """Round wallet money to cents with ROUND_HALF_UP before any balance change."""
    amount = _finite_decimal(value, field)
    try:
        return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"{field} cannot be represented as cents") from exc


def _stored_odds(value: Decimal | float | str, field: str) -> Decimal:
    """Normalize to Numeric(8, 4) before validating or calculating winnings."""
    amount = _finite_decimal(value, field)
    try:
        amount = amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"{field} cannot be represented with four decimal places") from exc
    if amount <= 1:
        raise ValueError(f"{field} must be greater than 1.0 after rounding to four decimal places")
    if amount >= Decimal("10000"):
        raise ValueError(f"{field} cannot be represented as Numeric(8, 4)")
    return amount

def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def bookmaker_options() -> pd.DataFrame:
    return query_df("SELECT id, name FROM bookmakers WHERE is_active = 1 ORDER BY name")


def accounts(active_only: bool = True) -> pd.DataFrame:
    where = "WHERE ba.is_active = 1" if active_only else ""
    return query_df(
        f"""
        SELECT ba.*, b.name AS bookmaker
        FROM bookmaker_accounts ba
        JOIN bookmakers b ON b.id = ba.bookmaker_id
        {where}
        ORDER BY b.name, ba.account_name
        """
    )




def wallet_transactions(account_id: int | None = None, limit: int = 200) -> pd.DataFrame:
    if account_id:
        return query_df(
            """
            SELECT wt.*, ba.account_name, b.name AS bookmaker
            FROM bookmaker_wallet_transactions wt
            JOIN bookmaker_accounts ba ON ba.id = wt.bookmaker_account_id
            JOIN bookmakers b ON b.id = ba.bookmaker_id
            WHERE wt.bookmaker_account_id = ?
            ORDER BY wt.transaction_time DESC, wt.id DESC
            LIMIT ?
            """,
            (int(account_id), int(limit)),
        )
    return query_df(
        """
        SELECT wt.*, ba.account_name, b.name AS bookmaker
        FROM bookmaker_wallet_transactions wt
        JOIN bookmaker_accounts ba ON ba.id = wt.bookmaker_account_id
        JOIN bookmakers b ON b.id = ba.bookmaker_id
        ORDER BY wt.transaction_time DESC, wt.id DESC
        LIMIT ?
        """,
        (int(limit),),
    )


def latest_model_ev_signals(limit: int = 200, min_ev: float | None = None) -> pd.DataFrame:
    from betting_app.core.models.registry import get_active_hybrid

    hybrid = get_active_hybrid()
    where = (
        "WHERE mes.status = 'new' AND cp.prediction_status = 'active'"
        " AND cp.model_name = ? AND cp.model_version = ?"
    )
    params: list[str | float | int] = [hybrid.hybrid_model_name, hybrid.hybrid_model_version]
    if min_ev is not None:
        where += " AND mes.ev >= ?"
        params.append(float(min_ev))
    params.append(int(limit))
    return query_df(
        f"""
        SELECT mes.id, mes.side, mes.odds, mes.model_prob, mes.market_prob, mes.ev, mes.tax_rate,
               mes.stake_suggestion, b.name AS bookmaker,
               os.bookmaker_id, os.offer_url, os.scraped_at,
               cm.id AS canonical_match_id, cm.team_a_name, cm.team_b_name,
               cm.league, cm.start_time_normalized,
               cp.model_name, cp.model_version
        FROM model_ev_signals mes
        JOIN odds_snapshots os ON os.id = mes.odds_snapshot_id
        JOIN bookmakers b ON b.id = mes.bookmaker_id
        JOIN canonical_matches cm ON cm.id = mes.canonical_match_id
        JOIN canonical_predictions cp ON cp.id = mes.canonical_prediction_id
        {where}
        ORDER BY mes.ev DESC, mes.id DESC
        LIMIT ?
        """,
        tuple(params),
    )




def tracked_bets() -> pd.DataFrame:
    return query_df(
        """
        SELECT bets.*, b.name AS bookmaker, ba.account_name, ba.currency,
               cm.team_a_name AS canonical_team_a, cm.team_b_name AS canonical_team_b,
               cm.start_time_normalized AS canonical_start_time
        FROM bets
        LEFT JOIN bookmakers b ON b.id = bets.bookmaker_id
        LEFT JOIN bookmaker_accounts ba ON ba.id = bets.bookmaker_account_id
        LEFT JOIN canonical_matches cm ON cm.id = bets.canonical_match_id
        ORDER BY bets.placed_at DESC, bets.id DESC
        """
    )



def place_wallet_bet(
    db: Session,
    *,
    bookmaker_account_id: int,
    canonical_match_id: int | None,
    team_a: str | None,
    team_b: str | None,
    league: str | None,
    match_start_time: str | None,
    side: str,
    stake: Decimal | float,
    odds: Decimal | float,
    model_prob: float | None,
    ev: float | None,
    tax_rate: Decimal | float,
    note: str | None,
) -> int:
    """Atomically reserve a stake, persist its bet, and audit the debit.

    The conditional update is the concurrency boundary.  A concurrent placement
    observes the post-debit balance and cannot overdraw the same wallet.

    Stake and payout/refund credits use Decimal ROUND_HALF_UP cents. Odds use
    Numeric(8, 4) precision and tax uses Numeric(4, 2) precision before validation
    and persistence. Settlement derives profit only after rounding the payout,
    using stored terms or a precision-normalized winning odds override.
    The caller commits the bet, wallet balance and ledger together.
    """
    stake = money_cents(stake, "Stake")
    if stake <= 0:
        raise ValueError("Stake must be greater than zero after rounding to cents")
    odds = _stored_odds(odds, "Odds")
    tax_rate = _finite_decimal(tax_rate, "Tax rate")
    if not 0 <= tax_rate < 1:
        raise ValueError("Tax rate must be at least zero and less than one")
    tax_rate = money_cents(tax_rate, "Tax rate")
    if tax_rate >= 1:
        raise ValueError("Tax rate must be less than one after rounding to two decimal places")
    if model_prob is not None:
        model_prob = _finite_decimal(model_prob, "Model probability")
    if ev is not None:
        ev = _finite_decimal(ev, "EV")
    now = utc_now_iso()
    balance_after = db.execute(
        text(
            """
            UPDATE bookmaker_accounts
            SET current_balance = current_balance - :stake,
                updated_at = :now
            WHERE id = :wallet_id
              AND is_active = 1
              AND current_balance >= :stake
            RETURNING current_balance
            """
        ),
        {
            "wallet_id": bookmaker_account_id,
            "stake": stake,
            "now": now,
        },
    ).scalar_one_or_none()
    if balance_after is None:
        wallet_exists = db.execute(
            text("SELECT 1 FROM bookmaker_accounts WHERE id = :wallet_id"),
            {"wallet_id": bookmaker_account_id},
        ).scalar_one_or_none()
        if wallet_exists is None:
            raise WalletNotFoundError(bookmaker_account_id)
        raise InsufficientWalletBalanceError(bookmaker_account_id)
    bet_id = db.execute(
        text(
            """
            INSERT INTO bets (
                bookmaker_account_id, canonical_match_id, team_a, team_b,
                league, match_start_time, side, stake, taken_odds,
                model_prob, ev, tax_rate, note, status, placed_at, profit
            ) VALUES (
                :bookmaker_account_id, :canonical_match_id, :team_a, :team_b,
                :league, :match_start_time, :side, :stake, :odds,
                :model_prob, :ev, :tax_rate, :note, 'open', :now, 0
            )
            RETURNING id
            """
        ),
        {
            "bookmaker_account_id": bookmaker_account_id,
            "canonical_match_id": canonical_match_id,
            "team_a": team_a,
            "team_b": team_b,
            "league": league,
            "match_start_time": match_start_time,
            "side": side,
            "stake": stake,
            "odds": odds,
            "model_prob": model_prob,
            "ev": ev,
            "tax_rate": tax_rate,
            "note": note,
            "now": now,
        },
    ).scalar_one()
    db.execute(
        text(
            """
            INSERT INTO bookmaker_wallet_transactions (
                bookmaker_account_id, bet_id, transaction_time,
                transaction_type, amount, balance_after, note
            ) VALUES (
                :wallet_id, :bet_id, :now, 'bet_placed',
                :amount, :balance_after, :note
            )
            """
        ),
        {
            "wallet_id": bookmaker_account_id,
            "bet_id": bet_id,
            "now": now,
            "amount": -stake,
            "balance_after": balance_after,
            "note": note,
        },
    )
    return int(bet_id)


def settle_wallet_bet_atomic(
    db: Session,
    *,
    bet_id: int,
    result: str,
    settlement_odds: Decimal | float | None,
) -> None:
    """Settle once; only winnings need odds/tax, refunds credit the stored stake."""
    bet = db.execute(
        text("SELECT * FROM bets WHERE id = :id"),
        {"id": bet_id},
    ).mappings().one_or_none()
    if bet is None:
        raise BetNotFoundError(bet_id)
    if bet["status"] != "open":
        raise BetAlreadySettledError(bet_id)
    if result not in {"won", "lost", "void", "cancelled"}:
        raise ValueError(f"Invalid settlement result: {result}")
    stake = money_cents(bet["stake"], "Stored stake")
    if result == "won":
        odds = _stored_odds(
            bet["taken_odds"] if settlement_odds is None else settlement_odds,
            "Settlement odds",
        )
        tax_rate = _finite_decimal(bet["tax_rate"], "Stored tax rate")
        payout = money_cents(stake * odds * (Decimal("1") - tax_rate), "Payout")
    elif result in {"void", "cancelled"}:
        payout = stake
    else:
        payout = Decimal("0.00")
    profit = payout - stake
    now = utc_now_iso()
    updated = db.execute(
        text(
            """
            UPDATE bets
            SET status = :result, result = :result, profit = :profit,
                settled_at = :now
            WHERE id = :id AND status = 'open'
            """
        ),
        {
            "result": result,
            "profit": profit,
            "now": now,
            "id": bet_id,
        },
    )
    if updated.rowcount != 1:
        raise BetAlreadySettledError(bet_id)
    if not payout or not bet.get("bookmaker_account_id"):
        return
    balance_after = db.execute(
        text(
            """
            UPDATE bookmaker_accounts
            SET current_balance = current_balance + :payout,
                updated_at = :now
            WHERE id = :wallet_id
            RETURNING current_balance
            """
        ),
        {
            "payout": payout,
            "now": now,
            "wallet_id": bet["bookmaker_account_id"],
        },
    ).scalar_one()
    db.execute(
        text(
            """
            INSERT INTO bookmaker_wallet_transactions (
                bookmaker_account_id, bet_id, transaction_time,
                transaction_type, amount, balance_after, note
            ) VALUES (
                :wallet_id, :bet_id, :now, :kind,
                :payout, :balance_after, NULL
            )
            """
        ),
        {
            "wallet_id": bet["bookmaker_account_id"],
            "bet_id": bet_id,
            "now": now,
            "kind": f"settled_{result}",
            "payout": payout,
            "balance_after": balance_after,
        },
    )