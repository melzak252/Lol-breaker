"""Router: /api/wallets, /api/bets."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from betting_app.api.deps import get_db, query_df, query_one
from betting_app.api.schemas import BetCreate, BetResponse, BetSettle, WalletResponse
from betting_app.services.wallet_service import (
    BetAlreadySettledError,
    BetNotFoundError,
    InsufficientWalletBalanceError,
    WalletNotFoundError,
    money_cents,
    place_wallet_bet,
    settle_wallet_bet_atomic,
)

router = APIRouter(tags=["wallets"])


# ── GET /api/wallets ────────────────────────────────────────────────────────


@router.get("/wallets", response_model=list[WalletResponse])
def list_wallets(db=Depends(get_db)):
    rows = query_df(
        db,
        """
        SELECT ba.*, b.name AS bookmaker_name
        FROM bookmaker_accounts ba
        LEFT JOIN bookmakers b ON b.id=ba.bookmaker_id
        WHERE ba.is_active=1
        ORDER BY ba.account_name
        """,
    )
    return [
        WalletResponse(
            id=r["id"],
            bookmaker=r.get("bookmaker_name"),
            account_name=r["account_name"],
            currency=r.get("currency", "PLN"),
            current_balance=float(r.get("current_balance", 0)),
            is_active=bool(r["is_active"]),
        )
        for r in rows
    ]


@router.post("/wallets", response_model=WalletResponse, status_code=201)
def create_wallet(
    bookmaker_id: int | None = None,
    account_name: str = "Default",
    opening_balance: Decimal = Decimal("100.00"),
    db=Depends(get_db),
):
    if not opening_balance.is_finite():
        raise HTTPException(status_code=400, detail="Opening balance must be finite")
    if opening_balance < 0:
        raise HTTPException(status_code=400, detail="Opening balance cannot be negative")
    try:
        opening_balance = money_cents(opening_balance, "Opening balance")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        row = db.execute(
            text("""
            INSERT INTO bookmaker_accounts (bookmaker_id, account_name, currency, opening_balance, current_balance, is_active, created_at, updated_at)
            VALUES (:bid, :name, 'PLN', :bal, :bal, 1, :now, :now)
            RETURNING id
            """),
            {"bid": bookmaker_id, "name": account_name, "bal": opening_balance, "now": now},
        ).mappings().one_or_none()
        if not row:
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create wallet")
        wallet_id = row["id"]
        db.commit()
        # fetch the inserted row by exact id
        r = query_one(
            db,
            """
            SELECT ba.*, b.name AS bookmaker_name
            FROM bookmaker_accounts ba
            LEFT JOIN bookmakers b ON b.id=ba.bookmaker_id
            WHERE ba.id = :id
            """,
            {"id": wallet_id},
        )
        if not r:
            raise HTTPException(status_code=500, detail="Failed to retrieve created wallet")
        return WalletResponse(
            id=r["id"],
            bookmaker=r.get("bookmaker_name"),
            account_name=r["account_name"],
            currency=r.get("currency", "PLN"),
            current_balance=float(r.get("current_balance", 0)),
            is_active=True,
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Wallet already exists")
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise


# ── GET /api/bets ───────────────────────────────────────────────────────────


@router.get("/bets", response_model=list[BetResponse])
def list_bets(status: str | None = None, limit: int = 50, db=Depends(get_db)):
    where = ""
    params: dict = {"lim": limit}
    if status:
        where = "WHERE b.status=:status"
        params["status"] = status
    rows = query_df(
        db,
        f"""
        SELECT b.*
        FROM bets b
        {where}
        ORDER BY b.placed_at DESC
        LIMIT :lim
        """,
        params,
    )
    return [
        BetResponse(
            id=r["id"],
            bookmaker_account_id=r["bookmaker_account_id"],
            canonical_match_id=r.get("canonical_match_id"),
            team_a=r.get("team_a"),
            team_b=r.get("team_b"),
            stake=float(r["stake"]),
            odds=float(r["taken_odds"]),
            side=r["side"],
            status=r["status"],
            profit=float(r["profit"]) if r.get("profit") is not None else None,
            placed_at=r.get("placed_at"),
            settled_at=r.get("settled_at"),
            note=r.get("note"),
        )
        for r in rows
    ]

@router.post("/bets", response_model=BetResponse, status_code=201)
def place_bet(body: BetCreate, db=Depends(get_db)):
    try:
        bet_id = place_wallet_bet(
            db,
            bookmaker_account_id=body.bookmaker_account_id,
            canonical_match_id=body.canonical_match_id,
            team_a=body.team_a,
            team_b=body.team_b,
            league=body.league,
            match_start_time=body.match_start_time,
            side=body.side,
            stake=body.stake,
            odds=body.odds,
            model_prob=body.model_prob,
            ev=body.ev,
            tax_rate=body.tax_rate,
            note=body.note,
        )
        db.commit()
    except WalletNotFoundError:
        db.rollback()
        raise HTTPException(status_code=404, detail="Wallet not found")
    except InsufficientWalletBalanceError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Insufficient balance")
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        db.rollback()
        raise
    row = query_one(db, "SELECT * FROM bets WHERE id=:id", {"id": bet_id})
    if not row:
        raise HTTPException(status_code=500, detail="Bet placed but could not be retrieved")
    return _bet_from_row(row)


create_bet = place_bet


@router.post("/bets/{bet_id}/settle", response_model=BetResponse)
def settle_bet(bet_id: int, body: BetSettle, db=Depends(get_db)):
    try:
        settle_wallet_bet_atomic(
            db,
            bet_id=bet_id,
            result=body.result,
            settlement_odds=body.settlement_odds,
        )
        db.commit()
    except BetNotFoundError:
        db.rollback()
        raise HTTPException(status_code=404, detail="Bet not found")
    except BetAlreadySettledError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Bet already settled")
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        db.rollback()
        raise
    row = query_one(db, "SELECT * FROM bets WHERE id=:id", {"id": bet_id})
    if not row:
        raise HTTPException(status_code=500, detail="Settled bet could not be retrieved")
    return _bet_from_row(row)


def _bet_from_row(r: dict) -> BetResponse:
    return BetResponse(
        id=r["id"],
        bookmaker_account_id=r["bookmaker_account_id"],
        canonical_match_id=r.get("canonical_match_id"),
        team_a=r.get("team_a"),
        team_b=r.get("team_b"),
        stake=float(r["stake"]),
        odds=float(r["taken_odds"]),
        side=r["side"],
        status=r["status"],
        profit=float(r["profit"]) if r.get("profit") is not None else None,
        placed_at=r.get("placed_at"),
        settled_at=r.get("settled_at"),
        note=r.get("note"),
    )
