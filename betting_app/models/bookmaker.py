"""Bookmaker, wallet and bet models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text as sa_text,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class Bookmaker(Base):
    __tablename__ = "bookmakers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(500))
    is_active: Mapped[int] = mapped_column(Integer, server_default="1")


class BookmakerAccount(Base):
    __tablename__ = "bookmaker_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False)
    account_name: Mapped[str] = mapped_column(String(100), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), server_default="PLN")
    opening_balance: Mapped[Decimal] = mapped_column(Numeric(12, 2), server_default="0")
    current_balance: Mapped[Decimal] = mapped_column(Numeric(12, 2), server_default="0")
    is_active: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"), onupdate=sa_text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("bookmaker_id", "account_name"),
        CheckConstraint("current_balance >= 0", name="chk_bookmaker_account_balance_non_negative"),
    )


class Bet(Base):
    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(Integer)
    model_ev_signal_id: Mapped[int | None] = mapped_column(Integer)
    bookmaker_account_id: Mapped[int | None] = mapped_column(ForeignKey("bookmaker_accounts.id"))
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"))
    bookmaker_id: Mapped[int | None] = mapped_column(Integer)
    side: Mapped[str] = mapped_column(String(1), CheckConstraint("side IN ('a','b')"))
    stake: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    taken_odds: Mapped[Decimal] = mapped_column(Numeric(8, 4))
    status: Mapped[str] = mapped_column(String(20), server_default="open")
    result: Mapped[str | None] = mapped_column(String(20))
    profit: Mapped[Decimal] = mapped_column(Numeric(12, 2), server_default="0")
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    team_a: Mapped[str | None] = mapped_column(String(200))
    team_b: Mapped[str | None] = mapped_column(String(200))
    league: Mapped[str | None] = mapped_column(String(100))
    match_start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_prob: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    ev: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    tax_rate: Mapped[Decimal] = mapped_column(Numeric(4, 2), server_default="0.12")
    note: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50), server_default="manual")


class WalletTransaction(Base):
    __tablename__ = "bookmaker_wallet_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_account_id: Mapped[int] = mapped_column(ForeignKey("bookmaker_accounts.id"))
    bet_id: Mapped[int | None] = mapped_column(ForeignKey("bets.id"))
    transaction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"))
    transaction_type: Mapped[str] = mapped_column(String(50))
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    balance_after: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    note: Mapped[str | None] = mapped_column(Text)
