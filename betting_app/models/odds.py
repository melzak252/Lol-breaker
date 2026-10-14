"""Odds snapshot and scrape run models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text as sa_text,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False, index=True)
    match_id: Mapped[int | None] = mapped_column(Integer)
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"), index=True)
    market_type: Mapped[str | None] = mapped_column(String(50), server_default="match_winner")
    raw_team_a: Mapped[str | None] = mapped_column(String(200))
    raw_team_b: Mapped[str | None] = mapped_column(String(200))
    odds_a: Mapped[float | None] = mapped_column(Float)
    odds_b: Mapped[float | None] = mapped_column(Float)
    is_live: Mapped[int | None] = mapped_column(Integer)
    scraped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, server_default=sa_text("NOW()"))
    source_url: Mapped[str | None] = mapped_column(String(500))
    offer_url: Mapped[str | None] = mapped_column(String(500))
    raw_payload: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index(
            "idx_odds_unique_snapshot", "canonical_match_id", "bookmaker_id", "scraped_at",
            unique=True,
            postgresql_where=sa_text("market_type = 'match_winner'"),
            sqlite_where=sa_text("market_type = 'match_winner'"),
        ),
    )


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"))
    scraper_name: Mapped[str] = mapped_column(String(100))
    scraper_version: Mapped[str | None] = mapped_column(String(50))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=sa_text("NOW()"))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), server_default='running')
    source_url: Mapped[str | None] = mapped_column(String(500))
    request_url: Mapped[str | None] = mapped_column(String(500))
    items_seen: Mapped[int] = mapped_column(Integer, server_default="0")
    items_inserted: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)


class BookmakerEvent(Base):
    __tablename__ = "bookmaker_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"))
    bookmaker_event_id: Mapped[str] = mapped_column(String(100))
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"))
    raw_team_a: Mapped[str] = mapped_column(String(200))
    raw_team_b: Mapped[str] = mapped_column(String(200))
    match_start_time: Mapped[str | None] = mapped_column(String(50))
    sport_id: Mapped[str | None] = mapped_column(String(20))
    sport_name: Mapped[str | None] = mapped_column(String(100))
    category_id: Mapped[str | None] = mapped_column(String(20))
    category_name: Mapped[str | None] = mapped_column(String(100))
    league_id: Mapped[str | None] = mapped_column(String(20))
    league_name: Mapped[str | None] = mapped_column(String(100))
    first_seen_at: Mapped[str | None] = mapped_column(String(50))
    last_seen_at: Mapped[str | None] = mapped_column(String(50))
    offer_url: Mapped[str | None] = mapped_column(String(500))

    __table_args__ = (UniqueConstraint("bookmaker_id", "bookmaker_event_id"),)


class BookmakerMarket(Base):
    __tablename__ = "bookmaker_markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker_event_id: Mapped[str] = mapped_column(String(100), index=True)
    bookmaker_market_key: Mapped[str] = mapped_column(String(200))
    market_name: Mapped[str | None] = mapped_column(String(200))
    market_type: Mapped[str | None] = mapped_column(String(50), server_default='match_winner')
    line_id: Mapped[str | None] = mapped_column(String(50))
    line_name: Mapped[str | None] = mapped_column(String(200))
    is_extra_market: Mapped[bool | None] = mapped_column(Integer, server_default="0")

    __table_args__ = (UniqueConstraint("bookmaker_event_id", "bookmaker_market_key"),)


class OddsOutcomeSnapshot(Base):
    __tablename__ = "odds_outcome_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scrape_run_id: Mapped[int | None] = mapped_column(ForeignKey("scrape_runs.id"))
    bookmaker_event_id: Mapped[str] = mapped_column(String(100), index=True)
    bookmaker_market_key: Mapped[str] = mapped_column(String(200), index=True)
    outcome_key: Mapped[str] = mapped_column(String(200))
    scraped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, server_default=sa_text("NOW()"))
    source_url: Mapped[str | None] = mapped_column(String(500))
    offer_url: Mapped[str | None] = mapped_column(String(500))
    outcome_name: Mapped[str | None] = mapped_column(String(200))
    outcome_side: Mapped[str | None] = mapped_column(String(10))
    decimal_odds: Mapped[float | None] = mapped_column(Float)
    raw_payload: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("scrape_run_id", "outcome_key"),)


class PropOddsSnapshot(Base):
    """Snapshot of a bookmaker proposition market line (kills, handicap, duration, etc.)."""

    __tablename__ = "prop_odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False, index=True)
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"), index=True)
    market_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    map_number: Mapped[int] = mapped_column(Integer, server_default="1", index=True)
    line: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    target_team: Mapped[str | None] = mapped_column(String(200), index=True)
    raw_team_a: Mapped[str | None] = mapped_column(String(200))
    raw_team_b: Mapped[str | None] = mapped_column(String(200))
    odds_over: Mapped[float | None] = mapped_column(Float)
    odds_under: Mapped[float | None] = mapped_column(Float)
    prob_over_novig: Mapped[float | None] = mapped_column(Float)
    prob_under_novig: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    raw_market_name: Mapped[str | None] = mapped_column(String(255))
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, server_default=sa_text("NOW()"))
    is_live: Mapped[int | None] = mapped_column(Integer, server_default="0")
    source_url: Mapped[str | None] = mapped_column(String(500))
    outcomes_payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_prop_odds_match_market_time", "canonical_match_id", "market_type", "scraped_at"),
        Index("ix_prop_odds_line_search", "canonical_match_id", "market_type", "line", "scraped_at"),
    )
