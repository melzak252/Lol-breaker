"""Canonical match and upcoming match models."""

from __future__ import annotations

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text as sa_text,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class CanonicalMatch(Base):
    __tablename__ = "canonical_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    team_a_name: Mapped[str] = mapped_column(String(200), nullable=False)
    team_b_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_a: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_b: Mapped[str] = mapped_column(String(200), nullable=False)
    start_time_normalized: Mapped[str | None] = mapped_column(String(50))
    league: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), server_default="upcoming")
    winner_name: Mapped[str | None] = mapped_column(String(200))
    loser_name: Mapped[str | None] = mapped_column(String(200))
    winner_normalized: Mapped[str | None] = mapped_column(String(200))
    winner_side: Mapped[str | None] = mapped_column(String(20))
    result_source: Mapped[str | None] = mapped_column(String(50))
    result_source_match_id: Mapped[str | None] = mapped_column(String(50))
    result_recorded_at: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    match_confidence: Mapped[float] = mapped_column(Float, server_default="1.0")
    best_of: Mapped[int | None] = mapped_column(SmallInteger, server_default="1")


class UpcomingMatch(Base):
    __tablename__ = "upcoming_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False)
    bookmaker_match_key: Mapped[str] = mapped_column(String(200), nullable=False)
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"))
    raw_team_a: Mapped[str] = mapped_column(String(200), nullable=False)
    raw_team_b: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_a: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_b: Mapped[str] = mapped_column(String(200), nullable=False)
    match_start_time: Mapped[str | None] = mapped_column(String(50))
    league: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(String(500))
    offer_url: Mapped[str | None] = mapped_column(String(500))
    is_live: Mapped[bool] = mapped_column(Integer, server_default="0")
    last_seen_at: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    team_a_golgg_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("golgg_teams.id"))
    team_b_golgg_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("golgg_teams.id"))

    __table_args__ = (UniqueConstraint("bookmaker_match_key"),)
