"""Durable team, competition, and mapping-review identity models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class CanonicalTeam(Base):
    __tablename__ = "canonical_teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    squad_type: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SourceTeamIdentity(Base):
    __tablename__ = "source_team_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_team_id: Mapped[int] = mapped_column(ForeignKey("canonical_teams.id"), nullable=False)
    source_system: Mapped[str] = mapped_column(String(50), nullable=False)
    source_team_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    competition_scope: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    valid_from: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    valid_to: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    review_status: Mapped[str] = mapped_column(String(50), nullable=False, default="approved")

    __table_args__ = (
        UniqueConstraint("source_system", "source_team_id", "competition_scope", "valid_from", "valid_to"),
    )


class CanonicalCompetition(Base):
    __tablename__ = "canonical_competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    family: Mapped[str] = mapped_column(String(100), nullable=False)


class SourceCompetitionIdentity(Base):
    __tablename__ = "source_competition_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_competition_id: Mapped[int] = mapped_column(ForeignKey("canonical_competitions.id"), nullable=False)
    source_system: Mapped[str] = mapped_column(String(50), nullable=False)
    source_competition_id: Mapped[str] = mapped_column(String(200), nullable=False)
    source_name: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized_source_name: Mapped[str] = mapped_column(String(300), nullable=False)

    __table_args__ = (UniqueConstraint("source_system", "source_competition_id"),)


class MappingReviewDecision(Base):
    __tablename__ = "mapping_review_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_match_id: Mapped[int] = mapped_column(ForeignKey("canonical_matches.id"), nullable=False)
    old_golgg_match_id: Mapped[str | None] = mapped_column(String(50))
    new_golgg_match_id: Mapped[str | None] = mapped_column(String(50))
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    operator: Mapped[str] = mapped_column(String(100), nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
