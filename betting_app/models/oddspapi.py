"""OddsPapi integration models for external fixture mapping and budget audit trail."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    text as sa_text,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class OddspapiFixtureMapping(Base):
    """Maps OddsPapi fixture identifiers to canonical matches."""

    __tablename__ = "oddspapi_fixture_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    canonical_match_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_matches.id"), index=True)
    sport_id: Mapped[int] = mapped_column(Integer, server_default="18")
    league: Mapped[str | None] = mapped_column(String(100))
    provider_team_1: Mapped[str] = mapped_column(String(200), nullable=False)
    provider_team_2: Mapped[str] = mapped_column(String(200), nullable=False)
    provider_team_1_is_a: Mapped[int | None] = mapped_column(Integer)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    has_odds: Mapped[int] = mapped_column(Integer, server_default="1")
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )


class OddspapiRequestLog(Base):
    """Immutable audit trail for OddsPapi HTTP requests to enforce quota boundaries."""

    __tablename__ = "oddspapi_request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    fixture_id: Mapped[str | None] = mapped_column(String(100), index=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=sa_text("CURRENT_TIMESTAMP"),
        index=True,
    )
