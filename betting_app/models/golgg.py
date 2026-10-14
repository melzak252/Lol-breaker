"""GOL.GG match/game/player/team models."""

from __future__ import annotations

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class GolggTeam(Base):
    __tablename__ = "golgg_teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    team_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    normalized_name: Mapped[str | None] = mapped_column(String(200))


class GolggMatch(Base):
    __tablename__ = "golgg_matches"

    match_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    date: Mapped[str | None] = mapped_column(String(20))
    tournament_name: Mapped[str | None] = mapped_column(String(300))
    patch: Mapped[str | None] = mapped_column(String(20))
    team1_name: Mapped[str | None] = mapped_column(String(200))
    team2_name: Mapped[str | None] = mapped_column(String(200))
    team1_id: Mapped[str | None] = mapped_column(String(50))
    team2_id: Mapped[str | None] = mapped_column(String(50))
    team1_score: Mapped[int | None] = mapped_column(Integer)
    team2_score: Mapped[int | None] = mapped_column(Integer)
    team1_win: Mapped[bool | None] = mapped_column(Integer)
    team2_win: Mapped[bool | None] = mapped_column(Integer)
    draw: Mapped[bool | None] = mapped_column(Integer)
    games_played: Mapped[int | None] = mapped_column(Integer)
    best_of: Mapped[int | None] = mapped_column(Integer)
    winner_name: Mapped[str | None] = mapped_column(String(200))
    loser_name: Mapped[str | None] = mapped_column(String(200))
    source_link: Mapped[str | None] = mapped_column(String(500))
    raw_json: Mapped[str | None] = mapped_column(Text)


class GolggGame(Base):
    __tablename__ = "golgg_games"

    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("golgg_matches.match_id"), nullable=False, index=True)
    date: Mapped[str | None] = mapped_column(String(20))
    tournament_name: Mapped[str | None] = mapped_column(String(300))
    patch: Mapped[str | None] = mapped_column(String(20))
    team1_name: Mapped[str | None] = mapped_column(String(200))
    team2_name: Mapped[str | None] = mapped_column(String(200))
    team1_id: Mapped[str | None] = mapped_column(String(50))
    team2_id: Mapped[str | None] = mapped_column(String(50))
    team1_win: Mapped[bool | None] = mapped_column(Integer)
    team2_win: Mapped[bool | None] = mapped_column(Integer)
    draw: Mapped[bool | None] = mapped_column(Integer)
    team1_side: Mapped[str | None] = mapped_column(String(10))
    team2_side: Mapped[str | None] = mapped_column(String(10))
    game_duration: Mapped[int | None] = mapped_column(Integer)
    team1_stats_json: Mapped[str | None] = mapped_column(Text)
    team2_stats_json: Mapped[str | None] = mapped_column(Text)
    raw_json: Mapped[str | None] = mapped_column(Text)


class GolggGamePlayer(Base):
    __tablename__ = "golgg_game_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("golgg_games.game_id"), nullable=False, index=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("golgg_matches.match_id"), index=True)
    team_id: Mapped[str | None] = mapped_column(String(50))
    team_name: Mapped[str | None] = mapped_column(String(200))
    side: Mapped[str] = mapped_column(String(5), index=True)
    role: Mapped[str | None] = mapped_column(String(20))
    player_id: Mapped[str | None] = mapped_column(String(50), index=True)
    player_name: Mapped[str | None] = mapped_column(String(200))
    champion_id: Mapped[str | None] = mapped_column(String(50))
    champion_name: Mapped[str | None] = mapped_column(String(100))
    champion_image: Mapped[str | None] = mapped_column(String(500))
    stats_json: Mapped[str | None] = mapped_column(Text)
    raw_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("game_id", "side", "role"),)


class GolggMatchMapping(Base):
    __tablename__ = "golgg_match_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_match_id: Mapped[int] = mapped_column(
        ForeignKey("canonical_matches.id"), nullable=False, unique=True
    )
    golgg_match_id: Mapped[str] = mapped_column(
        ForeignKey("golgg_matches.match_id"), nullable=False, unique=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    mapped_by: Mapped[str] = mapped_column(String(50), nullable=False, default="auto")
    mapped_at: Mapped[str | None] = mapped_column(String(50))


class TeamAlias(Base):
    __tablename__ = "team_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False)
    alias: Mapped[str] = mapped_column(String(200), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_name: Mapped[str | None] = mapped_column(String(200))
    normalized_alias: Mapped[str | None] = mapped_column(String(200))
    source_system: Mapped[str | None] = mapped_column(String(50))
    league_pattern: Mapped[str | None] = mapped_column(String(200))
    tournament_pattern: Mapped[str | None] = mapped_column(String(200))
    valid_from: Mapped[str | None] = mapped_column(String(20))
    valid_to: Mapped[str | None] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[int] = mapped_column(Integer, default=1)
    is_blocked: Mapped[int] = mapped_column(Integer, default=0)
    review_status: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column(String(50))
    updated_at: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (UniqueConstraint("normalized_name", "source"),)
