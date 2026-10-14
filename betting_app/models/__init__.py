"""SQLAlchemy ORM models for the betting database."""

from betting_app.models.base import Base, get_sync_session, get_async_session, is_timescale, is_sqlite

from betting_app.models.bookmaker import Bookmaker, BookmakerAccount, Bet, WalletTransaction
from betting_app.models.golgg import (
    GolggGame,
    GolggGamePlayer,
    GolggMatch,
    GolggMatchMapping,
    GolggTeam,
    TeamAlias,
)
from betting_app.models.match import CanonicalMatch, UpcomingMatch
from betting_app.models.odds import (
    OddsSnapshot,
    ScrapeRun,
    BookmakerEvent,
    BookmakerMarket,
    OddsOutcomeSnapshot,
    PropOddsSnapshot,
)
from betting_app.models.prediction import (
    ModelArtifact,
    RatingRun,
    EntityRating,
    TeamRollingFeature,
    UpcomingMatchFeature,
    MatchRosterOverride,
    TeamCurrentRosterPlayer,
    CanonicalPrediction,
    ModelEvSignal,
    ModelPropPrediction,
    ForecastEvidenceSnapshot,
    SourceCapture,
    OpportunityEvent,
)
from betting_app.models.automation import AutomationRun, AutomationCommand
from betting_app.models.identity import (
    CanonicalCompetition,
    CanonicalTeam,
    MappingReviewDecision,
    SourceCompetitionIdentity,
    SourceTeamIdentity,
)
from betting_app.models.oddspapi import (
    OddspapiFixtureMapping,
    OddspapiRequestLog,
)
from betting_app.models.alerts import AlertConfig, ValueAlertLog

__all__ = [
    "Base",
    "get_sync_session",
    "get_async_session",
    "is_timescale",
    "is_sqlite",
    "AlertConfig",
    "ValueAlertLog",
    "Bookmaker",
    "BookmakerAccount",
    "Bet",
    "WalletTransaction",
    "GolggTeam",
    "GolggMatch",
    "GolggGame",
    "GolggGamePlayer",
    "GolggMatchMapping",
    "TeamAlias",
    "CanonicalCompetition",
    "CanonicalTeam",
    "MappingReviewDecision",
    "SourceCompetitionIdentity",
    "SourceTeamIdentity",
    "CanonicalMatch",
    "UpcomingMatch",
    "OddsSnapshot",
    "ScrapeRun",
    "BookmakerEvent",
    "BookmakerMarket",
    "OddsOutcomeSnapshot",
    "PropOddsSnapshot",
    "ModelArtifact",
    "RatingRun",
    "EntityRating",
    "TeamRollingFeature",
    "UpcomingMatchFeature",
    "MatchRosterOverride",
    "CanonicalPrediction",
    "ModelEvSignal",
    "ModelPropPrediction",
    "ForecastEvidenceSnapshot",
    "SourceCapture",
    "OpportunityEvent",
    "AutomationRun",
    "AutomationCommand",
    "OddspapiFixtureMapping",
    "OddspapiRequestLog",
]
