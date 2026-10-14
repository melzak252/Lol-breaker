"""In-game proposition prediction models and distribution engines (IDEA-018)."""

from __future__ import annotations

from betting_app.ml.props.schemas import (
    HandicapLinePrediction,
    KillDistributionParams,
    KillDistributionPrediction,
    LeaguePaceContext,
    MatchPropContext,
    MatchupSpreadFeatures,
    OverUnderLinePrediction,
    PaceFeatures,
    RecentGameSummary,
    TeamKillsSpreadPrediction,
    TeamRecentForm,
)

__all__ = [
    "HandicapLinePrediction",
    "KillDistributionParams",
    "KillDistributionPrediction",
    "LeaguePaceContext",
    "MatchPropContext",
    "MatchupSpreadFeatures",
    "OverUnderLinePrediction",
    "PaceFeatures",
    "RecentGameSummary",
    "TeamKillsSpreadPrediction",
    "TeamRecentForm",
]
