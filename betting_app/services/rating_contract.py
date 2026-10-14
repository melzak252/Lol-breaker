"""Immutable identifiers for the regional operational-rating contract."""

from __future__ import annotations

OPERATIONAL_RATINGS_VERSION = "ratings-v2"
OPERATIONAL_FEATURE_VERSION = "player-team-ratings-w20-v0.3"
OPERATIONAL_MODEL_NAME = "Operational-PlayerTeamRatings-W20"
OPERATIONAL_MODEL_VERSION = "v0.4-binom-series"
OPERATIONAL_BACKFILL_FEATURE_VERSION = "operational-ratings-v2-chronological-v1"
OPERATIONAL_BACKFILL_MODEL_VERSION = "v0.4-binom-series-chronological-v1"
REGIONAL_GLICKO_SYSTEM = "gl"
RAW_RATING_SYSTEMS = ("elo", "ts", "os", "pl", "tm")
PUBLIC_RATING_SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
REGIONAL_ENGINE = "family-calibrated-glicko2-v1"
