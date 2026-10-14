"""Prediction / rating / feature models."""

from __future__ import annotations
from datetime import UTC, datetime
import json
from typing import Any
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    DDL,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from betting_app.models.base import Base


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    feature_schema_json: Mapped[str | None] = mapped_column(Text)
    model_params_json: Mapped[str | None] = mapped_column(Text)
    training_cutoff_at: Mapped[str | None] = mapped_column(String(50))
    metrics_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), server_default='registered')

    __table_args__ = (UniqueConstraint("model_name", "model_version"),)


class RatingRun(Base):
    __tablename__ = "rating_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ratings_version: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    source: Mapped[str | None] = mapped_column(String(100))
    data_cutoff_at: Mapped[str | None] = mapped_column(String(50))
    started_at: Mapped[str | None] = mapped_column(String(50))
    finished_at: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), server_default='running')
    systems_json: Mapped[str | None] = mapped_column(Text)
    matches_processed: Mapped[int] = mapped_column(Integer, server_default="0")
    games_processed: Mapped[int] = mapped_column(Integer, server_default="0")
    players_processed: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)


class EntityRating(Base):
    __tablename__ = "entity_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rating_run_id: Mapped[int | None] = mapped_column(ForeignKey("rating_runs.id"))
    ratings_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    snapshot_at: Mapped[str | None] = mapped_column(String(50))
    entity_type: Mapped[str] = mapped_column(String(20), index=True)
    entity_name: Mapped[str] = mapped_column(String(200))
    normalized_entity_name: Mapped[str] = mapped_column(String(200), index=True)
    team_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str | None] = mapped_column(String(20))
    rating_system: Mapped[str] = mapped_column(String(20), index=True)
    rating_value: Mapped[float | None] = mapped_column(Float)
    rd: Mapped[float | None] = mapped_column(Float)
    sigma: Mapped[float | None] = mapped_column(Float)
    games_played: Mapped[int] = mapped_column(Integer, server_default="0")
    last_match_at: Mapped[str | None] = mapped_column(String(50))
    state_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("ratings_version", "entity_type", "normalized_entity_name", "rating_system"),)


class TeamRollingFeature(Base):
    __tablename__ = "team_rolling_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    window_size: Mapped[int] = mapped_column(Integer, server_default="20")
    data_cutoff_at: Mapped[str | None] = mapped_column(String(50))
    matches_count: Mapped[int] = mapped_column(Integer, server_default="0")
    games_count: Mapped[int] = mapped_column(Integer, server_default="0")
    win_rate: Mapped[float | None] = mapped_column(Float)
    avg_kills: Mapped[float | None] = mapped_column(Float)
    avg_deaths: Mapped[float | None] = mapped_column(Float)
    avg_gd15: Mapped[float | None] = mapped_column(Float)
    avg_dpm: Mapped[float | None] = mapped_column(Float)
    avg_vspm: Mapped[float | None] = mapped_column(Float)
    avg_gold: Mapped[float | None] = mapped_column(Float)
    avg_towers: Mapped[float | None] = mapped_column(Float)
    avg_dragons: Mapped[float | None] = mapped_column(Float)
    avg_nashors: Mapped[float | None] = mapped_column(Float)
    avg_game_duration: Mapped[float | None] = mapped_column(Float)
    features_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("feature_version", "normalized_team_name", "window_size"),)


class UpcomingMatchFeature(Base):
    __tablename__ = "upcoming_match_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_match_id: Mapped[int] = mapped_column(ForeignKey("canonical_matches.id"), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    ratings_version: Mapped[str] = mapped_column(String(100), nullable=False)
    data_cutoff_at: Mapped[str | None] = mapped_column(String(50))
    team_a_golgg_name: Mapped[str | None] = mapped_column(String(200))
    team_b_golgg_name: Mapped[str | None] = mapped_column(String(200))
    feature_status: Mapped[str] = mapped_column(String(50), server_default='pending')
    missing_reason: Mapped[str | None] = mapped_column(Text)
    features_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("canonical_match_id", "feature_version", "ratings_version"),)


class MatchRosterOverride(Base):
    """Manually confirmed roster used for an upcoming canonical match.

    The automatic pipeline normally uses the latest observed GOL.GG roster.
    This separate table keeps an operator correction auditable and prevents a
    scraper/feature refresh from silently overwriting it.
    """

    __tablename__ = "match_roster_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_match_id: Mapped[int] = mapped_column(
        ForeignKey("canonical_matches.id"), nullable=False, index=True
    )
    team_side: Mapped[str] = mapped_column(String(1), nullable=False)
    roster_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (UniqueConstraint("canonical_match_id", "team_side"),)


class TeamCurrentRosterPlayer(Base):
    """Current, role-specific roster membership for a GOL.GG team.

    This is deliberately separate from ``match_roster_overrides``: an override
    protects one prediction, while this table is the reusable best-known squad
    for every future match of a team.  GOL.GG refreshes update it from the most
    recent completed game; an operator may replace it before an announced
    roster has appeared on GOL.GG.
    """

    __tablename__ = "team_current_roster_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[str | None] = mapped_column(String(50), index=True)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_team_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    player_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default="auto")
    source_match_id: Mapped[str | None] = mapped_column(String(50))
    source_game_id: Mapped[str | None] = mapped_column(String(50))
    source_match_date: Mapped[str | None] = mapped_column(String(50))
    updated_at: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (UniqueConstraint("normalized_team_name", "role"),)


class CanonicalPrediction(Base):
    __tablename__ = "canonical_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_match_id: Mapped[int] = mapped_column(ForeignKey("canonical_matches.id"), nullable=False, index=True)
    model_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"))
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_at: Mapped[str | None] = mapped_column(String(50))
    prob_a: Mapped[float | None] = mapped_column(Float)
    prob_b: Mapped[float | None] = mapped_column(Float)
    features_version: Mapped[str | None] = mapped_column(String(100))
    ratings_version: Mapped[str | None] = mapped_column(String(100))
    data_cutoff_at: Mapped[str | None] = mapped_column(String(50))
    prediction_status: Mapped[str] = mapped_column(String(50), server_default='active')
    diagnostics_json: Mapped[str | None] = mapped_column(Text)


class ModelEvSignal(Base):
    __tablename__ = "model_ev_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_match_id: Mapped[int] = mapped_column(ForeignKey("canonical_matches.id"), nullable=False, index=True)
    canonical_prediction_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_predictions.id"))
    odds_snapshot_id: Mapped[int | None] = mapped_column(Integer)
    bookmaker_id: Mapped[int] = mapped_column(ForeignKey("bookmakers.id"), nullable=False)
    side: Mapped[str] = mapped_column(String(5), nullable=False)
    odds: Mapped[float | None] = mapped_column(Float)
    model_prob: Mapped[float | None] = mapped_column(Float)
    market_prob: Mapped[float | None] = mapped_column(Float)
    ev: Mapped[float | None] = mapped_column(Float)
    tax_rate: Mapped[float] = mapped_column(Float, server_default="0.12")
    stake_suggestion: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(50), server_default='new')
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    market_context_json: Mapped[str | None] = mapped_column(Text)


class ModelPropPrediction(Base):
    """Statistical pace, kill spread, and distribution model predictions for a match map (IDEA-018)."""

    __tablename__ = "model_prop_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_match_id: Mapped[int] = mapped_column(ForeignKey("canonical_matches.id"), nullable=False, index=True)
    map_number: Mapped[int] = mapped_column(Integer, server_default="1", index=True)
    model_name: Mapped[str] = mapped_column(String(100), server_default="kill_spread_distribution")
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, server_default="prop-v1.0")
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        index=True,
    )
    expected_total_kills: Mapped[float] = mapped_column(Float, nullable=False)
    mu_team_a: Mapped[float] = mapped_column(Float, nullable=False)
    mu_team_b: Mapped[float] = mapped_column(Float, nullable=False)
    expected_spread: Mapped[float] = mapped_column(Float, nullable=False)
    league_avg_kills: Mapped[float | None] = mapped_column(Float)
    pace_category: Mapped[str | None] = mapped_column(String(50))
    distribution_json: Mapped[str | None] = mapped_column(Text)
    signals_json: Mapped[str | None] = mapped_column(Text)

class ForecastEvidenceSnapshot(Base):
    """Append-only local evidence envelope; this is not admission certification."""

    __tablename__ = "forecast_evidence_snapshots"
    __table_args__ = (UniqueConstraint("evidence_key", name="uq_forecast_evidence_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evidence_key: Mapped[str] = mapped_column(String(200), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )

event.listen(
    ForecastEvidenceSnapshot.__table__,
    "after_create",
    DDL(
        """
        CREATE OR REPLACE FUNCTION deny_forecast_evidence_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'forecast evidence snapshots are append-only';
        END;
        $$
        """
    ).execute_if(dialect="postgresql"),
)
event.listen(
    ForecastEvidenceSnapshot.__table__,
    "after_create",
    DDL(
        """
        DROP TRIGGER IF EXISTS forecast_evidence_no_mutation
        ON forecast_evidence_snapshots;
        CREATE TRIGGER forecast_evidence_no_mutation
        BEFORE UPDATE OR DELETE ON forecast_evidence_snapshots
        FOR EACH ROW EXECUTE FUNCTION deny_forecast_evidence_mutation()
        """
    ).execute_if(dialect="postgresql"),
)


class SourceCapture(Base):
    """Append-only exact response bytes and per-attempt provenance clocks."""

    __tablename__ = "source_captures"
    __table_args__ = (
        UniqueConstraint("capture_id", name="uq_source_capture_id"),
        Index(
            "ix_source_capture_revision_first_observed",
            "source_id",
            "sha256",
            "source_revision",
            "first_observed_at",
        ),
        CheckConstraint(
            "(sha256 IS NULL AND content_base64 IS NULL AND body_uri IS NULL) OR "
            "(sha256 IS NOT NULL AND content_base64 IS NOT NULL AND body_uri IS NOT NULL)",
            name="ck_source_capture_body_reference",
        ),
        CheckConstraint(
            "(request_started_at IS NULL OR response_received_at IS NULL OR "
            "request_started_at <= response_received_at) AND "
            "(request_started_at IS NULL OR observed_at IS NULL OR "
            "request_started_at <= observed_at) AND "
            "(response_received_at IS NULL OR observed_at IS NULL OR "
            "response_received_at <= observed_at) AND "
            "(first_observed_at IS NULL OR observed_at IS NULL OR "
            "first_observed_at <= observed_at) AND "
            "(retrieved_at IS NULL OR retrieved_at = response_received_at)",
            name="ck_source_capture_clock_order",
        ),
        CheckConstraint(
            "attempt_status IN ('captured', 'failed')",
            name="ck_source_capture_attempt_status",
        ),
        CheckConstraint(
            "source_authentication_status IN ('local_fixture', 'unverified', 'authenticated')",
            name="ck_source_capture_authentication_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    capture_id: Mapped[str] = mapped_column(String(200), nullable=False)
    source_id: Mapped[str] = mapped_column(String(200), nullable=False)
    evidence_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    content_base64: Mapped[str | None] = mapped_column(Text)
    body_uri: Mapped[str | None] = mapped_column(Text)
    body_representation: Mapped[str] = mapped_column(String(40), nullable=False)
    content_encoding: Mapped[str | None] = mapped_column(String(100))
    character_encoding: Mapped[str | None] = mapped_column(String(80))
    request_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    response_received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_revision: Mapped[str | None] = mapped_column(String(200))
    http_status: Mapped[int | None] = mapped_column(Integer)
    source_class: Mapped[str] = mapped_column(String(50), nullable=False)
    availability_basis: Mapped[str] = mapped_column(String(100), nullable=False)
    attempt_status: Mapped[str] = mapped_column(String(20), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    request_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    response_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_authentication_status: Mapped[str] = mapped_column(String(30), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )


class OpportunityEvent(Base):
    """Immutable discovery/origin/decision event independent of forecast success."""

    __tablename__ = "opportunity_events"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_opportunity_event_key"),
        CheckConstraint(
            "NOT primary_decision OR origin_at IS NOT NULL",
            name="ck_opportunity_primary_origin_required",
        ),
        CheckConstraint(
            "event_type NOT IN ('schedule_revision', 'correction') OR parent_event_key IS NOT NULL",
            name="ck_opportunity_revision_parent_required",
        ),
        CheckConstraint(
            "event_type IN ('discovery', 'schedule_revision', 'origin_pending', "
            "'primary_decision', 'correction', 'outcome_release')",
            name="ck_opportunity_event_type",
        ),
        Index(
            "ix_opportunity_event_protocol_recorded",
            "protocol_id",
            "recorded_at",
            "id",
        ),
        Index(
            "uq_opportunity_primary_origin",
            "protocol_id",
            "provider_id",
            "fixture_id",
            "origin_at",
            unique=True,
            postgresql_where=text("primary_decision"),
        ),
        Index(
            "uq_opportunity_primary_fixture",
            "protocol_id",
            "provider_id",
            "fixture_id",
            unique=True,
            postgresql_where=text("primary_decision"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_key: Mapped[str] = mapped_column(String(200), nullable=False)
    protocol_id: Mapped[str] = mapped_column(String(120), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(120), nullable=False)
    fixture_id: Mapped[str] = mapped_column(String(200), nullable=False)
    origin_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    primary_decision: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    parent_event_key: Mapped[str | None] = mapped_column(
        String(200), ForeignKey("opportunity_events.event_key", name="fk_opportunity_parent_event")
    )
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    forecast_evidence_key: Mapped[str | None] = mapped_column(
        String(200),
        ForeignKey("forecast_evidence_snapshots.evidence_key", name="fk_opportunity_forecast_evidence"),
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.clock_timestamp()
    )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


for _append_only_table, _append_only_trigger in (
    (SourceCapture.__table__, "source_captures_no_mutation"),
    (OpportunityEvent.__table__, "opportunity_events_no_mutation"),
):
    event.listen(
        _append_only_table,
        "after_create",
        DDL(
            """
            CREATE OR REPLACE FUNCTION deny_plan09_evidence_mutation()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only';
            END;
            $$
            """
        ).execute_if(dialect="postgresql"),
    )
    event.listen(
        _append_only_table,
        "after_create",
        DDL(
            f"""
            DROP TRIGGER IF EXISTS {_append_only_trigger}
            ON {_append_only_table.name};
            CREATE TRIGGER {_append_only_trigger}
            BEFORE UPDATE OR DELETE ON {_append_only_table.name}
            FOR EACH ROW EXECUTE FUNCTION deny_plan09_evidence_mutation()
            """
        ).execute_if(dialect="postgresql"),
    )
