"""Pydantic request/response schemas for the betting API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# ── Match board ─────────────────────────────────────────────────────────────


class MatchBoardItem(BaseModel):
    canonical_match_id: int
    match: str
    league: str | None = None
    start_time_normalized: str | None = None
    best_of: int | None = None

    team_a_name: str | None = None
    team_b_name: str | None = None
    team_a_golgg_name: str | None = None
    team_b_golgg_name: str | None = None
    team_a_mapping_source: str | None = None
    team_b_mapping_source: str | None = None
    has_unmapped_teams: bool = False
    match_confidence: float | None = None

    bookmaker_count: int = 0
    diagnostic_best_odds_a: float | None = None
    diagnostic_best_odds_b: float | None = None
    quote_status: Literal["actionable", "diagnostic", "unavailable"] = "unavailable"

    best_odds_a: float | None = None
    best_bookmaker_a: str | None = None
    best_offer_url_a: str | None = None
    avg_odds_a: float | None = None

    best_odds_b: float | None = None
    best_bookmaker_b: str | None = None
    best_offer_url_b: str | None = None
    avg_odds_b: float | None = None

    arb_no_tax: bool = False
    arb_after_tax: bool = False
    arb_margin_no_tax: float | None = None
    arb_margin_after_tax: float | None = None

    # Model/hybrid probabilities and EV (enriched per row in board)
    model_prob_a: float | None = None
    model_prob_b: float | None = None
    hybrid_prob_a: float | None = None
    hybrid_prob_b: float | None = None
    hybrid_ev_a: float | None = None
    hybrid_ev_b: float | None = None
    
    # Fusion SymAug probabilities and EV
    fusion_symaug_prob_a: float | None = None
    fusion_symaug_prob_b: float | None = None
    fusion_symaug_ev_a: float | None = None
    fusion_symaug_ev_b: float | None = None
    # Conformal Risk Bounds and 1/4 Kelly Stake (EXP-040)
    conformal_prob_low_a: float | None = None
    conformal_prob_low_b: float | None = None
    conformal_ev_a: float | None = None
    conformal_ev_b: float | None = None
    is_conformal_value_a: bool = False
    is_conformal_value_b: bool = False
    recommended_stake_a: float | None = None
    recommended_stake_b: float | None = None
    recommended_side: str | None = None
    recommended_team: str | None = None
    recommended_bookmaker: str | None = None
    recommended_odds: float | None = None
    recommended_ev: float | None = None

    has_props: bool = False
    prop_lines_count: int = 0

    last_scraped_at: str | None = None

class MatchBoardResponse(BaseModel):
    total: int
    matches: list[MatchBoardItem]


# ── Match detail ────────────────────────────────────────────────────────────


class BookmakerOddsRow(BaseModel):
    bookmaker: str
    odds_snapshot_id: int | None = None
    quote_scraped_at: str | None = None
    quote_status: Literal["actionable", "diagnostic", "unavailable"] = "unavailable"
    raw_team_a: str | None = None
    raw_team_b: str | None = None
    canonical_odds_a: float | None = None
    canonical_odds_b: float | None = None
    scraped_at: str | None = None
    source_url: str | None = None
    offer_url: str | None = None
    ev_a: float | None = None
    ev_b: float | None = None
    kelly_a: float | None = None
    kelly_b: float | None = None


class PredictionRow(BaseModel):
    model_name: str
    model_version: str
    prob_a: float | None = None
    prob_b: float | None = None
    predicted_at: str | None = None
    ev_a: float | None = None
    ev_b: float | None = None
    kelly_a: float | None = None
    kelly_b: float | None = None
    raw_diagnostics: dict[str, Any] | None = None


class RosterPlayer(BaseModel):
    player_id: str | None = None
    player_name: str | None = None
    role: str | None = None
    champion_name: str | None = None
    elo_rating: float | None = None
    glicko_rating: float | None = None
    glicko_rd: float | None = None
    trueskill_rating: float | None = None
    rating_uncertainty: float | None = None
    games_played: int | None = None


class RosterInfo(BaseModel):
    team_name: str | None = None
    source_match_id: str | None = None
    source_date: str | None = None
    source_tournament: str | None = None
    roster_source: str | None = None
    avg_elo: float | None = None
    avg_glicko: float | None = None
    avg_glicko_rd: float | None = None
    players_with_rating: int | None = None
    players: list[RosterPlayer] = []


class RosterOverridePlayerInput(BaseModel):
    """One manually confirmed player for an upcoming match roster."""

    player_id: str | None = Field(default=None, max_length=100)
    player_name: str = Field(min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=30)


class MatchRosterOverrideRequest(BaseModel):
    """Confirmed five-player roster for one side of an upcoming match."""

    team_side: Literal["a", "b"]
    players: list[RosterOverridePlayerInput] = Field(min_length=5, max_length=5)


class MatchRosterOverrideResponse(BaseModel):
    canonical_match_id: int
    team_side: Literal["a", "b"]
    roster: RosterInfo
    message: str



class RosterVerificationRequest(BaseModel):
    """Parameters for on-demand roster verification and sync."""

    team_names: list[str] | None = Field(default=None, description="Optional list of team names. If None, checks upcoming matches.")
    source: Literal["auto", "fandom", "liquipedia"] = Field(default="auto")
    dry_run: bool = Field(default=False, description="If True, reports discrepancies without committing to DB")
    force: bool = Field(default=False, description="If True, forces overwrite even if recent timestamp exists")
    limit: int = Field(default=50, ge=1, le=200)


class RosterStatusResponse(BaseModel):
    """Status of upcoming match rosters in the system."""

    upcoming_matches_count: int
    total_teams: int
    teams_with_roster: int
    teams_missing_roster: int
    source_counts: dict[str, int]
    teams_missing: list[str]
    teams_verified: list[dict[str, Any]] = []

class TeamRecentStats(BaseModel):
    team_name: str | None = None
    matches_count: int | None = None
    games_count: int | None = None
    win_rate: float | None = None
    avg_kills: float | None = None
    avg_deaths: float | None = None
    avg_gd15: float | None = None
    avg_dragons: float | None = None
    avg_nashors: float | None = None
    avg_towers: float | None = None
    avg_game_duration: float | None = None
    last_match_at: str | None = None


class TeamMappingInfo(BaseModel):
    canonical_name: str | None = None
    golgg_name: str | None = None
    confidence: float | None = None
    source: str | None = None  # 'alias', 'builtin', 'blocked', or None when unmapped


class TeamComparisonInfo(BaseModel):
    team_a: TeamMappingInfo | None = None
    team_b: TeamMappingInfo | None = None
    team_a_rating: float | None = None
    team_b_rating: float | None = None
    rating_system: str | None = None
    team_a_elo: float | None = None
    team_b_elo: float | None = None
    team_a_glicko: float | None = None
    team_b_glicko: float | None = None
    team_a_glicko_rd: float | None = None
    team_b_glicko_rd: float | None = None
    team_a_games_played: int | None = None
    team_b_games_played: int | None = None
    rating_probabilities: dict[str, float] = {}


class BookmakerEvSide(BaseModel):
    ev: float | None = None
    odds: float | None = None
    model_prob: float | None = None
    kelly: float | None = None
    market_prob: float | None = None
    odds_snapshot_id: int | None = None
    scraped_at: str | None = None


class BookmakerEvDetail(BaseModel):
    side_a: BookmakerEvSide = BookmakerEvSide()
    side_b: BookmakerEvSide = BookmakerEvSide()


class MatchResultItem(BaseModel):
    canonical_match_id: int
    probability_basis: Literal["historical_quote_recomputation"] = "historical_quote_recomputation"
    probability_unavailable_reason: str | None = None
    team_a_name: str | None = None
    team_b_name: str | None = None
    league: str | None = None
    start_time_normalized: str | None = None
    best_of: int | None = None
    status: str | None = None
    winner_name: str | None = None
    loser_name: str | None = None
    winner_side: str | None = None  # "team_a" or "team_b"
    team_a_score: int | None = None
    team_b_score: int | None = None
    result_source: str | None = None
    result_recorded_at: str | None = None
    # EV signals (best across all bookmakers)
    best_ev_a: float | None = None
    best_ev_b: float | None = None
    bookmakers_with_ev: list[str] = []
    best_odds_a: float | None = None
    best_odds_b: float | None = None
    # Model probabilities (for Kelly staking)
    model_prob_a: float | None = None
    model_prob_b: float | None = None
    # Kelly fractions (full Kelly, after tax)
    kelly_a: float | None = None
    kelly_b: float | None = None
    # Per-bookmaker EV/odds details
    bookmaker_ev_details: dict[str, BookmakerEvDetail] = {}
    model_name: str | None = None
    model_version: str | None = None
    odds_mode: str | None = None


class MatchResultsResponse(BaseModel):
    total: int
    days_back: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    odds_mode: str | None = None
    results: list[MatchResultItem]


# ── Financial backtest ─────────────────────────────────────────────────────


class FinancialLedgerEntry(BaseModel):
    canonical_match_id: int
    start_time: str | None = None
    league: str | None = None
    team_a_name: str | None = None
    team_b_name: str | None = None
    bookmaker: str
    side: str
    entry_odds: float
    close_odds: float | None = None
    hours_before: float | None = None
    horizon: str
    model_prob: float
    target_prob: float | None = None
    uncertainty_basis: Literal[
        "mean", "measured_bounds", "assumed_research_haircut", "missing_bounds"
    ]
    uncertainty_reason: str
    market_prob: float
    ev: float
    stake: float
    profit: float
    bankroll_after: float
    won: bool
    clv_odds_pct: float | None = None
    entry_scraped_at: str | None = None
    close_scraped_at: str | None = None


class FinancialBucket(BaseModel):
    key: str
    label: str
    bets: int
    matches: int
    staked: float
    profit: float
    roi: float | None = None
    hit_rate: float | None = None
    avg_ev: float | None = None
    avg_clv_odds_pct: float | None = None
    median_clv_odds_pct: float | None = None


class FinancialAnalysisResponse(BaseModel):
    methodology: str
    evaluation_kind: Literal["historical_quote_simulation"]
    result_status: Literal["computed", "no_eligible_bets"]
    uncertainty_basis: Literal[
        "mean", "measured_bounds", "assumed_research_haircut", "missing_bounds"
    ]
    uncertainty_reason: str
    historical_diagnostics_status: Literal["available", "unknown"]
    data_scope: str
    days_back: int
    odds_mode: str
    model_name: str
    model_version: str
    staking_mode: str
    min_ev: float
    initial_bankroll: float
    final_bankroll: float
    total_bets: int
    total_matches: int
    total_staked: float
    total_profit: float
    roi: float | None = None
    hit_rate: float | None = None
    max_drawdown_pct: float | None = None
    avg_clv_odds_pct: float | None = None
    positive_clv_rate: float | None = None
    max_open_stake: float = 0.0
    max_open_bets: int = 0
    temporal_exclusions: dict[str, int] = {}
    horizon_buckets: list[FinancialBucket] = []
    bookmaker_buckets: list[FinancialBucket] = []
    league_buckets: list[FinancialBucket] = []
    bankroll_curve: list[dict[str, float | str | int | None]] = []
    ledger: list[FinancialLedgerEntry] = []


class MatchBestOfUpdate(BaseModel):
    best_of: int = Field(ge=1, le=7)


class BettingRecommendation(BaseModel):
    has_value: bool
    verdict: str  # "value_bet" | "no_bet" | "unmapped" | "no_odds"
    verdict_label: str
    side: Literal["a", "b"] | None = None
    recommended_team: str | None = None
    opponent_team: str | None = None
    bookmaker: str | None = None
    best_odds: float | None = None
    offer_url: str | None = None

    # Probabilities
    model_prob: float | None = None
    hybrid_prob: float | None = None
    market_prob: float | None = None

    # Expected Value & Edge
    ev: float | None = None
    pure_model_ev: float | None = None
    edge_percentage_points: float | None = None
    min_odds_required: float | None = None

    # Staking
    half_kelly: float | None = None
    quarter_kelly: float | None = None
    suggested_stake_pct: float | None = None

    # Conformal bounds (if available from Venn-Abers / EXP-040)
    conformal_prob_low: float | None = None
    conformal_ev: float | None = None
    is_conformal_safe: bool = False

    # Human-readable breakdown
    summary: str
    reasons: list[str] = []
    threshold_info: str | None = None
    # Optimal Strategy Dual-Paradigm Guards
    thesis_prob: float | None = None
    thesis_disagreement: float | None = None
    thesis_guard_passed: bool | None = None
    guard_b_passed: bool | None = None
    logit_disagreement: float | None = None
    raw_clv: float | None = None

class MatchDetailResponse(BaseModel):
    canonical_match_id: int
    team_a_name: str | None = None
    team_b_name: str | None = None
    league: str | None = None
    start_time_normalized: str | None = None
    status: str | None = None
    best_of: int | None = None

    odds: list[BookmakerOddsRow] = []
    predictions: list[PredictionRow] = []
    roster_a: RosterInfo | None = None
    roster_b: RosterInfo | None = None
    roster_a_is_manual: bool = False
    roster_b_is_manual: bool = False
    recent_stats_a: TeamRecentStats | None = None
    recent_stats_b: TeamRecentStats | None = None
    team_comparison: TeamComparisonInfo | None = None
    recommendation: BettingRecommendation | None = None


# ── Odds history (line movement) ────────────────────────────────────────────


class OddsHistoryPoint(BaseModel):
    bookmaker: str
    scraped_at: str
    odds_a: float | None = None
    odds_b: float | None = None
    canonical_odds_a: float | None = None
    canonical_odds_b: float | None = None


# ── Prediction & EV history timeline ────────────────────────────────────────


class PredictionHistoryPoint(BaseModel):
    timestamp: str
    model_name: str
    model_version: str
    prob_a: float | None = None
    prob_b: float | None = None
    avg_odds_a: float | None = None
    avg_odds_b: float | None = None
    market_prob_a: float | None = None
    market_prob_b: float | None = None
    ev_a: float | None = None
    ev_b: float | None = None
    raw_diagnostics: dict[str, Any] | None = None


# ── Predictions / EV+ signals ───────────────────────────────────────────────


class EVSignal(BaseModel):
    canonical_match_id: int
    odds_snapshot_id: int
    quote_scraped_at: str
    evaluated_at: str
    match: str
    league: str | None = None
    start_time_normalized: str | None = None
    model_name: str
    model_version: str
    side: str  # "a" or "b"
    odds: float
    bookmaker: str
    model_prob: float
    prob_conservative: float | None
    prob_for_ev: float | None
    uncertainty_basis: Literal[
        "mean", "measured_bounds", "assumed_research_haircut", "missing_bounds"
    ]
    uncertainty_reason: str
    market_prob: float | None = None
    ev: float
    kelly: float = 0.0
    offer_url: str | None = None


class EVSignalResponse(BaseModel):
    total: int
    signals: list[EVSignal]


# ── Team and player rankings ────────────────────────────────────────────────


class RankingEntry(BaseModel):
    rank: int
    entity_type: Literal["team", "player"]
    entity_name: str
    normalized_entity_name: str
    team_name: str | None = None
    role: str | None = None
    rating_system: str
    rating_value: float
    rd: float | None = None
    sigma: float | None = None
    system_count: int = 1
    games_played: int
    last_match_at: str | None = None
    region_family: str | None = None
    region_tier: str | None = None
    regional_offset: float | None = None
    regional_uncertainty: float | None = None


class RankingsResponse(BaseModel):
    entity_type: Literal["team", "player"]
    rating_system: str
    ratings_version: str | None = None
    data_cutoff_at: str | None = None
    active_since: str | None = None
    squad_scope: Literal["major", "regional_academy", "regional", "development", "all", "main"] = "major"
    snapshot_at: str | None = None
    total: int
    available_rating_systems: list[str] = []
    rankings: list[RankingEntry] = []



# ── Player comparison and rating history ─────────────────────────────────────


class PlayerSearchItem(BaseModel):
    player_id: str
    player_name: str
    team_name: str | None = None
    role: str | None = None
    games_played: int
    current_elo: float | None = None
    current_gl: float | None = None
    last_match_at: str | None = None


class TopChampionItem(BaseModel):
    champion_name: str
    games: int
    wins: int
    win_rate: float


class PlayerSystemRating(BaseModel):
    system: str
    rating_value: float
    rd: float | None = None
    sigma: float | None = None
    rank: int | None = None
    percentile: float | None = None


class PlayerProfileDetail(BaseModel):
    player_id: str
    player_name: str
    team_name: str | None = None
    role: str | None = None
    games_played: int
    career_wins: int
    career_losses: int
    career_win_rate: float
    career_first_date: str | None = None
    career_last_date: str | None = None
    career_years: float | None = None
    teams: list[str] = []
    top_champions: list[TopChampionItem] = []
    ratings: dict[str, PlayerSystemRating] = {}
    peak_elo: float | None = None
    peak_elo_date: str | None = None
    peak_gl: float | None = None
    peak_gl_date: str | None = None


class H2HGameItem(BaseModel):
    game_id: str
    match_id: str
    date: str | None = None
    tournament_name: str | None = None
    team_a: str | None = None
    champ_a: str | None = None
    team_b: str | None = None
    champ_b: str | None = None
    winner: Literal["a", "b"]


class H2HSummary(BaseModel):
    total_games: int
    wins_a: int
    wins_b: int
    win_rate_a: float
    win_rate_b: float
    recent_games: list[H2HGameItem] = []


class SystemAdvantage(BaseModel):
    system: str
    system_label: str
    value_a: float
    value_b: float
    difference: float
    favors: Literal["a", "b", "tied"]
    win_prob_a: float


class ModelVerdict(BaseModel):
    better_player: Literal["a", "b", "tied"]
    better_player_id: str | None = None
    better_player_name: str | None = None
    win_probability_a: float
    win_probability_b: float
    systems_favor_a: int
    systems_favor_b: int
    systems_tied: int
    total_systems: int
    advantage_summary: str
    system_advantages: list[SystemAdvantage] = []
    h2h_winner: Literal["a", "b", "tied"]
    summary_pl: str


class RatingTimelinePoint(BaseModel):
    date: str
    match_id: str | None = None
    team_name: str | None = None
    games_count: int
    elo: float
    gl: float
    gl_rd: float | None = None
    ts_mu: float | None = None
    os_mu: float | None = None
    pl_mu: float | None = None
    tm_mu: float | None = None


class PlayerComparisonResponse(BaseModel):
    player_a: PlayerProfileDetail
    player_b: PlayerProfileDetail
    verdict: ModelVerdict
    h2h: H2HSummary
    timeline_a: list[RatingTimelinePoint] = []
    timeline_b: list[RatingTimelinePoint] = []
    available_rating_systems: list[str] = []


# ── Tax-amortized parlay recommendations (IDEA-019) ──────────────────────────


class ParlayLeg(BaseModel):
    canonical_match_id: int
    match_name: str
    league: str | None = None
    start_time: str | None = None
    side: Literal["a", "b"]
    team_name: str
    opponent_name: str
    odds: float
    model_prob: float
    single_ev: float
    is_favorite: bool = True


class ParlayRecommendation(BaseModel):
    id: str
    bookmaker: str
    legs: list[ParlayLeg]
    combined_odds: float
    effective_odds: float
    joint_prob: float
    breakeven_prob: float
    ev: float
    tax_amortization_gain: float
    suggested_stake: float
    quarter_kelly: float
    confidence_badge: str
    summary_pl: str


class ParlayRecommendationsResponse(BaseModel):
    count: int
    top_parlay: ParlayRecommendation | None = None
    parlays: list[ParlayRecommendation] = []
    tax_rate: float = 0.12
    explanation: str
# ── System status ───────────────────────────────────────────────────────────


class SystemStatusResponse(BaseModel):
    counts: dict[str, int]
    last_scrape_runs: list[dict[str, Any]] = []
    last_automation_runs: list[dict[str, Any]] = []


class AutomationTriggerResponse(BaseModel):
    status: str
    message: str


# ── Wallets and bets ────────────────────────────────────────────────────────


class WalletResponse(BaseModel):
    id: int
    bookmaker: str | None = None
    account_name: str
    currency: str = "PLN"
    current_balance: float
    is_active: bool = True


class BetCreate(BaseModel):
    bookmaker_account_id: int
    canonical_match_id: int | None = None
    team_a: str | None = None
    team_b: str | None = None
    league: str | None = None
    match_start_time: str | None = None
    side: str = Field(pattern="^(a|b)$")
    stake: Decimal = Field(
        gt=0, allow_inf_nan=False,
        description="Manual stake rounded to cents with ROUND_HALF_UP; must remain positive.",
    )
    odds: Decimal = Field(
        gt=1, allow_inf_nan=False,
        description=(
            "Rounded to four decimals with ROUND_HALF_UP before persistence; "
            "must remain greater than one and fit Numeric(8, 4)."
        ),
    )
    model_prob: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    ev: float | None = Field(default=None, allow_inf_nan=False)
    tax_rate: Decimal = Field(
        default=Decimal("0.12"), ge=0, lt=1, allow_inf_nan=False,
        description="Rounded to two decimals with ROUND_HALF_UP; must remain less than one.",
    )
    note: str | None = None


class BetResponse(BaseModel):
    id: int
    bookmaker_account_id: int
    canonical_match_id: int | None = None
    team_a: str | None = None
    team_b: str | None = None
    stake: float
    odds: float
    side: str
    status: str
    profit: float | None = None
    placed_at: str | None = None
    settled_at: str | None = None
    note: str | None = None


class BetSettle(BaseModel):
    result: str = Field(pattern="^(won|lost|void|cancelled)$")
    settlement_odds: Decimal | None = Field(
        default=None, gt=1, allow_inf_nan=False,
        description=(
            "Winning payout override rounded to four decimals with ROUND_HALF_UP; "
            "must remain greater than one and fit Numeric(8, 4). "
            "Not used for lost, void or cancelled settlements."
        ),
    )


# ── Bookmakers ──────────────────────────────────────────────────────────────


class BookmakerStatus(BaseModel):
    id: int
    name: str
    base_url: str | None = None
    last_scraped_at: str | None = None
    snapshot_count: int = 0


# ── Scheduler ───────────────────────────────────────────────────────────────


class SchedulerTaskResponse(BaseModel):
    id: str
    task_id: str
    name: str
    description: str = ""
    schedule: str
    interval_minutes: int | None = None
    cron_trigger: str | None = None
    enabled: bool = True


class SchedulerJobResponse(BaseModel):
    id: str
    name: str
    enabled: bool = True
    next_run_time: str | None = None
    last_run_at: str | None = None
    last_run_status: str | None = None
    is_running: bool = False
    trigger: str
    pending: bool = False


class SchedulerTriggerResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ── Single-match prediction ────────────────────────────────────────────────


class PredictResponse(BaseModel):
    status: str = "ok"
    message: str = ""
    prob_a: float | None = None
    prob_b: float | None = None
    hybrid_prob_a: float | None = None
    hybrid_prob_b: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    diagnostics: dict[str, Any] | None = None



# ── Custom Matchup Simulation ───────────────────────────────────────────────


class MatchupSimulationRequest(BaseModel):
    team_a_name: str = Field(min_length=1, description="Team A name or GOL.GG identifier")
    team_b_name: str = Field(min_length=1, description="Team B name or GOL.GG identifier")
    best_of: Literal[1, 3, 5] = Field(default=1, description="Native C0 series length: 1, 3, or 5")
    team_a_team_row_id: int | None = Field(default=None, description="Internal GOL.GG team row ID")
    team_b_team_row_id: int | None = Field(default=None, description="Internal GOL.GG team row ID")
    native_team_a_id: str | None = Field(default=None, description="Explicit provider-native team ID")
    native_team_b_id: str | None = Field(default=None, description="Explicit provider-native team ID")
    league: str | None = Field(default=None, description="Optional tournament/league context")
    team_a_roster_override: dict[str, Any] | None = None
    team_b_roster_override: dict[str, Any] | None = None


class MatchupSimulationResponse(BaseModel):
    team_a_name: str
    team_b_name: str
    best_of: int
    map_prob_a: float | None = None
    map_prob_b: float | None = None
    series_prob_a: float
    series_prob_b: float
    model_name: str
    model_version: str
    roster_a: RosterInfo | None = None
    roster_b: RosterInfo | None = None
    recent_stats_a: TeamRecentStats | None = None
    recent_stats_b: TeamRecentStats | None = None
    team_comparison: TeamComparisonInfo | None = None
    components: dict[str, Any] = Field(default_factory=dict)


class SimulateFearlessRequest(BaseModel):
    best_of: Literal[3, 5] = Field(default=5, description="Best-of series length: 3 or 5")
    n_simulations: int = Field(default=5000, ge=100, le=50000, description="Monte Carlo simulation count")
    seed: int = Field(default=42, description="Random seed")


class SimulateFearlessResponse(BaseModel):
    canonical_match_id: int | None = None
    team_a_name: str
    team_b_name: str
    best_of: int
    n_simulations: int
    series_prob_a: float
    series_prob_b: float
    scorelines: dict[str, float]
    scoreline_pmf: dict[str, float]
    map_win_probabilities: dict[int, float]
    map_reach_probabilities: dict[int, float]
    mean_unique_champions: float
    champions_drafted_distribution: dict[int, float]
# ── Team alias mapping ──────────────────────────────────────────────────────


class AliasCreateRequest(BaseModel):
    raw_name: str = Field(min_length=1, description="Bookmaker/raw team name to map")
    golgg_team_name: str = Field(min_length=1, description="GolGG canonical team name to map to")
    source_system: str | None = Field(
        default=None,
        max_length=50,
        description="Source family for a scoped alias, for example bookmaker",
    )
    league_pattern: str | None = Field(
        default=None,
        max_length=200,
        description="Competition scope required for collision-prone short aliases",
    )
    valid_from: str | None = Field(default=None, max_length=20)
    valid_to: str | None = Field(default=None, max_length=20)


class AliasCreateResponse(BaseModel):
    id: int
    normalized_name: str
    alias: str
    source: str


class AliasDeleteRequest(BaseModel):
    raw_name: str = Field(min_length=1, description="Bookmaker/raw team name to unmap")


class AliasBlockRequest(BaseModel):
    raw_name: str = Field(min_length=1, description="Bookmaker/raw team name to mark as blocked")


class GolggTeamsResponse(BaseModel):
    teams: list[str]


# ── Match mapping ───────────────────────────────────────────────────────────


class UnmappedMatchItem(BaseModel):
    canonical_match_id: int
    team_a_name: str
    team_b_name: str
    team_a_mapping: TeamMappingInfo | None = None
    team_b_mapping: TeamMappingInfo | None = None
    start_time_normalized: str | None = None
    league: str | None = None
    status: str
    bookmakers: list[str] = []


class UnmappedMatchesResponse(BaseModel):
    total: int
    matches: list[UnmappedMatchItem]


class GolggMatchCandidate(BaseModel):
    match_id: str
    team1_name: str
    team2_name: str
    date: str
    team1_win: bool | None = None
    team2_win: bool | None = None


class GolggMatchCandidatesResponse(BaseModel):
    candidates: list[GolggMatchCandidate]


class MatchMappingRequest(BaseModel):
    canonical_match_id: int
    golgg_match_id: str


class MappingCheckResponse(BaseModel):
    is_mapped: bool
    canonical_match_id: int | None = None
    team_a: str | None = None
    team_b: str | None = None
    start_time: str | None = None


class MappingReviewItem(BaseModel):
    canonical_match_id: int
    mapping_id: int
    golgg_match_id: str
    confidence: float
    mapped_by: str | None = None
    canonical_team_a: str
    canonical_team_b: str
    canonical_date: str | None = None
    canonical_competition: str | None = None
    golgg_team_a: str | None = None
    golgg_team_b: str | None = None
    golgg_date: str | None = None
    golgg_competition: str | None = None
    reasons: list[str]
    prediction_count: int
    feature_count: int
    signal_count: int
    bet_count: int


class MappingReviewResponse(BaseModel):
    total: int
    items: list[MappingReviewItem]


class MappingReviewDecisionRequest(BaseModel):
    canonical_match_id: int
    decision: str = Field(pattern="^(retain|replace|invalidate)$")
    reason: str = Field(min_length=8, max_length=2000)
    operator: str = Field(min_length=2, max_length=100)
    new_golgg_match_id: str | None = Field(default=None, max_length=50)


class MappingReviewDecisionResponse(BaseModel):
    decision_id: int
    canonical_match_id: int
    decision: str
    old_golgg_match_id: str | None
    new_golgg_match_id: str | None


# ── Alerts / Notifications ──────────────────────────────────────────────────


class AlertConfigResponse(BaseModel):
    is_enabled: bool
    min_ev: float
    min_odds: float
    max_odds: float | None = None
    cooldown_hours: float
    discord_enabled: bool
    discord_configured: bool
    discord_webhook_url_masked: str | None = None
    telegram_enabled: bool
    telegram_configured: bool
    telegram_bot_token_masked: str | None = None
    telegram_chat_id_masked: str | None = None
    updated_at: str | None = None


class AlertConfigUpdateRequest(BaseModel):
    is_enabled: bool | None = None
    min_ev: float | None = None
    min_odds: float | None = None
    max_odds: float | None = None
    cooldown_hours: float | None = None
    discord_enabled: bool | None = None
    discord_webhook_url: str | None = None
    telegram_enabled: bool | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


class AlertLogEntry(BaseModel):
    id: int
    created_at: str
    canonical_match_id: int | None = None
    match_label: str
    league: str | None = None
    match_start_at: str | None = None
    side: str | None = None
    team_name: str | None = None
    bookmaker_name: str | None = None
    odds: float | None = None
    model_prob: float | None = None
    market_prob: float | None = None
    ev: float | None = None
    suggested_stake: float | None = None
    channels: str
    status: str
    error_message: str | None = None


class AlertHistoryResponse(BaseModel):
    total: int
    alerts: list[AlertLogEntry]


class AlertCheckResponse(BaseModel):
    dispatched: int
    skipped: int
    failed: int
    message: str
    alerts: list[dict[str, Any]]


class AlertTestResponse(BaseModel):
    results: dict[str, Any]


# ── Proposition Markets (IDEA-018) ──────────────────────────────────────────


class PropOddsSnapshotItem(BaseModel):
    id: int
    bookmaker: str
    canonical_match_id: int | None = None
    market_type: str
    map_number: int = 1
    line: float
    target_team: str | None = None
    raw_team_a: str | None = None
    raw_team_b: str | None = None
    odds_over: float | None = None
    odds_under: float | None = None
    prob_over_novig: float | None = None
    prob_under_novig: float | None = None
    margin: float | None = None
    raw_market_name: str | None = None
    scraped_at: str | None = None


class PropOddsTimelineResponse(BaseModel):
    canonical_match_id: int
    market_type: str | None = None
    line: float | None = None
    total_snapshots: int
    timeline: list[PropOddsSnapshotItem]


class PropOddsLatestResponse(BaseModel):
    canonical_match_id: int
    map_number: int = 1
    total_lines: int
    lines: list[PropOddsSnapshotItem]


class PropSignalItem(BaseModel):
    bookmaker: str
    market_type: str
    selection: str
    odds: float
    model_prob: float
    fair_odds: float
    net_ev_tax12: float
    target_team: str | None = None


class PropEvaluatedLineItem(PropOddsSnapshotItem):
    model_prob_over: float | None = None
    model_prob_under: float | None = None
    ev_over_net: float | None = None
    ev_under_net: float | None = None


class PropModelExpectations(BaseModel):
    expected_total_kills: float
    mu_team_a: float
    mu_team_b: float
    expected_spread_a_minus_b: float
    league_avg_kills: float | None = None
    league_pace_category: str | None = None
    distribution: dict[str, Any] | None = None


class MatchPropAnalysisResponse(BaseModel):
    canonical_match_id: int
    saved_prediction_id: int | None = None
    team_a: str
    team_b: str
    league: str | None = None
    map_number: int = 1
    model_expectations: PropModelExpectations
    evaluated_lines: list[PropEvaluatedLineItem]
    signals: list[PropSignalItem]


class IngestPropOddsRequest(BaseModel):
    bookmaker: str
    raw_team_a: str
    raw_team_b: str
    map_number: int = 1
    league: str | None = None
    match_start_time: str | None = None
    source_url: str | None = None
    lines: list[dict[str, Any]]

