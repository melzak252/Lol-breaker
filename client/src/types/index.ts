// Match board types
export interface MatchBoardItem {
  canonical_match_id: number;
  match: string;
  league: string | null;
  start_time_normalized: string | null;
  best_of: number | null;
  team_a_name: string | null;
  team_b_name: string | null;
  team_a_golgg_name: string | null;
  team_b_golgg_name: string | null;
  team_a_mapping_source: string | null;
  team_b_mapping_source: string | null;
  has_unmapped_teams: boolean;
  match_confidence: number | null;
  bookmaker_count: number;
  diagnostic_best_odds_a: number | null;
  diagnostic_best_odds_b: number | null;
  quote_status: 'actionable' | 'diagnostic' | 'unavailable';
  best_odds_a: number | null;
  best_bookmaker_a: string | null;
  avg_odds_a: number | null;
  best_odds_b: number | null;
  best_bookmaker_b: string | null;
  avg_odds_b: number | null;
  arb_no_tax: boolean;
  arb_after_tax: boolean;
  arb_margin_no_tax: number | null;
  arb_margin_after_tax: number | null;
  model_prob_a: number | null;
  model_prob_b: number | null;
  hybrid_prob_a: number | null;
  hybrid_prob_b: number | null;
  hybrid_ev_a: number | null;
  hybrid_ev_b: number | null;
  recommended_side?: string | null;
  recommended_team?: string | null;
  recommended_bookmaker?: string | null;
  recommended_odds?: number | null;
  recommended_ev?: number | null;
  has_props?: boolean;
  prop_lines_count?: number;
  last_scraped_at: string | null;
}

export interface MatchBoardResponse {
  total: number;
  matches: MatchBoardItem[];
}

// Match detail types
export interface BookmakerOddsRow {
  bookmaker: string;
  odds_snapshot_id: number | null;
  quote_scraped_at: string | null;
  quote_status: 'actionable' | 'diagnostic' | 'unavailable';
  raw_team_a: string | null;
  raw_team_b: string | null;
  canonical_odds_a: number | null;
  canonical_odds_b: number | null;
  scraped_at: string | null;
  source_url: string | null;
  offer_url: string | null;
  ev_a: number | null;
  ev_b: number | null;
  kelly_a: number | null;
  kelly_b: number | null;
}

export interface PredictionRow {
  model_name: string;
  model_version: string;
  prob_a: number | null;
  prob_b: number | null;
  predicted_at: string | null;
  raw_diagnostics: Record<string, unknown> | null;
  ev_a: number | null;
  ev_b: number | null;
  kelly_a: number | null;
  kelly_b: number | null;
}

export interface RosterPlayer {
  player_id: string | null;
  player_name: string | null;
  role: string | null;
  champion_name: string | null;
  elo_rating: number | null;
  glicko_rating: number | null;
  glicko_rd: number | null;
  trueskill_rating: number | null;
  rating_uncertainty: number | null;
  games_played: number | null;
}

export interface RosterInfo {
  team_name: string | null;
  source_match_id: string | null;
  source_date: string | null;
  source_tournament: string | null;
  roster_source: string | null;
  avg_elo: number | null;
  avg_glicko: number | null;
  avg_glicko_rd: number | null;
  players_with_rating: number | null;
  players: RosterPlayer[];
}

export interface RosterOverridePlayerInput {
  player_id?: string | null;
  player_name: string;
  role?: string | null;
}

export interface RosterPlayerCandidate {
  player_id: string;
  player_name: string;
  role: string | null;
  team_name: string | null;
  is_expected_team: boolean;
}

export interface MatchRosterOverrideResponse {
  canonical_match_id: number;
  team_side: 'a' | 'b';
  roster: RosterInfo;
  message: string;
}

export interface TeamRecentStats {
  team_name: string | null;
  matches_count: number | null;
  games_count: number | null;
  win_rate: number | null;
  avg_kills: number | null;
  avg_deaths: number | null;
  avg_gd15: number | null;
  avg_dragons: number | null;
  avg_nashors: number | null;
  avg_towers: number | null;
  avg_game_duration: number | null;
  last_match_at: string | null;
}

export interface TeamMappingInfo {
  canonical_name: string | null;
  golgg_name: string | null;
  confidence: number | null;
  source: string | null;
}

export interface TeamComparisonInfo {
  team_a: TeamMappingInfo | null;
  team_b: TeamMappingInfo | null;
  team_a_rating: number | null;
  team_b_rating: number | null;
  rating_system: string | null;
  team_a_elo: number | null;
  team_b_elo: number | null;
  team_a_glicko: number | null;
  team_b_glicko: number | null;
  team_a_glicko_rd: number | null;
  team_b_glicko_rd: number | null;
  team_a_games_played: number | null;
  team_b_games_played: number | null;
  rating_probabilities: Record<string, number>;
}

export interface BettingRecommendation {
  has_value: boolean;
  verdict: string;
  verdict_label: string;
  side: 'a' | 'b' | null;
  recommended_team: string | null;
  opponent_team: string | null;
  bookmaker: string | null;
  best_odds: number | null;
  offer_url: string | null;
  model_prob: number | null;
  hybrid_prob: number | null;
  market_prob: number | null;
  ev: number | null;
  pure_model_ev: number | null;
  edge_percentage_points: number | null;
  min_odds_required: number | null;
  half_kelly: number | null;
  quarter_kelly: number | null;
  suggested_stake_pct: number | null;
  conformal_prob_low: number | null;
  conformal_ev: number | null;
  is_conformal_safe: boolean;
  summary: string;
  reasons: string[];
  threshold_info: string | null;
  logit_disagreement?: number | null;
  thesis_prob?: number | null;
  thesis_disagreement?: number | null;
  thesis_guard_passed?: boolean | null;
  guard_b_passed?: boolean | null;
  raw_clv?: number | null;
}

export interface ParlayLeg {
  canonical_match_id: number;
  match_name: string;
  league?: string | null;
  start_time?: string | null;
  side: 'a' | 'b';
  team_name: string;
  opponent_name: string;
  odds: number;
  model_prob: number;
  single_ev: number;
}

export interface ParlayRecommendation {
  bookmaker: string;
  legs: [ParlayLeg, ParlayLeg];
  combined_odds: number;
  effective_odds: number;
  joint_prob: number;
  ev: number;
  tax_amortization_gain: number;
  quarter_kelly: number;
  suggested_stake: number;
  confidence_badge: string;
  analysis_text: string;
}

export interface ParlayRecommendationsResponse {
  count: number;
  top_parlay: ParlayRecommendation | null;
  parlays: ParlayRecommendation[];
  tax_rate: number;
  explanation: string;
}

export interface MatchDetailResponse {
  canonical_match_id: number;
  team_a_name: string | null;
  team_b_name: string | null;
  league: string | null;
  start_time_normalized: string | null;
  status: string | null;
  best_of: number | null;
  odds: BookmakerOddsRow[];
  predictions: PredictionRow[];
  roster_a: RosterInfo | null;
  roster_b: RosterInfo | null;
  roster_a_is_manual: boolean;
  roster_b_is_manual: boolean;
  recent_stats_a: TeamRecentStats | null;
  recent_stats_b: TeamRecentStats | null;
  team_comparison: TeamComparisonInfo | null;
  recommendation?: BettingRecommendation | null;
}

// System status types
export interface SystemStatusResponse {
  counts: Record<string, number>;
  last_scrape_runs: Array<{
    name: string;
    task_name: string;
    status: string;
    last_scraped_at: string | null;
    started_at: string | null;
    duration_seconds: number | null;
    snapshot_count: number;
  }>;
  last_automation_runs: Array<{
    run_type: string;
    task_name: string;
    status: string;
    started_at: string | null;
    finished_at: string | null;
    duration_seconds: number | null;
    summary: string | null;
  }>;
}

export interface BookmakerStatus {
  id: number;
  name: string;
  base_url: string | null;
  last_scraped_at: string | null;
  last_scrape: string | null;
  snapshot_count: number;
}

// Scheduler types
export interface SchedulerTask {
  id: string;
  task_id: string;
  name: string;
  description: string;
  schedule: string;
  interval_minutes: number | null;
  cron_trigger: string | null;
  enabled: boolean;
}

export interface SchedulerJob {
  id: string;
  name: string;
  enabled: boolean;
  next_run_time: string | null;
  last_run_at: string | null;
  last_run_status: string | null;
  is_running: boolean;
  trigger: string;
  pending: boolean;
}

export interface SchedulerRun {
  id: number;
  run_type: string;
  task_name: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  duration_seconds: number | null;
}

export interface SchedulerCommand {
  id: number;
  command: string;
  status: string;
  step_order: number;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  duration_seconds: number | null;
  output: string | null;
}

export interface SchedulerTriggerResponse {
  task_id: string;
  status: string;
  message: string;
}

// Match movement — NEW FORMAT with deviation % from closing
export interface OddsMovementPoint {
  scraped_at: string;
  hours_before_match: number | null;
  bookmaker: string;
  odds_a: number;
  odds_b: number;
  deviation_a_pct?: number;
  deviation_b_pct?: number;
}

// Horizon accuracy types
export interface HorizonBin {
  label: string;                 // e.g. "0-2h", "2-6h", "6-12h"
  hours_start: number;
  hours_end: number | null;      // null for 48h+ (unbounded)
  snapshot_count: number;
  match_count: number;
  avg_logloss: number | null;
  avg_auc: number | null;
  avg_prob_winner: number | null;
  avg_prob_loser: number | null;
}

export interface ModelReferenceMetrics {
  model_name: string;
  model_version: string;
  avg_logloss: number | null;
  avg_auc: number | null;
  n_matches: number;
}

export interface HybridBinMetrics {
  label: string;
  hours_start: number;
  hours_end: number | null;
  snapshot_count: number;
  match_count: number;
  avg_logloss: number | null;
  avg_auc: number | null;
}

export interface HybridModelBins {
  model_name: string;
  model_version: string;
  bins: HybridBinMetrics[];
}

export interface BookmakerBinMetrics {
  bookmaker_id: number;
  bookmaker_name: string;
  bins: HybridBinMetrics[];
}

export interface MarketCloseCompetitor {
  name: string;
  display_name: string;
  n_matches: number;
  avg_logloss: number | null;
  avg_auc: number | null;
  avg_brier: number | null;
  accuracy: number | null;
  rank: number;
}

export interface MarketCloseBookmaker extends MarketCloseCompetitor {
  bookmaker_id: number;
  bookmaker_name: string;
}

export interface MarketCloseComparison {
  sample_definition: string;
  n_matches: number;
  min_matches: number;
  avg_bookmakers_per_match: number | null;
  model_delta_logloss_vs_market: number | null;
  hybrid_delta_logloss_vs_market: number | null;
  status: 'model_better' | 'model_on_market_level' | 'model_worse' | 'no_data' | 'unknown';
  competitors: MarketCloseCompetitor[];
  bookmakers: MarketCloseBookmaker[];
}

export interface ModelVsBookmakerTest {
  id: string;
  label: string;
  metric: 'logloss' | 'brier' | string;
  model_name: string;
  baseline_name: string;
  alternative: string;
  interpretation: string;
  n: number;
  df: number;
  mean_diff: number;
  sd_diff: number;
  sem_diff: number;
  t_stat: number | null;
  p_value_one_sided: number;
  alpha: number;
  t_critical_95_one_sided: number | null;
  significant: boolean;
}

export interface CalibrationBin {
  bin_index: number;
  bin_low: number;
  bin_high: number;
  label: string;
  avg_predicted: number | null;
  empirical_rate: number | null;
  calibration_error: number | null;
  count: number;
}

export interface MetricSummary {
  n_matches: number;
  avg_logloss: number | null;
  avg_auc: number | null;
  avg_brier: number | null;
  accuracy: number | null;
  ece?: number | null;
  calibration_status?: 'well_calibrated' | 'acceptable' | 'overconfident_miscalibrated' | 'miscalibrated' | 'unknown' | string;
  calibration_bins?: CalibrationBin[];
}

export interface HistoricalModelMetrics extends MetricSummary {
  key: string;
  label: string;
  description?: string;
  model_name: string;
  model_version: string;
  features_version: string;
  temporal_eligible_matches: number;
  segments?: {
    tier_1: MetricSummary | null;
    regional_erl: MetricSummary | null;
  };
  formats?: {
    bo1: MetricSummary | null;
    bo3: MetricSummary | null;
    bo5: MetricSummary | null;
  };
}

export interface ExecutiveInsight {
  id: string;
  type: 'warning' | 'positive' | 'recommendation' | 'neutral';
  title: string;
  text: string;
}

export interface HistoricalModelComparison {
  evaluation_scope: {
    kind: string;
    warning: string;
    temporal_rule: string;
  };
  target_model_key?: string;
  models: HistoricalModelMetrics[];
  common_cohort: {
    n_matches: number;
    target_model_key?: string;
    target_model_title?: string;
    target_model?: MetricSummary | null;
    target_minus_exp039_logloss?: number | null;
    target_minus_exp039_brier?: number | null;
    exp039: MetricSummary | null;
    operational_regional: MetricSummary | null;
    market_closing?: MetricSummary | null;
    disagreement?: {
      n_matches: number;
      disagreement_rate: number;
      model_wins: number;
      market_wins: number;
      model_win_rate: number;
      market_win_rate: number;
    } | null;
    naive_50_50?: MetricSummary | null;
    operational_minus_exp039_logloss: number | null;
    operational_minus_exp039_brier: number | null;
    exp039_vs_naive_logloss?: number | null;
    operational_vs_naive_logloss?: number | null;
    segments?: {
      tier_1: {
        n_matches: number;
        target_model?: MetricSummary | null;
        exp039: MetricSummary | null;
        operational_regional: MetricSummary | null;
      };
      regional_erl: {
        n_matches: number;
        target_model?: MetricSummary | null;
        exp039: MetricSummary | null;
        operational_regional: MetricSummary | null;
      };
    };
  };
  executive_insights?: ExecutiveInsight[];
}

export interface MatchupSimulationRequest {
  team_a_name: string;
  team_b_name: string;
  best_of?: number;
  league?: string;
}

export interface MatchupSimulationResponse {
  team_a_name: string;
  team_b_name: string;
  best_of: number;
  map_prob_a: number | null;
  map_prob_b: number | null;
  series_prob_a: number;
  series_prob_b: number;
  model_name: string;
  model_version: string;
  roster_a: RosterInfo | null;
  roster_b: RosterInfo | null;
  recent_stats_a: TeamRecentStats | null;
  recent_stats_b: TeamRecentStats | null;
  team_comparison: TeamComparisonInfo | null;
  components: Record<string, unknown>;
}

export interface FearlessSimulationResponse {
  canonical_match_id?: number | null;
  team_a_name: string;
  team_b_name: string;
  best_of: number;
  n_simulations: number;
  series_prob_a: number;
  series_prob_b: number;
  scorelines: Record<string, number>;
  scoreline_pmf: Record<string, number>;
  map_win_probabilities: Record<number | string, number>;
  map_reach_probabilities: Record<number | string, number>;
  mean_unique_champions: number;
  champions_drafted_distribution: Record<number | string, number>;
}

export interface TournamentStanding {
  team: string;
  champion_prob: number;
  top2_prob: number;
  top3_prob: number;
  top4_prob: number;
}

export interface BracketMatch {
  id: string;
  name: string;
  round_name: string;
  bracket_section: 'upper' | 'lower' | 'final';
  best_of: number;
  team1: string | null;
  team2: string | null;
  winner: string | null;
  score1: number | null;
  score2: number | null;
}

export interface TournamentSimulationResponse {
  tournament_id: string;
  tournament_name: string;
  simulations: number;
  standings: TournamentStanding[];
  bracket: Record<string, BracketMatch>;
  source?: string;
  status?: string;
  synced_at?: string | null;
  sync_message?: string | null;
  updated_matches?: number;
  cached?: boolean;
  cached_at?: string | null;
  available_depths?: number[];
  provenance?: {
    probability_model?: string;
    prediction_source?: string;
    [key: string]: unknown;
  };
  joint_finalists?: Array<{ pair: string; prob: number }>;
  calibration?: {
    calibrated: boolean;
    mode: string;
    pairwise_cap: number;
    temperature: number;
    entropy_dampening: number;
    regional_gamma: number;
  };
}

export interface TournamentSummary {
  id: string;
  name: string;
  region: string;
  format: string;
  teams: string[];
}

export interface WorldsTeamInput {
  team: string;
  region: string;
  pool?: number;
}

export interface WorldsStanding {
  team: string;
  region: string;
  stage: 'direct_swiss' | 'play_in';
  pool: number | null;
  play_in_qualifier_prob: number;
  champion_prob: number;
  top2_prob: number;
  top4_prob: number;
  top8_swiss_prob: number;
}

export interface WorldsSimulationResponse {
  tournament_id: string;
  tournament_name: string;
  format: string;
  simulations: number;
  teams: string[];
  direct_teams: WorldsTeamInput[];
  play_in_teams: WorldsTeamInput[];
  play_in_winner_pool: number;
  standings: WorldsStanding[];
}

export interface CrystalBallOutcomeItem {
  candidate: string;
  wins: number;
  probability: number;
  fair_odds: number;
}

export interface CrystalBallCategoryResult {
  category_name: string;
  ranking: CrystalBallOutcomeItem[];
}

export interface CrystalBallPercentiles {
  mean: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  min: number;
  max: number;
}

export interface WorldsCrystalBallResponse {
  n_simulations: number;
  player_most_kills_single_game: CrystalBallCategoryResult;
  player_highest_tournament_kda: CrystalBallCategoryResult;
  player_most_unique_champions: CrystalBallCategoryResult;
  player_highest_kill_participation: CrystalBallCategoryResult;
  team_most_unique_champions: CrystalBallCategoryResult;
  team_most_kills_single_game: CrystalBallCategoryResult;
  team_highest_kda: CrystalBallCategoryResult;
  tournament_shortest_game_seconds: CrystalBallPercentiles;
  tournament_longest_game_seconds: CrystalBallPercentiles;
  tournament_total_pentakills: Record<string, number>;
  tournament_total_barons: CrystalBallPercentiles;
  champion_most_picked: CrystalBallCategoryResult;
  champion_most_banned: CrystalBallCategoryResult;
  champion_highest_winrate: CrystalBallCategoryResult;
  champion_most_deaths: CrystalBallCategoryResult;
}

export interface EncSelectedPlayer {
  role: 'TOP' | 'JUNGLE' | 'MID' | 'ADC' | 'SUPPORT';
  player: string;
  normalized_player_id: string;
  rating: number;
  games_played: number;
  rating_source: 'gl' | 'default';
}

export interface EncTeamConfiguration {
  nation: string;
  entry_stage: 'group_stage' | 'play_in';
  ranking: number | null;
  source_roster: string[];
  selected_roster: EncSelectedPlayer[];
  missing_roles: string[];
  selection_status: 'ready' | 'incomplete' | 'defaulted';
  roster_rating: number | null;
}

export interface EncFormat {
  participants: number;
  invited: number;
  direct_group_stage: number;
  online_qualifiers: number;
  wildcards: string[];
  play_in: string;
  group_stage: string;
  playoffs: string;
  draw_and_tiebreak_policy: string;
}

export interface EncConfigurationResponse {
  tournament_id: string;
  tournament_name: string;
  source_url: string;
  format: EncFormat;
  ratings_version: string | null;
  data_cutoff_at: string | null;
  default_rating: number;
  default_rating_policy: string;
  teams: EncTeamConfiguration[];
  simulation_ready: boolean;
  blocking_issues: string[];
}

export interface EncStanding {
  nation: string;
  entry_stage: 'group_stage' | 'play_in';
  roster_rating: number;
  group_stage_prob: number;
  playoff_prob: number;
  top4_prob: number;
  top2_prob: number;
  champion_prob: number;
}

export interface EncSimulationResponse {
  tournament_id: string;
  tournament_name: string;
  format: EncFormat;
  simulations: number;
  standings: EncStanding[];
}

export interface HorizonAccuracyResponse {
  evaluation_scope?: {
    kind: string;
    prediction_source: string;
    prediction_rule: string;
    warning: string;
  };
  total_matches_with_odds: number;
  total_finished_matches: number;
  total_odds_processed: number;
  bins: HorizonBin[];
  min_matches_per_bin: number;
  model_references: ModelReferenceMetrics[];
  hybrid_model_bins: HybridModelBins[];
  bookmaker_bins: BookmakerBinMetrics[];
  model_vs_bookmaker_tests: ModelVsBookmakerTest[];
  market_close_comparison: MarketCloseComparison;
}

// Backend-generated analysis keys are open-ended so archived historical model keys remain readable.
export type ModelAnalysisKey = string

export interface ModelClvBin {
  model_key: ModelAnalysisKey | string;
  model_label: string;
  label: string;
  hours_start: number;
  hours_end: number | null;
  entry_count: number;
  match_count: number;
  observation_count: number;
  avg_hours_before: number | null;
  avg_clv_odds_pct: number | null;
  median_clv_odds_pct: number | null;
  avg_clv_probability_pp: number | null;
  median_clv_probability_pp: number | null;
  positive_clv_rate: number | null;
  avg_ev: number | null;
  aggregation_level: 'model_match_horizon' | string;
}
export interface BookmakerClvBreakdown {
  bookmaker_name: string;
  entry_count: number;
  match_count: number;
  avg_clv_odds_pct: number | null;
  median_clv_odds_pct: number | null;
  positive_clv_rate: number | null;
  positive_clv_count: number;
  avg_clv_probability_pp: number | null;
  avg_ev: number | null;
  avg_taken_odds: number | null;
  avg_closing_odds: number | null;
}

export interface OddsTierClvBreakdown {
  tier_label: string;
  odds_min: number;
  odds_max: number | null;
  entry_count: number;
  match_count: number;
  avg_clv_odds_pct: number | null;
  median_clv_odds_pct: number | null;
  positive_clv_rate: number | null;
  positive_clv_count: number;
  avg_ev: number | null;
  avg_taken_odds: number | null;
  avg_closing_odds: number | null;
}

export interface ModelConditionsSummary {
  best_horizon?: {
    label: string;
    avg_clv_odds_pct: number | null;
    positive_clv_rate: number | null;
    match_count: number;
    entry_count: number;
  };
  worst_horizon?: {
    label: string;
    avg_clv_odds_pct: number | null;
    positive_clv_rate: number | null;
    match_count: number;
    entry_count: number;
  };
  top_bookmakers?: BookmakerClvBreakdown[];
  best_odds_tier?: OddsTierClvBreakdown;
  overall_entry_count?: number;
  overall_matches_count?: number;
  overall_beat_closing_rate?: number | null;
  overall_avg_clv_odds_pct?: number | null;
  recommendations?: string[];
}

export interface ModelClvModelSummary {
  model_key: ModelAnalysisKey | string;
  model_label: string;
  bins: ModelClvBin[];
  bookmaker_breakdown?: BookmakerClvBreakdown[];
  odds_tier_breakdown?: OddsTierClvBreakdown[];
  conditions_summary?: ModelConditionsSummary;
}

export interface ModelClvByHorizonResponse {
  metadata: {
    max_days_back: number;
    max_odds_age_hours: number;
    tax_rate: number;
    min_ev: number;
    aggregation_level: string;
    entry_definition: string;
    closing_definition: string;
    clv_odds_pct_definition: string;
    aggregation_definition?: string;
  };
  total_predictions_scanned: number;
  total_entries: number;
  models: ModelClvModelSummary[];
  bins: ModelClvBin[];
  bookmaker_breakdown?: BookmakerClvBreakdown[];
  odds_tier_breakdown?: OddsTierClvBreakdown[];
  skips: Record<string, number>;
}

export interface AuditOddsBracket {
  bracket_label: string;
  odds_min: number;
  odds_max: number;
  n_bets: number;
  n_wins: number;
  win_rate_pct: number;
  avg_odds: number;
  avg_model_prob_pct: number;
  avg_market_prob_pct: number;
  calibration_bias_pp: number;
  expected_net_roi_pct: number;
  realized_net_roi_pct: number;
  realized_gross_roi_pct: number;
  pnl_units: number;
  avg_clv_pct: number;
  status: 'RECOMMENDED' | 'QUARANTINE' | 'NORMAL';
  flags: string[];
}

export interface AuditSeriesFormat {
  format_label: string;
  n_bets: number;
  n_wins: number;
  win_rate_pct: number;
  avg_odds: number;
  avg_model_prob_pct: number;
  avg_market_prob_pct: number;
  calibration_bias_pp: number;
  realized_net_roi_pct: number;
  pnl_units: number;
}

export interface AuditLeague {
  league: string;
  n_bets: number;
  n_wins: number;
  win_rate_pct: number;
  avg_odds: number;
  avg_model_prob_pct: number;
  avg_market_prob_pct: number;
  calibration_bias_pp: number;
  realized_net_roi_pct: number;
  pnl_units: number;
}

export interface AuditAnomaly {
  match_id: number;
  match_title: string;
  bet_side: string;
  team_selected: string;
  date: string;
  league: string;
  best_of: string;
  odds: number;
  odds_close: number;
  model_prob_pct: number;
  market_prob_pct: number;
  discrepancy_pp: number;
  player_consensus_pct: number | null;
  team_consensus_pct: number | null;
  idi_pct: number | null;
  won: boolean;
  pnl_net: number;
  reasons: string[];
}

export interface ModelProfitabilityAuditResponse {
  model_key: string;
  model_title: string;
  parameters: {
    tax_rate: number;
    min_ev: number;
    max_days_back: number;
    timing_window: string;
  };
  overall: {
    total_bets: number;
    n_wins: number;
    win_rate_pct: number;
    avg_odds: number;
    avg_model_prob_pct: number;
    avg_market_prob_pct: number;
    calibration_bias_pp: number;
    realized_net_roi_pct: number;
    realized_gross_roi_pct: number;
    expected_net_roi_pct: number;
    pnl_net_units: number;
    max_drawdown_units: number;
    brier_score: number;
    market_brier_score: number;
    log_loss: number;
    market_log_loss: number;
    positive_clv_pct: number;
    avg_clv_pct: number;
  } | null;
  filter_impact: {
    raw_bets: number;
    raw_roi_net_pct: number;
    raw_pnl_units: number;
    quarantined_bets: number;
    quarantined_roi_net_pct: number;
    quarantined_pnl_units: number;
    filtered_bets: number;
    filtered_roi_net_pct: number;
    filtered_pnl_units: number;
    pnl_improvement_units: number;
  } | null;
  odds_brackets: AuditOddsBracket[];
  series_formats: AuditSeriesFormat[];
  leagues: AuditLeague[];
  anomalies: AuditAnomaly[];
  operational_rules: string[];
  error?: string;
}

// Prediction & EV history timeline
export interface PredictionHistoryPoint {
  timestamp: string;
  model_name: string;
  model_version: string;
  prob_a: number | null;
  prob_b: number | null;
  avg_odds_a: number | null;
  avg_odds_b: number | null;
  market_prob_a: number | null;
  market_prob_b: number | null;
  ev_a: number | null;
  ev_b: number | null;
  raw_diagnostics: Record<string, unknown> | null;
}

export interface MatchMovementResponse {
  match_id: number;
  team_a: string | null;
  team_b: string | null;
  start_time: string | null;
  movement_points: OddsMovementPoint[];
  summary: {
    total_snapshots?: number;
    first_snapshot?: string;
    last_snapshot?: string;
    opening_odds_a?: number;
    opening_odds_b?: number;
    closing_odds_a?: number;
    closing_odds_b?: number;
    opening_deviation_a_pct?: number;
    opening_deviation_b_pct?: number;
    total_drift_a?: number;
    total_drift_b?: number;
    message?: string;
  };
}

// Match result types
export interface BookmakerEvSide {
  ev: number | null;
  odds: number | null;
  model_prob: number | null;
  kelly: number | null;
  market_prob?: number | null;
  odds_snapshot_id?: number | null;
  scraped_at?: string | null;
}

export interface BookmakerEvDetail {
  side_a: BookmakerEvSide;
  side_b: BookmakerEvSide;
}

export interface MatchResultItem {
  canonical_match_id: number;
  probability_basis: 'historical_quote_recomputation';
  probability_unavailable_reason: string | null;
  team_a_name: string | null;
  team_b_name: string | null;
  league: string | null;
  start_time_normalized: string | null;
  best_of: number | null;
  status: string | null;
  winner_name: string | null;
  loser_name: string | null;
  winner_side: string | null;
  team_a_score: number | null;
  team_b_score: number | null;
  result_source: string | null;
  result_recorded_at: string | null;
  best_ev_a: number | null;
  best_ev_b: number | null;
  bookmakers_with_ev: string[];
  best_odds_a: number | null;
  best_odds_b: number | null;
  model_prob_a: number | null;
  model_prob_b: number | null;
  kelly_a: number | null;
  kelly_b: number | null;
  bookmaker_ev_details: Record<string, BookmakerEvDetail>;
  model_name?: string | null;
  model_version?: string | null;
  odds_mode?: string | null;
}

export interface MatchResultsResponse {
  total: number;
  days_back: number;
  model_name?: string | null;
  model_version?: string | null;
  odds_mode?: string | null;
  results: MatchResultItem[];
}

export interface FinancialBucket {
  key: string;
  label: string;
  bets: number;
  matches: number;
  staked: number;
  profit: number;
  roi: number | null;
  hit_rate: number | null;
  avg_ev: number | null;
  avg_clv_odds_pct: number | null;
  median_clv_odds_pct: number | null;
}

export interface FinancialLedgerEntry {
  canonical_match_id: number;
  start_time: string | null;
  league: string | null;
  team_a_name: string | null;
  team_b_name: string | null;
  bookmaker: string;
  side: string;
  entry_odds: number;
  close_odds: number | null;
  hours_before: number | null;
  horizon: string;
  model_prob: number;
  target_prob: number | null;
  uncertainty_basis: 'mean' | 'measured_bounds' | 'assumed_research_haircut' | 'missing_bounds';
  uncertainty_reason: string;
  market_prob: number;
  ev: number;
  stake: number;
  profit: number;
  bankroll_after: number;
  won: boolean;
  clv_odds_pct: number | null;
  entry_scraped_at: string | null;
  close_scraped_at: string | null;
}

export interface FinancialAnalysisResponse {
  methodology: string;
  evaluation_kind: 'historical_quote_simulation';
  result_status: 'computed' | 'no_eligible_bets';
  uncertainty_basis: 'mean' | 'measured_bounds' | 'assumed_research_haircut' | 'missing_bounds';
  uncertainty_reason: string;
  historical_diagnostics_status: 'available' | 'unknown';
  data_scope: string;
  days_back: number;
  odds_mode: string;
  model_name: string;
  model_version: string;
  staking_mode: string;
  min_ev: number;
  initial_bankroll: number;
  final_bankroll: number;
  total_bets: number;
  total_matches: number;
  total_staked: number;
  total_profit: number;
  roi: number | null;
  hit_rate: number | null;
  max_drawdown_pct: number | null;
  avg_clv_odds_pct: number | null;
  positive_clv_rate: number | null;
  max_open_stake: number;
  max_open_bets: number;
  temporal_exclusions: Record<string, number>;
  horizon_buckets: FinancialBucket[];
  bookmaker_buckets: FinancialBucket[];
  league_buckets: FinancialBucket[];
  bankroll_curve: Array<{ index: number; bankroll: number; profit: number; start_time: string | null; canonical_match_id?: number | null }>;
  ledger: FinancialLedgerEntry[];
}

// Team and player rankings
export type RankingEntityType = 'team' | 'player';
export type RatingSystem = 'unified' | 'elo' | 'gl' | 'ts' | 'os' | 'pl' | 'tm';
export type RankingSquadScope = 'major' | 'regional_academy' | 'regional' | 'main' | 'development' | 'all';

export interface RankingEntry {
  rank: number;
  entity_type: RankingEntityType;
  entity_name: string;
  normalized_entity_name: string;
  team_name: string | null;
  role: string | null;
  rating_system: string;
  rating_value: number;
  rd: number | null;
  sigma: number | null;
  system_count: number;
  games_played: number;
  last_match_at: string | null;
  region_family: string | null;
  region_tier: string | null;
  regional_offset: number | null;
  regional_uncertainty: number | null;
}

export interface RankingsResponse {
  entity_type: RankingEntityType;
  rating_system: string;
  ratings_version: string | null;
  data_cutoff_at: string | null;
  snapshot_at: string | null;
  active_since: string | null;
  squad_scope: RankingSquadScope;
  total: number;
  available_rating_systems: string[];
  rankings: RankingEntry[];
}

// Alias mapping types
export interface AliasCreateRequest {
  raw_name: string;
  golgg_team_name: string;
  source_system?: string;
  league_pattern?: string;
  valid_from?: string;
  valid_to?: string;
}

export interface AliasCreateResponse {
  id: number;
  normalized_name: string;
  alias: string;
  source: string;
}

export interface AliasDeleteRequest {
  raw_name: string;
}

export interface AliasBlockRequest {
  raw_name: string;
}

export interface GolggTeamsResponse {
  teams: string[];
}

// ─── Bootstrap Analysis Types ─────────────────────────────────────────

export interface HorizonBootstrapBin {
  model_label: string;
  model_name: string;
  comparison: string;
  label: string;
  hours_start: number | null;
  hours_end: number | null;
  sample_size: number | null;
  n_blocks: number | null;
  model_logloss: number | null;
  benchmark_logloss: number | null;
  observed_difference: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p_one_sided: number | null;
  significant_05: boolean;
}

export interface HorizonBootstrapMonthly {
  month: string;
  n_snapshots: number | null;
  n_matches: number | null;
  model_logloss: number | null;
  bookmaker_logloss: number | null;
  mean_difference: number | null;
  horizon_bin: string;
  model_label: string;
}

export interface HorizonBootstrapResponse {
  metadata?: {
    aggregation_level?: string;
    sample_size_definition?: string;
    bootstrap_unit?: string;
    snapshot_role?: string;
  };
  bins: HorizonBootstrapBin[];
  monthly: HorizonBootstrapMonthly[];
  last_updated: string | null;
  plot_available: boolean;
  match_stats: {
    upcoming: number | null;
    finished: number | null;
    expired: number | null;
  };
}

// Match mapping types
export interface UnmappedMatchItem {
  canonical_match_id: number;
  team_a_name: string;
  team_b_name: string;
  team_a_mapping?: TeamMappingInfo | null;
  team_b_mapping?: TeamMappingInfo | null;
  start_time_normalized: string | null;
  league: string | null;
  status: string;
  bookmakers?: string[];
}

export interface UnmappedMatchesResponse {
  total: number;
  matches: UnmappedMatchItem[];
}

export interface GolggMatchCandidate {
  match_id: number;
  team1_name: string;
  team2_name: string;
  date: string;
  team1_win: boolean | null;
  team2_win: boolean | null;
}

export interface GolggMatchCandidatesResponse {
  candidates: GolggMatchCandidate[];
}

export interface MatchMappingRequest {
  canonical_match_id: number;
  golgg_match_id: number;
}

export interface MappingCheckResponse {
  is_mapped: boolean;
  canonical_match_id: number | null;
  team_a: string | null;
  team_b: string | null;
  start_time: string | null;
}

export interface MappingReviewItem {
  canonical_match_id: number;
  mapping_id: number;
  golgg_match_id: string;
  confidence: number;
  mapped_by: string | null;
  canonical_team_a: string;
  canonical_team_b: string;
  canonical_date: string | null;
  canonical_competition: string | null;
  golgg_team_a: string | null;
  golgg_team_b: string | null;
  golgg_date: string | null;
  golgg_competition: string | null;
  reasons: string[];
  prediction_count: number;
  feature_count: number;
  signal_count: number;
  bet_count: number;
}

export interface MappingReviewResponse {
  total: number;
  items: MappingReviewItem[];
}

// ─── Embedding Diagnostics ───────────────────────────────────────────

export interface ChampionEmbeddingPoint {
  champion_id: string;
  champion_name: string;
  role: string | null;
  x: number;
  y: number;
  x_norm: number;
  y_norm: number;
  n_games: number | null;
  recent_games: number | null;
  recent_window_days: number | null;
  recent_date_max: string | null;
  win_rate: number | null;
  fallback_level: string | null;
  window_days: number | null;
  shrinkage_weight_observed: number | null;
  age_days_mean: number | null;
  kda: number | null;
  damage_share: number | null;
  gold_share: number | null;
  kill_participation: number | null;
  cluster_id: number | null;
  cluster_label: string | null;
}

export interface ChampionEmbeddingProjectionResponse {
  metadata: {
    artifact_path: string;
    method: 'umap' | 'tsne' | 'pca';
    requested_method: 'umap' | 'tsne' | 'pca';
    preset: 'local' | 'balanced' | 'global';
    preset_config: {
      label: string;
      description: string;
      umap_n_neighbors: number;
      umap_min_dist: number;
      umap_metric: string;
      tsne_perplexity: number;
    };
    projection_warning?: string | null;
    snapshot: string;
    available_snapshots: string[];
    role: string;
    min_games: number;
    min_games_column: string;
    recent_window_days: number | null;
    total_points: number;
    cluster_count: number;
    cluster_counts: Record<string, number>;
    available_roles: string[];
    source_rows: number | null;
    reference_date: string | null;
    embedding_dim: number | null;
    model_name: string;
    model_version: string;
    fallback_counts: Record<string, number>;
  };
  points: ChampionEmbeddingPoint[];
}

// ─── Player Comparison & Rating Trajectory ──────────────────────────────

export interface PlayerSearchItem {
  player_id: string;
  player_name: string;
  team_name: string | null;
  role: string | null;
  games_played: number;
  current_elo: number | null;
  current_gl: number | null;
  last_match_at: string | null;
}

export interface TopChampionItem {
  champion_name: string;
  games: number;
  wins: number;
  win_rate: number;
}

export interface PlayerSystemRating {
  system: string;
  rating_value: number;
  rd: number | null;
  sigma: number | null;
  rank: number | null;
  percentile: number | null;
}

export interface PlayerProfileDetail {
  player_id: string;
  player_name: string;
  team_name: string | null;
  role: string | null;
  games_played: number;
  career_wins: number;
  career_losses: number;
  career_win_rate: number;
  career_first_date: string | null;
  career_last_date: string | null;
  career_years: number | null;
  teams: string[];
  top_champions: TopChampionItem[];
  ratings: Record<string, PlayerSystemRating>;
  peak_elo: number | null;
  peak_elo_date: string | null;
  peak_gl: number | null;
  peak_gl_date: string | null;
}

export interface H2HGameItem {
  game_id: string;
  match_id: string;
  date: string | null;
  tournament_name: string | null;
  team_a: string | null;
  champ_a: string | null;
  team_b: string | null;
  champ_b: string | null;
  winner: 'a' | 'b';
}

export interface H2HSummary {
  total_games: number;
  wins_a: number;
  wins_b: number;
  win_rate_a: number;
  win_rate_b: number;
  recent_games: H2HGameItem[];
}

export interface SystemAdvantage {
  system: string;
  system_label: string;
  value_a: number;
  value_b: number;
  difference: number;
  favors: 'a' | 'b' | 'tied';
  win_prob_a: number;
}

export interface ModelVerdict {
  better_player: 'a' | 'b' | 'tied';
  better_player_id: string | null;
  better_player_name: string | null;
  win_probability_a: number;
  win_probability_b: number;
  systems_favor_a: number;
  systems_favor_b: number;
  systems_tied: number;
  total_systems: number;
  advantage_summary: string;
  system_advantages: SystemAdvantage[];
  h2h_winner: 'a' | 'b' | 'tied';
  summary_pl: string;
}

export interface RatingTimelinePoint {
  date: string;
  match_id: string | null;
  team_name: string | null;
  games_count: number;
  elo: number;
  gl: number;
  gl_rd: number | null;
  ts_mu: number | null;
  os_mu: number | null;
  pl_mu: number | null;
  tm_mu: number | null;
}

export interface PlayerComparisonResponse {
  player_a: PlayerProfileDetail;
  player_b: PlayerProfileDetail;
  verdict: ModelVerdict;
  h2h: H2HSummary;
  timeline_a: RatingTimelinePoint[];
  timeline_b: RatingTimelinePoint[];
  available_rating_systems: string[];
}

export interface AlertConfigResponse {
  is_enabled: boolean;
  min_ev: number;
  min_odds: number;
  max_odds: number | null;
  cooldown_hours: number;
  discord_enabled: boolean;
  discord_configured: boolean;
  discord_webhook_url_masked: string | null;
  telegram_enabled: boolean;
  telegram_configured: boolean;
  telegram_bot_token_masked: string | null;
  telegram_chat_id_masked: string | null;
  updated_at: string | null;
}

export interface AlertConfigUpdateRequest {
  is_enabled?: boolean;
  min_ev?: number;
  min_odds?: number;
  max_odds?: number;
  cooldown_hours?: number;
  discord_enabled?: boolean;
  discord_webhook_url?: string;
  telegram_enabled?: boolean;
  telegram_bot_token?: string;
  telegram_chat_id?: string;
}

export interface AlertLogEntry {
  id: number;
  created_at: string;
  canonical_match_id: number | null;
  match_label: string;
  league: string | null;
  match_start_at: string | null;
  side: string | null;
  team_name: string | null;
  bookmaker_name: string | null;
  odds: number | null;
  model_prob: number | null;
  market_prob: number | null;
  ev: number | null;
  suggested_stake: number | null;
  channels: string;
  status: string;
  error_message: string | null;
}

export interface AlertHistoryResponse {
  total: number;
  alerts: AlertLogEntry[];
}

export interface AlertDispatchedItem {
  match: string;
  team: string;
  bookmaker: string;
  odds: number;
  ev: number;
  channels: string[];
  status: string;
}

export interface AlertCheckResponse {
  dispatched: number;
  skipped: number;
  failed: number;
  message: string;
  alerts: AlertDispatchedItem[];
}

export interface AlertChannelTestResult {
  ok?: boolean;
  error?: string | null;
}

export interface AlertTestResponse {
  results: Record<string, AlertChannelTestResult>;
}

// --- Proposition Odds & In-Game Models (IDEA-018) --------------------------

export interface PropOddsSnapshotItem {
  id: number;
  bookmaker: string;
  canonical_match_id: number;
  market_type: string;
  map_number: number;
  line: number;
  target_team: string | null;
  odds_over: number | null;
  odds_under: number | null;
  margin: number | null;
  prob_over_novig: number | null;
  prob_under_novig: number | null;
  is_live: boolean;
  scraped_at: string | null;
  source_url: string | null;
  raw_market_name: string | null;
}

export interface PropSignalItem {
  bookmaker: string;
  market_type: string;
  selection: string;
  odds: number;
  model_prob: number;
  fair_odds: number;
  net_ev_tax12: number;
  target_team: string | null;
}

export interface PropEvaluatedLineItem extends PropOddsSnapshotItem {
  model_prob_over: number | null;
  model_prob_under: number | null;
  ev_over_net: number | null;
  ev_under_net: number | null;
}

export interface PropDistributionLine {
  line: number;
  prob_over: number;
  prob_under: number;
  fair_odds_over: number | null;
  fair_odds_under: number | null;
}

export interface PropDensityPoint {
  kills: number;
  prob: number;
}

export interface PropDistributionData {
  total_lines?: PropDistributionLine[];
  density_curve?: PropDensityPoint[];
  quantiles?: {
    p10?: number;
    p25?: number;
    p50?: number;
    p75?: number;
    p90?: number;
  };
}

export interface PropModelExpectations {
  expected_total_kills: number;
  mu_team_a: number;
  mu_team_b: number;
  expected_spread_a_minus_b: number;
  league_avg_kills?: number | null;
  league_pace_category?: string | null;
  distribution?: PropDistributionData | null;
}

export interface MatchPropAnalysisResponse {
  canonical_match_id: number;
  saved_prediction_id?: number | null;
  team_a: string;
  team_b: string;
  league?: string | null;
  map_number: number;
  model_expectations: PropModelExpectations;
  evaluated_lines: PropEvaluatedLineItem[];
  signals: PropSignalItem[];
}

export interface PropOddsTimelineResponse {
  canonical_match_id: number;
  market_type: string;
  map_number: number;
  total_snapshots: number;
  timeline: PropOddsSnapshotItem[];
}

export interface PropOddsLatestResponse {
  canonical_match_id: number;
  map_number: number;
  total_lines: number;
  lines: PropOddsSnapshotItem[];
}

export interface SimilarPlayerSummary {
  player_name: string;
  team_name: string;
  role: string;
  similarity: number;
}

export interface PlayerEmbeddingPoint {
  player_name: string;
  team_name: string;
  role: string;
  x: number;
  y: number;
  cluster: number;
  is_major?: boolean;
  n_games: number;
  glicko_rating: number;
  stats: {
    kda: number;
    dpm: number;
    dmg_pct: number;
    gold_pct: number;
    csm: number;
    vspm: number;
    gd15: number;
    csd15: number;
    xpd15: number;
    kp_pct: number;
    mitigated: number;
    [key: string]: number | undefined;
  };
  similar_players?: SimilarPlayerSummary[];
}

export interface PlayerClusterSummary {
  cluster_id: number;
  size: number;
  pct: number;
  dominant_role?: string;
  label?: string;
  characteristic_features?: Array<{
    feature: string;
    label: string;
    level: string;
    direction: string;
    z_score: number;
    pct_diff: number;
  }>;
  [key: string]: unknown;
}

export interface PlayerEmbeddingProjectionResponse {
  points: PlayerEmbeddingPoint[];
  summary: {
    method: string;
    preset: string;
    requested_role: string;
    requested_tier: string;
    major_count: number;
    days_active: number | null;
    total_matching_filters?: number;
    total_points: number;
    cluster_count: number;
    cluster_algorithm: string;
    clusters?: PlayerClusterSummary[];
    [key: string]: unknown;
  };
}

export interface EVSignal {
  canonical_match_id: number;
  match: string;
  league: string | null;
  start_time_normalized: string | null;
  model_name: string;
  model_version: string;
  side: 'a' | 'b';
  odds: number;
  bookmaker: string;
  model_prob: number;
  prob_conservative: number | null;
  prob_for_ev: number;
  uncertainty_basis: 'mean' | 'measured_bounds' | 'assumed_research_haircut' | 'missing_bounds';
  uncertainty_reason: string;
  market_prob: number | null;
  ev: number;
  kelly: number;
  offer_url: string | null;
  odds_snapshot_id: number;
  quote_scraped_at: string;
  evaluated_at: string;
}
