import type {
  MatchBoardResponse,
  MatchDetailResponse,
  MatchRosterOverrideResponse,
  RosterPlayerCandidate,
  MatchResultsResponse,
  SystemStatusResponse,
  BookmakerStatus,
  SchedulerTask,
  SchedulerJob,
  SchedulerRun,
  SchedulerCommand,
  SchedulerTriggerResponse,
  MatchMovementResponse,
  HorizonAccuracyResponse,
  HistoricalModelComparison,
  PredictionHistoryPoint,
  AliasCreateRequest,
  AliasCreateResponse,
  GolggTeamsResponse,
  HorizonBootstrapResponse,
  ModelClvByHorizonResponse,
  ModelProfitabilityAuditResponse,
  ChampionEmbeddingProjectionResponse,
  PlayerEmbeddingProjectionResponse,
  FinancialAnalysisResponse,
  RankingEntityType,
  RankingsResponse,
  RankingSquadScope,
  RatingSystem,
  EncConfigurationResponse,
  EncSimulationResponse,
  MatchupSimulationResponse,
  FearlessSimulationResponse,
  TournamentSimulationResponse,
  TournamentSummary,
  WorldsSimulationResponse,
  WorldsCrystalBallResponse,
  WorldsTeamInput,
  PlayerSearchItem,
  PlayerProfileDetail,
  RatingTimelinePoint,
  PlayerComparisonResponse,
  AlertConfigResponse,
  AlertConfigUpdateRequest,
  AlertHistoryResponse,
  AlertCheckResponse,
  AlertTestResponse,
  MatchPropAnalysisResponse,
  PropOddsTimelineResponse,
  PropOddsLatestResponse,
  ParlayRecommendationsResponse,
} from '../types';

const API_BASE = '/api';

// ─── Matches ────────────────────────────────────────────────

export async function fetchMatches(
  minBooks: number = 1,
  daysAhead: number = 14,
  bookmaker?: string,
  signal?: AbortSignal
): Promise<MatchBoardResponse> {
  const params = new URLSearchParams({
    min_books: minBooks.toString(),
    days_ahead: daysAhead.toString(),
  });
  if (bookmaker) params.set('bookmaker', bookmaker);
  const response = await fetch(`${API_BASE}/matches?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch matches: ${response.statusText}`);
  return response.json();
}

export async function fetchMatchDetail(matchId: number, signal?: AbortSignal): Promise<MatchDetailResponse> {
  const response = await fetch(`${API_BASE}/matches/${matchId}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch match ${matchId}: ${response.statusText}`);
  return response.json();
}

export async function fetchParlayRecommendations(
  bookmaker?: string,
  signal?: AbortSignal
): Promise<ParlayRecommendationsResponse> {
  const params = new URLSearchParams();
  if (bookmaker) params.set('bookmaker', bookmaker);
  const response = await fetch(`${API_BASE}/matches/recommendations/parlays?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch parlay recommendations: ${response.statusText}`);
  return response.json();
}

export async function updateMatchBestOf(matchId: number, bestOf: number): Promise<{ best_of: number }> {
  const response = await fetch(`${API_BASE}/matches/${matchId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ best_of: bestOf }),
  });
  if (!response.ok) throw new Error(`Failed to update best_of: ${await response.text()}`);
  return response.json();
}

export async function updateMatchRoster(
  matchId: number,
  teamSide: 'a' | 'b',
  players: Array<{ player_id?: string | null; player_name: string; role?: string | null }>,
): Promise<MatchRosterOverrideResponse> {
  const response = await fetch(`${API_BASE}/matches/${matchId}/roster`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ team_side: teamSide, players }),
  });
  if (!response.ok) throw new Error(`Failed to update roster: ${await response.text()}`);
  return response.json();
}

export async function resetMatchRoster(matchId: number, teamSide: 'a' | 'b'): Promise<{ ok: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/matches/${matchId}/roster/${teamSide}`, { method: 'DELETE' });
  if (!response.ok) throw new Error(`Failed to reset roster: ${await response.text()}`);
  return response.json();
}

export async function searchRosterPlayers(
  matchId: number,
  teamSide: 'a' | 'b',
  query: string,
  role?: string | null,
  signal?: AbortSignal,
): Promise<RosterPlayerCandidate[]> {
  const params = new URLSearchParams({ query, team_side: teamSide });
  if (role) params.set('role', role);
  const response = await fetch(`${API_BASE}/matches/${matchId}/roster/players?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to search GOL.GG players: ${await response.text()}`);
  const data = await response.json();
  return data.players || [];
}

export async function fetchMatchResults(
  daysBack: number = 30,
  oddsMode: string = 'close',
  modelName: string = 'Hybrid-Operational-Market',
  modelVersion: string = 'c0-native-2026-w32-e12-v1-a0.50-t1.00',
  signal?: AbortSignal
): Promise<MatchResultsResponse> {
  const params = new URLSearchParams({
    days_back: daysBack.toString(),
    odds_mode: oddsMode,
    model_name: modelName,
    model_version: modelVersion,
  });
  const response = await fetch(`${API_BASE}/matches/results?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch match results: ${response.statusText}`);
  return response.json();
}

export async function fetchFinancialAnalysis(options: {
  daysBack: number;
  oddsMode: string;
  stakingMode: string;
  minEv: number;
  initialBankroll: number;
  fixedStake: number;
  modelName: string;
  modelVersion: string;
  dataScope: string;
}, signal?: AbortSignal): Promise<FinancialAnalysisResponse> {
  const params = new URLSearchParams({
    days_back: options.daysBack.toString(),
    odds_mode: options.oddsMode,
    staking_mode: options.stakingMode,
    min_ev: options.minEv.toString(),
    initial_bankroll: options.initialBankroll.toString(),
    fixed_stake: options.fixedStake.toString(),
    model_name: options.modelName,
    model_version: options.modelVersion,
    data_scope: options.dataScope,
  });
  const response = await fetch(`${API_BASE}/financial/analysis?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch financial analysis: ${response.statusText}`);
  return response.json();
}

// ─── Team and player rankings ───────────────────────────────

export async function fetchRankings(options: {
  entityType: RankingEntityType;
  ratingSystem: RatingSystem;
  search?: string;
  minGames?: number;
  activeWithinMonths?: number;
  squadScope?: RankingSquadScope;
  limit?: number;
  signal?: AbortSignal;
}): Promise<RankingsResponse> {
  const params = new URLSearchParams({
    entity_type: options.entityType,
    rating_system: options.ratingSystem,
    min_games: String(options.minGames ?? 1),
    active_within_months: String(options.activeWithinMonths ?? 6),
    squad_scope: options.squadScope ?? 'major',
    limit: String(options.limit ?? 100),
  });
  if (options.search?.trim()) params.set('search', options.search.trim());
  const response = await fetch(`${API_BASE}/rankings?${params}`, { signal: options.signal });
  if (!response.ok) throw new Error(`Failed to fetch rankings: ${response.statusText}`);
  return response.json();
}

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error(`Failed to fetch health: ${response.statusText}`);
  return response.json();
}

// ─── System ─────────────────────────────────────────────────

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  const response = await fetch(`${API_BASE}/system/status`);
  if (!response.ok) throw new Error(`Failed to fetch system status: ${response.statusText}`);
  return response.json();
}

export async function fetchBookmakers(signal?: AbortSignal): Promise<BookmakerStatus[]> {
  const response = await fetch(`${API_BASE}/bookmakers`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch bookmakers: ${response.statusText}`);
  return response.json();
}

export async function triggerLightCycle(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/automation/light-cycle`, { method: 'POST' });
  if (!response.ok) throw new Error(`Failed to trigger light cycle: ${response.statusText}`);
  return response.json();
}

export async function triggerBackup(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/automation/backup`, { method: 'POST' });
  if (!response.ok) throw new Error(`Failed to trigger backup: ${response.statusText}`);
  return response.json();
}

// ─── Scheduler ──────────────────────────────────────────────

export async function fetchSchedulerTasks(): Promise<SchedulerTask[]> {
  const response = await fetch(`${API_BASE}/scheduler/tasks`);
  if (!response.ok) throw new Error(`Failed to fetch scheduler tasks: ${response.statusText}`);
  return response.json();
}

export async function fetchSchedulerJobs(): Promise<SchedulerJob[]> {
  const response = await fetch(`${API_BASE}/scheduler/jobs`);
  if (!response.ok) throw new Error(`Failed to fetch scheduler jobs: ${response.statusText}`);
  return response.json();
}

export async function fetchSchedulerRuns(): Promise<SchedulerRun[]> {
  const response = await fetch(`${API_BASE}/scheduler/runs`);
  if (!response.ok) throw new Error(`Failed to fetch scheduler runs: ${response.statusText}`);
  return response.json();
}

export async function fetchSchedulerRunCommands(runId: number): Promise<SchedulerCommand[]> {
  const response = await fetch(`${API_BASE}/scheduler/runs/${runId}/commands`);
  if (!response.ok) throw new Error(`Failed to fetch run commands: ${response.statusText}`);
  return response.json();
}

export async function triggerSchedulerTask(taskId: string): Promise<SchedulerTriggerResponse> {
  const response = await fetch(`${API_BASE}/scheduler/trigger/${taskId}`, { method: 'POST' });
  if (!response.ok) throw new Error(`Failed to trigger task: ${response.statusText}`);
  return response.json();
}

// ─── Predict ─────────────────────────────────────────────────

export interface PredictResult {
  status: string;
  message: string;
  prob_a: number | null;
  prob_b: number | null;
  hybrid_prob_a: number | null;
  hybrid_prob_b: number | null;
  model_name: string | null;
  model_version: string | null;
  diagnostics: Record<string, unknown> | null;
}

export async function predictMatch(matchId: number): Promise<PredictResult> {
  const response = await fetch(`${API_BASE}/matches/${matchId}/predict`, { method: 'POST' });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to predict match ${matchId}: ${response.statusText}`);
  }
  return response.json();
}

export async function simulateMatchup(payload: {
  team_a_name: string;
  team_b_name: string;
  team_a_team_row_id?: number;
  team_b_team_row_id?: number;
  native_team_a_id?: string;
  native_team_b_id?: string;
  best_of?: number;
  league?: string;
}): Promise<MatchupSimulationResponse> {
  const response = await fetch(`${API_BASE}/matches/matchup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Nie udało się przeprowadzić symulacji: ${response.statusText}`);
  }
  return response.json();
}

export async function simulateFearlessMatch(
  matchId: number,
  payload?: { best_of?: number; n_simulations?: number; seed?: number },
): Promise<FearlessSimulationResponse> {
  const response = await fetch(`${API_BASE}/matches/${matchId}/simulate-fearless`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Nie udało się przeprowadzić symulacji Fearless: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchActiveTeams(): Promise<{ teams: Array<{ team_row_id: number; native_team_id: string | null; name: string; rating: number | null; games?: number | null; last_active?: string | null }> }> {
  const response = await fetch(`${API_BASE}/matches/active-teams`);
  if (!response.ok) {
    throw new Error(`Nie udało się pobrać listy aktywnych drużyn: ${response.statusText}`);
  }
  return response.json();
}

// ─── Prediction History ──────────────────────────────────────

export async function fetchPredictionHistory(matchId: number, signal?: AbortSignal): Promise<PredictionHistoryPoint[]> {
  const response = await fetch(`${API_BASE}/matches/${matchId}/prediction-history`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch prediction history: ${response.statusText}`);
  return response.json();
}

// ─── Timing Analysis ────────────────────────────────────────

export async function fetchMatchMovement(matchId: number): Promise<MatchMovementResponse> {
  const response = await fetch(`${API_BASE}/timing/match/${matchId}/movement`);
  if (!response.ok) throw new Error(`Failed to fetch match movement: ${response.statusText}`);
  return response.json();
}

export async function fetchHorizonAccuracy(
  daysBack: number = 90,
  minMatchesPerBin: number = 10
): Promise<HorizonAccuracyResponse> {
  const params = new URLSearchParams({
    max_days_back: daysBack.toString(),
    min_matches_per_bin: minMatchesPerBin.toString(),
  });
  const response = await fetch(`${API_BASE}/timing/horizon-accuracy?${params}`);
  if (!response.ok) throw new Error(`Failed to fetch horizon accuracy: ${response.statusText}`);
  return response.json();
}

export async function fetchHistoricalModelComparison(options?: {
  maxDaysBack?: number;
  league?: string;
  bestOf?: number;
  targetModel?: string;
}): Promise<HistoricalModelComparison> {
  const params = new URLSearchParams();
  if (options?.maxDaysBack) params.set('max_days_back', String(options.maxDaysBack));
  if (options?.league) params.set('league', options.league);
  if (options?.bestOf) params.set('best_of', String(options.bestOf));
  if (options?.targetModel) params.set('target_model', options.targetModel);
  const query = params.toString() ? `?${params.toString()}` : '';
  const response = await fetch(`${API_BASE}/timing/model-comparison${query}`);
  return response.json();
}

export async function fetchModelClvByHorizon(
  maxDaysBack: number = 90,
  maxOddsAgeHours: number = 4,
  taxRate: number = 0.12,
  minEv: number = 0
): Promise<ModelClvByHorizonResponse> {
  const params = new URLSearchParams({
    max_days_back: maxDaysBack.toString(),
    max_odds_age_hours: maxOddsAgeHours.toString(),
    tax_rate: taxRate.toString(),
    min_ev: minEv.toString(),
  });
  const response = await fetch(`${API_BASE}/timing/model-clv-by-horizon?${params}`);
  if (!response.ok) throw new Error(`Failed to fetch model CLV by horizon: ${response.statusText}`);
  return response.json();
}

export async function fetchModelProfitabilityAudit(options?: {
  modelKey?: string;
  taxRate?: number;
  minEv?: number;
  maxDaysBack?: number;
}): Promise<ModelProfitabilityAuditResponse> {
  const params = new URLSearchParams();
  if (options?.modelKey) params.set('model_key', options.modelKey);
  if (options?.taxRate !== undefined) params.set('tax_rate', String(options.taxRate));
  if (options?.minEv !== undefined) params.set('min_ev', String(options.minEv));
  if (options?.maxDaysBack !== undefined) params.set('max_days_back', String(options.maxDaysBack));
  const query = params.toString() ? `?${params.toString()}` : '';
  const response = await fetch(`${API_BASE}/timing/model-profitability-audit${query}`);
  if (!response.ok) throw new Error(`Failed to fetch model profitability audit: ${response.statusText}`);
  return response.json();
}

export interface ValidationReportResponse {
  summary: {
    matches: number;
    model_logloss: number;
    market_logloss: number;
    delta_logloss: number;
    model_brier: number;
    market_brier: number;
    model_accuracy_pct: number;
    market_accuracy_pct: number;
  };
  betting_totals: {
    result_status: 'computed' | 'no_eligible_bets';
    uncertainty_basis: 'mean' | 'measured_bounds' | 'assumed_research_haircut' | 'missing_bounds';
    uncertainty_reason: string;
    assumed_sigma_z: number | null;
    total_bets: number;
    total_wins: number;
    win_rate_pct: number | null;
    total_staked_pln: number;
    total_pnl_pln: number;
    roi_pct: number | null;
    avg_clv_pct: number | null;
    pos_clv_pct: number | null;
  };
  odds_brackets: Array<{
    label: string;
    bets: number;
    wins: number;
    win_rate_pct: number;
    avg_odds: number;
    expected_net_roi_pct: number;
    realized_net_roi_pct: number;
    pnl_pln: number;
    clv_pct: number;
  }>;
  bookmakers: Array<{
    bookmaker: string;
    bets: number;
    wins: number;
    win_rate_pct: number;
    avg_odds: number;
    pnl_pln: number;
    roi_pct: number;
    clv_pct: number;
  }>;
  horizons: Array<{
    label: string;
    matches: number;
    quotes: number;
    model_logloss: number;
    market_logloss: number;
    delta_logloss: number;
    avg_clv_pct: number;
    pos_clv_pct: number;
  }>;
  calibration_deciles: Array<{
    label: string;
    count: number;
    avg_predicted_pct: number;
    observed_rate_pct: number;
    gap_pp: number;
  }>;
  error?: string;
}

export async function fetchValidationReport(options?: {
  daysBack?: number;
  taxRate?: number;
  minEv?: number;
  modelName?: string;
  modelVersion?: string;
}, signal?: AbortSignal): Promise<ValidationReportResponse> {
  const params = new URLSearchParams();
  if (options?.daysBack !== undefined) params.set('days_back', String(options.daysBack));
  if (options?.taxRate !== undefined) params.set('tax_rate', String(options.taxRate));
  if (options?.minEv !== undefined) params.set('min_ev', String(options.minEv));
  if (options?.modelName) params.set('model_name', options.modelName);
  if (options?.modelVersion) params.set('model_version', options.modelVersion);
  const query = params.toString() ? `?${params.toString()}` : '';
  const response = await fetch(`${API_BASE}/timing/validation-report${query}`, { signal });
  if (!response.ok) throw new Error(`Failed to fetch validation report: ${response.statusText}`);
  return response.json();
}

// ─── Alias Mapping ───────────────────────────────────────────

export async function createTeamAlias(request: AliasCreateRequest): Promise<AliasCreateResponse> {
  const response = await fetch(`${API_BASE}/matches/alias`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to create alias: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteTeamAlias(raw_name: string): Promise<{ ok: boolean; deleted: boolean }> {
  const response = await fetch(`${API_BASE}/matches/alias`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_name }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to delete alias: ${response.statusText}`);
  }
  return response.json();
}

export async function blockTeamAlias(raw_name: string): Promise<{ ok: boolean; blocked: boolean }> {
  const response = await fetch(`${API_BASE}/matches/alias/block`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_name }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to block alias: ${response.statusText}`);
  }
  return response.json();
}

export async function unblockTeamAlias(raw_name: string): Promise<{ ok: boolean; unblocked: boolean }> {
  const response = await fetch(`${API_BASE}/matches/alias/block`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_name }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to unblock alias: ${response.statusText}`);
  }
  return response.json();
}

export async function searchGolggTeams(q: string = '', limit: number = 50, signal?: AbortSignal): Promise<GolggTeamsResponse> {
  const params = new URLSearchParams({ q, limit: limit.toString() });
  const response = await fetch(`${API_BASE}/matches/golgg-teams?${params}`, { signal });
  if (!response.ok) throw new Error(`Failed to search GolGG teams: ${response.statusText}`);
  return response.json();
}

// ─── Bootstrap Analysis ────────────────────────────────────────

export async function fetchHorizonBootstrap(): Promise<HorizonBootstrapResponse> {
  const response = await fetch(`${API_BASE}/bootstrap/horizon`);
  if (!response.ok) throw new Error(`Failed to fetch bootstrap results: ${response.statusText}`);
  return response.json();
}

// ─── Embedding Diagnostics ───────────────────────────────────

export async function fetchChampionEmbeddings(
  method: 'umap' | 'tsne' | 'pca' = 'umap',
  preset: 'local' | 'balanced' | 'global' = 'balanced',
  role: string = 'ALL',
  minGames: number = 0,
  snapshot: string = 'latest',
  signal?: AbortSignal
): Promise<ChampionEmbeddingProjectionResponse> {
  const params = new URLSearchParams({
    method,
    preset,
    role,
    min_games: minGames.toString(),
    snapshot,
  });
  const response = await fetch(`${API_BASE}/embeddings/champions?${params}`, { signal });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch champion embeddings: ${response.statusText}`);
  }
  return response.json();
}

// ─── Tournaments & Bracket Simulation ─────────────────────────

export async function fetchTournaments(): Promise<TournamentSummary[]> {
  const response = await fetch(`${API_BASE}/tournaments`);
  if (!response.ok) throw new Error('Failed to fetch tournaments');
  return response.json();
}

export async function fetchTournamentBracket(id: string, simulations: number = 10000): Promise<TournamentSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/${id}?simulations=${simulations}`);
  if (!response.ok) throw new Error('Failed to fetch tournament bracket');
  return response.json();
}
export async function syncTournamentBracket(
  id: string,
  source: string = 'auto',
  force: boolean = false,
  rawContent?: string,
): Promise<TournamentSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/${id}/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, force, raw_content: rawContent }),
  });
  if (!response.ok) throw new Error('Nie udało się zsynchronizować drabinki turniejowej');
  return response.json();
}

export async function simulateTournament(
  id: string,
  simulations: number = 10000,
  manual_overrides?: Record<string, string>,
  calibrate: boolean = true,
): Promise<TournamentSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/${id}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ simulations, manual_overrides, calibrate }),
  });
  if (!response.ok) throw new Error('Failed to run tournament simulation');
  return response.json();
}

export async function recalculateTournament(
  id: string,
  simulations: number = 10000,
  manual_overrides?: Record<string, string>,
  calibrate: boolean = true,
): Promise<TournamentSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/${id}/recalculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ simulations, manual_overrides, calibrate }),
  });
  if (!response.ok) throw new Error('Nie udało się przeliczyć symulacji turniejowej');
  return response.json();
}

export async function simulateWorlds(
  directTeams: WorldsTeamInput[],
  playInTeams: WorldsTeamInput[],
  playInWinnerPool: number,
  simulations: number = 5000,
): Promise<WorldsSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/worlds/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      direct_teams: directTeams,
      play_in_teams: playInTeams,
      play_in_winner_pool: playInWinnerPool,
      simulations,
    }),
  });
  if (!response.ok) throw new Error(`Failed to run Worlds simulation: ${await response.text()}`);
  return response.json();
}

export async function simulateWorldsCrystalBall(
  simulations: number = 1000,
  seed: number = 42,
  directTeams?: WorldsTeamInput[],
  playInTeams?: WorldsTeamInput[],
): Promise<WorldsCrystalBallResponse> {
  const response = await fetch(`${API_BASE}/tournaments/worlds/crystal-ball`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      n_simulations: simulations,
      seed,
      direct_teams: directTeams,
      play_in_teams: playInTeams,
    }),
  });
  if (!response.ok) throw new Error(`Failed to simulate Worlds Crystal Ball: ${await response.text()}`);
  return response.json();
}

export async function fetchEncConfiguration(): Promise<EncConfigurationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/enc`);
  if (!response.ok) throw new Error(`Failed to load ENC configuration: ${response.statusText}`);
  return response.json();
}

export async function simulateEnc(simulations: number = 5000): Promise<EncSimulationResponse> {
  const response = await fetch(`${API_BASE}/tournaments/enc/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ simulations }),
  });
  if (!response.ok) throw new Error(`Failed to run ENC simulation: ${await response.text()}`);
  return response.json();
}

// ─── Player Comparison ───────────────────────────────────────

export async function searchPlayers(query: string, limit: number = 15): Promise<PlayerSearchItem[]> {
  const clean = query.trim();
  if (!clean) return [];
  const params = new URLSearchParams({ query: clean, limit: limit.toString() });
  const response = await fetch(`${API_BASE}/players/search?${params}`);
  if (!response.ok) throw new Error(`Failed to search players: ${response.statusText}`);
  return response.json();
}

export async function fetchPlayerProfile(playerId: string): Promise<PlayerProfileDetail> {
  const response = await fetch(`${API_BASE}/players/${encodeURIComponent(playerId)}`);
  if (!response.ok) throw new Error(`Failed to fetch player profile: ${response.statusText}`);
  return response.json();
}

export async function fetchPlayerHistory(playerId: string, limit: number = 250): Promise<RatingTimelinePoint[]> {
  const response = await fetch(`${API_BASE}/players/${encodeURIComponent(playerId)}/history?limit=${limit}`);
  if (!response.ok) throw new Error(`Failed to fetch player history: ${response.statusText}`);
  return response.json();
}

export async function fetchPlayerComparison(
  playerA: string,
  playerB: string,
  ratingSystem: string = 'unified'
): Promise<PlayerComparisonResponse> {
  const params = new URLSearchParams({
    player_a: playerA,
    player_b: playerB,
    rating_system: ratingSystem,
  });
  const response = await fetch(`${API_BASE}/players/compare?${params}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to compare players: ${response.statusText}`);
  }
  return response.json();
}

// ─── Value Bet Alerts ──────────────────────────────────────

export async function fetchAlertConfig(): Promise<AlertConfigResponse> {
  const response = await fetch(`${API_BASE}/alerts/config`);
  if (!response.ok) throw new Error(`Failed to fetch alert config: ${response.statusText}`);
  return response.json();
}

export async function updateAlertConfig(payload: AlertConfigUpdateRequest): Promise<AlertConfigResponse> {
  const response = await fetch(`${API_BASE}/alerts/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to update alert config: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchAlertHistory(limit: number = 50, status?: string): Promise<AlertHistoryResponse> {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (status) params.set('status', status);
  const response = await fetch(`${API_BASE}/alerts/history?${params}`);
  if (!response.ok) throw new Error(`Failed to fetch alert history: ${response.statusText}`);
  return response.json();
}

export async function triggerAlertCheck(dryRun: boolean = false): Promise<AlertCheckResponse> {
  const params = new URLSearchParams({ dry_run: dryRun.toString() });
  const response = await fetch(`${API_BASE}/alerts/check?${params}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error(`Failed to trigger alert check: ${response.statusText}`);
  return response.json();
}

export async function triggerAlertTest(channel: string = 'both'): Promise<AlertTestResponse> {
  const params = new URLSearchParams({ channel });
  const response = await fetch(`${API_BASE}/alerts/test?${params}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error(`Failed to send test alert: ${response.statusText}`);
  return response.json();
}

// ─── Proposition Odds & In-Game Models (IDEA-018) ──────────────

export async function fetchMatchPropAnalysis(
  canonicalMatchId: number,
  options?: { mapNumber?: number; taxRate?: number; recalculate?: boolean }
): Promise<MatchPropAnalysisResponse> {
  const params = new URLSearchParams();
  if (options?.mapNumber != null) params.set('map_number', options.mapNumber.toString());
  if (options?.taxRate != null) params.set('tax_rate', options.taxRate.toString());
  const response = await fetch(`${API_BASE}/matches/${canonicalMatchId}/props/analysis?${params}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to fetch match prop analysis: ${response.statusText}`);
  }
  return response.json();
}

export async function triggerMatchPropPrediction(
  canonicalMatchId: number,
  mapNumber: number = 1,
  taxRate: number = 0.12
): Promise<MatchPropAnalysisResponse> {
  const params = new URLSearchParams({
    map_number: mapNumber.toString(),
    tax_rate: taxRate.toString(),
  });
  const response = await fetch(`${API_BASE}/matches/${canonicalMatchId}/props/predict?${params}`, {
    method: 'POST',
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to trigger prop prediction: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchPropOddsTimeline(
  canonicalMatchId: number,
  options?: { marketType?: string; line?: number; mapNumber?: number }
): Promise<PropOddsTimelineResponse> {
  const params = new URLSearchParams();
  if (options?.marketType) params.set('market_type', options.marketType);
  if (options?.line != null) params.set('line', options.line.toString());
  if (options?.mapNumber != null) params.set('map_number', options.mapNumber.toString());
  const response = await fetch(`${API_BASE}/matches/${canonicalMatchId}/props/timeline?${params}`);
  if (!response.ok) throw new Error(`Failed to fetch prop timeline: ${response.statusText}`);
  return response.json();
}

export async function fetchPropOddsLatest(
  canonicalMatchId: number,
  options?: { marketType?: string; mapNumber?: number }
): Promise<PropOddsLatestResponse> {
  const params = new URLSearchParams();
  if (options?.marketType) params.set('market_type', options.marketType);
  if (options?.mapNumber != null) params.set('map_number', options.mapNumber.toString());
  const response = await fetch(`${API_BASE}/matches/${canonicalMatchId}/props/latest?${params}`);
  if (!response.ok) throw new Error(`Failed to fetch latest prop odds: ${response.statusText}`);
  return response.json();
}

export interface ActiveModelInfo {
  status: string;
  active_operational_model: string;
  active_hybrid_model: string;
  operational_model: {
    name: string;
    version: string;
    family: string;
    feature_version: string;
    ratings_version: string;
    target: string;
    has_epistemic_uncertainty: boolean;
  };
  hybrid_model: {
    name: string;
    version: string;
    alpha: number;
    temperature: number;
    blending_mode: string;
  };
}

export async function fetchActiveModel(): Promise<ActiveModelInfo> {
  const response = await fetch(`${API_BASE}/matches/models/active`);
  if (!response.ok) throw new Error(`Failed to fetch active model: ${response.statusText}`);
  return response.json();
}

export interface RegisteredModelItem {
  name: string;
  version: string;
  family: string;
  feature_version: string;
  ratings_version: string;
  target: string;
  has_epistemic_uncertainty: boolean;
  description?: string;
}

export interface RegisteredModelsResponse {
  status: string;
  active_operational_model: string;
  active_hybrid_model: string;
  models: RegisteredModelItem[];
}

export async function fetchRegisteredModels(): Promise<RegisteredModelsResponse> {
  const response = await fetch(`${API_BASE}/matches/models/registered`);
  if (!response.ok) throw new Error(`Failed to fetch registered models: ${response.statusText}`);
  return response.json();
}

export async function fetchPlayerEmbeddings(
  method: string = 'pca',
  preset: string = 'balanced',
  role: string = 'ALL',
  minGames: number = 15,
  maxPoints: number = 2000,
  daysActive: number = 365,
  nClusters: number | null = null,
  clusterAlgorithm: string = 'gmm',
  minClusterSize: number | null = 15,
  tier: string = 'all',
  signal?: AbortSignal,
): Promise<PlayerEmbeddingProjectionResponse> {
  const params = new URLSearchParams({
    method,
    preset,
    role,
    min_games: minGames.toString(),
    max_points: maxPoints.toString(),
    cluster_algorithm: clusterAlgorithm,
    tier,
  });
  if (daysActive > 0) {
    params.set('days_active', daysActive.toString());
  }
  if (nClusters !== null) {
    params.set('n_clusters', nClusters.toString());
  }
  if (minClusterSize !== null) {
    params.set('min_cluster_size', minClusterSize.toString());
  }

  const response = await fetch(`${API_BASE}/embeddings/players/projection?${params.toString()}`, { signal });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Nie udało się pobrać projekcji graczy');
  }
  return response.json();
}
