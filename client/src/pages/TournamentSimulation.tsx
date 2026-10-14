import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  fetchActiveTeams,
  fetchEncConfiguration,
  fetchTournamentBracket,
  fetchTournaments,
  recalculateTournament,
  simulateEnc,
  simulateWorlds,
  simulateWorldsCrystalBall,
  syncTournamentBracket,
} from '../api/client';
import type {
  BracketMatch,
  EncConfigurationResponse,
  EncSimulationResponse,
  TournamentSimulationResponse,
  TournamentSummary,
  WorldsSimulationResponse,
  WorldsCrystalBallResponse,
  CrystalBallCategoryResult,
  WorldsTeamInput,
} from '../types';
import './TournamentSimulation.css';

function diagnosticRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function diagnosticText(value: unknown): string {
  return typeof value === 'string' && value.length > 0 ? value : 'nieznane (unknown)';
}

function TournamentInputProvenance({ response }: { response: unknown }) {
  const payload = diagnosticRecord(response);
  const provenance = payload.provenance === undefined ? payload : diagnosticRecord(payload.provenance);
  const pairs = Array.isArray(provenance.native_prediction_provenance)
    ? provenance.native_prediction_provenance.map(diagnosticRecord)
    : [];
  const sources = [...new Set(pairs.map((pair) => {
    const diagnostics = diagnosticRecord(pair.diagnostics);
    const source = diagnosticRecord(diagnostics.feature_provenance);
    const freshness = diagnosticRecord(source.dataset_freshness);
    return [
      `Źródło: ${diagnosticText(source.source_kind)}`,
      `maks. dzień udostępnionej historii: ${diagnosticText(freshness.raw_history_max_effective_release_day)}`,
      `historia użyta do cech: ${diagnosticText(source.feature_source_max_day ?? diagnostics.feature_source_max_day)}`,
      `SHA256: ${diagnosticText(source.raw_history_sha256)}`,
      `zegar żądania: ${diagnosticText(diagnostics.requested_decision_at)}`,
    ].join(' · ');
  }))];
  const rosterBasis = diagnosticRecord(provenance.roster_basis);
  const basisLabels: Record<string, string> = {
    current_stored_roster: 'obecny zapisany skład (mutable)',
    explicit_scenario_roster: 'jawny skład scenariusza (manual)',
    configured_role_selection: 'wybór ról z konfiguracji scenariusza',
  };

  return (
    <aside className="scenario-tip" aria-label="Pochodzenie danych symulacji">
      <strong>Pochodzenie danych C0 — scenariusz hipotetyczny, nie prognoza live</strong>
      <p>
        Bieżący czas obliczenia ani synchronizacja drabinki nie odświeżają historii modelu.
        Publikacja składów przed decyzją i uwierzytelnienie ogłoszeń: nie wykazane (unknown).
        Brak certyfikacji point-in-time i dowodu historycznej kalibracji turniejowej native C0.
      </p>
      {sources.length > 0 ? sources.map((source) => <p key={source}>{source}</p>) : (
        <p>Źródło, hash, maksymalny dzień udostępnionej historii i zegar żądania: nieznane (unknown) — brak diagnostyki w zapisanym wyniku.</p>
      )}
      <p>
        Podstawa składów: {Object.keys(rosterBasis).length > 0
          ? Object.entries(rosterBasis).map(([team, basis]) => `${team}: ${basisLabels[diagnosticText(basis)] ?? diagnosticText(basis)}`).join('; ')
          : 'nieznana (unknown) — brak danych o podstawie składów'}
      </p>
      {pairs.length > 0 && (
        <details>
          <summary>Rzeczywiste wejścia i źródła dla {pairs.length} par (ID, kolejność tensorów, zegary)</summary>
          {pairs.map((pair, index) => (
            <details key={index}>
              <summary>{diagnosticText(pair.team1_id)} vs {diagnosticText(pair.team2_id)} · BO{String(pair.best_of ?? '?')}</summary>
              <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                {JSON.stringify({
                  roster_provenance: pair.roster_provenance ?? 'unknown',
                  native_diagnostics: pair.diagnostics ?? 'unknown',
                }, null, 2)}
              </pre>
            </details>
          ))}
        </details>
      )}
    </aside>
  );
}

const INITIAL_DIRECT_TEAMS: WorldsTeamInput[] = [
  { team: 'Gen.G', region: 'LCK', pool: 1 },
  { team: 'Bilibili Gaming', region: 'LPL', pool: 1 },
  { team: 'LYON', region: 'LCS', pool: 1 },
  { team: 'G2 Esports', region: 'LEC', pool: 1 },
  { team: 'Team Secret Whales', region: 'LCP', pool: 2 },
  { team: 'Los Grandes', region: 'CBLOL', pool: 2 },
  { team: 'Hanwha Life Esports', region: 'LCK', pool: 2 },
  { team: "Anyone's Legend", region: 'LPL', pool: 2 },
  { team: 'Team Liquid', region: 'LCS', pool: 3 },
  { team: 'Karmine Corp', region: 'LEC', pool: 3 },
  { team: 'T1', region: 'LCK', pool: 3 },
  { team: 'Top Esports', region: 'LPL', pool: 3 },
  { team: 'CTBC Flying Oyster', region: 'LCP', pool: 4 },
  { team: 'Dplus KIA', region: 'LCK', pool: 4 },
  { team: 'LGD Gaming', region: 'LPL', pool: 4 },
];

const DIRECT_SLOT_LABELS = [
  'LCK #1', 'LPL #1', 'LCS #1', 'LEC #1',
  'LCP #1', 'CBLOL #1', 'LCK #2', 'LPL #2',
  'LCS #2', 'LEC #2', 'LCK #3', 'LPL #3',
  'LCP #2', 'LCK #4', 'LPL #4',
];

const INITIAL_PLAY_IN_TEAMS: WorldsTeamInput[] = [
  { team: 'Cloud9', region: 'LCS' }, { team: 'GIANTX', region: 'LEC' },
  { team: 'MVK Esports', region: 'LCP' }, { team: 'LOUD', region: 'CBLOL' },
];

const PLAY_IN_SLOT_LABELS = ['LCS #3', 'LEC #3', 'LCP #3', 'CBLOL #2'];

function formatSec(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function CrystalBallCard({
  icon,
  title,
  outcome,
  isExpanded,
  onToggle,
}: {
  icon: string;
  title: string;
  categoryKey?: string;
  outcome?: CrystalBallCategoryResult;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  if (!outcome || !outcome.ranking || outcome.ranking.length === 0) {
    return (
      <div className="crystal-ball-card">
        <div className="crystal-ball-card-header">
          <span className="crystal-ball-card-title">{icon} {title}</span>
        </div>
        <p className="subtitle-sm" style={{ color: '#64748b' }}>Brak danych symulacji</p>
      </div>
    );
  }

  const topPick = outcome.ranking[0];
  const displayItems = isExpanded ? outcome.ranking : outcome.ranking.slice(0, 5);
  const maxProb = topPick.probability > 0 ? topPick.probability : 1.0;

  return (
    <div className="crystal-ball-card">
      <div className="crystal-ball-card-header">
        <span className="crystal-ball-card-title">{icon} {title}</span>
        <span className="crystal-ball-badge-count">{outcome.ranking.length} kandydatów</span>
      </div>

      <div className="crystal-ball-top-pick">
        <div className="crystal-ball-top-name">
          👑 {topPick.candidate}
        </div>
        <div className="crystal-ball-top-stats">
          <span className="crystal-ball-prob-badge">{(topPick.probability * 100).toFixed(1)}%</span>
          <span className="crystal-ball-odds-badge">Kurs {topPick.fair_odds.toFixed(2)}</span>
        </div>
      </div>

      <table className="crystal-ball-table">
        <thead>
          <tr>
            <th className="crystal-ball-rank">#</th>
            <th>Kandydat</th>
            <th className="crystal-ball-bar-cell">Rozkład</th>
            <th style={{ textAlign: 'right' }}>Szansa</th>
            <th style={{ textAlign: 'right' }}>Kurs</th>
          </tr>
        </thead>
        <tbody>
          {displayItems.map((item, idx) => {
            const barWidth = Math.max(3, Math.min(100, Math.round((item.probability / maxProb) * 100)));
            return (
              <tr key={item.candidate}>
                <td className="crystal-ball-rank">{idx + 1}</td>
                <td className="crystal-ball-candidate">{item.candidate}</td>
                <td className="crystal-ball-bar-cell">
                  <div className="crystal-ball-bar-track">
                    <div className="crystal-ball-bar-fill" style={{ width: `${barWidth}%` }} />
                  </div>
                </td>
                <td style={{ textAlign: 'right', fontWeight: 600, color: '#38bdf8' }}>
                  {(item.probability * 100).toFixed(1)}%
                </td>
                <td style={{ textAlign: 'right', fontWeight: 600, color: '#10b981' }}>
                  {item.fair_odds.toFixed(2)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {outcome.ranking.length > 5 && (
        <button
          type="button"
          className="crystal-ball-expand-btn"
          onClick={onToggle}
        >
          {isExpanded ? '▲ Zwiń ranking' : `▼ Pokaż pozostałych (${outcome.ranking.length - 5})`}
        </button>
      )}
    </div>
  );
}

export default function TournamentSimulation() {
  const [activeTab, setActiveTab] = useState<'regional' | 'worlds' | 'enc'>('regional');
  // Layout states for spacious tables and flexible views
  const [worldsView, setWorldsView] = useState<'results' | 'editor' | 'split' | 'crystalBall'>('results');
  const [crystalBallData, setCrystalBallData] = useState<WorldsCrystalBallResponse | null>(null);
  const [crystalBallSimulating, setCrystalBallSimulating] = useState<boolean>(false);
  const [crystalBallSimCount, setCrystalBallSimCount] = useState<number>(1000);
  const [crystalBallError, setCrystalBallError] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});

  const handleSimulateCrystalBall = async () => {
    setCrystalBallSimulating(true);
    setCrystalBallError(null);
    try {
      const res = await simulateWorldsCrystalBall(
        crystalBallSimCount,
        42,
        directWorldsTeams,
        playInWorldsTeams,
      );
      setCrystalBallData(res);
    } catch (err: unknown) {
      setCrystalBallError(err instanceof Error ? err.message : 'Nie udało się wykonać symulacji Crystal Ball');
    } finally {
      setCrystalBallSimulating(false);
    }
  };

  const toggleCategoryExpand = (catKey: string) => {
    setExpandedCategories((prev) => ({ ...prev, [catKey]: !prev[catKey] }));
  };
  const [encView, setEncView] = useState<'results' | 'rosters' | 'split'>('results');

  // Regional state
  const [tournaments, setTournaments] = useState<TournamentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>('lck_2026_playoffs');
  const [data, setData] = useState<TournamentSimulationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [simulating, setSimulating] = useState<boolean>(false);
  const [simCount, setSimCount] = useState<number>(10000);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [syncSource, setSyncSource] = useState<'auto' | 'fandom' | 'liquipedia'>('auto');
  const [syncSuccessMessage, setSyncSuccessMessage] = useState<string | null>(null);
  const [manualModalOpen, setManualModalOpen] = useState<boolean>(false);
  const [manualText, setManualText] = useState<string>('');
  // Editable participant slots define a hypothetical scenario, not verified qualification.
  const [directWorldsTeams, setDirectWorldsTeams] = useState<WorldsTeamInput[]>(INITIAL_DIRECT_TEAMS);
  const [playInWorldsTeams, setPlayInWorldsTeams] = useState<WorldsTeamInput[]>(INITIAL_PLAY_IN_TEAMS);
  const [worldsData, setWorldsData] = useState<WorldsSimulationResponse | null>(null);
  const [worldsSimulating, setWorldsSimulating] = useState<boolean>(false);
  const [worldsSimCount, setWorldsSimCount] = useState<number>(5000);
  const [activeTeams, setActiveTeams] = useState<Array<{ name: string; rating: number | null }>>([]);
  const [teamSuggestionError, setTeamSuggestionError] = useState<string | null>(null);
  const [activeSuggestionSlot, setActiveSuggestionSlot] = useState<{
    stage: 'direct' | 'playIn';
    index: number;
  } | null>(null);

  const [encConfiguration, setEncConfiguration] = useState<EncConfigurationResponse | null>(null);
  const [encConfigurationError, setEncConfigurationError] = useState<string | null>(null);
  const [encData, setEncData] = useState<EncSimulationResponse | null>(null);
  const [encLoading, setEncLoading] = useState<boolean>(false);
  const encLoadInFlight = useRef(false);
  const encConfigurationAttempted = useRef(false);
  const regionalActionInFlight = useRef(false);
  const regionalContext = useRef({ activeTab, selectedId, simCount });
  useLayoutEffect(() => {
    regionalContext.current = { activeTab, selectedId, simCount };
  }, [activeTab, selectedId, simCount]);
  const [encSimulating, setEncSimulating] = useState<boolean>(false);
  const [encSimCount, setEncSimCount] = useState<number>(1000);
  useEffect(() => {
    fetchActiveTeams()
      .then((response) => setActiveTeams(response.teams))
      .catch((reason: unknown) => {
        setTeamSuggestionError(
          reason instanceof Error ? reason.message : 'Nie udało się pobrać aktualnych drużyn.',
        );
      });
  }, []);

  useEffect(() => {
    fetchTournaments()
      .then((list) => {
        setTournaments(list);
        if (list.length > 0) {
          setSelectedId((current) => current || list[0].id);
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Błąd ładowania turniejów'));
  }, []);

  const loadEncConfiguration = useCallback(async (explicitRetry = false) => {
    if (encLoadInFlight.current
      || (encConfigurationAttempted.current && !explicitRetry)) return;
    encConfigurationAttempted.current = true;
    encLoadInFlight.current = true;
    setEncLoading(true);
    setEncConfigurationError(null);
    try {
      setEncConfiguration(await fetchEncConfiguration());
    } catch (reason: unknown) {
      setEncConfigurationError(
        reason instanceof Error ? reason.message : 'Nie udało się pobrać konfiguracji ENC.',
      );
    } finally {
      encLoadInFlight.current = false;
      setEncLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'enc') {
      void loadEncConfiguration();
    }
  }, [activeTab, loadEncConfiguration]);


  useEffect(() => {
    if (activeTab !== 'regional' || !selectedId) return;
    let current = true;
    setLoading(true);

    fetchTournamentBracket(selectedId, simCount)
      .then((res) => {
        if (!current) return;
        setData(res);
        setOverrides({});
        setLoading(false);
      })
      .catch((err) => {
        if (!current) return;
        setError(err instanceof Error ? err.message : 'Błąd ładowania drabinki');
        setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [activeTab, selectedId, simCount]);
  const [simElapsed, setSimElapsed] = useState<number>(0);
  const [worldsElapsed, setWorldsElapsed] = useState<number>(0);
  const [encElapsed, setEncElapsed] = useState<number>(0);

  const LEAGUE_PILLS = [
    { id: 'lck_2026_playoffs', region: 'LCK', name: 'LCK Play-offs', flag: '🇰🇷' },
    { id: 'lec_2026_summer_playoffs', region: 'LEC', name: 'LEC Summer', flag: '🇪🇺' },
    { id: 'lpl_2026_split3_playoffs', region: 'LPL', name: 'LPL Split 3', flag: '🇨🇳' },
    { id: 'lcs_2026_championship', region: 'LCS', name: 'LCS Championship', flag: '🇺🇸' },
  ];

  useEffect(() => {
    if (!simulating) {
      setSimElapsed(0);
      return;
    }
    const start = Date.now();
    const interval = setInterval(() => {
      setSimElapsed((Date.now() - start) / 1000);
    }, 100);
    return () => clearInterval(interval);
  }, [simulating]);

  useEffect(() => {
    if (!worldsSimulating) {
      setWorldsElapsed(0);
      return;
    }
    const start = Date.now();
    const interval = setInterval(() => {
      setWorldsElapsed((Date.now() - start) / 1000);
    }, 100);
    return () => clearInterval(interval);
  }, [worldsSimulating]);

  useEffect(() => {
    if (!encSimulating) {
      setEncElapsed(0);
      return;
    }
    const start = Date.now();
    const interval = setInterval(() => {
      setEncElapsed((Date.now() - start) / 1000);
    }, 100);
    return () => clearInterval(interval);
  }, [encSimulating]);

  const handleSyncBracket = async (sourceOverride?: 'auto' | 'fandom' | 'liquipedia') => {
    if (!selectedId || regionalActionInFlight.current) return;
    regionalActionInFlight.current = true;
    const requestContext = { activeTab, selectedId, simCount };
    setSyncing(true);
    setError(null);
    setSyncSuccessMessage(null);
    try {
      const chosenSource = sourceOverride || syncSource;
      const res = await syncTournamentBracket(selectedId, chosenSource, false);
      if (regionalContext.current.activeTab !== requestContext.activeTab
        || regionalContext.current.selectedId !== requestContext.selectedId
        || regionalContext.current.simCount !== requestContext.simCount) return;
      setData(res);
      setOverrides({});
      if (res.sync_message) {
        setSyncSuccessMessage(res.sync_message);
      }
    } catch (err) {
      if (regionalContext.current.activeTab === requestContext.activeTab
        && regionalContext.current.selectedId === requestContext.selectedId
        && regionalContext.current.simCount === requestContext.simCount) {
        setError(err instanceof Error ? err.message : 'Błąd synchronizacji drabinki');
      }
    } finally {
      regionalActionInFlight.current = false;
      setSyncing(false);
    }
  };

  const handleManualImport = async () => {
    if (!selectedId || !manualText.trim() || regionalActionInFlight.current) return;
    regionalActionInFlight.current = true;
    const requestContext = { activeTab, selectedId, simCount };
    setSyncing(true);
    setError(null);
    try {
      const res = await syncTournamentBracket(selectedId, 'liquipedia', false, manualText);
      if (regionalContext.current.activeTab !== requestContext.activeTab
        || regionalContext.current.selectedId !== requestContext.selectedId
        || regionalContext.current.simCount !== requestContext.simCount) return;
      setData(res);
      setOverrides({});
      setManualModalOpen(false);
      setManualText('');
      if (res.sync_message) {
        setSyncSuccessMessage(res.sync_message);
      }
    } catch (err) {
      if (regionalContext.current.activeTab === requestContext.activeTab
        && regionalContext.current.selectedId === requestContext.selectedId
        && regionalContext.current.simCount === requestContext.simCount) {
        setError(err instanceof Error ? err.message : 'Błąd importu ręcznego');
      }
    } finally {
      regionalActionInFlight.current = false;
      setSyncing(false);
    }
  };
  const handleSimulate = async () => {
    if (!selectedId || regionalActionInFlight.current) return;
    regionalActionInFlight.current = true;
    const requestContext = { activeTab, selectedId, simCount };
    setSimulating(true);
    try {
      const res = await recalculateTournament(selectedId, simCount, overrides, true);
      if (regionalContext.current.activeTab === requestContext.activeTab
        && regionalContext.current.selectedId === requestContext.selectedId
        && regionalContext.current.simCount === requestContext.simCount) {
        setData(res);
      }
    } catch (err) {
      if (regionalContext.current.activeTab === requestContext.activeTab
        && regionalContext.current.selectedId === requestContext.selectedId
        && regionalContext.current.simCount === requestContext.simCount) {
        setError(err instanceof Error ? err.message : 'Błąd symulacji');
      }
    } finally {
      regionalActionInFlight.current = false;
      setSimulating(false);
    }
  };

  const handleSimulateWorlds = async () => {
    setWorldsSimulating(true);
    try {
      const res = await simulateWorlds(
        directWorldsTeams,
        playInWorldsTeams,
        4,
        worldsSimCount,
      );
      setWorldsData(res);
      setWorldsView('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd symulacji Worlds');
    } finally {
      setWorldsSimulating(false);
    }
  };

  const handleSimulateEnc = async () => {
    if (!encConfiguration?.simulation_ready) return;
    setEncSimulating(true);
    try {
      const res = await simulateEnc(encSimCount);
      setEncData(res);
      setEncView('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd symulacji ENC');
    } finally {
      setEncSimulating(false);
    }
  };

  const handleToggleWinner = (matchId: string, team: string | null) => {
    if (!team) return;
    setOverrides((prev) => {
      const updated = { ...prev };
      if (updated[matchId] === team) {
        delete updated[matchId];
      } else {
        updated[matchId] = team;
      }
      return updated;
    });
  };

  const handleUpdateWorldsTeam = (
    stage: 'direct' | 'playIn',
    index: number,
    value: string,
  ) => {
    const updateTeams = (teams: WorldsTeamInput[]) => teams.map((team, teamIndex) => (
      teamIndex === index ? { ...team, team: value } : team
    ));
    if (stage === 'direct') {
      setDirectWorldsTeams(updateTeams);
    } else {
      setPlayInWorldsTeams(updateTeams);
    }
    setWorldsData(null);
  };

  const activeSuggestionValue = activeSuggestionSlot
    ? (activeSuggestionSlot.stage === 'direct'
      ? directWorldsTeams[activeSuggestionSlot.index].team
      : playInWorldsTeams[activeSuggestionSlot.index].team)
    : '';
  const visibleTeamSuggestions = activeTeams
    .filter((team) => team.name.toLocaleLowerCase().includes(activeSuggestionValue.toLocaleLowerCase()))
    .slice(0, 8);

  const handleSelectSuggestedTeam = (
    stage: 'direct' | 'playIn',
    index: number,
    teamName: string,
  ) => {
    handleUpdateWorldsTeam(stage, index, teamName);
    setActiveSuggestionSlot(null);
  };

  const worldsReady = [
    ...directWorldsTeams,
    ...playInWorldsTeams,
  ].every((team) => team.team.trim());

  useEffect(() => {
    if (activeTab === 'worlds' && !worldsData && !worldsSimulating && worldsReady) {
      handleSimulateWorlds();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, worldsData, worldsSimulating, worldsReady]);

  const bracket = data?.bracket || {};
  const allMatches = Object.values(bracket);

  // Group matches chronologically by progression stage
  const upperR1 = allMatches.filter((m) => m.bracket_section === 'upper' && m.round_name.includes('Round 1'));
  const upperR2 = allMatches.filter((m) => m.bracket_section === 'upper' && (m.round_name.includes('Round 2') || m.round_name.includes('Semifinal')));
  const upperFinal = allMatches.filter((m) => m.bracket_section === 'upper' && m.round_name.includes('Upper Final'));

  const lowerR1 = allMatches.filter((m) => m.bracket_section === 'lower' && m.round_name.includes('Lower Round 1'));
  const lowerR2 = allMatches.filter((m) => m.bracket_section === 'lower' && m.round_name.includes('Lower Round 2'));
  const lowerSemi = allMatches.filter((m) => m.bracket_section === 'lower' && (m.round_name.includes('Lower Round 3') || m.round_name.includes('Lower Semifinal')));
  const lowerFinal = allMatches.filter((m) => m.bracket_section === 'lower' && m.round_name.includes('Lower Final'));

  const grandFinal = allMatches.filter((m) => m.bracket_section === 'final');

  const renderMatchCard = (m: BracketMatch) => {
    const isCompleted = m.winner !== null;
    const manualWinner = overrides[m.id];
    const effectiveWinner = manualWinner || m.winner;

    const isGrandFinal = m.bracket_section === 'final';

    return (
      <div key={m.id} className={`bracket-match-card ${isCompleted ? 'completed' : ''} ${isGrandFinal ? 'grand-final-card' : ''} ${manualWinner ? 'overridden' : ''}`}>
        <div className="match-card-header">
          <span className="round-badge">{isGrandFinal ? '👑 Mecz o Mistrzostwo' : m.round_name}</span>
          <span className="bo-badge">Bo{m.best_of}</span>
        </div>
        <div className="match-card-teams">
          <div
            className={`team-row ${effectiveWinner === m.team1 ? 'winner' : ''} ${m.team1 ? 'clickable' : ''}`}
            onClick={() => handleToggleWinner(m.id, m.team1)}
            title={m.team1 ? `Kliknij, aby wymusić wygraną ${m.team1}` : ''}
          >
            <span className="team-name">{m.team1 || 'TBD'}</span>
            {m.score1 !== null && <span className="team-score">{m.score1}</span>}
            {effectiveWinner === m.team1 && <span className="crown-icon">👑</span>}
          </div>
          <div
            className={`team-row ${effectiveWinner === m.team2 ? 'winner' : ''} ${m.team2 ? 'clickable' : ''}`}
            onClick={() => handleToggleWinner(m.id, m.team2)}
            title={m.team2 ? `Kliknij, aby wymusić wygraną ${m.team2}` : ''}
          >
            <span className="team-name">{m.team2 || 'TBD'}</span>
            {m.score2 !== null && <span className="team-score">{m.score2}</span>}
            {effectiveWinner === m.team2 && <span className="crown-icon">👑</span>}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="tournament-container">
      <header className="tournament-header">
        <div>
          <h1>🏆 Symulacja Turniejów i Mistrzostw Świata</h1>
          <p className="subtitle">
            Pobieraj rzeczywisty stan drabinek ligowych lub symuluj pełne Mistrzostwa Świata (Worlds) z własną listą drużyn.
          </p>
        </div>

        <div className="tab-switcher">
          <button
            className={`tab-btn ${activeTab === 'regional' ? 'active' : ''}`}
            onClick={() => setActiveTab('regional')}
          >
            Drabinki Regionalne (LCK / LEC / LPL / LCS)
          </button>
          <button
            className={`tab-btn ${activeTab === 'worlds' ? 'active' : ''}`}
            onClick={() => setActiveTab('worlds')}
          >
            🌍 Worlds 2026 (Swiss + Knockout)
          </button>
          <button
            className={`tab-btn ${activeTab === 'enc' ? 'active' : ''}`}
            onClick={() => setActiveTab('enc')}
          >
            🏳️ ENC 2027 (reprezentacje)
          </button>
        </div>

        {activeTab === 'regional' ? (
          <>
            <div className="tournament-controls">
              <div className="league-quick-pills">
                {LEAGUE_PILLS.map((pill) => (
                  <button
                    key={pill.id}
                    type="button"
                    className={`league-pill ${selectedId === pill.id ? 'active' : ''}`}
                    onClick={() => setSelectedId(pill.id)}
                    title={`Przełącz na ${pill.name}`}
                  >
                    <span className="pill-flag">{pill.flag}</span>
                    <span className="pill-region">{pill.region}</span>
                  </button>
                ))}
              </div>
              <select
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
                className="tournament-select"
              >
                {tournaments.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name} ({t.region})
                  </option>
                ))}
              </select>
              <div className="sim-config">
                <label>Próby:</label>
                <select
                  value={simCount}
                  onChange={(e) => setSimCount(Number(e.target.value))}
                  className="sim-count-select"
                >
                  <option value={5000}>5 000 (Szybka)</option>
                  <option value={10000}>10 000 (Standard C0)</option>
                  <option value={20000}>20 000 (Precyzyjna)</option>
                  <option value={100000}>100 000 (Głęboka)</option>
                </select>
              </div>
              <button
                className="simulate-btn"
                onClick={handleSimulate}
                disabled={simulating || syncing || loading}
              >
                {simulating ? '⏳ Przeliczanie C0...' : '⚡ Przelicz symulację (C0)'}
              </button>
              {data?.cached && (
                <span style={{ fontSize: '0.78rem', color: '#10b981', padding: '4px 8px', background: 'rgba(16,185,129,0.1)', borderRadius: '4px', border: '1px solid rgba(16,185,129,0.2)' }} title={`Zapisano z pamięci podręcznej: ${data.cached_at || ''}`}>
                  ⚡ Pamięć podręczna O(1)
                </span>
              )}
              {Object.keys(overrides).length > 0 && (
                <button className="reset-btn" onClick={() => setOverrides({})}>
                  Reset scenariuszy ({Object.keys(overrides).length})
                </button>
              )}
            </div>
            <p className="subtitle-sm">
              Odczyt drabinki i symulacja korzystają z lokalnych danych — nie pobierają danych ze źródła.
              Synchronizacja źródła jest wyłącznie ręczna i respektuje pamięć podręczną oraz limity częstotliwości.
            </p>

            <div className="sync-toolbar">
              <div className="sync-source-group">
                <label>Źródło danych:</label>
                <select
                  value={syncSource}
                  onChange={(e) => setSyncSource(e.target.value as 'auto' | 'fandom' | 'liquipedia')}
                  className="sync-source-select"
                >
                  <option value="auto">⚡ Auto (LoL Fandom / Liquipedia)</option>
                  <option value="fandom">LoL Fandom Cargo (drabinka)</option>
                  <option value="liquipedia">📖 Liquipedia MediaWiki</option>
                </select>
              </div>
              <button
                className="sync-btn"
                onClick={() => handleSyncBracket()}
                disabled={syncing || loading || simulating}
                title="Pobierz i zaktualizuj bieżący stan meczów, wyników i awansów z wybranego źródła"
              >
                {syncing ? '⏳ Pobieranie z API...' : '🔄 Pobierz stan drabinki'}
              </button>
              <p className="subtitle-sm">
                Synchronizacja jest jawna; żądania mogą korzystać z pamięci podręcznej lub zostać odroczone przez limity źródła.
              </p>
 
              
              <button
                className="manual-import-btn"
                onClick={() => setManualModalOpen(true)}
                disabled={syncing || simulating || loading}
                title="Wklej ręcznie wikitext szablonu Bracket lub fragment HTML z Liquipedii"
              >
                📋 Import Wikitext / HTML
              </button>
              <div className="sync-status-badges">
                {data?.source && (
                  <span
                    className={`sync-badge ${
                      data.source.includes('fandom')
                        ? 'fandom'
                        : data.source.includes('manual')
                        ? 'manual'
                        : data.source.includes('liquipedia')
                        ? 'liquipedia'
                        : 'cache'
                    }`}
                  >
                    {data.source.includes('fandom')
                      ? 'LoL Fandom (drabinka)'
                      : data.source.includes('manual')
                      ? '📋 Wikitext/HTML'
                      : data.source.includes('liquipedia')
                      ? '📖 Liquipedia'
                      : '💾 Pamięć podręczna'}
                  </span>
                )}
                {data?.synced_at && (
                  <span className="sync-meta-chip">
                    🕒 Zaktualizowano:{' '}
                    {new Date(data.synced_at).toLocaleTimeString('pl-PL', {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </span>
                )}
                {data?.updated_matches !== undefined && (
                  <span className="sync-meta-chip matches">
                    ✓ Zsynchronizowano {data.updated_matches} meczów
                  </span>
                )}
              </div>
            </div>
          </>
        ) : activeTab === 'worlds' ? (
          <div className="tournament-controls">
            {worldsView === 'crystalBall' ? (
              <>
                <div className="sim-config">
                  <label htmlFor="cb-simulation-count">Próby Crystal Ball:</label>
                  <select
                    id="cb-simulation-count"
                    value={crystalBallSimCount}
                    onChange={(event) => setCrystalBallSimCount(Number(event.target.value))}
                    className="sim-count-select"
                    disabled={crystalBallSimulating}
                  >
                    <option value={200}>200</option>
                    <option value={500}>500</option>
                    <option value={1000}>1 000</option>
                    <option value={2000}>2 000</option>
                  </select>
                </div>
                <button
                  className="simulate-btn crystal-ball-btn"
                  onClick={handleSimulateCrystalBall}
                  disabled={crystalBallSimulating || !worldsReady}
                >
                  {crystalBallSimulating
                    ? '⏳ Symulacja Crystal Ball...'
                    : `🔮 Symuluj Crystal Ball (${crystalBallSimCount.toLocaleString('pl-PL')} prób)`}
                </button>
              </>
            ) : (
              <>
                <div className="sim-config">
                  <label htmlFor="worlds-simulation-count">Próby Worlds:</label>
                  <select
                    id="worlds-simulation-count"
                    value={worldsSimCount}
                    onChange={(event) => setWorldsSimCount(Number(event.target.value))}
                    className="sim-count-select"
                  >
                    <option value={1000}>1 000</option>
                    <option value={5000}>5 000</option>
                    <option value={10000}>10 000</option>
                    <option value={20000}>20 000</option>
                  </select>
                </div>
                <button
                  className="simulate-btn"
                  onClick={handleSimulateWorlds}
                  disabled={worldsSimulating || !worldsReady}
                >
                  {worldsSimulating
                    ? '⏳ Symulacja Worlds...'
                    : `⚡ Symuluj Worlds (${worldsSimCount.toLocaleString('pl-PL')} prób)`}
                </button>
              </>
            )}
            <div className="simulation-subtabs" role="group" aria-label="Widok Worlds">
              <button
                type="button"
                className={`subtab-btn ${worldsView === 'results' ? 'active' : ''}`}
                onClick={() => setWorldsView('results')}
              >
                📊 Tabela wyników
              </button>
              <button
                type="button"
                className={`subtab-btn ${worldsView === 'editor' ? 'active' : ''}`}
                onClick={() => setWorldsView('editor')}
              >
                ⚙️ Edytor slotów
              </button>
              <button
                type="button"
                className={`subtab-btn ${worldsView === 'split' ? 'active' : ''}`}
                onClick={() => setWorldsView('split')}
              >
                ⊞ Obok siebie
              </button>
              <button
                type="button"
                className={`subtab-btn ${worldsView === 'crystalBall' ? 'active' : ''}`}
                onClick={() => setWorldsView('crystalBall')}
              >
                🔮 Crystal Ball
              </button>
            </div>
            {!worldsReady && (
              <span className="worlds-validation-hint">
                Uzupełnij drużyny dla wszystkich 15 slotów Swiss i 4 slotów Play-In.
              </span>
            )}
          </div>
        ) : (
          <div className="tournament-controls">
            <div className="sim-config">
              <label htmlFor="enc-simulation-count">Próby ENC:</label>
              <select
                id="enc-simulation-count"
                value={encSimCount}
                onChange={(event) => setEncSimCount(Number(event.target.value))}
                className="sim-count-select"
              >
                <option value={1000}>1 000</option>
                <option value={5000}>5 000</option>
                <option value={10000}>10 000</option>
                <option value={20000}>20 000</option>
              </select>
            </div>
            <button
              className="simulate-btn"
              onClick={handleSimulateEnc}
              disabled={encSimulating || !encConfiguration?.simulation_ready}
            >
            {encSimulating ? '⏳ Symulacja ENC...' : `⚡ Symuluj ENC (${encSimCount.toLocaleString('pl-PL')} prób)`}
            </button>
            <div className="simulation-subtabs" role="group" aria-label="Widok ENC">
              <button
                type="button"
                className={`subtab-btn ${encView === 'results' ? 'active' : ''}`}
                onClick={() => setEncView('results')}
              >
                🏆 Tabela wyników
              </button>
              <button
                type="button"
                className={`subtab-btn ${encView === 'rosters' ? 'active' : ''}`}
                onClick={() => setEncView('rosters')}
              >
                👥 Składy nacji
              </button>
              <button
                type="button"
                className={`subtab-btn ${encView === 'split' ? 'active' : ''}`}
                onClick={() => setEncView('split')}
              >
                ⊞ Obok siebie
              </button>
            </div>
          </div>
        )}
      </header>
      {error && (
        <div className="error-banner">
          <div className="banner-content">
            <span className="banner-icon">⚠️</span>
            <span>{error}</span>
          </div>
          <button className="banner-close-btn" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}
      {syncSuccessMessage && (
        <div className="sync-success-banner">
          <div className="banner-content">
            <span className="banner-icon">✓</span>
            <span>{syncSuccessMessage}</span>
          </div>
          <button className="banner-close-btn" onClick={() => setSyncSuccessMessage(null)}>
            ✕
          </button>
        </div>
      )}

      {manualModalOpen && (
        <div className="modal-backdrop" onClick={() => setManualModalOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📋 Ręczny import stanu drabinki (Liquipedia / LoL Fandom)</h3>
              <button className="modal-close" onClick={() => setManualModalOpen(false)}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p className="modal-hint">
                Wklej kod źródłowy szablonu drabinki (np. <code>{'{{Bracket|r1m1team1=...}}'}</code>) lub tabelę
                HTML skopiowaną ze strony Liquipedii.
              </p>
              <textarea
                className="manual-textarea"
                placeholder="Wklej kod wikitext lub HTML tutaj..."
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                rows={10}
              />
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setManualModalOpen(false)}>
                Anuluj
              </button>
              <button
                className="btn-primary"
                onClick={handleManualImport}
                disabled={!manualText.trim() || syncing}
              >
                {syncing ? 'Importowanie...' : 'Zastosuj import'}
              </button>
            </div>
          </div>
        </div>
      )}
      {activeTab === 'regional' ? (
        loading && !data ? (
          <div className="regional-skeleton-container">
            <div className="simulation-loading-banner">
              <div className="computing-spinner" />
              <div className="loading-text">
                <strong>Wczytywanie drabinki {tournaments.find((t) => t.id === selectedId)?.name || 'regionalnej'} i ratingów GL...</strong>
                <span>Łączenie węzłów bracketu, pobieranie aktualnego stanu meczów i generowanie rozkładów Monte Carlo</span>
              </div>
            </div>
            <div className="standings-skeleton-card">
              <div className="skeleton-header-bar" />
              <div className="skeleton-table">
                {[1, 2, 3, 4, 5, 6].map((i) => (
                  <div key={i} className="skeleton-table-row">
                    <div className="skeleton-cell rank" />
                    <div className="skeleton-cell team" />
                    <div className="skeleton-cell prob" />
                    <div className="skeleton-cell prob" />
                    <div className="skeleton-cell prob" />
                    <div className="skeleton-cell prob" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className={`regional-stacked-layout ${simulating ? 'simulating-active' : ''}`}>
            {simulating && (
              <div className="simulation-computing-banner">
                <div className="computing-header-row">
                  <div className="computing-spinner" />
                  <div className="computing-meta">
                    <strong>Trwa symulacja Monte Carlo ({simCount.toLocaleString('pl-PL')} prób)...</strong>
                    <span className="computing-sub">
                      Obliczanie losowań match-upów, przepływu Upper/Lower bracket i prawdopodobieństw tytułu
                    </span>
                  </div>
                  <div className="computing-timer-chip">
                    ⏱️ <strong>{simElapsed.toFixed(1)} s</strong>
                  </div>
                </div>
                <div className="computing-progress-track">
                  <div className="computing-progress-bar" />
                </div>
              </div>
            )}
            <div className={simulating ? 'simulating-dimmed' : ''}>
              {/* 1. STANDINGS TABLE ALWAYS AT THE TOP */}
          <section className="standings-panel full-width">
            <div className="standings-panel-header">
              <h2>📊 Szanse na końcowy wynik ({data?.simulations.toLocaleString('pl-PL')} symulacji)</h2>
            </div>
            {data && <TournamentInputProvenance response={data} />}
            {data?.joint_finalists && data.joint_finalists.length > 0 && (
              <div className="joint-finalists-bar">
                <span className="joint-finalists-label">🏆 Najbardziej prawdopodobny Finał:</span>
                <div className="joint-finalists-chips">
                  {data.joint_finalists.map((f, i) => (
                    <span key={f.pair} className={`finalist-chip ${i === 0 ? 'top-chip' : ''}`}>
                      <strong>{f.pair}</strong> ({(f.prob * 100).toFixed(1)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div className="table-responsive">
              <table className="standings-table">
                <thead>
                  <tr>
                    <th className="th-rank">Poz.</th>
                    <th className="th-team">Drużyna</th>
                    <th className="th-num">🥇 Mistrz</th>
                    <th className="th-num">🥈 Top 2</th>
                    <th className="th-num">🥉 Top 3</th>
                    <th className="th-num">Top 4</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.standings.map((s, idx) => (
                    <tr key={s.team}>
                      <td className="rank-cell">{idx + 1}</td>
                      <td className="team-cell"><strong>{s.team}</strong></td>
                      <td className="prob-cell champ-prob">
                        <div className="prob-with-bar">
                          <div className="mini-bar-bg">
                            <div className="mini-bar-fill gold" style={{ width: `${Math.min(100, s.champion_prob * 100)}%` }} />
                          </div>
                          <span>{(s.champion_prob * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="prob-cell">{(s.top2_prob * 100).toFixed(1)}%</td>
                      <td className="prob-cell">{(s.top3_prob * 100).toFixed(1)}%</td>
                      <td className="prob-cell">{(s.top4_prob * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="scenario-tip">
              💡 <strong>Tryb scenariuszy (What-if):</strong> Kliknij na dowolną drużynę w drabince, aby wymusić jej zwycięstwo i przeliczyć resztę turnieju.
            </p>
          </section>

          {/* 2. LOL FANDOM BRACKET TREE: UPPER & LOWER HORIZONTAL STREAMS TO GRAND FINAL ON THE RIGHT */}
          <section className="bracket-panel full-width">
            <div className="bracket-panel-header">
              <h2>🎮 Drzewo Turniejowe (LoL Fandom Double Elimination)</h2>
              <span className="bracket-hint">Przepływ Upper & Lower Bracket $\rightarrow$ Wielki Finał</span>
            </div>

            <div className="fandom-bracket-board">
              {/* Left side: Upper and Lower branches */}
              <div className="fandom-branches-col">

                {/* UPPER BRACKET ROW */}
                <div className="fandom-bracket-branch upper-branch">
                  <div className="fandom-branch-header">
                    <div className="branch-title-wrap">
                      <span className="branch-indicator upper">▲ UPPER BRACKET</span>
                      <span className="branch-sub">Drabinka wygranych</span>
                    </div>
                  </div>
                  <div className="fandom-stages-row">
                    {upperR1.length > 0 && (
                      <div className="fandom-stage-col">
                        <div className="fandom-stage-title">Upper Runda 1</div>
                        <div className="fandom-stage-matches">{upperR1.map(renderMatchCard)}</div>
                      </div>
                    )}
                    {upperR2.length > 0 && (
                      <div className="fandom-stage-col">
                        <div className="fandom-stage-title">Półfinały Upper</div>
                        <div className="fandom-stage-matches">{upperR2.map(renderMatchCard)}</div>
                      </div>
                    )}
                    {upperFinal.length > 0 && (
                      <div className="fandom-stage-col upper-final-stage">
                        <div className="fandom-stage-title highlight-upper">Finał Upper</div>
                        <div className="fandom-stage-matches">{upperFinal.map(renderMatchCard)}</div>
                      </div>
                    )}
                  </div>
                </div>

                {/* LOWER BRACKET ROW */}
                <div className="fandom-bracket-branch lower-branch">
                  <div className="fandom-branch-header">
                    <div className="branch-title-wrap">
                      <span className="branch-indicator lower">▼ LOWER BRACKET</span>
                      <span className="branch-sub">Drabinka eliminacji (ostatnia szansa)</span>
                    </div>
                  </div>
                  <div className="fandom-stages-row">
                    {lowerR1.length > 0 && (
                      <div className="fandom-stage-col">
                        <div className="fandom-stage-title">Lower Runda 1</div>
                        <div className="fandom-stage-matches">{lowerR1.map(renderMatchCard)}</div>
                      </div>
                    )}
                    {lowerR2.length > 0 && (
                      <div className="fandom-stage-col">
                        <div className="fandom-stage-title">Lower Runda 2</div>
                        <div className="fandom-stage-matches">{lowerR2.map(renderMatchCard)}</div>
                      </div>
                    )}
                    {lowerSemi.length > 0 && (
                      <div className="fandom-stage-col">
                        <div className="fandom-stage-title">Półfinał Lower</div>
                        <div className="fandom-stage-matches">{lowerSemi.map(renderMatchCard)}</div>
                      </div>
                    )}
                    {lowerFinal.length > 0 && (
                      <div className="fandom-stage-col lower-final-stage">
                        <div className="fandom-stage-title highlight-lower">Finał Lower</div>
                        <div className="fandom-stage-matches">{lowerFinal.map(renderMatchCard)}</div>
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* CONNECTING BRIDGE TO GRAND FINAL */}
              <div className="fandom-connector-zone">
                <div className="connector-bracket-bracket">
                  <div className="connector-tag">FINAŁ</div>
                </div>
              </div>

              {/* RIGHT SIDE: GRAND FINAL (WIELKI FINAŁ) */}
              <div className="fandom-grand-final-stage">
                <div className="fandom-branch-header grand-final">
                  <span className="branch-indicator gold">🏆 WIELKI FINAŁ</span>
                  <span className="branch-sub">Mecz o Mistrzostwo</span>
                </div>
                <div className="grand-final-card-wrapper">
                  <div className="trophy-crest">
                    <span className="trophy-crest-icon">🏆</span>
                    <span className="trophy-crest-text">PUCHAR MISTRZA</span>
                  </div>
                  <div className="grand-final-match-box">
                    {grandFinal.map(renderMatchCard)}
                  </div>
                </div>
              </div>

            </div>
          </section>
            </div>
          </div>
        )
      ) : activeTab === 'worlds' ? (
        worldsView === 'crystalBall' ? (
          <section className="crystal-ball-panel full-width">
            <div className="crystal-ball-header">
              <h2>🔮 Worlds 2026 Crystal Ball — Predykcje i Rozkłady Ekstremalne</h2>
              <p className="subtitle-sm">
                Hierarchiczna generatywna symulacja statystyk graczy (Kills, Deaths, Assists, CS, Pentakille) sprzężona
                z zasadami Hard Fearless Draft oraz pełną drabinką Worlds 2026 (Play-In → Swiss → Knockout).
                Wylicza dokładne prawdopodobieństwa P i kursy Fair Odds (1/P).
              </p>
            </div>

            {crystalBallError && (
              <div className="error-banner" role="alert" style={{ marginBottom: '1rem' }}>
                <p>⚠️ {crystalBallError}</p>
              </div>
            )}

            {crystalBallSimulating ? (
              <div className="empty-state" style={{ padding: '3rem 1rem' }}>
                <div className="sync-spinner" style={{ margin: '0 auto 1.5rem auto' }} />
                <h3>Symulacja Worlds Crystal Ball w toku...</h3>
                <p>
                  Przeliczanie {crystalBallSimCount.toLocaleString('pl-PL')} pełnych turniejów Worlds pod zasadami Hard Fearless Draft.
                  Generowanie sekwencji map z blokadami bohaterów i losowaniem statystyk dla 10 graczy...
                </p>
              </div>
            ) : !crystalBallData ? (
              <div className="empty-state" style={{ padding: '3rem 1rem' }}>
                <h3>🔮 Brak wyników symulacji Crystal Ball</h3>
                <p style={{ maxWidth: '640px', margin: '0 auto 1.5rem auto' }}>
                  Wybierz liczbę prób w pasku powyżej i kliknij <strong>🔮 Symuluj Crystal Ball</strong>, aby wygenerować
                  rozkłady prawdopodobieństw i kursy fair odds dla 15 oficjalnych kategorii Riot Worlds Pick'Em:
                  rekordów graczy, statystyk drużynowych, mety bohaterów oraz zdarzeń turniejowych.
                </p>
                <button
                  type="button"
                  className="simulate-btn crystal-ball-btn"
                  onClick={handleSimulateCrystalBall}
                  disabled={!worldsReady}
                >
                  🔮 Uruchom symulację Crystal Ball ({crystalBallSimCount.toLocaleString('pl-PL')} prób)
                </button>
              </div>
            ) : (
              <>
                <div className="crystal-ball-meta-bar">
                  <div className="crystal-ball-meta-item">
                    <span>Liczba symulacji:</span>
                    <strong>{crystalBallData.n_simulations.toLocaleString('pl-PL')}</strong>
                  </div>
                  <div className="crystal-ball-meta-item">
                    <span>Średni najkrótszy czas gry:</span>
                    <strong>{formatSec(crystalBallData.tournament_shortest_game_seconds.mean)}</strong>
                  </div>
                  <div className="crystal-ball-meta-item">
                    <span>Średni najdłuższy czas gry:</span>
                    <strong>{formatSec(crystalBallData.tournament_longest_game_seconds.mean)}</strong>
                  </div>
                  <div className="crystal-ball-meta-item">
                    <span>Średnia liczba Baronów:</span>
                    <strong>{crystalBallData.tournament_total_barons.mean.toFixed(1)}</strong>
                  </div>
                </div>

                {/* Sekcja 1: Gracze */}
                <div className="crystal-ball-section">
                  <h3 className="crystal-ball-section-title">👤 Rekordy Graczy (Player Records)</h3>
                  <div className="crystal-ball-grid">
                    <CrystalBallCard
                      icon="🗡️"
                      title="Najwięcej zabójstw w 1 grze"
                      categoryKey="player_kills"
                      outcome={crystalBallData.player_most_kills_single_game}
                      isExpanded={!!expandedCategories.player_kills}
                      onToggle={() => toggleCategoryExpand('player_kills')}
                    />
                    <CrystalBallCard
                      icon="🎖️"
                      title="Najwyższe KDA turnieju (min. 5 gier)"
                      categoryKey="player_kda"
                      outcome={crystalBallData.player_highest_tournament_kda}
                      isExpanded={!!expandedCategories.player_kda}
                      onToggle={() => toggleCategoryExpand('player_kda')}
                    />
                    <CrystalBallCard
                      icon="🎭"
                      title="Najwięcej unikalnych bohaterów"
                      categoryKey="player_champs"
                      outcome={crystalBallData.player_most_unique_champions}
                      isExpanded={!!expandedCategories.player_champs}
                      onToggle={() => toggleCategoryExpand('player_champs')}
                    />
                    <CrystalBallCard
                      icon="🎯"
                      title="Najwyższy udział w zabójstwach (KP%)"
                      categoryKey="player_kp"
                      outcome={crystalBallData.player_highest_kill_participation}
                      isExpanded={!!expandedCategories.player_kp}
                      onToggle={() => toggleCategoryExpand('player_kp')}
                    />
                  </div>
                </div>

                {/* Sekcja 2: Drużyny */}
                <div className="crystal-ball-section">
                  <h3 className="crystal-ball-section-title">🛡️ Statystyki Drużynowe (Team Records)</h3>
                  <div className="crystal-ball-grid">
                    <CrystalBallCard
                      icon="🎭"
                      title="Najwięcej unikalnych bohaterów (Team)"
                      categoryKey="team_champs"
                      outcome={crystalBallData.team_most_unique_champions}
                      isExpanded={!!expandedCategories.team_champs}
                      onToggle={() => toggleCategoryExpand('team_champs')}
                    />
                    <CrystalBallCard
                      icon="💥"
                      title="Najwięcej zabójstw w 1 grze (Team)"
                      categoryKey="team_kills"
                      outcome={crystalBallData.team_most_kills_single_game}
                      isExpanded={!!expandedCategories.team_kills}
                      onToggle={() => toggleCategoryExpand('team_kills')}
                    />
                    <CrystalBallCard
                      icon="🛡️"
                      title="Najwyższe KDA drużynowe"
                      categoryKey="team_kda"
                      outcome={crystalBallData.team_highest_kda}
                      isExpanded={!!expandedCategories.team_kda}
                      onToggle={() => toggleCategoryExpand('team_kda')}
                    />
                  </div>
                </div>

                {/* Sekcja 3: Bohaterowie */}
                <div className="crystal-ball-section">
                  <h3 className="crystal-ball-section-title">⚔️ Statystyki Bohaterów (Champions)</h3>
                  <div className="crystal-ball-grid">
                    <CrystalBallCard
                      icon="⭐"
                      title="Najczęściej wybierany bohater"
                      categoryKey="champ_pick"
                      outcome={crystalBallData.champion_most_picked}
                      isExpanded={!!expandedCategories.champ_pick}
                      onToggle={() => toggleCategoryExpand('champ_pick')}
                    />
                    <CrystalBallCard
                      icon="🚫"
                      title="Najczęściej banowany (Fearless lockouts)"
                      categoryKey="champ_ban"
                      outcome={crystalBallData.champion_most_banned}
                      isExpanded={!!expandedCategories.champ_ban}
                      onToggle={() => toggleCategoryExpand('champ_ban')}
                    />
                    <CrystalBallCard
                      icon="🏆"
                      title="Najwyższy winrate (min. 5 gier)"
                      categoryKey="champ_wr"
                      outcome={crystalBallData.champion_highest_winrate}
                      isExpanded={!!expandedCategories.champ_wr}
                      onToggle={() => toggleCategoryExpand('champ_wr')}
                    />
                    <CrystalBallCard
                      icon="💀"
                      title="Najwięcej zgonów"
                      categoryKey="champ_deaths"
                      outcome={crystalBallData.champion_most_deaths}
                      isExpanded={!!expandedCategories.champ_deaths}
                      onToggle={() => toggleCategoryExpand('champ_deaths')}
                    />
                  </div>
                </div>

                {/* Sekcja 4: Turniej Ogółem */}
                <div className="crystal-ball-section">
                  <h3 className="crystal-ball-section-title">📊 Podsumowanie Turnieju (Tournament Totals)</h3>
                  <div className="crystal-ball-grid">
                    {/* Pentakille */}
                    <div className="crystal-ball-card">
                      <div className="crystal-ball-card-header">
                        <span className="crystal-ball-card-title">⚡ Rozkład Pentakilli w Turnieju</span>
                      </div>
                      <div style={{ marginTop: '0.5rem' }}>
                        {Object.entries(crystalBallData.tournament_total_pentakills).map(([count, prob]) => (
                          <div key={count} className="pentakill-pmf-row">
                            <span className="pentakill-pmf-label">{count} {Number(count) === 1 ? 'pentakill' : 'pentakilli'}</span>
                            <div className="pentakill-pmf-bar-track">
                              <div className="pentakill-pmf-bar-fill" style={{ width: `${Math.round(prob * 100)}%` }} />
                            </div>
                            <span className="pentakill-pmf-val">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Długość gier */}
                    <div className="crystal-ball-card">
                      <div className="crystal-ball-card-header">
                        <span className="crystal-ball-card-title">⏱️ Ekstremalny Czas Gry (Mapy)</span>
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <div style={{ background: '#1e293b', padding: '0.65rem', borderRadius: '6px' }}>
                          <strong style={{ color: '#38bdf8' }}>Najkrótsza gra turnieju:</strong>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                            <span>Średnia: <strong>{formatSec(crystalBallData.tournament_shortest_game_seconds.mean)}</strong></span>
                            <span>Mediana: <strong>{formatSec(crystalBallData.tournament_shortest_game_seconds.p50)}</strong></span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                            Zakres: {formatSec(crystalBallData.tournament_shortest_game_seconds.min)} – {formatSec(crystalBallData.tournament_shortest_game_seconds.max)}
                          </div>
                        </div>
                        <div style={{ background: '#1e293b', padding: '0.65rem', borderRadius: '6px' }}>
                          <strong style={{ color: '#ec4899' }}>Najdłuższa gra turnieju:</strong>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                            <span>Średnia: <strong>{formatSec(crystalBallData.tournament_longest_game_seconds.mean)}</strong></span>
                            <span>Mediana: <strong>{formatSec(crystalBallData.tournament_longest_game_seconds.p50)}</strong></span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                            Zakres: {formatSec(crystalBallData.tournament_longest_game_seconds.min)} – {formatSec(crystalBallData.tournament_longest_game_seconds.max)}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Barony */}
                    <div className="crystal-ball-card">
                      <div className="crystal-ball-card-header">
                        <span className="crystal-ball-card-title">🐉 Łączna Liczba Baronów</span>
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #334155' }}>
                          <span>Średnia w turnieju:</span>
                          <strong style={{ color: '#38bdf8' }}>{crystalBallData.tournament_total_barons.mean.toFixed(1)}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #334155' }}>
                          <span>Mediana (p50):</span>
                          <strong>{crystalBallData.tournament_total_barons.p50}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #334155' }}>
                          <span>Przedział 80% (p10 – p90):</span>
                          <strong>{crystalBallData.tournament_total_barons.p10} – {crystalBallData.tournament_total_barons.p90}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
                          <span>Wartość min – max:</span>
                          <span style={{ color: '#94a3b8' }}>{crystalBallData.tournament_total_barons.min} – {crystalBallData.tournament_total_barons.max}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>
        ) : (
          <div className={`worlds-layout ${worldsView}`}>
          {(worldsView === 'editor' || worldsView === 'split') && (
            <section className={`worlds-teams-editor ${worldsView === 'editor' ? 'full-width' : ''}`}>
            <h2>Skład Worlds 2026 — wybierz drużyny do oficjalnych slotów</h2>
            <p className="subtitle-sm">
              Wstępnie wypełnione zespoły odpowiadają slotom Leaguepedia/Fandom. Możesz je zmienić przed
              symulacją; regiony, seedy i pule pozostają zgodne z opublikowaną strukturą Worlds 2026.
            </p>

            <div className="worlds-stage-heading">
              <h3>Swiss — 15 bezpośrednich miejsc</h3>
              <span>Układ slotów Leaguepedia/Fandom</span>
            </div>
            <div className="worlds-pool-grid">
              {[1, 2, 3, 4].map((pool) => (
                <div key={pool} className="worlds-pool-card">
                  <h4>Pula {pool} <span>({pool === 4 ? '3 + Play-In' : '4'})</span></h4>
                  {directWorldsTeams.map((team, index) => (
                    team.pool === pool && (
                      <div key={index} className="worlds-team-slot">
                        <span className="worlds-slot-label">{DIRECT_SLOT_LABELS[index]}</span>
                        <div className="worlds-suggestion-container">
                          <input
                            type="text"
                            value={team.team}
                            onChange={(event) => handleUpdateWorldsTeam('direct', index, event.target.value)}
                            onFocus={() => setActiveSuggestionSlot({ stage: 'direct', index })}
                            onBlur={() => {
                              window.setTimeout(() => {
                                setActiveSuggestionSlot((current) => (
                                  current?.stage === 'direct' && current.index === index ? null : current
                                ));
                              }, 150);
                            }}
                            className="team-text-input"
                            placeholder={`Wybierz ${DIRECT_SLOT_LABELS[index]}`}
                            aria-label={`Drużyna Swiss ${index + 1}`}
                            autoComplete="off"
                          />
                          {activeSuggestionSlot?.stage === 'direct'
                            && activeSuggestionSlot.index === index
                            && visibleTeamSuggestions.length > 0 && (
                              <div className="worlds-suggestion-menu" role="listbox">
                                {visibleTeamSuggestions.map((suggestion) => (
                                  <button
                                    key={suggestion.name}
                                    type="button"
                                    className="worlds-suggestion-option"
                                    onMouseDown={(event) => event.preventDefault()}
                                    onClick={() => handleSelectSuggestedTeam('direct', index, suggestion.name)}
                                  >
                                    <span>{suggestion.name}</span>
                                    {suggestion.rating !== null && <small>GL {Math.round(suggestion.rating)}</small>}
                                  </button>
                                ))}
                              </div>
                            )}
                        </div>
                      </div>
                    )
                  ))}
                </div>
              ))}
            </div>

            <div className="worlds-stage-heading">
              <h3>Play-In — 4 zespoły</h3>
              <span>Double elimination Bo5; zwycięzca trafia do puli 4 Swiss</span>
            </div>
            <div className="play-in-grid">
              {playInWorldsTeams.map((team, index) => (
                <div key={index} className="worlds-team-slot">
                  <span className="worlds-slot-label">{PLAY_IN_SLOT_LABELS[index]}</span>
                  <div className="worlds-suggestion-container">
                    <input
                      type="text"
                      value={team.team}
                      onChange={(event) => handleUpdateWorldsTeam('playIn', index, event.target.value)}
                      onFocus={() => setActiveSuggestionSlot({ stage: 'playIn', index })}
                      onBlur={() => {
                        window.setTimeout(() => {
                          setActiveSuggestionSlot((current) => (
                            current?.stage === 'playIn' && current.index === index ? null : current
                          ));
                        }, 150);
                      }}
                      className="team-text-input"
                      placeholder={`Wybierz ${PLAY_IN_SLOT_LABELS[index]}`}
                      aria-label={`Drużyna Play-In ${index + 1}`}
                      autoComplete="off"
                    />
                    {activeSuggestionSlot?.stage === 'playIn'
                      && activeSuggestionSlot.index === index
                      && visibleTeamSuggestions.length > 0 && (
                        <div className="worlds-suggestion-menu" role="listbox">
                          {visibleTeamSuggestions.map((suggestion) => (
                            <button
                              key={suggestion.name}
                              type="button"
                              className="worlds-suggestion-option"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => handleSelectSuggestedTeam('playIn', index, suggestion.name)}
                            >
                              <span>{suggestion.name}</span>
                              {suggestion.rating !== null && <small>GL {Math.round(suggestion.rating)}</small>}
                            </button>
                          ))}
                        </div>
                      )}
                  </div>
                </div>
              ))}
            </div>
            <a
              className="worlds-source-link"
              href="https://lol.fandom.com/wiki/2026_Season_World_Championship"
              target="_blank"
              rel="noreferrer"
            >
              Źródło slotów: Leaguepedia / Fandom — Worlds 2026
            </a>
            {activeTeams.length > 0 && (
              <p className="worlds-suggestion-note">
                Podpowiedzi: {activeTeams.length} drużyn z dokładną, kompletną obecną piątką.
              </p>
            )}
            {teamSuggestionError && (
              <p className="worlds-suggestion-error">{teamSuggestionError}</p>
            )}
            </section>
          )}

          {(worldsView === 'results' || worldsView === 'split') && (
            <section className={`worlds-results-panel ${worldsView === 'results' ? 'full-width' : ''}`}>
              <div className="results-panel-header">
                <div>
                  <h2>🏆 Wyniki Symulacji Worlds 2026</h2>
                  {worldsData && (
                    <p className="subtitle-sm">
                      Symulacja Monte Carlo ({worldsData.simulations.toLocaleString('pl-PL')} prób): Play-In Bo5, Swiss Bo1/Bo3 i puchar Bo5.
                    </p>
                  )}
                </div>
                {worldsData && worldsView === 'results' && (
                  <button
                    type="button"
                    className="ghost-action-btn"
                    onClick={() => setWorldsView('editor')}
                  >
                    ✏️ Edytuj sloty drużyn
                  </button>
                )}
              </div>
              {worldsData && <TournamentInputProvenance response={worldsData} />}

              {worldsSimulating && (
                <div className="worlds-computing-dashboard">
                  <div className="worlds-computing-header">
                    <div className="computing-spinner large" />
                    <div className="worlds-computing-title-wrap">
                      <h3>Trwa symulacja Mistrzostw Świata Worlds 2026 ({worldsSimCount.toLocaleString('pl-PL')} prób)</h3>
                      <p>Konfigurowany scenariusz: Play-In Bo5 → 16-drużynowy Swiss Bo1/Bo3 → Faza pucharowa Bo5; reguły niezweryfikowane.</p>
                    </div>
                    <div className="computing-timer-chip large">
                      ⏱️ Czas obliczeń: <strong>{worldsElapsed.toFixed(1)} s</strong>
                    </div>
                  </div>
                  <div className="worlds-stage-indicators">
                    <div className={`worlds-stage-pill ${worldsElapsed >= 0 ? 'active' : ''}`}>
                      <span className="stage-num">1</span>
                      <span className="stage-icon">⚔️</span>
                      <div className="stage-info">
                        <strong>Play-In Stage</strong>
                        <small>4 pretendentów, podwójna eliminacja Bo5</small>
                      </div>
                    </div>
                    <div className={`worlds-stage-pill ${worldsElapsed >= 0.5 ? 'active' : ''}`}>
                      <span className="stage-num">2</span>
                      <span className="stage-icon">🔄</span>
                      <div className="stage-info">
                        <strong>Swiss Stage (5 rund)</strong>
                        <small>System Buchholza, mecze awansu/odpadnięcia Bo3</small>
                      </div>
                    </div>
                    <div className={`worlds-stage-pill ${worldsElapsed >= 1.2 ? 'active' : ''}`}>
                      <span className="stage-num">3</span>
                      <span className="stage-icon">🏆</span>
                      <div className="stage-info">
                        <strong>Faza Pucharowa (Knockout)</strong>
                        <small>Single Elimination Bo5 do Pucharu Świata</small>
                      </div>
                    </div>
                  </div>
                  <div className="computing-progress-track">
                    <div className="computing-progress-bar" />
                  </div>
                </div>
              )}

              {worldsData ? (
                <div className={worldsSimulating ? 'simulating-dimmed' : ''}>
                  <div className="summary-cards-grid">
                    <div className="summary-card gold-border">
                      <span className="card-label">Główny faworyt</span>
                      <strong className="card-value">{worldsData.standings[0]?.team || '—'}</strong>
                      <span className="card-sub">
                        {((worldsData.standings[0]?.champion_prob || 0) * 100).toFixed(1)}% szans na mistrzostwo
                      </span>
                    </div>
                    <div className="summary-card silver-border">
                      <span className="card-label">Drugi pretendent</span>
                      <strong className="card-value">{worldsData.standings[1]?.team || '—'}</strong>
                      <span className="card-sub">
                        {((worldsData.standings[1]?.champion_prob || 0) * 100).toFixed(1)}% mistrz · {((worldsData.standings[1]?.top2_prob || 0) * 100).toFixed(1)}% finał
                      </span>
                    </div>
                    <div className="summary-card">
                      <span className="card-label">Uczestnicy</span>
                      <strong className="card-value">{worldsData.standings.length} zespołów</strong>
                      <span className="card-sub">15 Swiss + 4 Play-In</span>
                    </div>
                    <div className="summary-card">
                      <span className="card-label">Liczba symulacji</span>
                      <strong className="card-value">{worldsData.simulations.toLocaleString('pl-PL')}</strong>
                      <span className="card-sub">Scenariusz, niezweryfikowane reguły</span>
                    </div>
                  </div>

                  <div className="table-responsive">
                    <table className="standings-table">
                      <thead>
                        <tr>
                          <th className="th-rank">Poz.</th>
                          <th className="th-team">Drużyna</th>
                          <th className="th-meta">Region i etap</th>
                          <th className="th-num">Play-In → Swiss</th>
                          <th className="th-num">🥇 Puchar Worlds</th>
                          <th className="th-num">🥈 Finał</th>
                          <th className="th-num">Top 4</th>
                          <th className="th-num">Swiss Top 8</th>
                        </tr>
                      </thead>
                      <tbody>
                        {worldsData.standings.map((standing, idx) => (
                          <tr key={standing.team}>
                            <td className="rank-cell">{idx + 1}</td>
                            <td className="team-cell"><strong>{standing.team}</strong></td>
                            <td className="meta-cell">
                              <span className="badge region-badge">{standing.region}</span>
                              <span className="badge stage-badge">
                                {standing.stage === 'play_in' ? 'Play-In' : `Swiss P${standing.pool}`}
                              </span>
                            </td>
                            <td className="prob-cell">{(standing.play_in_qualifier_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell champ-prob">
                              <div className="prob-with-bar">
                                <div className="mini-bar-bg">
                                  <div
                                    className="mini-bar-fill gold"
                                    style={{ width: `${Math.min(100, standing.champion_prob * 100)}%` }}
                                  />
                                </div>
                                <span>{(standing.champion_prob * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                            <td className="prob-cell">{(standing.top2_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell">{(standing.top4_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell">{(standing.top8_swiss_prob * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : worldsSimulating ? (
                <div className="standings-skeleton-card">
                  <div className="skeleton-header-bar" />
                  <div className="skeleton-table">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
                      <div key={i} className="skeleton-table-row">
                        <div className="skeleton-cell rank" />
                        <div className="skeleton-cell team" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <p>
                    Wpisz własne 15 bezpośrednich uczestników i 4 zespoły Play-In (lub skorzystaj z gotowych slotów Fandom).
                    Kliknij <strong>⚡ Symuluj Worlds</strong> w pasku powyżej, aby wygenerować pełną tabelę wyników.
                  </p>
                  {worldsView === 'results' && (
                    <button
                      type="button"
                      className="ghost-action-btn"
                      style={{ marginTop: '1rem' }}
                      onClick={() => setWorldsView('editor')}
                    >
                      ⚙️ Otwórz edytor slotów drużyn
                    </button>
                  )}
                </div>
              )}
            </section>
          )}
        </div>
        )
      ) : (
        <div className={`enc-layout ${encView}`}>
          {encConfigurationError && (
            <div className="enc-blocking-issues" role="alert">
              <p>{encConfigurationError}</p>
              <button type="button" className="ghost-action-btn" onClick={() => void loadEncConfiguration(true)} disabled={encLoading}>
                {encLoading ? 'Ładowanie...' : 'Spróbuj ponownie'}
              </button>
            </div>
          )}
          {(encView === 'rosters' || encView === 'split') && (
            <section className={`enc-rosters-panel ${encView === 'rosters' ? 'full-width' : ''}`}>
            <h2>ENC 2027 — najwyższy aktualny GL w roli z Fandom</h2>
            {encLoading && <div className="empty-state"><p>Ładowanie składów i snapshotu GL...</p></div>}
            {encConfiguration && (
              <>
                <p className="subtitle-sm">
                  Każda rola wybiera najwyższy rating GL wyłącznie wśród zawodników przypisanych do niej w publicznej tabeli kadry Fandom.
                  Zawodnik bez wpisu GL dostaje jawny rating domyślny {encConfiguration.default_rating.toFixed(1)}; średnia pięciu ról jest wskaźnikiem
                  siły kadry, nie predykcją modelu EXP-039.
                </p>
                <div className="enc-format-grid">
                  <div><strong>Play-In</strong><span>{encConfiguration.format.play_in}</span></div>
                  <div><strong>Group Stage</strong><span>{encConfiguration.format.group_stage}</span></div>
                  <div><strong>Playoffs</strong><span>{encConfiguration.format.playoffs}</span></div>
                </div>
                {!encConfiguration.simulation_ready && (
                  <div className="enc-blocking-issues">
                    <strong>Symulacja wymaga przypisania wszystkich pięciu ról.</strong>
                    <ul>{encConfiguration.blocking_issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
                  </div>
                )}
                {encConfiguration.teams.some((team) => (
                  team.selected_roster.some((player) => player.rating_source === 'default')
                )) && (
                  <p className="enc-default-note">
                    <strong>GL domyślny {encConfiguration.default_rating.toFixed(1)}:</strong> użyty wyłącznie tam, gdzie ogłoszony zawodnik nie ma
                    aktualnego wpisu GL. Wiersze są oznaczone „domyślny”.
                  </p>
                )}
                <a
                  className="worlds-source-link"
                  href={encConfiguration.source_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Źródło struktury i kadr: Leaguepedia / Fandom — Esports Nations Cup 2027
                </a>
                <p className="enc-draw-note">{encConfiguration.format.draw_and_tiebreak_policy}</p>
                <div className="enc-team-grid">
                  {encConfiguration.teams.map((team) => (
                    <article key={team.nation} className={`enc-team-card ${team.selection_status}`}>
                      <div className="enc-team-heading">
                        <h3>{team.nation}</h3>
                        <span>{team.entry_stage === 'group_stage' ? 'bezpośrednio Group Stage' : 'Play-In'}</span>
                      </div>
                      {team.ranking !== null && <p className="enc-ranking">Ranking zaproszeń #{team.ranking}</p>}
                      {team.selected_roster.map((player) => (
                        <div className="enc-player-row" key={player.role}>
                          <span>{player.role}</span>
                          <strong>{player.player}</strong>
                          <small>
                            GL {player.rating.toFixed(1)}
                            {player.rating_source === 'default' ? ' · domyślny' : ''}
                          </small>
                        </div>
                      ))}
                      {team.missing_roles.length > 0 && (
                        <p className="enc-missing-roles">Brak ratingu/roli: {team.missing_roles.join(', ')}</p>
                      )}
                      {team.roster_rating !== null && <p className="enc-roster-rating">Średnia GL: {team.roster_rating.toFixed(1)}</p>}
                      {team.source_roster.length > 0 && (
                        <details>
                          <summary>Ogłoszona kadra ({team.source_roster.length})</summary>
                          <p>{team.source_roster.join(' · ')}</p>
                        </details>
                      )}
                    </article>
                  ))}
                </div>
              </>
            )}
            </section>
          )}

          {(encView === 'results' || encView === 'split') && (
            <section className={`enc-results-panel ${encView === 'results' ? 'full-width' : ''}`}>
              <div className="results-panel-header">
                <div>
                  <h2>🏆 Wyniki ENC 2027</h2>
                  {encData && (
                    <p className="subtitle-sm">
                      Symulacja Monte Carlo ({encData.simulations.toLocaleString('pl-PL')} prób): Play-In Bo1, grupy Bo3 i puchar Bo3/Bo5.
                    </p>
                  )}
                </div>
                {encData && encView === 'results' && (
                  <button
                    type="button"
                    className="ghost-action-btn"
                    onClick={() => setEncView('rosters')}
                  >
                    👥 Zobacz składy 16 reprezentacji
                  </button>
                )}
              </div>
              {encData && <TournamentInputProvenance response={encData} />}

              {encSimulating && (
                <div className="simulation-computing-banner">
                  <div className="computing-header-row">
                    <div className="computing-spinner" />
                    <div className="computing-meta">
                      <strong>Trwa symulacja European Nations Cup ({encSimCount.toLocaleString('pl-PL')} prób)...</strong>
                      <span className="computing-sub">
                        Faza grupowa 32 nacji (8 grup po 4) → Drabinka pucharowa 16 reprezentacji
                      </span>
                    </div>
                    <div className="computing-timer-chip">
                      ⏱️ <strong>{encElapsed.toFixed(1)} s</strong>
                    </div>
                  </div>
                  <div className="computing-progress-track">
                    <div className="computing-progress-bar" />
                  </div>
                </div>
              )}

              {encData ? (
                <div className={encSimulating ? 'simulating-dimmed' : ''}>
                  <div className="summary-cards-grid">
                    <div className="summary-card gold-border">
                      <span className="card-label">Główny faworyt ENC</span>
                      <strong className="card-value">{encData.standings[0]?.nation || '—'}</strong>
                      <span className="card-sub">
                        {((encData.standings[0]?.champion_prob || 0) * 100).toFixed(1)}% mistrz · GL {encData.standings[0]?.roster_rating?.toFixed(1) ?? 'nieznane'}
                      </span>
                    </div>
                    <div className="summary-card silver-border">
                      <span className="card-label">Drugi pretendent / Finał</span>
                      <strong className="card-value">{encData.standings[1]?.nation || '—'}</strong>
                      <span className="card-sub">
                        {((encData.standings[1]?.champion_prob || 0) * 100).toFixed(1)}% mistrz · {((encData.standings[1]?.top2_prob || 0) * 100).toFixed(1)}% finał
                      </span>
                    </div>
                    <div className="summary-card">
                      <span className="card-label">Reprezentacje</span>
                      <strong className="card-value">{encData.standings.length} nacji</strong>
                      <span className="card-sub">Konfigurowany scenariusz ENC</span>
                    </div>
                    <div className="summary-card">
                      <span className="card-label">Próby Monte Carlo</span>
                      <strong className="card-value">{encData.simulations.toLocaleString('pl-PL')}</strong>
                      <span className="card-sub">Play-In → Grupy Bo3 → Puchar</span>
                    </div>
                  </div>

                  <div className="table-responsive">
                    <table className="standings-table">
                      <thead>
                        <tr>
                          <th className="th-rank">Poz.</th>
                          <th className="th-team">Reprezentacja</th>
                          <th className="th-meta">Etap wejścia</th>
                          <th className="th-rating">GL składu</th>
                          <th className="th-num">🥇 Mistrz ENC</th>
                          <th className="th-num">🥈 Finał</th>
                          <th className="th-num">Top 4</th>
                          <th className="th-num">Playoffs</th>
                          <th className="th-num">Group Stage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {encData.standings.map((standing, idx) => (
                          <tr key={standing.nation}>
                            <td className="rank-cell">{idx + 1}</td>
                            <td className="team-cell"><strong>{standing.nation}</strong></td>
                            <td className="meta-cell">
                              <span className={`badge ${standing.entry_stage === 'group_stage' ? 'stage-group-badge' : 'stage-playin-badge'}`}>
                                {standing.entry_stage === 'group_stage' ? 'Faza grupowa' : 'Play-In'}
                              </span>
                            </td>
                            <td className="rating-cell">
                              <span className="rating-badge">{standing.roster_rating.toFixed(1)}</span>
                            </td>
                            <td className="prob-cell champ-prob">
                              <div className="prob-with-bar">
                                <div className="mini-bar-bg">
                                  <div
                                    className="mini-bar-fill gold"
                                    style={{ width: `${Math.min(100, standing.champion_prob * 100)}%` }}
                                  />
                                </div>
                                <span>{(standing.champion_prob * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                            <td className="prob-cell">{(standing.top2_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell">{(standing.top4_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell">{(standing.playoff_prob * 100).toFixed(1)}%</td>
                            <td className="prob-cell">{(standing.group_stage_prob * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : encSimulating ? (
                <div className="standings-skeleton-card">
                  <div className="skeleton-header-bar" />
                  <div className="skeleton-table">
                    {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                      <div key={i} className="skeleton-table-row">
                        <div className="skeleton-cell rank" />
                        <div className="skeleton-cell team" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                        <div className="skeleton-cell prob" />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <p>
                    Uruchom symulację, aby rozegrać cztery grupy Play-In Bo1, cztery grupy główne Bo3 i puchar
                    Bo3/Bo5 zgodnie z opublikowanym formatem.
                  </p>
                  {encView === 'results' && (
                    <button
                      type="button"
                      className="ghost-action-btn"
                      style={{ marginTop: '1rem' }}
                      onClick={() => setEncView('rosters')}
                    >
                      👥 Otwórz składy reprezentacji
                    </button>
                  )}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
