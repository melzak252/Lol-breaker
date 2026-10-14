import { useState, useEffect, useRef } from 'react';
import { fetchActiveTeams, simulateMatchup } from '../api/client';
import type { MatchupSimulationResponse } from '../types';
import './MatchupSimulator.css';

interface TeamOption {
  team_row_id: number;
  native_team_id: string | null;
  name: string;
  rating: number | null;
  games?: number | null;
  last_active?: string | null;
}


export default function MatchupSimulator() {
  const [activeTeams, setActiveTeams] = useState<TeamOption[]>([]);
  const [teamA, setTeamA] = useState('');
  const [teamB, setTeamB] = useState('');
  const [bestOf, setBestOf] = useState<number>(3);
  const [loadingTeams, setLoadingTeams] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MatchupSimulationResponse | null>(null);
  const latestRequest = useRef(0);
  const clearCurrentResult = () => {
    latestRequest.current += 1;
    setResult(null);
    setLoading(false);
    setError(null);
  };

  // Load active teams; no local team or rating fallback is a valid C0 input.
  useEffect(() => {
    fetchActiveTeams()
      .then((res) => {
        const teams = res?.teams ?? [];
        setActiveTeams(teams);
        setTeamA(teams[0]?.name ?? '');
        setTeamB(teams[1]?.name ?? '');
        setError(teams.length >= 2 ? null : 'Są potrzebne co najmniej dwie aktywne drużyny.');
      })
      .catch((err: unknown) => {
        setActiveTeams([]);
        setTeamA('');
        setTeamB('');
        setError(err instanceof Error && err.message ? err.message : 'Nie udało się pobrać aktywnych drużyn.');
      })
      .finally(() => {
        setLoadingTeams(false);
      });
  }, []);

  const handleSimulate = async (nameA = teamA, nameB = teamB, bo = bestOf) => {
    const requestId = ++latestRequest.current;
    setResult(null);
    if (!nameA.trim() || !nameB.trim()) {
      setLoading(false);
      setError('Wybierz obie drużyny.');
      return;
    }
    if (nameA === nameB) {
      setLoading(false);
      setError('Wybierz dwie różne drużyny do zestawienia.');
      return;
    }
    if (!activeTeams.some((team) => team.name === nameA) || !activeTeams.some((team) => team.name === nameB)) {
      setLoading(false);
      setError('Wybierz drużyny z załadowanej listy aktywnych zespołów.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const selectedA = activeTeams.find((team) => team.name === nameA);
      const selectedB = activeTeams.find((team) => team.name === nameB);
      const data = await simulateMatchup({
        team_a_name: nameA,
        team_b_name: nameB,
        team_a_team_row_id: selectedA?.team_row_id,
        team_b_team_row_id: selectedB?.team_row_id,
        native_team_a_id: selectedA?.native_team_id ?? undefined,
        native_team_b_id: selectedB?.native_team_id ?? undefined,
        best_of: bo,
      });
      if (requestId === latestRequest.current) setResult(data);
    } catch (err: unknown) {
      if (requestId === latestRequest.current) {
        setError(err instanceof Error && err.message ? err.message : 'Błąd podczas symulacji starcia.');
      }
    } finally {
      if (requestId === latestRequest.current) setLoading(false);
    }
  };

  // Simulate only after loading at least two distinct, real team entries.
  useEffect(() => {
    if (loadingTeams || activeTeams.length < 2 || !teamA || !teamB) return;
    if (!activeTeams.some((team) => team.name === teamA) || !activeTeams.some((team) => team.name === teamB)) return;
    if (teamA === teamB) {
      setResult(null);
      setError('Wybierz dwie różne drużyny do zestawienia.');
      return;
    }
    handleSimulate(teamA, teamB, bestOf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTeams, loadingTeams, teamA, teamB, bestOf]);

  const swapTeams = () => {
    const temp = teamA;
    setTeamA(teamB);
    setTeamB(temp);
  };

  return (
    <div className="matchup-page">
      <header className="matchup-header">
        <p className="eyebrow">Symulator starć bezpośrednich</p>
        <h1>Matchup Simulator (H2H)</h1>
        <p className="subtitle">
          Wybierz aktywne drużyny, ustaw format serii i sprawdź bezpośrednią predykcję C0.
        </p>
      </header>

      <section className="matchup-controls-card">
        {loadingTeams && <p className="status-loading">Ładowanie aktywnych drużyn…</p>}
        <div className="matchup-dropdowns-row">
          <div className="team-select-container">
            <label htmlFor="select-team-a">Drużyna A</label>
            <div className="select-wrapper">
              <select
                id="select-team-a"
                value={teamA}
                onChange={(e) => {
                  clearCurrentResult();
                  setTeamA(e.target.value);
                }}
                disabled={loadingTeams || activeTeams.length < 2}
              >
                <option value="" disabled>Wybierz drużynę</option>
                {activeTeams.map((t) => (
                  <option key={`a-${t.name}`} value={t.name}>
                    {t.name} {t.rating ? `(Glicko ${Math.round(t.rating)})` : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="button"
            className="swap-teams-btn"
            onClick={() => {
              clearCurrentResult();
              swapTeams();
            }}
            title="Zamień drużyny"
            aria-label="Zamień drużyny"
            disabled={loadingTeams || activeTeams.length < 2}
          >
            ⇄
          </button>

          <div className="team-select-container">
            <label htmlFor="select-team-b">Drużyna B</label>
            <div className="select-wrapper">
              <select
                id="select-team-b"
                value={teamB}
                onChange={(e) => {
                  clearCurrentResult();
                  setTeamB(e.target.value);
                }}
                disabled={loadingTeams || activeTeams.length < 2}
              >
                <option value="" disabled>Wybierz drużynę</option>
                {activeTeams.map((t) => (
                  <option key={`b-${t.name}`} value={t.name}>
                    {t.name} {t.rating ? `(Glicko ${Math.round(t.rating)})` : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="matchup-options-bar">
          <div className="format-selection-group">
            <span className="label">Format serii:</span>
            <div className="bo-pills">
              {[1, 3, 5].map((b) => (
                <button
                  key={b}
                  type="button"
                  className={`pill-btn ${bestOf === b ? 'active' : ''}`}
                  onClick={() => {
                    clearCurrentResult();
                    setBestOf(b);
                  }}
                >
                  Bo{b}
                </button>
              ))}
            </div>
          </div>

          <div className="quick-info">
            {loading ? (
              <span className="status-loading">⏳ Obliczanie predykcji C0...</span>
            ) : (
              <span className="status-ready">Model: {result ? `${result.model_name} (${result.model_version})` : 'Causal-C0 (c0-native-2026-w32-e12-v1)'}</span>
            )}
          </div>
        </div>

        {error && <div className="matchup-error">{error}</div>}
      </section>

      {result && (
        <section className="matchup-view-section">
          {/* Main probability banner */}
          <div className="matchup-hero-card">
            <div className="team-hero-box left">
              <span className="side-indicator">Drużyna A</span>
              <h2>{result.team_a_name}</h2>
              <div className="prob-big">{(result.series_prob_a * 100).toFixed(1)}%</div>
              <div className="prob-sub">Prawdopodobieństwo całej serii C0</div>
            </div>

            <div className="vs-center-col">
              <div className="series-pill">Seria Bo{result.best_of}</div>
              <div className="h2h-bar">
                <div
                  className="bar-fill-a"
                  style={{ width: `${result.series_prob_a * 100}%` }}
                />
              </div>
              <span className="binomial-formula-hint">C0 prognozuje wybraną serię bezpośrednio; prawdopodobieństwo mapy jest niedostępne.</span>
            </div>

            <div className="team-hero-box right">
              <span className="side-indicator">Drużyna B</span>
              <h2>{result.team_b_name}</h2>
              <div className="prob-big">{(result.series_prob_b * 100).toFixed(1)}%</div>
              <div className="prob-sub">Prawdopodobieństwo całej serii C0</div>
            </div>
          </div>

          <div className="breakdown-cards-row">
            <div className="breakdown-card">
              <span className="bd-title">Predykcja natywnego C0</span>
              <small className="bd-desc">
                Prawdopodobieństwo serii wyliczono bezpośrednio z natywnego snapshotu historii i składu.
                Model nie zwraca prawdopodobieństwa pojedynczej mapy ani rozkładu niepewności.
              </small>
            </div>
          </div>

          {/* Rosters comparison - PROMINENT 5v5 VIEW */}
          <div className="rosters-comparison-container">
            <div className="roster-block">
              <div className="roster-header">
                <div>
                  <h3>Skład {result.team_a_name}</h3>
                  <p className="roster-meta">
                    Średni Glicko: <strong>{result.roster_a?.avg_glicko ? Math.round(result.roster_a.avg_glicko) : '—'}</strong>
                    {result.roster_a?.avg_glicko_rd ? ` (RD ±${Math.round(result.roster_a.avg_glicko_rd)})` : ''}
                  </p>
                </div>
              </div>
              <div className="roster-table">
                <div className="roster-row-header">
                  <span>Rola</span>
                  <span>Zawodnik</span>
                  <span className="text-right">Glicko</span>
                  <span className="text-right">Pewność (RD)</span>
                </div>
                {result.roster_a?.players?.length ? (
                  result.roster_a.players.map((player, idx) => (
                    <div className="roster-row" key={`ra-${idx}`}>
                      <span className={`role-badge ${player.role?.toLowerCase()}`}>{player.role || '—'}</span>
                      <span className="player-title">{player.player_name || 'Nieznany'}</span>
                      <span className="player-rating text-right">
                        {player.glicko_rating ? Math.round(player.glicko_rating) : '—'}
                      </span>
                      <span className="player-rd text-right">
                        {player.glicko_rd ? `±${Math.round(player.glicko_rd)}` : '—'}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="no-roster-msg">Brak zapisanego składu w bazie</div>
                )}
              </div>
            </div>

            <div className="roster-block">
              <div className="roster-header">
                <div>
                  <h3>Skład {result.team_b_name}</h3>
                  <p className="roster-meta">
                    Średni Glicko: <strong>{result.roster_b?.avg_glicko ? Math.round(result.roster_b.avg_glicko) : '—'}</strong>
                    {result.roster_b?.avg_glicko_rd ? ` (RD ±${Math.round(result.roster_b.avg_glicko_rd)})` : ''}
                  </p>
                </div>
              </div>
              <div className="roster-table">
                <div className="roster-row-header">
                  <span>Rola</span>
                  <span>Zawodnik</span>
                  <span className="text-right">Glicko</span>
                  <span className="text-right">Pewność (RD)</span>
                </div>
                {result.roster_b?.players?.length ? (
                  result.roster_b.players.map((player, idx) => (
                    <div className="roster-row" key={`rb-${idx}`}>
                      <span className={`role-badge ${player.role?.toLowerCase()}`}>{player.role || '—'}</span>
                      <span className="player-title">{player.player_name || 'Nieznany'}</span>
                      <span className="player-rating text-right">
                        {player.glicko_rating ? Math.round(player.glicko_rating) : '—'}
                      </span>
                      <span className="player-rd text-right">
                        {player.glicko_rd ? `±${Math.round(player.glicko_rd)}` : '—'}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="no-roster-msg">Brak zapisanego składu w bazie</div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
