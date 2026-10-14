import { useEffect, useMemo, useState } from 'react';
import {
  blockTeamAlias,
  createTeamAlias,
  searchGolggTeams,
  unblockTeamAlias,
} from '../api/client';
import type {
  GolggMatchCandidate,
  MappingCheckResponse,
  MappingReviewItem,
  MappingReviewResponse,
  UnmappedMatchItem,
  UnmappedMatchesResponse,
} from '../types';
import './ManualMapping.css';

const API_BASE = '/api';

type StatusFilter = 'upcoming' | 'expired' | 'finished';
type AliasSide = 'a' | 'b';

const STATUS_LABELS: Record<StatusFilter, string> = {
  upcoming: 'Nadchodzące',
  expired: 'Wygasłe',
  finished: 'Zakończone',
};

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

const formatDateTime = (iso: string | null) => {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('pl-PL', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const formatDate = (iso: string | null) => {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('pl-PL', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
};

const isTeamMapped = (mapping: UnmappedMatchItem['team_a_mapping']) => Boolean(mapping?.golgg_name);

const needsTeamAttention = (mapping: UnmappedMatchItem['team_a_mapping']) => !isTeamMapped(mapping);

const getInitialAliasSide = (match: UnmappedMatchItem): AliasSide => (
  needsTeamAttention(match.team_a_mapping) ? 'a' : needsTeamAttention(match.team_b_mapping) ? 'b' : 'a'
);

const mappingBadge = (mapping: UnmappedMatchItem['team_a_mapping']) => {
  if (mapping?.golgg_name) {
    return {
      className: 'mapped',
      label: `OK → ${mapping.golgg_name}`,
    };
  }
  if (mapping?.source === 'blocked') {
    return {
      className: 'blocked',
      label: 'Zablokowane / brak mapowania',
    };
  }
  return {
    className: 'missing',
    label: 'Brak aliasu',
  };
};

const matchDiagnosis = (match: UnmappedMatchItem) => {
  const teamAMissing = needsTeamAttention(match.team_a_mapping);
  const teamBMissing = needsTeamAttention(match.team_b_mapping);
  if (teamAMissing && teamBMissing) return 'Brakuje mapowania obu drużyn';
  if (teamAMissing) return `Brakuje mapowania: ${match.team_a_name}`;
  if (teamBMissing) return `Brakuje mapowania: ${match.team_b_name}`;
  return 'Drużyny są rozpoznane — brakuje tylko mapowania meczu do GOL.GG';
};

export default function ManualMapping() {
  const [unmappedMatches, setUnmappedMatches] = useState<UnmappedMatchItem[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<UnmappedMatchItem | null>(null);
  const [candidates, setCandidates] = useState<GolggMatchCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [candidateLoading, setCandidateLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('upcoming');
  const [searchQuery, setSearchQuery] = useState('');
  const [onlyUnrecognizedTeams, setOnlyUnrecognizedTeams] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [reviewItems, setReviewItems] = useState<MappingReviewItem[]>([]);
  const [selectedReview, setSelectedReview] = useState<MappingReviewItem | null>(null);
  const [reviewReason, setReviewReason] = useState('');
  const [reviewOperator, setReviewOperator] = useState('');
  const [reviewToken, setReviewToken] = useState('');
  const [replacementId, setReplacementId] = useState('');
  const [reviewSaving, setReviewSaving] = useState(false);

  const [aliasSide, setAliasSide] = useState<AliasSide>('a');
  const [aliasQuery, setAliasQuery] = useState('');
  const [aliasResults, setAliasResults] = useState<string[]>([]);
  const [selectedGolggTeam, setSelectedGolggTeam] = useState('');
  const [aliasSaving, setAliasSaving] = useState(false);

  const [manualId, setManualId] = useState('');
  const [checkResult, setCheckResult] = useState<MappingCheckResponse | null>(null);
  const [checking, setChecking] = useState(false);
  const [mappingMatch, setMappingMatch] = useState(false);

  const rawAliasName = useMemo(() => {
    if (!selectedMatch) return '';
    return aliasSide === 'a' ? selectedMatch.team_a_name : selectedMatch.team_b_name;
  }, [aliasSide, selectedMatch]);

  const filteredMatches = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return unmappedMatches.filter((m) => {
      if (onlyUnrecognizedTeams && !needsTeamAttention(m.team_a_mapping) && !needsTeamAttention(m.team_b_mapping)) {
        return false;
      }
      if (!q) return true;
      return `${m.canonical_match_id} ${m.team_a_name} ${m.team_b_name} ${m.league || ''} ${m.bookmakers?.join(' ') || ''}`
        .toLowerCase()
        .includes(q);
    });
  }, [onlyUnrecognizedTeams, searchQuery, unmappedMatches]);

  const fetchUnmapped = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ status: statusFilter, limit: '250' });
      const data = await fetchJson<UnmappedMatchesResponse>(`${API_BASE}/matches/unmapped?${params}`);
      setUnmappedMatches(data.matches);
      if (selectedMatch && !data.matches.some((m) => m.canonical_match_id === selectedMatch.canonical_match_id)) {
        setSelectedMatch(null);
        setCandidates([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się pobrać listy meczów');
    } finally {
      setLoading(false);
    }
  };

  const fetchReview = async () => {
    try {
      const data = await fetchJson<MappingReviewResponse>(`${API_BASE}/matches/mapping-review?limit=100`);
      setReviewItems(data.items);
      if (selectedReview && !data.items.some((item) => item.canonical_match_id === selectedReview.canonical_match_id)) {
        setSelectedReview(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się pobrać kolejki kontroli');
    }
  };

  const handleReviewDecision = async (decision: 'retain' | 'replace' | 'invalidate') => {
    if (!selectedReview || reviewReason.trim().length < 8 || reviewOperator.trim().length < 2) return;
    setReviewSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await fetchJson(`${API_BASE}/matches/mapping-review/decision`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Identity-Review-Token': reviewToken,
        },
        body: JSON.stringify({
          canonical_match_id: selectedReview.canonical_match_id,
          decision,
          reason: reviewReason.trim(),
          operator: reviewOperator.trim(),
          new_golgg_match_id: decision === 'replace' ? replacementId.trim() : null,
        }),
      });
      setSuccess(`Zapisano decyzję ${decision} dla #${selectedReview.canonical_match_id}`);
      setReviewReason('');
      setReplacementId('');
      setSelectedReview(null);
      await fetchReview();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się zapisać decyzji');
    } finally {
      setReviewSaving(false);
    }
  };

  useEffect(() => {
    fetchUnmapped();
    fetchReview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter]);

  useEffect(() => {
    if (!selectedMatch) return;
    const initialQuery = aliasSide === 'a' ? selectedMatch.team_a_name : selectedMatch.team_b_name;
    setAliasQuery(initialQuery || '');
    setSelectedGolggTeam('');
    setAliasResults([]);
  }, [aliasSide, selectedMatch]);

  useEffect(() => {
    const q = aliasQuery.trim();
    if (!q) {
      setAliasResults([]);
      return;
    }
    const handle = window.setTimeout(async () => {
      try {
        const data = await searchGolggTeams(q, 20);
        setAliasResults(data.teams);
      } catch {
        setAliasResults([]);
      }
    }, 250);
    return () => window.clearTimeout(handle);
  }, [aliasQuery]);

  const handleSelectMatch = async (match: UnmappedMatchItem) => {
    setSelectedMatch(match);
    setAliasSide(getInitialAliasSide(match));
    setCandidates([]);
    setManualId('');
    setCheckResult(null);
    setError(null);
    setSuccess(null);
    setCandidateLoading(true);
    try {
      const data = await fetchJson<{ candidates: GolggMatchCandidate[] }>(
        `${API_BASE}/matches/${match.canonical_match_id}/mapping-candidates`
      );
      setCandidates(data.candidates);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się pobrać kandydatów GOL.GG');
    } finally {
      setCandidateLoading(false);
    }
  };

  const handleSaveAlias = async () => {
    if (!rawAliasName || !selectedGolggTeam) return;
    setAliasSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await createTeamAlias({
        raw_name: rawAliasName,
        golgg_team_name: selectedGolggTeam,
        source_system: 'bookmaker',
        league_pattern: selectedMatch?.league || undefined,
      });
      setSuccess(`Dodano alias: ${rawAliasName} → ${selectedGolggTeam}`);
      await fetchUnmapped();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się zapisać aliasu');
    } finally {
      setAliasSaving(false);
    }
  };

  const handleBlockAlias = async () => {
    if (!rawAliasName) return;
    setAliasSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await blockTeamAlias(rawAliasName);
      setSuccess(`Zablokowano automapowanie dla: ${rawAliasName}`);
      await fetchUnmapped();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się zablokować aliasu');
    } finally {
      setAliasSaving(false);
    }
  };

  const handleUnblockAlias = async () => {
    if (!rawAliasName) return;
    setAliasSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await unblockTeamAlias(rawAliasName);
      setSuccess(`Odblokowano automapowanie dla: ${rawAliasName}`);
      await fetchUnmapped();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się odblokować aliasu');
    } finally {
      setAliasSaving(false);
    }
  };

  const handleCheckId = async () => {
    if (!manualId.trim()) return;
    setChecking(true);
    setError(null);
    setCheckResult(null);
    try {
      const data = await fetchJson<MappingCheckResponse>(`${API_BASE}/matches/mapping-check/${manualId.trim()}`);
      setCheckResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się sprawdzić ID GOL.GG');
    } finally {
      setChecking(false);
    }
  };

  const handleMapMatch = async (golggMatchId: string | number) => {
    if (!selectedMatch) return;
    setMappingMatch(true);
    setError(null);
    setSuccess(null);
    try {
      await fetchJson<{ ok: boolean }>(`${API_BASE}/matches/map`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          canonical_match_id: selectedMatch.canonical_match_id,
          golgg_match_id: String(golggMatchId),
        }),
      });
      setSuccess(`Zmapowano mecz #${selectedMatch.canonical_match_id} → GOL.GG #${golggMatchId}`);
      setSelectedMatch(null);
      setCandidates([]);
      setManualId('');
      setCheckResult(null);
      await fetchUnmapped();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Nie udało się zmapować meczu');
    } finally {
      setMappingMatch(false);
    }
  };

  return (
    <div className="manual-mapping-page">
      <div className="mapping-page-header">
        <div>
          <h1>Mapowanie</h1>
          <p className="mapping-subtitle">
            Najpierw napraw mapowania drużyn aliasami. Mapowanie meczu do GOL.GG zostaw jako awaryjny krok dla wyników historycznych.
          </p>
        </div>
        <button className="secondary-btn" onClick={fetchUnmapped} disabled={loading}>
          {loading ? 'Odświeżanie…' : 'Odśwież'}
        </button>
      </div>

      {(error || success) && (
        <div className={`mapping-alert ${error ? 'error' : 'success'}`}>
          {error || success}
        </div>
      )}
      <section className="mapping-card">
        <div className="section-header">
          <div>
            <h2>Kontrola istniejących mapowań</h2>
            <span>{reviewItems.length} linków poza bezpieczną bramką</span>
          </div>
          <button className="secondary-btn" onClick={fetchReview} type="button">Odśwież kontrolę</button>
        </div>
        <div className="mapping-container">
          <div className="scroll-list">
            {reviewItems.map((item) => (
              <button
                className={`match-item ${selectedReview?.canonical_match_id === item.canonical_match_id ? 'selected' : ''}`}
                key={item.mapping_id}
                onClick={() => setSelectedReview(item)}
                type="button"
              >
                <div className="match-row-top"><span className="match-id">#{item.canonical_match_id}</span><span>{item.confidence.toFixed(3)}</span></div>
                <div className="match-teams">{item.canonical_team_a} vs {item.canonical_team_b}</div>
                <div className="match-sources">{item.canonical_date} · {item.canonical_competition}</div>
                <div className="match-diagnosis">{item.reasons.join(', ')}</div>
              </button>
            ))}
          </div>
          <div className="mapping-workspace">
            {!selectedReview ? <div className="empty-state">Wybierz link wymagający kontroli.</div> : (
              <div className="mapping-card priority-card">
                <h3>Canonical #{selectedReview.canonical_match_id}</h3>
                <p>{selectedReview.canonical_team_a} vs {selectedReview.canonical_team_b} · {selectedReview.canonical_date} · {selectedReview.canonical_competition}</p>
                <h3>GOL.GG #{selectedReview.golgg_match_id}</h3>
                <p>{selectedReview.golgg_team_a} vs {selectedReview.golgg_team_b} · {selectedReview.golgg_date} · {selectedReview.golgg_competition}</p>
                <p>Predykcje: {selectedReview.prediction_count} · cechy: {selectedReview.feature_count} · sygnały: {selectedReview.signal_count} · zakłady: {selectedReview.bet_count}</p>
                <label className="field-label">Operator<input value={reviewOperator} onChange={(event) => setReviewOperator(event.target.value)} /></label>
                <label className="field-label">Token kontroli<input type="password" autoComplete="off" value={reviewToken} onChange={(event) => setReviewToken(event.target.value)} /></label>
                <label className="field-label">Powód decyzji<textarea value={reviewReason} onChange={(event) => setReviewReason(event.target.value)} /></label>
                <label className="field-label">Nowe GOL.GG ID — tylko dla zamiany<input value={replacementId} onChange={(event) => setReplacementId(event.target.value)} /></label>
                <div className="action-row">
                  <button className="primary-btn" disabled={reviewSaving || !reviewToken.trim() || reviewReason.trim().length < 8 || reviewOperator.trim().length < 2} onClick={() => handleReviewDecision('retain')} type="button">Zatwierdź obecne</button>
                  <button className="secondary-btn" disabled={reviewSaving || !reviewToken.trim() || !replacementId.trim() || reviewReason.trim().length < 8 || reviewOperator.trim().length < 2 || selectedReview.bet_count > 0} onClick={() => handleReviewDecision('replace')} type="button">Zamień link</button>
                  <button className="danger-btn" disabled={reviewSaving || !reviewToken.trim() || reviewReason.trim().length < 8 || reviewOperator.trim().length < 2 || selectedReview.bet_count > 0} onClick={() => handleReviewDecision('invalidate')} type="button">Unieważnij</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>


      <div className="mapping-toolbar">
        <div className="status-tabs" role="tablist" aria-label="Status meczów">
          {(Object.keys(STATUS_LABELS) as StatusFilter[]).map((status) => (
            <button
              key={status}
              className={statusFilter === status ? 'active' : ''}
              onClick={() => setStatusFilter(status)}
              type="button"
            >
              {STATUS_LABELS[status]}
            </button>
          ))}
        </div>
        <input
          className="mapping-search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Szukaj po drużynie, lidze, ID, bukmacherze…"
        />
        <button
          className={`team-attention-toggle ${onlyUnrecognizedTeams ? 'active' : ''}`}
          onClick={() => setOnlyUnrecognizedTeams((enabled) => !enabled)}
          type="button"
          aria-pressed={onlyUnrecognizedTeams}
        >
          Tylko nierozpoznane drużyny
        </button>
      </div>

      <div className="mapping-container">
        <section className="unmapped-list-section">
          <div className="section-header">
            <div>
              <h2>Niezmapowane mecze</h2>
              <span>
                {filteredMatches.length} / {unmappedMatches.length} dla statusu: {STATUS_LABELS[statusFilter]}
                {onlyUnrecognizedTeams ? ' · tylko z brakującą drużyną' : ''}
              </span>
            </div>
          </div>

          {loading ? (
            <div className="empty-state">Ładowanie…</div>
          ) : filteredMatches.length === 0 ? (
            <div className="empty-state">Brak meczów dla wybranego filtra.</div>
          ) : (
            <div className="scroll-list">
              {filteredMatches.map((m) => (
                <button
                  key={m.canonical_match_id}
                  className={`match-item ${selectedMatch?.canonical_match_id === m.canonical_match_id ? 'selected' : ''}`}
                  onClick={() => handleSelectMatch(m)}
                  type="button"
                >
                  <div className="match-row-top">
                    <span className="match-id">#{m.canonical_match_id}</span>
                    <span className="match-time">{formatDate(m.start_time_normalized)}</span>
                  </div>
                  <div className="match-teams">{m.team_a_name} vs {m.team_b_name}</div>
                  <div className="match-diagnosis">{matchDiagnosis(m)}</div>
                  <div className="team-map-badges">
                    <span className={`team-map-badge ${mappingBadge(m.team_a_mapping).className}`}>
                      A: {mappingBadge(m.team_a_mapping).label}
                    </span>
                    <span className={`team-map-badge ${mappingBadge(m.team_b_mapping).className}`}>
                      B: {mappingBadge(m.team_b_mapping).label}
                    </span>
                  </div>
                  <div className="match-league">{m.league || 'Nieznana liga'}</div>
                  {m.bookmakers && m.bookmakers.length > 0 && (
                    <div className="match-sources">Bukmacherzy: {m.bookmakers.join(', ')}</div>
                  )}
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="mapping-workspace">
          {!selectedMatch ? (
            <div className="empty-state">
              Wybierz mecz z listy. Po prawej dostaniesz szybkie mapowanie aliasów drużyn i opcjonalne mapowanie meczu do GOL.GG.
            </div>
          ) : (
            <>
              <div className="selected-match-info">
                <div>
                  <span className="match-id">#{selectedMatch.canonical_match_id}</span>
                  <h2>{selectedMatch.team_a_name} vs {selectedMatch.team_b_name}</h2>
                  <p>{formatDateTime(selectedMatch.start_time_normalized)} · {selectedMatch.league || 'Nieznana liga'}</p>
                  <p className="selected-diagnosis">{matchDiagnosis(selectedMatch)}</p>
                </div>
                <a href={`/matches/${selectedMatch.canonical_match_id}`} className="detail-link">
                  Szczegóły meczu →
                </a>
              </div>

              <div className="mapping-card priority-card">
                <div className="card-title-row">
                  <div>
                    <h3>1. Alias drużyny</h3>
                    <p>To jest główny workflow dla obecnych predykcji — mapuje nazwę bukmachera na nazwę GOL.GG.</p>
                  </div>
                </div>

                <div className="side-picker">
                  <button className={aliasSide === 'a' ? 'active' : ''} onClick={() => setAliasSide('a')} type="button">
                    <span>{selectedMatch.team_a_name}</span>
                    <small className={`team-map-badge ${mappingBadge(selectedMatch.team_a_mapping).className}`}>
                      {mappingBadge(selectedMatch.team_a_mapping).label}
                    </small>
                  </button>
                  <button className={aliasSide === 'b' ? 'active' : ''} onClick={() => setAliasSide('b')} type="button">
                    <span>{selectedMatch.team_b_name}</span>
                    <small className={`team-map-badge ${mappingBadge(selectedMatch.team_b_mapping).className}`}>
                      {mappingBadge(selectedMatch.team_b_mapping).label}
                    </small>
                  </button>
                </div>

                <label className="field-label">
                  Surowa nazwa
                  <input value={rawAliasName} readOnly />
                </label>

                <label className="field-label">
                  Szukaj drużyny GOL.GG
                  <input
                    value={aliasQuery}
                    onChange={(e) => {
                      setAliasQuery(e.target.value);
                      setSelectedGolggTeam('');
                    }}
                    placeholder="Wpisz nazwę z GOL.GG…"
                  />
                </label>

                <div className="alias-results">
                  {aliasResults.length === 0 ? (
                    <span>Brak wyników — zmień frazę albo wpisz dokładniejszą nazwę.</span>
                  ) : aliasResults.map((team) => (
                    <button
                      key={team}
                      className={selectedGolggTeam === team ? 'active' : ''}
                      onClick={() => setSelectedGolggTeam(team)}
                      type="button"
                    >
                      {team}
                    </button>
                  ))}
                </div>

                <div className="action-row">
                  <button
                    className="primary-btn"
                    onClick={handleSaveAlias}
                    disabled={!selectedGolggTeam || aliasSaving}
                    type="button"
                  >
                    {aliasSaving ? 'Zapisywanie…' : `Zapisz alias${selectedGolggTeam ? ` → ${selectedGolggTeam}` : ''}`}
                  </button>
                  <button className="danger-btn" onClick={handleBlockAlias} disabled={aliasSaving} type="button">
                    Oznacz jako brak mapowania
                  </button>
                  <button className="secondary-btn" onClick={handleUnblockAlias} disabled={aliasSaving} type="button">
                    Odblokuj
                  </button>
                </div>
              </div>

              <div className="mapping-card">
                <h3>2. Opcjonalnie: mapowanie meczu do GOL.GG</h3>
                <p>Używaj głównie dla wygasłych/zakończonych meczów, gdy chcesz przypiąć konkretny wynik GOL.GG.</p>

                <div className="manual-id-entry">
                  <div className="input-group">
                    <input
                      type="number"
                      placeholder="GOL.GG Match ID"
                      value={manualId}
                      onChange={(e) => {
                        setManualId(e.target.value);
                        setCheckResult(null);
                      }}
                    />
                    <button onClick={handleCheckId} disabled={!manualId || checking} type="button">
                      {checking ? 'Sprawdzanie…' : 'Sprawdź ID'}
                    </button>
                  </div>

                  {checkResult && (
                    <div className={`check-result ${checkResult.is_mapped ? 'warning' : 'success'}`}>
                      <div>
                        {checkResult.is_mapped ? (
                          <>
                            <strong>To ID jest już przypisane</strong><br />
                            {checkResult.team_a} vs {checkResult.team_b} ({formatDate(checkResult.start_time)})
                          </>
                        ) : (
                          <><strong>ID jest wolne.</strong> Możesz je przypisać do wybranego meczu.</>
                        )}
                      </div>
                      <button className="map-manual-btn" onClick={() => handleMapMatch(manualId)} disabled={mappingMatch} type="button">
                        Mapuj to ID
                      </button>
                    </div>
                  )}
                </div>

                <div className="candidates-list">
                  <h4>Sugerowani kandydaci ±3 dni</h4>
                  {candidateLoading ? (
                    <div className="empty-state compact">Ładowanie kandydatów…</div>
                  ) : candidates.length === 0 ? (
                    <div className="empty-state compact">Brak kandydatów w oknie czasowym.</div>
                  ) : (
                    candidates.map((c) => (
                      <div key={c.match_id} className="candidate-item">
                        <div className="candidate-info">
                          <div className="candidate-date">{formatDate(c.date)} · GOL.GG #{c.match_id}</div>
                          <div className="candidate-teams">{c.team1_name} vs {c.team2_name}</div>
                          <div className="candidate-result">
                            {c.team1_win === null && c.team2_win === null
                              ? 'Brak wyniku'
                              : c.team1_win
                                ? 'Wygrała drużyna 1'
                                : 'Wygrała drużyna 2'}
                          </div>
                        </div>
                        <button onClick={() => handleMapMatch(c.match_id)} disabled={mappingMatch} type="button">
                          Mapuj
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
