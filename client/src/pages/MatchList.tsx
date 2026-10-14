import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchMatches, updateMatchBestOf, fetchBookmakers, fetchParlayRecommendations } from '../api/client';
import type { MatchBoardItem, BookmakerStatus, ParlayRecommendationsResponse } from '../types';
import './MatchList.css';

export default function MatchList() {
  const [matches, setMatches] = useState<MatchBoardItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingBoMatchId, setEditingBoMatchId] = useState<number | null>(null);
  const [savingBo, setSavingBo] = useState(false);
  const [bookmakers, setBookmakers] = useState<BookmakerStatus[]>([]);
   const [selectedBookmaker, setSelectedBookmaker] = useState<string>('');
   const [onlyProps, setOnlyProps] = useState(false);
   const [parlays, setParlays] = useState<ParlayRecommendationsResponse | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const mountedGeneration = useRef(0);
  useEffect(() => {
    const generation = ++mountedGeneration.current;
    let cancelled = false;
    const controller = new AbortController();
    fetchBookmakers(controller.signal)
      .then(data => { if (!cancelled) setBookmakers(data); })
      .catch((err) => {
        if (!cancelled && err.name !== 'AbortError') {
          setMutationError(err instanceof Error ? err.message : 'Nie udało się pobrać bukmacherów.');
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
      mountedGeneration.current = generation + 1;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setMatches([]);
    fetchMatches(1, 14, selectedBookmaker || undefined, controller.signal)
      .then((data) => {
        if (!cancelled) setMatches(data.matches);
      })
      .catch((err) => {
        if (!cancelled && err.name !== 'AbortError') setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedBookmaker]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setParlays(null);
    fetchParlayRecommendations(selectedBookmaker || undefined, controller.signal)
      .then((data) => {
        if (!cancelled) setParlays(data);
      })
      .catch((err) => {
        if (!cancelled && err.name !== 'AbortError') {
          setParlays(null);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedBookmaker]);


  const formatDateTime = (iso: string | null) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString('pl-PL', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };


  return (
    <div className="match-list">
      <h1>Nadchodzące mecze</h1>
      <p className="subtitle">{matches.length} meczów</p>
      {mutationError && <p className="error" role="alert">{mutationError}</p>}

       <div className="filters-bar">
         <label className="filter-label">
           Bukmacher:
           <select
             value={selectedBookmaker}
             onChange={(e) => {
               setSelectedBookmaker(e.target.value);
               setLoading(true);
             }}
             className="filter-select"
           >
             <option value="">Wszyscy</option>
             {bookmakers.map((bk) => (
               <option key={bk.id} value={bk.name}>
                 {bk.name}
               </option>
             ))}
           </select>
         </label>
 
         <label className={`filter-checkbox-label ${onlyProps ? 'active' : ''}`}>
           <input
             type="checkbox"
             checked={onlyProps}
             onChange={(e) => setOnlyProps(e.target.checked)}
             style={{ cursor: 'pointer' }}
           />
           <span>🎯 Tylko z rynkami pobocznymi ({matches.filter((m) => m.has_props).length})</span>
         </label>
       </div>
      {loading && <div className="loading">Ładowanie meczów...</div>}
      {error && <div className="error" role="alert">Błąd: {error}</div>}
      {!loading && !error && matches.length === 0 && <div className="empty">Brak nadchodzących meczów</div>}

      {parlays?.top_parlay && (
        <section className="parlay-recommendation-card">
          <div className="parlay-header">
            <div className="parlay-title-wrap">
              <span className="parlay-icon">🛡️</span>
              <div>
                <h2 className="parlay-heading">Rekomendowany Dubel Dnia (Safe AKO)</h2>
                <p className="parlay-subheading">Optymalizacja podatku obrotowego 12% przez łączenie faworytów o wysokim prawdopodobieństwie</p>
              </div>
            </div>
            <div className="parlay-badges">
              <span className="parlay-bm-badge">{parlays.top_parlay.bookmaker}</span>
              <span className="parlay-confidence-badge">{parlays.top_parlay.confidence_badge}</span>
            </div>
          </div>

          <div className="parlay-metrics-grid">
            <div className="parlay-metric-item">
              <span className="parlay-metric-label">Kurs łączny</span>
              <span className="parlay-metric-val highlight-cyan">{parlays.top_parlay.combined_odds.toFixed(2)}</span>
              <span className="parlay-metric-sub">efekt. {parlays.top_parlay.effective_odds.toFixed(2)} po pod.</span>
            </div>
            <div className="parlay-metric-item">
              <span className="parlay-metric-label">P(Wygranej AKO)</span>
              <span className="parlay-metric-val highlight-gold">{(parlays.top_parlay.joint_prob * 100).toFixed(1)}%</span>
              <span className="parlay-metric-sub">łączny model</span>
            </div>
            <div className="parlay-metric-item">
              <span className="parlay-metric-label">Oczekiwana Wartość (EV)</span>
              <span className="parlay-metric-val highlight-green">+{(parlays.top_parlay.ev * 100).toFixed(1)}%</span>
              <span className="parlay-metric-sub">po odliczeniu 12%</span>
            </div>
            <div className="parlay-metric-item">
              <span className="parlay-metric-label">Amortyzacja Podatku</span>
              <span className="parlay-metric-val highlight-green">+{(parlays.top_parlay.tax_amortization_gain * 100).toFixed(1)} p.p.</span>
              <span className="parlay-metric-sub">względem 2 singli</span>
            </div>
            <div className="parlay-metric-item">
              <span className="parlay-metric-label">Sugerowana Stawka</span>
              <span className="parlay-metric-val">{parlays.top_parlay.suggested_stake} PLN</span>
              <span className="parlay-metric-sub">{(parlays.top_parlay.quarter_kelly * 100).toFixed(1)}% Quarter-Kelly</span>
            </div>
          </div>

          <div className="parlay-legs-wrapper">
            <Link to={`/matches/${parlays.top_parlay.legs[0].canonical_match_id}`} className="parlay-leg-card">
              <div className="parlay-leg-top">
                <span className="parlay-leg-league">{parlays.top_parlay.legs[0].league || 'Mecz 1'}</span>
                <span className="parlay-leg-time">{formatDateTime(parlays.top_parlay.legs[0].start_time || null)}</span>
              </div>
              <div className="parlay-leg-pick">
                <span className="parlay-leg-team">{parlays.top_parlay.legs[0].team_name}</span>
                <span className="parlay-leg-odds">@ {parlays.top_parlay.legs[0].odds.toFixed(2)}</span>
              </div>
              <div className="parlay-leg-opp">vs {parlays.top_parlay.legs[0].opponent_name}</div>
              <div className="parlay-leg-stats">
                <span className="parlay-leg-stat">Model: <strong>{(parlays.top_parlay.legs[0].model_prob * 100).toFixed(0)}%</strong></span>
                <span className="parlay-leg-stat">Single EV: <strong>{(parlays.top_parlay.legs[0].single_ev * 100).toFixed(1)}%</strong></span>
              </div>
            </Link>

            <div className="parlay-multiply-sign" title="Kupon AKO: iloczyn kursów faworytów">×</div>

            <Link to={`/matches/${parlays.top_parlay.legs[1].canonical_match_id}`} className="parlay-leg-card">
              <div className="parlay-leg-top">
                <span className="parlay-leg-league">{parlays.top_parlay.legs[1].league || 'Mecz 2'}</span>
                <span className="parlay-leg-time">{formatDateTime(parlays.top_parlay.legs[1].start_time || null)}</span>
              </div>
              <div className="parlay-leg-pick">
                <span className="parlay-leg-team">{parlays.top_parlay.legs[1].team_name}</span>
                <span className="parlay-leg-odds">@ {parlays.top_parlay.legs[1].odds.toFixed(2)}</span>
              </div>
              <div className="parlay-leg-opp">vs {parlays.top_parlay.legs[1].opponent_name}</div>
              <div className="parlay-leg-stats">
                <span className="parlay-leg-stat">Model: <strong>{(parlays.top_parlay.legs[1].model_prob * 100).toFixed(0)}%</strong></span>
                <span className="parlay-leg-stat">Single EV: <strong>{(parlays.top_parlay.legs[1].single_ev * 100).toFixed(1)}%</strong></span>
              </div>
            </Link>
          </div>

          <div className="parlay-rationale-box">
            <strong>💡 Przewaga podatkowa:</strong> W polskim prawie bukmacherskim 12% podatku potrącane jest jednorazowo z wygranej kuponu.
            Skumulowany kurs brutto <strong>{parlays.top_parlay.combined_odds.toFixed(2)}</strong> obniża relatywny narzut podatku do <strong>{((0.12 / parlays.top_parlay.combined_odds) * 100).toFixed(1)}%</strong> (vs <strong>{((0.12 / parlays.top_parlay.legs[0].odds) * 100).toFixed(1)}%</strong> w singlu),
            przekształcając zakłady na faworytów w kupon o wysokiej wartości oczekiwanej (+{(parlays.top_parlay.ev * 100).toFixed(1)}% EV).
          </div>
        </section>
      )}

       <div className="matches-grid">
         {matches
           .filter((m) => !onlyProps || m.has_props)
           .map((m) => {
          const hasRecommendation = Boolean(
            m.recommended_side && m.recommended_ev && m.recommended_ev > 0
          );
          const unmappedTeams = [
            !m.team_a_golgg_name ? m.team_a_name : null,
            !m.team_b_golgg_name ? m.team_b_name : null,
          ].filter(Boolean).join(', ');

          const cardClass = `match-card${hasRecommendation ? ' ev-highlight' : ''}${m.has_unmapped_teams ? ' mapping-warning' : ''}`;

          return (
            <Link
              key={m.canonical_match_id}
              to={`/matches/${m.canonical_match_id}`}
              className={cardClass}
            >
              <div className="match-header">
                <span className="league">{m.league || 'Nieznana liga'}</span>
                {m.match_confidence !== null && m.match_confidence < 1.0 && (
                  <span className="confidence-badge" title={`Pewność mapowania: ${(m.match_confidence * 100).toFixed(0)}%`}>
                    🎯 {(m.match_confidence * 100).toFixed(0)}%
                  </span>
                )}
                {m.has_unmapped_teams && (
                  <span className="mapping-warning-badge" title={`Brak mapowania: ${unmappedTeams}. Kliknij mecz i dodaj alias.`}>
                    ⚠️ Dodaj mapowanie
                  </span>
                )}
                {m.has_props && (
                  <span
                    className="props-badge"
                    title={`Dostępne rynki poboczne (Props): ${m.prop_lines_count || 11} linii kursowych (zabójstwa, handicapy)`}
                  >
                    🎯 Props ({m.prop_lines_count || 11})
                  </span>
                )}
                {editingBoMatchId === m.canonical_match_id ? (
                  <span className="best-of-picker" onClick={(e) => e.preventDefault()}>
                    {[1, 3, 5, 7].map((bo) => (
                      <button
                        key={bo}
                        className={`bo-pill${(m.best_of || 1) === bo ? ' active' : ''}${savingBo ? ' saving' : ''}`}
                        onClick={async (e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (savingBo || (m.best_of || 1) === bo) {
                            setEditingBoMatchId(null);
                            return;
                          }
                          const generation = mountedGeneration.current;
                          setSavingBo(true);
                          setMutationError(null);
                          try {
                            const result = await updateMatchBestOf(m.canonical_match_id, bo);
                            if (mountedGeneration.current !== generation) return;
                            setMatches(prev => prev.map(mt =>
                              mt.canonical_match_id === m.canonical_match_id
                                ? { ...mt, best_of: result.best_of }
                                : mt
                            ));
                          } catch (err: unknown) {
                            if (mountedGeneration.current === generation) {
                              setMutationError(err instanceof Error ? err.message : 'Nie udało się zmienić BO.');
                            }
                          } finally {
                            if (mountedGeneration.current === generation) {
                              setSavingBo(false);
                              setEditingBoMatchId(null);
                            }
                          }
                        }}
                      >
                        Bo{bo}
                      </button>
                    ))}
                  </span>
                ) : (
                  <span
                    className={`best-of-badge bo${m.best_of || 1}`}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setEditingBoMatchId(m.canonical_match_id);
                    }}
                    title="Kliknij aby zmienić"
                  >
                    Bo{m.best_of || 1}
                  </span>
                )}
                <span className="datetime">{formatDateTime(m.start_time_normalized)}</span>
              </div>

              <div className="match-teams">
                <div className="team">
                  <span className="team-name">
                    {m.team_a_name || '?'}
                    {!m.team_a_golgg_name && (
                      <span className="team-unmapped" title="Brak mapowania do GOL.GG — kliknij mecz i dodaj alias">?</span>
                    )}
                  </span>
                  <div className="odds-block">
                    {m.best_odds_a && (
                      <span className="odds">{m.best_odds_a.toFixed(2)}</span>
                    )}
                    {m.best_odds_a == null && m.diagnostic_best_odds_a != null && (
                      <span className="odds" title="Cena diagnostyczna — brak świeżej oferty do rekomendacji">
                        {m.diagnostic_best_odds_a.toFixed(2)} <small>diagnostyczny</small>
                      </span>
                    )}
                  </div>
                </div>
                <span className="vs">vs</span>
                <div className="team">
                  <span className="team-name">
                    {m.team_b_name || '?'}
                    {!m.team_b_golgg_name && (
                      <span className="team-unmapped" title="Brak mapowania do GOL.GG — kliknij mecz i dodaj alias">?</span>
                    )}
                  </span>
                  <div className="odds-block">
                    {m.best_odds_b && (
                      <span className="odds">{m.best_odds_b.toFixed(2)}</span>
                    )}
                    {m.best_odds_b == null && m.diagnostic_best_odds_b != null && (
                      <span className="odds" title="Cena diagnostyczna — brak świeżej oferty do rekomendacji">
                        {m.diagnostic_best_odds_b.toFixed(2)} <small>diagnostyczny</small>
                      </span>
                    )}
                  </div>
                </div>
              </div>
              {m.recommended_side && m.recommended_ev && m.recommended_ev > 0 ? (
                <div className="match-recommendation-bar">
                  <span className="rec-badge">🎯 Typ</span>
                  <span className="rec-team">{m.recommended_team}</span>
                  {m.recommended_odds && <span className="rec-odds">@{m.recommended_odds.toFixed(2)}</span>}
                  {m.recommended_bookmaker && <span className="rec-bm">w {m.recommended_bookmaker}</span>}
                  <span className="rec-ev">EV +{(m.recommended_ev * 100).toFixed(1)}%</span>
                </div>
              ) : null}

              <div className="match-footer">
                <span className="bookmakers">
                  {m.bookmaker_count} bukmacher{m.bookmaker_count !== 1 ? 'ów' : ''}
                </span>
                {m.best_bookmaker_a && m.best_bookmaker_b && m.best_bookmaker_a === m.best_bookmaker_b ? (
                  <span className="best-bookmaker" title="Najlepsze kursy na obie strony">
                    {m.best_bookmaker_a}
                  </span>
                ) : (
                  <>
                    {m.best_bookmaker_a && (
                      <span className="best-bookmaker" title="Najlepszy kurs na {m.team_a_name}">
                        {m.best_bookmaker_a}
                      </span>
                    )}
                    {m.best_bookmaker_b && (
                      <span className="best-bookmaker" title="Najlepszy kurs na {m.team_b_name}">
                        {m.best_bookmaker_b}
                      </span>
                    )}
                  </>
                )}
                {m.arb_after_tax && (
                  <span className="arb-badge">ARB</span>
                )}
                {m.has_unmapped_teams && (
                  <span className="mapping-footer-note">brak predykcji do czasu mapowania</span>
                )}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
