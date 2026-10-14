import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  fetchMatchPropAnalysis,
  triggerMatchPropPrediction,
  fetchPropOddsTimeline,
} from '../api/client';
import type {
  MatchPropAnalysisResponse,
  PropEvaluatedLineItem,
  PropOddsSnapshotItem,
  PropDistributionLine,
  PropSignalItem,
} from '../types';
import './MatchPropsAnalysis.css';

interface MatchPropsAnalysisProps {
  canonicalMatchId: number;
  teamAName: string;
  teamBName: string;
  league?: string | null;
}

export default function MatchPropsAnalysis({
  canonicalMatchId,
  teamAName,
  teamBName,
  league,
}: MatchPropsAnalysisProps) {
  const [mapNumber, setMapNumber] = useState<number>(1);
  const [marketFilter, setMarketFilter] = useState<string>('all');
  const [data, setData] = useState<MatchPropAnalysisResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Line movement timeline modal state
  const [selectedTimelineMarket, setSelectedTimelineMarket] = useState<string | null>(null);
  const [selectedTimelineLine, setSelectedTimelineLine] = useState<number | null>(null);
  const [timelineData, setTimelineData] = useState<PropOddsSnapshotItem[]>([]);
  const [timelineLoading, setTimelineLoading] = useState<boolean>(false);

  const loadAnalysis = useCallback(async (mapNum: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchMatchPropAnalysis(canonicalMatchId, { mapNumber: mapNum });
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd ładowania analizy rynków prop');
    } finally {
      setLoading(false);
    }
  }, [canonicalMatchId]);

  useEffect(() => {
    loadAnalysis(mapNumber);
  }, [loadAnalysis, mapNumber]);

  const handlePredictAndSave = async () => {
    setPredicting(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const res = await triggerMatchPropPrediction(canonicalMatchId, mapNumber, 0.12);
      setData(res);
      const nowStr = new Date().toLocaleTimeString('pl-PL');
      setSuccessMessage(
        `Prognoza statystyczna wygenerowana i zapisana w bazie (ID: #${res.saved_prediction_id ?? 'OK'}, ${nowStr})`
      );
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd przeliczania modelu');
    } finally {
      setPredicting(false);
    }
  };

  const openTimeline = async (marketType: string, line: number) => {
    setSelectedTimelineMarket(marketType);
    setSelectedTimelineLine(line);
    setTimelineLoading(true);
    try {
      const res = await fetchPropOddsTimeline(canonicalMatchId, {
        marketType,
        line,
        mapNumber,
      });
      setTimelineData(res.timeline);
    } catch (err) {
      console.error('Failed to load timeline:', err);
    } finally {
      setTimelineLoading(false);
    }
  };

  const closeTimeline = () => {
    setSelectedTimelineMarket(null);
    setSelectedTimelineLine(null);
    setTimelineData([]);
  };

  // Filter evaluated lines
  const filteredLines = useMemo(() => {
    if (!data?.evaluated_lines) return [];
    if (marketFilter === 'all') return data.evaluated_lines;
    return data.evaluated_lines.filter((l) => l.market_type === marketFilter);
  }, [data?.evaluated_lines, marketFilter]);

  const expectations = data?.model_expectations;
  const distributionLines: PropDistributionLine[] = expectations?.distribution?.total_lines || [];
  const quantiles = expectations?.distribution?.quantiles;

  return (
    <div className="props-analysis-container" id="in-game-props">
      {/* Header bar with title, map switcher, and recalculate action */}
      <div className="props-header-bar">
        <div className="props-title-group">
          <div className="props-badge">IDEA-018: In-Game Modeling {league ? `• ${league}` : ''}</div>
          <h2 className="props-main-title">
            Analiza Rynków Pobocznych & Model Pace / Zabójstw
          </h2>
          <p className="props-subtitle">
            Statystyczny model Poisson/NegBinomial dla sumy zabójstw, handicapów i tempa meczu w oparciu o GOL.GG
          </p>
        </div>

        <div className="props-controls-group">
          <div className="map-selector">
            <span className="control-label">Mapa:</span>
            {[1, 2, 3].map((num) => (
              <button
                key={num}
                type="button"
                className={`map-btn ${mapNumber === num ? 'active' : ''}`}
                onClick={() => setMapNumber(num)}
              >
                Mapa {num}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="predict-save-btn"
            onClick={handlePredictAndSave}
            disabled={predicting || loading}
          >
            {predicting ? (
              <span className="spinner-inline">Przeliczanie...</span>
            ) : (
              '⚡ Przelicz & Zapisz w Bazie'
            )}
          </button>
        </div>
      </div>

      {successMessage && (
        <div className="props-success-banner">
          <span className="success-icon">✓</span> {successMessage}
        </div>
      )}

      {error && (
        <div className="props-error-banner">
          <span className="error-icon">⚠️</span> {error}
        </div>
      )}

      {loading && !data && (
        <div className="props-loading-state">
          <div className="props-spinner" />
          <p>Trwa pobieranie linii bukmacherskich i szacowanie modeli statystycznych...</p>
        </div>
      )}

      {data && (
        <>
          {/* Section 1: Model Voice & Expectations Summary Cards */}
          <div className="model-voice-panel">
            <div className="model-voice-header">
              <span className="model-icon">🧠</span>
              <span className="model-tag">Głos Modelu Statystycznego</span>
              {data.saved_prediction_id ? (
                <span className="saved-badge">Zapisano w bazie #{data.saved_prediction_id}</span>
              ) : (
                <span className="saved-badge unsaved">Tymczasowy szacunek</span>
              )}
              {expectations?.league_pace_category && (
                <span className={`pace-pill pace-${expectations.league_pace_category}`}>
                  Tempo ligi: {expectations.league_pace_category}
                </span>
              )}
            </div>

            <div className="model-metrics-grid">
              <div className="metric-box total-kills-box">
                <div className="metric-label">Oczekiwana Suma Zabójstw</div>
                <div className="metric-value">
                  {expectations?.expected_total_kills != null
                    ? expectations.expected_total_kills.toFixed(1)
                    : '-'}
                </div>
                <div className="metric-subtext">
                  Średnia ligowa: {expectations?.league_avg_kills?.toFixed(1) ?? '27.4'} kills
                </div>
              </div>

              <div className="metric-box team-split-box">
                <div className="metric-label">Rozbicie Drużynowe (Expected μ)</div>
                <div className="team-split-row">
                  <div className="team-split-col">
                    <span className="team-split-name">{teamAName}</span>
                    <span className="team-split-score">
                      {expectations?.mu_team_a?.toFixed(1) ?? '-'}
                    </span>
                  </div>
                  <div className="team-split-vs">vs</div>
                  <div className="team-split-col">
                    <span className="team-split-name">{teamBName}</span>
                    <span className="team-split-score">
                      {expectations?.mu_team_b?.toFixed(1) ?? '-'}
                    </span>
                  </div>
                </div>
                <div className="metric-subtext">
                  Oczekiwany handicap: {expectations?.expected_spread_a_minus_b != null
                    ? `${expectations.expected_spread_a_minus_b > 0 ? '+' : ''}${expectations.expected_spread_a_minus_b.toFixed(1)} kills`
                    : '-'}
                </div>
              </div>

              <div className="metric-box quantiles-box">
                <div className="metric-label">Kwantyle Rozkładu (Pace Range)</div>
                <div className="quantiles-chips">
                  <div className="quantile-chip">
                    <span className="q-label">P10</span>
                    <span className="q-val">{quantiles?.p10 ?? '-'}</span>
                  </div>
                  <div className="quantile-chip">
                    <span className="q-label">P25</span>
                    <span className="q-val">{quantiles?.p25 ?? '-'}</span>
                  </div>
                  <div className="quantile-chip highlight">
                    <span className="q-label">Mediana</span>
                    <span className="q-val">{quantiles?.p50 ?? '-'}</span>
                  </div>
                  <div className="quantile-chip">
                    <span className="q-label">P75</span>
                    <span className="q-val">{quantiles?.p75 ?? '-'}</span>
                  </div>
                  <div className="quantile-chip">
                    <span className="q-label">P90</span>
                    <span className="q-val">{quantiles?.p90 ?? '-'}</span>
                  </div>
                </div>
                <div className="metric-subtext">Zakres 80% gier (P10 - P90)</div>
              </div>
            </div>
          </div>

          {/* Section 2: Value Signals Banner if Net EV > 0 */}
          {data.signals && data.signals.length > 0 && (
            <div className="value-signals-section">
              <div className="section-title-with-badge">
                <h3 className="section-title">⚡ Wykryte Sygnały Value (+EV po 12% podatku)</h3>
                <span className="signals-count">{data.signals.length} okazji</span>
              </div>
              <div className="signals-grid">
                {data.signals.map((sig: PropSignalItem, idx: number) => (
                  <div key={idx} className="signal-card">
                    <div className="signal-card-top">
                      <span className={`bookmaker-tag bookmaker-${sig.bookmaker.toLowerCase()}`}>
                        {sig.bookmaker.toUpperCase()}
                      </span>
                      <span className="signal-market">{sig.market_type}</span>
                      <span className="net-ev-pill">Net EV: +{sig.net_ev_tax12.toFixed(1)}%</span>
                    </div>
                    <div className="signal-selection">{sig.selection}</div>
                    <div className="signal-details">
                      <div className="detail-item">
                        <span className="detail-label">Kurs bukmachera:</span>
                        <span className="detail-val highlight">{sig.odds.toFixed(2)}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Fair Odds (Model):</span>
                        <span className="detail-val">{sig.fair_odds.toFixed(2)}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Prawdopodobieństwo:</span>
                        <span className="detail-val">{(sig.model_prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Section 3: Distribution Curve & Fair Odds Table */}
          {distributionLines.length > 0 && (
            <div className="distribution-table-section">
              <h3 className="section-title">
                📊 Model Rozkładu Prawdopodobieństwa Zabójstw (Fair Odds)
              </h3>
              <p className="section-desc">
                Czyste prawdopodobieństwo wyliczone przez model dla poszczególnych linii zabójstw przed nałożeniem marży bukmachera
              </p>

              <div className="table-responsive">
                <table className="props-table">
                  <thead>
                    <tr>
                      <th>Linia Sumy Zabójstw</th>
                      <th>Prawdopodobieństwo OVER</th>
                      <th>Wizualizacja Rozkładu</th>
                      <th>Prawdopodobieństwo UNDER</th>
                      <th>Fair Odds OVER</th>
                      <th>Fair Odds UNDER</th>
                    </tr>
                  </thead>
                  <tbody>
                    {distributionLines.map((row) => {
                      const overPct = Math.round(row.prob_over * 100);
                      const underPct = 100 - overPct;
                      const isMedianLine = row.line === 26.5 || row.line === 28.5;

                      return (
                        <tr key={row.line} className={isMedianLine ? 'median-row' : ''}>
                          <td className="line-cell">
                            <strong>{row.line}</strong>
                            {isMedianLine && <span className="median-badge">Główna linia</span>}
                          </td>
                          <td className="prob-cell prob-over">
                            {(row.prob_over * 100).toFixed(1)}%
                          </td>
                          <td className="bar-cell">
                            <div className="prob-ratio-bar">
                              <div
                                className="bar-over"
                                style={{ width: `${overPct}%` }}
                                title={`Over: ${overPct}%`}
                              />
                              <div
                                className="bar-under"
                                style={{ width: `${underPct}%` }}
                                title={`Under: ${underPct}%`}
                              />
                            </div>
                          </td>
                          <td className="prob-cell prob-under">
                            {(row.prob_under * 100).toFixed(1)}%
                          </td>
                          <td className="fair-cell">
                            {row.fair_odds_over != null ? row.fair_odds_over.toFixed(2) : '-'}
                          </td>
                          <td className="fair-cell">
                            {row.fair_odds_under != null ? row.fair_odds_under.toFixed(2) : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Section 4: Cross-Bookmaker Board & Model Edge Evaluation */}
          <div className="market-board-section">
            <div className="board-header">
              <div>
                <h3 className="section-title">
                  🏦 Tablica Kursów Bukmacherów vs Model (Edge & Value)
                </h3>
                <p className="section-desc">
                  Zestawienie aktualnych ofert z eFortuna, STS i Betclic z kalkulacją marży i przewagi modelu (Net EV z podatkiem 12%)
                </p>
              </div>

              <div className="market-filter-tabs">
                {[
                  { id: 'all', label: 'Wszystkie' },
                  { id: 'total_kills', label: 'Suma Zabójstw' },
                  { id: 'team_kills', label: 'Zabójstwa Drużyn' },
                  { id: 'handicap_kills', label: 'Handicap Zabójstw' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={`filter-tab ${marketFilter === tab.id ? 'active' : ''}`}
                    onClick={() => setMarketFilter(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {filteredLines.length === 0 ? (
              <div className="empty-market-box">
                Brak zarejestrowanych linii bukmacherskich dla wybranego filtru na mapie {mapNumber}.
              </div>
            ) : (
              <div className="table-responsive">
                <table className="props-table board-table">
                  <thead>
                    <tr>
                      <th>Bukmacher</th>
                      <th>Rynek</th>
                      <th>Linia / Drużyna</th>
                      <th>Kurs OVER / 1</th>
                      <th>Kurs UNDER / 2</th>
                      <th>Marża Rynku</th>
                      <th>Model OVER (EV)</th>
                      <th>Model UNDER (EV)</th>
                      <th>Akcje</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLines.map((lineItem: PropEvaluatedLineItem) => {
                      const hasEvOver = lineItem.ev_over_net != null && lineItem.ev_over_net > 0;
                      const hasEvUnder = lineItem.ev_under_net != null && lineItem.ev_under_net > 0;

                      return (
                        <tr
                          key={lineItem.id}
                          className={hasEvOver || hasEvUnder ? 'value-row' : ''}
                        >
                          <td className="bookmaker-cell">
                            <span
                              className={`bookmaker-tag bookmaker-${lineItem.bookmaker.toLowerCase()}`}
                            >
                              {lineItem.bookmaker}
                            </span>
                          </td>
                          <td className="market-name-cell">
                            {lineItem.market_type === 'total_kills' && 'Suma zabójstw'}
                            {lineItem.market_type === 'team_kills' && 'Zabójstwa drużyny'}
                            {lineItem.market_type === 'handicap_kills' && 'Handicap zabójstw'}
                            {lineItem.market_type === 'duration' && 'Czas gry'}
                          </td>
                          <td className="line-value-cell">
                            <span className="line-pill">
                              {lineItem.target_team ? `${lineItem.target_team} ` : ''}
                              {lineItem.line > 0 && lineItem.market_type === 'handicap_kills' ? `+${lineItem.line}` : lineItem.line}
                            </span>
                          </td>
                          <td className="odds-cell">
                            {lineItem.odds_over != null ? (
                              <div className="odds-stack">
                                <span className="odds-num">{lineItem.odds_over.toFixed(2)}</span>
                                {lineItem.prob_over_novig != null && (
                                  <span className="no-vig-hint">
                                    nv: {(lineItem.prob_over_novig * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                            ) : (
                              '-'
                            )}
                          </td>
                          <td className="odds-cell">
                            {lineItem.odds_under != null ? (
                              <div className="odds-stack">
                                <span className="odds-num">{lineItem.odds_under.toFixed(2)}</span>
                                {lineItem.prob_under_novig != null && (
                                  <span className="no-vig-hint">
                                    nv: {(lineItem.prob_under_novig * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                            ) : (
                              '-'
                            )}
                          </td>
                          <td className="margin-cell">
                            {lineItem.margin != null
                              ? `${(lineItem.margin * 100).toFixed(1)}%`
                              : '-'}
                          </td>
                          <td className="model-eval-cell">
                            {lineItem.model_prob_over != null ? (
                              <div className="eval-stack">
                                <span className="model-prob">
                                  {(lineItem.model_prob_over * 100).toFixed(1)}%
                                </span>
                                {lineItem.ev_over_net != null && (
                                  <span
                                    className={`ev-pill ${hasEvOver ? 'ev-positive' : 'ev-negative'}`}
                                  >
                                    {lineItem.ev_over_net > 0 ? '+' : ''}
                                    {(lineItem.ev_over_net * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="muted">-</span>
                            )}
                          </td>
                          <td className="model-eval-cell">
                            {lineItem.model_prob_under != null ? (
                              <div className="eval-stack">
                                <span className="model-prob">
                                  {(lineItem.model_prob_under * 100).toFixed(1)}%
                                </span>
                                {lineItem.ev_under_net != null && (
                                  <span
                                    className={`ev-pill ${hasEvUnder ? 'ev-positive' : 'ev-negative'}`}
                                  >
                                    {lineItem.ev_under_net > 0 ? '+' : ''}
                                    {(lineItem.ev_under_net * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="muted">-</span>
                            )}
                          </td>
                          <td className="actions-cell">
                            <button
                              type="button"
                              className="timeline-trigger-btn"
                              title="Pokaż historię kursów dla tej linii"
                              onClick={() => openTimeline(lineItem.market_type, lineItem.line)}
                            >
                              📈 Ruch linii
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Section 5: Line Movement Timeline Modal */}
          {selectedTimelineMarket != null && selectedTimelineLine != null && (
            <div className="timeline-modal-backdrop" onClick={closeTimeline}>
              <div
                className="timeline-modal-content"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="timeline-modal-header">
                  <div>
                    <h3 className="timeline-title">
                      📈 Historia Ruchu Kursów: {selectedTimelineMarket} ({selectedTimelineLine})
                    </h3>
                    <p className="timeline-sub">
                      Zarejestrowane zmiany kursów u polskich bukmacherów w czasie
                    </p>
                  </div>
                  <button
                    type="button"
                    className="modal-close-btn"
                    onClick={closeTimeline}
                  >
                    ✕
                  </button>
                </div>

                {timelineLoading ? (
                  <div className="timeline-loading">Ładowanie historii kursów...</div>
                ) : timelineData.length === 0 ? (
                  <div className="timeline-empty">Brak zarejestrowanych zmian dla tej linii.</div>
                ) : (
                  <div className="timeline-table-wrapper">
                    <table className="props-table timeline-table">
                      <thead>
                        <tr>
                          <th>Czas zapisu (UTC)</th>
                          <th>Bukmacher</th>
                          <th>Kurs OVER</th>
                          <th>Kurs UNDER</th>
                          <th>Marża</th>
                          <th>No-Vig OVER</th>
                          <th>No-Vig UNDER</th>
                        </tr>
                      </thead>
                      <tbody>
                        {timelineData.map((pt) => (
                          <tr key={pt.id}>
                            <td className="time-cell">
                              {pt.scraped_at
                                ? new Date(pt.scraped_at).toLocaleString('pl-PL', {
                                    day: '2-digit',
                                    month: '2-digit',
                                    hour: '2-digit',
                                    minute: '2-digit',
                                    second: '2-digit',
                                  })
                                : '-'}
                            </td>
                            <td>
                              <span
                                className={`bookmaker-tag bookmaker-${pt.bookmaker.toLowerCase()}`}
                              >
                                {pt.bookmaker}
                              </span>
                            </td>
                            <td className="odds-num">
                              {pt.odds_over != null ? pt.odds_over.toFixed(2) : '-'}
                            </td>
                            <td className="odds-num">
                              {pt.odds_under != null ? pt.odds_under.toFixed(2) : '-'}
                            </td>
                            <td>
                              {pt.margin != null
                                ? `${(pt.margin * 100).toFixed(1)}%`
                                : '-'}
                            </td>
                            <td>
                              {pt.prob_over_novig != null
                                ? `${(pt.prob_over_novig * 100).toFixed(1)}%`
                                : '-'}
                            </td>
                            <td>
                              {pt.prob_under_novig != null
                                ? `${(pt.prob_under_novig * 100).toFixed(1)}%`
                                : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
