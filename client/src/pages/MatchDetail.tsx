import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchMatchDetail, fetchPredictionHistory, updateMatchBestOf, updateMatchRoster, resetMatchRoster, predictMatch, createTeamAlias, deleteTeamAlias, unblockTeamAlias, searchGolggTeams, searchRosterPlayers, simulateFearlessMatch } from '../api/client';
import type { MatchDetailResponse, PredictionHistoryPoint, RosterOverridePlayerInput, RosterPlayerCandidate, FearlessSimulationResponse } from '../types';
import MatchPropsAnalysis from '../components/MatchPropsAnalysis';
import './MatchDetail.css';

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function diagnosticObject(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function provenanceValue(value: unknown): string {
  if (value == null) return 'nieznane';
  return typeof value === 'string' ? value : JSON.stringify(value) ?? 'nieznane';
}

function PredictionProvenance({ diagnostics, predictedAt }: {
  diagnostics: Record<string, unknown> | null;
  predictedAt: string | null;
}) {
  const source = diagnosticObject(diagnostics?.feature_provenance);
  const freshness = diagnosticObject(source.dataset_freshness);
  const request = diagnosticObject(diagnostics?.c0_request);
  const roster = diagnosticObject(diagnostics?.roster_provenance);
  return (
    <div className="prediction-provenance">
      <p>Znacznik zapisania predykcji: {provenanceValue(predictedAt)}. Nie oznacza odświeżonej historii ani certyfikacji forward/live.</p>
      {!diagnostics && <p>Brak zapisanych historycznych diagnostics — pochodzenie wejść nieznane.</p>}
      <dl>
        <dt>Źródło historii (deklarowane)</dt><dd>{provenanceValue(source.source_kind ?? source.status)}</dd>
        <dt>Ścieżka historii</dt><dd>{provenanceValue(source.raw_history_path)}</dd>
        <dt>SHA256 historii</dt><dd><code>{provenanceValue(source.raw_history_sha256)}</code></dd>
        <dt>Ostatni dzień gry w źródle</dt><dd>{provenanceValue(freshness.raw_history_max_source_game_day)}</dd>
        <dt>Ostatni dzień wydania historii</dt><dd>{provenanceValue(freshness.raw_history_max_effective_release_day)}</dd>
        <dt>Historia faktycznie użyta do cech</dt><dd>{provenanceValue(diagnostics?.feature_source_max_day ?? source.feature_source_max_day)}</dd>
        <dt>Zegar decyzji żądania C0</dt><dd>{provenanceValue(request.decision_at)}</dd>
        <dt>Zegar decyzji modelu</dt><dd>{provenanceValue(diagnostics?.requested_decision_at)}</dd>
        <dt>Wyłączny cutoff historii (UTC)</dt><dd>{provenanceValue(diagnostics?.data_cutoff_at)}</dd>
        <dt>Zegar rosteru / start</dt><dd>{provenanceValue(roster.decision_at)} / {provenanceValue(roster.start_at)}</dd>
        <dt>Prawdopodobieństwo faktycznie użyte do EV</dt><dd>{provenanceValue(diagnostics?.prob_for_ev)}</dd>
        <dt>Podstawa / powód niepewności</dt><dd>{provenanceValue(diagnostics?.uncertainty_basis)} / {provenanceValue(diagnostics?.uncertainty_reason)}</dd>
      </dl>
      {(['a', 'b'] as const).map(side => {
        const team = diagnosticObject(roster[`team_${side}`]);
        return (
          <div key={side}>
            <h4>Roster {side.toUpperCase()} — deklarowana podstawa: {provenanceValue(team.basis)}</h4>
            <p>Powiązanie rosteru: {provenanceValue(team.binding)}</p>
            <p>ID drużyny: {provenanceValue(team.team_id)} · ID zawodników (źródłowa kolejność): {provenanceValue(team.player_ids)}</p>
            <p>Kolejność tensorów: {provenanceValue(team.tensor_player_order)} · ID w żądaniu C0: {provenanceValue(request[`roster_${side}`])}</p>
            <p>Źródło: {provenanceValue(team.source)} · mecz/gry: {provenanceValue(team.source_match_id)} / {provenanceValue(team.source_game_id)}</p>
            <p>Data meczu źródłowego: {provenanceValue(team.source_match_date)} · aktualizacja: {provenanceValue(team.updated_at)}</p>
            <p>Dokładne fakty źródłowe zawodników: {provenanceValue(team.source_players)}</p>
            <p>Publikacja: {provenanceValue(team.published_at)} · obserwacja: {provenanceValue(team.observed_at)} · uwierzytelnienie: {provenanceValue(team.authenticated)}</p>
          </div>
        );
      })}
      <details><summary>Dokładne zapisane raw_diagnostics</summary><pre>{diagnostics ? JSON.stringify(diagnostics, null, 2) : 'nieznane'}</pre></details>
    </div>
  );
}

// ─── Prediction History Chart Component ──────────────────────

interface ChartProps {
  data: PredictionHistoryPoint[];
  teamA: string;
  teamB: string;
}

function PredictionHistoryChart({ data, teamA, teamB }: ChartProps) {
  // Separate data by model
  const models = useMemo(() => {
    const map = new Map<string, PredictionHistoryPoint[]>();
    for (const pt of data) {
      const key = pt.model_name;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(pt);
    }
    // Sort each model's points by timestamp
    for (const pts of map.values()) {
      pts.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    }
    return map;
  }, [data]);

  const modelNames = Array.from(models.keys());

  // Build unique time axis from all data points, bucketed by minute
  const timeLabels = useMemo(() => {
    const tsSet = new Set(data.map(d => d.timestamp));
    return Array.from(tsSet).sort();
  }, [data]);

  // Compute dynamic Y range from actual data
  const { yMin, yMax, yTicks } = useMemo(() => {
    const allProbs: number[] = [];
    for (const pt of data) {
      if (pt.prob_a != null) allProbs.push(pt.prob_a);
      if (pt.prob_b != null) allProbs.push(pt.prob_b);
      if (pt.market_prob_a != null) allProbs.push(pt.market_prob_a);
      if (pt.market_prob_b != null) allProbs.push(pt.market_prob_b);
    }
    if (allProbs.length === 0) {
      return { yMin: 0, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1] };
    }
    const dataMin = Math.min(...allProbs);
    const dataMax = Math.max(...allProbs);
    // Add 5% padding on each side, but ensure at least 10% range
    const range = dataMax - dataMin;
    const padding = Math.max(range * 0.15, 0.05);
    let lo = Math.max(0, dataMin - padding);
    let hi = Math.min(1, dataMax + padding);
    // Ensure minimum 10% visible range
    if (hi - lo < 0.10) {
      const center = (lo + hi) / 2;
      lo = Math.max(0, center - 0.05);
      hi = Math.min(1, center + 0.05);
    }
    // Round to nice tick values (every 5% or 10%)
    const tickStep = (hi - lo) > 0.4 ? 0.10 : (hi - lo) > 0.2 ? 0.05 : 0.02;
    const niceLo = Math.floor(lo / tickStep) * tickStep;
    const niceHi = Math.ceil(hi / tickStep) * tickStep;
    const ticks: number[] = [];
    for (let v = niceLo; v <= niceHi + tickStep * 0.01; v += tickStep) {
      ticks.push(Math.round(v * 1000) / 1000);
    }
    return { yMin: niceLo, yMax: niceHi, yTicks: ticks };
  }, [data]);

  // Compute EV+ stats for the info table
  const evStats = useMemo(() => {
    const evPlusPoints = data.filter(
      pt => (pt.ev_a != null && pt.ev_a > 0) || (pt.ev_b != null && pt.ev_b > 0)
    );

    // Min/max prob_a and prob_b across all data points
    const probAs = data.filter(pt => pt.prob_a != null).map(pt => ({ value: pt.prob_a!, ts: pt.timestamp, model: pt.model_name }));
    const probBs = data.filter(pt => pt.prob_b != null).map(pt => ({ value: pt.prob_b!, ts: pt.timestamp, model: pt.model_name }));

    const minProbA = probAs.length > 0 ? probAs.reduce((a, b) => a.value < b.value ? a : b) : null;
    const maxProbA = probAs.length > 0 ? probAs.reduce((a, b) => a.value > b.value ? a : b) : null;
    const minProbB = probBs.length > 0 ? probBs.reduce((a, b) => a.value < b.value ? a : b) : null;
    const maxProbB = probBs.length > 0 ? probBs.reduce((a, b) => a.value > b.value ? a : b) : null;

    // Current (latest) values per model
    const currentByModel = new Map<string, PredictionHistoryPoint>();
    for (const pt of data) {
      const existing = currentByModel.get(pt.model_name);
      if (!existing || pt.timestamp > existing.timestamp) {
        currentByModel.set(pt.model_name, pt);
      }
    }
    const currentPoints = Array.from(currentByModel.values());

    // EV+ periods
    const evPlusA = evPlusPoints.filter(pt => pt.ev_a != null && pt.ev_a > 0);
    const evPlusB = evPlusPoints.filter(pt => pt.ev_b != null && pt.ev_b > 0);

    // Max EV seen
    const maxEvA = evPlusA.length > 0 ? evPlusA.reduce((a, b) => (a.ev_a ?? 0) > (b.ev_a ?? 0) ? a : b) : null;
    const maxEvB = evPlusB.length > 0 ? evPlusB.reduce((a, b) => (b.ev_b ?? 0) > (a.ev_b ?? 0) ? b : a) : null;

    // Market probability stats
    const marketProbsA = data.filter(pt => pt.market_prob_a != null).map(pt => ({ value: pt.market_prob_a!, ts: pt.timestamp }));
    const marketProbsB = data.filter(pt => pt.market_prob_b != null).map(pt => ({ value: pt.market_prob_b!, ts: pt.timestamp }));
    const currentMarketA = marketProbsA.length > 0 ? marketProbsA[marketProbsA.length - 1] : null;
    const currentMarketB = marketProbsB.length > 0 ? marketProbsB[marketProbsB.length - 1] : null;
    const minMarketA = marketProbsA.length > 0 ? marketProbsA.reduce((a, b) => a.value < b.value ? a : b) : null;
    const maxMarketA = marketProbsA.length > 0 ? marketProbsA.reduce((a, b) => a.value > b.value ? a : b) : null;
    const minMarketB = marketProbsB.length > 0 ? marketProbsB.reduce((a, b) => a.value < b.value ? a : b) : null;
    const maxMarketB = marketProbsB.length > 0 ? marketProbsB.reduce((a, b) => a.value > b.value ? a : b) : null;

    return {
      evPlusCount: evPlusPoints.length,
      evPlusACount: evPlusA.length,
      evPlusBCount: evPlusB.length,
      minProbA, maxProbA, minProbB, maxProbB,
      currentPoints,
      maxEvA, maxEvB,
      evPlusA, evPlusB,
      currentMarketA, currentMarketB,
      minMarketA, maxMarketA, minMarketB, maxMarketB,
    };
  }, [data]);


  // Chart dimensions
  const W = 900;
  const H = 420;
  const margin = { top: 30, right: 30, bottom: 50, left: 55 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const yScale = (v: number) => margin.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
  const xScale = (i: number) => margin.left + (i / (timeLabels.length - 1)) * plotW;

  // Known active C0 identities; unmatched names remain visible as archived recorded series.
  const modelColors: Record<string, string> = {
    'Causal-C0': '#38bdf8',
    'Hybrid-Operational-Market': '#f59e0b',
  };
  const palette = ['#38bdf8', '#f59e0b', '#e040fb', '#66bb6a', '#ef5350'];
  modelNames.forEach((name, i) => {
    if (!modelColors[name]) {
      modelColors[name] = palette[i % palette.length];
    }
  });
  // Market color
  const marketColor = '#888888';

  // Build SVG path for a series
  const buildPath = (points: { timeIdx: number; value: number }[]) => {
    return points
      .map((p, i) => {
        const x = xScale(p.timeIdx);
        const y = yScale(p.value);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  };

  // Build EV zone rectangles (where ev_a > 0.05 or ev_b > 0.05)
  const evZones = useMemo(() => {
    const zones: { startIdx: number; endIdx: number; side: 'a' | 'b' }[] = [];
    // For each consecutive pair of time indices, check if any model has EV > 5%
    for (let i = 0; i < timeLabels.length - 1; i++) {
      const ts = timeLabels[i];
      const tsNext = timeLabels[i + 1];
      // Check all points at this timestamp
      const pointsAtTs = data.filter(d => d.timestamp === ts);
      const pointsAtTsNext = data.filter(d => d.timestamp === tsNext);
      const hasEvA = pointsAtTs.some(p => p.ev_a != null && p.ev_a > 0.05) ||
                     pointsAtTsNext.some(p => p.ev_a != null && p.ev_a > 0.05);
      const hasEvB = pointsAtTs.some(p => p.ev_b != null && p.ev_b > 0.05) ||
                     pointsAtTsNext.some(p => p.ev_b != null && p.ev_b > 0.05);
      if (hasEvA) zones.push({ startIdx: i, endIdx: i + 1, side: 'a' });
      if (hasEvB) zones.push({ startIdx: i, endIdx: i + 1, side: 'b' });
    }
    return zones;
  }, [data, timeLabels]);
  if (timeLabels.length < 2) {
    const pt = data[0];
    if (!pt) {
      return (
        <section className="prediction-history-section">
          <h2>Archiwum predykcji i EV</h2>
          <p className="no-data">Brak zapisanych predykcji dla tego spotkania. Użyj przycisku „🔮 Predykcja” powyżej.</p>
        </section>
      );
    }
    const pA = pt.prob_a ?? 0.5;
    const pB = pt.prob_b ?? 0.5;
    const mA = pt.market_prob_a;
    const mB = pt.market_prob_b;
    const evA = pt.ev_a;
    const evB = pt.ev_b;
    return (
      <section className="prediction-history-section">
        <h2>Predykcja i Zgodność Modeli</h2>
        <p className="no-data">Wizualizacja predykcji (zapisano: {pt.timestamp}) · model: {pt.model_name}</p>
        <div style={{ padding: '1.25rem', background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '10px', marginTop: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontWeight: 700, fontSize: '1rem' }}>
            <span style={{ color: '#38bdf8' }}>{teamA}: {(pA * 100).toFixed(1)}%</span>
            <span style={{ color: '#f59e0b' }}>{teamB}: {(pB * 100).toFixed(1)}%</span>
          </div>
          <div style={{ height: '20px', width: '100%', background: '#334155', borderRadius: '10px', overflow: 'hidden', display: 'flex', marginBottom: '0.75rem' }}>
            <div style={{ width: `${(pA * 100).toFixed(1)}%`, background: 'linear-gradient(90deg, #0284c7, #38bdf8)', transition: 'width 0.4s' }} />
            <div style={{ width: `${(pB * 100).toFixed(1)}%`, background: 'linear-gradient(90deg, #f59e0b, #d97706)', transition: 'width 0.4s' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
            {mA != null && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
                <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>CENA RYNKOWA FAIR (SHIN)</div>
                <div style={{ fontWeight: 600, marginTop: '2px' }}>{teamA}: {(mA * 100).toFixed(1)}% | {teamB}: {((mB ?? (1 - mA)) * 100).toFixed(1)}%</div>
              </div>
            )}
            {evA != null && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
                <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>OCZEKIWANA WARTOŚĆ (EV A)</div>
                <div style={{ fontWeight: 700, marginTop: '2px', color: evA > 0 ? '#4ade80' : '#e2e8f0' }}>{evA > 0 ? '+' : ''}{(evA * 100).toFixed(1)}% po podatku 12%</div>
              </div>
            )}
            {evB != null && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
                <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>OCZEKIWANA WARTOŚĆ (EV B)</div>
                <div style={{ fontWeight: 700, marginTop: '2px', color: evB > 0 ? '#4ade80' : '#e2e8f0' }}>{evB > 0 ? '+' : ''}{(evB * 100).toFixed(1)}% po podatku 12%</div>
              </div>
            )}
          </div>
        </div>
      </section>
    );
  }

  // Format time label
  const formatTimeLabel = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  };

  // Show every Nth time label to avoid overlap
  const labelStep = Math.max(1, Math.floor(timeLabels.length / 10));

  return (
    <section className="prediction-history-section">
      <h2>Archiwum predykcji i EV</h2>
      <p className="no-data">Zapisane prognozy dla tego meczu; starsze modele są wyłącznie historycznymi zapisami, nie bieżącą predykcją C0.</p>
      <div className="chart-container">
        <svg viewBox={`0 0 ${W} ${H}`} className="pred-chart">
          {/* EV zones - green for team A EV > 5%, red for team B EV > 5% */}
          {evZones.map((zone, i) => {
            const x1 = xScale(zone.startIdx);
            const x2 = xScale(zone.endIdx);
            const fill = zone.side === 'a' ? 'rgba(76,175,80,0.12)' : 'rgba(239,83,80,0.12)';
            return (
              <rect
                key={`evzone-${i}`}
                x={x1}
                y={margin.top}
                width={x2 - x1}
                height={plotH}
                fill={fill}
              />
            );
          })}

          {/* Y grid lines */}
          {yTicks.map(v => (
            <g key={`ygrid-${v}`}>
              <line
                x1={margin.left}
                y1={yScale(v)}
                x2={W - margin.right}
                y2={yScale(v)}
                stroke="#2a2a4a"
                strokeWidth={1}
              />
              <text
                x={margin.left - 8}
                y={yScale(v) + 4}
                textAnchor="end"
                fill="#888"
                fontSize={11}
              >
                {v >= 0.01 ? `${(v * 100).toFixed(0)}%` : `${(v * 100).toFixed(1)}%`}
              </text>
            </g>
          ))}

          {/* 50% reference line (only if within visible range) */}
          {yMin <= 0.5 && yMax >= 0.5 && (
            <line
              x1={margin.left}
              y1={yScale(0.5)}
              x2={W - margin.right}
              y2={yScale(0.5)}
              stroke="#4a4a6a"
              strokeWidth={1.5}
              strokeDasharray="6,4"
            />
          )}

          {/* Market probability line (team A) */}
          {(() => {
            const marketPoints: { timeIdx: number; value: number }[] = [];
            for (let i = 0; i < timeLabels.length; i++) {
              const pts = data.filter(d => d.timestamp === timeLabels[i] && d.market_prob_a != null);
              if (pts.length > 0) {
                const avg = pts.reduce((s, p) => s + (p.market_prob_a ?? 0), 0) / pts.length;
                marketPoints.push({ timeIdx: i, value: avg });
              }
            }
            if (marketPoints.length < 2) return null;
            return (
              <path
                d={buildPath(marketPoints)}
                fill="none"
                stroke={marketColor}
                strokeWidth={2}
                strokeDasharray="8,4"
              />
            );
          })()}

          {/* Model probability lines (team A prob_a) */}
          {modelNames.map(modelName => {
            const pts = models.get(modelName)!;
            const linePoints: { timeIdx: number; value: number }[] = [];
            for (const pt of pts) {
              const idx = timeLabels.indexOf(pt.timestamp);
              if (idx >= 0 && pt.prob_a != null) {
                linePoints.push({ timeIdx: idx, value: pt.prob_a });
              }
            }
            if (linePoints.length < 2) return null;
            return (
              <path
                key={`model-${modelName}`}
                d={buildPath(linePoints)}
                fill="none"
                stroke={modelColors[modelName]}
                strokeWidth={2.5}
              />
            );
          })}

          {/* Data points for model lines */}
          {modelNames.map(modelName => {
            const pts = models.get(modelName)!;
            const dots: { x: number; y: number; evA: number | null }[] = [];
            for (const pt of pts) {
              const idx = timeLabels.indexOf(pt.timestamp);
              if (idx >= 0 && pt.prob_a != null) {
                dots.push({
                  x: xScale(idx),
                  y: yScale(pt.prob_a),
                  evA: pt.ev_a,
                });
              }
            }
            return dots.map((d, i) => (
              <circle
                key={`dot-${modelName}-${i}`}
                cx={d.x}
                cy={d.y}
                r={d.evA != null && d.evA > 0.05 ? 4 : 2.5}
                fill={modelColors[modelName]}
                stroke={d.evA != null && d.evA > 0.05 ? '#4caf50' : 'none'}
                strokeWidth={d.evA != null && d.evA > 0.05 ? 2 : 0}
              />
            ));
          })}

          {/* Data points for market line */}
          {(() => {
            const dots: { x: number; y: number }[] = [];
            for (let i = 0; i < timeLabels.length; i++) {
              const pts = data.filter(d => d.timestamp === timeLabels[i] && d.market_prob_a != null);
              if (pts.length > 0) {
                const avg = pts.reduce((s, p) => s + (p.market_prob_a ?? 0), 0) / pts.length;
                dots.push({ x: xScale(i), y: yScale(avg) });
              }
            }
            return dots.map((d, i) => (
              <circle
                key={`market-dot-${i}`}
                cx={d.x}
                cy={d.y}
                r={2}
                fill={marketColor}
              />
            ));
          })()}

          {/* X axis labels */}
          {timeLabels.map((ts, i) => {
            if (i % labelStep !== 0 && i !== timeLabels.length - 1) return null;
            return (
              <text
                key={`xlabel-${i}`}
                x={xScale(i)}
                y={H - margin.bottom + 20}
                textAnchor="middle"
                fill="#888"
                fontSize={10}
              >
                {formatTimeLabel(ts)}
              </text>
            );
          })}

          {/* Axis labels */}
          <text
            x={margin.left - 40}
            y={margin.top + plotH / 2}
            textAnchor="middle"
            fill="#aaa"
            fontSize={12}
            transform={`rotate(-90, ${margin.left - 40}, ${margin.top + plotH / 2})`}
          >
            Prawdopodobieństwo ({teamA})
          </text>
        </svg>

        {/* Legend */}
        <div className="chart-legend">
          {modelNames.map(name => (
            <span key={name} className="legend-item">
              <span
                className="legend-line"
                style={{ backgroundColor: modelColors[name] }}
              />
              {name}
            </span>
          ))}
          <span className="legend-item">
            <span className="legend-line legend-dashed" style={{ backgroundColor: marketColor }} />
            Rynek (średnia)
          </span>
          <span className="legend-item">
            <span className="legend-ev ev-positive" />
            EV {teamA} &gt; 5%
          </span>
          <span className="legend-item">
            <span className="legend-ev ev-negative" />
            EV {teamB} &gt; 5%
          </span>
        </div>

        {/* EV+ Info Table */}
        {evStats.evPlusCount > 0 && (
          <div className="ev-info-table">
            <h3>Statystyki EV+</h3>
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>{teamA}</th>
                  <th>{teamB}</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="ev-info-label">Punkty EV+</td>
                  <td className="ev-info-value ev-positive">{evStats.evPlusACount}</td>
                  <td className="ev-info-value ev-negative">{evStats.evPlusBCount}</td>
                </tr>
                {evStats.maxEvA && (
                  <tr>
                    <td className="ev-info-label">Max EV</td>
                    <td className="ev-info-value ev-positive">{((evStats.maxEvA.ev_a ?? 0) * 100).toFixed(1)}%</td>
                    <td className="ev-info-value ev-negative">{((evStats.maxEvB?.ev_b ?? 0) * 100).toFixed(1)}%</td>
                  </tr>
                )}
                {evStats.minProbA && evStats.maxProbA && (
                  <tr>
                    <td className="ev-info-label">Prawd. min</td>
                    <td className="ev-info-value">{(evStats.minProbA.value * 100).toFixed(1)}% <span className="ev-info-ts">({new Date(evStats.minProbA.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })})</span></td>
                    <td className="ev-info-value">{(evStats.minProbB!.value * 100).toFixed(1)}% <span className="ev-info-ts">({new Date(evStats.minProbB!.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })})</span></td>
                  </tr>
                )}
                {evStats.minProbA && evStats.maxProbA && (
                  <tr>
                    <td className="ev-info-label">Prawd. max</td>
                    <td className="ev-info-value">{(evStats.maxProbA.value * 100).toFixed(1)}% <span className="ev-info-ts">({new Date(evStats.maxProbA.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })})</span></td>
                    <td className="ev-info-value">{(evStats.maxProbB!.value * 100).toFixed(1)}% <span className="ev-info-ts">({new Date(evStats.maxProbB!.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })})</span></td>
                  </tr>
                )}
                {evStats.currentPoints.length > 0 && (
                  <tr>
                    <td className="ev-info-label">Aktualne prawd.</td>
                    <td className="ev-info-value">
                      {evStats.currentPoints.map((pt, i) => (
                        <span key={i}>{pt.prob_a != null ? `${(pt.prob_a * 100).toFixed(1)}%` : '—'}{i < evStats.currentPoints.length - 1 ? ' / ' : ''}</span>
                      ))}
                    </td>
                    <td className="ev-info-value">
                      {evStats.currentPoints.map((pt, i) => (
                        <span key={i}>{pt.prob_b != null ? `${(pt.prob_b * 100).toFixed(1)}%` : '—'}{i < evStats.currentPoints.length - 1 ? ' / ' : ''}</span>
                      ))}
                    </td>
                  </tr>
                )}
                {evStats.currentPoints.length > 0 && (
                  <tr>
                    <td className="ev-info-label">Aktualne EV</td>
                    <td className="ev-info-value">
                      {evStats.currentPoints.map((pt, i) => (
                        <span key={i} className={pt.ev_a != null && pt.ev_a > 0 ? 'ev-positive' : ''}>{pt.ev_a != null ? `${(pt.ev_a * 100).toFixed(1)}%` : '—'}{i < evStats.currentPoints.length - 1 ? ' / ' : ''}</span>
                      ))}
                    </td>
                    <td className="ev-info-value">
                      {evStats.currentPoints.map((pt, i) => (
                        <span key={i} className={pt.ev_b != null && pt.ev_b > 0 ? 'ev-negative' : ''}>{pt.ev_b != null ? `${(pt.ev_b * 100).toFixed(1)}%` : '—'}{i < evStats.currentPoints.length - 1 ? ' / ' : ''}</span>
                      ))}
                    </td>
                  </tr>
                )}
                {evStats.currentMarketA != null && (
                  <tr>
                    <td className="ev-info-label">Rynek (aktualne)</td>
                    <td className="ev-info-value">
                      {evStats.currentMarketA != null ? `${(evStats.currentMarketA.value * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="ev-info-value">
                      {evStats.currentMarketB != null ? `${(evStats.currentMarketB.value * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                )}
                {evStats.minMarketA && (
                  <tr>
                    <td className="ev-info-label">Rynek min</td>
                    <td className="ev-info-value">
                      {(evStats.minMarketA.value * 100).toFixed(1)}%
                      <span className="ev-info-ts"> {new Date(evStats.minMarketA.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                    </td>
                    <td className="ev-info-value">
                      {evStats.minMarketB ? `${(evStats.minMarketB.value * 100).toFixed(1)}%` : '—'}
                      {evStats.minMarketB && <span className="ev-info-ts"> {new Date(evStats.minMarketB.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>}
                    </td>
                  </tr>
                )}
                {evStats.maxMarketA && (
                  <tr>
                    <td className="ev-info-label">Rynek max</td>
                    <td className="ev-info-value">
                      {(evStats.maxMarketA.value * 100).toFixed(1)}%
                      <span className="ev-info-ts"> {new Date(evStats.maxMarketA.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                    </td>
                    <td className="ev-info-value">
                      {evStats.maxMarketB ? `${(evStats.maxMarketB.value * 100).toFixed(1)}%` : '—'}
                      {evStats.maxMarketB && <span className="ev-info-ts"> {new Date(evStats.maxMarketB.ts).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

// ─── Match Detail Page ──────────────────────────────────────

export default function MatchDetail() {
  const { id } = useParams<{ id: string }>();
  const [match, setMatch] = useState<MatchDetailResponse | null>(null);
  const [predHistory, setPredHistory] = useState<PredictionHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingBestOf, setEditingBestOf] = useState(false);
  const [savingBestOf, setSavingBestOf] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [aliasModalSide, setAliasModalSide] = useState<'a' | 'b' | null>(null);
  const [aliasSearchQuery, setAliasSearchQuery] = useState('');
  const [aliasSearchResults, setAliasSearchResults] = useState<string[]>([]);
  const [aliasSaving, setAliasSaving] = useState(false);
  const [editingRosterSide, setEditingRosterSide] = useState<'a' | 'b' | null>(null);
  const [rosterDraft, setRosterDraft] = useState<RosterOverridePlayerInput[]>([]);
  const [rosterCandidates, setRosterCandidates] = useState<Record<number, RosterPlayerCandidate[]>>({});
  const [searchingRosterPlayer, setSearchingRosterPlayer] = useState<number | null>(null);
  const [savingRoster, setSavingRoster] = useState(false);
  const [rosterMessage, setRosterMessage] = useState<string | null>(null);
  const [mutationMessage, setMutationMessage] = useState<string | null>(null);
  const [fearlessResult, setFearlessResult] = useState<FearlessSimulationResponse | null>(null);
  const [simulatingFearless, setSimulatingFearless] = useState(false);
  const [fearlessBestOf, setFearlessBestOf] = useState<3 | 5>(5);
  const [fearlessError, setFearlessError] = useState<string | null>(null);

  const handleSimulateFearless = async () => {
    if (!match?.canonical_match_id) return;
    setSimulatingFearless(true);
    setFearlessError(null);
    try {
      const res = await simulateFearlessMatch(match.canonical_match_id, {
        best_of: fearlessBestOf,
        n_simulations: 5000,
        seed: 42,
      });
      setFearlessResult(res);
    } catch (err: unknown) {
      setFearlessError(errorMessage(err, 'Nie udało się zasymulować Fearless Draft.'));
    } finally {
      setSimulatingFearless(false);
    }
  };
  const route = useRef({ id, generation: 0 });
  if (route.current.id !== id) route.current = { id, generation: route.current.generation + 1 };
  const refreshGeneration = useRef(0);
  const refreshController = useRef<AbortController | null>(null);
  const aliasSearchGeneration = useRef(0);
  const rosterSearchGeneration = useRef(0);
  const searchControllers = useRef<Set<AbortController>>(new Set());
  const captureRoute = () => {
    const identity = route.current;
    const generation = identity.generation;
    return () => route.current === identity && identity.generation === generation;
  };
  const invalidateRefresh = () => {
    refreshGeneration.current++;
    refreshController.current?.abort();
  };
  const refreshMatch = useCallback(async () => {
    if (!id) return false;
    const identity = route.current;
    const routeGeneration = identity.generation;
    const generation = ++refreshGeneration.current;
    refreshController.current?.abort();
    const controller = new AbortController();
    refreshController.current = controller;
    const isCurrent = () => route.current === identity
      && identity.generation === routeGeneration && refreshGeneration.current === generation;
    try {
      const [matchData, historyData] = await Promise.all([
        fetchMatchDetail(Number(id), controller.signal),
        fetchPredictionHistory(Number(id), controller.signal),
      ]);
      if (!isCurrent()) return false;
      setMatch(matchData);
      setPredHistory(historyData);
      return true;
    } catch (err: unknown) {
      if (!isCurrent() || controller.signal.aborted) return false;
      throw err;
    } finally {
      if (isCurrent()) refreshController.current = null;
    }
  }, [id]);

  const mutateAlias = async (action: () => Promise<unknown>, closeModal = false) => {
    const isCurrent = captureRoute();
    invalidateRefresh();
    setAliasSaving(true);
    setMutationMessage(null);
    try {
      await action();
      if (!isCurrent()) return;
      if (closeModal) {
        aliasSearchGeneration.current++;
        setAliasModalSide(null);
      }
      await refreshMatch();
    } catch (err: unknown) {
      if (isCurrent()) setMutationMessage(errorMessage(err, 'Nie udało się zmienić mapowania.'));
    } finally {
      if (isCurrent()) setAliasSaving(false);
    }
  };

  const searchAliases = async (query: string) => {
    const isRouteCurrent = captureRoute();
    const generation = ++aliasSearchGeneration.current;
    setAliasSearchQuery(query);
    setAliasSearchResults([]);
    if (query.length < 2) return;
    const controller = new AbortController();
    searchControllers.current.add(controller);
    try {
      const result = await searchGolggTeams(query, 50, controller.signal);
      if (isRouteCurrent() && aliasSearchGeneration.current === generation) setAliasSearchResults(result.teams);
    } catch (err: unknown) {
      if (isRouteCurrent() && aliasSearchGeneration.current === generation && !controller.signal.aborted) {
        setMutationMessage(errorMessage(err, 'Nie udało się wyszukać drużyny.'));
      }
    } finally {
      searchControllers.current.delete(controller);
    }
  };

  useEffect(() => {
    const identity = route.current;
    identity.generation++;
    const generation = identity.generation;
    const controllers = searchControllers.current;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setMatch(null);
    setPredHistory([]);
    setMutationMessage(null);
    setRosterMessage(null);
    setEditingBestOf(false);
    setSavingBestOf(false);
    setPredicting(false);
    setSavingRoster(false);
    setEditingRosterSide(null);
    setRosterCandidates({});
    setSearchingRosterPlayer(null);
    setAliasModalSide(null);
    setAliasSaving(false);
    refreshMatch()
      .catch((err: unknown) => {
        if (!cancelled && route.current === identity && identity.generation === generation) {
          setError(errorMessage(err, 'Nie udało się pobrać meczu.'));
        }
      })
      .finally(() => {
        if (!cancelled && route.current === identity && identity.generation === generation) setLoading(false);
      });
    return () => {
      cancelled = true;
      identity.generation++;
      refreshController.current?.abort();
      controllers.forEach(controller => controller.abort());
      controllers.clear();
    };
  }, [id, refreshMatch]);

  useEffect(() => {
    if (!loading && match && (window.location.hash === '#in-game-props' || window.location.pathname.endsWith('/props'))) {
      const el = document.getElementById('in-game-props');
      if (el) {
        const timer = setTimeout(() => el.scrollIntoView({ behavior: 'smooth' }), 100);
        return () => clearTimeout(timer);
      }
    }
  }, [loading, match]);

  if (loading) {
    return <div className="loading">Ładowanie szczegółów meczu...</div>;
  }

  if (error) {
    return <div className="error">Błąd: {error}</div>;
  }

  if (!match || match.canonical_match_id !== Number(id)) {
    return <div className="empty">Nie znaleziono meczu</div>;
  }

  const formatDateTime = (iso: string | null) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString('pl-PL', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatScrapedAt = (iso: string | null) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString('pl-PL', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const fmtNum = (value: number | null | undefined, digits = 1) => (
    value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
  );

  const fmtPct = (value: number | null | undefined, digits = 1) => (
    value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`
  );

  const ratingProbRows = Object.entries(match.team_comparison?.rating_probabilities || {})
    .filter(([, value]) => Number.isFinite(value))
    .sort(([a], [b]) => a.localeCompare(b));

  const renderTeamSummary = (side: 'a' | 'b') => {
    const roster = side === 'a' ? match.roster_a : match.roster_b;
    const stats = side === 'a' ? match.recent_stats_a : match.recent_stats_b;
    const comparison = match.team_comparison;
    const teamName = side === 'a' ? match.team_a_name : match.team_b_name;
    const elo = side === 'a' ? comparison?.team_a_elo : comparison?.team_b_elo;
    const glicko = side === 'a' ? comparison?.team_a_glicko : comparison?.team_b_glicko;
    const rd = side === 'a' ? comparison?.team_a_glicko_rd : comparison?.team_b_glicko_rd;
    const games = side === 'a' ? comparison?.team_a_games_played : comparison?.team_b_games_played;
    return (
      <article className={`team-detail-card ${side === 'a' ? 'team-a-card' : 'team-b-card'}`}>
        <div className="team-detail-head">
          <h3>{teamName || roster?.team_name || '—'}</h3>
          <span>{roster?.team_name && roster.team_name !== teamName ? roster.team_name : 'GOL.GG / DB'}</span>
        </div>
        <div className="rating-kpis">
          <div><strong>{fmtNum(elo, 0)}</strong><span>Team Elo</span></div>
          <div><strong>{fmtNum(glicko, 0)}</strong><span>Glicko2</span></div>
          <div><strong>{fmtNum(rd, 0)}</strong><span>RD</span></div>
          <div><strong>{games ?? '—'}</strong><span>Gry ratingu</span></div>
        </div>
        {stats && (
          <div className="recent-stats-grid">
            <div><span>W20 winrate</span><strong>{fmtPct(stats.win_rate)}</strong></div>
            <div><span>Mecze / gry</span><strong>{stats.matches_count ?? '—'} / {stats.games_count ?? '—'}</strong></div>
            <div><span>K / D</span><strong>{fmtNum(stats.avg_kills)} / {fmtNum(stats.avg_deaths)}</strong></div>
            <div><span>GD@15</span><strong>{fmtNum(stats.avg_gd15, 0)}</strong></div>
            <div><span>Smoki</span><strong>{fmtNum(stats.avg_dragons)}</strong></div>
            <div><span>Nashory</span><strong>{fmtNum(stats.avg_nashors)}</strong></div>
          </div>
        )}
        {roster && (
          <div className="roster-meta">
            <span>Roster: {roster.roster_source || '—'}</span>
            <span>Źródło: {roster.source_date ? formatDateTime(roster.source_date) : '—'}</span>
            {roster.source_tournament && <span>{roster.source_tournament}</span>}
          </div>
        )}
      </article>
    );
  };

  const renderRoster = (roster: MatchDetailResponse['roster_a'], side: 'a' | 'b') => {
    const teamName = side === 'a' ? match.team_a_name : match.team_b_name;
    const isManual = side === 'a' ? match.roster_a_is_manual : match.roster_b_is_manual;
    const isEditing = editingRosterSide === side;
    const beginEdit = () => {
      const players = roster?.players || [];
      setRosterDraft(Array.from({ length: 5 }, (_, index) => {
        const player = players[index];
        return {
          player_id: player?.player_id || null,
          player_name: player?.player_name || '',
          role: player?.role || ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'][index],
        };
      }));
      setRosterMessage(null);
      rosterSearchGeneration.current++;
      setSearchingRosterPlayer(null);
      setRosterCandidates({});
      setEditingRosterSide(side);
    };
    const updateDraft = (index: number, field: keyof RosterOverridePlayerInput, value: string) => {
      setRosterDraft(current => current.map((player, idx) => idx === index
        ? { ...player, [field]: value, ...(field === 'player_name' ? { player_id: null } : {}) }
        : player));
      if (field === 'player_name') setRosterCandidates(current => ({ ...current, [index]: [] }));
      rosterSearchGeneration.current++;
      setSearchingRosterPlayer(null);
    };
    const findRosterPlayers = async (index: number) => {
      const player = rosterDraft[index];
      if (!player?.player_name.trim() || player.player_name.trim().length < 2) {
        setRosterMessage('Wpisz co najmniej 2 znaki nazwy zawodnika.');
        return;
      }
      const isRouteCurrent = captureRoute();
      const generation = ++rosterSearchGeneration.current;
      const controller = new AbortController();
      searchControllers.current.add(controller);
      const isCurrent = () => isRouteCurrent() && rosterSearchGeneration.current === generation;
      setSearchingRosterPlayer(index);
      try {
        const candidates = await searchRosterPlayers(match.canonical_match_id, side, player.player_name, player.role, controller.signal);
        if (!isCurrent()) return;
        setRosterCandidates(current => ({ ...current, [index]: candidates }));
        if (!candidates.length) setRosterMessage(`Brak kandydatów GOL.GG dla „${player.player_name}”.`);
      } catch (err: unknown) {
        if (isCurrent() && !controller.signal.aborted) setRosterMessage(errorMessage(err, 'Nie udało się wyszukać zawodnika GOL.GG.'));
      } finally {
        searchControllers.current.delete(controller);
        if (isCurrent()) setSearchingRosterPlayer(null);
      }
    };
    const selectRosterPlayer = (index: number, candidate: RosterPlayerCandidate) => {
      rosterSearchGeneration.current++;
      setSearchingRosterPlayer(null);
      setRosterDraft(current => current.map((player, idx) => idx === index
        ? { ...player, player_id: candidate.player_id, player_name: candidate.player_name }
        : player));
      setRosterCandidates(current => ({ ...current, [index]: [] }));
    };
    const saveRoster = async () => {
      if (rosterDraft.some(player => !player.player_name.trim())) {
        setRosterMessage('Uzupełnij nazwę każdego z 5 zawodników.');
        return;
      }
      const isCurrent = captureRoute();
      const matchId = match.canonical_match_id;
      let committedMessage: string | null = null;
      invalidateRefresh();
      setSavingRoster(true);
      setRosterMessage(null);
      try {
        const result = await updateMatchRoster(matchId, side, rosterDraft);
        if (!isCurrent()) return;
        invalidateRefresh();
        committedMessage = result.message;
        // The PUT is committed independently of the subsequent prediction.
        setMatch(current => current?.canonical_match_id === matchId ? {
          ...current,
          ...(side === 'a'
            ? { roster_a: result.roster, roster_a_is_manual: true }
            : { roster_b: result.roster, roster_b_is_manual: true }),
        } : current);
        setEditingRosterSide(null);
        setRosterMessage(result.message);
        const prediction = await predictMatch(matchId);
        if (!isCurrent()) return;
        setRosterMessage(`${result.message} ${prediction.message}`);
        if (prediction.status === 'ok') await refreshMatch();
      } catch (err: unknown) {
        if (isCurrent()) setRosterMessage(`${committedMessage ? `${committedMessage} Predykcja nie została przeliczona: ` : ''}${errorMessage(err, 'Nie udało się zapisać składu.')}`);
      } finally {
        if (isCurrent()) setSavingRoster(false);
      }
    };
    return (
      <article className="roster-card">
        <div className="roster-card-head">
          <div>
            <h3>{teamName || roster?.team_name || '—'}</h3>
            {isManual && <span className="manual-roster-badge">Override operatora</span>}
          </div>
          <div className="roster-actions">
            <span>avg Elo {fmtNum(roster?.avg_elo, 0)} · avg Glicko {fmtNum(roster?.avg_glicko, 0)}</span>
            {!isEditing && <button className="roster-edit-btn" onClick={beginEdit}>Edytuj skład</button>}
            {!isEditing && isManual && (
              <button
                className="roster-reset-btn"
                disabled={savingRoster}
                onClick={async () => {
                  const isCurrent = captureRoute();
                  invalidateRefresh();
                  setSavingRoster(true);
                  setRosterMessage(null);
                  try {
                    const result = await resetMatchRoster(match.canonical_match_id, side);
                    if (!isCurrent()) return;
                    setRosterMessage(result.message);
                    await refreshMatch();
                  } catch (err: unknown) {
                    if (isCurrent()) setRosterMessage(errorMessage(err, 'Nie udało się przywrócić składu.'));
                  } finally {
                    if (isCurrent()) setSavingRoster(false);
                  }
                }}
              >Usuń override meczu</button>
            )}
          </div>
        </div>
        {isEditing && (
          <div className="roster-editor">
            <p>Wyszukaj i wybierz zawodnika z GOL.GG — zapisywane jest stabilne ID, nie sama nazwa. Dzięki temu homonimy nie zostaną pomylone.</p>
            {rosterDraft.map((player, index) => (
              <div className="roster-editor-row" key={`${side}-${index}`}>
                <select value={player.role || ''} onChange={event => updateDraft(index, 'role', event.target.value)}>
                  {['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'].map(role => <option key={role} value={role}>{role}</option>)}
                </select>
                <input
                  value={player.player_name}
                  placeholder="Nazwa zawodnika z GOL.GG"
                  onChange={event => updateDraft(index, 'player_name', event.target.value)}
                />
                <button type="button" className="roster-search-btn" onClick={() => findRosterPlayers(index)} disabled={searchingRosterPlayer === index}>
                  {searchingRosterPlayer === index ? 'Szukam…' : 'Szukaj GOL.GG'}
                </button>
                <small>{player.player_id ? `Wybrano ID ${player.player_id}` : 'Wybierz wynik, aby zapisać ID'}</small>
                {(rosterCandidates[index] || []).length > 0 && (
                  <div className="roster-candidates">
                    {rosterCandidates[index].map(candidate => (
                      <button type="button" key={candidate.player_id} onClick={() => selectRosterPlayer(index, candidate)}>
                        <strong>{candidate.player_name}</strong> · {candidate.role || '—'} · {candidate.team_name || 'brak drużyny'}
                        {candidate.is_expected_team && <em> aktualna drużyna</em>}
                        <small>ID {candidate.player_id}</small>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div className="roster-editor-actions">
              <button className="roster-cancel-btn" onClick={() => setEditingRosterSide(null)} disabled={savingRoster}>Anuluj</button>
              <button className="roster-save-btn" onClick={saveRoster} disabled={savingRoster}>
                {savingRoster ? 'Zapisywanie i liczenie…' : 'Zapisz i przelicz predykcję'}
              </button>
            </div>
          </div>
        )}
        {!roster || roster.players.length === 0 ? (
          <p className="no-data small">Brak przewidywanego rosteru w features_json.</p>
        ) : (
          <div className="players-table">
            <div className="players-header">
              <span>Rola</span><span>Zawodnik</span><span>Elo</span><span>Glicko2</span><span>RD</span><span>Gry</span>
            </div>
            {roster.players.map((player, idx) => (
              <div key={`${player.player_id || player.player_name || idx}`} className="players-row">
                <span className="role-pill">{player.role || '—'}</span>
                <span className="player-name">{(player.player_name || player.player_id || '—').replace(/&nbsp;/g, ' ').replace(/\u00a0/g, ' ')}</span>
                <span>{fmtNum(player.elo_rating, 0)}</span>
                <span>{fmtNum(player.glicko_rating, 0)}</span>
                <span>{fmtNum(player.glicko_rd, 0)}</span>
                <span>{player.games_played ?? '—'}</span>
              </div>
            ))}
          </div>
        )}
      </article>
    );
  };

  return (
    <div className="match-detail">
      <Link to="/" className="back-link">
        ← Powrót do listy meczów
      </Link>

      <div className="match-header-card">
        <div className="league-info">
          <span className="league">{match.league || 'Nieznana liga'}</span>
          <span className="status">{match.status}</span>
          {!editingBestOf && (
            <span
              className={`best-of-badge bo${match.best_of || 1}`}
              onClick={() => setEditingBestOf(true)}
              title="Kliknij aby zmienić"
            >
              Bo{match.best_of || 1}
            </span>
          )}
          {editingBestOf && (
            <span className="best-of-picker">
              {[1, 3, 5, 7].map(bo => (
                <button
                  key={bo}
                  className={`bo-pill ${(match.best_of || 1) === bo ? 'active' : ''} ${savingBestOf ? 'saving' : ''}`}
                  onClick={async () => {
                    if (savingBestOf || (match.best_of || 1) === bo) {
                      setEditingBestOf(false);
                      return;
                    }
                    const isCurrent = captureRoute();
                    const matchId = match.canonical_match_id;
                    invalidateRefresh();
                    setSavingBestOf(true);
                    setMutationMessage(null);
                    try {
                      const result = await updateMatchBestOf(matchId, bo);
                      if (!isCurrent()) return;
                      invalidateRefresh();
                      setMatch(current => current?.canonical_match_id === matchId ? { ...current, best_of: result.best_of } : current);
                    } catch (err: unknown) {
                      if (isCurrent()) setMutationMessage(errorMessage(err, 'Nie udało się zmienić BO.'));
                    } finally {
                      if (isCurrent()) {
                        setSavingBestOf(false);
                        setEditingBestOf(false);
                      }
                    }
                  }}
                  disabled={savingBestOf}
                >
                  Bo{bo}
                </button>
              ))}
            </span>
          )}
          <button
            className={`predict-btn ${predicting ? 'predicting' : ''}`}
            onClick={async () => {
              if (predicting) return;
              const isCurrent = captureRoute();
              invalidateRefresh();
              setPredicting(true);
              setMutationMessage(null);
              try {
                const result = await predictMatch(match.canonical_match_id);
                if (!isCurrent()) return;
                setMutationMessage(result.message);
                if (result.status === 'ok') await refreshMatch();
              } catch (err: unknown) {
                if (isCurrent()) setMutationMessage(errorMessage(err, 'Predykcja nie powiodła się'));
              } finally {
                if (isCurrent()) setPredicting(false);
              }
            }}
            disabled={predicting}
          >
            {predicting ? '⏳ Predykcja...' : '🔮 Predykcja'}
          </button>
          <a
            href="#in-game-props"
            className="props-quicklink-btn"
            title="Przejdź do analizy rynków in-game i modeli statystycznych"
          >
            🎯 Analiza In-Game (Props)
          </a>
        </div>
        <h1>
          {match.team_a_name || '?'} vs {match.team_b_name || '?'}
        </h1>
        <div className="datetime">{formatDateTime(match.start_time_normalized)}</div>
      </div>
      {mutationMessage && <p className="error" role="status">{mutationMessage}</p>}

      {/* ── Rekomendacja zakładu (Co obstawiać) ─────────────────────────── */}
      {match.recommendation && (
        <section
          className={`recommendation-section ${
            match.recommendation.has_value
              ? 'value-bet-card'
              : match.recommendation.verdict === 'unmapped'
              ? 'unmapped-card'
              : 'no-bet-card'
          }`}
        >
          <div className="recommendation-header">
            <div
              className={`recommendation-badge ${
                match.recommendation.has_value
                  ? 'value-badge'
                  : match.recommendation.verdict === 'unmapped'
                  ? 'unmapped-badge'
                  : 'no-bet-badge'
              }`}
            >
              <span className="badge-icon">
                {match.recommendation.has_value
                  ? '🎯'
                  : match.recommendation.verdict === 'unmapped'
                  ? '⚠️'
                  : '⏸️'}
              </span>
              <span className="badge-text">{match.recommendation.verdict_label}</span>
            </div>
            {match.recommendation.threshold_info && (
              <div
                className={`recommendation-summary-pill ${
                  match.recommendation.has_value
                    ? 'highlight'
                    : 'neutral'
                }`}
              >
                {match.recommendation.threshold_info}
              </div>
            )}
          </div>

          {match.recommendation.has_value ? (
            <div className="recommendation-content">
              <div className="recommendation-main-grid">
                <div className="recommendation-pick-box">
                  <div className="pick-label">Sugerowany zakład</div>
                  <div className="pick-team">{match.recommendation.recommended_team}</div>
                  <div className="pick-sub">vs {match.recommendation.opponent_team}</div>
                </div>

                <div className="recommendation-offer-box">
                  <div className="offer-label">Najlepszy dostępny kurs</div>
                  <div className="offer-odds-row">
                    <span className="odds-number">{match.recommendation.best_odds?.toFixed(2)}</span>
                    <span className="offer-bookmaker">w {match.recommendation.bookmaker?.toUpperCase()}</span>
                  </div>
                  {match.recommendation.offer_url && (
                    <a
                      href={match.recommendation.offer_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="offer-link-btn"
                    >
                      Przejdź do oferty w {match.recommendation.bookmaker?.toUpperCase()} →
                    </a>
                  )}
                </div>
              </div>

              <div className="recommendation-kpis">
                <div className="kpi-box ev-kpi">
                  <div className="kpi-label">Oczekiwana wartość (EV)</div>
                  <div className="kpi-val positive">
                    {match.recommendation.ev !== null && match.recommendation.ev !== undefined
                      ? `${match.recommendation.ev > 0 ? '+' : ''}${(match.recommendation.ev * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <div className="kpi-note">po potrąceniu podatku 12%</div>
                </div>

                <div className="kpi-box prob-kpi">
                  <div className="kpi-label">Szansa wygranej (Model)</div>
                  <div className="kpi-val">
                    {match.recommendation.hybrid_prob !== null && match.recommendation.hybrid_prob !== undefined
                      ? `${(match.recommendation.hybrid_prob * 100).toFixed(1)}%`
                      : match.recommendation.model_prob !== null && match.recommendation.model_prob !== undefined
                      ? `${(match.recommendation.model_prob * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <div className="kpi-note">
                    fair rynek: {match.recommendation.market_prob ? `${(match.recommendation.market_prob * 100).toFixed(1)}%` : '—'}
                    {match.recommendation.edge_percentage_points !== null &&
                      match.recommendation.edge_percentage_points !== undefined && (
                        <span className="edge-text">
                          {' '}
                          ({match.recommendation.edge_percentage_points > 0 ? '+' : ''}
                          {match.recommendation.edge_percentage_points} p.p.)
                        </span>
                      )}
                  </div>
                </div>

                <div className="kpi-box stake-kpi">
                  <div className="kpi-label">Sugerowana stawka (1/4 Kelly)</div>
                  <div className="kpi-val">
                    {match.recommendation.suggested_stake_pct !== null &&
                    match.recommendation.suggested_stake_pct !== undefined
                      ? `${(match.recommendation.suggested_stake_pct * 100).toFixed(2)}%`
                      : '—'}
                  </div>
                  <div className="kpi-note">zarządzanie ryzykiem bankrolla</div>
                </div>

                <div className="kpi-box threshold-kpi">
                  <div className="kpi-label">Próg opłacalności</div>
                  <div className="kpi-val">
                    kurs ≥ {match.recommendation.min_odds_required?.toFixed(2) || '—'}
                  </div>
                  <div className="kpi-note">kurs poniżej tej wartości daje ujemne EV</div>
                </div>
                {match.recommendation.logit_disagreement !== null &&
                  match.recommendation.logit_disagreement !== undefined && (
                    <div className="kpi-box logit-kpi">
                      <div className="kpi-label">Rozbieżność Logit (Guard B)</div>
                      <div className="kpi-val">
                        Δ = {match.recommendation.logit_disagreement.toFixed(3)}
                      </div>
                      <div className="kpi-note">limit bezpieczeństwa ≤ 0.85</div>
                    </div>
                  )}
                {match.recommendation.thesis_disagreement !== null &&
                  match.recommendation.thesis_disagreement !== undefined && (
                    <div className="kpi-box thesis-kpi">
                      <div className="kpi-label">Thesis Model Guard</div>
                      <div className={`kpi-val ${match.recommendation.thesis_guard_passed ? 'positive' : ''}`}>
                        {match.recommendation.thesis_guard_passed ? '🟢 ZGODA' : '🔴 WETO'} (Δ = {(match.recommendation.thesis_disagreement * 100).toFixed(1)}%)
                      </div>
                      <div className="kpi-note">próg weta: rozbieżność ≤ 10.0%</div>
                    </div>
                  )}
                {match.recommendation.raw_clv !== null &&
                  match.recommendation.raw_clv !== undefined && (
                    <div className="kpi-box clv-kpi">
                      <div className="kpi-label">Oczekiwany Raw CLV</div>
                      <div className="kpi-val positive">
                        +{((match.recommendation.raw_clv) * 100).toFixed(1)}%
                      </div>
                      <div className="kpi-note">przewaga nad kursem zamknięcia</div>
                    </div>
                  )}
              </div>

              {match.recommendation.reasons && match.recommendation.reasons.length > 0 && (
                <div className="recommendation-reasons">
                  <div className="reasons-title">Dlaczego ten typ? (Analiza systemu)</div>
                  <ul className="reasons-list">
                    {match.recommendation.reasons.map((reason, idx) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="no-bet-content">
              <p className="no-bet-summary">{match.recommendation.summary}</p>
              {match.recommendation.reasons && match.recommendation.reasons.length > 0 && (
                <ul className="no-bet-reasons">
                  {match.recommendation.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </section>
      )}

      <section className="odds-section">
        <h2>Kursy bukmacherów</h2>
        {match.odds.length === 0 ? (
          <p className="no-data">Brak dostępnych kursów</p>
        ) : (
          <div className="odds-table">
            <div className="odds-header">
              <span>Bukmacher</span>
              <span>{match.team_a_name}</span>
              <span>EV A</span>
              <span>Kelly A</span>
              <span>{match.team_b_name}</span>
              <span>EV B</span>
              <span>Kelly B</span>
              <span>Aktualizacja</span>
              <span>Link</span>
            </div>
            {match.odds.map((odd, idx) => (
              <div key={idx} className="odds-row">
                <span className="bookmaker-name">
                  {odd.bookmaker}
                  <small>{odd.quote_status === 'actionable' ? ' świeży' : odd.quote_status === 'diagnostic' ? ' diagnostyczny' : ' niedostępny'}</small>
                </span>
                <span className="odds-value">
                  {odd.canonical_odds_a ? odd.canonical_odds_a.toFixed(2) : '—'}
                </span>
                <span className={`ev-value ${odd.ev_a && odd.ev_a > 0 ? 'positive' : ''}`}>
                  {odd.ev_a ? `${(odd.ev_a * 100).toFixed(1)}%` : '—'}
                </span>
                <span className={`kelly-value ${odd.kelly_a && odd.kelly_a > 0 ? 'positive' : ''}`}>
                  {odd.kelly_a ? `${(odd.kelly_a * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="odds-value">
                  {odd.canonical_odds_b ? odd.canonical_odds_b.toFixed(2) : '—'}
                </span>
                <span className={`ev-value ${odd.ev_b && odd.ev_b > 0 ? 'positive' : ''}`}>
                  {odd.ev_b ? `${(odd.ev_b * 100).toFixed(1)}%` : '—'}
                </span>
                <span className={`kelly-value ${odd.kelly_b && odd.kelly_b > 0 ? 'positive' : ''}`}>
                  {odd.kelly_b ? `${(odd.kelly_b * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="scraped-at" title={odd.quote_scraped_at ?? undefined}>
                  {formatScrapedAt(odd.quote_scraped_at ?? odd.scraped_at)}
                  {odd.odds_snapshot_id != null && <small> #{odd.odds_snapshot_id}</small>}
                </span>
                <span className="link-cell">
                  {odd.offer_url && (
                    <a href={odd.offer_url} target="_blank" rel="noopener noreferrer">
                      Oferta →
                    </a>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {match.predictions.length > 0 && (
        <section className="predictions-section">
          <h2>Predykcje modeli</h2>
          <div className="predictions-table">
            <div className="predictions-header">
              <span>Model</span>
              <span>Prawd. A</span>
              <span>Prawd. B</span>
              <span>Min kurs A</span>
              <span>Min kurs B</span>
              <span>EV A</span>
              <span>EV B</span>
              <span>Kelly A</span>
              <span>Kelly B</span>
            </div>
            {match.predictions.map((pred, idx) => (
              <div key={idx} className="predictions-row">
                <span className="model-name">
                  {pred.model_name} <small>({pred.model_version})</small>
                </span>
                <span className="prob-value">
                  {pred.prob_a ? `${(pred.prob_a * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="prob-value">
                  {pred.prob_b ? `${(pred.prob_b * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="min-odds-value">
                  {pred.prob_a ? (1 / (pred.prob_a * 0.88)).toFixed(2) : '—'}
                </span>
                <span className="min-odds-value">
                  {pred.prob_b ? (1 / (pred.prob_b * 0.88)).toFixed(2) : '—'}
                </span>
                <span className={`ev-value ${pred.ev_a && pred.ev_a > 0 ? 'positive' : ''}`}>
                  {pred.ev_a ? `${(pred.ev_a * 100).toFixed(1)}%` : '—'}
                </span>
                <span className={`ev-value ${pred.ev_b && pred.ev_b > 0 ? 'positive' : ''}`}>
                  {pred.ev_b ? `${(pred.ev_b * 100).toFixed(1)}%` : '—'}
                </span>
                <span className={`kelly-value ${pred.kelly_a && pred.kelly_a > 0 ? 'positive' : ''}`}>
                  {pred.kelly_a ? `${(pred.kelly_a * 100).toFixed(1)}%` : '—'}
                </span>
                <span className={`kelly-value ${pred.kelly_b && pred.kelly_b > 0 ? 'positive' : ''}`}>
                  {pred.kelly_b ? `${(pred.kelly_b * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
            ))}
          </div>
          {match.predictions.map((pred, idx) => (
            <article key={`${pred.model_name}-${pred.model_version}-${idx}`}>
              <h3>Pochodzenie wejść: {pred.model_name} ({pred.model_version})</h3>
              <PredictionProvenance diagnostics={pred.raw_diagnostics} predictedAt={pred.predicted_at} />
            </article>
          ))}
        </section>
      )}

      <section className="fearless-section">
        <div className="fearless-header">
          <div className="fearless-title-row">
            <h2>⚔️ Fearless Draft — Dokładny Wynik i Symulacja Map</h2>
            <span className="fearless-badge">Hard Fearless Rules</span>
          </div>
          <p className="fearless-desc">
            Sekwencyjna symulacja Markovowska uwzględniająca blokowanie zgranych bohaterów (10 banów na mapę),
            wybór strony przez przegranego i spadek ratingu z powodu wyczerpania puli komfortowej.
          </p>
          <div className="fearless-controls">
            <label>
              Format serii:
              <select
                value={fearlessBestOf}
                onChange={(e) => setFearlessBestOf(Number(e.target.value) as 3 | 5)}
                disabled={simulatingFearless}
                className="fearless-select"
              >
                <option value={5}>Best of 5 (Bo5)</option>
                <option value={3}>Best of 3 (Bo3)</option>
              </select>
            </label>
            <button
              type="button"
              className="btn-primary"
              onClick={handleSimulateFearless}
              disabled={simulatingFearless}
            >
              {simulatingFearless ? '⏳ Symulowanie 5000 ścieżek...' : '🎲 Symuluj Fearless Draft'}
            </button>
          </div>
        </div>

        {fearlessError && <p className="error-message">{fearlessError}</p>}

        {fearlessResult && (
          <div className="fearless-results-container">
            <div className="fearless-summary-card">
              <div className="fearless-series-probs">
                <div className="fearless-team-prob">
                  <span className="team-name">{fearlessResult.team_a_name}</span>
                  <span className="prob-pct">{(fearlessResult.series_prob_a * 100).toFixed(1)}%</span>
                  <span className="fair-odds">kurs: {(1 / fearlessResult.series_prob_a).toFixed(2)}</span>
                </div>
                <span className="vs-label">VS</span>
                <div className="fearless-team-prob">
                  <span className="team-name">{fearlessResult.team_b_name}</span>
                  <span className="prob-pct">{(fearlessResult.series_prob_b * 100).toFixed(1)}%</span>
                  <span className="fair-odds">kurs: {(1 / fearlessResult.series_prob_b).toFixed(2)}</span>
                </div>
              </div>
              <div className="fearless-meta-stats">
                <span>Średnia liczba unikalnych czempionów: <strong>{fearlessResult.mean_unique_champions}</strong></span>
                <span>Prób Monte Carlo: <strong>{fearlessResult.n_simulations}</strong></span>
              </div>
            </div>

            <div className="fearless-scorelines-section">
              <h3>Dokładny wynik (Correct Score Odds)</h3>
              <div className="fearless-scoreline-cards">
                {Object.entries(fearlessResult.scorelines).map(([score, prob]) => {
                  const fairOdds = prob > 0 ? (1 / prob).toFixed(2) : '—';
                  const minOddsTax = prob > 0 ? (1 / (prob * 0.88)).toFixed(2) : '—';
                  const isFavoredA = score.startsWith(String(Math.floor(fearlessResult.best_of / 2) + 1));
                  return (
                    <div key={score} className={`scoreline-card ${isFavoredA ? 'team-a-lead' : 'team-b-lead'}`}>
                      <div className="scoreline-label">{score}</div>
                      <div className="scoreline-prob">{(prob * 100).toFixed(1)}%</div>
                      <div className="scoreline-odds-row">
                        <span className="odds-label">Fair: <strong>{fairOdds}</strong></span>
                        <span className="odds-label-tax" title="Z uwzględnieniem 12% podatku">Min: <strong>{minOddsTax}</strong></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="fearless-maps-section">
              <h3>Prawdopodobieństwo wygrania mapy P(Wygrana Mapy t | Mapa jest grana)</h3>
              <div className="fearless-maps-grid">
                {Object.entries(fearlessResult.map_win_probabilities).map(([mapNum, probA]) => {
                  const m = Number(mapNum);
                  const probB = 1 - probA;
                  const reachProb = fearlessResult.map_reach_probabilities[m] ?? (m === 1 ? 1 : 0);
                  return (
                    <div key={mapNum} className="fearless-map-card">
                      <div className="map-card-header">
                        <span className="map-number">Mapa {mapNum}</span>
                        <span className="map-reach">Prawdop. zagrania: {(reachProb * 100).toFixed(0)}%</span>
                      </div>
                      <div className="map-prob-bar">
                        <div className="bar-team-a" style={{ width: `${(probA * 100).toFixed(1)}%` }}>
                          {(probA * 100).toFixed(1)}%
                        </div>
                        <div className="bar-team-b" style={{ width: `${(probB * 100).toFixed(1)}%` }}>
                          {(probB * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="map-teams-legend">
                        <span>{fearlessResult.team_a_name}: {(probA * 100).toFixed(1)}%</span>
                        <span>{fearlessResult.team_b_name}: {(probB * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </section>

      {predHistory.length > 0 && (
        <>
          <PredictionHistoryChart
            data={predHistory}
            teamA={match.team_a_name || 'A'}
            teamB={match.team_b_name || 'B'}
          />
          <details>
            <summary>Pochodzenie historycznych predykcji (wyłącznie zapisane diagnostics)</summary>
            {predHistory.map((point, idx) => (
              <details key={`${point.model_name}-${point.timestamp}-${idx}`}>
                <summary>{point.model_name} ({point.model_version}) · {point.timestamp}</summary>
                <PredictionProvenance diagnostics={point.raw_diagnostics} predictedAt={point.timestamp} />
              </details>
            ))}
          </details>
        </>
      )}

      {(match.team_comparison || match.recent_stats_a || match.recent_stats_b) && (
        <section className="match-intel-section">
          <h2>Intel przedmeczowy</h2>
          <div className="team-detail-grid">
            {renderTeamSummary('a')}
            {renderTeamSummary('b')}
          </div>
          {ratingProbRows.length > 0 && (
            <div className="rating-prob-strip">
              {ratingProbRows.map(([system, value]) => (
                <div key={system} className="rating-prob-item">
                  <span>{system.toUpperCase()}</span>
                  <strong>{fmtPct(value)}</strong>
                  <small>{match.team_a_name || 'A'}</small>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      <section className="rosters-section">
        <h2>Przewidywane składy i ratingi zawodników</h2>
        <p className="section-hint">
          Skład może pochodzić z aktualnego zapisu lub wcześniejszego meczu GOL.GG. Możesz ręcznie potwierdzić lub poprawić ID zawodników. Zapis składu jest oddzielny od ponownego obliczenia native C0; jeśli predykcja się nie powiedzie, zatwierdzony skład pozostaje zapisany. Ręczne potwierdzenie nie oznacza uwierzytelnionego ogłoszenia.
        </p>
        {rosterMessage && <p className="roster-message">{rosterMessage}</p>}
        <div className="rosters-grid">
          {renderRoster(match.roster_a, 'a')}
          {renderRoster(match.roster_b, 'b')}
        </div>
      </section>

      {match.team_comparison && (
        <section className="comparison-section">
          <h2>Porównanie drużyn</h2>
          <div className="comparison-table">
            <div className="comparison-header">
              <span></span>
              <span>{match.team_a_name}</span>
              <span>{match.team_b_name}</span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">Nazwa kanoniczna</span>
              <span>{match.team_comparison.team_a?.canonical_name || '—'}</span>
              <span>{match.team_comparison.team_b?.canonical_name || '—'}</span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">Nazwa gol.gg</span>
              <span>{match.team_comparison.team_a?.golgg_name || '—'}</span>
              <span>{match.team_comparison.team_b?.golgg_name || '—'}</span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">Pewność mapowania</span>
              <span className="mapping-cell">
                {match.team_comparison.team_a?.confidence
                  ? `${(match.team_comparison.team_a.confidence * 100).toFixed(0)}%`
                  : '—'}
                <button
                  className="alias-link-btn"
                  onClick={() => {
                    aliasSearchGeneration.current++;
                    setAliasModalSide('a');
                    setAliasSearchQuery('');
                    setAliasSearchResults([]);
                  }}
                  title="Połącz z drużyną GolGG"
                >
                  🔗
                </button>
                {match.team_comparison.team_a?.source === 'alias' && (
                  <button
                    className="alias-delete-btn"
                    disabled={aliasSaving}
                    onClick={() => mutateAlias(() => deleteTeamAlias(match.team_comparison?.team_a?.canonical_name || match.team_a_name || ''))}
                    title="Usuń mapowanie"
                  >
                    🗑️
                  </button>
                )}
                {match.team_comparison.team_a?.source === 'blocked' && (
                  <button
                    className="alias-unblock-btn"
                    disabled={aliasSaving}
                    onClick={() => mutateAlias(() => unblockTeamAlias(match.team_comparison?.team_a?.canonical_name || match.team_a_name || ''))}
                    title="Odblokuj mapowanie"
                  >
                    ✅
                  </button>
                )}
                {aliasModalSide === 'a' && (
                  <div className="alias-dropdown">
                    <input
                      type="text"
                      placeholder="Szukaj drużyny GolGG..."
                      value={aliasSearchQuery}
                      onChange={e => searchAliases(e.target.value)}
                      autoFocus
                    />
                    {aliasSearchResults.length > 0 && (
                      <ul className="alias-results">
                        {aliasSearchResults.slice(0, 20).map((team) => (
                          <li key={team}>
                            <button
                              disabled={aliasSaving}
                              onClick={() => mutateAlias(() => createTeamAlias({
                                raw_name: match.team_comparison?.team_a?.canonical_name || match.team_a_name || '',
                                golgg_team_name: team,
                                source_system: 'bookmaker',
                                league_pattern: match.league || undefined,
                              }), true)}
                            >
                              {team}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    <button className="alias-close-btn" onClick={() => { aliasSearchGeneration.current++; setAliasModalSide(null); }}>✕</button>
                  </div>
                )}
              </span>
              <span className="mapping-cell">
                {match.team_comparison.team_b?.confidence
                  ? `${(match.team_comparison.team_b.confidence * 100).toFixed(0)}%`
                  : '—'}
                <button
                  className="alias-link-btn"
                  onClick={() => {
                    aliasSearchGeneration.current++;
                    setAliasModalSide('b');
                    setAliasSearchQuery('');
                    setAliasSearchResults([]);
                  }}
                  title="Połącz z drużyną GolGG"
                >
                  🔗
                </button>
                {match.team_comparison.team_b?.source === 'alias' && (
                  <button
                    className="alias-delete-btn"
                    disabled={aliasSaving}
                    onClick={() => mutateAlias(() => deleteTeamAlias(match.team_comparison?.team_b?.canonical_name || match.team_b_name || ''))}
                    title="Usuń mapowanie"
                  >
                    🗑️
                  </button>
                )}
                {match.team_comparison.team_b?.source === 'blocked' && (
                  <button
                    className="alias-unblock-btn"
                    disabled={aliasSaving}
                    onClick={() => mutateAlias(() => unblockTeamAlias(match.team_comparison?.team_b?.canonical_name || match.team_b_name || ''))}
                    title="Odblokuj mapowanie"
                  >
                    ✅
                  </button>
                )}
                {aliasModalSide === 'b' && (
                  <div className="alias-dropdown">
                    <input
                      type="text"
                      placeholder="Szukaj drużyny GolGG..."
                      value={aliasSearchQuery}
                      onChange={e => searchAliases(e.target.value)}
                      autoFocus
                    />
                    {aliasSearchResults.length > 0 && (
                      <ul className="alias-results">
                        {aliasSearchResults.slice(0, 20).map((team) => (
                          <li key={team}>
                            <button
                              disabled={aliasSaving}
                              onClick={() => mutateAlias(() => createTeamAlias({
                                raw_name: match.team_comparison?.team_b?.canonical_name || match.team_b_name || '',
                                golgg_team_name: team,
                                source_system: 'bookmaker',
                                league_pattern: match.league || undefined,
                              }), true)}
                            >
                              {team}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    <button className="alias-close-btn" onClick={() => { aliasSearchGeneration.current++; setAliasModalSide(null); }}>✕</button>
                  </div>
                )}
              </span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">
                Rating {match.team_comparison.rating_system || ''}
              </span>
              <span className={`rating-value ${
                match.team_comparison.team_a_rating && match.team_comparison.team_b_rating
                  ? match.team_comparison.team_a_rating > match.team_comparison.team_b_rating
                    ? 'rating-higher' : 'rating-lower'
                  : ''
              }`}>
                {match.team_comparison.team_a_rating
                  ? match.team_comparison.team_a_rating.toFixed(1)
                  : '—'}
              </span>
              <span className={`rating-value ${
                match.team_comparison.team_a_rating && match.team_comparison.team_b_rating
                  ? match.team_comparison.team_b_rating > match.team_comparison.team_a_rating
                    ? 'rating-higher' : 'rating-lower'
                  : ''
              }`}>
                {match.team_comparison.team_b_rating
                  ? match.team_comparison.team_b_rating.toFixed(1)
                  : '—'}
              </span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">Team Elo</span>
              <span>{fmtNum(match.team_comparison.team_a_elo, 0)}</span>
              <span>{fmtNum(match.team_comparison.team_b_elo, 0)}</span>
            </div>
            <div className="comparison-row">
              <span className="comparison-label">Team Glicko2 / RD</span>
              <span>{fmtNum(match.team_comparison.team_a_glicko, 0)} / {fmtNum(match.team_comparison.team_a_glicko_rd, 0)}</span>
              <span>{fmtNum(match.team_comparison.team_b_glicko, 0)} / {fmtNum(match.team_comparison.team_b_glicko_rd, 0)}</span>
            </div>
            {match.team_comparison.team_a_rating && match.team_comparison.team_b_rating && (
              <div className="comparison-row">
                <span className="comparison-label">Różnica ratingów</span>
                <span className="rating-diff">
                  {(match.team_comparison.team_a_rating - match.team_comparison.team_b_rating) > 0 ? '+' : ''}
                  {(match.team_comparison.team_a_rating - match.team_comparison.team_b_rating).toFixed(1)}
                </span>
                <span className="rating-diff-muted">
                  {match.team_comparison.rating_system || 'rating'}
                </span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── Analiza In-Game / Rynki Poboczne & Modele Statystyczne (IDEA-018) ── */}
      <MatchPropsAnalysis
        canonicalMatchId={parseInt(id!)}
        teamAName={match.team_a_name || 'Team A'}
        teamBName={match.team_b_name || 'Team B'}
        league={match.league}
      />
    </div>
  );
}
