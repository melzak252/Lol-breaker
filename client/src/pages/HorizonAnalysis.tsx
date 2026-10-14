import { useState, useEffect, useCallback } from 'react';
import { HorizonAccuracyResponse, ModelVsBookmakerTest, HybridBinMetrics } from '../types';
import { fetchHorizonAccuracy } from '../api/client';
import './HorizonAnalysis.css';

/* ─── Color palette ─────────────────────────────────────── */
const COLORS = {
  bar:    '#4fc3f7',
  logloss:'#e040fb',
  auc:    '#ff9800',
  grid:   '#2a2a4a',
  axis:   '#888',
  label:  '#aaa',
  title:  '#fff',
  modelHybrid:  '#00e676',
  modelBase:    '#fdd835',
  modelThesis:  '#ff6b9d',
  modelThesisHybrid: '#00d4ff',
};

/* ─── Chart dimensions ──────────────────────────────────── */
const CHART_W = 800;
const CHART_H = 340;
const PAD = { top: 30, right: 120, bottom: 70, left: 60 };

/* ─── Helpers ───────────────────────────────────────────── */
function fmt(v: number, d = 3): string {
  return v.toFixed(d);
}

function yScale(domain: [number, number]): (v: number) => number {
  const [ymin, ymax] = domain;
  const rangeH = CHART_H - PAD.top - PAD.bottom;
  return (v: number) => PAD.top + rangeH - ((v - ymin) / (ymax - ymin)) * rangeH;
}

function xPos(i: number, total: number): number {
  const rangeW = CHART_W - PAD.left - PAD.right;
  return PAD.left + (i + 0.5) * (rangeW / total);
}

function logGamma(z: number): number {
  const p = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < p.length; i++) x += p[i] / (z + i + 1);
  const t = z + p.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function studentTPdf(x: number, df: number): number {
  const logCoef = logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(logCoef - ((df + 1) / 2) * Math.log(1 + (x * x) / df));
}

/* ─── Stat card ─────────────────────────────────────────── */
function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

/* ─── Horizon bin y-ticks ───────────────────────────────── */
function yAxisTicks(domain: [number, number]): number[] {
  const [lo, hi] = domain;
  const raw = Math.abs(hi - lo);
  const rough = raw / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const nice = [1, 2, 2.5, 5, 10].find(v => v * mag >= rough)! * mag;
  const start = Math.floor(lo / nice) * nice;
  const ticks: number[] = [];
  for (let v = start; v <= hi + nice * 0.5; v = +(v + nice).toFixed(6)) {
    ticks.push(v);
  }
  return ticks;
}

/* ════════════════════════════════════════════════════════════
   Component
   ════════════════════════════════════════════════════════════ */
export default function HorizonAnalysis() {
  const [data, setData] = useState<HorizonAccuracyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [daysBack, setDaysBack] = useState(90);
  const [minMatches, setMinMatches] = useState(10);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchHorizonAccuracy(daysBack, minMatches);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, [daysBack, minMatches]);

  useEffect(() => { load(); }, [load]);

  /* ─── Bin availability info ──────────────────────────── */
  const allBins = data?.bins ?? [];
  const shownBins = allBins.filter(b => b.match_count >= minMatches);
  const shownCount = shownBins.length;
  const skippedCount = allBins.length - shownCount;

  /* ─── Loading / Error ────────────────────────────────── */
  if (loading) {
    return (
      <div className="horizon-page">
        <h1>Odds Horizon Accuracy</h1>
        <p className="loading-text">Loading…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="horizon-page">
        <h1>Odds Horizon Accuracy</h1>
        <p className="error-text">{error ?? 'No data'}</p>
      </div>
    );
  }

  /* ─── Histogram chart (match count per bin) ────────── */
  function HistogramChart() {
    const bins = allBins;
    const n = bins.length;
    if (n === 0) return <p className="no-data">No bins returned.</p>;
    const maxCount = Math.max(...bins.map(b => b.match_count));
    const domain: [number, number] = [0, maxCount * 1.15 || 1];
    const y = yScale(domain);
    const ticks = yAxisTicks(domain);
    const barW = Math.min(50, (CHART_W - PAD.left - PAD.right) / n * 0.7);

    return (
      <div className="chart-section">
        <h3>Histogram — Match count per horizon bin</h3>
        <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="horizon-chart">
          {/* Y grid */}
          {ticks.map(v => (
            <g key={v}>
              <line x1={PAD.left} y1={y(v)} x2={CHART_W - PAD.right} y2={y(v)} stroke={COLORS.grid} strokeWidth={1} />
              <text x={PAD.left - 6} y={y(v) + 4} textAnchor="end" fill={COLORS.axis} fontSize={11}>{Math.round(v)}</text>
            </g>
          ))}
          {/* Bars */}
          {bins.map((b, i) => {
            const cx = xPos(i, n);
            const barH = CHART_H - PAD.top - PAD.bottom - (y(b.match_count) - PAD.top);
            const barY = y(b.match_count);
            const isSkipped = b.match_count < minMatches;
            return (
              <g key={b.label}>
                <rect x={cx - barW / 2} y={barY} width={barW} height={barH}
                  fill={isSkipped ? '#555' : COLORS.bar} opacity={isSkipped ? 0.5 : 0.85} rx={2}>
                  <title>
                    {b.label}: {b.match_count} matches, {b.snapshot_count} snapshots
                    {isSkipped ? ' (SKIPPED — <10 matches)' : ''}
                  </title>
                </rect>
                <text x={cx} y={barY - 6} textAnchor="middle" fill={COLORS.label} fontSize={10}>
                  {b.match_count}
                </text>
                <text x={cx} y={CHART_H - 10} textAnchor="end" fill={COLORS.axis} fontSize={9}
                  transform={`rotate(-30, ${cx}, ${CHART_H - 10})`}>
                  {b.label}
                </text>
              </g>
            );
          })}
          {/* Y-axis title */}
          <text x={14} y={CHART_H / 2} textAnchor="middle" fill={COLORS.label} fontSize={11}
            transform={`rotate(-90, 14, ${CHART_H / 2})`}>Matches</text>
        </svg>
      </div>
    );
  }

  /* ─── Model reference lines (pure models only) ───────── */
  function ModelRefLines({ metricKey, domain }: {
    metricKey: 'avg_logloss' | 'avg_auc';
    domain: [number, number];
  }) {
    const refs = data?.model_references ?? [];
    if (refs.length === 0) return null;
    const y = yScale(domain);

    return (
      <>
        {refs.map(r => {
          const val = r[metricKey];
          if (val === null || val === undefined) return null;
          const cy = y(val);
          const nameLower = r.model_name.toLowerCase();
          // Skip hybrid models — they're drawn as dynamic lines
          if (nameLower.includes('hybrid')) return null;
          
          let color: string;
          let label: string;
          if (nameLower.includes('thesis') || nameLower.includes('sym-cal')) {
            color = COLORS.modelThesis;
            label = 'Thesis';
          } else {
            color = COLORS.modelBase;
            label = 'Base model';
          }
          return (
            <g key={r.model_name}>
              {/* Dashed reference line */}
              <line x1={PAD.left} y1={cy} x2={CHART_W - PAD.right + 80} y2={cy}
                stroke={color} strokeWidth={1.5} strokeDasharray="6,4" opacity={0.8} />
              {/* Label + value */}
              <text x={CHART_W - PAD.right + 4} y={cy - 4} fill={color} fontSize={10}>
                {label}
              </text>
              <text x={CHART_W - PAD.right + 4} y={cy + 8} fill={color} fontSize={9}
                opacity={0.8}>
                {fmt(val)} ({r.n_matches} matches)
              </text>
            </g>
          );
        })}
      </>
    );
  }

  /* ─── Dynamic hybrid lines (per-bin metrics) ─────────── */
  function HybridDynamicLines({ metricKey, domain }: {
    metricKey: 'avg_logloss' | 'avg_auc';
    domain: [number, number];
  }) {
    const hybridBins = data?.hybrid_model_bins ?? [];
    if (hybridBins.length === 0) return null;
    const y = yScale(domain);
    const shownBins = data?.bins ?? [];
    const n = shownBins.length;
    if (n === 0) return null;

    return (
      <>
        {hybridBins.map(hb => {
          const nameLower = hb.model_name.toLowerCase();
          const color = nameLower.includes('thesis') ? COLORS.modelThesisHybrid : COLORS.modelHybrid;
          const label = nameLower.includes('thesis') ? 'Thesis Hybrid' : 'Hybrid';
          
          // Build path from bins — match by label to the shown bins x-positions
          const points: { x: number; y: number; bin: HybridBinMetrics }[] = [];
          hb.bins.forEach((bin) => {
            const val = bin[metricKey];
            if (val !== null && val !== undefined && bin.match_count >= minMatches) {
              // Find the x-position by matching the bin label to the shown bins
              const binIdx = shownBins.findIndex(sb => sb.label === bin.label);
              if (binIdx < 0) return;
              const cx = xPos(binIdx, n);
              const cy = y(val);
              points.push({ x: cx, y: cy, bin });
            }
          });
          
          if (points.length === 0) return null;
          
          const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
          
          return (
            <g key={hb.model_name}>
              {/* Dashed line connecting bins */}
              <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} 
                strokeDasharray="6,4" opacity={0.8} />
              {/* Points at each bin */}
              {points.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={3} fill={color} stroke="#1a1a2e" strokeWidth={1}>
                  <title>{p.bin.label}: {metricKey}={fmt(p.bin[metricKey] ?? 0)} ({p.bin.match_count} matches, {p.bin.snapshot_count} snapshots)</title>
                </circle>
              ))}
              {/* Label at last point */}
              {points.length > 0 && (
                <text x={points[points.length - 1].x + 8} y={points[points.length - 1].y - 6} 
                  fill={color} fontSize={9} opacity={0.9}>
                  {label}
                </text>
              )}
            </g>
          );
        })}
      </>
    );
  }

  /* ─── Per-bookmaker lines (subtle) ───────────────────── */
  function BookmakerLines({ metricKey, domain }: {
    metricKey: 'avg_logloss' | 'avg_auc';
    domain: [number, number];
  }) {
    const bkBins = data?.bookmaker_bins ?? [];
    if (bkBins.length === 0) return null;
    const y = yScale(domain);
    const shownBins = data?.bins ?? [];
    const n = shownBins.length;
    if (n === 0) return null;

    // Subtle color palette for bookmakers
    const bkColors = [
      '#8e99a4', '#a3b1bf', '#7c8c9a', '#9ea8b3', '#6b7b8a',
      '#b0bec5', '#90a4ae', '#78909c', '#607d8b', '#546e7a',
    ];

    return (
      <>
        {bkBins.map((bk, bkIdx) => {
          const color = bkColors[bkIdx % bkColors.length];

          // Build path from bins — match by label to the shown bins x-positions
          const points: { x: number; y: number; bin: HybridBinMetrics }[] = [];
          bk.bins.forEach((bin) => {
            const val = bin[metricKey];
            if (val !== null && val !== undefined && bin.match_count >= minMatches) {
              // Find the x-position by matching the bin label to the shown bins
              const binIdx = shownBins.findIndex(sb => sb.label === bin.label);
              if (binIdx < 0) return;
              const cx = xPos(binIdx, n);
              const cy = y(val);
              points.push({ x: cx, y: cy, bin });
            }
          });

          if (points.length === 0) return null;

          const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');

          return (
            <g key={`bk-${bk.bookmaker_id}`}>
              <path d={pathD} fill="none" stroke={color} strokeWidth={1}
                strokeDasharray="3,3" opacity={0.45} />
              {points.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={2} fill={color} opacity={0.4}>
                  <title>{bk.bookmaker_name} {p.bin.label}: {metricKey}={fmt(p.bin[metricKey] ?? 0)} ({p.bin.match_count} matches)</title>
                </circle>
              ))}
            </g>
          );
        })}
      </>
    );
  }

  /* ─── Shared line chart ───────────────────────────────── */
  function MetricLineChart({
    title,
    metricKey,
    color,
    yLabel,
    domain,
  }: {
    title: string;
    metricKey: 'avg_logloss' | 'avg_auc';
    color: string;
    yLabel: string;
    domain: [number, number];
  }) {
    const bins = shownBins;
    const n = bins.length;
    if (n === 0) return <p className="no-data">No bins with ≥{minMatches} matches.</p>;
    const y = yScale(domain);
    const ticks = yAxisTicks(domain);

    const pathD = bins.map((b, i) => {
      const cx = xPos(i, n);
      const val = b[metricKey];
      if (val === null || val === undefined) return '';
      const cy = y(val);
      return `${i === 0 ? 'M' : 'L'}${cx},${cy}`;
    }).filter(d => d !== '').join(' ');

    return (
      <div className="chart-section">
        <h3>{title}</h3>
        <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="horizon-chart">
          {/* Y grid */}
          {ticks.map(v => (
            <g key={v}>
              <line x1={PAD.left} y1={y(v)} x2={CHART_W - PAD.right} y2={y(v)} stroke={COLORS.grid} strokeWidth={1} />
              <text x={PAD.left - 6} y={y(v) + 4} textAnchor="end" fill={COLORS.axis} fontSize={11}>{fmt(v)}</text>
            </g>
          ))}
          {/* Model reference lines (pure models) */}
          <ModelRefLines metricKey={metricKey} domain={domain} />
          {/* Dynamic hybrid lines (per-bin) */}
          <HybridDynamicLines metricKey={metricKey} domain={domain} />
          {/* Per-bookmaker lines (subtle, behind main line) */}
          <BookmakerLines metricKey={metricKey} domain={domain} />
          {/* Line */}
          <path d={pathD} fill="none" stroke={color} strokeWidth={2.5} />
          {/* Points + labels */}
          {bins.map((b, i) => {
            const cx = xPos(i, n);
            const val = b[metricKey];
            if (val === null || val === undefined) return null;
            const cy = y(val);
            return (
              <g key={b.label}>
                <circle cx={cx} cy={cy} r={4} fill={color} stroke="#1a1a2e" strokeWidth={1.5}>
                  <title>{b.label}: {metricKey}={fmt(val)} ({b.match_count} matches)</title>
                </circle>
                <text x={cx + 8} y={cy - 6} fill={color} fontSize={10}>
                  {fmt(val)}
                </text>
                <text x={cx} y={CHART_H - 10} textAnchor="end" fill={COLORS.axis} fontSize={9}
                  transform={`rotate(-30, ${cx}, ${CHART_H - 10})`}>
                  {b.label}
                </text>
              </g>
            );
          })}
          {/* Y-axis title */}
          <text x={14} y={CHART_H / 2} textAnchor="middle" fill={COLORS.label} fontSize={11}
            transform={`rotate(-90, 14, ${CHART_H / 2})`}>{yLabel}</text>
        </svg>
      </div>
    );
  }

  /* ─── Statistical tests: model vs average bookmaker ───── */
  function StudentTDistributionChart({ tests }: { tests: ModelVsBookmakerTest[] }) {
    const drawable = tests.filter(t => t.t_stat !== null && t.t_critical_95_one_sided !== null && t.df > 0);
    if (drawable.length === 0) return <p className="no-data">No t-test distributions available.</p>;

    const W = 800;
    const H = 310;
    const pad = { top: 25, right: 35, bottom: 42, left: 45 };
    const xs = Array.from({ length: 241 }, (_, i) => -4 + (8 * i) / 240);
    const maxY = Math.max(...drawable.flatMap(t => xs.map(x => studentTPdf(x, t.df)))) * 1.12;
    const sx = (x: number) => pad.left + ((x + 4) / 8) * (W - pad.left - pad.right);
    const sy = (y: number) => H - pad.bottom - (y / maxY) * (H - pad.top - pad.bottom);
    const colors = ['#00d4ff', '#ff6b9d', '#00e676', '#fdd835'];

    return (
      <div className="chart-section stat-tests-chart-section">
        <h3>Gdzie jesteśmy na rozkładzie t-Studenta?</h3>
        <p className="stat-tests-subtitle">
          Test jednostronny: H₁ = model ma mniejszy błąd niż średni bukmacher.
          Dodatnia statystyka t oznacza przewagę modelu; pionowa linia przerywana to próg α=0.05.
        </p>
        <svg viewBox={`0 0 ${W} ${H}`} className="horizon-chart t-dist-chart">
          {[ -3, -2, -1, 0, 1, 2, 3 ].map(x => (
            <g key={x}>
              <line x1={sx(x)} y1={pad.top} x2={sx(x)} y2={H - pad.bottom} stroke={COLORS.grid} strokeWidth={1} />
              <text x={sx(x)} y={H - 16} fill={COLORS.axis} fontSize={10} textAnchor="middle">{x}</text>
            </g>
          ))}
          <line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke={COLORS.axis} strokeWidth={1} />
          <text x={W / 2} y={H - 2} fill={COLORS.label} fontSize={11} textAnchor="middle">statystyka t</text>

          {drawable.map((t, idx) => {
            const color = colors[idx % colors.length];
            const pathD = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x)},${sy(studentTPdf(x, t.df))}`).join(' ');
            const tStat = Math.max(-4, Math.min(4, t.t_stat ?? 0));
            const critical = Math.max(-4, Math.min(4, t.t_critical_95_one_sided ?? 0));
            const labelY = pad.top + 15 + idx * 16;
            return (
              <g key={t.id}>
                <path d={pathD} fill="none" stroke={color} strokeWidth={1.7} opacity={0.75} />
                <line x1={sx(critical)} y1={pad.top} x2={sx(critical)} y2={H - pad.bottom}
                  stroke={color} strokeWidth={1} strokeDasharray="5,4" opacity={0.55} />
                <line x1={sx(tStat)} y1={pad.top} x2={sx(tStat)} y2={H - pad.bottom}
                  stroke={color} strokeWidth={2.4} opacity={0.95} />
                <circle cx={sx(tStat)} cy={sy(studentTPdf(tStat, t.df))} r={4} fill={color} stroke="#111" strokeWidth={1}>
                  <title>{t.label}: t={fmt(t.t_stat ?? 0, 3)}, p={t.p_value_one_sided.toFixed(4)}, df={t.df}</title>
                </circle>
                <text x={W - pad.right - 255} y={labelY} fill={color} fontSize={10}>
                  {t.metric.toUpperCase()} {t.model_name.includes('Hybrid') ? 'Hybrid' : 'Thesis'}: t={fmt(t.t_stat ?? 0, 2)}, p={t.p_value_one_sided.toFixed(4)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    );
  }

  function ModelVsBookmakerTestsSection() {
    const tests = data?.model_vs_bookmaker_tests ?? [];
    if (tests.length === 0) return null;
    return (
      <div className="stat-tests-section">
        <h3>🧪 Testy statystyczne: model vs średni bukmacher</h3>
        <p className="stat-tests-subtitle">
          Każdy test jest sparowany po meczu. Dla każdego meczu liczymy różnicę:
          <strong> błąd średniego bukmachera − błąd modelu</strong>. Jeśli średnia różnica jest dodatnia,
          model ma mniejszy błąd. Test t-Studenta sprawdza jednostronnie, czy ta przewaga jest istotna.
        </p>
        <div className="stat-tests-grid">
          {tests.map(t => (
            <div key={t.id} className={`stat-test-card ${t.significant ? 'significant' : 'not-significant'}`}>
              <div className="stat-test-header">
                <span className="stat-test-title">{t.label}</span>
                <span className="stat-test-badge">{t.significant ? 'istotne' : 'nieistotne'}</span>
              </div>
              <div className="stat-test-metrics">
                <span>n: <strong>{t.n}</strong></span>
                <span>df: <strong>{t.df}</strong></span>
                <span>średnia Δ: <strong>{t.mean_diff.toFixed(4)}</strong></span>
                <span>t: <strong>{t.t_stat === null ? '—' : t.t_stat.toFixed(3)}</strong></span>
                <span>p: <strong>{t.p_value_one_sided.toFixed(4)}</strong></span>
                <span>t kryt.: <strong>{t.t_critical_95_one_sided?.toFixed(3) ?? '—'}</strong></span>
              </div>
              <p className="stat-test-note">
                {t.mean_diff > 0 ? 'Model ma niższy średni błąd w tej metryce.' : 'Średni bukmacher ma niższy średni błąd w tej metryce.'}
              </p>
            </div>
          ))}
        </div>
        <StudentTDistributionChart tests={tests} />
      </div>
    );
  }

  function MarketCloseComparisonSection() {
    const comp = data?.market_close_comparison;
    if (!comp || comp.n_matches === 0) return null;

    const statusColors = {
      model_better: '#00e676',
      model_on_market_level: '#fdd835',
      model_worse: '#ff5252',
      no_data: '#888',
      unknown: '#888',
    };

    const statusLabels = {
      model_better: 'Model lepszy od rynku',
      model_on_market_level: 'Model na poziomie rynku',
      model_worse: 'Model gorszy od rynku',
      no_data: 'Brak danych',
      unknown: 'Nieznany',
    };

    return (
      <div className="market-close-section">
        <div className="market-close-header">
          <h3>🏁 Porównanie na zamknięcie rynku (Market Close)</h3>
          <div className="market-close-status" style={{ backgroundColor: statusColors[comp.status] }}>
            {statusLabels[comp.status]}
          </div>
        </div>
        
        <p className="market-close-subtitle">
          To jest najbardziej rzetelne porównanie: model vs średnia z ostatnich kursów przed meczem (closing odds).
          Wszystkie metryki są liczone na <strong>identycznej próbie {comp.n_matches} meczów</strong>.
        </p>

        <div className="market-close-summary-grid">
          <div className="market-close-main-stats">
            <div className="mc-stat">
              <span className="mc-label">Liczba meczów</span>
              <span className="mc-value">{comp.n_matches}</span>
            </div>
            <div className="mc-stat">
              <span className="mc-label">Śr. bukmacherów / mecz</span>
              <span className="mc-value">{comp.avg_bookmakers_per_match?.toFixed(1)}</span>
            </div>
            <div className="mc-stat">
              <span className="mc-label">Δ LogLoss (Model vs Rynek)</span>
              <span className={`mc-value ${comp.model_delta_logloss_vs_market && comp.model_delta_logloss_vs_market < 0 ? 'better' : 'worse'}`}>
                {comp.model_delta_logloss_vs_market?.toFixed(4)}
              </span>
            </div>
          </div>

          <div className="market-close-table-wrapper">
            <table className="market-close-table">
              <thead>
                <tr>
                  <th>Poz.</th>
                  <th>Zawodnik / Model</th>
                  <th>LogLoss</th>
                  <th>AUC</th>
                  <th>Brier</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {comp.competitors.map(c => (
                  <tr key={c.name} className={c.name === 'MODEL' ? 'row-highlight-model' : c.name === 'HYBRID' ? 'row-highlight-hybrid' : ''}>
                    <td>{c.rank}.</td>
                    <td><strong>{c.display_name}</strong></td>
                    <td>{c.avg_logloss?.toFixed(4)}</td>
                    <td>{c.avg_auc?.toFixed(4)}</td>
                    <td>{c.avg_brier?.toFixed(4)}</td>
                    <td>{((c.accuracy ?? 0) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {comp.bookmakers.length > 0 && (
          <div className="market-close-bookmakers">
            <h4>Ranking bukmacherów (na tej samej próbie)</h4>
            <div className="mc-bookmakers-grid">
              {comp.bookmakers.map(bk => (
                <div key={bk.bookmaker_id} className="mc-bk-card">
                  <div className="mc-bk-rank">#{bk.rank}</div>
                  <div className="mc-bk-info">
                    <span className="mc-bk-name">{bk.bookmaker_name}</span>
                    <span className="mc-bk-metrics">
                      LL: {bk.avg_logloss?.toFixed(4)} | AUC: {bk.avg_auc?.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  /* ══════════════ RENDER ════════════════════════════════ */
  return (
    <div className="horizon-page">
      <h1>Odds Horizon Accuracy</h1>
      <p className="subtitle">
        Archiwalne zestawienie historycznych kursów i modeli. Wyniki nie są walidacją ani bieżącą predykcją C0.
      </p>

      {/* Controls */}
      <div className="horizon-controls">
        <label>
          Days back:
          <select value={daysBack} onChange={e => setDaysBack(+e.target.value)}>
            {[30, 60, 90, 180, 365].map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <label>
          Min matches per bin:
          <input type="number" min={1} max={100} value={minMatches}
            onChange={e => setMinMatches(Math.max(1, +e.target.value))} />
        </label>
        <button onClick={load} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {/* Stats row */}
      <div className="stats-row">
        <StatCard label="Finished matches" value={data.total_finished_matches} />
        <StatCard label="With odds" value={data.total_matches_with_odds} />
        <StatCard label="Odds snapshots" value={data.total_odds_processed} />
        <StatCard label="Bins shown" value={`${shownCount}/${allBins.length}`} />
        {skippedCount > 0 && <StatCard label="Bins skipped (<10 matches)" value={skippedCount} />}
      </div>

      {/* ─── Chart 1: Histogram ──────────────────────────── */}
      <HistogramChart />

      {/* ─── Chart 2: LogLoss ────────────────────────────── */}
      <MetricLineChart
        title="LogLoss by hours before match (lower is better)"
        metricKey="avg_logloss"
        color={COLORS.logloss}
        yLabel="Avg LogLoss"
        domain={(() => {
          const vals = shownBins.map(b => b.avg_logloss).filter(v => v !== null) as number[];
          data?.model_references?.forEach(r => { if (r.avg_logloss !== null) vals.push(r.avg_logloss); });
          if (vals.length === 0) return [0, 1] as [number, number];
          const min = Math.min(...vals);
          const max = Math.max(...vals);
          const pad = (max - min) * 0.15 || 0.05;
          return [Math.max(0, +(min - pad).toFixed(4)), +(max + pad).toFixed(4)] as [number, number];
        })()}
      />

      {/* ─── Chart 3: AUC ────────────────────────────────── */}
      <MetricLineChart
        title="AUC by hours before match (higher is better)"
        metricKey="avg_auc"
        color={COLORS.auc}
        yLabel="Avg AUC"
        domain={[0.4, 1]}
      />

      {/* ─── Model Reference Summary ──────────────────────── */}
      {data.model_references.length > 0 && (
        <div className="model-refs-section">
          <h3>Overall model prediction accuracy</h3>
          <p className="model-refs-subtitle">
            Horizontal dashed lines show each model&apos;s overall LogLoss / AUC across all finished matches.
          </p>
          <div className="model-refs-grid">
            {data.model_references.map(r => {
              const nameLower = r.model_name.toLowerCase();
              let color: string;
              let label: string;
              if (nameLower.includes('thesis') && nameLower.includes('hybrid')) {
                color = '#00d4ff';
                label = 'Hybrid-Thesis-Market';
              } else if (nameLower.includes('thesis') || nameLower.includes('sym-cal')) {
                color = '#ff6b9d';
                label = 'Sym-Cal LR-ElasticNet-W20-Binomial';
              } else if (nameLower.includes('hybrid')) {
                color = '#00e676';
                label = 'Hybrid-PlayerTeam-W20-Market';
              } else {
                color = '#fdd835';
                label = 'Operational-PlayerTeamRatings-W20';
              }
              return (
                <div key={r.model_name} className="model-ref-card">
                  <span className="model-dot" style={{ backgroundColor: color }} />
                  <span className="model-name">{label}</span>
                  <span className="model-metric">LogLoss: {fmt(r.avg_logloss ?? 0)}</span>
                  <span className="model-metric">AUC: {fmt(r.avg_auc ?? 0)}</span>
                  <span className="model-matches">({r.n_matches} matches)</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── Statistical Tests ───────────────────────────── */}
      <ModelVsBookmakerTestsSection />

      {/* ─── Market Close Comparison ─────────────────────── */}
      <MarketCloseComparisonSection />

      {/* ─── Bookmaker Results ─────────────────────────────── */}
      {data.bookmaker_bins.length > 0 && (
        <div className="bookmaker-results-section">
          <h3>📊 Per-bookmaker accuracy</h3>
          <p className="bookmaker-results-subtitle">
            Average LogLoss and AUC for each bookmaker across all horizon bins.
            Dashed lines on the charts above correspond to these bookmakers.
          </p>
          <div className="bookmaker-results-grid">
            {data.bookmaker_bins
              .map(bk => {
                // Compute weighted averages across bins
                const binsWithData = bk.bins.filter(b => b.avg_logloss !== null && b.match_count >= minMatches);
                if (binsWithData.length === 0) return null;
                const totalMatches = binsWithData.reduce((s, b) => s + b.match_count, 0);
                const avgLL = binsWithData.reduce((s, b) => s + (b.avg_logloss ?? 0) * b.match_count, 0) / totalMatches;
                const avgAUC = binsWithData.reduce((s, b) => s + (b.avg_auc ?? 0) * b.match_count, 0) / totalMatches;
                const totalSnapshots = binsWithData.reduce((s, b) => s + b.snapshot_count, 0);
                return { ...bk, avgLL, avgAUC, totalMatches, totalSnapshots, binsWithData: binsWithData.length };
              })
              .filter(Boolean)
              .sort((a, b) => (a?.avgLL ?? 999) - (b?.avgLL ?? 999))
              .map((bk, i) => {
                if (!bk) return null;
                // Color matching the chart lines
                const bkColors = [
                  '#8e99a4', '#a3b1bf', '#7c8c9a', '#9ea8b3', '#6b7b8a',
                  '#b0bec5', '#90a4ae', '#78909c', '#607d8b', '#546e7a',
                ];
                const color = bkColors[i % bkColors.length];
                return (
                  <div key={bk.bookmaker_id} className="bookmaker-result-card">
                    <div className="bookmaker-result-header">
                      <span className="bookmaker-result-rank">#{i + 1}</span>
                      <span className="bookmaker-result-dot" style={{ backgroundColor: color }} />
                      <span className="bookmaker-result-name">{bk.bookmaker_name}</span>
                    </div>
                    <div className="bookmaker-result-metrics">
                      <div className="bookmaker-result-metric">
                        <span className="metric-label">LogLoss</span>
                        <span className="metric-value">{bk.avgLL.toFixed(4)}</span>
                      </div>
                      <div className="bookmaker-result-metric">
                        <span className="metric-label">AUC</span>
                        <span className="metric-value">{bk.avgAUC.toFixed(4)}</span>
                      </div>
                      <div className="bookmaker-result-metric">
                        <span className="metric-label">Matches</span>
                        <span className="metric-value">{bk.totalMatches}</span>
                      </div>
                      <div className="bookmaker-result-metric">
                        <span className="metric-label">Snapshots</span>
                        <span className="metric-value">{bk.totalSnapshots}</span>
                      </div>
                    </div>
                    {/* Per-bin breakdown */}
                    <details className="bookmaker-bin-details">
                      <summary>{bk.binsWithData} bins with data</summary>
                      <table>
                        <thead>
                          <tr>
                            <th>Bin</th>
                            <th>Matches</th>
                            <th>LogLoss</th>
                            <th>AUC</th>
                          </tr>
                        </thead>
                        <tbody>
                          {bk.bins
                            .filter(b => b.avg_logloss !== null && b.match_count >= minMatches)
                            .map(b => (
                              <tr key={b.label}>
                                <td>{b.label}</td>
                                <td>{b.match_count}</td>
                                <td>{(b.avg_logloss ?? 0).toFixed(4)}</td>
                                <td>{(b.avg_auc ?? 0).toFixed(4)}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </details>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* ─── Legend ──────────────────────────────────────── */}
      <div className="legend-section">
        <h3>Legend</h3>
        <ul>
          <li><strong>LogLoss</strong> — lower = better calibrated predictions. Random guessing = 0.69 (for 50/50).</li>
          <li><strong>AUC</strong> — higher = better discrimination between winners/losers. 0.5 = random, 1.0 = perfect.</li>
          <li>Each point aggregates all pre-match odds snapshots across all bookmakers for finished matches.</li>
          <li>Dashed horizontal lines indicate overall model prediction accuracy (all horizons combined).</li>
          <li>Dashed subtle lines on charts show per-bookmaker accuracy (see table above for details).</li>
          <li>Bins with fewer than {minMatches} matches are hidden but visible in the histogram.</li>
        </ul>
      </div>

      {/* ─── Raw data table ──────────────────────────────── */}
      {allBins.length > 0 && (
        <div className="raw-data">
          <h3>Raw data</h3>
          <table>
            <thead>
              <tr>
                <th>Bin</th>
                <th>Hours</th>
                <th>Snapshots</th>
                <th>Matches</th>
                <th>Avg LogLoss</th>
                <th>Avg AUC</th>
                <th>Avg P(winner)</th>
                <th>Avg P(loser)</th>
              </tr>
            </thead>
            <tbody>
              {allBins.map(b => (
                <tr key={b.label} className={b.match_count < minMatches ? 'row-skipped' : ''}>
                  <td>{b.label}</td>
                  <td>{b.hours_start}–{b.hours_end ?? '∞'}</td>
                  <td>{b.snapshot_count}</td>
                  <td>{b.match_count}</td>
                  <td>{fmt(b.avg_logloss ?? 0)}</td>
                  <td>{fmt(b.avg_auc ?? 0)}</td>
                  <td>{fmt(b.avg_prob_winner ?? 0, 3)}</td>
                  <td>{fmt(b.avg_prob_loser ?? 0, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
