import { useState, useEffect, useCallback } from 'react';
import { HorizonBootstrapResponse, HorizonBootstrapBin } from '../types';
import { fetchHorizonBootstrap } from '../api/client';
import './BootstrapPanel.css';

/* ─── Color palette ─────────────────────────────────────── */
const COLORS = {
  thesis_model:  '#ff6b9d',
  thesis_bench:  '#ff8aad',
  hybrid_model:  '#00d4ff',
  hybrid_bench:  '#33ddff',
  sig_bg:        '#1a3a2a',
  nonsig_bg:     '#3a1a1a',
  ci_bar:        'rgba(255,255,255,0.3)',
  ci_cap:        'rgba(255,255,255,0.5)',
  axis:          '#888',
  grid:          '#2a2a4a',
  label:         '#aaa',
  title:         '#fff',
  zero_line:     '#666',
};

/* ─── Helpers ───────────────────────────────────────────── */
function fmt(v: number | null, d = 3): string {
  if (v === null || v === undefined) return '—';
  return v.toFixed(d);
}

function fmtP(v: number | null): string {
  if (v === null || v === undefined) return '—';
  if (v < 0.0001) return '<0.0001';
  return v.toFixed(4);
}

/* ─── Stat Card ─────────────────────────────────────────── */
function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="bootstrap-stat-card" style={color ? { borderLeftColor: color } : {}}>
      <span className="bootstrap-stat-label">{label}</span>
      <span className="bootstrap-stat-value">{value}</span>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   CI Chart
   ════════════════════════════════════════════════════════════ */
function BootstrapCIChart({ bins, modelLabel, color }: {
  bins: HorizonBootstrapBin[];
  modelLabel: string;
  color: string;
}) {
  const modelBins = bins.filter(b => b.model_label === modelLabel);
  if (modelBins.length === 0) return <p className="bootstrap-no-data">No data for {modelLabel}.</p>;

  const CHART_W = 800;
  const CHART_H = 300;
  const PAD = { top: 30, right: 40, bottom: 60, left: 70 };

  // Find domain
  const allVals = modelBins.flatMap(b => [b.ci_low, b.ci_high, b.observed_difference, 0]).filter(v => v !== null) as number[];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const pad = Math.max(Math.abs(maxVal - minVal) * 0.15, 0.005);
  const domain: [number, number] = [minVal - pad, maxVal + pad];

  const yScale = (v: number) => PAD.top + (CHART_H - PAD.top - PAD.bottom) - ((v - domain[0]) / (domain[1] - domain[0])) * (CHART_H - PAD.top - PAD.bottom);
  const xPos = (i: number) => PAD.left + (i + 0.5) * ((CHART_W - PAD.left - PAD.right) / modelBins.length);

  // Y ticks
  const raw = domain[1] - domain[0];
  const rough = raw / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const nice = [1, 2, 2.5, 5, 10].find(v => v * mag >= rough)! * mag;
  const start = Math.floor(domain[0] / nice) * nice;
  const ticks: number[] = [];
  for (let v = start; v <= domain[1] + nice * 0.5; v = +(v + nice).toFixed(6)) ticks.push(v);

  const zeroY = yScale(0);

  return (
    <div className="bootstrap-ci-chart-wrapper">
      <h3 style={{ color }}>Δ LogLoss — {modelLabel}</h3>
      <p className="bootstrap-ci-subtitle">
        Każdy punkt to średnia różnica LogLoss (benchmark − model) w danym horyzoncie.
        Dodatnia wartość = model lepszy. Słupek to 95% CI z 10,000 bootstrap resampli
        miesięcznych bloków.
      </p>
      <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="bootstrap-ci-chart">
        {/* Grid */}
        {ticks.map(v => {
          if (Math.abs(v) < nice * 0.01) return null;
          return (
            <g key={v}>
              <line x1={PAD.left} y1={yScale(v)} x2={CHART_W - PAD.right} y2={yScale(v)}
                stroke={COLORS.grid} strokeWidth={1} />
              <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fill={COLORS.axis} fontSize={11}>
                {fmt(v, 4)}
              </text>
            </g>
          );
        })}

        {/* Zero line */}
        <line x1={PAD.left} y1={zeroY} x2={CHART_W - PAD.right} y2={zeroY}
          stroke={COLORS.zero_line} strokeWidth={1.5} strokeDasharray="6,4" />
        <text x={CHART_W - PAD.right + 2} y={zeroY - 3} fill={COLORS.zero_line} fontSize={10}>0</text>

        {/* Data */}
        {modelBins.map((b, i) => {
          const cx = xPos(i);
          const ciLow = b.ci_low ?? 0;
          const ciHigh = b.ci_high ?? 0;
          const obsDiff = b.observed_difference ?? 0;
          const yLow = yScale(ciLow);
          const yHigh = yScale(ciHigh);
          const yObs = yScale(obsDiff);
          const isSig = b.significant_05;

          return (
            <g key={`${b.model_label}-${b.label}`}>
              {/* CI bar */}
              <line x1={cx} y1={yLow} x2={cx} y2={yHigh}
                stroke={isSig ? COLORS.sig_bg : COLORS.nonsig_bg}
                strokeWidth={12} opacity={0.6} rx={3} />
              {/* CI cap lines */}
              <line x1={cx - 8} y1={yLow} x2={cx + 8} y2={yLow}
                stroke={COLORS.ci_cap} strokeWidth={1.5} />
              <line x1={cx - 8} y1={yHigh} x2={cx + 8} y2={yHigh}
                stroke={COLORS.ci_cap} strokeWidth={1.5} />
              {/* Point estimate */}
              <circle cx={cx} cy={yObs} r={5} fill={color} stroke="#1a1a2e" strokeWidth={1.5}>
                <title>
                  {b.label}: ΔLogLoss={fmt(obsDiff, 4)} [{fmt(ciLow, 4)}, {fmt(ciHigh, 4)}]
                  p={fmtP(b.p_one_sided)} {isSig ? '✓' : '✗'}
                </title>
              </circle>
              {/* Label */}
              <text x={cx} y={yScale(domain[0]) - 2} textAnchor="middle"
                fill={COLORS.label} fontSize={10}>
                {b.label}
              </text>
              {/* Value above point */}
              <text x={cx} y={yObs - 9} textAnchor="middle" fill={color} fontSize={9}
                fontWeight="bold">
                {fmt(obsDiff, 4)}
              </text>
              {/* Significance asterisk */}
              {isSig && (
                <text x={cx + 10} y={yObs + 3} fill="#00e676" fontSize={13} fontWeight="bold">*</text>
              )}
            </g>
          );
        })}

        {/* Y axis */}
        <text x={14} y={CHART_H / 2} textAnchor="middle" fill={COLORS.label} fontSize={11}
          transform={`rotate(-90, 14, ${CHART_H / 2})`}>
          Δ LogLoss (benchmark − model)
        </text>
      </svg>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Component
   ════════════════════════════════════════════════════════════ */
export default function BootstrapPanel() {
  const [data, setData] = useState<HorizonBootstrapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [triggering, setTriggering] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchHorizonBootstrap();
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load bootstrap results');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleRefresh = async () => {
    setTriggering(true);
    try {
      // Trigger the scheduler task
      const resp = await fetch('/api/scheduler/trigger/horizon_bootstrap', { method: 'POST' });
      const result = await resp.json();
      alert(`Zadanie bootstrap uruchomione: ${result.message}. Odśwież stronę za ~30s.`);
    } catch (e: unknown) {
      alert(`Błąd: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTriggering(false);
    }
  };

  /* ─── Loading / Error ────────────────────────────────── */
  if (loading) {
    return (
      <div className="bootstrap-page">
        <h1>Bootstrap Analysis</h1>
        <p className="bootstrap-loading-text">Loading bootstrap results…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bootstrap-page">
        <h1>Bootstrap Analysis</h1>
        <div className="bootstrap-error-box">
          <p className="bootstrap-error-text">{error}</p>
          <p className="bootstrap-error-hint">
            Uruchom najpierw zadanie scheduler: <strong>horizon_bootstrap</strong>.
            Możesz to zrobić ręcznie przez zakładkę System → Scheduler → "Horizon Bootstrap Analysis" → Trigger.
          </p>
          <button onClick={load} className="bootstrap-retry-btn">Retry</button>
        </div>
      </div>
    );
  }

  if (!data || data.bins.length === 0) {
    return (
      <div className="bootstrap-page">
        <h1>Bootstrap Analysis</h1>
        <div className="bootstrap-error-box">
          <p className="bootstrap-error-text">Brak wyników bootstrapu.</p>
          <p className="bootstrap-error-hint">
            Uruchom zadanie scheduler: <strong>horizon_bootstrap</strong> przez System → Scheduler → Trigger.
          </p>
          <button onClick={handleRefresh} disabled={triggering} className="bootstrap-retry-btn">
            {triggering ? 'Uruchamianie…' : 'Uruchom bootstrap'}
          </button>
        </div>
      </div>
    );
  }

  /* ─── Data ────────────────────────────────────────────── */
  const allBins = data.bins;

  // Group by model label for stats
  const modelLabels = [...new Set(allBins.map(b => b.model_label))];

  /* ══════════════ RENDER ════════════════════════════════ */
  return (
    <div className="bootstrap-page">
      <h1>Horizon Bootstrap Analysis</h1>
      <p className="bootstrap-subtitle">
        Monthly block bootstrap porównujący modele (Thesis i Hybrid) vs średni bukmacher
        w 6 horyzontach czasowych. 10,000 resampli bloków miesięcznych.
        Dodatnia ΔLogLoss = model lepszy od bukmachera.
      </p>

      {/* ─── Controls ──────────────────────────────────────── */}
      <div className="bootstrap-controls">
        <button onClick={handleRefresh} disabled={triggering} className="bootstrap-refresh-btn">
          {triggering ? 'Uruchamianie…' : '🔄 Uruchom bootstrap (scheduler)'}
        </button>
        <button onClick={load} className="bootstrap-refresh-btn bootstrap-refresh-btn-secondary">
          ⟳ Odśwież wyniki
        </button>
      </div>

      {/* ─── Stats Cards ────────────────────────────────────── */}
      <div className="bootstrap-stats-row">
        <StatCard label="Ostatnia aktualizacja" value={data.last_updated ? new Date(data.last_updated).toLocaleString('pl-PL') : '—'} />
        <StatCard label="Horyzonty (bins)" value={allBins.length} color="#4fc3f7" />
        <StatCard label="Modele" value={modelLabels.length} color="#ff6b9d" />
        <StatCard label="Mecze upcoming" value={data.match_stats.upcoming ?? '?'} color="#ff9800" />
        <StatCard label="Mecze finished" value={data.match_stats.finished ?? '?'} color="#00e676" />
        <StatCard label="Mecze expired" value={data.match_stats.expired ?? '?'} color="#888" />
      </div>

      {/* ─── CI Charts ────────────────────────────────────── */}
      {modelLabels.map(label => {
        const binCount = allBins.filter(b => b.model_label === label).length;
        if (binCount === 0) return null;
        const color = label.toLowerCase().includes('thesis') ? COLORS.thesis_model : COLORS.hybrid_model;
        return (
          <div key={label} className="bootstrap-model-section">
            <BootstrapCIChart bins={allBins} modelLabel={label} color={color} />
          </div>
        );
      })}

      {/* ─── Summary Table ──────────────────────────────────── */}
      <div className="bootstrap-table-section">
        <h3>📊 Szczegółowe wyniki</h3>
        <div className="bootstrap-table-wrapper">
          <table className="bootstrap-results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Horyzont</th>
                <th>Godziny</th>
                <th>N (snapshots)</th>
                <th>N (mecze)</th>
                <th>N (bloki)</th>
                <th>Model LogLoss</th>
                <th>Benchmark LogLoss</th>
                <th>Δ LogLoss</th>
                <th>95% CI (dół)</th>
                <th>95% CI (góra)</th>
                <th>p (jednostr.)</th>
                <th>Istotne (α=0.05)</th>
              </tr>
            </thead>
            <tbody>
              {allBins.map((b, i) => (
                <tr key={i}
                  className={b.significant_05 ? 'bootstrap-row-sig' : 'bootstrap-row-nonsig'}>
                  <td><strong>{b.model_label}</strong></td>
                  <td>{b.label}</td>
                  <td>{fmt(b.hours_start)}–{b.hours_end !== null ? fmt(b.hours_end) : '∞'}</td>
                  <td>{b.sample_size !== null ? b.sample_size.toLocaleString() : '—'}</td>
                  <td>—</td>
                  <td>{fmt(b.n_blocks, 0)}</td>
                  <td>{fmt(b.model_logloss, 4)}</td>
                  <td>{fmt(b.benchmark_logloss, 4)}</td>
                  <td style={{ color: (b.observed_difference ?? 0) > 0 ? '#00e676' : '#ff5252' }}>
                    {fmt(b.observed_difference, 4)}
                  </td>
                  <td>{fmt(b.ci_low, 4)}</td>
                  <td>{fmt(b.ci_high, 4)}</td>
                  <td>{fmtP(b.p_one_sided)}</td>
                  <td>
                    {b.significant_05
                      ? <span className="bootstrap-sig-yes">✓ Istotne</span>
                      : <span className="bootstrap-sig-no">✗ Nieistotne</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ─── Monthly Data ──────────────────────────────────── */}
      {data.monthly.length > 0 && (
        <div className="bootstrap-monthly-section">
          <h3>📅 Obserwowane różnice miesięczne</h3>
          <p className="bootstrap-monthly-subtitle">
            Średnie różnice LogLoss (benchmark − model) dla każdego miesiąca.
            To są surowe dane przed bootstrapem — każdy miesiąc to jeden blok.
          </p>
          <div className="bootstrap-table-wrapper">
            <table className="bootstrap-results-table">
              <thead>
                <tr>
                  <th>Miesiąc</th>
                  <th>Model</th>
                  <th>Horyzont</th>
                  <th>N snapshots</th>
                  <th>N mecze</th>
                  <th>Model LogLoss</th>
                  <th>Bookmaker LogLoss</th>
                  <th>Śr. różnica</th>
                </tr>
              </thead>
              <tbody>
                {data.monthly.map((m, i) => (
                  <tr key={i}>
                    <td><strong>{m.month}</strong></td>
                    <td>{m.model_label}</td>
                    <td>{m.horizon_bin}</td>
                    <td>{m.n_snapshots?.toLocaleString() ?? '—'}</td>
                    <td>{fmt(m.n_matches, 0)}</td>
                    <td>{fmt(m.model_logloss, 4)}</td>
                    <td>{fmt(m.bookmaker_logloss, 4)}</td>
                    <td style={{ color: (m.mean_difference ?? 0) > 0 ? '#00e676' : '#ff5252' }}>
                      {fmt(m.mean_difference, 4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ─── Plot (if available) ────────────────────────────── */}
      {data.plot_available && (
        <div className="bootstrap-plot-section">
          <h3>📈 Wykres CI</h3>
          <img src="/api/bootstrap/horizon/plot" alt="Horizon Bootstrap CI"
            className="bootstrap-plot-img"
            onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
        </div>
      )}

      {/* ─── Legend ──────────────────────────────────────────── */}
      <div className="bootstrap-legend">
        <h3>Legenda</h3>
        <ul>
          <li><strong>Δ LogLoss</strong> = LogLoss(benchmark) − LogLoss(model). Dodatnia = model ma niższy błąd.</li>
          <li><strong>Monthly block bootstrap</strong> — resamplujemy całe miesiące (nie pojedyncze mecze), by zachować autokorelację w obrębie miesiąca.</li>
          <li><strong>95% CI</strong> — przedział ufności z 2.5 i 97.5 percentyla rozkładu bootstrap.</li>
          <li><strong>p (jednostronny)</strong> — proporcja resampli gdzie Δ ≤ 0 (model nie jest lepszy).</li>
          <li><strong>* (asterysk)</strong> — istotne statystycznie na poziomie α=0.05.</li>
          <li>Tylko 2 bloki miesięczne (Maj–Czerwiec 2026) → szerokie przedziały ufności. Wyniki będą bardziej wiarygodne z większą liczbą miesięcy.</li>
        </ul>
      </div>
    </div>
  );
}
