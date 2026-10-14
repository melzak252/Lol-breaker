import { useEffect, useRef, useState } from 'react';
import { fetchValidationReport } from '../api/client';
import type { ValidationReportResponse } from '../api/client';
import './ModelAnalysis.css';

function conditionalPercent(value: number | null, digits: number, signed = false): string {
  return value === null ? 'N/D — brak kwalifikujących zakładów' : `${signed && value > 0 ? '+' : ''}${value.toFixed(digits)}%`;
}

export default function ModelAnalysis() {
  const [report, setReport] = useState<ValidationReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const requestGeneration = useRef(0);
  const requestController = useRef<AbortController | null>(null);

  // Filters
  const [daysBack, setDaysBack] = useState<number>(3650);
  const [taxRate, setTaxRate] = useState<number>(0.12);
  const [minEv, setMinEv] = useState<number>(0.05);

  const loadData = async (d = daysBack, t = taxRate, ev = minEv) => {
    const generation = ++requestGeneration.current;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const res = await fetchValidationReport({
        daysBack: d,
        taxRate: t,
        minEv: ev,
      }, controller.signal);
      if (generation !== requestGeneration.current || controller.signal.aborted) return;
      if (res.error) {
        setError(res.error);
      } else {
        setReport(res);
      }
    } catch (err) {
      if (generation === requestGeneration.current && !controller.signal.aborted) {
        setError(err instanceof Error ? err.message : 'Błąd ładowania raportu walidacyjnego');
      }
    } finally {
      if (generation === requestGeneration.current && !controller.signal.aborted) setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const controllerSlot = requestController;
    return () => controllerSlot.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDaysChange = (d: number) => {
    setDaysBack(d);
    loadData(d, taxRate, minEv);
  };

  const handleTaxChange = (t: number) => {
    setTaxRate(t);
    loadData(daysBack, t, minEv);
  };

  const handleEvChange = (ev: number) => {
    setMinEv(ev);
    loadData(daysBack, taxRate, ev);
  };

  const s = report?.summary;
  const b = report?.betting_totals;

  return (
    <div className="report-container">
      {/* 1. Header & Filter Bar */}
      <header className="report-header">
        <div className="header-titles">
          <h1>📊 Archiwalne zestawienie walidacji modeli</h1>
          <span className="header-subtitle">
            Historyczne metryki Model vs Rynek, CLV i ROI. Ten raport nie jest walidacją ani bieżącą predykcją natywnego C0.
            {b && ` Niepewność: ${b.uncertainty_basis}; ${b.uncertainty_reason}${b.assumed_sigma_z === null ? '' : ` (założone sigma=${b.assumed_sigma_z})`}. Status: ${b.result_status}.`}
          </span>
        </div>

        <div className="report-filters">
          <div className="filter-item">
            <label>Zakres czasu:</label>
            <select
              value={daysBack}
              onChange={(e) => handleDaysChange(Number(e.target.value))}
              className="report-select"
            >
              <option value={3650}>Cała historia (850 m.)</option>
              <option value={365}>Ostatni rok (365 dni)</option>
              <option value={180}>Ostatnie 6 miesięcy (180 dni)</option>
              <option value={90}>Ostatnie 90 dni</option>
              <option value={30}>Ostatnie 30 dni</option>
            </select>
          </div>

          <div className="filter-item">
            <label>Podatek:</label>
            <select
              value={taxRate}
              onChange={(e) => handleTaxChange(Number(e.target.value))}
              className="report-select"
            >
              <option value={0.12}>12% (Polska standard)</option>
              <option value={0.0}>0% (Promocja Betclic / Gross)</option>
            </select>
          </div>

          <div className="filter-item">
            <label>Min EV:</label>
            <select
              value={minEv}
              onChange={(e) => handleEvChange(Number(e.target.value))}
              className="report-select"
            >
              <option value={0.0}>Wszystkie (0%)</option>
              <option value={0.03}>Min +3%</option>
              <option value={0.05}>Min +5% ★</option>
              <option value={0.10}>Min +10%</option>
            </select>
          </div>

          <button
            type="button"
            className="report-btn"
            onClick={() => loadData()}
            disabled={loading}
          >
            {loading ? '⏳ Ładowanie...' : '🔄 Odśwież'}
          </button>
        </div>
      </header>

      {error && (
        <div className="report-error-bar">
          ⚠️ <span>{error}</span>
        </div>
      )}

      {/* 2. Top Summary KPI Tiles */}
      {s && b && (
        <section className="kpi-strip">
          <div className="kpi-box">
            <span className="kpi-label">Model LogLoss</span>
            <strong className="kpi-val primary">{s.model_logloss.toFixed(4)}</strong>
            <span className="kpi-sub">
              Rynek: {s.market_logloss.toFixed(4)} (
              <span className={s.delta_logloss <= 0 ? 'good' : 'bad'}>
                {s.delta_logloss > 0 ? '+' : ''}
                {s.delta_logloss.toFixed(4)}
              </span>
              )
            </span>
          </div>

          <div className="kpi-box">
            <span className="kpi-label">Brier & Accuracy</span>
            <strong className="kpi-val">{s.model_brier.toFixed(4)}</strong>
            <span className="kpi-sub">
              Trafność: <strong>{s.model_accuracy_pct.toFixed(1)}%</strong> (Rynek: {s.market_accuracy_pct.toFixed(1)}%)
            </span>
          </div>

          <div className="kpi-box">
            <span className="kpi-label">Liczba typów & Win Rate</span>
            <strong className="kpi-val">
              {b.total_bets}{' '}
              <small style={{ fontSize: '0.85rem', fontWeight: 600 }}>({b.total_wins} wygranych)</small>
            </strong>
            <span className="kpi-sub">
              Skuteczność: <strong>{conditionalPercent(b.win_rate_pct, 1)}</strong> na {s.matches} meczach
            </span>
          </div>

          <div className="kpi-box">
            <span className="kpi-label">Zysk Netto / ROI ({taxRate === 0 ? '0% tax' : '12% tax'})</span>
            <strong className={`kpi-val ${b.total_pnl_pln >= 0 ? 'good' : 'bad'}`}>
              {b.total_pnl_pln > 0 ? '+' : ''}
              {b.total_pnl_pln.toLocaleString('pl-PL', { minimumFractionDigits: 2 })} zł
            </strong>
            <span className="kpi-sub">
              Yield / ROI: <strong className={b.roi_pct === null ? '' : b.roi_pct >= 0 ? 'good' : 'bad'}>{conditionalPercent(b.roi_pct, 2, true)}</strong>
            </span>
          </div>

          <div className="kpi-box">
            <span className="kpi-label">Średnie CLV (Ruch kursu)</span>
            <strong className={`kpi-val ${b.avg_clv_pct === null ? '' : b.avg_clv_pct >= 0 ? 'good' : 'bad'}`}>
              {conditionalPercent(b.avg_clv_pct, 1, true)}
            </strong>
            <span className="kpi-sub">
              Pobicie zamknięcia: <strong>{conditionalPercent(b.pos_clv_pct, 1)}</strong> zakładów
            </span>
          </div>
        </section>
      )}

      {/* 3. Table 1: Reality Check across Odds Brackets */}
      {report && (
        <section className="report-table-section">
          <div className="section-head">
            <h2>🛡️ 1. Reality Check: Expected vs Reality w Przedziałach Kursowych</h2>
            <span className="section-sub">
              Porównanie oczekiwanego zwrotu (Expected Net EV) z rzeczywistym zyskiem (Realized Net ROI) na czysto.
            </span>
          </div>
          <div className="table-responsive">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Przedział kursowy</th>
                  <th className="num">Liczba typów</th>
                  <th className="num">Trafione</th>
                  <th className="num">Skuteczność</th>
                  <th className="num">Śr. kurs</th>
                  <th className="num">Oczekiwane EV Netto</th>
                  <th className="num">Rzeczywisty ROI Netto</th>
                  <th className="num">Zysk Netto (100 zł/bet)</th>
                  <th className="num">Mediana CLV</th>
                </tr>
              </thead>
              <tbody>
                {report.odds_brackets.map((row) => (
                  <tr key={row.label}>
                    <td>
                      <strong>{row.label}</strong>
                    </td>
                    <td className="num">{row.bets}</td>
                    <td className="num">{row.wins}</td>
                    <td className="num">{row.win_rate_pct.toFixed(1)}%</td>
                    <td className="num">{row.avg_odds.toFixed(2)}</td>
                    <td className="num">{row.expected_net_roi_pct > 0 ? '+' : ''}{row.expected_net_roi_pct.toFixed(1)}%</td>
                    <td className={`num ${row.realized_net_roi_pct >= 0 ? 'good' : 'bad'}`}>
                      <strong>{row.realized_net_roi_pct > 0 ? '+' : ''}{row.realized_net_roi_pct.toFixed(1)}%</strong>
                    </td>
                    <td className={`num ${row.pnl_pln >= 0 ? 'good' : 'bad'}`}>
                      <strong>{row.pnl_pln > 0 ? '+' : ''}{row.pnl_pln.toFixed(2)} zł</strong>
                    </td>
                    <td className={`num ${row.clv_pct >= 0 ? 'good' : 'bad'}`}>
                      {row.clv_pct > 0 ? '+' : ''}{row.clv_pct.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 4. Table 2: Bookmakers Performance Breakdown */}
      {report && (
        <section className="report-table-section">
          <div className="section-head">
            <h2>🏢 2. Wydajność według Bukmacherów (Kto daje największy zysk)</h2>
            <span className="section-sub">
              Identyfikacja podmiotów najwolniej korygujących linie, gdzie model generuje najwyższy realny yield.
            </span>
          </div>
          <div className="table-responsive">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Bukmacher</th>
                  <th className="num">Liczba typów</th>
                  <th className="num">Trafione</th>
                  <th className="num">Skuteczność</th>
                  <th className="num">Średni kurs</th>
                  <th className="num">Zysk Netto (zł)</th>
                  <th className="num">Rzeczywisty ROI</th>
                  <th className="num">Średni CLV</th>
                </tr>
              </thead>
              <tbody>
                {report.bookmakers.map((row) => (
                  <tr key={row.bookmaker}>
                    <td>
                      <span className="bookmaker-badge">{row.bookmaker}</span>
                    </td>
                    <td className="num">{row.bets}</td>
                    <td className="num">{row.wins}</td>
                    <td className="num">{row.win_rate_pct.toFixed(1)}%</td>
                    <td className="num">{row.avg_odds.toFixed(2)}</td>
                    <td className={`num ${row.pnl_pln >= 0 ? 'good' : 'bad'}`}>
                      <strong>{row.pnl_pln > 0 ? '+' : ''}{row.pnl_pln.toFixed(2)} zł</strong>
                    </td>
                    <td className={`num ${row.roi_pct >= 0 ? 'good' : 'bad'}`}>
                      <strong>{row.roi_pct > 0 ? '+' : ''}{row.roi_pct.toFixed(1)}%</strong>
                    </td>
                    <td className={`num ${row.clv_pct >= 0 ? 'good' : 'bad'}`}>
                      {row.clv_pct > 0 ? '+' : ''}{row.clv_pct.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 5. Table 3: CLV and Accuracy across Timing Horizons */}
      {report && (
        <section className="report-table-section">
          <div className="section-head">
            <h2>⏱️ 3. Dokładność i CLV według Horyzontu Czasowego</h2>
            <span className="section-sub">
              Porównanie błędu LogLoss modelu z rynkiem od wczesnego otwarcia (&gt;48h) do kursu zamknięcia (&lt;2h).
            </span>
          </div>
          <div className="table-responsive">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Horyzont czasowy</th>
                  <th className="num">Meczów</th>
                  <th className="num">Notowań</th>
                  <th className="num">Model LogLoss</th>
                  <th className="num">Rynek LogLoss</th>
                  <th className="num">Δ LogLoss</th>
                  <th className="num">Średni CLV</th>
                  <th className="num">Pobicie zamknięcia</th>
                </tr>
              </thead>
              <tbody>
                {report.horizons.map((row) => (
                  <tr key={row.label}>
                    <td>
                      <strong>{row.label}</strong>
                    </td>
                    <td className="num">{row.matches}</td>
                    <td className="num">{row.quotes}</td>
                    <td className="num">{row.model_logloss.toFixed(4)}</td>
                    <td className="num">{row.market_logloss.toFixed(4)}</td>
                    <td className={`num ${row.delta_logloss <= 0 ? 'good' : 'bad'}`}>
                      <strong>{row.delta_logloss > 0 ? '+' : ''}{row.delta_logloss.toFixed(4)}</strong>
                    </td>
                    <td className={`num ${row.avg_clv_pct >= 0 ? 'good' : 'bad'}`}>
                      {row.avg_clv_pct > 0 ? '+' : ''}{row.avg_clv_pct.toFixed(1)}%
                    </td>
                    <td className="num">{row.pos_clv_pct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 6. Table 4: Calibration Deciles */}
      {report && (
        <section className="report-table-section">
          <div className="section-head">
            <h2>⚖️ 4. Kalibracja Prawdopodobieństwa (Decyle 0% – 100%)</h2>
            <span className="section-sub">
              Weryfikacja rzetelności prognoz: deklarowane prawdopodobieństwo modelu vs rzeczywisty odsetek wygranych.
            </span>
          </div>
          <div className="table-responsive">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Przedział prawdopodobieństwa</th>
                  <th className="num">Liczba meczów</th>
                  <th className="num">Śr. prognoza modelu</th>
                  <th className="num">Rzeczywisty Win Rate</th>
                  <th className="num">Różnica kalibracji (Gap)</th>
                </tr>
              </thead>
              <tbody>
                {report.calibration_deciles.map((row) => (
                  <tr key={row.label}>
                    <td>
                      <strong>{row.label}</strong>
                    </td>
                    <td className="num">{row.count}</td>
                    <td className="num">{row.avg_predicted_pct.toFixed(1)}%</td>
                    <td className="num">{row.observed_rate_pct.toFixed(1)}%</td>
                    <td className={`num ${Math.abs(row.gap_pp) <= 5 ? 'good' : 'bad'}`}>
                      {row.gap_pp > 0 ? '+' : ''}{row.gap_pp.toFixed(1)} p.p.
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
