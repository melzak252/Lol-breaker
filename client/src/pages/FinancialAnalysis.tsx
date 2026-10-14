import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchFinancialAnalysis } from '../api/client'
import type { FinancialAnalysisResponse, FinancialBucket } from '../types'
import './FinancialAnalysis.css'

const DAYS = [30, 60, 90, 180]
const MODEL = { name: 'Hybrid-Operational-Market', version: 'c0-native-2026-w32-e12-v1-a0.50-t1.00' }

const pct = (value: number | null | undefined, digits = 1) => value == null ? '—' : `${value * 100 >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`
const money = (value: number | null | undefined) => value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)} PLN`

const uncertaintyLabel: Record<FinancialAnalysisResponse['uncertainty_basis'], string> = {
  mean: 'Średnia modelu — bez zastosowania granic niepewności',
  measured_bounds: 'Zapisane zmierzone granice Venn–Abers',
  assumed_research_haircut: 'Jawnie założony haircut badawczy — nie jest zmierzonym przedziałem',
  missing_bounds: 'Brak wymaganych granic niepewności — wykluczenie, nie ujemny EV',
}

function MiniCurve({ data }: { data: FinancialAnalysisResponse['bankroll_curve'] }) {
  if (data.length < 2) return <div className="financial-empty-chart">Brak kwalifikujących się zakładów. Krzywa nie stanowi oceny kalibracji ani rentowności.</div>
  const width = 760, height = 210, pad = 12
  const values = data.map(point => point.bankroll)
  const min = Math.min(...values), max = Math.max(...values)
  const span = max - min || 1
  const points = data.map((point, index) => {
    const x = pad + (index / (data.length - 1)) * (width - pad * 2)
    const y = height - pad - ((point.bankroll - min) / span) * (height - pad * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const positive = data[data.length - 1].bankroll >= data[0].bankroll
  return <div className="financial-chart-wrap">
    <div className="financial-chart-scale"><span>{min.toFixed(0)} PLN</span><span>{max.toFixed(0)} PLN</span></div>
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Krzywa bankrolla">
      <defs><linearGradient id="bankroll-fill" x1="0" y1="0" x2="0" y2="1"><stop stopColor={positive ? '#35d39e' : '#ff6b81'} stopOpacity=".3"/><stop offset="1" stopColor={positive ? '#35d39e' : '#ff6b81'} stopOpacity="0"/></linearGradient></defs>
      <polyline points={`${pad},${height - pad} ${points} ${width - pad},${height - pad}`} fill="url(#bankroll-fill)" stroke="none" />
      <polyline points={points} fill="none" stroke={positive ? '#35d39e' : '#ff6b81'} strokeWidth="3" vectorEffect="non-scaling-stroke" />
    </svg>
  </div>
}

function BucketTable({ title, subtitle, rows, limit }: { title: string; subtitle: string; rows: FinancialBucket[]; limit?: number }) {
  const shown = limit ? rows.slice(0, limit) : rows
  return <section className="financial-section">
    <div className="financial-section-title"><div><h2>{title}</h2><p>{subtitle}</p></div></div>
    <div className="financial-table-scroll"><table className="financial-table"><thead><tr><th>Segment</th><th>Mecze</th><th>Zakłady</th><th>ROI</th><th>P&L</th><th>CLV</th><th>Skut.</th></tr></thead><tbody>
      {shown.map(row => <tr key={row.key} className={(row.roi ?? 0) > 0 ? 'row-positive' : ''}><td><strong>{row.label}</strong><small>EV śr. {pct(row.avg_ev)}</small></td><td>{row.matches}</td><td>{row.bets}</td><td className={(row.roi ?? 0) >= 0 ? 'positive' : 'negative'}>{pct(row.roi)}</td><td className={(row.profit ?? 0) >= 0 ? 'positive' : 'negative'}>{money(row.profit)}</td><td className={(row.avg_clv_odds_pct ?? 0) >= 0 ? 'positive' : 'negative'}>{pct(row.avg_clv_odds_pct)}</td><td>{pct(row.hit_rate, 0)}</td></tr>)}
      {!shown.length && <tr><td colSpan={7} className="table-empty">Brak danych dla wybranego wariantu.</td></tr>}
    </tbody></table></div>
  </section>
}

export default function FinancialAnalysis() {
  const [daysBack, setDaysBack] = useState(90)
  const [oddsMode, setOddsMode] = useState('mid')
  const [dataScope, setDataScope] = useState('historical')
  const [stakingMode, setStakingMode] = useState('kelly')
  const [minEv, setMinEv] = useState(0.03)
  const [bankroll, setBankroll] = useState(1000)
  const [fixedStake, setFixedStake] = useState(10)
  const requestKey = JSON.stringify([daysBack, oddsMode, stakingMode, minEv, bankroll, fixedStake, dataScope, MODEL.name, MODEL.version])
  const [requestState, setRequestState] = useState<{
    key: string
    data: FinancialAnalysisResponse | null
    loading: boolean
    error: string | null
  } | null>(null)
  const requestGeneration = useRef(0)
  const requestController = useRef<AbortController | null>(null)
  const currentRequest = requestState?.key === requestKey ? requestState : null
  const data = currentRequest?.data ?? null
  const loading = currentRequest?.loading ?? true
  const error = currentRequest?.error ?? null

  const load = useCallback(() => {
    const generation = ++requestGeneration.current
    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller
    setRequestState({ key: requestKey, data: null, loading: true, error: null })
    fetchFinancialAnalysis({ daysBack, oddsMode, stakingMode, minEv, initialBankroll: bankroll, fixedStake, modelName: MODEL.name, modelVersion: MODEL.version, dataScope }, controller.signal)
      .then(response => {
        if (generation === requestGeneration.current && !controller.signal.aborted) {
          setRequestState({ key: requestKey, data: response, loading: false, error: null })
        }
      })
      .catch((err: unknown) => {
        if (generation === requestGeneration.current && !controller.signal.aborted) {
          setRequestState({ key: requestKey, data: null, loading: false, error: err instanceof Error ? err.message : String(err) })
        }
      })
  }, [daysBack, oddsMode, stakingMode, minEv, bankroll, fixedStake, dataScope, requestKey])
  useEffect(() => {
    load()
    return () => {
      requestController.current?.abort()
    }
  }, [load])
  const bestHorizon = useMemo(() => data?.horizon_buckets.filter(row => row.matches >= 20).sort((a, b) => (b.roi ?? -999) - (a.roi ?? -999))[0], [data])

  return <div className="financial-page">
    <header className="financial-hero"><div><span className="eyebrow">SYMULACJA HISTORYCZNYCH KURSÓW</span><h1>Analiza finansowa</h1><p>Retrospektywne przeliczenie zapisanych prognoz i kursów. Wynik symulacji nie oznacza wydania ani realizacji zakładów, potwierdzonej kalibracji lub gwarancji zysku.</p></div><Link to="/results" className="financial-link">Otwórz pojedyncze wyniki →</Link></header>

    <section className="financial-controls"><div className="control-group"><label>Okres</label><div className="financial-pills">{DAYS.map(days => <button className={daysBack === days ? 'active' : ''} onClick={() => setDaysBack(days)} key={days}>{days}d</button>)}</div></div>
      <div className="control-group"><label>Kurs wejścia</label><div className="financial-pills">{[['open','Open'],['mid','Mid'],['close','Close']].map(([value,label]) => <button className={oddsMode === value ? 'active' : ''} onClick={() => setOddsMode(value)} key={value}>{label}</button>)}</div></div>
      <div className="control-group"><label>Staking</label><div className="financial-pills"><button className={stakingMode === 'kelly' ? 'active' : ''} onClick={() => setStakingMode('kelly')}>¼ Kelly</button><button className={stakingMode === 'fixed' ? 'active' : ''} onClick={() => setStakingMode('fixed')}>Fixed</button></div></div>
      <div className="control-group"><label>Źródło danych</label><div className="financial-pills"><button className={dataScope === 'live' ? 'active' : ''} onClick={() => setDataScope('live')}>Live</button><button className={dataScope === 'historical' ? 'active' : ''} onClick={() => { setDataScope('historical'); setOddsMode('mid') }}>Historia · mid</button></div></div>
      <label className="financial-number">Min. EV<input type="number" min="0" max="0.5" step="0.01" value={minEv} onChange={event => setMinEv(Number(event.target.value))} /><span>{pct(minEv, 0)}</span></label>
      <label className="financial-number">Bankroll<input type="number" min="100" step="100" value={bankroll} onChange={event => setBankroll(Number(event.target.value))} /><span>PLN</span></label>
      {stakingMode === 'fixed' && <label className="financial-number">Stawka<input type="number" min="1" step="1" value={fixedStake} onChange={event => setFixedStake(Number(event.target.value))} /><span>PLN</span></label>}
      <button className="financial-run" onClick={load} disabled={loading}>{loading ? 'Przeliczanie…' : 'Przelicz symulację'}</button>
    </section>

    {error && <div className="financial-error">Nie udało się pobrać analizy: {error}</div>}
    {loading && !data && <div className="financial-loading">Tworzę audytowalny ledger zakładów…</div>}
    {data && <>
      <div className="financial-method research-warning"><strong>Symulacja historycznych kursów ({data.evaluation_kind}):</strong> {data.methodology}</div>
      <div className="financial-method">Dokładna zapisana prognoza · {data.model_name} · {data.model_version}</div>
      <div className={`financial-method ${data.uncertainty_basis === 'missing_bounds' || data.uncertainty_basis === 'assumed_research_haircut' ? 'research-warning' : ''}`}><strong>Podstawa niepewności:</strong> {uncertaintyLabel[data.uncertainty_basis]} · {data.uncertainty_reason}</div>
      <div className="financial-method"><strong>Historyczne diagnostyki:</strong> {data.historical_diagnostics_status === 'unknown' ? 'Nieznane — brak zapisanych diagnostyk dla części lub całej próby.' : 'Diagnostyki zapisane; sama obecność nie potwierdza wydania lub realizacji zakładu.'}</div>
      {data.result_status === 'no_eligible_bets' && <div className="financial-method research-warning"><strong>Brak kwalifikujących się zakładów.</strong> Saldo i P&amp;L bez działania pozostają prawidłowe; ROI, skuteczność i inne miary warunkowe są niedostępne. To nie jest sukces kalibracji ani ocena rentowności.</div>}
      <section className="financial-kpis"><article><span>Symulowany wynik netto</span><strong className={data.result_status === 'no_eligible_bets' ? '' : data.total_profit >= 0 ? 'positive' : 'negative'}>{money(data.total_profit)}</strong><small>ROI {pct(data.roi)}</small></article><article><span>Bankroll końcowy</span><strong>{data.final_bankroll.toFixed(2)} PLN</strong><small>start: {data.initial_bankroll.toFixed(0)} PLN</small></article><article><span>Próba</span><strong>{data.total_matches} mecz.</strong><small>{data.total_bets} zakładów · skut. {pct(data.hit_rate, 0)}</small></article><article><span>CLV</span><strong className={data.avg_clv_odds_pct == null ? '' : data.avg_clv_odds_pct >= 0 ? 'positive' : 'negative'}>{pct(data.avg_clv_odds_pct)}</strong><small>dodatni CLV: {pct(data.positive_clv_rate, 0)}</small></article><article><span>Max drawdown</span><strong className={data.max_drawdown_pct == null ? '' : 'negative'}>{data.max_drawdown_pct == null ? '—' : `-${pct(data.max_drawdown_pct, 1).replace('+','')}`}</strong><small>¼ Kelly ograniczone do 5% bankrolla</small></article></section>
      <section className="financial-section">
        <div className="financial-section-title">
          <div>
            <h2>Wykluczenia czasowe i kwalifikacyjne</h2>
            <p>Brak wymaganych granic niepewności nie oznacza ujemnego EV. Wykluczone pozycje nie są liczone w ROI.</p>
          </div>
        </div>
        <div className="financial-table-scroll">
          <table className="financial-table">
            <thead><tr><th>Powód</th><th>Liczba</th></tr></thead>
            <tbody>
              {Object.entries(data.temporal_exclusions).filter(([, count]) => count > 0).map(([reason, count]) => (
                <tr key={reason}><td>{reason.replace(/_/g, ' ')}</td><td>{count}</td></tr>
              ))}
              {!Object.values(data.temporal_exclusions).some(count => count > 0) && <tr><td colSpan={2} className="table-empty">Brak wykluczeń.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      <section className="financial-section bankroll-section"><div className="financial-section-title"><div><h2>Krzywa bankrolla</h2><p>{stakingMode === 'kelly' ? '¼ Kelly, maks. 5% dostępnego kapitału na zakład' : `Fixed ${fixedStake} PLN na zakład`} · maks. otwarte: {data.max_open_bets} zakł. / {data.max_open_stake.toFixed(2)} PLN</p></div>{bestHorizon && <div className="best-horizon"><span>Najlepszy segment*</span><strong>{bestHorizon.label}</strong><small>ROI {pct(bestHorizon.roi)} · {bestHorizon.matches} mecz.</small></div>}</div><MiniCurve data={data.bankroll_curve}/><p className="financial-note">* ranking wymaga co najmniej 20 meczów w próbie; ich niezależność nie została potwierdzona.</p></section>
      <BucketTable title="Segmenty horyzontu w symulacji" subtitle="Godziny liczone od snapshotu kursu do rozpoczęcia meczu; opis próby, nie zalecenie wejścia." rows={data.horizon_buckets}/>
      <div className="financial-two-columns"><BucketTable title="Bukmacherzy" subtitle="ROI jest symulowany; CLV porównuje kurs wejścia z ostatnim kursem tego samego bukmachera przed startem." rows={data.bookmaker_buckets}/><BucketTable title="Ligi" subtitle="Najbardziej reprezentowane ligi w aktualnym filtrze." rows={data.league_buckets} limit={10}/></div>
      <section className="financial-section">
        <div className="financial-section-title">
          <div>
            <h2>Ledger zakładów</h2>
            <p>Jeden najwyższy dodatni EV wybór na mecz; kapitał pozostaje zarezerwowany do dostępności wyniku.</p>
          </div>
          <span className="ledger-count">{data.ledger.length} pozycji</span>
        </div>
        <div className="financial-table-scroll ledger">
          <table className="financial-table">
            <thead><tr><th>Mecz</th><th>Bukmacher</th><th>Wejście</th><th>EV</th><th>CLV</th><th>Stawka</th><th>Wynik</th></tr></thead>
            <tbody>
              {data.ledger.slice().reverse().slice(0, 100).map(row => (
                <tr key={`${row.canonical_match_id}-${row.bookmaker}-${row.side}`}>
                  <td><strong>{row.team_a_name} <span className="vs">vs</span> {row.team_b_name}</strong><small>{row.league} · {row.horizon}</small></td>
                  <td>{row.bookmaker}</td>
                  <td>{row.entry_odds.toFixed(2)}</td>
                  <td>{pct(row.ev)}<small>{uncertaintyLabel[row.uncertainty_basis]} · {row.uncertainty_reason}</small><small>P użyte do EV: {pct(row.target_prob)}</small></td>
                  <td>{pct(row.clv_odds_pct)}</td>
                  <td>{money(row.stake)}</td>
                  <td className={row.profit >= 0 ? 'positive' : 'negative'}>{money(row.profit)}</td>
                </tr>
              ))}
              {!data.ledger.length && <tr><td colSpan={7} className="table-empty">Brak kwalifikujących się zakładów; przyczyny podano w wykluczeniach.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </>}
  </div>
}
