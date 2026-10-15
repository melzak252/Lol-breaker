import { useState, useEffect, useMemo, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { fetchMatchResults } from '../api/client'
import type { MatchResultItem } from '../types'
import './MatchResults.css'

const DAYS_OPTIONS = [7, 14, 30, 60, 90]
const MODEL_OPTIONS = [
  {
    label: 'MarketShrunk-Hybrid-50 (Attention + Market)',
    name: 'Hybrid-Operational-Market',
    version: 'c0-native-2026-w32-e12-v1-a0.50-t1.00',
    description: '50/50 logit shrinkage: PlayerHistory-Attention-Ensemble (C0) + konsensus rynku (Shin)',
  },
  {
    label: 'PlayerHistory-Attention-Ensemble (C0)',
    name: 'Causal-C0',
    version: 'c0-native-2026-w32-e12-v1',
    description: '3-seed sieć uwagi historii graczy (16 map) + eksperci Glicko-2/Elo/MLP',
  },
]
const ODDS_MODE_OPTIONS = [
  { value: 'open', label: 'Open' },
  { value: 'mid', label: 'Mid' },
  { value: 'close', label: 'Close' },
]


export default function MatchResults() {
  const [results, setResults] = useState<MatchResultItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [daysBack, setDaysBack] = useState(30)
  const [total, setTotal] = useState(0)
  const [showPositiveEvOnly, setShowPositiveEvOnly] = useState(false)
  const [selectedBookmaker, setSelectedBookmaker] = useState<string | null>(null)
  const [oddsMode, setOddsMode] = useState('close')
  const [selectedModelIndex, setSelectedModelIndex] = useState(0)
  const selectedModel = MODEL_OPTIONS[selectedModelIndex]

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setResults([])
    setTotal(0)
    fetchMatchResults(daysBack, oddsMode, selectedModel.name, selectedModel.version, controller.signal)
      .then((data) => {
        if (cancelled) return
        setResults(data.results)
        setTotal(data.total)
      })
      .catch((err) => {
        if (!cancelled && err.name !== 'AbortError') setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [daysBack, oddsMode, selectedModel.name, selectedModel.version])

  // Extract unique bookmakers from results
  const availableBookmakers = useMemo(() => {
    const bookmakers = new Set<string>()
    for (const r of results) {
      for (const b of Object.keys(r.bookmaker_ev_details || {})) {
        bookmakers.add(b)
      }
    }
    return Array.from(bookmakers).sort()
  }, [results])

  // Helper: get the effective EV/odds/kelly/prob for a match, respecting bookmaker filter
  const getEffectiveEvOdds = useCallback((r: MatchResultItem): {
    evA: number | null; evB: number | null;
    oddsA: number | null; oddsB: number | null;
    kellyA: number | null; kellyB: number | null;
    probA: number | null; probB: number | null;
  } => {
    if (selectedBookmaker && r.bookmaker_ev_details[selectedBookmaker]) {
      const detail = r.bookmaker_ev_details[selectedBookmaker]
      return {
        evA: detail.side_a.ev,
        evB: detail.side_b.ev,
        oddsA: detail.side_a.odds,
        oddsB: detail.side_b.odds,
        kellyA: detail.side_a.kelly,
        kellyB: detail.side_b.kelly,
        probA: detail.side_a.model_prob,
        probB: detail.side_b.model_prob,
      }
    }
    return {
      evA: r.best_ev_a,
      evB: r.best_ev_b,
      oddsA: r.best_odds_a,
      oddsB: r.best_odds_b,
      kellyA: r.kelly_a,
      kellyB: r.kelly_b,
      probA: r.model_prob_a,
      probB: r.model_prob_b,
    }
  }, [selectedBookmaker])


  // Filter results
  const filteredResults = useMemo(() => {
    return results.filter((r) => {
      const { evA, evB } = getEffectiveEvOdds(r)
      if (showPositiveEvOnly && !(evA !== null && evA > 0) && !(evB !== null && evB > 0)) {
        return false
      }
      if (selectedBookmaker && !r.bookmaker_ev_details[selectedBookmaker]) {
        return false
      }
      return true
    })
  }, [results, showPositiveEvOnly, selectedBookmaker, getEffectiveEvOdds])

  const positiveEvCount = useMemo(() => {
    return results.filter((r) => {
      const { evA, evB } = getEffectiveEvOdds(r)
      return (evA !== null && evA > 0) || (evB !== null && evB > 0)
    }).length
  }, [results, getEffectiveEvOdds])


  // Determine EV outcome for a match: 'won' | 'lost' | null
  const getEvOutcome = (r: MatchResultItem): 'won' | 'lost' | null => {
    const { evA, evB } = getEffectiveEvOdds(r)
    if (!(evA !== null && evA > 0) && !(evB !== null && evB > 0)) return null

    // Check if any +EV side won
    if (evA !== null && evA > 0 && r.winner_side === 'team_a') return 'won'
    if (evB !== null && evB > 0 && r.winner_side === 'team_b') return 'won'

    // If we had +EV but the +EV side didn't win
    return 'lost'
  }

  // Group results by date
  const groupedResults = useMemo(() => {
    const groups: { date: string; label: string; results: MatchResultItem[] }[] = []
    const seen = new Map<string, MatchResultItem[]>()

    for (const r of filteredResults) {
      const d = r.start_time_normalized ? new Date(r.start_time_normalized) : null
      const key = d
        ? `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
        : 'unknown'

      if (!seen.has(key)) {
        seen.set(key, [])
      }
      seen.get(key)!.push(r)
    }

    for (const [key, items] of seen) {
      const d = key === 'unknown' ? null : new Date(key + 'T12:00:00')
      const label = d
        ? d.toLocaleDateString('pl-PL', {
            weekday: 'long',
            day: 'numeric',
            month: 'long',
          })
        : 'Nieznana data'
      groups.push({ date: key, label: label.charAt(0).toUpperCase() + label.slice(1), results: items })
    }

    return groups
  }, [filteredResults])

  const formatTime = (dateStr: string | null) => {
    if (!dateStr) return '—'
    const d = new Date(dateStr)
    return d.toLocaleTimeString('pl-PL', {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getWinnerClass = (side: string | null, teamSide: 'a' | 'b') => {
    if (!side) return ''
    if (side === 'team_a' && teamSide === 'a') return 'winner'
    if (side === 'team_b' && teamSide === 'b') return 'winner'
    return 'loser'
  }

  const hasPositiveEv = (r: MatchResultItem) => {
    const { evA, evB } = getEffectiveEvOdds(r)
    return (evA !== null && evA > 0) || (evB !== null && evB > 0)
  }

  const formatEv = (ev: number | null) => {
    if (ev === null || ev === undefined) return null
    const pct = (ev * 100).toFixed(1)
    return ev > 0 ? `+${pct}%` : `${pct}%`
  }


  return (
    <div className="match-results-page">
      <div className="results-header">
        <div className="results-title-row">
          <div className="results-title">
            <div>
              <h1>Wyniki meczów</h1>
              <p>Historyczne przeliczenie przy obserwacji wybranego kursu (historical_quote_recomputation), nie bieżąca rekomendacja ani odtworzenie wydanej prognozy. Konsensus obejmuje wyłącznie znane wtedy kwotowania wszystkich bukmacherów.</p>
            </div>
            <span className="results-count">
              {filteredResults.length} {filteredResults.length === 1 ? 'mecz' : filteredResults.length < 5 ? 'mecze' : 'meczów'}
              {total !== filteredResults.length && ` z ${total}`}
            </span>
          </div>
          <button
            className={`ev-filter-btn${showPositiveEvOnly ? ' active' : ''}`}
            onClick={() => setShowPositiveEvOnly(!showPositiveEvOnly)}
            title="Pokaż tylko mecze z +EV"
          >
            <span className="ev-filter-icon">⚡</span> +EV
            {showPositiveEvOnly && positiveEvCount > 0 && (
              <span className="ev-filter-count">{positiveEvCount}</span>
            )}
          </button>
        </div>


        <div className="results-controls">
          <span className="controls-label">Okres:</span>
          <div className="days-pills">
            {DAYS_OPTIONS.map((d) => (
              <button
                key={d}
                className={`days-pill${daysBack === d ? ' active' : ''}`}
                onClick={() => setDaysBack(d)}
              >
                {d}d
              </button>
            ))}
          </div>
          <div className="staking-toggle">
            <span className="controls-label">Model:</span>
            <div className="staking-pills">
              {MODEL_OPTIONS.map((model, idx) => (
                <button
                  key={`${model.name}:${model.version}`}
                  className={`staking-pill${selectedModelIndex === idx ? ' active' : ''}`}
                  onClick={() => setSelectedModelIndex(idx)}
                  title={model.description}
                >
                  {model.label}
                </button>
              ))}
            </div>
          </div>
          <div className="staking-toggle">
            <span className="controls-label">Kurs:</span>
            <div className="staking-pills">
              {ODDS_MODE_OPTIONS.map((mode) => (
                <button
                  key={mode.value}
                  className={`staking-pill${oddsMode === mode.value ? ' active' : ''}`}
                  onClick={() => setOddsMode(mode.value)}
                  title="Wybór stanu przed startem; Mid używa czasowego środka między pierwszą obserwacją a startem. Nieaktywnego stanu nie zastępuje starszy kurs."
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>
          <Link className="ev-filter-btn" to="/financial">
            Otwórz ledger finansowy
          </Link>
          {availableBookmakers.length > 0 && (
            <div className="bookmaker-filter">
              <span className="controls-label">Bukmacher:</span>
              <div className="bookmaker-pills">
                <button
                  className={`bookmaker-pill${!selectedBookmaker ? ' active' : ''}`}
                  onClick={() => setSelectedBookmaker(null)}
                >
                  Wszyscy
                </button>
                {availableBookmakers.map((b) => (
                  <button
                    key={b}
                    className={`bookmaker-pill${selectedBookmaker === b ? ' active' : ''}`}
                    onClick={() => setSelectedBookmaker(selectedBookmaker === b ? null : b)}
                  >
                    {b}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {loading && <div className="results-loading">Ładowanie wyników...</div>}
      {error && <div className="results-error">Błąd: {error}</div>}

      {!loading && !error && filteredResults.length === 0 && (
        <div className="results-empty">
          {showPositiveEvOnly || selectedBookmaker
            ? 'Brak wyników pasujących do filtrów.'
            : 'Brak wyników w wybranym okresie.'}
        </div>
      )}

      {!loading && !error && groupedResults.map((group) => (
        <div key={group.date} className="results-date-group">
          <div className="date-group-header">{group.label}</div>
          <div className="results-grid">
            {group.results.map((r) => {
              const teamAClass = getWinnerClass(r.winner_side, 'a')
              const teamBClass = getWinnerClass(r.winner_side, 'b')
              const hasResult = r.team_a_score !== null && r.team_b_score !== null
              const isPositiveEv = hasPositiveEv(r)
              const evOutcome = getEvOutcome(r)

              // Determine card class based on EV outcome
              let cardEvClass = ''
              if (evOutcome === 'won') cardEvClass = ' ev-won'
              else if (evOutcome === 'lost') cardEvClass = ' ev-lost'

              const { evA, evB, oddsA, oddsB, probA, probB, kellyA, kellyB } = getEffectiveEvOdds(r)

              return (
                <Link
                  key={r.canonical_match_id}
                  to={`/matches/${r.canonical_match_id}`}
                  className={`result-card${cardEvClass}`}
                >
                  <div className="result-card-header">
                    <span className="result-league">{r.league || 'Nieznana liga'}</span>
                    {r.best_of && (
                      <span className={`result-bo-badge bo${r.best_of}`}>
                        Bo{r.best_of}
                      </span>
                    )}
                    {isPositiveEv && (
                      <span className={`result-ev-indicator${evOutcome === 'won' ? ' ev-won' : evOutcome === 'lost' ? ' ev-lost' : ''}`}>
                        {evOutcome === 'won' ? '✓' : evOutcome === 'lost' ? '✗' : '⚡'} +EV
                      </span>
                    )}
                    <span className="result-time">{formatTime(r.start_time_normalized)}</span>
                  </div>

                  <div className="result-card-body">
                    <div className={`result-team ${teamAClass}`}>
                      <span className="result-team-name">{r.team_a_name || '?'}</span>
                      {hasResult && (
                        <span className={`result-score ${teamAClass}`}>
                          {r.team_a_score}
                        </span>
                      )}
                    </div>
                    <div className="result-vs-divider">
                      <span className="result-vs">vs</span>
                    </div>
                    <div className={`result-team ${teamBClass}`}>
                      <span className="result-team-name">{r.team_b_name || '?'}</span>
                      {hasResult && (
                        <span className={`result-score ${teamBClass}`}>
                          {r.team_b_score}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="result-ev-row">
                    <span>Historyczne przeliczenie kursu</span>
                    <span>A: {oddsA === null ? '—' : oddsA.toFixed(2)} · EV {formatEv(evA) ?? 'niedostępne'}</span>
                    <span>B: {oddsB === null ? '—' : oddsB.toFixed(2)} · EV {formatEv(evB) ?? 'niedostępne'}</span>
                    <span>P(A): {probA === null ? 'niedostępne' : `${(probA*100).toFixed(1)}%`} · Kelly {kellyA === null ? 'niedostępne' : `${(kellyA*100).toFixed(1)}%`}</span>
                    <span>P(B): {probB === null ? 'niedostępne' : `${(probB*100).toFixed(1)}%`} · Kelly {kellyB === null ? 'niedostępne' : `${(kellyB*100).toFixed(1)}%`}</span>
                    {r.probability_unavailable_reason && <span>{r.probability_unavailable_reason}</span>}
                  </div>
                  {/* EV signals row */}
                  {isPositiveEv && (
                    <div className="result-ev-row">
                      {evA !== null && evA > 0 && (
                        <span className={`result-ev-badge ev-a${r.winner_side === 'team_a' ? ' ev-bet-won' : ' ev-bet-lost'}`}>
                          {r.team_a_name?.split(' ').slice(-1)[0]}: {formatEv(evA)}
                          {oddsA !== null && <span className="ev-odds">@{oddsA.toFixed(2)}</span>}
                        </span>
                      )}
                      {evB !== null && evB > 0 && (
                        <span className={`result-ev-badge ev-b${r.winner_side === 'team_b' ? ' ev-bet-won' : ' ev-bet-lost'}`}>
                          {r.team_b_name?.split(' ').slice(-1)[0]}: {formatEv(evB)}
                          {oddsB !== null && <span className="ev-odds">@{oddsB.toFixed(2)}</span>}
                        </span>
                      )}
                      {r.bookmakers_with_ev.length > 0 && (
                        <span className="result-ev-bookmakers">
                          {r.bookmakers_with_ev.join(', ')}
                        </span>
                      )}
                    </div>
                  )}

                  <div className="result-card-footer">
                    {r.result_source && (
                      <span className="result-source-badge">{r.result_source}</span>
                    )}
                    {hasResult && r.winner_name && (
                      <span className="result-winner-label">
                        🏆 {r.winner_name}
                      </span>
                    )}
                  </div>
                </Link>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
