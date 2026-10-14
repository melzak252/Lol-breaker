import React, { useEffect, useMemo, useState } from 'react';
import { fetchPlayerComparison, searchPlayers } from '../api/client';
import type {
  PlayerComparisonResponse,
  PlayerSearchItem,
  RatingTimelinePoint,
} from '../types';
import './PlayerComparison.css';

interface PresetMatchup {
  nameA: string;
  nameB: string;
  label: string;
}

const PRESET_MATCHUPS: PresetMatchup[] = [
  { nameA: 'Faker', nameB: 'Chovy', label: 'Faker vs Chovy' },
  { nameA: 'Ruler', nameB: 'Viper', label: 'Ruler vs Viper' },
  { nameA: 'Caps', nameB: 'Perkz', label: 'Caps vs Perkz' },
  { nameA: 'Canyon', nameB: 'Oner', label: 'Canyon vs Oner' },
  { nameA: 'Scout', nameB: 'Rookie', label: 'Scout vs Rookie' },
];

type TimelineMetric = 'elo' | 'gl' | 'ts_mu' | 'os_mu' | 'pl_mu' | 'tm_mu';

type TimeRange = 'all' | '5y' | '2y' | '1y';

const METRIC_LABELS: Record<TimelineMetric, string> = {
  elo: 'Elo Rating',
  gl: 'Glicko-2',
  ts_mu: 'TrueSkill (μ)',
  os_mu: 'OpenSkill (μ)',
  pl_mu: 'Plackett–Luce (μ)',
  tm_mu: 'Thurstone–Mosteller (μ)',
};

export const PlayerComparison: React.FC = () => {
  // Player IDs or names
  const [playerAId, setPlayerAId] = useState<string>('48'); // Faker
  const [playerBId, setPlayerBId] = useState<string>('1629'); // Chovy

  // Search input state
  const [searchA, setSearchA] = useState<string>('Faker');
  const [searchB, setSearchB] = useState<string>('Chovy');
  const [dropdownAOpen, setDropdownAOpen] = useState<boolean>(false);
  const [dropdownBOpen, setDropdownBOpen] = useState<boolean>(false);
  const [resultsA, setResultsA] = useState<PlayerSearchItem[]>([]);
  const [resultsB, setResultsB] = useState<PlayerSearchItem[]>([]);

  // Comparison data & states
  const [comparison, setComparison] = useState<PlayerComparisonResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Chart controls
  const [chartMetric, setChartMetric] = useState<TimelineMetric>('elo');
  const [timeRange, setTimeRange] = useState<TimeRange>('all');
  const [hoveredPoint, setHoveredPoint] = useState<{
    x: number;
    y: number;
    date: string;
    player: 'a' | 'b';
    playerName: string;
    val: number;
    team: string | null;
  } | null>(null);

  // Debounced search for Player A
  useEffect(() => {
    if (!dropdownAOpen || searchA.trim().length < 2) {
      setResultsA([]);
      return;
    }
    const timer = setTimeout(() => {
      searchPlayers(searchA)
        .then((res) => setResultsA(res))
        .catch(() => setResultsA([]));
    }, 200);
    return () => clearTimeout(timer);
  }, [searchA, dropdownAOpen]);

  // Debounced search for Player B
  useEffect(() => {
    if (!dropdownBOpen || searchB.trim().length < 2) {
      setResultsB([]);
      return;
    }
    const timer = setTimeout(() => {
      searchPlayers(searchB)
        .then((res) => setResultsB(res))
        .catch(() => setResultsB([]));
    }, 200);
    return () => clearTimeout(timer);
  }, [searchB, dropdownBOpen]);

  // Load comparison data
  useEffect(() => {
    if (!playerAId || !playerBId) return;
    setLoading(true);
    setError(null);

    fetchPlayerComparison(playerAId, playerBId)
      .then((data) => {
        setComparison(data);
        setSearchA(data.player_a.player_name);
        setSearchB(data.player_b.player_name);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Błąd podczas pobierania danych porównania.');
        setLoading(false);
      });
  }, [playerAId, playerBId]);

  const selectPlayerA = (player: PlayerSearchItem) => {
    setPlayerAId(player.player_id);
    setSearchA(player.player_name);
    setDropdownAOpen(false);
  };

  const selectPlayerB = (player: PlayerSearchItem) => {
    setPlayerBId(player.player_id);
    setSearchB(player.player_name);
    setDropdownBOpen(false);
  };

  const handlePresetClick = (preset: PresetMatchup) => {
    setSearchA(preset.nameA);
    setSearchB(preset.nameB);
    setPlayerAId(preset.nameA);
    setPlayerBId(preset.nameB);
  };

  // Filter timelines by time range
  const filteredTimelines = useMemo(() => {
    if (!comparison) return { a: [], b: [] };
    const { timeline_a, timeline_b } = comparison;
    if (timeRange === 'all') return { a: timeline_a, b: timeline_b };

    const nowYear = 2026;
    let minYear = 2013;
    if (timeRange === '5y') minYear = nowYear - 5;
    if (timeRange === '2y') minYear = nowYear - 2;
    if (timeRange === '1y') minYear = nowYear - 1;

    const minDateStr = `${minYear}-01-01`;
    return {
      a: timeline_a.filter((p) => p.date >= minDateStr),
      b: timeline_b.filter((p) => p.date >= minDateStr),
    };
  }, [comparison, timeRange]);

  // SVG Chart Computations
  const chartData = useMemo(() => {
    const listA = filteredTimelines.a;
    const listB = filteredTimelines.b;
    if (listA.length === 0 && listB.length === 0) return null;

    const getVal = (p: RatingTimelinePoint): number => {
      const v = p[chartMetric];
      return v !== null && v !== undefined ? Number(v) : 0;
    };

    const allVals = [...listA.map(getVal), ...listB.map(getVal)].filter((v) => v > 0);
    if (allVals.length === 0) return null;

    const rawMin = Math.min(...allVals);
    const rawMax = Math.max(...allVals);
    const padding = (rawMax - rawMin) * 0.1 || 20;
    const yMin = Math.floor(rawMin - padding);
    const yMax = Math.ceil(rawMax + padding);

    // Date range
    const allDates = [...listA.map((p) => p.date), ...listB.map((p) => p.date)].sort();
    const minDate = allDates[0];
    const maxDate = allDates[allDates.length - 1];

    const minTime = new Date(minDate).getTime();
    const maxTime = new Date(maxDate).getTime();
    const timeSpan = maxTime - minTime || 1;

    const width = 900;
    const height = 300;
    const padL = 50;
    const padR = 25;
    const padT = 20;
    const padB = 40;
    const innerW = width - padL - padR;
    const innerH = height - padT - padB;

    const getX = (dateStr: string) => {
      const t = new Date(dateStr).getTime();
      return padL + ((t - minTime) / timeSpan) * innerW;
    };

    const getY = (val: number) => {
      return padT + innerH - ((val - yMin) / (yMax - yMin)) * innerH;
    };

    const buildPath = (pts: RatingTimelinePoint[]) => {
      if (pts.length === 0) return '';
      return pts
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.date).toFixed(1)} ${getY(getVal(p)).toFixed(1)}`)
        .join(' ');
    };

    const pathA = buildPath(listA);
    const pathB = buildPath(listB);

    // Y ticks (4-5 ticks)
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => {
      const val = yMin + pct * (yMax - yMin);
      return { val: Math.round(val), y: getY(val) };
    });

    // X date ticks (approx 5-6 ticks)
    const xTicks: Array<{ date: string; x: number }> = [];
    const step = Math.max(1, Math.floor(allDates.length / 5));
    for (let i = 0; i < allDates.length; i += step) {
      xTicks.push({
        date: allDates[i].slice(0, 7), // YYYY-MM
        x: getX(allDates[i]),
      });
    }

    return {
      width,
      height,
      padL,
      padR,
      padT,
      padB,
      pathA,
      pathB,
      yTicks,
      xTicks,
      listA,
      listB,
      getX,
      getY,
      getVal,
    };
  }, [filteredTimelines, chartMetric]);

  return (
    <div className="player-comp-page">
      {/* Hero Header */}
      <div className="player-comp-hero">
        <div className="player-comp-eyebrow">Analityka & Ratingi Graczy</div>
        <h1>Zestawienie Graczy: Ewolucja Formy & H2H</h1>
        <p>
          Porównanie historycznych trajektorii ratingowych graczy, konsensus modeli probabilistycznych
          (Elo, Glicko-2, TrueSkill, OpenSkill, Plackett–Luce, Thurstone–Mosteller) oraz bezpośredni bilans meczowy.
        </p>
      </div>

      {/* Preset Matchups */}
      <div className="player-comp-presets">
        <span className="player-comp-presets-label">Szybkie zestawienia:</span>
        {PRESET_MATCHUPS.map((preset) => (
          <button
            key={preset.label}
            className={`preset-btn ${searchA === preset.nameA && searchB === preset.nameB ? 'active' : ''}`}
            onClick={() => handlePresetClick(preset)}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Player Search Bar */}
      <div className="player-search-grid">
        {/* Player A Box */}
        <div className="search-box">
          <label>Gracz A (Niebieski)</label>
          <div className="search-input-wrapper">
            <input
              type="text"
              className="search-input active-player-a"
              value={searchA}
              onChange={(e) => {
                setSearchA(e.target.value);
                setDropdownAOpen(true);
              }}
              onFocus={() => setDropdownAOpen(true)}
              placeholder="Wyszukaj gracza A (np. Faker, Caps)..."
            />
          </div>
          {dropdownAOpen && resultsA.length > 0 && (
            <div className="search-dropdown">
              {resultsA.map((p) => (
                <div key={p.player_id} className="search-item" onClick={() => selectPlayerA(p)}>
                  <div className="search-item-info">
                    <span className="search-item-name">{p.player_name}</span>
                    <span className="search-item-sub">
                      {p.team_name || 'Brak drużyny'} {p.role ? `• ${p.role}` : ''}
                    </span>
                  </div>
                  <div className="search-item-meta">
                    <span className="search-item-elo">Elo {p.current_elo ? p.current_elo.toFixed(0) : '—'}</span>
                    <span className="search-item-games">{p.games_played} gier</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* VS Badge */}
        <div className="vs-badge">VS</div>

        {/* Player B Box */}
        <div className="search-box">
          <label>Gracz B (Pomarańczowy)</label>
          <div className="search-input-wrapper">
            <input
              type="text"
              className="search-input active-player-b"
              value={searchB}
              onChange={(e) => {
                setSearchB(e.target.value);
                setDropdownBOpen(true);
              }}
              onFocus={() => setDropdownBOpen(true)}
              placeholder="Wyszukaj gracza B (np. Chovy, Ruler)..."
            />
          </div>
          {dropdownBOpen && resultsB.length > 0 && (
            <div className="search-dropdown">
              {resultsB.map((p) => (
                <div key={p.player_id} className="search-item" onClick={() => selectPlayerB(p)}>
                  <div className="search-item-info">
                    <span className="search-item-name">{p.player_name}</span>
                    <span className="search-item-sub">
                      {p.team_name || 'Brak drużyny'} {p.role ? `• ${p.role}` : ''}
                    </span>
                  </div>
                  <div className="search-item-meta">
                    <span className="search-item-elo">Elo {p.current_elo ? p.current_elo.toFixed(0) : '—'}</span>
                    <span className="search-item-games">{p.games_played} gier</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-muted)' }}>
          Wczytywanie pełnego profilu i historii rankingowej graczy...
        </div>
      )}

      {error && (
        <div style={{ padding: '1.5rem', background: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px', color: '#f87171' }}>
          {error}
        </div>
      )}

      {!loading && !error && comparison && (
        <>
          {/* Main Verdict Banner */}
          <div className="verdict-banner">
            <div className="verdict-header">
              <div className="verdict-title-wrap">
                <span className="verdict-title-eyebrow">Werdykt Modelu: Kto jest brany za lepszego gracza</span>
                <h2 className="verdict-main-text">
                  {comparison.verdict.better_player === 'a'
                    ? `Wskazanie: ${comparison.player_a.player_name}`
                    : comparison.verdict.better_player === 'b'
                    ? `Wskazanie: ${comparison.player_b.player_name}`
                    : 'Remis / Brak jednoznacznego faworyta'}
                </h2>
              </div>
              <div className="verdict-badges">
                <span className="badge-consensus">{comparison.verdict.advantage_summary}</span>
                {comparison.h2h.total_games > 0 && (
                  <span className="badge-h2h">
                    H2H: {comparison.h2h.total_games} gier ({comparison.h2h.wins_a} - {comparison.h2h.wins_b})
                  </span>
                )}
              </div>
            </div>

            {/* Win Probability Bar */}
            <div className="prob-meter-container">
              <div className="prob-meter-labels">
                <span className="prob-label-a">
                  {comparison.player_a.player_name}: {(comparison.verdict.win_probability_a * 100).toFixed(1)}%
                </span>
                <span className="prob-label-b">
                  {comparison.player_b.player_name}: {(comparison.verdict.win_probability_b * 100).toFixed(1)}%
                </span>
              </div>
              <div className="prob-bar">
                <div
                  className="prob-fill-a"
                  style={{ width: `${comparison.verdict.win_probability_a * 100}%` }}
                />
                <div
                  className="prob-fill-b"
                  style={{ width: `${comparison.verdict.win_probability_b * 100}%` }}
                />
              </div>
            </div>

            {/* Polish Summary Card */}
            <div className="verdict-summary-card">
              <div dangerouslySetInnerHTML={{ __html: formatMarkdownBold(comparison.verdict.summary_pl) }} />
            </div>
          </div>

          {/* Player Cards (Stats & Champions) */}
          <div className="comp-sections-grid">
            {/* Player A Card */}
            <div className="player-card player-a-border">
              <div className="player-card-header">
                <div>
                  <h3 className="player-card-name" style={{ color: '#38bdf8' }}>
                    {comparison.player_a.player_name}
                  </h3>
                  <div className="player-card-sub">
                    {comparison.player_a.team_name || 'Brak drużyny'}
                    {comparison.player_a.career_years ? ` • ${comparison.player_a.career_years} lat kariery` : ''}
                  </div>
                </div>
                {comparison.player_a.role && (
                  <span className="player-card-role-badge">{comparison.player_a.role}</span>
                )}
              </div>

              <div className="player-stats-grid">
                <div className="stat-item">
                  <span className="stat-item-label">Gry kariery</span>
                  <span className="stat-item-value">{comparison.player_a.games_played}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-item-label">Win Rate</span>
                  <span className="stat-item-value">{(comparison.player_a.career_win_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-item-label">Peak Elo</span>
                  <span className="stat-item-value">
                    {comparison.player_a.peak_elo ? comparison.player_a.peak_elo.toFixed(0) : '—'}
                  </span>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  NAJCZĘŚCIEJ GRANI CHAMPIONI
                </h4>
                <div className="champions-list">
                  {comparison.player_a.top_champions.map((c) => (
                    <div key={c.champion_name} className="champion-row">
                      <span>{c.champion_name}</span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        {c.games} gier ({(c.win_rate * 100).toFixed(0)}% WR)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Player B Card */}
            <div className="player-card player-b-border">
              <div className="player-card-header">
                <div>
                  <h3 className="player-card-name" style={{ color: '#f97316' }}>
                    {comparison.player_b.player_name}
                  </h3>
                  <div className="player-card-sub">
                    {comparison.player_b.team_name || 'Brak drużyny'}
                    {comparison.player_b.career_years ? ` • ${comparison.player_b.career_years} lat kariery` : ''}
                  </div>
                </div>
                {comparison.player_b.role && (
                  <span className="player-card-role-badge">{comparison.player_b.role}</span>
                )}
              </div>

              <div className="player-stats-grid">
                <div className="stat-item">
                  <span className="stat-item-label">Gry kariery</span>
                  <span className="stat-item-value">{comparison.player_b.games_played}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-item-label">Win Rate</span>
                  <span className="stat-item-value">{(comparison.player_b.career_win_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-item-label">Peak Elo</span>
                  <span className="stat-item-value">
                    {comparison.player_b.peak_elo ? comparison.player_b.peak_elo.toFixed(0) : '—'}
                  </span>
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  NAJCZĘŚCIEJ GRANI CHAMPIONI
                </h4>
                <div className="champions-list">
                  {comparison.player_b.top_champions.map((c) => (
                    <div key={c.champion_name} className="champion-row">
                      <span>{c.champion_name}</span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        {c.games} gier ({(c.win_rate * 100).toFixed(0)}% WR)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Multi-System Matrix */}
          <div className="matrix-card">
            <h3>Porównanie w Systemach Ratingowych</h3>
            <div style={{ overflowX: 'auto' }}>
              <table className="matrix-table">
                <thead>
                  <tr>
                    <th>Model / System</th>
                    <th>{comparison.player_a.player_name}</th>
                    <th>{comparison.player_b.player_name}</th>
                    <th>Różnica</th>
                    <th>Prawdopodobieństwo (A)</th>
                    <th>Faworyt Modelu</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.verdict.system_advantages.map((adv) => (
                    <tr key={adv.system}>
                      <td>
                        <strong>{adv.system_label}</strong>
                      </td>
                      <td style={{ color: '#38bdf8', fontWeight: 600 }}>{adv.value_a.toFixed(1)}</td>
                      <td style={{ color: '#f97316', fontWeight: 600 }}>{adv.value_b.toFixed(1)}</td>
                      <td>{adv.difference > 0 ? `+${adv.difference.toFixed(1)}` : adv.difference.toFixed(1)}</td>
                      <td>{(adv.win_prob_a * 100).toFixed(1)}%</td>
                      <td>
                        {adv.favors === 'a' ? (
                          <span className="badge-favors-a">{comparison.player_a.player_name}</span>
                        ) : adv.favors === 'b' ? (
                          <span className="badge-favors-b">{comparison.player_b.player_name}</span>
                        ) : (
                          <span className="badge-tied">Remis</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* SVG Rating Trajectory Timeline */}
          <div className="timeline-card">
            <div className="timeline-header">
              <div className="timeline-title-wrap">
                <h3>Trajektoria Formy w Czasie</h3>
                <p>Ewolucja umiejętności na przestrzeni lat na bazie meczów oficjalnych</p>
              </div>
              <div className="timeline-controls">
                <select
                  className="timeline-select"
                  value={chartMetric}
                  onChange={(e) => setChartMetric(e.target.value as TimelineMetric)}
                >
                  <option value="elo">Elo Rating</option>
                  <option value="gl">Glicko-2</option>
                  <option value="ts_mu">TrueSkill (μ)</option>
                  <option value="os_mu">OpenSkill (μ)</option>
                  <option value="pl_mu">Plackett–Luce (μ)</option>
                  <option value="tm_mu">Thurstone–Mosteller (μ)</option>
                </select>

                <select
                  className="timeline-select"
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value as TimeRange)}
                >
                  <option value="all">Cała historia</option>
                  <option value="5y">Ostatnie 5 lat</option>
                  <option value="2y">Ostatnie 2 lata</option>
                  <option value="1y">Ostatni rok</option>
                </select>
              </div>
            </div>

            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-dot-a" />
                <span>{comparison.player_a.player_name}</span>
              </div>
              <div className="legend-item">
                <div className="legend-dot-b" />
                <span>{comparison.player_b.player_name}</span>
              </div>
            </div>

            {chartData ? (
              <div className="timeline-svg-wrap">
                <svg
                  viewBox={`0 0 ${chartData.width} ${chartData.height}`}
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                >
                  {/* Grid lines & Y labels */}
                  {chartData.yTicks.map((tick) => (
                    <g key={tick.val}>
                      <line
                        x1={chartData.padL}
                        y1={tick.y}
                        x2={chartData.width - chartData.padR}
                        y2={tick.y}
                        stroke="rgba(255, 255, 255, 0.07)"
                        strokeDasharray="4 4"
                      />
                      <text
                        x={chartData.padL - 8}
                        y={tick.y + 4}
                        fill="#94a3b8"
                        fontSize="10"
                        textAnchor="end"
                      >
                        {tick.val}
                      </text>
                    </g>
                  ))}

                  {/* X Date Labels */}
                  {chartData.xTicks.map((xtick, idx) => (
                    <text
                      key={idx}
                      x={xtick.x}
                      y={chartData.height - 12}
                      fill="#94a3b8"
                      fontSize="10"
                      textAnchor="middle"
                    >
                      {xtick.date}
                    </text>
                  ))}

                  {/* Curve A (Blue) */}
                  {chartData.pathA && (
                    <path
                      d={chartData.pathA}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}

                  {/* Curve B (Orange) */}
                  {chartData.pathB && (
                    <path
                      d={chartData.pathB}
                      fill="none"
                      stroke="#f97316"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}

                  {/* Interactive Points on Player A */}
                  {chartData.listA.map((p, i) => {
                    const cx = chartData.getX(p.date);
                    const cy = chartData.getY(chartData.getVal(p));
                    return (
                      <circle
                        key={`a-${i}`}
                        cx={cx}
                        cy={cy}
                        r="3"
                        fill="#38bdf8"
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={() =>
                          setHoveredPoint({
                            x: cx,
                            y: cy,
                            date: p.date,
                            player: 'a',
                            playerName: comparison.player_a.player_name,
                            val: chartData.getVal(p),
                            team: p.team_name,
                          })
                        }
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    );
                  })}

                  {/* Interactive Points on Player B */}
                  {chartData.listB.map((p, i) => {
                    const cx = chartData.getX(p.date);
                    const cy = chartData.getY(chartData.getVal(p));
                    return (
                      <circle
                        key={`b-${i}`}
                        cx={cx}
                        cy={cy}
                        r="3"
                        fill="#f97316"
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={() =>
                          setHoveredPoint({
                            x: cx,
                            y: cy,
                            date: p.date,
                            player: 'b',
                            playerName: comparison.player_b.player_name,
                            val: chartData.getVal(p),
                            team: p.team_name,
                          })
                        }
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    );
                  })}
                </svg>

                {hoveredPoint && (
                  <div
                    className="timeline-tooltip"
                    style={{
                      left: `${(hoveredPoint.x / chartData.width) * 100}%`,
                      top: `${hoveredPoint.y - 45}px`,
                      transform: 'translateX(-50%)',
                    }}
                  >
                    <div style={{ fontWeight: 700, color: hoveredPoint.player === 'a' ? '#38bdf8' : '#f97316' }}>
                      {hoveredPoint.playerName}
                    </div>
                    <div>
                      {METRIC_LABELS[chartMetric]}: <strong>{hoveredPoint.val.toFixed(1)}</strong>
                    </div>
                    <div style={{ color: 'var(--text-muted)' }}>
                      Data: {hoveredPoint.date} {hoveredPoint.team ? `(${hoveredPoint.team})` : ''}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                Brak punktów trajektorii dla wybranego zakresu czasu.
              </div>
            )}
          </div>

          {/* Head-to-Head Recent Games Table */}
          {comparison.h2h.total_games > 0 ? (
            <div className="h2h-card">
              <h3>
                Bezpośrednie Mecze H2H ({comparison.h2h.total_games} rozegranych map)
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table className="h2h-table">
                  <thead>
                    <tr>
                      <th>Data</th>
                      <th>Turniej</th>
                      <th>{comparison.player_a.player_name} (Team / Champ)</th>
                      <th>{comparison.player_b.player_name} (Team / Champ)</th>
                      <th>Zwycięzca</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.h2h.recent_games.slice(0, 15).map((g) => (
                      <tr key={g.game_id}>
                        <td>{g.date || '—'}</td>
                        <td>{g.tournament_name || '—'}</td>
                        <td>
                          <strong>{g.team_a || '—'}</strong> ({g.champ_a || '—'})
                        </td>
                        <td>
                          <strong>{g.team_b || '—'}</strong> ({g.champ_b || '—'})
                        </td>
                        <td>
                          {g.winner === 'a' ? (
                            <span className="winner-pill-a">{comparison.player_a.player_name}</span>
                          ) : (
                            <span className="winner-pill-b">{comparison.player_b.player_name}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="h2h-card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
              Gracze nie rozegrali dotąd bezpośredniego oficjalnego meczu przeciwko sobie w bazie danych.
            </div>
          )}
        </>
      )}
    </div>
  );
};

function formatMarkdownBold(text: string): string {
  if (!text) return '';
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}
export default PlayerComparison;
