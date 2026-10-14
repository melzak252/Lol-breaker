import { useState, useEffect, useRef } from 'react'
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import MatchList from './pages/MatchList'
import MatchDetail from './pages/MatchDetail'
import MatchResults from './pages/MatchResults'
import SystemPage from './pages/SystemPage'
import ModelAnalysis from './pages/ModelAnalysis'
import ManualMapping from './pages/ManualMapping'
import ChampionEmbeddings from './pages/ChampionEmbeddings'
import FinancialAnalysis from './pages/FinancialAnalysis'
import Rankings from './pages/Rankings'
import TournamentSimulation from './pages/TournamentSimulation'
import MatchupSimulator from './pages/MatchupSimulator'
import PlayerComparison from './pages/PlayerComparison'
import AlertsCenter from './pages/AlertsCenter'
import './App.css'

interface NavSingle {
  type: 'single'
  to: string
  label: string
}

interface NavDropdownItem {
  to: string
  label: string
  desc: string
}

interface NavDropdown {
  type: 'dropdown'
  id: string
  label: string
  items: NavDropdownItem[]
}

type NavEntry = NavSingle | NavDropdown

const NAV_ENTRIES: NavEntry[] = [
  { type: 'single', to: '/', label: 'Mecze' },
  {
    type: 'dropdown',
    id: 'analytics',
    label: 'Analityka',
    items: [
      { to: '/financial', label: 'Finanse & Bankroll', desc: 'PnL, stawki Kelly, statystyki portfela' },
      { to: '/results', label: 'Wyniki & Backtest', desc: 'Historyczna skuteczność modeli i rynku' },
      { to: '/horizon', label: 'Walidacja Modelu & Reality Check', desc: 'LogLoss vs bukmacherzy, CLV, Kelly, Expected vs Reality' },
    ],
  },
  {
    type: 'dropdown',
    id: 'simulations',
    label: 'Symulacje & Ratingi',
    items: [
      { to: '/tournaments', label: 'Drabinka / Turnieje', desc: 'Symulacje Monte Carlo Worlds 2026 i ENC' },
      { to: '/matchup', label: 'Matchup (H2H)', desc: 'Symulator bezpośrednich pojedynków' },
      { to: '/players/compare', label: 'Porównanie Graczy', desc: 'Ewolucja ratingów w czasie, werdykt modelu i H2H' },
      { to: '/rankings', label: 'Rankingi', desc: 'Glicko-2, Elo drużyn oraz graczy' },
      { to: '/embeddings/champions', label: 'Champion Embeddings', desc: 'Reprezentacje wektorowe ról i postaci' },
    ],
  },
  {
    type: 'dropdown',
    id: 'system',
    label: 'Zarządzanie',
    items: [
      { to: '/system', label: 'Monitoring & Scheduler', desc: 'Status scraperów, bazy i zadania cron' },
      { to: '/alerts', label: 'Centrum Alertów', desc: 'Powiadomienia Value Bet (Telegram & Discord)' },
      { to: '/mapping', label: 'Mapowania drużyn', desc: 'Aliasy i powiązania z GOL.GG' },
    ],
  },
]

function isItemActive(to: string, pathname: string): boolean {
  if (to === '/') {
    return pathname === '/' || pathname.startsWith('/matches/')
  }
  if (to === '/horizon') {
    return pathname === '/horizon' || pathname === '/bootstrap'
  }
  return pathname === to || pathname.startsWith(`${to}/`)
}

function isDropdownActive(dropdown: NavDropdown, pathname: string): boolean {
  return dropdown.items.some((item) => isItemActive(item.to, pathname))
}

function Nav() {
  const location = useLocation()
  const [openDropdown, setOpenDropdown] = useState<string | null>(null)
  const navRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    setOpenDropdown(null)
  }, [location.pathname])

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (navRef.current && !navRef.current.contains(event.target as Node)) {
        setOpenDropdown(null)
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpenDropdown(null)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return (
    <nav className="main-nav" ref={navRef}>
      <div className="nav-brand">
        <Link to="/">
          <span className="brand-mark">LB</span>
          <span className="brand-copy">
            <strong>LoL-Breaker</strong>
            <small>betting intelligence</small>
          </span>
        </Link>
      </div>
      <div className="nav-links">
        {NAV_ENTRIES.map((entry) => {
          if (entry.type === 'single') {
            const active = isItemActive(entry.to, location.pathname)
            return (
              <Link
                key={entry.to}
                to={entry.to}
                className={active ? 'active' : ''}
              >
                {entry.label}
              </Link>
            )
          }

          const active = isDropdownActive(entry, location.pathname)
          const isOpen = openDropdown === entry.id

          return (
            <div
              key={entry.id}
              className={`nav-dropdown${entry.id === 'system' ? ' nav-dropdown-align-right' : ''}`}
            >
              <button
                type="button"
                className={`nav-dropdown-trigger${active ? ' active' : ''}${isOpen ? ' is-open' : ''}`}
                onClick={() => setOpenDropdown(isOpen ? null : entry.id)}
                aria-expanded={isOpen}
                aria-haspopup="true"
              >
                <span>{entry.label}</span>
                <svg
                  className="dropdown-chevron"
                  viewBox="0 0 12 12"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <path
                    d="M2.5 4.5L6 8L9.5 4.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              {isOpen && (
                <div className="nav-dropdown-menu" role="menu">
                  {entry.items.map((item) => {
                    const itemActive = isItemActive(item.to, location.pathname)
                    return (
                      <Link
                        key={item.to}
                        to={item.to}
                        className={`nav-dropdown-item${itemActive ? ' active' : ''}`}
                        role="menuitem"
                        onClick={() => setOpenDropdown(null)}
                      >
                        <span className="dropdown-item-title">{item.label}</span>
                        <span className="dropdown-item-desc">{item.desc}</span>
                      </Link>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </nav>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<MatchList />} />
          <Route path="/matches/:id" element={<MatchDetail />} />
          <Route path="/matches/:id/props" element={<MatchDetail />} />
          <Route path="/results" element={<MatchResults />} />
          <Route path="/financial" element={<FinancialAnalysis />} />
          <Route path="/matchup" element={<MatchupSimulator />} />
          <Route path="/tournaments" element={<TournamentSimulation />} />
          <Route path="/rankings" element={<Rankings />} />
          <Route path="/players/compare" element={<PlayerComparison />} />
          <Route path="/players" element={<PlayerComparison />} />
          <Route path="/mapping" element={<ManualMapping />} />
          <Route path="/embeddings/champions" element={<ChampionEmbeddings />} />
          <Route path="/system" element={<SystemPage />} />
          <Route path="/horizon" element={<ModelAnalysis />} />
          <Route path="/bootstrap" element={<ModelAnalysis />} />
          <Route path="/alerts" element={<AlertsCenter />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}

export default App
