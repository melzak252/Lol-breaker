import { useEffect, useState, useMemo } from 'react';
import {
  fetchSystemStatus,
  fetchBookmakers,
  fetchSchedulerTasks,
  fetchSchedulerJobs,
  fetchSchedulerRuns,
  fetchSchedulerRunCommands,
  triggerSchedulerTask,
  triggerLightCycle,
  triggerBackup,
} from '../api/client';
import type {
  SystemStatusResponse,
  BookmakerStatus,
  SchedulerTask,
  SchedulerJob,
  SchedulerRun,
  SchedulerCommand,
} from '../types';
import './SystemPage.css';

// ─── Health helpers ──────────────────────────────────────────

type HealthLevel = 'healthy' | 'warning' | 'critical' | 'unknown';

function getBookmakerHealth(book: BookmakerStatus): HealthLevel {
  if (book.snapshot_count === 0) return 'unknown';
  if (!book.last_scraped_at) return 'critical';
  const hoursSince = (Date.now() - new Date(book.last_scraped_at).getTime()) / 3600000;
  if (hoursSince > 48) return 'critical';
  if (hoursSince > 6) return 'warning';
  return 'healthy';
}

function getTimeSince(dateStr: string | null): string {
  if (!dateStr) return 'Never';
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function isZombieRun(run: SchedulerRun): boolean {
  if (run.status !== 'running') return false;
  if (!run.started_at) return false;
  const hours = (Date.now() - new Date(run.started_at).getTime()) / 3600000;
  return hours > 1; // running for more than 1 hour = zombie
}

function getSchedulerHealth(runs: SchedulerRun[]): HealthLevel {
  const hasZombie = runs.some(isZombieRun);
  const hasError = runs.some(r => r.status === 'error');
  if (hasZombie) return 'critical';
  if (hasError) return 'warning';
  return 'healthy';
}

// ─── Component ──────────────────────────────────────────────

export default function SystemPage() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [bookmakers, setBookmakers] = useState<BookmakerStatus[]>([]);
  const [tasks, setTasks] = useState<SchedulerTask[]>([]);
  const [jobs, setJobs] = useState<SchedulerJob[]>([]);
  const [runs, setRuns] = useState<SchedulerRun[]>([]);
  const [selectedRunCommands, setSelectedRunCommands] = useState<SchedulerCommand[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'bookmakers' | 'scheduler'>('overview');

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const [statusData, bookmakersData, tasksData, jobsData, runsData] = await Promise.all([
        fetchSystemStatus(),
        fetchBookmakers(),
        fetchSchedulerTasks(),
        fetchSchedulerJobs(),
        fetchSchedulerRuns(),
      ]);
      setStatus(statusData);
      setBookmakers(bookmakersData);
      setTasks(tasksData);
      setJobs(jobsData);
      setRuns(runsData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load system data');
    } finally {
      setLoading(false);
    }
  }

  async function handleTrigger(taskId: string) {
    try {
      const result = await triggerSchedulerTask(taskId);
      setActionMessage(`Triggered ${taskId}: ${result.status}`);
      setTimeout(() => setActionMessage(null), 5000);
      setTimeout(loadData, 2000);
    } catch (err) {
      setActionMessage(`Error: ${err instanceof Error ? err.message : 'Failed'}`);
    }
  }

  async function handleLightCycle() {
    try {
      const result = await triggerLightCycle();
      setActionMessage(`Light cycle: ${result.message}`);
      setTimeout(() => setActionMessage(null), 5000);
      setTimeout(loadData, 2000);
    } catch (err) {
      setActionMessage(`Error: ${err instanceof Error ? err.message : 'Failed'}`);
    }
  }

  async function handleBackup() {
    try {
      const result = await triggerBackup();
      setActionMessage(`Backup: ${result.message}`);
      setTimeout(() => setActionMessage(null), 5000);
    } catch (err) {
      setActionMessage(`Error: ${err instanceof Error ? err.message : 'Failed'}`);
    }
  }

  async function handleViewCommands(runId: number) {
    if (selectedRunId === runId) {
      setSelectedRunId(null);
      setSelectedRunCommands([]);
      return;
    }
    try {
      const commands = await fetchSchedulerRunCommands(runId);
      setSelectedRunCommands(commands);
      setSelectedRunId(runId);
    } catch (err) {
      setActionMessage(`Error loading commands: ${err instanceof Error ? err.message : 'Failed'}`);
    }
  }

  // ─── Derived health state ──────────────────────────────────

  const bookmakerHealth = useMemo(() => {
    const counts = { healthy: 0, warning: 0, critical: 0, unknown: 0 };
    bookmakers.forEach(b => { counts[getBookmakerHealth(b)]++; });
    return counts;
  }, [bookmakers]);

  const schedulerHealth = useMemo(() => getSchedulerHealth(runs), [runs]);

  const overallHealth = useMemo((): HealthLevel => {
    if (bookmakerHealth.critical > 0 || schedulerHealth === 'critical') return 'critical';
    if (bookmakerHealth.warning > 0 || schedulerHealth === 'warning') return 'warning';
    return 'healthy';
  }, [bookmakerHealth, schedulerHealth]);

  if (loading) return <div className="sys-loading">Loading system data...</div>;
  if (error) return <div className="sys-error">Error: {error}</div>;
  if (!status) return <div className="sys-error">No data</div>;

  return (
    <div className="system-page">
      {/* ─── Header ──────────────────────────────────────────── */}
      <div className="sys-header">
        <h1>System</h1>
        <div className="sys-header-actions">
          <button onClick={handleLightCycle} className="sys-btn">
            ⚡ Light Cycle
          </button>
          <button onClick={handleBackup} className="sys-btn sys-btn-secondary">
            💾 Backup
          </button>
          <button onClick={loadData} className="sys-btn sys-btn-ghost">
            ↻ Refresh
          </button>
        </div>
      </div>

      {actionMessage && (
        <div className="sys-toast">{actionMessage}</div>
      )}

      {/* ─── Health Banner ────────────────────────────────────── */}
      <div className={`health-banner health-${overallHealth}`}>
        <div className="health-indicator">
          <span className="health-dot" />
          <span className="health-label">
            {overallHealth === 'healthy' ? 'All Systems Operational' :
             overallHealth === 'warning' ? 'Attention Needed' :
             overallHealth === 'critical' ? 'Issues Detected' : 'Unknown'}
          </span>
        </div>
        <div className="health-summary">
          <span className="health-stat">
            <span className="health-stat-value">{bookmakerHealth.healthy}</span> OK
          </span>
          <span className="health-stat health-stat-warn">
            <span className="health-stat-value">{bookmakerHealth.warning}</span> Stale
          </span>
          <span className="health-stat health-stat-crit">
            <span className="health-stat-value">{bookmakerHealth.critical + bookmakerHealth.unknown}</span> Down
          </span>
          {runs.some(isZombieRun) && (
            <span className="health-stat health-stat-crit">
              <span className="health-stat-value">⚠</span> Zombie Run
            </span>
          )}
        </div>
      </div>

      {/* ─── Tabs ─────────────────────────────────────────────── */}
      <div className="sys-tabs">
        <button
          className={`sys-tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`sys-tab ${activeTab === 'bookmakers' ? 'active' : ''}`}
          onClick={() => setActiveTab('bookmakers')}
        >
          Bookmakers
          {bookmakerHealth.warning + bookmakerHealth.critical + bookmakerHealth.unknown > 0 && (
            <span className="tab-badge">
              {bookmakerHealth.warning + bookmakerHealth.critical + bookmakerHealth.unknown}
            </span>
          )}
        </button>
        <button
          className={`sys-tab ${activeTab === 'scheduler' ? 'active' : ''}`}
          onClick={() => setActiveTab('scheduler')}
        >
          Scheduler
          {schedulerHealth !== 'healthy' && (
            <span className="tab-badge tab-badge-warn">!</span>
          )}
        </button>
      </div>

      {/* ─── Overview Tab ─────────────────────────────────────── */}
      {activeTab === 'overview' && (
        <>
          {/* Bookmaker quick status */}
          <section className="sys-section">
            <h2>Bookmakers</h2>
            <div className="bm-quick-grid">
              {bookmakers.map(book => {
                const health = getBookmakerHealth(book);
                return (
                  <div key={book.name} className={`bm-quick-card bm-${health}`}>
                    <div className="bm-quick-indicator" />
                    <div className="bm-quick-info">
                      <span className="bm-quick-name">{book.name}</span>
                      <span className="bm-quick-time">{getTimeSince(book.last_scraped_at)}</span>
                    </div>
                    <span className="bm-quick-count">{book.snapshot_count.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {/* DB Metrics */}
          <section className="sys-section">
            <h2>Database</h2>
            <div className="metrics-grid">
              {Object.entries(status.counts).map(([key, value]) => (
                <div key={key} className="metric-card">
                  <div className="metric-value">{value.toLocaleString()}</div>
                  <div className="metric-label">{key.replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Recent runs combined */}
          <section className="sys-section">
            <h2>Recent Activity</h2>
            <div className="runs-combined">
              {status.last_scrape_runs && status.last_scrape_runs.length > 0 && (
                <div className="runs-group">
                  <h3>Scrape Runs</h3>
                  <table className="sys-table">
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {status.last_scrape_runs.slice(0, 5).map((run, idx) => (
                        <tr key={idx}>
                          <td>{run.task_name}</td>
                          <td><span className={`sys-badge sys-badge-${run.status}`}>{run.status}</span></td>
                          <td className="cell-time">{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                          <td>{run.duration_seconds?.toFixed(1)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {status.last_automation_runs && status.last_automation_runs.length > 0 && (
                <div className="runs-group">
                  <h3>Automation Runs</h3>
                  <table className="sys-table">
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {status.last_automation_runs.slice(0, 5).map((run, idx) => (
                        <tr key={idx}>
                          <td>{run.task_name}</td>
                          <td><span className={`sys-badge sys-badge-${run.status}`}>{run.status}</span></td>
                          <td className="cell-time">{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                          <td>{run.duration_seconds?.toFixed(1)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        </>
      )}

      {/* ─── Bookmakers Tab ───────────────────────────────────── */}
      {activeTab === 'bookmakers' && (
        <section className="sys-section">
          <div className="bm-detail-grid">
            {bookmakers.map(book => {
              const health = getBookmakerHealth(book);
              const hoursSince = book.last_scraped_at
                ? (Date.now() - new Date(book.last_scraped_at).getTime()) / 3600000
                : null;
              return (
                <div key={book.name} className={`bm-detail-card bm-detail-${health}`}>
                  <div className="bm-detail-header">
                    <div className="bm-detail-status">
                      <span className={`health-dot health-dot-${health}`} />
                      <h3>{book.name}</h3>
                    </div>
                    <span className={`sys-badge sys-badge-${health === 'healthy' ? 'success' : health === 'warning' ? 'warning' : health === 'critical' ? 'error' : 'unknown'}`}>
                      {health === 'healthy' ? 'OK' : health === 'warning' ? 'STALE' : health === 'critical' ? 'DOWN' : 'NEVER'}
                    </span>
                  </div>
                  <div className="bm-detail-body">
                    <div className="bm-detail-row">
                      <span className="bm-detail-label">Snapshots</span>
                      <span className="bm-detail-value">{book.snapshot_count.toLocaleString()}</span>
                    </div>
                    <div className="bm-detail-row">
                      <span className="bm-detail-label">Last Scrape</span>
                      <span className="bm-detail-value">
                        {book.last_scraped_at
                          ? new Date(book.last_scraped_at).toLocaleString()
                          : 'Never'}
                      </span>
                    </div>
                    <div className="bm-detail-row">
                      <span className="bm-detail-label">Time Since</span>
                      <span className={`bm-detail-value ${health !== 'healthy' ? 'value-warn' : ''}`}>
                        {getTimeSince(book.last_scraped_at)}
                      </span>
                    </div>
                    {hoursSince !== null && hoursSince > 6 && (
                      <div className="bm-detail-alert">
                        {hoursSince > 48
                          ? `⚠ Last scrape was ${Math.round(hoursSince / 24)} days ago — scraper may be broken`
                          : `⏳ Last scrape was ${Math.round(hoursSince)}h ago — may be falling behind`}
                      </div>
                    )}
                    {book.snapshot_count === 0 && (
                      <div className="bm-detail-alert">
                        ⚠ No snapshots — this bookmaker has never been scraped
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ─── Scheduler Tab ────────────────────────────────────── */}
      {activeTab === 'scheduler' && (
        <>
          {/* Zombie run alerts */}
          {runs.filter(isZombieRun).map(run => (
            <div key={run.id} className="sys-alert sys-alert-critical">
              <span className="sys-alert-icon">🧟</span>
              <div>
                <strong>Zombie Run Detected</strong>
                <p>Run #{run.id} ({run.task_name}) has been running since {run.started_at ? new Date(run.started_at).toLocaleString() : 'unknown'}. This is likely stuck.</p>
              </div>
            </div>
          ))}

          {/* Error run alerts */}
          {runs.filter(r => r.status === 'error').slice(0, 3).map(run => (
            <div key={run.id} className="sys-alert sys-alert-warning">
              <span className="sys-alert-icon">⚠</span>
              <div>
                <strong>Failed Run</strong>
                <p>Run #{run.id} ({run.task_name}) failed{run.error ? `: ${run.error}` : ''}</p>
              </div>
            </div>
          ))}

          {/* Registered tasks */}
          <section className="sys-section">
            <h2>Tasks</h2>
            <div className="tasks-grid">
              {tasks.map(task => (
                <div key={task.task_id} className={`task-card ${task.enabled ? '' : 'task-disabled'}`}>
                  <div className="task-header">
                    <div>
                      <h3>{task.task_id}</h3>
                      <span className="task-schedule">{task.schedule}</span>
                    </div>
                    <button
                      onClick={() => handleTrigger(task.task_id)}
                      className="sys-btn sys-btn-small"
                      disabled={!task.enabled}
                    >
                      ▶ Trigger
                    </button>
                  </div>
                  {task.description && (
                    <p className="task-description">{task.description}</p>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* Active jobs */}
          <section className="sys-section">
            <h2>Active Jobs</h2>
            {jobs.length === 0 ? (
              <p className="sys-empty">No active jobs</p>
            ) : (
              <table className="sys-table">
                <thead>
                  <tr>
                    <th>Job</th>
                    <th>Next Run</th>
                    <th>Trigger</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(job => (
                    <tr key={job.id}>
                      <td className="cell-mono">{job.id}</td>
                      <td className="cell-time">
                        {job.next_run_time
                          ? new Date(job.next_run_time).toLocaleString()
                          : '—'}
                      </td>
                      <td className="cell-mono">{job.trigger}</td>
                      <td>
                        {job.is_running
                          ? <span className="sys-badge sys-badge-running">Running</span>
                          : job.pending
                            ? <span className="sys-badge sys-badge-warning">Pending</span>
                            : <span className="sys-badge sys-badge-success">Idle</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* Recent runs */}
          <section className="sys-section">
            <h2>Recent Runs</h2>
            {runs.length === 0 ? (
              <p className="sys-empty">No runs recorded yet</p>
            ) : (
              <table className="sys-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Task</th>
                    <th>Status</th>
                    <th>Started</th>
                    <th>Duration</th>
                    <th>Error</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => (
                    <tr key={run.id} className={`run-row ${isZombieRun(run) ? 'run-zombie' : ''} ${selectedRunId === run.id ? 'run-selected' : ''}`}>
                      <td className="cell-mono">#{run.id}</td>
                      <td>{run.task_name}</td>
                      <td>
                        <span className={`sys-badge sys-badge-${run.status}`}>
                          {isZombieRun(run) ? '🧟 Zombie' : run.status}
                        </span>
                      </td>
                      <td className="cell-time">{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                      <td>
                        {run.duration_seconds != null
                          ? `${run.duration_seconds.toFixed(1)}s`
                          : '—'}
                      </td>
                      <td className="cell-error">
                        {run.error ? run.error.slice(0, 80) : '—'}
                      </td>
                      <td>
                        <button
                          onClick={() => handleViewCommands(run.id)}
                          className="sys-btn sys-btn-ghost sys-btn-small"
                        >
                          {selectedRunId === run.id ? '✕' : '⋯'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {selectedRunId && selectedRunCommands.length > 0 && (
              <div className="commands-panel">
                <h3>Commands — Run #{selectedRunId}</h3>
                <table className="sys-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Command</th>
                      <th>Status</th>
                      <th>Duration</th>
                      <th>Output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedRunCommands.map(cmd => (
                      <tr key={cmd.id} className={cmd.status === 'error' ? 'cmd-error' : ''}>
                        <td>{cmd.step_order}</td>
                        <td className="cell-mono">{cmd.command}</td>
                        <td><span className={`sys-badge sys-badge-${cmd.status}`}>{cmd.status}</span></td>
                        <td>{cmd.duration_seconds != null ? `${cmd.duration_seconds.toFixed(1)}s` : '—'}</td>
                        <td className="cell-output">
                          {cmd.error
                            ? <span className="cell-error">{cmd.error.slice(0, 150)}</span>
                            : cmd.output
                              ? cmd.output.slice(0, 150)
                              : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
