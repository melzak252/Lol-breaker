import { useEffect, useState } from 'react';
import { fetchSystemStatus, fetchBookmakers, triggerLightCycle, triggerBackup } from '../api/client';
import type { SystemStatusResponse, BookmakerStatus } from '../types';
import './SystemStatus.css';

export default function SystemStatus() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [bookmakers, setBookmakers] = useState<BookmakerStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const [statusData, bookmakersData] = await Promise.all([
        fetchSystemStatus(),
        fetchBookmakers(),
      ]);
      setStatus(statusData);
      setBookmakers(bookmakersData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load system status');
    } finally {
      setLoading(false);
    }
  }

  async function handleLightCycle() {
    try {
      const result = await triggerLightCycle();
      setActionMessage(`Light cycle: ${result.message}`);
      setTimeout(() => setActionMessage(null), 5000);
      loadData();
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

  if (loading) return <div className="loading">Loading system status...</div>;
  if (error) return <div className="error">Error: {error}</div>;
  if (!status) return <div className="error">No data</div>;

  return (
    <div className="system-status">
      <h1>System Status</h1>

      {actionMessage && (
        <div className="action-message">{actionMessage}</div>
      )}

      <div className="actions">
        <button onClick={handleLightCycle} className="btn">
          Run Light Cycle
        </button>
        <button onClick={handleBackup} className="btn">
          Run Backup
        </button>
        <button onClick={loadData} className="btn btn-secondary">
          Refresh
        </button>
      </div>

      <section className="metrics">
        <h2>Database Metrics</h2>
        <div className="metrics-grid">
          {Object.entries(status.counts).map(([key, value]) => (
            <div key={key} className="metric-card">
              <div className="metric-label">{key.replace(/_/g, ' ')}</div>
              <div className="metric-value">{value.toLocaleString()}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="scrapers">
        <h2>Scraper Status</h2>
        <div className="bookmakers-table">
          {bookmakers.map((book) => (
            <div key={book.name} className="bookmaker-row">
              <div className="bookmaker-name">{book.name}</div>
              <div className="bookmaker-status">
                {book.last_scrape ? (
                  <>
                    <span className="status-badge success">✓</span>
                    <span className="scrape-time">
                      {new Date(book.last_scrape).toLocaleString()}
                    </span>
                  </>
                ) : (
                  <span className="status-badge warning">No data</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {status.last_scrape_runs && status.last_scrape_runs.length > 0 && (
        <section className="recent-runs">
          <h2>Recent Scrape Runs</h2>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Started</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {status.last_scrape_runs.slice(0, 10).map((run, idx) => (
                <tr key={idx}>
                  <td>{run.task_name}</td>
                  <td>
                    <span className={`status-badge ${run.status}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                  <td>{run.duration_seconds?.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {status.last_automation_runs && status.last_automation_runs.length > 0 && (
        <section className="recent-runs">
          <h2>Recent Automation Runs</h2>
          <table className="runs-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Started</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {status.last_automation_runs.slice(0, 10).map((run, idx) => (
                <tr key={idx}>
                  <td>{run.task_name}</td>
                  <td>
                    <span className={`status-badge ${run.status}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                  <td>{run.duration_seconds?.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
