import { useEffect, useState } from 'react';
import {
  fetchSchedulerTasks,
  fetchSchedulerJobs,
  fetchSchedulerRuns,
  fetchSchedulerRunCommands,
  triggerSchedulerTask,
} from '../api/client';
import type { SchedulerTask, SchedulerJob, SchedulerRun, SchedulerCommand } from '../types';
import './SchedulerPanel.css';

export default function SchedulerPanel() {
  const [tasks, setTasks] = useState<SchedulerTask[]>([]);
  const [jobs, setJobs] = useState<SchedulerJob[]>([]);
  const [runs, setRuns] = useState<SchedulerRun[]>([]);
  const [selectedRunCommands, setSelectedRunCommands] = useState<SchedulerCommand[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [triggerMessage, setTriggerMessage] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const [tasksData, jobsData, runsData] = await Promise.all([
        fetchSchedulerTasks(),
        fetchSchedulerJobs(),
        fetchSchedulerRuns(),
      ]);
      setTasks(tasksData);
      setJobs(jobsData);
      setRuns(runsData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scheduler data');
    } finally {
      setLoading(false);
    }
  }

  async function handleTrigger(taskId: string) {
    try {
      const result = await triggerSchedulerTask(taskId);
      setTriggerMessage(`Triggered ${taskId}: ${result.status}`);
      setTimeout(() => setTriggerMessage(null), 5000);
      // Reload runs after a short delay
      setTimeout(loadData, 2000);
    } catch (err) {
      setTriggerMessage(`Error: ${err instanceof Error ? err.message : 'Failed'}`);
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
      setTriggerMessage(`Error loading commands: ${err instanceof Error ? err.message : 'Failed'}`);
    }
  }

  if (loading) return <div className="loading">Loading scheduler data...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div className="scheduler-panel">
      <h1>Scheduler Panel</h1>

      {triggerMessage && (
        <div className="action-message">{triggerMessage}</div>
      )}

      <button onClick={loadData} className="btn btn-secondary">
        Refresh
      </button>

      <section className="tasks-section">
        <h2>Registered Tasks</h2>
        <div className="tasks-grid">
          {tasks.map((task) => (
            <div key={task.task_id} className="task-card">
              <div className="task-header">
                <h3>{task.task_id}</h3>
                <button
                  onClick={() => handleTrigger(task.task_id)}
                  className="btn btn-small"
                >
                  Trigger
                </button>
              </div>
              <div className="task-details">
                <span className="task-schedule">{task.schedule}</span>
                {task.description && (
                  <p className="task-description">{task.description}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="jobs-section">
        <h2>Active Jobs (APScheduler)</h2>
        {jobs.length === 0 ? (
          <p className="empty-state">No active jobs</p>
        ) : (
          <table className="jobs-table">
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Next Run</th>
                <th>Trigger</th>
                <th>Pending</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id}>
                  <td>{job.id}</td>
                  <td>
                    {job.next_run_time
                      ? new Date(job.next_run_time).toLocaleString()
                      : '—'}
                  </td>
                  <td>{job.trigger}</td>
                  <td>{job.pending ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="runs-section">
        <h2>Recent Runs</h2>
        {runs.length === 0 ? (
          <p className="empty-state">No runs recorded yet</p>
        ) : (
          <table className="runs-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Started</th>
                <th>Finished</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id} className={selectedRunId === run.id ? 'selected' : ''}>
                  <td>{run.task_name}</td>
                  <td>
                    <span className={`status-badge ${run.status}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>{run.started_at ? new Date(run.started_at).toLocaleString() : '—'}</td>
                  <td>
                    {run.finished_at
                      ? new Date(run.finished_at).toLocaleString()
                      : '—'}
                  </td>
                  <td>
                    {run.duration_seconds != null
                      ? `${run.duration_seconds.toFixed(1)}s`
                      : '—'}
                  </td>
                  <td>
                    <button
                      onClick={() => handleViewCommands(run.id)}
                      className="btn btn-small"
                    >
                      {selectedRunId === run.id ? 'Hide' : 'Commands'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {selectedRunId && selectedRunCommands.length > 0 && (
          <div className="commands-detail">
            <h3>Commands for Run #{selectedRunId}</h3>
            <table className="commands-table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Command</th>
                  <th>Status</th>
                  <th>Duration</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {selectedRunCommands.map((cmd) => (
                  <tr key={cmd.id}>
                    <td>{cmd.step_order}</td>
                    <td className="command-text">{cmd.command}</td>
                    <td>
                      <span className={`status-badge ${cmd.status}`}>
                        {cmd.status}
                      </span>
                    </td>
                    <td>
                      {cmd.duration_seconds != null
                        ? `${cmd.duration_seconds.toFixed(1)}s`
                        : '—'}
                    </td>
                    <td className="command-output">
                      {cmd.output ? cmd.output.slice(0, 200) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
