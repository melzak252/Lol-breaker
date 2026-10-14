import React, { useState, useEffect } from 'react';
import {
  fetchAlertConfig,
  updateAlertConfig,
  fetchAlertHistory,
  triggerAlertCheck,
  triggerAlertTest,
} from '../api/client';
import type {
  AlertConfigResponse,
  AlertHistoryResponse,
  AlertCheckResponse,
} from '../types';
import './AlertsCenter.css';

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export default function AlertsCenter() {
  const [config, setConfig] = useState<AlertConfigResponse | null>(null);
  const [history, setHistory] = useState<AlertHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testingChannel, setTestingChannel] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [banner, setBanner] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [scanResult, setScanResult] = useState<AlertCheckResponse | null>(null);

  // Form local state
  const [minEvPct, setMinEvPct] = useState<number>(5.0);
  const [minOdds, setMinOdds] = useState<number>(1.25);
  const [maxOdds, setMaxOdds] = useState<number>(12.0);
  const [cooldownHours, setCooldownHours] = useState<number>(6.0);
  const [discordWebhookUrl, setDiscordWebhookUrl] = useState('');
  const [telegramToken, setTelegramToken] = useState('');
  const [telegramChatId, setTelegramChatId] = useState('');

  const loadData = async () => {
    setLoading(true);
    try {
      const [cfg, hist] = await Promise.all([
        fetchAlertConfig(),
        fetchAlertHistory(50),
      ]);
      setConfig(cfg);
      setHistory(hist);
      setMinEvPct(Number((cfg.min_ev * 100).toFixed(1)));
      setMinOdds(cfg.min_odds);
      setMaxOdds(cfg.max_odds ?? 12.0);
      setCooldownHours(cfg.cooldown_hours);
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd wczytywania konfiguracji alertów: ${getErrorMessage(err)}` });
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSaveThresholds = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const updated = await updateAlertConfig({
        min_ev: minEvPct / 100.0,
        min_odds: minOdds,
        max_odds: maxOdds,
        cooldown_hours: cooldownHours,
      });
      setConfig(updated);
      setBanner({ type: 'success', message: 'Parametry progowe silnika zostały pomyślnie zaktualizowane.' });
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd zapisu progów: ${getErrorMessage(err)}` });
      setSaving(false);
    }
  };

  const handleToggleGlobal = async () => {
    if (!config) return;
    setSaving(true);
    try {
      const updated = await updateAlertConfig({ is_enabled: !config.is_enabled });
      setConfig(updated);
      setBanner({
        type: 'info',
        message: updated.is_enabled ? 'Powiadomienia Value Bet zostały włączone.' : 'Powiadomienia Value Bet zostały wstrzymane.',
      });
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd zmiany statusu: ${getErrorMessage(err)}` });
      setSaving(false);
    }
  };

  const handleSaveDiscord = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!discordWebhookUrl.trim()) return;
    setSaving(true);
    try {
      const updated = await updateAlertConfig({ discord_webhook_url: discordWebhookUrl.trim() });
      setConfig(updated);
      setDiscordWebhookUrl('');
      setBanner({ type: 'success', message: 'Nowy Discord Webhook URL został zapisany i zamaskowany.' });
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd zapisu Discord Webhook: ${getErrorMessage(err)}` });
      setSaving(false);
    }
  };

  const handleSaveTelegram = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!telegramToken.trim() && !telegramChatId.trim()) return;
    setSaving(true);
    try {
      const updated = await updateAlertConfig({
        telegram_bot_token: telegramToken.trim() || undefined,
        telegram_chat_id: telegramChatId.trim() || undefined,
      });
      setConfig(updated);
      setTelegramToken('');
      setTelegramChatId('');
      setBanner({ type: 'success', message: 'Dane bota Telegram zostały zaktualizowane.' });
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd zapisu Telegram: ${getErrorMessage(err)}` });
      setSaving(false);
    }
  };

  const handleToggleChannel = async (channel: 'discord' | 'telegram') => {
    if (!config) return;
    setSaving(true);
    try {
      const payload = channel === 'discord'
        ? { discord_enabled: !config.discord_enabled }
        : { telegram_enabled: !config.telegram_enabled };
      const updated = await updateAlertConfig(payload);
      setConfig(updated);
      setBanner({ type: 'info', message: `Kanał ${channel.toUpperCase()} został ${payload[channel === 'discord' ? 'discord_enabled' : 'telegram_enabled'] ? 'włączony' : 'wyłączony'}.` });
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd przełączania kanału: ${getErrorMessage(err)}` });
      setSaving(false);
    }
  };

  const handleTestAlert = async (channel: string) => {
    setTestingChannel(channel);
    try {
      const res = await triggerAlertTest(channel);
      const chRes = res.results[channel] || res.results;
      if (chRes && chRes.ok === false) {
        setBanner({ type: 'error', message: `Test ${channel}: ${chRes.error || 'Niepowodzenie'}` });
      } else {
        setBanner({ type: 'success', message: `Pomyślnie wysłano testowe powiadomienie na kanał: ${channel.toUpperCase()}!` });
      }
      const hist = await fetchAlertHistory(50);
      setHistory(hist);
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd podczas testu ${channel}: ${getErrorMessage(err)}` });
      setTestingChannel(null);
    }
  };

  const handleScan = async (dryRun: boolean) => {
    setScanning(true);
    setScanResult(null);
    try {
      const res = await triggerAlertCheck(dryRun);
      setScanResult(res);
      setBanner({
        type: res.failed > 0 ? 'error' : 'success',
        message: res.message,
      });
      const hist = await fetchAlertHistory(50);
      setHistory(hist);
    } catch (err: unknown) {
      setBanner({ type: 'error', message: `Błąd skanowania sygnałów: ${getErrorMessage(err)}` });
      setScanning(false);
    }
  };

  if (loading && !config) {
    return (
      <div className="alerts-center">
        <div className="empty-state">Ładowanie konfiguracji centrum alertów...</div>
      </div>
    );
  }

  return (
    <div className="alerts-center">
      {/* Header */}
      <div className="alerts-header">
        <div>
          <h1>🔥 Centrum Alertów Value Bet</h1>
          <p>Automatyczne powiadomienia Discord i Telegram o zyskownych kursach (EV+) po scrapingu bukmacherskim.</p>
        </div>
        <div className="alerts-actions">
          <button
            className="btn-primary"
            onClick={() => handleScan(false)}
            disabled={scanning || saving}
          >
            {scanning ? '⏳ Skanowanie...' : '⚡ Skanuj & Wyślij (Live)'}
          </button>
          <button
            className="btn-secondary"
            onClick={() => handleScan(true)}
            disabled={scanning || saving}
          >
            🧪 Symulacja (Dry Run)
          </button>
          <button
            className="btn-outline"
            onClick={loadData}
            disabled={loading}
          >
            🔄 Odśwież
          </button>
        </div>
      </div>

      {/* Banner */}
      {banner && (
        <div className={`banner banner-${banner.type}`}>
          <span>{banner.message}</span>
          <button
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: '1.1rem' }}
            onClick={() => setBanner(null)}
          >
            ✕
          </button>
        </div>
      )}

      {/* Scan Summary Banner if present */}
      {scanResult && (
        <div className="banner banner-info" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
          <div style={{ fontWeight: 600 }}>📋 Raport z ostatniego skanowania:</div>
          <div>
            Wysłano: <strong>{scanResult.dispatched}</strong> | Pominięto (cooldown): <strong>{scanResult.skipped}</strong> | Błędów: <strong>{scanResult.failed}</strong>
          </div>
          {scanResult.alerts.length > 0 && (
            <div style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
              Zidentyfikowane okazje: {scanResult.alerts.map((a, idx) => (
                <span key={idx} style={{ marginRight: '1rem', background: 'rgba(255,255,255,0.08)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                  {a.match} ({a.team} @ {a.odds.toFixed(2)} - EV +{(a.ev * 100).toFixed(1)}%)
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Cards Grid */}
      <div className="cards-grid">
        {/* Card 1: Engine Rules */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">⚙️ Reguły Wykrywania EV</h3>
            <span
              className={`badge ${config?.is_enabled ? 'badge-active' : 'badge-danger'}`}
              style={{ cursor: 'pointer' }}
              onClick={handleToggleGlobal}
              title="Kliknij, aby przełączyć status globalny"
            >
              {config?.is_enabled ? 'Silnik Aktywny' : 'Wstrzymany'}
            </span>
          </div>

          <form onSubmit={handleSaveThresholds}>
            <div className="form-group">
              <label>Minimalna Wartość Oczekiwana (EV % po podatku 12%):</label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="50"
                className="form-control"
                value={minEvPct}
                onChange={(e) => setMinEvPct(parseFloat(e.target.value) || 0)}
              />
              <small style={{ color: '#64748b', fontSize: '0.78rem' }}>
                Rekomendowane: 5.0% (po odliczeniu podatku obrotowego)
              </small>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="form-group">
                <label>Min Kurs:</label>
                <input
                  type="number"
                  step="0.05"
                  min="1.01"
                  max="10"
                  className="form-control"
                  value={minOdds}
                  onChange={(e) => setMinOdds(parseFloat(e.target.value) || 1.01)}
                />
              </div>
              <div className="form-group">
                <label>Max Kurs:</label>
                <input
                  type="number"
                  step="0.5"
                  min="1.5"
                  max="50"
                  className="form-control"
                  value={maxOdds}
                  onChange={(e) => setMaxOdds(parseFloat(e.target.value) || 12.0)}
                />
              </div>
            </div>

            <div className="form-group">
              <label>Cooldown powiadomień (godziny):</label>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="48"
                className="form-control"
                value={cooldownHours}
                onChange={(e) => setCooldownHours(parseFloat(e.target.value) || 1.0)}
              />
              <small style={{ color: '#64748b', fontSize: '0.78rem' }}>
                Chroni przed spamem tego samego meczu i bukmachera
              </small>
            </div>

            <button
              type="submit"
              className="btn-primary"
              style={{ width: '100%', marginTop: '0.5rem' }}
              disabled={saving}
            >
              {saving ? 'Zapisywanie...' : 'Zapisz Parametry Silnika'}
            </button>
          </form>
        </div>

        {/* Card 2: Discord Webhook */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">💬 Kanał Discord</h3>
            <span className={`badge ${config?.discord_configured ? 'badge-active' : 'badge-warning'}`}>
              {config?.discord_configured ? 'Skonfigurowany' : 'Brak URL'}
            </span>
          </div>

          <div className="toggle-row">
            <span className="toggle-label">Wysyłaj na Discord:</span>
            <button
              type="button"
              className={`badge ${config?.discord_enabled ? 'badge-active' : 'badge-inactive'}`}
              onClick={() => handleToggleChannel('discord')}
              style={{ cursor: 'pointer', border: 'none' }}
            >
              {config?.discord_enabled ? 'WŁĄCZONE' : 'WYŁĄCZONE'}
            </button>
          </div>

          {config?.discord_webhook_url_masked && (
            <div style={{ marginBottom: '1rem', fontSize: '0.85rem', color: '#94a3b8' }}>
              Aktywny Webhook: <code style={{ color: '#38bdf8' }}>{config.discord_webhook_url_masked}</code>
            </div>
          )}

          <form onSubmit={handleSaveDiscord}>
            <div className="form-group">
              <label>Nowy Webhook URL:</label>
              <input
                type="password"
                placeholder="https://discord.com/api/webhooks/..."
                className="form-control"
                value={discordWebhookUrl}
                onChange={(e) => setDiscordWebhookUrl(e.target.value)}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                type="submit"
                className="btn-secondary"
                style={{ flex: 1 }}
                disabled={saving || !discordWebhookUrl.trim()}
              >
                Zapisz Webhook
              </button>
              <button
                type="button"
                className="btn-outline"
                onClick={() => handleTestAlert('discord')}
                disabled={testingChannel !== null || !config?.discord_configured}
              >
                {testingChannel === 'discord' ? 'Wysyłanie...' : '🧪 Test Discord'}
              </button>
            </div>
          </form>
        </div>

        {/* Card 3: Telegram Bot */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">✈️ Kanał Telegram</h3>
            <span className={`badge ${config?.telegram_configured ? 'badge-active' : 'badge-warning'}`}>
              {config?.telegram_configured ? 'Skonfigurowany' : 'Brak Danych'}
            </span>
          </div>

          <div className="toggle-row">
            <span className="toggle-label">Wysyłaj na Telegram:</span>
            <button
              type="button"
              className={`badge ${config?.telegram_enabled ? 'badge-active' : 'badge-inactive'}`}
              onClick={() => handleToggleChannel('telegram')}
              style={{ cursor: 'pointer', border: 'none' }}
            >
              {config?.telegram_enabled ? 'WŁĄCZONE' : 'WYŁĄCZONE'}
            </button>
          </div>

          {config?.telegram_chat_id_masked && (
            <div style={{ marginBottom: '1rem', fontSize: '0.85rem', color: '#94a3b8' }}>
              Chat ID: <code style={{ color: '#38bdf8' }}>{config.telegram_chat_id_masked}</code>
            </div>
          )}

          <form onSubmit={handleSaveTelegram}>
            <div className="form-group">
              <label>Bot Token:</label>
              <input
                type="password"
                placeholder="123456789:ABCDefGhIJKlmno..."
                className="form-control"
                value={telegramToken}
                onChange={(e) => setTelegramToken(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Chat ID / Kanał ID:</label>
              <input
                type="text"
                placeholder="-1001234567890 lub @twoj_kanal"
                className="form-control"
                value={telegramChatId}
                onChange={(e) => setTelegramChatId(e.target.value)}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                type="submit"
                className="btn-secondary"
                style={{ flex: 1 }}
                disabled={saving || (!telegramToken.trim() && !telegramChatId.trim())}
              >
                Zapisz Dane
              </button>
              <button
                type="button"
                className="btn-outline"
                onClick={() => handleTestAlert('telegram')}
                disabled={testingChannel !== null || !config?.telegram_configured}
              >
                {testingChannel === 'telegram' ? 'Wysyłanie...' : '🧪 Test Telegram'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* History Audit Table */}
      <div className="history-section">
        <div className="history-header">
          <h2>📜 Dziennik Wysłanych Powiadomień (Audit Trail)</h2>
          <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Łącznie: {history?.total ?? 0} wpisów
          </span>
        </div>

        {(!history || history.alerts.length === 0) ? (
          <div className="empty-state">
            Brak zarejestrowanych powiadomień w historii. Użyj przycisku &quot;Skanuj &amp; Wyślij (Live)&quot; lub wykonaj test połączenia.
          </div>
        ) : (
          <div className="table-responsive">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Czas wysłania</th>
                  <th>Status</th>
                  <th>Mecz &amp; Liga</th>
                  <th>Typ</th>
                  <th>Bukmacher</th>
                  <th>Kurs</th>
                  <th>EV (%)</th>
                  <th>Stawka Kelly</th>
                  <th>Kanały</th>
                  <th>Uwagi / Błędy</th>
                </tr>
              </thead>
              <tbody>
                {history.alerts.map((entry) => (
                  <tr key={entry.id}>
                    <td style={{ whiteSpace: 'nowrap', fontSize: '0.82rem', color: '#94a3b8' }}>
                      {entry.created_at.replace('T', ' ').substring(0, 19)}
                    </td>
                    <td>
                      <span className={`badge ${
                        entry.status === 'sent' ? 'badge-active' :
                        entry.status === 'simulated' ? 'badge-info' :
                        entry.status === 'test' ? 'badge-warning' :
                        'badge-danger'
                      }`}>
                        {entry.status}
                      </span>
                    </td>
                    <td>
                      <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{entry.match_label}</div>
                      <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{entry.league || 'LoL'}</div>
                    </td>
                    <td>
                      <span style={{ fontWeight: 600 }}>{entry.team_name}</span>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', marginLeft: '0.35rem' }}>
                        ({(entry.side || '').toUpperCase()})
                      </span>
                    </td>
                    <td>{entry.bookmaker_name}</td>
                    <td>
                      <span className="odds-pill">{entry.odds ? entry.odds.toFixed(2) : '-'}</span>
                    </td>
                    <td>
                      <span className="ev-pill">
                        {entry.ev !== null ? `+${(entry.ev * 100).toFixed(1)}%` : '-'}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.85rem' }}>
                      {entry.suggested_stake ? `${entry.suggested_stake.toFixed(1)}%` : '-'}
                    </td>
                    <td style={{ fontSize: '0.82rem', color: '#38bdf8' }}>
                      {entry.channels}
                    </td>
                    <td style={{ fontSize: '0.78rem', color: entry.error_message ? '#f87171' : '#64748b', maxWidth: '240px' }}>
                      {entry.error_message || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
