/* TimescaleDB initialisation — tables ordered by FK dependencies. */

CREATE EXTENSION IF NOT EXISTS timescaledb;

/* ── Zero-dependency tables ──────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS bookmakers (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100) UNIQUE NOT NULL,
    base_url   VARCHAR(500),
    is_active  INTEGER NOT NULL DEFAULT 1
);
INSERT INTO bookmakers(name, base_url) VALUES
    ('manual',NULL),('sts','https://www.sts.pl/'),('betclic','https://www.betclic.pl/'),
    ('superbet','https://superbet.pl/'),('efortuna','https://www.efortuna.pl/'),
    ('fortuna','https://www.efortuna.pl/'),('betfan','https://betfan.pl/'),
    ('totalbet','https://totalbet.pl/'),('lebull','https://www.lebull.pl/')
ON CONFLICT (name) DO NOTHING;

CREATE TABLE IF NOT EXISTS canonical_matches (
    id                    SERIAL PRIMARY KEY,
    canonical_key         VARCHAR(100) UNIQUE NOT NULL,
    team_a_name           VARCHAR(200) NOT NULL,
    team_b_name           VARCHAR(200) NOT NULL,
    normalized_team_a     VARCHAR(200) NOT NULL,
    normalized_team_b     VARCHAR(200) NOT NULL,
    start_time_normalized VARCHAR(50),
    league                VARCHAR(100),
    best_of               SMALLINT DEFAULT 1,
    status                VARCHAR(50) DEFAULT 'upcoming',
    winner_name           VARCHAR(200),
    loser_name            VARCHAR(200),
    winner_normalized     VARCHAR(200),
    winner_side           VARCHAR(20),
    result_source         VARCHAR(50),
    result_source_match_id VARCHAR(50),
    result_recorded_at    TIMESTAMPTZ,
    match_confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS golgg_teams (
    id              SERIAL PRIMARY KEY,
    team_name       VARCHAR(200) UNIQUE NOT NULL,
    team_id         INTEGER UNIQUE,
    normalized_name VARCHAR(200)
);

CREATE TABLE IF NOT EXISTS golgg_matches (
    match_id        VARCHAR(50) PRIMARY KEY,
    date            VARCHAR(20),
    tournament_name VARCHAR(300),
    patch           VARCHAR(20),
    team1_name      VARCHAR(200), team2_name      VARCHAR(200),
    team1_id        VARCHAR(50),  team2_id        VARCHAR(50),
    team1_score     INTEGER,      team2_score     INTEGER,
    team1_win       INTEGER,      team2_win       INTEGER,
    draw            INTEGER,
    games_played    INTEGER,      best_of         INTEGER,
    winner_name     VARCHAR(200), loser_name      VARCHAR(200),
    source_link     VARCHAR(500), raw_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_golgg_matches_date ON golgg_matches(date);
CREATE INDEX IF NOT EXISTS idx_golgg_matches_tournament ON golgg_matches(tournament_name);

CREATE TABLE IF NOT EXISTS golgg_games (
    game_id         VARCHAR(50) PRIMARY KEY,
    match_id        VARCHAR(50) NOT NULL REFERENCES golgg_matches(match_id),
    date            VARCHAR(20), tournament_name VARCHAR(300), patch VARCHAR(20),
    team1_name      VARCHAR(200), team2_name VARCHAR(200),
    team1_id        VARCHAR(50),  team2_id   VARCHAR(50),
    team1_win       INTEGER,      team2_win  INTEGER, draw INTEGER,
    team1_side      VARCHAR(10),  team2_side VARCHAR(10),
    game_duration   INTEGER,
    team1_stats_json TEXT, team2_stats_json TEXT, raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_golgg_games_match ON golgg_games(match_id);

CREATE TABLE IF NOT EXISTS golgg_game_players (
    id              SERIAL PRIMARY KEY,
    game_id         VARCHAR(50) NOT NULL REFERENCES golgg_games(game_id),
    match_id        VARCHAR(50) REFERENCES golgg_matches(match_id),
    team_id         VARCHAR(50), team_name VARCHAR(200),
    side            VARCHAR(5) NOT NULL,
    role            VARCHAR(20),
    player_id       VARCHAR(50), player_name VARCHAR(200),
    champion_id     VARCHAR(50), champion_name VARCHAR(100),
    champion_image  VARCHAR(500),
    stats_json      TEXT, raw_json TEXT,
    UNIQUE(game_id, side, role)
);
CREATE INDEX IF NOT EXISTS idx_golgg_game_players_player ON golgg_game_players(player_id);

CREATE TABLE IF NOT EXISTS team_aliases (
    id              SERIAL PRIMARY KEY,
    normalized_name VARCHAR(200) NOT NULL,
    alias           VARCHAR(200) NOT NULL,
    source          VARCHAR(255) NOT NULL,
    raw_name        VARCHAR(200),
    normalized_alias VARCHAR(200),
    source_system   VARCHAR(50),
    league_pattern  VARCHAR(200),
    tournament_pattern VARCHAR(200),
    valid_from      VARCHAR(20),
    valid_to        VARCHAR(20),
    confidence      DOUBLE PRECISION DEFAULT 1,
    is_active       INTEGER NOT NULL DEFAULT 1,
    is_blocked      INTEGER NOT NULL DEFAULT 0,
    review_status   VARCHAR(50) DEFAULT 'approved',
    notes           TEXT,
    created_at      VARCHAR(50) DEFAULT CURRENT_TIMESTAMP,
    updated_at      VARCHAR(50) DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(normalized_name, source)
);
CREATE INDEX IF NOT EXISTS idx_team_aliases_lookup ON team_aliases(normalized_name, is_active, source_system, league_pattern);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id                SERIAL PRIMARY KEY,
    model_name        VARCHAR(200) NOT NULL,
    model_version     VARCHAR(50) NOT NULL,
    artifact_path     VARCHAR(500),
    feature_schema_json TEXT, model_params_json TEXT,
    training_cutoff_at VARCHAR(50), metrics_json TEXT,
    status VARCHAR(50) DEFAULT 'registered',
    UNIQUE(model_name, model_version)
);

CREATE TABLE IF NOT EXISTS automation_runs (
    id                SERIAL PRIMARY KEY,
    run_type          VARCHAR(50) NOT NULL,
    trigger_source    VARCHAR(50) DEFAULT 'scheduler',
    status            VARCHAR(50) DEFAULT 'running',
    started_at        TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    interval_seconds  INTEGER, next_run_at TIMESTAMPTZ,
    host VARCHAR(200), pid INTEGER,
    commands_total INTEGER DEFAULT 0, commands_failed INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS rating_runs (
    id                SERIAL PRIMARY KEY,
    ratings_version   VARCHAR(100) UNIQUE NOT NULL,
    source            VARCHAR(100), data_cutoff_at VARCHAR(50),
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running',
    systems_json TEXT,
    matches_processed INTEGER DEFAULT 0, games_processed INTEGER DEFAULT 0,
    players_processed INTEGER DEFAULT 0, error TEXT
);

CREATE TABLE IF NOT EXISTS team_rolling_features (
    id                    SERIAL PRIMARY KEY,
    feature_version       VARCHAR(100) NOT NULL,
    team_name             VARCHAR(200) NOT NULL,
    normalized_team_name  VARCHAR(200) NOT NULL,
    window_size           INTEGER DEFAULT 20,
    data_cutoff_at        VARCHAR(50),
    matches_count INTEGER DEFAULT 0, games_count INTEGER DEFAULT 0,
    win_rate REAL, avg_kills REAL, avg_deaths REAL, avg_gd15 REAL,
    avg_dpm REAL, avg_vspm REAL, avg_gold REAL, avg_towers REAL,
    avg_dragons REAL, avg_nashors REAL, avg_game_duration REAL,
    features_json TEXT,
    UNIQUE(feature_version, normalized_team_name, window_size)
);

/* ── Single FK dependency ────────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS bookmaker_accounts (
    id               SERIAL PRIMARY KEY,
    bookmaker_id     INTEGER NOT NULL REFERENCES bookmakers(id),
    account_name     VARCHAR(100) NOT NULL,
    currency         VARCHAR(10) NOT NULL DEFAULT 'PLN',
    opening_balance  NUMERIC(12,2) NOT NULL DEFAULT 0,
    current_balance  NUMERIC(12,2) NOT NULL DEFAULT 0 CONSTRAINT chk_bookmaker_account_balance_non_negative CHECK (current_balance >= 0),
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(bookmaker_id, account_name)
);

/* ── GOL.GG → canonical match linking ───────────────────────────────────── */
CREATE TABLE IF NOT EXISTS golgg_match_mappings (
    id                  SERIAL PRIMARY KEY,
    canonical_match_id  INTEGER NOT NULL REFERENCES canonical_matches(id),
    golgg_match_id      VARCHAR(50) NOT NULL REFERENCES golgg_matches(match_id),
    confidence          REAL DEFAULT 1.0,
    mapped_by           VARCHAR(50) DEFAULT 'auto',
    mapped_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(canonical_match_id),
    UNIQUE(golgg_match_id)
);

CREATE TABLE IF NOT EXISTS upcoming_matches (
    id                  SERIAL PRIMARY KEY,
    bookmaker_id        INTEGER NOT NULL REFERENCES bookmakers(id),
    bookmaker_match_key VARCHAR(200) NOT NULL,
    canonical_match_id  INTEGER REFERENCES canonical_matches(id),
    raw_team_a VARCHAR(200) NOT NULL, raw_team_b VARCHAR(200) NOT NULL,
    normalized_team_a VARCHAR(200) NOT NULL, normalized_team_b VARCHAR(200) NOT NULL,
    match_start_time VARCHAR(50), league VARCHAR(100),
    source_url VARCHAR(500), offer_url VARCHAR(500),
    is_live INTEGER DEFAULT 0,
    last_seen_at TIMESTAMPTZ,
    team_a_golgg_id INTEGER REFERENCES golgg_teams(id),
    team_b_golgg_id INTEGER REFERENCES golgg_teams(id),
    UNIQUE(bookmaker_match_key)
    
);

/* ── Multi-FK tables ─────────────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS bets (
    id                    SERIAL PRIMARY KEY,
    signal_id             INTEGER,
    model_ev_signal_id    INTEGER,
    bookmaker_account_id  INTEGER REFERENCES bookmaker_accounts(id),
    canonical_match_id    INTEGER REFERENCES canonical_matches(id),
    placed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bookmaker_id          INTEGER,
    side                  VARCHAR(1) CHECK (side IN ('a','b')) NOT NULL,
    stake                 NUMERIC(12,2) NOT NULL,
    taken_odds            NUMERIC(8,4) NOT NULL,
    status                VARCHAR(20) NOT NULL DEFAULT 'open',
    result                VARCHAR(20),
    profit                NUMERIC(12,2) NOT NULL DEFAULT 0,
    settled_at            TIMESTAMPTZ,
    team_a VARCHAR(200), team_b VARCHAR(200), league VARCHAR(100),
    match_start_time TIMESTAMPTZ,
    model_prob NUMERIC(6,4), ev NUMERIC(10,4),
    tax_rate NUMERIC(4,2) NOT NULL DEFAULT 0.12,
    note TEXT, source VARCHAR(50) NOT NULL DEFAULT 'manual'
);
CREATE INDEX IF NOT EXISTS idx_bets_account ON bets(bookmaker_account_id, status, placed_at DESC);

CREATE TABLE IF NOT EXISTS bookmaker_wallet_transactions (
    id                    BIGSERIAL,
    bookmaker_account_id  INTEGER NOT NULL REFERENCES bookmaker_accounts(id),
    bet_id                INTEGER REFERENCES bets(id),
    transaction_time      TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    transaction_type      VARCHAR(50) NOT NULL,
    amount                NUMERIC(12,2) NOT NULL,
    balance_after         NUMERIC(12,2) NOT NULL,
    note                  TEXT
);
SELECT create_hypertable('bookmaker_wallet_transactions', 'transaction_time', if_not_exists => TRUE, migrate_data => TRUE);

/* ── Odds (time-series) ──────────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id                  BIGSERIAL,
    bookmaker_id        INTEGER NOT NULL REFERENCES bookmakers(id),
    match_id            INTEGER,
    canonical_match_id  INTEGER REFERENCES canonical_matches(id),
    market_type         VARCHAR(50) DEFAULT 'match_winner',
    raw_team_a VARCHAR(200), raw_team_b VARCHAR(200),
    odds_a DOUBLE PRECISION, odds_b DOUBLE PRECISION,
    is_live             INTEGER,
    scraped_at          TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    source_url VARCHAR(500), offer_url VARCHAR(500),
    raw_payload         TEXT
);
SELECT create_hypertable('odds_snapshots', 'scraped_at', if_not_exists => TRUE, migrate_data => TRUE);
CREATE INDEX IF NOT EXISTS idx_odds_bookmaker ON odds_snapshots(bookmaker_id, scraped_at DESC);
CREATE INDEX IF NOT EXISTS idx_odds_canonical ON odds_snapshots(canonical_match_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_unique_snapshot ON odds_snapshots(canonical_match_id, bookmaker_id, scraped_at) WHERE market_type = 'match_winner';

CREATE TABLE IF NOT EXISTS prop_odds_snapshots (
    id                  SERIAL PRIMARY KEY,
    bookmaker_id        INTEGER NOT NULL REFERENCES bookmakers(id),
    canonical_match_id  INTEGER REFERENCES canonical_matches(id),
    market_type         VARCHAR(50) NOT NULL,
    map_number          INTEGER NOT NULL DEFAULT 1,
    line                DOUBLE PRECISION NOT NULL,
    target_team         VARCHAR(200),
    raw_team_a          VARCHAR(200),
    raw_team_b          VARCHAR(200),
    odds_over           DOUBLE PRECISION,
    odds_under          DOUBLE PRECISION,
    prob_over_novig     DOUBLE PRECISION,
    prob_under_novig    DOUBLE PRECISION,
    margin              DOUBLE PRECISION,
    raw_market_name     VARCHAR(255),
    scraped_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_live             INTEGER DEFAULT 0,
    source_url          VARCHAR(500),
    outcomes_payload    TEXT
);
CREATE INDEX IF NOT EXISTS ix_prop_odds_match_market_time ON prop_odds_snapshots(canonical_match_id, market_type, scraped_at);
CREATE INDEX IF NOT EXISTS ix_prop_odds_line_search ON prop_odds_snapshots(canonical_match_id, market_type, line, scraped_at);

CREATE TABLE IF NOT EXISTS model_prop_predictions (
    id                      SERIAL PRIMARY KEY,
    canonical_match_id      INTEGER NOT NULL REFERENCES canonical_matches(id),
    map_number              INTEGER NOT NULL DEFAULT 1,
    model_name              VARCHAR(100) NOT NULL DEFAULT 'kill_spread_distribution',
    model_version           VARCHAR(50) DEFAULT 'prop-v1.0' NOT NULL,
    predicted_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expected_total_kills    DOUBLE PRECISION NOT NULL,
    mu_team_a               DOUBLE PRECISION NOT NULL,
    mu_team_b               DOUBLE PRECISION NOT NULL,
    expected_spread         DOUBLE PRECISION NOT NULL,
    league_avg_kills        DOUBLE PRECISION,
    pace_category           VARCHAR(50),
    distribution_json       TEXT,
    signals_json            TEXT
);
CREATE INDEX IF NOT EXISTS ix_model_prop_preds_match ON model_prop_predictions(canonical_match_id, map_number, predicted_at DESC);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id              SERIAL PRIMARY KEY,
    bookmaker_id    INTEGER REFERENCES bookmakers(id),
    scraper_name    VARCHAR(100) NOT NULL,
    scraper_version VARCHAR(50),
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running',
    source_url VARCHAR(500), request_url VARCHAR(500),
    items_seen INTEGER DEFAULT 0, items_inserted INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS bookmaker_events (
    id                  SERIAL PRIMARY KEY,
    bookmaker_id        INTEGER NOT NULL REFERENCES bookmakers(id),
    bookmaker_event_id  VARCHAR(100) NOT NULL,
    canonical_match_id  INTEGER REFERENCES canonical_matches(id),
    raw_team_a VARCHAR(200) NOT NULL, raw_team_b VARCHAR(200) NOT NULL,
    match_start_time VARCHAR(50),
    sport_id VARCHAR(20), sport_name VARCHAR(100),
    category_id VARCHAR(20), category_name VARCHAR(100),
    league_id VARCHAR(20), league_name VARCHAR(100),
    first_seen_at TIMESTAMPTZ, last_seen_at TIMESTAMPTZ,
    offer_url VARCHAR(500),
    UNIQUE(bookmaker_id, bookmaker_event_id)
);

CREATE TABLE IF NOT EXISTS bookmaker_markets (
    id                    BIGSERIAL PRIMARY KEY,
    bookmaker_event_id    VARCHAR(100) NOT NULL,
    bookmaker_market_key  VARCHAR(200) NOT NULL,
    market_name VARCHAR(200),
    market_type VARCHAR(50) DEFAULT 'match_winner',
    line_id VARCHAR(50), line_name VARCHAR(200),
    is_extra_market INTEGER DEFAULT 0,
    UNIQUE(bookmaker_event_id, bookmaker_market_key)
);

CREATE TABLE IF NOT EXISTS odds_outcome_snapshots (
    id                  BIGSERIAL,
    scrape_run_id       INTEGER REFERENCES scrape_runs(id),
    bookmaker_event_id  VARCHAR(100) NOT NULL,
    bookmaker_market_key VARCHAR(200) NOT NULL,
    outcome_key         VARCHAR(200) NOT NULL,
    scraped_at          TIMESTAMPTZ DEFAULT NOW(),
    source_url VARCHAR(500), offer_url VARCHAR(500),
    outcome_name VARCHAR(200), outcome_side VARCHAR(10),
    decimal_odds REAL, raw_payload TEXT
);
SELECT create_hypertable('odds_outcome_snapshots', 'scraped_at', if_not_exists => TRUE, migrate_data => TRUE);
CREATE INDEX IF NOT EXISTS idx_outcome_event ON odds_outcome_snapshots(bookmaker_event_id, scraped_at DESC);

/* ── Entity ratings (time-series) ────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS entity_ratings (
    id                    BIGSERIAL,
    rating_run_id         INTEGER REFERENCES rating_runs(id),
    ratings_version       VARCHAR(100) NOT NULL,
    snapshot_at           TIMESTAMPTZ DEFAULT NOW(),
    entity_type           VARCHAR(20) NOT NULL,
    entity_name           VARCHAR(200) NOT NULL,
    normalized_entity_name VARCHAR(200) NOT NULL,
    team_name VARCHAR(200), role VARCHAR(20),
    rating_system         VARCHAR(20) NOT NULL,
    rating_value REAL, rd REAL, sigma REAL,
    games_played INTEGER DEFAULT 0,
    last_match_at VARCHAR(50), state_json TEXT
);
SELECT create_hypertable('entity_ratings', 'snapshot_at', if_not_exists => TRUE, migrate_data => TRUE);
CREATE INDEX IF NOT EXISTS idx_rating_version ON entity_ratings(ratings_version, entity_type, rating_system);

/* ── Prediction tables ───────────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS upcoming_match_features (
    id                    SERIAL PRIMARY KEY,
    canonical_match_id    INTEGER NOT NULL REFERENCES canonical_matches(id),
    feature_version       VARCHAR(100) NOT NULL,
    ratings_version       VARCHAR(100) NOT NULL,
    data_cutoff_at        VARCHAR(50),
    team_a_golgg_name     VARCHAR(200), team_b_golgg_name VARCHAR(200),
    feature_status        VARCHAR(50) DEFAULT 'pending',
    missing_reason TEXT, features_json TEXT,
    UNIQUE(canonical_match_id, feature_version, ratings_version)
);
CREATE INDEX IF NOT EXISTS idx_upcoming_features_match ON upcoming_match_features(canonical_match_id);

CREATE TABLE IF NOT EXISTS canonical_predictions (
    id                    SERIAL PRIMARY KEY,
    canonical_match_id    INTEGER NOT NULL REFERENCES canonical_matches(id),
    model_artifact_id     INTEGER REFERENCES model_artifacts(id),
    model_name            VARCHAR(200) NOT NULL,
    model_version         VARCHAR(50) NOT NULL,
    predicted_at          TIMESTAMPTZ DEFAULT NOW(),
    prob_a DOUBLE PRECISION, prob_b DOUBLE PRECISION,
    features_version VARCHAR(100), ratings_version VARCHAR(100),
    data_cutoff_at VARCHAR(50),
    prediction_status VARCHAR(50) DEFAULT 'active',
    diagnostics_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_canonical_pred_match ON canonical_predictions(canonical_match_id, predicted_at DESC);

CREATE TABLE IF NOT EXISTS model_ev_signals (
    id                    SERIAL PRIMARY KEY,
    canonical_match_id    INTEGER NOT NULL REFERENCES canonical_matches(id),
    canonical_prediction_id INTEGER REFERENCES canonical_predictions(id),
    odds_snapshot_id      INTEGER,
    bookmaker_id          INTEGER NOT NULL REFERENCES bookmakers(id),
    side                  VARCHAR(5) NOT NULL,
    odds DOUBLE PRECISION, model_prob DOUBLE PRECISION, market_prob DOUBLE PRECISION,
    ev DOUBLE PRECISION, tax_rate DOUBLE PRECISION NOT NULL DEFAULT 0.12,
    stake_suggestion DOUBLE PRECISION, status VARCHAR(50) NOT NULL DEFAULT 'new',
    evaluated_at TIMESTAMPTZ, market_context_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_signal_match ON model_ev_signals(canonical_match_id, status, ev DESC);

CREATE TABLE IF NOT EXISTS automation_commands (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES automation_runs(id),
    command     VARCHAR(500) NOT NULL,
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'pending',
    exit_code INTEGER, output TEXT, error TEXT
);

/* ── APScheduler job store ───────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS apscheduler_jobs (
    id            VARCHAR(191) PRIMARY KEY,
    next_run_time DOUBLE PRECISION,
    job_state     BYTEA NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_apscheduler_jobs_next_run_time ON apscheduler_jobs(next_run_time);

/* ── Rating pipeline tracking ────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS rating_processed_matches (
    ratings_version VARCHAR(100) NOT NULL,
    match_id        VARCHAR(64) NOT NULL,
    match_date      VARCHAR(50),
    processed_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (ratings_version, match_id)
);
 
/* ── Append-only local forecast evidence ────────────────────────────────── */
CREATE TABLE IF NOT EXISTS forecast_evidence_snapshots (
    id            SERIAL PRIMARY KEY,
    evidence_key  VARCHAR(200) NOT NULL,
    content_sha256 VARCHAR(64) NOT NULL,
    payload_json  TEXT NOT NULL,
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_forecast_evidence_key UNIQUE (evidence_key)
);

CREATE OR REPLACE FUNCTION deny_forecast_evidence_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'forecast evidence snapshots are append-only';
END;
$$;

DROP TRIGGER IF EXISTS forecast_evidence_no_mutation ON forecast_evidence_snapshots;
CREATE TRIGGER forecast_evidence_no_mutation
BEFORE UPDATE OR DELETE ON forecast_evidence_snapshots
FOR EACH ROW EXECUTE FUNCTION deny_forecast_evidence_mutation();

/* ── Plan09 exact source captures and opportunity ledger ─────────────────── */
CREATE TABLE IF NOT EXISTS source_captures (
    id SERIAL PRIMARY KEY,
    capture_id VARCHAR(200) NOT NULL,
    source_id VARCHAR(200) NOT NULL,
    evidence_uri TEXT NOT NULL,
    sha256 VARCHAR(64),
    content_base64 TEXT,
    body_uri TEXT,
    body_representation VARCHAR(40) NOT NULL,
    content_encoding VARCHAR(100),
    character_encoding VARCHAR(80),
    request_started_at TIMESTAMPTZ,
    response_received_at TIMESTAMPTZ,
    retrieved_at TIMESTAMPTZ,
    first_observed_at TIMESTAMPTZ,
    observed_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    source_effective_at TIMESTAMPTZ,
    source_revision VARCHAR(200),
    http_status INTEGER,
    source_class VARCHAR(50) NOT NULL,
    availability_basis VARCHAR(100) NOT NULL,
    attempt_status VARCHAR(20) NOT NULL,
    failure_reason TEXT,
    request_metadata_json TEXT NOT NULL,
    response_metadata_json TEXT NOT NULL,
    source_authentication_status VARCHAR(30) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_source_capture_id UNIQUE (capture_id),
    CONSTRAINT ck_source_capture_body_reference CHECK (
        (sha256 IS NULL AND content_base64 IS NULL AND body_uri IS NULL)
        OR (sha256 IS NOT NULL AND content_base64 IS NOT NULL AND body_uri IS NOT NULL)
    ),
    CONSTRAINT ck_source_capture_clock_order CHECK (
        (request_started_at IS NULL OR response_received_at IS NULL OR request_started_at <= response_received_at)
        AND (request_started_at IS NULL OR observed_at IS NULL OR request_started_at <= observed_at)
        AND (response_received_at IS NULL OR observed_at IS NULL OR response_received_at <= observed_at)
        AND (first_observed_at IS NULL OR observed_at IS NULL OR first_observed_at <= observed_at)
        AND (retrieved_at IS NULL OR retrieved_at = response_received_at)
    ),
    CONSTRAINT ck_source_capture_attempt_status CHECK (attempt_status IN ('captured', 'failed')),
    CONSTRAINT ck_source_capture_authentication_status CHECK (
        source_authentication_status IN ('local_fixture', 'unverified', 'authenticated')
    )
);
CREATE INDEX IF NOT EXISTS ix_source_capture_revision_first_observed
    ON source_captures(source_id, sha256, source_revision, first_observed_at);

CREATE TABLE IF NOT EXISTS opportunity_events (
    id SERIAL PRIMARY KEY,
    event_key VARCHAR(200) NOT NULL,
    protocol_id VARCHAR(120) NOT NULL,
    provider_id VARCHAR(120) NOT NULL,
    fixture_id VARCHAR(200) NOT NULL,
    origin_at TIMESTAMPTZ,
    event_type VARCHAR(40) NOT NULL,
    primary_decision BOOLEAN NOT NULL DEFAULT FALSE,
    parent_event_key VARCHAR(200) REFERENCES opportunity_events(event_key),
    payload_json TEXT NOT NULL,
    payload_sha256 VARCHAR(64) NOT NULL,
    forecast_evidence_key VARCHAR(200) REFERENCES forecast_evidence_snapshots(evidence_key),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_opportunity_event_key UNIQUE (event_key),
    CONSTRAINT ck_opportunity_primary_origin_required CHECK (NOT primary_decision OR origin_at IS NOT NULL),
    CONSTRAINT ck_opportunity_revision_parent_required CHECK (
        event_type NOT IN ('schedule_revision', 'correction') OR parent_event_key IS NOT NULL
    ),
    CONSTRAINT ck_opportunity_event_type CHECK (
        event_type IN ('discovery', 'schedule_revision', 'origin_pending', 'primary_decision', 'correction', 'outcome_release')
    )
);
CREATE INDEX IF NOT EXISTS ix_opportunity_event_protocol_recorded
    ON opportunity_events(protocol_id, recorded_at, id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_opportunity_primary_origin
    ON opportunity_events(protocol_id, provider_id, fixture_id, origin_at)
    WHERE primary_decision;
CREATE UNIQUE INDEX IF NOT EXISTS uq_opportunity_primary_fixture
    ON opportunity_events(protocol_id, provider_id, fixture_id)
    WHERE primary_decision;

CREATE OR REPLACE FUNCTION deny_plan09_evidence_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only';
END;
$$;

DROP TRIGGER IF EXISTS source_captures_no_mutation ON source_captures;
CREATE TRIGGER source_captures_no_mutation
BEFORE UPDATE OR DELETE ON source_captures
FOR EACH ROW EXECUTE FUNCTION deny_plan09_evidence_mutation();

DROP TRIGGER IF EXISTS opportunity_events_no_mutation ON opportunity_events;
CREATE TRIGGER opportunity_events_no_mutation
BEFORE UPDATE OR DELETE ON opportunity_events
FOR EACH ROW EXECUTE FUNCTION deny_plan09_evidence_mutation();

/* ── Plan11 application tables omitted from the historical bootstrap ─────── */
CREATE TABLE IF NOT EXISTS alert_configs (
    id INTEGER PRIMARY KEY,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    min_ev DOUBLE PRECISION NOT NULL DEFAULT 0.05,
    min_odds DOUBLE PRECISION NOT NULL DEFAULT 1.25,
    max_odds DOUBLE PRECISION,
    cooldown_hours DOUBLE PRECISION NOT NULL DEFAULT 6,
    discord_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    discord_webhook_url VARCHAR(500),
    telegram_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    telegram_bot_token VARCHAR(200),
    telegram_chat_id VARCHAR(100),
    updated_at VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS value_alert_logs (
    id SERIAL PRIMARY KEY,
    created_at VARCHAR(50) NOT NULL,
    canonical_match_id INTEGER,
    match_label VARCHAR(255) NOT NULL,
    league VARCHAR(100),
    match_start_at VARCHAR(50),
    side VARCHAR(10),
    team_name VARCHAR(150),
    bookmaker_name VARCHAR(100),
    odds DOUBLE PRECISION,
    model_prob DOUBLE PRECISION,
    market_prob DOUBLE PRECISION,
    ev DOUBLE PRECISION,
    suggested_stake DOUBLE PRECISION,
    channels VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    message_payload TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS ix_value_alert_logs_created_at ON value_alert_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_value_alert_logs_canonical_match_id ON value_alert_logs(canonical_match_id);

CREATE TABLE IF NOT EXISTS canonical_teams (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(200) NOT NULL,
    normalized_name VARCHAR(200) NOT NULL UNIQUE,
    squad_type VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS canonical_competitions (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(200) NOT NULL,
    normalized_name VARCHAR(200) NOT NULL UNIQUE,
    family VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS source_team_identities (
    id SERIAL PRIMARY KEY,
    canonical_team_id INTEGER NOT NULL REFERENCES canonical_teams(id),
    source_system VARCHAR(50) NOT NULL,
    source_team_id VARCHAR(100) NOT NULL,
    source_name VARCHAR(200) NOT NULL,
    normalized_source_name VARCHAR(200) NOT NULL,
    competition_scope VARCHAR(200) NOT NULL DEFAULT '',
    valid_from VARCHAR(20) NOT NULL DEFAULT '',
    valid_to VARCHAR(20) NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1,
    review_status VARCHAR(50) NOT NULL DEFAULT 'approved',
    CONSTRAINT uq_source_team_identity_scope UNIQUE (source_system, source_team_id, competition_scope, valid_from, valid_to)
);
CREATE INDEX IF NOT EXISTS ix_source_team_identity_lookup ON source_team_identities(source_system, normalized_source_name, competition_scope);
CREATE TABLE IF NOT EXISTS source_competition_identities (
    id SERIAL PRIMARY KEY,
    canonical_competition_id INTEGER NOT NULL REFERENCES canonical_competitions(id),
    source_system VARCHAR(50) NOT NULL,
    source_competition_id VARCHAR(200) NOT NULL,
    source_name VARCHAR(300) NOT NULL,
    normalized_source_name VARCHAR(300) NOT NULL,
    CONSTRAINT uq_source_competition_identity UNIQUE (source_system, source_competition_id)
);
CREATE TABLE IF NOT EXISTS mapping_review_decisions (
    id SERIAL PRIMARY KEY,
    canonical_match_id INTEGER NOT NULL REFERENCES canonical_matches(id),
    old_golgg_match_id VARCHAR(50),
    new_golgg_match_id VARCHAR(50),
    decision VARCHAR(20) NOT NULL,
    reason TEXT NOT NULL,
    operator VARCHAR(100) NOT NULL,
    evidence_json TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_mapping_review_match_time ON mapping_review_decisions(canonical_match_id, decided_at);

CREATE TABLE IF NOT EXISTS match_roster_overrides (
    id SERIAL PRIMARY KEY,
    canonical_match_id INTEGER NOT NULL REFERENCES canonical_matches(id),
    team_side VARCHAR(1) NOT NULL,
    roster_json TEXT NOT NULL,
    updated_at VARCHAR(50),
    CONSTRAINT uq_match_roster_override_side UNIQUE (canonical_match_id, team_side)
);
CREATE INDEX IF NOT EXISTS ix_match_roster_overrides_canonical_match_id ON match_roster_overrides(canonical_match_id);
CREATE TABLE IF NOT EXISTS team_current_roster_players (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(50),
    team_name VARCHAR(200) NOT NULL,
    normalized_team_name VARCHAR(200) NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(200),
    role VARCHAR(20) NOT NULL,
    source VARCHAR(20) NOT NULL DEFAULT 'auto',
    source_match_id VARCHAR(50),
    source_game_id VARCHAR(50),
    source_match_date VARCHAR(50),
    updated_at VARCHAR(50),
    CONSTRAINT uq_team_current_roster_role UNIQUE (normalized_team_name, role)
);
CREATE INDEX IF NOT EXISTS ix_team_current_roster_players_team_id ON team_current_roster_players(team_id);
CREATE INDEX IF NOT EXISTS ix_team_current_roster_players_normalized_team_name ON team_current_roster_players(normalized_team_name);
CREATE INDEX IF NOT EXISTS ix_team_current_roster_players_player_id ON team_current_roster_players(player_id);

CREATE TABLE IF NOT EXISTS oddspapi_fixture_mappings (
    id SERIAL PRIMARY KEY,
    fixture_id VARCHAR(100) NOT NULL CONSTRAINT uq_oddspapi_fixture_id UNIQUE,
    canonical_match_id INTEGER REFERENCES canonical_matches(id),
    sport_id INTEGER NOT NULL DEFAULT 18,
    league VARCHAR(100),
    provider_team_1 VARCHAR(200) NOT NULL,
    provider_team_2 VARCHAR(200) NOT NULL,
    provider_team_1_is_a INTEGER,
    start_time TIMESTAMPTZ,
    has_odds INTEGER NOT NULL DEFAULT 1,
    last_synced_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_oddspapi_fixture_mappings_canonical_match_id ON oddspapi_fixture_mappings(canonical_match_id);
CREATE TABLE IF NOT EXISTS oddspapi_request_logs (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    fixture_id VARCHAR(100),
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_oddspapi_request_logs_created_at ON oddspapi_request_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_oddspapi_request_logs_fixture_id ON oddspapi_request_logs(fixture_id);
