# Betting app ML layer

This package is the production-facing ML/data-science layer for the betting
application. It is deliberately kept outside `scripts/`, because those scripts
are thesis/research artefacts and should not be reorganized in-place.

Initial responsibilities:

- deterministic historical backtests against collected bookmaker odds,
- shared metrics for model-vs-market evaluation,
- explicit staking and odds-selection policies,
- future model registry, retraining, shadow prediction and promotion workflows.

Current first slice:

- `backtesting.engine`: evaluates existing model probabilities against historical
  odds and settled match results,
- `backtesting.loaders`: read-only loaders from current application tables
  (`canonical_predictions`, `canonical_matches`, `odds_snapshots`),
- `backtesting.comparison`: model-vs-market probability metrics using no-vig
  bookmaker probabilities,
- `backtesting.odds_selection`: decides **when** a simulated bet would have been
  placed from available snapshots,
- `backtesting.staking`: fixed / bankroll-percent / fractional Kelly stake sizing,
- `registry`: lightweight model-version and evaluation-run registry tables,
- `registry.gates`: promotion gates for candidate/shadow models,
- `pipelines.evaluation`: Docker/Kedro-friendly historical evaluation pipeline,
- `training`: feature loading, candidate models, walk-forward validation and
  artifact saving for regular retraining,
- `pipelines.weekly_retrain`: weekly retraining runner that selects the best
  candidate and registers it as candidate/shadow,
- `metrics`: probability and betting summary metrics.

CLI example:

```bash
python -m betting_app.ml.backtesting.cli \
  --model-name 'Sym-Cal LR-ElasticNet-W20-Binomial' \
  --model-version exp-039 \
  --include-stale \
  --days-back 365 \
  --min-ev 0.0 \
  --staking fractional_kelly \
  --json
```

Pipeline example with registry logging:

```bash
python -m betting_app.ml.pipelines.evaluate_existing_model \
  --model-name 'Sym-Cal LR-ElasticNet-W20-Binomial' \
  --model-version exp-039 \
  --days-back 365 \
  --min-ev 0.0 \
  --staking fractional_kelly \
  --json
```

Generic tabular retraining is not an active application model path. If it is
used for an isolated experiment, the caller must supply a new explicit model
name; it must not reuse the retired
`Operational-Retrained-Tabular` identifier.

Each experimental artifact directory contains:

```text
model.joblib             # trained estimator + feature order
metadata.json            # model, candidate, metrics and dataset references
train_dataset.jsonl      # exact materialized training rows used by the model
feature_names.json       # exact feature order/schema
dataset_metadata.json    # row count, feature count, dataset_hash, format
```


`dataset_hash` is computed from the materialized training rows and feature
values, not just source ids. This gives us reproducibility now and a clean path
to DVC later: DVC can track these artifact directories without changing the
training code.

The pipeline:

1. loads finished match labels,
2. loads historical predictions,
3. loads historical bookmaker odds,
4. runs the betting backtest,
5. compares model probability metrics against no-vig market probabilities,
6. evaluates promotion gates,
7. writes `ml_model_versions` and `ml_evaluation_runs` records.

This is intentionally plain Python, so the same code can run as:

- a Docker command,
- a scheduler task,
- a cron job,
- future Kedro nodes/pipelines,
- future CI/CD model evaluation job.

The backtest output includes betting metrics (`bets`, `total_staked`, `roi`,
`hit_rate`, `max_drawdown`) and model-vs-market probability metrics
(`model_log_loss`, `market_log_loss`, `model_brier`, `market_brier`,
`model_accuracy`, `market_accuracy`).

Default anti-leakage assumptions:

- use finished matches only,
- use only odds snapshots mapped to the same `canonical_match_id`,
- ignore invalid odds (`<= 1.0`),
- by default take the latest pre-match quote per bookmaker,
- by default deduplicate predictions to the latest prediction per
  `(canonical_match_id, model_name, model_version)`, so stale prediction history
  does not simulate repeatedly betting the same match.

Next production slices should add:

1. shadow predictions for candidate models,
2. production promotion workflow,
3. API/UI reports for historical model-vs-bookmaker performance,
4. optional Kedro project wrapper around these plain-Python nodes.

Design rule: add new production abstractions here first, then gradually make old
services delegate to them. Do not break scheduler/API flows while migrating.
