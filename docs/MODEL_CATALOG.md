# Models, completed studies and retention

Updated: **2026-10-08**. Start here to choose a model or find an experiment; use [RESEARCH.md](RESEARCH.md) for the full evidence journal, [architecture](PROJECT_ARCHITECTURE.md) for code ownership, and [scripts](../scripts/README.md) for exact interfaces.

**Working choice: unchanged native C0 for match winner; the preserved standalone prop bundle for Map 1 kills/duration and series length.** These are separate predictors, not one calibrated joint model. The champion-history multitask challenger did not pass its replacement gates.

This catalog organizes existing decisions. It does not promote a model, restart experiments or establish today's server state. All inspected historical years remain development-exposed. Data and recovery binaries stay outside Git; their absence in a fresh clone is not permission to regenerate or replace frozen hashes.

## 1. Which models remain relevant?

| Model / artifact identity | Task | Position now | Source of truth / interface |
|---|---|---|---|
| **Native C0**, `c0-native-2026-w32-e12-v1`; original `a0_data_focus_20260929_192443/control` W32/E12 plus shared experts/calibration | BO1/3/5 series winner | **KEEP_REFERENCE; current local sports configuration.** Not annual A0. No change in this cleanup; no remote inspection/deployment claim. | [Serving config](../conf/base/c0_serving.json), [registry](../betting_app/core/models/registry.py), [application contract](../betting_app/README.md#aktualny-kontrakt-c0) |
| **Standalone props**, `prop_distribution_20261007_01` | Map 1 total kills | **KEEP_BASELINE.** CAL selected ridge-1 Poisson-mean/NB. `NO_ESTABLISHED_INCREMENTAL_GAIN` means no gain beyond the regularized control, not an unusable kill predictor. | [Selected bundle](../data/06_models/prop_distribution/prop_distribution_20261007_01/selected_model.joblib), [selection](../data/06_models/prop_distribution/prop_distribution_20261007_01/selection.json), [decision](../data/08_reporting/prop_distribution/prop_distribution_20261007_01/decision.json) |
| Same standalone prop bundle | Map 1 duration | **KEEP_RESEARCH_CANDIDATE.** Selected 15-leaf LightGBM/lognormal improves the regularized control; interval reliability remains unresolved. | Same frozen bundle and decision; [recovery below](#2-preserved-standalone-prop-model) |
| Same standalone prop bundle | BO3/BO5 series length; BO1 deterministic | **KEEP_RESEARCH_CANDIDATE.** Selected width128/epoch12 three-seed distribution mixture improves the regularized control, but historical Pinnacle comparison is negative. | Same frozen bundle and decision; [market evidence](../data/08_reporting/prop_market_comparison/prop_market_comparison_20261007_01/decision.json) |
| **Annual Causal A0** | Historical series-winner benchmark | **KEEP_CONTROL.** Canonical annual benchmark/reference, not the local C0 artifact. | [Pinned benchmark manifest](../conf/base/research_benchmark.json); `scripts/run_model_benchmark.py --suite --doctor`; external artifacts resolved via `data/research_root.txt`, environment or `--research-root` |
| **Frozen EXP-039** and **annual039 research derivative** | Thesis baseline / common-support research control | **KEEP_CONTROL, distinct identities.** Frozen039 is not the annual derivative. Derivative covers 9,907/11,550; missing 1,643 predictions cannot be treated as zero loss. | [Frozen application artifact](../betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib), [manifest](../conf/base/research_benchmark.json), [dated audit](../reports/production_readiness_and_market_benchmark_report_2026.md) |
| **Calibrated player Glicko / Elo** | Research controls | **KEEP_CONTROL.** Use the manifest's actual implementation/cohort, not a generic name-only comparison. | [Canonical benchmark](../scripts/run_model_benchmark.py), [research status](RESEARCH.md) |
| **Fixed50 C0 + market hybrid**, `c0-native-2026-w32-e12-v1-a0.50-t1.00` | Separate market-assisted local inference | **KEEP_LOCAL_CONFIGURATION, not odds-free.** One 50/50 logit-space blend; not a trained prop head or certified profitable policy. | [Registry](../betting_app/core/models/registry.py), [application contract](../betting_app/README.md#aktualny-kontrakt-c0) |
| **Fixed A0/raw connector03**, `a0_fixed_calibration_connector_20261001_03` | Historical series winner | **KEEP_HISTORICAL_CANDIDATE_ONLY.** Development candidate retained by its original study. Not established superior to native C0; July–September2026 selected heads are not a current/future recipe. | [Original decision](../data/08_reporting/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/decision.json), [research report](../reports/lol_model_improvement_research_20260927.md) |
| **Optimal Synergy Strategy (C0 Shrunk Hybrid + Thesis Guard)** | Decision-time series betting | **ADOPTED_LOCAL_STRATEGY.** Combines native C0 with market shrinkage ($\alpha=0.50$), Guard B ($\Delta\text{logit}\le 0.85$), and Thesis Model Veto ($|p_{\text{C0}} - p_{\text{Thesis}}| \le 0.10$). Generates $+4.83\%$ Net CLV and $+23.79\%$ Yield. | [Execution Report](../reports/optimal_synergy_c0_thesis_simulation_20261012.md), [Implementation](../betting_app/core/ev.py) |

No rejected challenger is on an automatic fitting queue. Better duration/length scores do not establish a better winner model or betting readiness.

## 2. Preserved standalone prop model

**Original run stays in place.** No aliases, moved modules, rewritten model files or changed scientific decisions.

| Item | Location |
|---|---|
| Original selected model | `data/06_models/prop_distribution/prop_distribution_20261007_01/selected_model.joblib` |
| Frozen task selection | `data/06_models/prop_distribution/prop_distribution_20261007_01/selection.json` |
| Protocol / input bindings | [protocol](../data/05_model_input/prop_distribution/prop_distribution_20261007_01/protocol.json), [binding](../data/05_model_input/prop_distribution/prop_distribution_20261007_01/binding.json) |
| Scientific result and limits | [decision](../data/08_reporting/prop_distribution/prop_distribution_20261007_01/decision.json), [metrics](../data/08_reporting/prop_distribution/prop_distribution_20261007_01/metrics.json), [comparisons](../data/08_reporting/prop_distribution/prop_distribution_20261007_01/comparisons.json), [sealed verification](../data/08_reporting/prop_distribution/prop_distribution_20261007_01/verification/sealed_verification.json) |
| Recovery bundle | [prop_distribution_20261007_01.zip](../data/06_models/model_preservation/prop_distribution_20261007_01.zip) |
| Executed recovery proof | [verification.json](../data/08_reporting/model_preservation/prop_distribution_20261007_01/verification.json) |
| Preservation tool | [CLI](../scripts/preserve_prop_model.py), [implementation](../src/analysis/prop_model_preservation.py), [integrity regressions](../betting_app/tests/test_prop_model_preservation.py) |

Bundle: **63,386,728 bytes; 154 ZIP members**. It contains all five study stages 04–08, including the 40,636-row prepared feature dataset, selected model and candidate/control models, distribution outputs, protocol/bindings, audits, failed-verifier records and numerical receipts. Local import dependencies/package initializers, core package versions, Python version and complete-distribution recovery examples are included. Research/script documentation is an as-packaged snapshot, not a replacement for this current catalog.

- Bundle SHA-256: `a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232`.
- Original selected-model SHA-256: `9260e43dcd926a662c75a46d3e747f0629757fbedb061207c9234542570d110b`.
- Original selection SHA-256: `8b28801e6e1975084e9f897687a7f4f2512bca3475ce50b582ae682d97291a81`.

**Measured recovery:** validate every member, extract into an owned temporary directory, run a fresh Python process importing restored sources, and compare nine BO1/3/5 historical cases. All 25 arrays/275 numerical values match exactly: selected NB/lognormal component parameters, mixture weights, legal length PMFs and fixed-line probabilities. Fifteen protected model/configuration/receipt hashes remain unchanged. This verifies those recovery cases and whole-archive integrity—not all 40,636 inference rows or fresh training. Forty focused preservation/existing-prop tests passed, 10 warnings; existing tests include isolated synthetic software fits, not a new formal experiment.

### Verify or restore

Run from the repository root with the recorded environment; these commands do **not** fit or activate models:

```bash
.venv/bin/python scripts/preserve_prop_model.py verify \
  --bundle data/06_models/model_preservation/prop_distribution_20261007_01.zip \
  --sha256 a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232

# Destination must not exist. Choose an owned location outside existing run directories.
.venv/bin/python scripts/preserve_prop_model.py restore \
  --bundle data/06_models/model_preservation/prop_distribution_20261007_01.zip \
  --sha256 a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232 \
  --destination /tmp/prop-model-restored
```

`verify` exercises extraction and saved inference automatically. The extracted `scripts/preserve_prop_model.py replay --root RESTORED --output NEW.npz` replays the bundled examples using those source files and the recorded environment. `create` requires a new output filename and validates the original frozen study before archiving; do not replace the bundle above.

**Recovery boundary:** portable inference from prepared historical features, not a live roster adapter or a full raw-source/deployment backup. The manifest lists unbundled raw/feature-bank/C0 guard assets needed for raw-to-feature reconstruction and the original runner. Core dependencies/Python are recorded; the virtual environment is not bundled. Joblib can execute pickle code: use only a trusted bundle and trusted external digest. This same-disk copy is **not an off-machine backup**; copy the ZIP and its verification receipt together to separate storage for disaster recovery.

### What the preserved model actually measured

Same 3,103 historical 2026 Map 1 targets; 2,358 BO3/BO5 targets for primary length scoring. TRAIN 36,205 before 2025-07-01; CAL 1,328 in 2025H2; strictly prior whole-series daily releases. Five non-neural fits plus six continuous 24-epoch neural trajectories; nine CAL candidates, one choice per task; no assessment/C0 fit. Primary 98.333333% intervals use 5,000 paired monthly-block draws/nine months.

| Task | Selected NLL | Other evidence | Research decision |
|---|---:|---|---|
| Map 1 kills |3.590488045| CRPS 5.010185; predictive-mean MAE 7.0483 kills; 80% interval coverage 79.57% | Retain simple baseline; no incremental learned-family gain |
| Map 1 duration |3.076194163| Selected−regularized NLL −0.011915977, interval [−0.025358976,−0.001450657]; MAE 4.2373 minutes; 80% coverage 75.99% | Retain development candidate; undercoverage unresolved |
| Series length |0.714579615| RPS 0.261305154; selected−regularized NLL −0.026560189, interval [−0.035573171,−0.017970164] | Retain development candidate, not a winner/correct-score model |

[Historical bookmaker comparison](../data/08_reporting/prop_market_comparison/prop_market_comparison_20261007_01/decision.json): on 1,318 Pinnacle total-map fixtures, binary line LL is model 0.669133345 versus market 0.658158740; paired 97.5% interval for model−market [+0.003025793,+0.023563745]. These are threshold scores, not categorical length NLL. Actual starts are missing on 423 primary fixtures; equal-month LL interval crosses zero. Two kill-price fixtures are insufficient; duration prices are unavailable. Preserve this negative evidence; no bookmaker superiority or betting certification.

## 3. What was checked, and what is retired?

**ARCHIVE_IN_PLACE** means removed from the current candidate shortlist while retaining executable sources, models, predictions and evidence at their bound paths. It is not physical deletion or a new scientific decision. The table groups related fixed studies; original decisions keep their exact wording.

### Recent outcome/statistics and representation studies

| Study | Tested / how | Result / disposition | Evidence |
|---|---|---|---|
| `optimal_synergy_c0_thesis_simulation_20261012` | C0 Shrunk Hybrid ($\alpha=0.50$) with Guard B and Thesis Model Guard ($\Delta \le 0.10$); 101-day chronological simulation on 10,000 PLN capital with 1/4 Kelly | **ADOPTED_LOCAL_STRATEGY**. 50 bets placed, 58.0% win rate, **+23.79% Yield / +17.12% ROI** (+4.83% Net CLV). Vetoed 9 bets that had 11.1% win rate (-67.7% loss). Max DD 8.21%. | [Report](../reports/optimal_synergy_c0_thesis_simulation_20261012.md), [Ledger](../data/08_reporting/optimal_strategy_simulation_ledger.csv) |
| `oddsportal_open_close_simulation_20261012` | Multi-year 5-season simulation on OddsPortal (11,609 matches, 2021-2026) and 2024-2026 benchmark (2,275 matches) comparing open vs close | **COMPLETED_LARGE_SAMPLE_VALIDATION**. At 0% tax: **+10.15% Yield / +42.5% CAGR** (3,528 bets). At 12% tax: **+9.65% Yield** (684 bets), doubling bankroll (+96.4% ROI) with 87.4% positive CLV rate. | [Report](../reports/oddsportal_open_close_simulation_20261012.md) |
| `monthly_refit_c0_production_odds_since_may_20261011` | Census of 234,435 quotes across 1,600 canonical matches; out-of-time monthly walk-forward C0 retraining across 645 matched series (May-Sept 2026) | **COMPLETED_CENSUS_AND_BENCHMARK**. Shrunk Hybrid ($\alpha=0.50$) achieves LogLoss 0.557877 vs market 0.570054 and C0 0.573307 (-1.22 pp gain). Mean Raw CLV +19.13%. | [Report](../reports/monthly_refit_c0_production_odds_since_may_20261011.md) |
| `champion_history_20261007_01` | Shared last 16 released-player-map encoder; paired champion-hidden/included × score-only/multitask; 12 completed trajectories, 8 CAL scalars; 3,103 assessment series | All four tasks and bundle **NOT_ADOPTED**. CAL-selected winner LL 0.546320524 versus C0 0.535448701. Duration beats old prop reference but not its matched control. **ARCHIVE_IN_PLACE**. | [Executed report](../reports/champion_history_prediction_20261007.md), [decision](../data/08_reporting/champion_history/champion_history_20261007_01/decision.json), [verification](../data/08_reporting/champion_history/champion_history_20261007_01/sealed_verification.json) |
| `draft_prop_20261007_01` | Actual-composition oracle vs matched no-draft and old props; 4 mean regressors/4 dispersion fits; 3,103 assessment rows | Information gate failed; **NOT_ADOPTED**. Draft policy and sequential Fearless simulation **NOT_RUN_BY_FROZEN_GATE**. Not a history-only or valid ordinary pre-match input study. **ARCHIVE_IN_PLACE**. | [Report](../reports/draft_prop_20261007_01.md), [decision](../data/08_reporting/draft_prop/draft_prop_20261007_01/decision.json) |
| `a0_champion_20261005_01` / IDEA-031 |30 champion summary features +12 rating differences, specialist plus CAL logit connector to C0 | **NOT_ADOPTED**; primary interval/Brier gates fail. Preserve the negative champion-pool result. **ARCHIVE_IN_PLACE**. | [Decision](../data/08_reporting/a0_champion/a0_champion_20261005_01/decision.json), [research journal](RESEARCH.md) |
| `c0_meta_20261007_094552_integrity_v2` | Frozen C0 hidden states plus 30-day role/meta profiles; 6 heads/2 CAL amplitudes; 3,103 primary targets | **NOT_ADOPTED**; both primary intervals cross zero, meta−C0 point NLL increases. **ARCHIVE_IN_PLACE**. | [Evaluation decision](../data/08_reporting/c0_meta/c0_meta_20261007_094552_integrity_v2/evaluation/decision.json), [execution](../data/08_reporting/c0_meta/c0_meta_20261007_094552_integrity_v2/execution.json) |
| `c0_meta_stable_20261007_01` | Fixed 180-day player support/profiles with unchanged 30-day meta; same fitting budget | **NOT_ADOPTED**; both CAL amplitudes zero, selected outputs exact C0. Not proof the representations are universally useless. **ARCHIVE_IN_PLACE**. | [Evaluation decision](../data/08_reporting/c0_meta_stable/c0_meta_stable_20261007_01/evaluation/decision.json), [execution](../data/08_reporting/c0_meta_stable/c0_meta_stable_20261007_01/execution.json) |
| `native_capacity_20261004_01` / IDEA-028/029 | Width16/32/64 ×12/24 endpoints, 3 paired seeds/9 continuous trajectories; controlled native recipe | Capacity **NOT_ADOPTED**; primary intervals cross zero. Distillation **SKIPPED_BY_FROZEN_EVALUATION_GATE**, zero KD fits. **ARCHIVE_IN_PLACE**. | [Decision](../data/08_reporting/a0_capacity/native_capacity_20261004_01/decision.json), [report](../reports/model_targets_capacity_and_distillation_research_20261004.md) |
| `context_reliability_20261004_02` | Detect 2024H2, replicate 2025; 6 diagnostic global fits; fixed common-support confidence-bin weights | Persistence not confirmed; **CONDITIONAL_FITS_SKIPPED_PERSISTENCE_GATE**, not a failed fitted conditional predictor. Preserve source blockers/protocol. | [Decision](../data/08_reporting/a0_context_reliability/context_reliability_20261004_02/decision.json), [research journal](RESEARCH.md) |
| `c0_raw_connector_20261002_033658` / Plan09 | Existing raw trio plus C0 connector and matched C0-only calibration control; 72 scalar heads, zero base fits | **NOT_ADOPTED** despite negative primary LL contrasts: BO5/A0 and source-unseen/A0 guards fail. **ARCHIVE_IN_PLACE**. | [Report](../reports/plan09_c0_connector_and_forward_evidence_20261002.md), [decision](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/decision.json) |
| Sports correction runs01/02/03 | History-specialist; raw-trio flexible correction; finally fixed-native-calibration learned connector, each separately frozen | Runs01/02 **NOT_ADOPTED**; run03 historical development candidate only, not proven above C0 or future-valid. Retain all adaptive attempts, not only03. | [Report §10](../reports/lol_model_improvement_research_20260927.md), [run01](../data/08_reporting/a0_sports_correction/a0_sports_correction_20261001_01/decision.json), [run02](../data/08_reporting/a0_sports_correction/a0_fresh_correction_20261001_02/decision.json), [run03](../data/08_reporting/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/decision.json) |
| `reliability_tradeoff_20261001_092401_integrity_v2` / Plan08 | Native6 vs native12 C0; full 11,550 rows; confidence-cell reliability and proper-score/cost gates | **NOT_ADOPTED**; LL worsens, reliability gain below required minimum. **ARCHIVE_IN_PLACE**. | [Decision](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/decision.json), [report §13](../reports/noisy_data_reliability_research_20260929.md) |
| Plan07 native6+12, FamilyMapBridge, integration | Earlier-origin replication; matched player vs family bridge; separately CAL-fitted integration with native A0 |6+12 replication fails. Standalone family bridge improves its rating control; calibration tolerance unresolved; A0 integration fails. Preserve these distinct positive/negative findings, no replacement. | [Plan07](../TODO/07_model_prediction_reliability.md), [report §11](../reports/noisy_data_reliability_research_20260929.md) |
| `a0_history_reconstruction_20261001_010503` / IDEA-026 | Matched-update temporary statistics reconstruction; native12 vs6+12 vs reconstruction+12; three paired seeds | Reconstruction rejected. Initial6+12 secondary signal later fails Plan07 replication. **ARCHIVE_IN_PLACE**. | [Report §10](../reports/noisy_data_reliability_research_20260929.md), [research journal](RESEARCH.md) |
| `a0_data_focus_20260929_192443` | High-tier/market-tier neural training, history-token dropout and combination vs uniform; 5 assembled models/3 seeds | Focus/dropout not adopted. **Do not trash this run:** its uniform control/shared models are the retained native C0 artifacts. | [Decision/report](../reports/bookmaker_noise_and_provider_training_20260927.md), [completed receipt](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/completed.json) |
| `native_refit_20260929_122617` | Annual/monthly × uniform/H365 native expert; 270 fits; identical chronological CAL policy | H365 rejected; monthly-uniform gain inconclusive. Expert comparison is not a whole-A0 replacement. **ARCHIVE_IN_PLACE**. | [Report §9](../reports/bookmaker_noise_and_provider_training_20260927.md), [research journal](RESEARCH.md) |
| `prematch_architecture_20260927_run001` | History-token input fusion, separate encoders and learned specialist connector; 11,550 matched targets | None beats annual A0. Preserve initial unexplained replay caveat and negative architectures. **ARCHIVE_IN_PLACE**. | [Report](../reports/ground_up_lol_prediction_20260927.md), [research journal](RESEARCH.md) |
| Ground-up, symmetry repair and fixed75/25 blend | Raw logistic/tree/role-network recipes; neutral-quadrature repair; separately fixed blend diagnostic | Raw standalone models lose to annual A0. Fixed blend has favorable exposed 2026 monthly signal but source-proxy interval crosses zero; not a promoted recipe. Preserve pre/post-repair outputs separately. | [Report](../reports/ground_up_lol_prediction_20260927.md), [research journal](RESEARCH.md) |
| Older calibration, roster/form, opponent-history, ratings/graph studies | Tier temperatures, isotonic optimism checks, representation/rating experiments and past-only roster/form corrections | No blanket calibration/flattening or repeated-hypothesis shortcut. Rejected results remain evidence; diagnostic research is not a fitted replacement. | [Research journal and external-root map](RESEARCH.md), [legacy experiment index](04_experiments/README.md), [future-idea statuses](future_ideas.md) |

### Tournament, market and operational evidence

| Study | Tested / how | Result / disposition | Evidence |
|---|---|---|---|
| `c0_tournament_copula_20261007_01` | One TRAIN-only player-shared copula scalar; complete-event Monte Carlo, 28 available forecasts/50k paths each | **NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE**; CAL guards also fail. Correctness fixes retained, not tournament-quality promotion. | [Decision](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/decision.json), [report §14](../reports/reliable_tournament_simulation_research_20261006.md) |
| `tournament_c0_20261006_01` | Native C0-anchored player residual filter/score law vs independent controls; eight requested editions | **NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE**; Brier guard fails, only one primary edition. **ARCHIVE_IN_PLACE**. | [Decision](../data/08_reporting/tournament_improvement/tournament_c0_20261006_01/decision.json), [research journal](RESEARCH.md) |
| `tournament_numerical_20261007_01` | Exact fixed-format DP/enumeration and Worlds Swiss-MC/exact-knockout hybrid; rational/route oracles | **KEEP_NUMERICAL_IMPLEMENTATION**: historical numerical integration only. Not a new trained model or predictive certification. | [Decision](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/decision.json), [report §13](../reports/reliable_tournament_simulation_research_20261006.md) |
| `tournament_dist_20261005_01` | Historical Glicko-derived regional proxy/calibration and bracket benchmark | **ARCHIVE_QUALIFIED_HISTORY**. Native C0/A0 attribution withdrawn; original `TOURNAMENT_CALIBRATION_ADOPTED` used incomplete gates and does not certify native/tournament calibration. Do not turn this stale decision into an active candidate. | [Corrected application explanation](../betting_app/README.md#historyczny-benchmark-proxy-nie-jest-ścieżką-native-c0--plan11), [current research corrections](RESEARCH.md) |
| `c0_betting_development_20261006_01`, `c0_input_aligned_20261006_02`, `c0_ev_loss_20261006_01` | Separately frozen sports+price vs market-only heads; actual-roster/prior-known inputs; likelihood vs return-MSE objectives | All **NOT_ADOPTED**; predictive/selected-return/after-cost guards not established. **ARCHIVE_IN_PLACE**, no change to local C0/Fixed50. | [Report and reviewed decisions](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md), [Plan10](../TODO/10_c0_ev_loss_objective.md) |
| Plans05/06/09 forward evidence | Production read-only reconstruction, selected-bet EV/Kelly, identity/context repairs; local capture/append-only ledger/native interfaces | Completed historical/local work retained; old loop stopped 10/10, **G0/source activation BLOCKED**, zero genuine future forecasts. Local software readiness is not authentic issuance. | [Plan05](../TODO/05_betting_ev_reliability.md), [Plan06](../TODO/06_evidence_admission_after_a0_audit.md), [Plan09](../TODO/09_c0_connector_and_forward_evidence.md), [reliability report](../reports/noisy_data_reliability_research_20260929.md) |
| Plan11 and C0 integrity audits | Application/source/clock/model identity, schema/accounting and input-admission repairs, not formal native-model fitting | **KEEP_SOFTWARE_FIXES_AND_EVIDENCE**; local completion with external gates blocked. Dated EXP081 exports stay historical, not today's server identity. | [Roadmap](../TODO/README.md), [architecture](PROJECT_ARCHITECTURE.md), [research journal](RESEARCH.md) |

## 4. Archive versus trash

### Retire from the shortlist, but retain evidence

Champion-history A/B/C/D, actual-draft oracle, champion specialist, 30-/180-day meta heads, rejected capacity endpoints, H365, reconstruction, failed connectors and tournament residual/copula candidates are **archive-only** for their completed recipes. Do not rerun them as untested ideas. Keep their protocols, exact source bindings, weights/checkpoints, complete predictions, metrics, fit/verification receipts and retained failures together.

**No model has been certified safe to physically delete in this pass.** A rejected model may be a replay dependency or mandatory matched control. In particular C0 lives inside a study that also contains rejected variants.

### Concrete trash candidates and blocked items

| Paths / category | Disposition | Why / condition |
|---|---|---|
| `.pytest_cache/`, `.ruff_cache/`, Python `__pycache__/` and `*.pyc`/`*.pyo` | **TRASH_CANDIDATE: regenerable cache** | Not model evidence. Scope to verified cache files, never adjacent sources. No physical removal in this pass. |
| `thesis/main.aux`, `.bbl`, `.blg`, `.bcf`, `.fdb_latexmk`, `.fls`, `.lof`, `.log`, `.lot`, `.out`, `.run.xml`, `.synctex.gz`, `.toc` | **TRASH_CANDIDATE: generated build intermediates** | Rebuilt by the existing latexmk workflow. Preserve `.tex`, bibliography, figures, generated research inputs and final PDF. Do not recursively prune backup trees. |
| Root `models/fusion_v2*.pt`, `models/transformer_best.pt` | **HOLD_DEPENDENCY_AUDIT** | Legacy inference/training/population scripts reference their exact paths. Not current C0, but not proven orphaned. |
| `experiments/*.py`, numbered `scripts/`, `docs/assets/exp-*` source snapshots | **ARCHIVE_IN_PLACE / HOLD** | Study records, dynamic imports and pickle module names bind paths. Moving a module can break restored model loading. |
| `data/{01_raw,03_primary,04_feature,05_model_input,06_models,07_model_output,08_reporting}`, external research archive, manifests/receipts | **KEEP** | Raw evidence, models, controls, predictions and provenance. Ignored-by-Git does not mean trash; failed checks are evidence too. |
| `thesis-backup-20260922/`, `thesis-codex-elasticnet-20260922/`, `backups/`, user changes and unclassified files | **HOLD_OWNERSHIP / KEEP_BACKUP** | Not established disposable; no dump/credential inspection, recursive move or deletion. |

A later physical cleanup needs an exact candidate list, dependency/manifest audit, verified off-machine copy where required, and explicit destructive-action approval. This pass completed **preservation and navigation cleanup**, with zero original models/files moved or deleted.

## 5. Where to read next

1. This catalog: working model choice, study disposition, recovery and trash boundaries.
2. [RESEARCH.md](RESEARCH.md): detailed methods, numbers, independent checks, limitations and retained incidents; latest qualified evidence can supersede stale historical adoption claims.
3. [scripts/README.md](../scripts/README.md): canonical interfaces and exact historically exercised commands. Old freeze/train examples are records, not authorization to rerun completed namespaces.
4. [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) / [application README](../betting_app/README.md): implementation/inference contracts, separate from scientific accuracy claims.
5. [TODO index](../TODO/README.md), [legacy thesis experiment index](04_experiments/README.md) and [idea catalog](future_ideas.md): plans and older experiments, not a list of current model candidates or an automatic execution queue.
