# Model Catalog & Research Registry

**Updated:** 2026-10-14  
**Source of Truth:** [`conf/base/c0_serving.json`](../conf/base/c0_serving.json), [`betting_app/core/models/registry.py`](../betting_app/core/models/registry.py), and [`reports/`](../reports/).

This catalog is the definitive index of active production models, canonical research baselines, and completed empirical studies across **LoL-Breaker**.

---

## 1. Active Operational Models & Strategies

| Role / Scope | Model Identifier | Specification & Mechanics | Primary Metrics | Status & Interface |
|---|---|---|---|---|
| **Active Sports Match-Winner** | **`Causal-C0`** (`c0-native-2026-w32-e12-v1`) | Width-32 / Epoch-12 3-seed causal neural history-attention (16 prior maps/player, Beta terminal score + $0.1$ SmoothL1 aux loss) + shared rating/MLP experts. | **LogLoss 0.535449**, Brier **0.181034** on 3,103 canonical 2026 series. | **`KEEP_REFERENCE`**<br/>Active local sports default. Output is whole-series BO1/3/5 win probability.<br/>Config: [`conf/base/c0_serving.json`](../conf/base/c0_serving.json) |
| **Market-Assisted Operational Hybrid** | **`Hybrid-Operational-Market`** (`c0-native-2026-w32-e12-v1-a0.50-t1.00`) | 50/50 logit-space shrinkage combining Native C0 with Shin-devigged market consensus: `logit(p_Fixed50) = 0.50 · logit(p_C0) + 0.50 · logit(q_shin)`. | LogLoss **0.557877** vs Market 0.570054 on 2026 production quotes ($-1.22$ pp gain). | **`KEEP_LOCAL_CONFIGURATION`**<br/>Interface: [`betting_app/core/models/registry.py`](../betting_app/core/models/registry.py) |
| **Leading Decision-Time Betting Policy** | **Optimal Synergy Strategy** | Fixed50 Hybrid + Shin 2-way de-vigging + **Guard B** (`\|logit(p_C0) - logit(q_shin)\| ≤ 0.85`) + **Thesis Veto** (`\|p_C0 - p_039\| ≤ 0.10`) + `EV_net ∈ [5%, 25%]` + `Odds ≤ 4.00` + 1/4 Kelly. | **+4.83% Net CLV** ($t=3.00, p=0.0042$), **+23.91% Realized Yield** on 50-bet production ledger. Max DD 6.10%. | **`ADOPTED_LOCAL_STRATEGY`**<br/>Report: [`reports/optimal_synergy_c0_thesis_simulation_20261012.md`](../reports/optimal_synergy_c0_thesis_simulation_20261012.md)<br/>Code: [`betting_app/core/ev.py`](../betting_app/core/ev.py) |
| **Standalone Props: Map 1 Total Kills** | `prop_distribution_20261007_01` (Kills) | Regularized Ridge-1 Poisson-mean / Negative Binomial dispersion over 306 invariant pre-match pace features. | NLL **3.590488** (predictive-mean MAE 7.05 kills, 80% coverage 79.57%). | **`KEEP_BASELINE`**<br/>Bundle: [`data/06_models/prop_distribution/prop_distribution_20261007_01/`](../data/06_models/prop_distribution/prop_distribution_20261007_01/) |
| **Standalone Props: Map 1 Duration** | `prop_distribution_20261007_01` (Duration) | 15-leaf LightGBM / LogNormal distribution. | NLL **3.076194** (MAE 4.24 min, 80% coverage 75.99%). | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Standalone Props: Series Length (BO3/5)** | `prop_distribution_20261007_01` (Length) | Width-128 / Epoch-12 3-seed neural distribution mixture. | NLL **0.714580**, RPS **0.261305** on 2,358 BO3/5 matches. | **`KEEP_RESEARCH_CANDIDATE`**<br/>Code: [`src/models/prop_distribution.py`](../src/models/prop_distribution.py) |
| **Advanced Tournament & Pick'Em Simulator** | `tournament_numerical_20261007_01` + `crystal_ball_simulator` | • Sequential Riot Swiss draw lottery (sliding-slot mechanics, no-rematch backtracking).<br/>• Hard Fearless Draft map simulator.<br/>• Exact $O(2^K)$ knockout bracket DP.<br/>• Hierarchical player stats generator for Riot Worlds Pick'Em. | Rao-Blackwellized exact rational distributions; calibrated fair odds for all Worlds Pick'Em categories. | **`KEEP_NUMERICAL_IMPLEMENTATION`**<br/>Code: [`src/models/advanced_tournament_simulator.py`](../src/models/advanced_tournament_simulator.py), [`src/models/crystal_ball_simulator.py`](../src/models/crystal_ball_simulator.py) |

---

## 2. Canonical Research Controls & Baselines

| Model Identifier | Target / Scope | Architecture & Method | Benchmarks | Role & Governance |
|---|---|---|---|---|
| **Annual Causal A0** (`p_full`) | Historical match winner | Annual walk-forward causal ensemble (7 base checkpoints across 3 experts; quarterly cross-fitted calibration). | Full 11,550 series (2024–2026): **LogLoss 0.551246**, Brier **0.187103**.<br/>Common 9,907 series: **0.559920**. | **`KEEP_CONTROL` (Canonical Benchmark Reference)**<br/>Pinned in [`conf/base/research_benchmark.json`](../conf/base/research_benchmark.json). Evaluated via `scripts/run_model_benchmark.py --suite --doctor`. |
| **Frozen Thesis Baseline** (`EXP-039`) | Thesis reference & macro veto guard | Map-level symmetric calibrated ElasticNet with $W20$ team form + binomial BO3/BO5 series projection. | Common 9,907 series: LogLoss **0.567337** (16 deep tail losses). Missing 1,643 series lacking historical W20. | **`KEEP_CONTROL` (Immutable Thesis Baseline)**<br/>Preserved in `betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib`. Evaluated as independent sanity check in Thesis Model Veto. |
| **Calibrated Player Glicko-2** | Rating baseline | Match-level player rating engine with rating deviation (RD), volatility tracking, and logistic link. | Full 11,550 series: LogLoss **0.575888**, Brier **0.196548**.<br/>Common 9,907 series: **0.577998**. | **`KEEP_CONTROL`**<br/>Baseline rating control; evaluated in benchmark manifest. |
| **Calibrated Player Elo** | Rating baseline | Match-level calibrated Elo with player-level K-factor optimization. | Full 11,550 series: LogLoss **0.588946**, Brier **0.201930**.<br/>Common 9,907 series: **0.591198**. | **`KEEP_CONTROL`**<br/>Baseline rating control; evaluated in benchmark manifest. |

---

## 3. Preserved Standalone Prop Model Recovery Bundle

The standalone prop model is preserved in a byte-exact, self-contained recovery archive:

- **Location:** `data/06_models/model_preservation/prop_distribution_20261007_01.zip` (63.4 MB, 154 members)
- **SHA-256:** `a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232`
- **Selected Model Hash:** `9260e43dcd926a662c75a46d3e747f0629757fbedb061207c9234542570d110b`
- **Verification CLI:**
  ```bash
  .venv/bin/python scripts/preserve_prop_model.py verify \
    --bundle data/06_models/model_preservation/prop_distribution_20261007_01.zip \
    --sha256 a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232
  ```

---

## 4. Completed Empirical Studies & Archived Research

All historical studies remain reproducible at their immutable data stages in `data/` and documented in `reports/`. Dispositions follow scientific evaluation gates:

| Study Identifier | Research Objective | Outcome & Scientific Verdict | Evidence & Full Report |
|---|---|---|---|
| `optimal_synergy_c0_thesis_20261012` | C0 Shrunk Hybrid ($\alpha=0.50$) + Guard B + Thesis Veto on 10,000 PLN portfolio. | **ADOPTED_LOCAL_STRATEGY**: 50 bets, 58.0% win rate, **+23.91% Yield / +4.83% Net CLV**. Vetoed 9 bets with 11.1% win rate. | [`reports/optimal_synergy_c0_thesis_simulation_20261012.md`](../reports/optimal_synergy_c0_thesis_simulation_20261012.md) |
| `oddsportal_multiyear_20261012` | 5-season simulation on OddsPortal ($N=11,609$, 2021–2026) testing open vs close. | **COMPLETED_LARGE_SAMPLE_VALIDATION**: At 0% tax: **+10.15% Yield / +42.5% CAGR** ($N=3,528$). At 12% tax: **+9.65% Yield** ($N=684$). | [`reports/oddsportal_open_close_simulation_20261012.md`](../reports/oddsportal_open_close_simulation_20261012.md) |
| `c0_monthly_refit_20261011` | Census of 234,435 quotes across 1,600 matches; monthly walk-forward C0 retraining. | **COMPLETED_CENSUS_AND_BENCHMARK**: Shrunk Hybrid LogLoss **0.557877** vs Market 0.570054 ($-1.22$ pp gain). Mean Raw CLV $+19.13\%$. | [`reports/monthly_refit_c0_production_odds_since_may_20261011.md`](../reports/monthly_refit_c0_production_odds_since_may_20261011.md) |
| `c0_tournament_copula_20261007_01` | Player-shared Gaussian copula over complete tournament Monte Carlo paths. | **NOT_ADOPTED**: CAL path NLL slightly improves, but champion log-loss and placement RPS degrade. CAL guards fail. | [`reports/reliable_tournament_simulation_research_20261006.md`](../reports/reliable_tournament_simulation_research_20261006.md) §14 |
| `champion_history_20261007_01` | Multitask representation learning for joint winner and prop prediction. | **NOT_ADOPTED**: CAL-selected winner LogLoss 0.546321 vs C0 0.535449. Failed replacement gates. | [`reports/champion_history_prediction_20261007.md`](../reports/champion_history_prediction_20261007.md) |
| `c0_meta_stable_20261007_01` | 180-day player support profiles with 30-day role-meta features. | **NOT_ADOPTED**: Both calibration amplitudes zero; representations yield identity C0 probabilities. | [`docs/RESEARCH.md`](RESEARCH.md#2026-10-07--c0-meta-stable-follow-up) |
| `c0_ev_loss_20261006_01` | Training sports betting heads with after-tax return-MSE objective vs likelihood. | **NOT_ADOPTED**: Multiplicity-adjusted risk CI $[-0.0008, +0.0016]$ crosses zero; after-cost guards fail. | [`reports/c0_selected_betting_reliability_research_and_audit_20261006.md`](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md) §11 |
| `native_capacity_20261004_01` | Scaling model capacity to Width 64 and Epoch 24 + knowledge distillation. | **NOT_ADOPTED**: Bonferroni confidence intervals cross zero; distillation skipped by evaluation gate. | [`reports/model_targets_capacity_and_distillation_research_20261004.md`](../reports/model_targets_capacity_and_distillation_research_20261004.md) |
| `c0_raw_connector_20261002_033658` | Learned specialist connector combining C0 with fresh logistic/tree ensemble. | **NOT_ADOPTED**: Primary LL contrasts negative, but BO5 and source-unseen replacement guards fail. | [`reports/plan09_c0_connector_and_forward_evidence_20261002.md`](../reports/plan09_c0_connector_and_forward_evidence_20261002.md) |
| `reliability_tradeoff_20261001` | Comparing Native-6 vs Native-12 epochs for calibration-first trade-off. | **NOT_ADOPTED**: Native-6 LogLoss worsens by $+0.000867$; RMS calibration gain below required $0.5$ pp. | [`reports/noisy_data_reliability_research_20260929.md`](../reports/noisy_data_reliability_research_20260929.md) §13 |
| `a0_data_focus_20260929_192443` | High-tier / market-tier neural filtering and history token dropout. | **NOT_ADOPTED** (Challengers): Uniform control is retained as the **active Native C0 baseline**. | [`reports/bookmaker_noise_and_provider_training_20260927.md`](../reports/bookmaker_noise_and_provider_training_20260927.md) §10 |
| `native_refit_20260929_122617` | Exponential time decay (H365) vs uniform weighting on neural history. | **NOT_ADOPTED**: H365 worsens performance across all cadences. Retain uniform history weighting. | [`reports/bookmaker_noise_and_provider_training_20260927.md`](../reports/bookmaker_noise_and_provider_training_20260927.md) §9 |

---

## 5. Developer Navigation & Documentation Map

- **Comprehensive Empirical Journal:** [`docs/RESEARCH.md`](RESEARCH.md) — Exhaustive statistical receipts, CI bounds, and verification logs.
- **Architectural Specification & Contracts:** [`docs/PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) — Data contracts, schema definitions, and process isolation.
- **Master Research Reports Index:** [`reports/README.md`](../reports/README.md) — Index of all 40+ research papers and econometric audits.
- **Application Backend Contracts:** [`betting_app/README.md`](../betting_app/README.md) — API routes, scrapers, models, and local test instructions.
- **Executable Script Reference:** [`scripts/README.md`](../scripts/README.md) — Command-line interface definitions and usage guides.
