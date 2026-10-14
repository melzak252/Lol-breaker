# LoL-Breaker — Data Catalog & Provenance Architecture

**Version:** 1.0.0 (Kedro 8-Layer + DVC Orchestration)  
**Last Updated:** 2026-10-15  
**Remote Storage:** Google Drive ([`gdrive://1Koo0-3d-1BssznDWXzz9PactUfmZbn7g`](https://drive.google.com/drive/folders/1Koo0-3d-1BssznDWXzz9PactUfmZbn7g))
**Declarative Catalogs:** [`conf/base/catalog.yml`](../conf/base/catalog.yml), [`conf/base/parameters.yml`](../conf/base/parameters.yml)  
**DVC Pipeline:** [`dvc.yaml`](../dvc.yaml), [`dvc.lock`](../dvc.lock)

---

## 1. Executive Summary & Storage Architecture

This catalog is the definitive reference for all datasets, model inputs, out-of-time probability panels, feature tensors, and verification receipts tracked across **LoL-Breaker**.

To guarantee 100% offline reproducibility without committing heavy data dumps into Git, the repository implements the **Kedro 8-Layer Data Engineering Architecture** paired with **DVC (Data Version Control)**. Raw bytes and parquet/zip archives (~117 MB total) are content-addressed and hosted in a dedicated Google Drive remote, while lightweight cryptographic pointer files (`*.dvc` and `dvc.lock`) are tracked in Git.
The remote storage folder contains:
1. **`files/`**: DVC content-addressed storage (`files/md5/...`) matching `dvc.lock` for automated `dvc pull` synchronization.
2. **`curated_data/`**: Human-readable directory tree for browsing individual CSV and Parquet files directly in Google Drive.
3. **`lol_breaker_complete_data_bundle.zip` (77.7 MB)**: Single-click full archive download for instant setup.

```text
Google Drive Remote Storage (gdrive://1Koo0-3d-1BssznDWXzz9PactUfmZbn7g)
  │
  ├── dvc pull / dvc push (Transfers raw data bytes via OAuth2)
  │
Local Repository (lol-breaker/)
  ├── conf/base/catalog.yml           <- Declarative dataset catalog with SHA-256 pins
  ├── dvc.yaml & dvc.lock             <- 5-stage simulation pipeline DAG & content locks
  └── data/                           <- Kedro 8-Layer Directory Tree (gitignored)
      ├── 01_raw/                     <- Immutable raw scrapes (GOL.GG, OddsPortal, bookmakers)
      ├── 02_intermediate/            <- Parsed & typed quotes, normalized team identities
      ├── 03_primary/                 <- Canonical aligned match series & targets
      ├── 04_feature/                 <- Feature tensors, pre-match 2h quote trajectories
      ├── 05_model_input/             <- Consolidated 11,550-match walk-forward benchmark
      ├── 06_models/                  <- Frozen neural weights & standalone prop bundles
      ├── 07_model_output/            <- Out-of-time prediction panels & joint cohorts
      └── 08_reporting/               <- Simulation ledgers, metric tables, & Markdown reports
```

---

## 2. Complete Curated Data Bundle Inventory

The table below lists all 11 curated datasets managed by DVC and consumed by the simulation pipeline:

| Catalog Dataset Key | Layer | File Path & DVC Pointer | Format | Size | Rows | SHA-256 Digest |
|---|---|---|---|---:|---:|---|
| **`canonical_benchmark_paired`** | `05_model_input` | `data/05_model_input/canonical_benchmark/` (`.dvc`) | Directory (382 files) | 43.52 MB | 11,550 | *Verified via internal files (see §3.1)* |
| **`prop_distribution_bundle`** | `06_models` | `data/06_models/model_preservation/prop_distribution_20261007_01.zip` (`.dvc`) | ZIP archive | 60.45 MB | N/A | `a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232` |
| **`c0_2026_assessment_joined`** | `08_reporting` | `data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/all_2026_joined.parquet` (`.dvc`) | Parquet | 1.79 MB | 3,103 | `5fbcd5e2ac213ab9dadbc425e68edb33d84943e1006b9b903d1b6bc57e5479a8` |
| **`c0_2026_market_t1_joined`** | `08_reporting` | `data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet` (`.dvc`) | Parquet | 1.02 MB | 1,668 | `91b7c364b5900363fe5d1e383752cfb3a5f2592aeb3cd85615bf88115e400491` |
| **`polish_bookmaker_panel`** | `07_model_output` | `data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet` (`.dvc`) | Parquet | 0.21 MB | 2,752 | `6c8334fcd207997003c3ab6e97b5b987dab5cd3950ca8f23adecbe4cc3cbc5f4` |
| **`oddsportal_joint_cohorts`** | `07_model_output` | `data/07_model_output/oddsportal/c0_thesis_books_joint_20261009_115922Z/joint_scoring_cohorts.parquet` (`.dvc`) | Parquet | 0.16 MB | 2,023 | `b99e32cce63157a802bb3913363cfc40b467a02a3ab94390ecd26f837afac321` |
| **`multiyear_oddsportal_odds`** | Root Data | `data/odds.csv` (`.dvc`) | CSV | 5.51 MB | 16,375 | `33504cb7f6b6e6af16ccae4c266b5ceebc2b3a9f821a1cd9a63860c510e20ed4` |
| **`multiyear_model_predictions`** | Root Data | `data/model_predictions_2020_2026.parquet` (`.dvc`) | Parquet | 1.64 MB | 26,503 | `f23d50467a0bf2236b13bd7b3c01cfb065f49d4d51a1160f418a763cb995e917` |
| **`production_2h_trajectory`** | `04_feature` | `data/04_feature/production_2h_odds_trajectory.parquet` (`.dvc`) | Parquet | 0.18 MB | 7,505 | `bde52975de0331a7d1238bdbfcd735190f964b27b0e74c2bf8d75d1e83a821c5` |
| **`optimal_synergy_ledger`** | `08_reporting` | `data/08_reporting/optimal_strategy_simulation_ledger.csv` (`.dvc`) | CSV | 0.01 MB | 50 | `d4b960e204b99ae7288cfbb3912eac0b3296a9b08666a1d2d0dff9b8dc9cd434` |
| **`prop_distribution_metrics`** | `08_reporting` | `data/08_reporting/prop_distribution/prop_distribution_20261007_01/metrics.json` (`.dvc`) | JSON | 0.01 MB | N/A | `cde886218dd5c22fe6093fc3c670897927ac2b61d95fd0c442c568f43e5e438b` |

---

## 3. Dataset Profiles & Provenance Deep-Dives

### 3.1 Canonical Benchmark Consolidated Archive (`data/05_model_input/canonical_benchmark`)

- **Format:** Directory containing 382 files (43.52 MB).
- **Where It Comes From:** Consolidated from the canonical research archive (`rating-foundation-20260915`, `a0-phase-walkforward-20260914`, `nonlinear-history-20260912`, `unified-ratings-20260911`, and `auxiliary-player-20260912`).
- **Core File:** `rating-foundation-20260915/data/08_reporting/paired.parquet` (11,550 series, 1.78 MB).
  - **SHA-256:** `8041b6c7ffd4a99e4c67d55c3665c22c8cd653c8dffe4f153b10dfd9b264dfaf` (Pinned in `conf/base/research_benchmark.json`).
- **Key Columns in `paired.parquet`:**
  - `golgg_match_id` (int64): Unique GOL.GG series identifier.
  - `date` (str): Series scheduled match date (`YYYY-MM-DD`).
  - `y` (int64): Binary series outcome ($1$ = Team 1 won, $0$ = Team 2 won).
  - `p_full` (float64): Annual Causal A0 ensemble walk-forward prediction.
  - `p_039` (float64): Frozen Thesis Baseline (`EXP-039`) prediction ($9,907$ non-null rows).
  - `p_glicko`, `p_elo` (float64): Calibrated rating baseline predictions.
  - `market_open`, `market_close` (float64): Archival OddsPortal opening and closing implied probabilities ($2,673$ non-null rows; $2,275$ with non-null `p_039`).
- **Temporal Scope:** 2024-01-01 to 2026-09-08 (covers all major and secondary professional LoL leagues).
- **Cutoff Guarantees:** Features and ratings strictly respect `effective_release_day < prediction_day`. Roster lineups are validated historical records.

---

### 3.2 Canonical 2026 Assessment Panel (`all_2026_joined.parquet`)

- **Path:** `data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/all_2026_joined.parquet`
- **Size:** 1.79 MB | **Rows:** 3,103 series | **Columns:** 103
- **Where It Comes From:** Output of study `a0_data_focus_20260929_192443` (completed 2026-09-29), which evaluated high-tier neural focus, market weights, and history dropout against the uniform control.
- **Key Columns:**
  - `golgg_match_id` (int64): Match identifier.
  - `y` (int64): Realized binary match winner.
  - `control__full__full` (float64): **Active Native Causal-C0** baseline forecast (W32/E12 3-seed history attention).
  - `p_a0_full` (float64): Annual Causal A0 reference forecast.
  - `p_039` (float64): EXP-039 thesis forecast ($N=2,699$ supported).
  - `p_glicko`, `p_elo` (float64): Glicko-2 and Elo rating forecasts.
- **Role in Simulation:** Stage 2 Cohort A evaluation (computes unclipped LogLoss, Brier, Accuracy, and bootstrap $\Delta\text{LL}$).

---

### 3.3 Pinnacle / OddsPapi T-1h Joined Panel (`market_T1_joined.parquet`)

- **Path:** `data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/market_T1_joined.parquet`
- **Size:** 1.02 MB | **Rows:** 1,668 series | **Columns:** 106
- **Where It Comes From:** Join between canonical 2026 series and OddsPapi archival quotes at exact $T-1\text{h}$ pre-match cutoff from Pinnacle.
- **Key Columns:**
  - `control__full__full`: Native C0 probability.
  - `q_consensus`: Pinnacle market consensus probability (Shin de-vigged, strictly available at $T-1\text{h}$).
  - `y`: Match winner outcome.
- **Role in Simulation:** Stage 2 Cohort C Bayesian shrinkage proper-score grid on sharp bookmaker lines ($N=1,668$).

---

### 3.4 Polish Bookmaker Paired Panel (`paired_probability_panel.parquet`)

- **Path:** `data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet`
- **Size:** 0.21 MB | **Rows:** 2,752 series | **Columns:** 24
- **Where It Comes From:** Generated by Plan 12 (`scripts/run_model_bookmaker_competitiveness.py`), capturing pre-match odds across 7 Polish licensed sportsbooks (STS, Fortuna, Superbet, Betclic, Totalbet, LV BET, Forbet).
- **Key Columns:**
  - `canonical_match_id`: Harmonized fixture identifier.
  - `prob_c0_mean`: Native C0 win probability.
  - `prob_thesis_039`: EXP-039 thesis baseline probability.
  - `odds_home`, `odds_away`: Available decimal odds at decision time.
  - `p_consensus`: Devigged consensus market probability.
- **Role in Simulation:** Cross-bookmaker market margin audits, overround calculation, and competitiveness diagnostics.

---

### 3.5 OddsPortal Joint Bookmaker Consensus Cohort (`joint_scoring_cohorts.parquet`)

- **Path:** `data/07_model_output/oddsportal/c0_thesis_books_joint_20261009_115922Z/joint_scoring_cohorts.parquet`
- **Size:** 0.16 MB | **Rows:** 2,023 series | **Columns:** 28
- **Where It Comes From:** Joint scoring analysis of OddsPortal multi-bookmaker lines across 2024–2026 (`c0_thesis_books_joint_20261009`).
- **Key Columns:**
  - `p_C0`: Native C0 model probability.
  - `p_consensus`: Multi-bookmaker devigged consensus probability.
  - `p_pinnacle`, `p_bet365`, `p_1xbet`, `p_bwin`, `p_unibet`: Individual bookmaker lines.
  - `book_count`: Number of synchronized bookmaker quotes available at cutoff.
- **Role in Simulation:** Stage 2 Cohort C Bayesian shrinkage proper-score grid ($N=2,023$, establishing joint minimum LogLoss at $\alpha=0.50$).

---

### 3.6 Multi-Year OddsPortal Odds (`odds.csv`) & Predictions (`model_predictions_2020_2026.parquet`)

- **Path:** `data/odds.csv` (5.51 MB, 16,375 rows) & `data/model_predictions_2020_2026.parquet` (1.64 MB, 26,503 rows).
- **Where It Comes From:** Historical scrape of OddsPortal archive from 2020 to 2026 merged with out-of-time model backtest forecasts from `EXP-039` and `EXP-081 Siamese`.
- **Key Columns:**
  - `golgg_match_id`: Match series ID.
  - `avg_open_home`, `avg_open_away`: Market opening prices.
  - `avg_odds_home`, `avg_odds_away`: Market closing prices.
  - `prob_exp039`, `prob_exp081_siamese`: Historical model forecasts.
- **Role in Simulation:** Stage 3 Simulation B multi-year stability analysis ($N=3,002$ bets over 6 seasons, 2021–2026).

---

### 3.7 Production 2-Hour Odds Trajectory (`production_2h_odds_trajectory.parquet`)

- **Path:** `data/04_feature/production_2h_odds_trajectory.parquet`
- **Size:** 0.18 MB | **Rows:** 7,505 quote snapshots | **Columns:** 22
- **Where It Comes From:** Scraped by operational scrapers (`betting_app/scrapers/`) tracking price movements within the final 2 hours before match start across Polish bookmakers.
- **Key Columns:**
  - `scraped_at`: High-resolution UTC observation timestamp.
  - `start_time_normalized`: Official scheduled match start time.
  - `minutes_to_start`: Discrete decision horizon (e.g. 120m, 60m, 15m).
  - `bookmaker`: Provider name (`sts`, `efortuna`, `superbet`, `betclic`, `totalbet`).
  - `odds_a`, `odds_b`: Available decimal odds.
  - `p_model`: As-of model forecast.
- **Role in Simulation:** Input to `HistoricalProductionSimulator` for discrete time-step operational queue replay.

---

### 3.8 Production Optimal Synergy Ledger (`optimal_strategy_simulation_ledger.csv`)

- **Path:** `data/08_reporting/optimal_strategy_simulation_ledger.csv`
- **Size:** 0.01 MB | **Rows:** 50 bets | **Columns:** 20
- **Where It Comes From:** Generated by the Optimal Synergy study (`reports/optimal_synergy_c0_thesis_simulation_20261012.md`), recording all 50 bets qualified between 2026-05-30 and 2026-09-08 under the adopted production rules (Shrunk Hybrid $\alpha=0.50$, Guard B $\le 0.85$, Thesis Veto $\le 0.10$, 12% Polish tax, 1/4 Kelly, 150 PLN cap).
- **Key Columns:**
  - `bet_id` (int): Sequential bet identifier.
  - `date`, `month` (str): UTC timestamp and calendar month.
  - `league`, `team_selected`, `opponent`, `bookmaker`: Match context.
  - `odds_taken`, `closing_odds` (float): Line taken vs. closing price.
  - `p_c0`, `p_thesis`, `p_hybrid` (float): C0, EXP-039, and 50/50 shrunk probability.
  - `model_ev_pct`: Expected value under post-tax hybrid probability.
  - `raw_clv_pct`, `net_clv_pct`: Price-beating CLV vs. post-tax closing value.
  - `stake`, `won`, `profit`, `bankroll`, `drawdown_pct`: Realized financial trajectory.
- **Role in Simulation:** Ground truth ledger for Stage 3 validation and Stage 4 econometric decomposition.

---

### 3.9 Preserved Prop Distribution Bundle (`prop_distribution_20261007_01.zip`)

- **Path:** `data/06_models/model_preservation/prop_distribution_20261007_01.zip`
- **Size:** 60.45 MB | **SHA-256:** `a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232`
- **Where It Comes From:** Preserved by `scripts/preserve_prop_model.py` from study `prop_distribution_20261007_01`. Contains complete source code, frozen weights, feature extractors, and test fixtures for odds-free LoL prop distributions across 306 invariant pre-match features.
- **Models Inside:**
  1. **Map 1 Total Kills:** Regularized Ridge-1 Poisson/NB dispersion (NLL `3.590488`).
  2. **Map 1 Duration:** 15-leaf LightGBM / LogNormal (NLL `3.076194`).
  3. **Series Length (BO3/5):** Width-128 / Epoch-12 3-seed mixture (NLL `0.714580`, RPS `0.261305`).
- **Companion File:** `data/08_reporting/prop_distribution/prop_distribution_20261007_01/metrics.json` (SHA-256: `cde886218dd5c22fe6093fc3c670897927ac2b61d95fd0c442c568f43e5e438b`).

---

## 4. Information Availability & Cutoff Invariants

To eliminate lookahead bias and survivorship distortion, every dataset conforms to the sports modeling contract:

```text
source_release_time <= data_cutoff_at <= predicted_at < target_start_at
```

1. **Strictly Prior Feature Cutoffs:** Player statistics, ratings, and attendance histories update as-of the match eve; no information from future games or later tournament phases leaks into prior rows.
2. **Side Symmetry ($A \leftrightarrow B$):** All model inputs preserve mathematical symmetry:
   $$p(A, B) + p(B, A) = 1.0 \pm 10^{-7}$$
   Swapping team slots inverts the logit identically without artificial home/away bias.
3. **Price Availability Synchronization:** Odds snapshots used for EV qualification were observed and recorded prior to match start; closing lines ($p_{\text{close}}$) are used exclusively for settlement and CLV audit, never for bet qualification.

---

## 5. DVC & Data Management Operations

### 5.1 Pulling Data on a Fresh Clone

To download all datasets from Google Drive onto a new workstation:

```bash
# 1. Clone repository
git clone https://github.com/melzak252/Lol-breaker.git
cd Lol-breaker

# 2. Pull all DVC-tracked datasets from Google Drive
.venv/bin/dvc pull

# 3. Verify benchmark manifest doctor
.venv/bin/python scripts/run_model_benchmark.py --suite conf/base/research_benchmark.json --doctor
```

### 5.2 Reproducing the Simulation Pipeline

```bash
# Reproduce full 5-stage DAG and update dvc.lock
.venv/bin/dvc repro

# View tracked summary metrics
.venv/bin/dvc metrics show
```

### 5.3 Uploading Updates to Google Drive

When adding or updating datasets:

```bash
# Track modified dataset with DVC
.venv/bin/dvc add data/<path_to_dataset>

# Push data files to Google Drive
.venv/bin/dvc push

# Commit updated pointer files and locks to Git
git add data/<path_to_dataset>.dvc dvc.lock
git commit -m "data: update <dataset_name> in DVC storage"
git push origin dev
```
