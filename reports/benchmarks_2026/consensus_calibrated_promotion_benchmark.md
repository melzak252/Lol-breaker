# Model Benchmark Report: Calibrated Consensus Rating (12-way) (consensus_calibrated)

- **Cohort Period:** `2024-01-14` to `2026-06-14`
- **Sample Size:** 9,482 matches

## 1. Core Performance & Calibration

| Metric | Value | Reference Target |
| :--- | :---: | :---: |
| **LogLoss** | **0.58433** | Primary loss (min) |
| **Brier Score** | **0.20004** | Quadratic error (min) |
| -- Brier Reliability | 0.00082 | Calibration component (< 0.010) |
| -- Brier Resolution | 0.04868 | Discrimination component (max) |
| **ROC-AUC** | **0.7562** | Ranking quality (max) |
| **Accuracy (P >= 0.50)** | **68.53%** | Threshold 0.50 |
| **ECE (10 bins)** | **0.0267** | Calibration error (<= 0.030) |
| **MCE (Max Bin Error)** | 0.0441 | Worst-bin deviation |
| **Calibration Slope** | **1.001** | Target 1.0 (range [0.85, 1.15]) |
| **Calibration Intercept** | 0.108 | Target 0.0 (bias) |

## 2. Paired Comparison vs EXP-039 Frozen Baseline

- **Delta LogLoss:** `+0.025067` (negative is better)
- **95% Monthly-block Bootstrap CI:** `[+0.018783, +0.033067]`
- **p-value (one-sided):** `1.0000`
- **Statistically Superior:** `NO (CI contains >= 0)`
- **Delta Brier:** `+0.010207`
- **Delta AUC:** `-0.0257`
- **Delta Accuracy:** `-2.17 p.p.`

## 3. Diagnostic Slices ('Gdzie model się myli')

### Slice: Confidence / Expected Prob

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Heavy Underdog (P < 0.25) | 1,191 | 0.4340 | 0.1323 | 84.2% | 0.155 | 0.158 | +0.003 |
| Moderate Underdog (0.25 <= P < 0.40) | 1,785 | 0.6405 | 0.2243 | 65.2% | 0.329 | 0.348 | +0.019 |
| Coin-Flip / Close (0.40 <= P <= 0.60) | 3,032 | 0.6864 | 0.2466 | 54.6% | 0.502 | 0.532 | +0.030 |
| Moderate Favorite (0.60 < P <= 0.75) | 1,928 | 0.6070 | 0.2083 | 70.3% | 0.670 | 0.703 | +0.033 |
| Heavy Favorite (P > 0.75) | 1,546 | 0.4070 | 0.1225 | 85.5% | 0.846 | 0.855 | +0.009 |

### Slice: Series Format (BoN)

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bo1 | 4,586 | 0.6041 | 0.2085 | 66.6% | 0.503 | 0.530 | +0.027 |
| Bo3 | 3,693 | 0.5535 | 0.1870 | 71.5% | 0.514 | 0.530 | +0.016 |
| Bo5 | 1,203 | 0.6036 | 0.2079 | 66.7% | 0.572 | 0.593 | +0.021 |

### Slice: Competition Tier

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| development | 2,041 | 0.6094 | 0.2101 | 67.1% | 0.511 | 0.527 | +0.016 |
| international | 197 | 0.6329 | 0.2214 | 65.0% | 0.548 | 0.574 | +0.026 |
| major | 1,772 | 0.5845 | 0.1992 | 69.2% | 0.517 | 0.528 | +0.011 |
| minor_top_level | 1,080 | 0.5234 | 0.1758 | 72.4% | 0.520 | 0.526 | +0.006 |
| regional | 4,392 | 0.5854 | 0.2007 | 68.1% | 0.516 | 0.548 | +0.032 |

### Slice: Player-Team Disagreement

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| High Agreement (Delta <= 0.05) | 4,086 | 0.5594 | 0.1892 | 71.0% | 0.525 | 0.541 | +0.016 |
| Moderate Agreement (0.05 < Delta <= 0.12) | 2,983 | 0.5891 | 0.2023 | 68.3% | 0.511 | 0.530 | +0.019 |
| High Disagreement (Delta > 0.12) | 2,413 | 0.6207 | 0.2156 | 64.6% | 0.507 | 0.543 | +0.036 |
| Severe Disagreement (Delta > 0.20) | 952 | 0.6194 | 0.2150 | 65.8% | 0.514 | 0.550 | +0.037 |

## 4. Promotion Gate Evaluation

**Status:** **FAILED (Promotion blocked)**

| Gate Criterion | Status |
| :--- | :---: |
| Calibration slope within [0.85, 1.15] | PASS |
| ECE within threshold (<= baseline ECE) | PASS |
| Statistically superior (95% Bootstrap CI upper bound < 0) | FAIL |
| Underdog quarantine safety (no false favorites on odds > 3.50) | PASS |

**Evaluation Notes & Gate Diagnostics:**
- 95% bootstrap CI [+0.018783, +0.033067] does not prove significant superiority
