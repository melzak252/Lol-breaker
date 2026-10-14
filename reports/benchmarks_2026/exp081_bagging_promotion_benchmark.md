# Model Benchmark Report: EXP-081 Bagging Recipe (exp081_bagging)

- **Cohort Period:** `2024-01-14` to `2026-06-14`
- **Sample Size:** 9,482 matches

## 1. Core Performance & Calibration

| Metric | Value | Reference Target |
| :--- | :---: | :---: |
| **LogLoss** | **0.56188** | Primary loss (min) |
| **Brier Score** | **0.19094** | Quadratic error (min) |
| -- Brier Reliability | 0.00066 | Calibration component (< 0.010) |
| -- Brier Resolution | 0.05772 | Discrimination component (max) |
| **ROC-AUC** | **0.7792** | Ranking quality (max) |
| **Accuracy (P >= 0.50)** | **70.49%** | Threshold 0.50 |
| **ECE (10 bins)** | **0.0241** | Calibration error (<= 0.030) |
| **MCE (Max Bin Error)** | 0.0382 | Worst-bin deviation |
| **Calibration Slope** | **1.050** | Target 1.0 (range [0.85, 1.15]) |
| **Calibration Intercept** | 0.119 | Target 0.0 (bias) |

## 2. Paired Comparison vs EXP-039 Frozen Baseline

- **Delta LogLoss:** `+0.002615` (negative is better)
- **95% Monthly-block Bootstrap CI:** `[+0.001038, +0.004199]`
- **p-value (one-sided):** `0.9984`
- **Statistically Superior:** `NO (CI contains >= 0)`
- **Delta Brier:** `+0.001102`
- **Delta AUC:** `-0.0028`
- **Delta Accuracy:** `-0.21 p.p.`

## 3. Diagnostic Slices ('Gdzie model się myli')

### Slice: Confidence / Expected Prob

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Heavy Underdog (P < 0.25) | 1,614 | 0.4279 | 0.1316 | 84.0% | 0.162 | 0.160 | -0.002 |
| Moderate Underdog (0.25 <= P < 0.40) | 1,600 | 0.6546 | 0.2309 | 63.8% | 0.328 | 0.362 | +0.034 |
| Coin-Flip / Close (0.40 <= P <= 0.60) | 2,535 | 0.6878 | 0.2473 | 54.8% | 0.502 | 0.530 | +0.028 |
| Moderate Favorite (0.60 < P <= 0.75) | 1,859 | 0.6132 | 0.2113 | 69.4% | 0.673 | 0.694 | +0.021 |
| Heavy Favorite (P > 0.75) | 1,874 | 0.3768 | 0.1114 | 86.9% | 0.839 | 0.869 | +0.030 |

### Slice: Series Format (BoN)

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bo1 | 4,586 | 0.5824 | 0.1996 | 68.5% | 0.504 | 0.530 | +0.026 |
| Bo3 | 3,693 | 0.5305 | 0.1778 | 73.5% | 0.513 | 0.530 | +0.017 |
| Bo5 | 1,203 | 0.5802 | 0.1984 | 68.7% | 0.562 | 0.593 | +0.031 |

### Slice: Competition Tier

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| development | 2,041 | 0.5882 | 0.2016 | 68.8% | 0.512 | 0.527 | +0.015 |
| international | 197 | 0.5942 | 0.2040 | 70.1% | 0.550 | 0.574 | +0.024 |
| major | 1,772 | 0.5790 | 0.1980 | 69.5% | 0.513 | 0.528 | +0.015 |
| minor_top_level | 1,080 | 0.5152 | 0.1713 | 73.4% | 0.514 | 0.526 | +0.012 |
| regional | 4,392 | 0.5528 | 0.1874 | 71.0% | 0.515 | 0.548 | +0.033 |

### Slice: Player-Team Disagreement

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| High Agreement (Delta <= 0.05) | 4,086 | 0.5449 | 0.1837 | 72.0% | 0.521 | 0.541 | +0.019 |
| Moderate Agreement (0.05 < Delta <= 0.12) | 2,983 | 0.5684 | 0.1936 | 70.6% | 0.510 | 0.530 | +0.020 |
| High Disagreement (Delta > 0.12) | 2,413 | 0.5825 | 0.2000 | 67.9% | 0.509 | 0.543 | +0.034 |
| Severe Disagreement (Delta > 0.20) | 952 | 0.5725 | 0.1953 | 69.1% | 0.517 | 0.550 | +0.034 |

## 4. Promotion Gate Evaluation

**Status:** **FAILED (Promotion blocked)**

| Gate Criterion | Status |
| :--- | :---: |
| Calibration slope within [0.85, 1.15] | PASS |
| ECE within threshold (<= baseline ECE) | PASS |
| Statistically superior (95% Bootstrap CI upper bound < 0) | FAIL |
| Underdog quarantine safety (no false favorites on odds > 3.50) | PASS |

**Evaluation Notes & Gate Diagnostics:**
- 95% bootstrap CI [+0.001038, +0.004199] does not prove significant superiority
