# Model Benchmark Report: Linear79 Symmetric Model (linear79)

- **Cohort Period:** `2024-01-14` to `2026-06-14`
- **Sample Size:** 9,482 matches

## 1. Core Performance & Calibration

| Metric | Value | Reference Target |
| :--- | :---: | :---: |
| **LogLoss** | **0.55948** | Primary loss (min) |
| **Brier Score** | **0.18985** | Quadratic error (min) |
| -- Brier Reliability | 0.00054 | Calibration component (< 0.010) |
| -- Brier Resolution | 0.05843 | Discrimination component (max) |
| **ROC-AUC** | **0.7816** | Ranking quality (max) |
| **Accuracy (P >= 0.50)** | **70.77%** | Threshold 0.50 |
| **ECE (10 bins)** | **0.0219** | Calibration error (<= 0.030) |
| **MCE (Max Bin Error)** | 0.0314 | Worst-bin deviation |
| **Calibration Slope** | **1.006** | Target 1.0 (range [0.85, 1.15]) |
| **Calibration Intercept** | 0.115 | Target 0.0 (bias) |

## 2. Paired Comparison vs EXP-039 Frozen Baseline

- **Delta LogLoss:** `+0.000217` (negative is better)
- **95% Monthly-block Bootstrap CI:** `[-0.001449, +0.002035]`
- **p-value (one-sided):** `0.6136`
- **Statistically Superior:** `NO (CI contains >= 0)`
- **Delta Brier:** `+0.000014`
- **Delta AUC:** `-0.0003`
- **Delta Accuracy:** `+0.06 p.p.`

## 3. Diagnostic Slices ('Gdzie model się myli')

### Slice: Confidence / Expected Prob

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Heavy Underdog (P < 0.25) | 1,617 | 0.4289 | 0.1315 | 83.9% | 0.150 | 0.161 | +0.011 |
| Moderate Underdog (0.25 <= P < 0.40) | 1,584 | 0.6496 | 0.2285 | 64.5% | 0.329 | 0.355 | +0.026 |
| Coin-Flip / Close (0.40 <= P <= 0.60) | 2,533 | 0.6874 | 0.2471 | 55.2% | 0.501 | 0.531 | +0.029 |
| Moderate Favorite (0.60 < P <= 0.75) | 1,846 | 0.6155 | 0.2124 | 69.0% | 0.672 | 0.690 | +0.017 |
| Heavy Favorite (P > 0.75) | 1,902 | 0.3707 | 0.1091 | 87.3% | 0.851 | 0.873 | +0.022 |

### Slice: Series Format (BoN)

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bo1 | 4,586 | 0.5780 | 0.1977 | 69.3% | 0.504 | 0.530 | +0.025 |
| Bo3 | 3,693 | 0.5273 | 0.1764 | 73.2% | 0.513 | 0.530 | +0.017 |
| Bo5 | 1,203 | 0.5877 | 0.2012 | 68.9% | 0.569 | 0.593 | +0.024 |

### Slice: Competition Tier

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| development | 2,041 | 0.5846 | 0.2003 | 68.7% | 0.513 | 0.527 | +0.014 |
| international | 197 | 0.5761 | 0.1971 | 70.1% | 0.558 | 0.574 | +0.016 |
| major | 1,772 | 0.5827 | 0.1987 | 69.5% | 0.513 | 0.528 | +0.015 |
| minor_top_level | 1,080 | 0.5166 | 0.1717 | 73.1% | 0.515 | 0.526 | +0.011 |
| regional | 4,392 | 0.5482 | 0.1856 | 71.7% | 0.517 | 0.548 | +0.031 |

### Slice: Player-Team Disagreement

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| High Agreement (Delta <= 0.05) | 4,086 | 0.5446 | 0.1834 | 72.2% | 0.523 | 0.541 | +0.018 |
| Moderate Agreement (0.05 < Delta <= 0.12) | 2,983 | 0.5631 | 0.1915 | 70.8% | 0.511 | 0.530 | +0.019 |
| High Disagreement (Delta > 0.12) | 2,413 | 0.5803 | 0.1987 | 68.2% | 0.510 | 0.543 | +0.033 |
| Severe Disagreement (Delta > 0.20) | 952 | 0.5669 | 0.1926 | 70.1% | 0.517 | 0.550 | +0.033 |

## 4. Promotion Gate Evaluation

**Status:** **FAILED (Promotion blocked)**

| Gate Criterion | Status |
| :--- | :---: |
| Calibration slope within [0.85, 1.15] | PASS |
| ECE within threshold (<= baseline ECE) | PASS |
| Statistically superior (95% Bootstrap CI upper bound < 0) | FAIL |
| Underdog quarantine safety (no false favorites on odds > 3.50) | PASS |

**Evaluation Notes & Gate Diagnostics:**
- 95% bootstrap CI [-0.001449, +0.002035] does not prove significant superiority
