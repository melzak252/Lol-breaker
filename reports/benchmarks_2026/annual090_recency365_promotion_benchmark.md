# Model Benchmark Report: Annual Recency 365d Model (annual090_recency365)

- **Cohort Period:** `2024-01-14` to `2026-06-14`
- **Sample Size:** 9,482 matches

## 1. Core Performance & Calibration

| Metric | Value | Reference Target |
| :--- | :---: | :---: |
| **LogLoss** | **0.56206** | Primary loss (min) |
| **Brier Score** | **0.19060** | Quadratic error (min) |
| -- Brier Reliability | 0.00073 | Calibration component (< 0.010) |
| -- Brier Resolution | 0.05808 | Discrimination component (max) |
| **ROC-AUC** | **0.7803** | Ranking quality (max) |
| **Accuracy (P >= 0.50)** | **70.99%** | Threshold 0.50 |
| **ECE (10 bins)** | **0.0238** | Calibration error (<= 0.030) |
| **MCE (Max Bin Error)** | 0.0437 | Worst-bin deviation |
| **Calibration Slope** | **0.929** | Target 1.0 (range [0.85, 1.15]) |
| **Calibration Intercept** | 0.117 | Target 0.0 (bias) |

## 2. Paired Comparison vs EXP-039 Frozen Baseline

- **Delta LogLoss:** `+0.002797` (negative is better)
- **95% Monthly-block Bootstrap CI:** `[+0.000380, +0.005502]`
- **p-value (one-sided):** `0.9866`
- **Statistically Superior:** `NO (CI contains >= 0)`
- **Delta Brier:** `+0.000770`
- **Delta AUC:** `-0.0017`
- **Delta Accuracy:** `+0.28 p.p.`

## 3. Diagnostic Slices ('Gdzie model się myli')

### Slice: Confidence / Expected Prob

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Heavy Underdog (P < 0.25) | 1,740 | 0.4458 | 0.1376 | 83.1% | 0.144 | 0.169 | +0.025 |
| Moderate Underdog (0.25 <= P < 0.40) | 1,511 | 0.6641 | 0.2353 | 62.6% | 0.327 | 0.374 | +0.047 |
| Coin-Flip / Close (0.40 <= P <= 0.60) | 2,448 | 0.6865 | 0.2467 | 56.5% | 0.501 | 0.525 | +0.024 |
| Moderate Favorite (0.60 < P <= 0.75) | 1,752 | 0.6111 | 0.2104 | 69.2% | 0.674 | 0.692 | +0.018 |
| Heavy Favorite (P > 0.75) | 2,031 | 0.3935 | 0.1181 | 85.8% | 0.858 | 0.858 | +0.000 |

### Slice: Series Format (BoN)

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bo1 | 4,586 | 0.5785 | 0.1980 | 69.5% | 0.505 | 0.530 | +0.025 |
| Bo3 | 3,693 | 0.5293 | 0.1767 | 73.5% | 0.514 | 0.530 | +0.016 |
| Bo5 | 1,203 | 0.5999 | 0.2053 | 69.1% | 0.567 | 0.593 | +0.025 |

### Slice: Competition Tier

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| development | 2,041 | 0.5919 | 0.2028 | 69.2% | 0.513 | 0.527 | +0.014 |
| international | 197 | 0.5706 | 0.1952 | 70.1% | 0.563 | 0.574 | +0.011 |
| major | 1,772 | 0.5891 | 0.2000 | 69.6% | 0.512 | 0.528 | +0.016 |
| minor_top_level | 1,080 | 0.5180 | 0.1720 | 73.4% | 0.512 | 0.526 | +0.013 |
| regional | 4,392 | 0.5478 | 0.1855 | 71.8% | 0.518 | 0.548 | +0.030 |

### Slice: Player-Team Disagreement

| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| High Agreement (Delta <= 0.05) | 4,086 | 0.5480 | 0.1846 | 72.3% | 0.523 | 0.541 | +0.017 |
| Moderate Agreement (0.05 < Delta <= 0.12) | 2,983 | 0.5653 | 0.1924 | 70.9% | 0.511 | 0.530 | +0.019 |
| High Disagreement (Delta > 0.12) | 2,413 | 0.5819 | 0.1986 | 68.9% | 0.511 | 0.543 | +0.032 |
| Severe Disagreement (Delta > 0.20) | 952 | 0.5704 | 0.1930 | 70.9% | 0.517 | 0.550 | +0.034 |

## 4. Promotion Gate Evaluation

**Status:** **FAILED (Promotion blocked)**

| Gate Criterion | Status |
| :--- | :---: |
| Calibration slope within [0.85, 1.15] | PASS |
| ECE within threshold (<= baseline ECE) | PASS |
| Statistically superior (95% Bootstrap CI upper bound < 0) | FAIL |
| Underdog quarantine safety (no false favorites on odds > 3.50) | PASS |

**Evaluation Notes & Gate Diagnostics:**
- 95% bootstrap CI [+0.000380, +0.005502] does not prove significant superiority
