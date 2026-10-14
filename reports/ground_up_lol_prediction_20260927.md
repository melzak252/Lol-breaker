# Sports prediction from first principles: research, data audit, and a trained LoL model

**Executed: 2026-09-27. Original run: `ground_up_lol_20260927_run001`. Latest architectural run: `prematch_architecture_20260927_run001`. Status: completed historical experiments; not deployed.**

## Executive result

A new predictor was built and trained from raw LoL history, without inherited model weights or a previous feature bank. The development-selected recipe is an **equal-weight average of logistic regression C=0.1, 15-leaf LightGBM, and a width-32 role-aware neural network**. All three use their **raw** probability variants: fitted temperatures were compared but are not applied by this selected recipe.

On **3,103 identical 2026 series, 2026-01-08–2026-09-08**, its log loss is **0.543517** and Brier score **0.183347**. The development-selected player-rating reference has log loss **0.577810** and Brier **0.197337**. The paired monthly loss difference is **-0.034292**, 95% interval **[-0.049184, -0.022332]**.

Two qualifications are material. First, the best single LightGBM is almost tied at **0.544260**; the ensemble-versus-tree interval crosses zero, so the extra neural/linear components are not independently established as necessary. Second, the primary input supplies the realized first-map lineup as a known-lineup scenario. With only previous released lineups, ensemble loss is **0.565827**.

**The source is not a prospective feed:** 2,553 of these 3,103 final rows (**82.28%**) are cache-recovered. “Fresh source” means fresh scraping in the preserved recovery dataset, not capture at the original forecast time. Neither source branch certifies roster announcements or publication times. Prior project exposure also prevents treating 2026 as a newly independent holdout.

**Subsequent user-requested A0 comparison:** the fresh ensemble does **not** beat frozen annual Causal A0 on these same 3,103 series. A0's LL is **0.536757**, versus **0.543517**; the paired difference is detailed below. This comparison did not retrain or reselect either model.

**Standalone improvement attempts:** neither recency fitting nor roster-summary features beat A0. Verification also found and repaired a numerical cold-start symmetry defect. After rebuilding the feature bank and refitting the frozen recipes, the corrected original ensemble scores **0.543863 LL / 0.183493 Brier**, and its pooled-tree variant **0.544162 / 0.183530**. These are separate versions, not replacements for the original results below. Keep the numerical correction; do not claim improved predictive accuracy or promote either model over A0.

**Subsequent combination diagnostic:** at the user's request, a fixed **75% annual A0 + 25% corrected fresh ensemble** scored **LL 0.535472 / Brier 0.180892**, versus A0's **0.536757 / 0.181573**. This promising historical result is separate from the failed standalone replacements. Monthly uncertainty favors the blend, but source-tournament LL sensitivity includes zero; no weight was fitted or model promoted.

**Latest, requested architectural experiment — not fixed averaging:** expanded-input fusion, separate history/summary encoders, and a learned specialist connector were trained and scored on all **11,550 canonical 2024–2026 matches**. Their calibrated LLs are **0.566403**, **0.562094**, and **0.557196**, respectively; annual A0 remains better at **0.551246**. Separate encoders improve match-weighted loss over input fusion, but the equal-month LL effect reverses sign. None meets the predeclared replacement criterion. The detailed architecture section below records the full population, intervals, artifacts, and a retained initial numerical-replay discrepancy.

### Original study: 2026 common-cohort comparison

| Model | Series | Log loss | Brier | Accuracy | ROC AUC | LL >= 2.5 |
|---|---|---|---|---|---|---|
| Selected equal-weight ensemble | 3103 | 0.543517 | 0.183347 | 72.09% | 0.7978 | 12 |
| LightGBM 15 leaves, raw | 3103 | 0.544260 | 0.183467 | 72.16% | 0.7978 | 8 |
| Role-aware neural width 32, raw | 3103 | 0.550465 | 0.186168 | 71.48% | 0.7925 | 25 |
| Logistic C=0.1, raw | 3103 | 0.562544 | 0.189508 | 71.41% | 0.7854 | 42 |
| Player BT, temperature (reference) | 3103 | 0.577810 | 0.197337 | 68.26% | 0.7637 | 15 |
| Organization BT, temperature | 3103 | 0.605533 | 0.209611 | 66.87% | 0.7307 | 1 |
| Constant 0.5 | 3103 | 0.693147 | 0.250000 | 53.21% | 0.5000 | 0 |

All rows above use the same realized-first-lineup scenario and target IDs. Lower log loss and Brier are better. Accuracy uses the ordinary 0.5 threshold; a low deep-loss count is not itself a good model—the constant forecast has zero such losses and much worse proper scores.

**Uncertainty and interpretation:**

- Calendar-month blocking: **9 blocks**, 5,000 paired resamples; match-weighted ΔLL **-0.034292**, equal-month ΔLL **-0.036038**.
- Source tournament label within year: **143 groups**, interval **[-0.046268, -0.022996]**; equal-group effect **-0.034190**. These are not independently verified tournament editions.
- The selected recipe beats the reference in each of the nine observed months. This describes this historical sample, not nine independent prospective trials.
- Post-selection diagnostic against LightGBM: ΔLL **-0.000742**, monthly interval **[-0.005571, 0.002471]**. It establishes neither superiority nor equivalence. The development-selected recipe was **not changed** after this calculation.

### Roster availability materially changes the result

The same selected model on prior lineups has LL **0.565827**, Brier **0.192009**, and accuracy **70.35%**. A player-rating control receiving those **same prior lineups** has LL **0.588796**. Their paired monthly ΔLL is **-0.022969**, interval **[-0.035900, -0.013619]**. This is the fair same-information comparison for that secondary scenario, rather than comparing it only with a reference that receives actual lineups.

Changing the ensemble from realized to prior lineups costs **0.022310** LL, interval **[0.009751, 0.037633]**. Both prior lineups exist for **92.94%** of final targets; both five-player memberships match the realized first-map memberships for **70.93%**. Membership overlap does not require unchanged role order. Supplying a genuinely announced lineup is therefore a substantive input improvement, not an incidental interface field.

### Calibration and remaining difficult populations

For the selected forecast, ECE10 is **2.60 percentage points**, diagnostic calibration slope **0.9717**, and intercept **+0.1196** in the source A/B orientation. Mean predicted A-win probability is **51.03%**, versus **53.21%** observed. These diagnostics are not corrections fitted back into the scored forecasts.

A separate favored-team confidence view contains one observation per series:

| Forecast confidence | Series | Mean confidence | Observed correct |
|---|---|---|---|
| [0.5, 0.6) | 721 | 55.08% | 54.23% |
| [0.6, 0.7) | 699 | 64.86% | 65.24% |
| [0.7, 0.8) | 646 | 74.96% | 75.23% |
| [0.8, 0.9) | 633 | 84.97% | 83.73% |
| [0.9, 1.0) | 404 | 93.55% | 92.57% |

These bin averages are descriptive; they are not independent confidence guarantees or a calibration test for an odds-selected betting subset.

| Profile | Series | Log loss | Brier | ECE (pp) |
|---|---|---|---|---|
| best_of: 1 | 745 | 0.584199 | 0.199212 | 3.50 |
| best_of: 3 | 1873 | 0.516413 | 0.173090 | 2.30 |
| best_of: 5 | 485 | 0.585698 | 0.198585 | 6.19 |
| history_support: all players >50 maps | 1228 | 0.593849 | 0.202619 | 4.16 |
| history_support: includes debutant | 334 | 0.466890 | 0.154017 | 5.84 |
| history_support: minimum 1-10 maps | 743 | 0.495056 | 0.164468 | 3.85 |
| history_support: minimum 11-50 maps | 798 | 0.543258 | 0.183542 | 5.23 |

The artifact label `includes debutant` means **zero previous maps in this source for at least one player**, not a verified career debut. The lower loss in sparse-history rows does not prove cold starts are easier: competition, format, and opponent mismatch differ across these observational slices. BO5's ECE **6.19 pp** on 485 series and weaker loss deserve scrutiny; no BO-specific correction was fitted on these outcomes.

### Source-origin sensitivity

| Source tag | Series | Dates | Ensemble LL | Reference LL |
|---|---|---|---|---|
| fresh_source | 550 | 2026-07-14–2026-09-08 | 0.545730 | 0.582730 |
| validated_database_cache | 2553 | 2026-01-08–2026-08-24 | 0.543040 | 0.576750 |

Both source-origin slices favor the ensemble over the reference in point loss, but their dates and opponents differ. The fresh subset covers only July–September and is not an independent replication. Full source-lineage membership is preserved in `source_lineage.parquet`; the complete final cohort remains the headline population.

![Historical final calibration and monthly paired effects](../data/08_reporting/ground_up_lol_20260927_run001/final_diagnostics.png)

## Requested comparison with frozen annual Causal A0

After completion of the from-scratch experiment, the user requested a comparison with A0. The reference is **annual Causal A0 `p_full` from the canonical manifest**, not monthly A0, A1, or the deployed EXP081 hybrid. Its input hash matches `conf/base/research_benchmark.json`; the new prediction hash matches the completed run receipt.

The join covers **3,103/3,103 series, 2026-01-08–2026-09-08**, with no missing predictions or duplicate IDs. Dates, effective release days, BO, outcomes, and both first-map player memberships agree. **1,579 A0 forecasts required side reversal** to match the new source orientation; 1,524 already matched. Each reversal applies `1 − p`, not a second reversal of labels or prices.

| Metric | Annual Causal A0 | New selected ensemble |
|---|---:|---:|
| Log loss ↓ | **0.536757** | 0.543517 |
| Brier ↓ | **0.181573** | 0.183347 |
| ROC AUC ↑ | **0.8016** | 0.7978 |
| Accuracy ↑ | 71.90% | **72.09%** |
| ECE10, descriptive pp ↓ | 2.86 | **2.60** |
| Losses with LL ≥ 2.5 | **10** | 12 |

The new model's slightly higher accuracy means six more correct threshold decisions. Among 288 non-tied disagreements, it is correct 147 times and A0 141 times. That does not outweigh its worse probability scores; accuracy discards confidence. ECE is also a descriptive binned statistic, not a standalone selection criterion.

**Paired effects are new minus A0; positive loss differences favor A0.** All calculations use the same 1e-6 probability clipping, 5,000 resamples, and seed 20260927:

- **Log loss, 9 calendar months:** Δ **+0.006761**, 95% interval **[+0.002726, +0.010184]**; equal-month effect **+0.003938**.
- **Log loss, 143 source-tournament labels within year:** interval **[+0.001428, +0.012230]**; equal-group effect **+0.007715**. These labels are not verified independent editions.
- **Brier, calendar months:** Δ **+0.001774**, interval **[+0.000022, +0.003228]**. Source-tournament sensitivity **[−0.000516, +0.004025]** crosses zero, so Brier uncertainty is less robust than the log-loss result.

| Format | Series | A0 LL | New LL | New − A0 |
|---|---:|---:|---:|---:|
| BO1 | 745 | 0.582456 | 0.584199 | +0.001742 |
| BO3 | 1,873 | 0.510016 | 0.516413 | +0.006397 |
| BO5 | 485 | 0.569825 | 0.585698 | +0.015873 |

A0 has lower point log loss in all three formats, all three observed quarters (Q3 is partial), and seven of nine months. The largest per-series format gap is BO5. These are descriptive profiles, not separately adjusted hypothesis tests.

**Decision:** the ground-up recipe improved on its new rating controls but did not improve on A0 in this matched historical comparison. Keep that negative result; do not replace A0 or blend/reselect models after observing it.

This is not a controlled architecture ablation. The methods have different history representations, fitting/calibration schedules, and feature contracts. The new model receives first-map role slots; the canonical A0 roster policy derives roles from prior games. Matching player membership does not make those role inputs identical. Both remain retrospective known-lineup scenarios, and prior exposure to 2026 prevents prospective claims. Do not compare the new **2026-only** LL with A0's **11,550-row 2024–2026** LL of 0.551246.

Evidence: [`a0_comparison/comparison.json`](../data/08_reporting/ground_up_lol_20260927_run001/a0_comparison/comparison.json), [`paired_predictions.parquet`](../data/08_reporting/ground_up_lol_20260927_run001/a0_comparison/paired_predictions.parquet), [`profiles.csv`](../data/08_reporting/ground_up_lol_20260927_run001/a0_comparison/profiles.csv), and [`verification.json`](../data/08_reporting/ground_up_lol_20260927_run001/a0_comparison/verification.json). An independent ID lookup and NumPy rescore reproduced LL, Brier, accuracy, and tail counts; a separate block-multiplicity calculation reproduced all four intervals within 1e-12. No fitting, deployment, or permanent test-suite execution was needed for this stored-forecast comparison.

## Production bookmaker comparison and ECE interpretation

This subsequent user-requested analysis uses the **saved production-odds snapshot captured on 2026-09-26**, not a new server query. “Production” describes the odds source; the two research models were not necessarily available when those prices were recorded. No models were trained, recalibrated, or selected again.

Each two-way bookmaker quote is normalized as `(1/odds_A) / (1/odds_A + 1/odds_B)`, aligned once to GOL.GG team1, and averaged with **one probability per available bookmaker**, not one vote per tick. The raw quote probabilities and consensus were independently reconstructed. All selected quotes satisfy their stored cutoff/start-time rules. OPEN means the first **observed** quote instant, not a certified original opening price; **506/514** matched OPEN fixtures contain only one bookmaker.

### Same 275 fixtures across every horizon

The primary common population runs from **2026-05-31 through 2026-09-08**. The model rows below are the same retrospective match-day predictions at every horizon, not four separately generated quote-time forecasts.

| Forecast | N | Log loss ↓ | Brier ↓ | Accuracy | ECE10, pp |
|---|---:|---:|---:|---:|---:|
| New ensemble | 275 | 0.561300 | 0.191114 | 71.64% | 5.69 |
| Annual A0 | 275 | 0.545250 | 0.185929 | 69.09% | 4.05 |
| Bookmaker consensus OPEN | 275 | 0.570488 | 0.195259 | 68.36% | 6.17 |
| Bookmaker consensus 48h | 275 | 0.554200 | 0.188167 | 70.18% | 4.78 |
| Bookmaker consensus 24h | 275 | 0.550888 | 0.186633 | 70.55% | 4.78 |
| Bookmaker consensus CLOSE | 275 | 0.548262 | 0.186217 | 69.82% | 5.07 |

On this population the new model has lower point loss than observed OPEN but higher loss than the later market horizons. A0 has lower point loss than the consensus in each horizon. **Every monthly-block model-versus-market loss interval includes zero**: there are only five months. For example, new-minus-CLOSE is **+0.013038 [−0.012428, +0.027998]**; A0-minus-CLOSE is **−0.003012 [−0.025915, +0.022609]**. These results do not establish a market advantage.

### Full matched coverage at each horizon

Within a row, all three forecasts use identical fixtures. Across rows, the populations differ; do not interpret changes in the models' scores as changing forecasts.

| Horizon | N | Market LL | A0 LL | New LL | Market ECE10, pp |
|---|---:|---:|---:|---:|---:|
| OPEN | 514 | 0.571850 | 0.562546 | 0.571960 | 3.82 |
| 48h | 343 | 0.552789 | 0.544745 | 0.559177 | 5.22 |
| 24h | 379 | 0.542886 | 0.536376 | 0.549545 | 5.34 |
| CLOSE | 409 | 0.566611 | 0.573952 | 0.582119 | 2.57 |

Only **57 OPEN, 1 at 48h, 2 at 24h, and 409 CLOSE** satisfy both models' conservative prior-day history/origin conditions. This still does not certify roster announcements, source publication times, or historical forecast existence. The one- and two-match slices cannot establish calibration or model superiority.

### Individual bookmakers at CLOSE

**This is not a bookmaker ranking:** each bookmaker has different coverage. The adjacent model losses use exactly that bookmaker's rows; the complete score tables also retain OPEN, 48h, and 24h.

| Bookmaker | N | Book LL | Matched A0 LL | Matched new LL | Book ECE10, pp |
|---|---:|---:|---:|---:|---:|
| Betclic | 186 | 0.613943 | 0.630310 | 0.630270 | 7.85 |
| Betfan | 220 | 0.550451 | 0.565276 | 0.572301 | 3.63 |
| eFortuna | 171 | 0.597297 | 0.609663 | 0.620528 | 6.02 |
| Lebull | 271 | 0.580064 | 0.589291 | 0.596177 | 5.60 |
| STS | 300 | 0.596952 | 0.608247 | 0.613677 | 5.19 |
| Superbet | 79 | 0.694783 | 0.697438 | 0.684826 | 13.10 |
| Totalbet | 325 | 0.570383 | 0.579303 | 0.577139 | 4.03 |

### Is ECE10 around 2.5 a default?

**No. The default is ten bins, not a score of 2.5.** `src.analysis.metrics.calculate_ece` defaults to ten equal-width bins, right-closed `(lower, upper]`. For bin count `n_b`, mean probability `p_b`, and observed win fraction `y_b`:

`ECE10 = sum_b (n_b / N) * abs(p_b - y_b)`.

A returned fraction of **0.025 means 2.5 percentage points**. It is neither a universal acceptance threshold nor a per-match probability interval. The infinite-population ideal is zero, but finite samples usually produce positive measured ECE even for perfectly calibrated probabilities. Bin counts, probability distribution, sample size, and dependence all matter; binning can also hide within-bin miscalibration.

To quantify the sampling effect, 20,000 simulations per scenario held the probabilities fixed and generated independent `Y_i ~ Bernoulli(p_i)`. The following are **conditional sampling references**, not confidence intervals for true calibration error:

| Forecast / cohort | N | Observed ECE10, pp | Mean under perfect calibration, pp | Central 95% sampling range, pp |
|---|---:|---:|---:|---:|
| New ensemble, full 2026 | 3,103 | 2.60 | 1.85 | 1.03–2.82 |
| A0, full 2026 | 3,103 | 2.86 | 1.84 | 1.03–2.81 |
| New ensemble, common production cohort | 275 | 5.69 | 6.16 | 3.46–9.44 |
| CLOSE consensus, full matched coverage | 409 | 2.57 | 5.07 | 2.70–7.86 |

Thus the measured **2.57 pp** for the full CLOSE market is an observed result, not a hardcoded/default ECE. The earlier roughly **2.5–7.0 pp** reference reported for a 514-match analysis was specific to that probability distribution and sample size, not a universal range. Samples can fall below or above a central reference interval.

The new model's full-cohort 2.60 pp lies within this independent-Bernoulli reference; A0's 2.86 pp is just above its upper endpoint. Neither is a dependence-robust calibration verdict: repeated teams, shared events, historical model development, and bin choice are not represented by the simulation. **Do not subtract the simulated mean from observed ECE** and call the difference “true ECE.” The full-2026 examples preserve the earlier comparison's orientation; production forecasts are all aligned to GOL.GG team1.

One numerical convention also matters: the older production report used **left-closed** bins. Its common-275 48h market ECE was **4.46 pp**, whereas the shared default's right-closed bins give **4.78 pp** because boundary-valued forecasts move bins. Both values are retained in the new evidence; the forecasts did not change.

### Evidence and verification

Outputs are under [`production_bookmakers/`](../data/08_reporting/ground_up_lol_20260927_run001/production_bookmakers/): `protocol.json`, `cohort_metrics.csv`, `bookmaker_metrics.csv`, `comparison_panel.parquet`, `bookmaker_quotes.parquet`, `paired_log_loss_intervals.csv`, `ece_sampling_reference.csv`, `ece_exact_moment_checks.csv`, and `verification.json`.

Independent calculations verified **117 metric rows / 819 numeric checks** (maximum difference **1.53e-16**) and **72 paired intervals**. All **21** ECE simulations were checked against exact independent Poisson-binomial bin expectations; the largest mean discrepancy was **1.78 Monte Carlo standard errors**. No new permanent tests, model changes, live queries, bets, or deployment were performed.

## Follow-up experiment: reclaiming unused calibration labels

The user subsequently requested continued model improvement toward beating A0. The first controlled hypothesis was to remove the six-month fitting gap for **raw-only** models: the original selected recipe does not apply its fitted temperatures. At the three 2026 origins, full-history fitting adds **1,328 / 1,185 / 2,287** available targets without admitting labels released on or after the origin.

Two variants were frozen before fitting: training through the origin with **730-day** or **365-day** half-life weights. Original features, roster assumptions, logistic C=0.1, 15-leaf tree, width-32 neural architecture, and seed remained fixed. All historical training rows retain positive weights. The weight reference moves with the TRAIN cutoff, as in the original implementation; this is not an ablation holding the absolute weights of older rows fixed.

On the same **7,119 development matches**, the equal-family ensemble remained selected:

| Fitting policy | Development LL | Development Brier |
|---|---:|---:|
| Original six-month gap, 730-day weights | 0.566147 | 0.192859 |
| Full history, 730-day weights — frozen primary | **0.564725** | **0.192385** |
| Full history, 365-day weights — diagnostic | 0.565732 | 0.192920 |

Selection was recorded **before either variant's 2026 fitting**. Both fixed variants were then assessed; the diagnostic variant could not replace the primary after seeing those outcomes. The original experiment's 2026 results were already known.

| Model / same 3,103 matches | 2026 LL | Brier | ECE10, pp | LL ≥ 2.5 |
|---|---:|---:|---:|---:|
| Annual A0 | **0.536757** | **0.181573** | 2.86 | 10 |
| Original ground-up ensemble | 0.543517 | 0.183347 | 2.60 | 12 |
| Full history, 730-day weights | 0.546303 | 0.184445 | 2.35 | 24 |
| Full history, 365-day weights | 0.550097 | 0.186139 | 2.42 | 24 |

**Negative result; neither variant is adopted.** Primary-minus-A0 ΔLL is **+0.009546**, monthly 95% interval **[+0.004705, +0.012686]**. Primary-minus-original is **+0.002786 [−0.001838, +0.006912]**: worse point scores, without a conclusive monthly interval against the original. One-year weighting is worse still. The reduction in ECE does not compensate for worse log loss/Brier.

Verification covered all **60 final raw/prior member replays**, side symmetry (maximum error **2.22e-16**), pure no-organization/cold-start queries, all 12 development partitions, independent scoring and **16 paired metric intervals**. The original saved CLI prediction remained **0.4612781332940849**. Full results and rejection decision: [`prematch_recency_20260927_run001/`](../data/08_reporting/prematch_recency_20260927_run001/).

The shared `run_model_benchmark.py` CLI also scored the complete 3,103-row joined table. Its legacy report uses bootstrap seed 42 and labels a resample fraction “p-value”; that label is not endorsed here. The primary intervals above use the frozen seed **20260927**, 5,000 resamples, and the shared ratio-bootstrap implementation, checked independently. Neither the legacy calibration thresholds nor its automatic gate constitute production evidence.

The runner now supports explicit `calibration_months: 0` only with `candidate_variants: ["raw"]`; no identity calibrator is fabricated. `feature_source_protocol` reuses a hash-verified feature bank without copying it. Original artifacts remain unchanged, and exact pre-/post-extension sources are preserved in the new experiment's model-input directory.

## Follow-up experiment: explicit roster summaries

The second frozen hypothesis added **28 means and 28 population standard deviations across the five players** to the constrained tree. Original role-specific fields remained; missing observations were excluded rather than filled with zero, and fully missing summaries remained missing. This is dispersion across current players' existing summaries, **not** a recovery of historical map-level dispersion or order.

This comparison retained the original six-month TRAIN/CAL split, 730-day weights, architectures and seed. Development loss improved from **0.566147 to 0.565418** on the same 7,119 matches. The selected recipe remained the raw equal-family ensemble, with the pooled tree replacing its original tree.

On all 3,103 historical 2026 matches, the ensemble instead scored **LL 0.544085, Brier 0.183505**, versus **0.543517 / 0.183347** originally. Pooled-tree-only LL was **0.545474**; the prior-lineup ensemble scored **0.566376**. ECE fell to **2.34 pp** and accuracy rose to **72.22%**, but these did not compensate for worse proper scores.

- Pooled-minus-original ΔLL: **+0.000568**, monthly 95% interval **[−0.000555, +0.001582]**.
- Pooled-minus-A0 ΔLL: **+0.007329**, monthly interval **[+0.003181, +0.010464]**; equal-month effect **+0.004257**.
- Source-tournament sensitivity: **143 groups**, pooled-minus-A0 interval **[+0.001898, +0.012852]**. Labels are not independently verified tournament editions.
- All intervals use 5,000 paired resamples and seed 20260927; **16** LL/Brier intervals were independently reproduced.

Unchanged neural refits reproduced the original development predictions exactly. Logistic refits did not: maximum individual difference **0.008701**, although development LL changed by only about **0.000008**. The precise cause was not established. A fixed-other-members diagnostic was declared before viewing the new final results: original logistic/neural probabilities plus the new tree, at unchanged equal weights. Its 2026 LL **0.544084** leads to the same rejection; no recipe was reselected.

**Not adopted, with an additional correctness failure discovered during verification.** All 90 stored member/CAL/prior forecast replays passed, and all three selected-member refits reproduced probabilities and temperatures exactly. However, an independently generated cold-start query did not complement its physically swapped query, despite passing algebraic swaps of already-generated feature rows. The failure is preserved in [`prematch_pooling_20260927_run001/artifact_verification.json`](../data/08_reporting/prematch_pooling_20260927_run001/artifact_verification.json), not relabeled as a successful verification.

### Numerical repair required before further comparisons

The original uncentered Gaussian quadrature produced a nominally neutral organization-control probability of **0.5000000000000001**. The tree could split on that one-ULP sign. Two identical unseen rosters consequently received **0.5004417379785517 in both orientations**: a swap residual of **0.0008834759571035**. This is a feature-generation numerical defect, not evidence of a roster advantage.

The repair integrates the **odd, centered series margin**, expressed as a polynomial in `tanh(log_odds / 2)`. Symmetric quadrature nodes then cancel exactly at equal strength; no team-name exception or probability override is used. Independent integration also exposed insufficient 24-node accuracy for high-variance BO5 inputs. Across **105 mathematical cases**, maximum error fell from **0.00182708** with 24 nodes to **3.50174e-8** with **128 nodes**. Node count was selected against numerical integrals, not match outcomes; this finite grid is not a universal error-bound proof.

The previously failing saved-model query now returns **0.5 in both orientations** without mutating state. Regression tests cover exact-neutral BO1/3/5 controls and independent BO3/BO5 integration at ordinary and long-inactive variances. **93 focused tests passed**, with one pre-existing Starlette/httpx deprecation warning.

Earlier banks, weights, predictions, recipes and metrics remain preserved as **pre-repair evidence**. Their stored scores are not silently recalculated under new source code. The corrected feature version and frozen-recipe reassessment are recorded under `prematch_symmetry_20260927_run001`; no additional architecture or weight search is authorized by this repair.

### Corrected feature version: frozen recipes refitted

`prematch_symmetry_20260927_run001` rebuilt the bank from the same raw source and release policy. All **40,636 metadata rows**, player arrays, team arrays and BO context are exactly unchanged. Only the two current/prior rating-control probability arrays change; maximum absolute difference on the actual bank is **9.65320e-7**. Tiny input changes can still alter optimized/tree models, so neither forecasts nor weights were reused.

Both original development-selected recipes were **explicitly inherited without reselection**. There was no new development search on the corrected bank. Six-month CAL windows, 730-day fitting weights, seed, model configurations and equal raw ensemble weights remain fixed. This executed **18 supervised fits** across the two variants and three final origins, plus **six repeated selected-member fits** for reproducibility.

| Same 3,103 series | Log loss | Brier | Accuracy | ECE10, pp | LL ≥ 2.5 |
|---|---:|---:|---:|---:|---:|
| Annual Causal A0 | **0.536757** | **0.181573** | 71.90% | 2.86 | 10 |
| Original ensemble, pre-repair historical result | 0.543517 | 0.183347 | 72.09% | 2.60 | 12 |
| Corrected original recipe — fixed primary | **0.543863** | **0.183493** | 72.16% | 2.48 | 13 |
| Corrected pooled-tree recipe — fixed comparison | 0.544162 | 0.183530 | 72.25% | 2.39 | 15 |

The corrected primary's ΔLL against A0 is **+0.007107**, monthly 95% interval **[+0.002956, +0.010601]**, equal-month effect **+0.004385**. Source-tournament sensitivity gives **[+0.001632, +0.012695]**, equal-group effect **+0.009006**. Blocking remains **9 months / 143 source labels**, with 5,000 draws and seed 20260927.

Corrected pooling minus corrected control is **+0.000299**, monthly interval **[−0.001128, +0.001492]**, equal-month effect **−0.000164**: no demonstrated gain or equivalence. The numerical repair itself is not a predictive improvement claim: corrected-control minus original is **+0.000346 [−0.000438, +0.001078]**. Prior-lineup LL is **0.566277** for corrected control and **0.566627** for corrected pooling.

**Decision:** retain the numerical correction and use the corrected control as the fresh-model research baseline. Reject recency fitting and roster aggregation as established improvements. **Annual A0 remains the research reference.** No canonical manifest, production configuration, odds or market forecasts were changed; 2026 remains repeatedly inspected historical development.

Verification reproduced **180 member/CAL/prior checkpoint forecasts exactly**, **130 scalar score checks**, and **20 independent paired intervals**. All **six selected-member refits** reproduced raw/calibrated forecasts and temperatures with zero observed difference. Fifty paired hypothetical queries recomputed features separately in each orientation, covering ordinary rosters, no organization, all BO1/3/5 cold starts and sampled lineups. All passed, with unchanged state; synthetic feature-swap error was at most **4.45e-16**. Actual cold-start CLI calls returned **0.5** for both artifacts; the corrected-control known-lineup example returned **0.46959853994795325**.

Current corrected control:

```bash
.venv/bin/python scripts/train_prematch_forecaster.py predict \
  --artifact data/06_models/prematch_symmetry_20260927_run001_control/selected_forecaster.joblib \
  --match-json data/08_reporting/ground_up_lol_20260927_run001/inference_query.json
```

Evidence: [`prematch_symmetry_20260927_run001/`](../data/08_reporting/prematch_symmetry_20260927_run001/) contains the feature-impact audit, matched predictions, metrics, profiles, intervals, decision and verification receipts. Per-variant `_control/` and `_pooling/` report directories contain fit receipts and full checkpoint/query replay checks. Protocols and the inherited-recipe provenance are under `data/05_model_input/prematch_symmetry_20260927_run001/`; code snapshots and prior evidence are preserved. The shared benchmark CLI also scored the corrected primary, subject to the legacy-label/gate caveat above.

## Combining A0 with the fresh model: fixed historical diagnostic

The approaches retain different representations. A0 uses individual recent-map histories through a player-history attention encoder, alongside ratings, experience, coplay and optional organization/W20 context. The fresh model primarily compresses histories into slow/fast Gaussian-BT ratings and rolling performance summaries. Its selected predictor averages a linear model, LightGBM and a role-aware MLP; the MLP preserves five role slots but does not receive the last-16-map tokens. Both use past sports information, not odds. Their fitting schedules and role-input conventions also differ, so this is not an isolated architecture ablation.

Recency fitting and roster pooling are modifications of that same fresh pipeline, not wholly independent modeling families. Probability-level combination with A0 is a separate question. A weaker standalone model can contribute when its probability errors differ from the stronger model's.

Before calculating the new blend scores, four diagnostic recipes were recorded: A0 weights **0.75 and 0.50**, combined with either the corrected fresh ensemble or its corrected single tree. **75/25 A0/fresh was the declared primary**; no continuous weight optimization, retraining, extra calibration, BO projection or post-result recipe change was performed.

| Forecast, same 3,103 matches | Log loss | Brier |
|---|---:|---:|
| Annual A0 | 0.536757 | 0.181573 |
| Corrected fresh ensemble | 0.543863 | 0.183493 |
| Corrected fresh tree | 0.544773 | 0.183690 |
| **75% A0 + 25% fresh ensemble — fixed primary** | **0.535472** | **0.180892** |
| 50% A0 + 50% fresh ensemble | 0.536246 | 0.180985 |
| 75% A0 + 25% fresh tree | 0.536074 | 0.181006 |
| 50% A0 + 50% fresh tree | 0.537240 | 0.181170 |

For the primary, blend-minus-A0 **ΔLL = −0.001285**. With 5,000 paired draws and seed 20260927:

- **9 calendar months:** 95% interval **[−0.002336, −0.000349]**; equal-month effect **−0.001965**.
- **143 source-tournament labels:** interval **[−0.002655, +0.000061]**; equal-group effect **−0.001108**. The interval includes zero; these labels are not verified independent editions.
- **ΔBrier = −0.000681**: monthly interval **[−0.001123, −0.000356]**, source-group interval **[−0.001278, −0.000105]**.

A0 and the fresh ensemble differ by **5.84 probability percentage points on average**, with RMS difference **7.87 pp**. Their threshold decisions disagree on **288** matches; the fresh model corrects **148** A0 errors but introduces **140** errors where A0 is correct. More importantly for probability quality, the independently checked Brier-mixture identity attributes **0.001161** reduction versus the weighted component Briers to forecast disagreement. This exceeds the weighted cost of including the weaker component on this sample. It does not establish the cause of their differences or a future advantage.

**Interpretation:** a simple probability blend is worth a controlled follow-up, not automatic promotion. The 50/50 tree blend worsens LL, so averaging does not guarantee improvement. All these 2026 outcomes were already exposed during development; the four recipes were fixed before this calculation, not prospectively preregistered before 2026. A defensible next experiment freezes the weight or selects it using earlier out-of-time forecasts, then seeks new prospective confirmation. This diagnostic did not build a deployed combined predictor or change the canonical reference.

Evidence: [`blend_diagnostic_01/`](../data/08_reporting/prematch_symmetry_20260927_run001/blend_diagnostic_01/) contains the pre-calculation protocol, matched predictions, all recipe metrics, 16 paired intervals, monthly effects, complementarity decomposition, decision and verification. Independent checks covered **35 scalar metrics**, all **16 intervals**, the Brier identity, complete ID/orientation alignment, and algebraic mixture symmetry.

## Requested architectural integration: histories, representations, and specialists

The user clarified that the desired combination was **at the input/architecture level, or through a learned connector**, not another fixed probability average. The separate frozen run is `prematch_architecture_20260927_run001`. A0 supplies its **raw historical-token bank**, not pretrained encoder weights or a probability input. Its stored annual forecasts are the external scoring reference.

### Same historical inputs, explicit player alignment

Both fusion networks receive the corrected fresh summaries above and up to **16 historical maps per player, with 33 token features and an explicit mask**. The banks contain the same **40,636 target matches and player memberships**, but their storage order differs: preparation resolved **19,991 side reversals** and **78,247 changed team-lineup orders** by match/team/player identity. It did not concatenate arrays by position or reduce the population to convenient rows.

The aligned histories follow the fresh model's five role slots. Observed tokens require recorded source bounds strictly before the target day. Those bounds are **global replay watermarks**, not evidence that every player has prior maps: all **32** rows with empty token histories but earlier global state were retained. Missing token payloads stay masked. Debutants' undefined inactivity is excluded from the maximum observed inactivity; an entirely unseen lineup receives zero for that maximum and zero minimum support, preserving the cold-start signal.

### Three fixed connecting designs

| Design | Learned connection |
|---|---|
| Expanded-input fusion | Masked attention summarizes each player's tokens; that vector joins normalized raw player summaries before the shared player MLP. |
| Separate representation encoders — primary | Summaries and token histories pass through separate encoders, then a small per-player fusion MLP. |
| Specialist connector | Regularized logistic regression connects calibrated representation-fusion, 15-leaf LightGBM, and fresh Gaussian-BT player probabilities, with context-dependent coefficients. |

Both neural designs preserve all five role slots and share the ordered-pair scoring construction, `logit(A,B) = (g(A,B) − g(B,A))/2`. Their direct target is the whole-series result; there is no second BO projection. A summary-only role MLP, the tree, and the player control are also retained.

The specialist has **15 coefficients and no intercept**: three expert logits plus their interactions with four side-invariant context fields—minimum log support, maximum observed log inactivity, mean slow-rating SD, and expert-probability SD. Context normalization is fitted only on the connector's earlier training rows. Its coefficients need not be positive or sum to one; this is learned logistic stacking, not a convex probability average.

**This is a whole-architecture comparison, not a parameter-matched fusion-placement ablation.** Input mode has **18,529 parameters**, representation mode **21,697**; the latter adds layers and a dropout path. Input mode expands the fresh role MLP; it is not an exact retraining of the original A0 network. The pre-training [review resolution](../data/05_model_input/prematch_architecture_20260927_run001/review_gate.json) records this scope before outcome scoring.

### Frozen chronology and assessment

The [protocol](../data/05_model_input/prematch_architecture_20260927_run001/protocol.json) and frozen implementation fix width **32**, **24 epochs**, dropout **0.1**, AdamW learning rate **0.001**, weight decay **0.001**, minibatches **512**, and seed **20260927**. Training used **two CPU threads**. TRAIN closes six months before each quarterly origin; CAL uses the following six months with labels released strictly before the origin. Training weights have a 730-day half-life. Raw and temperature outputs are retained; temperature bounds are **0.25–4**.

Four 2023 origins provide **4,836 genuine out-of-time expert forecasts** for connector warmup. Eleven 2024–2026 origins define the complete **11,550-match canonical assessment**. At each origin the connector fits only earlier saved forecasts with released labels; it never trains on in-sample predictions from its current experts. `history_representation__temperature` is the fixed primary candidate, and `history_input__temperature` its fixed architecture comparator. There is **no architecture, hyperparameter, calibration-variant, or ensemble-weight selection on these results**.

Headline loss, Brier, profiles and paired differences share the **1e-6** probability policy. Uncertainty uses **5,000 paired monthly-block resamples**, seed **20260927**, with match-weighted and equal-block effects. Source-tournament labels within year are a sensitivity analysis, not certified independent editions. Raw variants are predeclared sensitivities, not replacements selected after scoring.

All these years were previously exposed in the project. Origins describe simulated availability, not actual model creation dates. The same realized first-map roster scenario and daily release limitations remain. A0's fitting schedule and role inputs differ, so candidate-versus-A0 loss is a matched forecast comparison, not an isolated architecture effect. Single-seed bootstrap intervals do not measure training-seed variation. This run does not certify live forecasts, markets, or tournament distributions.

### Executed results: A0 remains the reference

All **11,550 matches, 2024-01-01–2026-09-08**, have forecasts from every row below. The run completed **67 base members: 52 supervised fits and 15 player-rating calibrations**, plus **11 learned connector fits**. Two additional neural refits were verification only, not selection or replacement.

| Model | Log loss | Brier | LL ≥ 2.5 |
|---|---:|---:|---:|
| Annual Causal A0 | **0.551246** | **0.187103** | 33 |
| Expanded input, raw | 0.567528 | 0.193695 | 68 |
| Expanded input, temperature | 0.566403 | 0.193276 | 41 |
| Separate encoders, raw | 0.562793 | 0.191657 | 45 |
| Separate encoders, temperature — primary | 0.562094 | 0.191364 | 34 |
| Summary-only neural, raw | 0.568386 | 0.193864 | 61 |
| Summary-only neural, temperature | 0.567203 | 0.193451 | 37 |
| 15-leaf LightGBM, raw | 0.560558 | 0.190392 | 15 |
| 15-leaf LightGBM, temperature | 0.560462 | 0.190484 | 32 |
| Gaussian-BT players, raw | 0.587783 | 0.201409 | 59 |
| Gaussian-BT players, temperature | 0.587247 | 0.201390 | 40 |
| Learned specialist connector | **0.557196** | **0.189308** | 44 |

The primary representation-minus-input difference is **ΔLL −0.004310**, monthly 95% interval **[−0.007553, −0.000943]**; ΔBrier **−0.001913**, interval **[−0.003242, −0.000563]**. However, the **equal-month ΔLL is +0.001568**: the favorable match-weighted result does not establish an improvement for a typical equally weighted month. Source-label sensitivity gives LL **[−0.007359, −0.001237]**, equal-label effect **−0.005757**. This compares the two complete designs, including their unequal capacity.

The following differences are **candidate minus A0**; positive is worse. There are **33 calendar-month blocks** and **444 source-tournament labels within year**.

| Calibrated design | ΔLL | Monthly 95% interval | Equal-month ΔLL | Source-label 95% interval |
|---|---:|---|---:|---|
| Expanded input | +0.015158 | [+0.011966, +0.017816] | +0.001983 | [+0.011064, +0.019375] |
| Separate encoders — primary | +0.010848 | [+0.007787, +0.013804] | +0.003551 | [+0.007266, +0.014465] |
| Learned specialist connector | +0.005950 | [+0.003818, +0.007633] | +0.000297 | [+0.003252, +0.008680] |

The connector also has **ΔBrier +0.002205**, monthly interval **[+0.001164, +0.003048]**. It has the best point scores among the new designs, but that does not establish superiority over its individual experts across seeds or future samples. In this historical run A0 has better LL and Brier in each year, each BO1/3/5 slice, and all four history-support strata.

For continuity with earlier experiments, the **3,103-match 2026 subset** is reported separately; it is not compared against the 11,550-row headline:

| Model, same 2026 rows | Log loss | Brier | LL ≥ 2.5 |
|---|---:|---:|---:|
| Annual A0 | **0.536757** | **0.181573** | 10 |
| Expanded input, temperature | 0.549320 | 0.186277 | 21 |
| Separate encoders, temperature | 0.548777 | 0.185788 | 16 |
| Learned specialist connector | 0.541236 | 0.182542 | 21 |

**Decision: do not replace A0 with any of these designs.** Preserve the trained architectures, the learned connector and the negative result. Do not select a different calibration variant after observing this table, conflate this trained experiment with the earlier fixed blend, or claim that a new architecture creates a new independent test.

### Verification, artifacts, and executable interface

- **63 focused tests passed**, covering benchmark contracts, forecasters, fusion, alignment, and stacking. The strengthened asymmetric-history-mask fixture subsequently passed with all **11** alignment tests. No production database or application-wide suite was used.
- Independent calculations agreed on **60 scalar metrics** and **48 paired intervals**. Three canonical suite runs retained **11,550/11,550** candidate coverage and reproduced headline LL/Brier to **1e-12**, all with status `DEVELOPMENT_ONLY`.
- The completed two-thread verification reproduced **121 checkpoint/variant forecasts exactly** across all 11 origins and all evaluation rows. Changing masked payloads to NaNs left forecasts unchanged; maximum observed side-swap error was **4.44e-16**.
- At the fixed **2026-01-01** origin, both fusion networks were retrained with the same seed on **36,205 TRAIN / 1,328 CAL rows**. Both reproduced saved temperatures and raw/calibrated forecasts on **1,034 target rows** exactly. This is same-data determinism, not independent-seed evidence.
- The actual `predict-batch` CLI exercised **three real empty-history rows and three supported-history rows**, all 11 output columns, and reversed sides. It matched saved forecasts within **1e-6** and had swap error **2.22e-16**.
- **Retained caveat:** an initial exact replay check differed on 255/2,109 first-origin input-fusion forecasts, maximum **4.88155e-6**. Repeated fresh-process checks, the numerical-audit prelude, and the later full replay did not reproduce it. Four/eight-thread probes showed only much smaller summary-control drift, not that failure. The cause remains unestablished; no source, artifact, protocol or exact-replay tolerance was changed to force a pass. See `replay_caveat.json`; universal bitwise portability is not certified.

All evidence is in [`data/08_reporting/prematch_architecture_20260927_run001/`](../data/08_reporting/prematch_architecture_20260927_run001/): `architecture_training.json`, `architecture_stacking.json`, `architecture_summary.json`, the metrics/interval/profile CSVs, paired and canonical prediction Parquets, three `canonical_*` suite directories, and the numerical, deterministic-refit, CLI and canonical verification receipts. `source_snapshot.zip` preserves the 16 frozen source files; `verification_source.zip` preserves the executed verification scripts. Canonical exports reverse rows to canonical A/B orientation: LL/Brier/tail counts are invariant, but AUC, binned ECE and calibration intercept need not match fresh-orientation summaries. The canonical suite's bootstrap seed is **20260915**, distinct from this experiment's **20260927**.

Re-score a candidate without training; the output directory must not already exist:

```bash
.venv/bin/python scripts/run_model_benchmark.py --suite \
  --candidate-data data/08_reporting/prematch_architecture_20260927_run001/canonical_stack_candidate.parquet \
  --candidate-col specialist_stack \
  --output-dir data/08_reporting/prematch_architecture_stack_rescore_002
```

Exercise the saved historical feature example, not a live roster forecast:

```bash
.venv/bin/python scripts/train_prematch_forecaster.py predict-batch \
  --artifact data/06_models/prematch_architecture_20260927_run001/architecture/2026-01-01/architecture_bundle.joblib \
  --features data/08_reporting/prematch_architecture_20260927_run001/inference_example.npz
```

The original run used `architecture-prepare`, `architecture-train`, `architecture-stack`, and `architecture-report`, in that order, with `--protocol data/05_model_input/prematch_architecture_20260927_run001/protocol.json`. Do not rerun fitting into these frozen evidence paths; a new experiment needs new run paths. There is no live raw-roster/history adapter, deployment, financial evaluation, or tournament-edition assessment in this deliverable.

## Original run: development selection

Selection used **7,119 series from 2024-01-01 through 2025-06-29**, with all selection labels released before 2025-07-01. It did not use 2025-H2 or 2026 outcomes. Six development origins produced 48 member checkpoints; three final origins produced 15. Of these scheduled members, 45 are supervised model fits and 18 are rating controls with quarterly calibrators. A further three selected-member refits were run strictly as reproducibility checks.

| Recipe | Development LL | Brier | LL >= 2.5 |
|---|---|---|---|
| ensemble_families | 0.566147 | 0.192859 | 14 |
| lgbm_15__raw | 0.568810 | 0.193866 | 5 |
| lgbm_15__temperature | 0.568831 | 0.194015 | 16 |
| lgbm_31__raw | 0.568986 | 0.193945 | 9 |
| lgbm_31__temperature | 0.569304 | 0.194133 | 19 |
| logistic_c01__raw | 0.573062 | 0.195185 | 33 |
| logistic_c01__temperature | 0.573707 | 0.195452 | 29 |
| roster_32__raw | 0.574887 | 0.196668 | 26 |
| roster_32__temperature | 0.574905 | 0.196659 | 17 |
| ensemble_player_anchor | 0.574956 | 0.196248 | 7 |
| logistic_c1__raw | 0.576889 | 0.196358 | 41 |
| logistic_c1__temperature | 0.577202 | 0.196593 | 31 |
| player_bt__temperature | 0.591911 | 0.203356 | 22 |
| player_bt__raw | 0.592167 | 0.203259 | 34 |
| roster_64__temperature | 0.595859 | 0.205144 | 27 |
| roster_64__raw | 0.605898 | 0.207377 | 82 |
| team_bt__temperature | 0.606680 | 0.209905 | 17 |
| team_bt__raw | 0.607031 | 0.209956 | 27 |

The wider neural configuration did not improve the development result; width, dropout, epochs, and weight decay differ jointly, so this is not an isolated causal claim about width. The player-anchor ensemble reduces deep losses to seven but has worse total loss than the selected recipe. Calibration is not automatically beneficial: the selected ensemble uses raw components. These are measured negative results, not reasons to quietly remove candidates from the report.

![All development recipes on their common cohort](../data/08_reporting/ground_up_lol_20260927_run001/development_comparison.png)

### Executed final fit partitions

| Simulated origin | TRAIN labels before | TRAIN | CAL | Forecasts |
|---|---|---|---|---|
| 2026-01-01 | 2025-07-01 | 36205 | 1328 | 1034 |
| 2026-04-01 | 2025-10-01 | 37382 | 1185 | 1253 |
| 2026-07-01 | 2026-01-01 | 37533 | 2287 | 816 |

**Origins are simulated availability origins, not actual fit dates.** Training and forecast generation were executed on 2026-09-27. Prediction files retain actual `generated_at` timestamps. Earlier final outcomes can enter the predeclared later history/CAL windows; the recipe itself remains frozen. The 2025-H2 window was reserved before any fitting for initial final calibration, not reused for configuration selection.

## Original artifact and pre-repair verification

Saved model: [`selected_forecaster.joblib`](../data/06_models/ground_up_lol_20260927_run001/selected_forecaster.joblib), **2,290,661 bytes**. It contains the three selected final-quarter models, their stored calibration objects, the raw-variant averaging recipe, and the final released-history state. The model's simulated origin is **2026-07-01**; its history snapshot ends **2026-09-08**. It does not fetch newer results automatically.

| Evidence | Path |
|---|---|
| Frozen experiment protocol | `data/05_model_input/ground_up_lol_20260927_run001/protocol.json` |
| Rebuilt features, history state, source audit | `data/04_feature/ground_up_lol_20260927_run001/` |
| Quarterly members and selected forecaster | `data/06_models/ground_up_lol_20260927_run001/` |
| Development, CAL, final and prior-lineup predictions | `data/07_model_output/ground_up_lol_20260927_run001/` |
| Metrics, profiles, plots, selection and verification | `data/08_reporting/ground_up_lol_20260927_run001/` |

Verification completed:

- **81 focused regression tests passed**, including 14 new forecasting-contract cases; one pre-existing Starlette/httpx deprecation warning was emitted. This was not the full application/database test suite.
- Independent NumPy scoring reproduced all **18 development recipes** and every final summary probability column; final target coverage is **3,103/3,103**, without duplicate IDs.
- Independent grouped calculations reproduced the **9-month and 143-source-group intervals**.
- **30 raw/temperature final-member forecast replays**, matching CAL replays, and corresponding prior-lineup replays passed. Maximum stored-forecast replay difference was **2.79e-08**, within the declared float32-network tolerance of 1e-6; CAL replay difference was zero.
- Maximum side-complementarity error over those final replays was **2.22e-16**; the 1e-12 symmetry requirement passed.
- Re-fitting **all three selected final-origin members** on the same pinned TRAIN/CAL reproduced probabilities and calibration temperatures with **zero observed difference**. This is a fitting check, not only a load/save check.
- Saved-artifact queries worked without organization history, returned **0.5** for symmetric wholly unseen rosters, preserved the state hash, and rejected a query whose history snapshot was not strictly prior.
- The actual `predict` CLI ran in a fresh Python process and reproduced **p(A)=0.4612781332940849** for the saved **hypothetical** example. This is not a claim about a real scheduled match or an announced September 27 lineup.

The tests were executed with:

```bash
.venv/bin/python -m pytest -q betting_app/tests/test_prematch_forecaster.py betting_app/tests/test_candidate_calibration.py betting_app/tests/test_model_benchmark.py betting_app/tests/test_research_benchmark_suite.py betting_app/tests/test_production_a0_calibration.py
```

That last pre-existing test module exercises shared scoring/bootstrap contracts; it did not supply a previous model or prediction to the new training run. The structured evidence is in `development_verification.json`, `verification.json`, and `cli_prediction.json` in the report directory.

Reproduce the demonstrated inference:

```bash
.venv/bin/python scripts/train_prematch_forecaster.py predict \
  --artifact data/06_models/ground_up_lol_20260927_run001/selected_forecaster.joblib \
  --match-json data/08_reporting/ground_up_lol_20260927_run001/inference_query.json
```

The query JSON requires `team_a`, `roster_a`, `team_b`, `roster_b`, `day`, and `best_of`. Rosters contain five distinct explicit IDs in TOP/JUNGLE/MID/ADC/SUPPORT order; ten players must be distinct across the matchup. Organization IDs may be null. The day must follow the loaded history snapshot. Use code/environment matching the recorded artifact provenance; this trusted-local joblib interface is not a cross-version migration or untrusted-upload service.

Figures are reproducible without fitting or reselection:

```bash
.venv/bin/python scripts/plot_prematch_forecaster.py --protocol data/05_model_input/ground_up_lol_20260927_run001/protocol.json
```

The corpus also changes over time: BO1 accounts for **66.09%** of 2024 binary targets but **24.01%** of 2026 targets. Annual headline losses are therefore not interchangeable populations.

![Measured source coverage, missingness, history support, and roster membership](../data/08_reporting/ground_up_lol_20260927_run001/dataset_overview.png)

## Original study: research question and evidence standard

The original request was to restart model design from raw sports evidence, rather than refine an inherited architecture or defend a thesis result. That original study therefore used **no earlier model weights, predictions, learned embeddings, or feature bank**. Existing raw parsing, metric definitions, and independently archived result-release corrections were infrastructure, not an inherited predictor. The later architectural integration explicitly reuses A0's raw history bank as described above. Earlier artifacts and the live application remain unchanged.

“Best” means the best recipe selected in the declared experiment—not the best possible sports model, not a bookmaker-beating system, and not a production certificate. The experiment targets **professional League of Legends series before draft**, because that is the local corpus with explicit player identities, role-labelled historical performance, and series structure. Other games were inventoried but were not pooled into LoL training.

Research was conducted on 2026-09-27 through five independent tracks: dynamic skill models, tabular/neural prediction, forecast validation, esports-specific signals, and local data inventory. Primary papers and official documentation were read, rather than treating search snippets as evidence. A separate two-track implementation review covered causal replay and the fit/calibration/selection contract. Literature recommendations and measured results are distinguished below.

### First choose the prediction, not the architecture

| Forecast | Appropriate target | Admissible information | Main evaluation |
|---|---|---|---|
| Football before kickoff | Outcome distribution or score distribution | Earlier matches, available lineup/venue/context | Log loss/Brier for outcomes; proper scores for the full score distribution |
| LoL before a series | Whole-series winner under a known format | Earlier released history and an explicitly stated roster scenario | Series-level log loss and Brier |
| LoL after a map draft | That map's winner | The finalized, timestamp-available draft and earlier history | A separate map-level benchmark |
| In-play forecasting | Conditional future outcome | Only events already observed in the current match | Evaluation at specified game states/times |
| Tournament forecasting | Joint wins/qualification/placement distributions | Rules and information available at the tournament origin | Edition-level scoring and independent format oracles |

Dixon–Coles models football goal counts through a dynamic Poisson-based construction; that does not make a Poisson goal model the natural likelihood for a LoL series. The transferable lessons are a target-appropriate generative model, changing strength, and out-of-period evaluation—not importing a reported historical betting return into another sport.[1] Dynamic paired-comparison models are a more direct starting point for binary head-to-head outcomes.[2][3]

## What the research supports

### Dynamic skill is a useful control, not a complete sports model

A Bradley–Terry-style model expresses the map win probability through a strength difference, `sigmoid(s_A − s_B)`. Dynamic rating systems add uncertainty and changing abilities. Glickman's work treats ability as time-varying; the Glicko-2 note explicitly treats games within a rating period as simultaneous.[2][3] TrueSkill illustrates probabilistic team-strength inference, and TrueSkill Through Time illustrates temporal skill modeling.[4][5]

These sources support three engineering choices: predict before consuming the result, compare both opponents from the same prior state, and allow uncertainty to grow during inactivity. They do **not** establish universal volatility constants, prove that averaging player strengths is optimal, or validate a diagonal approximation to all teammate/opponent covariances.

The fresh controls here are consequently labelled **independent diagonal-Gaussian dynamic Bradley–Terry approximations**, not Glicko-2, exact TrueSkill, or a reimplementation of either paper. Organization and player controls are separate. A player can transfer without losing their history; an organization can change its roster without being mistaken for the same five players. No nationality is inferred from competition participation.

### Trees deserve a serious comparison; neural networks do not deserve automatic preference

Grinsztajn, Oyallon, and Varoquaux compare methods on 45 tabular datasets and find tree methods strong on medium-sized tabular problems.[6] That is evidence for including a tree baseline, not a theorem about LoL. The LightGBM paper and its tuning documentation support histogram trees while warning about leaf-wise overfitting.[7][8] CatBoost's ordered boosting addresses a specific training-time prediction-shift mechanism; it does not make retrospectively constructed sports features causally available.[9]

Newer TabM work is a useful counterweight to the claim that deep learning is categorically unsuitable for tables: parameter-efficient MLP ensembles can be competitive.[10] It was researched, **not implemented** in this run. With numeric history summaries and no validated categorical patch/family variables in the frozen input, LightGBM was selected as the tree representative. CatBoost and LightGBM were not counted as two independent scientific families.

A compact roster network supplies a different inductive bias: share a player encoder, preserve five role slots, and learn nonlinear comparisons between the resulting lineups. Set Transformer motivates interaction-aware set processing, but LoL roles are not interchangeable.[11] The implemented model is a **role-aware MLP with concatenated role slots**, not a Set Transformer, a historical-token attention model, or a pretrained foundation model.

| Family | Useful inductive bias | Main risk | This experiment |
|---|---|---|---|
| Dynamic organization rating | Low-variance aggregate result history | Organization identity can outlive its roster | Fresh control |
| Dynamic player rating | Transferable player histories and uncertain cold starts | Correlated teammate abilities; disconnected competition pools | Fresh control |
| Regularized logistic regression | Additive signed evidence; stable small-data behavior | Missed nonlinear interactions | Two fixed regularization strengths |
| LightGBM | Thresholds and interactions in numeric histories | Overfitting sparse support, eras, and missingness patterns | Two constrained configurations |
| Role-aware neural MLP | Shared player representation and role interactions | Estimation variance and imperfect history compression | Two compact configurations |
| CatBoost | Native categorical processing | Causal availability still external to the learner | Researched; not run |
| Set/temporal attention | Flexible lineup/history interactions | Added complexity without demonstrated gain | Researched; not run |
| TabM | Parameter-efficient neural ensembles | Different method requiring its own fixed comparison | Researched; not run |
| TabPFN | Pretrained tabular prior | Full-cohort CPU practicality and separate weight terms | Researched; not run |

The current TabPFN documentation describes CPU use for moderate sample sizes, smaller than this full training corpus, and separately licensed weights. The inspected current weight licence restricts commercial/production use. Those are reasons not to treat it as an unrestricted full-cohort production candidate here, not a claim that all TabPFN variants are impossible to use.[12][13]

### Esports headline accuracies are often answers to different questions

Do et al. report 75.1% neural accuracy and 75.4% boosted-tree accuracy on 5,000 ranked NA matches, using a random 80/20 split and player–champion information after champion selection.[14] This is not professional pre-draft probability evaluation. It supports investigating player–champion experience when the champion is actually known, not importing 75% as an expected result on this corpus.

Role-relative historical performance is a defensible feature construction, but the paper by Bahrololloomi et al. does not establish a clean chronological professional benchmark: its reported role-weight optimization and shuffled evaluation limit that interpretation.[15] This experiment uses earlier same-role comparisons; it does not claim that the literature has established a universal ranking of role importance.

Riot also warns that Data Dragon updates are manual and that its versions are not always equivalent to regional client versions.[16] A historical patch field, a current static-data version, and an esports competition patch are not interchangeable. Target draft, target side, and target patch labels were therefore not silently promoted to pre-match inputs.

### Probability quality is not betting profitability

Strictly proper scoring rules reward honest predictive distributions. Gneiting and Raftery emphasize that competing forecasts must refer to the **same forecast situations** to be directly comparable.[17] Accuracy alone discards whether a correct forecast was 0.51 or 0.99 and cannot certify the usefulness of probabilities.

Calibration must use predictions from the exact fitted model that is subsequently evaluated, on data separate from its training and later assessment. Calibration research shows both the value of this step and the tendency of flexible calibration to overfit small calibration sets.[18] A one-parameter temperature map was chosen here instead of a flexible post-hoc correction search.

Even good sports forecasting does not imply reliable betting returns; the inspected football forecast study explicitly examines this distinction.[19] This run did not test price availability, tax-adjusted execution, Kelly allocation, liquidity, CLV, or live bet selection. No financial result is inferred from a lower log loss.

## Audited local data

### Primary LoL source

Source: `data/artifacts/golgg-database-recovery-20260909/matches.json`.

SHA-256: `991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca`.

The source was streamed in full; the following are measured counts, not estimates from a preview:

| Item | Count / range |
|---|---:|
| Series | 41,915 |
| Maps | 69,370 |
| Player-map records | 693,700 |
| Distinct explicit player IDs | 7,056 |
| Distinct explicit team IDs | 2,836 |
| Recorded dates | 2013-09-16–2026-09-08 |
| Fresh-source series / maps | 39,110 / 63,414 |
| Cache-recovered series / maps | 2,805 / 5,956 |
| Maps missing a patch label | 1,714 |
| Series with delayed effective release | 527 |
| Supplied release-correction entries | 7,866; none unused |
| Valid binary series targets | 40,636 |

The validation found no duplicate series IDs, duplicate map IDs, missing five-role slots, invalid ten-player map lineups, or inconsistent map win flags in the admitted source. These checks validate the recorded structure; they do not independently establish that every identifier refers to the correct real-world person across all eras.

The preserved source manifest still declares **incomplete mixed provenance**, including 309 unavailable/pending/error series. This experiment does not erase that incompleteness or invent records to fill it. It also does not certify publication timestamps or roster-announcement timestamps.

| Format | Source series | Binary-target treatment |
|---|---:|---|
| BO1 | 25,888 | Included |
| BO2 | 1,279 | All excluded as binary targets; all maps retained as history |
| BO3 | 11,009 | Included |
| BO5 | 3,739 | Included |

Of the BO2 series, 494 are draws and 785 are decisive. Keeping only decisive BO2 outcomes would condition the target sample on the result. Instead, the whole format is outside this binary target, while its map-level evidence remains usable history. BO2 prediction would need a separate three-outcome contract.

Historical source-header BO values can disagree with corrected source-map declarations. The independent audit identified 785 decisive BO2 records whose old header said BO3; their fresh map `source_best_of=2` originated from an explicit GOL.GG BO header, not an inference from winning 2–0. Recovered rows have less uniform provenance. The current pipeline does not infer BO from the eventual score, and the source remains labelled mixed-provenance.

### Historical performance coverage

Missingness is measured over all 693,700 player-map records. These are **past-map measurements** when used by the model, never measurements from the series being forecast.

| Source field | Missing | Missing share |
|---|---:|---:|
| CS/min | 16,750 | 2.41% |
| Gold/min | 16,750 | 2.41% |
| Damage/min | 16,750 | 2.41% |
| Kill participation | 16,750 | 2.41% |
| Damage share | 16,750 | 2.41% |
| Vision score/min | 120,810 | 17.42% |
| `gd@15` | 73,350 | 10.57% |
| `csd@15` | 91,680 | 13.22% |
| `xpd@15` | 91,680 | 13.22% |

Coverage changes by era. The source audit found nearly complete core CS/gold/damage coverage since 2020, with materially weaker coverage for some early-game and vision fields. Missing values remain missing, with support counts and train-fitted missingness indicators. They are not silently interpreted as zero performance.

The exact historical semantics of fields such as `gd@15` are not fully certified by a versioned source dictionary. The feature is therefore explicitly the **recorded own-role value minus the recorded opposing-role value**, not a newly asserted physical definition such as team gold lead. Such features can still shift when a source changes measurement conventions.

### Other available data: separate inventories, not extra LoL examples

The following table counts were independently read from local Parquet metadata:

| Table | Physical rows | Why not pooled into LoL training |
|---|---:|---|
| CS:GO maps | 127,469 | Different game; not CS2; stable team/series identity and player-side alignment limitations |
| Valorant maps | 127,813 | Different roles, maps, round mechanics, and measurement units |
| VCT series | 12,676 | A separate target population and chronology contract |
| Chronology-eligible VCT series table | 11,652 | Eligibility does not resolve every source/identity issue |
| VCT player-map/side records | 1,149,895 | Includes side/aggregate row types; not that many independent player maps |

A separate market snapshot declares 920 fixtures and 184,298 quotes, with 895 identity mappings. It was not used as a training input or for the original model selection. The later user-requested bookmaker comparison above uses its matched subset. Quotes are repeated observations of prices, not independent match outcomes.

Official esports feeds, public developer APIs, community downloads, and the preserved GOL.GG recovery have different provenance and usage terms. The Riot/GRID and Oracle's Elixir sources do not grant this repository a blanket redistribution licence.[16][20][21] No new scraping or redistribution was performed.

## From raw history to causal inputs

### The clock is an explicit scenario

For a genuine timestamped forecast, the desired contract is:

`source/release time <= data cutoff <= forecast time < target start`.

The source does not establish all those timestamps. The executed daily scenario is instead:

- Target day: earliest recorded header/map calendar date.
- Effective release: latest of the header date, every map date, and the independent correction.
- A series contributes history only if `effective_release_day < target_day`.
- All forecasts on a date use the same pre-date state; same-day results are excluded.
- Opponents' rating updates within a release-day batch are computed from frozen pre-day priors.

No target-date correction was required on the current source, but the conservative rule is implemented. The generated audit verified strict release-before-target history on every target. The maximum source release is 2026-09-08; the maximum historical release used by a target is 2026-09-07.

The 7,866 correction entries were copied byte-for-byte from an independently archived release-evidence file, not a prior model feature bank. Their SHA-256 is `9216682230264b160f3e83402fca63cbfece385c4b97422639078d93fe7b126c`. The archive-relative origin is recorded in the protocol. No personal archive path is hardcoded into the predictor.

### Two roster scenarios, never a hidden fallback

**Primary:** the actual first-map five is supplied as a known lineup. This is an explicit retrospective known-lineup scenario; it is not proof that the lineup was announced at the forecast time. Players who first appear later in the target series are not smuggled into the initial lineup.

**Secondary:** the latest released-day modal organization lineup is used, with a deterministic lexical tie-break. A missing historical lineup produces an unknown/cold lineup; it never falls back to the target's actual five. This secondary forecast measures sensitivity to roster availability, not a second population selected for favorable coverage.

History is credited to the players who actually appeared in **each historical map**, including substitutes. Hypothetical queries do not update any history. Querying two new organizations or a national roster is possible with five explicit role-ordered player IDs per side; organizational history is not required.

### Fresh feature specification

Each side contains five role-ordered players: TOP, JUNGLE, MID, ADC, SUPPORT. Each player has 28 features:

- Slow and fast rating mean/standard deviation.
- Log appearance count and log inactivity since the last released appearance.
- Opponent-strength and win-residual summaries over two map-history scales.
- Nine same-role source-stat differences with an EWMA-10 summary.
- Corresponding valid-observation counts.

Each team has 12 features: slow/fast organization rating and uncertainty, history count/inactivity, two result-residual summaries, previous-lineup overlap/availability, and mean/minimum previous pair coplay. BO1/3/5 is one-hot context. Two additional controls are the organization and player **whole-series** probabilities.

Source-stat fields are `csm`, `gpm`, `dpm`, `vspm`, `kp%`, `dmg%`, `gd@15`, `csd@15`, and `xpd@15`. Feature definitions and ordering are recorded in `audit.json`; they are not inferred later from model coefficient positions.

No target score, target performance, target champion, blue/red assignment, odds, prior-model probability, raw identity embedding, or future roster is a supervised input. Team/player IDs serve history lookup and alignment, not unrestricted categorical memorization.

### Map probability and series probability are different

The slow rating control uses a Gaussian strength-difference approximation, `Z ~ Normal(mu_A − mu_B, v_A + v_B)`. For player teams, strength is the mean of the five player strengths and variance uses the corresponding squared coefficients, under the declared diagonal assumption.

Conditional on a shared latent map-win probability `p = sigmoid(Z)`, the series functions are:

- BO1: `H_1(p) = p`.
- BO3: `H_3(p) = 3p² − 2p³`.
- BO5: `H_5(p) = 10p³ − 15p⁴ + 6p⁵`.

The original run evaluated `E[H_B(sigmoid(Z))]` with 24-node Gauss–Hermite quadrature; the numerical repair above replaces it with a centered, 128-node calculation. Neither applies `H_B` to an already averaged map probability or projects a series probability twice. Independent numerical-integration regression oracles check this distinction.

This likelihood assumes conditional exchangeability of maps at one latent strength. It does not model future drafts, alternating sides, momentum, deliberate opponent choices, or within-series roster changes. The supervised classifiers learn the series label directly and receive BO as context.

Slow and fast process-variance increments are fixed at 0.002 and 0.02 per elapsed day, with a variance cap of 4. Initial organization/player variances are 1 and 4. These are frozen control choices, not literature-certified optima. The experiment did not search rating hyperparameters on its final outcomes.

## Frozen training and selection protocol

Protocol: `data/05_model_input/ground_up_lol_20260927_run001/protocol.json`.

Seed: **20260927**. Scores consistently clip probabilities at **1e-6**, including headline loss, paired differences, and profile metrics. This avoids comparing differently clipped populations or making a small tail count the main objective.

### Three separate update schedules

1. **History/rating updates:** daily, after whole-series release, with frozen-opponent batches.
2. **Model fitting:** quarterly, using all eligible earlier TRAIN examples with positive two-year-half-life weights.
3. **Calibration:** the preceding six-month CAL window, applied to that exact fitted model; no refitting the base model after calibrating it.

For origin `o`, TRAIN targets and labels must both be before `o − 6 months`. CAL targets lie between that boundary and `o`, with labels released strictly before `o`. The following quarter is forecast. No early stopping, hyperparameter search, or ensemble weighting uses the forecast quarter.

Development origins are 2024-01-01, 2024-04-01, 2024-07-01, 2024-10-01, 2025-01-01, and 2025-04-01. Recipe selection uses only development targets **and label releases before 2025-07-01**. Before any candidate fitting, the initial broader development plan was tightened to reserve **2025-H2 exclusively for the first final calibrator**. Thus a selected configuration is not chosen using labels from its initial final calibration window.

Final origins are 2026-01-01, 2026-04-01, and 2026-07-01. Later quarters may use earlier 2026 outcomes for the predeclared rolling calibration/history updates, never for recipe reselection. This is prequential evaluation, not one unchanged model frozen for all of 2026.

The final year was already exposed in earlier project work. Starting a new model does not make those outcomes newly independent. The result is therefore labelled **historical locked-recipe assessment**, not a fresh prospective holdout.

### Candidate menu fixed before fitting

| Candidate | Fixed specification |
|---|---|
| Organization BT | Fresh slow dynamic organization control |
| Player BT | Fresh slow dynamic player control with five-player aggregation |
| Logistic C=0.1 / C=1 | L2, signed side differences and BO interactions, no intercept, TRAIN-only preprocessing |
| LightGBM 15 leaves | 500 trees, minimum child 150, L2=5 |
| LightGBM 31 leaves | 600 trees, minimum child 100, L2=10 |
| Both tree models | Learning rate 0.03, column fraction 0.8, deterministic mode, two threads, side-mirrored training |
| Neural width 32 | 24 epochs, dropout 0.1, weight decay 0.001 |
| Neural width 64 | 32 epochs, dropout 0.2, weight decay 0.003 |
| Both neural models | AdamW, learning rate 0.001, batch 512, CPU, two threads, TRAIN-only normalization |

Every base candidate has raw and scalar-temperature variants, yielding 16 variants. Temperature is fitted on CAL with bounds 0.25–4. Two additional recipes were predeclared: an equal-weight average of development-best logistic/tree/neural variants, and an equal-weight average of the best supervised variant with the best player control. Their constituents are selected only on development data. No learned stacking weights or extra ensemble calibrator are fitted.

Selection is pooled development log loss, then Brier, then deterministic name. The strongest development-selected rating control is the frozen reference. Final family representatives are likewise selected on development data; a family that looks better on final outcomes does not replace the selected recipe.

### Side symmetry is a construction, not a hope

For all models, swapping A/B—including both model-control probabilities—must yield `p(B,A) = 1 − p(A,B)`.

- Logistic regression uses signed differences and no free intercept.
- The neural logit is half the difference between the shared ordered-pair scorer on `(A,B)` and `(B,A)`.
- Tree inference uses `(q(A,B) + 1 − q(B,A))/2`; mirrored training alone is not treated as a guarantee.
- Temperature scaling preserves the signed-logit symmetry.

The two baseline probability columns index **organization/player models**, not two teams. A regression test with distinct, nonconstant controls caught that axis confusion before any full candidate fitting; it was fixed before this run. A test containing only 0.5 controls would have hidden it.

### Evaluation and uncertainty

Primary metrics are log loss and Brier, on identical target IDs. Accuracy, ROC AUC, calibration diagnostics, and losses above a fixed log-loss threshold are secondary. Profiles are by quarter, BO, identity-source provenance, confidence, and minimum player-history support. The metadata column named `source_provenance` describes header-derived versus map-derived **team identity**, not fresh-versus-cache origin; the independent `source_lineage.parquet` records the latter.

For selected-minus-reference loss, 5,000 paired calendar-month bootstrap resamples report both match-weighted and equal-month effects. A second sensitivity calculation groups source tournament label within year. These labels are **not independently verified tournament editions**; neither grouping removes every repeated-team dependency. A confidence interval containing zero proves neither improvement nor equivalence. A bootstrap nonnegative fraction is not presented as a classical p-value.

Calibration bins are descriptive and should be read with their counts. Sparse extreme bins cannot establish that a model is safe for high-stakes decisions. Covariate shift and concept drift are also distinct: covariate-shift methods assume an unchanged conditional relationship, which a patch or rule change may violate.[22]

## Implementation, reproducibility, and use

Reusable implementation is in the repository:

- `src/models/prematch_features.py`: exact-ID validation, release-batched replay, dynamic ratings, player/team histories, feature/state artifacts.
- `src/models/prematch_tabular.py`: signed logistic and symmetrized LightGBM models.
- `src/models/prematch_neural.py`: shared role-aware player encoder and antisymmetric pair scorer.
- `src/models/prematch_predictor.py`: chronological partitions, matching calibrated members, saved forecast artifact, pure matchup inference.
- `scripts/train_prematch_forecaster.py`: prepare/develop/final/predict CLI, provenance checks, fit receipts, common scoring/bootstrap functions.
- `scripts/plot_prematch_forecaster.py`: dataset overview, complete development comparison, and final calibration/monthly figures from stored artifacts; no fitting or reselection.
- `betting_app/tests/test_prematch_forecaster.py`: temporal, identity, symmetry, uncertainty-integration, checkpoint, and stage-lock contracts.

The stage lock hashes local calibration, probability, metric, bootstrap, parsing, and model dependencies—not just the orchestration script. Source, release, protocol, feature, model, and prediction hashes are recorded at their stages. Dependency versions are recorded in `prepared.json`. The earlier preparation receipt remains its actual historical receipt; later training/selection receipts include the strengthened code-dependency lock.

From the repository root, the executed stages are:

```bash
.venv/bin/python scripts/train_prematch_forecaster.py prepare --protocol data/05_model_input/ground_up_lol_20260927_run001/protocol.json
.venv/bin/python scripts/train_prematch_forecaster.py develop --protocol data/05_model_input/ground_up_lol_20260927_run001/protocol.json
.venv/bin/python scripts/train_prematch_forecaster.py final --protocol data/05_model_input/ground_up_lol_20260927_run001/protocol.json
```

The completed directories are intentionally not overwritten on rerun. To reproduce training, copy the protocol to a new run and change its run ID and output directories while retaining source hashes, partitions, candidate specifications, and seed; then run those three stages in order. Do not edit a selected protocol or implementation and call the result the same locked experiment.

The saved artifact contains the final-quarter selected members, their corresponding calibrators, and history through the source's last released day. It is a local inference artifact, not an application deployment. Load joblib only from a trusted local source. To extend history or refit on newly acquired data, create a new explicit replay/training run rather than pretending the saved state updates itself.

## Limits and next decision

1. **Historical evidence, not prospective independence.** Prior exposure to 2026 cannot be undone by changing architecture or naming a new run.
2. **Roster information is the largest availability qualification.** The primary scenario uses realized first-map lineups; only the separate prior-lineup scenario avoids that assumption. Neither establishes roster-announcement timestamps.
3. **Daily release is not a verified publication clock.** Source corrections are useful conservative evidence, but exact result/statistic arrival times remain missing.
4. **Source incompleteness and changing measurements remain.** The missing series, recovered rows, uncertain patch crosswalks, and old stat definitions are not repaired by model fitting.
5. **Gaussian ratings are approximate.** They omit teammate/opponent covariance and do not independently identify disconnected competition-family offsets. Inactivity is measured against released appearances, not certified real-world activity.
6. **History compression is a modeling choice.** EWMA summaries do not preserve the full map sequence, champion context, or richer opponent-dependent retrieval. Their adequacy is measured here, not assumed universal.
7. **Candidate search is finite.** The experiment does not exhaust CatBoost, TabM, state-space likelihoods, causal regional pooling, or sequence encoders. No unrun method is assigned a project score.
8. **No tournament or financial certification.** Match probability quality does not verify a bracket implementation, joint tournament distributions, executable prices, or profitable stakes.
9. **No deployment.** The live service, scheduler, production database, and previously loaded operational model were not changed.

The immediate useful next evidence is a frozen prospective capture of available rosters, result/statistic releases, and forecast timestamps. The requested history/input/representation and learned-specialist comparison has now been executed above and did not beat A0. Do not restart it as an untested idea. A further architecture experiment requires a distinct frozen hypothesis; more flexible corrections fitted to these already inspected losses would not be independent validation.

## Bibliography

[1] Dixon, M. J., and Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Journal of the Royal Statistical Society, Series C. https://doi.org/10.1111/1467-9876.00065

[2] Glickman, M. E. (1999). *Parameter estimation in large dynamic paired comparison experiments*. Applied Statistics. https://www.glicko.net/research/glicko.pdf

[3] Glickman, M. E. (2022 revision). *Example of the Glicko-2 system*. https://www.glicko.net/glicko/glicko2.pdf

[4] Herbrich, R., Minka, T., and Graepel, T. (2006). *TrueSkill: A Bayesian Skill Rating System*. NeurIPS. https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html

[5] Dangauthier, P., Herbrich, R., Minka, T., and Graepel, T. (2007). *TrueSkill Through Time: Revisiting the History of Chess*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2007/file/9f53d83ec0691550f7d2507d57f4f5a2-Paper.pdf

[6] Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

[7] Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS. https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

[8] LightGBM contributors (accessed 2026-09-27). *Parameters Tuning*. https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html

[9] Prokhorenkova, L., et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS. https://papers.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf

[10] Gorishniy, Y., Kotelnikov, A., and Babenko, A. (2024 preprint; ICLR 2025). *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*. https://arxiv.org/abs/2410.24210

[11] Lee, J., et al. (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML, PMLR 97. https://proceedings.mlr.press/v97/lee19d.html

[12] Prior Labs (accessed 2026-09-27). *TabPFN official repository and CPU guidance*. https://github.com/PriorLabs/TabPFN

[13] Prior Labs (accessed 2026-09-27). *TabPFN 3.5 model-weight licence*. https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE

[14] Do, T. D., Wang, S. I., Yu, D. S., McMillian, M. G., and McMahan, R. P. (2021). *Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends*. FDG. https://arxiv.org/abs/2108.02799

[15] Bahrololloomi, F., Sauer, S., Klonowski, F., Horst, R., and Dörner, R. (2022). *A Machine Learning based Analysis of e-Sports Player Performances in League of Legends for Winning Prediction based on Player Roles and Performances*. HUCAPP. https://doi.org/10.5220/0010895900003124

[16] Riot Games (accessed 2026-09-27). *League of Legends developer documentation and Data Dragon*. https://developer.riotgames.com/docs/lol

[17] Gneiting, T., and Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. Journal of the American Statistical Association. https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

[18] Niculescu-Mizil, A., and Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. ICML. https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf

[19] Reade, J. J., Singleton, C., and Vaughan Williams, L. (2020). *Betting markets for English Premier League results and scorelines: evaluating a forecasting model*. Economic Issues, 25(1), 87–106. https://centaur.reading.ac.uk/89738/1/reade_singleton_scorelines.pdf

[20] GRID (2023-11-30). *Riot Games and GRID Announce Exclusive Global Esports Data Partnership*. https://www.grid.gg/resources/newsroom/riot-games-and-grid-announce-exclusive-global-esports-data-partnership/

[21] GRID (2024-08-01). *Partner Spotlight: Oracle's Elixir Brings Cutting-Edge VALORANT Stats to Esports Fans*. https://www.grid.gg/resources/newsroom/oracles-elixir-brings-cutting-edge-valorant-stats-to-esports-fans/

[22] Sugiyama, M., Krauledat, M., and Müller, K.-R. (2007). *Covariate Shift Adaptation by Importance Weighted Cross Validation*. Journal of Machine Learning Research. https://jmlr.org/papers/v8/sugiyama07a.html
