# Native C0, corrected thesis models and Polish bookmakers — executed comparison

**Run:** `plan12_model_bookmaker_20261009_175136Z`. **Evidence level:** historical, development-exposed, source-conditional 2026 comparison. Betting profitability is not evaluated.

## 1. Answer

The collected Polish consensus has the best **observed** log loss and Brier. Native C0 is the closest of the three models; corrected thesis LR and MLP8 have larger point deficits. None of the twelve family-adjusted log-loss comparisons establishes a performance direction. This does not establish equality.

There is no adopted, justified practical loss budget. Formal competitiveness and equivalence are **NOT_ASSESSED**. Under every declared illustrative LL budget through0.020, the family-adjusted bounds fail to establish any model's non-inferiority or equivalence to the consensus. All three also have higher observed Brier, so the frozen overall advancement guard fails. These failures are not proof of unacceptable population inferiority.

The answer is **insufficient evidence for a unique best forecaster, practical model–bookmaker equivalence or a credible overall competitive-model claim**. No model is promoted, and no betting recommendation follows. Independent arithmetic/source verification is distinguished from authentication of issued forecasts or statistical assumptions.

## 2. Frozen identity, information and population

- **Native C0:** unchanged `c0-native-2026-w32-e12-v1`, original full-mode native-control `p_full_full`; not annual A0, a neural-only component or a substituted serving model.
- **Thesis models:** current corrected `lr_ts_standard_seed42` and `mlp_ts_standard_seed42`/MLP8 saved `p_calibrated`, with explicit feature/native participant orientation.
- **Book reference:** individually multiplicatively de-vigged Betfan, LV BET, STS, Superbet and available Fuksiarz probabilities, equal averaged. Primary requires at least two distinct books; sensitivity requires all fixed four: Betfan/LV BET/STS/Superbet. Fuksiarz appears on six admitted primary fixtures. Betclic/eFortuna are absent; Rainbet and Pinnacle are not silently substituted.
- **Quoted origin:** source-last stored prestart endpoint, named `close` in the collected files. It is **not authenticated CLOSE**, OPEN, actual quote arrival or a fixed-hour T−6h/T−12h/T−24h/T−48h origin.
- **Model origin:** saved origins no later than the source UTC event day; feature-history maximum days strictly before it. Actual first-map rosters remain an explicit conditional scenario. This does not prove historical model creation, roster announcement or synchronized quote/model availability. No forecasts are backdated.
- **Grouping calendar:** source UTC dates2026-01-14 through2026-09-07, inclusive. Four model-backed source events have source dates one day later than their original native target dates; three belong to both scoring cohorts. [Date-field diagnosis](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/source_vs_native_date_aliases.csv) retains these differences. The source-export date, rather than the native target date, is the declared grouping field; feature/release fences and fixture IDs remain checked. These differences do not authenticate actual start times.

### Source opportunity ledger

| Disposition | Fixtures |
|---|---:|
| Admitted primary Polish2plus |1467|
| No authenticated saved C0 endpoint/model match |258|
| Only one Polish bookmaker |22|
| Source date after known whole-series release day |2|
| Total frozen source close opportunities |1749|

The258 unavailable rows retain unknown model-support/source-clock fields; they are not imputed as neutral forecasts or fabricated timestamps. The two explicit date exclusions remain `8SqeE75l`/GOLGG76228 and `nwoBAP4D`/GOLGG76234. The fixed-four cohort has1285 of the1467 primary fixtures;182 primary rows lack all four. The2752 saved scoring rows are **nested-cohort repetitions**, not2752 independent or distinct matches. The all-endpoint quote verification inventory includes one additional open-only event; it is not an extra close opportunity.

Primary book-count composition is60 two-book,122 three-book,1279 four-book and6 five-book fixtures. Per-book admitted availability: Betfan1423, LV BET1410, STS1391, Superbet1402, Fuksiarz6. The panel is this frozen collected snapshot, not every2026 LoL match or every Polish bookmaker.

## 3. Probability construction and statistical contract

For a two-way series market with decimal prices $o_A,o_B$, use

$$p_A=\frac{1/o_A}{1/o_A+1/o_B}=\frac{o_B}{o_A+o_B},\qquad p_B=1-p_A.$$

Reconstruct each book separately, align participants by reviewed/native IDs, then average its probabilities. Do not average odds first, use outcomes to choose orientation, mix opening/closing prices or apply another BO projection. Tax/payout profitability is a separate question.

Natural-log binary LL is primary; $(p-y)^2$ Brier is secondary. Scores and paired differences use identical fixtures and **epsilon0/no clipping**. Half credit is assigned to an exactly0.5 forecast for diagnostic accuracy; all other forecasts use their strict favorite. Proper scores assess probability quality rather than merely the number of correct favorites. [1]

### Shared uncertainty

- 5,000 saved shared random-start samples, seed `20261016`; **seven consecutive calendar days**, not Monday–Sunday weeks.
- 237-day common calendar, 231 noncircular candidate seven-day windows; 34 blocks per replicate, with the final block retaining six days. Empty dates are retained: 23 primary and 28 fixed-four days.
- Each replicate's paired loss sum is divided by its sampled **fixture count**, not by the number of active days. Primary sampled counts range 978–1,982; fixed-four 863–1,737. All pairs/cohorts use the same saved starts.
- Nominal 95%/90% linear-quantile intervals are retained. Family conclusions use Bonferroni-style **99.583333% individual intervals across 12 LL contrasts**: six pairs in each of two nested cohorts. No independence of pairs/cohorts is assumed for this adjustment. It is not a Romano–Wolf/SPA/DM/MCS implementation or correction for all historical adaptive research. [4]
- Monthly sensitivity regenerates the previously declared seed `20261015` sequence: 5,000 draws of nine UTC calendar months, with match-weighted and equal-month results retained separately.

Seven days is the requested dependence assumption, **not a certified sufficient dependence length**. Calendar-block bootstrap validity still depends on the process/dependence/stability assumptions; repeated teams/tournaments and changes through a season are not made independent by this procedure. Only nine months are available for monthly sensitivity. Family-tail quantiles use roughly ten draws per tail at5000 replicates, so Monte Carlo resolution is limited. These are approximate bootstrap bounds, not exact finite-sample coverage or posterior probabilities. [2,3]

Noncircular windows underrepresent boundary dates. The observed point estimate remains the full-panel mean, not the bootstrap mean.

## 4. Absolute probability quality

### Primary:1467 identical series

| Forecaster | LL | Brier | Accuracy | ECE10 | LL≥2.5 |
|---|---|---|---|---|---|
| Polish consensus | 0.559604853 | 0.189266950 | 71.40% | 1.7123% | 0 |
| native C0 | 0.565904908 | 0.192123748 | 70.42% | 2.7190% | 5 |
| thesis LR | 0.570802361 | 0.193908558 | 70.35% | 1.5485% | 2 |
| thesis MLP8 | 0.573508843 | 0.195254691 | 69.53% | 2.1399% | 2 |

Lower LL/Brier is better. Accuracy/ECE and deep-loss counts are descriptive, not extra superiority tests or acceptance thresholds. LR has a smaller point ECE10 than the consensus while having worse LL/Brier: ECE alone does not identify the best probability forecaster.

### Fixed-four sensitivity:1285 identical series

| Forecaster | LL | Brier | Accuracy |
|---|---|---|---|
| Polish consensus | 0.562088251 | 0.189900115 | 71.48% |
| native C0 | 0.570740897 | 0.193537757 | 70.43% |
| thesis LR | 0.575596472 | 0.195682837 | 70.19% |
| thesis MLP8 | 0.575707093 | 0.196105772 | 69.18% |

This is a nested, different population and a separately defined consensus. Its absolute scores cannot be compared with the primary scores as an isolated improvement. [Full eight-vector scores](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/scores.csv) retain AUC, ECE, probability ranges and tail counts.

## 5. Every declared pair

Positive differences mean the challenger has higher loss. Relative excess is100×observed difference/reference mean loss, **not probability-point error or ROI**. It is a point effect; no ratio confidence interval is claimed without resampling its denominator. Display rounding never determines a decision.

### Log loss

| Cohort | Challenger − reference | ΔLL | Relative excess LL | Nominal95% interval | Family12 interval |
|---|---|---|---|---|---|
| primary | native C0 − Polish consensus | +0.006300054 | +1.126% | [-0.007429, +0.020895] | [-0.012585, +0.027929] |
| primary | thesis LR − Polish consensus | +0.011197508 | +2.001% | [-0.005247, +0.027307] | [-0.012643, +0.037644] |
| primary | thesis MLP8 − Polish consensus | +0.013903990 | +2.485% | [-0.002183, +0.030676] | [-0.008338, +0.040705] |
| primary | thesis LR − native C0 | +0.004897453 | +0.865% | [-0.003154, +0.012516] | [-0.006162, +0.016625] |
| primary | thesis MLP8 − native C0 | +0.007603936 | +1.344% | [-0.000594, +0.015397] | [-0.005220, +0.019407] |
| primary | thesis MLP8 − thesis LR | +0.002706482 | +0.474% | [-0.002919, +0.008392] | [-0.005197, +0.011944] |
| fixed4 | native C0 − Polish consensus | +0.008652646 | +1.539% | [-0.004013, +0.021371] | [-0.009261, +0.027759] |
| fixed4 | thesis LR − Polish consensus | +0.013508221 | +2.403% | [+0.000112, +0.026566] | [-0.006396, +0.033741] |
| fixed4 | thesis MLP8 − Polish consensus | +0.013618842 | +2.423% | [+0.000154, +0.026287] | [-0.005863, +0.033267] |
| fixed4 | thesis LR − native C0 | +0.004855575 | +0.851% | [-0.003400, +0.012383] | [-0.006384, +0.016772] |
| fixed4 | thesis MLP8 − native C0 | +0.004966196 | +0.870% | [-0.003600, +0.012655] | [-0.007794, +0.016652] |
| fixed4 | thesis MLP8 − thesis LR | +0.000110621 | +0.019% | [-0.005108, +0.005472] | [-0.007370, +0.008313] |

All six primary nominal95% intervals cross zero. On fixed-four, LR and MLP8 are worse than the consensus at the individual nominal95% level; neither result survives the declared12-contrast family bounds. All12 adjusted intervals cross zero. The point ranking is therefore not an established unique winner or proof that the forecasters are the same.

### Brier secondary guard

| Cohort | Challenger − reference | ΔBrier | Relative excess Brier | Nominal95% interval |
|---|---|---|---|---|
| primary | native C0 − Polish consensus | +0.002856798 | +1.509% | [-0.002806427, +0.008376185] |
| primary | thesis LR − Polish consensus | +0.004641608 | +2.452% | [-0.002186308, +0.011334141] |
| primary | thesis MLP8 − Polish consensus | +0.005987741 | +3.164% | [-0.000564949, +0.012569755] |
| primary | thesis LR − native C0 | +0.001784809 | +0.929% | [-0.001410740, +0.005049138] |
| primary | thesis MLP8 − native C0 | +0.003130943 | +1.630% | [-0.000001550, +0.006326675] |
| primary | thesis MLP8 − thesis LR | +0.001346133 | +0.694% | [-0.000996467, +0.003736469] |
| fixed4 | native C0 − Polish consensus | +0.003637642 | +1.916% | [-0.001511566, +0.008482187] |
| fixed4 | thesis LR − Polish consensus | +0.005782722 | +3.045% | [+0.000202173, +0.011287526] |
| fixed4 | thesis MLP8 − Polish consensus | +0.006205657 | +3.268% | [+0.000562614, +0.011650003] |
| fixed4 | thesis LR − native C0 | +0.002145080 | +1.108% | [-0.001048514, +0.005315952] |
| fixed4 | thesis MLP8 − native C0 | +0.002568015 | +1.327% | [-0.000801211, +0.005881456] |
| fixed4 | thesis MLP8 − thesis LR | +0.000422935 | +0.216% | [-0.001865435, +0.002803787] |

Brier intervals above are nominal/descriptive, not an additional primary family. The advancement rule uses the **observed** Brier nonincrease guard; each model–consensus comparison fails it. That project guard is not a statistical theorem or proof of population Brier harm.

## 6. Competitive deficit versus practical equivalence

For an adopted positive LL budget $\delta$ and applicable bounds $[L,U]$:

| Question | Strict criterion |
|---|---|
| Better |$U<0$|
| Worse |$L>0$|
| Non-inferior/acceptable LL deficit |$U<\delta$|
| Unacceptably inferior |$L>\delta$|
| Practically equivalent |$L>-\delta$ and $U<\delta$|

These are distinct questions. An interval crossing zero means unresolved direction, not equality. Bounds touching a margin do not pass. Standard individual TOST/one-sided NI conventions are not a license to replace the frozen simultaneous family bounds with a favorable90% illustration. [5,6]

No acceptable LL/Brier budget is adopted. The original illustrative grids remain LL0.005/0.010/0.020 and Brier0.001/0.002/0.005; no value is chosen after seeing which model passes. Every declared pair has separate non-inferiority and equivalence sensitivity files.

For all three models versus the consensus, the adjusted LL upper bounds exceed0.020 in both cohorts. Consequently no declared LL-grid illustration establishes model–market non-inferiority/equivalence. A C0 primary unadjusted90% interval fits a hypothetical0.020 LL budget, but the family bound does not; this is **not** a competitiveness pass. Some model–model hypothetical inclusions exist, especially fixed-four MLP8/LR, but no adopted margin or primary-cohort equality follows.

The observed C0 deficit may look small; the uncertainty still permits a materially larger deficit. That is an inconclusive practical result, not proof of an acceptable deficit. All three observed Brier guards fail as well. **Formal practical statuses: NOT_ASSESSED_NO_ADOPTED_JUSTIFIED_MARGIN; overall advancement NOT_ESTABLISHED.**

## 7. Monthly sensitivity—not a significance selector

| Cohort | Model − consensus | Monthly match-weighted95% interval | Equal-month ΔLL | Equal-month95% interval |
|---|---|---|---|---|
| primary | native C0 | [-0.007227, +0.023637] | +0.013250088 | [-0.006135, +0.035145] |
| primary | thesis LR | [-0.003661, +0.030437] | +0.019320440 | [-0.001535, +0.043408] |
| primary | thesis MLP8 | [-0.000456, +0.036090] | +0.025222956 | [+0.003938, +0.052790] |
| fixed4 | native C0 | [-0.004471, +0.025324] | +0.014959678 | [-0.002207, +0.033306] |
| fixed4 | thesis LR | [+0.000055, +0.031867] | +0.020996063 | [+0.002957, +0.042984] |
| fixed4 | thesis MLP8 | [+0.001355, +0.031942] | +0.024277549 | [+0.006898, +0.043596] |

Primary match-weighted monthly95% intervals cross zero for all three model–consensus comparisons. Equal-month MLP8 is nominally worse, but equal-month weighting asks a different question and gives small/partial months the same weight as busy months. Fixed-four LR/MLP8 also have some positive nominal monthly intervals; adjusted match-weighted monthly family bounds still all cross zero. None replaces the primary weekly/match-weighted conclusion. Full monthly results for every pair and both metrics are retained in JSON/Parquet.

## 8. Diagnostics and interpretation limits

[Descriptive profiles](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/descriptive_profiles.csv) contain500 rows, covering UTC month, BO, exact source competition labels, thesis-history support/unknown players/new organization/coplay, consensus confidence, model–market gaps and favorite agreement. [Calibration bins](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/calibration_bins.csv) contain80 rows: all ten fixed positive-class bins for each forecaster/cohort, including empty bins with undefined rates. Existing SSOT ECE uses right-closed bins $(0,.1],\ldots,(.9,1]$.

Primary BO support is365 BO1/839 BO3/263 BO5. Thesis minimum-prior-series proxies have34 zero/172 one-to-nine/526 ten-to49/735 at least50 fixtures;34 have positive thesis unknown-player counts and51 are flagged new organizations. These are **thesis-feature descriptive proxies**, not native C0 cold-start counts or proof of a new-roster cause. Exact source competition labels are not certified independent tournament editions or nationality/teamtier. Sparse groups are flagged; no outcome-picked exceptions, subgroup acceptance gates, new calibration or model fitting is introduced.

## 9. Independent evidence and execution limits

Two independently authored and actually executed CLI oracles use their own scalar math/Fraction/linear-quantile implementations, without importing the parent score/resampling helpers:

| Final verifier | Exercised evidence | Result |
|---|---|---|
| Numeric | Scalar LL/Brier/half-neutral accuracy;120000 weekly and240000 monthly paired statistics; counts, effect sizes, intervals, inherited monthly family6 preservation; strict margins and dependent/unequal-count counterexamples |PASS;472392 comparisons; max absolute error8.88178e-16|
| Source |13357 individual-table quote rows;11336 model-backed quote rows; decimal Fraction de-vig, explicit participant orientation, unchanged saved/direct-native/thesis outputs, source-day/origin/release fences, complete1749-row ledger and frozen input binding |PASS_BOUNDED_SOURCE_CONDITIONAL;701356 checks; max absolute error2.77556e-16|

Counts include repeated, nested, scalar and binding comparisons, not distinct independent matches. Native-control and thesis probability bindings have maximum error0; consensus floating reduction error is at most2.220446049250313e−16. Numerical tolerance remains2e−14; source tolerance remains1e−12. No tolerance, model, gate, cohort or source data changed to resolve verifier failures.

The source/raw endpoint-clock proof is **reused by frozen hash**, not freshly independently replayed: prior11336 eligible book endpoints reported22672 each mapping-price/mapping-clock/raw-price/raw-clock value checks. The new source oracle independently reconstructs probabilities from the pinned individual quote table; hashing that table/receipt does not authenticate actual source publication/arrival or every raw endpoint-selection rule.

Three failed source-verifier receipts are retained: (1) a generic IDnormalizer rejected null native IDs in unavailable rows; (2) the source-export grouping day was wrongly equated with the native target day, plus unknown-support Boolean handling; (3) paired missing source clocks were compared asNaT==NaT. Corrections preserve exact missingness, bind the proper saved source-export date, and retain required IDs/feature/release fences. Two parent descriptive-query/calculation incidents and initial calibration-bin metadata are also retained. They added **zero probability/model fits**. [Integration history](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/verifier_integration.json) and the failed receipts remain in the verification directory.

The immutable pre-scoring [reviewed protocol snapshot](../data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/reviewed_protocol.md) has the original Plan12Markdown hash. Final verifier guards bind40 live scientific source/code/model/receipt files plus that snapshot: the live Plan12journal is intentionally updated for execution status, not used to replace the frozen scientific contract. Eleven native artifact/receipt bytes are pinned. No weights are deserialized and no fresh native forward pass is claimed. AUC/ECE/support-profile auxiliaries and the adequacy/coverage of statistical assumptions are not all independently audited by these two oracles.

### Reproduce the delivered numerical/source checks

Run from repository root:

```bash
.venv/bin/python data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/verification/numeric_oracle.py --protocol data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/protocol.json
.venv/bin/python data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/verification/source_oracle.py --protocol data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/protocol.json
```

These commands verify saved artifacts; they do not train models, generate native forecasts, collect data, touch a database or deploy. Final receipts are [numeric](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/verification/numeric_verification.json) and [source](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/verification/source_oracle_receipt.json). No application-suite/build/lint claim is made for this analysis-only work.

## 10. Confirmation prerequisites and retained boundaries

All scored dates precede the execution freeze and have already been inspected during development. There are **zero genuine future forecasts** in this run. Conditional actual rosters and last-stored source endpoints cannot establish equal-time model/bookmaker competitiveness.

Before stronger claims, independently freeze an untouched future window and stop rule, justify/adopt the practical loss budget, authenticate model/input/roster/schedule/quote availability at one declared prestart origin, and obtain any needed source authorization. [Explicit missing-prerequisite record](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/confirmation_prerequisites.json). This is not an automatic acquisition/fitting queue or a claim that more data guarantees model success.

Existing Plan05/06/09 G0 and stopped10/10 rules remain untouched by this task. Readiness before2026-10-31T00:00Z and planned2026-11-01 start do not roll. This execution performs0model/calibration fits,0new model inference,0sports acquisition,0live activation/deployment/promotion; it changes no canonical model/manifest/serving recipe. Additional downloaded matches require a new frozen panel and cannot be relabelled untouched confirmation.

## 11. Durable outputs

| Artifact | Role |
|---|---|
| [Protocol](../data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/protocol.json) | Frozen hypothesis/identities/cohorts/origins/metrics/family/margins/41input bindings |
| [Executed source cells](../data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/calculation_cells.json) | Parent calculation provenance using existing SSOT metrics; not a competing benchmark CLI |
| [Probability panel](../data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/paired_probability_panel.parquet) | Preserved probabilities plus descriptive provenance/profile fields |
| [Fixture losses](../data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/fixture_losses.parquet) | Both losses for all four forecasters |
| [Weekly pairs](../data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/joint_weekly_pair_draws.parquet) / [monthly pairs](../data/07_model_output/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/joint_monthly_pair_draws.parquet) | All paired statistics on jointly indexed samples |
| [Coverage](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/coverage_ledger.csv) | Complete1749-row opportunity/absence inventory |
| [All pairwise results](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/pairwise_results.json) | Full-precision scores, every LL/Brier contrast, intervals/effects/sensitivities |
| [Competitiveness](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/competitiveness_margin_sensitivities.csv) / [equivalence](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/equivalence_margin_sensitivities.csv) | Separate declared-budget illustrations for every pair/metric/cohort |
| [Decision](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/decision.json) | Measured ranking, unresolved direction, unassessed practical claims and future limits |
| [Execution receipt](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/execution_receipt.json) / [sealed closure](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/sealed_verification.json) | Actual verification, delivery smoke/doc checks and immutable output digests |

## Bibliography

[1] Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation. https://doi.org/10.1198/016214506000001437

[2] Politis and White (2004), Automatic Block-Length Selection for the Dependent Bootstrap. https://doi.org/10.1081/ETC-120028836

[3] Patton, Politis and White (2009), Correction to Automatic Block-Length Selection for the Dependent Bootstrap. https://doi.org/10.1080/07474930802459016

[4] Romano and Wolf (2005), Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing. https://doi.org/10.1198/016214504000000539

[5] Lakens (2017), Equivalence Tests: A Practical Primer. https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/

[6] American Statistical Association (2016), Statement on Statistical Significance and P-Values. https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf
