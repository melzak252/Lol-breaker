# Plan 12: Model–bookmaker competitiveness protocol

**Created:** 2026-10-09  
**Status:** COMPLETED_HISTORICAL_CONDITIONAL_COMPARISON; practical competitiveness/equivalence NOT_ASSESSED; authenticated future confirmation NOT_RUN.  
**Authorization:** the user subsequently authorized the complete saved goal. The zero-fit saved-output comparison is executed below; no training, new model inference, source acquisition, deployment or prospective activation is authorized by this closure.

The [methodological research](../reports/model_competitiveness_tests_research_20261009.md) and its preserved-score arithmetic remain complete and unchanged. This document preserves the original procedure; §13 records the separately authorized historical execution without overwriting earlier sealed decisions. Stronger future confirmation remains a separate prerequisite.

## 1. Goal

Determine whether native C0 and the corrected thesis LR/MLP8 models predict League of Legends series outcomes better than, worse than, or practically as well as the collected Polish bookmaker consensus, on identical matches and at an explicitly stated information origin. Quantify the performance gap and its uncertainty, distinguish a small acceptable deficit from an inconclusive result, and identify whether any model can credibly be called competitive.

Success is an evidence-backed answer, not necessarily a positive result for our models. A valid conclusion may be that bookmakers are better, a model is better, the model is acceptably close, or the evidence is insufficient. Probability quality is the target; betting profitability and tournament-distribution quality are separate questions.

## 2. Scope and preserved evidence

### Forecasters

| Identity | Contract |
|---|---|
| C0 | Authenticated native research control already exported in the OddsPortal comparison; not research annual A0 or a substitute serving model |
| Thesis LR | Current corrected `lr_ts_standard_seed42` forecasts; no new reconstruction or retraining |
| Thesis MLP8 | Current corrected `mlp_ts_standard_seed42` forecasts; no new reconstruction or retraining |
| Polish consensus | Equal average of individually multiplicatively de-vigged available Polish-book probabilities |

The Polish pool is **Betfan, LV BET, STS, Superbet and Fuksiarz**. Require at least two distinct bookmakers for the primary consensus. Fuksiarz support is sparse; Betclic/eFortuna are absent from this collected 2026 snapshot. Do not describe the pool as every Polish bookmaker. Do not include Rainbet or silently replace a missing book with Pinnacle or another source.

### Cohorts

- **Primary — Polish2plus:** the same 1,467 saved 2026 matches for every forecaster, 2026-01-14 through 2026-09-07.
- **Sensitivity — PolishFixed4:** the same 1,285 saved matches with Betfan/LV BET/STS/Superbet all present, fixed equal weights.
- These are nested populations, not 2,752 independent matches. Compare all forecasters within each cohort; do not compare a score from one cohort against a score from the other.
- Report the source opportunity inventory and every exclusion/missing forecast separately. Missing predictions contribute neither zero loss nor invented neutral probabilities. Coverage changes are not accuracy gains.

### Input closure

Use the already sealed sources, predictions and identity alignment:

- [Original comparison protocol](../data/05_model_input/oddsportal/oddsportal_polish_consensus_20261009_152919Z/protocol.json).
- [Saved scoring cohorts](../data/07_model_output/oddsportal/oddsportal_polish_consensus_20261009_152919Z/polish_close_scoring_cohorts.parquet), SHA-256 `fa3f1deaf8f7daad1fa9253716679ba623911e7fe08ba47dccd4e05f78c494f6`.
- [Saved joint moving-block draws](../data/07_model_output/oddsportal/oddsportal_polish_mbb7_20261009_155531Z/moving_block_loss_difference_draws.parquet), SHA-256 `da72df2219882b61cd309dc5463141536f0e26cd79aebdbb2eee3eb8e19b88bd`.
- [Existing comparison and coverage receipt](../data/08_reporting/oddsportal/oddsportal_polish_consensus_20261009_152919Z/summary.json).
- [Existing seven-day resampling receipt](../data/08_reporting/oddsportal/oddsportal_polish_mbb7_20261009_155531Z/summary.json).
- [Completed literature/arithmetic verification](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/sealed_verification.json).

The ongoing download must not mutate this input closure. Additional downloaded matches require a separate versioned panel, eligibility inventory and receipt. Newly downloaded 2026 outcomes are not automatically independent confirmation of models already developed on 2026.

## 3. Information origin and admissibility

### Current retrospective scenario

The endpoint is **last stored source-prestart**, not authenticated market CLOSE. Saved model inputs use historical daily releases and realized first-map roster scenarios, not evidenced roster announcements or historical issuance.

Retain the existing participant-ID orientation, TRAIN/CAL origin checks, strictly prior feature-release days and target-already-released exclusions. Do not restore the two source-date conflicts excluded by the original protocol. Do not re-evaluate eligibility based on the outcome or whether a forecast looks good.

All scores and inference using this panel must be labelled **HISTORICAL / CONDITIONAL / DEVELOPMENT_ONLY**. Statistical tests cannot upgrade its information provenance. Proper forecast comparisons require the same forecast situations and an honest information contract. [1,2]

### Genuine equal-time confirmation

Before any separately authorized confirmation run, require timezone-aware evidence for:

```text
source/release time <= data_cutoff_at <= predicted_at < actual_target_start_at
```

The selected quote must have been available at the same declared decision origin; both outcome prices, exact participant bindings, roster/history availability, model validity and result release must be evidenced. Actual replay creation time remains distinct from a simulated historical origin. Do not backdate it.

Keep OPEN and −6h/−12h/−24h/−48h origins separate. A model may be scored at an earlier origin only if its actual input state and the quotes are admissible there. If that state cannot be reconstructed honestly, record absence. Do not pair target-day model inputs with earlier prices or manufacture intermediate quote histories.

Individual-book rankings, Pinnacle comparisons and additional horizons are separate comparison families, not extra rows in the present four-forecaster primary family. Their fixtures, coverage and family sizes must be declared before analysis. This document does not authorize their execution.

## 4. Odds transformation and metrics

For each bookmaker, transform its two decimal prices independently:

$$p_A=\frac{1/o_A}{1/o_A+1/o_B},\qquad p_B=1-p_A.$$

Then take the arithmetic mean of the eligible bookmakers' probabilities. Do not average raw odds, mix unrelated snapshot origins, count duplicate feeds as additional books, or reverse sides a second time.

Use:

1. **Primary:** unclipped binary log loss in natural-log units per series.
2. **Secondary guard:** binary Brier score `(p-y)^2`, not the two-class sum that doubles it.
3. **Diagnostics:** tie-aware accuracy, AUC, calibration profiles, confidence/tail losses, source/BO/time/roster-support profiles and coverage. These do not replace the primary decision.

For pair A/B, calculate the loss differential on every identical eligible fixture:

$$d_i=L(p_{A,i},y_i)-L(p_{B,i},y_i),\qquad\hat\Delta=\frac{1}{N}\sum_i d_i.$$

Negative means A performs better; positive means B performs better. Pair the differences before inference. Never decide from overlap/non-overlap of independent score intervals. Different proper scores can disagree, so LL remains primary even if another metric looks more favorable. [1]

Reject invalid probabilities with an explicit reason, not clipping. A valid probability 0/1 assigned to an impossible observed outcome produces infinite LL. Preserve it; if infinite differences make finite-mean inference undefined, report the relevant statistic as non-estimable rather than dropping the match or changing the scoring policy.

## 5. Dependence-aware uncertainty

### Preserved primary procedure

Reuse the saved **5,000 shared moving-block replicates**, seed `20261016`:

- Consecutive **seven calendar days**, with a uniformly sampled valid start date; not Monday–Sunday weeks, independent random dates or seven consecutive matches.
- Calendar 2026-01-14 through 2026-09-07: 237 days, 231 overlapping candidate windows.
- Sample 34 windows per replicate; retain six days from the last one for 237 total days.
- No circular wrapping. Keep empty dates and every match on each sampled date.
- Recompute the match-weighted statistic as total sampled paired loss divided by total sampled fixture count.
- Use identical sampled dates for every forecaster, pair and cohort. Obtain model–model draws by subtraction within the same saved replicate, not by independent resampling.
- Do not replace observed point scores with bootstrap means. Noncircular windows underrepresent boundary dates.
- A zero-denominator or nonfinite replicate cannot be silently discarded. Retain the failure and mark the affected interval non-estimable.

Report nominal 95% intervals. Keep the already computed monthly results as a separately labelled sensitivity; do not select whichever method yields significance. Equal-month summaries target a different weighting and must be separate from match-weighted estimates.

Seven days is a specified sensitivity, not a universal dependence guarantee. Repeated-team, patch and edition dependence may last longer. Overlapping candidate windows are not independent weeks; additional bootstrap repetitions do not increase the independent sports sample. [3,4]

### Confirmation procedure lock

Before accessing any protected confirmation outcomes, use development-only data to record a single primary dependence procedure and its diagnostics. A calendar-block bootstrap must justify its fixed/selected length; corrected block-length selection or a daily-ratio HAC analysis may be used only after the implementation and assumptions are verified. The locked procedure and declared sensitivities cannot change after scoring.

A daily HAC analysis must preserve the match-weighted ratio, not average unequal daily scores. Cluster/dyadic inference additionally requires authentic unit IDs, overlap-aware treatment and enough independent units; it is not interchangeable with treating teams/origins inside one edition as independent tournaments. No DM, HAC, cluster or new block-length computation is executed by saving this protocol. [2,3,4]

## 6. Comparison family and uncertainty levels

Among four forecasters there are six unordered pairs per cohort:

- C0 / Polish consensus.
- Thesis LR / Polish consensus.
- Thesis MLP8 / Polish consensus.
- Thesis LR / C0.
- Thesis MLP8 / C0.
- Thesis MLP8 / thesis LR.

The retrospective LL interpretation covers **all 12 pairs across the two declared cohorts**. Use two-sided Bonferroni intervals with confidence `1 - 0.05/12 = 99.583333%` for family-level claims. Nominal 95% intervals remain descriptive. This family does not adjust for all past model searches, leagues, horizons or choice of uncertainty method. [5]

The existing 5,000 draws leave relatively few observations in these extreme tails. Report that Monte Carlo precision limitation; do not imply certified tail accuracy. Changing draw count in a later execution requires a separately recorded procedure and new output namespace, not overwriting these draws.

Brier intervals and profiles are secondary diagnostics unless included in an explicitly frozen joint inferential family. A nonworse observed Brier score is an advancement guard, not proof of population Brier non-inferiority. No global family-wise guarantee across unspecified LL/Brier/profile hypotheses is claimed.

## 7. Better, competitive, equivalent or unresolved

### Margin contract

There is **no adopted practical loss margin** for this comparison. Do not invent one or choose it because the current model passes.

For historical sensitivity only, retain the previously recorded grids:

| Metric | Illustrative positive worsening budgets |
|---|---|
| LL | 0.005, 0.010, 0.020 |
| Brier | 0.001, 0.002, 0.005 |

These values do not define project acceptance. Before confirmatory competitiveness/equivalence claims, the domain decision must specify a positive LL budget, its justification and any separate Brier tolerance. If no budget is adopted, superiority/effect-size reporting remains possible, but formal practical labels are **NOT_ASSESSED**.

### Decision rules

For a fixed challenger/reference, let `[L,U]` be the applicable valid loss-difference interval and let `delta > 0` be the approved LL budget:

| Axis | Criterion | Meaning |
|---|---|---|
| Established better | `U < 0` | Challenger has lower expected loss |
| Established worse | `L > 0` | Challenger has higher expected loss |
| Direction unresolved | `L <= 0 <= U` | Difference not established; not proof of equality |
| LL non-inferiority | `U < delta` | Worsening is below the approved maximum |
| Unacceptable LL inferiority | `L > delta` | Worsening exceeds the approved maximum |
| Practical LL equivalence | `L > -delta` and `U < delta` | Both unacceptable advantage/deficit regions excluded |
| Practical status unresolved | Applicable margin criteria fail without proving the opposite | More than one practically different state remains possible |

Use the family-adjusted LL bounds for family-level decisions; the simultaneous-interval rule is deliberately conservative for non-inferiority/equivalence. Do not promote a favorable unadjusted 90% sensitivity result into a family-level pass.

For explanation only: individual alpha0.05 non-inferiority uses a valid one-sided upper bound; standard TOST uses both bounds of an appropriate 90% two-sided interval. Multiplicity across models is separate from TOST's two component tests. Bounds touching a margin do not pass strict inclusion. [6]

Keep direction and practical status as separate fields: a model can be significantly worse yet acceptably close, or significantly different yet equivalent within the approved bounds. Failure to prove non-inferiority does not prove inferiority; failure to reject equality does not prove equivalence.

**Overall advancement guard:** a claim that a model is a useful probability-quality competitor requires LL non-inferiority under the adopted budget and no observed Brier worsening against the same reference/cohort. If Brier rises, retain the LL-specific result but do not award an overall advancement pass. This observed Brier guard is a project decision rule, not an additional statistical theorem. An advancement pass does not authorize deployment or betting.

Do not interpret the fraction of ordinary bootstrap draws above/below zero as an automatic frequentist p-value or posterior probability of being best. A Model Confidence Set, SPA or Romano–Wolf analysis requires its own validated implementation/null construction and declared candidate universe; none is required or started here. [5,7]

## 8. Effect size and reporting

For every declared pair/cohort, report:

- Number of matched fixtures, calendar range, missingness and dependence units.
- Both forecasters' LL/Brier point scores, paired differences and intervals.
- Observed relative excess loss `100 * delta_loss / reference_mean_loss`; this is not a probability-point or ROI difference. Do not report a ratio interval without jointly resampling its denominator.
- Optional `exp(-delta_LL)`, explicitly labelled the geometric observed-outcome probability ratio, not a season joint likelihood or Bayes factor.
- Nominal versus family-adjusted conclusions, with the practical margin and evidence level stated.
- Separate primary/fixed-four, match-weighted/equal-month and origin-specific results; no mixing favorable pieces from different populations.
- Calibration, support, BO/competition/time/confidence/disagreement diagnostics with sparse/unknown categories retained. Diagnostic slices cannot repair a failed primary conclusion.

A broad surviving set or crossing-zero interval is an unresolved result, not evidence all forecasters are equally good. Statistical significance does not measure effect importance. [7,8]

## 9. Execution sequence and proof

The underlying historical comparison, weekly sensitivity and literature audit are **already completed**. The sequence below is for a separately authorized, new execution/version; it is not a request to repeat those runs.

1. **Freeze scope before new scoring.** Bind this Markdown, dataset/model/source/code hashes, forecaster identities, cohorts, eligibility rules, origins, metrics, family, dependence method, seeds, margins and evidence level. For confirmation also bind the untouched calendar window and fixed analysis/stop rule. Do not set or roll existing forward dates here.
2. **Build the admission ledger.** Preserve every source opportunity and explicit admission/absence reason. Validate participant identity/orientation, both-side quote clocks, label/feature releases and model origin validity before consulting forecast losses. Reuse existing mapping/source verification; do not restart completed alias/roster/refetch work.
3. **Create the paired panel.** One row per fixture/origin/cohort with IDs, label, model/consensus probabilities, book count, quote/model provenance and evidence-level flags. Check duplicates, legal probabilities, complement/orientation and saved model probabilities without fitting or substituting artifacts.
4. **Calculate scores and joint uncertainty.** Extend the existing benchmark/evaluation modules under `src/analysis/` and `scripts/run_model_benchmark.py` if implementation is needed; do not create a competing benchmark. Keep all pairs on the same resamples and keep observed scores independent of bootstrap means.
5. **Classify without rescue.** Apply the frozen interval family and approved margin rules. If margins are absent, produce effect-size/direction results and historical margin sensitivities only. Stop with an honest unresolved/failing result rather than changing models, cohorts, margins or block lengths.
6. **Independently verify.** Check scalar LL/Brier/de-vig arithmetic, same-replicate pair subtraction, match-count weighting, quantiles, strict margin boundaries and a small dependent/unequal-count oracle. Check unchanged input digests and representative future/target-input non-interference where new inputs are introduced. Preserve failures and exact verification limits; existing native replay proof may be reused by hash without claiming a fresh full replay.
7. **Publish and seal.** Produce comparison tables, coverage/profile ledgers, method/precision diagnostics, decision fields, source/clock limitations and an execution receipt. Update the research index with actual status; seal the new outputs. A negative or non-estimable result is reportable and does not authorize another attempt.

For a saved-output statistical comparison, the budget is **0 base/neural fits,0 calibration fits,0 probability-head fits and0 new model inference**. A future input adapter or forecast generation is a separately defined prerequisite, not an implicit addition to this zero-fit analysis.

## 10. Artifacts and acceptance

Use a new, nonexistent run namespace for any later execution:

| Stage | Deliverable |
|---|---|
| `data/05_model_input/` | Frozen protocol snapshot, input/code/model binding, cohort/origin/family inventory, method/margin/seed declarations |
| `data/07_model_output/` | Paired probability/loss panel and jointly indexed resample statistics, if newly computed |
| `data/08_reporting/` | Opportunity/absence ledger, scores/intervals/profiles, decisions, independent proof and sealed execution receipt |

Do not create new model artifacts in stage06 for a zero-fit comparison. Preserve existing stage01 raw bytes, old outputs, canonical manifest and models.

A completed execution must satisfy all of the following:

- Every planned forecaster/pair/cohort has either a result or an explicit absence/non-estimability reason.
- Point scores and differences use exactly the same population, orientation and probability policy.
- Complete declared families are reported, including unfavorable results.
- Dependence assumptions, effective-unit limitations, sparse tails and method sensitivities are explicit.
- Any competitiveness/equivalence label uses an adopted, justified margin and the applicable adjusted inference; otherwise it is not assessed.
- Results distinguish measured ranking, expected-loss evidence, practical status, Brier guard and historical versus authenticated information.
- Independent numerical/source-binding checks and an actual changed-interface/scenario smoke are recorded if evaluation code changes. No implementation/scoring execution is claimed from documentation checks alone.
- The report does not claim profit, tournament certification, universal bookmaker superiority or live readiness.

## 11. Confirmation and safety boundaries

All previously inspected 2026 results remain development-exposed. A genuinely unseen confirmation needs separately frozen valid model/input contracts, authentic source/roster/schedule/quote availability, approved acquisition and an untouched future window. No numeric NI/equivalence margin or new confirmation dates are approved by this document.

Keep Plan05's exhausted10/10 loop, G0 and the [Plan06 forward contracts](./06_evidence_admission_after_a0_audit.md) unchanged. The existing readiness deadline before2026-10-31T00:00Z and planned2026-11-01 start do not roll. This statistical comparison neither activates acquisition nor authorizes H/control fits, betting, model promotion, server changes or deployment.

## 12. Expected conclusion

The final answer must name the forecasters, same-match population and information origin; give the observed loss gap and uncertainty interval; distinguish established direction from unresolved direction; and state whether competitiveness/equivalence is established, unresolved or not assessed under the adopted budget. It must also report the Brier guard, coverage, dependence and information limits. No financial or deployment claim follows.

The original save-only phase is complete and preserved in the immutable reviewed snapshot. Its subsequent authorized historical execution is recorded below; genuinely unseen confirmation remains separately frozen and authorized.

## 13. Authorized historical execution — 2026-10-09

**Run:** `plan12_model_bookmaker_20261009_175136Z`, **COMPLETED_HISTORICAL_CONDITIONAL_COMPARISON**. [Complete executed report](../reports/model_bookmaker_competitiveness_execution_20261009.md), [frozen execution protocol](../data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/protocol.json), [pre-scoring reviewed snapshot](../data/05_model_input/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/reviewed_protocol.md), [decision](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/decision.json), [sealed verification](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/sealed_verification.json).

| Same 1,467 primary series | LL | Brier |
|---|---:|---:|
| Polish consensus | 0.559604853 | 0.189266950 |
| Native C0 | 0.565904908 | 0.192123748 |
| Corrected thesis LR | 0.570802361 | 0.193908558 |
| Corrected thesis MLP8 | 0.573508843 | 0.195254691 |

Model−consensus ΔLL is **+0.006300054/+0.011197508/+0.013903990**, relative excess mean loss **1.13%/2.00%/2.48%**, not probability-point error or ROI. All six primary nominal95% LL intervals and all twelve family-adjusted LL intervals across primary/fixed-four cohorts cross zero. The consensus is best on point scores and C0 closest among models; no unique best forecaster or equality is established.

Formal competitiveness and equivalence are **NOT_ASSESSED_NO_ADOPTED_JUSTIFIED_MARGIN**. No model−market family interval fits even the illustrative0.020 LL budget; each model has higher observed Brier in both cohorts, failing the frozen advancement guard. This does not prove an unacceptable population deficit. Fixed-four nominal and equal-month sensitivities do not replace the primary inference.

The original **1,749 source opportunities** retain 1,467 admissions/258 unavailable native-output contexts/22 one-book cases/two source-day exclusions. The fixed-four1,285 are nested within primary, not extra independent matches. Unknown support/source-clock fields remain missing; no neutral forecasts or timestamps are invented. Explicit native/source date aliases on four model-backed events are retained.

Actual final independent CLI verification: **472,392 numerical comparisons**, maximum absolute error **8.8818e−16**, and **701,356 source checks**, maximum **2.7756e−16**, zero final failures. Counts include repeated/nested comparisons. Three failed source-verifier attempts and descriptive preflight incidents are retained; repairs concern nullable fields/date aliases, not probabilities, cohorts, model fits, margins or tolerances. Native/thesis probability bindings are exact. Raw endpoint-clock proof is reused by hash, not freshly independently replayed; statistical assumptions/issuance are not certified.

The input closure is **40 live scientific/source/code/model/receipt pins plus the immutable original Markdown snapshot**. The live protocol/index is intentionally updated for execution status; final verifiers bind the frozen document bytes. Zero new model/calibration fits, native inference, weekly random-start draws, sports acquisition, deployment or promotion. Monthly draws regenerate the existing declared seed sequence; all pairs, both weightings, coverage and descriptive profiles are retained.

All 2026 remains development-exposed, using actual-roster/prior-day and source-last-prestart scenarios. Authenticated untouched-future confirmation is **NOT_RUN**, zero genuine future forecasts. It still requires a justified adopted budget, a frozen untouched window and authentic prestart model/input/quote evidence. No G0/10-of-10/forward-date change, live activation, financial or tournament certification follows.

## Bibliography

[1] Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation. https://doi.org/10.1198/016214506000001437

[2] Diebold (2012 author manuscript), Comparing Predictive Accuracy, Twenty Years Later. https://www.nber.org/papers/w18391

[3] Politis and White (2004), Automatic Block-Length Selection for the Dependent Bootstrap. https://doi.org/10.1081/ETC-120028836

[4] Patton, Politis and White (2009), Correction to Automatic Block-Length Selection for the Dependent Bootstrap. https://doi.org/10.1080/07474930802459016

[5] Romano and Wolf (2005), Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing. https://doi.org/10.1198/016214504000000539

[6] Lakens (2017), Equivalence Tests: A Practical Primer. https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/

[7] Hansen, Lunde and Nason (2011), The Model Confidence Set. https://doi.org/10.3982/ECTA5771

[8] American Statistical Association (2016), Statement on Statistical Significance and P-Values, official release. https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf
