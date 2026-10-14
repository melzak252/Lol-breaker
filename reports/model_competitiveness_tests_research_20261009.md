# Are C0 and the thesis models competitive with bookmakers?

**Online methodological research and preserved-score interpretation — 2026-10-09.**

Run: `model_competitiveness_research_20261009_161332Z`.
Status: **COMPLETED / HISTORICAL_METHODOLOGICAL_RESEARCH_ONLY**.

## 1. Decision and present evidence

**The collected Polish consensus has the best observed probability scores, followed by C0, thesis LR and thesis MLP8. The primary sample does not establish a statistically unique winner, model–bookmaker non-inferiority, or practical equivalence.** These are three different questions, not three interchangeable descriptions of an interval crossing zero.

The recommended comparison is **paired log-loss differences with dependence-aware uncertainty**, Brier as a secondary guard, a predeclared acceptable worsening budget for competitiveness, and a declared family of comparisons. Use an equivalence test only when practical sameness—not merely acceptable performance—is the question. A model can be competitive despite a small statistically detectable deficit; conversely, nonsignificance can coexist with an unresolved and economically unacceptable deficit. [1,3,10,11,14]

This study read 26 substantive records through four parallel literature tracks and parent checks: 22 full-text papers, author manuscripts, official guidance or source-authored documentation/code; four abstract-only records, labelled below. Three additional identified/fetched papers lacked substantive readable text and are not cited as evidence. No model or calibrator was fitted, no forecasts or new bootstrap samples were generated, and no sports data was collected.

### The existing primary comparison

All rows use **the same 1,467 2026 series**, 14 January–7 September. The consensus is the equal mean of individually multiplicatively de-vigged probabilities from at least two available Polish books: Betfan, LV BET, STS, Superbet and Fuksiarz. Fuksiarz has sparse support; Betclic and eFortuna are absent in the collected 2026 sample. This is not every Polish bookmaker.

For decimal odds, each book contributes

$$p_A=\frac{1/o_A}{1/o_A+1/o_B}.$$

Current corrected standard-TrueSkill thesis LR and MLP8 are retained, not a fresh reconstruction of a different thesis recipe. Native C0 remains a separate research reference. All differences below are **model minus consensus; positive is worse**. Log loss is in natural-log units per series; binary Brier is $(p-y)^2$, not the two-class sum that doubles it. Intervals use the preserved random-start seven-calendar-day block samples.

|Forecaster|Log loss|Brier|Excess LL versus consensus|Relative excess LL|Paired nominal 95% LL interval|
|---|---:|---:|---:|---:|---|
|Polish consensus|0.559605|0.189267|Reference|Reference|Reference|
|C0|0.565905|0.192124|+0.006300|+1.13%|[−0.007429, +0.020895]|
|Thesis LR|0.570802|0.193909|+0.011198|+2.00%|[−0.005247, +0.027307]|
|Thesis MLP8|0.573509|0.195255|+0.013904|+2.48%|[−0.002183, +0.030676]|

These percentages describe **relative mean loss**, not percentage-point probability errors, expected return, ROI or the fraction of matches won. Every model–consensus interval crosses zero. The uncertainty is large enough that “only 1–2% worse” is not a demonstrated competitiveness guarantee.

Source receipts: [original consensus comparison](../data/08_reporting/oddsportal/oddsportal_polish_consensus_20261009_152919Z/summary.json), [preserved seven-day sensitivity](../data/08_reporting/oddsportal/oddsportal_polish_mbb7_20261009_155531Z/summary.json), [new numerical examples](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_examples.json).

## 2. Define “better” before selecting a test

For event $i$, with outcome $y_i\in\{0,1\}$ and probability $p_i$:

$$L_{LL}(p_i,y_i)=-y_i\log p_i-(1-y_i)\log(1-p_i),\qquad L_B(p_i,y_i)=(p_i-y_i)^2.$$

For challenger $A$ and reference $B$:

$$d_i=L(p_{A,i},y_i)-L(p_{B,i},y_i),\qquad \hat\Delta=\frac1N\sum_i d_i.$$

The population target is expected paired loss on a **specified forecast population and information origin**. Negative $\Delta$ favors the challenger. Pairing removes irrelevant between-match difficulty variation; comparing independent error bars for two scores discards their covariance. Proper scores evaluate full probability forecasts, not only correct favorite selections. Different proper scores can rank imperfect forecasts differently; predeclare the primary score rather than selecting whichever favors the model. [1,2]

Gneiting and Raftery state: “Scores for competing forecast procedures are directly comparable if they refer to exactly the same set of forecast situations.” Here that means matching target, fixture population, participant orientation, BO interpretation, roster/feature availability and decision clock—not merely the year. [1]

### Effect size: how much better or worse?

Report the absolute paired difference and interval first. Useful additional descriptions are:

- **Relative excess mean loss:** $100\hat\Delta/\bar L_B$. This is an observed percentage with the reference score in the denominator, not a replacement proper score. An interval for the ratio would require joint resampling of that denominator; none is claimed here.
- **Geometric observed-outcome probability ratio:** $\exp(-\hat\Delta_{LL})$. By direct algebra this equals the geometric mean of $q_{A,i}/q_{B,i}$, where $q_i$ is the probability assigned to the outcome actually observed. It is not an average probability-point difference, a joint tournament likelihood, a Bayes factor or a betting return.
- **Brier difference:** interpretable on the declared score scale, but not generally a direct estimate of calibration error or squared error against the unknown true probabilities.

The geometric ratio identity uses only marginal forecast scores. Multiplying those marginal probabilities into an alleged season joint likelihood would require additional dependence and conditional-forecast assumptions. The ASA likewise distinguishes statistical significance from effect magnitude and importance. [1,14]

## 3. Superiority, competitiveness and practical sameness

Let $\delta>0$ be a justified maximum acceptable worsening in the chosen loss units.

|Question|Hypotheses for challenger minus reference|Evidence required|
|---|---|---|
|Any expected-loss difference?|$H_0:\Delta=0$ versus $\Delta\ne0$|Valid two-sided interval excludes zero; state direction|
|Challenger better?|$H_0:\Delta\ge0$ versus $\Delta<0$|Valid one-sided upper bound below zero|
|Challenger not unacceptably worse?|$H_0:\Delta\ge\delta$ versus $\Delta<\delta$|Valid one-sided upper bound below $\delta$|
|Challenger unacceptably worse?|$H_0:\Delta\le\delta$ versus $\Delta>\delta$|Valid one-sided lower bound above $\delta$|
|Practically equivalent?|$H_0:\Delta\le-\delta$ **or** $\Delta\ge\delta$, versus $-\delta<\Delta<\delta$|Both one-sided equivalence tests reject; interval strictly inside the bounds|

**Non-inferiority is the direct competitiveness question** when improvement is welcome. Equivalence additionally excludes a material advantage. A clearly better model can pass non-inferiority while failing symmetric equivalence because it is too much better; that is not a defect. Likewise a small nonzero difference can be statistically significant and practically equivalent if sufficiently precise. [10,11,13]

TOST is the two one-sided tests procedure identified in Schuirmann’s original abstract and explained in detail by Lakens. With each component at individual $\alpha=0.05$, an appropriate **90% two-sided interval** entirely inside the equivalence bounds implements the usual interval criterion. Do not split $\alpha$ again between these two component tests: they form an intersection–union decision. Multiplicity across separate model claims is a different issue. [10,12]

A one-sided $\alpha=0.05$ non-inferiority decision similarly uses the upper endpoint of an appropriate 90% two-sided interval. A stricter one-sided $\alpha=0.025$ uses a 95% two-sided interval; the FDA guidance illustrates that clinical convention. Its hypothesis logic is relevant, but its medical margins and regulatory context are **not** LoL standards. [13]

### Where does the acceptable budget come from?

It must be chosen before confirmatory results, with a domain justification: the maximum forecast-quality deficit acceptable for the intended use, weighed against genuine benefits such as independent odds-free availability or lower operating cost. There is no universal $0.005$ log-loss margin, “2% is close enough” rule, or conversion from log loss to a guaranteed betting ROI. A calibration or decision-value requirement must be separately defined and evaluated.

Do not choose a margin just larger than the observed interval. Instead report how much uncertainty remains, specify the externally justified budget, and classify the result as adequate or unresolved. Broad intervals can establish neither superiority nor competitiveness. Failure to prove non-inferiority is not proof of inferiority. Failure to reject equality is not proof of equivalence. [10,11,13,14]

Bayesian ROPE analysis is an alternative interval-hypothesis approach using a posterior and an explicitly justified practical region. The consulted Kruschke record is abstract-only. Bootstrap resampling frequencies are not a Bayesian posterior or a posterior probability that C0 is superior. No Bayesian analysis was fitted here. [25]

## 4. Dependent matches: which inference method fits?

Repeated teams, tournaments, patches and adjacent fixture days make independent-match uncertainty questionable. The inferential object should remain the **paired loss differential**, with the dependence unit and target population explicitly stated. None of the following methods authenticates missing issue times or turns reused historical development into independent confirmation.

|Method|Useful target|Main requirement or limitation here|
|---|---|---|
|DM-type mean-loss test with HAC variance|Average paired forecast-loss superiority/difference|Appropriate weak dependence, moments and consistent long-run variance; estimation schedule and nonstationarity matter|
|Calendar moving-block bootstrap|Intervals for match-weighted paired differences|Consecutive blocks, adequate length and suitable time-series assumptions; longer team/tournament dependence may remain|
|Tournament/cluster or dyadic inference|Population uncertainty when dependence follows editions/shared teams|Authentic unit IDs, sufficient independent units and overlap-aware theory; origins/teams within one edition are not separate editions|
|Giacomini–White conditional predictive ability|Whether relative quality changes with known contexts|Chosen conditioning variables and applicable limited-memory estimation framework; not an automatic test for every expanding refit|

### DM: compare forecasts, with the right variance

Diebold’s author manuscript emphasizes forecast comparison rather than an unrestricted claim about which model architecture is universally best. The mean paired loss can be tested with a heteroskedasticity/autocorrelation-consistent long-run variance. Approximate stability of the differential is more relevant than demanding that each individual loss stream be constant; shared changes can partly cancel. Conversely, a changing loss differential or estimation-induced nonstationarity needs explicit treatment. Covariance stationarity alone is not asserted sufficient for every possible dependent process: CLT/moment and variance-estimation conditions still matter. [3,4]

Our asynchronous games have unequal daily counts. A daily adaptation must preserve the event-weighted ratio, not replace it with an unweighted mean of daily scores. With daily paired loss totals $D_t$ and counts $N_t$, including empty days:

$$\hat\Delta=\frac{\sum_tD_t}{\sum_tN_t},\qquad g_t=D_t-\hat\Delta N_t.$$

A ratio-estimator HAC/delta-method formulation uses the long-run variance $\Omega_g$ of the centered daily influence series, giving approximately

$$\operatorname{SE}(\hat\Delta)=\sqrt{\frac{\Omega_g}{T\bar N^2}}.$$

This formula is an analytical adaptation, not a newly executed DM test. It assumes suitable stability and weak dependence of the daily totals/counts. Bandwidth selection must be justified for that process. Equal-month, equal-day, equal-edition and match-weighted averages answer different questions and must not be silently interchanged. [3,4]

**Package trap:** `forecast::dm.test` takes forecast residuals and a power loss. With raw binary residuals `e = y - p` and `power = 2`, it calculates exactly Brier differences. Passing already-squared Brier losses instead squares again; default squared-error mode is not log loss. The current implementation truncates covariance at `h - 1`, with `h` denoting forecast horizon. Setting `h=7` because a bootstrap block spans seven days confuses two unrelated parameters; using `h=1` also does not automatically protect against repeated-team or tournament dependence. [19,20]

### Seven-day moving blocks: sensible sensitivity, not a certification

The existing user-requested procedure resamples random-start **consecutive seven-calendar-day windows**, not Monday–Sunday weeks, isolated random dates, or seven consecutive games:

- 237 calendar days; 231 overlapping eligible start windows.
- 5,000 shared paired replicates, seed `20261016`.
- 34 sampled windows per replicate; the last retains six days, for 237 days total.
- Empty dates remain; each replicate recomputes total paired loss divided by total sampled fixture count.
- No circular year-end wrapping; boundary dates are sampled less often than interior dates. Observed scores remain the original full-cohort scores, not resample means.

Overlapping windows are **not 231 independent weeks**. Short blocks may miss longer edition/team effects; long blocks reduce effective independent resampling information. Politis–White provides automatic block-length selection under a stationary weak-dependence framework, with a subsequent correction that must accompany the original formula. Apply any such selection to the relevant differential/influence series, specify the procedure before confirmation, and inspect assumptions. It is not evidence that exactly seven days is always correct for esports. [5,6]

For a future protocol, retain monthly and declared calendar-block/HAC sensitivities and inspect temporal/regime and repeated-team structure. Do not choose the block length or method because it gives the desired significance. The current seven-day result is a requested historical sensitivity after prior results, not a newly pristine confirmatory analysis.

### Teams, editions and conditional quality

Cluster-robust precision is driven by independent clusters, not just the number of rows. Multi-way clustering allows multiple cluster dimensions under its assumptions; dyadic methods address observations sharing a member. Naively clustering separately by team-A and team-B slots can miss dependence when a team appears in either orientation. A global team dyad is not necessarily interchangeable with independent tournament editions, and authentic edition metadata is still required for edition-level inference. No universal eight- or thirty-edition guarantee follows from these papers. [7,8,9]

Giacomini–White tests a different question: relative expected loss conditional on chosen known information. It can be useful for predeclared league, cold-roster or BO regimes rather than an outcome-selected slice. Its original framework handles limited-memory/fixed/rolling estimation; it explicitly rules out ordinary expanding-window schemes, while some discounted estimators can have finite effective memory. Do not attach its label to arbitrary expanding neural retraining. [4]

## 5. Many models, many horizons and “which is best?”

The searched family includes model variants, metrics, cohorts, league slices, quote origins and uncertainty methods—not merely the three current models. Preserving that history matters; a correction for the latest shortlist does not repair all prior adaptive research. [14,15]

- **Holm:** adjusts valid individual p-values to control family-wise error; validity of each dependent-data test comes first. Romano–Wolf stepdown can exploit dependence using shared resampling, but requires a correctly constructed joint/null resampling implementation. [18]
- **White Reality Check / Hansen SPA:** benchmark-versus-many tests for a declared candidate universe. SPA studentization and its null construction address weaknesses of the Reality Check; neither authorizes dropping unsuccessful searched candidates after seeing them. Their stationary/estimation assumptions need checking, including limitations for recursively estimated forecasts. [15,16]
- **Model Confidence Set:** provides a set of surviving forecasts containing a best expected-loss member under its assumptions, not necessarily one champion. A large surviving set reflects limited information. The paper’s equal-loss testing is **not** margin-based practical equivalence: remaining in the set does not prove each member is acceptably close to the market. [17]

The MCS paper explicitly constructs null-bootstrap distributions by transforming paired losses to impose the null. An ordinary bootstrap distribution around the observed statistic is not automatically the relevant null distribution. Therefore “fraction of resamples where the model wins” is not automatically a frequentist p-value, a posterior superiority probability, or a search-adjusted probability of being best. [14,17,25]

This research adds a **retrospective illustration** covering all six LL pairs among the four forecasters in each of two cohorts: 12 contrasts. Two-sided Bonferroni intervals have 99.583333% nominal confidence, using the same saved joint draws. All 12 cross zero. This does not alter the original six-contrast family or its stored decision, and does not adjust for every earlier model experiment. With 5,000 draws the extreme family tails use relatively few draws; their Monte Carlo precision is not certified here.

### Tests that answer different questions

McNemar’s paired-proportion test targets categorical success/failure differences; DeLong targets correlated ROC AUC differences. Both consulted original records were abstract-only. Neither is a replacement test for expected log loss or Brier, and their familiar independent-case variance assumptions do not solve our time dependence. AUC measures ranking and is not generally a proper probability-forecast score; Byrne discusses the qualifications rather than a universal rejection of AUC. [21,22,23]

Demšar’s classifier recommendations concern comparisons across independent datasets. Treating games, reused folds, or overlapping seven-day windows as independent datasets would misapply that framework. A signed-rank test also does not generally target the mean loss difference for an arbitrary asymmetric differential distribution. [24]

Calibration measures remain supplementary. A constant base-rate forecast can be calibrated but uninformative; ECE depends on binning and alone cannot establish a useful bookmaker competitor. Report calibration/discrimination and proper-score performance together, with appropriate uncertainty. [1,26]

## 6. What the preserved results actually establish

### C0 versus the thesis models

Subtracting model–consensus differences within each **same** saved replicate gives the paired model–model intervals; independent marginal intervals are not subtracted.

|Primary cohort contrast|Observed ΔLL|Paired nominal 95% interval|
|---|---:|---|
|Thesis LR − C0|+0.004897|[−0.003154, +0.012516]|
|Thesis MLP8 − C0|+0.007604|[−0.000594, +0.015397]|
|Thesis MLP8 − thesis LR|+0.002706|[−0.002919, +0.008392]|

C0 has the best observed model score. **All six pairwise LL intervals among the four forecasters cross zero on the primary cohort.** That means neither a unique better forecast nor equality is established at the displayed nominal two-sided level.

### Illustrative competitiveness budgets—not acceptance gates

Margins `0.005`, `0.010`, `0.020` LL and `0.001`, `0.002`, `0.005` Brier were recorded before computing the new 90% summaries, but after the old historical comparisons were already known. They are a methodological illustration, **not a preregistered project definition of competitiveness**. The following table uses individual, unadjusted 90% interval inclusion, conditional on the existing bootstrap being adequate.

|Model versus primary consensus|LL 90% interval|NI within +0.005 LL?|NI within +0.010 LL?|NI within +0.020 LL?|
|---|---|---|---|---|
|C0|[−0.005694, +0.018225]|No|No|Illustrative inclusion only|
|Thesis LR|[−0.003046, +0.024772]|No|No|No|
|Thesis MLP8|[+0.000123, +0.027760]|No|No|No|

C0’s upper bound exceeds 0.010; accepting a deficit as large as 0.020 would be a materially different standard—about 3.57% of this consensus mean LL. No source says that standard is appropriate for our application. C0 also fails individual Brier non-inferiority interval inclusion for each illustrative budget through 0.005: its upper 90% Brier bound is 0.007441.

The “minimum needed margin” fields in the evidence are precision diagnostics only. Using them to choose a passing threshold would reverse the logic of non-inferiority. The MLP8 90% interval’s positive lower endpoint is also **not** conventional two-sided 95% significance; switching direction or confidence level after seeing the result does not create a confirmatory finding.

Symmetric equivalence requires both bounds. C0’s LL interval fits inside ±0.020 but not ±0.010; that remains an illustrative inclusion, not practical-sameness certification. On the primary cohort, LR–MLP8 does not fit inside ±0.005 LL either. [10,11]

### Fixed-four sensitivity is a different population

On 1,285 events with all four core books, the preserved consensus is Betfan/LV BET/STS/Superbet with fixed equal weights. LR and MLP8 deficits have nominal 95% intervals just above zero, respectively `[+0.000112,+0.026566]` and `[+0.000154,+0.026287]`; C0’s interval still crosses zero. All corresponding family-adjusted intervals cross zero.

MLP8–LR on this subset has a 90% interval `[−0.004283,+0.004501]`, which would satisfy an individually tested, justified ±0.005 equivalence budget. That does **not** transfer to the broader primary cohort or certify the models as equivalent to bookmakers. Fixed-four composition and match selection are distinct sensitivities; do not mix their favorable pieces into one claim.

## 7. Recommended confirmation protocol

This is a recommendation, not an automatically authorized new experiment or date change.

1. **Lock the question and population.** Match series target/BO, participant IDs and orientation, declared leagues/coverage, one fixed decision origin, and allowable model/market information. Keep OPEN, source-last and −6/12/24/48h comparisons separate; do not invent missing intermediate quotes or earlier feature snapshots.
2. **Lock the forecast recipes and candidate family.** Save actual model artifacts, full input closure and source observations before outcomes; declare which models, bookmakers, consensuses and horizons belong to each hypothesis family. Record absences rather than reducing to whichever intersection makes a model look best.
3. **Lock loss and margins.** LL primary, declared-scale Brier guard and secondary calibration profiles. Justify the competitiveness budget externally; choose one-sided α and multiplicity handling in advance. If practical sameness matters, declare symmetric/asymmetric equivalence bounds separately.
4. **Lock inference and sensitivity.** Preserve paired chronological rows; use calendar totals/counts for event weighting. Specify dependence diagnostics, primary block/bandwidth selection and alternate sensitivity rules before inspecting confirmation results. Use edition/dyadic methods only with authentic units and applicable independent-unit assumptions.
5. **Plan precision around the decision.** Use development-only pilot dependence and candidate gaps to simulate prospective interval width/power under the locked procedure. More bootstrap replicates reduce simulation noise, not uncertainty due to limited independent sports exposure. If the true deficit exceeds the acceptable margin, more evidence may confirm inferiority rather than make the model competitive.
6. **Evaluate without adaptive rescue.** Report effect sizes, intervals, all named absences and the full family. Use superiority, acceptable non-inferiority, unacceptable inferiority, practical equivalence or unresolved wording according to their actual criteria. Do not swap block length, margin, cohort or model after seeing confirmation outcomes.
7. **Keep predictive and financial claims separate.** A score-competitive model can still lack reliable price-selected EV or profitable execution. These require their own authenticated odds/issuance ledger and financial protocol; no H fitting or betting gate is opened by this research.

## 8. Availability and verification boundaries

The present records use conditional target-day historical features and realized first-map roster scenarios, not authenticated pre-match roster announcements or issued forecasts. The OddsPortal endpoint is **last stored source-prestart**, not certified market CLOSE or a synchronized fixed-hour snapshot. OPEN diagnostics are market-only; the saved target-day model inputs cannot certify earlier opening competitiveness. All 2026 model-development observations are exposed historical data. No inference method repairs these information or selection limitations. [3]

**Executed numerical verification:** an independent scalar/daily-calendar oracle replayed all 120,000 paired bootstrap values for six pairs × two metrics × two cohorts × 5,000 saved replicates. Maximum draw error was `5.20e−17`. It also checked 10,000 replicate fixture counts, 36 scalar scores including tie-aware accuracy, 24 point differences, 120 interval endpoints, 48 precision diagnostics, 144 margin predicates, 48 effect-size values and 16 mathematical boundary examples. Counts include repeated comparisons and nested populations, not that many distinct matches. Nine inherited input/protocol/receipt digests and both saved sampling files are bound in the numerical receipt.

The first oracle attempt mistakenly forced neutral probabilities to a positive-class accuracy decision rather than preserving half credit for neutral forecasts. Only that verifier line was corrected; the failed attempt, diagnosis and repair are retained. Probabilities, cohorts, original scores/draws, tolerances and frozen analysis scope did not change. A separate literature-track mistake about Brier residuals was corrected after reading the actual R source, and its correction incident is also retained.

This verifies arithmetic, pairing, quantiles and source bindings—not seven-day coverage validity, model-training provenance afresh, source publication authentication, predictive superiority, tournament laws or financial reliability. No DM/GW/SPA/MCS/Romano–Wolf/formal TOST/Bayesian test or full application suite was executed. This study makes no changes to canonical native C0/A0 artifacts, manifest, serving configuration, old decisions, G0 or forward dates.

### Reproducibility and retained evidence

- [Frozen numerical scope](../data/05_model_input/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_scope.json).
- [Independent numerical oracle](../data/05_model_input/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_oracle.py) and [executed receipt](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_verification.json).
- [Numerical examples](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_examples.json), [effect-size export](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/effect_sizes.csv) and [verifier incident](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_verifier_incident.json).
- [Literature/access/quotation ledger](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/source_records.json) and [claim support matrix](../data/08_reporting/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/claim_matrix.json). The four detailed literature-track artifacts remain alongside them.

Run from the repository root:

```bash
.venv/bin/python data/05_model_input/model_competitiveness_research/model_competitiveness_research_20261009_161332Z/numerical_oracle.py
.venv/bin/python "$HOME/.agents/skills/deep-research/scripts/verify_citations.py" reports/model_competitiveness_tests_research_20261009.md --json
```

The second command uses the local research-skill installation, not a project runtime dependency. Citation-structure checks do not independently authenticate publications or prove their assumptions apply. Exact read URLs, access depth, short quotations and methodological limits are preserved in the ledger; abstract-only entries below support only scope-level statements.

## Bibliography

[1] Strictly Proper Scoring Rules, Prediction, and Estimation — Tilmann Gneiting and Adrian E. Raftery (2007). https://doi.org/10.1198/016214506000001437

[2] Proper scoring rules for estimation and forecast evaluation — Kartik Waghmare; Johanna Ziegel (2026, arXiv v4 dated 10 May 2026). https://arxiv.org/abs/2504.01781

[3] Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold-Mariano Tests — Francis X. Diebold (2012). https://www.nber.org/papers/w18391

[4] Tests of Conditional Predictive Ability — Raffaella Giacomini; Halbert White (2006). https://doi.org/10.1111/j.1468-0262.2006.00718.x

[5] Automatic Block-Length Selection for the Dependent Bootstrap — Dimitris N. Politis; Halbert White (2004). https://doi.org/10.1081/ETC-120028836

[6] Correction to ‘Automatic Block-Length Selection for the Dependent Bootstrap’ by D. Politis and H. White — Andrew Patton; Dimitris N. Politis; Halbert White (2009). https://doi.org/10.1080/07474930802459016

[7] A Practitioner’s Guide to Cluster-Robust Inference — A. Colin Cameron; Douglas L. Miller (2015). https://doi.org/10.3368/jhr.50.2.317

[8] Robust Inference with Multi-way Clustering — A. Colin Cameron; Jonah B. Gelbach; Douglas L. Miller (2009 manuscript; 2011 journal publication). https://doi.org/10.1198/jbes.2010.07136

[9] Cluster-Robust Variance Estimation for Dyadic Data — Peter M. Aronow; Cyrus Samii; Valentina A. Assenova (2015). https://doi.org/10.1093/pan/mpv018

[10] Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses — Daniël Lakens (2017). https://doi.org/10.1177/1948550617697177

[11] Improving Your Statistical Inferences, Chapter 9: Equivalence Testing and Interval Hypotheses — Daniël Lakens (2022 online textbook; version may update). https://lakens.github.io/statistical_inferences/09-equivalencetest.html

[12] A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability — Donald J. Schuirmann (1987). https://doi.org/10.1007/BF01068419 **Abstract only.**

[13] Non-Inferiority Clinical Trials to Establish Effectiveness: Guidance for Industry — U.S. FDA CDER and CBER (2016). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials

[14] American Statistical Association Releases Statement on Statistical Significance and P-Values — American Statistical Association (2016). https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf

[15] A Reality Check for Data Snooping — Halbert White (2000). https://doi.org/10.1111/1468-0262.00152

[16] A Test for Superior Predictive Ability — Peter Reinhard Hansen (2005). https://doi.org/10.1198/073500105000000063

[17] The Model Confidence Set — Peter R. Hansen, Asger Lunde, and James M. Nason (2011). https://doi.org/10.3982/ECTA5771

[18] Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing — Joseph P. Romano and Michael Wolf (2005). https://doi.org/10.1198/016214504000000539

[19] Diebold-Mariano test for predictive accuracy — dm.test — forecast R package documentation; George Athanasopoulos and Kirill Kuroptev (current package documentation). https://pkg.robjhyndman.com/forecast/reference/dm.test.html

[20] forecast R package DM2.R implementation — Rob J Hyndman et al. (package source) (current master). https://github.com/robjhyndman/forecast/blob/master/R/DM2.R

[21] Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages — Quinn McNemar (1947). https://doi.org/10.1007/BF02295996 **Abstract only.**

[22] Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach — Elizabeth R. DeLong; David M. DeLong; Daniel L. Clarke-Pearson (1988). https://doi.org/10.2307/2531595 **Abstract only.**

[23] Empirical AUC for evaluating probabilistic forecasts — Simon Byrne (2016). https://doi.org/10.1214/16-EJS1109

[24] Statistical Comparisons of Classifiers over Multiple Data Sets — Janez Demšar (2006). https://jmlr.org/papers/v7/demsar06a.html

[25] Rejecting or accepting parameter values in Bayesian estimation — John K. Kruschke (2018). https://doi.org/10.1177/2515245918771304 **Abstract only.**

[26] Verified Uncertainty Calibration — Ananya Kumar, Percy Liang, and Tengyu Ma (2019). https://proceedings.neurips.cc/paper_files/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html
