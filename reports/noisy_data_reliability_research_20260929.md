# Noisy data and reliable A0 probabilities: evidence, limits, and next experiments

**Research date: 2026-09-29; updated 2026-10-01. Reconstruction failed; native6+12 failed the broader primary replication. A coherent family component improved its rating control, but incremental A0 integration failed. The reliability-first duration comparison also failed its primary reliability-gain gate. Keep annual A0; no promotion, deployment or betting certification.**

## 1. Decision

**Do not try to make match outcomes deterministic or make every probability less confident. Make the model sensitive to the quality of its evidence, while preserving useful confidence when the evidence supports it.**

The selected **training hypothesis** was structured, observed-statistic reconstruction pretraining for the existing history encoder, followed by the unchanged native outcome objective. This ReMasker-inspired adaptation was tested—not merely proposed—and did **not** improve assembled match probabilities over matched additional supervision. The literature motivated the mechanism, not a claim of transfer to A0.[11][12][16] The original comparison is in §10; §11 records the subsequently failed replication of its positive supervised-control signal and the completed family/integration program.

**Low historical-stat coverage and LPL membership remain almost inseparable.** The observed confidence error could reflect missing information, competition-specific model error, or both. Neither the source audit nor the reconstruction experiment establishes that missing values cause the error.

Priority order:

1. **Trace missing values and separate acquisition quality from competition/format effects.** The bounded source-to-target audit below is complete; broader source adjudication is not.
2. **Structured history-stat reconstruction and the broader native6+12 replication have now failed their fixed primary comparisons.** Preserve the results; neither recipe qualifies for promotion from these studies.
3. **Consider a small quality-conditioned probability correction only if past CAL independently supports it.** Do not infer a usable correction from the exposed 2026 association.
4. **Keep native-terminal-score last-layer Laplace as a secondary uncertainty experiment.** It addresses parameter uncertainty, not missing information or shared model bias.
5. **Do not adopt automatic high-loss filtering, blanket label smoothing, another larger seed ensemble, or a per-match “certainty” head as fixes.** These either address a different problem or lack supporting A0 evidence.

The reference remains **frozen annual Causal A0**. The separate, newly trained single-split control is a research assembly. Neither identity establishes which artifact serves the operational application.

## 2. What is actually new here

This extends—not replaces—the earlier [calibration/noise review](calibration_learning_and_noise_research_20260927.md) and the executed [data-focus experiment, §10](bookmaker_noise_and_provider_training_20260927.md#10-2026-09-29--executed-single-split-data-focus-and-history-masking).

Five independent research tracks covered labels/objectives, incomplete features, parameter uncertainty, distribution shift, and local A0 mechanics. The bibliography contains **33 unique primary sources**, including a 2025 imputation-uncertainty study and a 2026 tabular-model reliability benchmark. Evidence type matters: a theorem under stated assumptions, an empirical result on another domain, and an A0 observation are not interchangeable.

New executed work:

- Saved-forecast confidence diagnostics on **3,103 canonical 2026 matches through September 8**, including **1,668 market-linked matches**; both frozen annual A0 and the single-split uniform control.
- History-stat presence computed from the corrected bank, excluding padded history slots.
- **12 outcome-blind TRAIN series / 240 source-stat cells** checked against their aligned auxiliary targets and masks.
- **14 mathematical oracle cases**, including probability-target distortion, valid-upset filtering, bounded-loss gradients, Beta identifiability, and loss of resolution.
- Independent arithmetic checks of **116 descriptive estimates and 8 winning/losing controls**; source/input hashes and A/B reversal checks.
- **2026-10-01 follow-up:** six newly trained models plus three reused control members, all 3,103 canonical 2026 targets, matched optimizer budgets, saved-checkpoint replay and independent numeric verification. This is a separate historical model experiment, not completion of the live evidence-collection plan.

Previously reported training and bootstrap experiments were **not rerun** to confirm their negative results.

## 3. A0 is not uniformly overconfident in these saved forecasts

### Method and scope

For each non-neutral prediction, define favorite confidence \(f_i=\max(p_i,1-p_i)\), favorite-win indicator \(w_i\), and gap

\[
G=\frac{1}{n}\sum_i(f_i-w_i).
\]

Positive means that average favorite confidence exceeds the observed favorite-win rate. This is **confidence calibration**, not complete conditional calibration, a signed A/B calibration analysis, or a comparison of model log losses.

The protocol fixed confidence bins at 0.5/0.6/0.7/0.8/0.9/1, high-confidence thresholds at 0.8/0.9/0.95, and a historical-stat-presence split at 90%. Presence is the observed count of 12 statistic fields divided by 12 times the number of valid historical map tokens. Empty histories are a separate category.

An initial diagnostic exposed arbitrary favorite flips at floating-point representations of 0.5. **Protocol v2 excludes only numerically neutral predictions with \(|p-0.5|\le10^{-12}\) from favorite-based summaries**, reporting their counts. Four cold starts are neutral in each model; none are in the market subset. No probabilities, canonical scoring rows, or fitted artifacts were changed. All-cohort favorite diagnostics therefore use **3,099**, not 3,103, observations.

Intervals use the existing shared monthly-block bootstrap: **5,000 draws, seed 20260929**, match-weighted and equal-month estimands. These are **nominal, unadjusted descriptive intervals on already exposed development data**. The cohorts overlap; only nine months are available. Sparse-tail bootstrap intervals may be unstable. They are not prospective confirmation or per-match uncertainty intervals.

### Overall confidence gap

| Model and cohort | Non-neutral N | Mean favorite confidence | Favorite win rate | Gap, percentage points | 95% monthly interval, pp | Equal-month gap, pp |
|---|---:|---:|---:|---:|---:|---:|
| Annual A0, all 2026 | 3099 | 72.37% | 71.89% | +0.48 | −1.16 to +2.13 | +0.80 |
| Annual A0, market-linked | 1668 | 70.59% | 68.88% | +1.70 | −0.34 to +4.01 | +1.58 |
| Single-split control, all | 3099 | 72.68% | 72.06% | +0.63 | −1.15 to +2.65 | +0.65 |
| Single-split control, market | 1668 | 70.63% | 69.18% | +1.44 | −0.76 to +4.14 | +1.20 |

These intervals do not establish either perfect calibration or a universal need to flatten probabilities.

### High-confidence errors need their denominator

| Annual A0 cohort | Favorite threshold | N | Mean confidence | Observed upsets | Expected upsets if probabilities are correct |
|---|---:|---:|---:|---:|---:|
| All 2026 | ≥80% | 1028 | 88.78% | 120 | 115.38 |
| All 2026 | ≥90% | 414 | 94.31% | 16 | 23.58 |
| All 2026 | ≥95% | 161 | 97.26% | 1 | 4.40 |
| Market-linked | ≥80% | 475 | 87.40% | 67 | 59.86 |
| Market-linked | ≥90% | 137 | 93.41% | 8 | 9.03 |
| Market-linked | ≥95% | 34 | 96.69% | 1 | 1.13 |

Expected upsets are \(\sum_i(1-f_i)\), **conditional on the forecast probabilities being correct**. This expectation does not require independent match outcomes; a binomial significance test would require additional assumptions. The table is not such a test. Threshold sets are nested, not independent experiments. The market ≥95% slice spans eight months; the other rows span nine.

**Decision:** a few memorable confident losses do not justify suppressing all high-confidence forecasts. Here, the ≥90% and ≥95% annual-A0 groups have fewer observed upsets than their probabilities imply. This is descriptive counterevidence to blanket overconfidence, not proof that the model is correctly calibrated everywhere.

### The sharper warning is a low-coverage / LPL subgroup

| Annual A0 cohort | Historical statistic presence | N | Mean confidence | Favorite win rate | Gap, pp | 95% monthly interval, pp | Equal-month gap, pp |
|---|---|---:|---:|---:|---:|---:|---:|
| All 2026 | <90% | 232 | 70.92% | 63.36% | +7.56 | +3.02 to +10.87 | +6.59 |
| All 2026 | ≥90% | 2867 | 72.49% | 72.58% | −0.09 | −1.70 to +1.67 | +0.29 |
| Market-linked | <90% | 212 | 71.26% | 64.62% | +6.63 | +2.98 to +9.02 | +6.33 |
| Market-linked | ≥90% | 1456 | 70.49% | 69.51% | +0.99 | −1.16 to +3.54 | +0.90 |

The single-split control also has positive low-coverage gaps: **+6.40 pp** on all 232 and **+6.75 pp** on the 212 market rows. This is not independent replication: both models share substantial training information and are evaluated on the same outcomes.

**Confounding is decisive:** all 232 low-coverage matches are LPL, all are major-tier BO3/BO5, and they span all nine months. There are only **10 high-coverage LPL matches** in the full cohort, eight in the market cohort. Non-LPL matches have no low-coverage observations under this threshold. Consequently, these data do not identify a missing-stat effect separately from league, format, roster, or model-transfer effects. The 90% completeness score is not a validated measure of intrinsic match difficulty.

A **post-hoc source-tournament-label sensitivity** used the same 5,000 draws/seed and unchanged forecasts. Overall there are **143 labels** in the full cohort and **105** in the market cohort; the low-coverage subgroup has just **six labels** in either. Its annual-A0 match-weighted confidence-gap interval is **+5.49 to +12.99 pp** on all rows and **+5.47 to +9.66 pp** on market rows; equal-label gaps are **+11.03 / +8.65 pp**. All 12 model/cohort/stratum results are saved in [the sensitivity table](../data/08_reporting/noisy_data_reliability_20260929_205336/tournament_label_sensitivity.csv). These source strings can distinguish stages of the same event; they are **not certified independent editions**. Agreement of two blocking proxies does not remove the competition confound or establish prospective reliability.

![Confidence gaps by historical-stat coverage](../data/08_reporting/noisy_data_reliability_20260929_205336/reliability_diagnostics.png)

**Do not create an LPL exception or fit a correction on these 2026 results.** The later past-CAL support check in §10 also found weak competition/coverage overlap. If a competition effect remains, use the existing controlled player-only versus competition-family rating research contract, including its BO-likelihood repair—not a team/league-name residual selected after observing losses.

## 4. The raw-stat audit: missing is not the same as corrupted

Sampling was fixed before inspecting the selected outcomes: the first three chronological major-tier TRAIN series in each 2024/2025-H1 × present/missing CSD@15/XPD@15 group. Event and release dates must precede 2025-07-01.

The selected series were three Demacia Cup series and three LEC series in 2024, then three LPL and three LCK Cup series in 2025. For each, the audit aligned **bank index, match ID, first-map ID, team, player identity, roster slot, and statistic index**. CSD@15/XPD@15 are indices **8/9**, not the final two fields.

| Check | Result |
|---|---:|
| Series / individual statistic cells | 12 / 240 |
| Keys present with a null value in the immutable raw source | 120 |
| Finite numeric values in the raw source | 120 |
| Absent keys / string-valued cells / alternative normalized keys | 0 / 0 / 0 |
| Auxiliary-mask mismatches / observed-value mismatches | 0 / 0 |

For these cells, preprocessing did **not** discard available numeric CSD/XPD values: the missing cells were already null in `matches.json`. Numeric values match the float32 target storage exactly. The source checksum matches the pinned original.

**Limits:** this checks first-map auxiliary targets, not every historical-token value or the truth of every recorded match result. The history builder uses the same finite-numeric presence rule, but that static fact is not a full history-value audit. Nulls could still originate in upstream collection; original HTML/acquisition payloads were not adjudicated. This audit does not establish MCAR, MAR, or MNAR.

## 5. Which kind of “noise” needs which remedy?

| Mechanism | What it means for A0 | Appropriate response |
|---|---|---|
| Wrong record or mapping | Wrong winner, reversed side, player identity, units, duplicated event | Verify against independent source evidence; repair provenance and regenerate affected inputs |
| Missing measurements | Unknown CSD/XPD is not a measured zero | Retain masks; model missingness/support; test structured reconstruction without inventing observed facts |
| Finite, volatile performance | A small history of kills/damage/lane statistics varies even when correctly recorded | Regularize representation learning and estimate uncertainty; do not label difficult games corrupt |
| Unobserved pre-match information | Future draft/side, uncertain roster, unobserved condition | Integrate over causally available scenarios when a credible scenario model exists |
| Parameter uncertainty | Several parameter settings fit finite training data | Approximate posterior/ensemble diagnostics, with independent validation of their meaning |
| Shared bias / distribution shift | Every seed uses the same missing signal or wrong transfer assumption | Change or verify the information/model assumptions; extra seeds alone do not identify the bias |
| Genuine outcome randomness | An 80% favorite still loses 20% of comparable cases | Preserve valid upsets; evaluate probabilities with proper scores |

The target is a good estimate of \(P(Y\mid I_t)\), where \(I_t\) is information genuinely available at the forecast origin—not certainty about the one outcome that later happened. Proper scoring rules reward the correct conditional distribution, not hindsight agreement.[1]

## 6. Labels and losses: important distinctions

### Do not clean away legitimate upsets

Confident Learning estimates label-error structure under assumptions about a latent clean label and class-conditional corruption.[2] Ghosh et al.'s robustness results also have explicit loss/noise assumptions; some results require zero clean risk.[3] These are not automatic descriptions of sports outcomes.

Co-teaching exchanges small-loss examples between networks; generalized cross-entropy attenuates the influence of low-probability observed labels.[4][5] Both can suppress an erroneous record **or a correctly recorded upset**. A high loss is a useful review-queue signal, not an adjudication.

An executed population oracle illustrates the problem: for a genuine 80% favorite, a log-loss cutoff of 1 retains all favorite wins and removes all 20% upsets. The retained target rate becomes 100%. Moving the probability from 0.8 to 0.99 increases true expected log loss from **0.5004 to 0.9291**. Removing “noise” has made the model less reliable.

If actual labels flip with probability \(r\), observed-label probability is \(r+(1-2r)\eta\), not the clean probability \(\eta\). Changing to another proper loss alone does not identify or invert an unknown corruption process.

### Label smoothing and confidence penalties are regularizers, not truth recovery

With uniform binary smoothing of strength \(\epsilon\), the population cross-entropy target is

\[
p^*=(1-\epsilon)\eta+\epsilon/2.
\]

Thus smoothing systematically changes the target. It can still improve finite-sample generalization by reducing overfitting; the empirical benefits in neural classification do not contradict the target distortion.[6] Entropy/confidence penalties similarly modify the optimization objective.[7] Neither should be described as discovering which LoL labels are false.

### Focal loss: empirical calibration benefit does not imply probability propriety

Focal loss was designed around hard-example emphasis in imbalanced detection.[8] Mukhoti et al. report calibration gains on their vision/NLP benchmarks.[9] Charoenphakdee et al. prove that for positive focusing strength the raw focal loss is not strictly proper for posterior probabilities, while providing a corrective transformation.[10] Even the empirical paper notes this propriety limitation.

These findings are compatible: a biased objective can improve an overfit finite model. They do **not** establish focal loss as either a sports-label denoiser or a universal calibration improvement. A0's stochastic probability target makes a blind replacement particularly hard to justify.

### Brier is a defensible comparator, not a guaranteed cure

Log loss and Brier are proper for the distribution they score.[1] Brier is bounded. For a binary sigmoid logit \(z\):

\[
\partial_z L_{\log}=p-y,\qquad
\partial_z L_{\mathrm{Brier}}=2(p-y)p(1-p).
\]

The oracle checked these derivatives numerically. Binary log loss is unbounded as a reported score, but its logit gradient is bounded; Brier can produce **very small gradients on confidently wrong predictions**. These binary formulas are not a claim about the full native Beta-scoreline gradient.

A0 already uses **terminal stopped-series Beta NLL plus 0.1 × masked SmoothL1/Huber auxiliary loss**, and already masks W20/organization inputs. Recommending Huber or a missing-value mask as if absent would repeat existing functionality.

A later isolated objective study could compare native NLL with **categorical Brier over the legal terminal scorelines**, retaining the same architecture and auxiliary objective. Replacing it only with winner Brier discards scoreline information and changes the comparison. Boundedness neither corrects corrupted labels nor proves better forecasts.

## 7. The best-matched new representation hypothesis

### Structured observed-stat reconstruction, not whole-map dropout

GRU-D demonstrates the value—and context dependence—of missingness and elapsed-time modeling in clinical time series.[11] A0 already has presence bits and history ages; adding those names again is not new information. VIME supplies tabular mask/value pretext tasks.[12] ReMasker reconstructs synthetically hidden values that were originally observed, explicitly separating them from native missing values.[16]

**[INFERENCE] A0 adaptation:** use a temporary decoder to teach the existing history encoder to reconstruct selected observed statistic fields or field/time blocks. Then fine-tune with the unchanged supervised outcome head and native causal histories. This is representation pretraining, **not** treating imputed statistics as newly observed facts.

Maintain two masks: native presence \(M\) and artificial hiding \(C\). Encoder visibility is \(M(1-C)\); reconstruction targets are selected by \(MC\). Never train reconstruction against a synthetic zero at a genuinely absent cell. Keep map identity, player, role, age, opponent information, and causal order intact. The native presence and artificial corruption masks must remain separately auditable, even if the encoder consumes their combined visibility.

This differs materially from the failed 20% history-token dropout: that experiment removed entire map tokens and their information, with no observed-field reconstruction task. Its negative result is relevant, but does not test this mechanism.

**Do not force an unchanged win probability after deleting informative evidence.** A correct predictor may become less certain when information is removed. Similarly, SCARF's independent marginal feature replacements can create implausible player/map combinations; its tabular classification results do not establish that such augmentations preserve A0 semantics.[13]

The main risk is that reconstruction learns predictable but outcome-irrelevant correlations. ReMasker's own scope is imputation; downstream uncertainty is not automatically optimized.[16] A 2025 study across five numerical datasets with synthetic missingness also finds a disconnect between reconstruction accuracy and uncertainty calibration.[32] Its interval-coverage ECE concerns **imputed values**, not match-win probabilities. Both findings argue for scoring the final A0 forecast rather than celebrating a lower pretext loss.

### Missing-input integration is conceptually right, but assumptions matter

If a credible conditional model exists, prediction can average over missing features or possible announced rosters:

\[
P(Y\mid X_{\mathrm{obs}},M)
=\int P(Y\mid X_{\mathrm{obs}},X_{\mathrm{mis}},M)
\,dP(X_{\mathrm{mis}}\mid X_{\mathrm{obs}},M).
\]

MIWAE provides a probabilistic incomplete-data route under MAR; not-MIWAE explicitly models the observation mechanism for MNAR.[14][15] These assumptions cannot be inferred from an LPL association. Unknown observation mechanisms can be non-identifiable without domain constraints.

For A0, average **complete forecast distributions/probabilities**, not input means or logits. Plausible histories must preserve player/map/role constraints. We checked complementarity for a toy scenario mixture; **no imputer or roster-scenario model was fitted**. Full probabilistic imputation is lower priority than the bounded pretraining comparison because a wrong imputation distribution adds another shared bias.

## 8. What uncertainty methods can and cannot solve

Deep ensembles have useful empirical evidence, including under distribution shift, but no guarantee that member agreement means correctness.[17][18] Calibration and averaging also interact; the pooled predictor needs its own calibration assessment.[19] Locally, more seeds/bootstrap members and the tested input perturbations have not established an A0 reliability gain. Do not repeat them as untested solutions.

For an ensemble mixture, the identity

\[
\bar p(1-\bar p)
=E_m[p_m(1-p_m)]+\operatorname{Var}_m(p_m)
\]

holds by construction. It does **not** convert member spread into an observed per-match forecast error or a certified posterior interval.

**[INFERENCE] Secondary candidate: last-layer Laplace.** Freeze the history representation and non-neural experts; approximate uncertainty in a small output head under a stated prior and the **native terminal-score likelihood**.[21] Re-optimize the selected head under that coherent likelihood/prior if necessary; curvature of the composite auxiliary training objective is not automatically a Bayesian posterior. Pool terminal distributions, respect side symmetry and BO semantics, and calibrate the pooled predictor on separate causal CAL rather than blindly reusing a MAP calibrator. The generic Bernoulli/classification library path is not a drop-in replacement for the native likelihood.

This may expose parameter directions that seed dispersion misses, but it cannot repair the fixed encoder, missing inputs, wrong competition assumptions, or shared expert bias. SWAG requires a suitable optimization trajectory; SNGP adds distance awareness but requires a defensible LoL support representation.[20][22] Neither is an immediate, evidence-backed A0 upgrade.

**Do not add an unconstrained “certainty” head.** For a single Bernoulli likelihood, Beta distributions with the same mean and different concentrations yield the same marginal probability of one outcome. Our oracle used mean 0.8 and concentrations 2/10/100: the Bernoulli marginal is unchanged while latent variance differs. Evidential-learning critiques give broader reasons not to equate a learned concentration with calibrated epistemic uncertainty.[23][24]

**Important exception:** A0's existing Beta model scores terminal series scorelines, not just one Bernoulli winner. The same oracle showed that BO3 scoreline probabilities change with concentration while summing to one. The single-label example therefore does **not** invalidate A0's existing within-series overdispersion model. Identifiability of outcome dispersion is also not proof of epistemic calibration.

## 9. Calibration, shift, and abstention

Importance weighting targets covariate shift only when target support is covered and the conditional outcome mechanism remains stable; large/estimated ratios increase variance.[25] It cannot manufacture evidence for new regimes. The already failed market-tier weighting is not a reason to tune another weighting recipe on 2026.

Group DRO is a worst-group objective, not a calibration layer. Its empirical success can depend strongly on regularization and adequate group support.[26] Near-confounding between completeness and LPL makes a new worst-group objective particularly hard to interpret.

A **conditional** small correction worth considering is

\[
p'=\sigma\{\lambda(q)\operatorname{logit}(p)\},
\quad 0<\lambda(q)\le1,
\]

regularized toward the identity \(\lambda=1\), with an A/B-symmetric, genuinely pre-match quality measure \(q\). This is a proposed hypothesis, **not fitted here**. Past CAL must first show stable excessive extremity with adequate support; a mean confidence gap alone does not establish the required slope/shape. Calibration diagnostics should distinguish intercept, slope, flexible shape, and discrimination.[27] A free A/B intercept would violate side complementarity. A quality-varying slope can reorder pairs across quality groups, so ranking and resolution must also be checked.

A deliberately uninformative constant predictor can be calibrated. In the executed oracle with equally common true probabilities 0.2 and 0.8, both truthful and constant-0.5 forecasts are calibrated, but truthful forecasts achieve LL **0.5004** and Brier **0.16**, versus **0.6931 / 0.25** for the constant. Lower ECE or lower variance alone is not success.

Selective prediction is a separate operational contract: flag/abstain when support is inadequate and measure risk **and coverage**.[28] Never hide rejected matches from the full-cohort score. Standard selective/conformal guarantees need their sampling assumptions. Weighted conformal handles specified covariate shifts; structured conditional-conformal methods support specified classes of groups/shifts; time-series alternatives have their own dependence assumptions.[29][30][31] None is a distribution-free certificate that one LoL fixture's true win probability lies in a narrow interval.

A recent 112-dataset tabular benchmark reinforces the need to separate objectives: the studied foundation models had higher AUC but weaker size-stratified conformal coverage than GBDTs.[33] This is a result about those models, small/medium datasets, and LAC prediction sets—not a universal accuracy–calibration trade-off, a log-loss comparison, or evidence to replace A0 with a foundation model.

## 10. Executed matched reconstruction experiment — 2026-10-01

### Past-CAL support and the unresolved confound

The original proposal requested past-CAL support diagnostics. The completed descriptive check covers **1,328** CAL rows: **1,221** have ≥90% historical-stat coverage, **106** have lower coverage, and **one** has no history. Of the 106 low-coverage rows, **105 are LPL and one is Worlds**; their formats are **85 BO3 / 21 BO5**. Only **two of 107 LPL CAL rows** have high coverage. The competition/coverage overlap is therefore weak even before 2026. These diagnostics were executed **after the fixed experiment**, not used to select its rate, architecture or budget. C0 was calibrated on this same CAL, so its CAL scores are in-sample diagnostics, not validation. No source acquisition or ad hoc quality calibration was added. See [CAL support](../data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/cal_support.json).

### Selected training comparison: structured history-stat pretraining

**Hypothesis [INFERENCE]:** learning useful representations from partially observed historical statistics improves assembled A0 proper scores more than spending the same extra updates on ordinary supervision.

The following design was frozen before fitting:

| Arm | Training change | Purpose |
|---|---|---|
| C0 | Existing 12-epoch native uniform control | Reuse the saved reference; do not rerun merely to reproduce known results |
| C1 | Six native epochs, then twelve native epochs after optimizer reset | Matched 18-epoch-equivalent control |
| C2 | 6 epoch-equivalents of structured reconstruction, then 12 of native supervision | New candidate; no winner relabeling or filtering |

- Same three paired seeds **20260910/20260911/20260912**, broad TRAIN **36,205** with event/release before **2025-07-01**, CAL **1,328** in 2025 H2 with release before 2026, and all **3,103** exposed 2026 rows for development scoring.
- For a first fixed mechanism, select **25% of eligible player histories** and hide their originally observed **CSD@15/XPD@15** cells across valid historical tokens. Reconstruct only those hidden observed cells with a TRAIN-scaled masked loss. The 25% choice is an engineering hypothesis, not optimized or established by literature; do not tune it after seeing 2026.
- Preserve roles, identity, age, opponent context, other observed fields, native missingness records, and causal releases. Use an isolated corruption RNG and A/B-consistent masks. Do not pretrain on 2026 “unlabeled” histories: they contain future performance/results relative to the fitting origin.
- Use the same encoder family; discard the temporary reconstruction decoder for outcome inference. Preserve native terminal-score NLL plus 0.1 masked auxiliary SmoothL1 during supervised training. Shared non-neural experts and input normalizers remain the same.
- Match actual encoder optimizer-update counts between C1 and C2 and report runtime; equal steps are not equal FLOPs. Keep the calibration/assembly recipe identical across arms. Any mixture/calibration parameters are fitted only on CAL. Do not use CAL to select many pretext variants and then claim it remains untouched calibration data.
- Decoder, scaling, RNG, optimizer schedule and stopping were frozen in the [protocol](../data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json). No architecture search, early stopping, rate tuning or walk-forward fitting loop was run.

**Primary contrast:** C2−C1 assembled-A0 winner log loss on the complete canonical development cohort. Require a negative paired monthly interval and non-worse Brier point estimate to promote the candidate to a **future confirmation test**, not to production. Report C2−C0, raw/calibrated neural and full-assembly results, scoreline NLL, market-linked scores, coverage, ranking/resolution, and all fixed confidence/quality/BO/competition slices. Use at least 5,000 paired monthly resamples and report both weighting estimands, actual block counts, and tournament-group sensitivity without calling unaudited name groups independent editions.

**Reject or stop** if reconstruction improves without a downstream probability gain, gains are solely extra-optimization effects, cold starts/side symmetry/release causality fail, or apparent benefits depend on a post-hoc LPL exception. Follow any historical success with a frozen, genuinely unexposed future confirmation cohort. All of 2026 used here remains development data; repeated chronological splits do not restore independence.

The low-dimensional calibration hypothesis and native-head Laplace are **separate alternatives**, not extra components to bundle into C2. Combining all of them would destroy attribution.

The user explicitly promoted **IDEA-026** into execution on 2026-10-01. Run **`a0_history_reconstruction_20261001_010503`** is complete. Its [catalog entry](../docs/future_ideas.md#idea-026--structured-observed-stat-reconstruction-for-reliable-a0-forecasts) records a completed experiment with a rejected reconstruction recipe, not an adopted model.

### Implementation and actual budget

- Existing runner: `scripts/train_a0_data_focus.py`; new pretext implementation: `src/models/a0_history_reconstruction.py`. The existing evaluator was extended; shared benchmark metric/bootstrap functions were reused unchanged.
- History value channels **8/9** and presence channels **20/21** represent CSD@15/XPD@15. Native observation and artificial hiding remain separate. Both values and visibility flags are hidden; measured zeros remain targets, natural absence and padding do not.
- Temporary **Linear(64,2)** decoder: token embedding32 plus the unchanged native pooled-player representation32. Cell-mean SmoothL1 uses the **shared, full-TRAIN native history scaler**, including its ±10 clipping. Input scaling is distinct from the **unclipped probability scoring** policy.
- Pretext updates only `history_encoder`, attention, empty-history token and player encoder. Outcome/auxiliary heads and dependence parameter stay unchanged during pretext. The decoder is discarded before outcome inference; missing values are not persisted as observed facts.
- A/B share each canonical **player-slot** corruption draw; this is not a claim that opposing slots have the same role. Role, age, opponent context and the remaining observed fields are retained.
- **34,358 / 36,205 TRAIN rows** can supply observed reconstruction targets. Pretext cycles those eligible rows chronologically for exactly **1,698 nonempty updates**, rather than counting zero-target batches as learning. The final twelve native epochs add **3,396 updates**. C1 and C2 each receive **5,094 updates per seed**; equal updates do not imply equal FLOPs or label exposure.
- Both new arms reset AdamW and the native gate RNG at the phase boundary, retaining learned parameters. The final twelve native epochs have identical gate-mask streams. C0 and non-neural experts are reused from the pinned September29 run, with **zero new updates**.
- Actual selected-history fractions: **24.948% / 24.974% / 25.038%**. Pretext loss fell from **0.253263 / 0.247673 / 0.249999** to **0.208552 / 0.209079 / 0.209137**.
- Recorded fitting times across three seeds: C1 **304.99s**, C2 **393.67s**, CPU with two Torch threads. End-to-end training command: **730.28s**. Runtime is observed, not a transferable performance benchmark.

### Primary result: reconstruction did not help

All rows below are the **same 3,103 canonical 2026 targets**. The earlier 2024–2025 benchmark targets are not silently scored with models that trained on them.

| Model | Assembled winner LL | Brier | LL ≥2.5 |
|---|---:|---:|---:|
| Frozen annual A0 | 0.536757 | 0.181573 | 10 |
| C0: reused native12 | 0.535449 | 0.181034 | 12 |
| C1: native6 + native12 | **0.534535** | **0.180687** | 11 |
| C2: reconstruction6 + native12 | 0.535947 | 0.181261 | 11 |

**Prespecified C2−C1:** ΔLL **+0.001411**, monthly95% interval **[+0.000106,+0.002930]**; ΔBrier **+0.000574**, interval **[+0.000033,+0.001248]**. Both point scores worsened. The equal-month ΔLL is **+0.002269**, interval **[+0.000589,+0.003911]**. There are **nine months**, 5,000 paired resamples, seed20261001.

Tournament-name sensitivity uses **143 source-label proxies**, not independently validated editions: match-weighted ΔLL interval **[−0.000228,+0.003087]** crosses zero. This weakens a universal worsening claim; it does not establish the required reconstruction improvement. The formal result is **`PRESPECIFIED_PRIMARY_SUCCESS_NOT_ESTABLISHED`**.

C2−C0 is **+0.000498 [−0.000353,+0.001386]**. C2−frozenA0 is **−0.000810 [−0.002732,+0.000812]**. Neither interval establishes an improvement over its reference.

### Positive secondary finding: additional native supervision

C1−C0 is **−0.000913 [−0.001786,−0.000284]**, with ΔBrier **−0.000347**. C1−frozenA0 is **−0.002221 [−0.004252,−0.000618]**, with ΔBrier **−0.000886**. This is a **historical-development signal for the fixed native6+12 ensemble**, not proof of a new architecture or bookmaker-level accuracy. C0 is the cleaner recipe-matched reference; the frozen annual A0 has a different training/calibration schedule.

On the same **1,668 strict-prior-day T−1h market-linked rows**, LL is C0 **0.575423**, C1 **0.574484**, C2 **0.576243**, frozenA0 **0.576033**, consensus **0.568604**. C1−C0 interval **[−0.002370,+0.000326]** crosses zero; C2−C1 is **+0.001759 [−0.000599,+0.004073]**. Market-linked improvement is not established. FIRST407 and outside-T−1h1435 remain separate, complete cohorts; FIRST is not certified archival OPEN.

### Components, quality and scoreline checks

| Component, full context | C0 LL | C1 LL | C2 LL |
|---|---:|---:|---:|
| Raw neural ensemble winner probability | 0.536040 | 0.535360 | 0.536610 |
| Calibrated neural winner probability | 0.535863 | 0.535068 | 0.536484 |
| Raw neural terminal-scoreline NLL | 1.068707 | 1.067520 | 1.069312 |
| Calibrated neural terminal-scoreline NLL | 1.072205 | 1.071160 | 1.072759 |

Scoreline rows concern the **native neural distribution**, not an invented assembled-model joint distribution. C2 loses to C1 in all three paired raw members. C1's gain over C0 is an **ensemble** result; not every longer-trained individual seed improves.

- Low12-stat coverage: **232 targets**, still all LPL. LL C0 **0.615879**, C1 **0.613361**, C2 **0.617578**. Reconstruction did not fix the subgroup that motivated it.
- High coverage: **2,867 targets**. LL C0 **0.528720**, C1 **0.527936**, C2 **0.529122**.
- Unknown history: **four targets**, all retained in headline scores; each model gives essentially0.5 and LL0.693147.
- Assembled AUC C0/C1/C2: **0.799299 / 0.799957 / 0.798841**; calibration slopes **0.973401 / 0.990480 / 0.965766**. These secondary estimates are not universal production thresholds.
- Full/no-W20/no-organization, month, BO, competition, experience, organization, quality and fixed-A0-confidence profiles are retained in [metrics.csv](../data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/metrics.csv) and [profiles.csv](../data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/profiles.csv). Own-confidence profiles use each model's bins and are not paired-population contrasts. Per-seed and scoreline outputs remain separate.

### Verification, evidence and decision

- **45 targeted tests passed**; one existing Starlette/httpx deprecation warning. No full application-suite claim.
- Real smoke and saved-model probes cover all three modes and BO1/3/5, cold starts, empty histories, query nonmutation and side symmetry. Maximum full-run probe complement error **3.33e−8**.
- Independent replay checked **83,781 probabilities**, maximum error **1.45e−8**. Direct arithmetic reproduced **122 metric rows** and the primary monthly interval; **55 checksum comparisons** passed. Independent stopped-score product probabilities agreed to **6.67e−15** in NLL.
- A **one-update real-TRAIN pretext probe** remained bit-identical when post-origin features changed. It verified unchanged outcome/auxiliary heads and eta, and decoder disposal. This is not a claim to have retrained the full experiment under every possible future perturbation.
- [Protocol](../data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json), [30-file source snapshot](../data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/source_snapshot.zip), [evaluation](../data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/evaluation.json), [independent proof](../data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/independent_verification.json).

**Decision:** reject this fixed reconstruction recipe; preserve the negative result rather than retuning its mask rate on 2026. Retain native6+12 as a **research candidate** for a separately frozen confirmation, not as an adopted replacement. The canonical manifest, annual A0 and operational model are unchanged.

All2026 observations are repeatedly exposed historical development. Actual target rosters and strict-prior-day releases are explicit scenarios, not publication/issuance proof. CAL fits both calibration and mixture. There is no prospective, tournament-edition or executable-betting certification. The separate forward-evidence collection plan is **not a prerequisite for historical sports-model experiments**.

Executed from the repository root:

```bash
.venv/bin/python scripts/train_a0_data_focus.py --protocol data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json --smoke
.venv/bin/python scripts/train_a0_data_focus.py --protocol data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json
.venv/bin/python scripts/train_a0_data_focus.py --protocol data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json --evaluate-only
.venv/bin/python data/08_reporting/a0_data_focus/a0_history_reconstruction_20261001_010503/independent_verification.py
.venv/bin/python -m pytest -q betting_app/tests/test_a0_history_reconstruction.py betting_app/tests/test_a0_data_focus.py betting_app/tests/test_a0_target_objectives.py betting_app/tests/test_a0_native_refit.py
```

These commands document the executed run. Completed training, evaluation and proof artifacts are immutable; do not rerun over the same destinations or alter hashes to bypass their guards.

## 11. Executed model-reliability program — 2026-10-01

The user separately authorized [Plan07](../TODO/07_model_prediction_reliability.md): repaired prediction inputs, annual replication of native6+12, one coherent competition-family experiment, reliability gates and an evidence-backed forward decision. Run **`model_reliability_20261001_055321`** preserves the canonical manifest, frozen annual A0 and operational model. It is not a continuation of the exhausted selected-bet EV loop or a deployment.

### Repaired inputs and real recording proof

`scripts/predict_causal_a0.py` now builds declared-fixture snapshots through the existing roster/release replay and performs native A0 inference. It preserves exact identity, A/B orientation, BO, context and source dates; rejects ambiguous/conflicting metadata and invalid cutoffs; strips odds from sports-model inputs; and retains unavailable forecasts. Release-day input is an override map, not an invented complete publication ledger.

`SUPPORTED_INPUT`, `LIMITED_EVIDENCE` and `UNAVAILABLE` describe input support. **`predictive_reliability=NOT_ASSESSED` remains explicit.** History tokens do not prove observed statistics; nonzero raw features with missing source dates cannot pass by supplying an empty history mask. Daily release and declared roster scenarios are not authenticated publication or announcement evidence.

- Nine historical requests produced **four forecasts and five explicit roster-unavailable results**. Available requests cover BO1/3/5; no roster was invented to increase coverage.
- **Twelve real native envelopes**, four fixtures × three context modes, were committed on an isolated PostgreSQL/Timescale database migrated from empty through `f1e2d3c4b5a6`. Readback, retry idempotency, caller rollback and direct-SQL UPDATE/DELETE rejection passed.
- Rehydrated tensors reproduced the forecasts; empty-history inference remained finite and maximum A/B complement error was **1.33e−8**.
- A separate real-source probe used **2,235 series**, removed **76 unavailable under the declared daily cutoff**, and changed the future target result/actual roster. Earlier tensors and probability remained exactly equal.
- Feature snapshots must be built **before** native model namespace activation in a fresh process. A first combined proof encountered unsupported feature-backend reentry after inference; its already committed envelopes were verified without repeating writes in a separate continuation. This boundary is documented rather than concealed.
- A final run of the **actual unmodified CLI over the full frozen source** reproduced all nine request identities, evidence statuses and missing reasons; its four available probabilities matched the earlier smoke **exactly**. Runtime was **275.75s**. A240s attempt timed out; a separate periodically traced diagnostic attempt exited139 while emitting its second stack dump, with cause undetermined. The ordinary CLI subsequently completed without model/replay changes. Both failed attempts are retained and the temporary tracing wrapper was removed. This is an offline batch proof, not a live-latency guarantee. [CLI receipt](../data/08_reporting/model_reliability/model_reliability_20261001_055321/cli_smoke_verification.json).

Receipts: [combined proof](../data/08_reporting/model_reliability/model_reliability_20261001_055321/input_recording_proof.json), [recording continuation](../data/08_reporting/model_reliability/model_reliability_20261001_055321/input_recording_continuation.json), [future perturbation](../data/08_reporting/model_reliability/model_reliability_20261001_055321/future_input_verification.json). These establish software behavior on historical scenarios, **not genuine future issuance**.

### Annual replication: the native6+12 gain did not generalize to the primary cohort

The [neural protocol](../data/05_model_input/model_reliability/model_reliability_20261001_055321/neural_protocol.json) fixed origins January1 of2024/2025/2026, three paired seeds20260910/11/12, batch128 and the original native objective/calibration. TRAIN event and release precede July1 of the previous year; CAL events are in that year's second half and released before the origin. No early stopping, epoch search or assessment fitting was added.

| Origin | TRAIN | CAL | Canonical assessment | Execution |
|---|---:|---:|---:|---|
| 2024 | 27,705 | 1,379 | 4,642 | New matched native12 and native6+12 ensembles |
| 2025 | 32,625 | 1,103 | 3,805 | New matched native12 and native6+12 ensembles |
| 2026 | 36,205 | 1,328 | 3,103 | Reused pinned §10 artifacts; zero new updates |

All **207,900 exported component/context probabilities** were replayed from saved checkpoints, with maximum absolute discrepancy below **3e−8**. Each origin also passed side symmetry, cold/empty-history and nonmutation probes. The comparisons retain raw-neural, calibrated-neural and full-assembly probabilities in full/no-W20/no-organization modes, plus individual-seed and native-scoreline diagnostics.

**Primary replication, 8,447 rows in2024–2025:** C1−C0 ΔLL **+0.000154**, monthly95% interval **[−0.000442,+0.000702]**; ΔBrier **+0.000084**, interval **[−0.000170,+0.000318]**. There are **24 monthly blocks**,5,000 draws, seed20261001. Equal-month ΔLL is **−0.000490 [−0.002788,+0.001413]**. Neither estimand establishes improvement.

The full11,550-row point difference is **−0.000133**, but that aggregate includes the previously discovered2026 effect. It does not override the frozen replication gate. All four critical noninferiority guards passed; passing noninferiority is not a substitute for the failed improvement criterion. **Do not promote native6+12 from this program.** This does not prove equivalence or that every longer-training recipe is ineffective.

### Coherent family experiment: a positive structural result, not an A0 replacement

R0 and R1 share exactly one player-only Glicko-2 state and identical player updates. R1 adds `FamilyMapBridge`, with prior deviation150 rating points, process variance growth of one rating-point-squared per day, and zero initial means. Actual historical map participants supply fractional prior family affiliations. Daily frozen-prior updates use the player map logit plus the family contrast and weighted Bernoulli likelihood, with total series weight one. Map probabilities are projected to a series **once**.

Affiliations use previously released domestic evidence with a365-day TTL; international appearances do not overwrite them. Missing, conflicting, mixed and transferred affiliations are explicit. A tier change inside the same family is not a transfer. Query-time inactivity inflation does not mutate state. This diagonal weighted-map approximation is **not a joint Bayesian posterior or per-match uncertainty interval**.

Each arm receives one positive scalar temperature per year, fitted on identical released rows of the preceding calendar year: **4,836 / 4,642 / 3,805** CAL rows for2024/25/26. The calibrated object is the declared online algorithm, not falsely frozen rating weights; rating state continues to update from released history before each target.

| Full-context model, same 11,550 rows | Winner LL | Brier | LL ≥2.5 |
|---|---:|---:|---:|
| Frozen annual A0 | 0.551246 | 0.187103 | 33 |
| C0 native12 | 0.550351 | 0.186761 | 47 |
| C1 native6+12 | 0.550218 | 0.186729 | 44 |
| R0 player-only | 0.578667 | 0.197629 | 43 |
| R1 player + family | 0.574993 | 0.196209 | 42 |

**Primary R1−R0:** ΔLL **−0.003674 [−0.007128,−0.001264]**; ΔBrier **−0.001421 [−0.002830,−0.000473]**,33monthly blocks. Equal-month ΔLL is **−0.014114 [−0.025699,−0.004207]**. The family component passed the structural gate and all four critical noninferiority guards.

Tournament-name/year sensitivity uses **444 proxies**, not verified editions. Its match-weighted LL interval is **[−0.006354,−0.001312]**, while its equal-proxy interval **[−0.004854,+0.001440]** crosses zero. The result is not uniform improvement across independent tournaments.

R1 remains worse than A0 as a standalone predictor. It also **failed the source-unseen-player calibration-support gate**, despite support on the overall, BO3, BO5 and major-competition profiles. Structural improvement therefore does not establish general-purpose calibrated probabilities or justify replacing A0.

For source-unseen players, the independently reproduced favorite-confidence gap is **−2.27 percentage points [−5.94,+1.64]**, with908 non-neutral rows, six excluded neutral forecasts and30months. The point estimate is inside±5pp and suggests underconfidence, not overconfidence; the interval extends below−5pp and includes zero. The gate means **the required calibration tolerance was not established**, not that miscalibration or overconfidence was established.

The frozen [conditional design](../data/05_model_input/model_reliability/model_reliability_20261001_055321/integration_design.json) triggered a separate incremental test inside fixed C0. It was **executed**, with the result below; it is not remaining work.

### Conditional integration: no qualifying gain inside A0

The [integration protocol](../data/05_model_input/model_reliability/model_reliability_20261001_055321/integration_protocol.json) fixed three full-context arms: native12 alone, native12+player and native12+family. Each blend is `sigmoid((1−alpha)*logit(p_native12)+alpha*logit(p_rating))`. Annual alpha is fitted only on the identical native CAL subset, with both endpoints checked; no native retraining, temperature refit or assessment-label fitting occurs. Reusing CAL for native assembly and the scalar blend is explicitly **not cross-fitting**.

| Origin | Identical CAL rows | Player alpha | Family alpha |
|---|---:|---:|---:|
| 2024 | 1,379 | 0.228457 | 0.493181 |
| 2025 | 1,103 | 0 | 0.217351 |
| 2026 | 1,328 | 0 | 0.194669 |

| Same11,550 full-context targets | LL | Brier | LL ≥2.5 |
|---|---:|---:|---:|
| Native12 alone | 0.550351 | 0.186761 | 47 |
| Native12 + player | 0.550713 | 0.186835 | 43 |
| Native12 + family | 0.551099 | 0.186852 | 35 |

- **Family−base:** ΔLL **+0.000748 [−0.001129,+0.002271]**.
- **Family−player:** ΔLL **+0.000386 [−0.001227,+0.001622]**.
- Both prespecified match-weighted primary LL improvements and nonpositive Brier point differences failed. There are33monthly blocks and5,000 draws. The intervals establish neither worsening nor equivalence; the point estimates do not establish the required improvement.
- Equal-month sensitivity is favorable: family−base **−0.003471 [−0.007717,−0.000055]**, family−player **−0.003708 [−0.008215,−0.000160]**. This different estimand does **not** replace the frozen match-weighted primary criterion.
- BO3, BO5 and major noninferiority guards passed. **Source-unseen-player noninferiority was not established:**914rows/30months, ΔLL **+0.006180 [−0.004098,+0.016041]**, versus the chosen upper-bound margin+0.005.
- Native12+family passed the five required historical calibration-support profiles. Calibration support and fewer deep losses do not override failed accuracy/noninferiority gates.

Post-fit descriptive slices reuse the139 already fixed assessment profiles, producing417 metric rows in [integration profile metrics](../data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_profile_metrics.csv); [recipe/hashes](../data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_profile_recipe.json). Native12/family-blend annual LL is **0.563031/0.565337** in2024, **0.547035/0.546854** in2025 and **0.535449/0.535006** in2026. No profile-dependent correction or post-result selection was fitted. The integration itself covers **full context only**; the parent assessment separately covers all three native modes.

**Decision: reject promotion of this integration.** Preserve the standalone family component's positive structural result without declaring a stronger general-purpose A0 model. The lower tail count47→35 is not a substitute for probabilistic accuracy. [Completed execution](../data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_v1/completed.json), [gates](../data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_v1/gates.json).

### Coverage, failures and assessment provenance

The completed [assessment](../data/08_reporting/model_reliability/model_reliability_20261001_055321/assessment_v2/evaluation.json) includes **25 probability columns with zero missing forecasts** on all11,550 canonical targets; year counts are4,642/3,805/3,103. Common fixed-A0 confidence profiles remain paired; candidate-own confidence bins are separate calibration diagnostics. Low statistic coverage uses the original90% threshold, and neutral favorite forecasts use `abs(p−0.5) <= 1e−12`.

All headline scores and paired differences use strict interior probabilities **without clipping**. Calibration support requires at least200 non-neutral rows, six months and the whole95% mean favorite-confidence-minus-win-rate interval inside±0.05. These are predeclared engineering criteria on exposed development data, not universal guarantees or forecast-specific confidence intervals.

Two experimental failures were retained before separately frozen repairs:

1. The first family replay rejected a legitimate reversal between canonical and raw-source A/B orientation. Generic team/roster/outcome alignment was repaired, tested and frozen in **`model_reliability_20261001_055321_family_alignment_v2`**. A full preflight checked all41,915 raw series and16,386 forecast targets, including8,059 reversed orientations, before the successful replay.
2. Initial assessment reporting used the wrong component/context column order. It aborted before export. The repaired mapping was frozen in **`evaluation_protocol_v2.json`**, with unchanged model outputs, populations, resampling and gates. Both source snapshots and failure receipts remain available.

Evidence: [family protocol](../data/05_model_input/model_reliability/model_reliability_20261001_055321_family_alignment_v2/family_protocol.json), [family execution](../data/08_reporting/model_reliability/model_reliability_20261001_055321_family_alignment_v2/family/family_reliability_receipt.json), [evaluation protocol](../data/05_model_input/model_reliability/model_reliability_20261001_055321/evaluation_protocol_v2.json), [metrics](../data/08_reporting/model_reliability/model_reliability_20261001_055321/assessment_v2/metrics.csv), [contrasts](../data/08_reporting/model_reliability/model_reliability_20261001_055321/assessment_v2/contrasts.json), [calibration](../data/08_reporting/model_reliability/model_reliability_20261001_055321/assessment_v2/calibration.json).

### Independent arithmetic and provenance verification

A separate [oracle implementation](../data/08_reporting/model_reliability/model_reliability_20261001_055321/independent_assessment_audit.py), executed by Main without importing project scoring/bootstrap functions, reproduced **125 metric rows,54 paired-contrast intervals and55 calibration intervals**. Its1,264 numerical comparisons had maximum discrepancy **1.39e−16**;189 hash comparisons passed across184 distinct references, including the archived assessment sources and unchanged model/input artifacts.

The audit checked all11,550 canonical identities/orientations,16,386 family query receipts,160,105 affiliation release/TTL records and24,833 prediction/CAL receipt links. All six annual scalar temperatures had absolute objective gradient below7.5e−9; independent temperature transforms reproduced stored probabilities within2.23e−16. This is arithmetic and receipt consistency, not a second full rating replay or publication authentication. [Audit result](../data/08_reporting/model_reliability/model_reliability_20261001_055321/independent_assessment_audit.json).

The separate [integration oracle](../data/08_reporting/model_reliability/model_reliability_20261001_055321/verify_integration.py) replayed **all3,810 native CAL probabilities from real checkpoints**, maximum discrepancy **2.22e−8**. It independently checked six convex alpha objectives/endpoint KKT conditions, five metric rows,21 contrast intervals,25 calibration intervals and24 source/artifact hashes. Its357 numeric comparisons passed. An initial standalone invocation lacked the repository import path and failed before inference; the corrected proof script then completed. [Verification receipt](../data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_verification.json).

**Final regression:164passed,0failures,0errors,0skips**, one existing Starlette/httpx deprecation warning,4.61s pytest time. The command and JUnit hash are recorded in [final_regression.json](../data/08_reporting/model_reliability/model_reliability_20261001_055321/final_regression.json). Both database variables pointed to the isolated, migrated loopback instance; it was stopped afterward. This was a focused suite, not the entire application suite. Redundant plumbing-only tests were removed; the real checkpoint, future-perturbation and database proofs remain.

A separate integration code review and two complete-document reviews found no blockers; two documentation nits were corrected. Reviewers inspected evidence without rerunning tests or experiments. [Review dispositions](../data/08_reporting/model_reliability/model_reliability_20261001_055321/final_review.json); [initial introduced-link check](../data/08_reporting/model_reliability/model_reliability_20261001_055321/documentation_link_verification.json).

Executed integration command, from repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_model_benchmark.py \
  --reliability-integration data/05_model_input/model_reliability/model_reliability_20261001_055321/integration_protocol.json \
  --output-dir data/08_reporting/model_reliability/model_reliability_20261001_055321/integration_v1
```

The output is write-once; use a **new, nonexistent output directory** for a deliberate replay. The final integration sources are pinned. The earlier assessment used its own source ZIP before the integration CLI extension; replay that assessment with its archived source snapshot, never by rewriting its hashes.

The checkpoint/arithmetic proof commands below read experiment artifacts and emit JSON to stdout; they do not write results or database rows. Ordinary Python imports may create bytecode caches. Run from repository root:

```bash
.venv/bin/python data/08_reporting/model_reliability/model_reliability_20261001_055321/independent_assessment_audit.py
.venv/bin/python data/08_reporting/model_reliability/model_reliability_20261001_055321/verify_integration.py
```

### Final decision and forward boundary

The [sealed decision](../data/08_reporting/model_reliability/model_reliability_20261001_055321/decision.json) binds18 protocol/completion/proof/reference hashes. **No new general-purpose A0 replacement qualified, and no new candidate was frozen for promotion.** Frozen annual Causal A0 remains the unpromoted research reference. Neither its weights/calibration nor the canonical manifest nor the operational model changed.

The [forward design](../data/05_model_input/model_reliability/model_reliability_20261001_055321/forward_design.json) records first admissible immutable pre-start issuance, all requested fixtures including missing forecasts, real source/cutoff/generated/start clocks, and no odds inputs. Checkpoints are fixed: three complete months for integrity only; one quality assessment after six months and at least2,000 scored matches; administrative close at twelve months if support is inadequate. No interim tuning or retrospective issuance.

**Prospective confirmation: `NOT_RUN`; actual genuine future forecasts:0.** The twelve historical envelopes are not prospective evidence. Future fixture/start/roster/source observations, captured release updates, a compatible new-year artifact before out-of-year forecasts, elapsed observation periods and arriving outcomes are external prerequisites. The existing annual-year restriction remains intact. No live collection, deployment, financial evaluation or tournament-edition certification occurred, and no further experiment starts automatically.


## 12. Original September29 diagnostic evidence and remaining limits

Run from the repository root, using the existing research-root resolver:

```bash
.venv/bin/python data/08_reporting/noisy_data_reliability_20260929_205336/diagnostics.py
.venv/bin/python data/08_reporting/noisy_data_reliability_20260929_205336/verify_and_plot.py
.venv/bin/python data/08_reporting/noisy_data_reliability_20260929_205336/tournament_sensitivity.py
```

These commands were executed successfully. The first runs the source audit, confidence summaries, neutral/side-reversal checks, and 14 mathematical cases. The second independently checks 116 saved point estimates plus eight controls and produces the figure. The third records the post-hoc source-label blocking sensitivity and checks its 12 match-weighted/equal-label point estimates. These checks **do not independently reimplement the bootstrap interval algorithm**; that remains the existing shared `paired_cluster_interval`.

Artifacts: [protocol v2](../data/08_reporting/noisy_data_reliability_20260929_205336/protocol_v2.json), [diagnostic receipt](../data/08_reporting/noisy_data_reliability_20260929_205336/diagnostic_verification.json), [independent verification](../data/08_reporting/noisy_data_reliability_20260929_205336/independent_verification.json), [source/citation ledger](../data/08_reporting/noisy_data_reliability_20260929_205336/citation_audit.json), [research-track evidence](../data/08_reporting/noisy_data_reliability_20260929_205336/research_tracks.json).

The sensitivity has a separate [post-hoc protocol](../data/08_reporting/noisy_data_reliability_20260929_205336/tournament_sensitivity_protocol.json) and [verification receipt](../data/08_reporting/noisy_data_reliability_20260929_205336/tournament_sensitivity_verification.json).

The artifact directory also contains CSVs for confidence bins, input-quality strata, high-confidence counts, winning/losing controls, competition overlap, and the 240-cell source audit. Scripts and receipts preserve source/input hashes. These are local research artifacts, not new canonical model manifests.

Material limits:

- The September29 diagnostic study did not train or recalibrate models. The separately authorized October1 experiment in §10 did train six neural models and fit the fixed native CAL recipe. Neither study executed a financial simulation, tournament simulation or prospective test; no probabilistic imputer was fitted.
- Basic source integrity and sampled mask fidelity do not establish that all observed statistics or labels are correct.
- Low coverage is nearly confounded with LPL; nine-month descriptive intervals cannot identify the cause or support many new subgroup parameters.
- Actual rosters and daily releases are the existing research scenario, not evidence of roster announcements or production forecast issuance. Market-linked membership does not certify executable odds.
- Literature benchmarks are mostly tabular/vision/text/clinical tasks. Transfer recommendations are hypotheses. Recent preprints are labeled as such; no outside paper establishes an A0 improvement.
- The benchmark manifest, frozen models, research archive, operational application, and user Git work remain unchanged by this investigation.

## 13. Reliability versus log-loss (Plan08, 2026-10-01)

The separately authorized [Plan08](../TODO/08_reliability_first_model_tradeoff.md) tested whether the apparent native6+12 candidate advantage was primarily confidence/dispersion rather than better ranking, and whether six native epochs could buy reliability at acceptable log-loss cost. The accepted run is **`reliability_tradeoff_20261001_092401_integrity_v2`**. It compares six native epochs with the fixed twelve-epoch C0 reference; it does not claim twelve epochs is a universal optimum.

### Design, support and provenance

- Three seeds **20260910/20260911/20260912**, origins 2024/2025/2026; training uses the native terminal Beta scoreline NLL plus **0.1 masked auxiliary SmoothL1**. There are nine candidate fits, each one year × seed; C0 is reused. The confidence control is not another fitted network: within each year, it scales CAL logits by a single positive factor chosen to match native12's CAL favorite-confidence RMS. It preserves each candidate's ordering.
- Reliability is anchored to the **fixed annual-A0 favorite** and uses five fixed favorite-confidence strata crossed with calendar quarters. The 55 cells have **11,544 non-neutral assessment observations and six neutral forecasts**. The primary RMS aggregates fixed-cell squared favorite-gap error with assessment-support weights. Sparse cells remain influential according to those weights, and whole-month bootstrap resamples can leave some cells without support; this is a stated limitation, not a reason to drop or replace empty cells.
- Paired uncertainty uses **5,000 monthly-block draws**, seed **20261001**, across **33 months**. The name/year sensitivity has **444 uncertified tournament-name/year proxies**; they are not independently authenticated tournament editions. All 2024–2026 observations remain repeatedly exposed historical development.
- The initial run `reliability_tradeoff_20261001_092401` was retired before formal assessment after transitive 2026 artifact enforcement exposed a provenance gap. The accepted `_integrity_v2` run fixes that enforcement. The nine learned network/dependence parameter hashes match between attempts, but serialized checkpoint bytes do not (0/9 byte-identical); do not conflate matching learned states with byte-identical checkpoint files. Across the two attempts there were **18 candidate fits / 108 seed-epochs**, i.e. two executions of the same nine-fit recipe, not 18 recipes. Two additional three-seed, one-epoch smoke runs are separate preflights. No post-score tuning occurred.

### Primary reliability and proper-score results

| Full-context model | Winner LL | Brier | LL ≥2.5 |
|---|---:|---:|---:|
| Annual A0 | 0.551245583 | 0.187102933 | 33 |
| C0 native12 | 0.550350935 | 0.186761019 | 47 |
| Candidate native6 | 0.551217871 | 0.187049109 | 53 |
| CAL confidence control | 0.550321456 | 0.186750817 | 43 |

On all **11,550** targets, native6−C0 ΔLL is **+0.000866935**, monthly 95% CI **[+0.000153092,+0.001660772]**: native6 is worse on the primary proper score. C0 passes all shared-expert cost and absolute-support gates. Native6 meets the global LL/Brier and BO3, major and source-unseen-player canonical-A0 guards; the separate BO5 upper-bound guard fails below. Regardless of those passing guards, the candidate fails the primary C0-relative reliability-gain criterion. The confidence control is a diagnostic control, **not an eligible model or replacement**.

For fixed-A0 favorite calibration, quarter-cell RMS is **2.617194773%** for C0 and **2.574139930%** for native6, a point gain of just **0.043054843 percentage points**, below the frozen minimum **0.5 pp**. The monthly ΔRMS interval is **[−0.000971842,+0.000311238]** as a proportion (it crosses zero); the Δsquared-gap interval is **[−0.000066624,+0.000022024]**. Reliability improvement is not established; the candidate fails the frozen reliability-gain gate.

The separate canonical-A0 replacement guard also fails: on BO5 (**1,532** matches, **32** monthly blocks), native6−A0 ΔLL is **+0.002185692**, 95% CI **[−0.000633568,+0.005557397]**, exceeding the prespecified upper-bound margin **+0.005**. Keep this distinct from the primary C6-vs-C0 contrast and the RMS reliability gate. Across both LL-based and reliability-based selectors, C0 wins all origins: **zero** eligible evidence rows at 2024 origin, **4,642** at 2025, and **8,447** at 2026; both selectors select **native12__full__full** each year. The same hindsight-best constant is native12. This retrospective selection agreement is not independent validation.

The fixed-A0 anchor and common 55-cell construction avoid changing confidence bins when comparing candidates. The reported primary metric is not ECE and not an aggregate signed gap. Sparse quarter/confidence cells, nominal bootstrap coverage, the small number of calendar years, and the 444 uncertified name/year proxies limit generalization. This analysis makes **no latent-noise-cause claim** and supports neither a global overconfidence diagnosis nor a blanket flattening remedy.

### Training-loss and scoreline diagnostics are separate

For each of the three origins, raw-neural and full-assembly winner LL are lower for native12 than native6 on the TRAIN, CAL and ASSESS partitions: **18 of 18 origin × partition × output comparisons favor twelve epochs** (three origins × three partitions × two outputs; the earlier text incorrectly counted twelve). Thus the duration pattern is not the classic symptom of training LL worsening while assessment LL improves; this result alone does not identify a cause or establish a universal duration law. Raw/calibrated neural scoreline and winner-probability diagnostics are separate from the SGD objective: fitting minimizes native terminal scoreline NLL plus auxiliary loss, while the scored winner LL is an assembled probability after calibration/ensemble steps. Do not treat those quantities as interchangeable.

| Origin | Partition (N) | Raw neural LL: native12 / native6 | Full assembly LL: native12 / native6 |
|---|---|---:|---:|
| 2024 | TRAIN (27,705) | 0.586119154 / 0.590975736 | 0.586354419 / 0.590422544 |
| 2024 | CAL (1,379) | 0.575911683 / 0.576434215 | 0.573821808 / 0.574286275 |
| 2024 | ASSESS (4,642) | 0.563401936 / 0.563830344 | 0.563030533 / 0.563524888 |
| 2025 | TRAIN (32,625) | 0.582891465 / 0.587503154 | 0.584107553 / 0.588282664 |
| 2025 | CAL (1,103) | 0.544238759 / 0.546065137 | 0.542282347 / 0.543696576 |
| 2025 | ASSESS (3,805) | 0.547412386 / 0.547913795 | 0.547035021 / 0.547462992 |
| 2026 | TRAIN (36,205) | 0.580924704 / 0.583954275 | 0.580366286 / 0.583259782 |
| 2026 | CAL (1,328) | 0.538329555 / 0.538808745 | 0.537919885 / 0.538376611 |
| 2026 | ASSESS (3,103) | 0.536040437 / 0.537919867 | 0.535448701 / 0.537411277 |

These are descriptive scores over the protocol's origin-partition rows; TRAIN/CAL are not held-out independent confirmations and assessment years are development-exposed.

### Integrity, verification and decision

The first attempt's enforcement defect was corrected in provenance/validation, not by changing the production model, training recipe or frozen source protocol. The accepted training and assessment receipts, source snapshot, predictions, tables and regression evidence remain under the run's stage directories. The focused regression receipt records **43 passed, zero failures/errors/skips, one existing warning**. Selection-invariance proof passed six real-data checks (two selectors × three years), perturbing **11,550 / 6,908 / 3,103** inadmissible rows while preserving origin records/evidence IDs and current-year probabilities. Independent primary arithmetic reproduced **1,418 numeric values, 184 metric rows, 35 proper-score intervals, 8 nonlinear-calibration contrasts, 3 annual contrasts, 6 selector records and 19 gate booleans**, with maximum absolute error **1.11e−16**. Native checkpoint/software replay passed: **2,014,110 probabilities** replayed, maximum error **1.770588805e−6**, under the inherited `2e−6` tolerance; this is not an exact/bitwise replay claim and the discrepancy has no established numerical cause. **1,342,740** scoreline-NLL values matched to **9.77e−15**; 182 artifact/archive hashes checked. The replay also checked all nine C0 first-six-loss trajectories, inference for BO1/3/5 across three modes and origins, symmetry (maximum complement error **8.9817e−9**), neutral cold starts, finite empty histories, repeated-input exactness and input/parameter nonmutation. The bounded one-update future perturbation probe used128 selected TRAIN rows and excluded CAL/assessment features/labels/auxiliary poison; it did not rebuild raw source data. These are offline software/arithmetic/replay checks, not evidence of future forecast issuance or external source publication. Receipts: [native replay](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/native_replay_verification.json), [independent report comparison](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/independent_report_comparison.json), [selection invariance](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/selection_invariance.json), [regression](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/regression.json).
An additional post-goal-update smoke used the **actual frozen native loader** on BO1/3/5 across full/no-W20/no-organization modes: **27 probabilities checked**, maximum replay error **2.670756e−8** ([smoke receipt](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/post_documentation_smoke.json)).
An independent endpoint/profile audit reproduced **3,058 reported profile rows across 139 profiles / 11,550 assessment rows**, with maximum numeric error **1.11e−16**; it also independently checked all **72 endpoint accuracy summaries** and **144 raw/calibrated scoreline-NLL summaries** over **111,895 endpoint rows**. All nine confidence-origin/mode rank checks passed with zero ranking inversions or favorite flips. Native6 does flip the fixed-A0 favorite on **244 / 263 / 266** rows in full/no-W20/no-organization modes, respectively; this is why reliability anchoring holds the favorite fixed. The audit does **not** independently verify Brier decomposition, AUC, ECE/MCE, calibration slopes or intercepts. See [profile/endpoint receipt](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/profile_endpoint_verification.json).

**Decision: retain annual A0; do not promote native6 or the confidence control.** The canonical manifest and production implementation/artifacts remain untouched. The [sealed decision](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/decision.json) records `COMPLETED` / `NOT_ADOPTED` and binds the final evidence files; this report and its documentation verification are linked by [documentation_verification.json](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/documentation_verification.json). Source-genuine forward forecasts remain **0**; prospective confirmation is **NOT_RUN** ([forward boundary receipt](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/forward_confirmation.json)). No live collection, deployment, financial/tournament certification or automatic follow-up experiment is authorized.

From the repository root, the frozen protocol, write-once assessment and focused regression commands were:

```bash
.venv/bin/python scripts/train_a0_reliability_tradeoff.py --protocol data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json
.venv/bin/python scripts/run_model_benchmark.py --reliability-tradeoff data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json --output-dir data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment
.venv/bin/python -m pytest -q betting_app/tests/test_reliability_tradeoff.py betting_app/tests/test_a0_reliability.py betting_app/tests/test_a0_data_focus.py betting_app/tests/test_model_reliability_report.py betting_app/tests/test_model_benchmark.py betting_app/tests/test_research_benchmark_suite.py
```

The training and assessment destinations are completed/write-once. Do not rerun them over these paths. Paths above are repository-relative.

## 14. Deep integrity audit and local repairs (2026-10-01)

**Decision: repair the demonstrated input/provenance failures; retain frozen annual A0. No predictive or financial improvement is established.** The user requested an audit and attempts to fix the weaknesses identified in the model review. Run `model_integrity_audit_20261001_01` examined fixture clocks/context, roster evidence, quote/EV contracts, and native training/calibration/assembly. The [frozen protocol](../data/05_model_input/model_integrity_audit/model_integrity_audit_20261001_01/protocol.json) separates software correctness from predictive performance; its initial status is superseded by the [final decision](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/decision.json), not rewritten retrospectively.

### Reproduced failures and repairs

| Failure before repair | Local repair and observed result |
|---|---|
| `predict_fixture_inputs` silently preferred `decision_at` over a conflicting `cutoff_at`. A real-source feature reproduction declared cutoff December22 but admitted December22 history using the December24 decision origin. | These fields are aliases in this fixture API, not two supported independent clocks. Conflicting instants now raise; either alias alone and timezone-equivalent aliases work. Actual forecast creation remains separate. |
| A fixture's explicit competition context was overwritten by conflicting context in an existing metadata row, permitting a roster from the wrong requested context. | Conflict now raises before feature construction. No fuzzy name reconciliation or fallback to another competition is added. |
| When that metadata row had no context, the explicit fixture context was discarded, unnecessarily making a usable prior roster unavailable. | Explicit fixture context fills the missing metadata value without mutating either input frame. The same real-source reproduction now builds an available snapshot in the requested context. |
| A newer roster retraction without a repeated five-player list raised a schema exception before superseding an older announcement. | Retractions need not repeat the withdrawn lineup. The resolver preserves the evidence and returns `latest_announcement_retracted`; full feature replay withholds the forecast even when prior roster history exists. |
| The quote extractor copied provider `createdAt` into `observed_at`, falsely labeling a source-update clock as a receipt clock. | Per-side summaries expose `source_created_at`; observation times remain null when unknown. Source-state selection, prices and quote timestamps are preserved. The collection service's actual response/receipt clock remains separate. |

The first three failures were reproduced through the actual archived feature backend using one frozen source series and synthetic later requests, **not native model inference for that source's 2013 year**. Before/after traces and executable reproduction recipes are in [input_reproductions_before.json](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/input_reproductions_before.json), [input_reproductions_after.json](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/input_reproductions_after.json), and [input_integration_smoke.json](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/input_integration_smoke.json).

The quote change does not recover missing historical receipt evidence. Research quote availability and strict execution admission intentionally remain different: the former can describe an archived source-price scenario; the latter still needs bookmaker participant binding and decision-time evidence. No automatic quote reversal, missingness removal, odds threshold or probability flattening was introduced. Existing historical artifacts retain their original provenance fields and hashes; they were not silently rewritten or recertified.

### Native recipe and reporting audit

The [native audit](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/NativeContractAudit.json) examined release-closed TRAIN/CAL membership, TRAIN-only normalization, seed/mode assembly, optional-context masks, terminal-score mixture semantics, and favorite-calibration symmetry. Its bounded synthetic partition and complement checks passed. It found no reproducible assembly defect warranting a changed forecast. The large Plan07/08 native replay receipts remain **earlier evidence**, not new million-row replays in this audit.

Reusing CAL for expert-slope/mixture fitting and CAL diagnostics can make those diagnostics optimistic. This is an explicit limitation, not demonstrated leakage into assessment predictions. Organization/W20 profiles are reported but are not all primary promotion gates; that is a gate-design boundary, not proof of incorrect mode inference. Neither finding justifies an outcome-selected correction.

The saved Plan08 duration decision contains **18**, not12, raw-neural/full comparisons across three origins and three partitions; all favor native12. The [count verification](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/duration_count_verification.json) corrects the prose above and the research handoff without changing saved metrics, model selection or sealed evidence. Earlier documentation hashes identify their prior revisions.

### Verification and effect boundary

- **91 focused tests passed**, zero failures/errors/skips, one existing Starlette/httpx warning. Tests used an inert loopback database URL; no configured user database was accessed. This is not a full application or migrated-database regression.
- The actual `scripts/predict_causal_a0.py` completed a full frozen-history replay in145.65s: **nine historical requests, four available forecasts, five retained absences**, with exact probability, identity, clock, evidence-status and missing-reason parity against the prior CLI control. [Native CLI receipt](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/native_cli_verification.json). This run used `full` mode; it is not a new all-mode replay.
- The integrated feature smoke checked decision-only, cutoff-only and timezone-equivalent dual aliases in one batch, all **eight tensors** exactly equal and input frames unmodified. A current rosterless retraction withheld the forecast; a future retraction left earlier tensors unchanged.
- **15 real archived quote cases**, selected by horizon/source-state combination without outcomes, preserved prices, source clocks, states and availability. They cover active, inactive, unavailable, absent-market and invalid-price cases; ten distinct raw hashes matched. Unknown observation times remained null. [Quote smoke](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/quote_smoke_verification.json). This bounded regression is not a fresh full quote-corpus or financial replay.
- [Eight integrity comparisons](../data/08_reporting/model_integrity_audit/model_integrity_audit_20261001_01/reference_integrity.json) preserve the five canonical pinned inputs/evidence files, manifest, native adapter and core replay source. The native loader additionally enforced its own artifact pins. The immediately preceding [canonical rescore](../data/08_reporting/benchmark/model_health_review_20261001_01/REPORT.md) covered all11,550 rows: A0 LL **0.551245583**, Brier **0.187102933**, accuracy **71.13%**, ECE10 **1.15%**,33 deep losses. No new model fitting occurred.

The principal predictive weakness remains: the last repaired-context fixed387 selections overstate EV by **29.78pp**, with the existing monthly95% interval **[10.45,54.23]pp**. These local contract fixes do not erase that result, establish its sports cause, or validate a blanket recalibration. Missing historical announcements/bindings, dependence and limited independent tournament editions, repeated historical development, and the absence of genuine future forecasts remain unresolved evidence limits. No weights, calibrators, canonical manifest, old selection ledgers or production services changed; no live collection, H/market training, deployment, tournament certification, or restart of the completed10/10 loop occurred.

## Bibliography

Primary sources accessed 2026-09-29. Exact short quotations, access depth, and source locations are recorded in `citation_audit.json`. ArXiv links can refer to preprints or later revisions; venue status is not inferred from search snippets.

[1] Tilmann Gneiting, Adrian E. Raftery (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf.

[2] Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. https://www.jair.org/index.php/jair/article/download/12125/26676/26691.

[3] Aritra Ghosh, Himanshu Kumar, P. S. Sastry (2017). *Robust Loss Functions under Label Noise for Deep Neural Networks*. https://arxiv.org/abs/1712.09482.

[4] Zhilu Zhang, Mert R. Sabuncu (2018). *Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels*. https://arxiv.org/abs/1805.07836.

[5] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang, Masashi Sugiyama (2018). *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*. https://arxiv.org/abs/1804.06872.

[6] Rafael Müller, Simon Kornblith, Geoffrey Hinton (2019). *When Does Label Smoothing Help?*. https://arxiv.org/abs/1906.02629.

[7] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, Geoffrey Hinton (2017). *Regularizing Neural Networks by Penalizing Confident Output Distributions*. https://arxiv.org/abs/1701.06548.

[8] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2017). *Focal Loss for Dense Object Detection*. https://arxiv.org/abs/1708.02002.

[9] Jishnu Mukhoti et al. (2020). *Calibrating Deep Neural Networks using Focal Loss*. https://arxiv.org/abs/2002.09437.

[10] Nontawat Charoenphakdee, Jayakorn Vongkulbhisal, Nuttapong Chairatanakul, Masashi Sugiyama (2020 preprint). *On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective*. https://arxiv.org/abs/2011.09172.

[11] Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values*. https://www.nature.com/articles/s41598-018-24271-9.

[12] Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar (2020). *VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain*. https://proceedings.neurips.cc/paper_files/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf.

[13] Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler (2022). *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*. https://arxiv.org/abs/2106.15147.

[14] Pierre-Alexandre Mattei, Jes Frellsen (2019). *MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets*. https://proceedings.mlr.press/v97/mattei19a.html.

[15] Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen (2021). *not-MIWAE: Deep Generative Modelling with Missing Not at Random Data*. https://arxiv.org/abs/2006.12871.

[16] Tianyu Du, Luca Melis, Ting Wang (2024). *ReMasker: Imputing Tabular Data with Masked Autoencoding*. https://proceedings.iclr.cc/paper_files/paper/2024/file/258d234a229d1612a565f9d9a63e3e80-Paper-Conference.pdf.

[17] Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. https://papers.nips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf.

[18] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek (2019). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*. https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf.

[19] Rahul Rahaman, Alexandre H. Thiery (2021). *Uncertainty Quantification and Deep Ensembles*. https://arxiv.org/abs/2007.08792.

[20] Wesley J. Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson (2019). *A Simple Baseline for Bayesian Uncertainty in Deep Learning*. https://arxiv.org/abs/1902.02476.

[21] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig (2021). *Laplace Redux – Effortless Bayesian Deep Learning*. https://arxiv.org/abs/2106.14806.

[22] Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, Balaji Lakshminarayanan (2020). *Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness*. https://proceedings.neurips.cc/paper_files/paper/2020/file/543e83748234f7cbab21aa0ade66565f-Paper.pdf.

[23] Mira Jürgens, Nis Meinert, Viktor Bengs, Eyke Hüllermeier, Willem Waegeman (2024). *Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?*. https://arxiv.org/abs/2402.09056.

[24] Maohao Shen, J. Jon Ryu, Soumya Ghosh, Yuheng Bu, Prasanna Sattigeri, Subhro Das, Gregory W. Wornell (2024). *Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?*. https://arxiv.org/abs/2402.06160. Preprint.

[25] Masashi Sugiyama, Matthias Krauledat, Klaus-Robert Müller (2007). *Covariate Shift Adaptation by Importance Weighted Cross Validation*. https://www.jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf.

[26] Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang (2020 (ICLR; arXiv 2019)). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*. https://arxiv.org/abs/1911.08731.

[27] Ben Van Calster, David J. McLernon, Maarten van Smeden, Laure Wynants, Ewout W. Steyerberg, Patrick Bossuyt, Gary S. Collins, Petra MacAskill, Karel G. M. Moons, Andrew J. Vickers (2019). *Calibration: The Achilles Heel of Predictive Analytics*. https://doi.org/10.1186/s12916-019-1466-7.

[28] Yonatan Geifman, Ran El-Yaniv (2017). *Selective Classification for Deep Neural Networks*. https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf.

[29] Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas (2019 (arXiv version 2020)). *Conformal Prediction Under Covariate Shift*. https://arxiv.org/abs/1904.06019.

[30] Isaac Gibbs, John J. Cherian, Emmanuel J. Candès (2023 (arXiv v4, 2024)). *Conformal Prediction with Conditional Guarantees*. https://arxiv.org/abs/2305.12616.

[31] Chen Xu, Yao Xie (2021). *Conformal Prediction Interval for Dynamic Time-Series*. https://proceedings.mlr.press/v139/xu21h.html.

[32] Zarin Tahia Hossain, Mostafa Milani (2025). *Beyond Accuracy: An Empirical Study of Uncertainty Estimation in Imputation*. https://arxiv.org/abs/2511.21607. Preprint.

[33] José Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan (2026). *High Performance, Low Reliability: Uncertainty Benchmarking for Tabular Foundation Models*. https://arxiv.org/abs/2605.28554.
