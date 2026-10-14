# Learning calibration without learning noise: a research direction for A0

**Research date:** 2026-09-27. **Status:** literature review, completed saved-forecast diagnostics, and proposed experiments. No new deployable calibrator, neural training, model adoption, or deployment.

## Executive decision

**Do not replace probability learning with ECE minimization. Separate three jobs: learn useful probabilities, estimate their calibration reliably, and correct only errors that persist out of time.**

The most defensible new calibration hypothesis is a **small, identity-centered, partially pooled positive-slope model**: related competition categories share statistical strength instead of receiving five unrelated calibrators. This is a research proposal, not a demonstrated improvement. The previous unpooled-temperature experiment did not justify changing A0, and this review does not reverse that result.

The new numerical finding is especially important: on all **11,550** saved annual-A0 matches, fitting an isotonic curve to the evaluation outcomes improves Brier by **0.001120**. Under simulations in which the saved probabilities are correct, the same procedure improves Brier by **0.001144 on average** solely through fitting realized outcomes. All five recent competition groups show apparent improvements inside their respective illustrative reference ranges. A flexible fit can therefore look useful without identifying a transferable correction.

| Question | Finding | Decision |
|---|---|---|
| Is low ECE sufficient? | A constant 50/50 forecast has full-cohort ECE 2.44% and AUC 0.5. | Retain proper scores and discrimination. |
| Are high recent tier ECEs necessarily model failure? | Small calibration cells have substantial outcome noise; international 2026 has only 110 matches. | Diagnose uncertainty and persistence before fitting. |
| Should every tier get its own model? | The previous 110 past-only temperature fits did not establish a gain. | Test pooling, not another unconstrained partition. |
| Is a calibration-aware training loss the next priority? | Such losses exist, but neither their empirical benefits nor guarantees transfer automatically to this A0 recipe. | Preserve the existing likelihood; consider a controlled regularizer only after a persistent failure is identified. |
| Can uncertainty estimates remove upsets? | Outcome randomness, parameter uncertainty, and missing information are different. | Do not label legitimate losses as corrupt labels or seed spread as a posterior. |

**Evidence:** `data/08_reporting/calibration_learning_research_20260927_192446/`. Earlier annual-tier calibration: `data/08_reporting/a0_tier_calibration_20260927_185822/`.

## 1. Scope, method, and evidence labels

Four concurrent research tracks examined measurement, subgroup learning, temporal adaptation, and training/uncertainty. The selected bibliography contains **33 primary research sources**, including recent calibration-inference work and a July 2026 online-recalibration preprint. Some sources were available to a track only at abstract level; those claims are restricted accordingly. Access depth, short source excerpts, and qualifications are retained in `research_tracks.json` and `source_audit.json`. The PNAS CORP publisher page was inaccessible to a track; its author preprint and institutional publication record supplied the evidence.

The local object is **annual Causal A0 series-win probability**, `p_full`/`p_native`, for **2024-01-01 through 2026-09-08**. It is neither an unprojected map probability, monthly A0, nor the production EXP081/market hybrid. There is one record per series; reversing sides does not create another independent observation. Competition category is event metadata, not a team's historical affiliation.

All canonical years have informed previous development. No new chronological split can make those already-inspected outcomes an untouched prospective test. Realized first-map rosters and daily effective release remain scenario inputs, not independently verified announcement/publication timestamps.

Labels used below:

- **Observed:** computed in this investigation or recorded in the identified previous run.
- **Literature:** supported under the cited source's assumptions and setting.
- **[INFERENCE]:** a transfer argument or interpretation not established for LoL.
- **Proposed:** not implemented or evaluated as a candidate in this investigation.

No betting-policy, financial-execution, or tournament-distribution experiment ran. A correct match-level calibration result would not by itself validate joint tournament outcomes.

## 2. What “noise” means—and which kind can be reduced

### 2.1 Outcome randomness is not a mislabeled match

Let \(X\) be the information genuinely available before the forecast and \(r(X)=P(Y=1\mid X)\). A correctly estimated probability of 0.70 still permits the other team to win 30% of the time. That result is not evidence that its label is wrong. Label noise means a recording, identity, orientation, or target-definition defect; it requires a data-quality finding.

For squared probability error, the exact population identity is

\[
E[(Y-p(X))^2]
=E[\operatorname{Var}(Y\mid X)]+E[(r(X)-p(X))^2].
\]

The first term is irreducible **given that information set**; the second is probability-estimation error. Better available information can change the first term, but calibrating an existing score cannot reconstruct a future draft or unannounced roster. The analogous log-score decomposition is conditional entropy plus expected Bernoulli KL divergence. These identities explain why proper scores reward truthful probabilities without requiring every outcome to be predictable. [1,2,31]

Do not confuse the conditional variance above with the CORP `UNC` component later in this report: CORP's reference is marginal prevalence, not an estimate of irreducible uncertainty given all available features.

### 2.2 Model-estimation variance

Different finite training samples, regularization choices, optimizers, and seeds produce different estimates. More relevant historical observations, appropriate shrinkage, a lower-variance representation, and useful model diversity can reduce this variance. They do not remove shared misspecification.

For equal-variance estimates with common pairwise correlation \(\rho\),

\[
\operatorname{Var}(\bar\theta)=\sigma^2\left(\rho+\frac{1-\rho}{M}\right).
\]

This is an algebraic illustration, not an estimate of A0's seed correlation. It shows why adding correlated members cannot eliminate their shared component. Deep ensembles have useful empirical evidence, but finite seed dispersion is not automatically a calibrated posterior over the true win probability. Dataset-shift benchmarks also show that historical calibration need not survive a changed population. [19,30,31]

**Project evidence:** five-seed A0 did not establish improvement over the frozen reference. The block-bootstrap variant worsened LL relative to the seed control by **0.001526**, monthly interval **[0.000745, 0.002291]**. Its larger spread did not establish additional error information beyond favorite confidence. Those negative results constrain, but do not disprove, other uncertainty constructions.

### 2.3 Calibration-estimation variance

A recalibrator is another fitted statistical model. Splitting CAL into tier × confidence × format × roster cells reduces evidence per parameter. More flexibility can lower training error while increasing future error. Parametric-versus-isotonic learning curves and beta-calibration work both document the small-sample tradeoff; neither says a flexible map must improve a well-calibrated base model. [11,12]

Partial pooling addresses this component: a weakly supported group's parameter stays near a shared value. It trades variance against bias if groups actually differ. It is not a guarantee that all group errors disappear.

### 2.4 Evaluation noise

Even an unchanged, correctly calibrated model produces fluctuating bin frequencies, ECE, and observed losses. More bootstrap draws reduce Monte Carlo error in a resampling calculation; they do not increase the number of observed independent events. The existing benchmark has **33 calendar months**, and the recent window has **nine**, not 5,000 independent datasets.

| Noise/problem | Appropriate response | Incorrect shortcut |
|---|---|---|
| Finite evaluation outcomes | More protected future observations, support tables, dependence-aware uncertainty | Increase ECE bins or repeatedly choose the best partition |
| Noisy calibration coefficients | Low capacity, pooling, regularization, past-only selection | Fit separate curves to every small group |
| Unstable base-model estimates | Controlled shrinkage/diversity and better usable evidence | Treat more seeds as automatically better |
| Genuine unknown match outcome | A probability distribution and honest uncertainty | Remove or relabel upsets |
| Missing/shifted information | Improve cutoff-valid inputs and track forecast provenance | Add future draft/side or use output calibration as a substitute |

## 3. What our existing and new calculations establish

### 3.1 Previously completed chronological calibration

Native annual A0 has **LL 0.551246**, **Brier 0.187103**, and **ECE10 1.15%** on all 11,550 matches. In 2026, pooled ECE10 is **2.20%**; the sample-weighted average of within-category ECE10 is **4.31%**.

The earlier experiment fitted **110** actual positive temperatures using only earlier released outcomes. On the same 3,103 recent matches:

| Method | Pooled ECE10 | Category-conditional ECE10 | LL | Brier |
|---|---:|---:|---:|---:|
| Native A0 | **2.20%** | 4.31% | **0.536757** | **0.181573** |
| Global temperature | 2.51% | 4.57% | 0.537216 | 0.181737 |
| Independent category temperatures | 2.34% | **3.98%** | 0.536831 | 0.181630 |

Conditional ΔECE was **−0.335 percentage points**, with monthly resampling interval **[−0.947, +0.639] pp**. A slightly smaller conditional ECE did not come with a demonstrated proper-score benefit. **Native A0 remains the control.**

Earlier temperature, symmetric two-slope, and shrunk-isotonic controls on **monthly** A0 also failed to establish robust gains. They are relevant negative evidence, not the same experiment or a reason to claim these methods are wholly untested.

### 3.2 Actual calibration-cell support

New calculation, fixed ten equal-width bins, 2026 only:

| Group | Matches | Median matches/bin | Smallest bin | Median illustrative IID standard error of a bin residual |
|---|---:|---:|---:|---:|
| All | 3103 | 316 | 166 | 2.34 pp |
| Major | 704 | 74 | 16 | 5.03 pp |
| Regional | 1468 | 152.5 | 106 | 3.51 pp |
| Development | 457 | 44.5 | 30 | 6.00 pp |
| Other top-level | 364 | 35.5 | 11 | 7.13 pp |
| International | 110 | 14 | 2 | 12.63 pp |

The last column assumes independent Bernoulli outcomes with the saved probabilities correct; it is not a dependence-robust standard error for the real data. Its scale explains why a small observed bin discrepancy cannot be read as an accurately measured true error.

For a **single fixed-probability independent bin** near 50%, the normal approximation needs roughly **1,068 observations for a 95% half-width of 3 pp**, or **2,401 for 2 pp**. These are not requirements for achieving ECE below 3%, simultaneous ten-bin intervals, or an esports sample-complexity theorem. Dependence and multiple comparisons change the problem. They demonstrate the difference between a desired probability tolerance and the precision with which it can be checked.

### 3.3 New CORP decomposition: a flexible fit can learn outcome noise

Using the CORP/PAV approach, define the original score \(S_X\), the score after fitting an increasing isotonic map on the same observations \(S_C\), and the score of the sample-prevalence forecast \(S_R\). Then [5,32]

\[
S_X=\underbrace{(S_X-S_C)}_{MCB}
-\underbrace{(S_R-S_C)}_{DSC}
+\underbrace{S_R}_{UNC}.
\]

Here MCB is apparent miscalibration, DSC is discrimination relative to the constant reference, and UNC is that reference score. Brier uses \((p-y)^2\), not the doubled two-class-vector convention. Diagnostic PAV endpoints use the standard zero-log-zero convention; native predictions and benchmark scoring are unchanged.

| Full cohort, N=11,550 | Original score | Same-sample PAV score | MCB | DSC | UNC |
|---|---:|---:|---:|---:|---:|
| Brier | 0.187103 | 0.185983 | **0.001120** | 0.063421 | 0.249404 |
| Log loss | 0.551246 | 0.547561 | **0.003685** | 0.144394 | 0.691954 |

The PAV curve has **50 fitted levels**. Those are not 50 prevalidated probability regimes. This decomposition is descriptive: its discrimination includes same-sample estimation, and its MCB is not a forecast of future improvement or a bound on gains from richer inputs.

We simulated **2,000 independent label vectors per cohort**, keeping the probabilities fixed and generating \(Y_i\sim\operatorname{Bernoulli}(p_i)\). Each vector received the same PAV fit. This intentionally asks how much apparent improvement is produced even when no population correction is needed.

| Cohort | Observed Brier MCB | Perfect-calibration reference mean | Reference 95% range |
|---|---:|---:|---:|
| All 2024–2026 | 0.001120 | **0.001144** | 0.000956–0.001373 |
| All 2026 | 0.002775 | 0.002671 | 0.002119–0.003359 |
| Major 2026 | 0.007997 | 0.006982 | 0.005161–0.009539 |
| Regional 2026 | 0.003607 | 0.004258 | 0.003264–0.005583 |
| Development 2026 | 0.009604 | 0.009141 | 0.006688–0.012548 |
| Other top-level 2026 | 0.010664 | 0.010597 | 0.007673–0.015026 |
| International 2026 | 0.018213 | 0.022759 | 0.014992–0.034471 |

**Conclusion:** the magnitude of these apparent gains does not establish a transferable correction. This is not proof of perfect calibration, a dependence-robust test, or a noise value to subtract from the observed MCB. It is a direct counterexample to promoting a calibrator because it improves the outcomes on which it was fitted.

![Observed same-sample correction compared with independent perfect-calibration reference](../data/08_reporting/calibration_learning_research_20260927_192446/same_sample_calibration_optimism.png)

## 4. Better measurement: retain ECE, but stop treating it as ground truth

### 4.1 Why both upward and downward errors are possible

For a fixed partition, ECE is the weighted absolute difference between bin-average outcomes and probabilities. Coarsening can conceal opposite-signed errors within a bin; sampling noise creates positive absolute deviations even under perfect calibration. These effects oppose one another, so net finite-sample bias does not have a universal sign. Equal-mass bins, debiasing, and adaptive bin counts have evidence in particular settings, not universal superiority. [4,9,33]

The same distinction explains pooled versus category-conditional calibration. At a fixed probability, pooled signed error is an average of category errors. Taking the absolute value after that averaging can cancel errors. But separate categories also contain fewer observations. A larger conditional ECE can reflect cancellation, real differences, estimation noise, or all three; the two scalar numbers alone cannot identify the cause.

Do not silently replace ECE10 with a metric or bin count that makes the result look better. Preserve the historical definition; report sensitivity and the different estimand explicitly.

### 4.2 CORP, cumulative residuals, and smooth alternatives

**CORP/PAV:** useful for a stable monotone shape diagnostic and score decomposition. It assumes a nondecreasing calibration function; a genuinely nonmonotonic error can be hidden by that constraint. Its published uncertainty methods rely on independence/exchangeability conditions that must not be assumed for tournament data. This investigation uses PAV descriptively, not its IID inference as project certification. [5,32]

**Cumulative residuals:** sort forecasts and plot cumulative sums of \(y_i-p_i\). Such plots localize signed departures without arbitrary hard bins. Arrieta-Ibarra et al. develop related scalar statistics; Bröcker develops a uniform test for a particular small-lead-time forecasting setting with stationarity/ergodicity and sequential assumptions. Neither result licenses a generic IID test on these clustered, development-exposed matches. [6,7]

**Smooth/interval/kernel measures:** the distance-to-calibration framework identifies efficiently estimable alternatives with useful continuity properties. Their estimands differ from standard ECE; they are not new mechanisms for improving the underlying probability model. [8]

**Confidence intervals:** recent work on debiased squared calibration error obtains different asymptotics at calibrated and miscalibrated models. This is a warning against treating every small calibration statistic as an ordinary smooth mean. It is not a blanket theorem that all block bootstraps fail, nor a ready-made dependence-robust interval for our ECE10. [10]

**Recommendation:** keep LL/Brier and fixed ECE definitions; add support-aware reliability and a predeclared residual-shape diagnostic. Any confirmatory residual test needs a defensible dependence/null construction. Simply resampling uncentered residuals and calling the result a calibration p-value is not enough. The present IID references are illustrations only.

## 5. Learning calibration with less variance

### 5.1 The concrete lead: partial pooling toward global and identity

**[INFERENCE / Proposed]** Use event category to test a narrow hypothesis: *does A0 have stable differences in confidence scale between categories?* A low-capacity, convex penalized formulation is

\[
q_i=\sigma(a_{g_i}z_i),\qquad z_i=\operatorname{logit}(p_i),
\]

\[
\min_{a_0,a_1,\ldots,a_5\in[0.05,20]}
\sum_{i\in CAL}\ell(y_i,q_i)
+\frac{\lambda}{2}\sum_g(a_g-a_0)^2
+\frac{\lambda_0}{2}(a_0-1)^2.
\]

The bounds match the existing positive-slope family. The loss is binary log loss for the **series winner**. The regularization is on the summed loss scale; changing to mean loss without scaling the penalties would change the estimator.

- Well-supported groups can differ from the shared slope.
- Sparse groups shrink toward the shared slope instead of fitting freely.
- A group with no observations receives the shared slope; a wholly empty fit falls back to identity through the stated prior/penalty.
- Strong group pooling approaches a global fit; strong identity regularization suppresses unsupported correction.
- Positive slopes preserve within-group ordering and side symmetry. Different slopes can reorder forecasts across groups, so global AUC is not guaranteed unchanged.

Direct hierarchical calibration literature exists, but the identified study concerns **human probability judgments**, not esports. It demonstrated reduced bias/scatter without improved discrimination. Its unrestricted intercepts are not appropriate for our side-symmetric contract. The proposed penalty is a deliberately simpler transfer of the pooling principle, not a replication of that study or a Bayesian credible-interval guarantee. [14]

This lead remains low-confidence: shrinking already-unsuccessful tier temperatures may simply recover identity. That is an acceptable negative result. Partial pooling cannot manufacture a subgroup signal that does not persist.

### 5.2 Symmetry rules out some fashionable “upgrades”

For invariant context \(g\), side exchange requires

\[
H(-z,g)=-H(z,g),\qquad q=\sigma(H(z,g)).
\]

An ordinary affine logit map \(\sigma(b+az)\) therefore needs \(b=0\). A beta calibrator [12]

\[
q=\sigma(a\log p-b\log(1-p)+c)
\]

preserves complements for every \(p\) only when \(a=b\) and \(c=0\): it then reduces to temperature scaling. Thus beta calibration's usual three-parameter flexibility is not a free improvement here. This restriction is an algebraic project-contract result, not a claim that beta calibration is generally bad.

A legitimate nonlinear family can instead use an odd monotone logit function. For example, two positive slopes joined at a fixed confidence knot preserve symmetry. But similar symmetric-shape corrections already failed on monthly A0. They should not be proposed as a wholly new discovery or selected now from a broad knot sweep.

A score-only monotone calibrator also cannot repair incorrect within-group ranking. If actual matchup information identifies a favorite the base model ranks incorrectly, adding it creates an enriched prediction model, not merely a new temperature.

### 5.3 Method comparison

| Family | Capacity and useful property | Main risk here | Priority |
|---|---|---|---|
| Identity | No additional fit variance | Leaves genuine persistent error unchanged | Required control |
| Global positive temperature | One scalar; identity included; ranking/symmetry preserved | Cannot model category or shape differences; already tested | Required control, not a new claim [3] |
| Independent tier temperatures | Five local scales | Variance and sparse fallback; previous non-adoption | Historical control |
| Partially pooled positive slopes | Five scales linked to one identity-centered shared scale | Assumes useful similarity; shrinkage needs prior selection | Lead conditional experiment [14] |
| Ordinary beta/affine logistic | Flexible asymmetric shape/intercept | Violates complementarity unless constrained | Not a drop-in replacement [12] |
| Isotonic/monotone spline | Flexible shape | Same-sample optimism, sparse steps/ties, symmetry construction | Diagnostic; later challenger only [5,11] |
| Venn–Abers | Multiprobability prediction with a validity theory | Not an ordinary confidence interval for true p; guarantee depends on exchangeability/selector | Separate research contract [13] |
| Multicalibration | Controls declared overlapping groups and confidence strata | Sample support, group complexity, tuning, dependence | Audit first, not a broad default [15,18] |

### 5.4 Multicalibration: the right aspiration, not unlimited subgroup fitting

Multiaccuracy controls average residual bias over an auditing class. Multicalibration additionally checks calibration within prediction ranges inside the groups. The distinction matters: zero average group residual can hide overconfidence in one range and underconfidence in another. [15,16]

The theory supports rich overlapping classes subject to learnability, representation, and sufficient group mass. A clinical application supplies evidence at large scale, while a recent broad empirical study finds that already-calibrated models often need little additional multicalibration and that post-processing choices matter. None establishes a workable universal tolerance on our 110 international matches. [17,18]

Start with the existing five event categories and coarse **predeclared** confidence ranges. Do not multiply them immediately by team, country, patch, role, roster state, and format. Team-level Tier 1–Tier 2 matchups require chronological team/player affiliations, which were not reconstructed here. Missing affiliation must stay unknown, not become nationality or an event-tier proxy.

## 6. Should calibration be learned inside the neural training objective?

### 6.1 Proper loss is already probability learning

Expected log loss and Brier are minimized by the correct conditional probability in a sufficiently rich prediction class. This does not mean finite-sample optimization, misspecification, or regularization produces perfect calibration; it means the objectives do not intrinsically reward an uninformative constant forecast over a useful correct one. [1,2]

A0's native training is richer than winner-only cross-entropy: it uses a **Beta terminal-score likelihood plus auxiliary player-performance loss**, followed by calibrated seed/expert aggregation. The normalized scoreline likelihood supplies a log score for its target distribution; the auxiliary term changes representation learning and introduces its own bias–variance tradeoff. Do not describe the combined loss as automatically strictly proper for the winner probability alone.

Post-match statistics can be legitimate **supervised targets inside a closed TRAIN partition**. They cannot be inputs to that same game's pre-match forecast, and their labels must be available before model fitting. More correlated player targets from one series do not create more independent series outcomes.

### 6.2 What the alternatives actually offer

**MMCE and differentiable calibration penalties:** kernel calibration discrepancies can be optimized alongside likelihood; MMCE has a population calibration characterization and empirical results on neural classification. Its penalty weight, kernel/batch construction, and score-vs-confidence definition remain choices. A penalty can reduce a calibration statistic without increasing information or improving future proper scores. If tested, preserve A0's main likelihood and use one predeclared small regularizer ablation—not a replacement objective. [24]

**Expectation consistency:** work comparing expectation-consistency and temperature calibration finds similar practical performance in its studied settings. This reinforces the need to test a proposed principle against the simple existing control rather than assume a more elaborate derivation guarantees a better forecast. It is post-hoc evidence, not a proven advantage of a new A0 training loss. [25]

**Label smoothing:** published experiments report benefits, but with uniform binary smoothing the expected target becomes

\[
(1-\epsilon)r(X)+\epsilon/2,
\]

not \(r(X)\). This may regularize a finite neural fit but deliberately changes the probability target. It is not justified by correctly recorded upsets. [26]

**Mixup:** calibration benefits were reported in image/text tasks; simply interpolating LoL rosters, opponents, formats, and labels has no established semantic validity. It can construct impossible match contexts. A domain-valid augmentation needs its own information-preservation argument. [27]

**Focal loss:** some empirical studies report improved calibration, but theoretical work shows raw focal-risk minimizers are not generally the true class posterior. Classification calibration—correct decision boundary—is not probability calibration. A derived probability transformation has assumptions; it is not a guarantee for an arbitrary finite trained network. [28,29]

**Decision:** no objective overhaul is justified now. The project already tested richer supervision and ensembles; adding their names again is not a novel experiment. Training changes should follow a measured, repeatable failure of representation or likelihood, not an ECE threshold chosen after inspecting outcomes.

## 7. Time, versions, and the difference between adaptation and chasing noise

A short recent CAL window responds faster to a real change but estimates its parameters less precisely. An expanding window reduces variance if the calibration relationship is stable, but can average incompatible regimes. High 2026 ECE alone does not decide between them.

Dataset-shift research demonstrates that IID calibration can fail after shift. Importance-weighted calibration addresses a more specific covariate-shift assumption with overlap and defensible density ratios; patch/roster changes may alter the conditional outcome law itself. Reweighting unlabeled inputs cannot generally solve that. [19,20]

A calibrator belongs to a **forecast-generation protocol**, not just a string called A0. Retain model/checkpoint year, upstream calibrator/mixture identity, information mode, BO target, roster availability, forecast time, and result-availability time. Pooling past issued forecasts from an unchanged annual recipe is a prequential modeling assumption to test, not proof that every new checkpoint shares the same scale. Do not silently regenerate historical scores with a model trained on their outcomes.

At an actual recipe or information-contract change, the safe default is its validated native calibration until compatible calibration evidence exists. Refitting an extra stage on native CAL outputs that already used those labels requires careful nested/cross-fitted accounting; do not call those final outputs independently out of sample merely because base-model training ended earlier.

Recent online recalibration theory explicitly combines sequence calibration with excess proper-loss control relative to a supplied forecast stream. A July 2026 preprint gives improved rates for stated Lipschitz proper losses. This is relevant research, but its sequence-level metric, horizon, game order, and randomization conditions are not an immediate guarantee of 2% ECE for each sparse LoL subgroup. Unbounded log loss also is not globally Lipschitz without additional restrictions. [23]

**Conformal/ACI is a different task.** It controls inclusion of outcomes in prediction sets, possibly as a long-run frequency under change. A set containing both teams can cover the result without providing useful win probabilities. Distribution-free marginal coverage does not become pointwise conditional probability accuracy, and conditional-coverage impossibility results must not be misrepresented as saying probability estimation is impossible. [21,22]

## 8. A bounded experimental program—not another broad search

These are proposed experiments, not an automatically authorized execution queue. The first new implementation should address calibration measurement and provenance; the first candidate family, if justified, should be partial pooling.

### Experiment A: test persistence of calibration structure

**Question:** does a deviation recur at later forecast origins, or disappear when outcomes change?

- Freeze the five event categories, target, probability policy, one reliability convention, and a small residual-audit family.
- Use one series once, with issued probabilities and strictly earlier available results for each fit.
- Track both signed residuals and proper-score differences; a visually large small-bin excursion is not a fitted-model mandate.
- Separate detection, method selection, and later evaluation. A historical 2024/2025/2026 split remains development, even when chronology is mechanically correct.
- Report monthly block counts and sensitivity to genuine tournament editions if an edition mapping is available; source tournament names are not certified independent editions.

The existing PAV/null comparison supplies a noise warning, not the missing persistence result. This review did not execute a new residual-persistence test or claim a specific tier is truly miscalibrated.

### Experiment B: one partial-pooling candidate

**Hypothesis:** sharing tier information lowers estimation variance relative to independent slopes while retaining useful category differences.

**Fixed comparison:** native identity; global temperature; independent tier slopes; partially pooled slopes. Include an identity-regularized global control and matched independent regularization when needed to separate *pooling* from simply shrinking everything toward one. Use the same CAL rows, origin schedule, target, and base forecasts for every method.

For a concrete bounded historical design, fix \(\lambda_0=1\) on the summed-log-loss scale and allow only \(\lambda\in\{1,10,100\}\), selected using earlier prequential development origins. These are proposed engineering choices, **not literature-derived optimal constants**. Select once; do not extend the grid after viewing later outcomes. A prospective protocol must lock the choice before its evaluation outcomes arrive.

Do not simultaneously add nonlinear shape, recency weighting, new team affiliations, uncertainty gating, and a changed base model. A candidate that wins after all those changes would not identify what partial pooling contributed.

**Success criteria:** a clear paired proper-score improvement over identity and the credible global control, with Brier not worsened, supported conditional-calibration improvement, full-cohort coverage, and no gain explained solely by one tiny stratum. If noninferiority rather than strict improvement is desired, its substantive margin must be selected before evaluation—not invented afterward. A historical pass is not prospective adoption evidence.

**Failure outcome:** if the fitted group differences collapse toward identity, or conditional ECE falls while LL/Brier worsen, retain native A0 and stop this family search.

### Experiment C: recency only if temporal evidence supports it

A temporal correction is a different hypothesis. Compare expanding CAL with **one** prespecified decay schedule using the same low-capacity calibration family, not a window × tier × spline grid. A 12-month half-life is a possible design choice, not an established optimum. Validate version comparability and effective weighted sample size. Do not combine this ablation with the first pooling experiment.

### Experiment D: representation or training only after calibration is ruled insufficient

If supported error patterns depend on genuinely available context beyond the original score—rather than a stable monotone scale distortion—test a small contextual residual or one representation change against an equally credible A0-only calibration control. Signed features must reverse under side swap; invariant contexts may modulate slopes, not create arbitrary A/B intercepts.

A0 already receives ratings, experience/inactivity, coplay, history attention, optional organization/W20, richer scoreline supervision, and auxiliary player targets. The task is to identify a missing or noisy *mechanism*, not add another feature with one of those names. Earlier roster-form, retrieval, graph, bootstrap, and connector negatives remain relevant. This review identifies no newly proven input defect or trainable feature that already beats A0.

### Prospective boundary and deployment population

Freeze the selected algorithm, update schedule, hypotheses, and scoring population before collecting new evaluation outcomes. Prediction/result timestamps and a validated A0 inference path are prerequisites for claiming a live prospective test; a dated checkpoint alone is not enough. The research A0 adapter is not currently certified as the live production engine.

Use protected future matches for the final comparison. Algorithmic updates from already-released earlier results can be part of a frozen sequential evaluation, but human model/window selection on those same outcomes changes the evidence into development. Prespecify a calendar horizon and minimum information requirements without promising every sparse category will have adequate power.

If the practical target population prioritizes major leagues or international tournaments, specify its weights in advance and report a separate target-mixture proper score. Do not let the large regional population silently define that objective, and do not alter canonical benchmark weights after seeing which model wins. Event-category calibration still does not validate team-affiliation transfer or tournament joint probabilities.

## 9. Contradictions and limits resolved

| Apparent contradiction | Resolution |
|---|---|
| Temperature works in influential papers but fails to improve A0 | Their base networks and datasets differ; A0 is already internally calibrated. [3] |
| ECE is useful but not a selection objective | It diagnoses one aspect; proper scores reward both calibration and useful discrimination. [1,2] |
| Same-sample isotonic fitting cannot worsen its optimized proper score, yet may hurt future forecasts | It is optimized on those labels; prospective fitting variance is a separate problem. [5,11] |
| Partial pooling is principled but not proven here | It targets variance under a similarity assumption; the relevant signal may be absent. [14] |
| Multicalibration has guarantees but small groups remain uncertain | Guarantees have group-mass, complexity, and sampling requirements. [15,18] |
| Focal loss can help empirical ECE but is not a proper posterior score | Finite-network regularization and the population probability target are distinct. [28,29] |
| Online/conformal procedures tolerate shift but do not certify each match's probability | Their guarantees concern specified sequential losses/calibration or set-coverage contracts. [21-23] |

Two source-track target descriptions incorrectly called the canonical forecast map-level, and one misreported full international coverage as 110 rather than 282. They were rejected against the saved series contract and cohort evidence; raw tracks are retained for provenance, not treated as authoritative. The synthesized target and all calculations remain series-level.

**What remains unknown:** the true conditional calibration curve, whether any category effect persists prospectively, which missing pre-match information would be useful, and whether partial pooling improves future A0. None of these is settled by an IID simulation, a literature theorem with different assumptions, or an in-sample fit.

## 10. Verification and reproducibility

The diagnostic protocol was saved before the new decomposition/simulation calculations. It did not make previously inspected outcomes prospective. The executable diagnostic ran **24 observed isotonic fits** and **14,000 independent-label reference fits**. Five input hashes, including the canonical manifest and previous saved predictions, were preserved.

An independent stack-based PAV implementation, plus scikit-learn isotonic fitting, checked **240 numerical values across 24 subsets**. Maximum score discrepancy was **2.50e−16**; maximum independent-fit discrepancy **1.67e−16**. Side-reversal and permutation discrepancies were **0.0**. Fourteen simulated raw-score expectations passed the Monte Carlo sanity check. Duplicate-score pooling was checked explicitly. The figure was visually inspected.

Separate algebraic smoke checks confirmed monotonicity/complementarity for the proposed positive/odd-slope forms over 10,001 probability values. They are not a fitted-candidate evaluation. An unconstrained beta example violated complementarity by **0.1864**, illustrating the contract risk.

Reproduce from the repository root:

```bash
.venv/bin/python data/08_reporting/calibration_learning_research_20260927_192446/diagnose.py --protocol data/08_reporting/calibration_learning_research_20260927_192446/protocol.json
.venv/bin/python data/08_reporting/calibration_learning_research_20260927_192446/verify.py --directory data/08_reporting/calibration_learning_research_20260927_192446
```

Artifacts include `score_decomposition.csv`, `perfect_calibration_corp_reference.csv`, `calibration_cell_support_2026.csv`, `single_bin_sample_sizes.json`, `summary.json`, `verification.json`, raw research tracks, the source audit, and citation verification. Citation validation checks marker/reference integrity; source inspection and the evidence audit—not that script alone—support the claims.

**Final recommendation:** make calibration a well-measured conditional learning problem, not an ECE contest. Keep A0 unchanged now. Establish repeatable residual structure, then test one symmetry-preserving partial-pooling correction against strong controls. If that fails, improve available information or the base representation rather than repeatedly remapping the same probabilities.

## Bibliography

[1] Tilmann Gneiting and Adrian E. Raftery (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). JASA.

[2] Luciana Ferrer and Daniel Ramos (2024). [Evaluating Posterior Probabilities: Decision Theory, Proper Scoring Rules, and Calibration](https://arxiv.org/abs/2408.02841). Preprint.

[3] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML.

[4] Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, and Michael C. Mozer (2022). [Mitigating Bias in Calibration Error Estimation](https://proceedings.mlr.press/v151/roelofs22a.html). AISTATS.

[5] Timo Dimitriadis, Tilmann Gneiting, and Alexander I. Jordan (2021). Stable reliability diagrams for probabilistic classifiers. PNAS; author preprint titled [Evaluating probabilistic classifiers: Reliability diagrams and score decompositions revisited](https://arxiv.org/abs/2008.03033).

[6] Imanol Arrieta-Ibarra, Paman Gujral, Jonathan Tannen, Mark Tygert, and Cherie Xu (2022). [Metrics of Calibration for Probabilistic Predictions](https://jmlr.org/papers/v23/22-0658.html). JMLR 23(351).

[7] Jochen Bröcker (2022). [Uniform calibration tests for forecasting systems with small lead time](https://doi.org/10.1007/s11222-022-10144-9). Statistics and Computing 32:102.

[8] Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran (2023). [A Unifying Theory of Distance from Calibration](https://arxiv.org/abs/2211.16886). STOC; preprint first posted 2022.

[9] Futoshi Futami and Masahiro Fujisawa (2024). [Information-theoretic Generalization Analysis for Expected Calibration Error](https://arxiv.org/abs/2405.15709). Preprint.

[10] Yan Sun, Pratik Chaudhari, Ian J. Barnett, and Edgar Dobriban (2024). [A Confidence Interval for the ℓ₂ Expected Calibration Error](https://arxiv.org/abs/2408.08998). Preprint and subsequent revision accessed in 2026.

[11] Alexandru Niculescu-Mizil and Rich Caruana (2005). [Predicting Good Probabilities With Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf). ICML.

[12] Meelis Kull, Telmo Silva Filho, and Peter Flach (2017). [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html). AISTATS.

[13] Vladimir Vovk and Ivan Petej (2014). [Venn-Abers predictors](https://arxiv.org/abs/1211.0025). UAI; preprint first posted 2012.

[14] Edgar C. Merkle (2010). [Calibrating Subjective Probabilities Using Hierarchical Bayesian Models](https://ecmerkle.org/pub/Merkle2010.pdf). Social Computing, Behavioral Modeling, and Prediction.

[15] Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum (2018). [Multicalibration: Calibration for the (Computationally-Identifiable) Masses](https://proceedings.mlr.press/v80/hebert-johnson18a.html). ICML.

[16] Michael P. Kim, Amirata Ghorbani, and James Zou (2019). [Multiaccuracy: Black-Box Post-Processing for Fairness in Classification](https://arxiv.org/abs/1805.12317). AIES.

[17] Noam Barda, Gal Yona, Guy N. Rothblum, Philip Greenland, Morton Leibowitz, Ran Balicer, Eitan Bachmat, and Noa Dagan (2021). [Addressing bias in prediction models by improving subpopulation calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC7936516/). JAMIA 28(3):549–558.

[18] Dutch Hansen, Siddartha Devic, Preetum Nakkiran, and Vatsal Sharan (2024). [When is Multicalibration Post-Processing Necessary?](https://arxiv.org/abs/2406.06487). Preprint.

[19] Yaniv Ovadia et al. (2019). [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530). NeurIPS.

[20] Anusri Pampari and Stefano Ermon (2020). [Unsupervised Calibration under Covariate Shift](https://arxiv.org/abs/2006.16405). Preprint.

[21] Isaac Gibbs and Emmanuel Candès (2021). [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170). NeurIPS.

[22] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani (2021). [The limits of distribution-free conditional predictive inference](https://arxiv.org/abs/1903.04684). Annals of Statistics; preprint first posted 2019.

[23] Lunjia Hu, Kevin Tian, and Chutong Yang (2026). [Optimal Recalibration of an Online Predictor](https://arxiv.org/abs/2607.19689). July 2026 preprint.

[24] Aviral Kumar, Sunita Sarawagi, and Ujjwal Jain (2018). [Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings](https://proceedings.mlr.press/v80/kumar18a.html). ICML.

[25] Lucas Clarté, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová (2023). [Expectation consistency for calibration of neural networks](https://proceedings.mlr.press/v216/clarte23a.html). UAI.

[26] Rafael Müller, Simon Kornblith, and Geoffrey Hinton (2019). [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629). NeurIPS.

[27] Sunil Thulasidasan, Gopinath Chennupati, Jeff Bilmes, Tanmoy Bhattacharya, and Sarah Michalak (2019). [On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/36ad8b5f42db492827016448975cc22d-Paper.pdf). NeurIPS.

[28] Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H. S. Torr, and Puneet K. Dokania (2020). [Calibrating Deep Neural Networks Using Focal Loss](https://arxiv.org/abs/2002.09437). NeurIPS.

[29] Nontawat Charoenphakdee, Jayakorn Vongkulbhisal, Nuttapong Chairatanakul, and Masashi Sugiyama (2021). [On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective](https://arxiv.org/abs/2011.09172). CVPR; preprint first posted 2020.

[30] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474). NeurIPS.

[31] Alex Kendall and Yarin Gal (2017). [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977). NeurIPS.

[32] Tilmann Gneiting and Johannes Resin (2023). [Regression diagnostics meets forecast evaluation: conditional calibration, reliability diagrams, and coefficient of determination](https://arxiv.org/abs/2108.03210). Electronic Journal of Statistics 17(2):3226–3286.

[33] Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas B. Schön (2019). [Evaluating model calibration in classification](https://proceedings.mlr.press/v89/vaicenavicius19a.html). AISTATS.
