# Better LoL forecasts: what the current experiments get wrong

**Research date:** 2026-09-27; **authorized execution addendum:** 2026-10-01 (§10). Sections 1–9 preserve the original no-training investigation. Section 10 records three subsequent trained sports-model experiments; no production deployment.

**Current decision:** retain the learned single-weight connector as a **historical development candidate**, not a promoted model. It improves 2026 LL/Brier versus frozen A0 and its calibration-only control under the declared development gates. The first two recipes did not clear those gates. Canonical A0 remains unchanged; selected-bet reliability and independent confirmation are not established.

## Executive findings

1. **We changed much more than architecture.** The recent networks reuse A0's raw history tokens, but not its complete static feature contract, rating offset, scoreline objective, auxiliary player-performance objective, calibration or fitting schedule. Their failure is a valid negative result for those complete recipes—not an identified failure of attention, fusion or neural modeling.
2. **A0's advantage is already present in its native history-model forecasts.** Reconstructing its saved components reproduced all **11,550 full A0 forecasts exactly**. The three-seed history group scores **LL 0.551298**, almost the full system's **0.551246**. Individual seeds under the existing shared calibration score **0.551344–0.552695**, versus **0.562094** for the fresh separate-encoder model. More experts or more seeds alone is not a sufficient explanation.
3. **We have overemphasized spectacular failures.** Matches on which neither model has LL ≥ 2.5 account for **98.21%** of the separate-encoder model's net loss gap and **87.85%** of the connector's gap. Improving ordinary probability estimates matters more than moving a few losses below a threshold.
4. **The new forecasts contain a locally useful correction direction, but a deployable improvement is unproven.** Moving infinitesimally from A0 toward the fresh neural forecasts reduces historical mean loss. This is compatible with complementarity; it could also be calibration. A learned correction must beat an equally credible A0-only calibration control, not merely unmodified A0.
5. **Research validity is a separate bottleneck.** The canonical 2024–2026 outcomes have repeatedly informed development. Monthly bootstrap intervals do not reverse that exposure. Realized first-map rosters and daily release dates are scenarios, not proof of information available at a live forecast cutoff.

**Limits:** neither the original investigation nor the later execution identifies a unique cause of the model gap, establishes bookmaker superiority or financial reliability, or validates tournament distributions. The later predictive gain is exposed, adaptively developed historical evidence—not independent confirmation.

## 1. Scope, evidence and method

The question is how to improve **pre-match probabilities**, while retaining player-based generalization to new rosters and national teams. Future draft and map side are not ordinary pre-match inputs; W20 and organization history remain optional.

Six parallel research tracks covered native-versus-fresh recipes, raw-data/rating contracts, tabular/history representations, learned combinations, sports skill models, and evaluation methodology. The synthesis uses **34 selected primary sources**, including newer TabICL and TabPFN-2.5 evidence rather than treating older foundation-model limits as current. Some sources were available/read only at abstract level; their use is restricted to those claims. Source access depth and original track findings are retained in the evidence directory.

Three numerical investigations ran without fitting a model:

- **Error and compatibility diagnosis:** aligned saved forecasts, conditional loss accounting, high-confidence calibration accounting, and directional derivatives at A0. The protocol was saved before these calculations.
- **Native A0 decomposition:** a separately frozen addendum reconstructed the three calibrated expert groups and the final mixture from saved logits, gates and calibration records.
- **Seed decomposition:** a further pre-calculation addendum applied the existing jointly fitted history calibration to each native seed. No seed was selected or separately recalibrated.

The main cohort is **11,550 matches, 2024-01-01 through 2026-09-08**, spanning **33 calendar months**. Calculations use a common probability policy, ε = 10⁻⁶, and **5,000 paired monthly-block resamples**, seed 20260927. Reported intervals are descriptive historical uncertainty conditional on these saved predictions, not multiplicity-adjusted confirmatory tests. Match-weighted and equal-month quantities are distinct estimands.

Evidence directory, repository-relative:

`data/08_reporting/model_improvement_research_20260927_174646/`

The canonical manifest, pinned research inputs and frozen model artifacts were not changed. The research archive is resolved through `data/research_root.txt` or the established configuration, not a hardcoded path in reusable model code. Local evidence files can contain resolved provenance paths; they are not portable distribution packages.

### Evidence labels

- **Observed:** computed from saved forecasts, or directly established by source/artifact inspection.
- **[INFERENCE]:** interpretation, transfer argument or untested explanation.
- **Proposed:** an experiment not executed in this research pass.

## 2. What actually lost—and what the result establishes

All rows below use the **same 11,550 matches**. “Temperature” denotes the stored calibrated variant, not a post-hoc recalibration in this investigation.

| Forecast | Log loss | Brier | LL ≥ 2.5 |
|---|---:|---:|---:|
| Annual Causal A0 | **0.551246** | **0.187103** | 33 |
| Fresh input fusion, temperature | 0.566403 | 0.193276 | 41 |
| Fresh separate encoders, temperature | 0.562094 | 0.191364 | 34 |
| Learned specialist connector | 0.557196 | 0.189308 | 44 |
| Fresh LightGBM-15, temperature | 0.560462 | 0.190484 | 32 |
| Fresh player Gaussian-BT, temperature | 0.587247 | 0.201390 | 40 |

The connector does improve over its fresh components, but remains worse than A0: **ΔLL +0.005950**, monthly interval **[+0.003818, +0.007633]**. These results were already recorded by the architecture experiment; this report does not relabel that experiment as a new test.

Separate encoders beat input fusion in the match-weighted comparison: **ΔLL −0.004310**, monthly interval **[−0.007553, −0.000943]**. However, the equal-month difference is **+0.001568**. Also, the models have **21,697 versus 18,529 parameters**, respectively. The result is about complete candidate designs and weighting—not proof that moving the fusion point is beneficial at equal capacity, or that a typical month improved.

Both new fusion networks and the learned connector also lose to A0 in each reported year, BO1/3/5 and history-support stratum. Those descriptive slices do not identify the causal mechanism. The source-level review found no new reproducible masking, alignment or symmetry defect responsible for the remaining statistical gap.

The earlier **75% A0 + 25% corrected fresh ensemble** diagnostic is a different study on **3,103 2026 matches**: LL **0.535472** versus **0.536757**. Its monthly interval favored the blend, while source-tournament sensitivity included zero. It is neither a learned architecture nor independent confirmation. We do not recommend another fixed average in place of the requested learned integration.

## 3. The central comparison error: transferring tokens is not transferring A0

### 3.1 Recipe differences verified in source

| Dimension | Native annual A0 | Fresh history-fusion candidates |
|---|---|---|
| System | Seven fitted base checkpoints, grouped into three experts: three history seeds, one rating/logistic model, three full MLP seeds | One fresh neural seed per origin/variant; a separate connector combines fresh representation, tree and rating forecasts |
| History | Last 16 maps/player, 33 channels; nonlinear token projection, scalar content attention, learned empty sentinel | Same identity-aligned raw token bank, but a different static/history integration |
| Player/team encoding | History added residually to player encoding; team mean/SD aggregation | Five explicit role slots and shared antisymmetric pair scoring |
| Strength anchor | Learned roster/rating baseline enters as a map-logit offset | Fresh rating summaries/probabilities are learned inputs, not inherited A0 parameters |
| Neural supervision | Beta terminal-score likelihood **plus auxiliary player-performance loss** | Recency-weighted binary cross-entropy for the series winner |
| Fitting | Annual models; TRAIN before the preceding calendar year, that preceding year used for CAL | Quarterly refits; TRAIN closes six months before origin, followed by six months of CAL |
| Neural optimizer recipe | Three fixed seeds; 12 epochs; batch 128; AdamW, weight decay 0.01; gradient clipping | One configured seed; 24 epochs; batch 512; AdamW, weight decay 0.001 |
| Final probabilities | Joint scale/dependence calibration for history seeds, calibrated other experts, chronological cross-fitted mixture | Temperature scaling; connector learns a regularized, no-intercept combination from previous out-of-time expert predictions |
| Optional inputs | Separate full/no-W20/no-organization calibration and mixture paths | Different fresh feature and calibration contract |

Local implementation anchors:

- `src/models/prematch_history_fusion.py:245–314`: actual fresh fusion training loop and binary loss.
- `src/models/prematch_neural.py`: static encoders and antisymmetric scoring.
- Archive-relative `a0-phase-walkforward-20260914/code/causal_predictor.py:13–24`: actual annual artifact loading.
- Archive-relative `a0-phase-walkforward-20260914/code/causal_nn.py:14–78`: annual membership, preprocessing and neural fits.
- Archive-relative `nonlinear-history-20260912/code/src/models/beta_training.py:26–55`: terminal-score and auxiliary losses, gradient clipping and retained training diagnostics.
- Archive-relative `a0-phase-walkforward-20260914/code/causal_calibrate.py:55–81`: chronological calibration and mixtures.

The fresh loop does not persist per-epoch train/validation curves. The native Beta-training class does retain training and auxiliary losses. **Undertraining of the fresh network is therefore not established.** Nor does “24 versus 12 epochs” establish a larger optimization budget: batch sizes and training populations differ.

**[INFERENCE] The methodological error is changing many ingredients and then explaining the loss as an architecture verdict.** A0 has already tested several ideas that can sound like novel remedies: richer scoreline supervision, auxiliary player tasks, a rating-anchored residual, and multi-expert combination. A proposal must identify what changes beyond those existing mechanisms.

### 3.2 Counterevidence: A0 is not winning merely by having more experts

The component audit used native probability formulas, saved annual logits and the already fitted calibration/mixture parameters. It checked IDs, dates, BO, labels and orientation, reversing **5,695** rows to match the fresh table. The reconstructed full mixture had **maximum absolute error 0.0** against stored A0.

| Frozen native forecast | Log loss | Brier | ΔLL versus full A0 |
|---|---:|---:|---:|
| Full A0, reconstructed | **0.551246** | **0.187103** | 0 |
| Three-seed history group | **0.551298** | 0.187215 | +0.000052 |
| Rating/logistic group | 0.562023 | 0.190988 | +0.010777 |
| Three-seed full-MLP group | 0.555647 | 0.188931 | +0.004401 |
| History seed 20260910, shared calibration | 0.551344 | 0.187250 | +0.000099 |
| History seed 20260911, shared calibration | 0.551892 | 0.187488 | +0.000647 |
| History seed 20260912, shared calibration | 0.552695 | 0.187727 | +0.001449 |

History-group minus full-A0 ΔLL is **+0.000052**, monthly interval **[−0.000415, +0.000565]**. The interval does not establish equivalence. It does show that the observed full-system advantage over the fresh representation model is already almost entirely present when we inspect the native history group: the latter comparison's point gap is **0.010796**.

The final full-mode mixture weights for history/rating/MLP were approximately **70.8/20.6/8.6%** in 2024, **84.0/7.6/8.4%** in 2025 and **79.6/8.4/12.0%** in 2026. They were already fitted; none was reoptimized here.

**Important limitation:** individual seeds use the history group's joint calibration. These are not independently optimized standalone seed models. Nevertheless, their recorded forecasts are all substantially better in point LL than the fresh neural variant. “Just add more seeds” is not a demonstrated solution to its roughly 0.011 loss gap.

**[INFERENCE] The native feature/strength representation, objective, optimization and calibration recipe deserve priority over another change of architecture name.** The decomposition narrows the explanation; it does not isolate which ingredient matters.

### 3.3 Optional-context results are scenarios, not causal ablations

| Stored A0 mode | LL | ΔLL versus full | Monthly interval |
|---|---:|---:|---:|
| No W20 | 0.551764 | +0.000518 | [−0.000439, +0.001632] |
| No organization **and no W20** | 0.552188 | +0.000942 | [−0.000131, +0.002269] |

These use the same underlying checkpoints with altered gates and **mode-specific calibration/mixture parameters**. In `roster_optional.clean`, disabling organization also disables W20. They are not models retrained without those features, and `no_organization` is not an organization-only removal. They do not identify causal feature value or demonstrate that the inputs are unnecessary.

Their point scores do establish that native A0's advantage does not simply disappear in its recorded optional-context modes. New-roster functionality must remain an explicit contract, not an assumed property of organization history.

### 3.4 Previous controlled experiments constrain the explanation

The archive already contains relevant ablations. These are **reported historical results, not reruns on today's corrected inputs**; their differences must not be added together as a causal decomposition of the current gap.

| Earlier comparison | Reported full-cohort ΔLL and monthly interval | What it establishes |
|---|---|---|
| Add auxiliary player-stat supervision | −0.000790 [−0.001212, −0.000372] | Positive historical evidence for a component omitted by the fresh fusion networks |
| Coherent map-head/offset redesign, winner-only arm versus earlier auxiliary mixture | −0.001989 [−0.003398, −0.000864] | Positive historical evidence for the combined parameterization change, not for scoreline labels alone |
| Terminal-score versus winner-only supervision, same coherent family | −0.000184 [−0.000645, +0.000308] | No established overall advantage from the extra scoreline target |
| Beta-dependent training versus IID training, same Beta calibration | −0.000043 [−0.000152, +0.000067] | No established overall benefit from changing that training likelihood |
| Nonlinear token projection versus the preceding linear projection | −0.000507 [−0.000963, −0.000068] | Historical full-cohort improvement; market/phase advancement criteria still failed |
| Later native five-seed recipe versus canonical A0 | −0.000102 [−0.000321, +0.000123] | No established gain from that larger ensemble; its member calibration differs from canonical A0 |

Sources, relative to the configured research archive: `auxiliary-player-20260912/WYNIKI.md`, `coherent-series-20260912/WYNIKI.md`, `beta-trained-series-20260912/WYNIKI.md`, `nonlinear-history-20260912/WYNIKI.md`. The later five-seed result is repository-relative: `data/08_reporting/a0_uncertainty/a0_uncertainty_20260926_222155/benchmark_seed_only/report.json`. Their hashes and extracted claims are retained in `historical_recipe_evidence.json`.

**[INFERENCE] Preserve the auxiliary task and coherent strength parameterization as the starting control; do not assume that richer scoreline labels explain the new models' deficit.** Repeating winner-versus-score or IID-versus-Beta training is not a novel first experiment. The evidence favors preserving the accumulated recipe and isolating another disputed component.

## 4. What the new error analysis says

### 4.1 The loss gap is mostly outside extreme failures

For each candidate, divide matches into four exhaustive groups using whether its loss and A0's loss are at least 2.5. Sum each group's signed contribution to the **full-cohort mean** loss difference.

| Candidate | Matches where neither loss is deep | Share of net gap from those matches |
|---|---:|---:|
| Input fusion | 11,494 | **94.12%** |
| Separate encoders | 11,498 | **98.21%** |
| Learned connector | 11,492 | **87.85%** |
| LightGBM-15 | 11,499 | 102.81% |
| Player Gaussian-BT | 11,494 | 97.69% |

A contribution above 100% means the remaining groups offset part of the gap; it is not an arithmetic error. These are outcome-defined accounting groups, not causal subpopulations.

![Signed loss-gap accounting](../data/08_reporting/model_improvement_research_20260927_174646/loss_gap_attribution.png)

On A0-confidence bins from 0.5–0.6 through 0.9–1.0, both the fresh representation model and the connector also have positive mean loss differences in every bin. This is not only a problem of the most confident favorites.

**Decision:** stop selecting changes primarily because they reduce the count above LL 2.5. Keep deep losses as a diagnostic, while optimizing and evaluating the complete proper score. Proper scoring rules reward honest full predictive distributions, not cosmetically convenient tail thresholds.[11]

### 4.2 “Everything is overconfident” is not supported

At each model's confidence ≥ 0.8, compare actual upsets with the sum of its predicted upset probabilities:

| Model | Selected matches | Actual upsets | Predicted upsets | Excess upset rate, monthly 95% interval |
|---|---:|---:|---:|---:|
| A0 | 3,135 | 333 | 370.91 | −1.21 pp [−2.23, −0.16] |
| Separate encoders | 3,181 | 403 | 387.91 | +0.47 pp [−0.73, +1.78] |
| Connector | 3,311 | 398 | 396.98 | +0.03 pp [−1.23, +1.37] |
| Player Gaussian-BT | 2,304 | 317 | 282.30 | +1.51 pp [+0.12, +3.07] |

These confidence-selected populations **differ across models**; this is not a matched accuracy comparison. They are not independent Bernoulli trials, so no iid binomial p-value is asserted.

A0's high-confidence group has fewer upsets than it predicts. The connector's aggregate accounting is close to expectation even though its overall loss is worse. Thus blanket confidence shrinkage is not an identified repair. Calibration and discrimination are different properties; good aggregate calibration can coexist with worse loss. Temperature or beta calibration can help in studied settings, but neither is universally adequate under changing distributions.[8][9][10]

### 4.3 Complementarity: test a correction, not another replacement

Let `p₀` be frozen A0, `pₖ` a frozen candidate, and `y` the outcome. For a probability-space direction,

\[
p_\alpha = p_0 + \alpha(p_k-p_0),\qquad
\left.\frac{d\ell}{d\alpha}\right|_0
=\frac{(p_0-y)(p_k-p_0)}{p_0(1-p_0)}.
\]

For a logit-space direction,

\[
z_\alpha=z_0+\alpha(z_k-z_0),\qquad
\left.\frac{d\ell}{d\alpha}\right|_0=(p_0-y)(z_k-z_0).
\]

A negative average means a sufficiently small positive move lowers loss on this historical sample. No weight was fitted, no optimum was searched, and no new combined-model score is being claimed.

| Candidate direction from A0 | Probability-space LL derivative | Logit-space LL derivative, monthly interval |
|---|---:|---:|
| Input fusion | −0.008147 | −0.008924 [−0.013962, −0.005236] |
| Separate encoders | −0.008574 | **−0.009072 [−0.012717, −0.005898]** |
| LightGBM-15 | −0.004450 | −0.005221 [−0.009059, −0.002001] |
| Player Gaussian-BT | +0.005917 | +0.002710 [−0.001045, +0.006284] |
| Learned connector | −0.005327 | **−0.005859 [−0.008654, −0.003618]** |

![Historical directional derivatives](../data/08_reporting/model_improvement_research_20260927_174646/a0_directional_derivatives.png)

Log-loss and Brier gradients passed finite-difference checks on all 11,550 interior rows. A separate direct monthly-resampling implementation reproduced **20 directional intervals and 80 estimates/bounds** to absolute tolerance 10⁻¹³.

**[INFERENCE] There is a justified small experiment here:** preserve A0 and learn whether these disagreement directions improve it. Stacking literature supports learning combinations from held-out predictions and scoring predictive distributions, but its ordinary cross-validation guarantees cannot simply be imported into a drifting, repeatedly examined sports benchmark.[12][13][14]

**Critical competing explanation:** a candidate can point toward a useful recalibration of A0 without carrying additional conditional information. The next test needs an A0-only calibration/context head with comparable flexibility and tuning budget. A negative derivative does not prove that a neural connector will learn a transferable correction.

## 5. What the literature changes—and what it does not

### 5.1 There is no evidence-based mandate for a bigger Transformer

Tabular comparisons do not produce a universal winner. Grinsztajn et al. find strong tree performance on their medium-sized benchmark; Gorishniy et al. establish strong simple neural baselines; McElfresh et al., across 176 datasets and 19 algorithms, show that tuning and dataset properties can matter more than the broad NN-versus-tree label.[16][17][18]

TabM is relevant because it offers a comparatively simple, parameter-efficient MLP-ensemble baseline, rather than assuming attention is necessary. Its inspected benchmark contains **46 datasets**, with **37 random and nine domain-aware splits**; classification headline metrics are accuracy/ROC-AUC, not uniformly pre-match log loss. It tunes on validation data and evaluates multiple seeds. That is a reason to include a fair modern baseline, not evidence it will beat A0.[19]

TabReD is particularly relevant counterevidence to naive benchmark transfer: its eight industrial datasets expose temporal drift and engineered-feature structure, and time-based splits change method rankings. Simple MLP-like models and GBDTs perform well there. LoL is not one of those datasets.[20]

**[INFERENCE] Our one-width, one-seed, fixed-epoch neural run does not settle neural architecture quality.** However, the native A0 seed reconstruction also means we should not assume seed averaging alone will close the gap. Seed-variation studies in NLP and deep-ensemble work justify measuring training variability, not copying a numerical gain from unrelated tasks.[6][7][15]

### 5.2 Keep the correct invariances without deleting time or role

Deep Sets and Set Transformer justify set-aware representations when the target is invariant to element order.[21][22] They do not say that role identity or event age is irrelevant.

- Arbitrary ordering of players in a file should not change the forecast.
- A support player and a top laner are not interchangeable without their role/context traveling with them.
- Historical map order, age and environment can matter. Content attention that receives age is **not** completely time-blind, but it lacks an explicit sequential-transition structure.
- A/B orientation must preserve `p(A,B) + p(B,A) ≈ 1`; this is not the same as knowing blue/red map side.

Both the native and fresh history attention are content-scored rather than explicitly queried by the current opponent/roster. **Opponent-conditioned retrieval has already been tried in this repository's research archive with inconsistent results.** Another query-conditioned encoder is a possible controlled hypothesis, not an untouched idea or the default next step.

### 5.3 Foundation models deserve a bounded test, not stale dismissal or automatic adoption

The 2025 TabPFN Nature study focuses on tables up to 10,000 samples.[23] That limit must not be presented as applying to the entire current family:

- **TabICL** reports scaling to 500,000 samples and comparisons on 200 TALENT classification datasets, including a larger-table subset.[24]
- **TabPFN-2.5** reports a design range up to 50,000 rows and 2,000 features; the inspected HTML revision is dated 2026-02-05.[25]

Those claims make the approximately 40,636-row feature bank a plausible *size* for investigation. They do not establish temporal LoL log-loss performance, local hardware feasibility, suitability for linked player histories, or commercial licensing. The TabPFN-2.5 paper describes a non-commercial license; check the exact weights and terms before any application use. No cloud upload of local sports data or dependency installation was performed.

XTab provides evidence for cross-table pretraining, but column heterogeneity and source-task relevance matter.[26] Synthetic/table pretraining is not the same thing as causally learning professional player skill. A0 already uses an auxiliary player task; “add auxiliary supervision” alone is not a novel proposal.

### 5.4 Player skill needs a measurement model, not more labels called “form”

TrueSkill, TrueSkill 2 and dynamic/state-space rating work motivate evolving latent skill, uncertainty and explicit observation models.[27][28][29][30][31] TrueSkill 2's strong published improvement is on Halo 5 matchmaking, with additional shooter-specific information—not a transferable LoL improvement estimate.

The SIDO paper is domain-relevant: gold and damage production are embedded in teammate, opponent, champion and game-state context. It studies player-performance measurement, not proof of superior cutoff-valid pre-match log loss.[32] Regularized adjusted plus-minus provides another example of the need for regularization and out-of-sample assessment when teammates share exposure.[34]

**[INFERENCE] Stable five-player lineups make individual credit assignment weakly identified.** If two players nearly always appear together, redistributing their inferred effects while preserving the lineup sum can leave outcome predictions nearly unchanged. Disconnected competition groups pose a related problem: priors impose assumptions; they do not create cross-group evidence.

The current fresh Gaussian-BT control already has dynamic uncertainty and actual map updates. It also integrates series probability over uncertain map strength. These are not missing features. Ignoring player covariance remains an approximation; observing that approximation does not prove it caused the neural model's loss gap.

### 5.5 Coherent series targets matter—but this is partly implemented already

For fixed iid map probability, the BO winner distribution follows a stopped/majority-series calculation.[33] Under shared uncertain match strength, generally `E[F_BO(p)] ≠ F_BO(E[p])`. Native A0's history recipe uses a Beta terminal-score model; the fresh Gaussian-BT reference integrates a Gaussian map-logit distribution. Neither should be described as simply projecting a mean map probability twice.

A separate **legacy family-calibrated Glicko bridge** does mix map-level probability with a majority-series outcome. That is a real issue to resolve before claiming a coherent regional-rating comparison. It is **not** an established explanation for the new fusion models' binary-series loss gap.

Finally, accurate match probabilities do not certify series-length, qualification, placement or champion distributions. Whole tournament editions, legal matchups, format oracles and transition uncertainty require their own evaluation.

## 6. What data and process limitations remain

### Confirmed limitations

1. **Absolute availability:** realized first-map lineups and daily-effective release bounds are explicit historical scenarios. Exact roster-announcement/publication timestamps are not certified by this benchmark. This can limit deployment validity without explaining the relative ranking of two models using the same scenario.
2. **Repeated outcome exposure:** 2024–2026 has been used to choose and reject ideas. Finite-sample selection criteria themselves can be overfit; a reused historical holdout is not made independent by another chronological split or bootstrap.[1][2]
3. **Recipe attribution:** the experiment changes static features, strength anchor, objective, role pooling, calibration and schedule at once relative to A0.
4. **Optimization observability:** no retained fresh per-epoch curves; one training seed does not characterize algorithm variability. This does not prove a convergence failure.
5. **Effective sample size:** 41,915 source series, 69,370 maps and 40,636 feature-bank targets are different units. Ten player records per map are not ten independent match outcomes; overlapping histories also share information. No defensible single effective sample size was estimated here.
6. **Input coverage:** historical detailed statistics have missingness and era/competition-dependent coverage. Current fresh static tensors have no explicit target patch/family/tier/transfer-event field; tournament labels remain metadata. This narrow observation does not establish that all such context is absent from every A0 input or historical experiment.

### Plausible, not demonstrated causes

| Hypothesis | Supporting evidence | What prevents a causal conclusion |
|---|---|---|
| [INFERENCE] Native feature/offset/objective recipe is stronger | Native history seeds already beat fresh neural forecasts in point LL | Multiple recipe factors differ simultaneously |
| [INFERENCE] Fresh models relearn redundant signal inefficiently | A0 already knows ratings, experience, inactivity, coplay and map history | No controlled representation/information ablation |
| [INFERENCE] Availability, era and competition shifts weaken transfer | Scenario limitations and absent explicit fresh context fields | No new context model trained; prior tier/roster corrections often failed |
| [INFERENCE] Calibration explains some apparent complementarity | Negative historical directions can arise without new information | No matched A0-only calibration control fitted in this pass |
| [INFERENCE] Credit assignment limits new-roster generalization | Shared outcomes and diagonal rating approximations | No covariance-controlled or transfer-specific experiment here |

Do not replace unknown causes with a confident story about “too little data,” “too much confidence,” “a weak optimizer,” or “not enough attention.” Learning-curve crossings and time-dependent evaluation studies reinforce why those generalizations need measured controls.[4][5]

## 7. Ranked experiments proposed on September 27

These were **proposals** in the original no-training research pass. Rank 1 was subsequently authorized and executed as the three separately frozen recipes in §10; do not resume it as unexecuted work. Other proposals remain separate, not an automatic queue. The execution uses the existing canonical benchmark and preserves frozen A0.

### Rank 1 — Learn an A0-anchored correction, with a calibration-only opponent

**Question:** do the fresh forecasts improve A0 beyond flexible recalibration?

Use frozen A0 as an identity starting point:

\[
z_0=\operatorname{logit}(p_{A0}),\quad d_k=\operatorname{logit}(p_k)-z_0,\quad
z_{final}=z_0+\delta_\theta(d,c).
\]

Here `c` contains only symmetric, cutoff-valid context. The correction must be antisymmetric under swapping teams, with coefficients regularized toward zero. Zero correction is exactly A0; a swapped input negates the correction. Begin with a small linear/context-interaction head, not a large unconstrained network.

**Controls:** untouched A0; an A0-only calibration/context head; the anchored head with one predeclared fresh signal block. Give the calibration-only comparison credible flexibility and the same selection budget. For a larger nonlinear head, include a parameter-matched A0-only head. Do not claim new information merely because an underpowered temperature control loses.

**Training data:** genuine earlier out-of-time forecasts from *every* expert, including A0, with labels released before each fitting origin. The 4,836 fresh warmup forecasts from 2023 do **not** automatically provide 4,836 honest A0 warmup forecasts. Reconstruct a causally valid earlier A0 recipe if needed, or keep the correction at its identity solution until valid historical training data exist; disclose that startup period and retain complete coverage. Never backfill warmup with a checkpoint trained on those outcomes.

**Bound:** one fixed correction family and at most four predeclared regularization settings selected on earlier development origins. No new network training is necessary merely to test this forecast-level hypothesis. If training a feature-level adapter later, count that as another experiment.

**Success:** lower LL than both A0 and its calibration-only control on later origins, acceptable Brier and full coverage, followed by new prospective confirmation. Report equal-month and source-label sensitivity rather than selecting the most favorable uncertainty analysis.

**Stop:** if the correction adds no benefit beyond A0-only calibration, reject the claim of useful incremental forecast signal for that head. Do not respond by immediately widening the connector.

This is a learned combination, **not another fixed 75/25 average**. The earlier failed chronological A1 macro residual and roster/form corrections remain relevant negative controls: residual architecture by itself is not the new hypothesis; the new hypothesis concerns these particular out-of-time disagreement signals.

### Rank 2 — Identify what the native history recipe retains and the fresh recipe loses

**Question:** how much does the explicit native rating-offset path contribute when the remaining recipe is held fixed?

First establish one-origin native training-recipe parity: identical released rows, raw/static inputs, rating offset, normalizers, initialization policy, model, schedule and calibration. Reuse the native-training infrastructure in `src/models/a0_uncertainty.py` and `scripts/train_a0_uncertainty.py`, rather than porting the archive again; its existing K=5/member-calibration recipe is not an exact canonical-parity control. The component replay in this report establishes **inference replay**, not a parity retrain.

Then predeclare one factor:

- control: unchanged native map-logit offset, terminal-score likelihood and auxiliary player loss;
- variant: set the explicit map-logit offset to zero during both training and inference, retaining all other input tensors, losses, BO handling, optimization and calibration.

This measures the contribution of the implemented baseline path, including information available only through that path; it is **not** an equal-information architecture comparison. Use the same three predeclared seeds, report individual and aggregate forecasts, and retain train/validation curves and optimizer-step counts. A material deficit would justify transferring the native strength-anchor contract into the fresh design in a separately declared experiment.

Do not substitute another default winner/score or IID/Beta loss comparison: those have already been tested with small, inconclusive overall effects (§3.4). A repeat needs a specific new hypothesis about the corrected inputs or interaction, not the claim that A0 lacks richer supervision.

**Bound:** one parity origin before a full replay; then control plus one factor, not a factorial search over loss, architecture, region and calibration.

**Success:** a stable paired difference that survives the seed comparison and chronological development windows, with complete coverage and invariants. If the native recipe already cannot be reproduced, resolve that before interpreting architectural changes.

**Stop:** if the factor is unsupported, retain the result and move to a different evidence-backed factor. Do not relabel the same BCE/history model under a new architecture name.

### Rank 3 — Add demonstrably new, available information

**Question:** can a new information block—not a renamed existing summary—improve the preserved control?

Prioritize a provenance/coverage inventory before another learner. Possible blocks include historically known patch/environment context, explicit transfer/role-change events, and competition-family information with defensible affiliation timing. Add only one block to the same retained model/head.

For a regional-rating variant, follow the existing `REGION_PLAN.md`: identical player-update semantics and target-roster scenario in control and regional versions; coherent series likelihood; historical substitutes, mixed affiliations, inactivity and transfers explicitly handled. Competition family is not nationality. A common competition label can modulate calibration/context without automatically identifying player affiliation strength; these are different hypotheses.

For uncertain rosters, distinguish probabilities conditional on a specified lineup from a forecast marginalized over lineup scenarios known at the cutoff. A newly observed future five is not a legal way to improve an earlier forecast.

**Success:** incremental loss improvement beyond the same control, valid coverage, and benefit in predeclared transfer/cold-start slices without sacrificing the main cohort. Small cross-region samples limit what can be concluded.

**Stop:** missing historical availability or another repackaging of a rejected roster/form/tier correction. No team-name exceptions, future draft information or hindsight affiliations.

### Rank 4 — One fair modern tabular baseline, not an architecture tournament

A TabM-style baseline is a reasonable first modern comparator. TabICL/TabPFN-2.5 are additional bounded research options once local feasibility, weight provenance and licensing are understood.[19][24][25]

Use the same cutoff-valid static inputs and chronological protocol, enforce side symmetry, select on earlier origins, and score LL/Brier—not only AUC. A static-table comparison is not a controlled replacement of A0's history encoder. Freeze preprocessing, categorical/role treatment, capacity, seed count and calibration before evaluation. Do not claim a cross-domain benchmark win transfers to professional LoL.

**Stop:** no development improvement under the frozen comparison; do not broaden immediately to dozens of foundation models or an unconstrained retrieval/GNN search.

## 8. Evaluation and stopping contract

### Development versus confirmation

Use expanding/rolling chronological origins for development; preprocessing, ratings, neural fitting and calibration each need their own cutoff.[3][4] Shuffling minibatches *inside closed TRAIN* is legal. Reordering events that update historical ratings is not.

The canonical cohort remains useful for reproducible diagnosis. A new candidate run on it is a **new development comparison**, not a new independent test. Confirmation requires outcomes not already used in this research process—preferably a prospectively registered, append-only prediction cohort with exact data/roster availability records.

### Proposed decision rule—not a literature constant

Before running the next candidate, choose a minimum worthwhile effect and uncertainty policy. A concrete proposal is:

- target **ΔLL ≤ −0.001** versus the relevant control;
- require a predeclared paired interval below zero for confirmation;
- no material Brier deterioration, using a margin agreed in advance rather than invented after results;
- complete eligible coverage or explicit, separately assessed failure counts;
- no broken chronology, state immutability, side symmetry, cold start or missing-context behavior.

The 0.001 target is a **project proposal**, not a universal production threshold. Estimate required prospective duration from observed block variability and that target; do not declare success from a handful of favorable months. A predeclared fixed analysis date or valid sequential rule is preferable to repeatedly checking until an interval becomes favorable.

Monthly resampling reflects the observed distribution of months and their sizes. It does not refit networks, account for architecture selection, or model every dependency between adjacent months and repeated teams. Report seed variability separately; examine temporal-block sensitivity. Source tournament labels are not automatically independent editions. Simulation confirmation must block whole verified editions and check joint outcome constraints.

### Keep the negative-result registry active

Do not blindly repeat:

- roster-weighted form, inactivity and organization corrections from `roster-form-20260915`;
- opponent-conditioned retrieval from `opponent-history-20260912`;
- earlier player graph/pair and lineup-smoothing variants;
- raw recency/no-CAL-gap variants already scored on 2026;
- the failed chronological A1 macro residual;
- temperature/profile changes selected primarily to reduce a deep-loss count.

An actually different hypothesis can revisit a mechanism, but must specify the changed information, objective or control. A new name does not erase prior evidence.

## 9. Verification, artifacts and limitations

### What actually ran

From the repository root:

```bash
.venv/bin/python data/08_reporting/model_improvement_research_20260927_174646/diagnostics.py --protocol data/08_reporting/model_improvement_research_20260927_174646/diagnostic_protocol.json
.venv/bin/python data/08_reporting/model_improvement_research_20260927_174646/independent_audit.py --directory data/08_reporting/model_improvement_research_20260927_174646
.venv/bin/python data/08_reporting/model_improvement_research_20260927_174646/a0_components.py --directory data/08_reporting/model_improvement_research_20260927_174646
.venv/bin/python data/08_reporting/model_improvement_research_20260927_174646/plots.py --directory data/08_reporting/model_improvement_research_20260927_174646
.venv/bin/python data/08_reporting/model_improvement_research_20260927_174646/verify_report.py --directory data/08_reporting/model_improvement_research_20260927_174646
```

Observed verification:

- Six main-input hashes checked; all 11,550 rows aligned; no models fitted or weights selected.
- LL/Brier directional finite differences passed on 11,550 rows.
- Five gap decompositions and 20 intervals independently checked; 80 numerical estimates/bounds agreed to absolute tolerance 10⁻¹³.
- Fourteen component-audit inputs pinned; native full-A0 replay **exact on all 11,550 rows** after explicit orientation handling.
- Mean of the three reconstructed seed probabilities matched the history group within **3.34 × 10⁻¹⁶**.
- Both generated figures visually inspected; crowded gradient-axis labels corrected.
- Citation-structure verification passed: all 34 cited sources have bibliography entries, with no unused entries or invalid/placeholder URL syntax. This check does not independently validate every source's scientific claim.

Key files in the evidence directory:

| Artifact | Purpose |
|---|---|
| `diagnostic_protocol.json`, `diagnostic_results.json` | Main scope, input hashes, calibration accounting and checks |
| `loss_gap_attribution.csv` | Exhaustive signed gap accounting |
| `conditional_loss_profiles.csv`, `decision_disagreements.csv` | Descriptive slice and decision differences |
| `a0_directional_derivatives.csv` | All LL/Brier, probability/logit directions and block intervals |
| `probability_correlations.csv`, `probability_error_correlations.csv` | Supporting dependence diagnostics; not independent evidence of incrementality |
| `independent_audit.py`, `independent_audit.json` | Separate arithmetic/resampling oracle |
| `a0_component_protocol.json`, `a0_seed_protocol.json` | Frozen addenda before component/seed calculations |
| `a0_components.py`, `a0_component_predictions.parquet`, `a0_component_results.json` | Native frozen component reconstruction, weights and scores |
| `a0_component_verification.json` | Persisted mixture and seed-mean checks |
| `historical_recipe_evidence.json` | Hashes and reported claims from earlier controlled experiments; not new reruns |
| `verify_report.py`, `report_verification.json` | Citation/link structure, input preservation and report-to-metric checks |
| `source_ledger.json`, `tracks/` | Selected primary sources, access depth, raw research findings and corrected bibliography metadata |

The scripts are reproducibility artifacts for this diagnosis, not a second production benchmark. Future candidate evaluation belongs in the existing benchmark runner and versioned experiment pipeline.

### Material limitations

- All numerical findings are post-hoc exposed-history development. No new prospective labels were inspected or evaluated.
- Bootstrap intervals here are not adjusted for the entire history of candidate selection or all inspected diagnostics.
- Native seed forecasts retain joint calibration; optional-input scenarios retain mode-specific calibrations/mixtures. Neither is a causal retraining ablation.
- This investigation did not establish convergence of the fresh neural fits, relative value of each training loss, causal patch/region effects, or a prospective gain from a learned correction.
- The earlier architecture run's initial replay discrepancy up to **4.88155 × 10⁻⁶** remains unexplained. Exact reconstruction of **stored A0 component logits** here does not resolve that separate fresh-network runtime caveat.
- No full application test suite, new model fit, live scraping, financial simulation, tournament evaluation, migration or deployment ran in this pass. Existing application/test results remain separate evidence.
- The live engine was not inspected again. Research A0 must not be confused with the separately identified operational EXP081-based hybrid.

## Conclusion

The following is the original September 27 research conclusion; the subsequently authorized execution and current decision are recorded in §10.

**The next model should preserve the strong native probability mechanism and earn each additional degree of freedom.** The fresh experiments show that replacing it with a broadly similar token encoder loses useful predictive performance; they do not show that a larger network is needed.

The most direct improvement hypothesis is a **learned, shrinkage-controlled A0 extension that defeats an A0-only calibration control on chronological forecasts**. The most informative mechanism experiment is a **native-recipe, one-factor rating-offset ablation**. Better timestamped context is a separate information project. Modern tabular models are bounded comparators, not substitutes for those controls.

A0 remains the research reference. No candidate is promoted by this report.

## 10. Executed learned sports connectors — 2026-10-01

### Decision and scope

The user authorized continued historical model improvement. Three separately frozen recipes were executed; **the third clears the declared predictive development gates**. No odds enter any fit. This does not reopen the exhausted betting-EV loop, clear G0 for a sports-plus-market head, or authorize deployment.

All artifacts use `data/{05_model_input,06_models,07_model_output,08_reporting}/a0_sports_correction/<run>/`. Each run has its own immutable design, pre-fit execution/source hashes, saved heads, forecasts, standard canonical benchmark, numerical/checkpoint verification and final `decision.json`:

| Run | Change | Primary assessment | Result |
|---|---|---|---|
| `a0_sports_correction_20261001_01` | BO-specific anchored logit correction using the frozen separate-history encoder; equally selected A0-only calibration control | 6,908 matches, 2025–2026, 21 months | Neither primary LL interval excludes zero; not adopted |
| `a0_fresh_correction_20261001_02` | Same correction family, using the corrected raw logistic/tree/role-network specialist | 3,103 matches, 2026, nine months | Beats calibration-only control, but not unchanged A0 under the primary interval gate; not adopted |
| `a0_fixed_calibration_connector_20261001_03` | One learned convex probability weight; no additional A0 recalibration | Same 3,103 matches | Both LL intervals below zero, Brier improves, full coverage and four guards pass; retained for independent confirmation |

The third hypothesis was motivated after seeing the first two results. **These are three adaptive development recipes, not three independent confirmations.** Every recipe and negative result is retained. Nominal bootstrap intervals are not adjusted for this search or earlier exposure of 2026. No additional recipe was launched after the third result.

### Training, selection and the saved model

Run01 uses an odd, identity-centered, BO1/3/5-specific piecewise-logit correction with knot `logit(0.8)`, plus a nonnegative specialist-disagreement coefficient. Its A0-only opponent removes only that disagreement term. Four fixed penalties `[0.0001, 0.001, 0.01, 0.1]` are selected on earlier rolling-origin predictions; ties within `1e-12` choose stronger regularization. Run02 retains this family and control but changes the specialist.

The stronger specialist is the existing corrected raw equal-weight trio: `logistic_c01`, `lgbm_15`, `roster_32`. Run02 makes **24 genuine earlier base fits**—three members at eight quarterly 2024–2025 origins—and reuses the nine existing 2026 member checkpoints. The existing helper also fits 24 temperatures, **unused by this raw specialist**. TRAIN uses the inherited 730-day weights and seed `20260927`; six months remain reserved for CAL. A0 weights, calibration and mixture are untouched.

The trio design was originally selected using 2024–2025H1. Its newly generated earlier forecasts are fixed-architecture, weight-out-of-time inputs **inside a closed head-development partition**, not globally selection-unseen historical tests. Consequently runs02/03 leave **all 8,447 rows from 2024–2025 exactly equal to A0** and assess predictive gain only on the 3,103 later rows. Run01 instead has 4,642 identity-warmup rows in 2024. Actual first-map rosters and daily release bounds remain scenarios, not certified announcements or forecast issuance.

The retained connector learns

```text
p_final = (1 − alpha) * p_A0 + alpha * p_fixed_raw_families
objective = mean Bernoulli log loss + 0.5 * lambda * alpha²
0 <= alpha <= 1
```

This is a **learned specialist connector**, not another fixed 75/25 average. It uses the same four-penalty grid and past-only selection, leaves native A0 calibration intact, and consumes already projected **series-win** probabilities. There is no second BO projection or new clipping. It is a whole-connector comparison: reducing parameter count and changing logit/probability geometry means it does not isolate recalibration as the cause of earlier failures.

| Forecast origin | Selected lambda | Learned specialist weight |
|---|---:|---:|
| 2026-01-01 | 0.001 | 0.156353902507 |
| 2026-04-01 | 0.0001 | 0.178831378663 |
| 2026-07-01 | 0.0001 | 0.196613511594 |

The three runs require **180 head fits: 72 + 72 + 36**, in addition to run02's 24 new base fits and 24 unused temperature fits. Run03 adds no base refits and reuses run02's A0-only control exactly. Numerical verification refits and synthetic tests are separate from these training counts.

The usable July head is:

`data/06_models/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/2026-07-01/selected_candidate.json`

An exercised, read-only inference command from the repository root:

```bash
.venv/bin/python -m src.models.a0_probability_connector predict \
  --artifact data/06_models/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/2026-07-01/selected_candidate.json \
  --inputs data/08_reporting/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/smoke_inputs.json
```

Inputs contain matched, same-side `p_a0`, `p_specialist`, `date` and `best_of` arrays. The caller must generate the genuine base probabilities from aligned historical features; the CLI does not reconstruct rosters or certify source availability. Native and fresh backends were exercised in separate processes because of their namespace contracts. This artifact covers **2026-07-01 through 2026-09-30**, not a new Q4/future model. Only full-context winner probabilities are evaluated; optional-context modes, scoreline distributions and a live feature adapter are not newly certified.

### Matched predictive results and uncertainty

| Primary cohort / model | N | Log loss | Brier |
|---|---:|---:|---:|
| Run01 A0, 2025–2026 | 6,908 | 0.542502061 | 0.183706301 |
| Run01 candidate | 6,908 | 0.542017916 | 0.183462540 |
| Runs02/03 A0, 2026 | 3,103 | 0.536756581 | 0.181572637 |
| Runs02/03 A0-only calibration control | 3,103 | 0.537109868 | 0.181655590 |
| Run02 candidate | 3,103 | 0.536315909 | 0.181302703 |
| **Run03 retained connector** | **3,103** | **0.535644479** | **0.181012213** |

Run01 candidate-minus-A0 ΔLL is **−0.000484144 [−0.001475887, +0.000555646]**; versus its calibration control, **−0.000573780 [−0.001432866, +0.000318027]**. Run02 versus A0 is **−0.000440671 [−0.001477403, +0.000511112]**; versus its control, **−0.000793959 [−0.001421433, −0.000333618]**. Neither result clears its declared two-comparator gate. These failures are not equivalence findings.

For run03, 5,000 paired monthly draws over **nine calendar months**, seed `20261001`, give:

| Contrast | Match-weighted difference | 95% interval | Equal-month difference | Equal-month 95% interval |
|---|---:|---:|---:|---:|
| LL, connector − A0 | −0.001112102 | [−0.001834402, −0.000419645] | −0.001653825 | [−0.003350526, −0.000454470] |
| LL, connector − calibration control | −0.001465389 | [−0.002318951, −0.000656503] | −0.002091503 | [−0.003679267, −0.000877345] |
| Brier, connector − A0 | −0.000560423 | [−0.000870316, −0.000326550] | −0.000840361 | [−0.001664657, −0.000307498] |

Source-tournament-name/year sensitivity has **143 proxy blocks**, not verified independent editions. Its match-weighted LL intervals also exclude zero: versus A0 **[−0.002102319, −0.000189384]**, versus control **[−0.002721803, −0.000235318]**. The corresponding **equal-proxy-block intervals cross zero**; do not claim every estimand is robust.

All four predefined replacement guards have monthly LL upper bounds below `+0.005` against both controls: BO3 **N=1,873**, BO5 **485**, major **704**, source-unseen-player **334**. The largest checked upper bound is BO5 versus A0, **+0.003570345**. These guards are bounded noninferiority checks, not proof of improvement in every subgroup.

All **11,550** canonical rows remain available. Full-cohort A0/connector LL is **0.551245583 / 0.550946808**, Brier **0.187102933 / 0.186952371**; this includes 8,447 unchanged warmup rows, not 11,550 fresh assessment observations. Each run was scored through `scripts/run_model_benchmark.py --suite`, with status `DEVELOPMENT_ONLY`. This is not a claim to beat every historical recipe or the separately assembled Plan08 C0.

### Calibration and selected-EV limits

Run03's 2026 ECE10 improves pointwise **2.204653% → 2.069664%**, accuracy **71.833709% → 71.962617%**. Deep losses remain **10 → 10** in 2026 and **33 → 33** overall. However, fixed-A0 quarter/confidence-cell RMS **worsens slightly: 2.971577% → 3.001789%**, on 15 supported cells, 3,099 nonneutral rows plus four excluded neutral rows. Full-cohort ECE also rises slightly, **1.153374% → 1.165502%**. Do not call this a uniform calibration improvement or prescribe universal flattening.

The separately frozen market diagnostic retains all **11,973** source rows with exclusion reasons and admits **1,619** common 2026 fixtures across nine months: Pinnacle T−1h, strict-prior-day feature scenario, both quote sides no older than 24 hours, before recorded start. Prices, forecast/quote timing and side identities are aligned as an archival scenario, **not certified bookmaker bindings, observation clocks or execution**. It is not a replay of the old 387 prior-context selections.

The research policy chooses at most one side at raw after-12%-tax EV strictly above 5%; fixed A0 selections and each model's own selections are reported separately. These are unit-return diagnostics, not a bankroll/Kelly simulation or the actual application's recommendation policy.

| Same fixed A0 selections | N | A0 return overstatement | Run01 | Run02 | Run03 |
|---|---:|---:|---:|---:|---:|
| All EV >5% | 328 | 46.3061 pp | 43.1819 pp | 44.3232 pp | **44.8558 pp** |
| Original (5%,10%] EV band | 74 | 28.5020 pp | 26.8473 pp | 27.5758 pp | **27.5798 pp** |

For the retained model, paired error changes are **−1.450279 pp [−1.887753, −1.106142]** on the fixed 328 and **−0.922212 pp [−1.673291, −0.318662]** on the fixed 74. Realized returns cannot change at fixed IDs/sides/prices: they remain **−9.0598%** and **−21.3768%** respectively. Reduced predicted EV is not improved realized profit.

The remaining run03 overstatement intervals are **[+27.7764, +66.8265] pp** overall and **[+1.3317, +47.6348] pp** in the fixed near-threshold band—nowhere near the unchanged ±2 pp interval-containment target. Its own policy selects **320** bets with mean realized return **−8.451950%** and overstatement **45.4865 pp**; its own near-threshold set has **70** bets and error **26.8248 pp**. **Betting reliability remains unmet.** Earlier recipes reduce this diagnostic error more but fail the predictive gate; no financial-diagnostic reselection was performed.

On the same 1,619 fixtures, A0/connector/Pinnacle LL is **0.574880713 / 0.574151013 / 0.564260246**. Connector-minus-A0 **−0.000729701 [−0.001885071, +0.000151186]** is inconclusive on this market subset. Connector-minus-Pinnacle is **+0.009890767 [−0.000732073, +0.021555733]**. Neither the primary gain nor selected-EV reduction establishes bookmaker superiority.

### Verification and evidence boundaries

- **83 distinct focused tests passed**, with the existing Starlette/httpx warning: 79 integrated model/benchmark/history tests, one market-orientation/payoff regression and three convex-connector edge/optimum tests. No full application-suite or database claim.
- Real training/evaluation CLIs ran for all three recipes. Saved heads replay all 11,550 output rows per run; run03 is exact from its saved base probabilities. Actual prediction CLIs ran on six real-derived BO1/3/5, both-orientation inputs for each recipe.
- Run02 independently replays **11,550 raw-specialist rows from 33 actual checkpoints**, maximum error **1.460982e−8**. Genuine native and fresh inference covers **all 3,103 primary rows**, maximum errors **6.571312e−9 / 8.788428e−9**. Run03 reuses those authenticated base outputs, verifies equality to its own inputs and composes its actual saved heads: maximum final error **5.279e−9**. This is not a second independent base replay or a bitwise-runtime claim.
- Side symmetry, input nonmutation, BO support, synthetic neutral/cold behavior and genuine cold/new-roster examples are exercised. Empty history with nonzero static information need not produce exactly 0.5. The old unexplained `4.88155e−6` fresh-runtime caveat remains historical evidence; the current smaller replay error does not explain it.
- Independent fit oracles check **72 + 72 + 36** objectives, bounds and gradients. Run03's 36 convex roots match exactly; objective error is at most **1.11e−16**, projected gradient **1.06e−16**. Its oracle reproduces **32,044** validation-grid probabilities, **12** selection losses, three selections and **249,764** past-only membership occurrences. Representative future perturbations pass; not every origin/selection sequence was independently perturbed.
- Primary/all LL and Brier, monthly/source-proxy intervals, fixed-A0 cells, run03 ECE and canonical all/039-common contrasts are independently reproduced. A review caught source-proxy Brier intervals mislabeled as monthly in two oracle fields; corrected before sealing, with `oracle_reporting_correction.json` preserving the correction. Evaluator, forecasts and decisions did not change. This is not an independent audit of every AUC/decomposition/profile field.
- Run03 verifies **87** run02 prepared-input hashes, **85** native provenance paths and **14/14** source-snapshot members. Run02 preserves the 24 new/9 reused base identities and the inherited selection-exposure caveat. Run01's one bounded source-compatibility allowance records the archived/current benchmark-helper hashes and exact default-preserving change; other specialist code remains pinned.
- Separate Decimal/selection/bootstrap oracles verify **2,415 / 2,418 / 2,404** market-diagnostic rows for runs01/02/03, **12** selection sets and **36** monthly interval values per run; maximum payoff error **1.78e−15**. Counts include overlapping policies/bands, not that many distinct bets.

Final decisions: runs01/02 **`NOT_ADOPTED`**; run03 **`DEVELOPMENT_CANDIDATE_RETAINED`**, with nominal predictive gates passed but independent confirmation and financial reliability explicitly absent. Evidence and hashes are sealed in each reporting directory's `decision.json`; run03 is the retained inference artifact, not a canonical manifest replacement.

Frozen A0, the benchmark manifest, raw sources, research archive, operational EXP081-based hybrid and old 10/10 EV journal are unchanged. No live collection, H/market fitting, tournament certification or deployment occurred. Forward evidence is **`NOT_RUN`, zero genuine future forecasts**. A new independent assessment requires genuinely future released inputs/outcomes and a future-valid frozen recipe; no additional sweep starts automatically from this result.

## References

Primary sources; accessed 2026-09-27. Source-specific reading depth is recorded in `source_ledger.json`.

[1] Gavin C. Cawley; Nicola L. C. Talbot (2010). [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html).

[2] Cynthia Dwork; Vitaly Feldman; Moritz Hardt; Toniann Pitassi; Omer Reingold; Aaron Roth (2015). [Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629).

[3] Rob J Hyndman and George Athanasopoulos (2021). [Forecasting: Principles and Practice, third edition: Time series cross-validation](https://otexts.com/fpp3/tscv.html).

[4] Vitor Cerqueira; Luís Torgo; Igor Mozetič (2020). [Evaluating time series forecasting models: An empirical study on performance estimation methods](https://arxiv.org/abs/1905.11744).

[5] Claudia Perlich; Foster Provost; Jeffrey S. Simonoff (2003). [Tree Induction vs. Logistic Regression: A Learning-Curve Analysis](https://jmlr.org/papers/v4/perlich03a.html).

[6] Nils Reimers; Iryna Gurevych (2017). [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://aclanthology.org/D17-1035/).

[7] Jesse Dodge; Gabriel Ilharco; Roy Schwartz; Ali Farhadi; Hannaneh Hajishirzi; Noah Smith (2020). [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305).

[8] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

[9] Meelis Kull, Telmo de Menezes e Silva Filho, Peter Flach (2017). [Beta Calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf).

[10] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek (2019). [Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530).

[11] Tilmann Gneiting, Adrian E. Raftery (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

[12] Leo Breiman (1996). [Stacked Regressions](https://link.springer.com/article/10.1007/BF00117832).

[13] Mark J. van der Laan, Eric C. Polley, Alan E. Hubbard (2007). [Super Learner](https://pubmed.ncbi.nlm.nih.gov/17910531/).

[14] Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman (2018). [Using Stacking to Average Bayesian Predictive Distributions](https://arxiv.org/abs/1704.02030).

[15] Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474).

[16] Leo Grinsztajn, Edouard Oyallon, Gael Varoquaux (2022). [Why do tree-based models still outperform deep learning on typical tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).

[17] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko (2021). [Revisiting Deep Learning Models for Tabular Data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html).

[18] Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Benjamin Feuer, Chinmay Hegde, Ganesh Ramakrishnan, Micah Goldblum, Colin White (2023). [When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/html/2305.02997).

[19] Yury Gorishniy, Akim Kotelnikov, Artem Babenko (2025). [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/html/2410.24210).

[20] Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko (2024). [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks](https://arxiv.org/abs/2406.19380).

[21] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander J Smola (2017). [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).

[22] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, Yee Whye Teh (2019). [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://proceedings.mlr.press/v97/lee19d.html).

[23] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter, et al. (2025). [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6).

[24] Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan (2025). [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564).

[25] Léo Grinsztajn et al. (Prior Labs Team) (2025). [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/abs/2511.08667).

[26] Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, Mahsa Shoaran (2023). [XTab: Cross-table Pretraining for Tabular Transformers](https://arxiv.org/html/2305.06090).

[27] Ralf Herbrich, Tom Minka, Thore Graepel (2006/2007). [TrueSkill: A Bayesian Skill Rating System](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf).

[28] Tom Minka, Ryan Cleven, Yordan Zaykov (2018). [TrueSkill 2: An improved Bayesian skill rating system](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf).

[29] Pierre Dangauthier, Ralf Herbrich, Tom Minka, Thore Graepel (2007). [TrueSkill Through Time: Revisiting the History of Chess](https://proceedings.neurips.cc/paper_files/paper/2007/hash/9f53d83ec0691550f7d2507d57f4f5a2-Abstract.html).

[30] Blaž Krese, Erik Štrumbelj (2021). [A Bayesian approach to time-varying latent strengths in pairwise comparisons](https://pmc.ncbi.nlm.nih.gov/articles/PMC8136743/).

[31] Samuel Duffield, Samuel Power, Lorenzo Rimella (2023). [A State-Space Perspective on Modelling and Inference for Online Skill Rating](https://arxiv.org/abs/2308.02414).

[32] Amy X. Zhang, Parth Naidu (2024). [The SIDO Performance Model for League of Legends](https://arxiv.org/html/2403.04873v2).

[33] Roger W. Johnson (2018). [Playoff Series and the Incomplete Beta Function](https://ww2.amstat.org/meetings/proceedings/2018/data/assets/pdf/867069.pdf).

[34] Joseph Sill (2010). [Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing](https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing).
