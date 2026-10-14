# Sports probability models and betting reliability: research, weaknesses and bounded next studies

**Evidence date: 2026-10-09. Research status: COMPLETED / RESEARCH_ONLY. Proposed experiments: PROPOSED_NOT_FROZEN / NOT_AUTHORIZED.** This report synthesizes six literature/local-audit tracks, specifies concrete research hypotheses, and interprets independently checked mathematical illustrations. It does not authorize experiments, acquisition, source activation, deployment or wagers. The [research protocol](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/protocol.json), [research pack](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/research_pack.json) and [phase-one review](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/phase1_review.json) preserve the evidence and qualifications.

## 1. Executive decision

**Retain native C0 as the winner reference; do not interpret its disagreement with prices as a validated edge. Prioritize a bounded duration-distribution study, then a support-gated winner-representation study and a historical lineup-law feasibility study. Run sports research independently of, and alongside, betting-evidence development.** No current hypothesis establishes betting profitability. Better sports probabilities, informative market movement, reliable selected returns, positive after-cost policy value and authentic execution are separate achievements.

The strongest immediately formulable modeling question is **P1: conditional duration scale around the existing tree's fixed arithmetic mean**, with both the unchanged distribution and a matched mean-preserving global-scale control. Duration has a measured coverage shortfall and existing historical features/labels. That makes P1 better grounded than a generic claim that larger networks, recency or confidence flattening will repair winner forecasts. It is a **prop study, not a cure for C0 winner errors**; duration prices and genuine future issuance are absent.

**W1** asks whether learned compositional representations trained on held-out identities help a failed champion-history challenger generalize to natural out-of-vocabulary histories. The representation limitation is real, but belongs to that challenger, not selected C0. First count fixture-level support using saved evidence; do not infer support from large token counts. Even a successful OOV repair could leave the challenger behind C0. **R1** learns an exact-context historical lineup law and reports supported-roster bounds. Its finite softmax assigns positive OTHER mass to uncertain cases, so it supplies no complete unconditional scalar winner forecast there. It is feasibility research, not a deployable uncertainty wrapper.

A genuinely new betting signal could be **same-book pre-decision price trajectory**, but only with authentic all-state histories at both cutoffs. Give the same trajectory information to the market-only control; otherwise market news is incorrectly credited to the sports model. The conditional nine-head design is not an execution queue and cannot bypass blocked sources or expired dates.

The current position is not “wait years before learning anything.” Historical sports research can test precise mechanisms now, after separate authorization, with exposed development labels and honest controls. It cannot certify old issued forecasts or substitute for untouched confirmation. The original [Plan05](../TODO/05_betting_ev_reliability.md), [Plan06](../TODO/06_evidence_admission_after_a0_audit.md), G0, old 10/10 stopped loop and October 31/November 1 conditions remain intact.

## 2. Identity, population and information must precede interpretation

The [model catalog](../docs/MODEL_CATALOG.md) and [original C0 audit](c0_selected_betting_reliability_research_and_audit_20261006.md) distinguish the following identities:

| Identity | Meaning and boundary |
|---|---|
| Current local native C0 | `c0-native-2026-w32-e12-v1`, W32/E12 assembled native control from `a0_data_focus_20260929_192443`; full-mode series-winner probability. Frozen history ends 2026-09-08. |
| Annual A0 | Separate canonical research benchmark/control; its results are not automatically C0 results. |
| Fixed50 | `sigmoid(.5 logit(p_C0)+.5 logit(q))`, T=1: fixed diagnostic market hybrid, not an adopted learned betting model. |
| Old exported operational snapshot | EXP081-based `exp081-siamese-series-v1-a0.50-t1.00`; not evidence of current remote state. No new live-server inspection occurred. |
| Champion-history D | Separate width32/history16/epoch12 multitask challenger with its own vocabulary, terminal-score law and negative winner result. |
| Standalone props | Map1 kills/duration and series-length marginals; not a coherent full-C0 multi-market law or certified betting product. |

The current local registry also has a native-C0 Fixed50 hybrid identity; that does not retroactively change the older EXP081 export. The inspected old export contained zero C0 predictions, recorded bets and wallet transactions. Its financial portfolios are counterfactuals, not recorded production returns; this does not dispute user-reported manual losses.

The canonical sports assessment contains **3,103 series**, BO1/3/5 = **745/1,873/485**, from January 8 through September 8. C0's winner LL is **0.535448701** there. The later Polish market panel contains **1,467 identical series** under a different source/coverage contract; three historical betting-head assessments use **514 or 382 common score rows** but retain **523 mature cash opportunities**. The older strict-prior production replay has **945 origin rows / 387 distinct fixtures**; its first-observed, T−24h and T−1h panels are 318, 269 and 358. Repeated origins and nested cohorts are not independent matches. A score difference between populations cannot isolate model improvement.

A prediction must specify what was available: current players, released prior performance, competition/format, history origin, roster basis, quote clock and calibration recipe. Actual first-map players are a legitimate **conditional historical scenario**, not proof their identities were announced before a betting decision. A strictly earlier history date does not authenticate forecast creation or quote arrival. Conversely, lack of announcement evidence does not make historical player-history learning invalid; it limits the claim that it reproduces an issued live forecast.

Updating a frozen history state with earlier released assessment outcomes is **not weight refitting**. Future-safe comparisons must distinguish fixed model parameters from causal state updates, retain same-day release exclusions, and bind the complete model/calibrator/mixture recipe. No later mutable aggregate silently extends C0's frozen September history.

## 3. What a defensible sports probability model should learn

### 3.1 Causal player history, latent strength and noisy performance

Forecasts should be distributions conditional on a declared information set, evaluated on matched cases with strictly proper scores. LL and Brier assess winner probabilities; density NLL and CRPS assess continuous outcomes; terminal-score probabilities require legal normalized support. Accuracy is useful descriptively but cannot distinguish two probabilities choosing the same favorite. Calibration without sharpness is insufficient, and sharpness without calibration is dangerous. [1]

A player’s observed boxscore is not latent skill. It contains role, opponent, resource allocation, game length, team strategy and outcome-adjacent noise. Missing statistics are not zero performance; padding, missing observations, observed zero and previously unseen identities need distinct states. C0 already uses masks and auxiliary performance supervision. “Add masks” or “add historical player performance” therefore fails the novelty test in this repository.

Glickman's dynamic paired-comparison model illustrates the distinction between a latent state transition and an observation likelihood. Its random-walk variance grows with elapsed time, and causal filtering uses preceding outcomes; retrospective smoothing uses future observations. [2] A future LoL state filter could separately describe slowly changing skill and faster regime variation, but would need explicit context-dependent observation noise and uncertainty propagation. That is different from simply awarding rating credit for a postgame statistic, downweighting old supervised rows, or scheduling monthly retraining. It is also a much broader, currently deferred hypothesis—not one of the three first-line cards.

PandaSkill is useful for role-conditioned postgame representation and rating ideas, but its **90.74% accuracy** is explicitly explained by predicting game outcomes from end-game statistics. It is not prematch forecast accuracy. [4] Locally, direct PandaSkill-like performance credit already worsened map LL, and family/regional rating proposals have negative or unsupported records. A genuine measurement-noise filter would need a materially different mechanism and matched control; renaming rejected credit as “uncertainty-aware skill” does not supply one.

Score-driven football models likewise show how predictive-likelihood gradients can update strengths and how target/likelihood choice matters. Their expanding-sample forecasts re-estimate static parameters; that is not identical to a fixed C0 plus daily history updates. Football goal-count assumptions and results are methodological analogies, not LoL constants. [5] Guan and Tung's recent LoL preprint is closer in domain, but predicts **maps** with side and first-pick information. Its reported nonsignificant differences between architectures and Polymarket comparisons do not prove equality, transfer to series forecasts or define our usable pre-series inputs. [3]

### 3.2 Uncertainty is not one scalar

There are at least four distinct objects: randomness in the match conditional on inputs; uncertainty about latent/player states; uncertain roster or context inputs; and sampling uncertainty in an estimated aggregate score or return. A confidence interval for mean ROI is not an interval for an individual match's true probability. A central duration interval describes a predictive outcome distribution, not model-parameter confidence. A roster missing-mass bound is algebraic conditional on the assumed weight law, not a calibrated epistemic lower probability.

Expert agreement is not a calibration certificate. Experts share data, targets and much of the information basis; small disagreement can coexist with common bias. The ensemble theory inspected here is specifically about overparameterized random-feature regression and cautions against conventional uncertainty interpretations, not a proof that all neural ensembles fail. [21] Locally, twelve old deep C0 losses had known players, unanimous expert favorite direction and meaningful coplay. Neither unknown players nor disagreement explains all failures.

Uncertainty estimates should be tested through predictive scores, coverage **and width**, and their relation to supported contexts. Merely making every forecast closer to .5 or every interval wider can improve one diagnostic while damaging proper scores or usefulness. A scalar haircut invented from ensemble spread is not justified unless its conditional meaning is established.

### 3.3 Series and distribution semantics cannot be repaired by labels

Map winner, series winner, terminal score and total maps are different targets. A map probability cannot be used as a series probability without a stated series law. Conversely, already-projected C0 series probabilities must not receive a second BO transform. For a common latent map probability θ, BO3 winner probability is `3 E[θ²]−2 E[θ³]`; changing Beta dispersion while fixing `E[θ]` changes that series winner marginal. “Same map mean” is not “same winner forecast.”

The assembled C0 combines series experts; a component's terminal PMF is not automatically a coherent joint PMF for full C0. Constructing `P(score|winner) P_C0(winner)` preserves the winner probability by definition and therefore cannot improve its winner LL/Brier. It could investigate conditional score forecasts under separate controls, but prior score-law/residual studies already constrain that route.

Duration requires positive-support distributions and scores in the correct units. The current tree predicts log-location and calibrates global lognormal dispersion at **fixed log-location**, preserving median rather than arithmetic mean. Under `LogNormal(μ,σ²)`, `E[Y]=exp(μ+σ²/2)`. The generic `means_frozen:true` receipt does not override that identity. The evaluator correctly uses the arithmetic-mean formula. This is a source-level interpretation correction, not evidence of a broken evaluation or permission to mutate protected models.

Kills, duration and length marginals also do not identify their joint law. Dependence is needed for combined events, parlays and simultaneous exposures. Current standalone props cannot gain joint-market certification simply because each marginal is normalized or calibrated.

### 3.4 Strong controls identify information, not cosmetic improvement

A candidate must beat an equally informed control with the same cutoff, labels, support, tuning allowance and input scenario. A sports-plus-market head beating raw q may only recalibrate the market. The proper question is whether it adds information beyond a calibrated nonlinear market-only model. A market-offset connector with the existing feature span changes shrinkage or constraints, not information. A conditional-scale proposal must beat a matched global-scale arm, not only an older parametrization. An OOV proposal must separate extra capacity from held-identity training.

These controls are not administrative burdens. They distinguish the cause claimed by an experiment from an easier alternative explanation. A new feature is genuinely new information only if its authentic before-cutoff values were absent from the previous information set and are also offered to every relevant market control.

## 4. Confirmed weakness, limitation and unproved cause

The following matrix separates observed results from architecture gaps and causal conjectures. Local evidence is retained in the [model audit](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/tracks/ModelWeaknesses.md), [betting audit](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/tracks/BettingWeaknesses.md) and [availability/props audit](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/tracks/AvailabilityProps.md).

| Observation | Classification | What follows—and what does not |
|---|---|---|
| Selected C0 returns differ sharply from apparent EV; three learned-head studies fail adoption gates | Confirmed exposed-development weakness | Global sports quality does not establish selected reliability; no universal causal explanation follows. |
| Production T−1h selections have larger model–market gaps and lower favorite confidence than unselected rows | Confirmed selection pattern | Not simply “bets the most confident favorites”; no post-hoc disagreement threshold is justified. |
| Annual A0 low-history subgroup has a favorite-confidence gap | Confirmed subgroup warning | All 232 low-coverage rows are LPL major BO3/5; missingness is nearly confounded, not identified as the cause. |
| Selected duration central80 coverage 75.9910%, central50 46.5034% on 3,103 rows | Confirmed historical reliability shortfall | Conditional scale is plausible; location bias, tail shape and temporal shift remain alternatives. |
| Champion-history UNK rows received no valid TRAIN examples | Confirmed candidate-specific representation limit | Natural fixture-level support and causal loss contribution remain unproved; not C0's defect. |
| Native C0 conditions on exact five-player inputs, has no supplied per-match calibrated epistemic interval | Limitation | Does not explain known-player deep losses or authorize a probability haircut. |
| Service roster provenance is unauthenticated with null publication/observation clocks | Evidence limitation | Input consistency is not announcement authenticity; historical daily-history modeling remains feasible. |
| Only three betting-assessment monthly blocks, sparse actions and undefined near-band resamples | Precision limitation | Neither persistent error nor equivalence is established by a wide/null interval. |
| Active-only archived quote export omits live/unknown states | Source limitation | Latest retained quote is not necessarily latest source state; genuine availability is unproved. |
| Shared expert bias, two-timescale drift, roster uncertainty and performance measurement noise | Unproved mechanisms | Must be falsifiable under distinct controls; none currently explains all error. |

For scale, annual A0's low-history 232-row group has mean favorite confidence **70.92%** and wins **63.36%**, a **7.56pp** gap; its market-linked 212-row subset has a **6.63pp** gap. These are not C0-wide error estimates. In the separate production T−1h audit, 36 selected versus 322 unselected rows have favorite confidence **.64436 versus .72246** and absolute model–market gap **.13465 versus .07644**. Selected LL is **.75557 C0 versus .69496 market**. These descriptions locate a decision-population problem, not a threshold to optimize.

The older first-observed strict-prior portfolio has **34 C0 selections**, predicted EV **+24.54%**, flat ROI **−18.86%**, and point overstatement **43.40pp**. The same bets lose **−7.79% without tax**. Replacing ten rosters with retained actual lineups lowers predicted EV to **22.43%** but cannot change realized PnL on fixed prices/outcomes; unchanged-player cases still have substantial overstatement. Taxes and roster mismatch are therefore not sufficient explanations. The [original report](c0_selected_betting_reliability_research_and_audit_20261006.md) preserves their scenario-specific intervals and exclusions; they are not recomputed here.

**Props have a separate market weakness.** The [original prop-market decision](../data/08_reporting/prop_market_comparison/prop_market_comparison_20261007_01/decision.json) scores one price-selected Pinnacle total-maps line per series on **1,318 fixtures**: model/market binary LL **0.669133345/0.658158740**, model−market **+0.010974605**, 97.5% monthly interval **[+0.003025793,+0.023563745]**. The match-weighted comparison favors Pinnacle; its equal-month interval crosses zero. This is not the standalone 2,358-row full series-length distribution NLL or a universal market verdict. There are nine monthly blocks and 423 missing actual starts; issuance and executable availability remain uncertified. Kill prices cover only **two fixtures**, not 18 independent observations from their alternative lines; duration has no verified price comparison. Improving a prop distribution against another sports model therefore does not establish a betting advantage. These are retained earlier results, not a new market collection or replay.

## 5. The negative-experiment ledger constrains novelty

Research should preserve failed and skipped branches, not reopen them under attractive names. The following is a compressed current ledger; linked original records carry exact cohorts, guards and uncertainty.

| Completed line | Current interpretation and no-repeat consequence |
|---|---|
| [Recency/cadence](bookmaker_noise_and_provider_training_20260927.md) | 270 fits, 11,550 common matches: H365 worsened LL annually and monthly; cadence without decay was inconclusive. Not a clean rejection of every dynamic state model, but no same decay/monthly retry. |
| [Tier weighting/token dropout](../docs/MODEL_CATALOG.md) | High-tier-only, market-tier weighting, whole-history token dropout and combinations were not adopted; uniform native control retained. |
| [Reconstruction and native6+12](noisy_data_reliability_research_20260929.md) | Reconstruction lost to its matched extra-supervision control. That control's small exposed-2026 gain did not persist at earlier origins; do not promote it or retune corruption. |
| [Six-epoch confidence branch](noisy_data_reliability_research_20260929.md) | Plan08 had worse LL, insufficient reliability gain and a failed BO5 cost guard; evidence does not establish that twelve epochs overfit. |
| [Capacity/native KD](model_targets_capacity_and_distillation_research_20261004.md) | Adjusted capacity contrasts crossed zero; KD was **skipped with zero KD fits** after a frozen guard failed. Neither equality of capacities nor empirical KD failure follows. |
| [Context persistence](../data/08_reporting/a0_context_reliability/context_reliability_20261004_02/decision.json) | Six global diagnostic fits, persistence not confirmed, insufficient novelty support; **zero conditional fits**. Not a fitted conditional-model negative. |
| [Champion pool](../data/08_reporting/a0_champion/a0_champion_20261005_01/decision.json), [champion history](champion_history_prediction_20261007.md), [meta studies](../docs/RESEARCH.md) | Pool connector not adopted with a failed source-unseen guard. D loses to C0; 30-day meta unproved; stable180 selected zero amplitudes/exact C0. “Champions never used” is stale. |
| [Family/region/performance credit](noisy_data_reliability_research_20260929.md) | Bridge improvements against weaker controls did not establish incremental A0/C0 value; direct boxscore credit and regional Meta-OpenSkill were rejected or contract-limited. |
| [Plan09 connector](plan09_c0_connector_and_forward_evidence_20261002.md) | Primary signal does not override failed BO5/source-unseen guards. The original Plan09 report and its linked decision remain authoritative. |
| [Betting heads](c0_selected_betting_reliability_research_and_audit_20261006.md) | Three later head studies remain NOT_ADOPTED. Generic market head, prior-input alignment and return-MSE are already tested, not new proposals. |
| [Tournament/joint laws](../docs/MODEL_CATALOG.md) | Copula/residual/score-law studies rejected or insufficient; preserving winner marginals cannot improve them. No tournament copula as an untested winner cure. |

Champion-history D's LL **0.546320524** versus C0 **0.535448701** on the same 3,103 targets gives **+0.010871824**, historical interval **[+0.003984947,+0.019233993]**. Its assessment UNK-token counts are **13,662 champion / 418,999 patch / 422,104 competition**; these are not fixture denominators. Direct performance-credit map LL changed **.615081→.615761** on 20,200 maps. These failures narrow mechanisms, not the universe of useful sports information.

Training-only market-teacher IDEA-022 remains **deferred** and distinct from native expert KD and tested market/profit heads. A two-timescale noise-aware skill filter is also deferred. Neither is quietly added as a fourth first-line experiment.

## 6. Latest market comparison and three completed betting studies

### 6.1 What the Polish comparison actually says

[Plan12's original decision](../data/08_reporting/model_bookmaker_competitiveness/plan12_model_bookmaker_20261009_175136Z/decision.json) and [execution report](model_bookmaker_competitiveness_execution_20261009.md) are a historical source-conditional comparison, not a betting test. The primary **1,467-series** consensus separately de-vigs Betfan, LV BET, STS, Superbet and available Fuksiarz, then equally averages probabilities; at least two distinct books are required. Fixed-four sensitivity has **1,285** nested series. Of **1,749** frozen source close opportunities, 258 lack a saved model match, 22 have only one Polish book, and two fail the release-day date rule. No unavailable forecast is imputed.

| Forecaster, same primary 1,467 series | LL | Brier |
|---|---:|---:|
| Polish consensus | .559604853 | .189266950 |
| Native C0 | .565904908 | .192123748 |
| Corrected thesis LR | .570802361 | .193908558 |
| Corrected thesis MLP8 | .573508843 | .195254691 |

C0−consensus LL is **+.006300054**, nominal95% interval **[−.007429,+.020895]**, family12 interval **[−.012585,+.027929]**. All twelve family-adjusted LL intervals cross zero. **Nonsignificance is not equivalence.** No justified practical LL budget was adopted; competitive/equivalent status is explicitly not assessed on that basis. The consensus has the best observed points, not an established unique winner. LR's smaller ECE10 alongside worse LL/Brier illustrates why ECE alone cannot choose the best distribution.

The endpoint called `close` is last stored prestart source data, **not authenticated CLOSE**, arrival time or a fixed-hour quote. Models use a conditional actual-roster scenario and strict-prior history, not proved synchronized issuance. A real equal-cutoff/equal-information comparison is still needed. Nested scores cannot be summed as independent matches, and the requested seven-day block procedure is an assumption—not certified independence or untouched confirmation.

### 6.2 Completed head studies: keep the denominators separate

All three studies use the frozen C0 base, 12% stake-tax assumption, one flat unit per fixture, raw net EV **strictly >5%**, best aligned price and deterministic ties. They are exposed historical development. Their original [reviewed study-one decision](../data/08_reporting/c0_betting_development/c0_betting_development_20261006_01/reviewed_decision.json), [study-two decision](../data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_02/decision.json) and [study-three decision](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/decision.json) all remain **NOT_ADOPTED**.

| Study | Actual fits; cohort | Own-policy result | Why it fails |
|---|---|---|---|
| H-price likelihood | **18 head / 0 base**; 742/363/514 TRAIN/CAL/ASSESS common rows; phase opportunities 775/370/**523** | 15 bets; mean EV58.21%, ROI+129.09%,95% ROI[−100%,+240.71%]; selected error−70.89pp[−216.31,+106.67]; 5 near-band bets | LL superiority over M-price/M-shape unresolved; Brier worse versus M-shape; symmetric ±2pp reliability and value lower bounds fail. |
| Prior-input-aligned H-price | **18 head / 0 base**; native-prior661/304/386, common636/299/**382**; same523 cash opportunities | 14 bets; ROI+39.48%[−100%,+62.448%]; selected error+19.12904pp[+9.21488,+111.13557]; 6 near-band bets, interval not estimable | Both market LL contrasts unresolved and both Brier points worse; selected/near-band and value gates fail/null. |
| H-return-MSE | **12 head / 0 base**; same636/299/**382**, same523 cash opportunities | 10 bets; mean EV75.4883%, ROI+72.04%[−100%,+132.0267%]; error+3.4483pp[−21.9286,+106.0619]; 6 near-band bets | Matched H-likelihood risk superiority unresolved; market LL/Brier, symmetric reliability and value gates fail. |

Study two has **514 archival market-supported rows but 510 permitted market forecasts** after shared prohibitions. Its 312 roster absences and five already-released-target prohibitions remain explicit. Neither 510 nor 382 replaces 523 in cash opportunity accounting. The latest three-month assessment includes July, August and partial September; 5,000 monthly draws do not add independent months.

Study-one LL is **.575664709 H versus .576848730 M-price / .576828062 M-shape** on 514 rows. Its profit per cash opportunity is **.037024857**, interval **[−.021276596,+.089704348]**, and increment versus M-price **.030087954 [−.002306032,+.102250932]**. Positive point ROI therefore does not establish value. Study-two H LL **.582673346** exceeds M-price **.582432980** and M-shape **.582413548** on 382 rows; PPO is **.010568260**, with market increment **−.016942256**, both intervals spanning zero.

The third objective is precisely `0.5*.88²*(O_A²+O_B²)*(p_A−y)²`: odds-squared-weighted Brier, **not profit**. All four likelihood/return-MSE arms select penalties using the same CAL return-MSE criterion. H-return-MSE−H-likelihood risk is **−.000157271**,98.333% interval **[−.000847313,+.001575692]**. PPO **.013774379 [−.021276596,+.025147937]**, increment versus M-return-MSE **−.011824092 [−.064954037,+.042553191]**. Local stationarity checks do not prove global optimality of its nonconvex sigmoid objective.

The separate study-one transfer to 284 prior-context T−1h cash opportunities has **267 common forecasts/9 bets** at24h freshness and **81/4** at300s, ROI **−52.19%/−50.72%**. It changes roster, history/cutoff and prices together and overlaps historical fixtures; it is not independent replication or isolated roster causality. Fixed-selection diagnostics cannot change realized PnL. No old forecasts, confidence intervals or policies were rerun for this report.

## 7. Literature correction and transfer ledger

| Source or claim | Correct use |
|---|---|
| Walsh–Joshi original calibration paper [8] versus corrigendum [9] | The 2025 correction supersedes affected results: calibration-driven average ROI **−9.77%**, not +34.69%; accuracy-driven −26.78%. Calibration-driven **eighth-Kelly −42.67%**; fixed-stake best case still +23.13% versus +10.9%. Not “all corrected strategies lost,” not calibration useless, and no automatic calibration→profit/Kelly guarantee. |
| Zhao decision calibration [6] | Formal loss excludes direct X dependence and decisions use predicted p only. Varying odds make betting action `A(p,O)` and loss price-dependent; direct theorem transfer is invalid without a justified conditional/extended setup. Population-average loss matching is not per-bet probability correctness. |
| Multicalibration [7] | Guarantees concern a declared computational class with support/generalization conditions. They do not certify arbitrary post-hoc odds/EV groups or±2pp selected returns. |
| Guan–Tung [3] | LoL map preprint with side/first-pick covariates; nonsignificance is not equivalence and not C0 series or equal-time bookmaker proof. |
| PandaSkill [4] | End-game-statistic accuracy is postgame explanation, not prematch accuracy or independent vendor-model validation. |
| Overround adjustment [10] | Inspected publisher abstract reports power-method performance on three sports datasets. It supports sensitivity questions, not universal power superiority in LoL. |
| Robust/risk-constrained Kelly [14][15] | Explicit probability ambiguity set or assumed return process supplies guarantees. These papers do not validate our ambiguity set or repair C0 probabilities. |
| Conformal/sequence methods [17][19][20] | Boundedness, exchangeability/conditional-mean or shift-control assumptions matter. Time drift, chosen bets and repeated teams are not generically handled by a label. |

The catalog contains **25 external records**; this report cites a selected subset, not all records. The JRSS source that returned 403 and incompletely extracted Yale championship-series PDF are excluded from technical support, as recorded in the [phase-one review](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/phase1_review.json). Access restrictions are evidence limits, not silently filled knowledge. No inaccessible theorem is asserted. Legal/operator pages establish the ordinary 12%/.88 accounting assumption, not authentic offered/accepted prices or comprehensive current legal advice; inconsistent high-win threshold shorthand is not used. [11][12][13]

## 8. Six fresh analytic examples—mathematics, not new empirical results

All constants below are synthetic illustrations from [diagnostic_results_v2.json](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/diagnostic_results_v2.json), independently checked in [independent_verification_v2.json](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/independent_verification_v2.json). They demonstrate identities and counterexamples, not fitted project probabilities, future outcomes, new CIs or profitable policies.

**Example1—price magnifies small probability errors.** For an ordinary taxed single, `D=.88O`, net EV `E=Dp−1`, realized unit return `R=DY−1`, and error `E−R=D(p−Y)`. Break-even is `1/D`; the5% hurdle is `1.05/D`. If the true conditional win probability is π, expected-return bias is `D(p−π)`: a2pp conditional EV-bias budget corresponds to probability error `.02/D`, not2pp probability error.

| Decimal O | Break-even p | p at5% EV | Probability error for2pp EV error |
|---|---:|---:|---:|
|1.50|.757576|.795455|.015152|
|1.85|.614251|.644963|.012285|
|2.00|.568182|.596591|.011364|
|3.00|.378788|.397727|.007576|
|5.00|.227273|.238636|.004545|
|10.00|.113636|.119318|.002273|

Thus one probability point at odds10 changes estimated EV by **8.8pp**; at odds2 by1.76pp. A bound of `.02/D` on every conditional probability error is **sufficient, not necessary**, for a2pp aggregate expected-bias bound: signed errors can cancel. It does not remove outcome noise or guarantee containment of a finite-sample confidence interval for selected return error. These are sensitivity calculations, not measured per-match errors, a universal precision requirement, or grounds for introducing an odds cap.

**Example2—average conditional probabilities, not logits or player inputs.** With roster weights(.8,.2) and conditional winner probabilities(.8,.2), total probability is **.68**, whereas weighted-logit averaging gives **.696730455**. Side reversal gives **.32**. Total Bernoulli variance **.2176** splits into expected conditional variance **.16** and between-roster variance **.0576**. The latter is uncertainty induced by this specified roster law, not a measured C0 epistemic variance. Actual R1 weights must be learned from prior evidence, not these illustrative weights.

**Example3—same duration mean, different tails.** Hold arithmetic mean30 minutes and set `μ=log(30)−σ²/2`. At σ=.2/.3, `P(Y>35)` is **.191944407/.253397769**; central80 intervals are **[22.757325,37.997018] / [19.525707,42.125904]**. Mean MAE stays fixed, tails and widths do not. If μ instead stays fixed, the mean rises **30→30.759454** while median stays29.405960. Therefore current fixed-μ calibration is not a true fixed-mean control, and wider distributions alone cannot establish better coverage on real outcomes.

**Example4—marginals do not price a joint event.** With marginals .7 and .6, Fréchet bounds give joint probability **[.3,.6]**. Independence gives .42 but is an extra assumption. For a taxed coupon at decimal3, EV ranges **−.208 to+.584**; independence yields+.1088. Equally good marginal forecasts can imply opposite joint value. This is not a proposed parlay strategy; it shows why unsupported dependence cannot be replaced by multiplication.

**Example5—fixed map mean does not fix BO3 winner.** At map mean.7, iid BO3 winner is **.784**. A Beta latent law with concentration2 gives **.728**; concentration20 gives **.772727273**. Changing dispersion changes the winner marginal despite fixed map mean. No native C0 forecast was changed or replayed. Fixed-winner conditional-score modeling would avoid that change but cannot improve the fixed winner forecast.

**Example6—market-offset residual is not new information.** Let `m=u+v` be any fixed-knot market decomposition and `g=b1u+b2v+ce`. Then `g+γ(z−m)=(b1−γ)u+(b2−γ)v+ce+γz`, overlapping the existing H-price span subject to constraints. Four synthetic cases agree to **1.1102230246251565e−16**. The generic illustration's knot is in logit units; production H-price uses `k=logit(.8)`. This is an algebraic proof for any fixed knot, not a replay of native-head transforms or evidence that fitting procedures are identical. Freezing g can change statistical control/shrinkage, not add information.

## 9. Three first-line sports cards

The complete [sports experiment design](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/sports_experiment_design.md) specifies transforms, optimizer bounds, contract checks and stop rules. The summaries below are hypotheses, not adopted gate changes or authorizations.

### Shared chronology and proposed inference

Use existing **TRAIN: 36,205** canonical series played/released before 2025-07-01; **CAL: 1,328** from 2025-07-01 to 2026-01-01 exclusive with release fence; **ASSESS: 3,103** frozen 2026 series through September 8. TRAIN learns representations, normalizers, scale slopes or lineup laws. CAL may fit only stated scalars. ASSESS never fits or reselects; fixed causal history can update from earlier released observations, excluding same-day releases. All periods are development-exposed, not protected confirmation.

The proposed shared primary family reserves ten slots: P1 two, W1 four, R1 two and deferred IDEA-022 two. **99.5% paired intervals** reflect a suggested .05/10 allocation, with unused slots not recycled. Proposed uncertainty is 10,000 paired calendar-month draws, seed 20261009, all nine month universes retained, with match-weighted and equal-month estimands. Source-name/year is a proxy sensitivity, not certified independent editions. Proposed subgroup floors ≥200 series/six months for primary claims and ≥100/four for guards are feasibility suggestions, not power guarantees. Undefined required subgroup draws mean NOT_ESTIMABLE. None replaces inherited statistical gates or cures historical adaptivity.

### P1: fixed-tree conditional duration scale

**Mechanism:** uncertainty varies predictably with existing causal support, cross-side historical pace gap and duration missingness. Keep the selected 15-leaf tree's arithmetic mean `m_i=exp(μ0_i+σ0²/2)` fixed; every new scale uses `μ_i=log(m_i)−σ_i²/2`.

**Exact controls:** P0 unchanged tree/current CAL distribution; PG mean-preserving global scale with one CAL scalar; PC mean-preserving conditional scale with three TRAIN slopes and one CAL scalar. The three covariates are `log1p(x_pace_player_duration_minutes_count_mean_level)`, `x_pace_player_duration_minutes_mean_gap` and `x_pace_player_missing_duration_minutes_level`. No odds, target duration, new competition residual or tree refit. TRAIN normalization/imputation and fixed domain clipping are distinct from outcome exclusion.

**Budget:** four estimation operations—analytic TRAIN reference sigma, one joint TRAIN slope-head fit, PG CAL scalar and PC CAL scalar; **zero base fits**. TRAIN does not use the old CAL-derived mean anchor; slopes freeze before CAL. Maximum 500 iterations for declared fits; no alternate starts, bounds, seeds or families. In-sample tree residual optimism is disclosed, not repaired with unbudgeted cross-fits.

**Proposed success/falsifier:** PC−P0 and PC−PG density NLL upper 99.5% bounds both below **−.002**, equal-month deltas negative; **cohort-mean CRPS point estimates must not increase against either control**, with paired uncertainty reported. This is not per-observation dominance or demonstrated population noninferiority. Require 80% coverage-error improvement ≥1pp against P0 and nonworse PG, and 50% error worsening ≤.5pp. Report interval widths, PIT and fixed threshold Brier diagnostics. Mean forecasts remain equal rowwise. **Failure to beat PG closes the conditional-information claim**; widening or beating only the median-fixed parametrization is insufficient. A successful exposed result is only DEVELOPMENT_CANDIDATE. Missing duration prices prevent a betting conclusion, not this historical modeling question.

### W1: entity-held-out compositional OOV winner learning

**Stage0, zero fits:** count natural champion-OOV target series and seen-history support using valid strictly released historical identities, existing token channels and saved D/C0 predictions. Require proposed ≥200 series/six months in both groups; otherwise close without fitting. Existing token totals do not establish this support. No target champion/patch, current draft or outcome defines group membership.

**Mechanism/controls:** compose existing 36-vector token statistics/masks/role/pace/strength/ages into 8-dimensional identity representations with support-gated residuals. Hold entire deterministic identity residuals out for approximately 20% of TRAIN IDs; retain their numerical observations. D0 is saved original D. H uses held IDs routed to UNK without composition; C uses composition without holdout; HC combines both. C/HC match active capacity; added projections contribute 888 parameters, so HC versus H/D0 alone cannot identify the mechanism.

**Budget/chronology:** H/C/HC × seeds 20261007/20261008/20261009 = **nine neural fits, 108 seed-epochs**, plus exactly **three ensemble CAL temperatures**; inherited width32/history16/twelve epochs, objectives and optimization. No C0 refit, prop-scale fit or ASSESS fit. Temperatures transform member terminal PMFs before ensemble averaging/summing legal winner bins; no second BO conversion.

**Proposed success/falsifier:** four 99.5% primary LL upper bounds <0: HC−D0/H/C on the predeclared natural champion-OOV group and HC−unchanged C0 on all 3,103 targets, with nonworse Brier and supported guards. **Still losing globally to C0, failing HC−C, or losing the advantage relative to H closes the winner hypothesis.** The known D deficit makes failure plausible. No guarantee that solving a candidate-specific OOV limitation closes C0's gap; no favorable champion subset, history or capacity retry.

### R1: learned historical lineup law and incomplete-mixture bounds

**Mechanism:** at the strict-prior UTC-day origin, take each team's last ten released appearances in its **exact supplied competition context**. Candidates are the latest distinct exact five and next earlier different five, ordered without outcomes. The next actual lineup is only a later released supervised label: latest, previous or OTHER. No candidate means absence/abstention, never actual-roster fallback.

**Control/budget:** one six-parameter side-shared multinomial logistic law uses candidate counts and ages; one two-logit global-transition control uses identical candidates/masks. **Two TRAIN fits; zero CAL fits or C0 refits.** CAL audits recall, NLL and OTHER reliability once. Proposed support ≥1,000 TRAIN side-labels, ≥100 OTHER and ≥100 previous, then ≥200 assessment series/six months; actual exact-context denominators must be counted, not assumed equal canonical totals.

**Proposed success/falsifier:** conditional-minus-global lineup categorical NLL and OTHER-event Brier upper 99.5% bounds <0, with recall/class support reported first. Failure, low recall, unreliable OTHER or outcome-conditioned supported-roster filtering closes the mechanism claim. Separately, provisionally multiply side lineup laws, disclose independence, and infer frozen C0 for at most four supported pairs. These outputs are already series probabilities.

If `L=Σ_supported w_ab p_C0(a,b)` and `u=1−Σ_supported w_ab`, completion lies in **[L,L+u]**. Do not renormalize away OTHER or invent its probability. Finite softmax generally gives OTHER>0, so **uncertain-case unconditional scalar coverage is zero** under this card. Bounds are not confidence intervals or deployable p_low; known-roster cases retain a point mass. A complete mixture needs exhaustive supported provenance or a real OTHER forecast and validated joint weights. Historical lineup learning need not await public announcements, but announcement-conditioned live claims require authentic publication evidence.

## 10. Conditional betting study and three separate progression questions

The [betting program design](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/betting_program_design.md) specifies opportunity, identity, cash, chronology and multiplicity contracts. A sports score win does not establish conditional market value. Dmochowski's sports decision analysis separates choosing a side, whether a profitable bet exists and estimating its needed distribution; its NFL spread/total evidence does not prove LoL binary performance. [18]

### 10.1 One new signal, nine heads at most

The proposed signal is same-book probability movement between **T1=T−24h** and **T0=T−1h**, using the same frozen schedule vintage. Require authentic all-state bytes, source/observation clocks, latest-invalid rejection and two identical underlying books with full two-sided quotes within300s at both cutoffs. No CLOSE substitution, later quote, stale state behind suspension or missing movement=0. Earlier-origin capture is additional authorization scope, not implied by current capture machinery.

Let `Δ=mean_b[logit(q_b,A,T0)−logit(q_b,A,T1)]` on the matched-book intersection. Keep current q's frozen book rule unchanged. With `m=logit(q_A,T0)`, `k=logit(.8)`, `u=sign(m)min(|m|,k)`, `v=m−u`, `e=log(O_B,exec)−log(O_A,exec)` and frozen sports logit z, compare:

- **M0:** `sigmoid(b1u+b2v+ce)`;
- **MT:** same plus `dΔ`;
- **HT:** same plus `dΔ+az`.

No side intercept; b1,b2,a≥0 and c,d free. Three recipes × λ{.01,.1,1} give **maximum nine formal head fits**, zero base/calibrator fits/restarts. Fit all eligible prior fixtures, not selected winners/bets; CAL selects the fixed penalty using LL, ties favor stronger penalty, no refit. Valid full-recipe past-only sports forecast support is required; missing support stops HT rather than broadening budgets. MT−M0 isolates market movement; HT−MT isolates sports increment. Both proposed 97.5% LL upper bounds <0 with nonworse Brier are required for their respective claims; M-cal/M-shape remain strong-control guardrails.

MT alone succeeding means market information helped, not sports skill. Both succeeding means only progression to reliability/value assessment. Missing authentic trajectories are a prerequisite failure, not proof movement contains no information. Existing static heads remain negative; this design does not reopen their historical recipes.

### 10.2 Selected reliability is symmetric and policy-specific

For fixed selected tickets, `B=mean[D(p−Y)]`; changing the probability changes modeled error but not realized PnL. For each own policy on full universe U, evaluate `B_j=ΣA_jD(p_j−Y)/ΣA_j` and recompute selections inside sampled opportunity blocks. Do not transfer a fixed-selection interval to a new policy.

The inherited Plan05 target requires **both** overall and near-band `5%<EV≤10%` 95% intervals entirely inside **[−.02,+.02]**. Negative overstatement outside tolerance fails too; unexpectedly high realized ROI is not calibrated prediction. No actions make conditional error/ROI undefined, not zero. Under the distinct proposed two-policy joint claim, four overall/band comparisons would need a predeclared adjustment, such as 98.75% individual intervals. This is a proposal, not an adopted gate change. Price/EV/context diagnostics must be fixed in advance and supported; multicalibration does not license adaptive cells. [6][7]

### 10.3 Value, availability and confirmation are additional gates

Policy value uses **profit per eligible financial opportunity**, `PPO=ΣAR/|U|`, retaining no-action rows. Require supported positive after-cost PPO lower bound, positive paired PPO increment over an **equally informed** market-only policy, and estimable positive selected ROI with supported lower bound. Availability is separately reconciled source→admission→forecast→action→maturity. Show common-support information comparisons and market-extra coverage; selective low coverage or vanishing actions is not a successful business policy. No current evidence supplies a numeric business action floor, so none is invented.

Full-recipe chronology includes normalization, labels/releases, calibrator, mixture and series projection—not just head-fit dates. More bootstrap draws do not repair three months, dependence across teams/editions, undefined selected ratios or repeated research on exposed years. Backtest-overfitting analysis concerns relative strategy ranking under its resampling setup, not the probability a particular betting policy loses money. [16] Confidence sequences require the inspected bounded/conditional-mean assumptions; nonexchangeable conformal bounds pay a shift penalty, and selective conformal guarantees require their own exchangeability/shift conditions. [17][19][20] No theorem here supplies assumption-free±2pp LoL EV reliability.

Original E3 planned windows remain TRAIN **[2026-11-01,2027-11-01)**, CAL **[2027-12-01,2028-06-01)** and DEVELOPMENT **[2028-07-01,2029-07-01)**, with maturity buffers/locks in the detailed design. Readiness **and separate acquisition authorization before 2026-10-31T00:00Z** are necessary for the November 1 start; otherwise the version expires unstarted. No dates roll automatically. E4 requires separately authorized untouched twelve complete UTC months after final recipe/protocol freeze, not reused DEVELOPMENT, optional stopping for early profit or extending until success. Successful development is at most DEVELOPMENT_CANDIDATE; supported confirmation is cohort/policy-specific and still not automatic deployment.

## 11. Ranked next steps and explicit exclusions

1. **Freeze P1 after separate authorization:** reuse existing tree/data, demonstrate mean identity and rank support, then the four-operation comparison once. Most grounded reliability mechanism; no winner or profit claim.
2. **Authorize only W1 Stage0 first:** establish fixture-level natural-OOV/seen support and existing D−C0 deficit. A passing inventory permits consideration—not automatic execution—of the fixed nine-neural-fit/three-temperature comparison.
3. **Establish R1 exact-context recall and class support:** if feasible, compare two fixed lineup laws, retaining OTHER mass and zero uncertain scalar coverage. Do not pretend feasibility is a complete model.
4. **Resolve real equal-cutoff/source information prerequisites separately:** authentic all-state trajectories, complete sports chronology and opportunity coverage before any nine-head price study. Do not rebuild existing accounting as if it were new signal.
5. **Reserve genuine confirmation for a frozen successful candidate:** preserve original gates/dates, report applicability and uncertainty, and keep acquisition/deployment separately authorized.

Do **not** retune EV thresholds, odds caps, Kelly fractions, entropy discounts, global temperatures, league blacklists or favorable horizons to rescue exposed returns. Do not promote C0–market disagreement, CLV or consensus q as truth; multiply marginals into a parlay; average player tensors as roster marginalization; eraseOTHER; attach an epistemic label to expert spread; or declare equality because intervals cross zero. Robust/risk-constrained Kelly is exposure management conditional on a justified model, not probability repair. [14][15] Do not erase negative decisions, skipped branches or the old stopped loop. The objective is identifiable learning followed by honest evidence—not producing another impressive historical ROI.

## 12. Executed bounded studies and empirical resolutions — 2026-10-10

Following user authorization to orchestrate improvements, the three bounded sports cards (P1, W1, R1) were prepared, frozen, and executed alongside prerequisite evaluations for the market branches (T1, D1).

### 12.1 P1 Duration scale reliability (`duration_scale_20261009_220719Z` / `_p1_integrity_v3`)
- **Protocol & Design:** Preserves the arithmetic mean $m$ of the frozen 15-leaf LightGBM duration model from `prop_distribution_20261007_01` on Map 1 duration. Reparameterizes $\mu_{\text{new}} = \log(m) - \sigma_{\text{new}}^2 / 2$. Exactly four bounded estimation operations: P0 (reference tree baseline, scale $s=1$), PG (mean-preserving global scalar control, $s=0.96347101$), and PC (mean-preserving 3-covariate conditional scale using rolling pre-match pace features). Zero base fits, zero neural fits.
- **Evaluation on 3,103 canonical 2026 series:**
  - P0 NLL: 3.076194163, nominal 80% coverage: 75.991% [74.453%, 77.457%].
  - PG NLL: 3.074744795, nominal 80% coverage: 76.571% [75.050%, 78.026%].
  - PC NLL: 3.074351083, nominal 80% coverage: 76.829% [75.318%, 78.272%].
  - Contrast $\Delta\text{NLL}$ (PC − PG) = −0.000393712, 95% paired block bootstrap CI [−0.001642, +0.000854] crosses zero (9 monthly blocks, 5,000 resamples).
- **Decision:** `COMPLETED_NO_CONDITIONAL_GAIN_OVER_GLOBAL_SCALE`. Global scale mildly mitigates duration undercoverage (+0.58 pp), but conditional pace features do not establish statistically significant incremental value over the single global scalar. Retained as development evidence; not adopted. C0 winner reference and serving unchanged.

### 12.2 W1 Champion OOV support inventory (`sports_improvements_20261009_220719Z_w1_metadata_v3`)
- **Protocol & Support Gate:** Evaluated fixture-level support across all 3,103 canonical 2026 assessment series before considering any neural training.
  - Natural champion OOV series: 2,932 series across all 9 months (318, 377, 239, 586, 500, 129, 264, 446, 73).
  - Seen champion history series: **167 series** across 9 months (48, 36, 13, 10, 15, 12, 15, 17, 1). This fails the prespecified support floor of $\ge 200$ series.
  - Missing champion identity: 0 series. Cold history: 4 series.
- **Decision:** `CLOSED_BY_SUPPORT`. Zero neural fits executed (0/9 neural fits, 0/108 seed-epochs, 0/3 CAL temperatures, 0 C0 fits). Public saved inference strictly rejects requests: no fitted model exists after support closure. Primary slots [3, 4, 5, 6] remain unused; no slot reallocation.

### 12.3 R1 Historical lineup law and supported roster bounds (`sports_improvements_20261009_220719Z` / `_v3`)
- **Protocol & Execution:** Causal candidate builder from 41,915 source series / 69,370 source maps using strictly prior exact-context appearances.
  - Support: 63,131 TRAIN side labels (4,824 other, 2,041 previous, 56,266 latest), 2,110 CAL side labels, 2,520 ASSESS series. CAL class presence audit passed (1,950 latest, 124 other, 36 previous).
  - Exactly two TRAIN fits executed once: conditional 6-parameter law (objective 0.589805, KKT residual $< 10^{-5}$) and global 2-parameter transition law (objective 0.596043, KKT residual $< 10^{-5}$). Zero CAL fits, zero C0 fits.
  - Evaluated supported roster bounds $[L, L+u]$ on 2026 assessment series using native C0 scenario pairs. Where positive OTHER mass exists, unsupported mass $u = a_2 + b_2 - a_2 b_2 > 0$ and unconditional scalar probability is withheld (`prob_a = None`). Lineup law feasibility established; bounds reflect unsupported roster mass, not a confidence interval. Native C0 reference, model weights, config, and serving remain completely unchanged.

### 12.4 T1 & D1 Market branches
- Evaluated against prerequisite records under `prerequisite_resolution.json`:
  - T1: Authentic all-state coherent pre-decision quotes at T−24h and T−1h (both $\le 300\text{s}$ receipt age), multiple common underlying books, and complete sports OOF inputs remain unavailable. `BLOCKED_AUTHENTIC_INPUT_PREREQUISITES`. Zero head fits, zero base fits.
  - D1: Licensed/admitted closed-TRAIN teacher quotes with verified observation clocks and pre-2025-07-01 maturity remain unestablished. `BLOCKED_ADMITTED_TEACHER_PREREQUISITES`. Zero neural fits, zero base fits.
- Both market branches remain unexecuted with zero fits; preserved deadline before 2026-10-31T00:00:00Z.

## 13. Execution evidence, retained failure and reproduction

The [numerical execution receipt](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/numerical_execution.json) records **PASS: six mathematical families, fifteen named diagnostic checks, 565 independent assertions, 367 scalar comparisons, 165 numeric comparisons, zero failed assertions, 27 saved-fact records, maximum absolute numeric error 2.220446049250313e−16**. Counts include repeated structure/type/presence and scalar/source comparisons; they are not independent statistical experiments. Two synthetic distributions exercised the actual `DistributionPrediction.duration_over` API, maximum error **5.551115123125783e−17**.

The first independent attempt failed **two type assertions**: concentration constants 2/20 were serialized 2.0/20.0. Numeric values agreed; only integer serialization changed. Formulas, tolerance, oracle, model and evidence inputs did not. Original [diagnostics_attempt1.py](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/diagnostics_attempt1.py), [diagnostic_results.json](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/diagnostic_results.json), [independent_verification.json](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/independent_verification.json) and [failure receipt](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/verification_attempt1_failure.json) remain retained; successful outputs use new v2 names.

**Thirteen protected artifact hashes were checked unchanged at the numerical gate**, not across the whole checkout. This run records **0 fits, 0 sports/odds acquisition, 0 deployments, 0 genuine future forecasts**. Saved decision-field equality is not a metric/CI replay. No full-model inference, historical forecast regeneration, old statistical gate, application/UI suite or bankroll replay is claimed. Older whole-project smokes belong to the separate project-deep-research study and were not newly run here. The [scientific design review](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/scientific_review.json) and [resolution receipt](../data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01/scientific_review_resolution.json) clarify the P1 mean-CRPS guard; this is design review, not fresh performance certification.

To reproduce only the delivered diagnostic/oracle scope from repository root, choose fresh output names; both scripts refuse reuse. The following uses distinct review names, not existing or failed receipts:

```sh
E=data/08_reporting/sports_model_betting_reliability/sports_reliability_20261009_01
.venv/bin/python "$E/diagnostics.py" --root . --output "$E/diagnostic_results_review1.json"
.venv/bin/python "$E/independent_oracle.py" --root . --results "$E/diagnostic_results_review1.json" --output "$E/independent_verification_review1.json"
```

If a reproduction fails, retain its outputs and use another fresh pair after any justified correction. These commands are reproduction instructions, not additional executions reported here. Mathematical consistency establishes the cited illustrations and saved-fact equality; it does not validate statistical assumptions, authentic sports inputs or profitability.

## References

[1] Gneiting, T. and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf. Evidence/access/transfer scope: inspected primary full-text article; proper scoring, calibration and sharpness principles, not LoL or betting-profit results.

[2] Glickman, M. E. (1999). Parameter Estimation in Large Dynamic Paired Comparison Experiments. https://www.glicko.net/research/glicko.pdf. Evidence/access/transfer scope: author-hosted full text; dynamic uncertainty, causal filtering versus smoothing; no demonstrated C0 replacement.

[3] Guan and Tung (2026). Pre-game paired-comparison modeling of professional League of Legends map outcomes. https://arxiv.org/html/2609.08060v1. Evidence/access/transfer scope: inspected primary preprintv1; map target and side/first-pick information, not series/equal-time market equivalence.

[4] De Bois et al. (2025). PandaSkill — Player Performance and Skill Rating in Esports: Application to League of Legends. https://arxiv.org/html/2501.10049v2. Evidence/access/transfer scope: inspected vendor-affiliated author preprint; postgame performance and rating methodology, not independent prematch accuracy proof.

[5] Koopman, S. J. and Lit, R. (2019). Forecasting football match results in national league competitions using score-driven time series models. https://research.vu.nl/ws/files/151597369/Forecasting_football_match_results_in_national_league_competitions_using_scoredriven_time_series_models.pdf. Evidence/access/transfer scope: inspected primary full text; likelihood/state-update analogy from football, not LoL constants or fixed-weight equivalence.

[6] Zhao, S., Kim, M., Sahoo, R., Ma, T. and Ermon, S. (2021). Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration. https://proceedings.neurips.cc/paper_files/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf. Evidence/access/transfer scope: inspected NeurIPS full text; specified population-average loss/action classes, excludes directX loss dependence; no automatic varying-odds theorem transfer.

[7] Hébert-Johnson et al. (2018). Multicalibration: Calibration for the (Computationally-Identifiable) Masses. https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf. Evidence/access/transfer scope: inspected ICML full text; specified subgroup class/support conditions, not arbitrary selected-return certification.

[8] Walsh, C. and Joshi, A. (2023). Machine learning for sports betting: should model selection be based on accuracy or calibration? https://arxiv.org/html/2303.06021. Evidence/access/transfer scope: inspected original author preprint; NBA simulation design only; affected numerical results superseded by[9].

[9] Walsh, C. and Joshi, A. (2025). Corrigendum to “Machine learning for sports betting: should model selection be based on accuracy or calibration?” https://researchportal.bath.ac.uk/en/publications/corrigendum-to-machine-learning-for-sports-betting-should-model-s/. Evidence/access/transfer scope: inspected full correction on author-affiliated Bath record; corrected average, fixed-stake and eighth-Kelly NBA results, not LoL evidence.

[10] Clarke, S., Kovalchik, S. and Ingram, M. (2017). Adjusting Bookmaker’s Odds to Allow for Overround. https://www.sciencepg.com/article/10.11648/j.ajss.20170506.12. Evidence/access/transfer scope: inspected publisher metadata/abstract; three sports datasets and named adjustment methods only, not universal LoL superiority or inaccessible derivations.

[11] STS (2024). Jakie podatki są pobierane? https://pomoc.sts.pl/jakie-podatki-sa-pobierane/. Evidence/access/transfer scope: inspected operator help, updated2024-06-25;12% stake/.88 assumption only, not conflicting high-win threshold shorthand or accepted wagers.

[12] STS (2026). Kupon SOLO oraz AKO. https://pomoc.sts.pl/kupon-solo-oraz-ako/. Evidence/access/transfer scope: inspected operator help, updated2026-03-27; ordinary SOLO payout formula, not execution authenticity.

[13] Polish Ministry of Finance (2025). Stawki i limity. https://www.podatki.gov.pl/pozostale/podatek-od-gier/stawki-i-limity. Evidence/access/transfer scope: inspected official page, updated2025-12-15; named12% mutual-betting tax category, not comprehensive live legal/cash certification.

[14] Sun, Q. and Boyd, S. (2018). Distributional Robust Kelly Gambling. https://web.stanford.edu/~boyd/papers/pdf/robust_kelly.pdf. Evidence/access/transfer scope: inspected author manuscript; worst-case log growth over an explicit ambiguity set, not a validated C0 uncertainty set.

[15] Busseti, E., Ryu, E. K. and Boyd, S. (2016). Risk-Constrained Kelly Gambling. https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf. Evidence/access/transfer scope: inspected author manuscript; drawdown constraint under the assumed return process, not probability repair or universal live risk bound.

[16] Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2015). The Probability of Backtest Overfitting. https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf. Evidence/access/transfer scope: inspected February2015 author manuscript; strategy-rank overfitting/CSCV assumptions, not probability of losing money; journal version2017 noted in source catalog.

[17] Waudby-Smith, I. and Ramdas, A. (2023). Estimating means of bounded random variables by betting. https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf. Evidence/access/transfer scope: inspected RSS meeting manuscript; bounded mean/confidence-sequence assumptions, not arbitrary drifting selected LoL returns.

[18] Dmochowski, J. P. (2023). A statistical theory of optimal decision-making in sports betting. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0287601. Evidence/access/transfer scope: inspected primary open-access article; payout-sensitive theory and NFL spread/total analysis, not LoL binary profitability.

[19] Farinhas, A., Zerva, C., Ulmer, D. and Martins, A. F. T. (2024). Non-Exchangeable Conformal Risk Control. https://arxiv.org/html/2310.01262. Evidence/access/transfer scope: inspected author manuscriptv2; bounded monotone risk and explicit shift penalty, not assumption-free temporal calibration.

[20] Bai, T. and Jin, Y. (2026). Conformal Selective Prediction with General Risk Control. https://arxiv.org/html/2603.24704v1. Evidence/access/transfer scope: inspected primary preprint; selective/marginal bounded-risk guarantees under stated exchangeability/shift conditions, not automatic dependent sports deployment validity.

[21] Dern, N., Cunningham, J. P. and Pleiss, G. (2025). Theoretical Limitations of Ensembles in the Age of Overparameterization. https://proceedings.mlr.press/v267/dern25a.html. Evidence/access/transfer scope: inspected ICML/PMLR page and stated theory scope; random-feature regression caution, not a universal neural-ensemble failure theorem.
