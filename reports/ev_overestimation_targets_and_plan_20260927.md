# Reducing EV overestimation: learning targets, conditional calibration, and an executable research plan

Date: 2026-09-27. Status: **RESEARCH COMPLETE; EXPERIMENTS PROPOSED, NOT RUN.**

## Recommendation

**Keep the odds-free sports probability and series-outcome label. First test a separate, small market-aware probability model—not a raw-profit target, another arbitrary temperature, or an uncertainty haircut.**

The proposed change is to estimate the outcome conditional on both sports and contemporaneous market information, and assess calibration under the actual selection rule. It is not to declare bookmakers correct or push every underdog prediction toward the market. A learned correction must be allowed to fail.

Priority order:

1. Establish a usable decision-time ledger; distinguish historical as-of scenarios from genuinely recorded forecasts.
2. Keep native annual A0 fixed. Test one regularized, side-symmetric sports-plus-market head against strong enough market-only controls.
3. Measure probability quality, selection-conditional error, and actual policy value as three different questions.
4. Reopen a neural target ablation only for a new, specific hypothesis not answered by the earlier winner-versus-scoreline trial.
5. Freeze the surviving recipe before prospective shadow evaluation. Consider bankroll sizing only after flat-stake evidence and settlement correctness.

No retraining, new financial sweep, production access, scraping, deployment, or real wager was performed for this report. The old `uncertainty_betting_evaluation_run005` rankings remain invalid; this research does not rehabilitate them.

## 1. What the repository evidence establishes

The saved 48-hour diagnostic is annual research **Causal A0**, not the operational EXP081 market hybrid. On the common 275-fixture cohort, raw net EV greater than 5%, with 12% tax and no application gates, selected 82 bets:

| Quantity | Saved result |
|---|---:|
| Mean A0 probability on selected sides | 48.803% |
| Mean market reference probability | 33.401% |
| Wins | 32/82 = 39.024% |
| Mean predicted net EV | +31.698% |
| Realized flat-stake net return | −3.393% |
| Win-frequency minus predicted-probability gap | −9.779 percentage points |
| Realized-return minus estimated-EV gap | −35.091 percentage points |

These are **previously recorded results**, not a rerun. Only four selected monthly blocks support the reported intervals. Evidence for overstated EV is stronger than evidence for a negative population return: the saved actual-return intervals include zero. Neither the sample nor those intervals identify the cause as a wrong learning target.

More importantly, only **1 of 343** full 48-hour rows passes the existing strict-prior-day screen. Matchday research forecasts were retrospectively attached to earlier prices; actual forecast creation and roster-announcement times are not established. Filtering that table is not a substitute for generating forecasts from the correct earlier information set.

Local evidence and constraints:

- [Explanation of the saved 48-hour result](why_a0_48h_ev_bets_fail_20260927.md).
- [Previous calibration/learning/noise research](calibration_learning_and_noise_research_20260927.md) and [bookmaker/provider research](bookmaker_noise_and_provider_training_20260927.md).
- `coherent-series-20260912/WYNIKI.md`, relative to the configured research root: in a related controlled family, **scoreline minus winner ΔLL = −0.000184**, nominal monthly interval **[−0.000645, +0.000308]**. Both arms kept auxiliary player learning. The larger improvement over an older model also changed the head and format offset; it is not a target-only effect. This is not an untried idea, nor proof that the current native A0 recipe cannot benefit from a new, justified ablation.
- Native monthly no-decay versus annual training was inconclusive. H365/H730 experiments were exponential-weighting experiments with other differences, not an isolated native rolling-window test. Keep cadence/decay separate from this proposal.
- Previous global/tier calibration, fixed blending, and trained uncertainty experiments did not establish a general repair. Five-member spread or quantiles have not been shown to be per-bet confidence bounds or useful incremental error predictors.
- All repeatedly examined 2024–2026 results remain development evidence. A new chronological split inside those years does not create an untouched confirmatory test.

Research-root discovery remains `--research-root`, then `ENSEMBLE_RESEARCH_ROOT`, then `data/research_root.txt`. Frozen artifacts and the canonical manifest are unchanged. Of the previous delivery receipt’s 45 files, 43 research/evidence files retain their hashes; the two living handoff documents were intentionally extended. Their complete prior text and matching prior hashes are preserved in this run’s `handoff_before.json`.

## 2. What the literature supports—and does not

### 2.1 Estimate a probability before applying the payoff

Log loss and Brier loss are strictly proper: in the unrestricted population problem their unique optimum is the true conditional outcome probability. That does not guarantee calibration of a finite neural model, robustness to drift, or profitability after selection. Probability estimation and cost-sensitive decisions can nevertheless be separated when the relevant probability and payoff are known. [1,2]

For this application, keep these distinct:

- **Target:** realized series winner, terminal series score, player statistics, closing price, or another actual observed quantity.
- **Information set:** sports history alone versus sports plus a quote available at the decision time.
- **Loss:** log loss, Brier, scoreline likelihood, return-weighted loss, or a decision utility.
- **Decision:** pass/A/B, then stake size, execution, and settlement.

Native A0 already has richer supervision than a binary winner label: its neural component uses a Beta terminal-score likelihood and auxiliary player objectives, alongside other experts. Its combined training objective is not automatically strictly proper for the winner probability alone under model restriction and shared parameters. That is a possible research mechanism, not an established diagnosis of the 82 bets.

### 2.2 Calibration must survive the decision rule

Multiaccuracy controls residual correlations against a declared auditor class; ordinary group-average accuracy is weaker than calibration within score levels. Multicalibration adds those score-level conditions. Omnipredictor and decision-calibration results connect prediction quality to specified downstream decision classes, not arbitrary future policy searches. Representative data, subgroup support, learnability, and generalization assumptions matter. [9-12]

For betting, the relevant context includes both prices, chosen side, tax, probability, and selection threshold. Checking only global ECE or probability deciles can miss systematic error on the selected portfolio. Conversely, searching hundreds of tiny odds/league/confidence cells creates another estimation problem.

The optimizer’s curse describes a related but distinct mechanism: maximizing noisy value estimates can produce optimistic selected estimates even when the original estimates are unbiased. It is not proof that every independent, fixed betting policy must be biased. Hyperparameter/threshold search and conditional miscalibration are separate channels. The cited optimizer’s-curse source was inspected at abstract level, not as a verified full theorem proof. [13]

### 2.3 Direct decision learning is legitimate, but not a free calibration fix

Task-based learning and Smart Predict-then-Optimize provide real reasons to optimize downstream decisions when finite model capacity makes generic prediction loss misaligned with the task. Their demonstrated domains and consistency assumptions do not establish a cure for LoL EV overstatement. [6,7]

Do **not** claim that every utility-derived loss is improper: Bayes-act utility can induce a proper score. It need not be strictly proper. A pass/bet threshold or a stake cap often maps many probabilities to the same action, so good decisions need not identify a unique probability. Interior, unconstrained Kelly is a different case, with an invertible probability-to-optimal-stake mapping under its assumptions. Fractional Kelly, caps, overlap, and constraints change that argument. [1,8]

### 2.4 Sports papers are precedents, not transferable effect sizes

| Evidence | Relevant lesson | Transfer limit |
|---|---|---|
| Hubáček, Šourek, Železný: NBA winner prediction with a market-decorrelation objective | Changing the loss or market relationship can matter without changing the winner label. | Decorrelation is not proper market-offset calibration; excessive decorrelation can hurt. Historical NBA simulated returns do not validate this LoL candidate. [14] |
| Reade, Singleton, Vaughan Williams: EPL scoreline and result markets | Richer scoreline forecasting and forecast-encompassing evidence need not produce consistent betting returns. | Different sport, markets, margins, and a two-season sample. Multiple quoted scorelines are not independent matches. [15] |
| Simon: real-time MLB price movements | Closing prices need not improve monotonically; detectable movement is not automatically a tradable net profit. | Not a supervised closing-price-target comparison and not a LoL study. [16] |
| Walsh and Joshi: calibration-versus-accuracy model selection | Probability assessment matters more than winner classification accuracy for a price-dependent decision. | One betting season and selected configurations do not establish repeatable ROI, conditional EV calibration, or an A0 target replacement. [17] |

## 3. Should we change the learning target?

### 3.1 Return regression mostly changes weighting

Let `Y` be the selected side’s win indicator, `O` its executable decimal odds, and `c = 1 − tax`. Per-unit return and predicted EV are

\[
R=cOY-1,\qquad \widehat{EV}=cOp-1.
\]

If a return predictor is parameterized through the same probability,

\[
(R-\widehat{EV})^2=(cO)^2(Y-p)^2.
\]

Thus return MSE is **odds-squared-weighted Brier loss**. At fixed observed `X,O`, its unrestricted probability optimum is still `P(Y=1|X,O)`. With limited capacity it prioritizes high-odds observations and can amplify their influence. This derivation is not evidence that such weighting improves this dataset.

If the model omits informative odds and still trains that probability-parameterized return loss, for constant tax its optimum instead becomes

\[
p_{\mathrm{opt}}(X)=\frac{E[O^2Y\mid X]}{E[O^2\mid X]},
\]

not necessarily `P(Y=1|X)`. An unconstrained direct regressor of `R` without odds estimates `c E[OY|X]−1`, also not the return at an arbitrary new quote. Calling either construction odds-free because odds were omitted from runtime features would hide market-dependent training.

**Decision:** do not replace A0’s target with raw return as the first repair.

### 3.2 A market residual is usually a reparameterization

With quote-time `q` in the conditioning information, squared-error regression of `Y−q` estimates `P(Y=1|X,q)−q` in the unrestricted problem. A Bernoulli model with `logit(q)` as an offset also still predicts the winner. The useful change is the information set and restricted function class—not new outcome information.

An unconstrained probability-scale residual can leave the unit interval; a logistic model avoids that issue. Neither formulation implies that market probabilities are correct. Shrinkage toward a market reference is a regularization choice that must beat controls out of time.

### 3.3 Comparison of alternatives

| Alternative | What it actually learns or changes | Decision for this project |
|---|---|---|
| Series-win log loss/Brier | Conditional winner probability, subject to information and model restrictions | Keep as the evaluation primitive; retain native A0 as the frozen anchor. [1,2] |
| Winner instead of terminal-score main loss | Changes allocation of finite capacity within a coherent series model | Conditional later ablation; related comparison already inconclusive. Preserve auxiliary weight, head, experts, and schedule. |
| Market-offset/joint probability model | Winner probability conditional on sports and contemporaneous market information | **First modeling candidate**, separately identified from odds-free A0. |
| Raw return MSE | Quote-dependent mean return; under probability parameterization, weighted Brier | Not new label information; higher variance/influence at long odds. Not first. |
| Class weighting | With weights `w1,w0`, log-loss optimum is `w1 p / (w1 p + w0(1−p))` | Not a calibrated-p repair unless the sampling/tilt is explicitly corrected. [2] |
| Focal loss | Classification-calibrated decisions, but raw population scores generally not true posteriors | Not a default repair; a theoretical posterior-recovery transform exists, and finite-model gains remain empirical. [3] |
| Uniform label smoothing | For binary labels, population target becomes `(1−ε)p + ε/2` | Can regularize finite networks but also raises underdog probabilities toward 0.5; not automatic conservatism. [4] |
| Ranking/high-confidence classification | Ordering or class decisions, not necessarily moneyline probability | Cannot directly substitute for calibrated probabilities at price-dependent thresholds. [5] |
| Closing price/CLV | Future market price or movement, not the realized-outcome probability | Later auxiliary market-model hypothesis only; close may be a released past TRAIN label, never a future 48-hour input. [16] |
| Direct profit/Kelly/task loss | Utility of a specified policy, possibly without probability identification | Legitimate later policy research, after a valid payoff/settlement contract; not a small-sample cure. [6-8] |

Positive weights that are functions only of the complete conditioning information preserve an unrestricted proper-loss optimum. This statement does **not** cover label-dependent weights, informative omitted variables, or identical finite-capacity fits.

A label such as “this was a profitable bet” is usually just the realized win/loss transformed by payout; it is not a ground-truth label for “the true EV was positive.” One outcome does not reveal a match’s true probability.

## 4. The specific correction worth testing first

### 4.1 Two outputs, two identities

- `p_s`: native annual A0 **`p_full`**, including its existing expert mixture/calibration. Do not confuse this with a raw neural logit.
- `q`: existing same-cutoff market consensus from properly aligned, within-book no-vig probabilities, one selected quote per bookmaker.
- `p_h`: a new market-aware research prediction. It is not a new version of the odds-free sports engine and is not the existing operational hybrid.

Proposed candidate:

\[
p_h=\sigma\{u\,\operatorname{logit}(p_s)+v\,\operatorname{logit}(q)\},
\quad u,v\ge0.
\]

Fit Bernoulli log loss on **all eligible prior out-of-time fixtures**, not only selected bets. Regularize toward `(u,v)=(0,1)`:

\[
\frac1n\sum_i\ell_{\log}(Y_i,p_{h,i})
+\frac{\lambda}{2}\{u^2+(v-1)^2\}.
\]

This market-centered prior is a conservative complexity choice, **not a claim of market truth or betting safety**. The model can sharpen as well as soften predictions. Since `u+v` is not constrained to one, it is not necessarily a convex blend and need not lie between the inputs.

No arbitrary A/B intercept: A/B is data orientation, not a known physical advantage. Swapping teams negates both logits and complements the output. Nonnegative coefficients impose monotonicity in each signal; they are not what creates side symmetry. A real known advantage would need an appropriately signed, available input—not a constant orientation bias.

### 4.2 Controls that prevent a misleading win

| ID | Forecast | Purpose |
|---|---|---|
| S | Frozen native `p_full` | Preserve the actual sports reference. |
| S-cal | Positive-slope recalibration of `logit(p_s)` | Detect a gain that is only ordinary sports recalibration. |
| M | Raw same-cutoff consensus `q` | Transparent market reference, not a truth label. |
| M-cal | `sigmoid(v logit(q))` | Nested market-only primary comparator; same calibration chronology and penalty search. |
| M-shape | Side-symmetric monotone two-slope calibration of market logit | Guard against mistaking simple nonlinear market recalibration for added sports information. |
| Fixed50 | Existing 50/50 logit blend | Prior simple-blend reference, not the proposed learned model. |
| H | The two-coefficient candidate | The only new sports-plus-market candidate in the first experiment. |

For a bounded, reproducible M-shape control, propose one fixed knot `k=logit(0.8)` and

\[
g(z)=\operatorname{sign}(z)\{a\min(|z|,k)+b\max(|z|-k,0)\},\quad a,b\ge0;
\quad p=\sigma(g(z)).
\]

Regularize its slopes toward `(1,1)`. The knot is a declared engineering choice, not a source-derived optimum; do not tune it on the final period. This control is not a competing large model search. Even beating it establishes added predictive value relative to the declared market-only family, not independence from every possible market feature or calibration function.

Proposed fitting contract:

- Expanding strict-past data; no recency weighting in this experiment.
- Refit the correction monthly by **decision origin**, not target-match month; the sports model’s native annual schedule remains unchanged.
- Candidate regularization grid `{0.01, 0.1, 1.0}` on the **mean-loss** scale above. Choose only with inner expanding chronological log loss; tie goes to stronger regularization. Give fitted controls the same opportunities and log every fit. These are proposed protocol values, not optimized findings.
- All base predictions used for fitting must be out of time for the **whole native recipe**, including its calibrator and final mixture. In-sample native CAL predictions are not relabeled OOF.
- Outcome labels must have been released before each fit. At a year/month boundary, use the artifact actually permitted at the decision origin, not the artifact for the future match date.
- Missing fitting support produces `UNAVAILABLE/WARMUP`, not a fitted-looking fallback. Preserve those fixtures in the coverage ledger.
- Reuse the benchmark’s probability policy consistently in headline, paired, and profile metrics. Keep model/calibrator/quote IDs and coefficient histories.

This differs from another temperature or fixed 50/50 blend: it learns the conditional contribution of the sports forecast while explicitly allowing and testing market recalibration. It remains an **untested hypothesis**.

### 4.3 Audit the error that actually creates EV overstatement

For a fixed selected side,

\[
R-\widehat{EV}=cO(Y-p).
\]

Therefore check both `mean(Y−p)` and `mean(cO(Y−p))`. Zero unweighted residual does not imply accurate EV when odds vary. A payout-weighted auditor connects directly to the financial error, provided its policy class, boundedness/support, and inference assumptions are stated. [9-12]

Use a short, frozen audit list: selected versus unselected, fixed odds bands already used in the benchmark, model–market disagreement, and time. Report unknown profiles. Do not fit an unrestricted odds×league×BO×uncertainty correction to 82 bets.

Distinguish three populations:

1. **All common eligible fixtures:** paired log loss/Brier and coverage.
2. **Fixed reference-selected fixture IDs and sides:** compare conditional probability and EV-estimation errors. With identical odds, stakes, and outcomes, realized PnL is identical across predictors; improved calibration is not improved realized return on this fixed set.
3. **Each model’s own selected bets:** apply the same frozen EV rule separately. This measures policy value and changes membership. It is not the fixed-set comparison.

## 5. Conservative decisions: what can be guaranteed?

| Method | Actual scope | Why it is not a ready-made betting guarantee |
|---|---|---|
| Conformal risk control | Marginal expected bounded, monotone loss under exchangeability and stated regularity | Conditional selected-bet loss need not be monotone in the hurdle. No automatic per-match probability or ROI bound. [18] |
| Learn then Test | Selection among candidate rules using valid risk-test p-values and multiplicity control | Standard setup uses IID calibration; the dependence/drift here does not automatically satisfy it. An aggregate/selective risk guarantee is not a per-bet lower probability. [19] |
| Confidence sequences/e-processes | Time-uniform inference under conditional process assumptions | Optional-stopping protection does not validate an incorrect dependence model or unlogged policy search. The average estimand is not every match’s probability. [20] |
| Adaptive conformal inference | Long-run prediction-set coverage under its online update; stronger claims need more assumptions | Long-run coverage is not calibrated win probability, positive EV, or a safe stake. [21] |
| Bayesian/robust Kelly | Log-wealth optimization under a posterior model or a declared ambiguity set | Prior/likelihood and set validity matter. Ensemble quantiles do not certify that the true distribution lies in that set. [22,23] |
| Backtest-overfitting diagnostics | Diagnose selection over a specified strategy family | Not a replacement for chronological, untouched forward evaluation. [24] |

A valid lower bound would need a clearly named estimand and an independently validated construction. Until then, `p_low`, seed spread, bootstrap-member quantiles, and disagreement haircuts remain heuristics. Reducing a stake limits exposure; it does not fix the forecast’s bias.

## 6. Staged plan and acceptance gates

All work below is **proposed**, not implementation completed in this session.

### Stage A — Construct the decision-time evidence contract

**Deliverable:** one fixture/opportunity ledger for the 48-hour decision, with exclusion and missingness reasons. OPEN, 24-hour, and CLOSE are separate sensitivity populations; do not pool them to manufacture sample size.

Required lineage includes match/side/market identity; timezone-aware decision cutoff and scheduled start vintage; feature snapshot and latest release; roster policy plus evidence time; model and calibrator origin/hash; real forecast creation time; quote ID/book/first-observed time; selected executable payout; result availability and settlement status.

Two modes must remain explicit:

- **Historical scenario:** record the real current replay creation time separately from `simulated_origin`. Use only information allowed at that origin, with previously known rosters or evidenced announcements. Daily release assumptions remain labeled assumptions. Never backdate a real creation timestamp or call reconstructed prices recorded production decisions.
- **Prospective shadow:** append the actual pre-match forecast and quote records before outcomes. This requires a validated live A0 adapter; the current operational EXP081 hybrid is not one.

**Acceptance:** future outcomes, roster changes, quotes, and schedule revisions cannot change earlier snapshots; one quote side is aligned exactly once; a future monthly/yearly artifact cannot leak across the origin. Quote observation alone does not prove account-level executability. If usable history cannot be reconstructed, report the exact missing support and begin collecting genuine shadow evidence rather than certifying the 1/343 slice.

### Stage B — Fit one low-capacity conditional candidate

**Deliverable:** chronological predictions for S, S-cal, M, M-cal, M-shape, Fixed50, H; coefficient/calibration receipts; complete fit and missingness ledger.

Use the exact contract in §4. Keep A0 features, experts, training target, seed recipe, and annual schedule fixed. Fit the head only after Stage A can supply the needed data; do not treat outcome-selected rows or the final 82 bets as its training set.

**Acceptance:** symmetry, strict fit-label release, complete-recipe OOF status, cold/missing-history behavior inherited from the frozen sports contract, finite valid probabilities, and reproducible predictions. No claim of gain from the fitted sports coefficient being nonzero.

### Stage C — Decide whether the correction helps

**Primary predictive contrast:** H minus M-cal log loss on identical eligible fixtures. Require a negative point estimate and a nominal paired monthly-block 95% interval entirely below zero for a historical progression claim. Require the aggregate Brier point estimate not to worsen; report its full interval rather than calling that a noninferiority proof. Apply the same proper-score gate against M-shape before describing the result as added sports value beyond the declared market-only controls. Comparisons with S and Fixed50 are explicit secondary questions, not substitutes if the primary comparison fails.

**Calibration contrast:** fixed reference-selected sides and each policy’s own sides, separately. Report probability residual, payout-weighted residual, conditional proper scores, odds/disagreement slices, and their support. The existing practical tolerances of **±3 percentage points for win-frequency error and ±5 points for EV/yield error** can be retained as proposed task-specific targets. Claiming error lies inside a band requires the relevant interval to lie inside it; containing zero is not equivalence. These bands are not universal production criteria.

**Policy contrast:** flat unit stakes, raw net `EV > 0.05`, tax `0.12`, highest-EV side and the existing deterministic tie rule. No new uncertainty filter, odds cap, or threshold search in this first comparison. Preserve actual quote prices; never substitute `1/q` as an executable price.

- Primary policy summary: paired net PnL per eligible opportunity on the same opportunity universe; abstention contributes zero cashflow.
- Also report per-bet ROI, number and fraction of bets, distinct fixtures, months, stake exposure, concentration, and maturity.
- Zero bets means conditional ROI and conditional calibration are **undefined**, not zero. An abstain-all policy cannot pass merely by having zero error or zero losses. Before outcomes, set a coverage requirement from the intended operating volume and a precision plan; no universal numeric sample minimum is claimed here.
- Use at least 5,000 paired calendar-month bootstrap draws from the entire pre-selection quote-bearing universe. Keep no-bet months. If conditional statistics are undefined in replicates, disclose the fraction and leave that interval unavailable rather than silently discarding or imputing them. Report match-weighted and equal-month summaries.
- Add tournament/edition clustering sensitivity where identification permits; many bets or quotes on one series do not create independent outcomes. Few blocks, long dependence, or unstable tails yield `INCONCLUSIVE`. A bootstrap sign fraction is not a classical p-value.
- Score only matured outcomes under a fixed censoring rule. No instant-settlement Kelly or replaying shuffled months as if they formed a valid bankroll path.

**Separate policy-value gate:** for historical financial progression, require positive H net PnL per eligible opportunity and a positive paired increment over M-cal, with both nominal 95% lower interval bounds above zero, the predeclared coverage requirement met, and conditional ROI defined. Otherwise a predictive improvement is not an established betting edge. This gate is an approximate, dependence-sensitive progression rule—not a distribution-free profit guarantee or independent confirmation.

**Decision:** `REJECT` for integrity/contract failure or consistently worse performance; `INCONCLUSIVE` for unresolved comparisons/support; `DEVELOPMENT_CANDIDATE` only for the specified historical gates. None is a production certificate. A model can improve probability scores and still fail the net-value question after tax.

### Stage D — Conditional target experiment, not the first default

Open this stage only if the remaining well-supported error suggests a specific scoreline/auxiliary tradeoff and the protocol states what is new relative to the earlier controlled trial. If there is no new hypothesis, do not rerun that comparison under a new name.

**Two arms only:** unchanged native control versus the same current Beta/series head with the main terminal-score NLL replaced by series-winner NLL computed from that same coherent series distribution. Retain auxiliary coefficient, other experts, initialization pairing, data, update schedule, calibration procedure, and ensemble recipe. Do not add a new head and new supervision simultaneously and call that target-only. Do not name this candidate “A1,” which already has a different, withdrawn identity here.

Record main/auxiliary loss and gradient scales: retaining the numerical auxiliary coefficient does not preserve its relative gradient influence when the main loss changes. Interpret the result as a controlled loss-recipe comparison, not proof that richer labels alone caused an effect.

**Acceptance:** chronology, side symmetry, empty histories, valid scoreline mass, and no double BO projection first; then the canonical full-cohort benchmark and identical information-valid market rows. Report winner and scoreline/length quality separately. The candidate cannot inherit A0 tournament validity merely by improving winner LL.

Cadence/decay is a separate experiment lane. CLV auxiliary supervision and direct decision-focused objectives remain lower-priority hypotheses, not items to run automatically after an inconclusive result.

### Stage E — Forward shadow, then only the necessary financial layer

Freeze one selected recipe, data/quote policy, update schedule, primary comparison, coverage/precision requirements, and observation endpoint before unseen outcomes. Record every eligible fixture, including abstentions and missing predictions. Do not repeatedly stop at a favorable ROI; an anytime-valid alternative would first require a justified process model. [19,20,24]

Only after forward probability, conditional-error, coverage, and flat-policy results support the intended use should a separate bankroll simulation reserve overlapping stakes, use free capital, settle after actual result availability, and prove idempotent atomic accounting. Kelly/robust sizing is not evidence that the underlying edge exists. [8,22,23]

**No automatic deployment.** A successful result supports only its recorded population, horizon, execution assumptions, and policy. A wide interval or inadequate number of effective periods remains an unresolved research result.

### Integration boundaries for a later implementation

| Existing location | Reuse/extend; do not create a competing path |
|---|---|
| `src/analysis/production_market_benchmark.py` | Quote selection, within-book alignment, consensus, A0 panel/selection contracts, common populations, calibration reporting. |
| `src/analysis/thesis_models.py` | Existing positive-slope symmetric calibration and strict-past OOS fitting patterns; do not assume its generic model families already train native A0. |
| `src/analysis/unified_evaluation_engine.py` | Shared EV/decision definitions; current conservative-probability helpers remain heuristics without new validation. |
| `scripts/run_unified_betting_evaluation.py` | Extend the existing audited evaluation entry point with explicit candidate identity and protocol; no second evaluator in a dated folder. |
| `src/models/a0_accelerator.py` | Native target boundary for a later, explicitly versioned experiment. Preserve default frozen-recipe fidelity and the external archived implementation. |
| `scripts/run_model_benchmark.py`, canonical manifest | Rescore any actual later sports candidate under the existing contract. Do not modify canonical hashes to admit changed data. |
| Existing `test_production_market_benchmark.py`, `test_production_a0_calibration.py`, `test_thesis_models.py`, `test_unified_evaluation_engine.py`, `test_a0_uncertainty.py` | Extend observable chronology, symmetry, quote selection, conditional-statistic, and execution regressions when those paths are implemented. These suites were not run for this research-only task. |

Artifacts for later runs follow the repository’s `data/04_feature`, `05_model_input`, `06_models`, `07_model_output`, and `08_reporting` stages. Record full source hashes, membership IDs, fit-label availability, origins, every candidate tried, and explicit failure reasons.

## 7. Executed verification and limits of this report

Four independent literature tracks plus one repository/prior-experiment audit informed the synthesis. Four substantive reasoning reviews completed through the harness’s strongest `slow` completion setting; the harness does not disclose a named backend. Their recommendations were reviewed, not accepted by vote.

Important corrections during synthesis:

- Rejected a proposed arbitrary A/B intercept because it violates this repository’s orientation contract.
- Corrected “same threshold” versus “same selected IDs”: the former can reselect; the latter cannot change flat PnL at fixed quotes/stakes.
- Rejected adding a new winner head while retaining all old losses as an allegedly isolated target replacement.
- Reordered the proposal to conditional probability correction before another winner/scoreline experiment, given the prior local evidence.
- Added a nonlinear market-only control and limited the incremental-signal claim to the declared comparator family.
- Corrected the collected Reade-paper author metadata against the primary PDF: **J. James Reade, Carl Singleton, Leighton Vaughan Williams**.

Executed reproducible mathematical investigation:

```bash
.venv/bin/python data/08_reporting/ev_target_research_20260927_212506/verify_target_math.py --output-dir data/08_reporting/ev_target_research_20260927_212506
```

**PASS: 239 checks.** These are deterministic population/toy calculations, not trained-model performance or a financial backtest:

- Proper-loss optima, class tilting, smoothing, and raw focal-loss optima.
- Return-MSE/weighted-Brier identity, maximum numerical discrepancy `2.84e-14`.
- Omitted-odds example: ordinary sports probability `0.5`, odds-weighted optimum `0.24954`.
- A globally calibrated toy predictor whose EV-selected sides have estimated EV `+0.76` but true expected return `−0.296`; this proves possibility, not A0’s causal failure mechanism.
- Logit-blend side symmetry, maximum complement error `1.11e-16`.
- A counterexample in which shrinking an already-correct sports prediction toward an imperfect market worsens expected log loss by approximately `0.05834`.

Evidence, source ledgers, mathematical output, review records, and delivery checks are under `data/08_reporting/ev_target_research_20260927_212506/`. Bibliography integrity was checked separately from source content. Two bibliography entries, decision calibration and the optimizer’s curse, were accessed only at publication-record/abstract level; their full proofs were not verified here. Other papers had full-text access in the source collection; parent verification concentrated on decision-relevant claims and metadata rather than claiming every page was independently reread.

**Unresolved:** whether an eligible historical 48-hour cohort has enough support; whether sports information improves on calibrated market controls at that horizon; whether any candidate reduces both selected-bet error and net-return uncertainty; whether a prospective edge exists. No cited theorem or this report removes those empirical questions.

## Bibliography

[1] Gneiting, T., and Raftery, A. E. (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). JASA. Full-text source.

[2] Elkan, C. (2001). [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf). IJCAI. Full-text source.

[3] Charoenphakdee, N., Vongkulbhisal, J., Chairatanakul, N., and Sugiyama, M. (2020 preprint). [On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective](https://arxiv.org/abs/2011.09172). Full-text source; posterior-recovery conditions matter.

[4] Müller, R., Kornblith, S., and Hinton, G. (2019; accessed 2020 revision). [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629). Full-text source.

[5] Burges, C., et al. (2005). [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf). ICML. Full-text source.

[6] Donti, P. L., Amos, B., and Kolter, J. Z. (2017). [Task-based End-to-end Model Learning in Stochastic Optimization](https://arxiv.org/abs/1703.04529). NeurIPS. Full-text source.

[7] Elmachtoub, A. N., and Grigas, P. (2022; preprint 2017). [Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005). Management Science. Full-text source.

[8] Kelly, J. L., Jr. (1956). [A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf). Bell System Technical Journal. Full-text source.

[9] Kim, M. P., Ghorbani, A., and Zou, J. (2018 preprint). [Multiaccuracy: Black-Box Post-Processing for Fairness in Classification](https://arxiv.org/abs/1805.12317). Full-text source.

[10] Hébert-Johnson, U., Kim, M., Reingold, O., and Rothblum, G. (2018). [Multicalibration: Calibration for the (Computationally-Identifiable) Masses](https://proceedings.mlr.press/v80/hebert-johnson18a.html). ICML. Full-text source.

[11] Gopalan, P., Kalai, A. T., Reingold, O., Sharan, V., and Wieder, U. (2022). [Omnipredictors](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2022.79). ITCS. Full-text source.

[12] Zhao, S., Kim, M., Sahoo, R., Ma, T., and Ermon, S. (2021). [Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration](https://proceedings.neurips.cc/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html). NeurIPS. **Abstract/publication record only.**

[13] Smith, J. E., and Winkler, R. L. (2006). [The Optimizer’s Curse: Skepticism and Postdecision Surprise in Decision Analysis](https://scholars.duke.edu/publication/798952). Management Science. **Abstract/publication record only.**

[14] Hubáček, O., Šourek, G., and Železný, F. (2019). [Exploiting sports-betting market using machine learning](http://ida.felk.cvut.cz/zelezny/pubs/ijf.2019.pdf). International Journal of Forecasting. Full-text source.

[15] Reade, J. J., Singleton, C., and Vaughan Williams, L. (2020). [Betting markets for English Premier League results and scorelines: evaluating a forecasting model](https://www.reading.ac.uk/web/files/economics/emdp202003.pdf). Reading discussion-paper copy. Full-text source.

[16] Simon, J. (2024). [Inefficient Forecasts at the Sportsbook: An Analysis of Real-Time Betting Line Movement](https://doi.org/10.1287/mnsc.2022.00456). Management Science 70(12), 8583–8611. [Accepted manuscript](https://ndownloader.figshare.com/files/59330750), [repository metadata](https://api.figshare.com/v2/articles/30546293). Full-text source; repository upload date is not publication year.

[17] Walsh, C., and Joshi, A. (2024 revision; preprint 2023). [Machine learning for sports betting: should model selection be based on accuracy or calibration?](https://arxiv.org/abs/2303.06021). Full-text source; single-season empirical limitations.

[18] Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. (2022 preprint). [Conformal Risk Control](https://arxiv.org/abs/2208.02814). Full-text source.

[19] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., and Lei, L. (2021; accessed v5, 2022-09-29). [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052). Full-text source. The HTML rendering’s later internal date is not treated as the publication date.

[20] Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240). Annals of Statistics. Full-text source.

[21] Gibbs, I., and Candès, E. J. (2021). [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170). NeurIPS. Full-text source.

[22] Browne, S., and Whitt, W. (1996). [Portfolio Choice and the Bayesian Kelly Criterion](https://business.columbia.edu/sites/default/files-efs/pubfiles/6343/bayes_kelly.pdf). Full-text source.

[23] Sun, Q., and Boyd, S. (2018). [Distributional Robust Kelly Gambling: Optimal Strategy under Uncertainty in the Long-Run](https://arxiv.org/abs/1812.10371). Full-text source.

[24] Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2017; accessed 2015 revised manuscript). [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Full-text source.
