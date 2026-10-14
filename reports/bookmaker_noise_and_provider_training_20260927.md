# Bookmaker noise, provider training, and whether A0 needs monthly decay-weighted refits

**Original research date:** 2026-09-27; that investigation trained no forecasting model. **2026-09-29 updates:** the native-expert factorial and deeper bookmaker audit (§9), then a single-split data-focus/masking experiment (§10), are complete. Neither study established an adopted replacement.

**Model identity:** the original reference is annual research **Causal A0 series-win probabilities**. Section 9 evaluates four native neural-expert variants, not assembled full A0 models. Section 10 separately evaluates five new research assemblies and their neural components. None is production EXP081 or the operational market hybrid.

**Evidence:** [saved run](../data/08_reporting/bookmaker_noise_research_20260927_200205/), [source-access audit](../data/08_reporting/bookmaker_noise_research_20260927_200205/source_audit.json), [independent numerical verification](../data/08_reporting/bookmaker_noise_research_20260927_200205/verification.json).

## Executive conclusion

**We have not established that bookmakers are intrinsically less noisy or uniformly better calibrated, or that A0's expanding-history training is wrong.** On the same 2,673 archival matches, bookmaker OPEN consensus and A0 have very similar proper scores. CLOSE has better log loss and Brier, but a higher ECE10 point estimate than OPEN. That is evidence about historical forecast quality, not proof that closing prices are more calibrated or that monthly retraining explains their advantage.

The strength of the CLOSE-versus-A0 log-loss conclusion also depends on the blocking assumption: its monthly interval excludes zero, but a supplemental source-tournament-label interval **[−0.016807, +0.000055]** includes it. The point advantage is real in this dataset; robust superiority is not established.

A directly relevant disclosure exists: **Bayes Esports tested how much LoL history to use for map-winner odds**, reporting its best result around 5,000 games over 21 patches. It explicitly describes the tradeoff between recent relevance and sample size. It does **not** disclose exponential training weights, a monthly refit cadence, or an independently verified calibration result. Its map/side contract differs from our pre-match, unknown-side series forecast. [1]

Other providers publicly describe official data feeds, predictive models, human traders, market information, betting activity, and operator-specific liability adjustments. These describe a broader pricing system—not a single sports-only classifier whose displayed prices can be read as unbiased probabilities. Exact training objectives and calibration recipes mostly remain undisclosed. [3][6][8][9][10]

The code audit establishes a specific, testable distinction:

- Native A0's neural objective **has no explicit training-row age weights**.
- Its player ratings and historical features **already update with available history**; tokens include age. Annual refitting does not mean annual-only input updates.
- A previous monthly neural-expert trial, without sample decay, did not establish improvement over its January-only control.
- Previous 365/730-day experiments used **exponential weighting, not hard rolling windows**, but trained different recipes. They lost to A0 and do not identify the effect of weighting native A0.

**Original recommendation, 2026-09-27:** retain native A0 and test one bounded **annual/monthly × uniform/fixed-decay** experiment with the native recipe, common calibration rules and unchanged causal inputs. Expanding history and time decay can coexist; do not discard historical rating state. **Executed on 2026-09-29:** H365 worsened the controlled proper scores; cadence alone was inconclusive. Retain full A0; details in §9.

| Question | Answer supported now | Important boundary |
|---|---|---|
| Are the checked bookmakers perfectly calibrated? | No such conclusion follows from the diagnostics. | ECE is noisy, bin-dependent and conditional on the price conversion. |
| Does CLOSE contain a better historical forecast? | Lower archival LL/Brier points; CLOSE-versus-A0 LL uncertainty depends on blocking. | Archival quote timestamps are unavailable; this is not equal-time superiority. |
| Do providers use recency? | Bayes publicly investigated training-history length; dynamic sports methods explicitly model changing strength. | Neither establishes a universal LoL half-life or monthly cadence. |
| Is A0 missing any recency handling? | No: history age and evolving rating/history inputs already exist. | Explicit neural target-row decay is a distinct, absent mechanism in the audited fit. |
| Should we switch now? | No. The controlled H365 experiment now failed (§9); cadence alone is inconclusive. | All inspected historical years remain development-exposed. |

## 1. Scope, sequence, and evidence standards

The requested sequence was followed: **first evaluate bookmakers**, then investigate provider practice and temporal learning. New calculations use frozen saved forecasts and corrected quote identities; no provider account, production database, scheduler, scraper, or live betting service was used.

Five concurrent evidence collectors covered general providers, esports providers, market calibration research, dynamic sports methods, and actual repository training schedules. They were read-only `scout` workers, not the final research decision makers. In response to the request for stronger models, four substantive reviews used the documented **strongest available `slow` reasoning setting**: statistical review, refit design, provider synthesis, and native-decay critique. The exact backend model identity is not independently exposed. Their outputs and accepted/rejected conclusions are retained in [the review receipt](../data/08_reporting/bookmaker_noise_research_20260927_200205/strong_review_receipt.json).

Stronger reasoning was not treated as an oracle. One review incorrectly described the earlier H365/H730 runs as hard windows; source inspection showed `exp2(-age / half_life)`. That mistake was rejected. Suggestions to add hard-window searches and new dynamic architectures were also excluded from the bounded recommendation.

This report cites **26 external source records**, with access depth and transfer limits. Several are vendor disclosures, not independent performance studies. Clarke and Shin were accessible at abstract level; Dixon–Coles' original scan was not text-extractable, so its likelihood-weighting mechanism is additionally grounded in the fully read Koopman–Lit paper. No inaccessible equation, proprietary recipe, or vendor calibration score is invented.

## 2. First check: what the bookmakers actually do in our data

### 2.1 Cohorts and probability construction

The main population contains **2,673 identical archival fixtures**, spanning 2024-01-13 to 2026-03-20 across **26 active calendar months**. All three streams—annual A0, OPEN and CLOSE—cover those same fixtures. This is a subset of the 11,550 canonical forecasts, not the whole benchmark. The 2026 archival slice contains only **344 matches and three months**, not the full 3,103-match 2026 cohort.

Individual prices come from the corrected `market-identity-20260912/data/03_primary/aligned_book_quotes.parquet`: **25,736 book/timing/fixture quote rows**, six bookmakers. Their identities, orientation and reconstructed consensus were checked against the corrected snapshot and canonical probabilities. An older individual-bookmaker derivative predates documented identity repairs; it was not used. No archive files were changed.

For each bookmaker's two-sided decimal odds, first compute proportional de-vig probabilities:

\[
p_A = \frac{1/o_A}{1/o_A+1/o_B}.
\]

Then average the available books' probabilities per fixture and horizon. This is not the inverse of an average price, and not a synthetic pair of best odds from different books. It is a convenient consensus benchmark, **not an estimate with a known number of independent information sources**.

All retained probabilities satisfy `0 < p < 1`; the shared loss epsilon `1e-15` changes none. Main ECE uses the repository's right-closed equal-width bins `(lower, upper]`, with 5/10/20-bin sensitivity. The legacy left-closed ECE is saved separately. Logistic calibration slope/intercept fits are descriptive and are never applied back to scored forecasts.

### 2.2 Matched archival result

| Forecast, same N=2,673 | Log loss ↓ | Brier ↓ | AUC ↑ | ECE10 ↓ |
|---|---:|---:|---:|---:|
| Annual A0 | **0.590213** | **0.202548** | 0.746856 | 2.726% |
| Bookmaker OPEN consensus | 0.592155 | 0.203284 | 0.743056 | **1.795%** |
| Bookmaker CLOSE consensus | **0.581694** | **0.198706** | **0.756612** | 2.914% |

The lower OPEN ECE does not make OPEN the better forecast overall: CLOSE has better proper scores and discrimination. Conversely, the proper-score advantage does not isolate calibration as its cause.

Paired intervals use **5,000 calendar-month bootstrap draws**, seed `20260927`, and the same monthly draws for each pair. These are nominal historical intervals, not multiplicity-adjusted certification or proof that months are independent tournaments.

| Contrast | Match-weighted ΔLL | 95% monthly interval | Equal-month ΔLL |
|---|---:|---:|---:|
| OPEN − A0 | +0.001942 | [−0.009789, +0.012196] | −0.025149 |
| CLOSE − A0 | **−0.008519** | **[−0.017385, −0.001303]** | −0.033800 |
| CLOSE − OPEN | **−0.010461** | **[−0.014857, −0.006186]** | −0.008651 |

CLOSE − A0 Brier is **−0.003843**, interval **[−0.007733, −0.000844]**. OPEN − A0 is +0.000736, interval [−0.004241, +0.004869].

The equal-month estimand is intentionally different. Some archival months contain **one or two fixtures**, whereas others contain more than 200. Giving each month equal weight can reverse a small match-weighted comparison; neither estimand should be silently substituted for the other. Exact equal-month intervals are saved in `paired_comparisons.json`.

All three primary paired ECE10 differences have intervals crossing zero. For example, CLOSE − A0 ECE10 is **+0.188 percentage points**, interval **[−1.199, +1.393] pp**. The defensible finding is therefore **better archival CLOSE proper scores**, not a demonstrated calibration ordering.

**Dependence sensitivity:** a separately recorded, post-hoc calculation resampled **151 source tournament labels within calendar year**, not certified independent editions. CLOSE − A0 LL has interval **[−0.016807, +0.000055]**, crossing zero; its Brier interval remains negative, **[−0.007453, −0.000156]**. CLOSE − OPEN LL remains negative under this alternative, **[−0.014593, −0.006643]**. This does not replace the prespecified monthly analysis with a preferred interval: it shows that the LL superiority claim against A0 is not robust across these two dependence assumptions. Exact estimands, equal-label results and provenance are in `source_label_sensitivity.json`.

![Matched bookmaker diagnostic: reliability, proper-score contrasts, IID ECE reference and confidence exposure](../data/08_reporting/bookmaker_noise_research_20260927_200205/bookmaker_diagnostics.png)

### 2.3 How much observed calibration error could be sampling variation?

For each saved probability stream, 2,000 independent synthetic outcome vectors were drawn with `Y_i ~ Bernoulli(p_i)`. The generating probabilities are correct by construction, yet empirical ECE is positive:

| Stream | Observed ECE10 | Mean IID-reference ECE10 | Central 95% reference range |
|---|---:|---:|---:|
| A0 | 2.726% | 2.001% | [1.114%, 3.199%] |
| OPEN | 1.795% | 1.979% | [1.066%, 3.072%] |
| CLOSE | 2.914% | 1.997% | [1.074%, 3.063%] |

**This is an illustration, not a calibration test robust to dependence, not a universal noise floor, and not an amount to subtract from measured ECE.** The matches share teams, periods and competitions; the simulation deliberately does not recreate those dependencies. Being inside this range does not establish calibration or equivalence.

Binning matters. A0 ECE5/10/20 is **2.657/2.726/3.214%**; OPEN is **1.686/1.795/2.060%**; CLOSE is **2.437/2.914/2.914%**. Favorite-oriented ECE10, using each forecast's own favorite and excluding exact ties, is **1.129/1.534/1.939%** for A0/OPEN/CLOSE. That changes the apparent ordering again. Favorite-oriented and A-side calibration are different summaries, not interchangeable evidence.

Hard bin edges can also change ECE after side reversal when quotes lie exactly on an edge. The verification checks proper-score symmetry and records ECE reversal as a sensitivity rather than forcing a false invariant.

### 2.4 Are bookmakers avoiding extreme errors?

In this cohort they issue far fewer extreme **consensus** probabilities:

| Stream | Fixtures with favorite probability ≥91.79% | Actual losses with LL ≥2.5 | Expected such losses if saved probabilities were correct |
|---|---:|---:|---:|
| A0 | **98** | **9** | 5.333 |
| OPEN | **9** | **0** | 0.694 |
| CLOSE | **13** | **0** | 0.994 |

The threshold is `1 − exp(−2.5)`: a favorite's loss beyond that confidence yields LL at least 2.5. Zero bookmaker failures is weak evidence when only 9 or 13 fixtures are exposed. Under the independent reference, the probability of zero such losses is about **48.6% for OPEN** and **35.6% for CLOSE**. These are illustrative probabilities, not dependence-robust tail tests.

This does **not** mean A0 is simply more confident everywhere. Mean favorite confidence is **67.64% for A0, 68.27% for OPEN, 68.35% for CLOSE**. A0 differs especially in the extremes. A blanket temperature correction could alter many ordinary predictions to address relatively few extreme ones; it requires an out-of-time proper-score benefit, not just fewer threshold crossings.

### 2.5 Conditional results and individual bookmakers

Event categories expose heterogeneity that a single ECE obscures:

| Event category | N | A0 LL | OPEN LL | CLOSE LL | A0 / OPEN / CLOSE ECE10 |
|---|---:|---:|---:|---:|---:|
| Major | 1,118 | 0.596211 | 0.586800 | 0.586111 | 3.49 / 2.72 / 2.79% |
| Regional | 1,173 | 0.590107 | 0.606504 | 0.588691 | 2.88 / 3.04 / 4.88% |
| Other top-level | 243 | 0.564112 | 0.583807 | 0.563210 | 5.64 / 6.10 / 5.37% |
| International | 139 | 0.588498 | 0.528738 | 0.519437 | 12.69 / 6.25 / 7.49% |

These are descriptive, differently sized slices. International is a diagnostic priority, not proof of a missing regional feature or a precisely measured calibration defect. There are **no development-category matches** with archival odds here. Categories label events, not team nationality, historical affiliation, or independently adjudicated team strength.

Each bookmaker was also compared with A0 and consensus on that bookmaker's own OPEN/CLOSE intersection. Those intersections range from 747 to 2,600 fixtures; ranking their raw scores would confound difficulty and coverage. On the **same 669 fixtures shared by all six books at both horizons**:

| CLOSE forecast | LL | Brier | ECE10 |
|---|---:|---:|---:|
| A0 reference | 0.599675 | 0.205600 | 4.540% |
| Consensus | 0.587212 | 0.200514 | 3.566% |
| betclic | 0.586017 | 0.199975 | 3.837% |
| betfan | 0.585998 | 0.200020 | 4.003% |
| efortuna | 0.585800 | 0.199868 | 3.260% |
| lv_bet | 0.587348 | 0.200221 | 3.544% |
| sts | 0.589098 | 0.201439 | 4.132% |
| superbet | 0.592459 | 0.202761 | 3.701% |

This is not a best-book selection exercise. The small rank differences are not supported as stable superiority, and consensus is not automatically better than every constituent.

### 2.6 Quote disagreement, margin removal, and movement are not pure noise estimates

Across archival fixtures, the median within-fixture range between books is **2.62 pp at OPEN and 2.68 pp at CLOSE**. The 95th-percentile range is **9.31 pp and 6.32 pp**, respectively. Closing agreement is not uniformly tighter: its median range is slightly larger. Shared feeds, stale quotes, operator pricing and genuine information differences are possible contributors; their causal contributions were not identified.

Median observed overround is **8.14%** in archival quotes and **8.63%** in timestamped quotes. Proportional normalization is not a unique recovery of fair probabilities. As a fixed sensitivity, applying equal-margin additive adjustment to each two-sided book quote gives:

| Archival consensus conversion | OPEN LL / ECE10 | CLOSE LL / ECE10 |
|---|---:|---:|
| Proportional | 0.592155 / 1.795% | 0.581694 / 2.914% |
| Additive | 0.592684 / 2.153% | 0.581080 / 2.409% |

No additive probability was invalid or clipped. Eight archival quote pairs were underround; they remain recorded rather than silently removed. Additive and Shin adjustments are equivalent for two competitors under the relevant formulation, so they are not two independent checks. The Shin economic interpretation should not be casually assigned to underround data. No transform was selected on these outcomes. [14][15]

Published market studies also caution against a universal calibration assumption: favorite-longshot distortions occur in horse racing; selected football exchange markets can show the reverse pattern. Their settings differ from binary professional LoL. Those studies motivate checking our market, not importing a bias correction. [17][19]

### 2.7 Timestamped market panel: a separate, smaller question

The reused timestamped panel contains 514 OPEN, 343 48h, 379 24h and 409 CLOSE observations, with **275 fixtures common to all four horizons**. Its time span is 2026-05-29 to 2026-09-08, only **five active months**. Horizon rows overlap; adding their counts does not produce independent matches.

| Same 275 fixtures | LL | Brier | ECE10 |
|---|---:|---:|---:|
| A0 | **0.545250** | **0.185929** | 4.053% |
| Market OPEN | 0.570488 | 0.195259 | 6.171% |
| Market CLOSE | 0.548262 | 0.186217 | 5.075% |

CLOSE − OPEN LL is **−0.022226**, monthly interval **[−0.038489, +0.006544]**. CLOSE − A0 is **+0.003012**, interval **[−0.022609, +0.025915]**. These small-block comparisons do not establish either market or A0 superiority.

Horizon semantics were executed as checks, not assumed from labels:

- OPEN means **first observed quote**, not certified first market publication. At this horizon the median book count is one; only eight of the 514 fixtures have at least two books.
- 24h/48h use quotes at or before the specified cutoff with an allowed maximum age of 24 hours. Some are more than 23 hours old at that cutoff.
- CLOSE uses a quote strictly before the recorded start and no more than two hours earlier. It is not guaranteed to be the last executable market price.
- The daily feature-availability screen leaves only **57 OPEN**, **one 48h**, **two 24h**, and **409 CLOSE** observations in the full horizon cohorts. It still does not certify roster-announcement availability or real forecast issuance.

The median absolute OPEN→CLOSE probability movement on the common 275 fixtures is **3.67 pp**, with a 90th percentile of **12.98 pp**. Calling that “noise” would be unjustified. Information, stale-price replacement, changing bookmaker composition and pricing decisions can all contribute. In particular, most OPEN observations have one book while later horizons typically have several. This report does not identify a pure same-information, same-provider-set causal effect.

The same post-hoc source-label sensitivity has **42 labels** here. CLOSE − A0 LL still includes zero, **[−0.022842, +0.023159]**. CLOSE − OPEN becomes negative, **[−0.050390, −0.001903]**, unlike its monthly interval. Do not select the more favorable blocking scheme; this is sensitivity to assumed dependence, not new independent evidence.

## 3. What providers actually disclose

### 3.1 The closest answer to the user's training question: Bayes Esports

Bayes' provider-authored 2022 article describes a genuine LoL training experiment: divide history into approximately two-week patch slices, repeatedly train a map-winner model using increasing amounts of history, score the latest matches outside TRAIN, and repeat at different historical starting times. The reported best point is about **5,000 games / 21 patches**, described by the article as approximately **10.5 months**. Twelve recent patches provided too little data. [1]

The useful finding is the **relevance–variance tradeoff**, not the article's broad suggestion that one window will “always” be optimal. The article does not publish enough raw results, partition detail or uncertainty calculations to independently reproduce its significance claims; its convention that a Brier difference of 0.005 is significant is not a universal statistical threshold. Nor does the article reveal whether the operational model's parameters update monthly, after patches, or on another schedule. [1]

The target matters. Its discussion of changing red-side preference makes clear that side-sensitive map forecasts are involved. We generally do not know blue/red side or draft at a pre-match series origin. We cannot copy such inputs without changing the information contract. A map-window result also does not automatically transfer to a series likelihood, new-roster inference, or international tournaments. [1]

**What it supports:** a controlled test of the age and amount of historical evidence under patch change. **What it does not support:** replacing A0 with a fixed 21-patch window, calling Bayes Esports Bayesian, asserting monthly exponential-decay training, or claiming bookmaker probabilities are independently calibrated.

### 3.2 The wider pipeline: data → sports forecast → market/trading price

The following is a conceptual separation, not a claim that every provider implements the same architecture:

| Layer | Public evidence | What remains unknown |
|---|---|---|
| Official, structured observations | GRID/Riot partnership, GRID telemetry, GRID/Abios LoL relationship | Which exact feeds each checked book uses; point-in-time features in each predictor. [3][4][6] |
| Predictive model and history infrastructure | Bayes training-window article and AWS case study; Genius predictive systems | Full architectures, labels, hyperparameters, retraining and calibration schedules. [1][2][10] |
| Human domain interpretation | Abios LoL trader commentary about First Stand, Fearless Draft and regional context | Whether each item is a model input, discretionary override, or editorial explanation. [7] |
| Market and betting feedback | Kambi network-bet disclosure; Alpha Odds and Genius market/liability tools | Exact signal weights, which clients enable them, causal accuracy gains. [8][9][10] |
| Feed composition and operator pricing | OneFeed aggregation/conversion; customizable trading products | Independent information count behind multiple bookmaker quotes. [9][10][11] |
| Coherent related markets | Genius' consistent football/player pricing description | The joint generative distribution and whether it is calibrated. [13] |

Stats Perform illustrates another important distinction: granular data for modeling, trading and settlement is a **data product**, not a publicly audited probability model. Likewise, GRID's Pinnacle customer account establishes a data relationship in its stated esports context; it does not establish that GRID sets Pinnacle's prices or prove a specific Pinnacle LoL pipeline. [5][12]

Official low-latency telemetry matters enormously for in-play workflows. It does not give our pre-match model permission to use future kills, gold, objectives or draft. The appropriate lesson is better availability and identity control, not importing later information into an earlier forecast. [3][4][6]

### 3.3 Why prices can look disciplined without a magically noise-free model

Alpha Odds explicitly advertises operator-specific pricing informed by liabilities, customer behavior and market data. Genius describes predictive models and expert traders alongside liability-informed price changes. Kambi describes odds refinement from network betting activity. These corroborate that advertised modern pricing systems may use more than sports-only covariates, but they do not expose the contribution of each layer. [8][9][10]

A bettor's willingness to wager can be informative about an error; a liability limit can protect an operator without improving probability estimation. Price changes can do either or both. **[INFERENCE]** Better inputs, selective human review and external market feedback are plausible contributors to strong closing forecasts, but the local OPEN/CLOSE score difference does not identify them.

Do not assume all bookmakers mechanically balance stakes to equalize outcomes. Levitt's NFL contest evidence challenges that universal picture, while being far removed from current LoL customer-level staking. Exchange-versus-bookmaker studies also show that market structure matters; they do not make any market an oracle. [16][18]

Finally, revenue, increased trading margin, uptime, low feed latency and regulatory platform certification are not log loss, Brier, calibration curves, or independent prospective validation. None of the provider materials reviewed supplies an audited professional-LoL ECE/LL table comparable to this report. That is a bounded disclosure finding—not a claim that providers do not measure calibration internally.

## 4. What “handling noise” means in statistical terms

### 4.1 Four distinct problems

1. **Outcome randomness:** a correct 70% forecast still loses 30% of the time in its reference population. One observed outcome cannot reveal the latent probability of that individual match.
2. **Estimation uncertainty:** sparse players, rosters, patches or international pairings leave parameters weakly identified. More recent-only data can make this worse.
3. **Drift:** skill, rosters and the relationship between game statistics and future wins can change. Old examples can become less relevant.
4. **Measurement and pricing effects:** missing identities, stale quotes, unknown release times and margin allocation can distort the apparent comparison without reflecting a statistical training defect.

Smooth prices, low seed disagreement and small ECE do not separately identify these components. Neither does bookmaker agreement prove independent confirmation: multiple prices can contain shared information or common biases.

### 4.2 Recency mechanisms that must not be conflated

A training-row decay objective can be written as:

\[
w_i(T)=2^{-(T-t_i)/h},\qquad
L_T(\theta)=\frac{\sum_{i\in\mathrm{TRAIN}(T)} w_i(T)\,\ell_i(\theta)}{\sum_{i\in\mathrm{TRAIN}(T)} w_i(T)}.
\]

Here `h` is a half-life and `t_i` an explicitly chosen historical event-time field. Eligibility additionally requires label release before the fit cutoff. Every eligible row can remain in an **expanding** set with positive weight. A rolling window instead makes sufficiently old weights exactly zero. Likelihood recency weighting is established in sports modeling, but the literature does not supply A0's optimal `h`. [20][21]

An input-history age feature changes the representation of a current prediction. A rating filter changes the estimated current player state. A monthly refit changes when shared parameters are estimated. A calibrator changes the probability mapping. These are different interventions.

Glickman's original dynamic paired-comparison work explicitly grows skill uncertainty with elapsed inactivity and updates it with observations. This is not equivalent to lowering an old neural training row's loss. The paper is methodological background, not a claim that our particular Glicko-2 implementation has identical semantics. [22]

Koopman–Lit's football study is a useful counterexample to “expanding history means no adaptation”: it updates time-varying strengths and refits static parameters after each round **on an expanding sample**. Dynamic methods generally improved their forecast scores over static/semi-dynamic alternatives, but their stated betting simulation did not establish bookmaker-beating profitability. These are football goal models, not transferable LoL constants or a monthly-refit result. [21]

Retrospective smoothing is also not live filtering. TrueSkill Through Time uses later results to refine earlier skill estimates; using those smoothed values as past forecast inputs would leak future information. [23]

Adaptive windows such as ADWIN are another distinct choice: detect a change and remove older observations. That is not smooth decay and is not an established LoL improvement. Fading recent error in a monitoring statistic does not itself update a predictor. Evaluation research likewise supports choosing methods for the actual temporal regime, not a universal schedule. These distinctions argue for a small controlled test rather than introducing a drift detector, new rating engine and decay simultaneously. [24][25][26]

## 5. What our implementation already does—and what was actually tested

Archive paths below are relative to the root resolved through `data/research_root.txt`, `ENSEMBLE_RESEARCH_ROOT` or an explicit research-root argument. They are not repository-relative paths or production deployment locations.

### 5.1 Native annual A0

Direct source anchors:

- `a0-phase-walkforward-20260914/code/causal_nn.py:14-17,48-69`
- `a0-phase-walkforward-20260914/code/causal_calibrate.py:55-80`
- `nonlinear-history-20260912/code/src/models/nonlinear_history.py:23-28`
- `nonlinear-history-20260912/code/src/models/beta_training.py:26-55`
- `nonlinear-history-20260912/code/scripts/temporal_player_features.py:42-67`
- `nonlinear-history-20260912/code/scripts/prospective_sports_features.py:350-415`

For test year `Y`, the annual neural fit uses TRAIN before January 1 of `Y−1`, CAL during `Y−1` with labels released before `Y`, and forecasts during `Y`. Thus the 2026 mapping's TRAIN excludes 2025, which supplies calibration. This is more specific than “an expanding window”: it includes a preceding-year calibration separation.

The neural expert has a rating offset, a Beta terminal-score likelihood and auxiliary player-performance loss. Its training code takes the mean terminal negative log likelihood and adds the masked auxiliary loss with coefficient **0.1**. It uses three neural seeds and **12 epochs**. There is no sample-age argument or explicit time-decay factor in this objective. AdamW `weight_decay=0.01` regularizes parameters; it is not temporal example weighting.

At prediction time, ratings and histories are replayed as of the allowed cutoff. A player has at most **16 prior map tokens**, not 16 days. Tokens include `log1p(age in days)`, role, observed statistics and prior opponent rating. The existence of an age input does not prove a useful learned half-life, but it disproves the idea that all past history is represented without age information.

The final annual A0 is a calibrated multi-expert system, not just this one neural expert. Its native calibration and mixture are part of the forecast contract. A new weighted neural model cannot be declared an A0 reproduction merely because it consumes the same token bank.

### 5.2 A monthly refit was already tried

The archived `monthly-updates-20260914` experiment contains 33 first-of-month origins, expanding eligible TRAIN, fresh rather than warm-start fits, and three seeds. The frozen plan explicitly excludes new temporal sample decay. Both its monthly and January-only `annual_current` neural arms use the same monthly, past-only prior-12-month calibration policy based on saved earlier out-of-sample forecasts.

Saved full-mode monthly − annual-current LL is **+0.000545**, interval **[−0.001150, +0.002346]**. On its `all330_1h` cohort the difference is **+0.001081**, interval **[−0.018589, +0.006332]**. The stored decision has `frequency_point_support=false`, no independent holdout and no production promotion. These are inspected historical receipts, not new reruns.

The monthly candidate is one neural map expert subsequently converted/calibrated to series probabilities; the saved annual A0 comparator is the full multi-expert system. That distinction limits candidate-versus-A0 interpretation. The within-recipe monthly-versus-January comparison is more relevant to frequency, but still does not test neural sample-age weighting.

A **different frozen monthly replay** has independently checked LL **0.550316** on all 11,550 rows, versus annual A0 **0.551246**. That numerical result is real, but it is not an isolated refit-frequency contrast. Do not combine its headline score with the other experiment's control, calibration or recipe to assert a causal monthly advantage. Evidence: `data/08_reporting/a0_a1_audit_20260926_081429/monthly_replay_verification/independent_results_verification.json`.

### 5.3 Exponential decay was tried in different models

The current repository's `scripts/train_prematch_forecaster.py:208-210` explicitly computes `np.exp2(-age / training_half_life_days)`. The H365/H730 studies are **not hard rolling windows**.

On the same 3,103 2026 fixtures:

| Saved forecast | LL | Brier | ΔLL versus A0, monthly 95% interval |
|---|---:|---:|---:|
| Annual A0 | **0.536757** | **0.181573** | — |
| Fresh raw H730 | 0.546303 | 0.184445 | +0.009546 [ +0.004705, +0.012686 ] |
| Fresh raw H365 | 0.550097 | 0.186139 | +0.013341 [ +0.007032, +0.017712 ] |

Those candidates use quarterly origins and distinct raw-only logistic/LightGBM/neural/rating recipes. Another fresh architecture reused A0 history tokens with a 730-day half-life and six-month temperature calibration; its representation-fusion ΔLL was **+0.010848**, interval **[+0.007787, +0.013804]** on 11,550 rows. Neither study is a native A0 weighted-versus-unweighted ablation. Their failure must be preserved without overgeneralizing it.

Details and source anchors are retained in [the training audit](../data/08_reporting/bookmaker_noise_research_20260927_200205/training_schedule_sources.json). **As of 2026-09-27, [INFERENCE]:** no inspected result directly identified the combined effect of native A0 monthly refitting and target-row decay. The new, directly controlled experiment in §9 now answers this specific historical-development question; preserve the distinction from the older recipes.

## 6. The experiment that would answer the training question

**Design proposed on 2026-09-27, executed on 2026-09-29 in §9.** This tests a specific learning policy, not a promise to beat the market.

### 6.1 Bound the intervention

Use the native neural expert recipe and preserve its architecture, rating offset, scoreline likelihood, auxiliary target, three seeds, initialization policy and optional-context behavior. Do not substitute a fresh winner-only network. Keep unchanged annual full A0 as an external reference.

The smallest useful design is:

| Native-expert refit cadence | Uniform supervised objective | One fixed decay objective |
|---|---|---|
| January-only | A | B |
| Monthly | C | D |

The contrasts are B−A and D−C for weighting, C−A and D−B for cadence, and their interaction. The same four cells should be assessed at identical forecast origins and on identical fixtures. Do not compare a monthly single expert with a differently trained full ensemble and call the difference a cadence effect.

For the first hypothesis, **365 days is a defensible predeclared engineering choice, not an empirically established optimum**. It retains long history with decreasing influence and makes the new native-recipe test distinguishable from the failed fresh-model studies. It is not equivalent to Bayes' roughly 10.5-month inclusion window. Do not select between many half-lives using the final evaluated months.

The decay intervention should be named precisely: **neural supervised-row age weighting**, including a documented, mask-correct treatment of the auxiliary target and learned series-dependence parameter. Other expert objectives must not silently change. If later assembling a complete A0 candidate, carry all non-neural experts and the native mixture/calibration through an explicit, common contract and evaluate that assembled candidate separately. A neural-expert result alone is not an A0 replacement result.

### 6.2 Separate the clocks and data cutoffs

1. **Rating/history state:** advance on newly released evidence for every arm, preserving the same prior-state and simultaneous-opponent rules. Do not discard old player history merely because its target-row weight is small.
2. **Neural fit:** fresh fit at its declared annual/monthly origin, using only event and label-release dates before the fit cutoff. Use the same initialization and optimizer recipe across matched cells; warm starts are a separate policy.
3. **Calibration:** the same past-only monthly policy and window across all four cells, using genuinely earlier out-of-sample forecasts from each arm. Report the transfer assumption when calibrating a newly refitted model from earlier models' predictions. Do not reuse native annual calibration blindly after changing score distributions.
4. **Forecast issuance:** record the actual artifact/feature/forecast availability time, not only a nominal monthly origin. A forecast cannot be evaluated as available before fitting completed.
5. **Market benchmark:** use a separately defined observed quote horizon. OPEN remains primary for market research; CLOSE is a later-information comparator, not an earlier input to the sports model.

The annual control in this factorial must use the same eligibility/OOF-calibration policy as the monthly control. It is therefore an **annual-current experiment arm**, not automatically identical to native A0's preceding-year CAL exclusion. Reproduce the existing native output as a separate anchor so this distinction remains visible.

### 6.3 Avoid accidentally implementing ineffective decay

Native `fit_beta` traverses chronological batches of 128. If a batch contains nearly same-age rows and the weighted loss is divided by that batch's own sum of weights, a common age multiplier cancels. The code can then contain “sample weights” while substantially erasing the intended between-era weighting.

Use a **global TRAIN normalization**, e.g. `w / mean(w over TRAIN)`, and define the intended loss reduction before fitting. Preserve the masking denominator and relative auxiliary strength explicitly. The weighted objective and its gradients should agree with a small full-batch arithmetic oracle; splitting the same examples into batches must not redefine the intended objective. Adaptive optimizers and finite passes can still respond differently to gradient scaling, so loss normalization is necessary, not a guarantee of identical optimization dynamics.

Keep minibatch order, optimizer settings, regularization and stopping policy fixed across weighting cells. Report update counts and compute separately: monthly fresh fits naturally consume more total work than January-only fitting. Do not introduce shuffling, warm starts or new early-stopping selection only in the decay arm.

Record weighted effective sample size:

\[
N_{\mathrm{eff},w}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
\]

This diagnoses concentration of weights, **not** the number of independent matches. Report it globally and for sparse event/experience/roster groups. Aggressive forgetting can sacrifice scarce cross-region evidence while making recent domestic predictions look smoother.

### 6.4 Evaluation and stopping rule

Freeze the loss contract, half-life, origins, TRAIN/CAL release policy, forecast roster scenario and decision criteria before scoring. The already inspected 2024–2026 years can support a labeled historical development replay; they cannot be relabeled untouched TEST. Maintain a frozen future shadow period for independent confirmation.

Score both raw native-contract series probabilities and the corresponding calibrated series probabilities, applying the BO projection only where the contract requires it and never twice. Use primary LL, secondary Brier, coverage, discrimination, calibration sensitivity and tail exposure. Predeclare slices for time, BO, patch where available, experience/inactivity, roster continuity and event category; do not invent missing team affiliations. Use at least 5,000 paired monthly resamples, both match/equal-month estimands, and edition sensitivity only when a trustworthy edition mapping exists.

A candidate is not promoted because ECE falls or a CI crosses zero. It needs a predeclared practically useful proper-score improvement, no unacceptable coverage or important-stratum regression, causal/symmetry/cold-start checks, and independent future confirmation. If the result is inconclusive or only one exposed subgroup improves, retain native A0. No universal numeric gain or calibration threshold can be justified by the provider materials alone.

## 7. What we are actually missing, in priority order

| Priority | Missing or unresolved capability | Why this is the next question, not a claim of a proven cause |
|---|---|---|
| 1 | An equal-information, timestamped forecast/quote ledger with roster-announcement provenance | The current archival comparison cannot establish availability; most early timestamped quotes predate the daily feature scenario. Decay cannot fix unavailable inputs. |
| 2 | Native-recipe cadence × target-weighting contrast: now executed in §9 | H365 worsened both cadence arms; monthly refitting alone was inconclusive. Do not repeat the exact experiment as untested or substitute a new half-life selected from these results. |
| 3 | Separation of slow player ability from fast meta/patch response | **[INFERENCE]** Different timescales may be useful; one global decay discounts both. This is a later hypothesis, not permission for a combined architecture change now. |
| 4 | Persistent, out-of-time conditional error rather than a low ECE point | Small international and recent bins are noisy; corrections must survive later origins and proper-score controls. |
| 5 | Operational capability to issue the validated research forecast causally | Research replay is not a validated live A0 adapter. A correct offline objective does not supply artifact provenance or a real-time input path. |
| 6 | Independent prospective and tournament-edition evidence | A good historical match score does not establish executable EV, profitability, coherent futures or commercial readiness. |

Market feedback is a legitimate separate **hybrid** research direction, not a missing requirement of the odds-free sports model. Provider trading layers help explain why their product is different; copying those layers into A0 would change its identity. Likewise, human handling of announced rosters, drafts and format changes suggests availability-aware inputs, not team-name exceptions or corrections chosen from a few losses.

The main error to avoid is treating all unfavorable observations as one kind of “noise.” The current evidence points to a mix of outcome uncertainty, sparse calibration measurement, extreme-probability exposure, time/information mismatch and potentially stale shared parameters. A monthly job alone does not solve all of these.

## 8. Reproduction, verification, and unchanged state

New evidence lives in `data/08_reporting/bookmaker_noise_research_20260927_200205/`. The protocol was frozen at **2026-09-27T20:02:05.388964Z**, before the new diagnostic scoring. It pins seven input hashes, including the canonical table, corrected identity snapshot and quotes, timestamped panel and quotes, benchmark manifest and shared ECE-reference helper. The completion receipt also records the metric-module hashes.

Commands actually executed from the repository root:

```bash
.venv/bin/python data/08_reporting/bookmaker_noise_research_20260927_200205/diagnose.py --protocol data/08_reporting/bookmaker_noise_research_20260927_200205/protocol.json
.venv/bin/python data/08_reporting/bookmaker_noise_research_20260927_200205/verify.py --directory data/08_reporting/bookmaker_noise_research_20260927_200205
.venv/bin/python data/08_reporting/bookmaker_noise_research_20260927_200205/plot.py --directory data/08_reporting/bookmaker_noise_research_20260927_200205
.venv/bin/python data/08_reporting/bookmaker_noise_research_20260927_200205/sensitivity.py --directory data/08_reporting/bookmaker_noise_research_20260927_200205
```

The diagnostic supports `--research-root`; otherwise the archive root is resolved without embedding a personal path. These are saved one-off diagnostic scripts, not a competing benchmark/training system. Candidate adoption still belongs in the canonical benchmark.

**Independent oracle: PASS.** It checked **207 metric rows**, **486 paired contrasts**, **6,862 numerical comparisons**, and **44 exact Poisson-binomial ECE moment checks**. Maximum arithmetic difference was **2.67e−16**; maximum proper-score side-reversal difference **8.88e−16**. Seven pinned input hashes were preserved. Identity/orientation, quote-horizon selection and optimized-Python rejection checks passed. The rendered figure was visually inspected; an overlapping annotation was corrected and re-rendered.

The post-hoc source-label supplement separately verified **12 contrasts and 72 numerical values**, maximum difference **1.39e−17**, using the existing shared clustered-interval implementation and an independent arithmetic oracle. Its own frozen supplement protocol explicitly records that the primary results had already been inspected; the original protocol was not rewritten. **Citation integrity: PASS, 26/26 sources**, with no missing or unused bibliography entries. That checker validates citation structure, not source truth or full-text access.

The oracle independently computes losses, ranked AUC, bin membership, bootstrap arithmetic and Poisson-binomial reference moments. This verifies calculations under the stated assumptions; it does not validate unpublished raw quote semantics, bookmaker internals, independence of match outcomes, or actual historical roster announcement times.

In the original 2026-09-27 investigation, no neural forecasting fit, new deployable calibration model, live scraping, deployment, database migration, financial simulation or tournament evaluation ran. Its descriptive calibration regressions and IID simulations were diagnostics only. Section 9 records the subsequently authorized native refit, separately from those original diagnostics. Frozen native A0 checkpoints, archive sources and the canonical manifest remain unchanged.

### Key saved artifacts

- `protocol.json`, `completed.json`, `verification.json`: frozen scope and executed receipts.
- `metrics.json`, `paired_comparisons.json`, `evaluated_predictions.parquet`: same-row metrics and contrasts.
- `reliability.csv`, `iid_noise_reference.csv`, `exact_ece_reference_moments.json`, `tail_reference.json`: calibration and tail diagnostics.
- `quote_dispersion.csv`, `dispersion_summary.json`, `quote_timing.csv`, `market_movement.csv`, `movement_summary.json`, `devig_sensitivity.json`: market construction and sensitivity.
- `source_audit.json`, five `*_sources.json` ledgers, four `strong_*.txt` reviews and `strong_review_receipt.json`: source provenance and stronger-model review trail.
- `sensitivity_protocol.json`, `source_label_sensitivity.json`, `sensitivity_verification.json`: explicitly post-hoc alternate-block analysis.

## 9. 2026-09-29 — Executed native refit and multi-bookmaker audit

**Decision: keep full annual A0. The fixed H365 intervention failed; consensus has a lower T−1h point loss than A0, but not an established general advantage over A0 or every bookmaker.**

Evidence:

- [Frozen native protocol](../data/05_model_input/a0_native_refit/native_refit_20260929_122617/protocol.json), [completed results](../data/08_reporting/a0_native_refit/native_refit_20260929_122617/results.json), [executed commands](../data/08_reporting/a0_native_refit/native_refit_20260929_122617/execution_plan.json), [figure](../data/08_reporting/a0_native_refit/native_refit_20260929_122617/native_refit_results.png).
- [Deep bookmaker audit](../data/08_reporting/benchmark/bookmaker_consensus_deep_audit_20260929_100823/audit_summary.json), [disagreement figure](../data/08_reporting/benchmark/bookmaker_consensus_deep_audit_20260929_100823/consensus_audit_overview.png), [synchronized-horizon figure](../data/08_reporting/benchmark/bookmaker_consensus_deep_audit_20260929_100823/synchronized_open_diagnostics.png).

### 9.1 What was trained

The user selected the **native neural expert**, not a sports-plus-market head. A/B/C/D are not newly assembled full-A0 ensembles. The unchanged annual full A0 is the external reference.

The four arms use January-current versus monthly fresh fits, uniform versus globally normalized `exp2(-age/365)` supervised-row weights, three fixed seeds, 12 epochs, chronological batches of 128, unchanged architecture/rating offset, native stopped-series Beta likelihood, and mask-correct auxiliary loss with coefficient 0.1. The learned dependence parameter participates in the weighted objective. Rating state, history normalization, baseline offset and auxiliary scaling stay unweighted. No odds enter inputs or targets.

All arms use monthly trailing-12-month calibration from genuinely earlier OOS forecasts of their own arm. January raw forecasts/checkpoints are shared between cadences; calibrated January forecasts need not match because earlier OOS predictions differ. TRAIN event and release dates precede the fit origin; CAL event and release dates precede its origin. Nominal historical origins are separate from actual artifact-creation timestamps.

Executed **270 genuine fits**: 45 origins, January 2023–September 2026, two weightings and three seeds. The 2023 origins provide OOS calibration warmup. All four candidates cover the same **11,550** evaluation matches, including **4,642/3,805/3,103** in 2024/2025/2026. No cohort intersection was substituted.

### 9.2 Native-expert result

| Forecast | LL | Brier | Accuracy | LL ≥ 2.5 |
|---|---:|---:|---:|---:|
| Full annual A0, external reference | 0.551246 | 0.187103 | 71.13% | 33 |
| A: annual-current, uniform | 0.550811 | 0.186952 | 71.24% | 44 |
| B: annual-current, H365 | 0.558216 | 0.189851 | 70.57% | 47 |
| C: monthly, uniform | 0.551456 | 0.187067 | 71.17% | 53 |
| D: monthly, H365 | 0.559740 | 0.190377 | 70.23% | 55 |

The predeclared primary **D−A ΔLL is +0.008929**, monthly match-weighted 95% interval **[+0.004189, +0.013817]**. ΔBrier is **+0.003424 [+0.001587, +0.005237]**. The improvement criterion fails. Equal-month ΔLL is +0.005499 **[−0.001062, +0.011824]**: that estimand does not establish a difference. Tournament-name×year proxy sensitivity has 444 blocks and interval [+0.006170, +0.011708], not audited independent editions.

| Controlled contrast | ΔLL | Monthly match-weighted 95% interval |
|---|---:|---:|
| B−A: H365 within annual | +0.007405 | [+0.004828, +0.009927] |
| D−C: H365 within monthly | +0.008284 | [+0.005024, +0.011769] |
| C−A: cadence without decay | +0.000645 | [−0.001127, +0.002495] |
| D−B: cadence with H365 | +0.001525 | [−0.002866, +0.006557] |
| D−C−B+A: interaction | +0.000880 | [−0.002119, +0.004479] |

Both isolated H365 contrasts also have positive equal-month intervals. Cadence alone remains inconclusive, not equivalent. All intervals use 5,000 paired resamples, seed 20260929. The standard canonical runner separately uses its established bootstrap seed; its headline scores agree.

A's small point gain over full A0 is **−0.000434 [−0.001874, +0.000762]**, not established superiority. It also has 44 rather than 33 deep losses. Tail counts are not a standalone selection criterion: `tail_reference.csv` records forecast-implied expected counts and explicitly IID Poisson-binomial references. For example, D has 55 deep losses versus 44.35 expected under independent calibrated forecasts; this does not prove a causal failure mechanism.

The raw full-mode LLs are A/B/C/D **0.550796/0.559439/0.551249/0.560532**. H365's deficit is not created solely by monthly calibration. All three optional-context modes, raw and calibrated, are retained in `optional_mode_metrics.csv`; no best mode was selected after scoring. `profiles.csv` covers year, BO, competition family/tier/scope, experience, coplay, new organizations, unknown players, inactivity and W20 availability. Patch is unavailable, not inferred.

At September 2026, TRAIN has 40,562 series but H365 weighted ESS is 12,047. This measures weight concentration, not independent observations. Sparse-support records are preserved; they do not establish why a particular match was wrong.

### 9.3 How good is the bookmaker consensus?

The frozen definition is the equal mean of proportionally de-vigged probabilities from **at least two distinct bookmakers at one cutoff**. A single bookmaker never substitutes for consensus. The primary timestamped cohort is **1,668 T−1h matches**, nine monthly blocks, passing the daily history-availability proxy. It is distinct from the 2,673 archival OPEN cohort.

| Forecast, same 1,668 matches | LL | Brier | Accuracy | LL ≥ 2.5 |
|---|---:|---:|---:|---:|
| Full A0 | 0.576033 | 0.196853 | 68.88% | 5 |
| Consensus ≥2 | 0.568604 | 0.192872 | 70.56% | 2 |
| A: annual uniform expert | 0.577065 | 0.197462 | 68.53% | 3 |
| B: annual H365 expert | 0.584713 | 0.200717 | 68.23% | 2 |
| C: monthly uniform expert | 0.580774 | 0.198425 | 68.82% | 7 |
| D: monthly H365 expert | 0.592868 | 0.202910 | 67.75% | 11 |

Consensus−A0 ΔLL is **−0.007430 [−0.020487, +0.006968]** and ΔBrier −0.003981 [−0.009610, +0.001863]. Equal-month and tournament-proxy intervals also include zero. Lower point error and ECE10 (2.11% versus 3.81%) are not general superiority or calibration certificates.

A0–consensus absolute disagreement averages **8.18 percentage points**, median **6.19**, P90 **17.05**. Gaps ≥5/10/15/20 pp occur in **979/467/228/123** matches: **58.69%/28.00%/13.67%/7.37%**. They choose opposite favorites in **202/1,668 (12.11%)**; consensus is right in 115, A0 in 87. A0 nevertheless has lower realized loss in 917 matches versus 751: error magnitude matters.

BO5 has the largest point format deficit: A0−consensus **+0.036354** on 262 matches, interval [−0.005077, +0.060981]; equal-month weighting reverses the point sign. NACL and EMEA Masters have point deficits +0.083325/+0.082401 on 72/75 rows and contribute 98.3% of the **net** gap. Those post-hoc small strata are not 98.3% of all errors or proven causes; positive and negative contributions cancel.

D moves further from the market: mean absolute gap **9.58 pp**, opposite favorites **263/1,668 (15.77%)**. This native refit is not a successful market-alignment intervention.

### 9.4 Does consensus beat the individual books?

Exclude the focal book, then require **at least two other books**. Each row uses identical fixtures for that pair, but different rows have different coverage and remaining members; this is not a universal brand ranking.

| Excluded bookmaker | N | Focal LL | Remaining-consensus LL | ΔLL consensus−focal, monthly 95% interval |
|---|---:|---:|---:|---:|
| Fortuna.pl | 227 | 0.571361 | 0.569329 | −0.002032 [−0.002334, −0.001979] |
| Superbet.pl | 409 | 0.602563 | 0.602719 | +0.000156 [−0.001729, +0.000327] |
| Pinnacle | 1159 | 0.563119 | 0.567897 | +0.004778 [−0.002330, +0.008158] |
| bet365 | 486 | 0.567497 | 0.559200 | −0.008297 [−0.015543, +0.004286] |
| Unibet | 1106 | 0.572308 | 0.568475 | −0.003833 [−0.006167, +0.000169] |
| bwin | 676 | 0.557559 | 0.558718 | +0.001159 [−0.001414, +0.004126] |
| Betway | 1058 | 0.572682 | 0.562210 | −0.010472 [−0.018711, −0.001319] |
| 1xBet | 482 | 0.577173 | 0.575283 | −0.001891 [−0.005363, +0.002297] |
| Cloudbet | 444 | 0.580894 | 0.584248 | +0.003354 [−0.003174, +0.005181] |

Consensus has lower point LL in **5/9** comparisons, not all nine. Fortuna covers only **two months**; its tournament-proxy interval crosses zero. Pinnacle's sign changes under equal-month weighting. Most intervals include zero; all are nominal, without multiple-comparison adjustment. Very high book correlations do not prove independent information or common ownership.

### 9.5 Quote timing, coverage and data integrity

- The earliest synchronized `FIRST_2PLUS` replay has 1,706 fixtures, each with two books; median lead 41.97 hours. This is **first observed synchronized history, not certified OPEN**.
- Only **407** pass the daily availability proxy: consensus LL **0.589489**, A0 **0.578660**, consensus−A0 **+0.010829 [−0.016445, +0.038211]**. The four experts score 0.574010/0.574752/0.573650/0.573357; do not select D on this exposed small subset after its primary failure. T−24h has only **20** passing matches.
- On 1,615 identical fixtures with the original two books still present at T−1h, their LL improves **0.584114 → 0.572028**, difference **−0.012086 [−0.016210, −0.005225]**. Adding other books then gives **0.570714**. In this comparison later prices matter more than increasing provider count.
- The source has 3,302 fixtures and 29,718 logical fixture–book pairs. **17,826** have no history: 12,059 HTTP404 and 5,767 successful-empty. Missingness is not a bad probability and not a zero loss.
- All 2,428 accepted identities/orientations passed; 64 many-to-one fixture collisions and one ambiguous identity were excluded. Replay examined 2,082,721 winner-market updates, with 683 invalid prices and four same-time conflicts. Normal inactive updates are state changes, not invalid prices; prices are not carried through deactivation.
- Thirteen extreme raw overround states appear in neither primary T−1h nor `FIRST_2PLUS`. Large checked LIT/EWC disagreements were not side-flip errors; bookmaker favorites can lose.
- Requiring every original quote to be ≤24h old gives 1,286 primary fixtures. Dropping only stale members and rebuilding a ≥2 consensus gives 1,619. Delaying the earliest snapshot until fresh quotes appear is a third, distinct policy.
- Actual start is absent for **539/1,668** primary fixtures; those rely on retrospective scheduled time. Where both exist, the earlier time is used. On the 1,129 with actual start recorded, consensus−A0 ΔLL is **−0.003937 [−0.023593, +0.016181]**. This is coverage sensitivity, not proof of the direction of timestamp bias.
- Covered 2026 fixtures are harder for A0: LL **0.576033**, versus **0.491103** on the other 1,435 canonical matches. Market-cohort findings cannot be extrapolated to all 3,103 targets.

### 9.6 Verification and what remains

The completed run independently checked all 270 checkpoint and 90 raw-forecast artifacts across the uniform and H365 audits, 135 paired initialization hashes and exact TRAIN membership, and all 132 calibrator artifacts with 548,316 CAL membership appearances. All 396 mode fits had finite interior solutions; maximum selected projected gradient was **7.75e−9**.

Recalculation reproduced all **138,600** calibrated probabilities exactly; maximum probability-complement error **8.89e−16**. Restored early-2024 and late-2026 BO1/3/5 checkpoints passed all optional modes, cold-start neutrality and nonmutation. The late-2026 saved-inference maximum error was **2.39e−7**. **312** independent scalar/interval comparisons differed by at most **1.12e−16**. All four canonical benchmark runs completed, as did aggregate and verified resume. **49 targeted tests passed**, with one existing Starlette/httpx deprecation warning; no application database was used.

The bookmaker audit independently checked 13,870 arithmetic values, block-bootstrap arithmetic, all 3,412 early quotes / 6,824 sides against 2,927 raw files, and 32 quality-artifact hashes. Evidence receipts distinguish these direct checks from read-only review of reported numbers.

The earlier unscored run `native_refit_20260929_103444` was preserved after correcting a stale flat checkpoint-export path following atomic bundle publication. Training/scoring rules were not changed; a new immutable run was executed rather than modifying old receipts.

**Completed:** the requested data audit and native-expert factorial. **Not established:** equal-information issued forecasts, roster-announcement availability, prospective superiority, executable EV/CLV/Kelly, tournament quality or production readiness. The model still uses actual first-map rosters as an explicit scenario. Neither a new market-aware head nor another half-life search was run. Keep full A0 and preserve this negative H365 result; the timestamped decision/roster ledger and independent future evaluation remain prerequisites, not completed work.

## 10. 2026-09-29 — Executed single-split data focus and history masking

**Keep broad training data. None of the four interventions lowered full-context, assembled-A0 market log loss relative to the new uniform control.** High-tier filtering was the weakest arm. The combined market-weighting/dropout hypothesis failed its prespecified success criterion; this does not establish that every possible denoising method must fail.

Run: `a0_data_focus_20260929_192443`. [Governing protocol v2](../data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/protocol_v2.json), [complete evaluation](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/evaluation.json), [metrics](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/metrics.csv), [profiles](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/profiles.csv), [independent verification](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/independent_verification.json).

### 10.1 Fixed design: no walk-forward fitting

- **TRAIN:** 36,205 rows with event and result-release dates before **2025-07-01**. **CAL:** 1,328 July–December 2025 rows, with labels released before **2026-01-01**. **Validation:** all **3,103** canonical 2026 rows, **January 8–September 8**.
- Reserving 2025 H2 leaves a separate calibration population; 2025 Q4 alone contains only 151 rows. Consequently, this deliberately does not use every 2025 label for gradient fitting. No 2026 target enters TRAIN or CAL.
- Frozen weights/calibrators, no monthly refits, no early stopping or validation-selected variants. Causal player/rating/history inputs still advance before each target and may use earlier released 2026 results; that is not retraining on a future target.
- Three paired seeds **20260910/20260911/20260912**, **12 epochs** each, chronological target batches. The native terminal beta scoreline likelihood and **0.1 × masked SmoothL1/Huber auxiliary loss** remain unchanged.
- All arms share freshly fitted full-TRAIN baseline/input normalizers, raw-rating expert, and three warm/full-MLP expert pairs. There are **15 neural fits**, plus those shared experts. Interventions apply to the neural supervised component, not the whole history or every A0 expert.
- Native per-mode beta calibration, expert slopes and regularized three-group mixtures use the same fixed CAL. **CAL is reused for calibration and mixture fitting; there is no quarterly CAL crossfit.** This is a new single-split research assembly, not an exact reproduction of the frozen annual recipe. Its uniform control, not external annual A0, isolates the interventions.
- Primary endpoint was **weighting + dropout minus control**, assembled full-context A0 LL on the **1,668** strict-prior-day T−1h fixtures with at least two books. Required: monthly paired LL interval entirely below zero, nonpositive Brier difference, complete coverage; an all-purpose claim also required nonregression on all 3,103 rows.
- All probabilities are scored **without clipping**, consistently for headline metrics, profiles, deltas and tails. Some new probabilities fall below 0.001; the legacy benchmark helper's default 0.001 clipping must not be silently substituted.

| Arm | Neural supervised rows | Change |
|---|---:|---|
| Control | 36,205 | Uniform weights, native context masking only |
| High-tier only | 11,302 | Major + international targets; auxiliary target scaler fitted on those targets |
| Market weighted | 36,205 | TRAIN-only archival-market tier-share weighting |
| Token dropout | 36,205 | Independently hide 20% of valid historical map tokens during training |
| Weighted dropout | 36,205 | Both preceding interventions; prespecified primary candidate |

High-tier filtering keeps **all rating/history evidence and shared experts**. At the fixed 12 epochs it has **1,068 optimizer steps per seed**, versus **3,396** for the full-target arms. Its result therefore includes reduced supervision and update count; it does not isolate label quality at equal compute.

Market weights use only **1,931 TRAIN IDs** with archival OPEN coverage, not odds values, current odds, 2026 coverage or 2026 outcomes. Divide the covered tier share by the TRAIN tier share, clip to [0.5, 3], then normalize globally to mean one:

| Competition tier | TRAIN N | Covered TRAIN N | Final weight |
|---|---:|---:|---:|
| Development | 5,298 | 0 | 0.460098 |
| International | 1,699 | 75 | 0.761613 |
| Major | 9,603 | 792 | 1.422935 |
| Minor top-level | 7,919 | 185 | 0.460098 |
| Regional | 11,686 | 879 | 1.297746 |

The final lower weight is below 0.5 because clipping precedes normalization. Weight ESS is **30,549**, not an independent-match count. Both native main and correctly masked auxiliary losses receive these weights. Historical archival membership is not certified as-of availability of the weighting recipe.

Native A0 already masks organization/W20 contexts; it did not randomly remove historical map tokens. The new mask uses a separate RNG, intersects the existing validity mask, and zeros the hidden normalized token, including its missingness indicators. It never alters target statistics, observed-target masks, player identity/core features or inference inputs. Entirely empty histories retain the native empty token. Across the three dropout seeds, observed hide rates were **20.0153%, 20.0062%, 20.0046%**, over **62,726,844 valid-token presentations per seed**. The two dropout arms share masks; existing context-gate RNG streams remain paired with control.

### 10.2 Data integrity is not the same as market relevance

[Prescore quality audit](../data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/data_quality.json), [population composition](../data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/population_composition.csv), [recent-stat detail](../data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/recent_train_stat_detail.csv).

The 40,636-row bank had no duplicate match IDs, self-matches, wrong-size/duplicated/opposing-overlap rosters, nonfinite raw inputs or observed history/auxiliary values, or forward feature cutoffs. Ten null feature cutoffs are the source-free initial cold starts: all players unknown, no history, no context gates. Other unexplained null cutoffs are now rejected. These checks do **not** independently verify every original result or statistic.

Three material differences were observed before scoring:

1. The historical market-covered TRAIN sample contains **zero development matches**, but the 2026 T−1h cohort contains **294/1,668 (17.6%)**. “Market training” through this historical coverage proxy is not the current market population.
2. Full TRAIN is **68.15% BO1**; the market validation cohort is **63.91% BO3**. Native A0 models scorelines/series, but competition-tier weighting does not explicitly correct this format/era shift.
3. High tier does not imply complete features. For major-league targets in 2024–June 2025, CSD@15 and XPD@15 history-token presence is **64.97%**; auxiliary target presence is **64.55%**. Kills/deaths/assists and several rate statistics have full presence in that slice. Existing presence masks distinguish absent statistics from measured zeros. This is missingness evidence, not proof that those labels are wrong.

The failed experiments do not establish the cause of the remaining market gap. In particular, they do not justify removing lower-tier rows as “bad data.”

### 10.3 Full-context assembled-model results

Every row below uses the same population within its column. The consensus has no value on the full cohort because market coverage is incomplete.

| Model | All 2026 LL | All Brier | All LL ≥ 2.5 | Market T−1h LL | Market Brier | Market LL ≥ 2.5 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen annual A0, external reference | 0.536757 | 0.181573 | 10 | 0.576033 | 0.196853 | 5 |
| New uniform control | **0.535449** | **0.181034** | 12 | **0.575423** | 0.196552 | 6 |
| High-tier only | 0.539638 | 0.182496 | 15 | 0.578409 | 0.197452 | 7 |
| Market weighted | 0.536950 | 0.181570 | 11 | 0.576202 | 0.196771 | 4 |
| Token dropout | 0.535847 | 0.181154 | 12 | 0.575441 | 0.196503 | 5 |
| Weighted dropout | 0.537073 | 0.181579 | 12 | 0.576226 | 0.196735 | 5 |
| ≥2-book consensus | — | — | — | 0.568604 | 0.192872 | 2 |

Lower tails alone would favor weighting over control on the market subset, but LL/Brier worsen. That is not an improvement criterion.

![Assembled full-context A0 intervention effects](../data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/comparison_intervals.png)

**Prespecified primary:** market ΔLL **+0.000803**, monthly 95% interval **[−0.000850, +0.002290]**; ΔBrier **+0.000183 [−0.000492, +0.000778]**. Equal-month ΔLL **+0.001011 [−0.001404, +0.003218]**. Tournament-name proxy sensitivity: **105 groups**, LL interval **[−0.000729, +0.002403]**. These are not independently audited editions.

On all 3,103 rows, the primary candidate also regresses: ΔLL **+0.001624 [+0.000315, +0.002639]**, ΔBrier **+0.000545**. Neither the primary signal nor the broader nonregression guard passes.

| Secondary contrast vs new control | All-2026 ΔLL, monthly 95% interval | Market ΔLL, monthly 95% interval |
|---|---:|---:|
| High-tier only | +0.004190 [+0.000595, +0.007393] | +0.002986 [+0.000179, +0.006496] |
| Market weighted | +0.001501 [+0.000224, +0.002561] | +0.000780 [−0.000702, +0.002038] |
| Token dropout | +0.000398 [+0.000036, +0.000850] | +0.000018 [−0.000830, +0.000933] |

Monthly intervals use **5,000 paired resamples**, seed **20260929**, **nine calendar-month blocks**. Secondary intervals are nominal, not multiplicity-adjusted. High-tier equal-month intervals cross zero on both populations; do not claim every blocking/weighting choice establishes deterioration. The weighting×dropout interaction is explicitly **exploratory**, not a retroactively declared primary endpoint.

The new control's advantage over frozen annual A0 is small and inconclusive under match weighting: all-2026 **−0.001308 [−0.003149, +0.000122]**, market **−0.000611 [−0.003284, +0.001346]**. Control−consensus is **+0.006819 [−0.009429, +0.021275]**. Lower consensus point loss does not establish statistical superiority, equivalence or executable profitability.

### 10.4 Component, population and missing-context diagnostics

The calibrated neural component's all-2026 LL is **0.535863/0.549528/0.537868/0.536150/0.537880** in control/high-tier/weighted/dropout/combined order. Market LL is **0.576365/0.588104/0.577584/0.576260/0.577530**. Shared experts cushion the high-tier damage. Dropout has a tiny neural-only market point gain, but no assembled full-context LL gain; do not report the component as the whole model.

High-tier filtering also worsens the **major-only** point result: ΔLL **+0.004272** on 704 major validation matches, **+0.001949** on the 591 market-covered majors. Its market-development ΔLL is **+0.009732** on 294 matches. These are descriptive profiles, not subgroup-specific adoption tests.

| Assembled variant | All LL, no W20 | All LL, no organization | Market LL, no W20 | Market LL, no organization |
|---|---:|---:|---:|---:|
| Control | 0.537111 | 0.537622 | 0.576462 | 0.577388 |
| High-tier only | 0.541278 | 0.541451 | 0.578807 | 0.579272 |
| Market weighted | 0.538753 | 0.539087 | 0.577384 | 0.578061 |
| Token dropout | 0.537597 | 0.538002 | 0.576547 | 0.577293 |
| Weighted dropout | 0.538980 | 0.539292 | 0.577473 | 0.578092 |

`profiles.csv` also preserves all monthly, BO, tier, unknown-player, low-experience and new-organization slices. `metrics.csv` includes raw neural probabilities, all three context modes and the **1,435** matches outside T−1h market coverage.

On the distinct **407** strict-prior-day `FIRST_2PLUS` fixtures, assembled LL is **0.575976/0.580727/0.577916/0.577178/0.579018**, versus frozen A0 **0.578660** and consensus **0.589489**. Control−consensus **−0.013512 [−0.041848, +0.014894]** is inconclusive. This earliest observed synchronized cohort is neither the T−1h primary population nor certified market OPEN.

### 10.5 Execution, verification and retained evidence

Reusable implementation: `src/models/a0_data_focus.py`, `src/analysis/a0_data_focus.py`, `scripts/train_a0_data_focus.py`. Scoring delegates to the existing benchmark metric, identity and paired-bootstrap primitives; it does not introduce another loss definition.

Executed from the repository root:

```bash
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/train_a0_data_focus.py --protocol data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/protocol_v2.json
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/train_a0_data_focus.py --protocol data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/protocol_v2.json --evaluate-only
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q betting_app/tests/test_a0_data_focus.py betting_app/tests/test_a0_native_refit.py betting_app/tests/test_a0_target_objectives.py betting_app/tests/test_research_benchmark_suite.py betting_app/tests/test_model_benchmark.py
```

These completed artifacts are immutable: a new reproduction needs a new run ID/protocol, not a rerun that overwrites this directory. `--smoke` exercises reduced fitting and saved inference in a separate timestamped smoke run. Models are under `data/06_models/a0_data_focus/<run>/`; probabilities/logits under `data/07_model_output/a0_data_focus/<run>/`; reports under `data/08_reporting/a0_data_focus/<run>/`.

- **55 tests passed**, with one existing Starlette/httpx deprecation warning; no application database. Regression tests cover missing-cutoff cold starts, padding/NaN-safe masking, target-population weights, numeric market identities/orientation, extreme unclipped loss and completed-report immutability.
- All 15 neural checkpoints have 12 epochs and matched initial states within seed. Saved BO1/3/5 inference, side symmetry, repeatability, empty histories, neutral cold starts and nonmutation passed. Maximum checked symmetry error **9.85e−9**.
- A full saved-model replay checked **139,635 probabilities** across all five arms, three modes and three probability types. Maximum probability error **1.45e−8**; shared-expert logit error **2.39e−7**. Neural states were unchanged by replay.
- Independent scalar arithmetic checked **194 metric rows**, **276 profile rows** and **706 paired intervals**: maximum errors **1.67e−16**, **2.23e−16**, **2.78e−17** respectively. **72** recorded artifact/source hashes, the protocol hash and all training-input hashes matched.
- An actual two-fit future-perturbation check changed unused future labels, raw/history inputs and auxiliary targets/masks. TRAIN-only baseline/normalization and the one-epoch dropout neural fit stayed exactly identical in weights, losses and RNG traces. This is a neural/input-pipeline check, not an additional full-ensemble retraining.
- Training emitted a PyTorch warning when wrapping a non-writable NumPy weight array. The native loss only reads the weights. Two real weighted fits with read-only versus owned writable arrays gave identical trained state/losses and unchanged input bytes; the warning was not suppressed. Frozen training code was not rewritten after fitting.
- Original `protocol.json` and failed unscored smoke evidence were retained. Before full training, v2 corrected the stated auxiliary objective from MSE to native SmoothL1, pinned the quality audit, and tightened null-cutoff validation; no validation-score-driven objective change occurred. The evaluator initially rejected equal numeric identities such as integer `1` versus float `1.0`; a failing regression test and typed comparison corrected this without changing predictions or cohort membership.

**Decision:** retain canonical annual A0 as the research reference and the new uniform arm as this experiment's control. Do not adopt hard-tier filtering, these historical market-tier weights or this 20% token-dropout recipe. Retain the negative result. The measured statistic missingness and format/coverage shift justify examining causal source quality and representativeness next; they do not establish a successful repair, and no additional variant was fitted after seeing 2026 scores.

All 2026 observations remain repeatedly exposed **development data**. Actual rosters are a scenario, daily cutoffs do not certify announcement/model-issuance times, and 539 primary market fixtures still lack actual start times. No deployment, market-aware head, prospective evaluation, financial simulation or tournament validation ran. The canonical manifest, frozen source/models and operational application remain unchanged.

## Bibliography and access boundaries

All URLs were inspected in this research pass by the main researcher or the named evidence collectors. “Full” describes accessed text, not independent reproduction. Exact source-level limitations are in `source_audit.json`.

| ID | Source | Date | Access / relevance |
|---|---|---|---|
| [1] | Paprzycki / Bayes Esports, [How much League of Legends data is enough data?](https://igamingbusiness.com/esports/how-much-league-of-legends-data-is-enough-data/) | 2022-06-30 | Full provider-authored LoL training-window article; incomplete experimental disclosure. |
| [2] | AWS / Bayes Esports, [Case study](https://aws.amazon.com/solutions/case-studies/bayes-esports-case-study/) | Undated | Full vendor infrastructure case study, not calibration evidence. |
| [3] | GRID / Riot Games, [Global esports data partnership](https://www.grid.gg/resources/newsroom/riot-games-and-grid-announce-exclusive-global-esports-data-partnership/) | 2023-11-30 | Full official announcement. |
| [4] | GRID, [Live Data](https://www.grid.gg/products/live-data/) | Undated; accessed 2026-09-27 | Full product page; telemetry claims, not forecast validation. |
| [5] | GRID / Pinnacle, [Pinnacle's data-driven path into esports](https://www.grid.gg/resources/newsroom/pinnacles-data-driven-path-from-traditional-sports-into-esports/) | 2023-07-12 | Full customer disclosure; its game context must not be generalized to LoL. |
| [6] | GRID / Abios, [Official esports data partnership extension](https://www.grid.gg/resources/newsroom/grid-and-kambi-groups-abios-extend-official-esports-data-partnership/) | 2024-06-11 | Full announcement, including LoL data. |
| [7] | Abios, [LoL First Stand trading](https://abiosgaming.com/press/lol-first-stand-trading/) | 2025-02-25 | Full trader commentary, not a disclosed feature specification. |
| [8] | Kambi, [Odds Feed+](https://www.kambi.com/what-we-do/odds-feed/) | Undated; accessed 2026-09-27 | Full general Kambi product page; not proof of an Abios-LoL signal. |
| [9] | Sportradar, [Alpha Odds](https://sportradar.com/betting-gaming/trading-risk-management/insight-tech-services/alpha-odds/) | Undated; accessed 2026-09-27 | Full operator-specific pricing disclosure. |
| [10] | Genius Sports, [Genius Trading Services](https://www.geniussports.com/bet/genius-trading-services/) | Undated; accessed 2026-09-27 | Full predictive/trader/liability product description. |
| [11] | Sportradar, [OneFeed integration documentation](https://docs.sportradar.com/mts/onefeed/introduction-and-integration-specifics.md) | Living documentation | Full technical feed-aggregation documentation. |
| [12] | Stats Perform, [Bet Trading Data](https://www.statsperform.com/products/bet-trading-data/) | Undated; accessed 2026-09-27 | Full data-product description. |
| [13] | McKiernan / Genius Sports, [Football pricing and player betting](https://www.geniussports.com/content-hub/football-pricing-player-betting/) | 2026-08-20 | Full provider article on related-market consistency. |
| [14] | Clarke, Kovalchik & Ingram, [Adjusting Bookmaker's Odds to Allow for Overround](https://www.sciencepublishinggroup.com/article/10.11648/j.ajss.20170506.12) | 2017 | Publisher abstract/metadata only; de-vig methods. |
| [15] | Shin, [Prices of State Contingent Claims with Insider Traders, and the Favourite-Longshot Bias](https://ideas.repec.org/a/ecj/econjl/v102y1992i411p426-35.html) | 1992 | Abstract/metadata only; theoretical mechanism. |
| [16] | Levitt, [Why Are Gambling Markets Organised So Differently from Financial Markets?](https://www.nber.org/papers/w9422) | 2003 working paper; 2004 article | Record and full working paper; NFL contest setting. |
| [17] | Snowberg & Wolfers, [Explaining the Favorite-Longshot Bias](https://www.nber.org/papers/w15923) | 2010 | Record and full author-hosted paper; horse racing. |
| [18] | Franck, Verbeek & Nüesch, [Prediction accuracy of different market structures](https://www.unifr.ch/tim/en/assets/public/uploads/Publication%20list/2010/IJoF_Franck_Verbeek_Nuesch.pdf) | 2010 | Full paper; matched football book/exchange forecasts. |
| [19] | Angelini, De Angelis & Singleton, [Informational efficiency and behaviour within in-play prediction markets](https://www.carlsingletoneconomics.com/uploads/4/2/3/0/42306545/information_efficiency_angelini_de_angelis_singleton.pdf) | Manuscript May 2021 | Full paper; EPL exchange setting and selected biases. |
| [20] | Dixon & Coles, [Modelling Association Football Scores and Inefficiencies in the Football Betting Market](https://research.lancaster-university.uk/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-f/) | 1997 | University abstract; original scan inaccessible as text; mechanism corroborated by [21]. |
| [21] | Koopman & Lit, [Forecasting football match results using score-driven time series models](https://research.vu.nl/ws/portalfiles/portal/151597369/Forecasting_football_match_results_in_national_league_competitions_using_scoredriven_time_series_models.pdf) | 2019 | Full version of record; dynamic state and expanding refits. |
| [22] | Glickman, [Parameter Estimation in Large Dynamic Paired Comparison Experiments](https://www.glicko.net/research/glicko.pdf) | 1999 | Full original paper; evolving skill and inactivity uncertainty. |
| [23] | Dangauthier et al., [TrueSkill Through Time](https://proceedings.neurips.cc/paper_files/paper/2007/file/9f53d83ec0691550f7d2507d57f4f5a2-Paper.pdf) | 2007 | Full proceedings paper; smoothing is future-informed. |
| [24] | Bifet & Gavaldà, [Learning from Time-Changing Data with Adaptive Windowing](https://researchportal.ip-paris.fr/en/publications/learning-from-time-changing-data-with-adaptive-windowing/) | 2007 | Abstract plus coauthor's 2009 thesis ADWIN chapter; not publisher full text. |
| [25] | Gama, Sebastião & Rodrigues, [On evaluating stream learning algorithms](https://repositorio.inesctec.pt/bitstreams/435f0677-8283-40ed-9e42-503c8f38a1d6/download) | 2013 | Full paper; prequential/fading performance estimates. |
| [26] | Cerqueira, Torgo & Mozetič, [Evaluating time series forecasting models](https://kt.ijs.si/igor_mozetic/papers/CerTorMoz-TimeSer-MACH-20.pdf) | 2020 | Full paper; regime-dependent performance-estimation methods. |
