# Reliable League of Legends tournament simulation — deep research

**Research date:** 2026-10-06. **Research run:** `tournament_simulation_research_20261006_01`. The original research phase comprised five parallel external-evidence tracks, one read-only repository audit, 31 primary-source records and six independently executed mathematical check families; it performed no fitting, tournament replay or deployment. Subsequent implementation was executed on **2026-10-07** as `tournament_numerical_20261007_01` (§13), then the authorized simulator/C0 improvement as `c0_tournament_copula_20261007_01` (§14). Original proposals and receipts remain dated research evidence, not retrospectively relabeled experiments.

## 1. Decision

**Use a versioned generative tournament model, with exact propagation where tractable and Monte Carlo for genuinely adaptive state. Keep unchanged native C0 as the sports reference.** Reliability requires three separate proofs:

1. **Rules and information:** every path follows this edition’s rules using only the declared origin’s information.
2. **Predictive law:** the joint distribution, not merely match or qualification marginals, performs adequately on independent editions.
3. **Numerics:** reported probabilities accurately integrate that law, with numerical error distinguished from predictive uncertainty.

An exact algorithm proves neither the sports model nor its inputs correct. More simulations reduce integration error, not missing-roster, wrong-draw, scorelaw, or calibration error. Phylourny supplies an exact knockout algorithm rather than an improved pairwise sports model [1]. Calibration is a property of forecasts **and observations**, not of normalized output alone [29].

**Recommended architecture [INFERENCE]:** source-bound edition specification → frozen origin and entrants/roster assumptions → complete native C0 pair-series tables → explicit draw/choice kernels and optional separately validated joint state → exact small-state propagation or complete-path Monte Carlo → joint outcomes and marginal summaries → target-specific scoring and edition-level validation. Use exact knockout suffixes inside otherwise stochastic simulations when the remaining state permits it.

Do **not** automatically add tournament temperature, probability caps, regional discounts, or another residual fit. The earlier residual/score-law challenger `tournament_c0_20261006_01` remains **NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE**; its series Brier guard failed. The later marginal-preserving copula also fails its separate CAL and support gates (§14). These negatives concern the declared recipes, not a proof that tournament dependence is irrelevant or C0 has reached an informational ceiling.

## 2. Established results and their boundaries

| Question | Established result | Boundary for this project |
|---|---|---|
| Fixed knockout champion probabilities | Exact bottom-up propagation from a pairwise matrix; Phylourny gives an O(K²) method [1]. | Conditional independence and a specified tree/BO law; not full joint placements or real-world calibration. |
| Groups feeding knockouts | Enumerate small groups and retain dependencies when trajectories mix [2]. | Paper uses its own tie convention and omits scorelines; neither is a LoL rule. |
| Constrained draws | Greedy locally uniform choices can produce nonuniform complete assignments [3,28]. | Uniform feasible matchings and the organizer’s sequential lottery are different targets. |
| Swiss pairings | Pairings depend on previous results; matching formulations encode particular policies [4]. | Chess optimization is an algorithmic analogue, not Riot’s draw law. |
| Persistent uncertainty | Posterior predictive integration samples parameters and outcomes; shared state induces dependence [7]. | Useful only if the posterior/process is supported; added variance need not improve forecasts. |
| Tournament scoring | Proper scores evaluate specified targets [11,13]; marginals do not identify dependence [12,14]. | A good match score is a guard, not qualification/placement/joint validation. |
| Simulation precision | Binomial intervals require their sampling assumptions [16,21]; adaptive stopping needs time-uniform inference [17]. | Numerical intervals are not confidence intervals for tournament model accuracy. |

Research comparing tournament *designs* under assumed strengths is not validation of this project’s forecasts [5]. Combinatorial prediction-market algorithms illustrate why arbitrary joint queries can be harder than champion marginals; their pricing guarantees do not transfer wholesale to sports forecasting [6].

## 3. Specify the actual edition and information set

### Official rules change across editions

Directly inspected Riot rulebooks establish the following; sections and evidence quotes are retained in the source ledger.

| Edition | Verified rule | Simulation consequence |
|---|---|---|
| Worlds 2023 | Swiss Round 1 prohibits intra-region pairings; Rounds 2–5 have **no additional restrictions**. Quarterfinals draw 3–0 against 3–2 and put those two matches on opposite halves [22] (§§6.3.2–6.3.4). | Do not impose later editions’ no-rematch restriction. Represent draw-order/slot policy. |
| Worlds 2024 | Swiss prohibits rematches; officials **may waive** this when no eligible draw remains. Drawn teams move to the next available slot if a restriction would be violated [23] (§6.3.3). | A matching optimizer is not automatically the lottery. Automatic minimal relaxation is a scenario unless the official waiver policy is represented. |
| Worlds 2023/2024 | Quarterfinal draws happen after Swiss ends [22,23]. | Pre-Swiss forecasts must integrate qualifier sets and draws, not use eventual opponents. |
| MSI 2023 | Eight-team BO5 double-elimination bracket; Grand Final loser is eliminated as second, without reset. Play-in has two four-team BO3 brackets and a cross-bracket BO5 decider [24] (§§6.2–6.3). | A generic reset-final template is wrong here. Play-in and main bracket differ. |
| MSI 2024 | BO5 double elimination with one final; regional separation through upper Rounds 1/2 and lower Round 1; draw after May 5 matches [25] (§§6.1–6.3). | Restrict the appropriate reachable graph nodes, not merely the opening pairing. |
| MSI 2025 | BO5 play-in/main-bracket structure and one final; Fearless makes previously picked champions unavailable for the rest of the series [26] (§§6.1–6.3,7.5.4). | Future drafts are latent state if modeling maps; a winner-only series engine need not invent a draft forecast. |

MSI 2024 grants the upper-bracket entrant first-game side choice in the final; a coin flip decides the first choice in other specified rounds, and the previous game’s loser chooses subsequently [25]. **The right to choose is not the chosen blue/red side.** A pre-event forecast must integrate unknown choices/drafts or use a model already marginalizing them—not insert eventual choices.

**Access limit:** full Worlds 2025/2026 and MSI 2026 primer bodies were not accessible through fetched Riot pages/raw pages in this research. Initial snippet-based detailed claims were withdrawn. Detailed rules for these editions are **UNVERIFIED here**, not assumed unchanged from 2024. This is an access limitation, not proof that Riot published no rules. Accessible PDFs support the historical statements above.

### Forecast contract

Bind rulebook version/availability; entrant identities and seeds; legal phase graph; BO at each possible node; slot routing; byes/forfeits; tie and official tied-placement categories; rematch/regional restrictions; random draw and discretionary-choice policies; resets; and target definitions. A result-archive date does not authenticate prior publication of these inputs.

Use separate forecast classes:

- **Authenticated pre-event:** released rules/entrants/starting rosters and available frozen artifacts, with source time ≤ data cutoff ≤ prediction time < event start.
- **Conditional scenario:** explicitly fixed roster/entrants/draw assumptions without claiming historical availability.
- **Checkpoint:** condition on completed/released results and published draws by the checkpoint; keep remaining routes uncertain.

If forecasting before play-ins, propagate uncertain play-in qualifiers into the main stage. Forecasts starting with known main-stage entrants are conditional on that set, not unconditional forecasts of the whole event.

For national teams/new rosters, use player identities and the existing no-organization/W20-optional contract, not an invented organization or nationality-as-strength shortcut. First Stand’s regional club qualification and resident-player rules are **not national-team rules** [27]. Authentic national-team eligibility, announcements and lineups remain separate prerequisites.

## 4. Algorithm by format

### Fixed single elimination: exact dynamic programming

For disjoint child-subtree winner distributions L and R, and conditional series-win probability p(i,j), a team i from the left wins the parent with probability:

`q(i) = L(i) × Σ[j in right subtree] R(j) × p(i,j)`.

The corresponding right-team probability uses `1−p(i,j)`. Propagate leaves to root [1]. BO-specific series probabilities belong to each node. **C0 inputs are already series probabilities: never apply a second BO projection.** Exactness refers to arithmetic integration of the selected conditional law, not exact sports predictions.

A champion vector alone does not encode joint finalists, exact routes or all placements. Add the required joint events or enumerate complete legal outcome paths. Eight-team binary elimination has 2⁷=128 winner paths; sixteen teams have 2¹⁵=32,768. These counts exclude random seeds, scorelines, latent integration and extra phases.

### Groups, GSL and double elimination

A four-team binary round robin has six matches and 2⁶=64 outcome vectors. Enumerate them, apply the **actual** tie rules, and retain ordered qualifier-slot assignments. Brandes et al. explain why joint advancement must be retained when group trajectories mix [2]. Their uniformly random tie resolution is an example assumption, not a LoL tiebreak.

GSL and double elimination require the actual directed match graph. Sparse-state propagation can enumerate winners/losers routing, loss counts, pending slots and final-reset conditions. Merge only states with identical future-relevant identity/history/choice information. A binary knockout recurrence is insufficient for losers brackets.

**[INFERENCE]** Small fixed graphs permit useful independent enumeration oracles and may permit production exact propagation after benchmarking. Not all existing profiles are established as tractable. Never import a reset merely because a format is named double elimination; the inspected MSI finals do not reset [24–26].

### Swiss: history and pairing law

Carry entrant identity, record bucket, original seed, relevant ranking state, opponent history, eligibility/waiver state and publication status of fixed draws. Add bye/side/tiebreak history when the policy actually uses it. Counts of teams at 2–1 are insufficient: different identities and previous opponents imply different futures [4].

Monte Carlo is a practical default for adaptive 16-team Swiss **[INFERENCE]**. Exact state DP is possible in principle; tractability depends on reachable equivalence classes. “Swiss mathematically requires simulation” is too strong. Measure state growth rather than claiming a universal efficient exact solution.

### Legal matching is not a draw law

Our exact four-allocation example reproduces the distinction in Roberts–Rosenthal [28]. Slots Q/F/B, entrants M/S/U, and B–U forbidden: a feasible sequential first-available draw over six random input orders yields probabilities **1/3,1/3,1/6,1/6** over the four legal allocations. Uniform legal complete allocation gives **1/4 each**. Both obey the same final restriction.

For a uniform complete-matching target, choose an edge proportional to the number of feasible completions after it [3,28]. For optimizer-with-random-ties, count only optimal completions. For the organizer’s lottery, implement its sequence, slot shifts/lookahead and disclosed discretion; do not replace it with the uniform reference because that seems fairer.

Existing `_perfect_matching` is count-weighted among minimum-cost complete matchings, not greedy. The new tiny consumer exercise reached all four legal matchings using its four root tickets. This proves that example, **not agreement with Riot’s lottery or all formats**. Source-backed equivalence to the lottery remains to be established where claimed.

### Hybrid exact suffixes

Sample necessary stochastic prefixes—unknown draws, adaptive Swiss outcomes and supported latent state—then calculate a fixed knockout suffix exactly conditional on that prefix:

`P̂(E) = average_m P(E | sampled prefix_m)`.

The law of total variance shows that replacing a sampled suffix indicator with its correct conditional expectation cannot increase variance under the same prefix design. Structured tournament algorithms exploit related conditional completions [1,2,6]. **[INFERENCE]** This is a strong numerical optimization candidate, not a sports-model improvement; runtime still requires measurement. A history-dependent joint model needs sufficient latent/history state in the suffix.

## 5. Match probabilities do not identify the joint law

### Persistent state versus fresh noise

A Bayesian joint predictive distribution integrates shared state: `P(H|D) = ∫ P(H|θ,D) π(dθ|D)`. Reuse the relevant player/team/event draw across its appearances within a rollout [7]. Fresh independent draws each match define a different model. Outcome randomness remains necessary even when parameters are sampled.

Dynamic Bradley–Terry research models time-varying strength and evolution [8,10]. The adaptive-innovation paper reports NBA results, **not LoL validation**; online publication was June27,2026, despite December cover-date metadata [9]. None supplies an automatic C0-compatible event-variance estimate. Long-term drift, series shocks, persistent event effects, roster uncertainty and input/model error are distinct hypotheses. An identical additive shock to every team cancels in pairwise strength differences.

### Checked four-team counterexample

Bracket A–B and C–D, then final. Every potential A-versus-other pre-event probability is0.6; other pairs are0.5.

| Joint law | A champion | B champion | C champion | D champion |
|---|---:|---:|---:|---:|
| Independent outcomes at fixed probabilities | 0.360 | 0.200 | 0.220 | 0.220 |
| A persistent state: p=0.9 or0.3, each probability1/2 | 0.450 | 0.200 | 0.175 | 0.175 |

Both laws have identical **pre-event potential pair marginals** and total champion mass one. The second gives E[p²]=0.45 rather than E[p]²=0.36. Conditional on A reaching the final, its win probability is0.75, not0.6. Equal pre-event marginals do **not** imply equal reached-match conditional probabilities.

This is an exact counterexample, not evidence that positive correlation helps LoL. We independently compared DP with all eight binary paths for three probability matrices; the calculation passes.

### Frozen origin versus checkpoint inference

For a pre-event rollout, draw/evolve the declared generative state along hypothetical histories. At an actual checkpoint, posterior inference must condition on results released by that time. Forcing observed winners while retaining an unchanged latent mixture does not perform Bayes conditioning.

Automatically updating C0 ratings/features from simulated winners changes the frozen reference recipe. Do so only in a separately declared dynamic sports model with a valid observation/update law and release schedule. There is no general evidence that all mid-event updates help or harm.

**Project decision:** preserve unchanged C0 and the negative residual study. A later dependence experiment needs a separately frozen question, matched-independent control with the same marginals, match guards and usable edition-level evidence. This research does not start another q/variance/calibration sweep.

## 6. Winner, exact score and length are different targets

BO3 winner probability0.6 permits both:

- A2–0 with probability0.6; B0–2 with0.4: always two maps.
- A2–1 with probability0.6; B1–2 with0.4: always three maps.

These are illustrative laws, not fitted LoL descriptions. A C0 series-win table does not uniquely identify scorelines, draft sequences, fatigue or duration.

A compatible winner-conditioned law supplies `P(S=(r,ℓ))=p×P(ℓ|A wins)` and `P(S=(ℓ,r))=(1−p)×P(ℓ|B wins)`. The latter must correspond to swapped teams/1−p for symmetry. Fit on released earlier labels; preserve winner marginals. IID-map, Markov-momentum and Fearless-aware map laws are different assumptions.

For unreleased drafts/side, integrate a validated map-state model over legal policies or retain a clearly labeled marginal series/score model. Fearless makes future legal champion pools history-dependent [26]; it does not authorize actual future picks. Do not infer the assembled C0 exact-score law from one neural expert’s terminal likelihood without mixture compatibility proof.

**Existing result, not rerun:** continuous native-C0 2026 exact-score LL/length RPS: learned **1.070488252/0.162662216**, pooled TRAIN **1.093736045/0.171193413**, fixed Markov0.35 **1.065382436/0.162243865**. Learned beats pooled but loses to the fixed heuristic on these development diagnostics. No scorelaw is promoted.

The current engine differs from continuous diagnostics: learned `conditional_distributions()` uses **100 width-0.01 midpoint bins**; the fixed Markov law defaults to four intervals `[0,.5,.65,.8,1]`. The runner uses learned law for years≥2025 and fixed law earlier. Matching saved bin laws is not exact continuous propagation. The initial scout’s blanket four-bin claim was corrected; the correction is retained.

## 7. Numerical accuracy and simulation budgets

### Four uncertainty layers

1. **Outcome:** different legal real-world paths even with known strengths.
2. **Predictive state:** uncertain strengths/rosters/choices/parameters under a chosen model.
3. **Numerical approximation:** finite latent integration, binning and bracket simulation approximate that law.
4. **Empirical:** few observed editions limit calibration and superiority claims.

Wilson intervals describe an appropriate numerical Bernoulli estimator, not sports-model correctness. Posterior intervals depend on model/prior. Edition bootstraps address declared empirical estimands, not integration error.

### Independently calculated fixed-N planning

For **iid independent complete tournament paths under one frozen law**, worst-case p=0.5 pointwise normal planning is `N≈ceil(z(.975)²/(4ε²))`.

| Absolute half-width | Pointwise95% normal planning | Simultaneous16-event95% Hoeffding/union bound |
|---|---:|---:|
| ±0.5 percentage points | 38,415 | 129,230 |
| ±0.1 percentage points | 960,365 | 3,230,735 |

Normal planning is **not a finite-sample coverage guarantee**. The simultaneous column uses the conservative bounded-independent inequality `2K exp(−2Nε²)≤0.05`, fixed N and16 predeclared events; the16 events need not be mutually independent. Fixed-time and time-uniform concentration are developed in [17]; these arithmetic examples are calculated here, not LoL experiment sizes.

At N10,000, worst-case normal pointwise half-width is **0.980pp**. This does not establish uncertainty in the saved study’s score differences; analyze paired differences and the actual sampling design.

Zero hits are not impossibility: true p0.0001 has **36.786%** chance of no hits in10,000 iid paths. With zero hits, the one-sided exact-binomial95% upper bound is **0.000299528**, or **0.029953pp** [16,21]. Do not convert unsampled categories into structural zeros or add arbitrary floors to hide infinite log loss.

### Sampling design, stopping and variance reduction

Use predeclared fixed N, or an independent pilot followed by fixed N. Stopping based on interim estimates requires an appropriate confidence sequence; repeatedly inspected ordinary intervals are not anytime-valid [17]. Simultaneous claims across entrants/targets require multiplicity handling.

Current coherent simulation consumes an externally supplied strength vector **indexed per rollout**, not a fresh draw from the retained finite list. Conditional on that list, event probabilities can differ across rows: this is not automatically an iid binomial experiment for the finite-mixture target. If the external vectors themselves are independent posterior draws and uncertainty includes their generation, the target/error scope differ. **[INFERENCE]** Do not retrofit the table’s binomial interpretation onto saved coherent outputs without checking the estimator.

Common random numbers can reduce comparison variance under positive covariance, but are not universally beneficial [18]. Same seed alone does not establish semantically aligned draw/game randomness after routes diverge. **[INFERENCE]** Record coupling and estimate error from paired statistic differences, not two unpaired probability bars. This is not implemented here.

Exact suffix integration is the first numerical optimization to consider **[INFERENCE]**. Randomized quasi-Monte Carlo is benchmarkable, not a promised speedup for adaptive bracket logic. Independent randomizations—not dependent points within one net—supply its replication unit [19,20]. Adapt intervals/ensemble corrections to the actual design.

### Finite simulation changes score estimates

For iid Monte Carlo frequency p-hat, expected empirical Brier exceeds underlying-law Brier by p(1−p)/M. Ferro’s fair adjustment for random-sample ensembles is [30] (Eq.3):

`BS_fair = (p_hat−y)² − p_hat(1−p_hat)/(M−1)`.

Our exact binomial calculation at M10,p0.6,y1 gives underlying Brier **0.160**, expected empirical Brier **0.184**, adjusted expectation **0.160**. This distinguishes scoring an issued finite empirical forecast from estimating its underlying law. The adjustment is not automatically valid for arbitrary dependent/stratified/RQMC members. No stored project metric changed.

For realized qualifier sets/full paths, sampled frequencies can be zero despite positive model probability. Prefer exact event probabilities, explicit path-likelihood integration or a predeclared scoreable estimator where feasible. A marginal-Brier correction does not repair joint-log estimation, which also has finite-sample bias. Use log-sum-exp when integrating conditional path likelihoods [7].

## 8. Evaluate the requested targets

| Target | Primary evaluation | Not established alone |
|---|---|---|
| Match/series winner | LL/Brier and chronological guards | Tournament dependence/ranks/coverage. |
| Champion | Multicategory Brier/log score [11] | Full route/placements. |
| Qualification/finalist marginals | Mean-team binary Brier; frozen aggregation | Which teams jointly qualify. |
| Official placement bands or wins | RPS with ordered categories and fixed normalization [11,13] | Joint ranking; win-count and placement RPS are distinct. |
| Qualifier set/finalist pair | Joint categorical log/Brier where scoreable [11] | All path features. |
| Joint wins/placements vector | Energy plus targeted variogram/dependence diagnostics [12] | Universal joint-calibration guarantee. |
| Exact score/length | Scoreline log score and ordered length RPS | Champion/financial performance. |

Energy score is proper under its conditions but can weakly distinguish correlation errors. Variogram score is not generally strictly proper and can miss common shifts [12]. Freeze vector encodings, weights and any historical standardization; arbitrary nominal category numbering is not ordinal. Copula calibration is complementary [14]; continuous PIT/CopPIT cannot be applied unmodified to discrete tied placements.

### Checked qualifier-set counterexample

Four teams, exactly two qualifiers. U assigns1/6 to every legal pair; C assigns1/2 to AB and1/2 to CD. Both give every team qualification probability0.5 and, on observed AB, mean-team Brier **0.25**. Joint log losses differ: **log6=1.791759469** versus **log2=0.693147181**. This proves marginal scores cannot identify these laws—not empirical superiority of either.

Independently sampling each team Bernoulli0.5 preserves expected qualifier count two but produces the wrong count with probability **0.625**. Expected mass/flow is not per-path feasibility. Legal complete-path simulation supplies coherence; marginal feasibility LPs are necessary checks, not certificates of joint law or draw policy.

### Edition-level evidence

Freeze origin, artifacts, entrants, assumptions, missingness, targets, weights, comparisons and stopping rules before outcomes. Compute paired per-edition differences; predeclare equal-edition versus target-weighted estimands. Multiple phases/checkpoints/teams within one edition are not independent editions. Shared seasons/patches/players may create cross-edition dependence; record it and examine appropriate blocking sensitivities.

One edition can receive a proper score but cannot establish general calibration or a dependable comparison interval. Resampling one edition always returns the same difference. More simulation paths cannot manufacture editions. Few-cluster inference has no universal threshold; its literature stresses setting dependence and does not supply a tournament-specific guarantee [31]. **Eight editions is the frozen study’s local gate, not a theorem**; neither8 nor30 ensures adequate power. Plan precision using target effect, edition-level variation, heterogeneity and the prospective opportunity universe.

Repeatedly examined history remains development data. Another recipe chosen from those results incurs search optimism; nominal intervals do not undo it [15]. Genuine future confirmation requires forecasts recorded before outcomes, not relabeling another retrospective split.

## 9. Existing repository: guarantees and evidence boundaries

Read-only findings and anchors are retained in `LocalTournamentContract.json` and its correction. No new engine defect is asserted from an unexercised concern.

| Mechanism | Inspected state | Boundary |
|---|---|---|
| Matching | `tournament_formats.py:487–578` counts/samples optimal completions. | Specified measure≠certified organizer lottery. |
| Checkpoints | `:602–666,1261–1287,1902–1929` checks clocks/published draws/completed facts. | Forcing facts does not condition latent parameters; authentication is separate. |
| Native inputs | `tournament_native.py`, `tournament_research_cohort.py` distinguish identity, eligibility and conditional/publication semantics. | First-phase actual rosters/known-draw scenarios are not authenticated forecasts. |
| Scorelaw | `tournament_scorelines.py:467–486`; runner`:613–616` uses learned.01 bins for2025+, fixed law earlier. | Bin approximation, continuous scoring and exact winner marginals differ. |
| Joint paths | Core aggregates joint finalists/advancement; coherent arms reuse draws. | First trace is illustrative; hypothetical-pair reports are not reach-conditioned. |
| Flow LP | `tournament_metrics.py:295–397` checks marginal mass/capacity feasibility. | Does not authenticate all joint probabilities or convergence. |
| Oracles | `verify_tournament_distribution_oracles.py` has small independent examples. | Not a general exact Worlds/MSI forecasting backend. |

**Frozen study, not rerun:** `tournament_c0_20261006_01` requested eight editions×seven models: **56cells,29forecasts,27named absences**, seven native input cases but only **one supported primary2026 edition**. MSI2026 lacks first-requested-phase raw roster inputs; no later substitution. Whole-edition and requested-phase roster-oracle absences use different scopes.

On all3,103 canonical2026 series, residual−C0 LL **+0.000501948**, monthly95% interval **[+0.000113996,+0.000892632]**: worsening under the match-weighted estimand. Brier **0.181034493→0.181218284**. Equal-month LL interval crosses zero, not overriding the guard. Full11,550 scores include8,447 exact C0 warmup rows.

Single LCK qualification-phase Brier/placement RPS: coherent **0.066407343/0.060619163**, C0 **0.067723103/0.061895235**, matched-independent **0.066192843/0.060035727**. Small coherent-versus-C0 point gains do not establish incremental dependence benefit; coherent is worse than matched-independent on these two scores. This phase has no champion/final/Swiss target. Earlier651 focused tests and frozen numerical audits belong to that execution, **not this research**.

## 10. Priorities—proposals, not an execution queue

1. **Authenticate the contract:** bind edition rules, actual lottery/choice procedure, availability, complete entrant universe, roster status and targets. Retain absences and keep retrospective scenarios separate.
2. **Establish numerical correctness for unchanged C0:** reuse engine/oracle conventions; an exact knockout backend/suffix requires separately authorized implementation. Compare identical laws against enumeration; benchmark time/state growth and numerical error independently of sports-model changes.
3. **Establish baseline tournament evidence:** report every requested cell and information class; assess all supported marginal/joint targets across eligible editions. Historical breadth remains development evidence and does not repair the challenger’s frozen primary gate.
4. **Then freeze one joint-model hypothesis if supported:** coherent versus matched-independent with identical marginals/scorelaw and C0 match guards; or change only scorelaw with fixed winner marginals. Do not combine draw repair, variance, draft model and calibration into one improvement claim.
5. **Prospective independent editions:** record forecasts before outcomes under a separately frozen protocol, with coverage and precision rules. Do not move existing deadlines, restart G0/old10/10, fit financial heads or certify futures.

**Recommendation [INFERENCE]:** immediate software direction is a rule-auditable **hybrid exact/Monte-Carlo simulator**, not another probability-softening model. Immediate evidence direction is authentic, complete edition-level forecasts and joint-target evaluation. Neither requires pretending the rejected residual challenger succeeded.

## 11. Corrections and limitations

The [September16 report](tournament_simulation_deep_research_report.md) remains history. This report supersedes its unsupported universal tournament temperature, .85–.88 caps, .70 regional discount, iid single-match assertion, mathematical Monte-Carlo necessity, and universal30-cluster bootstrap claims. Match forecasts need not be iid; match LL remains a useful guard. No inspected source establishes those fixed remedies generally for LoL. Brandes’ inspected preprint authors are **Ulrik Brandes, Gordana Marmulla and Ivana Smokovic**, correcting older names [2]. Preserve historical evidence and negative experiments.

Exact algorithms/score identities hold under assumptions; Riot PDFs establish rules, not roster publication; other-sport models motivate hypotheses, not LoL transfer. Some primary records were inspected only at abstract/metadata scope, explicitly retained in the ledger. Inaccessible pages were not treated as read text. The original Hoeffding scan was not extractable and an alternative hit a bot check; it is not counted as inspected. The numerical illustration uses the inspected concentration framework in[17] and independently calculated arithmetic.

No universal variance, cap, edition count, guaranteed RQMC speedup or calibration certificate follows. At completion of the original research phase, no proposed exact-backend runtime benchmark or new predictive comparison had run; the subsequently authorized numerical implementation is recorded separately in §13. No new worker, source authorization, national-team rules contract, genuine future forecast or production change follows.

## 12. New verification and retained evidence

- **PASS:** six mathematical check families: three knockout matrices versus exhaustive paths; constrained draw enumeration/tiny matching consumer; qualification joint-law/feasibility; BO3 winner/length nonidentifiability; fixed-N/zero-hit arithmetic; finite-M Brier expectation.
- [Numerical results](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/mathematical_checks.json); [executed source archive](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/verification_source.zip).
- [31-source/claim ledger](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/source_ledger.json), six retained findings, and [source-audit correction](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/LocalTournamentContract_correction.json). Access gaps/withdrawals remain visible.
- [Citation verification](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/citation_verification.json) checks bibliography structure, **not truth/authentication**. [Research verification](../data/08_reporting/tournament_simulation_research/tournament_simulation_research_20261006_01/research_verification.json) records arithmetic/source-archive agreement, links and protected pins.
- The final pin comparison found `conf/base/c0_serving.json` differs from an earlier recorded hash. The current file still declares `c0-native-2026-w32-e12-v1`; this research did not edit it or establish live serving identity. The failed comparison is retained in the verification receipt, not silently replaced with a new baseline. The other four selected frozen pins match.
- The original research phase added no permanent tests or forecasting implementation and did not rerun the old study. Canonical manifest, frozen models, serving configuration, completed decisions and forward dates were unchanged by that phase. Subsequent implementation evidence is separate in §13.

## 13. Authorized implementation and historical numerical execution — 2026-10-07

**Completed:** `tournament_numerical_20261007_01`, **HISTORICAL_NUMERICAL_INTEGRATION_ONLY**. This implements the numerical direction above, not a new uncertainty/calibration recipe or promotion. Unchanged native C0, origin-eligible annual A0, prior-calibrated Glicko-format and uniform integrate the same retained origin pair tables from the closed `tournament_c0_20261006_01` study. There are **0 predictive fits, 0 new native input inferences and 0 residual-arm reruns**.

### Implemented contract

- `src/models/tournament_exact.py` supplies fixed-knockout dynamic programming and bounded exact small-state enumeration. GSL, deterministic win/head-to-head/seed round robin, graph, gauntlet and double elimination preserve complete-path routing and joint targets. Full fixed-knockout ranks are bounded to16 entrants; enumeration is bounded to20 series, and finite-vector mixtures to1,048,576 worst-case weighted paths. Unsupported exact requests reject explicitly, without a Monte Carlo fallback.
- `simulate_tournament_bracket(..., integration="monte_carlo"|"exact"|"hybrid")` and `TournamentBracketSimulator.integration` preserve the default Monte Carlo path. Hybrid samples adaptive Swiss and the quarterfinal draw, then integrates the fixed knockout suffix. The supported MSI main brackets are fully exact. Supplementary exact-reference rows reuse those MSI results, recording zero additional integration evaluations.
- Joint finalist-pair and qualifier-set log/Brier scores supplement champion, marginal qualification, placement RPS, Swiss-record and scoreline/length targets. Canonical unordered member sets retain dependence; sparse zero-probability observed targets produce **infinite unclipped log loss**, serialized as null with the explicit reason.
- `src/analysis/tournament_numerics.py` reports fixed-N iid complete-path Wilson and zero-hit bounds; iid-prefix bounded-contribution Hoeffding bounds; and zero **sampling**, not arithmetic/model, error for exact integration. Non-iid supplied-vector designs do not receive binomial intervals. Intervals are pointwise integration statements, not simultaneous or predictive-calibration guarantees.
- The new CLI namespace freezes forecasting pair inputs separately from observed-label identity and stores an exclusive snapshot of the reviewed Markdown. Old v3 pins remain lineage; the current execution separately binds v4 replay and checks hypothetical targets carry no actual match/result/release identity. It does not exempt released targets from the stricter guard or silently repin the closed study.

### Actual execution

```bash
.venv/bin/python scripts/run_tournament_distribution_benchmark.py --numerical-benchmark --run-id tournament_numerical_20261007_01 --source-run-id tournament_c0_20261006_01 --integrations monte_carlo hybrid --simulations 10000 --seed 20261008
```

Worlds2023/2024 and MSI2023/2024 × four models × two requested methods: **32 requested cells, 28 available, 4 named absences**. The absent annual-A02023 artifact accounts for both methods in both2023 editions; no proxy or later artifact substitutes for it. Seven supplementary MSI exact-reference rows yield35 stored forecast rows, not35 separate integration evaluations. Monte Carlo uses10,000 complete paths per available cell; Worlds hybrid uses10,000 independent prefixes, while MSI exact has no random-simulation count. Total reported runtime: **235.117s**.

Unchanged C0 descriptive results:

|Edition|Integration|Champion Brier|Placement RPS|Joint finalist LL|Joint qualification LL|
|---|---|---:|---:|---:|---:|
|Worlds2023|Hybrid|1.040889428|0.129369850|5.277338024|8.111728083|
|Worlds2024|Hybrid|1.111883059|0.119482424|3.871302935|7.013115795|
|MSI2023|Exact|0.609899272|0.119636860|3.086207565|No qualification target|
|MSI2024|Exact|0.651716150|0.111260552|1.552558758|No qualification target|

Across all supported models and reported scalar targets, maximum10,000-path MC-versus-exact MSI probability error is **1.462846pp**. Worlds MC-versus-hybrid disagreement reaches **2.180000pp**, but neither method supplies exact Swiss truth. Same seed is not certified common-random-number coupling. Exact/hybrid were **not faster at this N**: native C0 Worlds MC/hybrid took9.907/12.919s and9.721/12.726s; MSI MC/exact took2.570/6.502s and2.375/5.325s. Exact removes sampling error for the supported law, not prediction error.

The Worlds2023 uniform hybrid run missed the observed qualifier set, leaving its joint qualification log loss infinite. No clipping, smoothing or post-score N adjustment hides this finite-sampling limitation. MSI qualification/Swiss scores are separately absent because those main-bracket specs declare no such target.

### Verification and retained incidents

- **576 final focused/new-plus-existing tournament and benchmark tests passed**, one existing Starlette/httpx warning; not the full application suite. The real eight-path/prefix CLI smoke completed32 requested cells,28 available/4absent in37.357s. Public exact API also matched an independent8-path rational knockout oracle exactly.
- An independent stdlib-only MSI route enumerator checked **16,384 legal paths in each of7 supported cases**, champion/final/joint-final/full ranks/official placement bands and14-series mass: **1,092 numerical comparisons, maximum error1.665334537e−15**. It imports no production simulator.
- Independent saved-joint proper-score arithmetic checked **98 values**, including repeated exact-reference rows, maximum error2.220446049e−16 and the retained infinite score. This certifies arithmetic from recorded target keys, not their historical authentication.
- Static follow-up review found0 unresolved required findings. Repaired RED boundaries are retained: null digests bypassing source verification; an eliminated Swiss rank9 admitted without placement bands; rank-reference opening pairs sorting dictionaries; and excessive finite-vector exact work. The probability parser is now included in runtime bindings. Two independent-verifier launches initially failed solely because shell quoting passed literal newlines; corrected launches used unchanged verifier programs, with no simulator/model changes.
- [Decision](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/decision.json), [full metrics/comparisons](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/metrics.json), [execution binding receipt](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/receipt.json), [exercised checks/incidents](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/verification/execution.json), [independent MSI oracle](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/verification/independent_msi.json), [independent joint scores](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/verification/independent_joint_scores.json), [executed source archive](../data/08_reporting/tournament_numerical/tournament_numerical_20261007_01/verification/source.zip).

**Limits:** these are retrospective known-opening-draw/first-roster conditional scenarios, not authenticated pre-event issuance. Uniform feasible matching remains a declared scenario rather than the certified Riot lottery; detailed inaccessible rulebooks remain unverified. Four development-exposed editions, reused inputs and exact arithmetic do not establish predictive superiority, tournament calibration, national-team coverage or financial reliability. Source/model/manifest nonmutation guards passed; canonical C0/A0 artifacts, serving, old negative decisions, G0/10-of-10 and forward dates remain unchanged. No new fitting, genuine future forecasts, live acquisition, deployment, gate change or automatic follow-up sweep.

## 14. Authorized simulator repairs and marginal-preserving C0 tournament experiment — 2026-10-07

**Decision: retain the simulator correctness fixes; reject the learned challenger.** The user requested better simulations and tournament-capable C0. Run **`c0_tournament_copula_20261007_01`** completed one controlled historical experiment, with **`NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE`**. CAL performance guards also fail; obtaining more assessment editions alone would not make this fitted recipe eligible. This does not establish that C0 is great at tournaments or that all dependence models fail.

### Retained simulator changes

- Reject contradictory Swiss-quarterfinal draw/choice contracts instead of silently selecting a controller.
- Preserve the original pair orientation and smaller probability tail, including rare `1e-17` cases. Winner-conditioned score bins use the original winner-oriented probability, not a double-complemented or latent-shifted bin. The repair applies to default/zero-copula and positive-copula paths.
- `query_tournament_joint_numerics` reports uncertainty for the **actual queried finalist/qualifier set**, including feasible unsampled sets. It uses retained integer counts and authoritative target topology: fixed-N iid Wilson and zero-hit bounds, iid-prefix Hoeffding bounds, or exact zero sampling error. Structural impossibility and finite-sampling zero are distinct. Arbitrary supplied strength vectors do not acquire an iid certificate.
- A positive copula uses complete-path pre-event Monte Carlo. Unsupported checkpoint/exact/hybrid requests fail explicitly; there is no silent fallback. Numerical intervals condition on the chosen model, roster and rules; they do not certify predictive accuracy.

### Frozen model and chronology

The extension draws independent standard-normal **player states once per rollout**, with team state $U_A=\sum_{i\in A}Z_i/\sqrt{5}$ reused throughout the event. Its conditional series probability is

$$
p(A,B\mid U)=\Phi\left(
\frac{\Phi^{-1}(p_{C0}(A,B))+\sqrt{\rho}(U_A-U_B)/\sqrt{2}}
{\sqrt{1-\rho}}\right).
$$

Disjoint five-player rosters imply an exactly preserved pre-event marginal $E[p(A,B\mid U)]=p_{C0}(A,B)$; $\rho=0$ is identity. This is a learned **joint-law extension**, not a fixed probability blend or a C0 base refit. Static disjoint event rosters reduce it to independent team-normal states; this experiment does not identify player-specific variances or transfer benefits. Organization history is not required by the extension, but authentic national-team predictive performance was not evaluated.

One $\rho\in[0,0.5]$ was selected using TRAIN-only equal-edition mean normalized joint sports-response path NLL: four 2023–2024 editions/**108 series**. Two 2025 editions/**54 series** are CAL; requested assessment is LCK2026 Road to MSI and MSI2026. TRAIN releases precede CAL and CAL releases precede assessment under the explicit conservative daily-clock scenario. First actual rosters/day-only source clocks remain retrospective scenarios, not authenticated announcements or issuance.

The selected **$\rho=0.18832422863931964$** used a fixed65-point grid and bounded refinement of interior local minima. Fit quadrature was8 scrambled Sobol replicates × $2^{15}$; fixed-parameter verification8 × $2^{17}$. There was **one formal dependence fit, zero base fits**, 68.478s fitting and1203.398s producer execution. All controls share a fixed Markov score law, $\beta=0.35$, with original prior-probability bins; the old conditional CAL score-law recipe was not reused or refitted.

The canonical guard covers all **11,550 series** (4,642/3,805/3,103): analytic pre-event LL/Brier changes are exactly zero by the variance identity; independent128-node integration has maximum marginal error **3.330669074e−16**. This is a marginal-preservation proof, not a fresh native C0 replay or evidence of improved match accuracy.

### Complete requested evaluation and explicit absences

Actual formal output contains **32 requested event×model cells, 28 supported forecasts and four explicit MSI2026 absences**. Each supported forecast uses **50,000 complete paths**, totaling **1,400,000 paths** across seven editions and unchanged C0/copula/Glicko/uniform. MSI2026 is withheld for every model: `NATIVE_FIRST_EVENT_ROSTER_ABSENT: 2849`. No later roster replaces it. Only **one** complete primary assessment edition is supported versus the frozen minimum **eight**.

|Equal-edition CAL metric|Unchanged C0|Copula|Candidate−C0|
|---|---:|---:|---:|
|Normalized joint sports-path NLL|0.555157457|0.552265983|−0.002891474|
|Champion Brier|0.756896411|0.753132232|−0.003764180|
|Champion LL|1.674725413|1.680826341|+0.006100928|
|Placement RPS|0.102873673|0.104483241|+0.001609568|
|Worlds2025 qualification Brier only|0.152019764|0.152754613|+0.000734849|
|Worlds2025 joint qualification LL only|5.983496377|6.165817934|+0.182321557|

Thus **CAL fails champion LL, qualification Brier, joint qualification LL and placement RPS guards**, despite improved mean path NLL and champion Brier. The descriptive whole-edition95% path-NLL difference interval is **[−0.010957662,+0.005174715]**, only two editions; placement difference is **[+0.000340110,+0.002879025]**. Qualification targets exist on only one CAL edition, so no edition-bootstrap interval is reported there. Marginal series-score LL and series-length RPS remain unchanged by construction; that is not proof that joint tournament outcomes are unchanged.

|LCK2026 Road to MSI, five series|Path NLL/series|Qualification Brier|Joint qualification Brier|Joint qualification LL|Placement RPS|
|---|---:|---:|---:|---:|---:|
|Unchanged C0|0.634748078|0.066684330|0.447819390|0.855101564|0.061988427|
|Copula|0.668577915|0.076171790|0.523845766|0.987518241|0.065348512|
|Glicko|0.616982279|0.087492987|0.547304134|0.954511945|0.055921893|
|Uniform|0.693147181|0.034880662|0.291096523|0.690351093|0.064492798|

The challenger worsens all available LCK targets against C0. One edition supplies no reliable population interval or universal ranking of models. LCK declares qualification/placement, **not champion/finalist targets**; unavailable primary champion/finalist guards reflect the missing MSI edition, not measured LCK championship harm. Joint sports-path likelihood conditions on the observed legal route and omits common organizer-lottery/score terms; it is not an authenticated absolute probability of the complete real tournament.

### Exercised verification, with bounded claims

- **1,066 selected tournament/native/numerics/scoring/format/benchmark tests passed**, one existing Starlette/httpx warning; not the full application or database suite. Actual freeze/run CLIs and the public Python simulator/model/query interfaces were exercised.
- Independent alternative-factorization Cholesky/Genz orthant integration: **288 checks and28 comparisons**, seven fitted-copula paths and21 closed-form controls, zero numerical disagreements. Maximum normalized-NLL difference **0.000063150615**; agreement uses the declared Monte Carlo error policy. Producer fixed-rho maximum diagnostic SE is **0.000090241919**, below0.001. The verifier did not independently refit or certify every fit-grid optimum.
- Independent saved-count/score/bootstrap/hash audit: **526,285 checks** (including repeated assertions), zero failures, maximum scalar error **1.687538997e−14**. It reconstructs joint/marginal targets, placements, prior pair score laws and whole-edition5,000-draw intervals. It checks the first retained path trace, not every simulated path.
- Synthetic public smoke:8,192 copula paths with neutral priors, player rename/roster permutation/future-entrant noninterference/nonmutation; a feasible two-path zero-hit finalist query receives actual zero-hit bounds, while its exact reference is0.25. Generic supplied vectors do not receive fake iid intervals. No genuine future forecast or fresh native-input inference follows.
- The canonical suite completed **`DEVELOPMENT_ONLY`** with11,550 rows: A0/Glicko/Elo complete,039 has9,907 forecasts/1,643 absences. This unchanged annual-A0 suite is separate from the native C0 marginal guard. All30 executed code files are archived with exact producer-bound hashes.

**Retained incidents:** one source-smoke namespace was retired before fitting; three nonformal scalar fits followed (one failed pipeline on the date-only origin boundary, two completed smokes), separate from the one formal fit. Original orientation/bin, strict TRAIN and rare-tail defects were fixed before formal freeze. Initial independent verifiers incorrectly counted descriptive metadata strings as extra partitions; corrected verifier-only partition filtering passes, with original failures retained. A `python -c` launcher lacked `__file__`; the source-preserving temporary-file launcher corrected execution without changing the oracle. The public smoke initially queried an impossible finalist pair; only the fixture was corrected to a feasible set. No post-score model, tolerance or gate adjustment occurred.

**Evidence:** [frozen protocol](../data/05_model_input/tournament_copula/c0_tournament_copula_20261007_01/protocol.json), [decision](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/decision.json), [producer receipt](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/receipt.json), [canonical marginal guard](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/canonical_marginal_guard.json), [execution verification](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/verification/execution.json), [independent likelihood](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/verification/independent_orthant_corrected.json), [independent distributions](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/verification/independent_distributions_corrected.json), [public smoke](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/verification/public_smoke.json), [sealed verification](../data/08_reporting/tournament_copula/c0_tournament_copula_20261007_01/sealed_verification.json).

**Remaining trust gates:** a separately frozen successful recipe, complete independent edition support, authenticated pre-event rules/draws/rosters/source clocks and genuine future forecasts/outcomes. All historical years remain development-exposed. No automatic variance sweep, probability cap, league exception or extra exposed-data recipe follows. Canonical A0/C0 weights, manifest, serving, prior negatives, G0/old10-of-10 and forward dates remain unchanged by this study. No live acquisition, deployment, financial certification or tournament certification. The local implementation and negative experiment are complete; the broader user objective is **not established**.

## Bibliography

| ID | Primary record / inspected evidence | URL |
|---|---|---|
| [1] | Bettisworth, Jordan & Stamatakis. Phylourny (2023): exact fixed-knockout propagation. | https://pmc.ncbi.nlm.nih.gov/articles/PMC10186292/ |
| [2] | Brandes, Marmulla & Smokovic. Stop Simulating! Efficient Computation of Tournament Winning Probabilities. Inspected author preprint; journal page inaccessible. | https://arxiv.org/html/2307.10411 |
| [3] | Diaconis & Kolesnik. Randomized sequential importance sampling for estimating the number of perfect matchings in bipartite graphs. | https://arxiv.org/html/1907.02333 |
| [4] | Sauer, Cseh & Lenzner. Improving Ranking Quality and Fairness in Swiss-System Chess Tournaments. Algorithmic analogy only. | https://arxiv.org/html/2112.10522 |
| [5] | Sziklai, Biró & Csató. The efficacy of tournament designs. Abstract/design-study scope. | https://arxiv.org/abs/2103.06023 |
| [6] | Xia & Pennock. An Efficient Monte-Carlo Algorithm for Pricing Combinatorial Prediction Markets for Tournaments (IJCAI2011). | https://www.ijcai.org/Proceedings/11/Papers/083.pdf |
| [7] | Stan User’s Guide2.38. Posterior Predictive Sampling. | https://mc-stan.org/docs/2_38/stan-users-guide/posterior-prediction.html |
| [8] | Cattelan, Varin & Firth. Dynamic Bradley–Terry modelling of sports tournaments. | https://wrap.warwick.ac.uk/id/eprint/54660/1/WRAP_Firth_Dynamic_Bradley_j.1467-9876.2012.01046.x.pdf |
| [9] | Torelli. Bayesian dynamic Bradley-Terry model with commensurate spike-and-slab priors. OnlineJune27,2026; NBA evidence. | https://link.springer.com/article/10.1186/s40537-026-01486-6 |
| [10] | A Bayesian approach to time-varying latent strengths in pairwise comparisons (PLOS ONE2021). | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251945 |
| [11] | Gneiting & Raftery. Strictly Proper Scoring Rules, Prediction, and Estimation (2007). | https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf |
| [12] | Scheuerer & Hamill. Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities (2015). | https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf |
| [13] | Epstein. A Scoring System for Probability Forecasts of Ranked Categories (1969). Historical ranked-category reference; detailed formula also supported by[11,30]. | https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml |
| [14] | Ziegel & Gneiting. Copula Calibration. | https://arxiv.org/html/1307.7650 |
| [15] | White. A Reality Check for Data Snooping (2000). | https://users.ssc.wisc.edu/~behansen/718/White2000.pdf |
| [16] | NIST/SEMATECH e-Handbook7.2.4.1. Confidence intervals: Wilson/exact binomial. | https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm |
| [17] | Howard, Ramdas, McAuliffe & Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. Abstract and full author HTML inspected. | https://arxiv.org/html/1810.08240 |
| [18] | Glasserman & Yao. Some Guidelines and Guarantees for Common Random Numbers (1992). Author institution abstract inspected. | https://business.columbia.edu/faculty/research/some-guidelines-and-guarantees-common-random-numbers |
| [19] | Owen. Monte Carlo Book: The Quasi-Monte Carlo Parts, chapter17 (2019). | https://artowen.su.domains/mc/qmcstuff.pdf |
| [20] | Owen & Rudolf. A strong law of large numbers for scrambled net integration. Abstract/theorem-scope evidence. | https://arxiv.org/abs/2002.07859 |
| [21] | NIST Dataplot. Exact Binomial Confidence Limits. | https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm |
| [22] | Riot. 2023 World Championship Rules v1.0. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/1068e920b7b4826b2c34a22ec602646ded96cb3f.pdf |
| [23] | Riot. 2024 World Championship Rules v1.0. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news/f69130a70849462c75fbd834f740f170963b173e.pdf?accountingTag=lol_esports |
| [24] | Riot. 2023 Mid Season Invitational Rules. Parent inspected§§6.2–6.3 in addition to agent’s edition-scope inspection. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/96b920d226d09d3c875711aeaac0214975636d56.pdf |
| [25] | Riot. 2024 Mid Season Invitational Rules. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/c5eb4dfed0dd2574a94a145c6ac9a23818cbf8b8.pdf |
| [26] | Riot. 2025 Mid Season Invitational Tournament Rules. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/53e33fce18c6b894749de07156d6bb518d1487fc.pdf |
| [27] | Riot. 2025 First Stand Tournament Rules. Regional club eligibility, not national teams. | https://cmsassets.rgpub.io/sanity/files/dsfx7636/news/1696c93dc7040eba9f6194668d3402dcb8b74dca.pdf?accountingTag=lol_esports |
| [28] | Roberts & Rosenthal. Football group draw probabilities and corrections. Author postprint; sequential versus uniform draws. | https://probability.ca/jeff/ftpdir/fdrawrespub.pdf |
| [29] | Gneiting, Balabdaoui & Raftery. Probabilistic forecasts, calibration and sharpness (2007). | https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf |
| [30] | Ferro. Fair scores for ensemble forecasts. Online2013/journal2014; Eq.3 finite-M Brier. | https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf |
| [31] | Cameron & Miller. A Practitioner’s Guide to Cluster-Robust Inference (2015). Regression/few-cluster caution, not a tournament-specific guarantee. | https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf |
