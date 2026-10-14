# Plan 05: Reliable selected-bet EV

**Created:** 2026-09-29  
**Status:** G0_BLOCKED for new fitting; all ten separately tracked attempts are complete, including competition-context roster repair and raw-to-final frozen-A0 attribution. No new fitting, scraping, deployment or autonomous-loop reactivation. The wider reliability goal remains incomplete.
**Current priority:** obtain genuine competition-unit/lineup-announcement and bookmaker-participant/observation evidence. The exact-context fallback is repaired locally, with substantial coverage loss; raw expert probability error persists. Calibration raises selected neural EV while improving overall proper scores, so removing it is not justified. No further model change or automatic attempt11 is selected.

**Next-step plan:** [Plan 06 — evidence admission after the A0 audit](./06_evidence_admission_after_a0_audit.md) A1–A2 is complete under a separately saved goal, using three subagents and an independent metadata oracle. The bounded source review selected forward acquisition readiness E; no B/C replay or H/control fit followed. This document retains the H/control/reliability specification and the unchanged 10/10 journal. Annual recipe membership receipts exist, but exact historical input/quote/roster availability is not thereby established.

**Local readiness update:** Plan06 E1–E2 is also complete: genuine frozen-A0 native parity and append-only shadow evidence were exercised on isolated local fixtures. This does not reopen G0, change the10/10 journal, recover authentic quote/roster evidence or authorize fitting/live collection. Next is the acquisition/development protocol; see Plan06 for proof and the separate pre-existing migration-default caveat.

**Saved execution goal:** Phases1–4, at most ten implementation-and-verification attempts. Attempts **1–2** audited prerequisites; **3** exported production read-only; **4** executed the A0-main-engine counterfactual; **5** compared quarter Kelly; **6** diagnosed losses and evaluated OddsPapi; **7** completed the full audit; **8** repaired replay/roster/quote contracts; **9** diagnosed remaining errors; **10** repaired competition-context fallback and traced raw experts through calibration/series projection to fixed-selection EV. H/control comparison and prospective reliability remain incomplete. Preserve the original [G0 audit](../data/08_reporting/betting_ev_reliability/ev_reliability_20260929_221258/AUDIT.md), [saved objective](../data/05_model_input/betting_ev_reliability/ev_reliability_20260929_221258/saved_goal.json) and continued [10/10 attempt journal](../data/08_reporting/betting_ev_reliability/a0_context_stages_20260930_183251/attempts.json).

## Latest completed repair and diagnosis — attempt 10

[Results and decision](../data/08_reporting/betting_ev_reliability/a0_context_stages_20260930_183251/results.json), [verification](../data/08_reporting/betting_ev_reliability/a0_context_stages_20260930_183251/verification.json) and [detailed research handoff](../docs/RESEARCH.md). The [attempt9 audit](../data/08_reporting/betting_ev_reliability/a0_error_audit_20260930_101520/REPORT.md) remains historical evidence.

- `a0_input_repair_asof_v3` uses valid announcements first, otherwise strictly prior exact competition-context roster history with existing evidenced links. No org/family-only fallback and no Cup-history deletion. Explicit scheduled-target context is retained even without a target in the results archive; archived labels are not announcement proof.
- Gen.G–HLE: HLE99.5413%→48.2421% using prior same-context players, not actual target players. Two other Cup→LCK cases, including a saved winning control, are withheld. Origins remain4626; forecasts4500→3585, primary1852→1539. Same1539LL.572587→.569562, paired interval crosses0. Only8retained primary lineups change;1531unchanged probabilities match exactly.
- The fixed498selections retain387forecasts/153wins. On those same387, EV error30.0723→29.7832pp; remaining95%interval[10.4485,54.2349]. Most of the difference from old49839.39pp is coverage, not repair. Separate re-selection has387bets/152wins:385origin/side pairs are shared with the retained set, with2replacements. Near5–10%88bets still fails the reliability target.
- Raw neural IID/rating/MLP already overstate EV25.77/33.40/27.70pp on the fixed387. Frozen neural scale0.9093 softens logits; scale plus beta raises error to30.10pp while improving full1539LL.573972→.569517. Rating calibration reduces EV error. No specific omitted sports factor is established; no immediate model change is justified.
- 42targeted tests passed; both native traces reproduce all3modes exactly. Independent checks cover7170source-side rosters,208proper-score rows,220paired contrasts,10956selection/stage rows and2204monthly/edition-proxy intervals. Raw-map BO exclusions are separate from missing forecasts. No new full-suite or financial/prospective certification. Counter10/10; no automatic attempt11. The original H/control/prospective tasks below remain incomplete.

## Goal

When the system displays **+5% net EV**, it must be an empirically supported estimate of average after-tax return for comparable selected bets—not just an optimistic model/price disagreement.

Keep the odds-free Causal A0 sports model fixed. Test one small, separately identified market-aware betting-probability model. Validate the actual selection process, not only global probability calibration. A correct outcome is also **no demonstrated edge**; do not force a profitable-looking answer or a minimum number of bets.

The user's reported EV overstatement is the starting observation. This plan measures its mechanism and extent; it does not rerun rejected experiments to reconfirm that observation.

### Measurable target

For selected bet i, let O be its executable decimal odds, t the turnover-tax fraction, p the probability used for the displayed estimate, and Y the selected side's win indicator:

- Effective payout: `cO = (1 - t) * O`.
- Predicted net return: `E = cO * p - 1`.
- Realized unit return: `R = cO * Y - 1`.
- **EV overstatement:** `B = mean(E - R) = mean(cO * (p - Y))`. Positive means optimistic.
- Also report `mean(p - Y)`, proper probability scores, selected-bet count and selection coverage.

If a legacy displayed field is post-processed, also measure `mean(E_shown - R)` directly; do not pretend it obeys the probability identity above. H's new estimate uses the formula-based EV without an additional hidden discount. The binary-return formula applies to normally settled bets; voids/refunds and unmatured outcomes require explicit separate accounting and stay visible in the opportunity ledger.

**Target for a reliability claim:** the 95% interval for B lies entirely within **[-0.02, +0.02]**, both for the complete selected portfolio and the predeclared near-threshold band **0.05 < E <= 0.10**. This is a proposed engineering tolerance of **two percentage points of net return**, not a universal statistical standard. Compare realized return with the band's mean estimated EV, not its 5% lower threshold.

**Separate betting-value target:** positive out-of-time net return with its 95% lower interval bound above zero, plus evidence of added value over the declared market-only policy. Good calibration alone does not establish an edge; positive realized ROI alone does not establish accurate EV.

These are group/policy claims, never a guarantee about one match or every month. Inadequate precision, insufficient coverage or zero selected bets means **INCONCLUSIVE / NOT_ESTIMABLE**, not success.

### Precision and feasibility warning

At odds 2.50 and 12% turnover tax, +5% net EV corresponds to p = 47.73%; break-even is 45.45%. A 2.27 pp probability error erases the entire edge. A gross +5% estimate instead becomes **-7.6% net** under that tax contract.

Even an illustrative IID normal approximation with identical 2.50 odds and true 5% EV needs about **11,600 selected bets** for a 95% return interval with two-percentage-point half-width. Real dependence, varying odds and drift change that requirement. This is not a guaranteed sample size or a reason to loosen the target after observing results. A feasibility/precision assessment is required before promising prospective certification; a small historical portfolio can support diagnosis without certifying a 5% edge.

## Scope and explicit non-goals

- Preserve the canonical manifest, frozen A0 artifacts, raw source and research archive. H is a market-conditioned candidate, not a replacement for the full-coverage sports benchmark.
- Identify the actual model behind the application's displayed EV. Research A0, operational EXP081 and existing hybrids are different identities. No live-server identity is inferred from a dated report.
- No new neural training, target changes, history reconstruction, tier weighting, arbitrary league exceptions or seed/bootstrap ensembles in this comparison.
- No fixed uncertainty haircut, entropy discount or larger EV threshold as an assumed repair. No high-loss deletion or invented per-bet confidence bound.
- No Kelly optimization, bankroll simulation, real wagering, scraping, scheduler changes or deployment in the first experiment. Start with flat unit returns.
- This is the current execution-design reference. It replaces the older dispersion/P0–P5/Kelly sweep as the priority, not as a rewrite of historical evidence. It operationalizes the [EV-specific research](../reports/ev_overestimation_targets_and_plan_20260927.md); its tighter error target supersedes that report's proposed five-point EV tolerance. A 48-hour horizon is not imposed on records captured at other decision times.
- [IDEA-026 history-stat reconstruction](../docs/future_ideas.md#idea-026--structured-observed-stat-reconstruction-for-reliable-a0-forecasts) was separately authorized and completed on2026-10-01, outside this EV comparison. Reconstruction failed; native6+12 yielded a secondary historical-development signal. This does not clear G0 or authorize H/control fitting or deployment.

## Shared contracts

### Two distinct evaluation questions

1. **Displayed-signal audit:** preserve the exact historical UI/API probability, EV field, tax, eligibility configuration and artifact identity. Reproduce the actual rule where its inputs exist. Missing lineage is a stated limitation, not permission to substitute A0.
2. **Controlled A0 experiment:** use a fixed scientific selection policy across all forecast recipes to isolate the probability change. Report this separately from the application's additional gates; do not call the two portfolios equivalent.

### Decision-time opportunity ledger

One opportunity records match and side identities, bookmaker/quote IDs, both aligned prices, decision time, source/release and feature cutoffs, actual forecast creation, model/calibrator versions and hashes, market-consensus inputs, roster policy/evidence, start-time vintage, tax, displayed/computed EV fields, pass/select reason, and eventual result-release/settlement state. Retain every opportunity, including unavailable forecasts and passes.

- For recorded decisions, every forecast and quote must already exist at the decision; no forced ordering between their arrivals. Every relevant feature/label must satisfy its own origin and release constraint. The decision must precede the verified match start.
- Historical reconstruction uses separate `simulated_origin` and real `replay_created_at`; never backdate a newly created forecast. Daily release/actual-roster assumptions remain scenarios, not publication/announcement evidence.
- If an archived A0 forecast used information unavailable at the simulated origin, regenerate a new scenario forecast from an artifact permitted at that origin and strictly prior/evidenced inputs, including the roster. Keeping A0 fixed means preserving its recipe/weights, not reusing a causally invalid probability. Preserve the original archive and record the new replay's actual creation time; this does not turn the replay into a recorded production decision.
- Later CLOSE is permitted only for a clearly labeled post-decision CLV diagnostic, never an earlier input or gate. Unknown actual starts, execution availability and quote freshness remain explicit limitations/exclusions under the frozen policy.
- Same-cutoff consensus: align sides once, remove margin within each bookmaker, one selected quote per bookmaker, then equal-book averaging. Require at least two distinct books. A leave-executed-book-out sensitivity requires at least two remaining books; otherwise unavailable. Consensus is not an executable price.
- Primary research replay uses one common, model-independent opportunity per fixture. Prefer recorded decision times; absent those, a separately labeled T−1h reconstruction is the initial scenario. Freeze quote freshness and schedule-vintage rules from the source contract before outcome evaluation. Do not mix horizons or duplicate the same outcome across snapshots as independent evidence.
- Partition counts and eligibility are outputs of the audit, not promises of 4,374 or 1,645 valid rows from older plans. Do not revive invalid `run005` rankings or the known timing-invalid 48-hour reconstruction.

### Fixed scientific selection policy

For the first modeling contrast: flat **one-unit** stakes, net `EV > 0.05`, tax `0.12`; report tax `0` only as a separate fixed sensitivity, never as evidence for the taxed policy. Select at most one side/quote per fixture using the existing deterministic best-EV rule; freeze its complete quote/tie policy. No new odds cap, EV ceiling, uncertainty discount, dynamic hurdle or threshold search in this scientific arm. Label it as such, not as the operational gate configuration.

Keep the actual operational-rule replay as a separate audit arm with its original configuration. Do not feed a final closing price into it at an earlier origin. If exact reproduction requires unavailable inputs, label it unavailable rather than silently dropping the rule.

Freeze diagnostic EV bands `(5%,10%]`, `(10%,20%]`, `>20%`; odds bands `<1.80`, `[1.80,2.50]`, `(2.50,3.50]`, `>3.50`; selected/unselected, model–market disagreement, BO, competition and time. Subgroup support/intervals are required. No tuning against the final subgroup chart.

### Candidate and controls

Let `p_s` be native annual A0's complete `p_full`, not a raw neural component, and q the same-cutoff no-vig consensus.

| ID | Recipe | Role |
|---|---|---|
| S | Frozen A0 | Sports reference |
| S-cal | Positive-slope symmetric recalibration of S | Detect a calibration-only explanation |
| M | Raw q | Transparent market reference |
| M-cal | Positive-slope symmetric recalibration of q | Primary predictive comparator |
| M-shape | Side-symmetric monotone two-slope market calibration, fixed knot `logit(0.8)` | Check whether apparent sports gain is merely nonlinear market calibration |
| Fixed50 | Fixed 50/50 logit blend | Historical simple-blend reference |
| H | `sigmoid(a * logit(p_s) + b * logit(q))`, `a,b >= 0` | Only new sports-plus-market candidate |

H has no arbitrary A/B intercept. It is regularized toward `(a,b)=(0,1)` using mean Bernoulli log loss; this is a complexity prior, not a claim that the market is correct. Coefficients need not sum to one. Preserve A/B complementarity and calibrate the complete candidate, not one component under another model's calibration.

Use the existing report's bounded penalty grid `{0.01, 0.1, 1.0}`, with identity-centered penalties for fitted controls and equal tuning opportunities. Select by past-only log loss; ties prefer stronger regularization. Freeze the transform/scoring probability policy and use it consistently in all metrics. Do not round or clip only reported losses to hide extreme errors.

Fit on **all eligible prior fixtures**, not only selected bets. Full A0 predictions used for fitting must be out of time for weights, calibrators and mixture. In-sample CAL predictions are not OOF. TRAIN/CAL/development dates and memberships must be frozen after the availability audit, before fitting; no favorable chronological split search. Start with a single closed head-fit/evaluation split, not another neural walk-forward training campaign. If eligible historical support is insufficient, report `INSUFFICIENT_FIT_SUPPORT` and collect proper shadow data rather than relaxing chronology. All previously inspected 2024–2026 outcomes remain development data; the confirmation period starts only after recipe freeze.

## Phased work with deliverable proof

### Phase 1 — Establish what the displayed 5% means

- [x] **Trace and freeze the current displayed EV contract.** Follow the observed UI/API field to its probability, quote, tax, rounding, eligibility rule and artifact loader. Distinguish `ev_net`, conservative-score EV and any post-processed `ev_calibrated`; retain unavailable historical identity explicitly.
  - **Targets:** `betting_app/api/routers/matches.py`, its actual UI/schema callers, `betting_app/services/bet_qualification_service.py`, `src/analysis/unified_evaluation_engine.py`.
  - **Proof:** one actual response/fixture reconstructed end to end; independent gross/net arithmetic; no assertion that a research artifact is the serving model.
  - **Known code risk:** `UnifiedBettingEngine.qualify_quote` contains a legacy 0.73 EV multiplier, odds/entropy adjustments and clipping. Trace whether it affects the chosen surface; do not presume it does. Such post-processing is not itself reliability evidence and must not be stacked onto H's formula-based EV unnoticed.
  - **Current evidence:** the [observed production contract](../data/05_model_input/betting_ev_reliability/production_provenance_20260930_001359/observed_ev_contract.json) pins the live API identity, captured client/server code and on-disk EXP081 artifact hash. Ten populated EV values in an actual six-fixture board response independently reproduce from saved sidewise scores, odds and 12% tax. No qualified recommendation appeared in that sample. Cached/historical loaded bytes and the user's specific historical display remain unverified; this is not EV reliability certification.
- [ ] **Build the immutable opportunity and availability ledger.** Reuse current identity/quote selection, keeping separate recorded and reconstructed populations, missingness and rejection reasons.
  - **Targets:** `src/analysis/production_market_benchmark.py`, `scripts/run_unified_betting_evaluation.py`; stage-03/05 artifacts.
  - **Proof:** unique opportunity/quote alignment, complete denominator accounting, timestamp/fit-release checks, and perturbing future results/quotes/rosters cannot change earlier snapshots. Produce actual date ranges, eligible counts and TRAIN/CAL/development membership manifests.
  - **Current evidence:** complete admission ledgers retain 11,550 canonical matches and 3,302 provider fixtures at three separate horizons, with explicit exclusions. G0 admits none; no causal TRAIN/CAL/evaluation memberships or candidate future-perturbation proof are claimed.

**Gate G0:** only verified available information enters fitting or policy claims. A timing-invalid or identity-invalid cohort is rejected, not repaired by selecting a favorable intersection.

### Phase 2 — Quantify selection-conditional EV error

- [ ] **Add the fixed-policy reliability report.** Show mean predicted EV, mean unit return, B, probability residual, LL/Brier, count and coverage for the full selection and fixed bands. Preserve non-selected opportunities and winning/losing controls.
  - **Targets:** existing production-market evaluation functions and runner; extend rather than copy a second benchmark.
  - **Proof:** independent per-row payoff/side oracle; correct zero-bet undefined statistics; positive/negative bias direction; variable-odds cases where zero mean probability error does not imply zero EV error. Do not claim the source of bias from an association alone.
- [ ] **Freeze the statistical and precision protocol.** At least 5,000 paired monthly-block resamples, seed **20260929**, sampling the entire pre-selection opportunity universe; preserve no-bet months. Report match/bet-weighted and equal-month estimands separately. Record undefined-replicate fractions instead of discarding or filling them. Add edition-level sensitivity where verified; source-name proxies must be named as proxies.
  - **Proof:** cohort/block counts, deterministic arithmetic and realistic precision/support assessment. Bootstrap sign fractions are not classical p-values. Few blocks or unsupported dependence invalidate certification, even with a narrow computed interval. Infeasible precision is reported before an adoption claim.

**Deliverable:** a measured conditional-error baseline and frozen protocol, not a fitted correction or profitable-policy claim.

### Phase 3 — Fit the one bounded betting-probability candidate

- [ ] **Implement H and the declared controls on identical eligible inputs.** Reuse symmetric calibration conventions; save coefficients, transforms, fit memberships, penalties, input hashes and complete model identities. Do not change A0.
  - **Targets:** `src/analysis/thesis_models.py` and the existing unified runner/evaluation path; stage-06 model and stage-07 prediction artifacts.
  - **Proof:** orientation symmetry, finite valid predictions, empty/missing-market coverage behavior, cold-history contract, exact artifact replay, and no future labels entering fit/tuning. Missing q or fit support is unavailable, not a hidden fallback.
- [ ] **Produce three explicitly separate comparisons.** (a) All common eligible fixtures: probability scores. (b) Fixed S-selected IDs/sides: conditional EV/probability errors. (c) Each model's own selected portfolio: policy outcomes.
  - **Proof:** fixed IDs/prices/stakes imply identical realized PnL across forecasts; own-policy comparisons may change membership. Changing p downward must not pass merely because the signed overstatement becomes smaller. Report proper scores, two-sided error and coverage.

**Gate G1 — historical progression only:** H must improve paired LL against M-cal, then M-shape, with negative 95% intervals and non-worse Brier point estimates; report all intervals and S comparisons. Use the declared sequential order, not whichever control H happens to beat. Unresolved comparisons mean `INCONCLUSIVE`; no added sports value means do not promote H as an A0 improvement. Historical success is only `DEVELOPMENT_CANDIDATE`.

### Phase 4 — Evaluate betting reliability and value separately

- [ ] **Apply the error and policy gates without another parameter search.** Compute the goal's two-sided B intervals on each candidate's own selection, including near-threshold bets, and the fixed-reference diagnostics. Report mean profit per eligible opportunity as well as conditional ROI and coverage.
  - **Proof:** independently recomputed returns, no hidden dropped forecasts/bets, correct tax/void/maturity treatment, and identical pre-selection universes for paired policy contrasts. Report all loss periods and predeclared slices.

**Gate G2 — historical decision evidence:** distinguish (i) reduced diagnostic error, (ii) meeting the ±2 pp reliability target, and (iii) positive net policy value. For a betting-value claim require positive net profit per eligible opportunity and a positive paired increment over M-cal, with both lower interval bounds above zero, plus estimable positive selected-bet ROI. No-bet policies cannot pass. Historical uncertainty or insufficient near-threshold support remains inconclusive; do not tune a cutoff to escape it.

### Phase 5 — Freeze and collect genuine forward shadow evidence

- [ ] **Validate the live artifact/adapter and freeze the recipe.** A0 has no validated live adapter in the current handoff. Implement/verify that prerequisite before claiming an A0 shadow trial; do not substitute operational EXP081. A0's as-of inputs, roster policy, BO semantics, optional W20 and nonmutating queries must match the declared research contract.
  - **Targets:** existing `betting_app/core/models/` and `betting_app/services/upcoming_inference_service.py` paths; use existing persistence where possible. Any necessary schema change requires a separate migration review and isolated PostgreSQL/Timescale verification.
  - **Proof:** artifact parity, side symmetry/cold start, input-release and schedule checks, and timestamped forecast/quote records written before outcomes. No automatic live rollout is authorized by this planning request.
- [ ] **Run the preregistered shadow observation period.** Default confirmation window: twelve complete UTC calendar months starting after adapter validation and protocol freeze; record exact start/end and outcome-maturity rules before the first included result. Freeze the deployed coefficients; no outcome-driven refits, threshold changes or early success stop. Operational integrity checks may continue, but evaluated outcomes are not used for model development.
  - **Proof:** append-only opportunity coverage, unchanged protocol/model hashes, untouched outcomes until the declared assessment, and G0/G1/G2 reporting on genuinely future records. Twelve months does not guarantee adequate precision; underpowered results stay inconclusive, without an opportunistic extension until profit appears.

### Phase 6 — Decide; do not automatically deploy

- [ ] **Publish an adoption or rejection decision.** Outcomes: `REJECTED`, `INCONCLUSIVE`, `NO_ADDED_SPORTS_VALUE`, or `FORWARD_SUPPORTED` for the specified cohort/policy. Record uncertainty, regime coverage and unsupported contexts.
  - **Proof:** a complete decision report tracing each gate to artifacts, including failures. Improvement in global ECE, attractive CLV, lower drawdown or fewer bets is not a substitute for the goal.
- [ ] **Only if supported and separately authorized, plan the operational cutover.** Replace the chosen EV field's old calculation path cleanly, migrate every caller and label estimates without presenting group intervals as per-bet guarantees. Remove superseded candidate-path shims/heuristics; keep historical evidence. Bankroll sizing is a later, separate exercise with reserved overlapping stakes, free capital, real settlement availability and atomic/idempotent accounting.
  - **Proof:** affected API/UI behavior and existing regression suites; browser checks for a UI change; isolated database verification for persistence changes; normal review/Git/deployment/health sequence. No automatic wagers.

## Verification commands and artifact contract

During implementation, extend existing behavior-focused tests and exercise the actual runner on a frozen, isolated fixture before the eligible historical data. Existing test paths include:

```bash
.venv/bin/python -m pytest -q betting_app/tests/test_production_market_identity.py betting_app/tests/test_production_market_benchmark.py betting_app/tests/test_production_a0_calibration.py betting_app/tests/test_thesis_models.py betting_app/tests/test_unified_evaluation_engine.py
```

The five suites above plus `betting_app/tests/test_betting_recommendation.py` ran during the prerequisite audit: **57 passed**, one existing dependency deprecation warning. Both database variables used non-serving loopback port 1, not the configured database. Exact argv is in the run receipt. The existing CLI also passed a two-fixture synthetic flat-stake smoke; it did not evaluate a historical cohort or implement H. Never present a proposed/nonexistent CLI as executed proof.

Use one new run ID without overwriting earlier output:

| Stage | Planned artifacts |
|---|---|
| `data/03_primary/betting_ev_reliability/<run_id>/` | Opportunity ledger, eligibility/exclusion and identity/timing audit |
| `data/05_model_input/betting_ev_reliability/<run_id>/` | Frozen protocol, partitions/membership hashes, precision plan, exact policy/EV contract |
| `data/06_models/betting_ev_reliability/<run_id>/` | H/control coefficients, transforms, calibration and provenance |
| `data/07_model_output/betting_ev_reliability/<run_id>/` | All fixture forecasts, availability and selections, not just profitable bets |
| `data/08_reporting/betting_ev_reliability/<run_id>/` | Proper scores, EV reliability bands, fixed/own-policy comparisons, monthly/edition sensitivity, receipts and decision report |

Reuse `src/analysis/production_market_benchmark.py`, `src/analysis/unified_evaluation_engine.py`, `scripts/run_unified_betting_evaluation.py` and existing calibration conventions. No competing uncertainty-betting script or dated-directory production implementation. Resolve the external archive through the existing research-root resolver; never hardcode a personal path.

## Read-only production evidence — 2026-09-30

The authorized inspection/export is complete. [Audit and G0 decision](../data/08_reporting/betting_ev_reliability/production_provenance_20260930_001359/audit.json), [export verification](../data/08_reporting/betting_ev_reliability/production_provenance_20260930_001359/export_verification.json), and [EV arithmetic proof](../data/08_reporting/betting_ev_reliability/production_provenance_20260930_001359/ev_surface_verification.json) are local evidence, not a model comparison.

- Primary PostgreSQL snapshot: **2026-09-30 00:36:58.305864 UTC**, enforced `READ ONLY`, `REPEATABLE READ`; **739,113 projected records in 22 datasets**. It includes **373,618 predictions across 1,592 matches**, **234,434 non-live match-winner quotes**, **95,873 signals**, **1,595 current feature rows**, **6,375 current roster-player rows**, and seven feature backups. These are repeated records, not independent bets.
- Safe GETs for health, active identity and the board returned HTTP 200. The live configured model is **`Hybrid-Operational-Market:exp081-siamese-series-v1-a0.50-t1.00`**, not A0. The current on-disk EXP081 artifact SHA-256 is `546d19549965a5dec87a466a56ccec1513bee6a9c164b491db04a617b05998a2`; this does not prove historical or cached in-memory bytes.
- **4,353** predictions explicitly identify backfills; another **408** identify a backtest. **25** backfilled rows have `predicted_at` later than the export, extending to 2027. Do not interpret all stored prediction clocks as real issuance times.
- **9,438** signals persist mean `model_prob` with EV calculated from a different, saved sidewise lower score. Every such EV reconciles with that score. The fields must not be treated as one probability/EV pair or attributed to the separate CLI's `0.73` multiplier.
- A hash-matched supplement preserves **58,233** legacy diagnostic records. Eight prediction backups recover **13,859** unique base IDs compatible with **35,732** child forecasts; **4,738** child references remain unresolved in the inspected set. Targeted quote recovery retains **1,664** rows and finds **57** compatible identity/price candidates among **4,163** problematic signal references; side alignment and decision-time availability still require verification. No records were repaired.
- Quote payloads recover **43,786** `event.start_at` fields and **41,761** raw .NET `event.date` fields. This is useful schedule-vintage evidence, not proof of actual kickoff or complete schedule coverage. Current features, rosters and registry records remain mutable; signal decision timestamps and historical rendered payloads are absent from the inspected contract.
- All export counts, row hashes, completion footers and field-exclusion checks passed; the local derived SQLite index passed integrity checking. No outcomes, stakes, wallets, accounts or credentials were exported or scored. No server mutation commands, write API routes, scraper/scheduler triggers, migrations, restarts, model fits or deployments were issued.

Raw exports are under `data/01_raw/betting_ev_reliability/production_provenance_20260930_001359/`; the read-only query recipes and contract are under stage 05, and the local query index under stage 02. Supplements have their own capture times; they are not silently presented as the primary snapshot.

## Immediate next deliverable

**Resolve G0 before fitting.** The production export and separately requested prior-known-roster A0 counterfactual are now complete; do not wait for either again. Native A0 inference was reexecuted with its real creation time and explicit daily-history/settlement scenarios. This is not certification of historical issuance, execution or a causal fitting cohort. The original admission ledgers remain unchanged. H, controls, prospective reliability and the autonomous loop remain unstarted.

### Completed separate request: A0 as main betting engine

Run `a0_main_engine_20260930_013337`: [results](../data/08_reporting/betting_ev_reliability/a0_main_engine_20260930_013337/results.json), [financial oracle](../data/08_reporting/betting_ev_reliability/a0_main_engine_20260930_013337/financial_verification.json), [quote oracle](../data/08_reporting/betting_ev_reliability/a0_main_engine_20260930_013337/quote_clock_verification.json), and [research interpretation](../docs/RESEARCH.md).

This request authorizes a separate flat-stake bankroll diagnostic, not the H experiment or a relaxation of its two-book/G0 requirements. Genuine frozen annual A0 generated1,644 available prior-roster forecasts on1,645 origins /514 fixtures (29May–8Sep2026), matching the prior replay within1.11e−16. OPEN513/514,24h379/379 and48h343/343 were eligible for scoring; all409CLOSE origins equal start and are excluded, not artificially shifted.

With12%tax,100 flat stakes and10,000 initial capital, the current qualification function applied to A0 mean probabilities selected56 OPEN bets: **−1,352.24 PnL,−24.15%ROI**, versus raw EV>5%:136 bets,−12.62%. Filtered24h:47 bets,−13.18%; filtered48h:36 bets,+2.54%, with95% monthly interval[−84.78%,+29.35%]. Do not select timing after these results. No invented conservative probability or unavailable rating-disagreement guard was applied; unknown historical board visibility prevents an exact UI replay. Settlement is an explicit daily-release/start+12h scenario with overlapping stakes reserved.

OPEN mean predicted EV25.92% versus realized−24.15% supports an overstatement diagnosis for this selected portfolio, not a new corrected probability. The narrow(5%,10%] band has only8 bets, estimated7.86%, realized5.05%, and a very wide error interval; it does not certify the±2pp target. Current/original quote exports give identical selected sides, prices and financial returns, despite two small consensus changes. They are not independent replication. Independent arithmetic checked960 ledger rows,12 portfolios and6,864 per-book quotes. All2026 outcomes remain development-exposed. **Do not adopt A0 as a live EV engine from this evidence; no fitting or deployment occurred.**

### Completed follow-up: quarter Kelly on the same A0 bets

Run `a0_quarter_kelly_20260930_024021`: [results](../data/08_reporting/betting_ev_reliability/a0_quarter_kelly_20260930_024021/results.json), [independent verification](../data/08_reporting/betting_ev_reliability/a0_quarter_kelly_20260930_024021/verification.json).
The user explicitly authorized this staking comparison; it does not restart the old uncertainty grid or authorize fitting/deployment. Quarter Kelly uses a 2.5% cap on free capital, initial10,000 and12% tax, keeping all selected IDs/sides/probabilities/quotes and settlement scenarios fixed. No fresh inference or model fit was needed.

Core filters OPEN/24h/48h retain56/47/36 bets: quarter-Kelly PnL **−1,744.26 / −1,638.25 / −1,208.38**, ROI **−15.91% / −18.83% / −20.93%**, max drawdown **28.25% / 27.72% / 15.82%**. Each loses more capital than its flat100 control. Raw EV>5% also loses at all horizons. OPEN's higher point ROI is not a supported improvement: paired fixed-ledger monthly ΔROI **+8.24pp [−1.65,+28.29]**; this is not a bootstrap of capital paths. Mean OPEN stake195.82 and40/56cap hits show why it is not equal-risk or uncapped-Kelly evidence.

Independent arithmetic verified480 bets across6 portfolios; perturbing future outcomes preserved305 earlier stakes. Original artifacts and financial controls remain unchanged. The selected-bet probability/EV reliability problem is not repaired by this staking rule.

### Completed follow-up: loss diagnosis and OddsPapi

Run `a0_loss_oddspapi_20260930_030315`: [loss diagnosis](../data/08_reporting/betting_ev_reliability/a0_loss_oddspapi_20260930_030315/loss_diagnosis/loss_diagnosis.json), [OddsPapi financial comparison](../data/08_reporting/betting_ev_reliability/a0_loss_oddspapi_20260930_030315/oddspapi_financial_results.json), [matched proper scores](../data/08_reporting/betting_ev_reliability/a0_loss_oddspapi_20260930_030315/oddspapi_probability_results.json).

Original56 filtered OPEN bets: A0 expected32.82wins, market21.13, actual23; A0 overprediction17.54pp[2.53,27.74] under five-month bootstrap. Same bets still lose13.80%before tax. Removing odds≥5 leaves all six original portfolios negative. No checked side/BO/payout bug; alternative roster/history scenarios are diagnostic, not a causal roster ablation.

At unchanged native origins, OddsPapi covers122OPEN/32124h/27748h opportunities; core filters select19/37/34bets. FlatROI **−27.77%/−17.11%/−9.02%**; capped quarter-KellyROI **−12.64%/−33.31%/−18.07%**. Both price sources also compared on identical opportunity IDs. Raw policies lose too; no late quotes fill missing OPEN. Independent raw-state checks cover2658pairs and7368absences.

Broader OddsPapi probability scores use the separate canonical target-roster scenario: T−1h/prior-day Pinnacle n1632, A0LL.575282 versus market.566077, paired monthly ΔCI[−.003017,+.021820]. These results diagnose selected-bet overstatement, not universal overconfidence. No new API calls, fitting, inference, policy tuning or deployment; the original reliability target remains unmet.

### Completed follow-up: full audit of why A0 underperforms

Run `a0_full_audit_20260930_044702`: [full report](../data/08_reporting/betting_ev_reliability/a0_full_audit_20260930_044702/REPORT.md), [independent verification](../data/08_reporting/betting_ev_reliability/a0_full_audit_20260930_044702/verification.json). All3,302provider fixtures remain in the inventory. Native inference generated9,252prior/target rows and4,626separate aligned-prior rows without fitting.

**Confirmed source defect:** prior replay can combine ID-sorted raw player features with differently ordered history. Permuting history/mask/champions restores exact same-player-set parity on3,431origins. This is a diagnostic correction, **not a source repair**; `src/analysis/ev_asof_replay.py` remains unchanged. Earlier no-checked-bug findings had narrower scope.

Same1,930freshT−1h origins: LL prior.593640→aligned.592363→actual-lineup oracle.568127; market.565863. Alignment-only Δ−.001277 has CI crossing0. The remaining lineup gain is concentrated in61unknown-roster cases, including47with same-name history under different team IDs;13one-cold-side cases have alignedLL2.303920. Current lineup announcements remain unproven; do not merge academy/main teams or reuse old rosters blindly.

Eighteen price-only contradictory-quote fixtures change corrected-prior raw flatROI from+12.04%to+0.13%; retained540bets predict44.86%meanEV and270.27wins but achieve198wins. Near-target(5%,10%]EV:106bets, predicted7.08%, realized−8.51%, overstatement15.59pp[2.76,33.09]. QuarterKelly on540bets:−2.00%ROI/maxDD68.62%. OriginalOPENfilters after alignment:59bets/−26.32%; actual-lineup oracle51/−7.31%. No reliable/executable edge established.

Independent controls verified2,308financial cases (2,126nonempty and182empty),960flat/480Kelly original rows,11,973quote pairs/31,731absences,7,649source hashes and22unchanged protocol sources. Full raw-future-history perturbation and prospective/tournament tests are not claimed. **Next:** versioned replay alignment repair, temporal roster/source-ID and bookmaker participant provenance, then G0 admission and the separately specified H/control/shadow. Do not restart uncertainty/temperature/stake grids.

### Completed follow-up: versioned input repair and reevaluation

Run `a0_input_repair_20260930_074319`: [report](../data/08_reporting/betting_ev_reliability/a0_input_repair_20260930_074319/REPORT.md), [verification](../data/08_reporting/betting_ev_reliability/a0_input_repair_20260930_074319/verification.json). This supersedes the unrepaired source state in attempt7, not its historical evidence.

- Replay `a0_input_repair_asof_v2` aligns raw/history tensors using the same canonical string player-ID order and separates scenario decision clocks from actual generation. **4,500/4,626** origins available;123unknown/incomplete and3overlapping lineups withheld. Retained full-model forecasts match the earlier aligned control within1.26e−9.
- Explicit time-bounded roster links/announcements reject ambiguous, late, retracted or incomplete evidence; no name-only merge. Links extend roster lookup, not all organization history. No qualifying historical evidence was found, so unknown lineups remain unavailable.
- Shared quote parser preserves source participants/states and rejects latest invalid states or quotes at/after earliest known start. Explicit per-fixture/book/market/outcome bindings are required for local Pinnacle persistence; the ordinary caller supplies none. Actual observation and source quote clocks remain separate. No deployed change or old-row recertification.
- **43,704** quote origins, **11,973** matching active pairs, **31,731** absences, **7,649** raw hashes; zero verified bookmaker bindings. Complete3,302fixture inventory/66,040native opportunities and2,472parent rows retained. **660** financial cases, **42,030** independently checked Decimal payoffs; all200strict-execution cases empty/undefined. **272** focused tests passed on isolated PostgreSQL/Timescale.
- T−1h/best-price/frozen q25 exclusion: raw498bets predict42.62%EV/243.70wins, achieve186wins and+3.23%flatROI[−15.71%,+20.64%]; quarterKelly+0.20%/maxDD67.01%. Current filters301bets/−0.77%flatROI. Near-target104bets predict7.05%, return−7.81%, overstatement14.86pp[2.28,32.54]. No ±2pp reliability certification.
- Same1,852origin LL: original.580103→repaired.578901, market.565614; repairΔCI crosses0. Do not count exclusion of unknown rosters as model improvement. Frozen model/calibrator, canonical manifest and previous journals remain unchanged.

**Remaining gate:** genuine historical evidence, validated live A0/decision contract and separately authorized prospective shadow. H/control fits remain NOT_RUN. Attempt8 completes these requested repairs, not the wider EV-reliability goal or autonomous loop.

### Plan06 E3 continuation — protocol frozen, collection blocked

The [E3 protocol](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/protocol.json) and [decision](../data/08_reporting/betting_ev_reliability/forward_protocol_20260930_234022/decision.json) now specify source/access, one T−1h opportunity,300s quote freshness, complete missingness accounting and separate TRAIN/CAL/development windows. TRAIN2026-11-01→2027-11-01; CAL2027-12-01→2028-06-01; DEVELOPMENT2028-07-01→2029-07-01, all half-open UTC intervals with label locks/maturity buffers defined in [Plan06](./06_evidence_admission_after_a0_audit.md). These are planned windows, not collected memberships or authorization for a32-month campaign.

The one-off offline exercise passed79 checks, verified9 saved envelopes/72 tensors and admitted none of the synthetic fixtures. Eight source contracts have no verified live access; the current A0 adapter explicitly rejects2027 cutoffs. Capture/clock/binding/start/coverage, native tensor, future-valid fixed-artifact and no-forecast-ledger prerequisites remain. Exact EV ties follow the existing selector's side-A rule. H/controls, tax, flat stakes, reliability/value gates and bootstrap seed remain as specified above; no fitting or financial evaluation ran. Failure to meet the2026-10-31 readiness deadline expires this protocol; no automatic new dates. E4 confirmation is separately authorized only after recipe freeze and cannot reuse development observations.

## Evidence and planning status

- [Current research status](../docs/RESEARCH.md).
- [EV overestimation, targets and controlled market-aware proposal](../reports/ev_overestimation_targets_and_plan_20260927.md).
- [Noise/reliability research and subgroup confounding](../reports/noisy_data_reliability_research_20260929.md).
- [Application identity and live-adapter limitations](../betting_app/README.md).
- [Canonical research manifest](../conf/base/research_benchmark.json).

Original planning verification checked references and illustrative EV/precision arithmetic without executing a hypothesis. Its original text is preserved in `data/05_model_input/betting_ev_reliability/ev_reliability_20260929_221258/plan_at_origin.txt`. The saved offline goal produced the G0 audit, isolated interface/CLI proofs and targeted test receipt; the separately authorized third attempt produced the production evidence above. The canonical manifest, original attempt journal and saved objective retain their hashes. No model weights, research archive or application code were changed by the inspection.
