# Plan 07: Trustworthy general-purpose match probabilities

**Authorized:** 2026-10-01. The user requested a saved goal, a plan and execution of the complete ordered model-reliability program.  
**Run:** `model_reliability_20261001_055321`.  
**Status:** historical program complete; all conditional work executed. No A0 replacement qualified for promotion. Prospective confirmation remains `NOT_RUN`, with zero genuine future forecasts.

## Goal

Improve the odds-free sports model and establish where its probabilities are supported, rather than make uncertain predictions look safer. Deliver reproducible, causal inputs; a broader matched test of native6+12; one coherent player-plus-competition-family experiment; explicit reliability gates; and a frozen-forward decision with real recording proof. A valid outcome is rejection or insufficient evidence. No improvement or betting profit is promised.

This is a separately authorized program, not a restart of the exhausted Plan05 ten-attempt EV loop. Its historical modeling work does not depend on Plan06's multi-year collection proposal. Betting reliability, market-aware fitting and tournament-distribution certification remain separate claims.

## Invariants

- Preserve the canonical manifest, frozen annual A0, previous experiment artifacts and the external research archive. Preserve the dirty checkout and unrelated user work. No commit, push, live migration, scheduler, scraping or deployment is part of this execution.
- TRAIN uses released history only. Targets use their declared roster scenario; an actual historical first-map lineup is not an evidenced announcement. Earlier predictions must not depend on later results or roster changes.
- Existing repaired roster selection, native tensor construction, A0 inference and append-only persistence are reused. Do not create another operational model or competing benchmark.
- New rosters, legitimate cold starts, missing organization history and optional W20 remain supported. Ambiguous identities do not silently become normal forecasts. Model A/B orientation is not blue/red side.
- Preserve side symmetry and native terminal-series semantics. Do not apply BO projection twice. Odds never become sports-model inputs.
- The inspected 2024–2026 years remain historical development. Bootstrap resampling does not restore independence or certify publication/execution times.

## Work packages and proof

### A. Consistent inputs and evidence status

Use `src/analysis/ev_asof_replay.py`, `src/analysis/roster_provenance.py`, the native A0 adapter and `betting_app/services/a0_shadow_readiness.py`. Add one real entry point that constructs repaired native snapshots from declared fixtures and then performs inference/recording through existing contracts. Expose known versus assumed lineups, exact competition context, history/stat support and missing-input reasons. Evidence status is not an uncertainty interval or a claim of predictive accuracy.

Proof: real-source feature construction and native inference; announced/prior/ambiguous roster behavior; BO1/3/5, cold/new players and optional history; future perturbation and A/B reversal; exact metadata retained by the existing append-only recording path on an isolated database. No configured user database or production writes.

### B. Fixed supervised recipe across annual origins

At origins January1 of 2024, 2025 and 2026:

- TRAIN: event and effective release strictly before July1 of the preceding year.
- CAL: events from that July1 to the origin, with labels released strictly before the origin.
- Assessment: every canonical target in the following calendar year, through the frozen source endpoint.
- Three paired seeds: 20260910, 20260911, 20260912. Broad training population, batch128, no shuffle, CPU/two Torch threads.
- C0: native12. C1: native6, reset optimizer and native gate RNG, then native12. No epoch search, early stopping or post-result change. Shared non-neural experts and TRAIN-only normalizers are identical between arms at an origin; native calibration/assembly recipe is unchanged.
- Reuse the pinned 2026 C0/C1 artifacts where the complete contracts match; train the missing earlier origins. Never score a historical row using the 2025-trained candidate.

Primary replication contrast: C1 minus C0 on all canonical **2024–2025** rows, excluding the already discovered 2026 effect. Report 2026 separately and the full **11,550-row** aggregate. Annual A0 is an external reference, not the recipe-matched control. All populations, missing forecasts and fit schedules are explicit.

Proof: source/input/membership/checkpoint receipts, paired initialization and actual updates, chronological boundaries, saved-checkpoint replay for every exported probability, raw/calibrated/full components and all three context modes. Extend reusable training/evaluation code without editing the hash-pinned implementations needed to replay prior artifacts.

### C. One competition-family experiment

Fixed comparison: R0 player-only Glicko-2 versus R1 the same player state plus a learned, regularized competition-family bridge. R0 and R1 share **exact player updates**, so the experiment isolates the bridge rather than changing the rating engine too. Reuse `glicko2_core.py` and existing rating/release/affiliation machinery; no hardcoded tier or region multipliers, nationality inference, new graph or extra tier component.

The bridge uses actual historical map participants, fixed prior player logits and fractional per-player family affiliations. Affiliations come from previously released domestic evidence, with a365-day TTL and explicit conflict/unknown handling; international appearances do not overwrite them. Transfers update affiliation only after release. Queries inflate inactivity without mutating stored state. Historical substitute maps use their actual participants, not the first-map roster copied across a series.

Freeze a Gaussian-regularized additive family bridge before outcomes are scored: family prior deviation150 rating points, process deviation1 rating point per elapsed day (variance adds elapsed days), zero prior means; no tier effect or heuristic discount. Update against map outcomes with weight `1 / number_of_valid_maps_in_series`, in frozen daily release batches. The link is the player map logit plus family-offset contrast; the same Bernoulli link supplies gradients/information. This is a weighted-map, diagonal approximation, not a full joint Bayesian posterior or per-match confidence interval. Project map probabilities to series once at query; calibrate each arm on identical past CAL rows using the same fixed positive scalar temperature recipe.

Primary: calibrated R1−R0 LL on all11,550 canonical matches. Report raw outputs and international, domestic, mixed/unknown affiliation, transfer, BO and history-support profiles, with a graph-support inventory. This is a separate structural test, not an explanation assumed from LPL missingness.

Proof: independent gradient/probability oracles; identical player states; chronology, side/period-order invariance, actual substitutes, transfers, TTL, unknown/mixed affiliations, nonmutating inactivity and serialization. No heuristic legacy regional-discount result counts as this experiment.

**Conditional integration:** only if R1 passes the predeclared structural gate, compare its incremental contribution inside matched A0 using past CAL for the assembly and no assessment fitting. If R1 fails, record `NOT_TRIGGERED_GATE_FAILED`; do not force an A0 integration or tune another bridge.

### D. Reliability and promotion gates

Use existing shared metric/bootstrap definitions. No probability clipping in the new headline scores/differences; reject nonfinite/boundary forecasts explicitly. Keep scoreline likelihood separate from winner scoring.

- Paired monthly-block bootstrap,5000 draws, seed20261001. Report actual blocks, match-weighted and equal-month effects. Source-tournament-name/year groups are sensitivity proxies, not verified independent editions.
- Training gate: primary replication LL interval upper endpoint<0 and Brier point difference<=0; full-population LL must not worsen. Structural gate: primary full-population LL interval upper endpoint<0 and Brier difference<=0.
- Predeclared noninferiority guard: the upper95% paired LL difference must be below **+0.005** on BO3, BO5, major competitions and sparse-history targets (at least one source-unseen player). Insufficient coverage or fewer than six monthly blocks is `INSUFFICIENT_EVIDENCE`, not a pass. This is a chosen engineering tolerance, not a universal threshold.
- Calibration support: assess favorite-confidence bins[.5,.6,.7,.8,.9,1] and the same critical profiles, with neutral forecasts separately counted. At least200 rows and six months are required. A95% interval for mean favorite-confidence minus favorite-win-rate must lie entirely inside **±0.05** for `SUPPORTED_IN_DEVELOPMENT`. Otherwise report the observed failure or insufficient precision. This five-point probability tolerance is not the separate two-point net-EV reliability target.
- Comparative profiles use common fixed-A0 bins; candidate-own bins are calibration diagnostics, not paired populations. Keep unknown profiles and every canonical row in headline coverage.
- A successful historical candidate is only eligible for forward confirmation. None is automatically adopted, deployed or certified for betting.

### E. Frozen-forward evaluation and recording

After applying the gates, freeze the eligible candidate's exact recipe/artifact/source hashes and the permitted input contract. If no candidate qualifies, explicitly retain annual A0 as an unpromoted research reference and record why no candidate was frozen for promotion.

Exercise immutable recording using the real selected/reference inference path and the new evidence status on an isolated database. Synthetic fixtures remain labeled synthetic; replayed past matches remain historical. Genuine future predictions require available fixture, roster/source and start-time evidence. Missing inputs are recorded as missing, not replaced with invented forecasts. No future outcomes are fabricated or read as an untouched test after development has used them.

Define the later confirmation cohort and fixed evaluation checkpoints before collection, without claiming that a future observation period can finish now. Do not alter an annual artifact's year restriction to fake future-year support. Report separately: software recording proof, historical eligibility decision, actual genuine future forecasts (if admissible inputs exist), and outstanding evidence prerequisites.

## Artifacts and execution ownership

Stages: `data/{05_model_input,06_models,07_model_output,08_reporting}/model_reliability/model_reliability_20261001_055321/`. Save goal, frozen subprotocols, code/input hashes, memberships, fitted checkpoints, predictions, profiles, monthly and tournament-proxy contrasts, verification receipts and the final report. Freeze each executable subprotocol's source hashes before fitting; source changes require a new sub-run, not rewritten hashes.

Main owns this plan, neural training orchestration, shared reliability evaluation, integration and the final report. Independent input and family-rating implementations may proceed concurrently in disjoint files; validation and real experiment execution are centrally coordinated. No agent changes the canonical manifest or the hash-pinned prior neural sources.

## Completion

Each work package receives an executed result, a conditional-gate decision, or an exact externally unavailable prerequisite. Update this document, `docs/RESEARCH.md` and the handoff with observed outcomes. Preserve negative results. Do not call implementation tests financial, generalization or prospective certification.

### Executed result — 2026-10-01

| Work package | Observed result |
|---|---|
| A. Inputs and recording | Repaired fixture CLI, explicit input/evidence status, four real native historical forecasts from nine requests, five retained roster absences, twelve immutable envelopes on an isolated migrated database. Future perturbation, symmetry, empty history, rollback and idempotency passed. |
| B. Annual replication | New2024/25 paired ensembles; pinned2026 reused. Primary8,447-row C1−C0 ΔLL **+0.000154 [−0.000442,+0.000702]**, ΔBrier **+0.000084**. Improvement gate failed; no promotion of native6+12. |
| C. Family component | Same11,550 rows: R0/R1 LL **0.578667/0.574993**. ΔLL **−0.003674 [−0.007128,−0.001264]**; structural gate and four guards passed. Retain the positive component result, not an A0 replacement. |
| C. Conditional A0 integration | Executed, not deferred. Native12 / +player / +family LL **0.550351/0.550713/0.551099**. Family−base **+0.000748 [−0.001129,+0.002271]**; both primary contrasts failed, as did the unseen-player noninferiority guard. |
| D. Reliability | Complete11,550-row coverage for25 probability columns, all three native context modes and fixed profiles. R1's unseen-player calibration tolerance was not established: gap **−2.27pp [−5.94,+1.64]**. This is not proof of overconfidence. |
| E. Forward decision | Retain frozen annual A0 as an unpromoted research reference; **no new candidate frozen for promotion**. Software recording proved separately. Future design sealed; real future inputs, elapsed observation periods and outcomes remain unavailable prerequisites. |

All intervals above are95% paired monthly-block intervals with5,000 draws; the primary replication has24 blocks, full-cohort comparisons33. Equal-month and tournament-name/year sensitivities are in the report; they do not replace the prespecified match-weighted gates.

**Verification:**207,900 exported neural probabilities and3,810 native CAL probabilities replayed from real checkpoints. Independent arithmetic reproduced125 assessment metric rows,54 contrast intervals,55 calibration intervals, then five integration metric rows,21 contrast intervals and25 calibration intervals. Six integration alpha fits passed independent convex-objective/endpoint checks. **164 focused regression tests passed**, with one existing Starlette/httpx warning. The isolated database was stopped. No full application-suite, live collection, production migration or deployment was performed.

**Forward design:** three complete months for integrity only; one quality checkpoint after six months and at least2,000 scored matches; administrative close at twelve months if support is inadequate. No backdated issuance, interim tuning or relaxed annual-model year restriction. Genuine future forecasts in this execution: **0**.

- [Final sealed decision and evidence hashes](../data/08_reporting/model_reliability/model_reliability_20261001_055321/decision.json)
- [Full results, limitations and verification](../reports/noisy_data_reliability_research_20260929.md#11-executed-model-reliability-program--2026-10-01)
- [Forward design and exact external prerequisites](../data/05_model_input/model_reliability/model_reliability_20261001_055321/forward_design.json)

The original saved goal and executable source snapshots remain unchanged. Source/orientation and assessment-reporting failures were preserved as failed attempts before separately frozen repairs. Neither the canonical manifest nor frozen annual A0 nor the operational model changed. Plan05/06 remain separate; no further modeling or collection starts automatically.
