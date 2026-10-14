# Plan 06: Evidence admission after the A0 audit

**Created:** 2026-09-30  
**Status (updated 2026-10-02): A1–A2, local E1–E2 and E3 protocol COMPLETE; subsequent Plan09 local capture/ledger/fixed-artifact proof COMPLETE; acquisition activation BLOCKED.** Source access, live collection, B/C historical evidence recovery, D fitting and E4 confirmation remain unexecuted. The previous journal remains **10/10**; frozen admission rules/dates/source contracts are unchanged.

## Completed A1–A2 decision

[Saved goal](../data/05_model_input/betting_ev_reliability/evidence_feasibility_20260930_215718/saved_goal.json), [decision](../data/08_reporting/betting_ev_reliability/evidence_feasibility_20260930_215718/decision.json), [evidence matrix](../data/03_primary/betting_ev_reliability/evidence_feasibility_20260930_215718/evidence_matrix.json), and [independent verification](../data/08_reporting/betting_ev_reliability/evidence_feasibility_20260930_215718/inventory_verification.json).

- All 4,626 origins, 66,040 opportunities, 1,852 primary IDs and 498 saved IDs/sides/prices are retained. Original G0 ledgers remain separate: 34,650 canonical and 9,906 provider rows, with their original zero admissions.
- The 1,041 current absences comprise 915 newly unavailable, 123 previously incomplete/unknown and 3 previously overlapping rosters. Both sides are missing for 742 origins, one for 299. Of 1,783 missing side slots, 1,571 have an older complete unscoped five; these are not certified recoveries.
- No authentic unit bridge, announcement or bookmaker binding was recovered. Annual A0 recipe membership receipts do exist: all three receipt hashes and within-year TRAIN/CAL/TEST disjointness were checked. Exact historical source/label publication and schedule evidence remain separate gaps.
- Choose E readiness, not another historical replay or fit. Absence is bounded to the inspected sources; a late archive download or missing local issuance alone does not rule out transparently reconstructed research supported by authentic source evidence.
- Both metadata recipes exited 0. The independent oracle checked 9,252 side records, all opportunity joins, exact projected source values, 20 unchanged input/evidence hashes and 8 output hashes. No model/financial metrics or application tests ran.

## Completed E1–E2 local readiness

[Saved goal](../data/05_model_input/betting_ev_reliability/a0_local_readiness_20260930_222642/saved_goal.json), [final native pilot](../data/08_reporting/betting_ev_reliability/a0_local_readiness_20260930_222642/final/native_pilot.json), [immutable envelope export](../data/08_reporting/betting_ev_reliability/a0_local_readiness_20260930_222642/final/evidence_export.jsonl), [schema pilot](../data/08_reporting/betting_ev_reliability/a0_local_readiness_20260930_222642/schema_pilot.json), and [export oracle](../data/08_reporting/betting_ev_reliability/a0_local_readiness_20260930_222642/export_verification.json).

- Genuine annual `CausalA0Adapter` delegates to the archived predictor after manifest, source, checkpoint and calibration-receipt hash checks. Nine input-selected fixtures span2024–2026 and BO1/3/5; all three native modes agree exactly with direct native calls. Maximum complement error is2.66e−8; cold-history error from0.5 is1.11e−8. Retained2026 predictions agree within4.13e−8. Optional W20/organization absence, unchanged tensors, future non-input perturbations and strict declared release boundaries were exercised.
- `record_a0_snapshot` appends native tensors, roster/context, model/calibrator identity, schedule, quote/binding and exact-byte source captures to separate `forecast_evidence_snapshots`; it never writes active predictions. Nine final envelopes passed commit/readback and independent canonical-hash/source-link checks. Quotes/bindings are **synthetic local fixtures**, not recovered historical proof. Missing clocks remain null; actual generation and insertion clocks are distinct from simulated origins.
- The new table passed an empty-database migration chain, ORM `create_all` and full Timescale bootstrap. SQL UPDATE/DELETE, conflicting keys, concurrent writes, idempotency and rollback were exercised. `clock_timestamp()` avoids transaction-start backdating. Database-owner TRUNCATE/DDL remains outside this application guarantee.
- **136 targeted tests passed on the isolated ORM-created database.** On the isolated migration-created database,132 passed and4 API fixture setups failed because the pre-existing initial migration lacks the `bookmakers.is_active` default used by the fixture. New evidence-path tests and all three schema pilots passed. The unrelated bookmaker schema discrepancy was retained, not hidden or repaired incidentally; no full-suite or live-rollout claim follows.
- This is local software readiness, not G0 admission, source authentication, live A0 feature construction or profit certification. Run native archive imports in a separate local process; they use shared `src.models` names. Frozen artifacts, the canonical benchmark manifest, operational model and old10/10 journal were not changed.

## Completed E3 protocol — collection not activated

[Saved goal](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/saved_goal.json), [frozen protocol](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/protocol.json), [source contracts](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/source_contracts.json), [offline exercise](../data/08_reporting/betting_ev_reliability/forward_protocol_20260930_234022/protocol_exercise.json), [reproducible recipe](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/verification_recipe.json), and [decision](../data/08_reporting/betting_ev_reliability/forward_protocol_20260930_234022/decision.json).

- Two read-only source audits covered bookmaker/fixture access and sports/roster/history contracts. Eight source contracts are specified; **zero live sources are verified or authorized**. Official unit/announcement and actual-start source locators remain missing prerequisites, not invented APIs. The fixed candidate books are Pinnacle via OddsPapi, TOTALbet and Lebull. Require two distinct underlying bookmaker identities; an aggregator or duplicate feed is not an extra vote. Other scraper implementations are not automatic fallbacks.
- Primary population: all LoL series discovered in the approved OddsPapi fixture feed, including unmapped, late and unavailable records. This is not worldwide coverage. One primary **T−1h** origin per fixture; quote receipt age **≤300s**; forecast/persistence/decision finish within **60s** after cutoff and before independently evidenced actual start. Latest invalid/suspended state cannot revive an older quote. Schedule changes after a due origin do not rewrite it or create another primary decision.
- Preserve Plan05's H/controls, flat one-unit stakes, raw net EV **>5%**, tax **12%**, fixed bands, 5,000 paired monthly draws and seed **20260929**. Exact cross-side EV ties prefer **A**, matching the existing selector—not the superseded sweep's higher-odds tie. Numerical policy is explicitly symmetric epsilon **1e−6** throughout fit/selection/scoring; raw outputs remain preserved. No H/control fit or financial evaluation ran.

All dates below are UTC, start-inclusive/end-exclusive; membership uses the immutable nominal cutoff, not a later result or reschedule:

| Partition | Origin window | Label lock | Role |
|---|---|---|---|
| TRAIN | 2026-11-01 → 2027-11-01 | 2027-11-15 | Fit the fixed head/control grid on all admitted TRAIN fixtures |
| CAL | 2027-12-01 → 2028-06-01 | 2028-06-15 | Choose penalty by LL; ties prefer stronger penalty; no TRAIN+CAL refit |
| DEVELOPMENT | 2028-07-01 → 2029-07-01 | 2029-07-15 | One closed assessment of the selected frozen recipe |

November2027 and June2028 are maturity/fit-lock buffers, excluded from head fitting/tuning/scoring. The design has30 observation months over32 calendar months; this is **not an authorized collection commitment or precision guarantee**. There are no actual partition memberships yet. Readiness and separate acquisition authorization must be complete **before2026-10-31T00:00Z** for the planned2026-11-01 start; otherwise this version expires unstarted. No automatic date shifts, retrospective inclusion or outcome-driven extensions. E4's separate12-month confirmation dates remain unset until recipe freeze, gates and separate authorization.

**Local proof:** `env PYTHONPATH=. .venv/bin/python /tmp/ensemble-e3-protocol-234022.py` passed **79 checks**: calendar/clock/schedule/freshness boundaries; actual existing quote, consensus and selection functions; nine E1/E2 envelope hashes and72 tensor hashes; native snapshot validation; future-result/schedule/state perturbations. All **9/9 synthetic envelopes fail necessary forward-admission conditions**, with explicit reasons; this is not another full G0 audit. The actual current adapter rejects a2027 cutoff with `cutoff year differs from the annual A0 recipe year`. The one-off script is archived in the recipe and removed after verification.

**At the E3 protocol seal, eight activation blockers were recorded:** source authorization/entitlements/official locators; exact-byte dual-clock capture; authentic identity/participant bindings; actual-start/result evidence; two-book coverage and proven request capacity; live native tensor construction; future-valid unchanged2026 artifact contract; and immutable no-forecast opportunity ledger. Subsequent Plan09 local proof resolves the capture/ledger/software-artifact subset, not authentic acquisition or live operation. Existing annual validation stays intact; unknown publication remains null. The older full-application migration discrepancy remains a rollout limitation, not a prerequisite for isolated capture.

**Next gate after local implementation:** source-specific acquisition/retention authorization, verified entitlements/quotas and official locators, followed only then by capped authentic source probes and live-input/coverage proof. No network probe, scheduler or collection starts implicitly. Plan06 E4 confirmation cannot begin now.

## Subsequent Plan09 local implementation — 2026-10-02

[Plan09 E1–E4](./09_c0_connector_and_forward_evidence.md) is verified; [execution report](../reports/plan09_c0_connector_and_forward_evidence_20261002.md) and [BLOCKED activation decision](../data/08_reporting/betting_ev_reliability/forward_capture_20261002_033658/decision.json). Exact-byte dual-clock capture and a separate immutable nullable-forecast ledger passed three empty isolated PostgreSQL schema paths, concurrent writes, rollback, direct SQL immutability and retained-byte/future-state replay. Final18 local fixtures retain16 primary+2 pending,1 genuine native envelope+17 without forecast,0 admitted. Eight own-loopback requests across two transport attempts,0 external requests; stage01/03 evidence archived and owned database stopped.

The explicit `frozen-a0-2026-forward-v1` contract serves unchanged2026 artifacts for **[2026-11-01,2029-07-01)**.21 native/mode probabilities agree exactly; captured-history/future/cold/nonmutation checks pass. Annual unsupported-year rejection remains intact. This is local software compatibility, not a live worker,2027-trained model, source authentication or confirmation.168 distinct final focused tests cover the whole Plan09 session, including sports; not a full application suite.

E5 source probes remain blocked: all eight sources unapproved/unverified, missing permission/retention/quotas/official locators/authentic bindings/start/results. No H fits, actual future forecasts or serving changes. Plan09's separately rejected C0 head does not replace this plan's frozen annual A0. The original E3 calendar, deadline, admission/EV gates and all earlier sealed receipts remain unchanged.

## Decision and goal

The evidence work aims at **an evidence-backed admission decision**, not another model correction. Recover roster coverage only where competitive-unit continuity is supported; establish whether trustworthy historical inputs and quotes exist. The completed A1–A2 decision above selects forward readiness; the conditional B/C/D design below remains unexecuted.

[Plan 05](./05_betting_ev_reliability.md) remains the specification for H, controls, selection, statistical gates and prospective confirmation. This plan supplies its missing evidence work in executable slices; it does not create a second model search or replace those gates. The older uncertainty/Kelly sweep and historical Plan 04 are not execution instructions for this work.

### What the completed work establishes

Source: [attempt 10 results](../data/08_reporting/betting_ev_reliability/a0_context_stages_20260930_183251/results.json) and [verification](../data/08_reporting/betting_ev_reliability/a0_context_stages_20260930_183251/verification.json).

- Exact-context roster lookup corrects the Gen.G–HLE failure without actual target players. It leaves **3,585/4,626** origins available, versus 4,500 before the restriction: 915 additional absences, not 915 proven recoverable forecasts.
- On the same **387 retained selections**, EV overstatement remains **29.78 pp [10.45, 54.23]**. Raw neural/rating/MLP errors precede calibration. No specific omitted sports factor was established.
- Neural softening increases selected EV while improving proper scores on **1,539 scored primary matches**. Removing calibration, flattening every prediction or dropping leagues is not a supported fix.
- Bookmaker bindings and historical publication/observation evidence remain unverified. Existing source hashes and software tests do not certify them.

**Success of the immediate work:** every origin has an explicit admission or rejection reason supported by immutable evidence, and the next research branch is decided. Zero admissible rows is a valid negative result, not a reliability pass. Do not promise a recovery count or profit.

## Invariants and shared evidence contract

1. Keep A0 weights, calibrators, mixture, player-history ordering and all Cup/player/rating/W20 history unchanged. Keep research A0 distinct from operational EXP081/hybrid.
2. No team-name exceptions, fuzzy identity merges, family-only roster fallback, target-lineup oracle, outcome-based odds reversal or retrospectively chosen exclusions.
3. Preserve all **4,626 origins**, **66,040 opportunity rows**, the **1,852 primary IDs** and **498 saved selection IDs/sides/prices** in the attempt-10 diagnostic control. Separately retain the original G0 canonical/provider inventory; these inventories are not additive or interchangeable.
4. Evidence records require an exact scope, source locator and content hash, publication and observation clocks, and validity interval where applicable. Reuse `published_at`, `observed_at`, `valid_from`, `valid_to`, `evidence_uri` conventions in `roster_provenance.py`; quote bindings retain their existing exact fixture/book/market/outcome identifiers. Also record actual retrieval time. A hash proves bytes, not historical truth.
5. Never backdate today's observation. Only independently supported earlier capture/availability evidence can support an earlier origin. Missing clocks stay missing; no midnight/default timestamp converts a scenario into proof. A lineup published at T−30m cannot repair T−1h.
6. Keep **recorded decisions**, **evidence-backed historical reconstruction**, and **assumption-only scenarios** separate. A replay's actual creation time remains current; the simulated origin is a different field. Neither replay nor a quote binding proves historical account-level execution.
7. Freeze eligibility from identities, scope, clocks and source evidence before comparing results. Protected prospective outcomes stay unopened. All previously examined 2024–2026 outcomes remain development-exposed.
8. Extend existing modules and the unified runner. No second roster registry, competing benchmark script, parallel EV formula or alias-based migration. Before changing exported interfaces, enumerate references and migrate active callers/tests together.

## Dependency order

**A. Local evidence feasibility → B1 roster evidence and B2 quote evidence in parallel → C. Integrated replay/admission → decision.**

Only the shared evidence contract above is a prerequisite for both B tracks. They must not edit the same integration boundary concurrently: roster ownership is `roster_provenance.py` / `ev_asof_replay.py`; quote ownership is `quote_provenance.py`; the integration owner handles `ev_input_repair.py` and the unified CLI after both are ready.

- If G0 and fitting-support requirements pass: **D. One authorized H/control comparison** under Plan 05.
- If evidence is irrecoverable or support inadequate: **E. Forward acquisition readiness**, not another historical fit or audit loop.

## A. Freeze control and determine what evidence is recoverable

- [x] **A1 — Create a metadata-only evidence inventory.**
  - **Targets:** existing attempt-8/10 inputs, receipts and coverage tables; `src/analysis/ev_input_repair.py` for reusable inventory/admission logic; `src/analysis/a0_context_evaluation.py` for existing paired coverage accounting.
  - **Work:** reuse completed exports and audits, pin their hashes, and classify all 1,041 current roster absences, including the 915 newly lost origins. Distinguish missing competitive-unit evidence, missing released roster/history, identity conflicts, missing publication clocks and unavailable quotes. Track evidence gaps independently rather than collapsing them into a single success flag. Do not rerun the original model-error diagnosis.
  - **Proof:** exact inventory-ID reconciliation; every gap has an existing source locator or an explicit missing-source status; no p, win/loss or realized-return criterion determines recovery priority or eligibility.

- [x] **A2 — Make one bounded source-feasibility decision.**
  - **Targets:** `data/01_raw/betting_ev_reliability/production_provenance_20260930_001359/`, existing raw GOLGG/OddsPapi snapshots and the input manifests referenced by attempt 10.
  - **Work:** accept completed audit findings as known; inspect only named source/field gaps not already exhausted by those audits. Record `available_with_proof`, `conflicting`, or `not_present_in_existing_sources` for each requirement. If new external evidence is needed, list its exact provider/document/API fields and acquisition scope; do not repeat an exhaustive scan or request the same complete production export again.
  - **Proof:** a source/evidence matrix stating which origins could be admitted and which facts cannot be reconstructed. Missing historic publication or observation cannot be repaired by replaying the model. If there is no defensible recovery path, skip historical implementation/replay and take branch E.

## B. Implement only evidence-supported recovery

### B1. Roster context and competitive-unit continuity

- [ ] **B1a — Extend the existing roster evidence contract.**
  - **Targets:** `src/analysis/roster_provenance.py` (`load_roster_evidence`, `resolve_roster_evidence`); `betting_app/tests/test_roster_provenance.py`.
  - **Work:** add scoped competitive-unit evidence to the existing immutable evidence input, not to a new name-based registry. A proposed record must connect an exact team identity and source/target competition contexts to the same roster unit, with publication/observation, validity and source proof. Unit continuity is not nationality, competition family, organization identity or an assertion derived from actual target players. Resolve the exact schema from the sources found in A2 before implementation. Keep existing directional team-ID lineage separate from context continuity.
  - **Proof:** a supported cross-phase/main-unit link is admitted; an equally recent alternate Cup unit, wrong team, expired/future link, conflicting mapping or retracted evidence is rejected. Record source-authentication status separately from syntactic acceptance.

- [ ] **B1b — Integrate temporal unit-aware roster lookup.**
  - **Targets:** `src/analysis/ev_asof_replay.py` (`select_prior_roster`, `build_asof_targets`); `betting_app/tests/test_roster_context_replay.py`, `betting_app/tests/test_ev_asof_replay.py`.
  - **Work:** valid exact-match announcements retain precedence. Otherwise select the latest released roster among exact-context observations and independently evidenced same-unit contexts. Without that extra evidence, retain v3's exact-context restriction. Invalid latest evidence or an incomplete latest eligible roster must not silently revive an older five. Keep explicit scheduled-target context and its provenance when the target is absent from the archive. Version the changed replay contract.
  - **Proof:** preserve the Cup→LCK regression and a winning control; verify late announcements, boundary clocks, transfers, main/academy separation, conflict, missing context and cold player history. Adding future results/rosters/context links must not change earlier inputs or forecasts. With no new applicable evidence, retained v3 forecasts and eligibility must remain unchanged; raw/history player slots stay aligned and history queries remain nonmutating.

### B2. Bookmaker participant binding and availability

- [ ] **B2a — Build immutable bindings from authentic source payloads.**
  - **Targets:** `src/analysis/quote_provenance.py` (`extract_winner_quote`); `betting_app/tests/test_quote_provenance.py`; existing raw-source input manifests.
  - **Work:** supply the existing `participant_bindings` contract with exact provider and bookmaker fixture, market, outcome and participant identities for both sides. Preserve source bytes, locator, publication/observation and retrieval clocks. Do not infer binding from generic IDs 181/182, team-name similarity, price direction or the eventual winner. No backfill is labelled contemporaneous.
  - **Proof:** the correctly scoped two-side record admits the actual source orientation, including a genuinely evidenced reversal of provider order; contradictory or unsubstantiated swaps, duplicate/missing sides, wrong book/market/fixture, live/map markets and late evidence fail closed. Apply the evidenced orientation exactly once. The latest inactive/malformed quote cannot revive an older price. Outcome perturbation cannot affect binding or quote selection.

- [ ] **B2b — Thread evidence through offline quote reconstruction.**
  - **Targets:** `src/analysis/ev_input_repair.py` (`reconstruct_quotes`, `_quote_contract`); `scripts/run_unified_betting_evaluation.py`; `betting_app/tests/test_ev_input_repair.py`.
  - **Work:** consume the immutable binding manifest rather than inventing a second parser. Preserve source quote time, actual observation, schedule vintage, earliest known start and a same-cutoff consensus with one vote per bookmaker. Require two distinct books for the declared H comparison; unavailable consensus stays unavailable. Keep CLOSE out of earlier decisions.
  - **Proof:** all original quote/opportunity IDs remain accounted for; quote state, orientation and exclusion reason have an independent raw-payload oracle. If authentic evidence contradicts a saved side/price, retain that original record as invalid, exclude it from the same-price paired comparison and report corrected-price selections separately; never silently rewrite the control. No scrape, scheduler, database migration, server mutation or live ingestion is included in this offline slice.

## C. Freeze admission before one paired reevaluation

- [ ] **C1 — Produce and freeze the evidence-specific admission ledger.**
  - **Targets:** `src/analysis/ev_input_repair.py`, `src/analysis/production_market_benchmark.py` (`validate_fixture_cohort` and quote/consensus helpers), `src/analysis/ev_approach_data.py` (existing artifact/complete-recipe provenance), and the unified runner. Extend existing contracts; the current repair result's hard-coded G0 status is not an admission implementation.
  - **Work:** require evidence for team/roster context, source release and cutoff, schedule vintage/start, quote binding/state, and complete-recipe past-only training/calibration. Separate reconstructed research eligibility from actual issued/displayed forecast and execution evidence. Store all rejection reasons, missingness and explicit information-status classes. Freeze admission rules and chronological head-TRAIN/CAL/development memberships after the metadata audit, before any new performance comparison; do not search splits for favorable outcomes.
  - **Proof:** row-level admissions trace to source evidence; future-label/quote/roster mutations leave earlier admissions and inputs unchanged. No assumption-only row enters fitting. Include date/block counts and a pre-fit support/precision assessment, not just a positive admitted-row count.
  - **Decision:** `G0_BLOCKED`, `INSUFFICIENT_FIT_SUPPORT`, or `G0_PASSED_FOR_DECLARED_RESEARCH_COHORT`. The last is not production or executable-betting certification. A blocked or inadequately supported cohort goes to E without another full historical performance replay; admitted, supported inputs proceed to C2's end-to-end verification.

- [ ] **C2 — Run one frozen-model, paired input reevaluation.**
  - **Targets:** `src/analysis/ev_input_repair.py` (`replay_inputs`, `attach_repaired_predictions`, `selected_reliability`); `src/analysis/a0_context_evaluation.py`; the unified runner's existing `input-repair` entry point.
  - **Work:** use C1's frozen evidence, admission rules and memberships. Capture real generation time and all input/model/source hashes. Compare v3 with the evidence-supported contract on identical admitted/available IDs, and report newly available/unavailable rows separately. Retain original saved selections and report any reselected portfolio as a different population. Reuse the existing payoff and bootstrap definitions; no new Kelly simulation.
  - **Proof:** unchanged-input native parity, A/B symmetry, missing/cold-history behavior, strict chronology and future-data perturbation; all denominator totals reconcile. Independent checks reproduce LL/Brier and sidewise taxed unit returns. For this repair comparison use the attempt-10 control policy, epsilon, seed **20260930**, 5,000 draws and complete preselection block universe. More coverage is not a model-score gain.
  - **Decision:** publish the verified admission counts, common-cohort results and final G0 decision, then stop this workstream. Parity or chronology failures block progression; a score improvement cannot override them. Do not silently launch D.

## D. Conditional next model experiment — already specified, not authorized here

If C passes and execution is authorized, resume **Plan 05 Phases 2–4**, rather than inventing another candidate:

- Keep S = frozen complete A0. Fit H = `sigmoid(a*logit(p_s) + b*logit(q))`, `a,b >= 0`, and only the already declared fitted controls. Retain S-cal, M, M-cal, M-shape and Fixed50, the existing penalty grid and past-only selection rules.
- Fit on all eligible earlier fixtures, not only S-selected bets. Use the same admitted populations and probability policy; do not leak calibration/development outcomes into earlier forecasts. Preserve the Plan 05 statistical seed **20260929** for this separate experiment; do not silently inherit the repair-diagnostic seed.
- Report all-fixture proper scores, unchanged S-selected IDs/sides/odds, and each recipe's own portfolio separately. Keep strict raw net `EV > 0.05`, tax 0.12, one-unit returns and the predeclared bands. Operational filters remain a separately labelled comparison.
- Require Plan 05's sequential LL/Brier market-control gate, then its **±2 pp** EV-reliability interval for both all selected bets and the **(5%,10%]** band. Positive after-tax value and incremental value over market controls are separate gates. Preserve zero-bet months and undefined replicates; no bets or inadequate precision cannot pass.
- **Targets:** `src/analysis/thesis_models.py`, `src/analysis/production_market_benchmark.py`, `src/analysis/unified_evaluation_engine.py`, the unified runner and their existing tests. No neural retraining, odds-driven change to A0, regional exceptions, new threshold grid or selection of whichever stage looked best in attempt 10.
- **Proof/outcome:** the immutable recipe, partitions, controls, paired intervals and independent payoff oracle support `REJECTED`, `INCONCLUSIVE`, `NO_ADDED_SPORTS_VALUE`, or at most `DEVELOPMENT_CANDIDATE`. Exposed historical data cannot supply prospective confirmation.

## E. If historical admission cannot be established

Prepare the minimum **forward acquisition/development** path instead of repeatedly trying to repair unavailable history:

- [x] **E1 — Local native adapter validation.** Implemented in `betting_app/core/models/causal_a0.py`; native parity, symmetry, optional context, cold history, BO and nonmutation proofs are recorded above. Operational EXP081/hybrid remains separate and active.
- [x] **E2 — Local append-only evidence pilot.** Implemented in `betting_app/services/forecast_evidence.py` and `a0_shadow_readiness.py`, ORM model and migration `f1e2d3c4b5a6`. Actual PostgreSQL/Timescale verification and the synthetic-evidence limitation are recorded above; no live collection was performed.
- [x] **E3 — Freeze acquisition/development protocol before live collection.** Completed with exact dates, source contracts, T−1h/coverage rules and79 offline checks above. Activation remains blocked; no sources, memberships or outcomes were collected. A missed readiness deadline expires this version rather than shifting its periods.
- [ ] **E4 — Separate confirmation after recipe freeze.** Follow Plan05's separately authorized, preregistered protocol. Twelve months does not guarantee adequate precision. No automated wagers, hidden early stop, live deployment or server restart is authorized.

## Artifacts and verification

Use one new, non-overwriting run directory per explicitly authorized execution. Preserve attempt-10 and earlier outputs unchanged.

| Stage | Planned output |
|---|---|
| `data/01_raw/betting_ev_reliability/<run_id>/` | Newly authorized immutable source bytes and original observation metadata |
| `data/03_primary/betting_ev_reliability/<run_id>/` | Evidence inventory, scoped roster/quote evidence and complete admission ledger |
| `data/05_model_input/betting_ev_reliability/<run_id>/` | Frozen protocol, input hashes, information contract, decision gate and any admitted memberships |
| `data/07_model_output/betting_ev_reliability/<run_id>/` | All replay origins, availability, stage/model identity and actual creation clocks |
| `data/08_reporting/betting_ev_reliability/<run_id>/` | Coverage transitions, common-ID scores, fixed/own selections, G0 decision and independent receipts |

Model artifacts under stage 06 are only for separately authorized D, never an incidental product of evidence work.

Existing entry point to extend: `scripts/run_unified_betting_evaluation.py input-repair`. It currently accepts `--inputs`, `--stage replay|quotes|evaluate`, optional `--research-root` and `--roster-evidence`. A generalized evidence-admission action/binding-manifest interface is **proposed work**, not a currently implemented CLI flag. Do not execute the broad legacy financial matrix just to validate admission.

Run affected behavior tests with database variables set to an isolated/non-serving local URL; never the user's configured database. The existing suites are `test_roster_provenance.py`, `test_roster_context_replay.py`, `test_ev_asof_replay.py`, `test_quote_provenance.py`, `test_ev_input_repair.py` and `test_production_market_identity.py` under `betting_app/tests/`. Their current regressions ran with E1/E2 as reported above. The additional evidence-admission scenarios specified for deferred B/C remain planned; running existing tests does not execute those workstreams.

## Completed deliverables and next task

**A1–A2, local E1–E2 and E3 protocol are complete; Plan09 subsequently completed local capture/no-forecast-ledger and explicit fixed-artifact/native-input proof.** The [frozen E3 protocol](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/protocol.json) still controls future activation. Next is the source-specific authorization/authentic evidence gate, not another local implementation or historical audit. Live collection, H/control fits and E4 confirmation remain unexecuted; do not restart10/10 or roll the frozen calendar.
