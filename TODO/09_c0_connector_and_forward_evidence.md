# Plan 09: C0 learned connector and decision-time evidence

**Created:** 2026-10-02  
**Status:** AUTHORIZED HISTORICAL/LOCAL SLICES COMPLETE; EXTERNAL GATES BLOCKED — 2026-10-02.  
**Scope receipt:** [`plan09_execution_20261002_033658`](../data/05_model_input/plan09/plan09_execution_20261002_033658/scope.json). S1–S8 completed / **NOT_ADOPTED**; E1–E4 verified; E6 decision **BLOCKED**. E5/F1–F3/authentic T1 remain unexecuted for explicit prerequisites. [Execution report](../reports/plan09_c0_connector_and_forward_evidence_20261002.md). No serving-database migration, deployment or wagering.

**Next gate:** source-specific acquisition/retention authorization, actual entitlements/quotas and authenticated official unit/roster/start/result evidence. Local software is verified, not source-authenticated. Do not repeat the completed72-fit sports recipe, start a replacement sweep, clear betting G0 or relabel future/edition tasks complete.

## 1. Sources of truth and completed work

- [Next-step research report](../reports/next_model_steps_after_learned_connector_20261002.md), especially §§3,5,7–9; [sealed research decision](../data/08_reporting/model_next_steps/next_model_steps_20261002_01/decision.json).
- [IDEA-027](../docs/future_ideas.md#idea-027--c0-anchored-learned-raw-specialist-connector): catalog entry for Track S. This plan supplies its detailed task checklist, not a second competing experiment.
- [Current research handoff](../docs/RESEARCH.md) and [canonical manifest](../conf/base/research_benchmark.json).
- [Retained connector03 protocol](../data/05_model_input/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/protocol.json) and [matched comparator/source receipt](../data/08_reporting/model_next_steps/next_model_steps_20261002_01/comparator_protocol.json).
- [Plan06](./06_evidence_admission_after_a0_audit.md) remains authoritative for evidence admission and E3 activation. Its A1–A2, local E1–E2 and E3 protocol are complete. Track E below decomposes its **remaining** capture/ledger/tensor/activation work; it does not reopen its completed audit.
- [Plan05](./05_betting_ev_reliability.md) remains authoritative for the conditional H/control experiment, selected-bet gates and separate confirmation. G0 remains blocked; its completed attempt journal stays **10/10**.

Existing evidence, not work to rerun:

| Item | Recorded result | Consequence |
|---|---|---|
| Learned connector03 versus annual A0 | Same3,103 canonical2026 rows: LL0.535644479 versus0.536756581 | Retained historical development candidate, not production |
| Connector03 versus native12 C0 | C0 LL0.535448701; connector−C0 ΔLL+0.000195778, monthly interval[−0.001156409,+0.001791005] | No established gain over C0; test this anchor explicitly |
| C0→raw-trio local LL derivative | Negative point in2024/25/26; monthly interval crosses0 in2024, not2025/26 | Motivation only; no fitted weight or demonstrated finite-weight gain |
| Fixed A0 selections in the1,619-fixture market scenario | Connector overstatement44.8558pp on328;27.5798pp on the74 original near-band selections | Selected-EV reliability remains unresolved; not the old387 prior-context sample |
| Native6+12, H365, history dropout/data focus, reconstruction, larger correction heads | Completed comparisons with failed relevant gates | No automatic retry or new label for the same hypothesis |
| Local A0/envelope and E3 protocol readiness | Software/protocol evidence only; zero authenticated and authorized live sources at the recorded checkpoint | Finish capture and activation prerequisites; do not call fixtures live evidence |

All previously inspected2024–2026 outcomes remain development-exposed. Preserve negative results and old reports/receipts. Do not edit a sealed report to append a new experiment; use a separate execution report and update mutable handoffs.

## 2. Dependency order and ownership

```text
P1 scope and immutable inputs
  ├─ S1 input alignment → S2 protocol freeze → S3 head/runner → S4 evaluation
  │    → S5 pre-fit verification → S6 one bounded execution
  │    → S7 independent proof → S8 decision
  │
  └─ E1 exact capture → E2 full opportunity ledger → E3 local integrated proof
       → E4 native tensors/future-valid fixed-artifact contract
       → E5 separately authorized source probes → E6 activation decision

S8 does not authorize E5/E6 or H fitting.
E6 does not certify model/EV quality.
F1 sports confirmation and F2/F3 betting development/confirmation have separate gates.
T1 tournament certification is a separate branch, not a consequence of winner LL.
```

**Implementation ownership:** one sports owner for S1–S8; one evidence owner for E1–E6. After P1 their work can proceed independently. One integration owner owns shared persistence/schema changes, final documentation and sealing. Do not have two workers edit the same model/evidence integration file concurrently. Run the affected verification after a slice is integrated, not repeated full suites during parallel edits.

### P1 — Establish the execution boundary

- [x] **P1 — Record the authorized slice and preserve the current checkout.**
  - **Targets:** this plan, existing research receipts, `conf/base/research_benchmark.json`, the current branch/worktree.
  - **Work:** record whether the execution request covers S, local E1–E4, authorized probes, or a later F/T branch. Do not ask again for already explicit authorization, but do not interpret “create a plan” as permission to run it. Inspect branch/uncommitted changes; retain unrelated work. Pick a new non-existing run ID for each executed track, and record actual creation time separately from historical/scenario origins.
  - **Freeze:** baseline hashes for the manifest, canonical paired table, reused predictions, checkpoints/calibration receipts and previous decisions. Resolve the external research root through the established argument/environment/local-file contract; never hardcode a personal directory.
  - **Proof:** a scope/input receipt lists all reused sources and a zero-ambiguity output location. No source/model/report is overwritten. A failed input hash or an unknown mandatory source stops the affected slice before fitting or network access.

## 3. Track S — One bounded sports-only experiment

### 3.1 Question, controls and non-goals

**Question:** does the existing raw specialist improve the existing assembled native12 C0 forecast beyond unchanged C0, a global C0-only slope recalibration and connector03?

The hypothesis concerns an incremental predictive gain under these controls. It is not a causal proof of a missing sports factor, and it does not rule out every nonlinear calibration-only explanation.

| Arm | Probability | Learned parameters in this experiment |
|---|---|---:|
| Native12 C0 | Existing assembled full-context series probability |0|
| C0-cal | `sigmoid(s * logit(p_c0))`, `s >= 0`, identity `s=1` |1 per origin/penalty|
| C0 connector | `(1-alpha)*p_c0 + alpha*p_specialist`, `0 <= alpha <= 1`, identity `alpha=0` |1 per origin/penalty|
| Connector03 | Reused retained prediction, unchanged |0|
| Canonical annual A0 | Reused canonical prediction, unchanged |0|

Use uniform mean Bernoulli LL and identity-centred penalties:

- candidate: `mean_LL + 0.5*lambda*alpha**2`;
- calibration control: `mean_LL + 0.5*lambda*(s-1)**2`.

Both preserve A/B complementarity. A free logit intercept is excluded: at neutral p it shifts both orientations in the same direction. Do not introduce odds, league/tier identities, BO-specific coefficients, new experts, extra knots, a new neural objective, a second BO projection or a larger penalty grid. The old fixed75/25 diagnostic and rejected native6+12 remain references, not newly competing arms.

Scope is **full-context series-winner inference only**. Do not invent no-W20/no-organization modes for the new connector; canonical A0's existing optional-context modes remain unchanged.

### 3.2 Inputs and output identity

Exact input locators and hashes are pinned by the linked comparator receipt:

| Role | Existing input |
|---|---|
| C0 forecast | `data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment/joined_predictions.parquet`, column `native12__full__full` |
| Annual A0, specialist, connector03, release metadata | `data/07_model_output/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/predictions.parquet`, columns `p_a0`, `p_specialist`, `p_correction` and lineage fields |
| Raw specialist source | `data/05_model_input/a0_sports_correction/a0_fresh_correction_20261001_02/fixed_raw_specialist_predictions.parquet`, column `p_fixed_raw_families` |
| Prepared native/raw lineage | `data/05_model_input/a0_sports_correction/a0_fresh_correction_20261001_02/prepared.json`, plus the C0 protocol/checkpoint/calibration receipts |
| Canonical cohort | Manifest-resolved `rating-foundation-20260915/data/08_reporting/paired.parquet` |

**Proposed new experiment columns:** `p_a0`, `p_c0`, `p_specialist`, `p_connector03`, `p_c0_calibration`, `p_c0_connector`. These names must retain their literal model meanings. Do not overwrite `p_a0` with C0 to make an A0-specific runner appear compatible. The old `p_calibration` is A0-based and must not be reused or relabelled as the new C0 control.

Require exact one-to-one matching on `golgg_match_id`, `date`, `result_day`, `team1_id`, `team2_id`, `best_of`, `y`. Reindex only after verifying complete unique-ID sets. Preserve the documented raw-source side normalization exactly once; no outcome-based orientation repair, inner-join row loss or price-dependent eligibility.

### 3.3 Chronology, selection and fixed fit budget

- Complete cohort: **11,550** canonical rows; primary: all **3,103**2026 rows through September8; earlier **8,447** rows are warmup.
- Head-fit/rolling-validation origins: **2024-07-01,2024-10-01,2025-01-01,2025-04-01,2025-07-01,2025-10-01,2026-01-01,2026-04-01,2026-07-01**.
- Delivered active-head origins: **2026-01-01,2026-04-01,2026-07-01**. A head's target interval is its origin-inclusive/next-quarter-exclusive interval; the July head does not automatically cover October.
- Fit rows satisfy both `date < origin` and `head_label_release_day < origin`. Require at least **500** fit rows at every origin; otherwise stop, not an unrecorded partial fit.
- Penalties: **0.0001,0.001,0.01,0.1**. Select each arm's penalty separately on the same eligible earlier rolling-origin validation IDs; require `cv_origin < origin` and labels released before origin. Score selection by unpenalized match-weighted LL. Ties within **1e-12** choose the stronger penalty. With no past validation support, use the strongest penalty and record that reason.
- Exactly **9 origins × 4 penalties × 2 learned arms = 72 real head fits** if all inputs/support pass. There are **zero base fits**, no extra final refit and no head hyperparameter search after assessment. Reuse the already fitted origin/penalty artifact chosen by past validation.
- Mathematical/toy verification fits are separate from these72 data fits and must be labelled as such. If a genuine implementation defect forces a repeated execution, record every attempt and cost; retain the failed evidence and unchanged recipe. No silent reset or unrestricted retry-until-improved loop.
- New candidate and new calibration-control forecasts on the earlier8,447 rows are exact **C0 warmup**, with a literal C0 warmup status. Connector03's same rows remain exact A0. Full-cohort differences mix that anchor substitution with the2026 active-head effect; they are not8,447 new connector assessments.
- The inherited raw architecture was selected on2024–2025H1. Earlier weight-OOF forecasts are closed head-development inputs, not selection-unseen confirmation.
- Strict interior probabilities; **no new clipping** in fitting, selection, headline metrics, differences or profiles. Use stable logit/log-likelihood arithmetic. A numerically saturated/invalid output stops the run; do not fix it by silently clipping only the reported metric.

### 3.4 Metrics, bootstrap and decision gates

Freeze before fitting:

- Primary LL and Brier, candidate minus each of **C0, C0-cal, connector03**. For **all three** contrasts: the paired monthly95% LL interval upper endpoint must be **<0**, and Brier point difference must be **<=0**.
- Eight noninferiority checks: BO3, BO5, major and source-unseen-player, each versus **C0 and annual A0**. Require the LL interval upper endpoint **<+0.005**. Reuse the existing major-event classifier and `unknown_players > 0` definition; retain unknown-profile counts. Missing/undefined support does not pass.
- Exactly **5,000** paired calendar-month draws, **seed20261002**. Preserve all nine primary months and shared draw matrices across paired contrasts. Report match-weighted and equal-month effects. Use the existing fixed-month helpers, retaining zero-support months for conditional guards. Record every zero-support replicate; a required conditional interval with undefined replicates remains unresolved rather than silently discarding/redrawing them or substituting zero. Freeze this convention before fitting; do not replace blocks with IID row bootstrap.
- Source-tournament-name within year sensitivity:143primary proxies under the current unchanged labels. Report match-weighted/equal-proxy results. These are not verified independent tournament editions and cannot rescue a failed primary gate.
- Full11,550 and earlier8,447 scores are required descriptive coverage/warmup reporting, **not a replacement primary gate**. Do not accidentally inherit the current A0-specific evaluator's whole-cohort eligibility rule as if it were the new hypothesis.
- Diagnostics: year/origin, BO, event tier/family, experience, co-play, unknown players/new organization, W20 availability, confidence and expert disagreement; LL/Brier, AUC/accuracy, calibration support, fixed-A0 favorite/quarter-confidence cells, own-favorite gaps, ECE and LL>=2.5 tails. No ECE-only or tail-count promotion rule.
- All intervals are nominal exposed-development evidence, not selection-adjusted proof. Point improvement with a crossing interval means the gate failed, not equivalence.

**Optional archived market diagnostic, fixed before execution:** reuse only the existing1,619-fixture scenario and exact328/74 A0-selected IDs/sides/prices, then separately report the candidate's own selections. Preserve tax0.12, the archived freshness/tie/probability conventions and all pre-selection months. Do not replace it with the old387 prior-context sample, recompute a new Kelly portfolio, or let it affect sports fitting/selection. Optional means its execution is explicitly chosen before the run—not after viewing a favorable subset. It cannot clear G0 or establish live EV.

### 3.5 Implementation tasks and proofs

- [x] **S1 — Assemble and validate the C0/raw/control input bundle.**
  - **Targets:** `src/models/a0_correction_experiment.py`, the comparator receipt, existing stage05 preparation and source provenance.
  - **Work:** keep canonical A0, C0 and connector03 as separate columns/identities; enforce full-ID matching, source orientation, release clocks, strict probabilities and all transitive checkpoint/calibration hashes. Save exact per-origin fit and validation memberships, including maximum label-release dates. Trace C0's actual native inference recipe, not the operational EXP081 path or the annual-A0 adapter as a substitute.
  - **Proof:**11,550 exact IDs;3,103 primary;8,447 warmup; no key/BO/y mismatches or probability absences; reused values match pinned sources; shuffled input table order does not change the keyed bundle. An incompatible artifact or row is an explicit integrity failure, not a smaller cohort.

- [x] **S2 — Freeze the new protocol and separate runtime identity.**
  - **Targets:** a new `data/05_model_input/a0_sports_correction/<run_id>/protocol.json`; existing directory/provenance helpers.
  - **Work:** record §3.1–3.4 verbatim as machine-readable decisions, input hashes, new probability-column mapping, warmup policy, model/control schemas,72-fit budget, all origins/grid/seed, guards and the optional diagnostic choice. Preserve true `generated_at` separately from historical `origin`. Later seal implementation/dependency versions and a source snapshot **before data fits**.
  - **Proof:** a protocol validator rejects missing/changed controls, extra penalties, unsupported modes, changed cohort, conflicting recipe identity and already existing output directories. This is a new protocol/version, never a rewrite of run03's sealed protocol.

- [x] **S3 — Implement the C0 head/control and integrate the existing runner.**
  - **Targets:** `src/models/a0_probability_connector.py`; shared `training_indices`, `select_penalty` and artifact helpers in `src/models/a0_correction_experiment.py`.
  - **Work:** reuse the convex coefficient optimizer and add the identity-centred nonnegative slope control. Make anchor identity explicit throughout input validation, serialization, head status, selected artifacts and inference. Preserve original A0 schemas in their sealed artifacts/replay snapshots; the new active path uses explicit version/model identity, not misleading aliases that make C0 look like A0. Enumerate references before any exported-symbol change and migrate active callers/tests together.
  - **Proof:** known synthetic optima/boundaries; correct derivative/stationarity checks; alpha0/1, slope0/1 and identical-expert behavior; complementarity and neutrality; serialization round-trip reproduces probabilities. Inference rejects the wrong anchor artifact, BO and validity interval. No probability at a current/future origin participates in an earlier head fit or selection.

- [x] **S4 — Extend the existing evaluator with explicit comparator bindings.**
  - **Targets:** `src/analysis/a0_sports_correction.py`; existing bootstrap/scoring helpers and benchmark candidate export.
  - **Work:** remove the new recipe's dependency on hard-coded A0/control column assumptions without changing the recorded semantics of earlier recipes. Produce all three primary contrasts, eight guards, C0-versus-A0 warmup disclosure, calibration/profile support and optional fixed/own market diagnostics. Keep fixed-A0 favorite cells anchored to actual annual A0, not renamed C0. Export a complete candidate in the canonical benchmark contract with truthful scenario/creation timestamps and `train_end`/`calibration_end` release bounds derived from the complete recipe; preserve finer nested lineage separately. Never invent earlier bounds just to satisfy validation.
  - **Proof:** independent small-array LL/Brier/paired-interval checks, an intentional missing-control failure, unknown/empty-profile failure and a case where full-cohort improvement cannot override a failed2026 primary contrast. Extreme valid probabilities receive the same policy in every output.

- [x] **S5 — Verify the integrated path before the72 data fits.**
  - **Targets:** `betting_app/tests/test_a0_probability_connector.py`, `test_a0_correction_experiment.py`, `test_a0_sports_correction_evaluation.py`, `test_research_benchmark_suite.py`, `test_model_benchmark.py` in the same directory.
  - **Work:** add only consumer-visible regression cases for new model identity, chronology, selection and gates. Run focused checks once the code is integrated; assemble bounded toy inputs and exercise actual CLI training, save/load and prediction in a new isolated output directory. Do not score alternative recipes on the primary cohort while debugging.
  - **Proof:** actual CLI outputs match the mathematical oracle; both learned arms, origins and controls are distinct; future-label/probability perturbations leave earlier membership, selected penalty, coefficients and predictions unchanged. Existing frozen recipe replay remains valid without rewriting any old artifact or receipt.

- [x] **S6 — Execute exactly the frozen sports experiment once.**
  - **Targets:** existing `scripts/train_a0_sports_correction.py` and `src.models.a0_probability_connector` entry points; new stage05/06/07/08 run directories.
  - **Work:** seal code/environment/input hashes, fit the72 registered heads, persist all coefficients/optimizer diagnostics/memberships and all rolling-validation predictions, choose penalties past-only, then export11,550 delivered rows. Evaluate the fixed recipe; no look-driven extra fit or selection grid.
  - **Proof:**72 real fit records,36 per arm,0 base fits; complete quarter validity/lineage; exact8,447 C0 warmup values; no reused target directory; selected artifacts account for every primary ID. Missing fits, unexpected dependencies or changed sources stop assessment as an integrity failure.

- [x] **S7 — Independently verify arithmetic and genuine inference.**
  - **Targets:** new model/prediction/report artifacts; existing canonical benchmark runner and the receipt-declared native C0/raw-specialist inference paths.
  - **Work:** replay every stored head prediction from saved coefficients. Verify all72 scalar optima with an independent objective/gradient implementation; recompute every headline contrast/guard and declared calibration cell from retained rows. Exercise genuine C0 and raw-specialist checkpoints on input-selected examples covering all three active origins, BO1/3/5, cold/empty histories and swapped sides. Check native/reused input hashes and end-to-end composition; do not claim a probability-only API is a live roster builder.
  - **Proof:** native smoke examples agree under source-recipe tolerances fixed before execution; all11,550 coefficient-level predictions reproduce within absolute1e−12, and independently recomputed headline LL/Brier values agree within1e−12. Freeze optimizer/KKT and upstream native tolerances from their numerical contracts before data fits. Validate complementarity, actual cold behavior and nonmutation. Run canonical suite/doctor and candidate scoring on full coverage. Record exact maxima and case counts; tests alone or a tolerance enlarged after failure do not establish replay fidelity.

- [x] **S8 — Seal one development decision and stop.**
  - **Targets:** new `decision.json`, separate execution report, `docs/RESEARCH.md`, this checklist and IDEA-027.
  - **Work:** publish every arm, three primary contrasts, eight guards, warmup/coverage accounting, sensitivities, selection lineage and actual computational cost. Keep failure/inconclusive results. Record calibration/EV/tournament claims separately.
  - **Outcome:** integrity issue → `BLOCKED_INTEGRITY`; valid execution with any failed/undefined required gate → `NOT_ADOPTED` with the reason; all gates/proofs pass → at most `DEVELOPMENT_CANDIDATE_RETAINED`. No canonical manifest change, production activation, automatic follow-on recipe or H fit follows. Mark this plan's sports tasks complete only for work actually executed.

## 4. Track E — Finish local decision-time capture and activation prerequisites

This track keeps **Plan06's frozen annual A0** as its sports reference. A successful C0 experiment must not silently replace that model in E3/H. Any later change of the forward sports recipe needs a new explicit outcome-blind protocol before affected data/assessment—not an edit to the frozen Plan06 record.

### 4.1 Data contracts to implement

**Source capture:** reuse the E3 `capture_contract` in [source_contracts.json](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/source_contracts.json). Required fields include source/capture IDs, evidence URI, SHA256, exact body bytes/base64 and representation/encoding, actual request/response/retrieval/first-observation clocks, nullable publication/effective clocks, revision, HTTP status, source class and availability basis. Store the bytes returned by the HTTP client before JSON/text parsing; a rendered DOM must remain explicitly typed as such. Missing publication is not supplied from response time. Genuine response receipt proves availability then, not at an invented earlier publication time.

Record failed, denied, malformed, timed-out and empty-body attempts with their actual status/clocks and explicit reason; do not manufacture a successful empty payload. Exclude credentials/cookies/API keys from request metadata and logs. Raw evidence stays in restricted local data storage, never Git. Secret-bearing responses must not be printed or silently rewritten into supposedly exact source evidence.

**Opportunity ledger:** separate append-only discovery/origin/decision/outcome events for **all provider fixtures**, including unknown IDs, missing roster, no quote, late discovery and failed inference. Reuse the evidence service/schema conventions; do not force a fake forecast into `ForecastEvidenceSnapshot`, whose current contract requires a real forecast envelope.

- Stable protocol/provider/fixture identity and a unique primary-origin identity; link schedule/capture revisions by immutable IDs/hashes.
- Preserve the first discovered record even if no nominal origin can yet be assigned. Record actual discovery time; do not backdate a fixture discovered after its T−1h origin.
- Before the origin is due, scheduled changes append new evidence. Once the primary origin is due/frozen, later schedule changes do not rewrite it or create another primary decision.
- Store all applicable rejection reasons separately from the primary decision/status, known/missing clocks, model/feature availability, information class, and optional successful forecast-envelope reference. Do not hide rows because the market was absent or the model could not predict.
- Append later result/start corrections with actual receipt clocks; never mutate the original origin/evidence. Rescore only under an explicitly versioned policy if required, preserving the previous view.
- A repeated immutable event key with identical canonical payload is idempotent; the same key with different payload is a conflict. Enforce at most one primary decision for the declared fixture/origin. Corrections are new linked events, not UPDATE/DELETE.

**Persistence design for the local slice:** extend `betting_app/services/forecast_evidence.py` and the existing prediction/evidence model area with separate source-capture and opportunity-event records, not a competing forecast store. Freeze exact schema names/keys in E1/E2 before migration. Retain the existing successful-forecast envelope and link to it. Content-addressed bytes must exist and hash-verify before a committed metadata record references them; recovery must identify orphan bytes or incomplete attempts without fabricating a forecast. A derived current-state view may be recomputed; immutable event history is authoritative.

### 4.2 Implementation tasks and proofs

- [x] **E1 — Implement exact-byte capture with dual-clock evidence locally.**
  - **Targets:** `betting_app/services/forecast_evidence.py`, source client boundary in `betting_app/services/oddspapi_service.py`, and the already named Plan06 source interfaces. No new provider endpoint or second quote parser.
  - **Work:** define one reusable capture envelope and persist raw bytes plus safe request/response metadata before parsing. Make capture callable with local response fixtures so no network is needed for implementation. Add explicit failures and unknown clocks. Freeze the stored representation and availability meaning; a source timestamp must never be called actual observation time.
  - **Proof:** independent hashes reproduce exact binary/JSON/text bodies; encoding/whitespace differences remain distinct; duplicate bytes do not erase distinct capture attempts. Malformed JSON, HTTP errors, absent body, clock contradictions and redacted secret metadata produce truthful failure/evidence records. Mutating parsed objects cannot alter retained bytes. Zero external requests.

- [x] **E2 — Persist the complete no-forecast opportunity ledger.**
  - **Targets:** `betting_app/models/prediction.py`, `betting_app/models/__init__.py`, `betting_app/services/forecast_evidence.py`, a new revision under `betting_app/alembic/versions/`, `docker/timescale/init.sql`, and existing evidence tests.
  - **Work:** inspect the then-current ORM/migration/bootstrap schema before adding the capture/opportunity tables and constraints. Use the established append-only, canonical-payload/hash and database `clock_timestamp()` conventions. Keep a successful forecast reference nullable rather than weakening the existing envelope validation. Do not write `canonical_predictions` or `model_ev_signals`. Migrate all affected local schema creation paths together; do not amend the old f1e2d3c4b5a6 migration or add a competing schema path.
  - **Proof:** on an **empty isolated PostgreSQL/Timescale database**, migration, ORM construction and bootstrap paths create compatible affected tables. Test idempotency, conflicting payloads, concurrency, one-primary-decision uniqueness, rollback, and SQL UPDATE/DELETE rejection. Missing-forecast fixtures remain queryable/exportable. No claim of protection from database-owner DDL/TRUNCATE; no serving-database migration.

- [x] **E3 — Exercise discovery through terminal opportunity recording offline.**
  - **Targets:** capture/ledger service, `betting_app/services/a0_shadow_readiness.py`, `src/analysis/roster_provenance.py`, `src/analysis/quote_provenance.py`, `src/analysis/production_market_benchmark.py`, and their existing fixture tests.
  - **Work:** run a bounded local batch containing eligible, unknown-team, wrong/withdrawn roster, unbound participant, latest suspended quote, one-book-only, stale quote, late discovery, rescheduled, started and failed-model cases. Preserve every fixture and all reasons. Link genuine successful envelopes where inputs permit; synthetic source fixtures remain `local_fixture` and cannot enter forward TRAIN/CAL/development.
  - **Proof:** input fixture universe equals the disjoint accounted-for terminal outcomes plus explicitly pending/not-yet-due records; every due discovered origin receives exactly one primary decision. Future labels/rosters/quotes/reschedules cannot change earlier records. Independent reconstruction from retained bytes reproduces selected quote state/orientation and rejection reasons. A zero-forecast batch is accounted for, not labelled reliable or admitted.

- [x] **E4 — Validate native tensors and a future-valid fixed-artifact contract.**
  - **Targets:** `src/analysis/a0_prediction_input.py`, `scripts/predict_causal_a0.py`, `betting_app/core/models/causal_a0.py`, `betting_app/services/a0_shadow_readiness.py`, and receipt-declared native history code.
  - **Work:** extend the already repaired fixture-input path into a release-aware snapshot builder for genuinely captured history/roster inputs. Preserve canonical player-slot alignment, announced/exact-context roster precedence and retractions, cold states, all native modes, nonmutation and no future draft/side. Validate native tensor shape/order/dtype/masks/normalizers against the immutable source recipe, not an EXP081 feature approximation.
  - **Future artifact:** explicitly version and validate the serving contract for the **unchanged frozen2026 artifact** if it is to serve later cutoffs; freeze allowed input/calendar ranges, training/calibration identity and causal history update policy. Keep the current annual adapter's wrong-year rejection intact. Do not invent a2027-trained artifact, extend the July connector head past October1, or silently rotate to C0. If future-valid serving cannot be demonstrated under a separate explicit contract, activation remains blocked.
  - **Proof:** real native-vs-adapter parity on input-selected BO1/3/5 and optional-context/cold cases; repeated queries leave state unchanged; future perturbations leave earlier tensors/forecasts unchanged. The legacy adapter still rejects unsupported years, while only an explicitly supported new fixed-artifact contract can admit its declared range. Record the exact artifacts and tolerances, no financial-quality inference.

- [ ] **E5 — Perform only separately authorized, capped source probes.**
  - **Prerequisites:** E1–E4 local proof, explicit source-specific acquisition authorization, permitted retention/access, actual entitlements/quota, isolated capture persistence and named official locators. Existing code defaults are not account limits or permission. Official roster/unit and actual-start/result locators are currently unresolved; do not invent them.
  - **Targets:** the exact eight E3 source contracts: OddsPapi fixtures, Pinnacle via OddsPapi, TOTALbet, Lebull, GOLGG completed history, Liquipedia context-only, official unit/announcement, and official actual-start/settlement.
  - **Caps from the frozen source contract:** OddsPapi at most2 requests; TOTALbet1; Lebull4; sports at most1 per approved locator; unregistered locators0. No retries/pagination beyond the authorized cap; no whole scraper/synchronizer or configured database defaults. Select the first eligible scheduled fixture by `(start_time, exact fixture_id)`, never by odds or outcomes.
  - **Proof:** authentic captures establish source access, required participant/state fields and observed coverage—or record a precise blocker. Verify both sides and distinct underlying book identities. Liquipedia remains contextual, not authoritative announced-five/start evidence. Missing binding, insufficient quota, denied access or an unknown required locator stops that probe and blocks the relevant activation gate; there is no fallback to an unapproved book.
  - **Executed gate:** BLOCKED. Local E1–E4 proof is complete, but the eight contracts remain unapproved/unverified; official locators, retention/access permission and measured entitlements are absent. **Zero external requests.** No request cap was consumed against a real source.

- [x] **E6 — Make one forward-activation decision, not a reliability claim.**
  - **Targets:** new activation receipt linked to the immutable E3 protocol/source contracts, the complete opportunity ledger and quota/capture/tensor proofs.
  - **Required gates:** authorized sources/official locators; exact-byte clocks; authentic identity/participant bindings; independent actual-start/results; at least two distinct books with proven request capacity; live native tensors; future-valid unchanged-artifact contract; complete immutable no-forecast ledger.
  - **Contract:** one T−1h primary origin per fixture, quote receipt age<=300seconds, forecast/persistence/decision within60seconds after cutoff and before independently evidenced actual start. Latest invalid/suspended state cannot revive an old price. Retain all fixtures discovered in the approved feed; this is not worldwide coverage.
  - **Proof/decision:** `BLOCKED` with named unmet gates, or `READY_FOR_AUTHORIZED_FORWARD_ACQUISITION` with all evidence linked. Readiness is not collection, G0 passage, H fitting, deployment or profit. Start actual collection only under its explicit authorization and frozen calendar. Administrative/integrity monitoring does not expose protected outcomes for model selection.
  - **Sealed decision:** [BLOCKED](../data/08_reporting/betting_ev_reliability/forward_capture_20261002_033658/decision.json). Authentic two-book bindings/capacity, official start/results and live native-input operation remain unproved. Local exact-byte/ledger/native compatibility gates pass; no acquisition activation follows.

### 4.3 Frozen dates and incompatible conventions

All E3 dates are UTC, start-inclusive/end-exclusive, and membership follows the immutable nominal cutoff:

| Partition | Origin interval | Label lock |
|---|---|---|
| TRAIN |2026-11-01 →2027-11-01|2027-11-15|
| CAL |2027-12-01 →2028-06-01|2028-06-15|
| DEVELOPMENT |2028-07-01 →2029-07-01|2029-07-15|

November2027 and June2028 are buffers. Readiness and acquisition authorization must be complete **before2026-10-31T00:00Z** for this version's November1 start. Otherwise record `EXPIRED_UNSTARTED`; do not silently roll dates or retrospectively include a development period. A later schedule requires a separately versioned protocol before collection. Thirty observation months over32calendar months is neither a collection commitment nor a precision guarantee.

Keep E3/Plan05's **seed20260929**, tax0.12, strict raw net EV>0.05, one-unit returns, exact cross-side ties preferring A, symmetric epsilon1e−6 and all fixed bands. They differ from Track S's unclipped forecasts and the old1,619-fixture24-hour-freshness diagnostic. Do not merge those contracts or transfer admission between them.

## 5. Track F — Later development and genuine confirmation

- [ ] **F1 — Freeze a separate future sports confirmation protocol.**
  - **Prerequisites:** S8 complete, explicit choice/authorization of the recipe to confirm, future-valid genuine input/artifact support and a future assessment start not yet observed. A failed S experiment does not turn the retained older candidate into a confirmed model.
  - **Work:** freeze the complete model/normalization/calibration/refit schedule, model identity, comparator set, all-fixture admission, one fixed assessment horizon, proper-score/reliability/coverage gates and source/edition sensitivity before the first assessment forecast. Use a future-valid head and matching C0/raw artifacts; the July2026 artifact cannot simply be reused for Q4 or a new year. Keep missed/missing forecasts and actual creation clocks.
  - **Proof/outcome:** genuine issued-before-start paired forecasts and independent arithmetic support rejection, inconclusive evidence or a supported confirmation claim of its stated scope. Past-only refitting schedules must themselves be frozen; no outcome-driven early stop or extension. Do not substitute a repeatedly viewed nominal95% interval for a preregistered stopping rule. Exact dates/sample sufficiency are fixed at authorization; no future rows or results are fabricated by this plan.
  - **Current blocker:** no authorized confirmation recipe/horizon or future-valid matched C0/raw/head inputs. The new head is NOT_ADOPTED and expires at2026-10-01; no assessment dates or observations were fabricated.

- [ ] **F2 — Run Plan05's H/control development only after its admission gates pass.**
  - **Prerequisites:** E6 and authorized real collection; complete source/input/recipe evidence; frozen TRAIN/CAL memberships with released labels and adequate fit support; explicit execution authorization. Preserve all rejected/missing opportunities. Zero admissible rows or inadequate support means stop, not fit on scenarios.
  - **Targets:** existing Plan05/06 analysis/model helpers and `scripts/run_unified_betting_evaluation.py`; no second EV engine.
  - **Work:** retain S as the declared frozen annual A0 unless a new protocol explicitly changes it beforehand. Fit the already specified symmetric `H=sigmoid(a*logit(p_s)+b*logit(q))`, `a,b>=0`, shrunk toward `(0,1)`, with S-cal, M, M-cal, M-shape and Fixed50. The frozen penalties are `{0.01,0.1,1.0}`; the four learned recipes H/S-cal/M-cal/M-shape receive the same three-penalty opportunity, with M-shape's fixed knot `logit(0.8)`. Fit all admitted TRAIN fixtures, choose by CAL mean LL with stronger-penalty ties, no TRAIN+CAL refit, assess development once. No additional model/threshold/Kelly sweep.
  - **Proof/gates:** Plan05's sequential LL/Brier market-control gates, then selected-return error95% interval wholly within **±2pp** for both all selections and the **(5%,10%]** band; separate positive after-tax value and incremental value over market controls. Preserve complete monthly universes, zero-bet months and undefined replicates. Report unchanged S-selected IDs/sides/prices separately from every own-policy selection. A better sports score or a mean near zero cannot replace interval containment.
  - **Outcome:** rejected, inconclusive, no added sports value or at most a development candidate. No automatic betting/production activation or E4 confirmation follows.
  - **Current blocker:** E6 remains blocked; zero admitted released TRAIN/CAL observations, zero H/control fits. Synthetic software fixtures cannot fill this gap.

- [ ] **F3 — Execute separately authorized betting confirmation after recipe freeze.**
  - **Targets:** Plan06 E4 / Plan05 confirmation protocol and immutable production-independent evidence/decision ledger.
  - **Work:** freeze genuinely future dates, final selected head/control/selection/tax/settlement rules and monitoring/stopping policy before collecting the assessment. The existing separate12-month concept is not an adequate-precision guarantee; do not relabel TRAIN/CAL/development as confirmation. Continue to report all opportunities and unavailable decisions.
  - **Proof/outcome:** independently reproduce the proper-score, selected-reliability and positive-value gates with real information availability and recorded decisions. Sparse support, no bets or a wide interval cannot pass. Current illustrative IID counts25,516/9,843 assume zero future bias/stable variance and are not targets that guarantee reliability. More data alone cannot remove the currently observed bias. No automated wagers, hidden outcome peeking or deployment is part of this plan.
  - **Current blocker:** no final authorized betting recipe, future confirmation calendar or genuine assessment evidence/support; development periods cannot become confirmation.

## 6. Track T — Preserve the separate tournament objective

- [ ] **T1 — Specify and verify a tournament distribution contract before any promotion.**
  - **Prerequisites:** separate tournament scope and authenticated edition/rulebook/origin data; declared match/scoreline model suitable for the required outputs. A series-winner connector alone is insufficient.
  - **Targets:** existing `src/models/` tournament modules and `scripts/simulate_tournament.py`, with the affected format-specific tests selected from the actual implementation at execution time.
  - **Work:** enumerate editions, origins, phases, legal matchups, tiebreaks, transitions, roster policy and absent coverage. Define the joint scoreline/state distribution and placement/win-count targets before assessing. Never infer IID map probabilities from the pooled series winner, project BO twice, use actual future opponents or assume a scoreline/placement distribution from match LL.
  - **Proof/outcome:** independent small-format oracles, probability mass/joint constraints, legal transitions, future-result/draft/roster perturbation invariance and edition-level evaluation. Record missing editions and uncertain rules. No tournament or commercial-futures certification follows from S8, E6 or a source-label bootstrap.
  - **Executed subset:** [contract and local mechanics](../data/08_reporting/plan09/plan09_execution_20261002_033658/tournament_gate_decision.json) complete. Catalog91 profiles/234 phases/227 configured/**0 historically ready**. Authentic edition evaluation remains NOT_RUN without official editions/rulebooks/origins/rosters and an eligible coherent joint-scoreline adapter.

## 7. Artifact and verification contracts

### 7.1 Artifacts per executed track

Executed runs: sports `c0_raw_connector_20261002_033658`, evidence `forward_capture_20261002_033658`, integration `plan09_execution_20261002_033658`. The stage templates below use those run IDs. Only later F models/partitions remain conditional; no fabricated fit-support artifacts exist.

| Stage | Sports S | Evidence/forward E/F |
|---|---|---|
|01 raw|No new sports source collection|Exact immutable source bytes and capture failures under `data/01_raw/betting_ev_reliability/<run_id>/`|
|03 primary|Reuse canonical IDs, never overwrite|Opportunity/event exports, evidence identities, admission/missingness under `data/03_primary/betting_ev_reliability/<run_id>/`|
|04 feature|Reuse frozen base inputs|Versioned native snapshot validation and exact causal tensors when authorized|
|05 model input|`data/05_model_input/a0_sports_correction/<run_id>/`: protocol, joined input, memberships, hashes, execution snapshot|`data/05_model_input/betting_ev_reliability/<run_id>/`: capture/admission/activation protocols and actual partition memberships|
|06 models|`data/06_models/a0_sports_correction/<run_id>/`:72 fitted heads, diagnostics and selected artifacts|Only separately authorized F2 models; capture/ledger work does not fit a model|
|07 model output|`data/07_model_output/a0_sports_correction/<run_id>/`: all11,550 rows, CV grid, candidate benchmark export|All opportunities/statuses and genuine forecasts when available, never only successful predictions|
|08 reporting|`data/08_reporting/a0_sports_correction/<run_id>/`: scores, gates, replays, independent verification and decision|`data/08_reporting/betting_ev_reliability/<run_id>/`: capture/schema/admission/activation/fit/confirmation receipts as actually executed|

Each decision records actual timestamps, protocol/code/input/model hashes, environment/dependency versions, exact coverage, model identity, attempts and fit counts, verification actually run, limitations and next authorized state. Store raw arrays/datasets/checkpoints in ignored data paths, not Git. Do not mutate the sealed next-step research report or old decisions when writing a new execution report.

### 7.2 Executed interfaces and reproduction boundary

The following help commands already exist and are safe discovery commands; they do not execute this plan:

```bash
.venv/bin/python -m src.models.a0_probability_connector --help
.venv/bin/python scripts/run_model_benchmark.py --help
.venv/bin/python scripts/predict_causal_a0.py --help
```

**Implemented path:** explicit C0/control schema, three primary contrasts and eight guards now run through the existing sports-correction CLI family. Local capture/ledger services and the explicit fixed2026 native CLI contract are implemented. There is no generic authenticated-admission CLI or deployed collector implied by this document.

The following existing CLI family was exercised after S1–S5. `SPORTS_PROTOCOL` identifies the separately sealed C0 protocol; `RESEARCH_ROOT` resolves the existing archive; `CANDIDATE_DATA` is the complete truthful candidate export. Outputs are immutable: do not rerun training into the completed directories or treat these commands as authorization for another experiment. The suite does not forward legacy `--model-name`/`--model-version` options:

```bash
.venv/bin/python scripts/train_a0_sports_correction.py train --protocol "$SPORTS_PROTOCOL" --research-root "$RESEARCH_ROOT"
.venv/bin/python scripts/train_a0_sports_correction.py evaluate --protocol "$SPORTS_PROTOCOL"
.venv/bin/python scripts/run_model_benchmark.py --suite --doctor --research-root "$RESEARCH_ROOT"
.venv/bin/python scripts/run_model_benchmark.py --suite --research-root "$RESEARCH_ROOT" --candidate-data "$CANDIDATE_DATA" --candidate-col p_c0_connector --output-dir "$BENCHMARK_OUTPUT"
```

The canonical doctor passed and the candidate benchmark retained11,550 rows/zero absences as DEVELOPMENT_ONLY. The benchmark does not substitute for the three-control/eight-guard evaluator or imply promotion. C0 is never supplied under a false A0 identity.

Focused sports suites after implementation, with the test environment confirmed non-serving/isolated:

```bash
.venv/bin/python -m pytest -q betting_app/tests/test_a0_probability_connector.py betting_app/tests/test_a0_correction_experiment.py betting_app/tests/test_a0_sports_correction_evaluation.py betting_app/tests/test_research_benchmark_suite.py betting_app/tests/test_model_benchmark.py
```

Evidence changes additionally require affected existing `test_forecast_evidence.py`, `test_a0_prediction_input.py`, `test_a0_prediction_input_persistence.py`, `test_causal_a0_adapter.py` and source parser/provenance suites under `betting_app/tests/`, plus new behavior cases for capture/opportunity persistence. Before any DB-backed command, resolve and assert the target is an empty isolated PostgreSQL/Timescale instance; never inherit the user's configured database. Provision from the then-current supported test setup, not guessed credentials or a serving Compose stack. Existing migration-fixture limitations must be reported if encountered; they are not permission to bypass proof or repair unrelated schema opportunistically.

The execution report records actual training/evaluation/benchmark/native CLI runs, three isolated schema paths and168 distinct final focused tests. No full-application regression, live source probe or serving migration ran. The task-owned isolated database was archived to stages01/03 and stopped.

## 8. Stop conditions and completion checklist

| Condition | Required action |
|---|---|
|Unknown mandatory model/source identity, changed pinned bytes, row mismatch, future leakage or replay failure|Stop affected slice as integrity-blocked; preserve evidence; do not shrink cohort or widen tolerance|
|Any sports primary/guard gate fails or lacks support|Seal NOT_ADOPTED with negative/inconclusive result; no automatic next recipe|
|Missing source authorization, official locator, quota, participant binding, native tensors or future-valid artifact|Keep activation blocked; finish reachable local software work, do not scrape or invent missing evidence|
|2026-10-31T00:00Z readiness deadline missed|Expire E3 version unstarted; no quiet date roll|
|Too few admitted/selected observations, no bets, undefined/wide reliability intervals|Report insufficient evidence; do not relax±2pp or call it success|
|Successful local capture/schema/tensor tests|Software proof only; no source authentication, predictive or financial certification|
|Positive historical sports result|Development candidate only; no silent canonical/production/forward-recipe replacement|

**Definition of completion for each executed slice:** concrete deliverable exists; the actual changed interface was exercised; affected behavioral tests and independent oracles ran with explicit results; all IDs/absences reconcile; protocol/input/artifact hashes are sealed; report and handoffs state actual scope and limitations; only genuinely completed task boxes are checked. Once smoke proof is obtained, remove temporary runners/debug scaffolding while retaining the reproducible recipe and evidence. Do not leave a job described as running after it has stopped.

**Current state:** S1–S8 completed once:72 historical scalar heads/0 base fits,11,550 delivered rows/3,103 primary, three primary passes but two annual-A0 replacement guard failures → **NOT_ADOPTED**. E1–E4 local proof and one BLOCKED E6 decision are complete. E5/F1–F3/authentic T1 are explicitly blocked, not completed. No jobs remain active; no next recipe or live acquisition starts automatically. Full results, negative attempts, replay maxima, preserved inputs and prerequisite receipts are linked in the [execution report](../reports/plan09_c0_connector_and_forward_evidence_20261002.md).
