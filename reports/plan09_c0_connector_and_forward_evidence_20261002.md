# Plan09 execution: C0 connector and local forward evidence — 2026-10-02

## 1. Decision and exact scope

**Sports: COMPLETED / NOT_ADOPTED. Local evidence E1–E4: verified. Forward activation: BLOCKED.** This executes the user's “Orchestrate Execute plans” request under the [scope receipt](../data/05_model_input/plan09/plan09_execution_20261002_033658/scope.json) and [Plan09](../TODO/09_c0_connector_and_forward_evidence.md). It does not declare all future/edition work complete.

- Sports run: `c0_raw_connector_20261002_033658`; [frozen protocol](../data/05_model_input/a0_sports_correction/c0_raw_connector_20261002_033658/protocol.json), [sealed decision](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/decision.json).
- Evidence run: `forward_capture_20261002_033658`; [implemented local contract](../data/05_model_input/betting_ev_reliability/forward_capture_20261002_033658/evidence_contract.json), [activation decision](../data/08_reporting/betting_ev_reliability/forward_capture_20261002_033658/decision.json).
- Integration receipts: [`plan09_execution_20261002_033658`](../data/08_reporting/plan09/plan09_execution_20261002_033658/). **168 distinct final focused tests passed**, one existing Starlette warning; not a full application-suite claim.
- The three primary comparisons pass, but the BO5 and source-unseen replacement guards versus annual A0 fail their prespecified +0.005 upper-endpoint requirement. This is failure to establish the required noninferiority, not proof of harm or equivalence. No post-score variant or promotion.
- Canonical annual A0, frozen base models, benchmark manifest, operational EXP081/hybrid, old10/10 journal and Plan05/06 dates remain unchanged. No live source acquisition, H fitting, deployment, serving-database migration or wagers. **Zero genuine future forecasts.**

## 2. Frozen experiment and actual cost

The only new data fits were **36 scalar C0/raw connector heads +36 scalar C0-only calibration controls =72**, zero base-model fits and no final refit. Nine origins: 2024Jul/Oct, 2025Jan/Apr/Jul/Oct and 2026Jan/Apr/Jul. Four penalties: 0.0001/0.001/0.01/0.1. Date and effective label release must both precede each origin, minimum500 rows. Past-validation LL selects the penalty; ties within1e−12 prefer the stronger penalty.

Candidate: `p=(1-alpha)*p_C0+alpha*p_raw`, alpha in[0,1], objective meanLL +0.5*lambda*alpha². Control: `sigmoid(s*logit(p_C0))`, s≥0, penalty0.5*lambda*(s−1)². C0 and raw are already series probabilities; no second BO projection, intercept, optional-mode fabrication or odds input. C0 is the distinct native12 TRAIN/CAL/assembly recipe, not a renamed annual A0.

| Delivered origin | Candidate alpha | Candidate lambda | Control slope | Control lambda |
|---|---:|---:|---:|---:|
|2026-01-01|0.1769033853776533|0.0001|1.0035552617964951|0.1|
|2026-04-01|0.18268224240244837|0.0001|1.0076469407114867|0.1|
|2026-07-01|0.19731172697778213|0.0001|1.0031736237214188|0.1|

Each selected head ends at the next quarter, exclusively. The July head is **not valid on/after2026-10-01**. Coefficient-only public inference requires matched genuine C0/raw series probabilities and exact declared model identity, BO and validity range; it is not a live roster builder.

All11,550 canonical targets are retained. Primary assessment is the3,103 exposed2026 rows through September8. Earlier8,447 candidate/control rows remain **exactly C0 warmup**; connector03 instead has A0 warmup. The raw-trio design previously used2024–2025H1 selection; earlier weight-OOF rows are not selection-unseen confirmation. All years remain development-exposed; nominal intervals do not adjust for the prior search history.

Pre-data CLI proof fitted144 additional **synthetic** scalar heads across baseline/future-perturbed toy inputs, not extra historical recipes. The72 historical fits ran once. Independent numerical root solves are verification, not additional candidate selection.

## 3. Scores, contrasts and replacement guards

### Primary2026 — all3,103 rows

| Model | LL | Brier | N |
|---|---:|---:|---:|
|Canonical annual A0|0.536756581|0.181572637|3103|
|Native12 C0|0.535448701|0.181034493|3103|
|C0-only calibration|0.535480256|0.181039747|3103|
|New C0 connector|0.534477081|0.180552711|3103|
|Retained connector03|0.535644479|0.181012213|3103|
|Raw specialist|0.543863352|0.183493134|3103|

Exactly5,000 shared calendar-month bootstrap draws, seed20261002, nine primary months. Three gates require upper LL endpoint<0 and Brier point≤0:

| Candidate minus | ΔLL | Monthly95% interval | Equal-month ΔLL [95% interval] | ΔBrier |
|---|---:|---|---|---:|
|Native12 C0|-0.000971619|[-0.001681791, -0.000265418]|-0.001445642 [-0.003148836, -0.000279386]|-0.000481782|
|C0-only calibration|-0.001003174|[-0.001715505, -0.000290138]|-0.001459997 [-0.003138863, -0.000310821]|-0.000487037|
|Retained connector03|-0.001167398|[-0.002679503, -0.000008773]|-0.001344418 [-0.002689403, -0.000089477]|-0.000459502|

All three pass. Six of the eight required guards pass; all required actual guard resamples had defined support. Empty-support cases are rejected by the contract rather than dropped or counted as zero.

| Profile | Comparator | N | ΔLL | Upper95% endpoint | Pass (<0.005) |
|---|---|---:|---:|---:|---|
|best_of_3|Canonical annual A0|1873|-0.002861164|-0.001155738|yes|
|best_of_3|Native12 C0|1873|-0.001177347|-0.000466299|yes|
|best_of_5|Canonical annual A0|485|-0.000291740|+0.005183914|NO|
|best_of_5|Native12 C0|485|-0.000014052|+0.003731397|yes|
|major|Canonical annual A0|704|-0.000804577|+0.000899840|yes|
|major|Native12 C0|704|-0.000760212|+0.000489058|yes|
|source_unseen_player|Canonical annual A0|334|-0.004436504|+0.005193668|NO|
|source_unseen_player|Native12 C0|334|-0.002122585|+0.001214798|yes|

**Decision: NOT_ADOPTED.** Both failing checks are against annual A0. A lower global mean does not override profile replacement guards.

### Sensitivity —143 source-tournament-name/year proxies

| Comparator |143-proxy match-weighted LL interval|Equal-proxy ΔLL [95% interval]|
|---|---|---|
|Native12 C0|[-0.001955087, +0.000005193]|-0.000711506 [-0.002814551, +0.001194921]|
|C0-only calibration|[-0.001986579, -0.000027343]|-0.000754391 [-0.002881042, +0.001166763]|
|Retained connector03|[-0.002426466, +0.000098258]|-0.001306370 [-0.003359242, +0.000716464]|

Match-weighted proxy intervals cross zero versus C0 and connector03. All equal-proxy intervals cross zero. These are source-label proxies, **not independently certified tournament editions**; this sensitivity cannot rescue a failed primary/guard gate.

### Full and warmup cohorts — descriptive, not extra primary assessments

| Model | LL | Brier | N |
|---|---:|---:|---:|
|Canonical annual A0|0.551245583|0.187102933|11550|
|Native12 C0|0.550350935|0.186761019|11550|
|C0-only calibration|0.550359413|0.186762430|11550|
|New C0 connector|0.550089902|0.186631584|11550|
|Retained connector03|0.550946808|0.186952371|11550|
|Raw specialist|0.559324389|0.190068168|11550|

| Model | LL | Brier | N |
|---|---:|---:|---:|
|Canonical annual A0|0.556568108|0.189134484|8447|
|Native12 C0|0.555825262|0.188864654|8447|
|C0-only calibration|0.555825262|0.188864654|8447|
|New C0 connector|0.555825262|0.188864654|8447|
|Retained connector03|0.556568108|0.189134484|8447|
|Raw specialist|0.565003991|0.192483502|8447|

The full-cohort change mixes C0 warmup with the learned2026 head. Comparing it directly with connector03's A0 warmup would conflate the anchor replacement and learned-head increment.

## 4. Calibration, support and tails

| Model | Accuracy | AUC | ECE10 | Fixed-A0 quarter-cell RMS | LL≥2.5 |
|---|---:|---:|---:|---:|---:|
|Canonical annual A0|0.718337|0.798125|2.2047%|2.9716%|10|
|Native12 C0|0.720271|0.799299|2.2855%|2.9356%|12|
|C0-only calibration|0.720271|0.799309|2.4051%|2.9474%|12|
|New C0 connector|0.723816|0.800464|2.5280%|2.9557%|10|
|Retained connector03|0.719626|0.799477|2.0697%|3.0018%|10|
|Raw specialist|0.720915|0.794315|1.9123%|3.3702%|13|

The primary fixed-A0 quarter×confidence diagnostic has15 cells,3,099 non-neutral rows and4 neutral rows. The connector's fixed-A0 favorite confidence-minus-win-rate gap is0.5190pp, versus C0's0.7198pp; its own-favorite gap is0.0991pp, with a different favorite label. Pointwise ECE and fixed-cell RMS worsen versus C0. The connector has417≥90% favorites,17 upsets versus23.61 expected. These summaries do not establish uniform calibration improvement, global overconfidence, a flattening remedy or per-match reliability.

The evaluator retains profiles by origin/year, BO, family/tier, history/experience, co-play, unknown players/new organization, W20, confidence and disagreement, with unknown/missing support retained. Independent verification covers LL/Brier, paired intervals,174 profile groups and110 declared favorite-calibration cells; it does **not** independently certify every auxiliary AUC/accuracy/ECE/MCE/slope/intercept or binned Brier-decomposition statistic. These remain evaluator outputs, not extra oracle coverage.

## 5. Separate archived selected-EV diagnostic

This optional analysis was chosen **before fitting**. It reuses the exact1,619 Pinnacle T−1h / strict-prior-day /24-hour-freshness scenario,328 fixed A0 choices and74 original(5%,10%] choices,12% tax, unit returns. It is not the300-second future capture contract, the old387 prior-context selections, application qualification, certified issued forecasts or executable betting evidence. Nine monthly blocks and111 source-name/year proxies retain the full opportunity universe.

| Policy/model | Band | N | Predicted net return | Realized unit return | Overstatement [monthly95% interval] |
|---|---|---:|---:|---:|---|
|fixed_a0 / Canonical annual A0|all_EV_gt_5pct|328|37.2463%|-9.0598%|46.3061pp [28.8807, 68.2387]|
|fixed_a0 / New C0 connector|all_EV_gt_5pct|328|34.8635%|-9.0598%|43.9233pp [26.0968, 66.2183]|
|fixed_a0 / Canonical annual A0|EV_5_to_10pct|74|7.1252%|-21.3768%|28.5020pp [1.4804, 48.5423]|
|fixed_a0 / New C0 connector|EV_5_to_10pct|74|6.6856%|-21.3768%|28.0624pp [0.3684, 49.1895]|
|own / New C0 connector|all_EV_gt_5pct|330|35.9511%|-9.0099%|44.9609pp [25.6891, 66.7309]|
|own / New C0 connector|EV_5_to_10pct|76|7.5308%|-15.0395%|22.5703pp [2.1464, 39.8285]|

Fixed IDs/sides/prices have identical realized returns; changes there measure probability/EV attribution, not changed PnL. The connector's own330 choices and76 near-band choices are separate selections. Both the all-selection and near-band ±2pp reliability requirements remain unmet. No Kelly portfolio, market head, H fit, financial certification or outcome-selected policy was added.

[Market results](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/market/results.json) contain all24 diagnostics and model/market scores. [Independent market replay](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/market_verification_attempt_02.json) reconstructs11,973 source rows, fixed/own identities,4,974 selected diagnostic rows and Decimal unit payoffs (max discrepancy1.78e−15), plus the5,000-draw intervals and preserved hashes.

## 6. Sports implementation and executed proof

Existing paths were extended rather than replaced by a dated training implementation:

- `src/models/a0_probability_connector.py`: explicit C0 runtime identity, convex head and identity-centred nonnegative logit control, serialization/validity checks.
- `src/models/a0_correction_experiment.py`: keyed source preparation, strict frozen protocol, memberships/selection,72-fit accounting and complete benchmark export.
- `src/analysis/a0_sports_correction.py` and `a0_correction_market.py`: explicit comparator bindings, three primary contrasts/eight guards, preserved fixed-A0 cells and separately fixed/own selections.
- `scripts/train_a0_sports_correction.py`: existing prepare/train/evaluate/predict CLI supports the distinct C0 protocol; prior sealed A0 recipes remain separate.

Executed entry points (recorded runs are immutable; **do not rerun training into these directories**):

```bash
SPORTS_PROTOCOL=data/05_model_input/a0_sports_correction/c0_raw_connector_20261002_033658/protocol.json
.venv/bin/python scripts/train_a0_sports_correction.py train --protocol "$SPORTS_PROTOCOL"
.venv/bin/python scripts/train_a0_sports_correction.py evaluate --protocol "$SPORTS_PROTOCOL"
.venv/bin/python scripts/run_model_benchmark.py --suite --doctor
```

The canonical doctor returned READY; the complete candidate benchmark returned DEVELOPMENT_ONLY,11,550 coverage and0 missing. Original039 remains9,907 eligible/1,643 missing, not silently reduced to a common candidate cohort. [Benchmark report](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/canonical_benchmark/REPORT.md).

| Proof | Observed result | Receipt |
|---|---|---|
|Input preflight|208 hashes; exact11,550 IDs/orientation/BO/outcomes and all9 release-bounded memberships|[preflight](../data/08_reporting/plan09/plan09_execution_20261002_033658/c0_input_preflight.json)|
|Real CLI toy|144 synthetic heads;17 negative contract checks, immutable-output collision and future perturbation; mathematical agreement1e−12|[CLI proof](../data/08_reporting/plan09/plan09_execution_20261002_033658/c0_cli_smoke.json)|
|Independent scalar/artifact/evaluation oracle|72 optima,11,550 delivered predictions exactly reproduced; max coefficient difference3.03e−15; saved KKT violation2.80e−16;4,057 score/interval scalar checks,174 profile groups,110 cells|[arithmetic](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/arithmetic_verification_attempt_04.json)|
|Genuine native C0/raw checkpoints|12 cases covering active origins/BO1/3/5, swapped sides,5 source-unseen cases and2 natural empty-history cases; C0 max saved-p difference1.13e−8, raw1.71e−8|[native replay](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/native_replay.json)|
|Additional genuine cold cases|Raw ten-new-player/two-new-organization forecasts exactly0.5; C0 source-free native-bank tensors remain within2.91e−9 of0.5, including explicit BO5 scenario|[cold extension](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/native_cold_extension.json)|
|Saved wrapper composition|All6 selected candidate/control wrappers via actual public CLI,12 native cases and swaps; pure math1e−12 and inherited upstream bounds; no mutation|[composition](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/head_composition_verification.json)|

C0 replay used the predeclared2e−6 native tolerance; raw retained the inherited4.88155e−6 caveat before any historical fit, not a widened post-failure tolerance. Observed errors here are much smaller; no universal bitwise or explained-numerical-cause claim. C0 cold proof uses genuine native-bank inputs; it does not claim a new arbitrary-player-ID C0 feature builder.

## 7. E1–E3: exact bytes, immutable missingness and real local inference

`betting_app/services/forecast_evidence.py` now exposes exact-byte `capture_source_attempt`, `load_source_capture`, `audit_source_capture_store`, immutable `append_opportunity_event` / `load_opportunity_events`, bounded `orchestrate_local_opportunity_batch`, and `export_opportunity_ledger`. `OddsPapiClient.get_captured` is an explicit opt-in boundary with caller transaction and budget; ordinary `get`/live callers are not activated.

Separate `source_captures` and `opportunity_events` tables extend the existing prediction model area. New Alembic revision `d4e5f6a7b8c9` follows `f1e2d3c4b5a6`; ORM, migration and `docker/timescale/init.sql` agree. Body bytes are hashed/fsynced before SQL references, stored content-addressed with owner-only permissions. Attempt identities, response/retrieval/actual observation/first-observation clocks, unknown publication, status and encoding remain distinct. Canonical request metadata strips credentials; exact body bytes are not silently redacted. Recovery identifies orphan bodies, interrupted staging files, missing bodies and hash mismatches.

The ledger preserves unknown IDs, absent/withdrawn/conflicted rosters, malformed/unbound/suspended/stale quotes, insufficient or conflicting book identities, late discovery, starts, pending origins and failed/unavailable inference. Exact fixture/origin/roster/source-capture/model-envelope binding is required. A callback savepoint prevents partial failed envelopes from contaminating successful fixture accounting. Both genuine generated_at and database recorded_at must fall within the60-second origin window and before the known start; actual completion is checked again after inference. A stale supplied evaluation time cannot make late inference eligible. Once fired, the primary origin survives later missing or future-only schedule vintages. Every local fixture remains `forward_admitted=false`.

Executed proof:

- Three **fresh empty isolated PostgreSQL/Timescale schema paths**: ORM42 / migrated39 / bootstrap33 tables, six evidence immutability triggers. Runtime proofs cover concurrent identical-body captures retaining earliest observation, concurrent batches yielding one primary, rollback, parent/FK/clock constraints, nullable forecasts, exports and direct SQL UPDATE/DELETE rejection. Database-owner DDL/TRUNCATE remains outside the guarantee.
- Actual owned-loopback HTTP success403/malformed/repeat sequence: four requests in the final passing run, fifth blocked before transport. A previous verifier-only failure made four more local requests: **eight local requests total, zero external/provider requests**. Failed-attempt bytes and receipts remain retained, not hidden.
- Final genuine native local batch: **18 fixtures =16 primary +2 pending;1 actual frozen-native forecast and17 without a forecast;0 admitted.** Four callbacks exercise native success, model unavailable, genuine invalid-tensor failure and rejection of a foreign envelope. The genuine success is `p=0.8145407684821454`; quotes, announcements and receipt clocks are declared local fixtures, not authentic publication.
- Independent retained-byte replay:162 body hashes,58 sidewise quote-state/orientation checks,16 primary reason sets and2 pending;56 earlier events and the native envelope remain unchanged after future schedule/withdrawal/suspended-price/outcome additions. Zero model calls in this future mutation replay.
- Behavioral red cases exposed and fixed actual-clock precedence, incomplete-body recovery, source metadata idempotence, revision-parent/timeout clock constraints, malformed fixtures, book-identity conflicts, frozen-origin lookup and post-callback deadline enforcement. Final focused runs passed96,84 and29 cases with overlaps: **168 distinct tests**, not209 independent cases.

Receipts: [HTTP](../data/08_reporting/plan09/plan09_execution_20261002_033658/http_capture_verification.json), [three-schema runtime](../data/08_reporting/plan09/plan09_execution_20261002_033658/schema_runtime_verification.json), [final native orchestration](../data/08_reporting/plan09/plan09_execution_20261002_033658/local_native_orchestration_attempt_03.json), [independent byte replay](../data/08_reporting/plan09/plan09_execution_20261002_033658/local_ledger_independent_replay.json). The earlier schema/replay receipts identify their exact source versions; final native orchestration and completion-boundary regression exercise the final service version. Do not present the entire earlier proof as one later binary snapshot.

## 8. E4: unchanged native artifact with explicit future contract

`src/analysis/a0_prediction_input.py` consumes hash-verified captured history/roster records, respects actual availability/release floors, exact player slots and authoritative retractions, rejects destructive overlap/caller roster replacement, and preserves future invariance. CLI additions are `--history-captures`, `--roster-captures` and `--serving-contract fixed-2026`.

`FixedArtifactA0Adapter` explicitly identifies **frozen-a0-2026-forward-v1**, serving only `2026-11-01 <= cutoff < 2029-07-01` with unchanged2026 weights/calibrators. It is not a2027-trained artifact. The original annual adapter still rejects unsupported years; snapshots must be built before native archive namespace activation in a fresh process. This is not a deployed live feature worker.

Real native proof: seven input-selected cases, BO1/3/5 at the supported boundaries plus ten-new-player/two-new-organization cold start; **21 native/mode probabilities exactly match**. An821-series bounded captured-history fixture changes all8 native arrays, while future additions leave earlier tensors/probabilities unchanged. Repeated-query nonmutation, all3 annual2029 rejections, symmetry max9.11e−9 and cold deviation3.05e−10 pass. [CLI result](../data/08_reporting/plan09/plan09_execution_20261002_033658/fixed_native_cli.json), [independent native verification](../data/08_reporting/plan09/plan09_execution_20261002_033658/fixed_native_verification.json). These are software/artifact compatibility proofs, not future forecast accuracy or source admission.

## 9. External gates: do not fabricate the remaining work

| Plan item | Executed status | Exact remaining prerequisite |
|---|---|---|
|S1–S8|COMPLETED / NOT_ADOPTED|No automatic replacement recipe or follow-up fit|
|E1–E4|PASS_LOCAL_SOFTWARE_ONLY|Local synthetic inputs do not authenticate any source|
|E5 probes|BLOCKED /0 external requests|Source-specific acquisition/retention permission, actual entitlements/quotas, approved official unit/roster/start/result locators and authenticated samples|
|E6 activation decision|Completed decision: BLOCKED|Authentic participant/two-book identity, measured request capacity, independent official start/results and proven live native-input operation in addition to E5|
|F1 sports confirmation|NOT_RUN|Explicit recipe/horizon authorization, untouched future dates and future-valid matched C0/raw/head/input artifacts; the new head is rejected and expires at October1|
|F2 H/control development|NOT_RUN /0 fits|E6 and real collection, admitted released TRAIN/CAL data, fit support and execution authorization|
|F3 betting confirmation|NOT_RUN|Frozen final recipe/selection/settlement rules, separately authorized future assessment dates and genuine evidence/support|
|T1|Contract/local mechanics COMPLETE; authentic evaluation NOT_RUN|Official edition/rulebook/origin/roster corpus, coherent eligible joint-scoreline adapter and separate edition-level assessment|

All eight frozen source contracts remain unapproved/unverified. No credentials, endpoint guesses, entitlement assumptions or backdated issuance stand in for evidence. Frozen probe caps remain OddsPapi2, TOTALbet1, Lebull4 and one request per approved sports locator, zero unregistered locators; no automatic retries or pagination.

Frozen Plan06 calendar is unchanged: TRAIN2026-11-01→2027-11-01 (labels lockedNov15); CAL2027-12-01→2028-06-01 (Jun15 lock); DEVELOPMENT2028-07-01→2029-07-01 (Jul15 lock). November2027/June2028 are buffers. Readiness and authorization must be complete **before2026-10-31T00:00Z**, otherwise this planned November1 start is **EXPIRED_UNSTARTED**, not rolled forward. Plan05's ±2pp all-selection and(5%,10%] interval-containment gates, tax, epsilon, seed and independent positive-value gate are unchanged. This history is not prospective confirmation.

T1's [contract](../data/08_reporting/plan09/plan09_execution_20261002_033658/tournament_contract.json) and [gate decision](../data/08_reporting/plan09/plan09_execution_20261002_033658/tournament_gate_decision.json) cover legal matchups/transitions/joint targets, missing cells and edition-blocked evaluation. Read-only catalog audit finds91 profiles/234 phases,227 configured and **zero historically ready**. Actual simulator checks: BO5 terminal case20,000 simulations (p=.7, observed champion fraction.7008), exact swapped orientation, deterministic four-team round-robin→final,18 rank-mass checks and future-history rejection. This proves bounded mechanics, not real edition coverage or tournament probability quality. A scalar winner connector does not determine the missing joint scoreline distribution.

## 10. Preservation, provenance and completion limits

The initial checkout was `dev`,137 staged/66 unstaged/159 untracked entries. No reset, stage, commit, push or serving change was performed. Existing user work was retained. Two recovery incidents are explicit: a child removed two pre-existing untracked native files, restored exactly before legitimate edits; an oracle removed the frozen sports protocol, restored byte-for-byte to SHA256 `43855e45b29d7ddabdc0b549fb1f9a86866a041cdb2d0412d1a2136ca83534b5`, with no fits during absence. [Native recovery](../data/08_reporting/plan09/plan09_execution_20261002_033658/native_recovery/recovery.json), [protocol recovery](../data/08_reporting/a0_sports_correction/c0_raw_connector_20261002_033658/protocol_recovery_exact.json). These are recovered workflow defects, not hidden model variants.

Failed verifier attempts, the initial SQLAlchemy DDL percent-format issue, test collection error, observed behavioral RED cases and corrected fixture-only expectations remain in the execution receipts. No historical fit was rerun to repair them. The earlier pre-assessment activation receipt's missing-local-proof status is superseded by the final evidence decision; a subsequent accounting correction explicitly includes the four earlier local HTTP requests. The activation outcome remains BLOCKED throughout.

Before stopping the task-owned isolated database, exact local bodies/capture rows were preserved under `data/01_raw/betting_ev_reliability/forward_capture_20261002_033658/`, and complete event/envelope exports under stage03 with hashes. Five meaningful databases contain396 captures,205 events and5 envelope rows; **only the two native-smoke envelopes are genuine model outputs**, on synthetic fixtures. The other three envelopes are schema fixtures, not forecasts. Earlier failed-attempt records are intentionally included; final18-case proof counts are not archive totals. [Archive verification](../data/08_reporting/plan09/plan09_execution_20261002_033658/local_evidence_archive.json) reports no missing/corrupt/orphan archived bodies. [Shutdown](../data/08_reporting/plan09/plan09_execution_20261002_033658/isolated_database_shutdown.json) records successful stop of the owned loopback container and supervised process; no service is left running.

Reproducible verification scripts and immutable receipts remain in ignored stage08 evidence, not as competing production runners. No new datasets/model artifacts are added to Git. The original sealed October2 research report is unchanged; this is its separate execution report. Current status is **completed authorized historical/local slices, external gates blocked**, not full Plan09 completion, production readiness, reliable +5% EV or tournament certification.
