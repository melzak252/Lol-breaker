# Plan11: Project integrity and readiness backlog

**Created:** 2026-10-06  
**Status:** LOCAL_COMPLETE / EXTERNAL_GATES_BLOCKED —17 tasks retain explicit outcomes below. Actual3428-test full suite, CLI/API/PostgreSQL/browser/build/lint gates passed; owned services, containers/networks, browser tabs and throwaway helpers are closed. Source activation and authentic whole-edition/pre-start certification remain externally gated, not silently marked complete.
**Scope:** User-authorized orchestration of the exploration backlog's reachable local fixes. No new formal model experiment, external acquisition, configured-user database, production migration/deployment, wagers or restart of the stopped EV loop.

## Priorities and evidence

| Track | Priority | Problem | Execution boundary |
|---|---|---|---|
| T — Tournament benchmark integrity | P0 | Historical benchmark labels Glicko-derived formulas as native C0/A0; adoption ignores two recorded criteria | Local correction first; genuine model evaluation requires eligible artifacts and a separate frozen protocol |
| D — Documentation and task status | P1 | Architecture/index contain obsolete active-model, deployment and financial-certification claims | Documentation only; preserve dated evidence |
| S — Serving input provenance | P1 | Default C0 history stops on 2026-09-08; current rosters are not authenticated prior-available announcements | Local disclosure/contract checks; authentic activation remains externally blocked |
| O — API and operational isolation | P1 | API mutation routes depend on an unverified deployment access boundary; default Compose starts collectors on persistent storage | Local configuration and loopback checks only |
| B — Database and money contracts | P1 | Three schema paths need common behavioral proof; wallet money is Numeric but arithmetic includes floats | Empty, owned PostgreSQL/Timescale databases only |
| F — Financial/UI contract clarity | P2 | Native uncertainty is absent; historical simulation and assumed research haircuts must not appear certified | No strategy tuning, fabricated bounds or new wallet UI |

Confirmed code evidence from exploration:

- `src/models/tournament_pair_adapter.py:231-281`: the `a0` and `c0` branches derive probabilities from team Glicko states, hard-coded regional offsets and transformations; neither loads native artifacts.
- `scripts/run_tournament_distribution_benchmark.py:272-298`: adoption checks Uniform champion Brier and flow feasibility, but omits the computed Glicko champion-Brier and Swiss-RPS criteria.
- `docs/PROJECT_ARCHITECTURE.md:19-45`: obsolete active A0, 75/25 hybrid, uncertainty and verified-strategy descriptions.
- `conf/base/c0_serving.json` and `betting_app/README.md`: current local native C0 identity, static-history qualification and exact 50/50 logit hybrid; no remote deployment claim.
- `betting_app/api/main.py`, `docker-compose.yml`, `client/Caddyfile`: permissive API CORS, published API port and separate proxy-auth configuration; current remote protection was not inspected.
- `betting_app/README.md`: migrated alert-table absences and an isolated prop-analysis HTTP500 are recorded limitations, not newly reproduced findings. Do not rerun merely to reconfirm them; trace the cause and verify after a repair.

Schema, concurrent-wallet behavior and browser request races are verification tasks, not newly established runtime defects. Source anchors describe the inspected snapshot; re-read before implementation.

## Dependencies and ownership

1. T1 and D1 can start immediately. T2 precedes T3; T4 cannot establish native tournament results before its artifact/availability gate passes.
2. S1 precedes S2. S3 is an external gate owned by the existing forward-evidence plans, not a duplicate collector implementation.
3. O1 defines the access boundary before O2 changes local deployment configuration. O3 is independent of model experiments.
4. B1 precedes B2; B3 uses the reconciled schema. No routine schema audit of the configured user database.
5. F1 precedes F2 so the browser exposes the actual backend contract rather than another probability/uncertainty policy.
6. If executed in parallel, assign one integration owner for shared API, Compose, research-status and index edits. Run related checks once after the implementation wave; retain failed evidence.

## T — Correct tournament predictor provenance and decision semantics

### T1 — Correct the current scientific interpretation

- [x] **T1 / P0 / DONE:** Add a dated correction to current research documentation identifying `tournament_dist_20261005_01` as a Glicko-derived proxy comparison, not native C0/A0 calibration evidence.
  - **Targets:** `docs/RESEARCH.md`, `scripts/README.md`, relevant tournament descriptions in `betting_app/README.md` and `docs/PROJECT_ARCHITECTURE.md`.
  - **Change:** Preserve the original decision, metrics, receipts and model-output bytes. Explain the mismatch alongside the current claim and withdraw its native-model adoption interpretation. Keep the application's genuine C0 tournament path distinct.
  - **Acceptance:** Following the current documentation cannot imply that native C0 checkpoints were used in the four historical editions. No sealed artifact or canonical match manifest is rewritten.

### T2 — Give proxy predictors honest identities and receipts

- [x] **T2 / P0 / DONE:** Remove misleading native `c0`/`a0` identities from the proxy benchmark's current public contract and migrate its callers.
  - **Targets:** `src/models/tournament_pair_adapter.py`, `scripts/run_tournament_distribution_benchmark.py`, `betting_app/tests/test_tournament_distribution_engine.py`, `betting_app/tests/test_calibrated_tournament_model.py`.
  - **Change:** Use explicit Glicko/regional/transformed-proxy identities. New-run receipts bind predictor kind, formula/configuration, source data and relevant code hashes. Preserve archived outputs with a historical interpretation note; do not keep misleading aliases or overwrite the old run. Genuine native models require real artifact/input bindings, not a renamed proxy.
  - **Acceptance:** A real small proxy simulation returns the declared identity/provenance and complementary series probabilities. An unsupported native-model selection fails explicitly; it cannot fall back to Glicko.

### T3 — Make the declared decision contract match execution

- [x] **T3 / P0 / DONE:** Replace the misleading adoption result with an explicit, frozen proxy-study decision contract and exercise every mandatory criterion.
  - **Targets:** `scripts/run_tournament_distribution_benchmark.py`, `src/analysis/tournament_metrics.py`, existing tournament benchmark tests.
  - **Change:** Separate mathematical consistency from predictive evidence. A proxy diagnostic must not emit native `TOURNAMENT_CALIBRATION_ADOPTED`. For any separately authorized genuine comparison, freeze primary metrics, comparators, missingness and edition-level uncertainty before scoring; require every declared gate, including support/estimability. Four historical editions do not establish universal calibration.
  - **Acceptance:** Behavioral regression cases where one mandatory comparator/uncertainty gate fails cannot yield adoption. Probability-mass/LP success alone cannot promote a predictor. Run a small real simulator scenario after the fix; do not test source text or copied booleans.

### T4 — Establish eligibility before any genuine native tournament benchmark

- [x] **T4 / P1 / LOCAL_INVENTORY_COMPLETE; AUTHENTIC_CERTIFICATION_BLOCKED:** Record edition/origin-specific model and input eligibility before preparing a genuine native-model evaluation. The36-cell inventory is dated; its addendum preserves the concurrently updated eight-case native-study manifest, which Plan11 neither executed nor certified.
  - **Targets:** `conf/base/tournament_manifest.json`, `src/models/tournament_prediction.py`, `src/models/tournament_catalog.py`, `scripts/simulate_tournament.py`, `betting_app/tests/test_tournament_replay_readiness.py`, `betting_app/tests/test_tournament_historical_manifest.py`.
  - **Change:** Bind actual fit timestamps separately from closed TRAIN/CAL dates, roster scenarios/evidence, released histories, rules/draws and all legal counterfactual pairs. Original serving C0 uses pre-2026 TRAIN/CAL and rejects pre-2026 decisions; it cannot represent pre-event Worlds/MSI2023/24 forecasts. Keep explicit model/edition absences. Origin-eligible retrospective native research with declared scenarios is permitted; it is not authenticated historical issuance.
  - **Acceptance/outcome:** The local inventory records supported/missing cells and reasons without scoring a convenient intersection. Genuine artifacts/rules/input publication and independently supported editions remain prerequisites for authentic certification. The separately owned authorized native-study and completed LCK2026 phase are distinct, preserved work—not an unexecuted Plan11 proposal or Plan11 result.

## D — Reconcile current documentation without erasing history

- [x] **D1 / P1 / DONE:** Update the architecture guide and obsolete roadmap notices to match inspected contracts.
  - **Targets:** `docs/PROJECT_ARCHITECTURE.md`, `README.md`, `TODO/README.md`, `TODO/01_bet_qualification_hardening.md`, `TODO/02_regional_family_ratings.md`, `TODO/03_operational_pipeline_cutover.md`, `TODO/04_lineup_ingestion_and_tournament_readiness.md`.
  - **Change:** State annual A0 research versus local native C0 versus separate Fixed50 logit hybrid; separate local code, observed production history and deployment. Mark old yield/rating/certification claims and fallback instructions as historical/superseded, with links to corrected evidence. Do not advertise unsupported A1, rating fallback as A0, or automatically acquired confirmed lineups.
  - **Acceptance:** Read every revised document completely and verify its links/commands. Current instructions neither authorize collectors/deployment nor describe a profitable or certified strategy. Historical results and prior negative findings remain recoverable.

- [x] **D2 / P2 / DONE:** Reconcile Plan10 and later experiment status only from completed receipts.
  - **Targets:** `TODO/10_c0_ev_loss_objective.md`, `TODO/README.md`, `docs/RESEARCH.md`, existing betting research report.
  - **Change:** Plan10 is separate user-authorized work; this backlog must not execute or duplicate it. When its owner publishes a completed decision, link actual fits, verification, failures and limitations. Until then retain its explicit planned/in-progress state; code presence and modification dates are not execution proof.
  - **Acceptance:** Index, plan and report agree with the actual decision receipt; no new fit, repeated experiment or promotion is implied.

## S — Make serving input freshness and availability explicit

- [x] **S1 / P1 / DONE:** Trace history and roster provenance through persistence to the user-facing result, filling only actual disclosure gaps.
  - **Targets:** `betting_app/core/models/native_c0.py`, `betting_app/services/upcoming_inference_service.py`, `betting_app/api/schemas.py`, `client/src/types/index.ts`, `client/src/pages/MatchDetail.tsx`, `client/src/pages/TournamentSimulation.tsx`.
  - **Change:** Preserve input source/hash, maximum released-history day, decision clock and roster basis. Distinguish frozen history, operator overrides, current stored rosters and authenticated prior announcements. A recent `predicted_at` must not imply refreshed history or forward certification. Preserve historical missing diagnostics as unknown rather than backfilling claims.
  - **Acceptance:** An isolated native prediction and browser view disclose the static source through 2026-09-08 and the actual roster basis; current timestamps cannot produce a verified-live label. Existing model math, weights and series projection remain unchanged.

- [x] **S2 / P1 / DONE:** Verify that roster changes between feature construction and prediction cannot silently persist inconsistent inputs.
  - **Targets:** `betting_app/services/upcoming_inference_service.py`, `betting_app/services/current_roster_service.py`, `betting_app/tests/test_native_c0.py`, `betting_app/tests/test_a1_persistence_contracts.py`.
  - **Change:** Establish whether the existing request binds the roster snapshot used for tensors and persistence. Repair only demonstrated inconsistency: rebuild at the actual decision or explicitly withhold stale inputs. Keep exact player ordering, competition context, optional-organization routing and separate actual clocks.
  - **Acceptance:** Change or retract a roster between construction and inference in an isolated scenario; the saved forecast either binds the intended valid roster/tensors or is withheld with a reason. Future announcements cannot change an earlier request; inputs are not mutated.

- [x] **S3 / P1 / TRACKING_COMPLETE; ACTIVATION_BLOCKED_EXTERNAL_PREREQUISITES:** Track authentic native-history/source activation under the existing admission contracts. Recording the missing external prerequisites is complete; none of the eight sources is activated and G0 is not cleared.
  - **Owners:** [Plan06](./06_evidence_admission_after_a0_audit.md), [Plan09 Track E](./09_c0_connector_and_forward_evidence.md), current native-C0 readiness in `betting_app/services/c0_shadow_readiness.py`.
  - **Prerequisites:** Source-specific permission/retention/entitlements/quotas; approved locators and participant bindings; authentic released-history/roster builder and coverage; starts, settlements and locked future samples.
  - **Acceptance:** Readiness independently validates actual retained bytes and authority, not self-declared authentication or hashes. Readiness before `2026-10-31T00:00:00Z` and start `2026-11-01` remain fixed; otherwise report `EXPIRED_UNSTARTED`. Do not repeat completed local capture/schema audits, scrape without authorization, loosen G0, roll dates or reopen the old 10/10 loop.

## O — Verify the access boundary and isolate local operations

- [x] **O1 / P1 / DONE:** Establish the supported access boundary for every mutation/trigger route.
  - **Targets:** `betting_app/api/main.py`, API routers `bets.py`, `scheduler.py`, `matches.py`, `tournaments.py`, `client/Caddyfile`, `docker-compose.yml`.
  - **Change:** Inventory wallet/bet writes, mapping, roster updates, prediction, sync and task triggers. Check whether proxy authentication actually protects all of them and whether direct API exposure bypasses it. Prefer the existing authenticated-proxy boundary with an internal/loopback API; do not introduce a second identity system without a demonstrated need. Classify viewing versus work-triggering routes, including state-changing GET behavior.
  - **Acceptance:** A loopback-only unauthenticated request cannot reach a protected mutation through the supported entrypoint; authorized access works. Direct API exposure and allowed browser origins are explicit. No claim about the remote server without authorized inspection; CORS is not authentication.

- [x] **O2 / P1 / DONE:** Make the chosen local configuration enforce O1 without changing the user's live stack.
  - **Targets:** `docker-compose.yml`, `client/Caddyfile`, `client/Dockerfile`, `client/vite.config.ts`, `betting_app/api/main.py`, `betting_app/README.md`.
  - **Change:** Align published ports, proxy target, authentication and allowed origins. Keep the development Vite server distinct from static/proxy delivery. Verify the configuration, not just the existence of a Caddyfile. Preserve secrets outside the repository.
  - **Acceptance:** On an owned loopback stack, intended browser requests work and disallowed entrypoints/origins do not bypass the boundary. Do not build/publish/deploy images or restart the existing server under this backlog.

- [x] **O3 / P1 / DONE:** Provide a documented disposable API/UI smoke path that cannot start collection accidentally.
  - **Targets:** `docker-compose.yml`, `betting_app/scheduler/app.py`, `betting_app/scheduler/registry.py`, `betting_app/README.md`, `betting_app/tests/test_scheduler.py`.
  - **Change:** Use isolated storage, ports and unique process/container names; exclude scheduler, external transports and notification delivery from the smoke path. Explicitly identify normal scheduler startup's job-store reset and enabled collectors. Reuse existing configuration instead of creating a competing stack.
  - **Acceptance:** Start and exercise the owned API/UI surface with zero collector jobs or external requests and no mount of `/data/inzynierka/pgdata`; stop only task-owned resources. Ordinary scheduled operation remains unchanged unless separately requested.

## B — Verify schema and accounting end to end

- [x] **B1 / P1 / DONE:** Compare operational behavior across all three empty PostgreSQL schema paths.
  - **Targets:** `betting_app/models/`, `betting_app/alembic/versions/`, `docker/timescale/init.sql`, `betting_app/core/db.py`, `betting_app/tests/conftest.py`.
  - **Change:** On separate owned databases compare ORM initialization, `alembic upgrade head`, and Timescale bootstrap for tables, relevant columns/types/defaults, timezone semantics, foreign keys, uniqueness and immutable-evidence triggers. Focus on quotes/predictions/signals, accounts/bets/transactions, alerts and props used by the application. Do not assume an old initial migration represents the final chain.
  - **Acceptance:** The same parser → quote → native prediction/signal, wallet settlement, alert-with-disabled-delivery and affected prop request exercise works on each supported path, or a precise discrepancy is retained. No ORM `create_all` over a migrated database to mask missing tables. No configured-user DB or SQLite substitute.

- [x] **B2 / P1 / DONE:** Repair only demonstrated schema/default/timezone discrepancies with a forward migration and matching declarations/bootstrap.
  - **Depends on:** B1.
  - **Targets:** A new Alembic revision when needed, corresponding ORM models and bootstrap SQL; existing API/service regression tests.
  - **Change:** Do not rewrite applied migration history. Include known migrated alert-table absences after tracing the missing declarations and dependencies. Trace the recorded prop HTTP500 rather than hiding it or assuming its cause. Preserve stored money, identity and evidence bytes.
  - **Acceptance:** Empty migration chain and an owned upgrade fixture both exercise the affected real interfaces successfully. Invalid inputs/SQL failures remain distinguishable; no broad exception-to-empty-result fallback.

- [x] **B3 / P1 / DONE:** Verify cent-exact wallet arithmetic, concurrent reservation and settlement idempotency; fix failures at their cause.
  - **Targets:** `betting_app/services/wallet_service.py`, `betting_app/models/bookmaker.py`, `betting_app/api/routers/bets.py`, `betting_app/api/schemas.py`, `betting_app/tests/test_bets.py`.
  - **Change:** Derive payouts from persisted Numeric values using explicit Decimal/rounding semantics, preserving the public money contract. Validate monetary/odds/tax boundaries without NaN/infinity reaching persistence. Preserve the conditional sufficient-balance debit and open-only settlement transition; placement, balance and transaction ledger commit atomically.
  - **Acceptance:** Independent Decimal oracle for fractional-cent win/refund/loss cases; two simultaneous placements cannot overdraw; two settlements credit once; injected failure rolls back bet/balance/ledger together. Exercise actual HTTP placement/settlement on an owned PostgreSQL database, not just mocks. Manual records remain distinct from qualified recommendations.

## F — Align financial and browser semantics

- [x] **F1 / P2 / DONE:** Make uncertainty provenance and unavailable financial results explicit across current evaluation surfaces.
  - **Targets:** `src/analysis/unified_evaluation_engine.py`, `betting_app/services/bet_qualification_service.py`, `betting_app/api/routers/financial.py`, `client/src/pages/FinancialAnalysis.tsx`, `betting_app/tests/test_uncertainty_betting_evaluation.py`, `betting_app/tests/test_financial_analysis.py`.
  - **Change:** Keep native C0's absent uncertainty absent. Distinguish explicitly assumed research haircuts from measured bounds and from mean-probability qualification. A missing conformal bound or zero eligible bets must report exclusions/undefined conditional metrics, not successful calibration, zero error or profit certification. Preserve stored-hybrid no-reblend behavior and the existing financial policy; no tuning of thresholds, odds caps, Kelly or sigma values.
  - **Acceptance:** Same saved forecast exercised in mean and explicitly bounded/assumed scenarios yields the declared eligibility and reasons. Missing native bounds cannot create an invented lower bound; historical quote recomputation cannot become recorded issuance/execution.

- [x] **F2 / P2 / DONE:** Verify browser request ordering and refusal/error display, repairing demonstrated stale-state behavior.
  - **Targets:** `client/src/api/client.ts`, `client/src/pages/MatchList.tsx`, `MatchDetail.tsx`, `MatchResults.tsx`, `FinancialAnalysis.tsx`, related TypeScript response types.
  - **Change:** Preserve exact model/version selections, backend abstention reasons, valid prices and null probability/EV. Where requests overlap, cancel or guard superseded responses so old model/bookmaker/horizon data cannot replace the current selection. Surface failed BO/roster/predict mutations rather than silently showing success. Reuse page-local patterns; no new state-management framework or manual-wallet UI.
  - **Acceptance:** Browser exercise with deliberately reversed response completion, model/bookmaker/Open-Mid-Close changes, successful refusal and HTTP failure. Current controls match displayed data; unavailable EV remains null/visible as unavailable, prices remain visible, and failed mutations retain the correct state. Run frontend build/lint after changes.

## Repeatable verification commands

These commands describe repeatable gates; actual execution receipts are below, not inferred from this checklist. Database-dependent tests require `TEST_DATABASE_URL` pointing to an owned disposable PostgreSQL/Timescale database; the fixture truncates tables. Alembic also requires `DATABASE_URL` to name only that owned database. Never use the default/configured user database.

```bash
# Read-only canonical integrity check; does not train or create a scoring run:
.venv/bin/python scripts/run_model_benchmark.py --suite --doctor

# T: after predictor/decision changes:
.venv/bin/python -m pytest -q betting_app/tests/test_tournament_distribution_engine.py betting_app/tests/test_calibrated_tournament_model.py betting_app/tests/test_tournament_formats.py betting_app/tests/test_tournament_replay_readiness.py betting_app/tests/test_tournament_historical_manifest.py

# S: after serving contract changes:
.venv/bin/python -m pytest -q betting_app/tests/test_native_c0.py betting_app/tests/test_c0_evidence_transport.py betting_app/tests/test_a1_persistence_contracts.py betting_app/tests/test_c0_forward_readiness.py

# B/O/F: only on the explicitly owned isolated test database:
.venv/bin/python -m pytest -q betting_app/tests/test_bets.py betting_app/tests/test_forecast_evidence.py betting_app/tests/test_scheduler.py betting_app/tests/test_financial_analysis.py betting_app/tests/test_uncertainty_betting_evaluation.py

# Fresh migration path only after provisioning an empty owned database:
.venv/bin/python -m alembic upgrade head

# From client/, after frontend changes:
npm run build
npm run lint
```

Tests are necessary but not sufficient: each changed runtime path also needs its named actual CLI/API/browser/SQL smoke. Preserve failed attempts and report limits; do not claim a full suite, deployed model, live evidence, financial reliability or tournament certification from a narrower check.

## Completion and non-goals

- Each task is done only when its observable acceptance and relevant caller/docs updates are complete. Blocked tasks retain exact missing prerequisites and are not silently dropped.
- Keep original annual A0/C0 artifacts, canonical benchmark manifest, sealed historical reports/decisions, Plan05/06/09 contracts and old 10/10 journal unchanged unless separately authorized for a new version.
- Do not repeat completed quote/context/capture repairs, negative capacity/champion/reconstruction/connector experiments, or source-feasibility audits as unexecuted work.
- Do not add model fallbacks, universal probability flattening, team/league exceptions, new learned heads, automatic retries or benchmark variants under an integrity task.
- No manual-wallet frontend feature, research archive migration, broad service refactor, live collection, production schema change or deployment is included.

## Executed local results and retained external gates

Evidence root: [`data/08_reporting/project_integrity/plan11_integrity_20261006_195949/`](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/). Failed attempts and corrected independent-oracle assumptions remain retained, not overwritten.

| Gate | Observed final result | Receipt |
|---|---|---|
| Full application regression, owned PostgreSQL/Timescale | **3428 passed,36 warnings**, no skips/failures; existing dependency warnings | [final suite](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_application_suite_corrected.json) |
| Final frontend, after last disclosure edit | TypeScript/Vite build and ESLint passed | [build/lint](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_frontend_after_roster_disclosure.json) |
| Schema/upgrade behavior |51 distinct scoped cases pass on each ORM/migrations/bootstrap/upgrade path:204 executions, not204 distinct tests | `final_schema_orm.json`, `final_schema_migrated_final.json`, `final_schema_bootstrap.json`, `final_schema_upgrade.json` in evidence root |
| Actual genuine native pipeline on3 supported PG paths | Parser/quote/native/persistence/2 mean signals/refused uncertainty, HTTP wallet18.49 payout/108.49 balance, disabled alerts0 delivery and propssave/read | [runtime observations](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_three_schema_native_runtime.json) |
| Concurrent roster/fixture contract | Absent-state writer blocks until publication;5 concurrency/rollback regressions; both real maintenance utilities pass on owned fixtures | `final_absent_roster_publication_lock_complete_fixture.json`, `final_roster_maintenance_writer_smoke.json` |
| Actual proxy CLI |16 model×edition cases/32 draws,6 prior-CAL scalar BO fits; diagnostic completed, native adoption false; native alias refused | [CLI receipt](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/proxy_cli/plan11_proxy_smoke_20261006_01/receipt.json) |
| Independent saved proxy replay/current source |3552 probabilities/48 actual reverse queries,max2.22e-16; further current-format32-draw CLI champion mass1/finalist mass2 | [independent proof/protected hashes](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_independent_proxy_and_protected_hashes.json), `final_current_proxy_forecast.json` |
| Actual supported authentication/network boundary |80 API operations+5 UI/docs paths unauthenticated401; authenticated health/wallet/CORS exercised, API/DB no host port/egress, collectors0 | [auth](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_actual_authentication_boundary.json), `final_actual_network_boundary.json` |
| Actual Decimal wallet through Caddy |100.005→100.01,10.005→10.01,2.12345→2.1235,.125→.13,payout18.49; duplicate settle400/no credit; legacy void refunds | [independent money proof](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_actual_wallet_decimal_oracle.json) |
| Browser surface | Actual native DTO/history/roster; controlled stale model/book/horizon/route/30-day/refusal/BO-merge cases; tournament disclosure render only;8 durable screenshots | [actual native UI](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_browser_genuine_native_provenance.json), [BO merge](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_browser_bo_roster_merge.json), [screenshots](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_browser_screenshot_inventory.json) |
| Canonical/protected state | Doctor READY/five pinned source hashes; five protected config/old-decision/old-receipt/10-of-10 hashes unchanged | [doctor](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_canonical_benchmark_doctor.json), independent proof above |

Forward migrations are `e203c0060011` and `e203c0060012`; old applied history is not rewritten. Past INTEGER confidence precision cannot be restored. UTC upgrade preserves instants, original float quotes/raw payloads and disabled state; legacy naive clocks interpreted as UTC are not evidence of authentic publication. Source quarantines remain latest quote state, not permission to reuse an older price. Direct trusted-local API is not independently authenticated; CORS is not auth. Ordinary Compose remains persistent and includes enabled collectors; only the disposable override is an offline smoke.

**Retained failures:** initial full suite3318 passed/11 failed; focused806 passed/1 stale copied snapshot failed before removal; wrong standalone exact0.5/provider-binding/Float-vs-Decimal/owned-sequence/manual-opaque-ID/counter-alias oracles and unknown first standalone API failure. Genuine native captured cold p0.499999998892492, ordinary/evidence/replay difference0, reverse error2.215e-9; no model change or false exact-neutral claim. The final suite is a new successful execution, not a relabeling.

**Concurrent owner work preserved:** current native-study manifest SHA`802b317fb9dff16d1c33abf6901df850c25fd85610bb12ad2799882a56c29b1d`, separate Plan10 completed/NOT_ADOPTED and genuine C0 LCK2026 phase. The earlier proxy execution binds format code SHA`b0f282…`; later current-format smoke uses`d837496220dfa542d8b32da82daeddd6aad7ca8917be4b3a9bd84dfa73b69c78`. No re-pin, rollback, protected-artifact rewrite or new formal native/head/base fit. Six proxy scalar CAL fits are disclosed separately.

**Remaining external gates:** [blocked metadata](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/blocked_prerequisites.json) retains permissions/retention/entitlements/quotas, official locators/authentic fixture-unit-player bindings, actual capture/publication clocks, live release-aware builder/coverage, official starts/results/settlements and future samples. S3/G0 remains blocked; authentic whole-edition/pre-start certification and separately owned native-study are not Plan11 results. Readiness beforeOct31/startNov1, earlier negatives and stopped10/10 remain unchanged. No user DB, external collection/notifications/scheduler, deploy or server mutation occurred. Existing Docker image contains an unread baked `.env`; smoke masks it, not certifies a clean image.

Only task-owned resources were stopped: two managed API/Vite services, two browser tabs, four disposable containers and two owned Compose networks. Five owned `/tmp` throwaway scripts were removed after proof/source receipts were retained. No shared browser, user database/container, live server or user work was stopped/removed. Complete revised-document/link/command verification and actual shutdown are bound by [`final_closure.json`](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_closure.json).
