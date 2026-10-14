# Project Map & Architecture

Scope: LoL-Breaker repository, inspected local contracts as of 2026-10-06. **Local implementation, historical server observations and deployment are separate facts.** This guide does not authorize collectors, training, notifications, database changes or deployment.

Current model choices, completed study decisions, prop recovery and archive/trash boundaries: **[MODEL_CATALOG.md](MODEL_CATALOG.md)**. Historical reports and frozen paths remain in place; this architecture guide describes implementation contracts, not a current candidate shortlist or research promotion.

## 1. Entry points and ownership

| Function | Canonical source | Boundary |
|---|---|---|
| Research match benchmark | [`src/analysis/research_benchmark.py`](../src/analysis/research_benchmark.py), [`src/analysis/model_benchmark.py`](../src/analysis/model_benchmark.py); [`scripts/run_model_benchmark.py`](../scripts/run_model_benchmark.py) | Frozen annual A0/rating/039 comparison via the pinned [research manifest](../conf/base/research_benchmark.json); retrospective development, not serving certification |
| Historical financial evaluation | [`src/analysis/unified_evaluation_engine.py`](../src/analysis/unified_evaluation_engine.py); [`scripts/run_unified_betting_evaluation.py`](../scripts/run_unified_betting_evaluation.py) | Explicit study/protocol and cash-flow assumptions; not executed betting returns |
| Recommendation qualification | [`betting_app/services/bet_qualification_service.py`](../betting_app/services/bet_qualification_service.py) | `is_bet_eligible(...)` applies configurable policy; eligibility is not proof of reliable EV or profitability |
| Local sports/hybrid inference | [`registry.py`](../betting_app/core/models/registry.py), [`engine.py`](../betting_app/core/models/engine.py); [`run_upcoming_prediction_pipeline.py`](../betting_app/scripts/run_upcoming_prediction_pipeline.py) | Native C0 and separately identified Fixed50 hybrid; reads existing inputs and writes predictions/signals |
| Application tournament scenarios | [`betting_app/services/tournament_service.py`](../betting_app/services/tournament_service.py), [`API routes`](../betting_app/api/routers/tournaments.py) | Genuine native C0 series inference, not the historical research proxy benchmark |
| Historical tournament proxy benchmark | [`src/models/tournament_pair_adapter.py`](../src/models/tournament_pair_adapter.py); [`scripts/run_tournament_distribution_benchmark.py`](../scripts/run_tournament_distribution_benchmark.py) | Glicko-derived pair probabilities and bracket mechanics; not native C0/A0 calibration evidence |
| Manual betting/accounting and UI | [`betting_app/README.md`](../betting_app/README.md), [`betting_app/api/`](../betting_app/api/), [`client/`](../client/) | FastAPI + React; manual decisions, bet records, wallets and settlement. **No automatic bet placement** |

## 2. Model identities: three distinct contracts

### Local native C0

- The configured sports model is **`Causal-C0:c0-native-2026-w32-e12-v1`**, feature version `native-c0-history16-v1`, selected by the [registry](../betting_app/core/models/registry.py) and [serving configuration](../conf/base/c0_serving.json).
- It is the frozen **control W32/E12** from `a0_data_focus_20260929_192443`: three neural seeds, shared rating/MLP experts, and mode-specific calibration/mixtures. The configured model root is `data/06_models/a0_data_focus/a0_data_focus_20260929_192443`, not the newest experiment directory.
- The sports model does not consume market odds. It returns the winner probability of the **whole BO1/3/5 series**; do not apply another map-to-series projection.
- Supported modes are `full`, `no_w20`, and `no_organization`. Exact, distinct player IDs are required: five per side, with disjoint rosters. Missing organization identity with complete player identities uses explicit `no_organization`, not a rating fallback. Unsupported formats, pre-2026 decision cutoffs, missing inputs and invalid artifacts fail explicitly.
- Native C0 supplies **no operational lower bound or `epistemic_sigma_z`**. An assumed uncertainty haircut is not a native output or a certified confidence interval.

### Separate Fixed50 market hybrid

`Hybrid-Operational-Market:c0-native-2026-w32-e12-v1-a0.50-t1.00` applies exactly one transform to a saved native series probability and an admitted no-vig series-market consensus:

$$p_{\mathrm{Fixed50}}=\operatorname{sigmoid}\!\left(0.50\,\operatorname{logit}(p_{\mathrm{C0}})+0.50\,\operatorname{logit}(q_{\mathrm{market}})\right),\qquad T=1.00.$$

This is **50/50 in logit space**, not a linear probability average, the former 75/25 description, or a learned betting head. Already stored hybrid probabilities must not be blended again. Market context, quote orientation, timing and source-state validity are separate admission contracts; see the [application contract and dated local verification](../betting_app/README.md#naprawy-kontraktu-kursów-i-ev--2026-10-06).

### Annual A0 and frozen academic references

- **Annual Causal A0** is the separate, unpromoted research reference. Its frozen annual checkpoints live under the resolved research root at `a0-phase-walkforward-20260914/data/06_models/causal_a0/`. The canonical manifest evaluates annual `p_full`, not this local C0 checkpoint and not a live model.
- The September 26 audit reports annual A0 LL **0.551246 on 11,550** matches and approximately **0.5599 on the 9,907-match common 039 population**. These are different retrospective populations, not new forecasts or deployment evidence; [research status](RESEARCH.md) and the [dated audit](../reports/production_readiness_and_market_benchmark_report_2026.md) retain limitations and subsequent negative results.
- `Consolidated-A1` remains research-only and is rejected for operational inference; a rating consensus is not A0. EXP081 is not an active loader or fallback in the current C0 application. Historical names/records are not rewritten into C0.
- **EXP-039** remains the immutable thesis baseline, including `betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib` and historical scripts/reports. Do not delete it or confuse the manifest's annual research 039 derivative with that frozen application artifact.

## 3. Worker, native inputs and provenance

The API-side [`c0_inference.py`](../betting_app/services/c0_inference.py) calls an isolated JSON-lines [`c0_worker.py`](../betting_app/services/c0_worker.py); [`NativeC0Adapter`](../betting_app/core/models/native_c0.py) loads the pinned native implementations and artifacts inside that process. Isolation keeps archival `src.models` imports out of the API process.

The native snapshot has **eight tensors**, not a rating-only approximation:

| Family | Tensor | Per-match shape | Dtype |
|---|---|---|---|
| Raw | `players` | `(2, 5, 21)` | `float32` |
| Raw | `core` | `(2, 5)` | `float32` |
| Raw | `team` | `(2, 11)` | `float32` |
| Raw | `w20` | `(2, 10)` | `float32` |
| Raw | `gates` | `(2, 2)` | `float32` |
| Temporal | `history` | `(2, 5, 16, 33)` | `float16` |
| Temporal | `mask` | `(2, 5, 16)` | `uint8` |
| Temporal | `champions` | `(2, 5, 16)` | `uint16` |

Default history is `data/artifacts/golgg-database-recovery-20260909/matches.json`, pinned SHA-256 `991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca`: **41,915 series, source/release boundary 2026-09-08**. The directory's September 9 date is not the final history day. Replay also uses the pinned release-day map and consumes only history before the UTC decision-day cutoff.

Worker diagnostics preserve source/hash, maximum source/released-history days, consumed cutoff and player-slot order. Supplied snapshots are explicitly identified as caller-supplied native arrays. A recent `predicted_at` is an execution timestamp, **not refreshed history**. Current stored rosters or operator scenario overrides are not automatically authenticated prior-available lineup announcements. Updating the application's GOL.GG cache, ratings or W20 does not update native C0 history.

Research root resolution is `C0_RESEARCH_ROOT`, `ENSEMBLE_RESEARCH_ROOT`, `RESEARCH_ROOT`, then `data/research_root.txt`. Configuration/model/history/release overrides are explicit; no silent fallback to `data/golgg_matches.json`. Artifact validation and source activation remain distinct. See [current artifact prerequisites](../betting_app/README.md#artefakty-i-lokalne-źródła) and [forward-evidence limitations](RESEARCH.md).

## 4. Persistence: three schema construction paths

The current database layer is PostgreSQL via SQLAlchemy, with Timescale bootstrap support, not the archived SQLite MVP. [`db.py`](../betting_app/core/db.py) resolves `DATABASE_URL`; the old `alembic.ini` SQLite default must not be used as the operational database contract.

| Path | Source | Meaning |
|---|---|---|
| ORM initialization | [`betting_app/models/`](../betting_app/models/), `init_db()` / `Base.metadata.create_all` | Creates declared tables and seeds static bookmakers |
| Full Alembic chain | [`alembic.ini`](../alembic.ini), [`betting_app/alembic/versions/`](../betting_app/alembic/versions/) | Applies ordered revisions through `alembic upgrade head` with an explicit owned PostgreSQL `DATABASE_URL` |
| Empty Timescale bootstrap | [`docker/timescale/init.sql`](../docker/timescale/init.sql) | Initializes extension/tables/seeds for a fresh Timescale database |

They are separate paths, **not presumed equivalent because a subset of evidence tables passed**. Do not run ORM `create_all` over a migrated database to conceal missing tables. Plan11 repaired the demonstrated alert/prop/default/UTC discrepancies with forward revision `e203c0060011`; `e203c0060012` preserves new fractional identity confidence as DOUBLE PRECISION across migration and bootstrap paths. Past confidence already rounded by the old INTEGER schema cannot be recovered. The same 51 behavioral/schema cases passed on each of ORM, fresh migrations, bootstrap and an owned upgrade fixture (204 executions, not 204 distinct tests). Actual parser → persisted quote → genuine native prediction/two mean-EV signals, wallet settlement, disabled alerts and prop save/read passed on all three existing PostgreSQL paths without ORM schema repair. The non-UTC upgrade retained quote instants, original float prices and exact raw payloads. This is scoped operational parity, not an audit of every column or financial/source certification; [Plan11 receipts](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/) retains failed attempts and limits.

## 5. Tournament scenarios versus historical proxies

The application simulator calls `c0_inference.predict_c0` with exact player/team identity, decision clock, competition context and BO. It uses the native **series** output for Monte Carlo bracket winners. It does not invent a map probability, joint scoreline distribution or conditional probability for a partially played series; unsupported live-score conditioning fails explicitly. Stored current rosters make these scenarios, not authenticated historical editions.

The separate historical `tournament_dist_20261005_01` compared **Uniform (`uniform`), format Glicko (`glicko_format`), regional Glicko proxy (`regional_glicko_proxy`) and calibrated regional Glicko proxy (`calibrated_regional_glicko_proxy`)** on Worlds 2023/2024 and MSI 2023/2024. The archived labels “A0”/“C0” denoted the latter two rating formulas, not annual A0 or native W32/E12. The [October 6 attribution correction](RESEARCH.md#2026-10-05--tournament-distribution-and-bracket-simulation-benchmark-native-c0a0-attribution-withdrawn) preserves the original decision, scores and identity audit.

The calibrated proxy's champion Brier **0.828215** beats Uniform **0.903941**, but its Swiss RPS **0.117283** is worse than the regional proxy's **0.111770**. The archived `TOURNAMENT_CALIBRATION_ADOPTED` decision does not establish native adoption; its implementation omitted two recorded criteria from the conjunction. Keep those historical bytes and negative comparisons. Bracket feasibility, symmetry and software tests alone certify neither model calibration nor genuine native tournament evaluation. Do not use the obsolete native-sounding proxy aliases as instructions for a new run.

## 6. Qualification is policy, not a verified profitable strategy

Displayed mean EV under a simple turnover-tax assumption is $p\,O\,(1-\tau)-1$; the common local tax default is 12%. Qualification adds configurable hurdles and guards. The current base minimum is 5%, with an extra low-odds hurdle below 2.20; caller-specific caps and staking differ across application/research paths. There is **no universal optimal odds range, 15% EV ceiling, BO1 cap or quarter-Kelly rule enforced by every endpoint**.

Native uncertainty is absent. Policies that require verified conservative bounds must fail closed rather than substitute invented bounds. Historical assumed haircuts, Kelly portfolios and point ROI remain scenario diagnostics, not certified probabilities, execution or profitability.

The [October 6 C0 audit](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md) retains the strict-prior first-observed result: 34 C0 selections predicted **+24.54%** average net EV and realized **−18.86%** flat ROI, with a wide EV-error interval. The separately trained historical betting heads are **NOT_ADOPTED** and do not replace C0/Fixed50. [Research status](RESEARCH.md) preserves earlier A0 failures, G0/source-activation blocks and the unchanged forward calendar. No financial or tournament certification follows from the local cutover.

## 7. Local commands and deployment boundary

From the repository root, these inspect CLI contracts without inference, acquisition or database writes:

```bash
.venv/bin/python scripts/run_model_benchmark.py --help
.venv/bin/python -m betting_app.scripts.prepare_model_db --help
.venv/bin/python -m betting_app.scripts.run_upcoming_prediction_pipeline --help
```

With existing local research artifacts, the canonical source/hash check is:

```bash
.venv/bin/python scripts/run_model_benchmark.py --suite conf/base/research_benchmark.json --doctor
```

It requires `--research-root`, `ENSEMBLE_RESEARCH_ROOT` or `data/research_root.txt`; doctor does not score or train. Scoring additionally requires a **new** `--output-dir`; never overwrite sealed results. It remains retrospective development.

Only after preparing an explicitly owned, isolated PostgreSQL/Timescale database, the loopback API command is `.venv/bin/python -m uvicorn betting_app.api.main:app --host 127.0.0.1 --port 8000`. API startup does not create a schema or start the scheduler. Frontend development is separate in `client/`: `VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1`. Prediction/registration CLIs write data and also call ORM initialization; they must not be used to verify a clean migrated schema. Do not pass `--notify` in offline smoke.

The [ordinary Compose file](../docker-compose.yml) still mounts persistent `/data/inzynierka/pgdata`, uses fixed container names and includes a collector scheduler; its direct API port is now hard-bound to `127.0.0.1`. **`docker compose up` alone is not an isolated offline smoke** or permission to deploy. The [disposable override](../docker-compose.smoke.yml) removes the scheduler, uses unique names/tmpfs and an internal API/DB network with no published ports; only authenticated Caddy has loopback ingress. Actual owned-stack inspection and HTTP checks returned401 for all80 OpenAPI operations plus5 UI/docs paths before handler mutation. CORS is not authentication; the trusted-local direct API is not independently authenticated. The installed image's `/app/.env` is masked during smoke, not certified secret-free. No remote protection or fresh server state is claimed.

The [September 26 read-only production audit](../reports/production_readiness_and_market_benchmark_report_2026.md) observed EXP081 + market. The [October 5 export described in the October 6 audit](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md#2-scope-and-model-identities) also observed an EXP081-based hybrid, with zero C0 predictions, recorded bets or wallet transactions in that export. These dated observations do not establish today's server state or reconstruct unrecorded manual betting. No fresh server inspection or deployment is claimed here.
