---
type: scripts-index
tags: [scripts, reproducibility, thesis, project-structure]
project: EnsembleLegends
date: 2026-05-07
---

# Indeks skryptów

> [!abstract]
> Katalog `scripts/` jest pogrupowany według numerowanych rozdziałów pracy. Numer folderu odpowiada głównemu etapowi analizy: od przygotowania danych, przez EDA i modele, po symulacje finansowe, robustness oraz generowanie wizualizacji raportowych.

## Current models, preservation and historical tools

Use the **[model/study catalog](../docs/MODEL_CATALOG.md)** to choose a retained model and find completed methods/results. This script index is an interface reference—not a list of active model candidates or authorization to rerun frozen experiments.

| Purpose | Interface | Boundary |
|---|---|---|
| Canonical saved-prediction benchmark | `run_model_benchmark.py --suite --doctor` / an authorized new output directory | Annual A0/rating/039 controls; no training or deployment |
| Existing standalone prop inference | `train_prop_distribution.py predict --run-id prop_distribution_20261007_01 --match-id ID` | Frozen historical feature rows; Map1 kills/duration and series length, not a live adapter |
| Prop preservation/recovery | `preserve_prop_model.py create|verify|restore|replay` | Checksummed separate archive; never overwrite outputs, fit or activate models |
| Closed challenger experiments below | Their exact study runners and saved protocols | Archive/reproducibility tools; NOT_ADOPTED and gated NOT_RUN studies do not resume automatically |

The original prop run and its selected bundle stay in place. [Recovery ZIP, digest and proof](../docs/MODEL_CATALOG.md#2-preserved-standalone-prop-model) include all five study stages plus source dependencies and environment metadata. Actual create/verify and restored-source inference completed; old scientific receipts and retained failures are unchanged.

```bash
.venv/bin/python scripts/preserve_prop_model.py verify \
  --bundle data/06_models/model_preservation/prop_distribution_20261007_01.zip \
  --sha256 a115b26576b90b32997536aa9e1ee7eb1228dd6864bfa53e16e671b72982a232
```

`create --run-id ID --output NEW.zip` validates the original frozen study and refuses an existing output. `restore --bundle ZIP --sha256 HASH --destination NEW_DIR` checks all members before extracting; the directory must not exist. `verify` restores into its own temporary directory and compares full marginal components/PMFs in a fresh process. It does not certify live inputs, untested rows or a full raw-source rebuild. Copy the ZIP and receipt to separate storage for a real backup; only load trusted joblib/pickle.

## Aktualny punkt wejścia do badań

**`run_model_benchmark.py --suite conf/base/research_benchmark.json`** (ścieżka manifestu z katalogu głównego: `conf/base/research_benchmark.json`). Uruchamiaj z głównego katalogu repo według [instrukcji badań](../docs/RESEARCH.md).

- `--doctor`: sprawdzenie źródeł i hashy, bez liczenia i treningu.
- `--suite ... --output-dir data/08_reporting/benchmark/<nowy-run>`: wspólne metryki A0, ratingów,039 i OPEN z pokryciem.
- `--candidate-data ... --candidate-col p`: dołączenie kompletnego kandydata do tej samej próby.
- `run_unified_betting_evaluation.py production-calibration`: bazowy A0 na zapisanych
  kursach produkcyjnych, domyślnie ostatnie sześć miesięcy przed wymaganym
  `--end-date`; 12% podatku, surowe EV >5%, flat 100 PLN i osobne horyzonty.
  Wymaga `--panel`, `--fixture-coverage` oraz nowego `--output-dir`.
  Pełna komenda, pokrycie i wyniki: [stan badań](../docs/RESEARCH.md).
  To retrospektywna kalibracja, nie symulacja wykonania Kelly ani certyfikacja produkcji.
- Dawny tryb `--data ... --candidate-col ...` pozostaje kompatybilnym narzędziem pojedynczej kolumny; jego automatyczne bramki nie zastępują pełnego dopuszczenia produkcyjnego.

Poniższa numeracja opisuje historyczne narzędzia i wyspecjalizowane potoki, nie kilka konkurujących standardów benchmarku. Nowe eksperymenty powinny eksportować prognozy do jednego benchmarku.

### Bounded sports reliability improvement studies — 2026-10-10

- **Duration scale reliability (P1):**
  - Runner: `scripts/train_duration_scale.py` (`prepare`, `freeze`, `run`, `predict`).
  - Verifier: `scripts/verify_duration_scale_reliability.py`.
  - Evaluates mean-preserving scale reparameterization $\mu_{\text{new}} = \log(m) - \sigma_{\text{new}}^2 / 2$ holding frozen tree mean $m$ fixed: P0 (reference), PG (global scale control), PC (conditional scale candidate). Four estimation operations, zero base fits. Public CLI `predict` outputs calibrated parameters and survival quantiles.
- **Champion OOV and compositional representation (W1):**
  - Runner: `scripts/train_champion_oov.py` (`inventory`, `prepare`, `freeze`, `run`, `predict`).
  - Verifier: `scripts/verify_champion_oov_reliability.py`.
  - Evaluates natural champion-OOV vs seen-history support across 2026 assessment series. Strictly rejects fitted inference upon support gate closure. Independent streaming verifier audits 5.8M token lineage.
- **Exact-context historical lineup law and supported roster bounds (R1):**
  - Runner: `scripts/train_lineup_law.py` (`freeze`, `prepare`, `run`, `recover-evaluation`, `predict`).
  - Verifier: `scripts/verify_lineup_law_reliability.py`.
  - Learns conditional 6-parameter and global 2-parameter historical next-lineup laws from prior released appearances, CAL class presence audit, and evaluates supported roster bounds $[L, L+u]$ on native C0 scenario pairs. Supports technical evaluation recovery without refitting. Public CLI `predict` returns side laws and supported roster bounds.

### Native C0 EMEA raw-input reconstruction — 2026-10-10

`research/prepare_c0_emea_reconstruction.py --scope PATH [--research-root DIR]`
freezes the complete 3,103 native2026 targets, all89 primary EMEA Masters rows,
the complete1,467 Polish-primary cohort, source/model hashes and outcome-blind
control links. `research/run_c0_emea_reconstruction.py --protocol PATH` rebuilds
the eight native arrays from eligible raw maps and exercises genuine saved C0
inference in an isolated process. `research/run_c0_emea_diagnostics.py --protocol PATH`
retains favorable/unfavorable EMEA cases, exact-ID cross-competition histories,
all selected controls and the explicit cause/fit gate. Existing output namespaces
reject replacement; a copied namespace is not permission to repeat a study.

Recorded run:
[`c0_emea_reconstruction_20261009_230237Z_integrity_v2` protocol](../data/05_model_input/c0_emea_reconstruction/c0_emea_reconstruction_20261009_230237Z_integrity_v2/PROTOCOL.md),
[raw/native receipt](../data/08_reporting/c0_emea_reconstruction/c0_emea_reconstruction_20261009_230237Z_integrity_v2/tracks/reconstruction/receipt.json),
[cause ledger](../data/08_reporting/c0_emea_reconstruction/c0_emea_reconstruction_20261009_230237Z_integrity_v2/tracks/diagnostics/cause_ledger.json).
The89 matched links reuse19 unique controls, not89 independent controls.
These are reconstruction/diagnostic tools, not a new benchmark, live feature
adapter, model fit, league exception or source-acquisition authorization.


### Champion-aware historical outcomes and statistics — 2026-10-07

`train_champion_history.py` executes a separate, odds-free historical challenger.
Inputs are the last16 **released past maps per player**, not the upcoming draft:
champion/role, performance masks, pace, historical patch/competition and prior
opponent strength. Role-aware individual-player aggregation needs no organization
history. A legal terminal-score PMF supplies winner, length and map-play
probabilities; separate NB-kills and lognormal-duration marginals condition on
each map being played. Later maps share the same pre-series inputs.

The frozen A/B/C/D factorial hides/includes historical champions and
disables/enables prop supervision: three paired seeds, twelve completed
twelve-epoch trajectories and eight ensemble-level CAL scalars, zero C0/assessment
fits. A host restart required explicitly authorized identical-recipe recovery:
thirteen actual formal attempts include one lost partial fit of unknown progress;
three original checkpoints were restored and nine remaining fits completed.
TRAIN precedes2025-07-01, CAL is released2025H2, assessment is exposed2026 development data.
Exact rule evidence is optional context; UNKNOWN is not STANDARD. No draft
simulation, market input, live adapter, collection or production cutover.

```bash
# Recorded authorized run. prepare/freeze/smoke/run reject existing receipts;
# these commands are not authorization to repeat the completed experiment.
.venv/bin/python scripts/train_champion_history.py prepare --run-id champion_history_20261007_01
.venv/bin/python scripts/train_champion_history.py freeze --run-id champion_history_20261007_01
.venv/bin/python scripts/train_champion_history.py smoke --run-id champion_history_20261007_01
.venv/bin/python scripts/train_champion_history.py run --run-id champion_history_20261007_01
# Recorded one-time, explicitly authorized interruption recovery; do not repeat.
.venv/bin/python scripts/resume_champion_history.py
# Completed saved-only scoring repairs; zero neural/CAL fits.
.venv/bin/python scripts/score_champion_history.py --run-id champion_history_20261007_01
.venv/bin/python scripts/train_champion_history.py predict --run-id champion_history_20261007_01 --variant D --match-id 82590
```

`--research-root`, `ENSEMBLE_RESEARCH_ROOT`, then `data/research_root.txt` resolve
the archive. Feature, protocol, model, prediction and receipt stages use
`data/{04_feature,05_model_input,06_models,07_model_output,08_reporting}/champion_history/<run>`.
Saved replay checks frozen code/input/model hashes and rejects pre2026 rows.
It is a known-first-map-roster/day-release scenario, not issued live forecasts.
The existing prop baseline supplies first-map statistics only; later-map
evaluation retains explicit unsupported-baseline reasons. See the
[executed research report](../reports/champion_history_prediction_20261007.md).

The executed decision is **NOT_ADOPTED for every task and the complete bundle**.
The champion-aware D winner does not beat C0; B length improves against A but
not the existing length model; D duration beats the old model but not C without
champions. Models/output are retained for reproducibility, not serving.

The main benchmark owns saved-model rescoring; a new output directory is
mandatory. The command below ran successfully on3,103series/6,964maps and
recomputed twelve5,000-draw primary contrasts with maximum replay error0.

```bash
# Recorded execution; central_benchmark now exists and must not be reused.
.venv/bin/python scripts/run_model_benchmark.py --champion-history champion_history_20261007_01 --output-dir data/08_reporting/champion_history/champion_history_20261007_01/central_benchmark
.venv/bin/python scripts/report_champion_history_profiles.py --run-id champion_history_20261007_01
.venv/bin/python scripts/verify_champion_history_public.py
.venv/bin/python scripts/verify_champion_history_numerics.py --run-id champion_history_20261007_01
```

The temporal verifier's `data`, `perturb`, `saved-typed` and `saved` modes retain
full-source, physical input-mutation, typed invariant and actual-CLI receipts.
Original strict1e−12 C public equality failed(max4.5763e−7) and remains failed;
the separate pre-existing2e−6 public contract passes. The bounded process
variation is unexplained. TRAIN has no UNK1 champion/patch/competition tokens,
so corresponding OOV embeddings are not informatively trained. Do not call the
negative recipe exactly reproducible or silently patch/retrain it. The report
links all failures, numerical/temporal proofs and task-specific decisions.



### Stable player180 × unchanged meta30 C0 follow-up — 2026-10-07

`train_c0_meta_stable.py` implements the authorized, fixed180-day player-profile
follow-up. The original30-day meta features/normalization,100-observation local
pooling, hidden widths and fitting/CAL schedule stay unchanged. Separate activity
and support flags retain eligible C0 hidden history without recent games; both
player-only and meta-conditioned series-logit corrections use the same contract.
The original experiment and its negative result are preserved.

```bash
# Documentation of the bounded experiment, not authorization for another sweep.
# A separately authorized new execution requires a new, nonexistent run ID.
.venv/bin/python -B scripts/train_c0_meta_stable.py --freeze NEW_RUN_ID
.venv/bin/python -B scripts/train_c0_meta_stable.py --prepare data/05_model_input/c0_meta_stable/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/train_c0_meta_stable.py --smoke data/05_model_input/c0_meta_stable/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/train_c0_meta_stable.py --run data/05_model_input/c0_meta_stable/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/run_model_benchmark.py --c0-meta data/05_model_input/c0_meta_stable/NEW_RUN_ID/protocol.json --output-dir data/08_reporting/c0_meta_stable/NEW_RUN_ID/evaluation
.venv/bin/python -B scripts/train_c0_meta_stable.py --support-profiles data/05_model_input/c0_meta_stable/NEW_RUN_ID/protocol.json --benchmark-dir data/08_reporting/c0_meta_stable/NEW_RUN_ID/evaluation --output data/08_reporting/c0_meta_stable/NEW_RUN_ID/support_profiles.json
```

Archive resolution and isolated native-namespace preparation follow the original
CLI. No database, network, scheduler or server is used. The authenticated prior
seal/prepared features and unchanged native artifacts are prerequisites; freeze
rejects modified prior data even when a copied local hash inventory is repinned.
Source snapshots and six saved heads remain separate under `c0_meta_stable`.

Saved replay: `--predict data/06_models/c0_meta_stable/NEW_RUN_ID/bundle.json
--features data/04_feature/c0_meta_stable/NEW_RUN_ID/features.npz
--indices INDEX_A INDEX_B`; optional `--output` requires a new JSON path.
Replay checks the frozen model/source/input inventory and origin eligibility.
It exports the two corrections, unchanged C0 and role-meta embeddings; it is not
a live feature builder or roster adapter.

Completed run **`c0_meta_stable_20261007_01`** is **NOT_ADOPTED**: both genuine trained
heads receive exactly zero amplitude from the unchanged CAL procedure, so all
selected probabilities equal C0. Retaining inactive/native-only input support is
not a predictive gain. Results/proofs/limits: [research status](../docs/RESEARCH.md).


### Frozen-C0 30-day role-meta correction — 2026-10-07

`train_c0_meta.py` freezes, prepares, fits and replays one bounded historical
experiment. C0 weights stay fixed. The player-only and meta-conditioned branches
share topology; each learns a small antisymmetric correction to **series logits**,
not a probability blend or another BO projection. The meta branch learns 16-wide
global/local role embeddings and player-profile × meta interactions.

```bash
# New executions require a new, nonexistent run ID; do not overwrite this run.
.venv/bin/python -B scripts/train_c0_meta.py --freeze NEW_RUN_ID
.venv/bin/python -B scripts/train_c0_meta.py --prepare data/05_model_input/c0_meta/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/train_c0_meta.py --smoke data/05_model_input/c0_meta/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/train_c0_meta.py --run data/05_model_input/c0_meta/NEW_RUN_ID/protocol.json
.venv/bin/python -B scripts/run_model_benchmark.py --c0-meta data/05_model_input/c0_meta/NEW_RUN_ID/protocol.json --output-dir data/08_reporting/c0_meta/NEW_RUN_ID/evaluation
```

Archive resolution follows `--research-root`, `ENSEMBLE_RESEARCH_ROOT`, then
`data/research_root.txt`. Preparation alone activates the archived native namespace
in an isolated process; no database, network, scheduler or server is used.
Formal fitting is two branches × three seeds × 12 epochs on 2024 TRAIN, with two
bounded correction amplitudes fitted on 2025 CAL and 2026 assessment. Smoke uses
separate artifact directories and is not a recipe-selection result.

The fixed window is `[cutoff−30 days, cutoff)`, additionally requiring whole-series
release strictly before cutoff. Explicit historical patch labels are used only for
coverage/diversity/concentration; target patch is evaluation-only. Exact competition
context pools toward global context with 100 prior observations. The correction uses
players supported in the recent window; unchanged C0 retains its longer history.
The 2024 three-seed frozen history representations are concatenated, not averaged
across unaligned latent coordinates. Actual first-map rosters and daily release
remain conditional historical scenarios, not publication-certified inputs.

Saved inference requires the bound feature pack and origin-eligible row indices:
`--predict data/06_models/c0_meta/NEW_RUN_ID/bundle.json --features
data/04_feature/c0_meta/NEW_RUN_ID/features.npz --indices INDEX_A INDEX_B`.
It verifies source/model/input hashes and exports both corrected probabilities and
role-meta embeddings. A copied bundle cannot load checkpoints outside its verified
inventory. This is research replay, not a live roster adapter or production cutover.


### Manual OddsPortal collection with source datetimes — 2026-10-07

Maintained collectors now live here: `scrape_oddsportal.py` (current public league archives) and `refresh_oddsportal.py` (existing event URLs in a local CSV or validated inventory JSON). Shared implementation is in `src/utils/oddsportal_collection.py`, `oddsportal_inventory.py`, `oddsportal_bulk.py`, `oddsportal_current.py`, and `oddsportal_data.py`; no foreign checkout, database, scheduler or model is required. `requirements-betting.txt` includes the public-payload decryption dependency `pycryptodome`.

Run from the repository root, with a **new** run ID; these are bounded manual acquisition examples, not automatic jobs:

```bash
.venv/bin/python scripts/scrape_oddsportal.py \
  https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/ \
  --run-id oddsportal_lcs_manual_001 --max-matches 100 \
  --with-history --with-markets --concurrency 1 --delay-seconds 1
.venv/bin/python scripts/refresh_oddsportal.py \
  --input-csv data/05_model_input/oddsportal/oddsportal_live_20261007_083701Z/historical_year_sample.csv \
  --run-id oddsportal_refresh_manual_001 --max-matches 100 \
  --with-history --with-markets --concurrency 1 --delay-seconds 1
```

For **all available LoL events**, omit `--max-matches` and the archive URL. The unbounded CLI first inventories every published league/season/page, unions known historical URLs by exact event ID, then downloads every inventory record:

```bash
.venv/bin/python scripts/scrape_oddsportal.py \
  --known-events-csv data/01_raw/oddsportal/oddsportal_full_20261007_114857Z_seed/historical_matches.csv \
  --run-id oddsportal_full_NEW --with-markets --concurrency 4 --delay-seconds 1
```

The automatic inventory uses `<run-id>_inventory`; both namespaces must be new. To separate discovery and download, use `--inventory-only` with `scrape_oddsportal.py`, then `refresh_oddsportal.py --inventory-json data/02_intermediate/oddsportal/INVENTORY_RUN/matches.json --run-id DOWNLOAD_RUN --with-markets --concurrency 4 --delay-seconds 1`. Inventory JSON refresh does not accept a match limit. `--checkpoint-every` controls durable progress receipts (default100 events); settled records are appended to `settled-records.jsonl`, with one final CSV/JSON export instead of repeated full rewrites. Unexpected worker/disk errors propagate after preserving settled records.

`inventory-coverage.json` accounts for advertised leagues, seasons, pages, failures and empty source shells; `inventory-metadata.json` binds its exact event inventory. Legacy prices are never treated as fresh quotes. Source gaps can make archive completeness false even when every known event was attempted. A global access denial stops both discovery and later requests; no fallback claims full source history.

`--event-url URL` collects one exact event; an omitted archive URL discovers published LoL league slugs from the current esports catalogue. Archive collection follows actual advertised season/page locators, including renamed seasons and published locale-prefixed AJAX paths. Same-identity on-host locale redirects retain every response in the capture chain; unrelated-sport/host redirects are rejected. A stale current alias remains a named metadata failure, but its explicit historical LoL links are requested independently and validated under their own identities. Old empty shells are named source absences.401/403/429 or access challenges stop acquisition without bypass.

Each run retains pre-decryption HTTP-decoded entities and UTC capture clocks under `data/01_raw/oddsportal/<run-id>/`, decoded payloads and `matches.csv/json` under stage02, and `run.json` under stage08. `_open_at` is provider `openingChangeTime`; `_close_at` is provider **last change**, not certified official CLOSE. Unknown clocks remain null. Both array and exact string-indexed `0`/`1` two-outcome representations preserve source side order; arbitrary keyed pairs and invalid prices fail explicitly. Original archive/legacy aliases remain nested, not mixed with fresh normalized identity.

`--with-history` retains the rows actually returned, not a promised full price trajectory. `--with-markets` retains advertised additional payloads without guessing outcome labels, units or prop semantics. Only `COMPLETED` exits0; partial/failure/source-absence statuses exit1 and preserve ledgers. Existing run directories cannot be overwritten.

Actual pilot:151 distinct events/148 usable moneylines after retained-response normalization;2017–2026 coverage, eight observed bookmakers, source clocks and one additional three-outcome market. **Not a full16,659-event harvest.** Commands, source gaps, after-start clocks and verification: [research entry](../docs/RESEARCH.md#2026-10-07--oddsportal-live-datetime-pilot-and-local-collector). This manual acquisition does not authorize forward-source activation, training, production imports or deployment.

Full manual acquisition completed **2026-10-08**: **18,147 inventoried LoL events attempted, 18,004 usable moneylines, 143 explicit failures, zero events left unrequested**. All41 advertised league slugs and428 advertised archive pages were visited; source dates span2017-04-06 through2026-10-07. The combined [CSV](../data/02_intermediate/oddsportal/oddsportal_full_20261007_114857Z_complete_20261008_115636Z/matches.csv) and [JSON](../data/02_intermediate/oddsportal/oddsportal_full_20261007_114857Z_complete_20261008_115636Z/matches.json) retain all event IDs, including failed rows without legacy odds. The [receipt](../data/08_reporting/oddsportal/oddsportal_full_20261007_114857Z_complete_20261008_115636Z/run.json) remains **PARTIAL** because143 event failures,27 empty advertised season entries and one stale current alias remain explicit source gaps; it is not an unfinished download. [Measured verification and limitations](../docs/RESEARCH.md#2026-10-08--oddsportal-full-available-lol-acquisition-completed) include response hashes, source clocks, interrupted-run recovery and optional-market coverage. Full price history was not requested; no production import, model training or forward-source activation followed.

### Standalone prop distributions — 2026-10-07

`train_prop_distribution.py` is a separate **sports-only historical research** entry point, not a replacement moneyline benchmark or an application cutover. It predicts total **Map 1** kills, Map 1 in-game duration in minutes, and legal BO1/3/5 series map counts. It neither loads nor changes C0 weights/probabilities; the existing causal player/rating summaries and raw source are reused as inputs.

The fixed study is `prop_distribution_20261007_01`: TRAIN before2025-07-01, CAL2025H2, development assessment2026 through the frozen source end. Every target retains its task-specific labels/absences. All prior maps, including BO2, feed exact-ID player last16-map and optional organization last20-map pace; same-day results are withheld. Actual first-map rosters and effective daily releases are explicitly conditional research inputs, not authenticated historical publication.

Commands invoked for this study; **do not rerun freeze/run into these existing directories**:

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -B scripts/train_prop_distribution.py freeze --run-id prop_distribution_20261007_01
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -B scripts/train_prop_distribution.py smoke --run-id prop_distribution_20261007_01
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -B scripts/train_prop_distribution.py run --run-id prop_distribution_20261007_01
```

`freeze` requires an explicitly prepared `data/05_model_input/prop_distribution/<run-id>/protocol.json`; it validates the source/bank, creates a new stage04 dataset, and pins inputs/code plus protected C0 artifacts. `smoke` uses only TRAIN rows and records software checks separately. `run` selects each task once on CAL and never refits on assessment. New authorized experiments require a new protocol/run ID, not automatic retries.

Reusable implementation: `src/features/prop_distribution_data.py`, `src/models/prop_distribution.py`, `src/models/prop_distribution_runner.py`, and shared proper scores in `src/analysis/prop_distribution_evaluation.py`. Distribution objects expose negative-binomial kills, lognormal duration and legal categorical length probabilities; seed mixtures combine densities rather than average parameters. Saved selection can use different families per task. The model's `predict(frame)` accepts the frozen `x_*` feature schema and returns those task marginals; it is **not** a live roster/history adapter or a joint kills-duration/parlay law.

After a completed run, `predict --run-id NAME --match-id ID` verifies frozen inputs and bundle hashes and evaluates one retained historical row, without writes or fitting. Full protocols, model bundles, component forecasts, line diagnostics, profiles, intervals and decisions follow stages05/06/07/08 under `prop_distribution/`. Consult [the current research entry](../docs/RESEARCH.md) for measured results and limitations; old synthetic-line ROI is not market/profit evidence.

### Draft-aware prop information gate — 2026-10-07

`train_draft_prop_distribution.py` executes a separately frozen **first-map actual-composition diagnostic**, not a live pre-match draft policy. It reuses the existing standalone feature bank and unchanged series-length reference; it does not load or change C0 serving weights. Source/schema audit and exact edition/phase rule matrix precede fitting. `prepare` requires an explicit protocol and audited map-composition parquet; it does not scrape or reconstruct missing pick/ban sequences.

Executed run `draft_prop_20261007_01` is **COMPLETED / NOT_ADOPTED**. Four TRAIN mean regressors/four CAL dispersion fits; matched36,192/1,328/3,103. Stage2 joint gate fails, so Stage3 draft marginalization and Stage4 sequential Fearless simulation are **NOT_RUN_BY_FROZEN_GATE**, not implemented or implicitly scheduled.

Commands actually exercised; do not repeat immutable preparation/freeze/smoke/run into these existing directories:

```bash
.venv/bin/python scripts/train_draft_prop_distribution.py prepare --run-id draft_prop_20261007_01
.venv/bin/python scripts/train_draft_prop_distribution.py freeze --run-id draft_prop_20261007_01
.venv/bin/python scripts/train_draft_prop_distribution.py smoke --run-id draft_prop_20261007_01
.venv/bin/python scripts/train_draft_prop_distribution.py run --run-id draft_prop_20261007_01
.venv/bin/python scripts/train_draft_prop_distribution.py predict --run-id draft_prop_20261007_01 --match-id 82251
.venv/bin/python scripts/verify_draft_prop_distribution.py --run-id draft_prop_20261007_01
```

`predict` verifies frozen inputs, protected assets and model/output hashes, and evaluates one retained historical row. Actual compositions are labelled `ACTUAL_COMPOSITION_ORACLE_NOT_PREMATCH`; invalid compositions are explicit absences. It is not a roster/draft/patch availability adapter. The `matched_no_draft` variant is available through `--variant`; neither variant is promoted.

`verify_draft_prop_distribution.py` performs independent retained-parameter arithmetic and saved-model replay with no fits; it writes one immutable receipt. It verifies monthly/equal-month contrasts, not every source-proxy/profile statistic. Full source-feature reconstruction and bounded source-identity repair are separate receipts. The supplemental `audited_map_drafts.parquet` restores40quarantined identity cells without changing any prepared feature or the scored freeze; original frozen inputs remain provenance, not a silently rewritten recipe.

New implementation: `src/features/draft_prop_data.py`, `src/models/draft_prop_model.py`, `src/models/draft_prop_runner.py`; shared scores remain in `src/analysis/prop_distribution_evaluation.py`. Outputs follow stages04–08 under `draft_prop/`. [Executed report and limitations](../reports/draft_prop_20261007_01.md); no bookmaker superiority, betting readiness, future forecast or production cutover follows.

### Historyczny price-aware betting-head C0 — 2026-10-06

Istniejący punkt wejścia: `run_unified_betting_evaluation.py approach-comparison c0-historical-development`.
Implementacja: `src/analysis/c0_betting_development.py`, wspólne math/evaluation w `src/analysis/`.
To osobny historyczny eksperyment development-only, nie obejście G0 ani aktywacja collectorów.

Wykonane komendy oryginalnego treningu (nie powtarzać z ukończonym run ID):

```bash
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-historical-development \
  --freeze --run-id c0_betting_development_20261006_01
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-historical-development \
  --protocol data/05_model_input/c0_betting_development/c0_betting_development_20261006_01/protocol.json
```

18 fitów headów/0 bazowych, trzy λ na rodzinę, TRAIN/CAL742/363, CAL-only/no-refit.
Pierwotny protokół i kod treningu są zachowane; obecny kod ma poprawkę cash accounting,
więc nie należy przepisywać hashy starego protokołu, aby ponownie trenować.
Aktualny wynik **NOT_ADOPTED** i [dokładne metryki](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md).

Wykonane assessment-only po review, bez fitów i bez zmiany zapisanych headów:

```bash
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-historical-development \
  --reassess --protocol data/05_model_input/c0_betting_development/c0_betting_development_20261006_01/protocol.json \
  --execution-binding data/08_reporting/c0_betting_development/c0_betting_development_20261006_01/cash_review/reviewed_execution_binding.json
```

`reviewed_*.json/parquet` zachowują523 dojrzałe okazje, również zero-action przy brakach cen.
Stare wyniki514/57 pozostają provenance, nie aktualną estymacją cash. Rerun do istniejących
reviewed artefaktów jest odrzucany przed zapisem; zachowano też dokładne źródła rzeczywistego
assessmentu sprzed dodania tej blokady. Nie ma resume/overwrite/refit fallbacku.
Modele/λ, źródła i9 forward-pinned plików są weryfikowane osobno; sparse CI nie filtruje
zerowych denominatorów. Nowy eksperyment wymaga osobnej autoryzacji, nowego runu i protokołu.
Nie zmieniać dat Oct31/Nov1 ani starego10/10. Zależności z istniejącego `.venv`;
archiwum przez `--research-root`, `ENSEMBLE_RESEARCH_ROOT` lub `data/research_root.txt`.

### Tournament numerical integration — 2026-10-07

`run_tournament_distribution_benchmark.py` adds a separate numerical study of retained native
scenario inputs. It does **not** refit C0, rerun the closed residual study, authenticate
historical roster publication or promote a model.

```bash
# No persistent artifacts; eight paths/prefixes by default (--simulations overrides).
.venv/bin/python scripts/run_tournament_distribution_benchmark.py --numerical-smoke --seed 20261008
# Completed numerical run: for another execution, use a new, nonexistent run ID.
.venv/bin/python scripts/run_tournament_distribution_benchmark.py --numerical-benchmark --run-id tournament_numerical_20261007_01 --source-run-id tournament_c0_20261006_01 --integrations monte_carlo hybrid --simulations 10000 --seed 20261008
```

The default matrix requests Worlds 2023/2024 and MSI 2023/2024, unchanged native C0,
origin-eligible annual A0, prior-calibrated Glicko-format and uniform. Missing annual A0
2023 remains an absence. Stage05 freezes the pair inputs, observed-label identity and a
snapshot of the reviewed Markdown separately; stage07 stores forecasts; stage08 stores
scores, integration comparisons, decision and binding receipt. Existing namespaces cannot
be overwritten. Old replay pins remain historical lineage; the new runtime separately
binds current v4 and rejects hypothetical targets carrying actual result identity.

Public Python API: `simulate_tournament_bracket(..., integration="monte_carlo"|"exact"|"hybrid")`,
also available on `TournamentBracketSimulator`. Default Monte Carlo retains its RNG path.
Exact integration supports fixed knockout trees (up to 16 entrants) and bounded small
GSL, deterministic round-robin, graph, gauntlet and double-elimination programs.
Enumeration refuses more than 20 series or 1,048,576 worst-case finite-mixture paths;
there is no silent Monte Carlo fallback. Adaptive Swiss/random-draw exact requests,
score-dependent standings, checkpoints and unsupported observables are rejected explicitly.
Hybrid samples the adaptive prefix and quarterfinal draw, then integrates a supported
fixed-knockout suffix; an entirely supported MSI bracket is exact. Supplementary MSI exact
references reuse that result instead of running a duplicate integration.

`numerical_uncertainty` separates fixed-N iid complete-path Wilson/zero-hit bounds,
iid-prefix Hoeffding bounds and exact zero sampling error. Non-iid supplied-vector
designs do not receive fake binomial intervals. These are pointwise **integration**
intervals, not model-calibration or simultaneous confidence guarantees. Joint finalist
and qualifier-set log/Brier scores retain zero-probability infinite log losses without
clipping. Uniform feasible matching remains a declared draw scenario, not the organizer's
certified sequential lottery. [Research and execution](../reports/reliable_tournament_simulation_research_20261006.md).

### Marginal-preserving C0 tournament copula and actual joint queries — 2026-10-07

`run_tournament_distribution_benchmark.py` supports a separate frozen joint-law study:

```bash
# These commands actually ran; this occupied namespace cannot be overwritten.
.venv/bin/python scripts/run_tournament_distribution_benchmark.py --freeze-copula --run-id c0_tournament_copula_20261007_01
.venv/bin/python scripts/run_tournament_distribution_benchmark.py --run-copula --protocol data/05_model_input/tournament_copula/c0_tournament_copula_20261007_01/protocol.json
# An explicitly nonformal smoke uses --copula-smoke --protocol PATH, not --run-copula.
```

The completed run is **NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE**, with CAL guards
also failing:1formal TRAIN-only rho fit/0base fits,32cells/28forecasts/4absences,
50000complete paths each. More support alone does not repair the CAL failure.
Preserve this negative; these commands are not an automatic new fitting queue.
Standard `--research-root`/environment/local-root resolution remains supported.

`GaussianTournamentCopula` samples independent player-normal states for disjoint
five-player rosters and reuses them throughout a rollout. It preserves the supplied
pre-event C0 series margins; no organization history or second BO projection.
Use `copula_rho` on `simulate_tournament_bracket`/`TournamentBracketSimulator`;
positive rho requires pre-event Monte Carlo, rejecting exact/hybrid/checkpoints.
Arbitrary supplied strength vectors do not automatically become iid samples.

`query_tournament_joint_numerics(forecast, law, members, stage=...)` accepts
`joint_final_prob` or `joint_advance_prob`. It requires retained authoritative topology
and actual counts for MC confidence bounds; feasible unsampled sets receive zero-hit
bounds, structural zeros remain distinct, exact queries have zero sampling error.
Default/zero-copula simulation also preserves original pair orientation/rare tails
and original winner-oriented score bins, repairing boundary double-complementing.
These integration diagnostics do not certify model calibration or future tournament quality.

[Results, source incidents and verification](../reports/reliable_tournament_simulation_research_20261006.md#14-authorized-simulator-repairs-and-marginal-preserving-c0-tournament-experiment--2026-10-07).

### Plan11: uczciwe identyfikatory i zewnętrzne blokady — 2026-10-06

- Historyczny `run_tournament_distribution_benchmark.py` / `tournament_dist_20261005_01` porównywał **formuły Glicko-derived**, nie checkpointy C0/A0. Dla znormalizowanej różnicy ratingów `d`, różnicy stałych offsetów regionu `o` i nachylenia BO `s`, stary `a0` liczył `clip(sigmoid(s*(d+.70*o)/1.08),.001,.999)`; stary `c0` dodawał do tego samego bazowego `u` transform `clip(.92*sigmoid(logit(clip(u,.0001,.9999))/1.12)+.04,.12,.88)`. `glicko_format` to `clip(sigmoid(s*d),.001,.999)`. Pełne definicje, zakres historyczny i niezmienione metryki: [korekta tournament](../docs/RESEARCH.md#2026-10-05--tournament-distribution-and-bracket-simulation-benchmark-native-c0a0-attribution-withdrawn).
- Aktualne selektory proxy to **`uniform`, `glicko_format`, `regional_glicko_proxy`, `calibrated_regional_glicko_proxy`**; `--model c0/a0` jest odrzucane w ścieżce proxy. `glicko_format` używa nieujemnych nachyleń BO bez interceptu, dopasowanych wyłącznie na wcześniejszym CAL; brak formatu ma nazwany powód niedostępności. Nie przemianowywać archiwalnych artefaktów. Plan11 wykonał rzeczywisty CLI:16 model×edition scenarios,32 drawy/scenario,6 scalar prior-CAL fits, **`PROXY_DIAGNOSTIC_COMPLETED / native_adoption=false`**, oraz jawne odrzucenie `--model c0`. [Receipt wykonania](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/proxy_cli/plan11_proxy_smoke_20261006_01/receipt.json). Poniższe polecenia opisują inne możliwe operacje; Plan11 nie wykonał oddzielnego native-study:
  ```bash
  # Proxy: walidacja manifestu bez zapisu, nowy forecast i scoring wczytanego forecastu.
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --freeze-manifest
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --simulate --edition worlds-2024 --model regional_glicko_proxy --output data/07_model_output/tournament_distribution/new_proxy_forecast.json
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --score --forecast data/07_model_output/tournament_distribution/new_proxy_forecast.json --output data/08_reporting/tournament_distribution/new_proxy_score.json
  # Native: wykonane freeze/smoke/run run01; nie uruchamiać ponownie ukończonych katalogów.
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --freeze-study --run-id tournament_c0_20261006_01
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --study-smoke --protocol data/05_model_input/tournament_improvement/tournament_c0_20261006_01/protocol.json
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --run-study --protocol data/05_model_input/tournament_improvement/tournament_c0_20261006_01/protocol.json
  # Po wykonaniu frozen study: nowa symulacja i scoring rzeczywistego pliku.
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --native-study --simulate --protocol data/05_model_input/tournament_improvement/tournament_c0_20261006_01/protocol.json --edition lck-2026-road-to-msi --study-model residual_coherent --output data/07_model_output/tournament_improvement/new_lck_coherent.json
  .venv/bin/python scripts/run_tournament_distribution_benchmark.py --native-study --score --protocol data/05_model_input/tournament_improvement/tournament_c0_20261006_01/protocol.json --forecast data/07_model_output/tournament_improvement/new_lck_coherent.json --output data/08_reporting/tournament_improvement/new_lck_coherent_score.json
  ```
  Native `--benchmark --native-study --protocol PATH` deleguje do tego samego pełnego badania; nie uruchamia proxy. Opcjonalne `--prepare-native-2023 --run-id NAME` tworzy tylko osobny autoryzowany exact W32/E12 C0 control (3 seeds + shared experts); wykonać przed `--freeze-study --run-id NAME`, jeśli ten comparator ma być dostępny. Nie tworzy ani nie aliasuje A0 2023. Archive root rozwiązywany lokalnie przez `--research-root`, istniejące environment/helper lub `data/research_root.txt`, bez zewnętrznych zapytań/DB.
  Formalne stage05/06/07/08 są wyłączne, bez resume/overwrite; powyższe wyjścia native simulate/score także już istnieją, więc świadomy replay wymaga nowych ścieżek. **Wykonany run01: COMPLETED / NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE** ([wyniki i evidence](../docs/RESEARCH.md#2026-10-06--native-c0-tournament-challenger-not_adopted)). Extension TRAIN2024 dopasował trzy stałe filtry i jeden conditional score-law; CAL2025 wybrał q0.0001 raz po globalnym LL; ASSESS2026 tylko przewiduje/aktualizuje wcześniej wydane etykiety. Rollouts10k/seed20261006:56 komórek,29 forecastów/27 jawnych braków; tylko jedna primary2026 edycja versus minimum8. Canonical2026 LL0.535448701→0.535950648 i Brier0.181034493→0.181218284; guard Brier fails. Mniejsze coherent qualification/placement scores w LCK nie dowodzą poprawy, matched-independent jest lepszy, bootstrap jednej edycji degeneruje. MSI2026 jest wykluczone przez braki pierwszego raw rosteru; dlatego wykonany przykład CLI używa LCK. `first_event_actual_roster` to warunkowy scenariusz, nie proof publikacji. Exact-score diagnostics używają rzeczywistej skompilowanej binowej law i skończonych latent draws, oddzielnie od continuous TRAIN-law diagnostic; learned law przegrywa z fixed Markov0.35. Actual freeze/smoke/run/simulate-score i oba canonical suites wykonane,651 focused tests oraz dwa independent audits passed; brak promocji/deployment ani automatycznego rerun. Źródła independent verifierów zachowane w ZIP, nie jako kolejny benchmark skrypt.
- Stary status `TOURNAMENT_CALIBRATION_ADOPTED` wymagał tylko przewagi Champion Brier nad Uniform i flow feasibility. Ignorował zapisane kryteria Glicko Champion Brier i Uniform Swiss RPS; komentarz o Glicko Swiss RPS także nie był egzekwowanym gate'em. Wszystkie stare decision/metrics/output bytes pozostają dowodem historycznym, nie promocją native ani certyfikacją kalibracji. Rzeczywista aplikacyjna ścieżka C0 jest osobna; cutoff minimum2026-01-01 oryginalnego serving C0 wyklucza jego pre-event forecasts2023/24. Retrospektywny native research może korzystać z jawnych scenariuszy roster/history przy origin-eligible artefaktach, osobnym frozen protocol i pełnym wsparciu legalnych par; nie wymaga dowodu dawnej emisji, aby być research, ale bez niego nie jest autentycznym pre-start forecast. Osobna native-study gałąź i aktualny manifest są zachowane, nie wykonane lub certyfikowane przez Plan11. Autentyczna certyfikacja nadal wymaga rzeczywistych publikacji, bindings, rules/draws, pełnej macierzy coverage i niezależnych edycji; rerun proxy ani dogodna podpróba tego nie zastępują.
- Oddzielny Plan10: [smoke decision](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01_software_smoke/decision.json) pozostaje **SOFTWARE_ONLY /12 synthetic/0 formal fits**; [RED/support](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/red_and_support_receipt.json) nie jest wynikiem badania. Późniejszy [formal decision](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/decision.json) oraz [CLI receipt](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/actual_cli_verification.json) właściciela dają **COMPLETED /NOT_ADOPTED /12 formal heads/0 base fits/0 CAL refits**, wyłącznie exposed historical development. [Independent receipt](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/independent/receipt.json) raportuje PASS1,143,083 assertions, bez nowej rekonstrukcji native/globalnego optimum nonconvex return-MSE ani pełnej final application suite. Plan11 nie uruchamia, nie nadpisuje ani nie powiela tej pracy.
- S3: [źródłowy BLOCKED receipt](../data/08_reporting/a0_context_reliability/source_activation_20261004_01/receipt.json), [macierz8 źródeł](../data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json) i [native readiness](../data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_02/forward_readiness/readiness.json) są wcześniejszą proweniencją, nie nową weryfikacją. [Braki zewnętrzne Plan11](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/blocked_prerequisites.json): source-specific permission/retention/entitlements/quotas i remaining capacity; approved exact locators/bindings/captures; autentyczny live release-aware native builder/history/roster/coverage; official actual starts/settlements/revisions; locked przyszłe sample. Parser/transport/cache/tenant ID nie daje zgody ani autentyczności. Wszystkie8 źródeł nadal blocked; readiness **przed2026-10-31T00:00:00Z**, start **2026-11-01**, po niedotrzymaniu **EXPIRED_UNSTARTED**. Daty i future phase/maturity locks bez zmian; żadnego nowego sourceaudit, transportu, scheduler/notifications ani pozyskiwania.


### Model od surowych danych — eksperyment 2026-09-27

`train_prematch_forecaster.py` buduje nowe przyczynowe historie graczy,
trenuje kontrolne ratingi, regresję, LightGBM i sieci ról, a następnie
ocenia zamrożony wybór. Pierwotny przebieg nie ładował wcześniejszych wag
ani banku cech; kolejne kontrolowane porównania mogą jawnie używać jego
zweryfikowanych cech bez dziedziczenia wytrenowanych modeli.
Korzysta ze wspólnych metryk i bootstrapu w `src/analysis/`; nie ustanawia
drugiego standardu benchmarku.

- Etapy `prepare`, `develop`, `final` wymagają `--protocol PATH`.
  Wykonany protokół: `data/05_model_input/ground_up_lol_20260927_run001/protocol.json`.
- Każdy nowy trening wymaga nowego identyfikatora i katalogów wynikowych
  w kopii protokołu; ukończone przebiegi nie są nadpisywane.
- `feature_source_protocol` w nowym protokole pozwala `prepare` użyć tego
  samego banku bez kopiowania. Weryfikowane są źródła, polityka release,
  protokół źródłowy, kod cech oraz hashe wszystkich artefaktów.
- `calibration_months: 0` wymaga `candidate_variants: ["raw"]`.
  TRAIN obejmuje wtedy wszystkie etykiety wydane ściśle przed originem,
  bez nieużywanego okna CAL. Artefakt odrzuca żądanie kalibracji
  temperaturowej, zamiast tworzyć pozorny kalibrator tożsamościowy.
- `params.pool_players: true` w modelu tabularnym dodaje średnią i
  populacyjne odchylenie standardowe każdej cechy pięciu graczy.
  Braki nie stają się zerami; indywidualne pola ról pozostają wejściami.
  Jest to opcja eksperymentalna, nie automatyczna zmiana starego modelu.
- Bieżąca wersja cech całkuje wycentrowany margines BO1/3/5 przez
  kwadraturę 128-węzłową. Zachowuje dokładne 0.5 przy równych siłach;
  poprzednia kwadratura mogła wytworzyć znak błędu zaokrąglenia,
  który drzewo wzmacniało. Stare banki nie przechodzą kontroli hasha
  nowego kodu. Naprawę oceniono w osobnej wersji cech, nie przez
  nadpisanie wcześniejszych prognoz ani zmianę ich zapisanej selekcji.
- `predict --artifact PATH --match-json PATH` wykonuje czyste zapytanie
  o serię BO1/3/5 z pięcioma jawnymi identyfikatorami graczy na stronę.
  Dzień zapytania musi następować po stanie historii; zapisany snapshot
  nie odtwarza historycznych decyzji i nie pobiera nowych wyników.
- `plot_prematch_forecaster.py --protocol PATH` odtwarza wykresy z
  zapisanych wyników, bez treningu i ponownej selekcji.
- [Raport, wyniki, ograniczenia i komendy](../reports/ground_up_lol_prediction_20260927.md).
  [Aktualny handoff badań](../docs/RESEARCH.md).

### Łączenie historii i reprezentacji graczy

Ten sam `train_prematch_forecaster.py` udostępnia osobne etapy
`architecture-prepare`, `architecture-train`, `architecture-stack` i
`architecture-report`, każdy z `--protocol PATH`. Protokół:
`data/05_model_input/prematch_architecture_20260927_run001/protocol.json`.

- Przygotowanie łączy surowe tokeny historii z banku A0 z naprawionymi
  cechami modelu od zera. Dopasowuje identyfikatory meczu, strony i graczy,
  nie pozycje w tablicach; nie ładuje wytrenowanych encoderów A0.
- Porównanie obejmuje fuzję wejść, osobne encodery z łącznikiem oraz
  regresję łączącą prognozy specjalistów i symetryczny kontekst.
  Sieci różnią się też głębokością i liczbą parametrów: nie jest to
  eksperyment izolujący wyłącznie miejsce fuzji.
- Łącznik uczy się wyłącznie z wcześniejszych prognoz out-of-time,
  których etykiety były dostępne przed kolejnym originem.
- `predict-batch --artifact PATH --features PATH` przyjmuje plik NPZ
  zawierający `players`, `teams`, `context`, `baseline_series`,
  `history` oraz `history_mask`. Zwraca warianty prognoz serii.
  To interfejs poprawnie wyrównanych cech, nie adapter do bieżących
  składów; nie wdraża żadnego modelu w aplikacji.
- Raport eksportuje bazowe modele i łącznik osobno do wspólnego
  benchmarku, z różnymi rzeczywistymi granicami dostępności etykiet.
  Zamrożone A0 i manifest benchmarku pozostają bez zmian.

### Uczone korekty i łącznik zachowujący kalibrację A0

`train_a0_sports_correction.py` udostępnia etapy `prepare`, `train`,
`evaluate` i `predict`. Przygotowanie/trening/ocena wymagają `--protocol PATH`;
predykcja wymaga `--artifact PATH --inputs PATH`.
Eksperymenty zachowują A0 jako kontrolę, dopasowują strony/roster/release,
zapisują przeszłe membershipy oraz wykonują selekcję tylko na wcześniejszych
rolling-origin prognozach. Nie używają kursów do treningu.

- `python -m src.models.a0_fresh_correction --protocol PATH` rozszerza
  zamknięty head-development o wcześniejsze checkpointy ustalonego raw trio
  i trenuje korektę. Dziedziczona selekcja projektu trio na2024–2025H1
  wyklucza nazywanie tych samych lat niezależnym assessmentem.
- `python -m src.models.a0_probability_connector train --protocol PATH`
  uczy pojedynczą wagę convex pool bez dodatkowej kalibracji A0.
  Protokół wykonanego runu:
  `data/05_model_input/a0_sports_correction/a0_fixed_calibration_connector_20261001_03/protocol.json`.
  Nowy trening wymaga osobnego, zamrożonego protokołu/run ID; nie nadpisuj
  ukończonych przebiegów ani nie uruchamiaj automatycznej siatki.
- `python -m src.models.a0_probability_connector predict --artifact PATH --inputs PATH`
  przyjmuje tablice `p_a0`, `p_specialist`, `date`, `best_of`.
  To zgodne identyfikacyjnie i stronowo prawdopodobieństwa **serii**, nie map;
  nie stosuj ponownej projekcji BO. CLI sprawdza przedział dat artefaktu,
  lecz nie buduje cech i nie certyfikuje pochodzenia podanych prawdopodobieństw.
  Odtwarzanie natywnego A0 i świeżych modeli wymaga oddzielnych procesów.
- Lipcowy artefakt run03 obejmuje lipiec–wrzesień2026, nie przyszłyQ4.
  [Raport §10](../reports/lol_model_improvement_research_20260927.md#10-executed-learned-sports-connectors--2026-10-01)
  zawiera uruchomiony przykład CLI, wyniki dwóch odrzuconych receptur,
  zachowany development candidate i nadal niespełnioną wiarygodność EV.
- Każdy run eksportuje `predictions.parquet`/`p_correction` do istniejącego
  `run_model_benchmark.py --suite --candidate-data PATH --candidate-col p_correction`.
  Manifest i model operacyjny pozostają niezmienione; brak wdrożenia,
  certyfikacji turniejów lub zgody na H/market fitting.

### Natywna pojemność × ciągły czas treningu

`train_a0_capacity_experiment.py` wykonuje zamrożone IDEA-028 oraz warunkowe
IDEA-029. Uruchamiaj z katalogu głównego repo:

```bash
.venv/bin/python scripts/train_a0_capacity_experiment.py --freeze-run <nowy-run>
.venv/bin/python scripts/train_a0_capacity_experiment.py \
  --protocol data/05_model_input/a0_capacity/<nowy-run>/protocol.json --smoke
.venv/bin/python scripts/train_a0_capacity_experiment.py \
  --protocol data/05_model_input/a0_capacity/<nowy-run>/protocol.json --distill
```

- `--freeze-run` sprawdza bank, kanoniczne tożsamości/strony, oryginalne
  receipts C0 i hashe zależności przed zapisem protokołu. `--research-root`
  rozwiązuje archiwum; podczas wykonania musi zgadzać się z zamrożonym rootem.
- Formalny run to dziewięć trajektorii width16/32/64 × trzy seedy do24epok,
  z checkpointami12/24 bez resetowania AdamW ani RNG. Kontrola width32/12
  musi przejść CAL/assessment i zapisane Plan07 C0 przed treningiem innych szerokości.
- Zachowana jest natywna inicjalizacja i dokładny probability-mixture C0,
  nie nowe losowanie wyjścia ani zaokrąglona logit-mixture. Neural Beta CAL
  jest dopasowywane dla komórki; shared experts, ich slopes i mixture pozostają stałe.
- `--smoke` zapisuje osobny suffix `_smoke`: jeden seed/jedna epoka,
  ograniczony TRAIN/CAL, BO1/3/5 i wszystkie tryby. Trzy repliki tego samego
  endpointu służą wyłącznie interfejsowi inferencji; nie są trzema fitami.
- `--distill` jest częścią jednego formalnego wykonania, nie drugim treningiem
  już ukończonego runu. Gate może zapisać skip; przy dopuszczeniu teacher
  width32/12 używa wyłącznie TRAIN i surowych terminal distributions,
  a trzy KD fits porównuje się z istniejącą kontrolą width16/12.
- Etapy05–08: `a0_capacity/<run>/`. Parquet zawiera komórki i tryby,
  winner proper scores oraz osobne neural scoreline NLL. Sports gates
  i diagnostyczny Pinnacle EV na stałych330/76 wyborach są rozdzielone.
  G0, kanoniczny manifest i produkcja nie zmieniają się.
- Hashy ani ukończonych katalogów nie nadpisuj. Nowa komenda formalna
  wymaga osobnego run ID i autoryzacji; saved checkpoints zachowują optimizer/RNG.


### Historyczna diagnostyka persistence kontekstów C0 — run02

Moduł CLI: `src.models.a0_context_reliability_runner`. Uruchamiaj z katalogu głównego repozytorium; wymagane archiwum rozwiązuje się przez `--research-root`, `ENSEMBLE_RESEARCH_ROOT` lub `data/research_root.txt`.

```bash
.venv/bin/python -m src.models.a0_context_reliability_runner freeze --run-id <nowy-run-id>
.venv/bin/python -m src.models.a0_context_reliability_runner run \
  --protocol data/05_model_input/a0_context_reliability/<nowy-run-id>/protocol.json
.venv/bin/python -m src.models.a0_context_reliability_runner smoke \
  --output-dir data/08_reporting/a0_context_reliability/<nowy-katalog-smoke>
.venv/bin/python -m src.models.a0_context_reliability_runner predict \
  --artifact PATH --inputs PATH --output NEW_PATH
```

- Freeze tworzy protokół dla nowego identyfikatora; nie nadpisuj ukończonych runów. `run` wykonuje formalną zamrożoną recepturę i zapisuje artefakty pod `data/{05_model_input,06_models,07_model_output,08_reporting}/a0_context_reliability/<run-id>/`. `smoke` to deterministyczny dowód software na syntetycznych danych, nie wynik naukowy. `predict` używa zapisanego artefaktu, jawnych wejść i nowego wyjścia.
- **Historyczne/past-only:** `predict` dopuszcza wiersze wyłącznie w historycznym oknie `[scenario_origin, valid_until)` zapisanego artefaktu; daty poza nim są odrzucane. Run02 kończy zakres przed2026-09-09. To replay scenariusza, nie dowód historycznej emisji ani uprawnienie do bieżącej/przyszłej inferencji. Nie przenoś run02 na przyszłą kalibrację.
- Wykonany `context_reliability_20261004_02`: persistence **niepotwierdzona**, gałąź conditional `NOT_RUN`; sześć globalnych fitów diagnostycznych, zero conditional/context i base fitów. Zachowuj ten status. Brak confirmation nie jest dowodem braku efektów ani wynikiem predictive-calibration accuracy. Dokładne wyniki, evidence i source blockers: [RESEARCH handoff](../docs/RESEARCH.md#2026-10-04--context-reliability-persistence-not-confirmed), [decision](../data/08_reporting/a0_context_reliability/context_reliability_20261004_02/decision.json), [source prerequisite matrix](../data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json).
- Wszystkie osiem źródeł pozostaje zablokowanych. Nie wysyłaj zapytań ani nie wyciągaj zgody na pobieranie z dostępności parsera/CLI; deadline Oct31 i start Nov1 pozostają bez zmian.

### Trening A0 z niepewnością zespołu

`train_a0_uncertainty.py` uruchamia zamrożony eksperyment z rzeczywistą
architekturą Causal A0: pięciu członków `seed_only` oraz pięciu
`block_bootstrap`, roczne podziały TRAIN/CAL/TEST i kalibracja bez użycia TEST.
Zapisuje modele w `data/06_models/`, prognozy w `data/07_model_output/`,
a ocenę w `data/08_reporting/a0_uncertainty/`. Do metryk modeli wywołuje
powyższy wspólny benchmark — nie tworzy konkurencyjnego standardu oceny.

- Wymaga wcześniej zapisanego `--protocol` i archiwum wskazanego przez
  `--research-root`, `ENSEMBLE_RESEARCH_ROOT` lub `data/research_root.txt`.
- `--workers 2 --device cpu` mieści się w zamrożonym limicie zasobów.
  `--device cuda` wymaga `--workers 1` i zgodnego środowiska PyTorch/CUDA.
- `--resume-checkpoints` wznawia nieukończony przebieg po kontroli protokołu,
  danych, wersji i checkpointów. Zmiana kodu wymaga zachowania jego oryginalnej
  kopii; gotowy przebieg nie jest nadpisywany.
- Rozrzut członków i kwantyle nie są skalibrowanymi przedziałami ufności.
  Szczegóły i wyniki: [stan badań](../docs/RESEARCH.md).

### A0: stały podział, market focus i maskowanie historii

`train_a0_data_focus.py --protocol PATH` obsługuje zamrożone eksperymenty
bez walk-forward refitów. `a0_data_focus_v1` trenuje pięć wariantów:
uniform, targety major/international, market-tier weights, token dropout i kombinację.
`a0_history_reconstruction_v1` ponownie używa zapisanej kontroli native12
i wspólnych ekspertów; trenuje native6+12 oraz reconstruction6+native12,
po trzy seedy z identycznym budżetem aktualizacji.
`--evaluate-only` osobno ocenia zapisane prognozy przez istniejące metryki,
kontrakty tożsamości i bootstrap w `src/analysis/`; bez clippingu prawdopodobieństw.
`--smoke` wykonuje krótki fit i odtworzenie modeli w osobnym katalogu smoke.

- Wykonany protokół: `data/05_model_input/a0_data_focus/a0_data_focus_20260929_192443/protocol_v2.json`.
- TRAIN przed 2025-07-01, CAL 2025 H2, development validation 2026.
  Interwencje dotyczą komponentu neuronowego; raport pokazuje również
  nowe pełne złożenia A0, nie podszywa ich pod zamrożony roczny model.
- Archiwum rozwiązywane przez `--research-root`, `ENSEMBLE_RESEARCH_ROOT`
  lub `data/research_root.txt`. Modele, prognozy i raporty pod
  `data/{06_models,07_model_output,08_reporting}/a0_data_focus/<run>/`.
- Gotowych artefaktów nie nadpisuje się: ponowny eksperyment wymaga nowego
  run ID/protokołu. `--evaluate-only` odrzuca już ukończony raport.
- **Wynik negatywny:** żaden wariant nie poprawił market LL pełnego modelu
  względem uniform. [Wyniki i wykonane komendy, §10](../reports/bookmaker_noise_and_provider_training_20260927.md#10-2026-09-29--executed-single-split-data-focus-and-history-masking).

**IDEA-026, wykonana2026-10-01:** protokół
`data/05_model_input/a0_data_focus/a0_history_reconstruction_20261001_010503/protocol.json`.
CSD@15/XPD@15 są maskowane tylko tam, gdzie faktycznie je zaobserwowano;
tymczasowy dekoder nie jest częścią inferencji. Obie nowe receptury resetują
optymalizator na granicy faz i wykonują5,094aktualizacje/seed. Smoke ogranicza
każdą nową fazę do jednego równoważnika epoki; C0/eksperci pozostają zapisanymi referencjami.
Primary nawszystkich3,103targetach2026: reconstruction−matchedcontrol
ΔLL+0.001411[+0.000106,+0.002930], więc rekonstrukcja nieadoptowana.
Native6+12 dało wtórny sygnał historyczny, ale późniejszy Plan07 nie potwierdził
go na primary2024–25; nie promować tej receptury.
[Wynik pierwotny, §10](../reports/noisy_data_reliability_research_20260929.md#10-executed-matched-reconstruction-experiment--2026-10-01).

### Plan08: reliability-first comparison of native6 and native12

The completed [Plan08](../TODO/08_reliability_first_model_tradeoff.md) compares six versus twelve native epochs in run `reliability_tradeoff_20261001_092401_integrity_v2`. Primary LL and fixed-A0 quarter-cell RMS gates both fail; retain annual A0. [Full result and limitations, §13](../reports/noisy_data_reliability_research_20260929.md#13-reliability-versus-log-loss-plan08-2026-10-01), [sealed decision](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/decision.json).

- `train_a0_reliability_tradeoff.py --protocol PATH` runs the nine-fit frozen recipe; `--smoke` is a separate short preflight, not an assessed candidate.
- `run_model_benchmark.py --reliability-tradeoff PATH --output-dir NEW_PATH` evaluates saved tradeoff forecasts through the existing assessment CLI. The completed assessment is write-once; use a new nonexistent output directory only for an intentional replay.
- Protocol: `data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json`. Predictions and endpoint probabilities: `data/07_model_output/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/`. Assessment/evidence: `data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/`.
- Keep the retired pre-integrity attempt and its provenance correction explicit. Nine learned parameter hashes match; serialized checkpoint bytes do not. The observed replay discrepancy is below inherited tolerance but is not bitwise exact and has no established cause. No source-genuine future forecasts, production change, or automatic follow-up.

### Plan07: wejścia, replikacja roczna i komponent rodzin

Program jest zakończony: [plan i decyzje](../TODO/07_model_prediction_reliability.md),
[raport i dowody §11](../reports/noisy_data_reliability_research_20260929.md#11-executed-model-reliability-program--2026-10-01).
Nie uruchamiać kolejnego treningu automatycznie. Roczne A0 i manifest kanoniczny
pozostają bez zmian; żaden nowy ogólny kandydat nie przeszedł promocji.

- `predict_causal_a0.py --fixtures PATH --metadata PATH --source-matches PATH
  --release-days PATH --output NEW_PATH`: lokalna budowa snapshotów i natywna inferencja,
  bez kursów, sieci i bazy. Format/tryby/ograniczenia:
  [README aplikacji](../betting_app/README.md#repaired-fixture-input-i-status-dowodów--plan07).
  Status wejścia nie certyfikuje jakości prawdopodobieństwa ani publikacji źródeł.
- `train_a0_reliability.py --freeze-run RUN_ID`: zamraża już zapisany cel przed fitem;
  `--protocol PATH` wykonuje stałe native12/native6+12 dla rocznych originów.
  Opcjonalne `--smoke` jest oddzielnym, nieocenianym preflightem jednej epoki.
  Ukończony protokół: `data/05_model_input/model_reliability/model_reliability_20261001_055321/neural_protocol.json`.
- `experiment_family_reliability.py --write-protocol PATH --run-id NEW_RUN_ID`
  zamraża nowy kontrolowany eksperyment; `--protocol PATH` wykonuje replay,
  a `--smoke` uruchamia matematyczne oracles. Nie zmienia legacy engine.
  Ukończony run ma sufiks `_family_alignment_v2`; wcześniej nieudana próba pozostaje.
- `run_model_benchmark.py --reliability-study PATH --output-dir NEW_PATH`
  ocenia zapisane wyniki; `--reliability-integration PATH --output-dir NEW_PATH`
  wykonuje tylko zamrożoną warunkową integrację na przeszłym CAL i jej ocenę.
  To tryby wspólnego benchmarku, nie konkurencyjne standardy.
- Wszystkie subprotokoły przypinają źródła/wejścia; wyniki są niemodyfikowalne.
  Assessment v2 wymaga swojego sourceZIP sprzed dodania końcowego CLI integracji.
  Nie przepisywać hashy, aby dopasować stare wyniki do nowego kodu.
  Archiwum rozwiązywać przez istniejące `--research-root`/zmienną środowiskową/plik.

## Kolektory aplikacji — ograniczanie żądań Liquipedia

Stan lokalnego kodu po porządkowaniu 2026-10-05; bez uruchomienia kolektorów
na zewnętrznych źródłach i bez wdrożenia.

- `GET /api/tournaments/{id}`, `/simulate` i `/recalculate` korzystają wyłącznie
  z lokalnej drabinki oraz modelu C0. Brak lub wygaśnięcie cache nie uruchamia
  pobierania źródła. Przeliczana jest żądana liczba symulacji, nie cztery głębokości.
- Pobieranie drabinki wymaga jawnego `/sync`; domyślne `force=false` zachowuje
  cache. `force=true` nie omija budżetu żądań Liquipedia. Import wikitext/HTML
  pozostaje offline także przy pustym lub błędnym wejściu. Udana synchronizacja
  unieważnia poprzednią symulację, a nieudana zachowuje oryginalny czas snapshotu.
- `betting_app/services/liquipedia_transport.py` jest wspólnym transportem
  klienta meczów/składów i drabinek. Cache domyślnie znajduje się w
  `data/cache/liquipedia`; `LIQUIPEDIA_CACHE_DIR` pozwala wskazać wspólny katalog.
  API i scheduler muszą współdzielić ten trwały katalog. Budżet nie jest
  globalny między maszynami z osobnymi katalogami.
- Poprawne odpowiedzi klienta są przechowywane przez 6 godzin, a żądania
  drabinki przez 30 minut. Blokada per host zapobiega równoległemu pobraniu
  tego samego zapytania między procesami. Minimalne odstępy wynoszą 2 sekundy
  dla żądań i 30 sekund dla `parse`/`expandtemplates`.
- HTTP 403 wstrzymuje nowe żądania do hosta na 24 godziny. HTTP 429 wstrzymuje
  je na co najmniej godzinę lub dłuższy poprawny `Retry-After`. Błędy zapytania
  są zapamiętywane przez 5 minut; odpowiedzi API `ratelimited`/`maxlag` również
  wstrzymują host. Uszkodzony stan blokuje pobieranie zamiast zerować budżet.
- Zwykli klienci nie czekają na odstęp ani nie ponawiają błędu. Jawne CLI
  składów może odczekać odstęp do 30 sekund, lecz nie cooldown. Wyszukiwanie
  alternatywnego tytułu następuje tylko po błędzie brakującej strony, nie po
  timeout, 403, 429 ani ogólnym błędzie API.
- Dzienny job 05:00 aktualizuje tylko ticker/BO. Osobny job 05:15 weryfikuje
  składy raz, z limitem 50 unikalnych zespołów i timeoutem 7200 sekund.
  Jawne listy zespołów są przycinane i deduplikowane. Interfejs turniejów nie
  ponawia automatycznie nieudanego ENC; ponowienie wymaga działania użytkownika.

Bezpieczne sprawdzenie interfejsów CLI z katalogu głównego, bez pobierania:

```bash
.venv/bin/python -m betting_app.scripts.sync_liquipedia_bon --help
.venv/bin/python -m betting_app.scripts.verify_and_sync_team_rosters --help
```

To konserwatywna polityka lokalna, nie dowód uprawnień do źródła ani gwarancja
braku blokady po stronie Liquipedia. Rzeczywiste uruchomienie zbierania wymaga
osobnej autoryzacji; nie należy usuwać cache/cooldown w celu wymuszenia żądań.

## Zasada uruchamiania

Skrypty należy uruchamiać z katalogu głównego projektu, po aktywowaniu środowiska wirtualnego:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\03_dane_pipeline\00_profile_datasets_for_whitepaper.py
```

> [!note]
> Część skryptów wykorzystuje lokalne pliki danych z katalogu `data/`, np. `data/golgg_matches.json`, `data/odds.csv`, `data/golgg_y_predicts.csv` oraz `data/golgg_stacking_results.csv`. Katalog `data/` nie jest wersjonowany, dlatego trzeba go uzupełnić lokalnie przed uruchomieniem pipeline'u.

---

## Numeracja rozdziałów skryptowych

| Folder | Odpowiedni etap / rozdział | Rola |
|---|---|---|
| `03_dane_pipeline/` | Dane, mapowanie i jakość datasetów | Profilowanie danych, mapowanie GOL.GG ↔ OddsPortal, sanity checks. |
| `04_eda_rynek/` | EDA gry i rynku | Analiza rozkładu meczów, formatów BoN, rynku opening/closing, marż, arbitrażu i cech esportowych. |
| `05_ratingi_baseline/` | Ratingi i baseline'y | Generowanie ratingów, burn-in, porównanie player/team ratings, Market Open/Close i strojenie TrueSkill/OpenSkill. |
| `06_metamodel/` | Metamodel sportowy | Trening, diagnostyka, ablation studies, Optuna/walk-forward i wykresy metamodelu. |
| `07_model_hybrydowy/` | Hybryda model + rynek | Alpha sweep, temperature scaling, dynamic alpha, odds shopping i diagnostyki hybrydy. |
| `08_symulacje_finansowe/` | EV, staking i bankroll | Kelly, fixed stake, symulacje bankrolla, yield, ROI i wykresy finansowe. |
| `09_robustness_walidacja/` | Robustness i walidacja | Stress testy, CLV, segmentacja zysku, bootstrap i stabilność wyników. |
| `10_wizualizacje_raportowe/` | Materiały końcowe | Statyczne i prezentacyjne wizualizacje generowane z wyników eksperymentów. |

---

## Kolejność odtwarzania pipeline'u

Minimalna kolejność odtwarzania wyników wygląda następująco:

1. `03_dane_pipeline/` — przygotowanie i profil danych.
2. `04_eda_rynek/` — opis gry, rynku i jakości cen.
3. `05_ratingi_baseline/` — ratingi i baseline'y.
4. `06_metamodel/` — finalny model sportowy.
5. `07_model_hybrydowy/` — połączenie modelu z rynkiem.
6. `08_symulacje_finansowe/` — przejście od prawdopodobieństwa do decyzji bettingowej.
7. `09_robustness_walidacja/` — stress testy i sanity checks.
8. `10_wizualizacje_raportowe/` — figury końcowe do dokumentów.

> [!check]
> Taki układ utrzymuje zgodność między kodem a kolejnością etapów pracy: czytelnik może przejść od etapu pipeline'u do odpowiadającego mu folderu skryptów.

---

## Narzędzia benchmarkowe (w katalogu głównym `scripts/`)

W katalogu `scripts/` znajdują się bezpośrednie CLI do diagnostyki i ewaluacji modeli:

1. **`scripts/run_odds_bracket_benchmark.py`**:
   - Dzieli przestrzeń na 7 koszyków kursowych (od ciężkich faworytów $<1.40$ po wysokie underdogi $>3.50$).
   - Mierzy błąd kalibracji, Brier score, LogLoss modelu vs rynkowy LogLoss no-vig oraz identyfikuje koszyki z overconfidence.

2. **`scripts/run_financial_benchmark.py`**:
   - Kompleksowy benchmark finansowy symulujący zakłady o $\text{EV}_{\text{net}} \ge 5\%$ z uwzględnieniem 12% polskiego podatku obrotowego.
   - Segmentuje wyniki według przedziałów kursowych, poziomów EV oraz wskaźnika CLV (Closing Line Value).
   - Szczegółowe przykłady użycia i instrukcja porównywania modeli (czysty model vs hybryda) znajdują się w głównym `README.md`.

