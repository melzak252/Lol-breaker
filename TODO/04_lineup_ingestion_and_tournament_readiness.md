# TODO 04: Lineup Ingestion & Tournament Readiness

## Historical roadmap notice — 2026-10-06

**Status: SUPERSEDED / CONFIRMED-LINEUP AND TOURNAMENT CERTIFICATION NOT ESTABLISHED.** The proposed 45/30-minute listener, “verified starters” updates, A1-primary/cache checklist, latency and test counts below remain historical text. They are neither an implemented authenticated lineup feed nor current certification of rules, draws, tournament calibration or financial performance. Cache speed and passing software tests cannot establish point-in-time predictive evidence.

- The current [application contract](../betting_app/README.md#aktualny-kontrakt-c0) uses local native **Causal-C0** (`c0-native-2026-w32-e12-v1`) and a separate **Fixed50** logit hybrid (`Hybrid-Operational-Market`, `c0-native-2026-w32-e12-v1-a0.50-t1.00`), not active A1. Frozen annual A0 is a **research reference**; [current research](../docs/RESEARCH.md) distinguishes native paths from the historical tournament Glicko-derived proxy comparison and withdraws its native C0/A0 adoption interpretation.
- Native C0 already returns series probabilities. Stored current rosters, operator-provided scenario lineups and prior-observed rosters are **not authenticated prior-available announcements** merely because their player IDs are complete. Frozen native history ends **2026-09-08**; a fresh prediction/cache timestamp does not refresh it or certify a live forecast.
- [Plan11 corrective scope](./11_project_integrity_and_readiness.md) owns tournament identities/decisions/origin-specific eligibility (T1–T4) and history/roster provenance (S1/S2). Future-trained artifacts, later rosters/results or uncertified rules/draws cannot establish a historical pre-event forecast. Static hypothetical simulation is not independent tournament assessment.
- Authentic source activation remains externally gated under [Plan06](./06_evidence_admission_after_a0_audit.md) and [Plan09](./09_c0_connector_and_forward_evidence.md), not an automatic scraper task: permission/retention/quotas, approved locators/bindings and real released-history/announcement evidence are prerequisites. Neither this notice nor the old steps authorize collectors, automatic acquisition, scheduler/notifications, configured-user database operations, deployment, wagers or restarting completed experiments/the stopped 10/10 loop. Missing native inputs/artifacts remain explicit refusals, never EXP081, rating-consensus or randomly initialized model fallbacks.

## Preserved original roadmap (not current instructions)

**Priority:** P2 (Medium Operational & Research Impact)  
**Target Files:**
- `betting_app/scrapers/`
- `src/models/tournament_*.py`
- `scripts/simulate_tournament.py`
- `conf/base/tournament_formats.json`

---

## 1. Problem Statement
Two external structural gaps limit production automation:
1. **Lineup Uncertainty:** Pre-match models currently rely on a `previous_observed_roster` proxy. In high-stakes matches or regional playoffs, last-minute substitutions (e.g. academy subs, illness) cause severe market line shifts against the model.
2. **Tournament Quarantine:** While the tournament engine possesses 91 profiles across 227 phases, **zero phases currently hold verified point-in-time rulebook and draw certification**. Per `AGENTS.md`, match-level Log Loss alone cannot validate multi-round Joint RPS or tournament placement accuracy.

---

## 2. Specification & Implementation

### A. Lineup Ingestion Scraper (30–45 Minutes Prior):
1. Implement a lightweight scraper/listener for official lineup releases:
   - Riot Lolesports official match API / official team Twitter feeds.
   - Run $45$ and $30$ minutes prior to scheduled match start.
2. Update `team_current_roster_players` in the database with verified starters.
3. Trigger instant re-inference if any of the starting five differs from the default `previous_observed_roster`.

### B. Tournament Certification Audit:
1. Audit historical rulebooks and bracket draws in `data/artifacts/leaguepedia-tournament-rules-20260908`.
2. Implement independent point-in-time publication verification:
   - Prove that tournament rules, tiebreak formats, and Swiss draw mechanics were publicly released before the tournament start timestamp.
3. Compute Joint RPS (Ranked Probability Score) for official placements and win counts across complete tournament editions, rather than relying solely on match Log Loss.

---

## 3. Verification & Acceptance Criteria
1. Scraper test:
   - Verify that an announced substitution updates the starting lineup in `canonical_matches` before match start.
   - Prove that re-inference fires and updates the probability before the betting deadline.
2. Tournament A1 & Caching Engine (COMPLETED):
   - [X] Integrated `Consolidated-A1` as the primary pairwise probability engine in `TournamentSimulator`.
   - [X] Built `TournamentSimulationCacheService` pre-computing standardized 5k, 10k, 20k, and 100k paths.
   - [X] Cached retrieval latency reduced from 8,000ms to < 1ms ($O(1)$ JSON cache).
   - [X] Added on-demand recalculation endpoint `POST /tournaments/{id}/recalculate` and UI recalculate button.
   - [X] 54/54 tournament unit tests passing.
