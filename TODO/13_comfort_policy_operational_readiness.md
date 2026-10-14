# Plan 13: Comfort Policy Operational Readiness, 1/4 Kelly Sizing, and Live Execution Guardrails

**Created:** 2026-10-11  
**Status:** IN_PROGRESS  
**Priority:** P0 (Capital Risk & Operational Readiness)  
**Target Subsystems:** `betting_app/core/ev.py`, `betting_app/core/staking.py`, `betting_app/services/bet_qualification_service.py`, `betting_app/services/upcoming_inference_service.py`, `betting_app/scheduler/tasks/predict.py`, `client/src/pages/MatchDetail.tsx`.

---

## 1. Goal

Integrate, wire, and verify the empirically proven **Comfort Policy** across the live betting application and scheduler pipeline to completely neutralize the Optimizer's Curse, eliminate the historical 34.5 pp EV overstatement gap, and enforce safe 1/4 Kelly capital allocation under the mandatory 12% Polish turnover tax.

### Measurable Acceptance Criteria:
1. **Zero Phantom EV Traps:** Every qualified bet in live execution must satisfy the Dynamic Logit Disagreement Gate ($\Delta_{\text{logit}} \le 0.85$) and the EV window ($0.05 \le \text{EV}_{\text{net}} \le 0.25$), bounding expected vs realized EV gap to $\le 2.0$ pp.
2. **Shin De-vigging as the Production Reference:** All live API endpoints and signal generators must strip bookmaker overround using Shin (1992) / Additive closed-form devigging, eliminating the empirical favorite-longshot bias.
3. **1/4 Kelly Staking Enforcement:** Signal recommendations and automated execution must size stakes using 1/4 Kelly ($f^* = \frac{\text{EV}_{\text{net}}}{O \cdot 0.88 - 1.0}, \; \text{Stake} = 0.25 \cdot f^* \cdot \text{Free Capital}$) with an absolute single-bet ceiling of 150.00 PLN (1.5% of a 10,000 PLN bankroll).
4. **Strict Tier-1 Gating:** Automated prediction generation must reject Tier-2 and regional tournaments without referee verification (limiting action strictly to LCK, LPL, LEC, LCS, Worlds, and MSI).
5. **No Regressions:** All 4,374+ unit tests and the canonical benchmark suite doctor (`scripts/run_model_benchmark.py --suite --doctor`) must remain 100% green.

### Non-Goals:
- No automated live money placement (the app remains an analytics, research, and manual bet-tracking platform).
- No retraining of native C0 weights or neural feature extractors.
- No loose betting on unconfirmed lineups or Tier-2 academy leagues.

---

## 2. Empirical Grounding & Theoretical Invariants

- **The Optimizer's Curse (Smith & Winkler 2006):** Unconstrained positive-EV selection acts as an extreme-value order statistic filter on model error ($\mathbb{E}[\epsilon \mid \text{EV} > 5\%] \gg 0$). On 1,391 matches from the 2026 season, the raw model promised $+36.56\%$ EV but realized only $+2.05\%$ ROI (34.5 pp overstatement).
- **Guard B Tail-Sensitivity Proof:** Logit-space distance $\Delta_{\text{logit}} = |\text{logit}(P_{\text{model}}) - \text{logit}(P_{\text{market}})|$ dynamically scales allowable probability gaps: $\pm 19.0$ pp at 50/50 coinflips, but $\le 3.9$ pp on $95\%$ heavy favorites. Capping $\Delta_{\text{logit}} \le 0.85$ prunes 127 toxic outlier bets and collapses the EV gap from $34.5$ pp to $-6.04$ pp.
- **Shin 2-Way Resolution:** For $n = 2$ outcomes, Jullien & Salanié (1994) iteration divides by $(n - 2) = 0$. The exact analytical solution is closed-form: $z = S - 1$, and $\pi_1 = q_1 - (S - 1)/2$.
- **1/4 Kelly Optimization:** On the 103 Tier-1 season matches, 1/4 Kelly with a 1.5% cap (150 PLN) generated $+2,333.50$ PLN net profit (+17.62% ROI, +23.3% bankroll growth) with maximum drawdown capped at 12.26%.

---

## 3. Phased Implementation Plan

### Phase 1: Core Pipeline & Gating Hardening (Backend)
- [ ] **Task 1.1:** Audit `betting_app/services/upcoming_inference_service.py` and `betting_app/scheduler/tasks/predict.py` to ensure `is_bet_eligible` evaluates Guard B ($\Delta_{\text{logit}} \le 0.85$), EV cap $[0.05, 0.25]$, and Tier-1 gating on every polled match.
- [ ] **Task 1.2:** Ensure `model_ev_signals` table stores `stake_suggestion` computed from 1/4 Kelly with the 150 PLN / 1.5% bankroll cap and records `logit_disagreement` in diagnostics JSON.
- [ ] **Task 1.3:** Wire referee lineup confirmation status ($T-35\text{m}$) as a prerequisite before emitting qualified signals.

### Phase 2: Automated Shadow Forward-Ledger (Paper-Trading)
- [ ] **Task 2.1:** Implement `scripts/run_shadow_forward_ledger.py` to record live incoming quotes, compute Shin probabilities, log 1/4 Kelly recommendations, and track CLV against closing lines.
- [ ] **Task 2.2:** Add idempotent post-match settlement worker that resolves shadow bets upon match finish and updates running equity curves without database locks.

### Phase 3: Frontend Transparency (UI)
- [ ] **Task 3.1:** Update `client/src/pages/MatchDetail.tsx`, `Matches.tsx`, and types to display:
  - Suggested 1/4 Kelly stake (PLN) and raw EV vs haircut EV.
  - Fair market probability (Shin) alongside model probability.
  - Logit divergence meter ($\Delta_{\text{logit}}$) with green/yellow/red safety status.
  - Transparent quarantine reasons (e.g. `dynamic_logit_market_divergence`, `unconfirmed_lineup`).
- [ ] **Task 3.2:** Verify frontend builds cleanly with tsc and lint.

### Phase 4: Verification & Operational Sign-Off
- [ ] **Task 4.1:** Run full test suite across API routers, EV calculators, staking helpers, and operational schemas.
- [ ] **Task 4.2:** Execute live loopback API health checks (`/matches`, `/bets`, `/signals`) on an isolated database.
- [ ] **Task 4.3:** Verify benchmark suite doctor (`scripts/run_model_benchmark.py --suite --doctor`) passes with status `READY`.
