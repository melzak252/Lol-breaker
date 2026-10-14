# TODO 02: Regional Family Rating Engine (`REGION_PLAN.md`)

## Historical roadmap notice — 2026-10-06

**Status: SUPERSEDED / RATING AND DEPLOYMENT CLAIMS NOT RECERTIFIED.** Preserve the original diagnostic numbers, proposed γ=0.70 rule, checklist and server claim below as historical assertions. “Verified discount”, a single G2–Gen.G example and “COMPLETED / deployed” are **not current evidence of calibration, financial reliability or the remote server's state**, and do not authorize applying those offsets to current native probabilities.

- The current [application contract](../betting_app/README.md#aktualny-kontrakt-c0) is local native **Causal-C0** (`c0-native-2026-w32-e12-v1`) plus the separate **Fixed50** logit hybrid (`Hybrid-Operational-Market`, `c0-native-2026-w32-e12-v1-a0.50-t1.00`), not active A1 or frozen annual A0. Annual A0 remains a **research reference**; [current research](../docs/RESEARCH.md) retains the later coherent-family improvement to the rating control and the failed native integration/promotion gates.
- Native C0 already returns whole-series probabilities; do not project BO again, add this fixed regional discount or substitute ratings when its artifact/input contract fails. Research rating systems, native neural models and tournament Glicko-derived proxies are distinct identities, not interchangeable C0/A0 aliases.
- [Plan11 corrective scope](./11_project_integrity_and_readiness.md) owns documentation reconciliation (D1), tournament predictor/decision/eligibility integrity (T1–T4) and serving provenance (S). Local code and historical production observations do not establish a current deployment.
- No completed experiment may be restarted under this roadmap. It grants no permission for collectors, automatic acquisition, scheduler/notifications, configured-user database operations, deployment or wagering; no EXP081, rating-consensus or randomly initialized model fallback.

## Preserved original roadmap (not current instructions)

**Priority:** P1 (High Modeling Impact)  
**Target Files:**
- `src/ratings/family_calibrated_glicko2.py`
- `src/ratings/glicko2_core.py`
- `rating-foundation-20260915/REGION_PLAN.md`

---

## 1. Problem Statement
The diagnostic audit demonstrated a systemic failure mode in international/inter-regional tournaments (MSI, Worlds, EMEA Masters, EWC):
- Baseline Causal A0 predicted a **$59.28\%$ win rate** for major region teams facing regional/lower-tier teams.
- The actual observed win rate was **$72.94\%$** (a $-13.66\%$ structural deficit).
- Cause: Player and team ratings calibrated on domestic leagues carry inflated point pools ("regional rating bubble"). Dominant players in ERLs or minor regions do not scale 1:1 against major regions.

---

## 2. Specification & Implementation

### A. Resolve Best-of Likelihood Mismatch
In `family_calibrated_glicko2.py`, ensure map-level latent probabilities ($p_{\text{map}}$) are mapped to series probabilities ($P_{\text{series}}$) via the exact independent series projection:
$$P_{\text{Bo3}}(p) = p^2 (3 - 2p), \quad P_{\text{Bo5}}(p) = p^3 (10 - 15p + 6p^2)$$
Never train map likelihood directly against a multi-game series outcome without series combinatorics.

### B. Regional Tier Discounting Factor ($\gamma = 0.70$)
When a team from a lower-tier/regional competition family faces a team from a major competition family (LCK, LPL, LEC, LCS):
1. Apply the verified discount factor $\gamma = 0.70$ to the lower-tier team's effective rating differential:
   $$\Delta r_{\text{effective}} = r_{\text{major}} - \gamma \cdot r_{\text{regional}}$$
2. In logit space, this corresponds to an offset of $\approx +0.65$ in favor of the major-region team.

### C. Explicit Rules for Transfers, Substitutes, and Decay:
1. **Historical Substitutes:** Explicit player ID matching; do not inherit the starter's rating for an academy call-up.
2. **Cross-Region Transfers:** When a player transfers between regions, decay prior rating variance (increase Glicko-2 $\text{RD}$) toward the destination family mean.
3. **Inactivity Decay:** Increase rating deviation ($\text{RD}$) according to elapsed calendar time between tournaments without mutating history.

---

## 3. Verification & Acceptance Criteria
1. Implementation & Deployment Status: **COMPLETED**
   - [X] Created `src/models/regional_scaling.py` implementing canonical regional offsets and $\gamma = 0.70$.
   - [X] Integrated regional offsets into `A1PredictorEngine.predict(...)` and `PredictionEngine.predict_from_features(...)`.
   - [X] Integrated regional offsets into `TournamentSimulator.estimate_matchup_probability(...)`.
   - [X] Tested on G2 vs Gen.G: shifts G2 from an unrealistic $57.0\%$ domestic bubble to $48.9\%$, correctly positioning Gen.G as the favorite.
   - [X] Preserves zero-offset identity for domestic matchups (e.g. Gen.G vs T1, G2 vs Fnatic).
   - [X] Deployed and verified live on production server `192.168.1.17:8000`.
