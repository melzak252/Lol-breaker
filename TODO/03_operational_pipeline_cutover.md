# TODO 03: Live Operational Pipeline Cutover

## Historical roadmap notice — 2026-10-06

**Status: SUPERSEDED / NOT A LIVE CUTOVER AUTHORIZATION.** The EXP081 “currently” statement, annual-A0 hybrid registry example, sports-consensus fallback and pipeline/CLV instructions below describe an old proposal. They are not the active local configuration or a current inspection of the remote server. Preserve the recorded comparisons without treating them as financial certification or permission to run the historical pipeline.

- The current [application contract](../betting_app/README.md#aktualny-kontrakt-c0) is local native **Causal-C0** (`c0-native-2026-w32-e12-v1`) and the separate **Fixed50** logit hybrid (`Hybrid-Operational-Market`, `c0-native-2026-w32-e12-v1-a0.50-t1.00`): `sigmoid(0.50 * logit(p_C0) + 0.50 * logit(q_market))`, T=1.00, applied once. Annual A0 is a **research reference**, not the local operational sports model; [current research](../docs/RESEARCH.md) preserves later negative betting/adoption evidence.
- Native C0 already emits series probabilities and uses frozen released history through **2026-09-08**, not an automatically refreshed feed. Missing artifacts, roster IDs, supported BO or valid clocks cause explicit refusal; **the A0-offline rating-consensus fallback below is withdrawn**. EXP081/A1, default ratings and randomly initialized models are not replacements.
- A native sports prediction and a market-dependent hybrid have separate contracts. Missing valid market inputs do not produce a certified hybrid by relabeling sports-only output; previously saved hybrids must not be blended again.
- [Plan11 corrective scope](./11_project_integrity_and_readiness.md) owns serving provenance (S), access/operational isolation (O) and financial disclosure (F). Its owned local checks are separate from deployment. This old roadmap authorizes no collector, automatic acquisition, scheduler/notification, configured-user database operation, production cutover, wagering or restart of completed experiments/the stopped 10/10 loop.

## Preserved original roadmap (not current instructions)

**Priority:** P1 (High Operational Impact)  
**Target Files:**
- `betting_app/core/models/registry.py`
- `betting_app/services/upcoming_inference_service.py`
- `betting_app/scripts/run_upcoming_prediction_pipeline.py`
- `betting_app/api/routers/predictions.py`

---

## 1. Problem Statement
The live application database (`betting_app`) currently generates prospective predictions using the older, uncalibrated Siamese model (`EXP-081`). Our diagnostic benchmark proved that:
- Standalone EXP-081 underperforms Causal A0 by $+0.031$ Log Loss ($0.5856$ vs $0.5544$).
- Standalone EXP-081 underperforms even the baseline calibrated player Glicko-2 by $+0.010$ Log Loss.
- Meanwhile, the **Bayesian Shrunk Hybrid ($\alpha = 0.50$)** achieves Log Loss **$0.5608$**, outperforming both Causal A0 ($0.5702$) and Bookmaker Opening lines ($0.5721$).

---

## 2. Specification & Implementation

### A. Register Shrunk Hybrid in Model Registry (`registry.py`):
Update the active model spec to use the calibrated Shrunk Hybrid:
```python
ACTIVE_MODEL_NAME = "Hybrid-Bayesian-Shrunk-A0-Market"
ACTIVE_MODEL_VERSION = "hybrid-a0-mkt-v1-a0.50"
ACTIVE_MODEL_FAMILY = "bayesian_market_hybrid"
```

### B. Prediction Engine Logic (`upcoming_inference_service.py`):
1. When computing prospective probabilities for an upcoming fixture:
   - Compute base pre-match sports probability via Causal A0 (or symmetric rating consensus if A0 checkpoint is offline).
   - Fetch the latest available pre-match no-vig opening odds from `odds_snapshots`.
2. Evaluate Shrunk Probability in logit space:
   $$z_{\text{hybrid}} = 0.50 \cdot \text{logit}(p_{\text{sports}}) + 0.50 \cdot \text{logit}(p_{\text{market\_novig}})$$
   $$p_{\text{active}} = \sigma(z_{\text{hybrid}})$$
3. Fallback: If no bookmaker odds have been scraped yet ($> 48$h prior), use pure $p_{\text{sports}}$ until odds arrive.

### C. Automated CLV Tracking:
When match results are recorded:
- Fetch the final Closing line from `odds_snapshots` ($< 2$ hours prior).
- Record realized Closing Line Value:
  $$\text{CLV} = \frac{\text{odds}_{\text{entry}}}{\text{odds}_{\text{close}}} - 1.0$$
- Store CLV alongside prediction records in `canonical_predictions`.

---

## 3. Verification & Acceptance Criteria
1. Execute unit and integration tests:
   ```bash
   .venv/bin/python -m pytest -q betting_app/tests/test_model_registry.py betting_app/tests/test_predictions.py
   ```
2. Verification script:
   - Run `betting_app/scripts/run_upcoming_prediction_pipeline.py` on active canonical fixtures.
   - Verify that output predictions in `canonical_predictions` reflect the Shrunk Hybrid probabilities.
   - Verify that no NaN values, side reversals, or uncalibrated probabilities are inserted.
