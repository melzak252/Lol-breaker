# LoL-Breaker — League of Legends Quantitative Modeling & Betting Intelligence

Standalone application and modeling repository for League of Legends quantitative modeling and betting intelligence.

**LoL-Breaker** combines pre-match causal match-winner and prop-distribution models (`src/`, `conf/`) with a full-stack manual betting analytics & EV intelligence manager (`betting_app/`, `client/`). The application does **not** place bets automatically.

---

## 1. Repository Architecture (App + Models Together)

| Path | Responsibility |
|---|---|
| `src/` | Core modeling, rating engines (Glicko-2, Elo, FamilyMapBridge), tournament simulators (Exact DP + Swiss MC), Shin de-vigging, and Risk Guards A–D + Guard W (Wasserstein DRO). |
| `betting_app/` | FastAPI backend (`LoL-Breaker Betting API`), isolated Native C0 inference worker (`c0_worker.py`), SQLAlchemy/TimescaleDB models, bookmaker scrapers, scheduler, and EV/staking services. |
| `client/` | React 18 + TypeScript + Vite frontend (`LoL-Breaker — Betting Intelligence`). |
| `conf/base/` | Serving contract (`c0_serving.json`) and canonical benchmark manifest (`research_benchmark.json`). |
| `scripts/` | Benchmark runners (`run_model_benchmark.py`, `run_unified_betting_evaluation.py`), shadow forward ledger (`run_shadow_forward_ledger.py`), and prop preservation utilities. |
| `docs/`, `reports/` | Model catalog ([`docs/MODEL_CATALOG.md`](docs/MODEL_CATALOG.md)), research journal ([`docs/RESEARCH.md`](docs/RESEARCH.md)), architecture map ([`docs/PROJECT_ARCHITECTURE.md`](docs/PROJECT_ARCHITECTURE.md)), and econometric audit reports ([`reports/README.md`](reports/README.md)). |
| `data/` | Local datasets, model checkpoints, and reporting receipts (standalone local directory in `/home/melzak/dev/lol-breaker/data`). |

> **Standalone Environment**: `data/` and `.venv/` are standalone local directories in `/home/melzak/dev/lol-breaker` with no external repository dependencies or symlinks.

---

## 2. Active Production Model & Optimal Synergy + Wasserstein DRO Stack

| Layer | Configuration | Source |
|---|---|---|
| **Base Sports Model** | `Causal-C0` (`c0-native-2026-w32-e12-v1`, W32/E12 3-seed neural history attention + rating/MLP experts) | `conf/base/c0_serving.json`, `betting_app/core/models/registry.py` |
| **Market Hybrid** | `Hybrid-Operational-Market` (`c0-native-2026-w32-e12-v1-a0.50-t1.00`, 50/50 logit shrinkage with 2-way closed-form Shin de-vigging) | `betting_app/core/models/registry.py`, `betting_app/core/ev.py` |
| **Guard B (Logit Gate)** | $|\operatorname{logit}(p_{\text{C0}}) - \operatorname{logit}(q_{\text{shin}})| \le 0.85$ | `src/analysis/betting_guards_a_to_d.py`, `betting_app/services/bet_qualification_service.py` |
| **Thesis Model Veto** | $|p_{\text{C0}} - p_{\text{EXP-039}}| \le 0.10$ | `betting_app/core/ev.py`, `betting_app/services/upcoming_inference_service.py` |
| **Guard W (Wasserstein DRO)** | $\varepsilon = 0.025$ ($(\hat{p}_{\text{hybrid}} - 0.025)(1 - \tau)O - 1 > 0 \iff \text{EV}_{\text{net}} > \max(0.05, (1-\tau)O \cdot 0.025)$) | `src/analysis/betting_guards_a_to_d.py`, `betting_app/services/bet_qualification_service.py` |
| **Caps & Staking** | $\text{EV}_{\text{net}} \in [5\%, 25\%]$, $\text{Odds} \le 4.00$, whole-integer PLN step-compounded unit stake (`3%` of bankroll, capped at `150 PLN`) | `betting_app/core/staking.py` |

---

## 3. Quick Verification & Local Commands

```bash
# 1. Verify canonical research benchmark manifest & hashes:
.venv/bin/python scripts/run_model_benchmark.py --suite conf/base/research_benchmark.json --doctor

# 2. Run qualification, guards, and benchmark regression tests:
.venv/bin/python -m pytest -q betting_app/tests/test_bet_qualification.py betting_app/tests/test_betting_guards_a_to_d.py betting_app/tests/test_research_benchmark_suite.py

# 3. Frontend lint and production build:
cd client && npm run lint && npm run build
```
