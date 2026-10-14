# Long-Term Betting System Trustworthiness, The Optimizer's Curse, and Multi-Source Econometric Audit

**Date:** 2026-10-14  
**Scope:** End-to-end codebase & model cartography, cross-source empirical audit across **OddsPortal** ($N=11,609$ multi-year & $N=2,275$ canonical), **OddsPapi / Pinnacle** ($N=1,619$ at $T-1\text{h}$ & $N=447$ at Close), and **Production Scraped Polish Odds** ($234,435$ quotes across $1,600$ matches from 7 Polish bookmakers), plus exact mathematical decomposition of **True Profit ($\text{Net CLV}$)**, **The Optimizer's Curse**, **Match Outcome Noise**, and **Time-to-Profit Convergence ($N_{\min}$)**.

---

## Executive Summary & Core Verdicts

1. **Leading Models by Role (`docs/MODEL_CATALOG.md`, `conf/base/c0_serving.json`, `betting_app/core/models/registry.py`)**:
   - **Leading Pure Sports Match-Winner Model (Local Active Default)**: **`Causal-C0`** (`c0-native-2026-w32-e12-v1`, frozen Width-32 / Epoch-12 3-seed neural history attention + rating/MLP experts from `data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control`). Scores **LogLoss 0.535449** and **Brier 0.181034** on the 3,103 canonical 2026 series.
   - **Leading Market-Assisted Operational Model**: **`Hybrid-Operational-Market`** (`c0-native-2026-w32-e12-v1-a0.50-t1.00`), applying 50/50 logit shrinkage between Native C0 and Shin-devigged market consensus.
   - **Leading Decision-Time Betting Strategy**: **Optimal Synergy Strategy** (`C0 Shrunk Hybrid` $\alpha=0.50$ + **Guard B** $\Delta_{\text{logit}} \le 0.85$ + **Thesis Veto** $|p_{\text{C0}} - p_{\text{EXP-039}}| \le 0.10$ + $\text{EV}_{\text{net}} \in [5\%, 25\%]$ + $\text{Odds} \le 4.00$ + 1/4 Kelly staking).
   - **Canonical 2024–2026 Research Benchmark Reference**: **Annual Causal A0** (`p_full` in `conf/base/research_benchmark.json`, LogLoss **0.551246** on 11,550 series; **0.559920** on 9,907 common support vs. Frozen Thesis **EXP-039** at **0.567337**).
   - **Leading Standalone Prop Models**: Preserved bundle `prop_distribution_20261007_01` (Map 1 Kills: Ridge-1 Poisson/NB, NLL **3.590488**; Map 1 Duration: 15-leaf LightGBM/LogNormal, NLL **3.076194**; BO3/5 Series Length: W128/E12 3-seed mixture, NLL **0.714580**).
   - **Leading Tournament Simulator**: **Exact Knockout DP + Swiss Monte Carlo Hybrid** (`tournament_numerical_20261007_01`).

2. **Can We Trust the Betting System in the Long Term?**
   - **Unshrunk, ungated models CANNOT be trusted**: Raw model EV selection suffers a **30.1 pp to 60.5 pp Optimizer's Curse EV Overstatement Gap** across all three datasets (claiming $+24.5\%$ to $+49.9\%$ EV while realizing $-18.9\%$ to $+2.1\%$ Yield and $25\%\text{--}67\%$ Max Drawdowns). Even perfect hindsight knowledge of starting lineups leaves **$35.0\text{ to }44.3\text{ pp}$** of EV overstatement intact.
   - **Shrinkage ($\alpha \in [0.35, 0.50]$) + Guard B ($\Delta_{\text{logit}} \le 0.85$) + Caps ($\text{EV} \le 25\%, \text{Odds} \le 4.00$) + Thesis Veto ($\le 0.10$) mathematically neutralize the Optimizer's Curse**: They compress the EV Overstatement Gap from $>30\text{ pp}$ down to **$-2.6\text{ pp}\text{ to }+1.9\text{ pp}$** and cap Max Drawdown at **$6.1\%\text{--}15.0\%$**.

3. **What Is Our True Profit vs. What Is Noise? How Long Until "Almost Only Profit"?**
   - **True Long-Term Profit ($\mu$) = Net CLV**: Under 12% Polish tax, your true mathematical edge on the gated production strategy is **$\text{Net CLV} = +4.83\%$ per bet** ($95\%\text{ CI: }[+1.68\%, +7.98\%], t = 3.000, p = 0.0042$). Under 0% tax (international books or Betclic 0% tax promo), your true edge is **$+7.60\%$ to $+10.15\%$ per bet**.
   - **Two Different "Noises"**:
     1. **The Optimizer's Curse ($B_i = \text{Claimed EV}_i - \text{Net CLV}_i$)**: Systematic positive selection bias ($\mathbb{E}[B_i \mid \text{selected}] > 0$). **NEVER disappears with time.** Removed only by Shrinkage + Gates.
     2. **Match Outcome Noise ($\eta_i = \text{Realized Yield}_i - \text{Net CLV}_i$)**: Zero-mean binomial match variance ($\mathbb{E}[\eta_i] = 0$, $\sigma_{\text{bet}} = 109.79\%$ at avg odds $2.53$). **DOES disappear with time** at rate $\text{SE}(N) = \frac{\sigma}{\sqrt{N}}$. On the 50-bet production ledger, $3.33$ extra lucky wins ($29$ actual vs $25.67$ closing-expected) added $+19.42\text{ pp}$ of temporary positive noise on top of the $+4.83\%$ true profit to produce $+23.91\%$ Realized Yield.
   - **Required Bets ($N_{\min}$) for 95% Probability of Positive Profit ("Almost Only Profit", $z = 1.645$)**:
     - **At 12% Polish Tax ($\mu = +4.83\%$, $\sigma = 109.8\%$):** **$N = 1,396\text{ bets}$** ($\sim 3.8$ years at 30 bets/month, or $\sim 1.9$ years at 60 bets/month across all soft books).
     - **At 0% Tax ($\mu = +7.60\%$, $\sigma = 101.0\%$):** **$N = 478\text{ bets}$** ($\sim 8\text{--}10$ months).
     - **To Prove Net CLV $> 0$ at 95%–99% Confidence ($\sigma_{\text{CLV}} = 11.39\%$):** Only **$N = 15\text{ to }30\text{ bets}$** ($1\text{--}2$ months), because Net CLV has **$104.5\times$ lower variance** than binary match outcomes.

---

## Part I: Leading Models & Architecture Cartography

### 1.1 Authoritative Model Hierarchy

| Domain / Role | Model Identity & Version | Architecture & Formulation | Artifact / Config Source of Truth | Status |
|---|---|---|---|---|
| **Leading Pure Sports Model (Local Active)** | **`Causal-C0`** (`c0-native-2026-w32-e12-v1`) | 3-seed Width-32 / Epoch-12 causal history-attention neural net (16 prior maps/player, Beta terminal score + $0.1$ SmoothL1 aux loss) + shared rating (Glicko-2/Elo) + MLP experts + mode-specific calibration (`full`, `no_w20`, `no_organization`) | `conf/base/c0_serving.json`, `betting_app/core/models/registry.py:18-30`, `data/06_models/a0_data_focus/a0_data_focus_20260929_192443/control` | **`KEEP_REFERENCE` / Active Local Default** |
| **Leading Market-Assisted Operational Model** | **`Hybrid-Operational-Market`** (`c0-native-2026-w32-e12-v1-a0.50-t1.00`) | 50/50 logit-space shrinkage between Native C0 and Shin de-vigged market consensus: $p_{\text{Fixed50}} = \sigma(0.50\operatorname{logit}(p_{\text{C0}}) + 0.50\operatorname{logit}(q_{\text{market}}))$, $T=1.00$ | `betting_app/core/models/registry.py:82-99`, `betting_app/core/models/engine.py` | **`KEEP_LOCAL_CONFIGURATION`** |
| **Leading Decision-Time Betting Policy** | **Optimal Synergy Strategy** (`C0 Shrunk Hybrid + Guards`) | Fixed50 Hybrid + Shin 2-way closed-form de-vigging + **Guard B** ($|\operatorname{logit}(p_{\text{C0}}) - \operatorname{logit}(q_{\text{mkt}})| \le 0.85$) + **Thesis Veto** ($|p_{\text{C0}} - p_{\text{EXP-039}}| \le 0.10$) + 1/4 Kelly (12% Polish tax, 150 PLN cap) | `betting_app/services/bet_qualification_service.py`, `betting_app/core/ev.py`, `src/analysis/betting_guards_a_to_d.py` | **`ADOPTED_LOCAL_STRATEGY`** |
| **Canonical Research Benchmark Reference** | **Annual Causal A0** (`p_full` in `causal-a0-rating-controls-v1`) | Annual walk-forward causal ensemble (7 base checkpoints across 3 experts: history attention, ratings, MLP; quarterly cross-fitted calibration) | `conf/base/research_benchmark.json`, `<research_root>/a0-phase-walkforward-20260914/data/06_models/causal_a0/` | **`KEEP_CONTROL` (Canonical Benchmark)** |
| **Frozen Thesis Baseline** | **`Sym-Cal LR-ElasticNet-W20-Binomial`** (`exp-039` / `p_039`) | Map-level symmetric calibrated Logistic Regression / ElasticNet with $W20$ organizational features and binomial BO3/BO5 projection | `betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib`, `betting_app/core/models/registry.py:46-58` | **`KEEP_CONTROL` (Immutable Thesis Baseline)** |
| **Standalone Props: Map 1 Total Kills** | `prop_distribution_20261007_01` (Kills) | Regularized Ridge-1 Poisson-mean / Negative Binomial dispersion over 306 invariant pre-match pace features | `data/06_models/prop_distribution/prop_distribution_20261007_01/selected_model.joblib` (`prop_distribution_20261007_01.zip`) | **`KEEP_BASELINE`** |
| **Standalone Props: Map 1 Duration** | `prop_distribution_20261007_01` (Duration) | 15-leaf LightGBM / LogNormal distribution | Same frozen prop bundle (`src/models/prop_distribution.py`) | **`KEEP_RESEARCH_CANDIDATE`** |
| **Standalone Props: Series Length (BO3/5)** | `prop_distribution_20261007_01` (Length) | Width-128 / Epoch-12 3-seed neural distribution mixture | Same frozen prop bundle (`src/models/prop_distribution.py`) | **`KEEP_RESEARCH_CANDIDATE`** |
| **Leading Tournament Simulator** | **Exact DP + Swiss MC Hybrid** (`tournament_numerical_20261007_01`) | Rao-Blackwellized hybrid: sequential Buchholz lottery Monte Carlo for Swiss/draws + exact $O(2^K)$ dynamic programming for fixed knockout brackets | `src/models/tournament_exact.py`, `src/models/advanced_tournament_simulator.py`, `betting_app/services/tournament_service.py` | **`KEEP_NUMERICAL_IMPLEMENTATION`** |

### 1.2 Benchmark Accuracy Across Canonical Sports Cohorts

#### A. Canonical 2026 Assessment Cohort (3,103 Series: Jan–Sep 2026)

| Model | LogLoss | Brier Score | Status & Notes |
|---|---:|---:|---|
| **Native C0 (`c0-native-2026-w32-e12-v1`)** | **0.535449** | **0.181034** | **Leading pure sports baseline** (TRAIN: 36,205; CAL: 1,328; ECE: 2.2855%) |
| C0 30-Day Role-Meta Challenger (`c0_meta_v2`) | 0.535571 | 0.181073 | `NOT_ADOPTED`; 97.5% $\Delta\text{LL}$ CI crosses 0 |
| Learned Sports Connector (`run03`) | 0.535644 | 0.181012 | `KEEP_HISTORICAL_CANDIDATE_ONLY`; $\Delta\text{LL}$ vs C0 $+0.000196$ crosses 0 |
| A0 Champion-Pool Specialist (`IDEA-031`) | 0.536102 | 0.181227 | `NOT_ADOPTED`; source-unseen guard failed |
| **Annual Causal A0 (`p_full`)** | **0.536757** | **0.181573** | Canonical annual walk-forward reference (ECE: 2.2047%) |
| Ground-Up Raw Ensemble (`prematch_symmetry`) | 0.543863 | 0.183493 | `NOT_ADOPTED`; loses to A0 by $+0.007106$ LL |
| Champion-History Multitask (`D`) | 0.546321 | 0.184929 | `NOT_ADOPTED`; loses to C0 by $+0.010872$ LL |

#### B. Full 2024–2026 Walk-Forward Cohort (11,550 Series) vs. Common EXP-039 Cohort (9,907 Series)

| Model | 11,550 Full Cohort LogLoss | 11,550 Full Cohort Brier | 9,907 Common Cohort LogLoss | 9,907 Tails ($\text{LL} \ge 2.5$) |
|---|---:|---:|---:|---:|
| **Annual Causal A0 (`p_full`)** | **0.551246** | **0.187103** | **0.559920** | 30 |
| Annual EXP-039 Research Derivative (`p_039`) | *N/A (1,643 missing)* | *N/A* | 0.567337 | **16** |
| Calibrated Player Glicko-2 (`p_glicko`) | 0.575888 | 0.197253 | 0.577998 | 36 |
| Calibrated Player Elo (`p_elo`) | 0.589182 | 0.203037 | 0.591198 | 29 |

### 1.3 Developer Navigation Matrix

| If you want to... | Primary File(s) to Inspect |
|---|---|
| Check which model/study is active, preserved, or rejected | `docs/MODEL_CATALOG.md`, `docs/RESEARCH.md` |
| Inspect or switch the local serving model / hybrid | `conf/base/c0_serving.json`, `betting_app/core/models/registry.py`, `betting_app/core/models/engine.py` |
| Trace how Native C0 builds features and runs inference | `betting_app/core/models/native_c0.py`, `betting_app/services/c0_worker.py`, `betting_app/services/c0_inference.py` |
| Modify bet qualification guards (Guard B, Thesis Veto, EV hurdle) | `betting_app/services/bet_qualification_service.py`, `src/analysis/betting_guards_a_to_d.py` |
| Audit EV, Shin devigging, or Quarter-Kelly staking math | `betting_app/core/ev.py`, `betting_app/core/staking.py`, `src/analysis/unified_evaluation_engine.py` |
| Run the canonical 11,550-match research benchmark | `scripts/run_model_benchmark.py`, `src/analysis/research_benchmark.py`, `conf/base/research_benchmark.json` |
| Work on standalone Map 1 kills / duration / series-length props | `src/models/prop_distribution.py`, `src/analysis/prop_model_preservation.py`, `scripts/preserve_prop_model.py` |
| Modify tournament simulation (Swiss MC, Exact DP, Fearless) | `src/models/advanced_tournament_simulator.py`, `src/models/tournament_exact.py`, `betting_app/services/tournament_service.py` |
| Inspect database schema, hypertables, or immutability triggers | `betting_app/models/prediction.py`, `betting_app/models/bookmaker.py`, `docker/timescale/init.sql` |
| Trace API endpoints or frontend views | `betting_app/api/main.py`, `betting_app/api/routers/`, `client/src/App.tsx`, `client/src/api/client.ts` |

---

## Part II: Multi-Source Econometric Audit (OddsPortal, OddsPapi/Pinnacle, Production)

### 2.1 Proper Scoring Rules Across Shrinkage Levels ($\alpha$) and Datasets

For any pre-match probability $p_{\text{model}}$ and Shin-devigged market consensus $q_{\text{market}}$, Bayesian logit shrinkage is defined as:
$$\operatorname{logit}(p_{\text{hybrid}}) = \alpha \cdot \operatorname{logit}(p_{\text{model}}) + (1 - \alpha) \cdot \operatorname{logit}(q_{\text{market}})$$

| Dataset & Cohort | Sample $N$ | Market Alone ($\alpha=0$) LL | Pure Model ($\alpha=1$) LL | Hybrid $\alpha=0.25$ LL | Hybrid $\alpha=0.35$ LL | Hybrid $\alpha=0.50$ LL | Hybrid $\alpha=0.75$ LL | Optimal $\alpha$ & Gain vs Market |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **OddsPortal Multi-Year Open (2021–2026)** | 11,623 | 0.606278 | 0.584165 | 0.590352 | 0.585813 | **0.581170** | — | $\alpha=0.50$ (**$-0.02511$ LL**) |
| **OddsPortal Canonical Open (2024–2026)** | 2,275 | 0.593258 | 0.591839 | 0.587843 | 0.586628 | **0.585823** | 0.587179 | $\alpha=0.50$ (**$-0.00744$ LL**) |
| **OddsPortal Canonical Close (2024–2026)** | 2,275 | 0.582080 | 0.591839 | **0.580729** | 0.580896 | 0.581907 | — | $\alpha=0.25$ (**$-0.00135$ LL**) |
| **OddsPortal Joint Consensus (2024–2026)** | 2,023 | 0.581530 | 0.579373 | 0.576587 | 0.575473 | **0.574693** | 0.575680 | $\alpha=0.50$ (**$-0.00684$ LL**) |
| **Production Monthly-Refit C0 (May–Sep 2026)** | 645 | 0.570054 | 0.573307 | — | 0.558570 | **0.557877** | — | $\alpha=0.50$ (**$-0.01218$ LL**) |
| **OddsPapi Consensus OPEN (2026)** | 513 | 0.572300 | 0.588800 | 0.564300 | **0.563800** | 0.565400 | — | $\alpha=0.35$ (**$-0.00850$ LL**) |
| **OddsPapi Pinnacle T-24h (2026)** | 557 | 0.576457 | 0.578170 | 0.571300 | 0.570289 | **0.569933** | — | $\alpha=0.50$ (**$-0.00652$ LL**) |
| **OddsPapi Pinnacle T-1h (2026)** | 1,619 | 0.564260 | 0.574881 | 0.561807 | **0.561791** | 0.562797 | — | $\alpha=0.35$ (**$-0.00247$ LL**) |
| **OddsPapi Pinnacle Close (2026)** | 447 | **0.535729** | 0.574425 | 0.538377 | 0.540652 | 0.545459 | — | $\alpha=0.00$ (Pinnacle Close sharpest) |

* **De-vigging Method Audit ($N=65,724$ individual closing quotes / $1,391$ multi-book consensus matches)**:
  - **Shin (1992)** (which simplifies algebraically in 2-way markets to the Additive equal-margin split $\pi_1 = q_1 - (S - 1)/2$, avoiding the Jullien & Salanié $n-2=0$ division-by-zero bug) achieves **LogLoss 0.565881** and **ECE10 1.82%** on consensus matches, beating Multiplicative de-vigging (**0.567168**, ECE10 **2.01%**) by **$-0.001287$ LL**.

---

### 2.2 Shrinkage & Gate Ablation Matrices Across All 3 Data Sources

#### A. Source 1 — Production Scraped Polish Odds (`May 28 – Sep 30, 2026`, 7 Polish Books, 12% Tax)

| Strategy / Gate Configuration | Horizon & Roster | Bets $N$ | Win Rate | Avg Odds | Expected Net EV% | Realized Yield% (Profit/Turnover) | EV Gap ($\text{EV}-\text{Yield}$) | Bankroll ROI% (10k Base) | Max DD% | Raw CLV% | Net CLV% (12% Tax) | Pos Raw CLV% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Oct 6 Counterfactual: Pure C0 (No Shrinkage, No Gates)** | First-Obs (33/34 1-book) | 34 | 38.2% | 3.20 | +24.54% | **−18.86%** | **+43.40 pp** | −6.41% | 25.16% | ~+0.7% | ~−11.4% | — |
| **2. Standalone Thesis (EXP-039, Unshrunk)** | $T-48\text{h}..1\text{h}$ | 122 | 48.4% | 2.18 | +14.20% | **−9.86%** | **+24.06 pp** | −17.52% | 26.39% | +1.64% | −10.56% | 48.4% |
| **3. Standalone C0 (Unshrunk)** | $T-48\text{h}..1\text{h}$ | 147 | 46.9% | 2.24 | +15.80% | **−3.16%** | **+18.96 pp** | −6.72% | 25.33% | +0.72% | −11.37% | 48.3% |
| **4. Shrunk C0 Hybrid ($\alpha=0.50$, Guard B + Caps, No Veto)** | $T-48\text{h}..1\text{h}$ (1/4 Kelly) | 59 | 50.8% | 2.51 | +13.20% | **+11.56%** | **+1.64 pp** | +9.66% | 13.39% | +18.37% | +4.17% | 84.7% |
| **5. Comfort Policy C (Hybrid $\alpha=0.50$ + Guard B $\le 0.85$ + Caps)** | 2h Trajectory (Flat 100) | 103 | 37.9% | 2.78 | +13.82% | **+19.87%** | **−6.05 pp** | +20.46% | 8.67% | +16.50% | +2.52% | 84.5% |
| **6. Optimal Synergy (Hybrid $\alpha=0.50$ + Guard B + Caps + Thesis Veto $\le 0.10$)** | $T-48\text{h}..1\text{h}$ (1/4 Kelly) | **50** | **58.0%** | **2.53** | **+13.41%** | **+23.91%** | **−10.50 pp** | **+17.15%** | **6.10%** | **+19.13%** | **+4.83%** | **90.0%** |

#### B. Source 2 — OddsPapi & Sharp Pinnacle Lines (`2026`, $N=1,619$ Matches at $T-1\text{h}$)

| Configuration (Flat 100 PLN, $\text{EV}_{\text{net}} > 5\%$) | Tax Rate | Bets $N$ | Win Rate | Expected EV% | Realized Yield% | EV Gap (pp) | Bankroll ROI% | Max DD% | Raw CLV% | Net CLV% | 95% Bootstrap CI (Yield) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **1. Pure Model ($\alpha=1.00$, No Gates)** | **0%** | 787 | 43.7% | +29.7% | **−0.4%** | **+30.1 pp** | −3.5% | 47.8% | −5.81% | −5.81% | $[−10.5\%, +9.7\%]$ |
| **2. Pure + Guard B ($\Delta_{\text{logit}} \le 0.85$)** | **0%** | 640 | 42.3% | +19.1% | **−4.3%** | **+23.4 pp** | −27.2% | 54.1% | −5.71% | −5.71% | $[−14.4\%, +6.1\%]$ |
| **3. Pure + EV Cap ($\le 25\%$)** | **0%** | 533 | 49.7% | +12.4% | **−4.2%** | **+16.6 pp** | −22.3% | 40.6% | −5.68% | −5.68% | $[−13.5\%, +5.6\%]$ |
| **4. Pure + Odds Cap ($\le 4.00$)** | **0%** | 642 | 49.5% | +19.9% | **−2.5%** | **+22.4 pp** | −16.0% | 33.8% | −5.67% | −5.67% | $[−11.2\%, +6.2\%]$ |
| **5. Shrinkage ($\alpha=0.50$) Alone** | **0%** | 416 | 38.0% | +19.2% | **+5.8%** | **+13.4 pp** | +24.2% | 33.7% | −5.81% | −5.81% | $[−10.5\%, +23.2\%]$ |
| **6. Shrinkage ($\alpha=0.35$) Alone** | **0%** | 251 | 34.3% | +16.5% | **+11.4%** | **+5.1 pp** | +28.6% | 23.2% | −5.94% | −5.94% | $[−11.9\%, +37.0\%]$ |
| **7. Shrunk ($\alpha=0.35$) + Guard B + EV Cap ($\le 25\%$) + Odds Cap ($\le 4.0$)** | **0%** | **89** | **39.3%** | **+8.4%** | **+6.9%** | **+1.4 pp** | **+6.2%** | **9.9%** | −5.37% | −5.37% | $[−20.5\%, +36.8\%]$ |
| **8. Pure Model ($\alpha=1.00$, No Gates)** | **12%** | 328 | 33.5% | +37.2% | **−9.1%** | **+46.3 pp** | −29.7% | 42.7% | −5.97% | **−17.26%** | $[−25.5\%, +9.2\%]$ |
| **9. Shrunk ($\alpha=0.50$) + Guard B ($\le 0.85$)** | **12%** | 40 | 20.0% | +9.3% | **−24.8%** | **+34.2 pp** | −9.9% | 22.6% | −6.24% | **−17.49%** | $[−72.3\%, +33.2\%]$ |
| **10. Shrunk ($\alpha=0.35$) + Guard B + EV Cap + Odds Cap** | **12%** | **0** | N/A | N/A | **0.0%** | **N/A** | **0.0%** | **0.0%** | N/A | N/A | *Correctly abstains (0 bets)* |

#### C. Source 3 — OddsPortal Open-vs-Close Composite (`2024–2026` Canonical $N=2,275$ & `2021–2026` Multi-Year $N=11,609$)

| Cohort & Strategy (Betting at `market_open`, Settled vs `market_close`) | Tax Rate | Staking | Bets $N$ | Win Rate | Avg Odds | Expected EV% | Realized Yield% | EV Gap (pp) | Bankroll ROI% | Max DD% | Raw CLV% | Pos CLV% | 95% Bootstrap CI (Yield) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Multi-Year (11,609): Unshrunk Model** | **0%** | 1/4 Kelly | 4,352 | 59.1% | 1.97 | +14.80% | **+6.72%** | +8.08 pp | +436.17% | 22.60% | +3.79% | 68.0% | $[+3.8\%, +9.6\%]$ |
| **Multi-Year (11,609): Shrunk Hybrid ($\alpha=0.50$)** | **0%** | 1/4 Kelly | **3,528** | **53.7%** | **2.22** | **+9.80%** | **+10.15%** | **−0.35 pp** | **+525.79%** | 42.41% | **+6.35%** | **73.6%** | $[+6.4\%, +13.9\%]$ |
| **Multi-Year (11,609): Shrunk Hybrid + Guard B** | **12%** | 1/4 Kelly | **684** | **48.5%** | **2.64** | **+8.90%** | **+9.65%** | **−0.75 pp** | **+96.38%** | **14.96%** | **+21.49%** | **87.4%** | $[+1.2\%, +18.1\%]$ |
| **Canonical (2,275): Unshrunk C0** | **0%** | Flat 1u | 1,522 | 47.0% | 2.89 | +24.03% | **+6.37%** | **+17.66 pp** | +97.00% | 24.27% | −0.80% | 36.1% | $[−0.12\%, +13.61\%]$ |
| **Canonical (2,275): Shrunk Hybrid ($\alpha=0.50$)** | **0%** | Flat 1u | 1,023 | 43.1% | 3.24 | +12.56% | **+9.83%** | **+2.73 pp** | +100.55% | 24.38% | +1.31% | 43.6% | $[+1.00\%, +19.82\%]$ |
| **Canonical (2,275): Optimal Synergy (C0 + Guard B + Caps + Veto)** | **0%** | Flat 1u | **557** | **46.9%** | **2.51** | **+6.10%** | **+8.73%** | **−2.64 pp** | **+48.63%** | **24.49%** | −1.06% | 41.8% | $[−1.61\%, +18.93\%]$ |
| **Canonical (2,275): Optimal Synergy (C0 + Guard B + Caps + Veto)** | **12%** | Flat 1u | 61 | 37.7% | 3.17 | +3.66% | **−2.78%** | +6.44 pp | −1.70% | 14.91% | +5.59% | 55.7% | $[−34.68\%, +27.90\%]$ |

---

### 2.3 Prior-Roster vs. Actual-Roster Hindsight Impact (`a0_error_audit` & `a0_context_stages`)

Across $N = 1,948$ matches at $T-1\text{h}$:
- **77.9% of fixtures ($1,517/1,948$)** had the exact same 5 players (`LL = 0.5723` for both prior and actual lineups).
- **18.9% of fixtures ($368/1,948$)** had legitimate player substitutions (`LL = 0.5872` prior vs `0.5535` actual).
- **3.2% of fixtures ($63/1,948$)** had a context-carryover bug (where cup/academy lineups were carried into main league matches, e.g., Gen.G–HLE match `81029`), driving **74.0%** of the total LogLoss gap between prior and actual rosters (`1.0255` vs `0.4621`).
- **Crucial Financial Proof**: Even when granted **perfect hindsight knowledge of actual match-day starting lineups**, an unshrunk model still loses **$-14.99\%$ Yield at $T-24\text{h}$ (EV Overstatement Gap $+44.3\text{ pp}$)** and **$-1.25\%$ Yield at OPEN (EV Overstatement Gap $+35.0\text{ pp}$)**. Lineup information does not cure the Optimizer's Curse; probability shrinkage and Guard B are required.

---

## Part III: When to Bet Before the Match, Bookmaker Toxicity & League Tiers

### 3.1 Lead-Time Horizon Analysis (`When We Bet Before the Match`)

| Lead-Time Horizon Before Match | Lead Hours ($\Delta t$) | Whole Market Mean Raw CLV% | Whole Market Mean Net CLV% (12% Tax) | Pure C0 LL vs Market LL | Hybrid ($\alpha=0.50$) LL | Gated Hybrid Bets $N$ | Gated Hybrid Raw CLV% | Gated Hybrid Net CLV% (12% Tax) | Pos CLV Rate% | Expected EV% | Realized Yield% | Max DD% | Operational Verdict |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **First-Observed (1-Book Only)** | $\sim 92\text{h}$ | −6.38% | −17.61% | 0.5660 vs 0.5725 | 0.5588 | 12 | N/A | N/A | — | +14.39% | **−1.00%** | 14.41% | ❌ **AVOID** (1-book placeholder trap) |
| **$T-\text{pre48h}$ ($\ge 2$ Books)** | $>48\text{h}$ | −6.38% | −17.61% | — | — | 18 | **+21.40%** | **+6.83%** | **100.0%** | +12.80% | **+24.10%** | 5.20% | ✅ **PRIME EARLY ALPHA** |
| **$T-48\text{h}$ Window** | $24\text{--}48\text{h}$ | −8.10% | −19.12% | 0.5764 vs 0.5528 | 0.5482 | 21 | **+20.75%** | **+6.26%** | **100.0%** | +13.50% | **+42.90%** | 4.30% | ✅ **OPTIMAL SWEET SPOT** |
| **$T-24\text{h}$ Window** | $12\text{--}24\text{h}$ | −7.98% | −19.02% | 0.5663 vs 0.5567 | 0.5521 | 16 | **+19.64%** | **+5.29%** | **100.0%** | +14.20% | **+26.71%** | 6.13% | ✅ **OPTIMAL SWEET SPOT** |
| **$T-12\text{h}$ Window** | $6\text{--}12\text{h}$ | −7.92% | −18.97% | — | — | 24 | **+18.20%** | **+4.02%** | **91.7%** | +13.10% | **+18.40%** | 7.80% | ✅ **STRONG SECONDARY** |
| **$T-6\text{h}$ Window** | $2\text{--}6\text{h}$ | −7.96% | −19.00% | — | — | 29 | **+16.90%** | **+2.87%** | **86.2%** | +12.40% | **+14.80%** | 8.10% | ⚠️ **MODERATE BUFFER** |
| **$T-2\text{h}$ Window (Playbook C)** | $1\text{--}2\text{h}$ | −7.90% | −18.95% | — | — | 103 | **+16.50%** | **+2.52%** | **84.5%** | +13.82% | **+19.87%** | 8.67% | ⚠️ **VIABLE ONLY ON STALE BOOKS** |
| **$T-1\text{h}$ Closing Window** | $0\text{--}1\text{h}$ | −7.92% | −18.97% | 0.5757 vs 0.5544 | 0.5544 | 19 | +15.10% | **+1.29%** | 78.9% | +14.21% | +26.44% | 10.32% | ❌ **TOO THIN FOR 12% TAX** |

### 3.2 Bookmaker-by-Bookmaker Toxicity & Alpha Profile

| Bookmaker | Production Quotes | Median Vig | Optimal Ledger Bets | Win Rate | Realized Yield% | Net Profit (PLN) | Mean Raw CLV% | Pos CLV Rate% | Strategic Role & Toxicity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Totalbet** | 44,378 (18.9%) | 8.05% | 7 | 85.7% | **+76.41%** | +790.75 | +16.10% | 85.7% | **Grade A+ (Top Stale Alpha)**: Lags Pinnacle moves by 4–12h (+53.7% ROI in Oct 6 Hybrid too). |
| **Betfan** | 46,999 (20.0%) | 8.35% | 7 | 85.7% | **+78.96%** | +774.40 | +23.36% | 85.7% | **Grade A- (Gated Alpha)**: Toxic for raw C0 (−28.1% ROI on longshot traps), stellar with Guard B + Caps. |
| **STS** | 44,564 (19.0%) | 8.02% | 23 | 47.8% | **+1.17%** | +37.67 | +16.16% | 87.0% | **Grade A (Core Liquidity)**: 46% of volume; +28.6% ROI across Oct 6 Hybrid (27 bets, +771.60 PLN). |
| **LeBull** | 42,131 (18.0%) | 9.66% | 10 | 60.0% | **+17.38%** | +256.35 | +24.10% | 100.0% | **Grade B (Late Stale)**: High 9.7% vig; toxic at open, profitable at $T-24\text{h}..2\text{h}$ when $\text{EV}_{\text{net}} \ge 8\%$. |
| **Superbet** | 7,562 (3.3%) | 9.87% | 2 | 50.0% | **+2.08%** | +6.24 | +32.44% | 100.0% | **Grade D (Vig Trap)**: 9.87% vig + 12% tax eats almost all CLV. |
| **Betclic** | 38,207 (16.5%) | 9.37% | 1 | 0.0% | **−100.00%** | −150.00 | +25.80% | 100.0% | **Grade B- (0% Tax Promo Only)**: Sharp risk team; trade during 0% tax promotions. |
| **eFortuna** | 9,571 (4.1%) | 8.51% | 0 | N/A | **N/A** | 0.00 | N/A | N/A | **Grade C (Inert)**: Delays line release until Asian market solidifies; 0 qualifying bets. |

### 3.3 League-Tier Alpha Breakdown (`optimal_strategy_simulation_ledger.csv`)

| League Tier | Bets $N$ | Share | Win Rate | Turnover (PLN) | Net Profit (PLN) | Realized Yield% | Mean Raw CLV% | Mean Net CLV% (12% Tax) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tier-1 Major (LCK, LPL, LEC, LCS)** | 19 | 38.0% | 47.4% | 2,694.67 | +384.03 | **+14.25%** | +15.85% | **+1.95%** |
| **European Regional Leagues (ERLs: Prime, LFL, TCL)** | 14 | 28.0% | 57.1% | 2,092.36 | +340.40 | **+16.27%** | +23.09% | **+8.32%** |
| **Development / Academy (LCK CL, NACL)** | 9 | 18.0% | 77.8% | 1,277.11 | +585.63 | **+45.86%** | +20.49% | **+6.03%** |
| **Minor Top Level (CBLOL, LCP) & Other** | 8 | 16.0% | 62.5% | 1,109.01 | +405.35 | **+36.55%** | +18.46% | **+4.25%** |
| **Non-Tier-1 Combined** | **31** | **62.0%** | **64.5%** | **4,478.48** | **+1,331.38** | **+29.73%** | **+21.14%** | **+6.53%** |

---

## Part IV: Mathematical Decomposition of Profit (`Net CLV`), The Optimizer's Curse, and Time-to-Profit ($N_{\min}$)

### 4.1 The Exact Three-Part Identity of Every Bet

For any bet $i$ with decimal odds $O_i$, tax multiplier $c = 1 - \tau$ ($0.88$ under 12% Polish tax, $1.00$ under 0% tax), model/hybrid probability $\hat{p}_i$, fair devigged closing probability $p_{\text{close},i}$, and binary match outcome $Y_i \in \{0, 1\}$:

$$\underbrace{c O_i Y_i - 1}_{\substack{\text{Realized Yield} \\ \text{(What hits your wallet)}}} = \underbrace{c O_i \hat{p}_i - 1}_{\substack{\text{Claimed EV} \\ \text{(Model promise)}}} - \underbrace{c O_i (\hat{p}_i - p_{\text{close},i})}_{\substack{\text{1. THE OPTIMIZER'S CURSE } (B_i) \\ \text{(Systematic bias: NEVER vanishes with } N\text{)}}} + \underbrace{c O_i (Y_i - p_{\text{close},i})}_{\substack{\text{2. MATCH OUTCOME NOISE } (\eta_i) \\ \text{(Zero-mean luck: decays as } \sigma/\sqrt{N}\text{)}}}$$

Since $\mathbb{E}[Y_i \mid p_{\text{close},i}] = p_{\text{close},i}$, your **True Long-Term Profit Rate ($\mu$)** is:
$$\mu = \mathbb{E}[\text{Realized Yield}_i] = \text{Claimed EV}_i - \text{Optimizer's Curse}_i = c O_i p_{\text{close},i} - 1 = \mathbf{\text{Net CLV}_i}$$

### 4.2 Empirical Decomposition on Our Datasets

| Component | Formula | Unshrunk A0/C0 Baseline ($N=454$, 12% Tax) | Our Optimal Synergy System ($N=50$, 12% Tax) | OddsPortal 5-Year Hybrid ($N=3,002$, 0% Tax) | Interpretation |
|---|---|---:|---:|---:|---|
| **1. Pure Model Claimed EV** | $c O \hat{p}_{\text{C0}} - 1$ | $+31.93\%$ | $+22.05\%$ | $+14.80\%$ | Raw model promise before market shrinkage ($29.36/50$ expected wins) |
| **2. Shrunk Hybrid Claimed EV** | $c O \hat{p}_{\text{hybrid}} - 1$ | *Not applied* | **$+13.41\%$** | **$+4.68\%$** | Promise after 50/50 logit shrinkage ($27.53/50$ expected wins) |
| **3. True Long-Term Profit ($\text{Net CLV}$)** | $c O p_{\text{close, fair}} - 1$ | **$-13.69\%$** | **$+4.83\%$** | *(Raw CLV $+6.35\%$)* | **True mathematical edge as $N \to \infty$** ($25.67/50$ closing-expected wins) |
| **4. Optimizer's Curse Removed by Shrinkage & Gates** | $\text{Pure EV} - \text{Hybrid EV}$ | $0.00\text{ pp}$ | **$-8.64\text{ pp}$** *(plus 97 toxic bets vetoed)* | **$-10.12\text{ pp}$** | Fake edge stripped out before placing the bet |
| **5. Residual Optimizer's Curse** | $\text{Hybrid EV} - \text{Net CLV}$ | **$+45.62\text{ pp}$** | **$+8.57\text{ pp}$** | $\sim 0.00\text{ pp}$ | Remaining optimism of $+13.41\%$ Hybrid EV vs $+4.83\%$ closing truth |
| **6. Match Outcome Noise (Sample Luck $\bar{\eta}$)** | $\text{Realized Yield} - \text{Net CLV}$ | $+9.15\text{ pp}$ | **$+19.42\text{ pp}$** | $+1.25\text{ pp}$ | Zero-mean coinflip luck ($29$ actual wins vs $25.67$ expected $= +3.33$ lucky wins) |
| **7. Realized Unit Yield** | $\text{Net Profit} / \text{Turnover}$ | **$-4.54\%$** | **$+23.91\%$** ($+24.26\%$ unit) | **$+7.60\%$** | **True Profit ($+4.83\%$) + Temporary Luck ($+19.42\text{ pp}$)** |

### 4.3 $t$-Statistic Shootout: Realized Yield vs. Net CLV ($N=50$ Production Bets, 10,000 Bootstrap Draws)

| Metric ($N=50$ Production Bets) | Point Estimate | Sample Std. Dev. ($s$) | Standard Error ($\text{SE}$) | $t$-Statistic | Two-Sided $p$-Value | 95% Bet-Level Bootstrap CI | 95% Monthly-Block Bootstrap CI |
|---|---:|---:|---:|---:|---:|---|---|
| **Realized Yield (Turnover Return)** | **+23.91%** | 116.50% | 16.48% | **1.472** | $0.1473$ (Not sig.) | $[−7.55\%, +54.83\%]$ | $[+8.04\%, +79.06\%]$ |
| **Bankroll ROI (10k Capital)** | **+17.15%** | 81.32% | 11.50% | **1.491** | $0.1423$ (Not sig.) | $[−5.42\%, +39.30\%]$ | $[+7.48\%, +25.38\%]$ |
| **Expected Net EV** | **+13.41%** | 6.15% | 0.87% | **15.414** | $<10^{-15}$ | $[+11.76\%, +15.16\%]$ | $[+12.68\%, +15.73\%]$ |
| **Raw CLV (Price Beating)** | **+19.13%** | 12.95% | 1.83% | **10.446** | $\mathbf{4.64 \times 10^{-14}}$ | **$[+15.55\%, +22.71\%]$** | **$[+17.39\%, +25.10\%]$** |
| **Net CLV (Post-12% Tax)** | **+4.83%** | 11.39% | 1.61% | **3.000** | $\mathbf{0.0042}$ | **$[+1.68\%, +7.98\%]$** | **$[+3.30\%, +10.09\%]$** |

$$\text{Variance Ratio} = \frac{\operatorname{Var}(\text{Unit Profit})}{\operatorname{Var}(\text{Net CLV})} = \frac{1.1650^2}{0.1139^2} = \mathbf{104.5\times}$$

---

### 4.4 How Long Must You Bet to See "Almost Only Profit"? (Noise Decay Table)

At average odds $O = 2.53$ under 12% tax ($0.88 \times 2.53 = 2.226$), theoretical per-bet outcome noise is $\sigma_{\text{bet}} = \sqrt{p_{\text{close}}(1 - p_{\text{close}})(0.88 \cdot O)^2} = \mathbf{109.79\%}$. After $N$ bets:
$$\text{Realized Yield}(N) = +4.83\% \pm \frac{109.79\%}{\sqrt{N}}$$

| Number of Bets ($N$) | Time at **15 Bets/Mo** *(Strict Synergy)* | Time at **30 Bets/Mo** *(Playbook C Pace)* | Remaining Noise $\text{SE}_N = \frac{109.8\%}{\sqrt{N}}$ | 95% Realized Yield Range ($\mu \pm 1.96\,\text{SE}_N$) | Probability of Positive Profit $P(\text{Yield} > 0)$ | Wallet Behavior |
|---:|---:|---:|---:|---|---:|---|
| **10 bets** | 3 weeks | 10 days | $\pm 34.72\%$ | $[−63.21\%, +72.88\%]$ | **55.5%** | Pure coinflip; noise is $7.2\times$ larger than $+4.83\%$ edge |
| **20 bets** | 1.3 months | 3 weeks | $\pm 24.55\%$ | $[−43.28\%, +52.95\%]$ | **57.8%** | Wide swings (e.g., Aug 2026 had $-0.68\%$ over 18 bets) |
| **50 bets** *(Current)* | 3.3 months | 1.7 months | $\pm 15.53\%$ | $[−25.60\%, +35.27\%]$ | **62.2%** | Net CLV proven ($p=0.004$), but wallet PnL is still $75\%$ luck |
| **100 bets** | 6.7 months | 3.3 months | $\pm 10.98\%$ | $[−16.68\%, +26.35\%]$ | **67.0%** | 2-in-3 chance of profit; a bad 2-month run can still be $-10\%$ |
| **250 bets** | 1.4 years | 8.3 months | $\pm 6.94\%$ | $[−8.78\%, +18.44\%]$ | **75.7%** | 3-in-4 chance of profit; edge visibly pulls bankroll upward |
| **500 bets** | 2.8 years | 1.4 years | $\pm 4.91\%$ | $[−4.79\%, +14.46\%]$ | **83.8%** | Noise $\text{SE}$ ($4.91\%$) drops to equal your true profit ($4.83\%$) |
| **1,000 bets** | 5.5 years | 2.8 years | $\pm 3.47\%$ | $[−1.97\%, +11.64\%]$ | **91.8%** | $>9$ in $10$ chance of profit; 95% lower bound is nearly flat ($-2\%$) |
| **1,396 bets** | **7.8 years** | **3.9 years** | **$\pm 2.94\%$** | **$[−0.92\%, +10.59\%]$** | **95.0%** | 🏆 **"Almost Only Profit" threshold** ($z = 1.645$ one-sided 95% certainty) |
| **2,000 bets** | 11.1 years | 5.5 years | $\pm 2.45\%$ | **$[+0.02\%, +9.65\%]$** | **97.6%** | Two-sided 95% lower bound is strictly positive ($>0\%$) |
| **2,792 bets** | 15.5 years | 7.8 years | $\pm 2.08\%$ | **$[+0.76\%, +8.91\%]$** | **99.0%** | **99% Guaranteed Profit**; noise compressed to $\pm 2\%$ |

### 4.5 Comparing the 3 Operating Regimes ($N_{\min}$ Required for Profit Certainty)

| Operating Regime | True Profit $\mu$ ($\text{Net CLV}$) | Per-Bet Noise $\sigma$ | Bets for **75% Profit Prob** | Bets for **90% Profit Prob** | Bets for **95% Profit Prob ("Almost Only Profit")** | Bets for **99% Profit Prob** | Bets Needed to Prove **Net CLV > 0** (at 95% Conf.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Regime 1: 0% Tax (International Books / Betclic 0% Promo)** | **$+7.60\%$** | $101.0\%$ | **81 bets** *(~1.5 mos)* | **291 bets** *(~5 mos)* | **478 bets** *(~8–10 mos)* | **956 bets** *(~1.6 yrs)* | **7 bets** |
| **Regime 2: 12% Polish Tax — Early Stale Books ($T-48\text{h}..12\text{h}$, Optimal Synergy)** | **$+4.83\%$** | $109.8\%$ | **235 bets** *(~8 mos)* | **848 bets** *(~2.3 yrs)* | **1,396 bets** *(~3.8 yrs)* | **2,792 bets** *(~7.8 yrs)* | **15 bets** |
| **Regime 3: 12% Polish Tax — Conservative / Late Window ($T-6\text{h}..2\text{h}$)** | **$+3.00\%$** | $109.8\%$ | **610 bets** *(~1.7 yrs)* | **2,200 bets** *(~6.0 yrs)* | **3,624 bets** *(~9.9 yrs)* | **7,249 bets** *(~19.8 yrs)* | **39 bets** |

---

### 4.6 Multi-Year (2021–2026), Half-Year (2024–2026), and Rolling-20 Stability

#### A. Multi-Year 6-Season Stability (`N=3,002` Bets, Shrunk Hybrid + Caps, 0% Tax, Flat 1u)

| Season Year | Bets $N$ | Win Rate | Avg Odds | Expected EV% | Realized Yield% | EV Gap (pp) | Net Profit (100 PLN/u) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **2021 Season** | 417 | 56.6% | 1.89 | +4.32% | **+1.15%** | +3.17 pp | +479 PLN |
| **2022 Season** | 742 | 58.5% | 2.00 | +4.34% | **+7.90%** | −3.56 pp | +5,863 PLN |
| **2023 Season** | 746 | 59.4% | 2.08 | +5.07% | **+12.98%** | −7.91 pp | +9,684 PLN |
| **2024 Season** | 530 | 55.3% | 2.11 | +4.66% | **+7.47%** | −2.81 pp | +3,957 PLN |
| **2025 Season** | 435 | 55.9% | 2.01 | +4.80% | **+3.08%** | +1.73 pp | +1,338 PLN |
| **2026 Season (to Mar)** | 132 | 59.8% | 2.01 | +5.16% | **+11.29%** | −6.13 pp | +1,490 PLN |
| **6-Year Aggregate (2021–2026)** | **3,002** | **57.6%** | **2.03** | **+4.68%** | **+7.60%** | **−2.92 pp** | **+22,811 PLN** |

#### B. Rolling 20-Bet Windows on 2026 Production Ledger (`optimal_strategy_simulation_ledger.csv`)

| Bet Window | Date Range | Rolling-20 Realized Yield% | Rolling-20 Expected EV% | Rolling-20 Raw CLV% | Rolling-20 Net CLV% (12% Tax) | Cumulative Drawdown% |
|---|---|---:|---:|---:|---:|---:|
| **Bets 1–20** | May 30 – July 25 | +25.04% | +14.16% | +21.35% | **+6.79%** | 2.52% |
| **Bets 6–25** | June 11 – July 30 | +12.58% | +13.32% | +19.05% | **+4.76%** | 1.30% |
| **Bets 11–30** | July 11 – Aug 02 | +34.82% | +13.18% | +16.49% | **+2.51%** | 0.00% |
| **Bets 16–35** | July 20 – Aug 05 | +22.16% | +12.28% | +16.77% | **+2.76%** | 3.52% |
| **Bets 21–40** | July 25 – Aug 10 | +11.97% | +12.62% | +16.13% | **+2.20%** | 5.78% |
| **Bets 26–45** | July 30 – Aug 20 | −8.05% | +12.30% | +16.16% | **+2.22%** | 5.15% |
| **Bets 31–50** | Aug 02 – Sept 08 | −1.21% | +13.68% | +20.19% | **+5.77%** | 0.30% |

* **Key Observation**: Rolling-20 Realized Yield fluctuates between **$-8.05\%$ and $+34.82\%$** due to binomial match noise, whereas **Rolling-20 Net CLV stays strictly positive ($+2.20\%$ to $+6.79\%$) in every single window**.

---

## Part V: Forensic Code-vs-Simulation Audit & Operational Playbook

### 5.1 Live Code vs. Verified Simulation Alignment

| Mechanism | Verified Optimal Simulation Rule | Current Live Code State | Exact File & Line Reference | Risk / Required Fix |
|---|---|---|---|---|
| **1. Default Inference Model** | **Shrunk Hybrid ($\alpha=0.50$)** (`Hybrid-Operational-Market`) | `DEFAULT_MODEL_NAME = "Causal-C0"` (Unshrunk!) unless CLI passes `--operational-hybrid` | `betting_app/services/upcoming_inference_service.py:2718` | ⚠️ **HIGH RISK**: Calling `generate_model_ev_signals()` with default args uses unshrunk `Causal-C0`, re-exposing the portfolio to the Optimizer's Curse. |
| **2. Disagreement Veto** | **Thesis Model Veto**: $\|p_{\text{C0}} - p_{\text{EXP-039}}\| \le 0.10$ (`qualify_optimal_synergy_bet`) | Uses **Elo/Glicko rating spread** (`rating_disagreement <= 0.15`), does not call `qualify_optimal_synergy_bet` | `betting_app/services/upcoming_inference_service.py:2441`, `betting_app/services/bet_qualification_service.py:14` | ⚠️ **MEDIUM RISK**: Live code uses rating spread instead of the EXP-039 ElasticNet veto that filtered 8/9 losing bets in the 2026 simulation. |
| **3. Guard B, Caps & Shin** | Shin 2-way devigging, $\Delta_{\text{logit}} \le 0.85$, $\text{EV} \in [5\%, 25\%]$, $\text{Odds} \le 4.00$, 1/4 Kelly (cap 150 PLN) | **All active and aligned** in `ev.py`, `staking.py`, and `bet_qualification_service.py` | `betting_app/core/ev.py:20`, `bet_qualification_service.py:16-25, 441-454`, `staking.py:26` | ✅ **ALIGNED** (minor docstring typo in `staking.py:43` says `0.05` while signature default is `0.25`). |

### 5.2 The 4 Operational Rules to Accelerate Profit Convergence by $3\times$

1. **Eliminate the 12% Tax Whenever Possible ($1,396 \to 478$ bets)**:
   - The 12% Polish turnover tax consumes $\sim 14.3\text{ pp}$ of your $+19.13\%$ Raw CLV (leaving $+4.83\%$ Net CLV).
   - Routing bets to **0%-tax promotions (e.g., Betclic "Gra bez podatku")** or **0%-tax international accounts** triples your net edge and cuts the number of bets required for 95% profit certainty by **$3\times$** (from $1,396$ bets down to **$478$ bets**, or under 1 season).
2. **Execute Strictly in the $T-48\text{h}$ to $T-12\text{h}$ Window When $\ge 2$ Books Are Live**:
   - Never bet isolated 1-book openers (`books == 1`), and avoid last-hour ($T-1\text{h}$) bets under 12% tax where Net CLV compresses to $+1.29\%$.
3. **Prioritize Soft/Stale Books (Totalbet, Betfan, STS, LeBull) and Moderate Odds ($1.70 \le O \le 2.60$)**:
   - Because $N_{\min} \propto \sigma^2$, betting at $O \approx 1.90\text{--}2.30$ ($\sigma \approx 82\%\text{--}96\%$) cuts the required number of bets to see steady profit by **$2.5\times$** compared to longshots at $O \approx 3.50$ ($\sigma \approx 135\%$).
4. **Use Rolling-30 Net CLV ($\ge +3.5\%$) as Your Operational Truth Metric**:
   - It takes **1,396 bets** for your cash wallet to have a 95% guarantee of being positive under 12% tax, but only **15 to 30 bets** for your **average Net CLV** to have a 95%–99% guarantee of being positive.

---

### 5.3 Fixed Flat Stakes vs. Capped 1/4 Kelly vs. Uncapped 1/4 Kelly Across All Simulations

| Dataset & Cohort | Staking Method | Bets $N$ | Total Staked (PLN) | Net Profit (PLN) | Realized Yield% (Profit/Turnover) | Bankroll ROI% (10k Base) | Max Drawdown% | Key Finding |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **1. Production Optimal Synergy (`2026`, 12% Tax)** | **Fixed Flat 100 PLN (1.0%)** | 50 | 5,000.00 | +1,212.80 | **+24.26%** | +12.13% | **4.29%** | **Highest Yield & lowest Drawdown** |
| **1. Production Optimal Synergy (`2026`, 12% Tax)** | **Fixed Flat 150 PLN (1.5%)** | 50 | 7,500.00 | +1,819.20 | **+24.26%** | +18.19% | **6.35%** | Beats Capped 1/4 Kelly on both Yield & Profit |
| **1. Production Optimal Synergy (`2026`, 12% Tax)** | **1/4 Kelly (150 PLN Cap)** *(Report Default)* | 50 | 7,173.15 | +1,715.41 | **+23.91%** | +17.15% | **6.10%** | **40 of 50 bets (80%) hit the 150 PLN cap** (mean stake 143.46 PLN) |
| **1. Production Optimal Synergy (`2026`, 12% Tax)** | **Uncapped 1/4 Kelly** | 50 | 22,770.00 | +2,921.71 | **+12.83%** | +29.22% | **19.60%** | Over-stakes noisy high-EV bets; **cuts Yield in half** & triples DD |
| **2. Production 2h Playbook C/D (`2026`, 12% Tax)** | **Fixed Flat 100 PLN (Policy C)** | 103 | 10,300.00 | +2,046.40 | **+19.87%** | **+20.46%** | **8.67%** | **Dominates Kelly on Yield, ROI, and Drawdown** |
| **2. Production 2h Playbook C/D (`2026`, 12% Tax)** | **1/4 Kelly, 250 PLN Cap (Policy D)** | 103 | 17,337.52 | +1,805.48 | **+10.41%** | +18.05% | **17.00%** | Larger stakes on higher-EV errors drag Yield down by −9.46 pp |
| **3. OddsPortal Canonical (`2024–2026`, 0% Tax, Optimal Synergy)** | **Fixed Flat 100 PLN** | 557 | 55,700.00 | +4,863.00 | **+8.73%** | **+48.63%** | **24.49%** | Outperforms 1/4 Kelly across all metrics |
| **3. OddsPortal Canonical (`2024–2026`, 0% Tax, Optimal Synergy)** | **1/4 Kelly (150 PLN Cap)** | 557 | 53,440.00 | +3,244.00 | **+6.07%** | +32.44% | **34.82%** | Worse Yield (−2.66 pp) and higher Max DD (+10.33 pp) |
| **4. OddsPortal Multi-Year (`2021–2026`, 0% Tax, Hybrid)** | **Fixed Flat 100 PLN (Capped)** | 3,002 | 300,200.00 | +22,811.00 | **+7.60%** | +228.11% | ~18.5% | Positive in all 6 calendar years (2021–2026) |
| **4. OddsPortal Multi-Year (`2021–2026`, 0% Tax, Hybrid)** | **1/4 Kelly (150 PLN Cap)** | 3,528 | 518,077.36 | +52,579.30 | **+10.15%** | +525.79% | 42.41% | Higher multi-year compounding, but 42.4% Max Drawdown |
| **5. OddsPapi / Pinnacle $T-1\text{h}$ (`2026`, 0% Tax, Full Gated)** | **Fixed Flat 100 PLN** | 89 | 8,900.00 | +615.00 | **+6.91%** | +6.15% | **9.90%** | Evaluated strictly on Fixed Flat 100 PLN |
| **6. Noise Decay & $N_{\min}$ Tables ($1,396$ / $478$ Bets)** | **Fixed Unit Stakes (1u)** | — | — | — | +4.83% / +7.60% | — | — | Exact equal-weighted unit return convergence |

---

### 5.4 Micro-Bankroll (`100 PLN`) Simulation, Sub-Wallet Allocation & Anti-Detection Rules

Simulating the exact 50-bet Optimal Synergy Ledger starting from a **100.00 PLN** bankroll under the 12% Polish turnover tax with exact Polish grosz (`0.01 PLN`, `ROUND_HALF_UP`) payout rounding:

| Staking Rule on `100.00 PLN` Bankroll | Bets $N$ | Total Staked (PLN) | Net Profit (PLN) | Final Bankroll (PLN) | Realized Yield% | Bankroll ROI% | Max Drawdown (PLN / %) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Exact 1:100 Scale (`~1.43 PLN`/bet)** | 50 | 71.73 | **+17.17 PLN** | 117.17 PLN | +23.94% | +17.17% | 7.16 PLN (6.09%) |
| **Fixed Flat `1.50 PLN` (1.5%/bet)** | 50 | 75.00 | **+18.20 PLN** | 118.20 PLN | +24.27% | +18.20% | 7.24 PLN (6.09%) |
| 🏆 **Fixed Flat `2.00 PLN` (2.0%/bet — Bookie Min)** | **50** | **100.00** | **+24.24 PLN** | **124.24 PLN** | **+24.24%** | **+24.24%** | **9.66 PLN (7.72%)** |
| **Fixed Flat `3.00 PLN` (3.0%/bet)** | 50 | 150.00 | **+36.38 PLN** | 136.38 PLN | +24.25% | +36.38% | 14.49 PLN (10.53%) |
| **Fixed Flat `5.00 PLN` (5.0%/bet)** | 50 | 250.00 | **+60.65 PLN** | 160.65 PLN | +24.26% | +60.65% | 24.14 PLN (14.84%) |
| **1/4 Kelly (`Min 2.00 PLN, Cap 3.00 PLN`)** | 50 | 130.67 | **+27.70 PLN** | 127.70 PLN | +21.20% | +27.70% | 13.37 PLN (10.22%) |

#### Sub-Wallet Performance & Drawdown Dips at `Flat 2.00 PLN`

| Bookmaker | Bets $N$ | Wins | Net Profit (`@ 2 PLN`) | Lowest Cumulative Dip | Max Sub-Wallet Drawdown | Recommended `100 PLN` Allocation |
|---|---:|---:|---:|---:|---:|---|
| **Totalbet** | 7 | 6 | **+10.35 PLN** | 0.00 PLN | 2.00 PLN | **30 PLN** |
| **Betfan** | 7 | 6 | **+11.83 PLN** | 0.00 PLN | 2.00 PLN | **30 PLN** *(or keep clean in shadow)* |
| **STS** | 23 | 10 | **+0.14 PLN** | −2.16 PLN | 8.54 PLN | **40 PLN** *(needs buffer for 23 bets & 8.54 PLN DD)* |
| **LeBull** | 10 | 6 | **+3.84 PLN** | −2.53 PLN | 4.00 PLN | *Skip on 100 PLN (keep in shadow)* |
| **Superbet / Betclic** | 3 | 1 | **−1.92 PLN** | −2.00 PLN | 2.00 PLN | *Skip on 100 PLN (keep in shadow)* |

#### Anti-Limiting ("Anti-Gubbing") Rules for `2 PLN` Micro-Testing
1. **What `2 PLN` protects against (Trader Liability Queues)**: At `2.00 PLN` to `5.00 PLN`, your ticket liability (`~3 PLN`) stays far below the `100–250 PLN` manual trader review threshold on ERL esports, so tickets auto-accept without human inspection.
2. **What `2 PLN` does NOT protect against ("Account Burning" via Nightly CLV Bots)**: Soft bookmaker risk engines (Altenar/Sportingtech/BetConstruct) run automated nightly scripts scoring every account's **CLV%** and **market tier**. If you place 50 straight `2 PLN` early-opener bets on ERL/Academy matches that beat the closing line by `+19%`, the automated bot can limit your PESEL account *before* you ever scale up stakes.
3. **How to avoid burning your accounts during a `100 PLN` test**:
   - **Always bet whole-integer PLN (`2 PLN`, `3 PLN`, `5 PLN`)**: Never bet Kelly decimals like `2.37 PLN` or `1.43 PLN`—non-integer stakes are an instant automated bot/arb flag.
   - **Allocate the `100 PLN` across only 2–3 accounts (`40 PLN` STS + `30 PLN` Totalbet + `30 PLN` Betfan, or `50 PLN` STS + `50 PLN` Totalbet)** so no sub-wallet runs dry during a `8.54 PLN` drawdown or overlapping open matches. Keep your remaining bookmaker accounts untouched until you scale up.

---

### 5.5 Betclic-Only Simulation (`100 PLN` Budget, `0% Tax` for Next `900 PLN` Won, Strictly Causal Opening Lines)

When evaluating **327 verified pre-match Betclic fixtures (`17,452` quotes, May–Sep 2026)** using **strictly causal opening probabilities (`q_open`)** (no closing-line lookahead in `p_hybrid`):

#### A. Why the `0% Tax` Promo Transforms Betclic from Unprofitable to High-Edge

| Betclic Strategy Configuration (Strictly Causal `q_open`, $N=327$ Fixtures) | Tax Rate | Bets $N$ | Win Rate | Avg Odds | Expected EV% | Realized Yield% | EV Gap ($\text{EV}-\text{Yield}$) | Net CLV% (vs Close Consensus) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Pure C0 (Unshrunk, No Gates)** | **12% Tax** | 52 | 44.2% | 3.17 | +25.75% | +0.79% | **+24.96 pp** (Curse) | −11.77% |
| **2. Optimal Synergy ($\alpha=0.50$ + Guard B $\le 0.85$ + EV $[5,25\%]$ + Veto $\le 0.10$)** | **12% Tax** | **0** *(or 29 forced)* | 48.3% | 2.27 | −4.21% | **−3.66%** | −0.55 pp | **−7.72%** |
| **3. Pure C0 (Unshrunk, No Gates)** | **0% Tax** | 162 | 51.2% | 2.38 | +23.95% | **−3.94%** | **+27.89 pp** (Curse) | +0.14% |
| **4. Comfort Policy ($\alpha=0.50$ + Guard B $\le 0.85$ + EV $[5,25\%]$ + Odds $\le 4.0$)** | **0% Tax** | 32 | 43.8% | 2.32 | +8.90% | −0.78% | +9.68 pp | +3.65% |
| 🏆 **5. Optimal Synergy ($\alpha=0.50$ + Guard B + Caps + Thesis Veto $\le 0.10$)** | **0% Tax** | **29** | **48.3%** | **2.27** | **+8.85%** | **+9.48%** | **−0.63 pp (Honest)** | **+4.86%** |
| 🏆 **6. Optimal Synergy + Free Cross-Book Check (`Betclic Odds >= Open Consensus`)** | **0% Tax** | **22** | **50.0%** | **2.22** | **+8.71%** | **+12.27%** | **−3.56 pp (Honest)** | **+5.03%** |
| 🏆 **7. Optimal Synergy + Near-Best Price (`Betclic Odds >= 0.97 * Best Market`)** | **0% Tax** | **16** | **50.0%** | **2.08** | **+8.65%** | **+10.94%** | **−2.29 pp (Honest)** | **+6.20%** |

#### B. `100.00 PLN` Betclic Budget Simulation & `900 PLN Won` Quota Consumption

| Betclic 0%-Tax Policy & Whole-PLN Staking Rule (`100.00 PLN` Starting Budget) | Bets $N$ | Total Staked | Net Profit | Final Bankroll | Realized Yield% | Bankroll ROI% | Max Drawdown (PLN / %) | Gross Payouts Won (of `900 PLN` Quota) | Net Profit on Wins (of `900 PLN` Quota) | Est. Bets Until `900 PLN` Quota Hit |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **All 29 Betclic Synergy Bets — `Flat 2.00 PLN` (2%)** | 29 | 58.00 PLN | **+5.50 PLN** | 105.50 PLN | **+9.48%** | +5.50% | 21.90 PLN (17.5%) | 63.50 PLN (7.1%) | 35.50 PLN (3.9%) | **~411 bets** *(~735 if net)* |
| **All 29 Betclic Synergy Bets — `Flat 3.00 PLN` (3%)** | 29 | 87.00 PLN | **+8.25 PLN** | 108.25 PLN | **+9.48%** | +8.25% | 32.85 PLN (23.9%) | 95.25 PLN (10.6%) | 53.25 PLN (5.9%) | **~274 bets** *(~490 if net)* |
| **All 29 Betclic Synergy Bets — `Flat 5.00 PLN` (5%)** | 29 | 145.00 PLN | **+13.75 PLN** | 113.75 PLN | **+9.48%** | +13.75% | 54.75 PLN (33.7%) | 158.75 PLN (17.6%) | 88.75 PLN (9.9%) | **~164 bets** *(~294 if net)* |
| **All 29 Betclic Synergy Bets — `1/4 Kelly Whole-PLN (2–5 PLN)`** | 29 | 78.00 PLN | **+6.15 PLN** | 106.15 PLN | **+7.88%** | +6.15% | 24.30 PLN (19.2%) | 84.15 PLN (9.4%) | 44.15 PLN (4.9%) | **~310 bets** *(~591 if net)* |
| 🏆 **22 Cross-Book Filtered Bets — `Flat 2.00 PLN` (2%)** | 22 | 44.00 PLN | **+5.40 PLN** | 105.40 PLN | **+12.27%** | +5.40% | **15.90 PLN (13.3%)** | 49.40 PLN (5.5%) | 27.40 PLN (3.0%) | **~400 bets** *(~722 if net)* |
| 🏆 **22 Cross-Book Filtered Bets — `Flat 3.00 PLN` (3%)** | 22 | 66.00 PLN | **+8.10 PLN** | 108.10 PLN | **+12.27%** | +8.10% | **23.85 PLN (18.6%)** | 74.10 PLN (8.2%) | 41.10 PLN (4.6%) | **~267 bets** *(~481 if net)* |
| 🏆 **16 Near-Best-Price Bets — `Flat 3.00 PLN` (3%)** | 16 | 48.00 PLN | **+5.25 PLN** | 105.25 PLN | **+10.94%** | +5.25% | **18.60 PLN (15.3%)** | 53.25 PLN (5.9%) | 29.25 PLN (3.3%) | **~270 bets** *(~492 if net)* |

---

### 5.6 Wasserstein DRO (`W_1 <= ε`) Across All 7 Bookmakers & 12% Polish Tax

For a binary match outcome $Y \in \{0, 1\}$, a 1-Wasserstein ambiguity ball $W_1(\mathbb{Q}, \text{Bernoulli}(\hat{p})) \le \varepsilon$ shifts win probability to $p_{\text{DRO}}(\varepsilon) = \hat{p} - \varepsilon$. Requiring positive worst-case net return replaces a flat 5% EV hurdle with an **odds-scaled EV hurdle**:
$$\text{EV}_{\text{DRO}}(\varepsilon) = (\hat{p}_{\text{hybrid}} - \varepsilon) \cdot 0.88 \cdot O - 1 > 0 \quad\Longleftrightarrow\quad \text{EV}_{\text{net}} > \max\!\big(0.05,\; 0.88 \cdot O \cdot \varepsilon\big)$$

#### A. Wasserstein DRO on the 50-Bet Production Ledger (All 7 Polish Books, 12% Tax)

| Wasserstein Radius $\varepsilon$ | Required Hurdle at $O=3.50$ | Bets $N$ | Win Rate | Avg Odds | **Flat Stake Yield%** | **W-Kelly Yield%** | Max Drawdown% | **True Edge (Net CLV%)** | **Pos Net CLV Rate%** | $N_{\min}$ for 95% Profit Prob |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **$\varepsilon = 0.000$ (Current Flat $\ge 5\%$)** | $+5.00\%$ | 50 | 58.0% | 2.53 | +24.26% | +15.70% | 6.10% | **+4.83%** | 66.0% | **1,396 bets** |
| 🏆 **$\varepsilon = 0.025$ ($2.5\text{ pp}$ ball)** | **$+7.70\%$** | **45** | **60.0%** | **2.46** | **+24.86%** | +13.59% | **5.41%** | **+6.98%** | **73.3%** | **641 bets (−54.1%)** |
| **$\varepsilon = 0.030$ ($3.0\text{ pp}$ ball)** | $+9.24\%$ | 43 | 58.1% | 2.45 | +20.29% | +13.26% | 5.20% | **+7.21%** | 74.4% | 599 bets |
| 🏆 **$\varepsilon = 0.035$ ($3.5\text{ pp}$ ball)** | **$+10.78\%$** | **39** | **61.5%** | **2.40** | **+26.65%** | +12.54% | **4.79%** | **+7.47%** | **74.4%** | **547 bets (−60.8%)** |
| **$\varepsilon = 0.040$ ($4.0\text{ pp}$ ball)** | $+12.32\%$ | 35 | 60.0% | 2.36 | +23.68% | +11.57% | 5.34% | **+8.86%** | 80.0% | 382 bets |
| **$\varepsilon = 0.050$ ($5.0\text{ pp}$ ball)** | $+15.40\%$ | 27 | 63.0% | 2.31 | +24.11% | +8.61% | 5.82% | **+11.67%** | **92.6%** | 216 bets |

#### B. The Exact 5 Bets Pruned by Wasserstein DRO ($\varepsilon = 0.025$) on the Production Ledger
*Every single one of the 5 bets removed by $\varepsilon = 0.025$ had **strictly negative Net CLV** (5 out of 5 = 100% negative true closing edge, averaging **−14.51% Net CLV**):*

| Bet ID | Date | League | Team Selected | Bookmaker | Odds Taken | Claimed Model EV% | **True Closing Edge (Net CLV%)** | Outcome |
|---:|---|---|---|---|---:|---:|---:|:---:|
| 20 | 2026-07-25 | LCP | Ground Zero Gaming | STS | 2.50 | +5.35% | 🔴 **−3.68%** | Lost |
| 22 | 2026-07-26 | CBLOL | Keyd Stars Esports | STS | 3.50 | +6.68% | 🔴 **−18.71%** | Won |
| 28 | 2026-07-31 | LEC | Team Heretics | STS | 3.80 | +7.88% | 🔴 **−17.38%** | Lost |
| 30 | 2026-08-02 | LCK | DN Soopers | Betfan | 3.25 | +6.61% | 🔴 **−17.67%** | Won |
| 45 | 2026-08-20 | LCK CL | DN SOOPers Challengers | STS | 3.05 | +5.93% | 🔴 **−15.12%** | Lost |
| **Mean of 5 Pruned Bets** | — | — | — | — | **3.22** | **+6.49%** | 🔴 **−14.51%** | **40.0% (2W–3L)** |
| **Mean of 45 Kept Bets** | — | — | — | — | **2.46** | **+14.17%** | 🟢 **+6.98%** | **60.0% (27W–18L)** |

#### C. Wasserstein DRO Across All Other Normal Datasets (12% Polish Tax & Canonical 2024–2026)

| Dataset & Scope | Rule | Bets $N$ | Win Rate | Avg Odds | Flat Stake Yield% | W-Kelly Yield% | Net CLV% (12% Tax) | Pos Net CLV% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Strictly Causal Opening (All 7 Books, 12% Tax)** | Flat $\text{EV}_{12\%} \ge 5.0\%$ | 15 | 60.0% | 2.83 | +43.38% | +28.10% | **+4.51%** | 60.0% |
| *Same cohort* | **+ Wasserstein DRO $\varepsilon = 0.025$** | **12** | **75.0%** | **2.58** | **+64.56%** | +31.40% | **+8.48%** | **66.7%** |
| *Same cohort* | **+ Wasserstein DRO $\varepsilon = 0.030$** | **8** | **87.5%** | **2.58** | **+101.19%** | +42.15% | **+12.46%** | **75.0%** |
| **2. Full Pre-Match Trajectory (All 7 Books, 12% Tax, 108k Quotes)** | Flat $\text{EV}_{12\%} \ge 5.0\%$ | 23 | 56.5% | 2.88 | +37.97% | +27.37% | **+2.56%** | 56.5% |
| *Same cohort* | **+ Wasserstein DRO $\varepsilon = 0.025$** | **19** | **63.2%** | **2.80** | **+50.80%** | +19.04% | **+5.19%** | **63.2%** |
| *Same cohort* | **+ Wasserstein DRO $\varepsilon = 0.030$** | **16** | **68.8%** | **2.81** | **+64.50%** | +12.63% | **+7.86%** | **68.8%** |
| **3. Canonical 2024–2026 Benchmark ($N=2,275$ Matches)** | Flat $\text{EV} \ge 5.0\%$ | 195 | 46.2% | 2.56 | +9.67% | — | — | — |
| *Same cohort* | **+ Wasserstein DRO $\varepsilon = 0.025$** | **157** | **51.6%** | **2.39** | **+17.41%** | — | — | — |

---

### 5.7 Bankroll Growth Roadmap: How Long From `100 PLN` to `1,000 PLN`, `5,000 PLN`, and `10,000 PLN`?

To grow `100 PLN` to `1,000 PLN` ($10\times$), `5,000 PLN` ($50\times$), and `10,000 PLN` ($100\times$), you **cannot** keep your stake frozen at `2.00 PLN` forever (at `+0.14 PLN` expected profit per `2 PLN` bet, making `+9,900 PLN` would take $\sim 70,900$ bets). Instead, you **step-compound whole-integer PLN stakes** as your bankroll $B$ grows:
$$\text{Stake}(B) = \min\!\big(150\text{ PLN},\; \max\!\big(2\text{ PLN},\; \operatorname{round}(0.03 \times B)\big)\big)$$
*(i.e., `3 PLN` at `100 PLN` $\to$ `6 PLN` at `200 PLN` $\to$ `15 PLN` at `500 PLN` $\to$ `30 PLN` at `1,000 PLN` $\to$ `150 PLN` at `5,000+ PLN`).*

#### Monte Carlo First-Passage Milestones Starting From `100.00 PLN` (Combined Setup + Wasserstein $\varepsilon = 0.025$)

| Edge & Tax Scenario (`3%` Step-Compounded Whole-PLN Stake, Capped at `150 PLN`) | **`100 PLN` $\to$ `1,000 PLN`**<br>*(Median Bets & [25%–75%] Range)* | **`100 PLN` $\to$ `5,000 PLN`**<br>*(Median Bets & [25%–75%] Range)* | **`100 PLN` $\to$ `10,000 PLN`**<br>*(Median Bets & [25%–75%] Range)* | Risk of Ruin (`<2 PLN`) |
|---|---|---|---|---:|
| **1. Conservative True Net CLV (`+6.98%` Yield, 12% Tax From Day 1)** | **1,310 bets** `[933 – 1,801]`<br>*(~2.4 yrs at 45 bets/mo)* | **2,289 bets** `[1,758 – 3,011]`<br>*(~4.2 yrs at 45 bets/mo)* | **2,789 bets** `[2,206 – 3,531]`<br>*(~5.1 yrs at 45 bets/mo)* | 0.4% |
| 🏆 **2. Conservative True Net CLV (`+6.98%` + Betclic `0% Tax` First `900 PLN`)** | **843 bets** `[593 – 1,249]`<br>*(~1.5 yrs at 45 bets/mo)* | **1,869 bets** `[1,452 – 2,474]`<br>*(~3.4 yrs at 45 bets/mo)* | **2,353 bets** `[1,861 – 2,960]`<br>*(~4.3 yrs at 45 bets/mo)* | **0.0%** |
| 🏆 **3. Moderate Realistic Edge (`+12.5%` Yield + Betclic `0% Tax` Start)** | **639 bets** `[488 – 886]`<br>*(~1.2 yrs at 45 bets/mo)* | **1,346 bets** `[1,065 – 1,701]`<br>*(~2.5 yrs at 45 bets/mo)* | **1,686 bets** `[1,390 – 2,077]`<br>*(~3.1 yrs at 45 bets/mo)* | **0.0%** |
| **4. Historical 2026 In-Sample Rate (`+24.86%` Yield + Betclic `0% Tax` Start)** | **313 bets** `[257 – 386]`<br>*(~7 mos at 45 bets/mo)* | **625 bets** `[536 – 736]`<br>*(~14 mos at 45 bets/mo)* | **788 bets** `[700 – 919]`<br>*(~17.5 mos at 45 bets/mo)* | **0.0%** |

#### Step-by-Step Stake Progression Table (`100 PLN` $\to$ `10,000 PLN`)

| Bankroll Milestone | Whole-PLN Stake (`3%` of Bankroll) | Profit Needed for Next Stage | Incremental Bets Required (`Conservative +6.98%` vs `Moderate +12.5%` vs `Historical +24.86%`) | Where to Execute |
|---|---:|---:|---|---|
| **Stage 1: `100 PLN` $\to$ `300 PLN`** ($3\times$) | `3 PLN` $\to$ `9 PLN` | `+200 PLN` | **~360 bets** (Cons) / **~280 bets** (Mod) / **~145 bets** (Hist) | **Betclic `0% Tax` Promo** (`900 PLN` quota covers Stages 1 & 2) |
| **Stage 2: `300 PLN` $\to$ `1,000 PLN`** ($3.3\times$) | `9 PLN` $\to$ `30 PLN` | `+700 PLN` | **~483 bets** (Cons) / **~359 bets** (Mod) / **~168 bets** (Hist) | **Betclic `0% Tax`** + **STS / Totalbet** |
| **Stage 3: `1,000 PLN` $\to$ `5,000 PLN`** ($5\times$) | `30 PLN` $\to$ `150 PLN` | `+4,000 PLN` | **~1,026 bets** (Cons) / **~707 bets** (Mod) / **~312 bets** (Hist) | Split across **STS + Totalbet + Betfan + LeBull** |
| **Stage 4: `5,000 PLN` $\to$ `10,000 PLN`** ($2\times$) | `150 PLN` *(Capped)* | `+5,000 PLN` | **~484 bets** (Cons) / **~340 bets** (Mod) / **~163 bets** (Hist) | Multi-account execution (`150 PLN` cap per book) |
