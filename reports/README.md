# Master Research Reports Index & Evidence Navigation

This directory contains the primary research reports, statistical audits, econometric backtests, and architectural specifications produced across the **LoL-Breaker** project.

> **Research Governance Rules:**
> 1. **Do Not Delete or Overwrite Historical Reports:** Negative findings and rejected hypotheses must be preserved in place to prevent ungrounded experiment repetition.
> 2. **Current vs. Historical Status:** Refer to [`docs/MODEL_CATALOG.md`](../docs/MODEL_CATALOG.md) and [`docs/RESEARCH.md`](../docs/RESEARCH.md) for the active production and benchmark baselines. A dated report remains an immutable record of what was tested at that time.

---

## 1. Quick Navigation Matrix (Key Reference Documents)

| Category / Milestone | Primary Document | Model Evaluated | Date | Key Scientific Verdict |
|---|---|:---:|:---:|---|
| ⚖️ **Long-Term Trust & Noise Decay** | [`long_term_betting_trust_and_optimizers_curse_audit_20261014.md`](long_term_betting_trust_and_optimizers_curse_audit_20261014.md) | **C0 + Hybrid + Guards (All 3 Sources)** | 2026-10-14 | **Net CLV = +4.83% ($p=0.0042$, $104.5\times$ lower var than Yield)**; $N=1,396$ bets (12% tax) or $N=478$ (0% tax) for 95% profit certainty. |
| 🏆 **Optimal Betting Strategy** | [`optimal_synergy_c0_thesis_simulation_20261012.md`](optimal_synergy_c0_thesis_simulation_20261012.md) | **C0 Shrunk Hybrid + Thesis Guard** | 2026-10-12 | **+23.79% Yield / +17.12% ROI** (+4.83% Net CLV). Vetoes 88.9% of losing longshots. |
| 📊 **Multi-Year Market Backtest** | [`oddsportal_open_close_simulation_20261012.md`](oddsportal_open_close_simulation_20261012.md) | **Shrunk Hybrid (11.6k matches)** | 2026-10-12 | **+10.15% Yield / +42.5% CAGR** over 5 seasons (0% tax); **+9.65% Yield** under 12% tax. |
| 🔍 **Production Scrape Census** | [`monthly_refit_c0_production_odds_since_may_20261011.md`](monthly_refit_c0_production_odds_since_may_20261011.md) | **Monthly Walk-Forward C0** | 2026-10-11 | Audited 234,435 quotes across 1,600 matches. C0 refit monthly with zero spoilers. |
| 🧠 **Optimizer's Curse Solution** | [`how_professional_syndicates_lift_optimizers_curse_20261011.md`](how_professional_syndicates_lift_optimizers_curse_20261011.md) | **Benter / Smith & Winkler** | 2026-10-11 | Mathematical proof of how Bayesian shrinkage and Guard B lift the Optimizer's Curse. |
| 🎯 **In-Game Prop Distributions** | [`eda_prop_markets_idea018.md`](eda_prop_markets_idea018.md) | **Negative Binomial / LogNormal** | 2026-09-05 | Proved overdispersion ($\sigma^2/\mu = 3.02$) in kills. $r = +0.40$ with team pace. |
| 🏆 **Tournament Simulation** | [`reliable_tournament_simulation_research_20261006.md`](reliable_tournament_simulation_research_20261006.md) | **Exact Knockout DP + Swiss MC** | 2026-10-06 | Multi-phase bracket propagation; rejected native tournament copula. |

---

## 2. Thematic Catalog of Research Reports

### Group A: Match Prediction & Neural Architecture Studies

These reports investigate feature representations, neural transformers, capacity trade-offs, and multi-model consensus:

- [`optimal_synergy_c0_thesis_simulation_20261012.md`](optimal_synergy_c0_thesis_simulation_20261012.md): **Dual-Paradigm Synergy:** Combining C0 (micro-player attention) with the Thesis Model (macro W20 ElasticNet) to filter unmodeled noise and optimize betting return.
- [`monthly_refit_c0_production_odds_since_may_20261011.md`](monthly_refit_c0_production_odds_since_may_20261011.md): **Spoiler-Less Monthly C0 Refits:** Out-of-time evaluation across 645 matched canonical series from May to September 2026.
- [`model_bookmaker_competitiveness_execution_20261009.md`](model_bookmaker_competitiveness_execution_20261009.md): **Model vs. Bookmaker Competitiveness:** Statistical testing of native C0 vs Polish consensus.
- [`model_competitiveness_tests_research_20261009.md`](model_competitiveness_tests_research_20261009.md): Pre-experimental protocol for non-inferiority testing against market closing lines.
- [`champion_history_prediction_20261007.md`](champion_history_prediction_20261007.md): **Champion-History Multitask Challenger:** Evaluated joint winner and prop representations (**NOT ADOPTED**).
- [`c0_selected_betting_reliability_research_and_audit_20261006.md`](c0_selected_betting_reliability_research_and_audit_20261006.md): Comprehensive reliability and calibration audit of C0 selections.
- [`model_targets_capacity_and_distillation_research_20261004.md`](model_targets_capacity_and_distillation_research_20261004.md): Testing width 16/32/64 and knowledge distillation (**NOT ADOPTED**).
- [`plan09_c0_connector_and_forward_evidence_20261002.md`](plan09_c0_connector_and_forward_evidence_20261002.md): Learned sports connector and local decision-time capture implementation.
- [`next_model_steps_after_learned_connector_20261002.md`](next_model_steps_after_learned_connector_20261002.md): Analysis of directional derivatives and linear connectors.
- [`noisy_data_reliability_research_20260929.md`](noisy_data_reliability_research_20260929.md): Evaluating noise robustness, missing player statistics, and LPL coverage anomalies.
- [`ground_up_lol_prediction_20260927.md`](ground_up_lol_prediction_20260927.md): **Ground-Up Rebuild:** Evaluating 18 from-scratch model recipes against A0 baseline.
- [`a1_architecture_and_benchmark_report.md`](a1_architecture_and_benchmark_report.md): Audit and retirement of consolidated A1; formal reinstatement of Causal A0 as baseline.
- [`production_readiness_and_market_benchmark_report_2026.md`](production_readiness_and_market_benchmark_report_2026.md): Master production readiness audit and live server inspection.

---

### Group B: Quantitative Betting, Econometrics & Optimizer's Curse

These reports analyze market efficiency, Shin devigging, closing line value (CLV), Kelly capital allocation, and structural taxation:

- [`long_term_betting_trust_and_optimizers_curse_audit_20261014.md`](long_term_betting_trust_and_optimizers_curse_audit_20261014.md): **Multi-Source Trustworthiness & Time-to-Profit Audit:** Cross-source comparison (OddsPortal, OddsPapi/Pinnacle, Production), decomposition of True Profit ($\text{Net CLV}$) vs. Optimizer's Curse vs. Match Outcome Noise, and $N_{\min}$ convergence tables.
- [`how_professional_syndicates_lift_optimizers_curse_20261011.md`](how_professional_syndicates_lift_optimizers_curse_20261011.md): Theoretical synthesis of Smith & Winkler (2006) and Benter (1994) multi-stage market fusion.
- [`professional_betting_syndicates_and_bot_architecture_20261011.md`](professional_betting_syndicates_and_bot_architecture_20261011.md): Architectural blueprint for quantitative execution, scraper latency, and limit management.
- [`optimizers_curse_literature_and_sports_solutions_20261011.md`](optimizers_curse_literature_and_sports_solutions_20261011.md): Academic literature review of postdecision surprise in decision analysis and sports betting.
- [`historical_production_operational_playbook_20261011.md`](historical_production_operational_playbook_20261011.md): Standard operating procedure for opportunity ledgers and dual-clock provenance.
- [`optimizers_curse_devigging_and_guards_research_20261011.md`](optimizers_curse_devigging_and_guards_research_20261011.md): Mathematical derivation of Guard B ($\Delta\text{logit} \le 0.85$) and Shin insider devigging.
- [`sports_model_betting_reliability_research_20261009.md`](sports_model_betting_reliability_research_20261009.md): Analysis of selection bias and market residual overconfidence.
- [`bookmaker_noise_and_provider_training_20260927.md`](bookmaker_noise_and_provider_training_20260927.md): Empirical study on whether training models directly on bookmaker closing lines reduces prediction error.
- [`ev_overestimation_targets_and_plan_20260927.md`](ev_overestimation_targets_and_plan_20260927.md): Root-cause breakdown of why naive model selections overstate net expected value.
- [`why_a0_48h_ev_bets_fail_20260927.md`](why_a0_48h_ev_bets_fail_20260927.md): Demonstrating how early opening lines decay and reverse by kickoff.
- [`calibration_learning_and_noise_research_20260927.md`](calibration_learning_and_noise_research_20260927.md): Analysis of isotonic regression fitting optimism and beta calibration symmetry limits.
- [`zero_gap_ev_calibration_and_strategy_report.md`](zero_gap_ev_calibration_and_strategy_report.md): Methodology for zero-gap EV calibration.
- [`syndicate_ev_calibration_deep_research.md`](syndicate_ev_calibration_deep_research.md): Quantitative models used by Starlizard and professional syndicates.

---

### Group C: In-Game Props & Secondary Markets

These reports focus on Map 1 Total Kills, Game Duration, and Series Map Counts:

- [`draft_prop_20261007_01.md`](draft_prop_20261007_01.md): Testing whether post-draft champion compositions improve pre-match kill and duration forecasts (**NOT ADOPTED**).
- [`eda_prop_markets_idea018.md`](eda_prop_markets_idea018.md): **Foundational Prop Research:** Demonstrating overdispersion in kills, regional pace divergence, and rolling pace correlations ($r = +0.40$).

---

### Group D: Tournament Simulation & Distribution Models

These reports develop and audit bracket simulation algorithms:

- [`reliable_tournament_simulation_research_20261006.md`](reliable_tournament_simulation_research_20261006.md): **Exact Tournament Integration:** Knockout dynamic programming, exact GSL enumeration, and Swiss Monte Carlo rollout.
- [`tournament_simulation_deep_research_report.md`](tournament_simulation_deep_research_report.md): Official rulebook analysis (Worlds, MSI) and sequential lottery simulation.
- [`tournament_simulation_all_phases_report.md`](tournament_simulation_all_phases_report.md): Evaluation of group stage, Swiss stage, and knockout stages across 220 historical tournament phases.
- [`tournament_composite_calibration_audit_report.md`](tournament_composite_calibration_audit_report.md): Joint probability distribution calibration and rank probability scores (RPS).

---

### Group E: Preserved Archives & Legacy Subdirectories

The following subdirectories contain preserved historical artifacts and intermediate outputs:

- [`reports/archive/legacy_json_dumps/`](archive/legacy_json_dumps/): Consolidated historical JSON dumps from older experiment phases (e.g., `exp054`–`exp058`, neural player delta backtests, pandaskill tests).
- `exp039_*`: Historical backtests and sweeps for the EXP-039 baseline model.
- `exp081_*`: Historical artifacts for the legacy EXP-081 Siamese network.
- `exp063_`–`exp066_`: Context-aware player encoder and bootstrap comparison artifacts.
- `benchmarks_2026/`: 2026 benchmark output snapshots.
