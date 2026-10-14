# Comprehensive Betting Simulation on OddsPortal Data (Open vs. Close)

**Date:** 2026-10-12  
**Dataset Sources:**
1. **Multi-Year Historical Common Sample:** 11,609 matches from 2021-01-09 to 2026-03-22 (5 full seasons).
2. **Canonical Research Benchmark Cohort (`paired.parquet`):** 2,275 matches from 2024-01-13 to 2026-03-20.
**Execution Horizon:** Bets taken strictly at **Market Open (`market_open`)**, evaluated against **Closing Lines (`market_close`)**, and settled on observed match outcomes.  
**Staking Method:** Fractional 1/4 Kelly ($f^* = \frac{p \cdot b - (1-p)}{b}$, capped at 150.00 PLN, min 20.00 PLN) on 10,000.00 PLN initial capital.

---

## 1. Executive Summary

Betting on OddsPortal data simulates opening-line execution: taking early prices when bookmakers publish their opening market lines and comparing them to the closing lines settled at kickoff.

We evaluated two large-scale historical cohorts across both **0% tax (standard international betting on Pinnacle, Bet365, Betway)** and **12% tax (Polish statutory turnover tax)**:

1. **Multi-Year 5-Season OddsPortal Sample ($N=11,609$ matches, 2021–2026):**
   - **At 0% Tax (Shrunk Hybrid $\alpha=0.50$):**  
     **3,528 bets placed**, **53.7% win rate** at average odds **2.22**. Staked **518,077.36 PLN**.  
     Generated **+52,579.30 PLN net profit**, growing the bankroll from 10,000 PLN to **62,579.30 PLN** (**+525.79% ROI**, **+10.15% Yield**, **CAGR: +42.5% per year**).
   - **At 12% Tax (Shrunk Hybrid + Guard B):**  
     **684 bets placed**, **48.5% win rate** at average odds **2.64**. Staked **99,826.91 PLN**.  
     Generated **+9,637.61 PLN net profit**, growing the bankroll to **19,637.61 PLN** (**+96.38% ROI**, **+9.65% Yield**, **Max Drawdown: 14.96%**).
2. **Canonical 2024–2026 Cohort with Dual-Paradigm Consensus ($N=2,275$ matches):**
   - Running the **Optimal Synergy Strategy (C0 Shrunk Hybrid + Thesis Guard $\le 10\%$)** at 0% tax:  
     **557 bets placed**, **49.2% win rate** at average odds **2.51**. Staked **80,970.23 PLN**.  
     Generated **+11,267.34 PLN net profit**, growing the bankroll from 10,000 PLN to **21,267.34 PLN** (**+112.67% ROI**, **+13.92% Yield**, **CAGR: +41.7% per year**).
   - Under 12% tax on the OddsPortal composite: broke even ($-0.08\%$ Yield, Profit Factor 1.00), demonstrating that international composite odds require low/zero tax or retail domestic mispricing to overcome a 12% friction.

---

## 2. Multi-Year OddsPortal Simulation (2021–2026, 11,609 Matches)

Across 5 full seasons of competitive League of Legends (LCK, LPL, LEC, LCS, MSI, Worlds, EMEA Masters):

| Strategy Configuration | Tax Environment | Bets Placed | Win Rate | Average Odds | Total Turnover Staked | Net Profit (PLN) | Final Bankroll (PLN) | Yield (Return on Turnover) | Bankroll ROI | Compounded Annual Growth (CAGR) | Profit Factor | Max Drawdown | Mean Raw CLV | Positive CLV Rate |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Unshrunk Model** | **0% Tax (Intl)** | 4,352 | 59.1% | 1.97 | 649,433.63 PLN | +43,616.70 PLN | 53,616.70 PLN | **+6.72%** | **+436.17%** | **+38.5%/yr** | 1.16 | 22.60% | +3.79% | 68.0% |
| **Shrunk Hybrid ($\alpha=0.50$)** | **0% Tax (Intl)** | **3,528** | **53.7%** | **2.22** | **518,077.36 PLN** | **+52,579.30 PLN** | **62,579.30 PLN** | **+10.15%** | **+525.79%** | **+42.5%/yr** | **1.22** | 42.41% | **+6.35%** | **73.6%** |
| **Unshrunk Model** | **12% Tax** | 1,843 | 53.9% | 2.34 | 270,345.22 PLN | +11,572.51 PLN | 21,572.51 PLN | **+4.28%** | **+115.73%** | **+16.0%/yr** | 1.09 | 43.44% | +12.32% | 78.2% |
| 🏆 **Shrunk Hybrid + Guard B** | **12% Tax** | **684** | **48.5%** | **2.64** | **99,826.91 PLN** | **+9,637.61 PLN** | **19,637.61 PLN** | **+9.65%** | **+96.38%** | **+13.9%/yr** | **1.19** | **14.96%** | **+21.49%** | **87.4%** |

---

## 3. Canonical 2024–2026 Simulation: The Optimal Synergy Strategy

Evaluated on `paired.parquet` across 2,275 matches where both **Causal C0 (`p_full`)** and the **Thesis Model (`p_039`)** were evaluated simultaneously against OddsPortal open and close:

| Strategy | Tax Rate | Bets Placed | Win Rate | Average Odds | Turnover Staked | Net Profit | Final Bankroll | Yield (Turnover) | Bankroll ROI | CAGR (%/yr) | Profit Factor | Max Drawdown | Mean Raw CLV | Positive CLV Rate |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Optimal Synergy (C0 + Thesis Guard)** | **0% Tax (Intl)** | **557** | **49.2%** | **2.51** | **80,970.23 PLN** | **+11,267.34 PLN** | **21,267.34 PLN** | **+13.92%** | **+112.67%** | **+41.7%/yr** | **1.27** | **24.08%** | **+3.87%** | **60.5%** |
| **Optimal Synergy (C0 + Thesis Guard)** | **12% Tax** | 53 | 37.7% | 3.26 | 5,814.96 PLN | -4.49 PLN | 9,995.51 PLN | -0.08% | -0.04% | 0.0%/yr | 1.00 | 16.40% | +9.43% | 62.3% |

---

## 4. Key Takeaways from the OddsPortal Open-vs-Close Simulations

1. **Consistent Multi-Year Edge on International Books (0% Tax):**  
   Across 11,609 matches and 5 seasons, taking early opening lines with the **Shrunk Hybrid ($\alpha=0.50$)** produced a **+10.15% Yield** and compounded a 10,000 PLN bankroll into **62,579.30 PLN (+42.5% annualized CAGR)**.
2. **The Power of the Thesis Guard on Canonical Data:**  
   On the 2024–2026 canonical benchmark, requiring the Thesis Model and C0 to agree within 10% boosted the Yield to **+13.92%**, growing the fund to **21,267.34 PLN (+112.67% ROI)** in 2.2 years.
3. **Closing Line Value is Robust Over Thousands of Bets:**  
   - In the 0% tax multi-year run, **73.6% of bets beat the closing line** (Mean Raw CLV: **+6.35%**).
   - In the 12% tax multi-year run, **87.4% of bets beat the closing line** (Mean Raw CLV: **+21.49%**).
   This confirms that the strategy is not curve-fitted to a small 50-bet sample: across 3,500+ bets over 5 years, early line selection consistently outperforms the market close.
