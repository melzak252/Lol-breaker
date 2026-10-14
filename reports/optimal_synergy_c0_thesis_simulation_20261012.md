# Comprehensive Financial Audit & Simulation: Standalone vs. Hybrid vs. Optimal Synergy Strategy

**Date:** 2026-10-12  
**Dataset:** 2026 Production Scrape Archive (`production_rows.jsonl.gz`, May 28 – September 30, 2026, 101 Days)  
**Starting Capital:** 10,000.00 PLN  
**Staking Method:** Fractional 1/4 Kelly ($f^* = \frac{p \cdot b - (1-p)}{b}$, capped at 150.00 PLN, min 20.00 PLN)  
**Tax Environment:** 12% Polish Statutory Turnover Tax on Gross Payouts (`payout = stake * odds * 0.88`)

---

## 1. Precise Financial Definitions (Yield vs. ROI)

In quantitative finance and sports betting econometrics, **Yield** and **ROI** measure two distinct aspects of performance:

| Financial Metric | Meaning | Exact Formula | Practical Interpretation |
|---|---|:---:|---|
| **Yield** | **Return on Turnover** | $\frac{\text{Net Profit}}{\text{Total Capital Staked}}$ | Measures **betting edge / pricing efficiency**. How much profit is extracted per 100 PLN wagered across the slips. |
| **ROI** | **Return on Investment** | $\frac{\text{Net Profit}}{\text{Starting Bankroll}}$ | Measures **portfolio capital growth**. How much the total initial fund of 10,000 PLN expanded over the period. |
| **Annualized ROI** | **Compounded Annual Growth** | $(1 + \text{ROI})^{\frac{365}{\text{Days}}} - 1$ | Annualized capital growth rate based on 101 active operational days. |
| **Raw CLV** | **Closing Line Value** | $\frac{\text{odds}_{\text{taken}}}{\text{odds}_{\text{closing, fair}}} - 1$ | Measures price-beating ability against sharp, devigged closing consensus. |
| **Net CLV** | **Post-Tax Expectation** | $(\text{odds}_{\text{taken}} \times 0.88 \times p_{\text{closing, fair}}) - 1$ | Mathematical expectation after the 12% turnover tax is deducted. |

---

## 2. Head-to-Head Strategy Recalculation (5 Models Tested)

All models were evaluated on the same 603 matches from the 2026 Summer Split using identical execution rules (Net EV $\in [5\%, 25\%]$, Odds $\le 4.00$, 1/4 Kelly staking):

| Strategy Variant | Bets | Win Rate | Total Staked (Turnover) | Net Profit (Post-Tax) | Yield (Return on Turnover) | ROI (Capital Growth) | Annualized ROI (CAGR) | Profit Factor | Max Drawdown | Mean Raw CLV | Positive CLV Rate |
|---|:---:|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Standalone Thesis (039)** | 122 | 48.4% | 17,757.93 PLN | -1,751.54 PLN | **-9.86%** | **-17.52%** | -50.1% | 0.81 | 26.39% | +1.64% | 48.4% |
| **2. Standalone C0 Model** | 147 | 46.9% | 21,275.33 PLN | -672.10 PLN | **-3.16%** | **-6.72%** | -22.2% | 0.94 | 25.33% | +0.72% | 48.3% |
| **3. Shrunk Thesis Hybrid** | 46 | 47.8% | 6,460.01 PLN | -189.05 PLN | **-2.93%** | **-1.89%** | -6.7% | 0.94 | 12.29% | +18.43% | **97.8%** |
| **4. Shrunk C0 Hybrid (No Veto)** | 59 | 50.8% | 8,353.83 PLN | +965.99 PLN | **+11.56%** | **+9.66%** | +39.5% | 1.24 | 13.39% | +18.37% | 84.7% |
| 🏆 **5. Optimal Synergy (C0 + Thesis Guard)** | **50** | **58.0%** | **7,194.04 PLN** | **+1,711.67 PLN** | **+23.79%** | **+17.12%** | **+77.0%** | **1.57** | **8.21%** | **+19.13%** | **90.0%** |

---

## 3. Why the Optimal Synergy Strategy Outperforms

1. **Standalone Models Fail Against the 12% Tax:**  
   Both the Standalone Thesis Model (**Yield: -9.86%**, **ROI: -17.52%**) and Standalone C0 (**Yield: -3.16%**, **ROI: -6.72%**) lose money. They suffer from the Optimizer's Curse, betting too frequently (122 to 147 bets) on minor model-market disagreements (+1% to +2% raw CLV) that are completely wiped out by the 12% tax.
2. **C0 Shrunk Hybrid Generates Positive Yield (+11.56%):**  
   Anchoring C0 to the market via Bayesian logit shrinkage ($\alpha=0.50$) prunes low-conviction noise, boosting Yield to **+11.56%** and capital growth (ROI) to **+9.66%**.
3. **The Thesis Guard Acts as an Asymmetric Error Filter:**  
   When the Thesis Model vetoes C0 bets where disagreement exceeds 10% ($|p_{\text{C0}} - p_{\text{Thesis}}| > 0.10$), it removes 9 toxic bets that had an **11.1% win rate (1 win, 8 losses)** and a **-67.7% loss rate**.
   - **Win Rate Jumps:** 50.8% $\rightarrow$ **58.0%**
   - **Yield More than Doubles:** +11.56% $\rightarrow$ **+23.79%**
   - **Bankroll ROI Nearly Doubles:** +9.66% $\rightarrow$ **+17.12%**
   - **Max Drawdown Compresses:** 13.39% $\rightarrow$ **8.21%**

---

## 4. Month-by-Month Financial Progression (Optimal Strategy)

Across the 101 operational days (May 30 to September 8, 2026):

| Month | Bets Placed | Wins | Win Rate (%) | Turnover Staked (PLN) | Net Profit (PLN) | Yield (Return on Turnover) | ROI Contribution (% of 10k) | Month-End Bankroll (PLN) |
|---|:---:|:---:|:---:|---:|---:|:---:|:---:|:---:|
| **2026-05** | 2 | 2 | 100.0% | 300.00 PLN | +525.00 PLN | **+175.00%** | +5.25% | 10,525.00 PLN |
| **2026-06** | 6 | 4 | 66.7% | 884.81 PLN | +260.95 PLN | **+29.49%** | +2.61% | 10,785.95 PLN |
| **2026-07** | 20 | 12 | 60.0% | 2,818.94 PLN | +541.00 PLN | **+19.19%** | +5.41% | 11,326.95 PLN |
| **2026-08** | 18 | 8 | 44.4% | 2,569.40 PLN | -17.38 PLN | **-0.68%** | -0.17% | 11,309.57 PLN |
| **2026-09** | 4 | 3 | 75.0% | 600.00 PLN | +405.84 PLN | **+67.64%** | +4.06% | **11,715.41 PLN** |
| **TOTAL** | **50** | **29** | **58.0%** | **7,194.04 PLN** | **+1,711.67 PLN** | **+23.79%** | **+17.12%** | **11,715.41 PLN** |

---

## 5. Closing Line Value (CLV) & Price Execution

- **Average Odds Taken:** **2.53**
- **Average Closing Odds (Same Bookmaker):** **2.06** (lines shortened significantly after our taken bets)
- **Mean Raw CLV:** **+19.13%**
- **Median Raw CLV:** **+21.13%**
- **Mean Net CLV (After 12% Tax):** **+4.83%**
- **Positive Raw CLV Rate:** **90.0%** (45 of 50 bets beat the sharp closing price)
- **Positive Net CLV Rate:** **66.0%** (33 of 50 bets beat the closing price even after absorbing the 12% tax)

---

## 6. Takeaway & Practical Summary

- **Yield = +23.79%:** For every 100 PLN staked, the strategy generated **23.79 PLN of net profit** after tax.
- **ROI = +17.12%:** The total 10,000 PLN fund grew by **1,711.67 PLN** over 3.5 months, representing an **annualized return (CAGR) of +77.0%**.
- **Risk Profile:** By capping stakes at 150 PLN (1/4 Kelly) and enforcing the Thesis Model veto, max historical drawdown was restricted to **8.21%**, ensuring capital preservation.
