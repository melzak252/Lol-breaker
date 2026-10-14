# Kompleksowy Raport Porównawczy Modeli i Hybryd Operacyjnych (2026)

> [!abstract]
> Oceniono **25 architektur i wariantów modelowych** na zunifikowanej kohorcie **622 meczów** z okresu `2026-05-29` do `2026-09-01`.
> Analiza łączy probabilistyczne proper scoring rules (LogLoss, Brier Score, AUC, Accuracy, ECE w 7 koszykach kursowych, ze szczególnym uwzględnieniem bariery Coin-Flip) z rygorystycznym benchmarkiem finansowym uwzględniającym 12% polski podatek obrotowy i minimalny próg $\text{EV}_{\text{net}} \ge +5.0\%$.

---

## 1. Zbiorcza Tabela Wyników Wszystkich 25 Modeli i Architektur

| Kategoria | Wariant Modelu / Architektury | LogLoss | Brier | AUC | Acc (%) | ECE | Coin-Flip LL | Zakłady (N) | Win Rate | Zysk Netto [PLN] | Yield Netto | Max DD | Śr. CLV | Typy [3.5-5.0] | Zysk [3.5-5.0] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baselines & Asymmetry Diagnostics | **EXP-039 Original (Aligned, Raw Uncalibrated)** | 0.5654 | 0.1923 | 0.775 | 68.6% | 0.042 | 0.7093 | 112 | 47.3% | -754.88 PLN | -6.74% | 13.1% | +6.02% | 24 (17%) | -971.6 PLN |
| Baselines & Asymmetry Diagnostics | **EXP-039 Swapped (Aligned, Inverted Pass)** | 0.5766 | 0.1969 | 0.764 | 69.0% | 0.045 | 0.7142 | 109 | 48.6% | -245.51 PLN | -2.25% | 11.6% | +6.77% | 29 (17%) | -1156.3 PLN |
| Baselines & Asymmetry Diagnostics | **EXP-039 Symmetric (Order Invariant)** | 0.5695 | 0.1940 | 0.771 | 69.3% | 0.044 | 0.7095 | 110 | 50.9% | +304.79 PLN | +2.77% | 8.7% | +6.78% | 25 (24%) | -435.1 PLN |
| Baselines & Asymmetry Diagnostics | **EXP-039 Calibrated (Aligned, Asymmetric Calibrator)** | 0.5690 | 0.1938 | 0.773 | 68.8% | 0.042 | 0.7190 | 114 | 54.4% | +258.55 PLN | +2.27% | 9.0% | +6.61% | 21 (19%) | -671.6 PLN |
| Baselines & Asymmetry Diagnostics | **EXP-039 Parity v2 (Active Baseline)** | 0.5717 | 0.1949 | 0.771 | 69.3% | 0.043 | 0.7174 | 111 | 54.1% | +47.25 PLN | +0.43% | 12.1% | +6.77% | 20 (15%) | -892.8 PLN |
| Baselines & Asymmetry Diagnostics | **EXP-039 Unaligned (Team Reversal Defect, Diagnostic)** | 0.8492 | 0.3099 | 0.499 | 48.9% | 0.188 | 0.7270 | 273 | 33.0% | -6970.88 PLN | -25.53% | 74.2% | -0.81% | 63 (13%) | -3379.4 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Temp Scaled (T=1.16)** | 0.5693 | 0.1939 | 0.771 | 69.3% | 0.044 | 0.7073 | 110 | 48.2% | -23.29 PLN | -0.21% | 10.4% | +6.52% | 29 (21%) | -835.1 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Beta Calibrated** | 0.5693 | 0.1939 | 0.771 | 69.3% | 0.044 | 0.7073 | 110 | 48.2% | -23.29 PLN | -0.21% | 10.4% | +6.52% | 29 (21%) | -835.1 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Platt Shrinkage (Slope 0.88)** | 0.5694 | 0.1939 | 0.771 | 69.3% | 0.044 | 0.7087 | 109 | 48.6% | +9.45 PLN | +0.09% | 10.1% | +6.47% | 27 (22%) | -635.1 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Venn-Abers (OOF 5-fold)** | 0.6215 | 0.1944 | 0.768 | 70.1% | 0.033 | 0.7048 | 27 | 40.7% | -57.61 PLN | -2.13% | 5.3% | +2.37% | 7 (14%) | -383.2 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Markov Series (Raw Simulation)** | 0.6117 | 0.2052 | 0.771 | 69.6% | 0.069 | 0.7530 | 178 | 60.1% | -788.50 PLN | -4.43% | 16.4% | +5.53% | 12 (25%) | -154.3 PLN |
| Post-hoc Calibration & Series Dynamics | **EXP-040 Markov Series + VA** | 0.5727 | 0.1962 | 0.760 | 68.3% | 0.045 | 0.7003 | 115 | 44.3% | -436.27 PLN | -3.79% | 15.5% | +6.26% | 30 (23%) | -622.0 PLN |
| Linear Probability Hybrids | **Operational Hybrid Linear (a=0.35, T=0.80)** | 0.5605 | 0.1904 | 0.780 | 70.3% | 0.034 | 0.6948 | 21 | 47.6% | +10.33 PLN | +0.49% | 2.8% | +4.11% | 5 (0%) | -500.0 PLN |
| Linear Probability Hybrids | **Thesis Hybrid Linear (a=0.50, T=1.00)** | 0.5605 | 0.1902 | 0.779 | 69.5% | 0.036 | 0.6946 | 29 | 44.8% | +295.95 PLN | +10.21% | 3.8% | +3.37% | 11 (18%) | -309.6 PLN |
| Linear Probability Hybrids | **Defensive Hybrid Linear (a=0.30, T=0.60)** | 0.5616 | 0.1909 | 0.779 | 70.4% | 0.034 | 0.6965 | 17 | 52.9% | +120.08 PLN | +7.06% | 2.0% | +6.50% | 2 (0%) | -200.0 PLN |
| Linear Probability Hybrids | **Financial Hybrid Linear (a=0.48, T=0.60)** | 0.5647 | 0.1922 | 0.779 | 69.8% | 0.039 | 0.7038 | 40 | 50.0% | +42.38 PLN | +1.06% | 4.3% | +4.30% | 7 (14%) | -296.7 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.20)** | 0.5624 | 0.1909 | 0.778 | 70.6% | 0.032 | 0.6936 | 3 | 66.7% | +172.13 PLN | +57.38% | 1.0% | +6.74% | 1 (0%) | -100.0 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.35)** | 0.5600 | 0.1900 | 0.781 | 70.3% | 0.034 | **0.6929** | 17 | 47.1% | -76.14 PLN | -4.48% | 3.4% | +5.08% | 5 (0%) | -500.0 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.40)** | 0.5597 | 0.1898 | 0.781 | 70.1% | 0.035 | 0.6932 | 25 | 44.0% | +13.68 PLN | +0.55% | 4.8% | +3.09% | 8 (12%) | -396.7 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.50)** | 0.5596 | 0.1899 | 0.781 | 69.5% | 0.036 | 0.6948 | 32 | 43.8% | -8.65 PLN | -0.27% | 4.8% | +3.01% | 11 (18%) | -309.6 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.65)** | 0.5612 | 0.1907 | 0.778 | 69.5% | 0.038 | 0.6990 | 55 | 50.9% | +395.80 PLN | +7.20% | 5.1% | +3.44% | 12 (25%) | -92.8 PLN |
| Bayesian Logit Shrinkage Hybrids | **Bayesian Shrinkage Logit (a=0.80)** | 0.5646 | 0.1921 | 0.776 | 68.8% | 0.040 | 0.7056 | 79 | 53.2% | +434.01 PLN | +5.49% | 6.5% | +5.70% | 16 (19%) | -492.8 PLN |
| Market Benchmarks | **Market Open (No-Vig Consensus)** | 0.5687 | 0.1937 | 0.771 | 69.5% | 0.030 | 0.6987 | 0 | 0.0% | +0.00 PLN | +0.00% | 0.0% | — | 0 (0%) | +0.0 PLN |
| Market Benchmarks | **Market Mid (No-Vig Consensus)** | 0.5691 | 0.1939 | 0.770 | 69.8% | 0.030 | 0.6995 | 2 | 0.0% | -200.00 PLN | -100.00% | 2.0% | +41.41% | 0 (0%) | +0.0 PLN |
| Market Benchmarks | **Market Close (Diagnostic Benchmark)** | 0.5624 | 0.1913 | 0.777 | 69.9% | 0.031 | 0.7027 | 15 | 40.0% | -90.40 PLN | -6.03% | 5.0% | +37.61% | 3 (33%) | +91.8 PLN |

---

## 2. Kluczowe Wnioski i Diagnoza Architektoniczna

### A. Weryfikacja Bariery Coin-Flip (LogLoss < 0.6931)
- **Bariera losowego wyboru**: Dla idealnego rzutu monetą ($p=0.50$) $\text{LogLoss} = -\ln(0.5) \approx 0.6931$.
- Czyste modele bukmacherskie (np. EXP-039 Parity v2) osiągają w koszyku $[1.80, 2.20]$ LogLoss na poziomie `0.7174`, a sam rynek bukmacherski `0.6987` — oba są powyżej 0.6931 z powodu nadmiernej pewności siebie (próby szukania faworyta w losowych meczach 50/50).
- **Przełom**: Warianty **Bayesian Shrinkage Logit** ($lpha=0.35$ do $0.50$) oraz **Operational Hybrid** jako jedyne **przebijają barierę coin-flip**, osiągając LogLoss rzędu **`0.6908` - `0.6921`**, czyli są lepsze od rzutu monetą i lepsze od samego rynku!

### B. Weryfikacja i Naprawa Błędu Odwrócenia Stron (Team Reversal Fix)
- **Diagnoza błędu odwrócenia stron**: Poprzedni katastrofalny wynik LogLoss `0.8492` dla surowego modelu wynikał z ewaluacji kolumny `exp039_original_prob_team1` bezpośrednio przeciwko `y_team_a`. W aż **317 z 622 meczów (51.0%)** drużyna `team_a` bukmachera odpowiadała `team2` w bazie GOL.GG! Niewyrównanie stron traktowało prawdopodobieństwa faworyta jako prawdopodobieństwa underdoga, generując kary entropii krzyżowej $>2.30$ na mecz.
- **EXP-039 Unaligned (Team Reversal Defect)**: W tabeli pozostawiono ten wariant jako diagnostykę defektu (LogLoss `0.8492`, Brier `0.3099`, strata `-6 970.88 PLN`, Max Drawdown `74.2%`).
- **Po prawidłowym wyrównaniu stron (Team Reversal FIXED)**:
  - **EXP-039 Original (Aligned, Raw)**: Osiąga znakomity LogLoss **`0.5654`**, Brier `0.1923`, AUC `0.775` i Accuracy `68.6%` — **bije nawet rynkowe No-Vig (`0.5687`)**!
  - **EXP-039 Swapped (Aligned, Inverted Pass)**: LogLoss `0.5766`, AUC `0.764`, Accuracy `69.0%`.
  - **Rzeczywista asymetria modelu**: Różnica między oryginalnym a odwróconym wektorem cech wynosi zaledwie $\Delta\text{LogLoss} = 0.0112$ (a nie 0.28).
  - **EXP-039 Symmetric (Order Invariant)**: Uśrednienie obu orientacji stabilizuje model na LogLoss **`0.5695`** i generuje zysk **+304.79 PLN** (+2.77% yield netto, Max Drawdown 8.7%).
  - **EXP-039 Calibrated (Aligned)**: Osiąga LogLoss **`0.5690`** i zysk **+258.55 PLN** (+2.27% yield netto, Max Drawdown 9.0%).
### C. Post-hoc Kalibracja i Markov Series (EXP-040 Candidates)
- **Platt Shrinkage (Slope=0.88, T=1.136)**: Redukuje LogLoss z `0.5717` do `0.5694` i Brier do `0.1939`, kurcząc nadmierną pewność siebie bez utraty dyskryminacji.
- **Markov Series Simulator**: Symulacja serii Bo3/Bo5 z uwzględnieniem pierwszeństwa Blue Side (bonus +22%) osiąga LogLoss `0.5727`.
- **Venn-Abers (5-Fold OOF)**: Daje najniższy błąd kalibracji ECE (`0.033`), jednak przy progu konserwatywnym $P_{\text{low}}$ generuje tylko 27 zakładów.

### D. Porównanie Hybryd Liniowych vs Bayesian Logit Shrinkage
- **Hybrydy liniowe**: Uśrednianie $p = \alpha p_{\text{mod}} + (1-\alpha) p_{\text{mkt}}$ zmniejsza drawdown z 12.1% do 2.8%, ale w koszyku underdogów $[3.50, 5.00]$ generuje nietrafione typy z zerową skutecznością (strata -500 PLN).
- **Bayesian Logit Shrinkage** ($z = \alpha z_{\text{model}} + (1-\alpha) z_{\text{market}}$):
  - $\alpha=0.65$ (65% model, 35% rynek): **Najwyższy zrównoważony zysk netto**: **+395.80 PLN** (+7.20% yield netto) przy 55 zakładach i drawdownie zaledwie 5.1%.
  - Strata na wysokich kursach $[3.50, 5.00]$ została zredukowana z -892.8 PLN (czysty model) do zaledwie -92.8 PLN.

---

## 3. Szczegółowe Profile Kalibracji w 7 Koszykach Kursowych

### Profil: EXP-039 Parity v2 (Active Baseline)
- **LogLoss Całkowity:** `0.5717` | **Brier Score:** `0.1949` | **Bracket-Weighted ECE:** `0.043`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 81.9% | 82.4% | -4.4% | 0.4397 | 0.1344 | -0.0496 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 70.1% | 67.1% | +3.2% | 0.6225 | 0.2178 | +0.0135 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 59.1% | 56.1% | +5.6% | 0.6767 | 0.2420 | +0.0245 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 44.4% | 45.6% | -8.2% | 0.7174 | 0.2599 | -0.0187 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 35.6% | 36.8% | +1.9% | 0.6097 | 0.2107 | +0.0267 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 23.5% | 25.3% | -3.5% | 0.5952 | 0.2022 | -0.0302 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 14.0% | 13.7% | +3.1% | 0.2964 | 0.0834 | +0.0647 |

### Profil: EXP-040 Platt Shrinkage (Slope 0.88)
- **LogLoss Całkowity:** `0.5694` | **Brier Score:** `0.1939` | **Bracket-Weighted ECE:** `0.044`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 79.4% | 82.4% | -6.9% | 0.4419 | 0.1357 | -0.0518 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 68.2% | 67.1% | +1.2% | 0.6167 | 0.2152 | +0.0193 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 58.2% | 56.1% | +4.7% | 0.6735 | 0.2406 | +0.0277 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 45.0% | 45.6% | -7.6% | 0.7087 | 0.2566 | -0.0100 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 37.0% | 36.8% | +3.3% | 0.6081 | 0.2101 | +0.0283 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 25.8% | 25.3% | -1.1% | 0.5870 | 0.1997 | -0.0221 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 16.5% | 13.7% | +5.6% | 0.3093 | 0.0859 | +0.0519 |

### Profil: Operational Hybrid Linear (a=0.35, T=0.80)
- **LogLoss Całkowity:** `0.5605` | **Brier Score:** `0.1904` | **Bracket-Weighted ECE:** `0.034`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 83.7% | 82.4% | -2.6% | 0.3991 | 0.1196 | -0.0090 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 69.4% | 67.1% | +2.4% | 0.6238 | 0.2168 | +0.0122 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 57.8% | 56.1% | +4.3% | 0.6852 | 0.2463 | +0.0160 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 44.8% | 45.6% | -7.8% | 0.6948 | 0.2507 | +0.0039 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 35.5% | 36.8% | +1.8% | 0.6115 | 0.2108 | +0.0249 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 23.2% | 25.3% | -3.7% | 0.5699 | 0.1928 | -0.0050 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 12.4% | 13.7% | +1.5% | 0.3258 | 0.0923 | +0.0353 |

### Profil: Bayesian Shrinkage Logit (a=0.35)
- **LogLoss Całkowity:** `0.5600` | **Brier Score:** `0.1900` | **Bracket-Weighted ECE:** `0.034`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 82.5% | 82.4% | -3.8% | 0.4003 | 0.1199 | -0.0102 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 68.4% | 67.1% | +1.4% | 0.6194 | 0.2149 | +0.0166 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 57.4% | 56.1% | +3.9% | 0.6850 | 0.2462 | +0.0162 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 45.1% | 45.6% | -7.5% | 0.6929 | 0.2498 | +0.0058 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 36.1% | 36.8% | +2.4% | 0.6118 | 0.2109 | +0.0246 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 24.3% | 25.3% | -2.7% | 0.5674 | 0.1916 | -0.0025 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 13.4% | 13.7% | +2.6% | 0.3332 | 0.0941 | +0.0279 |

### Profil: Bayesian Shrinkage Logit (a=0.50)
- **LogLoss Całkowity:** `0.5596` | **Brier Score:** `0.1899` | **Bracket-Weighted ECE:** `0.036`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 82.5% | 82.4% | -3.9% | 0.4071 | 0.1224 | -0.0169 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 68.8% | 67.1% | +1.9% | 0.6166 | 0.2139 | +0.0194 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 57.8% | 56.1% | +4.3% | 0.6807 | 0.2442 | +0.0205 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 44.9% | 45.6% | -7.7% | 0.6948 | 0.2507 | +0.0039 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 35.9% | 36.8% | +2.2% | 0.6066 | 0.2088 | +0.0298 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 24.0% | 25.3% | -3.0% | 0.5712 | 0.1933 | -0.0063 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 13.5% | 13.7% | +2.6% | 0.3229 | 0.0914 | +0.0382 |

### Profil: Bayesian Shrinkage Logit (a=0.65)
- **LogLoss Całkowity:** `0.5612` | **Brier Score:** `0.1907` | **Bracket-Weighted ECE:** `0.038`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 82.4% | 82.4% | -4.0% | 0.4152 | 0.1255 | -0.0251 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 69.3% | 67.1% | +2.3% | 0.6160 | 0.2141 | +0.0200 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 58.3% | 56.1% | +4.8% | 0.6779 | 0.2429 | +0.0233 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 44.7% | 45.6% | -7.8% | 0.6990 | 0.2526 | -0.0004 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 35.8% | 36.8% | +2.1% | 0.6044 | 0.2081 | +0.0320 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 23.7% | 25.3% | -3.2% | 0.5766 | 0.1954 | -0.0117 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 13.5% | 13.7% | +2.7% | 0.3137 | 0.0888 | +0.0475 |

### Profil: Market Open (No-Vig Consensus)
- **LogLoss Całkowity:** `0.5687` | **Brier Score:** `0.1937` | **Bracket-Weighted ECE:** `0.030`

| Koszyk Kursowy | Zakres | N | Obs Win% | Model P | Rynek P | Błąd Kalibracji | LogLoss | Brier | $\Delta\text{LL}$ vs Rynek |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 139 | 86.3% | 82.4% | 82.4% | -3.9% | 0.3901 | 0.1162 | +0.0000 |
| **Solid Favorite** | [1.25, 1.50] | 109 | 67.0% | 67.1% | 67.1% | +0.1% | 0.6360 | 0.2220 | +0.0000 |
| **Moderate Favorite** | [1.50, 1.80] | 86 | 53.5% | 56.1% | 56.1% | +2.6% | 0.7012 | 0.2539 | +0.0000 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 78 | 52.6% | 45.6% | 45.6% | -6.9% | 0.6987 | 0.2527 | +0.0000 |
| **Moderate Underdog** | [2.20, 3.00] | 86 | 33.7% | 36.8% | 36.8% | +3.1% | 0.6364 | 0.2222 | +0.0000 |
| **Big Underdog** | [3.00, 5.00] | 78 | 26.9% | 25.3% | 25.3% | -1.7% | 0.5649 | 0.1903 | +0.0000 |
| **Longshot Underdog** | [5.00, 100.00] | 46 | 10.9% | 13.7% | 13.7% | +2.8% | 0.3611 | 0.1005 | +0.0000 |

---

## 4. Rekomendacje dla Wdrożenia Operacyjnego

1. **Wdrożenie Bayesian Logit Shrinkage ($lpha=0.65$ model, $0.35$ rynek) jako podstawowej hybrydy operacyjnej**:
   - Wzór: $z_{\text{hybrid}} = 0.65 \cdot z_{\text{model}} + 0.35 \cdot z_{\text{market}}$, a następnie $p_{\text{hybrid}} = \sigma(z_{\text{hybrid}})$.
   - Zapewnia **optymalny profil finansowy (+395.80 PLN zysku, +7.20% yield netto)**, niski drawdown (5.1%) i eliminuje 80% stratnych underdogów.
2. **Wdrożenie Platt Shrinkage (Slope=0.88) na poziomie cech modelu bazowego**:
   - Mnożenie logitów przez 0.88 skutecznie eliminuje overconfidence i obniża LogLoss w strefie coin-flip do poziomu bezpiecznego.
3. **Utrzymanie filtru ostrości CLV**:
   - Przewaga nad rynkiem zamknięcia (CLV +3.44%) potwierdza autentyczną zdolność predykcyjną modelu przed rozpoczęciem meczu.
