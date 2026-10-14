# Benchmark Rynkowy: Ewaluacja Modeli Predykcyjnych przeciwko Kwotowaniom Pinnacle (2026)

**Data ewaluacji:** 2026-09-07  
**Kohorta testowa:** $N = 447$ meczów League of Legends z okresu `2026-05-29` do `2026-09-01`  
**Źródła danych:**
- Kwotowania przedmeczowe Pinnacle: `data/oddspapi_lol_2026_model_audit/selected_pre_match_quotes.csv`
- Konsensus rynku polskiego (STS, Fortuna, Superbet, Betclic) i predykcje modeli: `reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv`
- Czysty model produkcyjny: `EXP-039 Symmetric` oraz `EXP-081 Siamese Series`

---

## 1. Wprowadzenie i Cel Badania

Pinnacle jest powszechnie uznawany za najostrzejszy rynek bukmacherski na świecie (sharp market maker), charakteryzujący się wysokimi limitami stawek, niską marżą bukmacherską oraz dynamicznym reagowaniem na przepływ kapitału graczy profesjonalnych.

Celem niniejszego audytu jest:
1. **Oszacowanie luki informacyjnej (Informational Gap)** pomiędzy czystym modelem statystyczno-ratingowym a rynkiem zamknięcia Pinnacle na identycznej próbie meczów.
2. **Porównanie efektywności rynkowej**: zestawienie marży i jakości probabilistycznej Pinnacle z konsensusem licencjonowanych bukmacherów w Polsce.
3. **Analiza sporów (Disagreement Analysis)**: zbadanie, kto ma rację w sytuacjach, gdy model i Pinnacle wskazują odmiennego faworyta.
4. **Weryfikacja hipotezy o zyskowności przeciwko Pinnacle**: empiryczne sprawdzenie, czy przewagi wyliczane przez model ($\text{EV} > 0$) są rzeczywistym zyskiem, czy artefaktem błędu modelu.

---

## 2. Zbiorcze Wyniki Probabilistyczne na Próbie $N = 447$ Meczów

Zastosowano proper scoring rules (LogLoss, Brier Score, ROC-AUC, Accuracy, ECE oraz stratę w strefie Coin-Flip $[45\%, 55\%]$):

| Predyktor / Benchmark | Typ Źródła | LogLoss | Brier Score | ROC-AUC | Accuracy | ECE (10 bins) | Coin-Flip LogLoss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pinnacle No-Vig Fair** | Rynek ostry (Closing Line) | **0.5357** | **0.1802** | **0.8073** | **72.93%** | 0.0419 | **0.6820** |
| **Pinnacle Normalized Implied** | Surowe kursy Pinnacle | 0.5357 | 0.1802 | 0.8072 | 72.93% | 0.0419 | 0.6820 |
| **Hybryda Bayesowska ($\alpha=0.35$ Pinnacle)** | Model + Pinnacle ($T=0.80$) | **0.5407** | **0.1816** | **0.8054** | **71.36%** | 0.0415 | **0.6921** |
| **Hybryda Liniowa (Model + Pinnacle)** | $0.35 p_{\text{mod}} + 0.65 p_{\text{pin}}$ | 0.5428 | 0.1826 | 0.8025 | 71.81% | 0.0330 | 0.6935 |
| **Hybryda Operacyjna (Model + PL Open)** | Model + Konsensus PL ($0.35/0.65$) | 0.5616 | 0.1906 | 0.7832 | 70.47% | 0.0447 | 0.7211 |
| **Konsensus Rynku Polskiego (Close)** | Średnia no-vig zamknięcia PL | 0.5620 | 0.1908 | 0.7827 | 70.47% | **0.0275** | 0.7166 |
| **Konsensus Rynku Polskiego (Open)** | Średnia no-vig otwarcia PL | 0.5663 | 0.1926 | 0.7763 | 70.02% | 0.0449 | 0.7257 |
| **EXP-039 Symmetric** | Czysty model (antysymetryczny) | 0.5724 | 0.1951 | 0.7710 | 69.35% | 0.0409 | 0.7188 |
| **EXP-039 Parity v2** | Czysty model bazowy (praca inż.) | 0.5744 | 0.1959 | 0.7710 | 69.35% | 0.0564 | 0.7236 |

---

## 3. Testy Istotności Statystycznej (Paired Bootstrap, $B = 5\,000$ Replikacji)

Obliczono różnice strat $\Delta \text{LogLoss}_i = \mathcal{L}_{A, i} - \mathcal{L}_{B, i}$ dla każdego meczu z osobna:

### A. Model vs Pinnacle
$$\Delta \text{LogLoss}(\text{Model} - \text{Pinnacle}) = +\mathbf{0.0367} \quad [95\%\text{ CI: } +0.0164, \, +0.0574], \quad p = 0.0004$$
*Przedział ufności znajduje się w całości powyżej zera ($p < 0.001$). Pinnacle jest bezwzględnie i istotnie statystycznie dokładniejszy niż czysty model sportowy.*

### B. Konsensus Polski (Close) vs Pinnacle
$$\Delta \text{LogLoss}(\text{PL Close} - \text{Pinnacle}) = +\mathbf{0.0262} \quad [95\%\text{ CI: } +0.0118, \, +0.0406], \quad p < 0.0001$$
*Pinnacle wykazuje znaczącą przewagę informacyjną nad konsensusem polskich bukmacherów detalicznych. Zysk informacyjny wynosi ponad $2.6$ punktu LogLoss, a trafność jest wyższa o niemal $2.5$ punktu procentowego.*

### C. Konsensus Polski (Open) vs Model
$$\Delta \text{LogLoss}(\text{PL Open} - \text{Model}) = -0.0061 \quad [95\%\text{ CI: } -0.0278, \, +0.0151], \quad p = 0.57$$
*Brak istotnej różnicy statystycznej pomiędzy kursem otwarcia w Polsce a czystym modelem. Model jest równorzędny z cenami otwarcia bukmacherów detalicznych.*

---

## 4. Analiza Sporów (Disagreement Analysis)

### A. Kto ma rację w meczach o odmiennej ocenie faworyta?
W **46 na 447 meczów (10.3%)** model i Pinnacle wskazały innego faworyta ($P \ge 0.50$):
- **Pinnacle trafił w 67.4% przypadków** (31 wygranych na 46 meczów), osiągając LogLoss **0.5941**.
- **Model trafił w 32.6% przypadków** (15 wygranych na 46 meczów), osiągając LogLoss **0.7701**.
- W przypadku bezpośredniego sporu Pinnacle jest ponad dwukrotnie skuteczniejszy.

### B. Rozbieżności rzędu $\ge 10$ punktów procentowych
- Wystąpiły w **110 meczach (24.6%)**.
- Średnia bezwzględna różnica pomiędzy prawdopodobieństwem modelu a Pinnacle wynosi **$7.90$ p.p.**
- Korelacja liniowa Pearsona pomiędzy predykcją modelu a kursem Pinnacle wynosi **$r = 0.8911$**.

---

## 5. Wyniki w Podziale na Format Serii (Best-of)

| Format | Próba (N) | Pinnacle LogLoss | Model LogLoss | PL Close LogLoss | Pinnacle Acc | Model Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bo1** | 79 | **0.5436** | 0.5906 | 0.5917 | **72.2%** | 65.8% |
| **Bo3** | 275 | **0.5410** | 0.5722 | 0.5626 | **73.5%** | 70.9% |
| **Bo5** | 93 | **0.5134** | 0.5577 | 0.5347 | **72.0%** | 67.7% |

Przewaga rynku ostrego jest widoczna we wszystkich formatach rozgrywek, osiągając najwyższą precyzję w długich seriach Bo5 ($\text{LogLoss} = 0.5134$).

---

## 6. Marża Bukmacherska i Symulacja Zakładów na Pinnacle

### A. Średnia marża bukmacherska (Vig)
- **Pinnacle**: średnio **$6.41\%$** (mediana $6.01\%$).
- **Polscy bukmacherzy (Open)**: średnio **$8.66\%$** (mediana $8.50\%$).
- **Polscy bukmacherzy (Close)**: średnio **$8.64\%$** (mediana $8.45\%$).
- Pinnacle pobiera marżę o **$2.25$ p.p. niższą**, oferując graczom znacznie wyższe kursy brutto.

### B. Symulacja zakładów płaską stawką przeciwko Pinnacle (bez podatku 12%)
Gdyby gracz zawierał zakłady bezpośrednio na Pinnacle, kierując się dodatnią wartością oczekiwaną z modelu ($\text{EV}_{\text{gross}} = p_{\text{mod}} \cdot \text{Odds}_{\text{pin}} - 1.0 > \text{próg}$):

| Próg $\text{EV}_{\text{gross}}$ | Liczba Zakładów | Skuteczność (Win Rate) | Zrealizowany ROI | Zysk/Strata (PnL) |
| :---: | :---: | :---: | :---: | :---: |
| $\ge 0.0\%$ | 327 | 37.6% | **-14.45%** | -47.25 u |
| $\ge +3.0\%$ | 271 | 35.4% | **-11.91%** | -32.27 u |
| $\ge +5.0\%$ | 243 | 34.2% | **-11.17%** | -27.15 u |
| $\ge +8.0\%$ | 212 | 30.7% | **-14.75%** | -31.27 u |

### Wniosek rynkowy:
Czysty model statystyczny nie posiada dodatniej wartości oczekiwanej wobec kursów Pinnacle. Rozbieżności pomiędzy modelem a rynkiem ostrym są objawem braków informacyjnych modelu (np. brak wiedzy o nagłych zmianach składów, draftach lub motywacji), a nie nieefektywności wyceny Pinnacle.

---

## 7. Rola Pinnacle w Rozwoju Modeli Operacyjnych

Audyt dowodzi, że Pinnacle nie powinien być traktowany jako rywal do "ogrania", lecz jako **zewnętrzne źródło wiedzy (oracle / anchor)**:
1. **Kotwica Bayesowska**: Połączenie predykcji modelu z kwotowaniem Pinnacle redukuje LogLoss do **$0.5407$** (znacznie poniżej konsensusu polskiego $0.5620$).
2. **Destylacja Wiedzy (Knowledge Distillation)**: Użycie prawdopodobieństw Pinnacle jako "soft targets" w procesie douczania sieci neuronowych pozwala na przeniesienie informacji rynkowej do wag modelu.
3. **Wykrywanie anomalii**: Duża rozbieżność $|p_{\text{mod}} - p_{\text{pin}}| \ge 0.10$ powinna działać jako sygnał ostrzegawczy (quarantine flag) blokujący zawieranie zakładów na rynku detalicznym.
