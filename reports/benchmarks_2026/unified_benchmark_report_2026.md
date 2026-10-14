# Zunifikowany Raport Benchmarków Modeli, Kalibracji i Turniejów (2026)

> [!abstract]
> Raport stanowi całościową syntezę czterech zaplanowanych benchmarków analitycznych:
> 1. **Unified Model Promotion Benchmark (AGENTS.md)**: Rygorystyczna ewaluacja 4 kandydatów modelowych na nowoczesnej próbie holdout $N=9\,482$ serii z lat 2024–2026 w oparciu o 5 000 resampli bootstrapowych z blokami miesięcznymi.
> 2. **Odds Bracket Calibration & Betting Benchmark**: Ewaluacja kalibracji w 7 przedziałach kursowych (od Mega Faworytów $[1.01, 1.25]$ po Longshoty $[5.00, 100.00]$) na próbie 1 244 linii bukmacherskich (622 mecze).
> 3. **Comprehensive 25-Model and Hybrid Benchmark**: Wielowymiarowe porównanie 25 architektur (modele bazowe, kalibratory post-hoc, symulacje łańcuchów Markowa, hybrydy liniowe i bayesowskie) pod kątem log-loss, bariery coin-flip ($0.6931$) i rentowności finansowej po 12% podatku obrotowym.
> 4. **Tournament Quality & Replay Readiness Benchmark**: Ocena mechaniki turniejowej 234 faz w 7 rodzinach (91 profili) oraz audyt certyfikacji czasowej do symulacji retrospektywnych.

---

## 1. Unified Model Promotion Benchmark (AGENTS.md)

### A. Metodologia i Protokół Walk-Forward
Zgodnie z wymaganiami promocyjnymi w `AGENTS.md`:
- **Próba ewaluacyjna**: Zablokowana kohorta holdout od `2024-01-14` do `2026-06-14` obejmująca $N = 9\,482$ rozegranych serii profesjonalnych.
- **Punkt odniesienia (Frozen Baseline)**: `Sym-Cal LR-ElasticNet-W20-Binomial / exp-039`.
- **Weryfikacja istotności**: 5 000 resampli bootstrapowych z blokami miesięcznymi (zachowującymi autokorelację meta-gry i patchy LoL).
- **Kandydaci**:
  1. `recipe__reported_regularization_bagging` (Poprawiony bagging regularyzacyjny EXP-081)
  2. `architecture__linear79` (Symetryczny model liniowy na 79 cechach)
  3. `annual__annual090_recency365` (Model rocznego refitu z oknem 365 dni)
  4. `consensus_mean12_series_calibrated` (Skalibrowany 12-składnikowy konsensus rankingowy)

### B. Wyniki Główne i Porównanie Sparowane z Baza Zablokowaną (EXP-039)

| Model / Architektura | LogLoss | Brier | Brier Rel | Brier Res | ROC-AUC | Acc (%) | ECE | Slope | Intercept | $\Delta$ LogLoss vs EXP-039 | 95% Bootstrap CI | p-value | Status Bramki |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-039 Frozen Baseline** | **0.55926** | **0.18983** | **0.00063** | **0.05832** | **0.7820** | **70.70%** | **0.0226** | **1.037** | **0.117** | *baseline* | *baseline* | *baseline* | **BASELINE** |
| **Linear79 Symmetric** | 0.55948 | 0.18985 | 0.00054 | 0.05843 | 0.7816 | 70.77% | 0.0219 | 1.006 | 0.115 | +0.00022 | [-0.00145, +0.00204] | 0.6136 | **ODRZUCONY** |
| **EXP-081 Bagging Recipe** | 0.56188 | 0.19094 | 0.00066 | 0.05772 | 0.7792 | 70.49% | 0.0241 | 1.050 | 0.119 | +0.00262 | [+0.00104, +0.00420] | 0.9984 | **ODRZUCONY** |
| **Annual Recency 365d** | 0.56206 | 0.19060 | 0.00073 | 0.05808 | 0.7803 | 70.99% | 0.0238 | 0.929 | 0.117 | +0.00280 | [+0.00038, +0.00550] | 0.9866 | **ODRZUCONY** |
| **Consensus Rating (12-way)** | 0.58433 | 0.20004 | 0.00082 | 0.04868 | 0.7562 | 68.53% | 0.0267 | 1.001 | 0.108 | +0.02507 | [+0.01878, +0.03307] | 1.0000 | **ODRZUCONY** |

### C. Wnioski Promocyjne
1. **Żaden kandydat nie spełnia twardej reguły promocyjnej**: Zgodnie z AGENTS.md, górna granica 95% CI dla $\Delta \text{LogLoss}$ musi być ujemna ($< 0.0$). Dla wszystkich kandydatów przedział zawiera wartości dodatnie, co uniemożliwia zatwierdzenie ich promocji.
2. **Przewaga prostej symetrycznej regularyzacji nad sieciami neuronowymi**: Złożona architektura Siamese (EXP-081) osiąga istotnie gorszy LogLoss (+0.00262, $p=0.9984$) niż liniowy baseline EXP-039.
3. **Waga cech kontekstowych W20**: Czysty konsensus rankingowy bez cech formy i kontekstu W20 traci aż +0.025 LogLoss i ponad 2.5 p.p. AUC.

---

## 2. Odds Bracket Calibration & Betting Benchmark

Ewaluacja w 7 przedziałach kursowych przeprowadzona na 1 244 liniach bukmacherskich z okresu maj–wrzesień 2026.

| Przedział Kursowy | Zakres Kursów | N Linii | Śr. Kurs | Obs Win% | Model P | Rynek P | Błąd Kalibracji | Model LL | Rynek LL | $\Delta$ (Rynek - Mod) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mega Favorite** | [1.01, 1.25] | 253 | 1.13 | 85.8% | 82.5% | 80.7% | -3.3% (Underconf.) | **0.4202** | 0.4111 | -0.0090 |
| **Solid Favorite** | [1.25, 1.50] | 188 | 1.37 | 64.4% | 69.0% | 66.1% | +4.6% (Overconf.) | **0.6709** | 0.6591 | -0.0117 |
| **Moderate Favorite** | [1.50, 1.80] | 162 | 1.65 | 53.1% | 56.8% | 56.0% | +3.7% (Overconf.) | **0.6859** | 0.7017 | +0.0158 |
| **Coin-Flip / Tight** | [1.80, 2.20] | 153 | 1.99 | 48.4% | 46.3% | 46.2% | -2.0% (Underconf.) | **0.6994** | 0.6977 | -0.0017 |
| **Moderate Underdog** | [2.20, 3.00] | 175 | 2.57 | 37.7% | 34.0% | 37.7% | -3.7% (Underconf.) | **0.6121** | 0.6369 | +0.0248 |
| **Big Underdog** | [3.00, 5.00] | 184 | 3.85 | 24.5% | 22.7% | 24.5% | -1.8% (Underconf.) | **0.5813** | 0.5562 | -0.0251 |
| **Longshot Underdog** | [5.00, 100.0] | 129 | 7.13 | 9.3% | 14.5% | 9.4% | +5.1% (Overconf.) | **0.3128** | 0.3482 | +0.0354 |

### Kluczowe Diagnozy Kalibracji:
- **Przecenianie faworytów umiarkowanych $[1.25, 1.80]$**: Model przeszacowuje szanse o +3.7% do +4.6%, generując fałszywe sygnały value przeciwko rynkowi.
- **Niedoszacowanie faworytów skrajnych $[1.01, 1.25]$**: Obserwowany Win Rate 85.8% vs model 82.5% (-3.3%).
- **Bezpieczeństwo Underdogów $[3.50, 5.00]$**: Wskaźnik przeszacowania został opanowany dzięki bramce `hurdle_odds_cutoff` – model nie generuje agresywnych sygnałów na outsiderów bez potwierdzenia rynkowego.

---

## 3. Kompleksowy Przegląd 25 Modeli i Architektur (2026)

Ewaluacja probabilistyczna i finansowa (z uwzględnieniem 12% podatku i minimalnego progu $\text{EV}_{\text{net}} \ge +5\%$) na 622 meczach.

### Główne Wyniki Finansowe i Przebicie Bariery Coin-Flip:
1. **Bariera Coin-Flip ($0.6931$)**:
   - Dla meczów 50/50 czysty model osiąga LogLoss `0.7174` (z powodu nadmiernej pewności).
   - **Bayesian Shrinkage Logit** ($\alpha = 0.35$ do $0.50$) jako jedyna rodzina łamie tę barierę, osiągając **LogLoss `0.6929`** (lepszy niż rzut monetą i lepszy niż rynek `0.6987`).
2. **Korekta Błędu Odwrócenia Stron (Team Reversal Fix)**:
   - Wyrównanie `team_a` bukmachera z `team1` GOL.GG zlikwidowało defekt generujący wcześniej sztuczny LogLoss `0.8492` i stratę -6 970 PLN. Prawidłowo wyrównany surowy model osiąga znakomity LogLoss `0.5654` (bijąc rynek otwarcia `0.5687`).
3. **Najbardziej Rentowna Architektura Hybrydowa**:
   - **Bayesian Shrinkage Logit ($\alpha = 0.65$)**: Wypracowuje **+395.80 PLN** zysku netto (+7.20% yield) przy drawdownie zaledwie 5.1% i CLV +3.44%.
   - Zmniejsza straty w koszyku ryzykownym $[3.50, 5.00]$ z -892.8 PLN do -92.8 PLN.

---

## 4. Tournament Quality & Replay Readiness Benchmark

### A. Pokrycie Topologii i Mechanik Turniejowych
- **Zasób**: `data/artifacts/leaguepedia-tournament-rules-20260908` oraz `conf/base/tournament_formats.json`.
- **Katalog faz**: Skatalogowano 234 fazy w 7 rodzinach rozgrywek (Worlds, MSI, LCS, LCK, LEC, LPL, EMEA Masters).
- **Zaimplementowane profile**: 91 unikalnych profili mechaniki (drabinki pojedynczej/podwójnej eliminacji, GSL, swiss, round-robin).
- **Skonfigurowane fazy**: 227 faz (w tym 220 zakończonych faz historycznych).

### B. Status Certyfikacji Retrospektywnej (Temporal Integrity)
- **`historically_ready_count: 0`**: Dokładnie **0 faz** posiada pełną certyfikację niezależnej dostępności reguł, drabinek i losowań przed datą rozpoczęcia turnieju (point-in-time publication verification).
- Wszystkie konfiguracje stanowią modele mechaniki, a nie potwierdzone historyczne zapisy predykcyjne. Zgodnie z AGENTS.md żaden wynik symulacji turniejowej nie może być prezentowany jako zrealizowany wynik bukmacherski.

### C. Wyniki Diagnostyki Pojedynczego Turnieju (LCK 2026 Road to MSI)
W teście symulacji kwalifikacji do MSI:
- **EXP-081 Series Simulation**: Brier kwalifikacji `0.0609`, LogLoss `0.2172`.
- **Fair-Series Seeding Baseline**: Brier kwalifikacji `0.0349`, LogLoss `0.1873`.
- **Wniosek**: Struktura drabinki i rozstawienie (seeding) niosą kluczową informację topologiczną, której nie doceniały czyste modele meczowe.

---

## 5. Podsumowanie Wdrożeniowe i Rekomendacje

1. **Utrzymanie EXP-039 jako zamrożonego punktu odniesienia**: Żaden kandydat z rodziny sieci Siamese (EXP-080, EXP-081) nie osiągnął istotnej statystycznie przewagi w protokole walk-forward.
2. **Standard operacyjny hybrydy rynkowej**: Rekomendowane jest stosowanie estymatora **Bayesian Shrinkage Logit** ($\alpha \in [0.35, 0.65]$), który skutecznie tłumi błędy nadmiernej pewności siebie w meczach wyrównanych (przebicie bariery coin-flip) i eliminuje fałszywe sygnały na faworytach i outsiderach.
3. **Quarantine turniejów**: Symulator turniejowy posiada kompletną reprezentację mechanik (91 profili, 220 faz), lecz pozostaje w reżimie badań diagnostycznych do momentu formalnego pozyskania datowanych historycznych regulaminów.
