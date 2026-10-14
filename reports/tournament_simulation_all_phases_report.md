---
type: tournament_benchmark_report
project: EnsembleLegends
date: 2026-09-09
status: invalidated-audit
evaluated_phases: 207
total_completed_phases: 220
methods:
  - flat_baseline_uniform
  - seed_aware_fair_series
  - sports_rating_frozen_glicko2
  - corrected_calibrated_markov_temperature
---

# Zbiorcza Ewaluacja Symulacji Turniejowych na 207 Fazach Rozgrywek (2020–2026)

> **UWAGA — wnioski raportu wycofane po audycie (2026-09-09).**
> Poniższa treść jest zachowana jako historyczny zapis błędnej ewaluacji, nie jako dowód jakości modeli. Faktyczne liczby faz z wynikami to **109 dla mistrza** i **44 dla awansu**, a nie 195 i 46. Top-2/Top-4 oznacza obecność późniejszego mistrza w rankingu faworytów, nie trafność przewidywania finalistów/półfinalistów. Nie wykonano pełnej ewaluacji czterech osi; konfiguracja formatu nie potwierdza poprawności historycznego odtworzenia ani braku wycieku danych. Raport nie ocenia EXP-081.
>
> Zobacz [audyt źródeł, pełny protokół i plan nowej ewaluacji](../docs/05_results/tournament_evaluation_audit_and_plan_20260909.md). Nie używać poniższych tabel do promocji modeli ani deklarowania wiarygodności prawdopodobieństw.

## Nowa implementacja i ograniczony pilot — 2026-09-09

**Status: mechanika wdrożona; pełna walidacja empiryczna nadal zablokowana.** Cztery subagenty wykonały implementację, dwa kolejne przejrzały kod; poprawione naruszenia stanu/czasu i integralności wznowienia są objęte regresjami. Przeszło **193 testy ukierunkowane**. CLI wykonało prognozowanie, ocenę i wznowienie dla dwóch syntetycznych edycji; nie są one dowodem jakości predykcji.

Pełny inwentarz obejmuje **234 fazy**, w tym **220 ukończonych**. Zapisano **1 170 wykluczeń faza/oś**; pełny manifest zakończył się jawnym `no_eligible_origins`, nie wynikiem na wybranym podzbiorze.

Wykonano osobny, retrospektywnie odtworzony **pilot LCK2026 Road to MSI od startu fazy**, nie całej edycji. Nowy chronologiczny fold EXP081 korzysta z rozdzielonych okresów dopasowania i kalibracji, kończących się przed 6 czerwca. Zachowano ograniczenia historii częściowej i proxy poprzednio obserwowanego składu.

| Sześć binarnych celów awansu do MSI, jeden start fazy | LogLoss | Brier |
|---|---:|---:|
| EXP081 | 0.211047 | 0.058874 |
| Fair series na tej samej drabince | 0.187476 | 0.035121 |
| Flat | 0.636514 | 0.222222 |

**W tym pilocie EXP081 jest gorszy od fair series w obu właściwych metrykach.** Jedna faza nie pozwala wyznaczyć wiarygodnego przedziału porównania ani potwierdzić kalibracji. Nie ma podstaw do promocji modelu.

Zapisano **90 wierszy dla czterech cutoffów**, w tym **18 jawnych nieudanych prognoz refreshed**. Odświeżanie historii od 7 czerwca odrzuca źródło `79152` bez obiektów gier; późniejszy `79275` ma trzy mapy mimo raportowanych czterech. Nie zastąpiono brakujących cech stanem frozen. Kontynuacje ze stałą macierzą startową są osobno oznaczone; siedem rozstrzygniętych celów wyłączono z oceny. Brakuje też dowodów dostępności wyników/losowań do ścisłej osi next-round.

![Ograniczony pilot: prawdopodobieństwa i właściwe metryki](../docs/assets/tournament_evaluation_20260909.png)

Pełny opis, wyniki osobnego kontrastu frozen–fair, komendy odtwarzania i ścieżki artefaktów: [wynik implementacji](../docs/05_results/tournament_evaluation_audit_and_plan_20260909.md#implementation-result). Dalszy pilot dzienny dodał jawne tryby `pre_draw` / `post_draw` / `daily_rollforward` / `post_round`; w Road to MSI wykonał 12 originów, 282 wiersze i nie odnowił zablokowanych cech EXP081. Wspólny scoreboard pozostaje `inconclusive`; dostępny kontrolny widok frozen–fair nie jest pełnym kohortowym porównaniem modeli. Końcowy werdykt: **inconclusive**, `production_qualified=false`. Wyników poniższego, wycofanego raportu nie przywrócono.

## Zamrożony protokół akademicki i pełny audyt kohorty

Wykonano offline `scripts/academic_cohort.py`, bez bazy, scrapera, treningu ani testów. Maszynowo czytelny protokół i pełny audyt zapisano w `data/artifacts/tournament-academic-validation-20260909/academic_cohort/` jako `academic_protocol.json` i `academic_cohort.json`. SHA-256 protokołu: `01676f4a991bd15ca2457de1dc7daca7d66d84c76f6c60fc48bc86ad89d2877c`.

Ponownie rozliczono **234 unikalne fazy i 1 170 wykluczeń faza/oś**. Inwentarz daje **206 kandydackich klastrów tytułów i 64 miesiące startu**, ale **zero zweryfikowanych pełnych edycji i zero autoryzowanych originów**. Klastry mogą rozdzielać fazy tej samej edycji; miesiące obejmują także zaplanowane fazy. Nie są to certyfikowane niezależne obserwacje. W źródle 90 z 8 016 wierszy oznaczonych jako ukończone nie ma poprawnej daty ISO; nie uzupełniono ich wymyślonym czasem.

Główny estimand dotyczy marginalnego LogLoss awansu EXP081 frozen względem fair-series przy starcie fazy po ustaleniu drabinki. **Nie zastępuje to pełnej walidacji turniejowej.** Protokół osobno wymaga mistrza, oficjalnych przedziałów miejsc, rzeczywistego dojścia do finału/węzłów, zestawień w węzłach, łącznych zbiorów kwalifikantów i par finalistów oraz celów kolejnej rundy i ścieżek. Brak stosownej rodziny celów lub rozkładu łącznego blokuje całościową deklarację zaufania. Tryby prognozy, frozen/refreshed, proxy/ogłoszone składy i rekonstrukcja/prospektywność nie są mieszane; kontrole operacyjne i EXP039 wymagają własnej poprawnej proweniencji.

Ponowna agregacja pilota nadal daje **jeden blok edycji i jeden miesiąc na kontrast**. Wariancja różnic między edycjami, moc i wymagane N są **nieestymowalne**, nie potwierdzone arbitralnym progiem liczby faz czy symulacji. Przyszła reguła obejmuje nowe, wcześniej nieoglądane pełne edycje rozpoczynające się **ściśle po 2026-09-10T00:00:00+00:00 oraz po faktycznym zamrożeniu protokołu/modelu**. Cały obecny inwentarz pozostaje danymi rozwojowymi. **Nie zebrano tutaj żadnego przyszłego holdoutu.**

Dokładne wagi dzienne/horyzontów, blokowy bootstrap, korekta wielokrotności, jednoczesne bramki kalibracyjne, wymagania precyzji i analiza mocy: [zamrożony protokół i wykonany audyt](../docs/05_results/tournament_evaluation_audit_and_plan_20260909.md#locked-academic-protocol-and-executed-full-inventory-cohort-audit). Wynik pozostaje **inconclusive**, `production_qualified=false`; poniższy wycofany raport nadal nie jest dowodem jakości.

---

## Archiwalna treść wycofanego raportu

> [!abstract]
> Przeprowadzono całościową symulację Monte Carlo na **207 ukończonych fazach turniejowych** z bazy Leaguepedia/Cargo i GOL.GG przy użyciu 4 niezależnych metod predykcyjnych. Model sportowy oparty na ratingach Glicko-2 osiąga **32.11% trafności Top-1** (wobec **12.47%** losowego Flat Baseline) oraz **79.82% pokrycia w Top-4**. Najwyższą jakość predykcji odnotowano w formatach **Double Elimination (Brier 0.6235 vs 0.8594)** oraz ligach **LCK (Brier 0.5142, 61.5% Top-1, 100% Top-4)** i **MSI (Brier 0.5255, 100% Top-4)**.

---

## 1. Architektura 4 Metod Symulacyjnych

Każda faza turniejowa została zasymulowana w $N_{\text{sim}} = 400$ pełnych przebiegach Monte Carlo według 4 metod:

1. **Sposób 1: Flat Baseline (Uniform $1/N$)**:
   - Prawdopodobieństwo $P = 1/N$ dla każdego uczestnika, ignorujące geometrię drabinki, rozstawienia i siłę zespołów.
2. **Sposób 2: Seed-Aware Fair Series Baseline ($P=0.50$)**:
   - Pełna symulacja Monte Carlo dokładnej topologii drabinki (`simulate_tournament`), w której każda seria meczowa ma równe szanse $P=0.50$. Izoluje czysty wpływ mechaniki turniejowej (byes, drabinka przegranych, gauntlet).
3. **Sposób 3: Sports Rating Model (Pre-Tournament Glicko-2 / Elo)**:
   - Ratingi zespołów zamrożone przed datą startu fazy (zero leakage), z bayesowskim shrinkage i regionalnymi offsetami (LCK/LPL faworyzowane względem LEC/LCS i minor regions). Konwersja mapy na serię Bo1/Bo3/Bo5 przez dwumianowy model serii.
4. **Sposób 4: Corrected / Calibrated Series Model**:
   - Model `CorrectedTournamentPredictor` z dynamiką momentum Markowa ($\beta = 0.35$ na serię) oraz kalibracją temperatury $T=0.85$ zapobiegającą nadmiernej pewności siebie na faworytach.

---

## 2. Wyniki Zbiorcze na 4 Osiach Ewaluacji

### Oś 1: Cały Turniej od Startu ($N=195$ faz ze zwycięzcą)

| Metoda | Brier Score (min) | LogLoss (min) | Top-1 Accuracy | Top-2 Coverage | Top-4 Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Flat Baseline ($1/N$)** | 0.8846 | 2.2484 | 12.47% (exp) | 24.94% (exp) | 48.35% (exp) |
| **2. Seed-Aware Fair Series ($P=0.50$)** | 0.8913 | 2.3296 | 11.01% | 26.61% | 52.29% |
| **3. Sports Rating Model** | **0.8321** | 2.8569 | **32.11%** | 51.38% | **79.82%** |
| **4. Corrected / Calibrated Model** | 0.8533 | 3.2059 | 31.19% | **52.29%** | **79.82%** |

- **Top-1 Accuracy**: Model sportowy trafia mistrza w **32.11%** przypadków – niemal **3-krotnie częściej** niż losowy wybór ($12.47\%$).
- **Top-4 Coverage**: Rzeczywisty mistrz turnieju znajdował się w czwórce faworytów modelu w **79.82%** wszystkich analizowanych faz turniejowych.
- **Wpływ samej drabinki (Fair Series)**: Sama struktura rozstawień daje faworytom podwojenie szans awansu, ale bez wiedzy o sile zespołów nie pozwala na skuteczne typowanie zwycięzcy (11.01% vs 32.11%).

---

### Oś 2: Fazy Awansu i Kwalifikacji ($N=46$ faz Swiss/Grupy/Play-In)

| Metoda | Brier Score Awansu | Względna zmiana błędu |
| :--- | :---: | :---: |
| **1. Flat Baseline ($k/N$)** | **0.24868** | punkt odniesienia |
| **2. Seed-Aware Fair Series** | 0.29334 | -17.96% |
| **3. Sports Rating Model** | 0.32052 | -28.89% |
| **4. Corrected / Calibrated Model** | 0.32553 | -30.91% |

- **Diagnoza faz kwalifikacyjnych**:
  - Na turniejach **Worlds (Swiss i Play-In)** model sportowy **pobija Flat Baseline** (Brier **`0.24156`** vs `0.25000`).
  - Na turniejach **MSI Play-In** model sportowy dorównuje Flat Baseline (Brier **`0.25149`**).
  - W turniejach regionalnych Tier-2 (EU Masters, ERL) brak wspólnej historii spotkań między drużynami z różnych lig narodowych prowadzi do domyślnych ratingów (1350) i podwyższonego błędu kalibracji.

---

### Oś 4: Przekroje Diagnostyczne

#### 1. Według formatu / mechaniki turnieju:

| Format turnieju | Liczba faz | Flat Brier | Fair Brier | Rating Brier | Rating Top-1 | Rating Top-4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Double Elimination** | 16 | 0.8594 | 0.8631 | **0.6235** | **50.0%** | **93.8%** |
| **Single Elimination** | 10 | 0.8565 | 0.8538 | **0.8107** | **50.0%** | **83.3%** |
| **Gauntlet (King of the Hill)** | 7 | 0.9000 | 0.9997 | **0.7923** | **28.6%** | **71.4%** |
| **Graph (Drabinki wieloetapowe)** | 58 | 0.8527 | 0.8403 | **0.8053** | **35.0%** | **90.0%** |
| **Round Robin (Grupy / Sezony)** | 87 | 0.9213 | 0.9410 | **0.8674** | **27.6%** | **62.1%** |
| **Swiss Stage** | 9 | 0.9453 | 0.9428 | 1.1081 | 0.0% | 75.0% |
| **GSL Groups** | 8 | 0.9479 | 0.9259 | 1.3574 | 0.0% | 66.7% |

#### 2. Według ligi / rodziny (Competition Family):

| Liga / Rodzina | Liczba faz | Flat Brier | Fair Brier | Rating Brier | Rating Top-1 | Rating Top-4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LCK (Korea)** | 40 | 0.8282 | 0.8711 | **0.5142** | **61.5%** | **100.0%** |
| **MSI (International)** | 13 | 0.8955 | 0.9046 | **0.5255** | **40.0%** | **100.0%** |
| **LEC (Europa)** | 36 | 0.8660 | 0.8584 | **0.6138** | **54.2%** | **91.7%** |
| **LPL (Chiny)** | 26 | 0.9010 | 0.9080 | **0.7779** | **33.3%** | **86.7%** |
| **LCS (Ameryka)** | 35 | 0.8479 | 0.8288 | 1.0599 | 10.0% | 80.0% |
| **Worlds** | 19 | 0.9375 | 0.9589 | 1.0898 | 8.3% | 50.0% |
| **EU Masters (Tier 2)** | 26 | 0.9461 | 0.9651 | 1.2053 | 13.3% | 46.7% |

#### 3. Według liczby uczestników ($N$):

| Koszyk uczestników | Liczba faz | Flat Brier | Rating Brier | Rating Top-1 Acc | Rating Top-4 Cov |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$N \le 6$ drużyn** | 55 | 0.8289 | **0.6557** | **50.0%** | **100.0%** |
| **$N = 8$ drużyn** | 32 | 0.8750 | 0.8966 | 22.7% | 77.3% |
| **$N = 10$ drużyn** | 52 | 0.9000 | **0.7887** | 26.7% | 80.0% |
| **$N \ge 12$ drużyn** | 56 | 0.9338 | 0.9696 | 23.7% | 63.2% |

---

## 3. Główne Wnioski i Diagnoza Błędów (Failure Modes)

1. **Dominacja w Double Elimination**: Formaty z podwójną eliminacją dają modelowi sportowemu największą przewagę (Brier redukowany z **`0.8594`** do **`0.6235`**). Wynika to z faktu, że podwójna drabinka filtruje losowość pojedynczych map i faworyzuje silniejsze zespoły.
2. **Niezrównana stabilność w LCK i MSI**:
   - W **LCK** model wskazał zwycięzcę w **61.5%** faz, a w **100%** faz rzeczywisty mistrz znalazł się w Top 4 modelu.
   - Na **MSI** w **100%** edycji mistrz był w Top 4 modelu, a Brier wyniósł znakomite **`0.5255`**.
3. **Identyfikacja punktów słabości**:
   - **Swiss i GSL**: Losowania rund 2-5 w Swiss i deciderów w GSL wprowadzają dużą wariancję. Pojedyncze gry Bo1 powodują, że model statyczny przed turniejem ponosi wysokie kary kwadratowe za niespodzianki.
   - **ERL / EU Masters**: Brak bezpośrednich meczów między drużynami z różnych lig regionalnych (Hiszpania, Francja, Polska, Niemcy) uniemożliwia precyzyjną kalibrację bez dynamicznych wag turniejowych.
   - **Worlds Cinderella Runs**: Wygrane T1 na Worlds 2023 i Worlds 2024 (jako 4. seed LCK po słabym sezonie letnim) to klasyczny przypadek meta-shiftu pomeczowego, którego żaden model zamrożony przed turniejem nie jest w stanie przewidzieć bez uwzględnienia adaptacji do patcha.
