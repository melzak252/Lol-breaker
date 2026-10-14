# Raport Finansowy i Benchmark Bukmacherski: EXP-039 (Parity v2 Retrospective) (v3_corrected)

- **Okres kohorty:** `2026-05-29` do `2026-09-01`
- **Liczba ocenionych meczów:** 622
- **Horyzont czasowy wejścia:** `opening (>12h)` (zakaz grania na linii zamknięcia!)
- **Podatek obrotowy:** 12.0% (współczynnik efektywny: 0.88)
- **Filtr opłacalności:** Minimalne $\text{EV} \ge +5.0\%$ (domyślnie +5.0%)
- **Strategia stawkowania:** `flat_100` (kapitał początkowy: 10,000.00 PLN)

## 1. Główne Wskaźniki Rentowności, Ryzyka i CLV

| Wskaźnik | Wartość | Uwagi / Interpretacja |
| :--- | :---: | :--- |
| **Postawione oferty (Liczba zakładów)** | **111** | 17.8% wszystkich meczów spełniło filtr EV |
| **Skuteczność (Win Rate)** | **54.05%** | 60 wygranych / 51 przegranych |
| **Średni kurs zagrany** | **2.45** | Średnia arytmetyczna kursów dziesiętnych |
| **Średnia stawka** | **100.00 PLN** | Średnia stawka pojedynczego zakładu |
| **Łączny obrót (Total Staked)** | **11,100.00 PLN** | Suma zainwestowanego kapitału |
| **Oczekiwany Yield (Expected Yield / EV)** | **+26.50%** | Średnia modelowa wartość oczekiwana netto |
| **Oczekiwany Zysk Netto** | **+2,941.11 PLN** | Oczekiwany zysk według modelu |
| **Rzeczywisty Yield Netto** | **+0.43%** | **Zysk netto / Obrót (z podatkiem 12%)** |
| **Zwrot z Kapitału (Net ROI)** | **+0.47%** | **(Kapitał końcowy - początkowy) / Początkowy** |
| **Zysk Całkowity Netto** | **+47.25 PLN** | Wynik portfela na czysto po odliczeniu podatku |
| **Maksymalne Obsunięcie (Max Drawdown)** | **12.07%** | Najgłębszy spadek bankrolla od szczytu |
| **Max Drawdown w PLN** | **1,293.36 PLN** | Największa nominalna strata od lokalnego peaku |
| **Kapitał Końcowy** | **10,047.25 PLN** | Stan portfela po zakończeniu próby |

### Jakość Przewagi Nad Rynkiem (Closing Line Value - CLV)

| Metryka Rynkowa (CLV) | Wartość | Interpretacja |
| :--- | :---: | :--- |
| **Średni CLV (Closing Line Value)** | **+6.77%** | Średnia różnica kursu zagranego vs kursu zamknięcia (wyżej = lepiej) |
| **Pobicie Linii Zamknięcia (Beat Close Rate)** | **64.0%** | Odsetek ofert zagranych po kursie wyższym niż kurs zamknięcia |
| **Weryfikacja Ostrości Modelu (Sharpness)** | `OSTRY MODEL (CLV > 0)` | Potwierdzenie, że rynek przesuwa się w stronę predykcji modelu |

## 2. Analiza Zakładów w Przedziałach Kursowych i Kalibracji

### Segment: Przedział Kursowy (Rentowność i Kalibracja Szans)

| Koszyk Kursowy | Oferty | Win Rate | Śr. Kurs | Prawd. Modelu | Implikowane (1/kurs) | Gap Kalibracji | Brier | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Śr. CLV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< 1.40 (Ciężki faworyt)** | 3 | 100.0% | 1.36 | 90.4% | 73.3% | **-9.6%** | 0.010 | 300 | +60.36 | **+20.12%** | +4.83% |
| **1.40 - 1.80 (Faworyt)** | 34 | 73.5% | 1.60 | 80.2% | 62.9% | **+6.7%** | 0.200 | 3,400 | +93.98 | **+2.76%** | +6.85% |
| **1.80 - 2.20 (Wyrównany)** | 17 | 58.8% | 2.02 | 69.0% | 49.7% | **+10.2%** | 0.239 | 1,700 | +84.93 | **+5.00%** | +8.71% |
| **2.20 - 2.80 (Lekki underdog)** | 27 | 48.1% | 2.46 | 61.1% | 40.8% | **+12.9%** | 0.256 | 2,700 | +74.59 | **+2.76%** | +6.34% |
| **2.80 - 3.50 (Złoty underdog)** | 10 | 60.0% | 3.01 | 53.6% | 33.4% | **-6.4%** | 0.272 | 1,000 | +626.14 | **+62.61%** | +9.09% |
| **3.50 - 5.00 (Wysoki underdog)** | 20 | 15.0% | 4.11 | 40.2% | 24.6% | **+25.2%** | 0.216 | 2,000 | -892.75 | **-44.64%** | +4.70% |

*Uwagi diagnostyczne do kalibracji kursowej:*
- **Gap Kalibracji** = $\bar{p}_{\text{model}} - \text{Win Rate}$. Wartość bliska $0.0\%$ oznacza idealną kalibrację w danym koszyku. Wartość $> +5.0\%$ sygnalizuje przeszacowanie szans (overconfidence), a $< -5.0\%$ niedoszacowanie.
- **Implikowane (1/kurs)** = Średnie prawdopodobieństwo surowe wyceniane przez bukmachera. Nadwyżka Prawd. Modelu nad Implikowanym to źródło EV.
- **Brier Score** = Średni błąd kwadratowy predykcji na zawartych zakładach (niżej = lepiej).

### Segment: Wielkość Przewagi (EV)

| Koszyk (Bucket) | Liczba Ofert | Win Rate | Śr. Kurs | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Śr. CLV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Umiarkowane EV (5% - 8%)** | 19 | 52.6% | 2.20 | 1,900 | -437.04 | **-23.00%** | +10.06% |
| **Średnie EV (8% - 15%)** | 35 | 62.9% | 2.12 | 3,500 | +171.71 | **+4.91%** | +10.08% |
| **Wysokie EV (> 15%)** | 57 | 49.1% | 2.73 | 5,700 | +312.58 | **+5.48%** | +3.65% |

### Segment: Wartość Względem Zamknięcia (CLV)

| Koszyk (Bucket) | Liczba Ofert | Win Rate | Śr. Kurs | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Śr. CLV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Silny CLV (> +5% wyższy kurs)** | 55 | 61.8% | 2.28 | 5,500 | +795.76 | **+14.47%** | +15.19% |
| **Lekki CLV (0% do +5%)** | 36 | 50.0% | 2.42 | 3,600 | -405.62 | **-11.27%** | +1.56% |
| **Ujemny CLV (< 0% - linia uciekła)** | 20 | 40.0% | 2.94 | 2,000 | -342.89 | **-17.14%** | -6.99% |
