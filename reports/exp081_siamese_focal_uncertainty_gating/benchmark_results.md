# EXP-081: korekta kontraktu i diagnostyka architektur

**Aktualizacja: 2026-09-07. Status: diagnostyka retrospektywna, bez zgody na promocję.**

## Wycofanie wcześniejszych wniosków

Poprzednie deklaracje „eliminacji Winner's Curse”, optymalności `kappa=0.75`, rekordowego yield `+54.48%` i redukcji wykonalnego ryzyka portfela nie są uzasadnione. Nie należy ich cytować jako wyników produkcyjnych ani dowodu przewagi EXP-081/EXP-082.

Audyt ujawnił rozbieżność trening–inference: brakujące W20 i interakcje formatu serii zastępowano zerami, a gradient Focal Loss pomijał pochodną czynnika modulującego. Jednostronna dolna ocena A była też błędnie dopełniana do oceny B. Brakuje pełnego point-in-time pochodzenia ratingów, timestampów dostępności składów/kwotowań i event-time ledger z rezerwacją kapitału. Historycznych tabel zysków nie traktujemy jako wykonalnych wyników finansowych.

## Naprawiony kontrakt

- Trening korzysta z tego samego `build_feature_mapping` i tej samej kolejności **79 cech** co inference; wymaga kompletnych surowych wejść i jawnego `best_of`.
- Dla logitów członków zespołu: `p_a = sigmoid(mean_z)`, `p_b = 1-p_a`, `low_a = sigmoid(mean_z-kappa*sd_z)`, `low_b = sigmoid(-mean_z-kappa*sd_z)`, `sd_z` z `ddof=0`.
- `low_a + low_b` nie musi wynosić 1. Są to heurystyczne oceny decyzyjne, **nie przedziały ufności ani gwarancja pokrycia**.
- Hybryda przekształca obie dolne oceny osobno, z odpowiednią stroną rynku i tym samym trybem temperatury/blendowania co średnią.
- Generator sygnałów, tablica meczów, szczegóły i endpoint sygnałów wymagają wspólnej kwalifikacji. EV i Kelly korzystają z dolnej oceny; średnia pozostaje prezentowanym prawdopodobieństwem i wejściem do kontroli rozbieżności z rynkiem. Brak wymaganej diagnostyki blokuje rekomendację.
- Rozbieżność ratingów to różnica średnich sześciu prawdopodobieństw drużyny i graczy, nie wyłącznie różnica Glicko.
- Domyślny próg EV generatora i CLI to `0.05` plus istniejąca kara niskich kursów. Nie jest to dowód optymalności tych parametrów. Model nie składa zakładów automatycznie.
- Eksport treningowy zapisuje nowy artefakt `experimental_not_qualified`, nigdy nie nadpisuje istniejących plików ani nie promuje modelu.

## Protokół wykonanego porównania

Odtworzono 27 717 kompletnych snapshotów z 40 158 wierszy ratingów. Wykluczono 4 873 serie z powtórnym uczestnictwem tej samej drużyny/gracza tego dnia, 7 550 z niepełnym W20, 4 z niepełnym składem i 14 niepoprawnych/niekompletnych serii. Predykcje całego dnia powstają przed aktualizacją historii tego dnia. Brakująca statystyka nie jest zerem ani powodem przesunięcia okna na starsze, wygodniejsze mecze.

Porównanie architektur ogranicza historię treningową do dat od **2020-01-01**. Expanding-window: trening przed poprzednim rokiem kalendarzowym, osobny poprzedni rok do kalibracji, kolejny rok do oceny. Test: **2024-01-14–2026-05-11, N=8 860**, trzy foldy roczne. Brak strojenia na tym teście. MLP: 35 epok, 5 członków, seedy 42/143/244/345/446; dodatnia kalibracja logitu bez interceptu. Regresja: `C=0.1`, bez interceptu. Skalowanie fitowane wyłącznie na treningu, bez centrowania dla zachowania antysymetrii.

Wariant 84 to 79 cech kanonicznych i pięć nieparzystych interakcji niezgodności ratingów. Dodają założenia funkcjonalne, **nie niezależne informacje o meczu**. Nie są wdrożonym ani zatwierdzonym EXP-082.

| Wariant | LogLoss ↓ | Brier ↓ | AUC ↑ | Accuracy ↑ | ECE10 ↓ | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| linear79 | 0.557858 | 0.189371 | 0.782531 | 0.707562 | 0.021779 | 1.020222 |
| bce79 | 0.597233 | 0.205281 | 0.751443 | 0.684876 | 0.051806 | 1.449558 |
| focal79 | 0.597799 | 0.205420 | 0.751461 | 0.684763 | 0.051772 | 1.468087 |
| focal84 | 0.601808 | 0.206952 | 0.748977 | 0.683747 | 0.055900 | 1.466992 |

Miesięczny block bootstrap: 5 000 resampli, 29 miesięcy; delty na wspólnych seriach, ujemne wartości lepsze:

- `focal79 − bce79`: ΔLogLoss **+0.000566**, 95% CI **[-0.001289, +0.002763]**, udział resampli Δ≥0: 0.702. Brak dowodu przewagi Focal Loss.
- `focal84 − bce79`: ΔLogLoss **+0.004575**, 95% CI **[+0.001098, +0.007777]**. W tym protokole wariant 84 jest gorszy.
- `bce79 − linear79`: ΔLogLoss **+0.039375**, 95% CI **[+0.032784, +0.045721]**. Wynik dotyczy tych konfiguracji, nie dowodzi uniwersalnej przewagi regresji nad sieciami.

Maksymalny błąd symetrii wszystkich wariantów: `2.22e-16`. Pełny JSON zawiera MCE, intercept, przybliżoną binowaną dekompozycję Briera z resztą, delty Briera, przekroje Bo1/3/5, poziomów rozgrywek, składów, confidence i niezgodności ratingów. Przekroje confidence są wspólne, ustalone według BCE79; nie dobieramy łatwiejszej grupy dla każdego modelu osobno.

Kursy dopasowano dokładnie po ID, dacie, stronach i wyniku dla **3 783** serii; 937 wymagało odwrócenia stron. Closing jest wyłącznie diagnostycznym benchmarkiem, nie wejściem modeli sportowych ani dowodem możliwej do zawarcia oferty. Raport zawiera korelacje i diagnostyczną krzywą mieszania; jej minimum nie zostało wybrane do produkcji.

## Dlaczego brak promocji

1. N=8 860 nie spełnia wymaganego N≥10 000 dla kohorty 2024+.
2. Nie można zweryfikować wersji i pre-update pochodzenia odziedziczonych ratingów. Składy odczytano z rozegranych gier, nie z timestampowanych ogłoszeń przedmeczowych.
3. Brak timestampów dostępności źródeł i kwotowań; nie można udowodnić pełnej nierówności temporalnej. Chronologiczny podział nie naprawia tego braku.
4. Kohorta była wcześniej oglądana; nie jest nietkniętym holdoutem. Brak point-in-time predykcji zamrożonego EXP-039 i właściwego baseline'u operacyjnego na tej kohorcie. `linear79` jest nowym kontrolnym treningiem, nie historycznym artefaktem EXP-078.
5. MLP nie spełniają wymagań kalibracji. Brak event-time ledger — nie raportujemy ROI, zysku, drawdown ani gwarantowanej ochrony faworytów.

**Nie podmieniono artefaktu aktywnego modelu i nie wykonano wdrożenia.** Poprawki kodu nie nadają historycznemu artefaktowi nowej walidacji. Do decyzji operacyjnej potrzebny jest wiarygodny point-in-time zbiór z baseline'ami oraz nowy, zatwierdzony eksperyment; nie wolno brakujących timestampów dopisywać wstecz.

## Reprodukcja lokalna, bez bazy i scrapowania

Wymagane istniejące pliki: `data/golgg_y_predicts.csv`, `data/golgg_matches.json`, `data/odds.csv`. Oba skrypty wymagają nieistniejącego katalogu wynikowego.

```bash
.venv/bin/python scripts/build_siamese_research_dataset.py --output-dir data/artifacts/siamese-integrity-repro
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_siamese_architectures.py \
  --snapshots data/artifacts/siamese-integrity-repro/snapshots.csv \
  --audit data/artifacts/siamese-integrity-repro/audit.json \
  --output-dir data/artifacts/siamese-integrity-repro/benchmark
```

Wyniki wykonanej sesji: `data/artifacts/siamese-integrity-v1/audit.json` i `benchmark/summary.json`, `benchmark/predictions.csv`. To lokalne, ignorowane artefakty; nie dodawać datasetów do Git. JSON przechowuje sumy SHA256 wejść i kodu z chwili uruchomienia.

## Zakres literatury

- [Focal Loss, Lin et al.](https://arxiv.org/abs/1708.02002): motywacja dotyczy nierównowagi łatwych/trudnych przykładów w detekcji obiektów, nie dowodu przewagi w prognozach LoL.
- [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html): kalibracja wymaga osobnej walidacji; temperatura `T<1` przy `logit/T` wyostrza, a nie spłaszcza rozkład.
- [Lakshminarayanan et al., deep ensembles](https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles): uzasadnia badanie niepewności zespołu, nie gwarantuje pokrycia heurystyki `mean_logit − 0.75*sd`.
