# EXP-040: architektura kandydata i operacje

## Status

`Sym-Cal LR-ElasticNet-W20-Binomial / exp-039` pozostaje zamrożonym artefaktem tezy. EXP-040 jest kandydatem; nie zastępuje EXP-039, nie składa zakładów i nie ma jeszcze kompatybilnego producenta cech dla predykcji nadchodzących spotkań.

Wcześniejsze liczby ROI, zysków i „zero ryzyka bankructwa” pochodziły z retrospektywnej symulacji na wybranych kursach historycznych. Nie spełniała ona pełnego kontraktu czasu decyzji, dlatego nie są miarą wykonalnej strategii ani podstawą promocji.

## Kontrakt danych i decyzji

Każdy wiersz kwalifikowany jako wykonawczy musi spełniać:

```text
feature/source time <= data_cutoff_at <= predicted_at <= quote_at < match_start_at
```

Brak któregokolwiek znacznika czasu wyklucza wiersz z trybu `live`. Kurs zamknięcia jest wyłącznie benchmarkiem diagnostycznym, nie wejściem do modelu ani ceną wykonania w symulacji.

Symulacja finansowa prowadzi księgę zdarzeń: przy złożeniu rezerwuje stawkę, nie używa kapitału z nierozstrzygniętych meczów i rozlicza po `result_recorded_at`. Widok `historical` jest jawnie oznaczony jako badawczy; tylko `live` egzekwuje pełny kontrakt czasu.

## Model EXP-040

1. **Cechy domenowe** — `candidate_features.py` oblicza kontekst strony, patch decay i ciągłość składu.
2. **Seria BoN** — `markov_series.py` modeluje wybór strony po grach. Jeżeli rzeczywisty priorytet Game 1 nie jest znany, prognoza uśrednia oba warianty, aby nie nadać arbitralnej przewagi stronie A.
3. **Kalibracja** — pipeline zapisuje temperaturę oraz Venn-Abers dopasowane do chronologicznych predykcji out-of-fold. Przedziały są emitowane w `canonical_predictions.diagnostics_json`.
4. **Bramka ryzyka** — używa wyłącznie zapisanych przedziałów Venn-Abers. Heurystyczny haircut punktowej predykcji nie jest przedziałem conformal i nie może oznaczać wartości.

Dla strony zakładu z dolnym ograniczeniem $p_{\mathrm{low}}$, kursem dziesiętnym $o$ i podatkiem obrotowym $t=0.12$:

$$
EV_{\mathrm{low}} = p_{\mathrm{low}}\,[o(1-t)] - 1.
$$

Sygnał przechodzi tylko gdy $EV_{\mathrm{low}}>0$ i szerokość przedziału nie przekracza $0.08$. Hybryda model+rynek nie przechodzi tej bramki, ponieważ po zmieszaniu prawdopodobieństw nie zachowuje pokrycia interwału pojedynczego modelu.

## OddsPapi

Integracja pobiera wyłącznie mapowania fixture i kursy Pinnacle dla obserwacji rynku; nie wykonuje zakładów.

- `oddspapi_fixture_sync`: odświeża fixture co trzy dni.
- `oddspapi_horizon_fetch`: pobiera mapowania mieszczące się w skonfigurowanym horyzoncie.
- `OddsPapiBudgetGuard`: blokuje wywołania po 8 żądaniach dziennie lub 250 w ruchomym oknie 30 dni.

Pinnacle jest benchmarkiem rynku. Nie trafia do wejść modelu sportowego. Porównanie rynku zwraca pola conformal jako `null`/`false`, dopóki najnowsza predykcja nie ma prawidłowych granic Venn-Abers.

## Analizy aplikacji

- **Finanse**: EXP-040 może być analizowany po wersji i `exp040-markov-va-v1`; włączona bramka conformal odrzuca prognozy bez zapisanych granic.
- **Horyzonty**: model candidate pojawia się dopiero po osiedleniu meczów z poprawnymi znacznikami czasu. Metryki są liczone na tych samych koszykach kursów, co benchmark rynku.
- **Turnieje**: symulator używa Markowa dla Bo3/Bo5 i nie wyprowadza pierwszeństwa strony z ratingu. Brak znanego wyboru strony daje średnią z obu wariantów.
- **Tablica meczów**: nie wyświetla sztucznej stawki opartej o stały bankroll. Wielkość stawki powstaje wyłącznie w księdze finansowej, która zna dostępny kapitał i otwarte rezerwacje.

## Operacje

```bash
.venv/bin/python -m betting_app.ml.pipelines.exp040_retrain_pipeline --min-date 2020-01-01
```

Przed rejestracją kandydata sprawdź chronologiczne metryki, kohortę, hash danych i artefakty. Promocja do `shadow` wymaga najpierw dostarczenia i przetestowania producenta cech zgodnego z 46-cechowym kontraktem treningowym; dopiero potem można uruchomić shadow inference na nowych feature rows i po rozstrzygnięciu meczów sprawdzić kalibrację, log loss, Brier, AUC, timing oraz no-vig market baseline na wspólnej kohorcie.

Promocja wymaga osobnej, udokumentowanej decyzji. Nie nadpisuje EXP-039.
