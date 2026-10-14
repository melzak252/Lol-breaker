# Plan trackowania kursów bukmacherskich

## Cel

Baza ma przechowywać historię kursów tak, żeby dało się później policzyć:

- najlepszy dostępny kurs w momencie decyzji,
- zmianę kursu w czasie,
- CLV względem kolejnych snapshotów / closing price,
- skuteczność scraperów i kompletność danych,
- mapowanie eventów bukmachera do kanonicznych meczów/modelu.

## Zasada główna

Nie zakładamy, że bukmacher/API zawsze zwraca pełny rynek `Team A` i `Team B` naraz. Realne API często zwraca pojedynczy selection/outcome. Dlatego kanonicznym poziomem zapisu jest:

```text
scrape run -> event -> market -> outcome odds snapshot
```

Dodatkowo rozróżniamy dwa URL-e:

- `source_url` — lista/API/snapshot, z którego pochodzi dany scrape,
- `offer_url` — deep link do konkretnej oferty/eventu; używany do ręcznego lub automatycznego pobrania close odds ok. 5 minut przed startem.

## Tabele

### `scrape_runs`

Audit każdego uruchomienia scrapera.

Najważniejsze pola:

- `bookmaker_id`,
- `scraper_name`, `scraper_version`,
- `started_at`, `finished_at`, `status`,
- `source_url`, `request_url`,
- `items_seen`, `items_inserted`, `error`.

### `bookmaker_events`

Wydarzenie tak, jak widzi je bukmacher.

Najważniejsze pola:

- `bookmaker_event_id` — stabilne ID z API, np. STS `matchId`,
- `raw_team_a`, `raw_team_b`,
- `mapped_team_a`, `mapped_team_b`,
- `match_start_time`,
- `sport_id/name`, `category_id/name`, `league_id/name`,
- `offer_url` — bezpośredni link do eventu u bukmachera,
- `match_id` — opcjonalne połączenie z `upcoming_matches`.

### `bookmaker_markets`

Rynek w obrębie eventu.

Najważniejsze pola:

- `bookmaker_market_key`,
- `market_name`,
- `market_type`, np. `match_winner`, `handicap`, `correct_score`, `map_prop`,
- `line_id`, `line_name`,
- `is_extra_market`.

### `odds_outcome_snapshots`

Pojedynczy tick kursu dla jednego outcome’u.

Najważniejsze pola:

- `scraped_at`,
- `outcome_key`,
- `outcome_name`,
- `outcome_side` (`a`, `b` lub `NULL`),
- `decimal_odds`,
- `source_url`, `offer_url`,
- `raw_payload`.

To ta tabela jest podstawą do line movement i CLV.

## STS prematch SBK snapshot

Aktualny scraper STS używa:

```text
https://sbk.sts.pl/sbk-exporter/v1/sports/ssr
```

Znaczenie ID:

- `sport_id=156` — Esport,
- `category_id=992` — League of Legends,
- `market_name=Zwycięzca meczu` — prematch match winner.

Snapshot ma strukturę kompaktową `B`/`P`: drzewo sportów/kategorii/turniejów/fixture'ów jest w `B.S`, a oferty/markety w `P`. Dla MVP parser przechodzi po `B.S.156.C.992.T.*.FX`, łączy fixture z `P[offer_id]` i zapisuje tylko market `Zwycięzca meczu`.

Scraper zapisuje równolegle:

- `odds_outcome_snapshots` — pojedyncze outcome ticki dla obu stron rynku,
- `odds_snapshots` — legacy dwustronny `odds_a/odds_b`, żeby obecny generator EV działał bez dodatkowego widoku.

`social-api.sts.pl/api/events` zostaje jako pomocnicze źródło popularnych typów, ale nie jest pełną bazą prematch match-winner.

## Docelowy pipeline

1. `python -m betting_app.scripts.scrape_odds --bookmaker sts`
2. Zapisz `scrape_runs`.
3. Upsertuj eventy i rynki.
4. Dopisz nowe outcome snapshots.
5. Mapuj `raw_team_a/raw_team_b` do GOL.GG przez `team_aliases`.
6. Dla match-winner z obiema stronami rynku generuj EV sygnały.
7. Po zakończeniu meczu rozlicz bety i policz CLV/ROI/yield.

## Następne ulepszenia

- ewentualnie rozgryźć websocket `prematch-ws` / match details dla pełnej oferty marketów,
- dodać tabelę `market_closing_odds` albo widok wyliczający closing z ostatniego ticka przed startem,
- dodać deduplikację ticków bez zmiany kursu, jeśli baza urośnie,
- dodać job co 15-60 min oraz osobny job tuż przed startem meczów.
