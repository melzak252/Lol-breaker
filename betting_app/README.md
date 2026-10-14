# LoL-Breaker — LoL Betting & EV Intelligence Manager

Lokalna aplikacja do ręcznej bukmacherki/analityki LoL. Aplikacja **nie stawia zakładów automatycznie** — służy do zbierania kursów, mapowania drużyn, liczenia EV, zapisywania decyzji i późniejszej analizy bankrolla/CLV.

> Stan lokalnego checkoutu po cutoverze 2026-10-05 i naprawach kontraktów 2026-10-06: model sportowy **Causal-C0**, hybryda **C0 + rynek**. To nie jest deklaracja wdrożenia na serwer. Obserwacje produkcyjne z 2026-09-26 i wcześniejsze procedury zachowano w archiwalnej części poniżej; nie opisują aktualnego modelu lokalnego.

## Aktualny kontrakt C0

| Ścieżka | Nazwa | Dokładna wersja |
|---|---|---|
| Model sportowy | `Causal-C0` | `c0-native-2026-w32-e12-v1` |
| Shrinkage model + rynek | `Hybrid-Operational-Market` | `c0-native-2026-w32-e12-v1-a0.50-t1.00` |

- Rejestr i wspólny silnik: `betting_app/core/models/registry.py` oraz `engine.py`. Upcoming, ręczne **Predict**, scheduler, symulacje, bieżące selektory API i finansów korzystają z C0 albo osobnej hybrydy C0. Stare rekordy EXP081 nie są przemianowywane ani dopuszczane jako aktualna wersja.
- Worker: `betting_app/services/c0_inference.py` / `c0_worker.py`; natywny adapter: `betting_app/core/models/native_c0.py`. Izolowany proces chroni API przed kolizją nazw archiwalnych modułów `src.models`.
- To oryginalny zamrożony **control W32/E12** z `a0_data_focus_20260929_192443`: trzy checkpointy neuronowe, wspólne eksperci rating/MLP i kalibracja/mieszanki zależne od trybu. Nie roczny A0, challenger pojemnościowy ani adapter EXP081.
- Wynik C0 jest już prawdopodobieństwem **całej serii** BO1/3/5. Nie wykonujemy drugiej projekcji BO. C0 nie udostępnia operacyjnych dolnych granic ani `epistemic_sigma_z`; brak takich danych nie jest naprawiany fikcyjną niepewnością.
- Model sportowy nie używa kursów. Osobna hybryda przekształca zapisany C0 dokładnie raz: `sigmoid(0.50 * logit(p_C0) + 0.50 * logit(q_market))`, T=1.00. Analizy korzystające z już zapisanej hybrydy nie shrinkują jej ponownie.
- Wymagane są prawdziwe identyfikatory pięciu różnych graczy na każdą stronę, rozłączne składy, obsługiwany BO i strefowe zegary decyzji/startu. Lokalny `golgg_teams.id` nie jest identyfikatorem źródłowym `team_id`.
- Przy braku ID organizacji po którejkolwiek stronie i kompletnych składach używany jest jawny tryb `no_organization`, nie rating domyślny ani model zastępczy. Pozostałe tryby to `full` i `no_w20`. Cutoff przed 2026-01-01, braki wejścia, nieobsługiwany BO lub błąd artefaktu pozostają jawną odmową prognozy.
- Symulacje używają natywnych prawdopodobieństw zwycięzcy serii. Nie wymyślają mapowej ani warunkowej dystrybucji wyniku do obsługi rozpoczętej serii; nieobsługiwany stan jest odrzucany. ENC wymaga dokładnych ID zawodników; błąd zapytania źródłowego jest błędem serwera, nie „niekompletnym składem”.
- EXP081 nie ma aktywnego loadera, domyślnego artefaktu ani aliasu w aplikacji. Historyczne raporty, rekordy i jawnie wybrane narzędzia badawcze zachowano bez recertyfikacji. EXP-039 pozostaje osobnym, zamrożonym baseline'em.

### Historyczny benchmark proxy nie jest ścieżką native C0 — Plan11

`tournament_dist_20261005_01` używał Glicko i stałych offsetów, **nie natywnych checkpointów C0/A0**. Stare `a0` to `clip(sigmoid(s*(d+.70*o)/1.08),.001,.999)`; stare `c0` bierze ten sam nieobcięty bazowy `u` i liczy `clip(.92*sigmoid(logit(clip(u,.0001,.9999))/1.12)+.04,.12,.88)`. `d` to znormalizowana różnica Glicko z RD, `o` to stała różnica regionów, `s` to historyczne stałe nachylenie BO; [pełna formuła i zachowana korekta native attribution](../docs/RESEARCH.md#2026-10-05--tournament-distribution-and-bracket-simulation-benchmark-native-c0a0-attribution-withdrawn). Bieżąca ścieżka proxy używa jawnych nazw **`uniform`, `glicko_format`, `regional_glicko_proxy`, `calibrated_regional_glicko_proxy`** i receiptów wiążących konfigurację, formułę, źródła i kod. `glicko_format` dopasowuje nieujemne nachylenia BO bez interceptu na wcześniejszym CAL; regionalne proxy zachowują opisane stałe. Plan11 wykonał rzeczywisty CLI: 16 model×edition scenarios,32 drawy na scenariusz,6 wcześniejszych scalar CAL fits i **`PROXY_DIAGNOSTIC_COMPLETED / native_adoption=false`**. `--model c0` jest jawnie odrzucone w tej ścieżce, bez fallbacku. To nie zastępuje native inferencji ani nie certyfikuje kalibracji turniejowej.

Zachowany stary decision `TOURNAMENT_CALIBRATION_ADOPTED` egzekwował tylko Uniform Champion Brier i flow feasibility, pomijając zapisane Glicko Champion Brier oraz Uniform Swiss RPS. Wspomniany w komentarzu Glicko Swiss RPS również nie stanowił egzekwowanego gate'u. Stare decision/metrics/receipts pozostają nietknięte, ale **nie certyfikują native C0/A0 ani kalibracji turniejowej**. Minimum cutoff oryginalnego serving C0 **2026-01-01** wyklucza jego prognozę pre-event Worlds/MSI2023/24. Retrospektywne badanie native może używać jawnych scenariuszy roster/history przy origin-eligible artefaktach, zamrożonym protokole i pełnym wsparciu legalnych par; brak dowodu historycznej emisji nie zakazuje takiego researchu, ale wyklucza jego przedstawienie jako autentycznie wydanych prognoz. Osobna bieżąca gałąź native-study i osiem przypadków manifestu pozostają pracą ich właściciela, nie wykonaniem Plan11. Plan11 zachowuje datowaną36-cell eligibility inventory i addendum zmiany manifestu; nie certyfikuje całych edycji. Autentyczna certyfikacja pre-start nadal wymaga rzeczywistych history/roster/context/rules/draws i zegarów publikacji.

### Badanie wyłącznie funkcji straty — 2026-10-06

**`c0_ev_loss_20261006_01`: COMPLETED / NOT_ADOPTED**,12 formalnych fitów headów,0 fitów C0,0 refitów CAL. Porównano likelihood i dwustronny after-tax return-MSE na tych samych prior-known wejściach, cechach i polityce; TRAIN636/CAL299/ASSESS382 common,1,668 provider opportunities/1,351 native/312 braków składów/5 znanych wyników,523 settled cash z abstencjami. Market-only own risk ma510 wierszy, nie382 matched. Rzeczywisty fit2026-10-06T19:51:52.720474Z–19:58:43.554501Z; wszystkie wyniki2026 są development-exposed. Historyczne24h/nonlive-only kursy i daily release/provider clocks nie stanowią authentic300s/all-state admission.

H-return-MSE−H-likelihood risk CI98.333% **[−0.000847313,+0.001575692]** przecina0; dwa matched market-risk CI są ujemne, ale LL/Brier, symetryczne±2pp i dodatnia wartość nie przechodzą bramek.10 zakładów/expected75.4883%/ROI72.04% nie dowodzi sukcesu: error+3.4483pp95% CI[−21.9286,+106.0619], near6/error−10.8538pp[−90.0573,+106.0619]. Obie funkcje celu korzystają z identycznej reguły selekcji CAL return-MSE; wybrane λ i parametry modeli są odrębne. Bez przewagi baseline wynikającej z selekcji LL.

Zapisany [H-return-MSE head](../data/06_models/c0_ev_loss/c0_ev_loss_20261006_01/H-return_mse_selected.json) wymaga **p_C0, q i wykonywalnych dwustronnych cen**; nie jest rejestracją operacyjną ani zamiennikiem C0/Fixed50 z tabeli. [Raport §11](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md#11-c0-ev-loss-objective-study--2026-10-06), [protocol](../data/05_model_input/c0_ev_loss/c0_ev_loss_20261006_01/protocol.json), [decision](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/decision.json), [execution](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/execution_receipt.json), [independent PASS](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/independent/receipt.json):1,143,083 assertions incl. repeats/8 verification resolves (6 convex/2 nonconvex local),0 dodatkowych formalnych fitów; max scalar1.14e−13/finite-diff5.95e−11/projected-gradient6.26e−8. Osobny smoke600 rows/12 synthetic fits/0 formal. [Final regression](../data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/final_regression.json):139 focused/model/benchmark PASS,0 skips/1 istniejący warning,191.27s pytest/198.54s wall; wcześniejsze pre-fit139 to osobne wywołanie, nie278 różnych testów. Bez pełnego application suite, nowej rekonstrukcji native, global optimality, G0 ani issuance/execution certification.

CLI ma **trzy rozłączne tryby `--freeze` / `--run` / `--smoke`**, `--run-id` tylko freeze, `--protocol` w pozostałych; **brak `--research-root` i `--output-dir`**, ścieżki wynikają z protokołu/run ID. Zapis kontraktu, nie polecenie ponownego zapisu ukończonego immutable runu:

```bash
RUN=c0_ev_loss_20261006_01
PROTOCOL="data/05_model_input/c0_ev_loss/$RUN/protocol.json"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-ev-loss-study --freeze --run-id "$RUN"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-ev-loss-study --smoke --protocol "$PROTOCOL"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-ev-loss-study --run --protocol "$PROTOCOL"
```

Wszystkie8 source contracts/G0 nadal blocked: uprawnienia/retention/entitlements/quotas, official bindings, live released-history/builder/coverage, start/settlement i locked unseen future samples pozostają wymagane. Oct31/Nov1 bez zmiany/roll;0 network/deployment/wagers/future forecasts. C0, manifest,17 original/prior71 checks i stare sealed NOT_ADOPTED pozostają bez zmian; brak automatycznej kolejki eksperymentów.

### Historyczny trening betting-head — 2026-10-06

Autoryzowany `c0_betting_development_20261006_01` wykonał **18 rzeczywistych fitów headów /0 fitów bazowego C0**. Wynik **NOT_ADOPTED**: H-price LL0.575664709 na514 wspólnych seriach, ale oba primary CI względem M-price/M-shape przecinają0; Brier względem M-shape,±2pp selected-EV i dodatnia wartość po kosztach nie przechodzą bramek. Dodatnia punktowa ROI na15 zakładach nie jest dowodem; zapisany head w diagnostyce prior-context daje−52.19%/−50.72% ROI (24h/300s).

Head używa natywnego prawdopodobieństwa serii, kształtu logitu rynku i kontrastu najlepszych kursów; market-only control ma te same cechy cenowe bez C0. Nie jest zarejestrowany w aplikacji i **nie zastępuje C0 ani Fixed50**. Wyniki są development-exposed, z retrospektywnymi składami/zegarami i niepełnym all-state eksportem kursów. Cash zachowuje523 dojrzałe okazje, również braki kursów jako zero działania; poprawka accounting została wykonana bez refitu, zmiany λ ani nadpisania starych artefaktów.

[Raport i źródła](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md), [aktualna decyzja](../data/08_reporting/c0_betting_development/c0_betting_development_20261006_01/reviewed_decision.json), [CLI](../scripts/README.md). **160 focused/model/benchmark tests passed**, rzeczywiste freeze/train/reassess oraz saved-head transfer; niezależny reviewed replay132,609 assertions/PASS. Rerun odrzucany przed nadpisaniem artefaktów. To nie jest nowy pełny application-suite, live activation ani certyfikacja finansowa. G0, stary10/10, daty Oct31/Nov1 i konfiguracja operational pozostają bez zmian.


### Badanie wejść prior-known — 2026-10-06

Osobno autoryzowany **`c0_input_aligned_20261006_02`** wykonał genuine native preparation oraz **18 formalnych fitów headów/0 fitów bazowych**, wybieranych tylko na CAL, bez refitu. **NOT_ADOPTED**: na382 wspólnych seriach H-price LL0.582673346 nie pokonuje M-price/M-shape (oba97.5% CI przecinają0, Brier wyższy);14 własnych zakładów/ROI+39.48% nie przechodzi±2pp ani bramek dodatniej wartości. Pełny mianownik523 cash zawiera braki jako abstencje. Prior native1,351/1,668;312 braków składów i5 już wydanych wyników blokuje prognozę, nie usuwa okazji. Poprzedni run01 zachowano jako **RETIRED_BEFORE_FORMAL_FITS/0 fits** po wykryciu target self-consumption; nie jest wynikiem modelu.

To nowa hipoteza dopasowania dystrybucji wejść, **nie naprawiona ocena starego NOT_ADOPTED**, dowód przyczyny strat ani aktywacja operacyjna. Historyczne24h kursy, daily release/provider clocks i wcześniej ujawnione wyniki2026 pozostają ograniczeniami;300s diagnostyka ma45 common/0 H-price bets, a matched actual-roster uses te same prior-trained heads bez fitu. **C0 i Fixed50 w tabeli powyżej bez zmian**, heady nie są rejestrowane w aplikacji. [Raport §10](../reports/c0_selected_betting_reliability_research_and_audit_20261006.md#10-deployment-input-aligned-historical-study--2026-10-06), [protokół](../data/05_model_input/c0_input_aligned/c0_input_aligned_20261006_02/protocol.json), [decyzja](../data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_02/decision.json).

Wykonano90 focused/model/benchmark tests (1 istniejący warning), [strict oracle PASS](../data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_02/independent/receipt_corrected_02.json)1,587,107 assertions/18 verification resolves (nie dodatkowe formalne fity); native replay18 reprezentatywnych wierszy max5.81462e−8.9 osobnych SQL cases dotyczą wyłącznie własnego bootstrap Timescale i unchanged core; nie całej aplikacji/migracji ani wszystkich niezależnie odbudowanych tensorów. Wszystkie8 autentycznych source contracts nadal blocked, deadline przed2026-10-31T00:00Z/start2026-11-01 bez przesunięcia;0 acquisition/0 deployment/0 wagering.

CLI ma **cztery rozłączne tryby**; `--run-id` tylko freeze, `--protocol` wymagany w pozostałych, opcjonalny `--research-root`, `--smoke-limit` tylko prepare. Nie ma `--output-dir`: stage paths wynikają z run ID. Poniżej zapis wykonanej sekwencji z rootu repo, **nie instrukcja ponownego zapisu zakończonego runu**; artefakty są immutable, nowy eksperyment wymaga odrębnej autoryzacji:

```bash
RUN=c0_input_aligned_20261006_02
PROTOCOL="data/05_model_input/c0_input_aligned/$RUN/protocol.json"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-input-aligned-study --freeze --run-id "$RUN"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-input-aligned-study --prepare --protocol "$PROTOCOL"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-input-aligned-study --run --protocol "$PROTOCOL"
.venv/bin/python scripts/run_unified_betting_evaluation.py approach-comparison c0-input-aligned-study --readiness --protocol "$PROTOCOL"
```

### Artefakty i lokalne źródła

Źródłem konfiguracji jest `conf/base/c0_serving.json`, nie nazwa najnowszego folderu:

- Model: `data/06_models/a0_data_focus/a0_data_focus_20260929_192443` — `control/seed_{20260910,20260911,20260912}/model.joblib`, `control/calibration/model.joblib`, `shared/model.joblib`, ich receipty i `execution.json`.
- Receipt zakończenia: `data/08_reporting/a0_data_focus/a0_data_focus_20260929_192443/completed.json`. Artefakty, receipty, zamrożone źródła treningowe i aktualna dozwolona ścieżka serving są weryfikowane osobno; nie przepisywać hashy, żeby ominąć błąd.
- Domyślna historia: `data/artifacts/golgg-database-recovery-20260909/matches.json`, SHA-256 `991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca`. Obejmuje 41 915 serii; ostatnia data źródła/release to **2026-09-08**. To jawnie zamrożona historia, nie świeży feed.
- Research root musi zawierać `nonlinear-history-20260912`, `unified-ratings-20260911`, `a0-phase-walkforward-20260914` i `auxiliary-player-20260912`. Mapa korekt release: `a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/release_days.json`.
- Root jest rozwiązywany przez `C0_RESEARCH_ROOT`, `ENSEMBLE_RESEARCH_ROOT`, `RESEARCH_ROOT`, następnie `data/research_root.txt`. Opcjonalne nadpisania: `C0_SERVING_CONFIG`, `C0_MODEL_ROOT`, `C0_RAW_HISTORY_PATH`, `C0_RELEASE_MAP_PATH`.
- Jawny alternatywny plik historii jest walidowany i jego hash/proweniencja są zapisywane. Nie ma cichego fallbacku do `data/golgg_matches.json`. Aktualizacja tego cache lub przebudowa `ratings-v2`/W20 nie aktualizuje automatycznie wejścia C0.
- Zimny replay historii w wykonanych smoke'ach trwał około 200 s. Osobne limity konfiguracji: startup 600 s, request 300 s; rekonstrukcja historii jest częścią requestu. Nie jest to gwarancja opóźnienia.

### Autentyczna aktywacja źródeł nadal BLOCKED — Plan11 S3

Wcześniejsze [source activation receipt](../data/08_reporting/a0_context_reliability/source_activation_20261004_01/receipt.json), [macierz8 źródeł](../data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json) i [native-C0 readiness](../data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_02/forward_readiness/readiness.json) są proweniencją statusu **BLOCKED_EXTERNAL_PREREQUISITES /G0_BLOCKED**; ta aktualizacja dokumentacji nie jest ponownym sourceaudit/transport probe ani zgodą na pozyskiwanie. [Metadata blokad](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/blocked_prerequisites.json) zachowuje braki dla każdego źródła: acquisition/use permission, raw-evidence retention, faktyczne account/territory/market entitlements, measured quotas i remaining capacity; approved exact locators i authentic fixture/unit/player/outcome bindings; retained bytes i rzeczywiste capture/publication/revision clocks.

Do tego konieczne są autentyczny **live release-aware native builder/history/roster i coverage**, match-specific announced five/retractions (nie samo current roster), official actual starts (nie schedule), authoritative results/corrections/settlement/void oraz locked genuinely future opportunity samples z pełnym no-forecast ledger. Parser, public tenant ID, cache, hash albo deklarowana autentyczność tego nie dostarczają. **Osiem źródeł pozostaje nieaktywowanych.** Readiness wymagane **przed2026-10-31T00:00:00Z**, start **2026-11-01**; po niedotrzymaniu **EXPIRED_UNSTARTED**, bez roll/backfill. Dotychczasowe TRAIN/CAL/development i maturity locks pozostają stałe; brak nowej autoryzacji source acquisition, scheduler/notifications, wagering lub deployment.

### Uruchamianie i kontrola bez aktywacji collectorów

Komendy wykonywać z rootu repo, z istniejącymi zależnościami i `DATABASE_URL` wskazującym wyłącznie własną, przygotowaną bazę PostgreSQL/Timescale. Nie inicjalizować, migrować ani przeliczać bazy użytkownika. Sekrety przechowywać poza repo.

```bash
# Inspekcja kontraktu CLI — bez zapisu ani pobierania danych:
.venv/bin/python -m betting_app.scripts.prepare_model_db --help
.venv/bin/python -m betting_app.scripts.run_upcoming_prediction_pipeline --help

# Zapis konfiguracji/predykcji wyłącznie do własnej izolowanej bazy:
.venv/bin/python -m betting_app.scripts.prepare_model_db --register-default-model
.venv/bin/python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05

# API tylko na loopback; nie uruchamia schedulera:
.venv/bin/python -m uvicorn betting_app.api.main:app --host 127.0.0.1 --port 8000
```

Frontend uruchamiany osobno w `client/`: `VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1`; lokalny Vite używa portu 3000. Pipeline zapisuje prognozy/sygnały na istniejących danych, ale nie pobiera nowych kursów; nie używać `--notify` w offline smoke. `--limit` ogranicza etap budowy features, nie wszystkie późniejsze zapytania/zapisy.

Compose przekazuje model/research root/konfigurację oraz montuje model, zamrożoną historię i archiwum research read-only. Hostowy research root wymaga `C0_RESEARCH_HOST_ROOT`; model można wskazać przez `C0_MODEL_HOST_ROOT`. **Obecny Compose montuje `/data/inzynierka/pgdata`, ma stałe nazwy kontenerów i zawiera collector scheduler** — nie traktować `docker compose up` jako disposable/offline smoke. Przed uruchomieniem należy oddzielnie przygotować izolowane storage, porty, sekrety i przejrzeć zadania. W cutoverze nie aktywowano schedulera ani nie wdrożono nowych obrazów.

Do **lokalnego HTTP smoke**, bez pobierania danych, służy jawne połączenie `-f docker-compose.yml -f docker-compose.smoke.yml` (Compose ≥2.24.4), z unikatowym `-p`, własnym `--env-file` spoza repo i wyłącznie już zainstalowanymi obrazami: `up --no-build --pull never`. Własny plik środowiska zastępuje automatyczne wczytanie repozytoryjnego `.env`; należy w nim przekazać tylko kontrolowane wartości wymagane przy interpolacji Compose, jednorazowe hasła PostgreSQL, lokalny research root oraz operatora/hash Caddy. Nie kopiować sekretów produkcyjnych. Smoke usuwa scheduler, sekrety i porty API/DB; PostgreSQL oraz zapisy `/app/data` mają wyłącznie własny tmpfs, a źródła/model/research są read-only. Nie używa `/data/inzynierka/pgdata` ani trwałych wolumenów Caddy.

Sieć smoke jest **rozdzielona**: `default` ma `internal: true`, a API i DB należą wyłącznie do niej, bez wyjścia do Internetu ani portów hosta. Tylko Caddy należy dodatkowo do zwykłego bridge `smoke-ingress` (`internal: false`), aby Docker rzeczywiście publikował `127.0.0.1:${SMOKE_CLIENT_PORT:-3000}:80`. Sam wpis `ports` na sieci wyłącznie internal nie gwarantuje aktywnego listenera hosta. Upstream pozostaje `betting-api:8000` przez sieć wewnętrzną; dodatkowy bridge nie daje API/DB egress i nie aktywuje collectorów. Smoke wymusza `CADDY_APP_HOSTNAME=http://:80`; UI i `/api/*` pozostają za Basic Auth, a dozwolone originy są ograniczone do jawnych lokalnych adresów/portu. Jest to HTTP wyłącznie na loopback, **nie publiczny deploy ani certyfikacja TLS**.

Root Compose publikuje API obowiązkowo na `127.0.0.1`; nie ma obejścia przez `BETTING_API_BIND_ADDRESS`. Uvicornowe `--host 0.0.0.0` dotyczy wyłącznie listenera wewnątrz kontenera. Ewentualny dostęp zdalny musi przechodzić przez uwierzytelniony Caddy i wymaga odrębnej decyzji operatora o hostname, bindzie/portach klienta oraz TLS. Obecne `local_certs` używa lokalnego CA, nie publicznie zaufanego certyfikatu; publiczny TLS nie jest częścią smoke. Nazwa operatora i hash hasła są wymagane poza Git, niezależnie od wyboru HTTP/TLS; CORS nie zastępuje uwierzytelnienia.

Zainstalowany obraz zawiera `/app/.env`, którego zawartości nie odczytano. Smoke maskuje go read-only bindem `/dev/null:/app/.env:ro` i zastępuje środowisko własnymi wartościami; nie oznacza to, że stary obraz jest pozbawiony sekretów. Aktualny `.dockerignore` wyklucza `.env*` z nowych kontekstów, lecz nie usuwa istniejących warstw. Przebudowa obrazu produkcyjnego wymaga osobnej autoryzacji, nie należy budować ani pobierać obrazu w tym smoke.

Kontrola rzeczywistego smoke musi potwierdzić listener hosta przez Docker inspect **i HTTP**, nie tylko `compose config`/health logi: brak uwierzytelnienia ma dawać 401 na UI i trasach API bez mutacji handlerów; z uwierzytelnieniem `/api/health` ma być osiągalne, a kontrolowana operacja wallet ma działać wyłącznie we własnym tmpfs. Osobno sprawdzić brak portów hosta i egress API/DB. To granica uruchomieniowa oprogramowania, nie dowód jakości prognoz, SCA ani gotowości źródeł.
### Plan11 — lokalne naprawy i finalna weryfikacja integralności

Wykonanie **`plan11_integrity_20261006_195949`** zamyka osiągalne lokalne naprawy integralności projektu ([RESEARCH.md](../docs/RESEARCH.md#2026-10-06--plan11-integrity-cutover)). **3428 passed,36 warnings** w pełnym `betting_app/tests` na własnym PostgreSQL/Timescale; finalne `npm run build` i `npm run lint` przeszły. To wynik regresji software, nie certyfikacja finansowa, turniejowa ani deploy.

- **Schemat:** forward-only `e203c0060011` naprawia brakujące alerts/props/defaults/UTC; `e203c0060012` wyrównuje fractional match confidence do DOUBLE PRECISION.51 tych samych przypadków przeszło na ORM, pustych migracjach, bootstrap i owned upgrade (204 executions,51 distinct). Dawnej precyzji utraconej przez INTEGER nie da się odzyskać. Legacy naive quote clocks interpretowane jako UTC nie stają się autentycznymi zegarami publikacji.
- **Rzeczywiste trzy ścieżki PostgreSQL:** parser→quote→genuine C0→persistence/2 mean-EV signals, brak wymaganych bounds→`missing_uncertainty`, HTTP wallet win→18.49 payout/108.49 balance, wyłączone alerts→0 dispatch i prop save/read przechodzą bez `create_all` nad migracjami.
- **Roster:** snapshot wiąże canonical fixture, orientację, BO, start/context i dokładne ID/kolejność. Posortowane transaction advisory locks chronią również brak rosteru; każdy fixture batcha ma oddzielną atomową transakcję. Błąd PostgreSQL nie może po cichu zwolnić tych locks przez wewnętrzny rollback. Rzeczywista blokada konkurencyjnego writera i oba maintenance writers przeszły smoke.
- **Money:** centy HALF_UP, kurs Numeric8,4 przed walidacją/payout, tax Numeric4,2 z jawnymi granicami. Won używa zapisanych wartości; void/cancel/loss nie walidują nieużywanych pól legacy. Sprawdzono brak overdraw/double credit i rollback bet/balance/ledger.
- **UI:** rzeczywisty Vite→loopback API→PostgreSQL pokazuje genuine C0, source SHA,2026-09-08 i `current_stored_roster`/`authenticated=false`. Kontrolowane kolejności odpowiedzi sprawdzają model/bookmaker/Open-Mid-Close/route/30-day validation oraz późne BO po zatwierdzonym rosterze. Oddzielny kontrolowany render provenance turniejowego nie jest symulacją. Brak p/EV/bounds/conditional ROI pozostaje nieznany/null; researchσ0.15 jest jawnie założony, nie zmierzony.
- **Boundary:** własny Compose/Caddy dał401 przed handlerem na80 operacjach OpenAPI i5 UI/docs paths; autoryzowane health/wallet oraz jawne CORS przeszły. API/DB mają tylko internal network, Caddy wyłącznie loopback ingress;0 collector jobs/request-log/scrape records. Trusted-local direct API nie ma odrębnego auth.
- **Proxy:**16 scenariuszy×32 drawy,6 prior-CAL scalar fits, żadnych nowych formalnych native/head/base fits. Niezależny saved replay:3552 probabilities,48 public reverse checks,max2.22e-16; nowy32-draw CLI przechodzi także po równoległej zmianie kodu formatów. Decision pozostaje diagnostic, nie native adoption.

[Dokładne receipts i zachowane błędne próby](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/), w tym [pełny suite](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_application_suite_corrected.json), [trzy native pipelines](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_three_schema_native_runtime.json) i [actual browser](../data/08_reporting/project_integrity/plan11_integrity_20261006_195949/final_browser_genuine_native_provenance.json). Zachowano wcześniejsze failed gates, błędne fixture/oracle assumptions i unknown standalone API failure; nic nie przemianowano na sukces. Pięć protected hashes i canonical doctor są niezmienione/READY. Równoległy manifest native-research i późniejszy kod formatów zachowano, nie przypięto wstecz do starego snapshotu.

Nie wykonano source acquisition, schedulera/notification transportu, deployu, operacji na bazie użytkownika ani oddzielnego native-study/Plan10. Source activation pozostaje **BLOCKED_EXTERNAL_PREREQUISITES/G0_BLOCKED**; autentyczna whole-edition/pre-start certyfikacja pozostaje niewykazana. Oryginalne C0/A0 artefakty, wcześniejsze negatywne wyniki, stopped10/10 i Oct31/Nov1 pozostają bez zmiany.


### Wykonana weryfikacja lokalnego cutoveru

- Pełny `betting_app/tests` na izolowanym PostgreSQL/Timescale: **2291 passed, 37 skipped, 36 warnings**. To regresja oprogramowania na schemacie ORM, nie nowy dowód migracji ani jakości finansowej.
- Frontend: `npm run build`, `npm run lint` oraz rzeczywista kontrola UI/API/persistence.
- Replay 3103 zapisanych przypadków × 3 tryby = **9309** prawdopodobieństw; maksymalna różnica `2.13e-9`. Rzeczywiste CLI predykcji, pojedynczy logit shrinkage i Monte Carlo zwycięzców serii przeszły smoke.
- Przypadek jednego brakującego ID organizacji: rzeczywisty native worker ma dokładną zgodność z jawnym `no_organization`; błąd komplementarności `5.82e-9`. Rzeczywisty błąd SQL w źródle ENC daje ogólne HTTP 500, bez ujawnienia diagnostyki.
- Niezależne przeglądy końcowe nie zgłaszają pozostałych actionable defects w sprawdzonym zakresie. Własne procesy i bazy smoke zostały zatrzymane.

Nie wykonano live acquisition, deployu, nowego formalnego treningu ani operacji na bazie użytkownika. Zamrożona historia, dzienne zegary release i scenariuszowo znane składy nie certyfikują autentycznej prognozy przyszłej. Polityki finansowe wymagające zweryfikowanych granic pozostają fail-closed; same średnie C0 ani nazwa „shrinkage” nie ustanawiają rentowności.

### Naprawy kontraktu kursów i EV — 2026-10-06

Run **`c0_market_contracts_20261006_01`**: lokalne naprawy i weryfikacja software, nie wynik strategii bukmacherskiej. Dowody: `data/08_reporting/c0_market_study/c0_market_contracts_20261006_01/`; wykonanie opisuje końcowa sekcja `reports/c0_selected_betting_reliability_research_and_audit_20261006.md`.

- Orientacja sprawdza obie nazwy źródłowe i obie kanoniczne w jednym kontekście aliasów (liga/źródło/data). Nieznana/niejednoznaczna para, jedna rozpoznana strona i niepoprawne ceny nie przechodzą przez fuzzy/raw fallback ani drugie odwrócenie.
- `prematch_quote_service.py` wybiera najnowszy **znany stan przed sprawdzeniem poprawności**. Późniejsze zawieszenie, live, niekompletna para i nieznane `is_live` nie przywracają starego kursu. Bieżąca akcja wymaga jawnego prematch, wieku ≤300 s, strefowych zegarów i znanego startu po decyzji.
- Hybryda zapisuje pełny kontekst konsensusu. Producer i konsumenci sprawdzają również aktualne wartości i stan źródłowych kwotowań, nie tylko ID. Gołe nadpisanie q nie wystarcza; zmiana ceny pod tym samym ID, stary peer lub niezgodna wersja unieważnia zapisany kontekst. Sygnał ma własny rzeczywisty `evaluated_at`; walidacja poprzedza limit wyników.
- Wyświetlany EV to surowe `p * odds * .88 - 1`; kwalifikacja jest osobną, niezmienioną polityką. Jej minimum obejmuje dodatkowy hurdle dla małych kursów. Nie wymyślamy lower bound ani sigma dla C0.
- Wyniki historyczne są **`historical_quote_recomputation`**, nie odtworzeniem wydanej prognozy. Każdy wybrany kurs korzysta z wcześniejszego C0 i znanych wtedy kwotowań wszystkich bukmacherów. Open/Mid/Close nie używają dzisiejszego filtra wieku; Mid oznacza środek czasu od pierwszej obserwacji do startu. Filtr bukmachera zmienia widok, nie konsensus. Brak p/EV jest `null`, nie0; poprawna cena pozostaje widoczna.
- Native evidence przygotowuje wejście i liczy C0 tylko raz: cztery pola transportu, osiem hashy tensorów, rzeczywiste sloty i odrębne zegary child/parent/DB. `c0_shadow_readiness.py` wymaga zachowanych bajtów, faktycznego buildera, autentycznego bindingu źródła i jawnej drogi announcement/prior-unit; sam receipt ani deklarowany hash nie ustanawia G0.
- Power devig rozwiązuje dodatni wykładnik także dla dużego overround; błąd nie zmienia metody po cichu. Dla1.01/1.02 wynik to0.6172218121/0.3827781879, nie proporcjonalne0.5024630542/0.4975369458.

**Wykonano:**491 focused/benchmark tests, jeden istniejący warning; build/lint frontendu; pełny canonical suite rescore11550 (`DEVELOPMENT_ONLY`,039 brak1643); rzeczywiste parser→DB→producer na ORM, pustych migracjach PostgreSQL i bootstrap SQL. Loopback HTTP:29 odpowiedzi,2 poprawne sygnały native,4 Fixed50,2 dodatnie syntetyczne parlay; alert service:4 symulacje przy wyłączonych kanałach i0 prób transportu zewnętrznego. Browser sprawdził oba modele, Open/Mid/Close, filtr bukmachera i widoczne ceny przy niedostępnym EV.

**Genuine native smoke:** ordinary/evidence parity0;9 przypadków mode×BO, side-swap max1.13e-8, rotation/timeout/nonmutation. W pełnym źródle41915 serii perturbacja74 przyszłych serii/253 map nie zmieniła żadnego z8 tensorów ani wcześniejszego p. Cold start0.500000000346004 oraz empty0.5000000109400433 mieszczą się w odziedziczonym2e-6; nie są bitowo0.5.

**Granice:** nie uruchomiono pełnego application suite ani migracji/deployu produkcyjnego. Czysty native migration chain nie zawiera istniejących tabel alertów; alert smoke użył deklarowanego schematu ORM, bez naprawczego DDL. Poboczny prop-analysis w izolowanej bazie zwrócił HTTP500; nie naprawiano go w tym zakresie.

### Deadline transportu i dostępność historycznego targetu — 2026-10-07

- `c0_inference.py` zapisuje ramki nieblokująco, obsługuje częściowe zapisy i używa jednego deadline dla serializacji ramki, zapisu i odpowiedzi po handshake. Startup pozostaje osobny. Shutdown nie blokuje bez końca przy pełnym pipe; ubijanie/reaping własnego child może dodać czas po upływie deadline. Limit nie obejmuje kolejki na lock ani przygotowania wejścia.
- Wspólny replay `a0_input_repair_asof_v4` sprawdza niezależny native release i znany label release przed lookupem składu oraz przed zapytaniem o cechy. Już opublikowany target pozostaje w ledgerze z `target_already_released_before_cutoff`; nie dostaje własnego wyniku w cechach ani p=.5. Granica jest dzienna UTC: release tego samego dnia nie wchodzi do historii. Historia targetu nadal służy późniejszym innym meczom.
- Przepięto wyłącznie bieżący hash buildera w config/loaderze i reuse historii turniejowej. Wagi, kalibratory, model/feature identity i zamrożone protokoły pozostają bez zmian. Dawne receipty v3 nie są przepisywane ani recertyfikowane.
- Rzeczywisty replay wszystkich3,103 wierszy2026 zachowuje primary full C0 bitowo: LL0.5354487005519923/Brier0.18103449300382884. Starszy ledger annual-A0 zachowuje4,626 originów, ale wstrzymuje6 już znanych targetów:3,585→3,579. Poprawione wcześniej1,668 okazji C0 nie zmienia eligibility.
- Public-client smoke:9 rzeczywistych BO×mode raw-history prognoz jest identycznych z runtime sprzed fixu; zatrzymany child/pipe4KiB daje `C0WorkerTimeout` dla1s write budget, a własny proces jest reapowany. Całość3.115s obejmuje termination, nie jest obietnicą końca wywołania w1s.
- Regresje:186 różnych focused cases PASS —175 przy standardowej konfiguracji,11 genuine cold-history contracts z jawnym900s budżetem **wyłącznie fixture testowego**, bez zmiany domyślnego request300s. Pierwsze równoległe timeouty są zachowane; dostępność cold-startu przy obciążeniu nadal nie jest certyfikowana. To nie jest pełny application suite.
- Dowody i zakres: `data/08_reporting/c0_integrity_fix/c0_integrity_20261007T073326Z/` oraz aktualna sekcja `docs/RESEARCH.md`. To naprawa software/admission, nie deploy, nowy model ani potwierdzenie EV.
- Canonical doctor READY; rzeczywisty suite rescore wszystkich11,550 wierszy pozostaje DEVELOPMENT_ONLY: annual A0 LL0.551246/Brier0.187103,039 missing1,643. To osobne zapisane kontrole, nie nowy model C0 ani trening. [Report](../data/08_reporting/benchmark/c0_integrity_20261007T073326Z/REPORT.md).


### Osobne, bramkowane badanie native C0 + rynek

`scripts/run_unified_betting_evaluation.py c0-market-study` używa nowego protokołu `native-c0-market-study-v1`; nie podstawia rocznego A0. S/S-cal/M/M-cal/M-shape/Fixed50/H, cztery rodziny × lambda0.01/0.1/1.0 = maksymalnie12 formalnych fitów TRAIN. CAL wybiera lambda bez refitu; zamrożone fazy, support, źródła, receipty i hash kodu są sprawdzane przed każdym etapem. Ocena ma pełny miesięczny wszechświat/5000 drawów, wspólną populację proper scores, denominator okazji z abstencjami oraz jawne niedefiniowane conditional CI.

W tym wykonaniu rzeczywisty `--freeze` i `--no-fit` dały **`G0_BLOCKED / BLOCKED_EXTERNAL_PREREQUISITES`,0 real fits/0 real models**. Oddzielny `--synthetic-smoke`:600 wierszy,12 synthetic fits,7 zapisanych recipes, **`SYNTHETIC_SOFTWARE_ONLY`**. Niezależnie odtworzono12 optima i wszystkie score/cash/bootstrap ścieżki zadeklarowane w `independent_numeric_receipt.json`;37734 numeric/exact assertions (obejmują powtórzenia), max scalar2.84e-14. To nie jest dowód przewagi H ani rentowności.

Protokół tego runu: `data/05_model_input/c0_market_study/c0_market_contracts_20261006_01/protocol.json`. Przykład oddzielnego, nowego wykonania software (zmienić RUN na nieistniejący; nie uruchamia realnego TRAIN):

```bash
RUN=c0_market_contracts_manual_02
.venv/bin/python scripts/run_unified_betting_evaluation.py c0-market-study --freeze --run-id "$RUN" --output-dir "data/05_model_input/c0_market_study/$RUN"
.venv/bin/python scripts/run_unified_betting_evaluation.py c0-market-study --protocol "data/05_model_input/c0_market_study/$RUN/protocol.json" --no-fit --output-dir "data/08_reporting/c0_market_study/$RUN/admission"
.venv/bin/python scripts/run_unified_betting_evaluation.py c0-market-study --protocol "data/05_model_input/c0_market_study/$RUN/protocol.json" --synthetic-smoke --output-dir "data/08_reporting/c0_market_study/$RUN/synthetic"
```

Aktywacja nadal wymaga prawdziwych uprawnień/retencji/limitów, locators/bindings, released history/rosters, startów/settlement i zamkniętych przyszłych próbek. Osiem source contracts pozostaje blocked, historia kończy się2026-09-08. Termin gotowości **przed2026-10-31T00:00Z**, start2026-11-01 i późniejsze locki TRAIN/CAL/development nie przesuwają się automatycznie.0 genuine future forecasts,0 external acquisition,0 wagering; stare10/10, ±2pp, manifest i oryginalne artefakty pozostają niezmienione.


### OddsPapi — klucz kolektora

- `ODDSPAPI_API_KEY` należy przechowywać poza Git w prywatnym `.env` z uprawnieniami `0600`. Przy przenoszeniu między maszynami kopiować tylko ten klucz, nie cały lokalny plik środowiska.
- Compose przekazuje klucz wyłącznie do `betting-scheduler`; samo dodanie go do hostowego `.env` nie wystarcza bez wpisu w `environment` i odtworzenia kontenera. Pusty klucz celowo pomija pobieranie.
- Po autoryzacji zbierania danych aktualizacja wyłącznie schedulera na zweryfikowanej produkcji (`melzak@192.168.1.17`, `/data/inzynierka/app`, projekt `inzynierka`) używa istniejącego obrazu, bez przebudowy/pull i bez restartu zależności:

  ```bash
  docker compose -p inzynierka -f docker-compose.yml up -d --no-deps --no-build --pull never --force-recreate betting-scheduler
  ```

- Horizon Pinnacle działa o `:05` i `:35` UTC, T−1h ±30 min, do dwóch zapytań na uruchomienie. Fixture sync ma cron `0 3 */3 * *`. Domyślne limity aplikacji pozostają 8 zapytań w ostatnich 24 h i 250 w ostatnich 30 dniach; liczniki innych instancji ani faktyczne uprawnienia konta dostawcy nie są przez nie weryfikowane.
- Weryfikacja produkcyjna **2026-10-08**: odtworzono tylko scheduler, runtime otrzymał lokalny klucz, pojedynczy fixture sync pobrał 9 fixtures i dopasował 7 do canonical matches. Smoke horizon nie miał kandydatów i nie pobrał/zapisał kursów. Sam status automation `completed` nie dowodzi pobrania — sprawdzać request logs, mappingi i zapisane kwotowania.
- To konfiguracja istniejącego kolektora, nie wdrożenie lokalnego adaptera kursów ani potwierdzenie G0, autentycznej emisji prognoz, przyszłego protokołu czy bezpieczeństwa zakładów.

## Archiwalne procedury i dowody

Poniżej zachowano starszą dokumentację MVP, operacji i eksperymentów. **To archiwum, nie instrukcje uruchomienia obecnego C0**: opisy SQLite, dawnych modeli, harmonogramów, domyślnych ratingów i polecenia aktywujące kolekcję mogą nie odpowiadać aktualnemu checkoutowi. Datowana obserwacja 2026-09-26 dotyczyła produkcyjnej hybrydy EXP081 + rynek; nie stanowi bieżącej inspekcji serwera. Współczesny kontrakt i ścieżki konfiguracji są wyłącznie w sekcji C0 powyżej.

## Historyczny status MVP

MVP v0.2:

- SQLite schema i lokalna baza `data/betting_app.sqlite3`,
- **FastAPI backend** (`betting_app/api/`) — REST API dla frontendu React,
- **React frontend** (`client/`) — dashboard z listą meczów, rankingami drużyn i zawodników, szczegółami, statusem systemu, schedulerem i analizą timingową,
- nadchodzące mecze, max/średnie kursy A/B, arbitraż, model, hybryda i karta szczegółów meczu,
- generowanie sygnałów EV+ jako diagnostyka,
- manualny bet tracker,
- osobne portfele/konta per bukmacher i historia transakcji,
- rozliczanie zakładów i bankroll events,
- znormalizowany tracking kursów: `scrape_runs` → `bookmaker_events` → `bookmaker_markets` → `odds_outcome_snapshots`,
- scraper STS League of Legends prematch przez `sbk-exporter/v1/sports/ssr`,
- scrapery NoDriver/API dla Betclic, Superbet, eFortuna, Betfan, TOTALbet i Lebull.

## Uruchomienie

### Docker Compose (rekomendowane)

```bash
docker compose up --build -d betting-api betting-scheduler client
```

Frontend będzie dostępny na `http://localhost:3000`, API na `http://localhost:8000`.

### Lokalnie (bez Dockera)

Backend:

```bash
python -m betting_app.scripts.init_db
uvicorn betting_app.api.main:app --reload --port 8000
```

Frontend:

```bash
cd client
npm install
npm run dev
```

Frontend będzie dostępny na `http://localhost:3000`.

Docelowo do scrapowania:

```bash
pip install nodriver
```

## Jak to działa — aktualny obraz systemu

System składa się z pięciu warstw:

1. **Dane GOL.GG**
   - surowy cache: `data/golgg_matches.json`,
   - relacyjna baza PostgreSQL/TimescaleDB: `golgg_matches`, `golgg_games`, `golgg_game_players`, `golgg_teams`,
   - aktualizacja zakończonych meczów co 2–3 dni przez `refresh_golgg_results`,
   - scraper GOL.GG jest skopiowany do projektu jako `betting_app/scrapers/golgg.py`, więc zwykłe działanie aplikacji nie wymaga zewnętrznego checkoutu `embedded-rift`.

2. **Kursy bukmacherów**
   - scrapery LoL prematch match-winner dla: STS, Betclic, Superbet, eFortuna, Betfan, TOTALbet, Lebull,
   - zapis do `odds_snapshots` oraz znormalizowanych tabel `scrape_runs`, `bookmaker_events`, `bookmaker_markets`, `odds_outcome_snapshots`,
   - każdy snapshot ma `source_url`, a jeśli da się ustalić także `offer_url` do konkretnej oferty.
   - lokalny adapter OddsPapi używa wspólnego parsera `src/analysis/quote_provenance.py`: ostatni stan nieaktywny/błędny unieważnia starszą cenę; kurs źródłowy i czas faktycznego odbioru pozostają osobnymi polami,
   - `createdAt` dostawcy porządkuje stany i jest zwracane jako `source_created_at` w podsumowaniu strony; nie jest zegarem odbioru. Parser pozostawia `observed_at=null`, gdy brak takiego dowodu. Rzeczywisty `response_observed_at`/`scraped_at` ustala osobno serwis podczas pobrania; naprawa nie certyfikuje archiwalnej dostępności kursu,
   - zweryfikowany binding Pinnacle wymaga jawnych `participant_bindings` w Pythonowym `fetch_pinnacle_horizon_odds`: fixture/book/market/outcome, identyfikatory uczestnika i bukmachera, nazwa gdy fixture nie zawiera identyfikatora uczestnika oraz `published_at`, `observed_at`, `evidence_uri`. Generyczne wyniki181/182 nie wystarczają. Bez zgodnego dowodu obserwacja jest zachowana jako **quarantined**, z niedopuszczonymi/null cenami; najnowszy taki stan nie przywraca starszego kursu. Liczniki to `saved = admitted + quarantined`; `admitted` opisuje binding, nie samodzielne uwierzytelnienie źródła ani aktualną wykonalność ceny,
   - dowód mapowania musi być dostępny nie później niż cut-off i przed startem. Przyjęcie takiego rekordu nie uwierzytelnia samo jego źródła i nie certyfikuje wykonania zakładu. Historycznych rekordów w bazie nie migrowano ani nie recertyfikowano; zmiany lokalne nie zostały wdrożone.

3. **Mapowanie meczów**
   - `canonical_matches` łączy ten sam mecz między bukmacherami,
   - kursy są wyrównywane do canonical team A/B, więc odwrócone strony u bukmachera nie psują best odds.

4. **Model operacyjny upcoming**
   - `entity_ratings`: Elo/Glicko-2/TrueSkill/OpenSkill/PL/TM dla teamów i graczy,
   - `team_rolling_features`: W20 z GOL.GG,
   - `upcoming_match_features`: feature vector dla przyszłego meczu,
   - roster upcoming = ostatni zaobserwowany roster drużyny w GOL.GG,
   - `canonical_predictions`: predykcje operational model oraz hybrydy model+rynek,
   - `model_ev_signals`: EV po podatku 12%.
   - audyt 2026-09-26: produkcja faktycznie używa `Hybrid-Operational-Market` z bazowym `Symmetrized-Siamese-Series-EXP081`, nie Causal A0/A1,
   - ówczesna lokalna naprawiona hybryda miała wersję `exp081-siamese-series-v1-a0.50-t1.00-contract-v2`; to historyczny kontrakt sprzed cutoveru C0, nie aktywna wersja checkoutu,
   - Consolidated A1 pozostaje badawczy i jest odrzucany w inferencji operacyjnej. Lokalny adapter zamrożonego Causal A0 jest zweryfikowany na fixture'ach; nie jest adapterem żywego buildera cech ani aktywnym silnikiem aplikacji. Zmian lokalnych nie wdrożono.

   Pełny [audyt architektury i produkcji](../reports/production_readiness_and_market_benchmark_report_2026.md) zawiera wyniki, ograniczenia i dowody. Zweryfikowany dostęp tylko do odczytu: `ssh melzak@192.168.1.17`, checkout `/data/inzynierka/app`; inwentaryzacja przez `docker ps`, health przez `curl -fsS http://127.0.0.1:8000/health`. Domyślne `docker compose ps` w checkoutcie pokazuje tylko klienta, więc nie zakładaj wspólnego zarządzania wszystkimi usługami.

   Osobno wykonany counterfactual A0 z30.09 nie uzasadnia podmiany silnika EV:
   natywny A0 na wcześniejszych składach, podatek12% i stawka100, po bieżących
   filtrach kwalifikacji: OPEN56 zakładów, **ROI−24,15%**;24h47,−13,18%;
   48h36,+2,54% z szerokim przedziałem niepewności. To scenariusz ze średnią
   prawdopodobieństwa A0, bez wymyślonej dolnej oceny i bez dowodu historycznej
   widoczności boardu/wykonania. Model produkcyjny nie został zmieniony.
   [Wyniki, pokrycie i ograniczenia](../docs/RESEARCH.md).

5. **UI / operacja**
   - React frontend (`client/`) — lista meczów, szczegóły, status systemu, scheduler, analiza timingowa.
   - FastAPI backend (`betting_app/api/`) — REST API z endpointami dla matches, predictions, bets, system, scheduler, timing.
   - historyczny widok ENC używał ratingów GL i domyślnego 1500.0. Po cutoverze model prawdopodobieństwa to C0 z rzeczywistymi ID graczy, bez takiego fallbacku; brak składu pozostaje jawny.

### Lokalny shadow A0 i niezmienne dowody — Plan06 E1/E2

- `betting_app.core.models.causal_a0.CausalA0Adapter(year, research_root=None)` przyjmuje natywny `AsOfSnapshot` z `src.analysis.ev_asof_replay`; lata2024–2026, BO1/3/5, tryby `full`, `no_w20`, `no_organization`. Wynik `predict_snapshot(snapshot, mode=...)` jest już prawdopodobieństwem całej serii; **nie wykonywać drugiej projekcji BO**. Nie udaje EXP081 i nie generuje dolnych granic niepewności.
- Loader weryfikuje przypięty manifest, receipt kalibracji, checkpointy i źródła natywne przed importem/unpickle. Wymaga pełnego lokalnego archiwum wskazanego przez `--research-root` w runnerach, `ENSEMBLE_RESEARCH_ROOT`, `data/research_root.txt` albo jawny argument adaptera. Nie zastępuje brakującego artefaktu innym modelem. Uruchamiać w oddzielnym procesie lokalnym: natywne moduły archiwum używają współdzielonych nazw importów `src.models`; to nie jest wdrożenie do workera API.
- `betting_app.services.a0_shadow_readiness.record_a0_snapshot(session, adapter, snapshot, evidence_key=..., sources=..., input_source_ids=..., schedule=..., quotes=..., mode=...)` łączy rzeczywistą inferencję z zapisem. Kursy i terminarz nie trafiają do modelu. Caller odpowiada za commit/rollback; nowa inferencja ma nowy `generated_at` i nowy klucz. Do ponowienia zapisu tego samego envelope służy `append_forecast_evidence`; `load_forecast_evidence` sprawdza integralność przy odczycie.
- Osobna tabela `forecast_evidence_snapshots`, migracja `f1e2d3c4b5a6`, przechowuje dokładne bajty/dtype/kształty tensorów, metadane składu i kontekstu, model/calibrator hashes, terminarz, dwustronne kursy/bindings oraz źródła `{source_id, evidence_uri, content_base64, sha256, retrieved_at, published_at, observed_at}`. Brakujące czasy pozostają `null`; rzeczywiste `generated_at` i serwerowe `recorded_at=clock_timestamp()` są oddzielne od `simulated_origin_at`.
- Ten sam klucz i treść są idempotentne; sprzeczna treść jest odrzucana. Trigger PostgreSQL blokuje UPDATE/DELETE, również przez bezpośredni SQL. To nie ochrona przed administratorem bazy: TRUNCATE, DDL lub wyłączenie triggera pozostają poza gwarancją. Hash potwierdza bajty, nie autentyczność źródła ani kwalifikację G0.
- Pilot obejmuje9fixture'ów, trzy roczne receptury i trzy tryby; adapter/native Δp=0. Osobno sprawdzono pustą bazę→pełne migracje, `metadata.create_all`, pełny bootstrap Timescale, konkurencyjne zapisy i zachowanie brakujących metadanych pandas. Kursy/bindings pilota są jawnie syntetyczne; nie odzyskano historycznych dowodów. [Receipts i eksport](../data/08_reporting/betting_ev_reliability/a0_local_readiness_20260930_222642/) oraz [Plan06](../docs/RESEARCH.md#2026-10-01--plan06-a1-a2).

Testy granic adaptera/persistence: `.venv/bin/python -m pytest -q betting_app/tests/test_causal_a0_adapter.py betting_app/tests/test_forecast_evidence.py` z `DATABASE_URL` i `TEST_DATABASE_URL` ustawionymi na wcześniej zmigrowaną, izolowaną bazę PostgreSQL. Nie używać konfiguracji bazy użytkownika. Live collection, budowa żywych tensorów A0, trening i wdrożenie pozostają poza wykonanym zakresem.

### Repaired fixture input i status dowodów — Plan07

`scripts/predict_causal_a0.py` wykonuje lokalny replay surowej historii, wybór wcześniej
znanego składu w dokładnym kontekście rozgrywek i natywną inferencję. Wymaga
`--fixtures`, `--metadata`, `--source-matches`, `--release-days` i nowej ścieżki
`--output`; opcjonalne są `--roster-evidence`, `--history-captures`,
`--roster-captures`, `--serving-contract annual|fixed-2026`, `--research-root`,
`--mode` i `--max-fixtures` (domyślnie250 żądań; limit nie ogranicza historii).
Pliki fixtures/metadata są tablicami JSON. Fixture zawiera jednoznaczne identyfikatory,
BO, strefowe `decision_at` lub `cutoff_at` oraz `start_at`; cutoff musi poprzedzać start.
`decision_at` i `cutoff_at` są tutaj aliasami jednego originu: jeśli podano oba,
muszą oznaczać tę samą chwilę. API nie obsługuje przez te aliasy dwóch odrębnych
originów; rzeczywisty czas utworzenia prognozy pozostaje osobnym `generated_at`.
Jawny kontekst rozgrywek fixture'a uzupełnia brak w istniejącym rekordzie metadata,
ale konflikt z metadata jest odrzucany, nie zastępowany innym kontekstem.
Mapa release jest mapą korekt: wspólna polityka bierze późniejszy dzień z korekty
i ostatniej mapy, także dla wejścia iterable. Nie dopisuje fikcyjnych dat publikacji.
Nowsze dostępne `status="retracted"` może nie zawierać piątki; blokuje prognozę
z przyczyną `latest_announcement_retracted`, zamiast przywracać starszy skład.

`src.analysis.a0_prediction_input` udostępnia `predict_fixture_inputs`,
`infer_fixture_snapshots` i `snapshot_evidence_status`. Status rozróżnia
`SUPPORTED_INPUT`, `LIMITED_EVIDENCE` i `UNAVAILABLE`; zawsze zachowuje
`predictive_reliability=NOT_ASSESSED` oraz nieuwierzytelnioną podstawę informacji.
Obecność tokenów historii nie oznacza obecności statystyk. Pusta maska nie ukrywa
braku daty źródła dla niezerowych cech surowych. Niepoprawne zegary scenariusza
i cechy dostępne dopiero po deklarowanym cutoffie są odrzucane.
`record_a0_snapshot` zachowuje ten status w `input.evidence_status`.

Najpierw zbudować wszystkie snapshoty, następnie uruchomić natywną inferencję
w tym samym izolowanym procesie. Ponowna aktywacja historycznego backendu cech
po inferencji nie jest obsługiwanym kontraktem; nie używać tego jako workera API.
CLI nie pobiera danych, nie zapisuje do bazy, nie nadpisuje istniejących wyników
i nie zmienia modelu operacyjnego.
Końcowy pełny replay CLI dla tych samych9 żądań trwał275.75s i dokładnie
odtworzył4 dostępne prawdopodobieństwa oraz5 przyczyn braku prognozy.
To narzędzie batch, nie dowód gotowości do inferencji live. Próba z limitem240s
przekroczyła deadline; osobny diagnostyczny proces z okresowym dumpem stosu
zakończył się kodem139 z nieustaloną przyczyną. Zwykły CLI następnie przeszedł
bez zmian kodu/modelu. [Receipt i zachowane próby](../data/08_reporting/model_reliability/model_reliability_20261001_055321/cli_smoke_verification.json).

Wykonany [dowód Plan07](../data/08_reporting/model_reliability/model_reliability_20261001_055321/input_recording_proof.json):
9 historycznych żądań,4 prognozy i5 zachowanych braków składu; BO1/3/5,
12 rzeczywistych envelopes w trzech trybach na izolowanej zmigrowanej bazie.
Maksymalny błąd symetrii wyniósł `1.33e-8`; sprawdzono pustą historię,
idempotencję, rollback i zakaz SQL UPDATE/DELETE. Usunięcie76 serii niedostępnych
według deklarowanego cutoffu z historii2235 serii oraz podmiana przyszłego wyniku/składów
nie zmieniły wcześniejszych tensorów ani prawdopodobieństwa. To dowód oprogramowania
na scenariuszu historycznym: **zero autentycznych prognoz przyszłych** i brak
certyfikacji predykcyjnej lub finansowej.

### Protokół pozyskania — Plan06 E3, bez aktywacji

[Protokół](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/protocol.json) i [kontrakty źródeł](../data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/source_contracts.json) definiują T−1h, świeżość odbioru kursu300s, okno zapisu/decyzji60s i minimum dwa różne booki z allowlisty Pinnacle/TOTALbet/Lebull. To plan, **nie uruchomiony collector ani nowe flagi CLI**. Aktualny helper OddsPapi pobiera wyłącznie Pinnacle i nie zapewnia takiego quorum.

E3 offline proof:79 kontroli,9 zweryfikowanych envelopes/72 tensory,zero dopuszczonych syntetycznych dowodów. Adapter roczny nadal odrzuca cutoff2027. Późniejszy Plan09 zweryfikował osobno niezmieniony artefakt2026 na jawnie ograniczonym przyszłym zakresie oraz lokalne captured tensors/bytes/ledger — opis poniżej. Nie jest to autentyczna kolekcja, aktywny worker ani spełnienie G0. Nie produkować sztucznego `forecast.generated_at`, aby upchnąć brak prognozy do envelope. Brak `published_at` pozostaje nieznany, nie jest kopią obserwacji.

Żaden dostęp do źródeł nie został zweryfikowany live. Wymagana osobna zgoda, warunki dostępu/retencji, dokładne oficjalne locatory i budżet przed ograniczonym probe'em; nie uruchamiać pełnych scraperów ani domyślnej bazy. [Plan06](../docs/RESEARCH.md#2026-10-01--plan06-a1-a2) podaje TRAIN/CAL/development, termin gotowości2026-10-31 i wygasający start2026-11-01. E4 potwierdzenie nie jest rozpoczęte ani automatycznie autoryzowane.

### Wykonany lokalny capture/ledger i fixed native contract — Plan09

[Raport wykonania](../reports/plan09_c0_connector_and_forward_evidence_20261002.md) i [decyzja aktywacji](../data/08_reporting/betting_ev_reliability/forward_capture_20261002_033658/decision.json): **lokalne E1–E4 zweryfikowane, E6 BLOCKED**,0 requestów do zewnętrznych źródeł i0 genuine future forecasts. Nie zmieniono modelu operacyjnego, schedulera ani serving DB.

- `forecast_evidence.capture_source_attempt` utrwala dokładne body bytes w prywatnym content-addressed store przed odwołaniem SQL. Osobne rzeczywiste request/response/retrieval/first-observation/observation clocks, nullable publication, typ reprezentacji i błędy transportu/parsera pozostają zachowane. `load_source_capture` weryfikuje hashe; `audit_source_capture_store` wykrywa brakujące/uszkodzone/osierocone bytes i przerwane pliki staging. Metadata usuwa sekrety, body nie jest przepisywane jako „dokładne” po redakcji.
- `OddsPapiClient.get_captured(..., session=..., capture_id=..., source_id=..., evidence_uri=..., body_store=...)` jest jawnym opt-in z caller-owned transakcją i request metering. Zapis poprzedza JSON parsing; HTTP errors/malformed/timeout nie stają się udanym pustym wynikiem. Zwykłe `get` i live callerzy nie zostały automatycznie przełączone.
- Osobne `source_captures` i `opportunity_events` w istniejącym obszarze modeli; migracja `d4e5f6a7b8c9` po `f1e2d3c4b5a6`. ORM, Alembic i `docker/timescale/init.sql` mają te same wymagane constraints/triggery. `append_opportunity_event`, `load_opportunity_events`, `orchestrate_local_opportunity_batch` i `export_opportunity_ledger` zachowują brakujące prognozy zamiast tworzyć fikcyjne envelopes. Commit/rollback należy do callera; SQL UPDATE/DELETE zabronione, owner DDL/TRUNCATE poza gwarancją.
- Orchestrator przyjmuje zweryfikowane capture IDs/body store, jawny protocol/provider/batch, `as_of`, dodatni `max_fixtures` i opcjonalny predictor. Predictor dostaje dokładny fixture/origin/schedule/known-start/roster/capture context oraz genuine feature payload; zwraca klucz rzeczywiście zapisanego envelope lub `None`. Callback ma savepoint; niepoprawny nowy envelope lub wyjątek nie zostawia częściowej prognozy. Fixture/BO/roster/source hashes i clocks muszą dokładnie pasować do envelope. Lokalny wynik zawsze ma `forward_admitted=false`.
- Jeden primary origin pozostaje niezmienny po wyzwoleniu, także gdy późniejszy feed nie powtarza starego terminarza. Brak ID/roster/quote, wycofanie, błędne bindings, latest suspended state, sprzeczne book identities, spóźnienie, start i model failure pozostają jawne. Rzeczywiste generated_at/recorded_at i completion muszą zmieścić się w60s od originu i przed znanym startem; samo stare `as_of` nie wystarcza.
- `--history-captures` i `--roster-captures` przyjmują JSON arrays z załadowanymi SourceCapture records (`content_base64` dla bajtów). Builder sprawdza hash/availability/release, zachowuje native player slots i retractions; nie pozwala nadpisać frozen history/callerem składu. Najpierw buduje wszystkie snapshoty, potem aktywuje archiwalny native namespace w izolowanym procesie.
- `FixedArtifactA0Adapter` / `--serving-contract fixed-2026` używa **niezmienionych2026 artefaktów**, wersja `frozen-a0-2026-forward-v1`, wyłącznie **2026-11-01 <= cutoff <2029-07-01**. Annual adapter i jego wrong-year rejection nie zostały poluzowane. To zgodność wejścia/artefaktu, nie2027-trained model, live feature worker lub certyfikacja przyszłej jakości.

Wykonano trzy świeże izolowane PostgreSQL/Timescale schema paths, concurrency/rollback/SQL immutability i rzeczywisty loopback HTTP. Finalny native batch zachował18 fixture'ów:16 primary+2 pending,1 genuine forecast+17 bez,0 admitted; niezależny replay bajtów i future perturbation przechodzi.21 native/mode probabilities są dokładnie zgodne;8 tensorów reaguje na dostępny captured history. **168 różnych final focused tests passed**, jeden zastany warning, nie pełny application suite. Eksporty stages01/03 zachowano, własną bazę zatrzymano.

Autentyczne source-specific uprawnienia/retencja/kwoty, oficjalne locatory, participant/two-book bindings, start/results oraz live-input operation pozostają niewykazane. Nie uruchamiać automatycznie źródeł, migracji, H fitting ani kolekcji; obowiązuje niezmieniony deadline/start Plan06.


## Kiedy co uruchamiać

Rekomendowany rytm, żeby ograniczyć ryzyko bana i nie odpytywać stron bez sensu:

| Co | Jak często | Po co |
|---|---:|---|
| Lekki scrape ofert bukmacherów | co 1–2h, domyślnie 2h | aktualna lista upcoming i orientacyjne kursy |
| Pipeline predykcji/EV bez scrapowania | dowolnie często | przeliczenie features/predykcji na istniejących kursach |
| Close odds check | ręcznie / selektywnie 5–15 min przed meczem | ostatni kurs do CLV i decyzji |
| Refresh GOL.GG zakończonych meczów | co 2–3 dni | nowe wyniki i gry |
| Import GOL.GG + rebuild ratingów + W20 | po refreshu GOL.GG | aktualizacja modelu sportowego |

Domyślny scheduler w Dockerze **nie scrapuje co 30 minut**. Ustawiony jest na 2h:

```text
BETTING_SCHEDULER_INTERVAL_SECONDS=7200
```

Jeżeli chcesz ostrożniej:

```bash
BETTING_SCHEDULER_INTERVAL_SECONDS=10800 docker compose up -d betting-scheduler  # 3h
```

Jeżeli chcesz tylko przeliczyć model/EV bez nowych requestów do bukmacherów:

```bash
python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05
```

Close odds najlepiej robić selektywnie: najpierw panel wskazuje ciekawe EV+, potem otwierasz `offer_url` albo odpalasz pojedynczy scraper przed startem. Nie ma potrzeby agresywnie odpytywać wszystkich bukmacherów co kilka minut.

## Docker Compose / najprostsze przenoszenie

Dodane są pliki:

- `Dockerfile.betting` — obraz Python 3.12 + Chromium + zależności scraperów/modelu,
- `requirements-betting.txt` — zależności aplikacji bettingowej,
- `docker-compose.yml` — API backend, scheduler, frontend React i opcjonalny heavy maintenance,
- `.dockerignore` — mniejszy kontekst buildu.

Najprostszy start:

```bash
docker compose up --build -d betting-api betting-scheduler client
```

Frontend będzie dostępny na:

```text
http://localhost:3000
```

API backend na:

```text
http://localhost:8000
```

Co robią kontenery:

- `betting-api` — FastAPI backend (REST API),
- `client` — React frontend (Vite dev server),
- `betting-scheduler` — co `BETTING_SCHEDULER_INTERVAL_SECONDS` sekund odpala lekki cykl; domyślnie co 2h, żeby nie spamować bukmacherów:
  1. scrape kursów: STS, Betclic, Superbet, eFortuna, Betfan, TOTALbet, Lebull,
  2. rebuild canonical matches,
  3. build upcoming features,
  4. predict operational model,
  5. generate hybrid model+market,
  6. generate EV signals.
- `betting-maintenance` — profil opcjonalny do cięższego cyklu: GOL.GG refresh, import JSON→SQLite, rebuild ratingów, rebuild W20, potem lekki cykl.

Dane są trzymane w bind-mount:

```text
./data:/app/data
```

czyli SQLite, GOL.GG JSON i debug scraperów zostają lokalnie poza kontenerem.

W `docker-compose.yml` jest też przygotowany serwis `timescaledb` (`timescale/timescaledb:2.17.2-pg16`) jako docelowy backend pod historię kursów/CLV. Na ten moment aplikacja nadal domyślnie działa na SQLite, bo kod ma dużo jawnych zapytań SQLite. Kolejny etap migracji to adapter DB + Alembic i przeniesienie tabel kursów/zakładów do Timescale.

Przydatne komendy:

```bash
# logi schedulerów
docker compose logs -f betting-scheduler

# ręczny lekki cykl jednorazowy
docker compose run --rm betting-scheduler \
  python -m betting_app.scripts.scheduler --mode light-once

# ręczny pipeline bez scrapowania, na istniejących kursach
docker compose run --rm betting-scheduler \
  python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05
```

Ciężki maintenance z GOL.GG korzysta teraz z lokalnego scrapera `betting_app/scrapers/golgg.py`:

```bash
docker compose --profile maintenance run --rm betting-maintenance
```

Zmienna `EMBEDDED_RIFT_ESPORT_DIR` może jeszcze występować w starych komendach dla kompatybilności, ale `refresh_golgg_results` jej już nie potrzebuje.

Zmienne środowiskowe:

```text
BETTING_APP_TAX_RATE=0.12
BETTING_APP_MIN_EV=0.05
BETTING_APP_BANKROLL=100.0
BETTING_SCHEDULER_INTERVAL_SECONDS=7200
BETTING_SCHEDULER_BOOKMAKERS=sts,betclic,superbet,efortuna,betfan,totalbet,lebull
```

Możesz skopiować przykład konfiguracji:

```bash
cp .env.betting.example .env
```

### Tryb laptop 24/7 bez SSH

Docelowy tryb użycia:

1. Na laptopie wgrywasz repo i bazowe dane w `data/`.
2. Uruchamiasz raz:

```bash
docker compose up --build -d betting-api betting-scheduler client
```

3. Upewniasz się, że Docker startuje po restarcie systemu:

```bash
sudo systemctl enable docker
```

4. Od tej pory kontenery mają `restart: unless-stopped`, więc po restarcie laptopa Docker powinien sam podnieść API i scheduler.
5. Korzystasz z frontendu React:
   - **Lista meczów** — wyniki, EV, best odds, linki do ofert,
   - **System status** — czy scheduler żyje, ostatnie błędy, ostatnie scrape'y,
   - **Scheduler** — status zadań, ręczne triggery, historia runów,
   - **Timing Analysis** — analiza zmian kursów w czasie, najlepsze okno do obstawiania.

Scheduler zapisuje swoje cykle do tabel:

- `automation_runs`,
- `automation_commands`.

Dzięki temu nie trzeba czytać logów Dockera, żeby zobaczyć czy automat działa. Logi Dockera zostają tylko jako awaryjna diagnostyka.

Backup lokalnej bazy:

```bash
python -m betting_app.scripts.backup_sqlite
```

Backupy trafiają do:

```text
data/backups/
```

Rekomendowany praktyczny model:

- lekki scheduler działa sam co 2h,
- GOL.GG / ratingi / W20 robisz co 2–3 dni przez maintenance,
- close odds odpalasz selektywnie z panelu przy interesujących EV+,
- wyniki oglądasz w React froncie, nie w terminalu.

## Testowy flow bez bukmachera

1. Inicjalizacja bazy:

```bash
python -m betting_app.scripts.init_db
```

2. Wrzucenie przykładowych kursów dry-run:

```bash
python -m betting_app.scripts.scrape_odds --bookmaker dry-run
```

3. Uruchom pipeline predykcji:

```bash
python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05
```

4. Sprawdź wyniki w froncie React na `http://localhost:3000`.

## Portfele per bukmacher i ręczne logowanie zakładów

Każdy bukmacher może mieć osobne konto/portfel:

- tabela `bookmaker_accounts` — saldo per bukmacher/konto,
- tabela `bookmaker_wallet_transactions` — wpłaty, wypłaty, stake postawiony, zwrot/wygrana,
- tabela `bets` — historia ręcznie wpisanych zakładów, kurs, stake, strona, wynik, profit.

Workflow przez REST API:

1. Utwórz portfel: `POST /api/wallets`
2. Po faktycznym ręcznym postawieniu kuponu: `POST /api/bets`
3. Rozlicz zakład: `POST /api/bets/{id}/settle`

To pozwala analizować osobno saldo i wyniki na każdym bukmacherze, a nie tylko jeden globalny bankroll.

## Finalny model z pracy inżynierskiej

W bazie rejestrowany jest finalny model pracy:

```text
Sym-Cal LR-ElasticNet-W20-Binomial / exp-039
```

Sprawdzenie artefaktów:

```bash
python -m betting_app.scripts.inspect_final_thesis_model --register
```

Przed cutoverem C0 predictor produkcyjny odnotowany w audycie był `Symmetrized-Siamese-Series-EXP081` z osobną hybrydą `Hybrid-Operational-Market`. To historyczna obserwacja, nie aktywna konfiguracja lokalnego checkoutu. EXP-039 pozostaje odrębnym, zamrożonym punktem odniesienia; dostępność jego kompletnego artefaktu należy sprawdzać, a nie wnioskować z nazwy w rejestrze.

## Rygorystyczny benchmark modelu względem rynku

```bash
.venv/bin/python -m betting_app.ml.pipelines.evaluate_existing_model \
  --model-name Operational-PlayerTeamRatings-W20 \
  --model-version <wersja> \
  --json
```

Benchmark odrzuca predykcje bez strefy czasowej lub bez pełnego łańcucha:

```text
data_cutoff_at <= predicted_at <= quote_at < match_start_at
```

Wybiera ostatnią kwalifikującą się predykcję przed startem meczu i porównuje ją
z no-vig probability kursu dostępnego po predykcji. Raport zawiera liczność
kohorty, powody wykluczeń, log loss, Brier, AUC i ECE dla modelu oraz rynku.
To benchmark jakości predykcji, nie dowód wykonalnego ROI: symulacja bankrolla
wymaga jeszcze ledgeru rozliczającego nakładające się mecze według czasu wyniku.

## TimescaleDB / Postgres

Dodany jest serwis TimescaleDB:

```bash
docker compose up -d timescaledb
```

Domyślne zmienne w `.env.betting.example`:

```text
POSTGRES_DB=betting
POSTGRES_USER=betting
POSTGRES_PASSWORD=betting_local_password
POSTGRES_PORT=5432
```

Docelowo Timescale powinien przejąć szczególnie:

- `odds_outcome_snapshots`,
- `odds_snapshots`,
- `scrape_runs`,
- `automation_runs`,
- `bets`,
- `bookmaker_wallet_transactions`.

Na teraz traktuj Timescale jako przygotowany fundament. Pełne przełączenie aplikacji wymaga jeszcze adaptera DB i migracji Alembic, żeby nie utracić kompatybilności z istniejącym SQLite MVP.

## STS League of Legends

STS LoL prematch działa przez snapshot SBK używany przez frontend do hydratacji oferty:

```text
https://sbk.sts.pl/sbk-exporter/v1/sports/ssr
```

ID używane przez STS:

- `sport_id=156` — Esport,
- `category_id=992` — League of Legends,
- market `Zwycięzca meczu` — prematch match winner.

Pobranie i zapis do SQLite:

```bash
python -m betting_app.scripts.scrape_odds --bookmaker sts
```

Scraper pobiera pełną listę nadchodzących meczów LoL ze snapshotu i zapisuje rynek `Zwycięzca meczu`. Dla każdego meczu tworzy:

- atomowe ticki w `odds_outcome_snapshots` — do historii kursów i CLV,
- dwustronny snapshot w `odds_snapshots` — dla prostego MVP generowania sygnałów EV.

Endpoint `social-api.sts.pl/api/events` zostaje traktowany tylko jako pomocnicze źródło popularnych typów; nie jest już głównym scraperem STS.

## Model bazy dla kursów

Kanoniczny tracking kursów jest w nowych tabelach:

1. `scrape_runs` — jeden job scrapera: kiedy, z jakiego URL-a, ile rekordów widział i ile zapisał.
2. `bookmaker_events` — wydarzenie u bukmachera: bookmaker event ID, drużyny, liga, start, kategoria.
3. `bookmaker_markets` — rynek w obrębie wydarzenia: zwycięzca meczu, handicap, dokładny wynik, mapa itd.
4. `odds_outcome_snapshots` — pojedynczy tick kursu dla outcome'u z timestampem `scraped_at`.

Stara tabela `odds_snapshots` zostaje dla prostego MVP/UI, gdzie mamy pełny dwustronny rynek match-winner (`odds_a`, `odds_b`). Dla realnych API preferuj `odds_outcome_snapshots`, bo pozwala liczyć line movement i CLV per outcome.

Dla closing odds zapisujemy dwa typy linków:

- `source_url` — URL listy/API/snapshotu użyty przez scraper,
- `offer_url` — bezpośredni link do konkretnego wydarzenia u bukmachera.

Linki do ręcznego odpalenia około 5 minut przed startem:

```bash
python -m betting_app.scripts.list_close_odds_targets --bookmaker sts
python -m betting_app.scripts.list_close_odds_targets --bookmaker betclic
python -m betting_app.scripts.list_close_odds_targets --bookmaker superbet
python -m betting_app.scripts.list_close_odds_targets --bookmaker efortuna
python -m betting_app.scripts.list_close_odds_targets --bookmaker betfan
python -m betting_app.scripts.list_close_odds_targets --bookmaker totalbet
python -m betting_app.scripts.list_close_odds_targets --bookmaker lebull
```

## Superbet, eFortuna, Betfan, TOTALbet i Lebull

Superbet oraz eFortuna są obecnie obsługiwane przez NoDriver i parser widocznej strony:

```bash
python -m betting_app.scripts.scrape_odds --bookmaker superbet --headless
python -m betting_app.scripts.scrape_odds --bookmaker efortuna --headless
python -m betting_app.scripts.scrape_odds --bookmaker betfan --headless
python -m betting_app.scripts.scrape_odds --bookmaker totalbet
python -m betting_app.scripts.scrape_odds --bookmaker lebull
```

Superbet zwraca bezpośrednie linki eventów typu `/kursy/league-of-legends/...`. eFortuna generuje per-event `offer_url` z URL-a ligi i slugów drużyn. Betfan jest renderowany w SPA, więc scraper klika zakładkę `LoL` i parseruje widoczne karty. TOTALbet korzysta z publicznego API `/dealer/bdata/v1/bet/events/esport`. Lebull korzysta z API `betting-platform.prod.sbteam.xyz` z publicznym tenant headerem pobieranym z SSR.

## NoDriver

Scraper STS nie wymaga NoDriver, bo korzysta ze snapshotu SBK. NoDriver zostaje dla Betclic i ewentualnych stron, gdzie trzeba renderować DOM:

```bash
python -m betting_app.scripts.scrape_odds --bookmaker betclic --no-headless
```

Obecnie `BetclicNoDriverScraper`:

- otwiera stronę bukmachera,
- zapisuje HTML/screenshot debug do `data/betting_scraper_debug/`,
- parsuje stronę Betclic LoL i zapisuje dwustronne snapshoty match-winner,
- próbuje przypiąć bezpośredni `offer_url` do każdej oferty na podstawie linków eventów w DOM.

Jeżeli Betclic zmieni HTML, parser zostawi debug `*_body.txt`, HTML i screenshot w katalogu debug.

## GOL.GG refresher

GOL.GG jest osobnym jobem aktualizującym wyniki co 2-3 dni. Scraper jest vendored w projekcie jako
`betting_app/scrapers/golgg.py`, a job domyślnie pobiera tylko brakujące `match_id` i nie refetchuje
meczów, które są już w `data/golgg_matches.json`.

```bash
python -m betting_app.scripts.refresh_golgg_results
```

Przydatny tryb kontrolny bez zapisu:

```bash
python -m betting_app.scripts.refresh_golgg_results --dry-run
```

Domyślny kontrakt:

1. pobierz najnowsze zakończone mecze,
2. porównaj `match_id` z lokalnym `data/golgg_matches.json`,
3. zapisz metadane tylko nowych meczów,
4. pobierz nested games tylko dla nowych meczów,
5. opcjonalnie użyj `--include-incomplete-existing`, żeby uzupełnić stare niekompletne rekordy,
6. potem przelicz ratingi/W20 i zapisz `ratings_version` oraz `data_cutoff_at` dla predykcji.

Wymagane zależności: `httpx`, `parsel`, `tqdm` — są wpisane w `requirements-betting.txt`.

## Relacyjna baza GOL.GG

Duży `data/golgg_matches.json` jest źródłem/cache, ale aplikacja nie powinna go
czytać przy każdym starcie. Import do SQLite:

```bash
python -m betting_app.scripts.import_golgg_to_db
```

Test na próbce:

```bash
python -m betting_app.scripts.import_golgg_to_db --limit 100
```

Importer wypełnia:

- `golgg_matches` — jeden rekord na match,
- `golgg_games` — pojedyncze mapy/gry,
- `golgg_game_players` — występy graczy per gra/rola,
- `golgg_teams` — nazwy drużyn do mapowania bookmakerów.

Po imporcie serwisy mapowania nazw korzystają najpierw z SQLite, a JSON jest
fallbackiem.

## Baza pod inference modelu upcoming

Struktura SQLite jest przygotowana pod uruchamianie modelu dla `canonical_matches`.
Inicjalizacja i kontrola gotowości:

```bash
python -m betting_app.scripts.prepare_model_db --register-default-model
```

Najważniejsze tabele operacyjne:

- `model_artifacts` — rejestr modeli, m.in. finalny `Sym-Cal LR-ElasticNet-W20-Binomial / exp-039`,
- `rating_runs` — metadane przebudowy ratingów po GOL.GG,
- `entity_ratings` — aktualne ratingi team/player dla Elo/Glicko-2/TrueSkill/OpenSkill/etc.,
- `team_rolling_features` — rolling W20 team context z GOL.GG,
- `upcoming_match_features` — gotowy feature vector per `canonical_match_id`,
- `canonical_predictions` — predykcje modelu dla cross-bookmaker canonical match,
- `model_ev_signals` — EV modelowe po zestawieniu predykcji z najlepszymi kursami.

Docelowy workflow:

```bash
python -m betting_app.scripts.refresh_golgg_results
python -m betting_app.scripts.import_golgg_to_db
python -m betting_app.scripts.scrape_odds --bookmaker sts
python -m betting_app.scripts.scrape_odds --bookmaker betclic --headless
python -m betting_app.scripts.scrape_odds --bookmaker superbet --headless
python -m betting_app.scripts.scrape_odds --bookmaker efortuna --headless
python -m betting_app.scripts.scrape_odds --bookmaker betfan --headless
python -m betting_app.scripts.scrape_odds --bookmaker totalbet
python -m betting_app.scripts.scrape_odds --bookmaker lebull
python -m betting_app.scripts.rematch_canonical_matches --rebuild
python -m betting_app.scripts.rebuild_regional_ratings --ratings-version ratings-v2
python -m betting_app.scripts.rebuild_w20_features --feature-version w20-latest --window-size 20
python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05
python -m betting_app.scripts.list_upcoming_model_predictions --positive-only
```

### Operacyjny kontrakt regionalny `ratings-v2`

`ratings-v2` jest pełnym, chronologicznym snapshotem sześciu systemów:
`elo`, regionalny `gl`, `ts`, `os`, `pl` i `tm`. `gl` używa
`family-calibrated-glicko2-v1`; jego niepewny offset rodziny/poziomu jest
stosowany dokładnie raz do prawdopodobieństw pięciu pozostałych systemów.
Prawdopodobieństwo `gl` nie dostaje drugiej korekty.

```bash
python -m betting_app.scripts.rebuild_regional_ratings \
  --ratings-version ratings-v2 \
  --source manual-regional-ratings-v2
python -m betting_app.scripts.rebuild_w20_features --feature-version w20-latest
python -m betting_app.scripts.run_upcoming_prediction_pipeline \
  --operational-hybrid --include-partial
```

Rebuild jest zawsze pełny. Nie uruchamiaj `rebuild_ratings` dla `ratings-v2`:
wszystkie systemy muszą odzwierciedlać identyczny kohortowy cutoff i ten sam
stan regionalnego Glicko. Zadanie `heavy_maintenance_cycle` wykonuje kolejno
odświeżenie GOL.GG, `rebuild_regional_ratings` oraz W20; zwykły
`prediction_pipeline` następnie tworzy features, predykcje modelu wskazanego
przez `betting_app/core/models/registry.py`, hybrydę rynku i sygnały EV.
W historycznym rejestrze był to EXP-081; jego obecność nie oznaczała
spełnienia kryteriów promocji. Audyt z 2026-09-07 nie zatwierdził nowego
artefaktu ani wariantu 84 cech — patrz
[`benchmark_results.md`](../reports/exp081_siamese_focal_uncertainty_gating/benchmark_results.md).
Obecna ścieżka **Predict** używa C0 i respektuje dokładne ID zatwierdzonego
składu. Braki natywnego wejścia są odmową predykcji, nie podstawieniem zer,
domyślnych ratingów lub zastępczego konsensusu.

EXP-039 (`Sym-Cal LR-ElasticNet-W20-Binomial`) pozostaje zamrożonym,
nieoperacyjnym baseline'em wyłącznie do porównań; nie jest już wybierany przez
interaktywną ani schedulerową ścieżkę predykcji. Żaden z tych kroków nie składa
zakładów automatycznie.

### Kontrakt rekomendacji i niepewności

Predykcja zachowuje komplementarne średnie `prob_a`, `prob_b`. Modele z
niepewnością muszą dodatkowo dostarczyć osobne `p_low_a`, `p_low_b`,
`epistemic_sigma_z` i `rating_disagreement` w `diagnostics_json`.
Dolne oceny **nie są komplementarne ani objęte gwarancją statystyczną**.
Hybryda przekształca każdą z nich z odpowiednią stroną rynku; nie używa
`1-p_low_a` jako konserwatywnej oceny B.

Generator, tablica, szczegóły meczu i `/predictions` używają wspólnej
kwalifikacji: dolna ocena do EV/Kelly, średnia do kontroli odchylenia od rynku,
rozbieżność konsensusu sześciu ratingów drużyny/graczy i kontrola Tier-1.
Brak lub błędna wymagana diagnostyka blokuje rekomendację. Stare jednostronne
diagnostyki wymagają ponownego obliczenia, nie dopisania dopełnienia.
Domyślny próg generatora/CLI wynosi +5% netto z dodatkową karą niskich kursów;
publiczne rekomendacje nie obniżają tej polityki filtrem `min_ev`.
Pola `model_prob` prezentują średnią, natomiast `ev` i `kelly` dotyczą
konserwatywnej oceny. `half_kelly` i `quarter_kelly` to odpowiednio 1/2 i 1/4
pełnego Kelly'ego, nie gwarantowane bezpieczne stawki.

Ta naprawa nie ustanawia historycznej dostępności danych, nie naprawia
event-time ledger i nie dowodzi rentowności. Nie wykonuje też migracji ani
przeliczenia skonfigurowanej bazy. Eksperymenty treningowe zapisują nowe,
niezatwierdzone artefakty poza katalogiem aktywnych modeli.

### Chronologiczne porównanie modeli

Po przebudowie ratingów i W20 można odtworzyć nowy model dla wszystkich
zmapowanych zakończonych meczów:

```bash
python -m betting_app.scripts.backfill_operational_predictions --apply
```

Backfill zapisuje osobny, niemutowalny wariant
`v0.4-binom-series-chronological-v1`. Każda data jest najpierw przewidywana z
ratingów i W20 z wcześniejszych pełnych dat, a dopiero potem aktualizowana;
wyniki wymagają `data_cutoff_at <= predicted_at < match_start_at`. Widok
**Model Analysis** pokazuje metryki EXP-039 i regionalnego replaya na ich
wspólnej kohorcie. To porównanie jakości modelu na etykietach historycznych,
nie dowód live forecastingu, CLV, ROI ani wykonania finansowego.

### Evidence and limitation

EXP-076 applies the shared uncertain family/tier offset to Player Elo,
TrueSkill, OpenSkill, Plackett-Luce and Thurstone-Mosteller only when both
sides have a prior domestic affiliation. Player Glicko is the regional
calibrated system; domestic same-family games receive no additional
probability offset.

```bash
python scripts/06_metamodel/06ak_multirating_family_symcal.py \
  --data-dir data \
  --output-dir reports/experiments/exp076_multirating_family_symcal
```

The 2024+ diagnostic result (`n=4311`) reduced LogLoss from `0.581427` to
`0.576200`, but its paired 95% CI was `[-0.013174, +0.000072]`; it does not
meet the global two-sided significance criterion. Cross-league rows (`n=526`)
improved from `0.593362` to `0.533148`, while domestic rows were weaker.
`ratings-v2` is therefore an explicitly versioned operational research
baseline, not evidence of financially executable performance or a replacement
for the frozen thesis result.

Skrócony runner:

```bash
# lekki tryb: scrape bukmacherów -> canonical matching -> features -> predykcje -> EV
# uruchamiaj ręcznie albo schedulerem co 1-2h, nie co kilka minut
python -m betting_app.scripts.run_daily_automation --operational-hybrid --min-ev 0.05

# samo przeliczenie predykcji/EV bez nowych requestów do bukmacherów
python -m betting_app.scripts.run_upcoming_prediction_pipeline --operational-hybrid --min-ev 0.05

# cięższy tryb po odświeżeniu zakończonych meczów GOL.GG
python -m betting_app.scripts.run_daily_automation \
  --refresh-golgg --reimport-golgg --rebuild-ratings --rebuild-w20 \
  --operational-hybrid --min-ev 0.05
```

## Zasada parserów HTML

Jeżeli scraper pracuje na HTML/DOM snapshotcie, ekstrakcja elementów powinna iść
przez `parsel.Selector`, a nie przez regex po HTML. Regex zostaje tylko do
parsowania tekstu liniowego (`document.body.innerText`), ID ze znanego URL-a albo
prostych etykiet/kursów.

## Ważne założenia finansowe

- EV liczone jest jako `p * odds * (1 - tax_rate) - 1`.
- Domyślny podatek: `12%`.
- Domyślny minimalny EV: `5%`.
- Domyślny suggested stake: Kelly 0.05 z limitami min/max.
- Wyniki są diagnostyczne; aplikacja nie dowodzi realnej przewagi bukmacherskiej.

## Zmienne środowiskowe

- `BETTING_APP_DB` — ścieżka do SQLite DB.
- `BETTING_APP_DEBUG_DIR` — katalog debug HTML/screenshot.
- `BETTING_APP_TAX_RATE` — domyślnie `0.12`.
- `BETTING_APP_MIN_EV` — domyślnie `0.05`.
- `BETTING_APP_BANKROLL` — domyślnie `100.0`.
- `BETTING_APP_HEADLESS` — `1`/`0` dla NoDriver.
