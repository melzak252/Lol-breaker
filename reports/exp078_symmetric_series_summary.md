# Podsumowanie wdrożenia i audytu modelu EXP-078 oraz anomalii ISSUE-001

Data: 2026-09-06
Status: Zakończone wdrożenie modelu sportowego EXP-078; częściowa mitygacja ISSUE-001 (otwarta bramka kapitałowa)
Artefakt modelu: `betting_app/models/exp078_symmetric_series_v1.json`

---

## 1. Wdrożenie modelu Symmetric-General-Series-EXP-078

### A. Problem bazowy
Poprzedni predictor operacyjny (`Operational-PlayerTeamRatings-W20` v0.4) łączył ratingi graczy (70%), ratingi drużyn (20%) i statystyki formy W20 (10%) sztywną średnią ważoną, po czym nakładał niezależną ekspansję dwumianową na format serii.
Audyt ujawnił:
1. Niespójność matematyczną: funkcje w `thesis_inference_service.py` liczyły prawdopodobieństwa ratingowe innymi wzorami niż historyczny `RatingManager`.
2. Sztuczną kompresję: format Bo1/Bo3/Bo5 nie był uczony wprost, lecz narzucany wzorem dwumianowym z założeniem niezależności map.
3. Uszkodzenia danych historycznych: 10 751 meczów w surowym eksporcie GOL.GG miało błędną orientację pól `t1_win`/`t1_score`.

### B. Architektura EXP-078
- **Czysty model sportowy**: Nie czyta kursów bukmacherskich, spreadów ani wolumenów rynkowych.
- **Symetria konstrukcyjna**: Wszystkie cechy są antysymetryczne ($f(B, A) = -f(A, B)$), a model stosuje regresję logistyczną L2 bez wyrazu wolnego (intercept = 0), co gwarantuje $P(A) + P(B) = 1.0$ z dokładnością numeryczną.
- **Zestaw 79 cech**:
  - Ratingi graczy i drużyn z 6 systemów (Elo, Glicko-2, TrueSkill, OpenSkill, Plackett–Luce, Thurstone–Mosteller);
  - 15 różnicowych wskaźników formy W20 z GOL.GG;
  - Interakcje z formatem serii (Bo1, Bo3, Bo5).
- **Kalibracja prekwencyjna**: Dobór regularyzacji $C=0.1$ wyłącznie na roku 2022. Predykcje na lata 2023–2026 generowane w trybie prekwencyjnym z kalibratorem slope-only trenowanym wyłącznie na wcześniejszych predykcjach out-of-fold.

### C. Wyniki na głównym kohorcie 2023–2026 ($N = 15\,341$)
| Metryka | Operational v0.4 (dwumianowy) | EXP-078 (symetryczny) | Zmiana ($\Delta$) | Istotność statystyczna (Bootstrap) |
|---|---:|---:|---:|---|
| **LogLoss** | 0.590876 | **0.565401** | **-0.025475** | 95% CI: `[-0.030471, -0.021228]`, 0 / 10 000 $\ge 0$ |
| **Brier Score** | 0.202304 | **0.192664** | **-0.009640** | 95% CI: `[-0.011711, -0.007876]` |
| **ROC-AUC** | 0.751982 | **0.774881** | **+0.022899** | Zauważalny wzrost dyskryminacji |
| **ECE-15** | 0.029681 | **0.019207** | **-0.010474** | Znacząca poprawa globalnej kalibracji |

### D. Benchmark rynkowy na próbie wspólnej ($N = 7\,125$)
| Model / Źródło | LogLoss | Brier Score | ROC-AUC |
|---|---:|---:|---:|
| **EXP-078** | **0.583349** | **0.200013** | **0.756227** |
| Zamrożony akademicki EXP-039 | 0.583362 | 0.199947 | 0.756086 |
| No-vig market close | 0.598668 | 0.206133 | 0.740552 |
| No-vig market open | 0.607455 | 0.209977 | 0.730228 |

---

## 2. Integralność danych i pipeline operacyjny

1. **Rekonstrukcja etykiet**: Wyniki meczów odtwarzane są bezpośrednio z map po wyrównaniu stron (`align_snapshot_odds`), eliminując 10 751 błędów orientacji. Nierozwiązywalny rekord `77747` został trwale wykluczony.
2. **Kolejność zdarzeń (temporal integrity)**: Mecze z tej samej daty są batchowane i nie aktualizują ratingów ani statystyk W20 wewnątrz tego samego dnia.
3. **Parytet wzorów**: Serwis predykcji operacyjnych (`betting_app/services/upcoming_inference_service.py`) został zaktualizowany i zweryfikowany testem parytetu (`test_operational_rating_probability_parity.py`) wobec `RatingManager`.
4. **Brak neutralnych fallbacków**: W przypadku braku dowolnej wymaganej cechy generowany jest jawny status `skipped`.

---

## 3. Wyniki audytu anomalii ISSUE-001 (wysokie underdogi 3.50–5.00)

Przeprowadzono szczegółowy audyt pod kątem zgłoszenia ISSUE-001 na kohorcie rynkowym 2023–2026 ($N=7\,125$).
Koszyk fair odds 3.50–5.00 zdefiniowano jako $P_{\text{market\_novig}} \in [0.20, 1/3.5 \approx 0.2857]$ ($N = 1\,381$ meczów):

### A. Wyniki w koszyku [3.50, 5.00]
- **Rzeczywisty win rate underdoga**: **25.20%**
- **Konsensus rynkowy (no-vig close)**: **24.55%**
- **Predykcja EXP-078**: **29.00%** (nadmierna pewność: **+3.80 p.p.**, 95% CI: `[+1.90, +5.59] p.p.`)
- **Dla porównania v0.4 operational**: 27.91% (w małej pierwotnej próbie ISSUE-001: 41.2%)
- **LogLoss**: EXP-078 **0.5356** vs v0.4 0.5561 vs no-vig market 0.5630.
- **Brier**: EXP-078 **0.1761** vs v0.4 0.1831 vs no-vig market 0.1837.

Wniosek: EXP-078 zredukował anomalię z +21.7–25.2 p.p. do +3.80 p.p. i ma lepszą dyskryminację niż rynek.

### B. Analiza podprzedziałów wewnątrz koszyka
| Przedział predykcji EXP-078 | $N$ | Średnia predykcja | Rzeczywisty win rate | Status kalibracji |
|---|---:|---:|---:|---|
| `[0.00, 0.20)` | 368 | 15.36% | 18.21% | Niedoszacowanie (-2.85 p.p.) |
| `[0.20, 0.30)` | 550 | 25.09% | 22.18% | Lekkie przeszacowanie (+2.91 p.p.) |
| **`[0.30, 0.40)`** | **267** | **34.12%** | **23.22%** | **Istotna nadmierna pewność (+10.90 p.p.)** |
| **`[0.40, 0.50)`** | **83** | **44.22%** | **27.71%** | **Ciężka nadmierna pewność (+16.51 p.p.)** |
| `[0.50, 1.00]` | 113 | 69.24% | 65.49% | Silny sygnał przewagi (dobra kalibracja) |

### C. Luka operacyjna i status ISSUE-001
1. W kodzie `betting_app/` nie istnieje funkcja `is_bet_eligible` ani filtr `quarantine_trap_issue_001`.
2. Status ISSUE-001 został formalnie zaktualizowany na **`partially-mitigated`**.
3. Aby zamknąć problem operacyjnie, konieczne jest wdrożenie hybrydy rynkowej (IDEA-021) oraz testowanej jednostkowo bramki kapitałowej w serwisie rekomendacji.
