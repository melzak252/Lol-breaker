"""Dedicated financial and betting benchmark for professional LoL prediction models.

Evaluates executable betting performance under the 12% Polish turnover tax:
- ROI (%) and Net Profit (PLN)
- Net Yield (%) = Total Profit / Total Staked * 100
- Maximum Drawdown (% and PLN)
- Offer volume: Bets count and Bet Rate (%)
- Expected ROI / Yield (%) = Mean EV of qualifying bets
- Closing Line Value (CLV %): Odds captured vs closing line
- Beat Closing Line Rate (%): How often model secures odds higher than close
- Time Horizon tracking (Opening >12h, Mid 2-12h vs Close)
- Segmented diagnostics & probability calibration analysis across odds tiers, EV brackets, and CLV tiers
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FinancialBenchmarkSummary:
    """Core financial, bankroll simulation, and CLV metrics."""

    tax_rate: float
    min_ev: float
    max_odds: float
    staking_strategy: str
    time_horizon: str
    initial_bankroll: float
    final_bankroll: float
    total_matches_evaluated: int
    bets_count: int
    bet_rate_pct: float
    wins_count: int
    losses_count: int
    win_rate_pct: float
    avg_odds: float
    avg_stake: float
    total_staked: float
    total_profit: float
    roi_pct: float
    yield_pct: float
    expected_yield_pct: float
    expected_profit: float
    max_drawdown_pct: float
    max_drawdown_pln: float
    avg_clv_pct: float | None = None
    beat_closing_rate_pct: float | None = None


@dataclass(frozen=True)
class FinancialSlice:
    """Segmented financial metric breakdown and calibration diagnostics for a subset of bets."""

    dimension: str
    bucket: str
    sample_size: int
    bets_count: int
    wins_count: int
    win_rate_pct: float
    avg_odds: float
    total_staked: float
    total_profit: float
    yield_pct: float
    expected_yield_pct: float
    avg_prob_pct: float | None = None          # Średnie prawdopodobieństwo modelu (%)
    implied_prob_pct: float | None = None      # Średnie prawdopodobieństwo implikowane kursem 1/kurs (%)
    calibration_gap_pct: float | None = None   # Gap = avg_prob_pct - win_rate_pct (+ oznacza przeszacowanie / overconfidence)
    brier_score: float | None = None           # Brier score dla zakładów w tym koszyku
    avg_clv_pct: float | None = None


@dataclass(frozen=True)
class FinancialBenchmarkReport:
    """Complete financial benchmark report."""

    model_name: str
    model_version: str
    cohort_start: str | None
    cohort_end: str | None
    summary: FinancialBenchmarkSummary
    slices: dict[str, list[FinancialSlice]] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_markdown(self) -> str:
        """Render a clean Markdown financial report with calibration analysis."""
        s = self.summary
        lines: list[str] = [
            f"# Raport Finansowy i Benchmark Bukmacherski: {self.model_name} ({self.model_version})",
            "",
            f"- **Okres kohorty:** `{self.cohort_start or 'N/A'}` do `{self.cohort_end or 'N/A'}`",
            f"- **Liczba ocenionych meczów:** {s.total_matches_evaluated:,}",
            f"- **Horyzont czasowy wejścia:** `{s.time_horizon}` (zakaz grania na linii zamknięcia!)",
            f"- **Podatek obrotowy:** {s.tax_rate * 100:.1f}% (współczynnik efektywny: {1.0 - s.tax_rate:.2f})",
            f"- **Filtr opłacalności:** Minimalne $\\text{{EV}} \\ge {s.min_ev * 100:+.1f}\\%$ (domyślnie +5.0%)",
            f"- **Strategia stawkowania:** `{s.staking_strategy}` (kapitał początkowy: {s.initial_bankroll:,.2f} PLN)",
            "",
            "## 1. Główne Wskaźniki Rentowności, Ryzyka i CLV",
            "",
            "| Wskaźnik | Wartość | Uwagi / Interpretacja |",
            "| :--- | :---: | :--- |",
            f"| **Postawione oferty (Liczba zakładów)** | **{s.bets_count:,}** | {s.bet_rate_pct:.1f}% wszystkich meczów spełniło filtr EV |",
            f"| **Skuteczność (Win Rate)** | **{s.win_rate_pct:.2f}%** | {s.wins_count:,} wygranych / {s.losses_count:,} przegranych |",
            f"| **Średni kurs zagrany** | **{s.avg_odds:.2f}** | Średnia arytmetyczna kursów dziesiętnych |",
            f"| **Średnia stawka** | **{s.avg_stake:.2f} PLN** | Średnia stawka pojedynczego zakładu |",
            f"| **Łączny obrót (Total Staked)** | **{s.total_staked:,.2f} PLN** | Suma zainwestowanego kapitału |",
            f"| **Oczekiwany Yield (Expected Yield / EV)** | **{s.expected_yield_pct:+.2f}%** | Średnia modelowa wartość oczekiwana netto |",
            f"| **Oczekiwany Zysk Netto** | **{s.expected_profit:+,.2f} PLN** | Oczekiwany zysk według modelu |",
            f"| **Rzeczywisty Yield Netto** | **{s.yield_pct:+.2f}%** | **Zysk netto / Obrót (z podatkiem 12%)** |",
            f"| **Zwrot z Kapitału (Net ROI)** | **{s.roi_pct:+.2f}%** | **(Kapitał końcowy - początkowy) / Początkowy** |",
            f"| **Zysk Całkowity Netto** | **{s.total_profit:+,.2f} PLN** | Wynik portfela na czysto po odliczeniu podatku |",
            f"| **Maksymalne Obsunięcie (Max Drawdown)** | **{s.max_drawdown_pct:.2f}%** | Najgłębszy spadek bankrolla od szczytu |",
            f"| **Max Drawdown w PLN** | **{s.max_drawdown_pln:,.2f} PLN** | Największa nominalna strata od lokalnego peaku |",
            f"| **Kapitał Końcowy** | **{s.final_bankroll:,.2f} PLN** | Stan portfela po zakończeniu próby |",
        ]

        if s.avg_clv_pct is not None:
            lines.extend(
                [
                    "",
                    "### Jakość Przewagi Nad Rynkiem (Closing Line Value - CLV)",
                    "",
                    "| Metryka Rynkowa (CLV) | Wartość | Interpretacja |",
                    "| :--- | :---: | :--- |",
                    f"| **Średni CLV (Closing Line Value)** | **{s.avg_clv_pct:+.2f}%** | Średnia różnica kursu zagranego vs kursu zamknięcia (wyżej = lepiej) |",
                    f"| **Pobicie Linii Zamknięcia (Beat Close Rate)** | **{s.beat_closing_rate_pct:.1f}%** | Odsetek ofert zagranych po kursie wyższym niż kurs zamknięcia |",
                    f"| **Weryfikacja Ostrości Modelu (Sharpness)** | `{'OSTRY MODEL (CLV > 0)' if s.avg_clv_pct > 0 else 'NIEPOBIŁ ZAMKNIĘCIA (CLV <= 0)'}` | Potwierdzenie, że rynek przesuwa się w stronę predykcji modelu |",
                ]
            )

        lines.append("")

        if self.slices:
            lines.extend(["## 2. Analiza Zakładów w Przedziałach Kursowych i Kalibracji", ""])
            for dim, slice_items in self.slices.items():
                has_clv = any(item.avg_clv_pct is not None for item in slice_items)
                has_calib = any(item.avg_prob_pct is not None for item in slice_items)

                if dim == "Przedział Kursowy" and has_calib:
                    lines.extend(
                        [
                            f"### Segment: {dim} (Rentowność i Kalibracja Szans)",
                            "",
                            (
                                "| Koszyk Kursowy | Oferty | Win Rate | Śr. Kurs | Prawd. Modelu | Implikowane (1/kurs) | Gap Kalibracji | Brier | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Śr. CLV |"
                                if has_clv
                                else "| Koszyk Kursowy | Oferty | Win Rate | Śr. Kurs | Prawd. Modelu | Implikowane (1/kurs) | Gap Kalibracji | Brier | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Oczekiwany Yield |"
                            ),
                            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                        ]
                    )
                    for item in slice_items:
                        prob_str = f"{item.avg_prob_pct:.1f}%" if item.avg_prob_pct is not None else "N/A"
                        imp_str = f"{item.implied_prob_pct:.1f}%" if item.implied_prob_pct is not None else "N/A"
                        gap_str = f"{item.calibration_gap_pct:+.1f}%" if item.calibration_gap_pct is not None else "N/A"
                        brier_str = f"{item.brier_score:.3f}" if item.brier_score is not None else "N/A"
                        if has_clv:
                            clv_str = f"{item.avg_clv_pct:+.2f}%" if item.avg_clv_pct is not None else "N/A"
                            lines.append(
                                f"| **{item.bucket}** | {item.bets_count:,} | {item.win_rate_pct:.1f}% | "
                                f"{item.avg_odds:.2f} | {prob_str} | {imp_str} | **{gap_str}** | {brier_str} | "
                                f"{item.total_staked:,.0f} | {item.total_profit:+,.2f} | **{item.yield_pct:+.2f}%** | {clv_str} |"
                            )
                        else:
                            lines.append(
                                f"| **{item.bucket}** | {item.bets_count:,} | {item.win_rate_pct:.1f}% | "
                                f"{item.avg_odds:.2f} | {prob_str} | {imp_str} | **{gap_str}** | {brier_str} | "
                                f"{item.total_staked:,.0f} | {item.total_profit:+,.2f} | **{item.yield_pct:+.2f}%** | {item.expected_yield_pct:+.2f}% |"
                            )
                    lines.extend(
                        [
                            "",
                            "*Uwagi diagnostyczne do kalibracji kursowej:*",
                            "- **Gap Kalibracji** = $\\bar{p}_{\\text{model}} - \\text{Win Rate}$. Wartość bliska $0.0\\%$ oznacza idealną kalibrację w danym koszyku. Wartość $> +5.0\\%$ sygnalizuje przeszacowanie szans (overconfidence), a $< -5.0\\%$ niedoszacowanie.",
                            "- **Implikowane (1/kurs)** = Średnie prawdopodobieństwo surowe wyceniane przez bukmachera. Nadwyżka Prawd. Modelu nad Implikowanym to źródło EV.",
                            "- **Brier Score** = Średni błąd kwadratowy predykcji na zawartych zakładach (niżej = lepiej).",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            f"### Segment: {dim}",
                            "",
                            (
                                "| Koszyk (Bucket) | Liczba Ofert | Win Rate | Śr. Kurs | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Śr. CLV |"
                                if has_clv
                                else "| Koszyk (Bucket) | Liczba Ofert | Win Rate | Śr. Kurs | Obrót [PLN] | Zysk Netto [PLN] | Yield Netto | Oczekiwany Yield |"
                            ),
                            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                        ]
                    )
                    for item in slice_items:
                        if has_clv:
                            clv_str = f"{item.avg_clv_pct:+.2f}%" if item.avg_clv_pct is not None else "N/A"
                            lines.append(
                                f"| **{item.bucket}** | {item.bets_count:,} | {item.win_rate_pct:.1f}% | "
                                f"{item.avg_odds:.2f} | {item.total_staked:,.0f} | {item.total_profit:+,.2f} | "
                                f"**{item.yield_pct:+.2f}%** | {clv_str} |"
                            )
                        else:
                            lines.append(
                                f"| **{item.bucket}** | {item.bets_count:,} | {item.win_rate_pct:.1f}% | "
                                f"{item.avg_odds:.2f} | {item.total_staked:,.0f} | {item.total_profit:+,.2f} | "
                                f"**{item.yield_pct:+.2f}%** | {item.expected_yield_pct:+.2f}% |"
                            )
                    lines.append("")
        return "\n".join(lines)


def calculate_quarter_kelly(
    probability: float,
    odds: float,
    tax_rate: float = 0.12,
    bankroll: float = 10_000.0,
    fraction: float = 0.25,
    max_cap_pct: float = 0.02,
) -> float:
    """Calculate Quarter-Kelly stake with Polish turnover tax.

    Net decimal return: b = odds * (1 - tax_rate) - 1.0.
    Full Kelly: f* = EV / b.
    Quarter Kelly: stake = bankroll * min(fraction * f*, max_cap_pct).
    """
    net_multiplier = odds * (1.0 - tax_rate)
    b = net_multiplier - 1.0
    if b <= 0.0 or probability <= 0.0 or probability >= 1.0:
        return 0.0

    ev = probability * net_multiplier - 1.0
    if ev <= 0.0:
        return 0.0

    full_kelly = ev / b
    if full_kelly <= 0.0:
        return 0.0

    stake_fraction = min(fraction * full_kelly, max_cap_pct)
    return float(bankroll * stake_fraction)


def compute_financial_slices(
    bets_df: pd.DataFrame,
) -> dict[str, list[FinancialSlice]]:
    """Generate financial diagnostic slices across odds tiers, EV brackets, and CLV tiers."""
    if bets_df.empty:
        return {}

    slices: dict[str, list[FinancialSlice]] = {}
    has_clv_col = "clv_pct" in bets_df.columns and bets_df["clv_pct"].notna().any()

    # Helper to build slice item
    def build_slice(dim: str, bucket: str, sub: pd.DataFrame) -> FinancialSlice:
        n_bets = len(sub)
        wins = int(sub["win"].sum())
        total_staked = float(sub["stake"].sum())
        total_profit = float(sub["profit"].sum())
        win_rate = float(wins / n_bets * 100.0)
        avg_odds = float(sub["odds"].mean())
        avg_prob = float(sub["prob"].mean() * 100.0) if "prob" in sub.columns else None
        implied_prob = float((1.0 / sub["odds"]).mean() * 100.0) if "odds" in sub.columns else None
        calib_gap = float(avg_prob - win_rate) if avg_prob is not None else None
        brier = float(((sub["prob"] - sub["win"]) ** 2).mean()) if "prob" in sub.columns and "win" in sub.columns else None
        avg_clv = float(sub["clv_pct"].mean()) if has_clv_col and sub["clv_pct"].notna().any() else None

        return FinancialSlice(
            dimension=dim,
            bucket=bucket,
            sample_size=n_bets,
            bets_count=n_bets,
            wins_count=wins,
            win_rate_pct=win_rate,
            avg_odds=avg_odds,
            total_staked=total_staked,
            total_profit=total_profit,
            yield_pct=float(total_profit / total_staked * 100.0) if total_staked > 0 else 0.0,
            expected_yield_pct=float(sub["ev"].mean() * 100.0),
            avg_prob_pct=avg_prob,
            implied_prob_pct=implied_prob,
            calibration_gap_pct=calib_gap,
            brier_score=brier,
            avg_clv_pct=avg_clv,
        )

    # 1. Odds tier - 7 canonical market tiers matching timing.py and ISSUE-001
    odds = bets_df["odds"].to_numpy()
    odds_buckets = [
        ("< 1.40 (Ciężki faworyt)", (odds >= 1.0) & (odds < 1.40)),
        ("1.40 - 1.80 (Faworyt)", (odds >= 1.40) & (odds < 1.80)),
        ("1.80 - 2.20 (Wyrównany)", (odds >= 1.80) & (odds < 2.20)),
        ("2.20 - 2.80 (Lekki underdog)", (odds >= 2.20) & (odds < 2.80)),
        ("2.80 - 3.50 (Złoty underdog)", (odds >= 2.80) & (odds < 3.50)),
        ("3.50 - 5.00 (Wysoki underdog)", (odds >= 3.50) & (odds <= 5.00)),
        ("> 5.00 (Ekstremalny longshot)", odds > 5.00),
    ]

    odds_slices: list[FinancialSlice] = []
    for bucket_name, mask in odds_buckets:
        sub = bets_df[mask]
        if len(sub) == 0:
            continue
        odds_slices.append(build_slice("Przedział Kursowy", bucket_name, sub))
    if odds_slices:
        slices["Przedział Kursowy"] = odds_slices

    # 2. EV bracket
    ev = bets_df["ev"].to_numpy()
    ev_buckets = [
        ("Umiarkowane EV (5% - 8%)", (ev >= 0.05) & (ev <= 0.08)),
        ("Średnie EV (8% - 15%)", (ev > 0.08) & (ev <= 0.15)),
        ("Wysokie EV (> 15%)", ev > 0.15),
    ]
    if (ev < 0.05).any():
        ev_buckets.insert(0, ("Niskie EV (< 5%)", ev < 0.05))

    ev_slices: list[FinancialSlice] = []
    for bucket_name, mask in ev_buckets:
        sub = bets_df[mask]
        if len(sub) == 0:
            continue
        ev_slices.append(build_slice("Wielkość Przewagi (EV)", bucket_name, sub))
    if ev_slices:
        slices["Wielkość Przewagi (EV)"] = ev_slices

    # 3. CLV bracket (if closing odds are available)
    if has_clv_col:
        clv_mask_valid = bets_df["clv_pct"].notna()
        clv_buckets = [
            ("Silny CLV (> +5% wyższy kurs)", clv_mask_valid & (bets_df["clv_pct"] > 5.0)),
            ("Lekki CLV (0% do +5%)", clv_mask_valid & (bets_df["clv_pct"] >= 0.0) & (bets_df["clv_pct"] <= 5.0)),
            ("Ujemny CLV (< 0% - linia uciekła)", clv_mask_valid & (bets_df["clv_pct"] < 0.0)),
        ]
        clv_slices: list[FinancialSlice] = []
        for bucket_name, mask in clv_buckets:
            sub = bets_df[mask]
            if len(sub) == 0:
                continue
            clv_slices.append(build_slice("Wartość Względem Zamknięcia (CLV)", bucket_name, sub))
        if clv_slices:
            slices["Wartość Względem Zamknięcia (CLV)"] = clv_slices

    return slices


def run_financial_benchmark(
    data: pd.DataFrame,
    candidate_prob_col: str,
    target_col: str = "y_true",
    odds_a_col: str | None = None,
    odds_b_col: str | None = None,
    odds_col: str | None = None,
    close_odds_a_col: str | None = None,
    close_odds_b_col: str | None = None,
    close_odds_col: str | None = None,
    date_col: str = "date",
    time_horizon: str = "opening (>12h)",
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    max_odds: float = 5.0,
    min_odds: float = 1.05,
    initial_bankroll: float = 10_000.0,
    staking_strategy: str = "flat_100",
    model_name: str = "Candidate Model",
    model_version: str = "v1.0",
) -> FinancialBenchmarkReport:
    """Simulate betting bankroll and compute complete financial & CLV metrics.

    Args:
        data: Evaluation dataset with predictions, targets, and bookmaker odds.
        candidate_prob_col: Column with predicted probability for team A / outcome 1.
        target_col: Binary outcome column (1 = Team A won, 0 = Team B won).
        odds_a_col: Decimal odds for Team A at the bet placement horizon.
        odds_b_col: Decimal odds for Team B at the bet placement horizon.
        odds_col: Single decimal odds column for the predicted side.
        close_odds_a_col: Optional decimal closing odds for Team A (for CLV calculation).
        close_odds_b_col: Optional decimal closing odds for Team B (for CLV calculation).
        close_odds_col: Optional single decimal closing odds column.
        date_col: Timestamp/date column for chronological ordering.
        time_horizon: Label describing entry horizon, e.g. "opening (>12h)" or "mid (2-6h)".
        tax_rate: Polish turnover tax (default 0.12).
        min_ev: Minimum net expected value to place a bet (default 0.05, i.e. +5%).
        max_odds: Upper odds cutoff for safety/quarantine (default 5.0).
        min_odds: Lower odds cutoff (default 1.05).
        initial_bankroll: Starting bankroll in PLN (default 10,000).
        staking_strategy: 'flat_100' (100 PLN flat), 'quarter_kelly' (25% Kelly, cap 2%),
                          or 'fixed_pct_1' (1% of bankroll).
        model_name: Name of the evaluated model.
        model_version: Model version string.

    Returns:
        FinancialBenchmarkReport with summary metrics, CLV analysis, and diagnostic slices.
    """
    sub = data.copy()

    # Determine odds columns
    has_two_sides = (odds_a_col is not None and odds_a_col in sub.columns) and (
        odds_b_col is not None and odds_b_col in sub.columns
    )
    has_single_odds = odds_col is not None and odds_col in sub.columns

    required = [candidate_prob_col, target_col]
    if has_two_sides:
        required.extend([odds_a_col, odds_b_col])  # type: ignore
    elif has_single_odds:
        required.append(odds_col)  # type: ignore
    else:
        raise ValueError(
            f"Either both odds_a_col ({odds_a_col}) and odds_b_col ({odds_b_col}), "
            f"or odds_col ({odds_col}) must exist in dataset."
        )

    has_close_two = (close_odds_a_col is not None and close_odds_a_col in sub.columns) and (
        close_odds_b_col is not None and close_odds_b_col in sub.columns
    )
    has_close_single = close_odds_col is not None and close_odds_col in sub.columns

    sub = sub.dropna(subset=required).copy()
    if date_col in sub.columns:
        sub = sub.sort_values(date_col).reset_index(drop=True)
    else:
        sub = sub.reset_index(drop=True)

    total_evaluated = len(sub)
    if total_evaluated == 0:
        raise ValueError("No valid rows remaining for financial benchmark.")

    bankroll = float(initial_bankroll)
    peak = float(initial_bankroll)
    max_drawdown_pct = 0.0
    max_drawdown_pln = 0.0

    bet_records: list[dict[str, Any]] = []
    clv_values: list[float] = []
    beat_closing_flags: list[int] = []

    for _, row in sub.iterrows():
        prob_a = float(row[candidate_prob_col])
        prob_b = 1.0 - prob_a
        actual_win_a = int(row[target_col]) == 1

        chosen_side: str | None = None
        chosen_prob = 0.0
        chosen_odds = 0.0
        chosen_close_odds: float | None = None
        chosen_ev = 0.0
        is_win = False

        if has_two_sides:
            o_a = float(row[odds_a_col])
            o_b = float(row[odds_b_col])
            ev_a = prob_a * o_a * (1.0 - tax_rate) - 1.0 if o_a > 1.0 else -1.0
            ev_b = prob_b * o_b * (1.0 - tax_rate) - 1.0 if o_b > 1.0 else -1.0

            # Pick the side with higher positive EV
            if ev_a >= ev_b and ev_a >= min_ev and min_odds <= o_a <= max_odds:
                chosen_side = "team_a"
                chosen_prob = prob_a
                chosen_odds = o_a
                chosen_ev = ev_a
                is_win = actual_win_a
                if has_close_two:
                    chosen_close_odds = float(row[close_odds_a_col])
            elif ev_b > ev_a and ev_b >= min_ev and min_odds <= o_b <= max_odds:
                chosen_side = "team_b"
                chosen_prob = prob_b
                chosen_odds = o_b
                chosen_ev = ev_b
                is_win = not actual_win_a
                if has_close_two:
                    chosen_close_odds = float(row[close_odds_b_col])
        else:
            o = float(row[odds_col])
            ev = prob_a * o * (1.0 - tax_rate) - 1.0 if o > 1.0 else -1.0
            if ev >= min_ev and min_odds <= o <= max_odds:
                chosen_side = "team_a"
                chosen_prob = prob_a
                chosen_odds = o
                chosen_ev = ev
                is_win = actual_win_a
                if has_close_single:
                    chosen_close_odds = float(row[close_odds_col])

        if chosen_side is None:
            continue

        # Calculate stake based on policy
        if staking_strategy == "flat_100":
            stake = min(100.0, bankroll)
        elif staking_strategy == "quarter_kelly":
            stake = calculate_quarter_kelly(
                probability=chosen_prob,
                odds=chosen_odds,
                tax_rate=tax_rate,
                bankroll=bankroll,
                fraction=0.25,
                max_cap_pct=0.02,
            )
            stake = min(stake, bankroll)
        elif staking_strategy == "fixed_pct_1":
            stake = bankroll * 0.01
        else:
            raise ValueError(f"Unknown staking strategy: {staking_strategy}")

        if stake <= 0.0:
            continue

        # Calculate PnL
        if is_win:
            net_profit = stake * (chosen_odds * (1.0 - tax_rate) - 1.0)
        else:
            net_profit = -stake

        bankroll += net_profit
        peak = max(peak, bankroll)
        current_dd_pln = peak - bankroll
        current_dd_pct = (current_dd_pln / peak) * 100.0 if peak > 0 else 0.0

        if current_dd_pct > max_drawdown_pct:
            max_drawdown_pct = current_dd_pct
        if current_dd_pln > max_drawdown_pln:
            max_drawdown_pln = current_dd_pln

        clv_pct: float | None = None
        if chosen_close_odds is not None and chosen_close_odds > 1.0:
            clv = (chosen_odds - chosen_close_odds) / chosen_close_odds
            clv_pct = float(clv * 100.0)
            clv_values.append(clv_pct)
            beat_closing_flags.append(int(chosen_odds > chosen_close_odds))

        bet_records.append(
            {
                "date": row.get(date_col, None),
                "side": chosen_side,
                "prob": chosen_prob,
                "odds": chosen_odds,
                "close_odds": chosen_close_odds,
                "clv_pct": clv_pct,
                "ev": chosen_ev,
                "stake": stake,
                "win": int(is_win),
                "profit": net_profit,
                "bankroll": bankroll,
            }
        )

    bets_df = pd.DataFrame(bet_records)
    n_bets = len(bets_df)

    avg_clv_pct = float(np.mean(clv_values)) if clv_values else None
    beat_closing_rate_pct = float(np.mean(beat_closing_flags) * 100.0) if beat_closing_flags else None

    if n_bets > 0:
        total_staked = float(bets_df["stake"].sum())
        total_profit = float(bets_df["profit"].sum())
        wins_count = int(bets_df["win"].sum())
        losses_count = n_bets - wins_count
        win_rate_pct = float(wins_count / n_bets * 100.0)
        avg_odds = float(bets_df["odds"].mean())
        avg_stake = float(bets_df["stake"].mean())
        roi_pct = float((bankroll - initial_bankroll) / initial_bankroll * 100.0)
        yield_pct = float(total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
        expected_yield_pct = float(bets_df["ev"].mean() * 100.0)
        expected_profit = float((bets_df["stake"] * bets_df["ev"]).sum())
        bet_rate_pct = float(n_bets / total_evaluated * 100.0)
    else:
        total_staked = 0.0
        total_profit = 0.0
        wins_count = 0
        losses_count = 0
        win_rate_pct = 0.0
        avg_odds = 0.0
        avg_stake = 0.0
        roi_pct = 0.0
        yield_pct = 0.0
        expected_yield_pct = 0.0
        expected_profit = 0.0
        bet_rate_pct = 0.0

    summary = FinancialBenchmarkSummary(
        tax_rate=tax_rate,
        min_ev=min_ev,
        max_odds=max_odds,
        staking_strategy=staking_strategy,
        time_horizon=time_horizon,
        initial_bankroll=float(initial_bankroll),
        final_bankroll=float(bankroll),
        total_matches_evaluated=total_evaluated,
        bets_count=n_bets,
        bet_rate_pct=bet_rate_pct,
        wins_count=wins_count,
        losses_count=losses_count,
        win_rate_pct=win_rate_pct,
        avg_odds=avg_odds,
        avg_stake=avg_stake,
        total_staked=total_staked,
        total_profit=total_profit,
        roi_pct=roi_pct,
        yield_pct=yield_pct,
        expected_yield_pct=expected_yield_pct,
        expected_profit=expected_profit,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_pln=max_drawdown_pln,
        avg_clv_pct=avg_clv_pct,
        beat_closing_rate_pct=beat_closing_rate_pct,
    )

    slices = compute_financial_slices(bets_df)

    cohort_dates = sorted(sub[date_col].dropna().astype(str).unique()) if date_col in sub.columns else []
    cohort_start = cohort_dates[0] if cohort_dates else None
    cohort_end = cohort_dates[-1] if cohort_dates else None

    return FinancialBenchmarkReport(
        model_name=model_name,
        model_version=model_version,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        summary=summary,
        slices=slices,
    )
