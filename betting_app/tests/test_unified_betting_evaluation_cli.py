"""Financial CLI formatting without replaying sealed studies or accessing databases."""

import json
import sys

import pytest

from scripts import run_unified_betting_evaluation as unified_cli
from scripts.scraped import evaluate_a1_scraped as a1_cli
from scripts.scraped import run_48h_betting_benchmark as h48_cli
from src.analysis.unified_evaluation_engine import CalibrationBin, SimulationSummary, UnifiedBettingEngine


@pytest.mark.parametrize(
    ("value", "spec", "suffix", "expected"),
    [
        (None, "+.2f", " zł", "N/A (no eligible bets)"),
        (None, ".1f", "%", "N/A (no eligible bets)"),
        (12.345, "+.2f", " zł", "+12.35 zł"),
        (50.0, ".1f", "%", "50.0%"),
        (0.0, "+.2f", "%", "+0.00%"),
    ],
)
def test_conditional_metric_preserves_numbers_and_labels_missing_values(value, spec, suffix, expected):
    assert unified_cli.format_conditional_metric(value, spec, suffix) == expected


def test_empty_legacy_cli_completes_with_undefined_conditions_and_zero_cash(tmp_path, monkeypatch, capsys):
    data = tmp_path / "empty.csv"
    data.write_text("match_id,best_a_open,best_b_open,p_full,y\n")
    monkeypatch.setattr(sys, "argv", ["run_unified_betting_evaluation.py", "--data", str(data)])

    unified_cli.main()

    output = capsys.readouterr().out
    assert "Liczba zakładów:                  0" in output
    for label in (
        "Skuteczność (Win Rate):",
        "Średni kurs:",
        "Surowe (niekalibrowane) EV / bet:",
        "Oczekiwane EV / bet (%):",
        "Oczekiwany zysk na 1 bet:",
        "Rzeczywiste EV / bet (ROI %):",
        "Rzeczywisty zysk na 1 bet:",
        "Błąd kalibracji na 1 bet:",
    ):
        assert next(line for line in output.splitlines() if line.startswith(label)).endswith("N/A (no eligible bets)")
    assert "Całkowity obrót:                  0.00 zł" in output
    assert "Zysk całkowity:                   +0.00 zł" in output


@pytest.mark.parametrize("module", [a1_cli, h48_cli])
def test_scraped_empty_portfolio_formatting_requires_no_financial_replay(module, capsys):
    summary = UnifiedBettingEngine().run_simulation([], initial_bankroll=1000.0)

    module.print_portfolio_summary(summary)

    output = capsys.readouterr().out
    assert "N/A (no eligible bets)" in output
    assert "Net Cash Profit:" in output
    assert "+0.00 zł" in output
    assert "1000.00 zł" in output
    assert "0.00%" in output  # Valid unconditional drawdown, not a conditional ROI.
    roi_line = next(line for line in output.splitlines() if line.startswith("Realized ROI"))
    assert roi_line.endswith("N/A (no eligible bets)")


def test_saved_populated_portfolio_keeps_displayed_numerics_without_replay(capsys):
    # Read the existing sealed result, never execute its historical financial study.
    path = unified_cli.PROJECT_ROOT / "data/08_reporting/48h_betting_benchmark_results.json"
    payload = json.loads(path.read_text())["tax_0"]
    for key in ("bins", "by_odds_bracket"):
        payload[key] = [CalibrationBin(**row) for row in payload[key]]
    summary = SimulationSummary(**payload)

    h48_cli.print_portfolio_summary(summary)

    output = capsys.readouterr().out
    for line in (
        "Qualified Bets:       18",
        "Average Odds Taken:   2.22",
        "Total Staked:         313.95 zł",
        "Net Cash Profit:      +164.33 zł",
        "Ending Bankroll:      1164.33 zł (from 1,000 zł initial)",
        "Realized ROI / Yield: +52.34%",
        "Expected ROI / EV:    +3.91%",
        "Calibration Gap (PLN):+8.45 zł / bet",
        "Maximum Drawdown:     1.83%",
        "Median Realized CLV:  +20.34% (vs market close)",
    ):
        assert line in output
    assert "N/A (no eligible bets)" not in output

    a1_cli.print_portfolio_summary(summary)

    output = capsys.readouterr().out
    assert "Win Rate:           66.7% (12/18)" in output
    assert "Realized ROI:       +52.34%" in output
    assert "Net Cash Profit:    +164.33 zł" in output
