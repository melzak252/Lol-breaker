"""Actual benchmark artifacts remain strict JSON and independently rescorable."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys

import pytest

from scripts import run_tournament_distribution_benchmark as benchmark
from src.analysis import tournament_metrics as metrics
from src.models import tournament_pair_adapter as pair_module
from src.models.tournament_pair_adapter import TournamentPairAdapter


def _strict_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_benchmark_zero_winner_reports_strict_json_and_saved_forecast_rescores_actual_law(
    tmp_path, monkeypatch,
):
    edition_id = "msi-2023"
    event_title = "Owned report-integrity cup"
    teams = ["GEN", "JDG", "MAD", "C9", "T1", "BLG", "G2", "GG"]
    source = tmp_path / "series_results.csv"
    with source.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "event_title", "status", "scheduled_at_utc", "stage", "section_path",
            "team1", "team2", "winner_side", "best_of", "team1_score", "team2_score",
        ])
        writer.writeheader()
        writer.writerow({
            "event_title": "Earlier domestic cup", "status": "completed",
            "scheduled_at_utc": "2023-04-01T00:00:00Z", "stage": "playoffs",
            "section_path": "Finals", "team1": "JDG", "team2": "BLG",
            "winner_side": "2", "best_of": 3, "team1_score": 1, "team2_score": 2,
        })
        writer.writerow({
            "event_title": event_title, "status": "completed",
            "scheduled_at_utc": "2023-05-21T00:00:00Z", "stage": "playoffs",
            "section_path": "Finals", "team1": "GEN", "team2": "T1",
            "winner_side": "1", "best_of": 5, "team1_score": 3, "team2_score": 1,
        })
    manifest = {
        "schema_version": 1,
        "sources": [{"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "editions": {edition_id: {
            "name": event_title, "event_title": event_title, "team_count": 8,
            "profile": "msi-2023-2026-double-elimination", "entrants": teams,
            "cutoff": "2023-05-08T00:00:00Z",
        }},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    source_before = source.read_bytes()
    manifest_before = manifest_path.read_bytes()
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(benchmark, "DEFAULT_EDITIONS", (edition_id,))
    monkeypatch.setattr(benchmark, "DEFAULT_MODELS", ("uniform",))
    monkeypatch.setattr(pair_module, "DEFAULT_SERIES_RESULTS_PATH", source)
    monkeypatch.setattr(metrics, "DEFAULT_SERIES_RESULTS_PATH", source)

    # One Monte Carlo draw cannot cover every possible actual champion. Find a
    # fixed small-budget counterexample using the real uniform predictor/bracket.
    adapter = TournamentPairAdapter(source)
    for seed in range(16):
        actual_forecast = benchmark.run_simulation(
            edition_id, "uniform", manifest, adapter, simulations=1, seed=seed,
        )
        if actual_forecast["champion_prob"]["GEN"] == 0:
            break
    else:
        pytest.fail("The deterministic counterexample did not expose an unseen champion")
    ground_truth = metrics.extract_tournament_ground_truth(event_title)
    actual_score = metrics.evaluate_tournament_edition(edition_id, actual_forecast, ground_truth)
    assert actual_score["champion_log_loss"] == math.inf
    law = next(row for row in actual_forecast["pair_scoreline_laws"]["rows"]
               if row["team_a"] == "GEN" and row["team_b"] == "T1" and row["best_of"] == 5)
    expected_scoreline_loss = -math.log(law["scoreline_pmf"]["3-1"])
    expected_length_rps = metrics.compute_series_length_rps(
        {int(games): p for games, p in law["series_length_pmf"].items()}, 4, 5,
    )
    assert actual_score["playoff_scoreline_log_loss"] == pytest.approx(expected_scoreline_loss)
    assert actual_score["playoff_series_length_rps"] == pytest.approx(expected_length_rps)

    run_id = "owned-zero-champion"
    report_dir = tmp_path / "reports" / run_id
    returned = benchmark.run_benchmark(run_id, simulations=1, seed=seed, output_base=tmp_path / "reports")
    persisted = {path.stem: _strict_json(path) for path in report_dir.glob("*.json")}
    assert returned == persisted["decision"]
    assert returned["native_adoption_eligible"] is False
    assert returned["model_comparison"]["uniform"]["mean_champion_log_loss"] is None
    assert returned["nonfinite_reasons"]["$.model_comparison.uniform.mean_champion_log_loss"] == (
        "positive_infinity_zero_observed_probability"
    )
    detailed = persisted["metrics"]["detailed_scores"][0]
    assert detailed["champion_log_loss"] is None
    assert persisted["metrics"]["nonfinite_reasons"]["$.detailed_scores[0].champion_log_loss"] == (
        "positive_infinity_zero_observed_probability"
    )
    assert detailed["playoff_scoreline_log_loss"] == pytest.approx(expected_scoreline_loss)
    assert detailed["playoff_series_length_rps"] == pytest.approx(expected_length_rps)
    assert actual_score["champion_log_loss"] == math.inf

    forecast_path = (tmp_path / "data/07_model_output/tournament_distribution" / run_id
                     / f"{edition_id}_uniform_distributions.json")
    loaded = _strict_json(forecast_path)
    rescored = metrics.evaluate_tournament_edition(edition_id, loaded, ground_truth)
    assert rescored["champion_log_loss"] == math.inf
    assert rescored["playoff_scoreline_log_loss"] == pytest.approx(expected_scoreline_loss)
    assert rescored["playoff_series_length_rps"] == pytest.approx(expected_length_rps)

    score_path = tmp_path / "loaded-score.json"
    monkeypatch.setattr(sys, "argv", [
        "run_tournament_distribution_benchmark", "--score", "--edition", edition_id,
        "--forecast", str(forecast_path), "--output", str(score_path),
    ])
    benchmark.main()
    cli_score = _strict_json(score_path)
    assert cli_score["playoff_scoreline_log_loss"] == pytest.approx(expected_scoreline_loss)
    assert cli_score["playoff_series_length_rps"] == pytest.approx(expected_length_rps)
    assert cli_score["champion_log_loss"] is None
    assert cli_score["nonfinite_reasons"]["$.champion_log_loss"] == (
        "positive_infinity_zero_observed_probability"
    )
    assert source.read_bytes() == source_before
    assert manifest_path.read_bytes() == manifest_before

    outputs_before = {path: path.read_bytes() for directory in (report_dir, forecast_path.parent)
                      for path in directory.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(run_id, simulations=1, seed=seed, output_base=tmp_path / "reports")
    assert {path: path.read_bytes() for path in outputs_before} == outputs_before
