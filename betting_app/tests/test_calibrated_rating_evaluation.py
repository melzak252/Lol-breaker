from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "05_ratingi_baseline"
    / "05j_evaluate_calibrated_glicko2.py"
)


def _load_evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "calibrated_rating_evaluation", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator at {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()


def _match(
    match_id: str,
    match_date: str,
    tournament: str,
    team_1: str,
    team_2: str,
    *,
    team_1_wins: bool,
) -> dict[str, Any]:
    players_1 = [f"{team_1}-p{index}" for index in range(5)]
    players_2 = [f"{team_2}-p{index}" for index in range(5)]
    return {
        "match_id": match_id,
        "date": match_date,
        "tournament": tournament,
        "tid_1": team_1,
        "tid_2": team_2,
        "name_1": team_1,
        "name_2": team_2,
        "players_1": players_1,
        "players_2": players_2,
        "games": [
            {
                "game_id": f"{match_id}-g1",
                "t1_id": team_1,
                "t2_id": team_2,
                "t1_win": team_1_wins,
                "t2_win": not team_1_wins,
            }
        ],
    }


def _write_matches(path: Path, matches: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(matches), encoding="utf-8")


def _write_baseline(
    path: Path, rows: list[tuple[str, str, float, int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("golgg_match_id", "date", "player_gl", "y_true"),
            lineterminator="\n",
        )
        writer.writeheader()
        for match_id, match_date, probability, y_true in rows:
            writer.writerow(
                {
                    "golgg_match_id": match_id,
                    "date": match_date,
                    "player_gl": probability,
                    "y_true": y_true,
                }
            )


def test_same_day_domestic_results_do_not_leak_affiliations_into_bridge(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "matches.json"
    _write_matches(
        matches_path,
        [
            _match("3", "2023-12-31", "Worlds 2023", "kr", "cn", team_1_wins=True),
            _match("2", "2023-12-31", "LPL 2023", "cn", "cn-other", team_1_wins=True),
            _match("1", "2023-12-31", "LCK 2023", "kr", "kr-other", team_1_wins=True),
            _match("4", "2024-01-01", "Worlds 2024", "kr", "cn", team_1_wins=False),
        ],
    )

    records, _ = evaluator.load_matches(matches_path)
    predictions = evaluator.replay_candidate(records)

    same_day_bridge = predictions["3"]
    assert same_day_bridge.affiliation_1 is None
    assert same_day_bridge.affiliation_2 is None

    next_day_bridge = predictions["4"]
    assert next_day_bridge.affiliation_1.family == "LCK"
    assert next_day_bridge.affiliation_2.family == "LPL"
    assert next_day_bridge.affiliation_1.source_date.isoformat() == "2023-12-31"
    assert next_day_bridge.affiliation_2.source_date.isoformat() == "2023-12-31"


def test_alignment_is_by_match_id_and_candidate_probability_tracks_team_1(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "matches.json"
    baseline_path = tmp_path / "baseline.csv"
    _write_matches(
        matches_path,
        [
            _match("1", "2023-12-31", "LCK 2023", "a", "b", team_1_wins=True),
            _match("10", "2024-01-10", "LCK 2024", "a", "b", team_1_wins=True),
            _match("11", "2024-01-11", "LCK 2024", "b", "a", team_1_wins=False),
        ],
    )
    # Deliberately reverse CSV order and use distinct probabilities. A positional
    # join would silently attach each control prediction to the wrong side.
    _write_baseline(
        baseline_path,
        [
            ("11", "2024-01-11", 0.2, 0),
            ("10", "2024-01-10", 0.8, 1),
        ],
    )

    records, _ = evaluator.load_matches(matches_path)
    candidates = evaluator.replay_candidate(records)
    baseline = evaluator.load_baseline(baseline_path)
    rows, counts = evaluator.align_evaluation_rows(records, candidates, baseline)

    assert [row.match.match_id for row in rows] == ["10", "11"]
    assert [row.legacy_probability for row in rows] == [0.8, 0.2]
    assert rows[0].candidate.probability > 0.5
    assert rows[1].candidate.probability < 0.5
    assert counts["diagnostic_rows_2024_plus"] == 2

    wrong_label = dict(baseline)
    wrong_label["10"] = evaluator.BaselinePrediction(
        match_id="10",
        match_date=baseline["10"].match_date,
        probability=0.8,
        y_true=0,
    )
    with pytest.raises(ValueError, match="label mismatch for 10"):
        evaluator.align_evaluation_rows(records, candidates, wrong_label)


def test_repeated_same_day_player_excludes_every_affected_comparison_row(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "matches.json"
    baseline_path = tmp_path / "baseline.csv"
    repeated_player_match = _match(
        "11", "2024-01-10", "LCK 2024", "c", "d", team_1_wins=False
    )
    repeated_player_match["players_1"][0] = "a-p0"
    _write_matches(
        matches_path,
        [
            _match("10", "2024-01-10", "LCK 2024", "a", "b", team_1_wins=True),
            repeated_player_match,
            _match("12", "2024-01-10", "LCK 2024", "e", "f", team_1_wins=True),
        ],
    )
    _write_baseline(
        baseline_path,
        [
            ("12", "2024-01-10", 0.55, 1),
            ("11", "2024-01-10", 0.45, 0),
            ("10", "2024-01-10", 0.60, 1),
            ("missing", "2024-01-10", 0.50, 1),
        ],
    )

    records, _ = evaluator.load_matches(matches_path)
    candidates = evaluator.replay_candidate(records)
    rows, counts = evaluator.align_evaluation_rows(
        records,
        candidates,
        evaluator.load_baseline(baseline_path),
    )

    assert set(candidates) == {"10", "11", "12"}
    assert [row.match.match_id for row in rows] == ["12"]
    assert counts["diagnostic_rows_before_same_day_exclusion"] == 3
    assert counts["diagnostic_rows_excluded_repeated_player_same_date"] == 2
    assert counts["diagnostic_rows_2024_plus"] == 1
    assert counts["baseline_without_eligible_match"] == 1


def test_artifacts_are_byte_deterministic_and_never_claim_promotion(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "matches.json"
    baseline_path = tmp_path / "baseline.csv"
    output_1 = tmp_path / "output-1"
    output_2 = tmp_path / "output-2"
    _write_matches(
        matches_path,
        [
            _match("1", "2023-12-31", "LCK 2023", "a", "b", team_1_wins=True),
            _match("2", "2024-01-01", "LCK 2024", "a", "b", team_1_wins=True),
            _match("3", "2024-02-01", "LCK 2024", "b", "a", team_1_wins=True),
        ],
    )
    _write_baseline(
        baseline_path,
        [
            ("3", "2024-02-01", 0.55, 1),
            ("2", "2024-01-01", 0.60, 1),
        ],
    )

    evaluator.run_evaluation(matches_path, baseline_path, output_1)
    evaluator.run_evaluation(matches_path, baseline_path, output_2)

    assert (output_1 / "summary.json").read_bytes() == (
        output_2 / "summary.json"
    ).read_bytes()
    assert (output_1 / "predictions.csv").read_bytes() == (
        output_2 / "predictions.csv"
    ).read_bytes()
    summary = json.loads((output_1 / "summary.json").read_text(encoding="utf-8"))
    assert summary["inputs"]["odds_files_read"] is False
    assert summary["inputs"]["odds_values_used"] is False
    assert summary["gates"]["prospective_promotion"]["available"] is False
    assert summary["operational_default_replacement"]["available"] is False
    assert set(summary["metrics"]["overall"]["candidate"]) >= {
        "n",
        "date_min",
        "date_max",
        "log_loss",
        "brier",
        "auc",
        "ece",
    }


def test_bridge_blocks_use_unordered_family_pair_and_year(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "matches.json"
    baseline_path = tmp_path / "baseline.csv"
    _write_matches(
        matches_path,
        [
            _match("1", "2023-12-31", "LCK 2023", "kr", "kr-other", team_1_wins=True),
            _match("2", "2023-12-31", "LPL 2023", "cn", "cn-other", team_1_wins=True),
            _match("3", "2024-01-10", "Worlds 2024", "kr", "cn", team_1_wins=True),
            _match("4", "2024-02-10", "MSI 2024", "cn", "kr", team_1_wins=False),
            _match("5", "2024-02-11", "LCK 2024", "kr", "kr-other", team_1_wins=True),
            _match("6", "2025-01-10", "Worlds 2025", "kr", "cn", team_1_wins=False),
        ],
    )
    _write_baseline(
        baseline_path,
        [
            ("3", "2024-01-10", 0.60, 1),
            ("4", "2024-02-10", 0.40, 0),
            ("5", "2024-02-11", 0.55, 1),
            ("6", "2025-01-10", 0.45, 0),
        ],
    )

    _, rows = evaluator.evaluate(matches_path, baseline_path)
    blocks_by_id = {
        row.match.match_id: evaluator.bootstrap_block_key(row) for row in rows
    }

    assert blocks_by_id["3"] == "family-pair:LCK|LPL/year:2024"
    assert blocks_by_id["4"] == blocks_by_id["3"]
    assert blocks_by_id["6"] == "family-pair:LCK|LPL/year:2025"
    assert blocks_by_id["5"] == "month:2024-02"
