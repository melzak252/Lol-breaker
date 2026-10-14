"""Behavioral contracts for the local Glicko-derived tournament diagnostic.

These tests use real predictors, scoring rules, brackets and filesystem receipts.
They do not run the archived study or authorize native-model adoption.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_tournament_distribution_benchmark as benchmark
from src.analysis.tournament_metrics import (
    ORDERED_SWISS_RECORDS,
    compute_champion_brier,
    compute_swiss_record_rps,
    verify_network_flow_conservation,
)
from src.models import tournament_pair_adapter as pair_module
from src.models.tournament_pair_adapter import TournamentPairAdapter


ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "uniform", "glicko_format", "regional_glicko_proxy",
    "calibrated_regional_glicko_proxy",
)
TARGET = "calibrated_regional_glicko_proxy"
CUTOFF = "2023-05-08T00:00:00Z"
TEAMS = ["GEN", "JDG", "MAD", "C9", "T1", "BLG", "G2", "GG"]
METRICS = ("mean_champion_brier", "mean_swiss_record_rps")
SUPPORT = ("champion_brier_edition_count", "swiss_record_rps_edition_count")
HEADER = "status,scheduled_at_utc,team1,team2,winner_side,best_of\n"
REQUIRED_CODE = (
    "src/models/tournament_pair_adapter.py",
    "src/models/calibrated_tournament_model.py",
    "src/ratings/glicko2_core.py",
    "scripts/run_tournament_distribution_benchmark.py",
    "src/models/tournament_catalog.py",
    "src/models/tournament_formats.py",
    "src/models/tournament_scorelines.py",
    "src/analysis/tournament_metrics.py",
    "src/models/tournament_bracket_simulator.py",
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "series_results.csv"
    # June outcomes initialize nonzero, side-symmetric earlier-only Glicko deltas.
    # Each prior-year CAL format has both wins and losses, not a separable fit.
    rows = [
        "completed,2022-06-01T00:00:00Z,GEN,G2,1,3\n",
        "completed,2022-06-01T00:00:00Z,T1,BLG,2,3\n",
    ]
    for day, bo in enumerate((1, 3, 5), start=1):
        rows.extend([
            f"completed,2022-07-0{day}T00:00:00Z,GEN,G2,1,{bo}\n",
            f"completed,2022-07-0{day}T00:00:00Z,T1,BLG,2,{bo}\n",
            f"completed,2022-07-0{day}T00:00:00Z,GEN,G2,2,{bo}\n",
        ])
    rows.extend([
        "completed,2023-01-01T00:00:00Z,GEN,G2,1,3\n",
        "completed,2023-01-02T00:00:00Z,T1,BLG,2,3\n",
    ])
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _manifest():
    return {"editions": {"local-proxy": {
        "profile": "msi-2023-2026-double-elimination",
        "entrants": TEAMS, "cutoff": CUTOFF,
    }}}


def _simulate(source, model):
    return benchmark.run_simulation(
        "local-proxy", model, _manifest(), TournamentPairAdapter(source),
        simulations=24, seed=117,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_receipt(receipt, source, model, teams, cutoff):
    assert receipt["schema_version"] == 1
    assert receipt["model_kind"] == model
    assert receipt["predictor_kind"] == (
        "reference_null" if model == "uniform" else
        "prior_calibrated_glicko_format" if model == "glicko_format" else
        "glicko_derived_proxy"
    )
    assert receipt["teams"] == teams
    assert receipt["cutoff"] == cutoff
    assert receipt["source"]["path"] == str(source)
    assert receipt["source"]["sha256"] == _sha(source)
    assert isinstance(receipt["formula"], str) and receipt["formula"].strip()
    assert receipt["formula_sha256"] == hashlib.sha256(
        receipt["formula"].encode("utf-8")
    ).hexdigest()
    config = receipt["configuration"]
    assert config["best_ofs"] == [1, 3, 5]
    assert receipt["configuration_sha256"] == hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if model == "glicko_format":
        adapter = TournamentPairAdapter(source)
        calibration = adapter.fitted_format_slopes(cutoff, (1, 3, 5))
        assert config["calibration"] == calibration
        assert config["format_slopes"] == {
            bo: parameters["slope"] for bo, parameters in calibration["formats"].items()
        }
        assert calibration["model_year"] == adapter._utc_clock(cutoff).year
        assert calibration["cal_start"] == f"{calibration['model_year'] - 1}-07-01"
        assert calibration["cal_end_exclusive"] == f"{calibration['model_year']}-01-01"
        for bo, parameters in calibration["formats"].items():
            values = []
            for match in adapter._matches:
                day = adapter._utc_clock(match.scheduled_at).date().isoformat()
                if match.best_of != int(bo) or not calibration["cal_start"] <= day < calibration["cal_end_exclusive"]:
                    continue
                earlier_ratings = adapter.compute_team_ratings(day + "T00:00:00Z")
                delta = adapter._rating_delta(
                    earlier_ratings[match.team1], earlier_ratings[match.team2],
                )
                assert delta != 0
                values.append((delta, match.score1, match.scheduled_at, match.team1, match.team2))
            assert parameters["rows"] == len(values) == 3
            assert parameters["slope"] > 0
            assert parameters["max_label_day"] == max(row[2] for row in values)
            assert parameters["membership_sha256"] == hashlib.sha256(
                json.dumps(values, separators=(",", ":")).encode()
            ).hexdigest()
        assert config["probability_clip"] is None
        assert config["calibration_policy"]["fit_intercept"] is False
        assert config["calibration_policy"]["slope_bounds"] == [0.0, None]
        assert "d_prior_day" in receipt["formula"]
    elif model != "uniform":
        assert config["format_slopes"] == {
            str(bo): slope for bo, slope in pair_module.FORMAT_SLOPES.items()
        }
    if model in ("regional_glicko_proxy", TARGET):
        assert config["regional_gamma"] == 0.70
        assert config["regional_temperature"] == 1.08
        assert config["regional_offsets"] == pair_module.REGIONAL_OFFSETS
    if model == TARGET:
        assert config["pairwise_cap"] == 0.88
        assert config["temperature"] == 1.12
        assert config["entropy_dampening"] == 0.08
    for relative_path in REQUIRED_CODE:
        assert receipt["code_sha256"][relative_path] == _sha(ROOT / relative_path)


@pytest.mark.parametrize("native_name", ["c0", "a0"])
def test_native_names_are_explicitly_refused(source, native_name):
    adapter = TournamentPairAdapter(source)
    with pytest.raises(ValueError, match="(?i)(native|unsupported|model_kind)"):
        adapter.predict_pair_table(TEAMS, CUTOFF, model_kind=native_name)
    with pytest.raises(ValueError, match="(?i)(native|unsupported|model_kind)"):
        _simulate(source, native_name)



@pytest.mark.parametrize("native_name", ["c0", "a0"])
def test_native_refusal_precedes_edition_lookup_and_bracket_construction(source, monkeypatch, native_name):
    def forbidden_simulator(*args, **kwargs):
        pytest.fail("native selection reached simulation")

    monkeypatch.setattr(benchmark, "TournamentBracketSimulator", forbidden_simulator)
    with pytest.raises(ValueError, match="(?i)(native|unsupported|model_kind)"):
        benchmark.run_simulation(
            "missing-edition", native_name, {"editions": {}}, TournamentPairAdapter(source),
            simulations=1,
        )

@pytest.mark.parametrize("native_name", ["c0", "a0"])
def test_cli_refuses_native_names_before_simulating(source, monkeypatch, native_name):
    monkeypatch.setattr(pair_module, "DEFAULT_SERIES_RESULTS_PATH", source)
    monkeypatch.setattr(benchmark.sys, "argv", [
        "run_tournament_distribution_benchmark.py", "--simulate",
        "--edition", "msi-2023", "--model", native_name, "--simulations", "1",
    ])
    with pytest.raises(SystemExit) as refusal:
        benchmark.main()
    assert refusal.value.code == 2


@pytest.mark.parametrize("model", MODELS)
def test_small_real_proxy_simulation_binds_identity_and_provenance(source, model):
    result = _simulate(source, model)
    assert result["model_kind"] == model
    _assert_receipt(result["predictor_receipt"], source, model, TEAMS, CUTOFF)
    assert sum(result["champion_prob"].values()) == pytest.approx(1.0, abs=1e-10)
    assert sum(result["final_prob"].values()) == pytest.approx(2.0, abs=1e-10)
    flow = verify_network_flow_conservation(result["champion_prob"], result["final_prob"])
    assert flow["feasible"] is True


@pytest.mark.parametrize("model", MODELS)
def test_proxy_cli_forecast_persists_actual_model_source_and_closed_cal_receipt(source, tmp_path, monkeypatch, model):
    manifest = {"editions": {"msi-2023": _manifest()["editions"]["local-proxy"]}}
    output = tmp_path / f"{model}_forecast.json"
    monkeypatch.setattr(benchmark, "load_manifest", lambda: manifest)
    monkeypatch.setattr(pair_module, "DEFAULT_SERIES_RESULTS_PATH", source)
    monkeypatch.setattr(benchmark.sys, "argv", [
        "run_tournament_distribution_benchmark.py", "--simulate",
        "--edition", "msi-2023", "--model", model, "--simulations", "24",
        "--seed", "117", "--output", str(output),
    ])
    benchmark.main()
    forecast = json.loads(output.read_text(encoding="utf-8"))
    assert forecast["edition_id"] == "msi-2023"
    assert forecast["model_kind"] == model
    assert forecast["cutoff"] == CUTOFF
    _assert_receipt(forecast["predictor_receipt"], source, model, TEAMS, CUTOFF)


@pytest.mark.parametrize("model", MODELS[1:])
def test_receipt_hash_binds_loaded_history_not_later_path_contents(source, model):
    adapter = TournamentPairAdapter(source)
    loaded_sha = _sha(source)
    before = adapter.predict_pair_table(TEAMS, CUTOFF, model)
    before_receipt = adapter.predictor_receipt(TEAMS, CUTOFF, model)
    source.write_text(HEADER, encoding="utf-8")
    result = benchmark.run_simulation(
        "local-proxy", model, _manifest(), adapter, simulations=24, seed=117,
    )
    assert result["predictor_receipt"] == before_receipt
    assert result["predictor_receipt"]["source"]["sha256"] == loaded_sha
    assert loaded_sha != _sha(source)
    assert adapter.predict_pair_table(TEAMS, CUTOFF, model) == before
    reloaded = TournamentPairAdapter(source)
    if model == "glicko_format":
        with pytest.raises(ValueError, match="unavailable_prior_calibrated_glicko_formats"):
            reloaded.predict_pair_table(TEAMS, CUTOFF, model)
    else:
        assert reloaded.predict_pair_table(TEAMS, CUTOFF, model) != before


def test_source_and_configuration_receipts_track_changed_predictions(source, monkeypatch):
    before = _simulate(source, TARGET)
    before_table = TournamentPairAdapter(source).predict_pair_table(TEAMS, CUTOFF, TARGET)
    source.write_text(
        source.read_text(encoding="utf-8")
        + "completed,2023-01-03T00:00:00Z,GEN,G2,2,3\n",
        encoding="utf-8",
    )
    after_source = _simulate(source, TARGET)
    after_table = TournamentPairAdapter(source).predict_pair_table(TEAMS, CUTOFF, TARGET)
    assert before_table["GEN", "G2", 3] != after_table["GEN", "G2", 3]
    assert before["predictor_receipt"]["source"]["sha256"] != after_source["predictor_receipt"]["source"]["sha256"]
    assert before["predictor_receipt"]["configuration_sha256"] == after_source["predictor_receipt"]["configuration_sha256"]

    monkeypatch.setattr(pair_module, "REGIONAL_OFFSETS", {
        **pair_module.REGIONAL_OFFSETS, "LCK": 0.0,
    })
    after_config = _simulate(source, TARGET)
    changed_table = TournamentPairAdapter(source).predict_pair_table(TEAMS, CUTOFF, TARGET)
    assert after_table["GEN", "G2", 3] != changed_table["GEN", "G2", 3]
    assert after_source["predictor_receipt"]["configuration_sha256"] != after_config["predictor_receipt"]["configuration_sha256"]
    assert after_source["predictor_receipt"]["source"]["sha256"] == after_config["predictor_receipt"]["source"]["sha256"]


def test_fitted_glicko_probabilities_use_receipted_slopes_not_fixed_proxy_slopes(source, monkeypatch):
    adapter = TournamentPairAdapter(source)
    receipt = adapter.predictor_receipt(TEAMS, CUTOFF, "glicko_format")
    table = adapter.predict_pair_table(TEAMS, CUTOFF, "glicko_format")
    ratings = adapter.compute_team_ratings(CUTOFF)
    delta = adapter._rating_delta(ratings["GEN"], ratings["G2"])
    for bo in (1, 3, 5):
        slope = receipt["configuration"]["format_slopes"][str(bo)]
        assert table["GEN", "G2", bo] == pytest.approx(1 / (1 + math.exp(-slope * delta)))
    monkeypatch.setattr(pair_module, "FORMAT_SLOPES", {1: 9.0, 3: 9.0, 5: 9.0})
    assert adapter.predict_pair_table(TEAMS, CUTOFF, "glicko_format") == table
    assert adapter.predictor_receipt(TEAMS, CUTOFF, "glicko_format") == receipt


@pytest.mark.parametrize("change", ["new_member", "changed_label"])
def test_prior_cal_membership_and_fitted_parameters_change_receipts(source, change):
    before = TournamentPairAdapter(source)
    before_receipt = before.predictor_receipt(TEAMS, CUTOFF, "glicko_format")
    before_table = before.predict_pair_table(TEAMS, CUTOFF, "glicko_format")
    contents = source.read_text(encoding="utf-8")
    if change == "new_member":
        contents += "completed,2022-08-01T00:00:00Z,GEN,G2,2,3\n"
    else:
        contents = contents.replace(
            "completed,2022-07-02T00:00:00Z,T1,BLG,2,3",
            "completed,2022-07-02T00:00:00Z,T1,BLG,1,3",
        )
    source.write_text(contents, encoding="utf-8")
    after = TournamentPairAdapter(source)
    after_receipt = after.predictor_receipt(TEAMS, CUTOFF, "glicko_format")
    assert before_receipt["source"]["sha256"] != after_receipt["source"]["sha256"]
    assert before_receipt["configuration_sha256"] != after_receipt["configuration_sha256"]
    assert before_receipt["formula_sha256"] == after_receipt["formula_sha256"]
    before_bo = before_receipt["configuration"]["calibration"]["formats"]["3"]
    after_bo = after_receipt["configuration"]["calibration"]["formats"]["3"]
    assert before_bo["membership_sha256"] != after_bo["membership_sha256"]
    assert before_bo["slope"] != after_bo["slope"]
    if change == "new_member":
        assert after_bo["rows"] == before_bo["rows"] + 1
        assert after_bo["max_label_day"] == "2022-08-01T00:00:00Z"
    else:
        assert after_bo["rows"] == before_bo["rows"]
        assert after_bo["max_label_day"] == before_bo["max_label_day"]
    assert before_table != after.predict_pair_table(TEAMS, CUTOFF, "glicko_format")


def test_current_year_ratings_change_without_reopening_closed_cal(source):
    source.write_text(
        source.read_text(encoding="utf-8")
        + "completed,2023-04-01T00:00:00Z,GEN,G2,2,3\n",
        encoding="utf-8",
    )
    adapter = TournamentPairAdapter(source)
    early_cutoff = "2023-04-01T00:00:00Z"
    early_receipt = adapter.predictor_receipt(TEAMS, early_cutoff, "glicko_format")
    late_receipt = adapter.predictor_receipt(TEAMS, CUTOFF, "glicko_format")
    assert early_receipt["cutoff"] == early_cutoff
    assert late_receipt["cutoff"] == CUTOFF
    assert early_receipt["source"] == late_receipt["source"]
    assert early_receipt["configuration_sha256"] == late_receipt["configuration_sha256"]
    assert early_receipt["configuration"]["calibration"] == late_receipt["configuration"]["calibration"]
    assert adapter.predict_pair_table(TEAMS, early_cutoff, "glicko_format") != adapter.predict_pair_table(
        TEAMS, CUTOFF, "glicko_format",
    )


@pytest.mark.parametrize("model", MODELS[1:])
def test_future_boundary_and_excluded_series_do_not_change_causal_predictions(source, model):
    adapter = TournamentPairAdapter(source)
    table = adapter.predict_pair_table(TEAMS, CUTOFF, model)
    receipt = adapter.predictor_receipt(TEAMS, CUTOFF, model)
    source.write_text(
        source.read_text(encoding="utf-8")
        + f"completed,{CUTOFF},GEN,G2,2,3\n"
        + "completed,2024-07-01T00:00:00Z,GEN,G2,2,1\n"
        + "scheduled,2022-08-01T00:00:00Z,GEN,G2,2,3\n"
        + "completed,2022-08-01T00:00:00Z,GEN,G2,,5\n"
        + "completed,2022-08-01T00:00:00Z,GEN,GEN,1,1\n",
        encoding="utf-8",
    )
    reloaded = TournamentPairAdapter(source)
    changed_receipt = reloaded.predictor_receipt(TEAMS, CUTOFF, model)
    assert changed_receipt["source"]["sha256"] != receipt["source"]["sha256"]
    assert changed_receipt["configuration_sha256"] == receipt["configuration_sha256"]
    assert changed_receipt["formula_sha256"] == receipt["formula_sha256"]
    assert reloaded.predict_pair_table(TEAMS, CUTOFF, model) == table


@pytest.mark.parametrize("deficiency", ["no_cal", "missing_bo5", "zero_prior_delta"])
def test_insufficient_prior_cal_is_refused_before_forecast(source, monkeypatch, deficiency):
    rows = source.read_text(encoding="utf-8").splitlines(keepends=True)
    if deficiency == "no_cal":
        rows = [rows[0], *(row for row in rows[1:] if ",2022-" not in row)]
        missing = "[1, 3, 5]"
    elif deficiency == "missing_bo5":
        rows = [row for row in rows if not row.endswith(",5\n")]
        missing = "[5]"
    else:
        rows = [HEADER, *(
            f"completed,2022-07-01T00:00:00Z,GEN,G2,1,{bo}\n" for bo in (1, 3, 5)
        )]
        missing = "[1, 3, 5]"
    source.write_text("".join(rows), encoding="utf-8")
    adapter = TournamentPairAdapter(source)
    expected = f"unavailable_prior_calibrated_glicko_formats:2023:{missing}"
    with pytest.raises(ValueError) as table_refusal:
        adapter.predict_pair_table(TEAMS, CUTOFF, "glicko_format")
    assert str(table_refusal.value) == expected
    with pytest.raises(ValueError) as receipt_refusal:
        adapter.predictor_receipt(TEAMS, CUTOFF, "glicko_format")
    assert str(receipt_refusal.value) == expected

    def forbidden_simulator(*args, **kwargs):
        pytest.fail("deficient prior CAL reached forecast simulation")

    monkeypatch.setattr(benchmark, "TournamentBracketSimulator", forbidden_simulator)
    with pytest.raises(ValueError) as simulation_refusal:
        _simulate(source, "glicko_format")
    assert str(simulation_refusal.value) == expected

    manifest = {"editions": {"msi-2023": _manifest()["editions"]["local-proxy"]}}
    output = source.parent / "refused_forecast.json"
    monkeypatch.setattr(benchmark, "load_manifest", lambda: manifest)
    monkeypatch.setattr(pair_module, "DEFAULT_SERIES_RESULTS_PATH", source)
    monkeypatch.setattr(benchmark.sys, "argv", [
        "run_tournament_distribution_benchmark.py", "--simulate",
        "--edition", "msi-2023", "--model", "glicko_format", "--simulations", "1",
        "--output", str(output),
    ])
    with pytest.raises(ValueError) as cli_refusal:
        benchmark.main()
    assert str(cli_refusal.value) == expected
    assert not output.exists()


def _aggregates():
    # Four champion editions and two Swiss editions, with explicit support.
    # The independently known scores are 2*(1-p)^2 and (1-p)^2.
    predictions = {
        "uniform": 0.50, "glicko_format": 0.65,
        "regional_glicko_proxy": 0.70, TARGET: 0.80,
    }
    result = {}
    for model, p in predictions.items():
        champion_scores = [compute_champion_brier({"A": p, "B": 1 - p}, "A") for _ in range(4)]
        records = {record: 0.0 for record in ORDERED_SWISS_RECORDS}
        records.update({"3-0": p, "0-3": 1 - p})
        swiss_scores = [compute_swiss_record_rps(
            {"A": records}, {"A": "3-0"},
        ) for _ in range(2)]
        assert champion_scores[0] == pytest.approx(2 * (1 - p) ** 2)
        assert swiss_scores[0] == pytest.approx((1 - p) ** 2)
        flow = verify_network_flow_conservation(
            {"A": p, "B": 1 - p}, {"A": 1.0, "B": 1.0},
        )
        result[model] = {
            "mean_champion_brier": sum(champion_scores) / len(champion_scores),
            "mean_swiss_record_rps": sum(swiss_scores) / len(swiss_scores),
            "champion_brier_edition_count": len(champion_scores),
            "swiss_record_rps_edition_count": len(swiss_scores),
            "all_flow_feasible": flow["feasible"],
        }
    return result


def _decision(aggregates):
    # Exercise the public decision with measured scoring/flow inputs.
    decision = benchmark.evaluate_proxy_decision(aggregates)
    assert decision["decision"] in {"PROXY_DIAGNOSTIC_COMPLETED", "PROXY_DIAGNOSTIC_FAILED"}
    assert decision["native_adoption_eligible"] is False
    return decision


def test_successful_measured_proxy_remains_diagnostic_and_native_ineligible():
    decision = _decision(_aggregates())
    assert decision["decision"] == "PROXY_DIAGNOSTIC_COMPLETED"
    for comparator in ("uniform", "glicko_format"):
        for metric in ("champion_brier", "swiss_rps"):
            assert decision["criteria"][f"{TARGET}_beats_{comparator}_{metric}"] is True
    assert decision["criteria"]["network_flow_feasibility"] is True
    assert decision["criteria"]["comparisons_estimable"] is True


@pytest.mark.parametrize("comparator", ["uniform", "glicko_format"])
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("margin", [0.0, 0.01], ids=["tie", "comparator_better"])
def test_each_mandatory_comparison_can_independently_fail(comparator, metric, margin):
    aggregates = _aggregates()
    aggregates[comparator][metric] = aggregates[TARGET][metric] - margin
    decision = _decision(aggregates)
    assert decision["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
def test_real_flow_violation_in_any_model_fails_diagnostic(model):
    aggregates = _aggregates()
    failed_flow = verify_network_flow_conservation(
        {"A": 0.8, "B": 0.2}, {"A": 0.5, "B": 1.5},
    )
    assert failed_flow["feasible"] is False
    aggregates[model]["all_flow_feasible"] = failed_flow["feasible"]
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("flow", [None, 1, "true"], ids=["missing", "integer", "text"])
def test_unverified_flow_cannot_pass(model, flow):
    aggregates = _aggregates()
    if flow is None:
        del aggregates[model]["all_flow_feasible"]
    else:
        aggregates[model]["all_flow_feasible"] = flow
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")], ids=["missing", "nan", "posinf", "neginf"])
def test_unestimable_metric_cannot_pass(model, metric, value):
    aggregates = _aggregates()
    aggregates[model][metric] = value
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("metric", METRICS)
def test_absent_metric_cannot_pass(model, metric):
    aggregates = _aggregates()
    del aggregates[model][metric]
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
def test_absent_model_cannot_pass_vacuous_comparisons_or_flow(model):
    aggregates = _aggregates()
    del aggregates[model]
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("support", SUPPORT)
@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), 1.5, True],
                         ids=["missing", "zero", "negative", "nan", "infinite", "fractional", "boolean"])
def test_missing_or_invalid_edition_support_cannot_pass(model, support, value):
    aggregates = _aggregates()
    if value is None:
        del aggregates[model][support]
    else:
        aggregates[model][support] = value
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


@pytest.mark.parametrize("comparator", ["uniform", "glicko_format"])
@pytest.mark.parametrize("support", SUPPORT)
def test_unmatched_edition_support_cannot_promote_reduced_intersection(comparator, support):
    aggregates = _aggregates()
    aggregates[comparator][support] -= 1
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


def test_unverified_extra_model_flow_cannot_be_ignored():
    aggregates = _aggregates()
    aggregates["additional_diagnostic"] = {"all_flow_feasible": False}
    assert _decision(aggregates)["decision"] == "PROXY_DIAGNOSTIC_FAILED"


def test_mathematical_flow_alone_cannot_complete_missing_predictive_evidence():
    aggregates = {model: {"all_flow_feasible": True} for model in MODELS}
    decision = _decision(aggregates)
    assert decision["criteria"]["network_flow_feasibility"] is True
    assert decision["criteria"]["comparisons_estimable"] is False
    assert decision["decision"] == "PROXY_DIAGNOSTIC_FAILED"


def test_new_benchmark_persists_proxy_receipts_and_leaves_archive_unchanged(tmp_path, monkeypatch, source):
    archive_dirs = [ROOT / "data" / stage / "tournament_distribution" / "tournament_dist_20261005_01"
                    for stage in ("07_model_output", "08_reporting")]
    archive_before = {path: _sha(path) for directory in archive_dirs
                      for path in directory.rglob("*") if path.is_file()}
    canonical_manifest = ROOT / "conf/base/tournament_manifest.json"
    manifest_before = canonical_manifest.read_bytes()
    manifest = deepcopy(benchmark.load_manifest())
    manifest["editions"] = {"worlds-2023": manifest["editions"]["worlds-2023"]}
    manifest["sources"] = [{"path": "series_results.csv", "sha256": _sha(source)}]
    local_manifest = tmp_path / "manifest.json"
    local_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    local_manifest_before = local_manifest.read_bytes()
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "DEFAULT_MANIFEST_PATH", local_manifest)
    monkeypatch.setattr(benchmark, "DEFAULT_EDITIONS", ("worlds-2023",))
    monkeypatch.setattr(benchmark, "DEFAULT_MODELS", MODELS)
    monkeypatch.setattr(pair_module, "DEFAULT_SERIES_RESULTS_PATH", source)
    # Redirect only filesystem boundaries; retain real source validation,
    # predictors, bracket simulation, scoring and decision.
    real_freeze = benchmark.freeze_manifest
    monkeypatch.setattr(benchmark, "freeze_manifest", lambda: real_freeze(local_manifest))

    decision = benchmark.run_benchmark("owned-proxy-contract", simulations=24, seed=117,
                                       output_base=tmp_path / "reporting")
    report = tmp_path / "reporting/owned-proxy-contract"
    receipt = json.loads((report / "receipt.json").read_text(encoding="utf-8"))
    assert decision["decision"] in {"PROXY_DIAGNOSTIC_COMPLETED", "PROXY_DIAGNOSTIC_FAILED"}
    assert decision["native_adoption_eligible"] is False
    stored_decision = json.loads((report / "decision.json").read_text(encoding="utf-8"))
    assert stored_decision == decision
    protocol = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    assert receipt["decision_contract"] == protocol
    assert protocol["native_adoption_eligible"] is False
    metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    aggregates = metrics["model_aggregates"]
    # Compute the diagnostic oracle from measured scores, not stored criteria.
    comparisons_pass = all(
        aggregates[TARGET][metric] < aggregates[comparator][metric]
        for comparator in ("uniform", "glicko_format") for metric in METRICS
    )
    flow_pass = all(row["network_flow_feasible"] is True
                    for row in metrics["detailed_scores"])
    assert decision["decision"] == (
        "PROXY_DIAGNOSTIC_COMPLETED" if comparisons_pass and flow_pass
        else "PROXY_DIAGNOSTIC_FAILED"
    )
    for model in MODELS:
        assert aggregates[model]["champion_brier_edition_count"] == 1
        assert aggregates[model]["swiss_record_rps_edition_count"] == 1
    edition = manifest["editions"]["worlds-2023"]
    for model in MODELS:
        predictor_receipt = receipt["predictor_receipts"]["worlds-2023"][model]
        _assert_receipt(predictor_receipt, source, model, edition["entrants"], edition["cutoff"])
        output = tmp_path / "data/07_model_output/tournament_distribution/owned-proxy-contract" / f"worlds-2023_{model}_distributions.json"
        distribution = json.loads(output.read_text(encoding="utf-8"))
        assert distribution["model_kind"] == model
        assert distribution["predictor_receipt"] == predictor_receipt
    assert canonical_manifest.read_bytes() == manifest_before
    assert local_manifest.read_bytes() == local_manifest_before
    outputs_before = {path: _sha(path) for directory in (
        report, tmp_path / "data/07_model_output/tournament_distribution/owned-proxy-contract",
    ) for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark("owned-proxy-contract", simulations=1, seed=999,
                                output_base=tmp_path / "reporting")
    assert {path: _sha(path) for path in outputs_before} == outputs_before
    assert {path: _sha(path) for directory in archive_dirs for path in directory.rglob("*")
            if path.is_file()} == archive_before
