"""Retained-source admission boundaries for numerical-only tournament runs."""
import copy

import pytest

from src.models import tournament_numerical_benchmark as benchmark


def retained_pair():
    event = {"teams": ["A", "B"], "spec": {"stages": [
        {"id": "final", "kind": "single_elimination", "entrants": ["A", "B"], "best_of": 3}]}}
    compiled = {"probability_unit": "series_win", "scope": "frozen_origin_unconditional_hypothetical_pair",
        "rows": [{"team_a": "A", "team_b": "B", "best_of": 3, "p": .6,
                  "scoreline_pmf": {"2-0": .6, "2-1": 0., "0-2": .4, "1-2": 0.},
                  "series_length_pmf": {"2": 1., "3": 0.}}]}
    payload = {"model": "uniform", "team_strength_draws": None,
               "pair_table": {"probability_unit": "series_win", "rows": [{"A": "A", "B": "B", "BO": 3, "p": .6}]},
               "forecast": {"conditional_score_distributions": None,
                            "series_score_distributions": {"3": [1., 0.]}, "pair_scoreline_laws": compiled}}
    return event, payload


def test_complete_pair_law_is_admitted_but_missing_and_duplicate_formats_are_rejected():
    event, payload = retained_pair()
    assert benchmark._validate_pairs(event, payload, {}) == {("A", "B", 3): .6}
    missing = copy.deepcopy(payload)
    missing["pair_table"]["rows"] = []
    with pytest.raises(ValueError, match="incomplete"):
        benchmark._validate_pairs(event, missing, {})
    duplicate = copy.deepcopy(payload)
    duplicate["pair_table"]["rows"].append({"A": "B", "B": "A", "BO": 3, "p": .4})
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark._validate_pairs(event, duplicate, {})


def test_supplied_score_law_must_equal_retained_pmf_not_merely_winner_marginal():
    event, payload = retained_pair()
    payload["forecast"]["series_score_distributions"]["3"] = [0., 1.]
    with pytest.raises(ValueError, match="Saved score law differs"):
        benchmark._validate_pairs(event, payload, {})


def test_compiled_score_law_cannot_change_series_win_probability():
    event, payload = retained_pair()
    row = payload["forecast"]["pair_scoreline_laws"]["rows"][0]
    row["scoreline_pmf"]["2-0"] = .7
    row["scoreline_pmf"]["0-2"] = .3
    with pytest.raises(ValueError, match="preserve"):
        benchmark._validate_pairs(event, payload, {})


def test_existing_numerical_stage_is_rejected_without_modifying_its_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    stage = tmp_path / "data/07_model_output/tournament_numerical/already-owned"
    stage.mkdir(parents=True)
    owned = stage / "user-input.json"
    owned.write_bytes(b'"original"\n')
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark.run_numerical_benchmark("already-owned")
    assert owned.read_bytes() == b'"original"\n'
    assert not (tmp_path / "data/05_model_input").exists()


def test_bound_source_change_is_rejected(tmp_path):
    source = tmp_path / "pairs.json"
    source.write_text('{"p":0.5}')
    expected = benchmark._sha(source)
    source.write_text('{"p":0.8}')
    with pytest.raises(ValueError, match="Sealed source bytes changed"):
        benchmark._bind(source, expected, {})


@pytest.mark.parametrize("expected", [None, "", "not-a-digest", "g" * 64, "A" * 64, 7])
@pytest.mark.parametrize("source_exists", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_invalid_source_digest_is_rejected_before_missing_file_or_cache_shortcut(
        tmp_path, expected, source_exists, cached):
    source = tmp_path / "source.json"
    if source_exists:
        source.write_bytes(b'{"p":0.5}')
    bindings = {str(source.resolve()): expected} if cached else {}
    with pytest.raises(ValueError, match="Expected source digest must be canonical SHA-256"):
        benchmark._bind(source, expected, bindings)


def test_joint_method_difference_keeps_categories_seen_by_only_one_estimator():
    comparison = benchmark._compare({"joint_final_prob": {'["A","B"]': 1.}},
                                    {"joint_final_prob": {'["A","B"]': .75, '["A","C"]': .25}}, "exact")
    target = comparison["targets"]["joint_final_prob"]
    assert target["l1_difference"] == pytest.approx(.5)
    assert target["max_absolute_difference"] == pytest.approx(.25)
    assert target["signed_differences"]['/["A","C"]'] == pytest.approx(-.25)
