"""Offline daily tournament-evaluation preparation contracts."""
from __future__ import annotations

import json

import pytest

from scripts.prepare_daily_tournament_evaluation import compile_daily_tournament_evaluation


def _input():
    return {
        "version": 1,
        "event": {
            "tournament_id": "fixture-cup-2026",
            "phase_id": "fixture-cup-main",
            "edition_start_at": "2026-01-02T12:00:00Z",
            "family": "fixture",
            "tier": "fixture",
            "format": "single_elimination",
            "forecast_scope": "full_tournament",
        },
        "pre_draw_cutoff": "2026-01-01T00:00:00Z",
        "post_draw_cutoff": "2026-01-02T00:00:00Z",
        "spec": {
            "version": 1,
            "id": "fixture-cup",
            "teams": ["A", "B", "C", "D"],
            "stages": [{
                "id": "main", "kind": "single_elimination", "entrants": ["A", "B", "C", "D"],
                "best_of": 1, "draw": {"policy": "uniform"},
            }],
            "champion": {"stage": "main", "group": "all", "rank": 1},
        },
        "seeds": ["A", "B", "C", "D"],
        "draws": [{
            "id": "main-round-1", "source_date": "2026-01-01", "stage": "main",
            "group": "all", "round": 1, "pairs": [["A", "B"], ["C", "D"]],
        }],
        "completed": [
            {"id": "semi-a", "source_date": "2026-01-03", "stage": "main", "group": "all",
             "round": 1, "team_a": "A", "team_b": "B", "best_of": 1, "weight": 1,
             "winner": "A", "loser": "B", "score_a": 1, "score_b": 0},
            {"id": "semi-b", "source_date": "2026-01-03", "stage": "main", "group": "all",
             "round": 1, "team_a": "C", "team_b": "D", "best_of": 1, "weight": 1,
             "winner": "C", "loser": "D", "score_a": 1, "score_b": 0},
            {"id": "final", "source_date": "2026-01-04", "stage": "main", "group": "all",
             "round": 2, "team_a": "A", "team_b": "C", "best_of": 1, "weight": 1,
             "winner": "A", "loser": "C", "score_a": 1, "score_b": 0},
        ],
        "targets": [{
            "target_id": "champion", "target_kind": "champion", "source": "champion_prob",
            "target_start_at": "2026-01-05T12:00:00Z",
        }],
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compiler_groups_completed_results_by_prior_utc_day_and_excludes_same_day(tmp_path):
    source = _write(tmp_path / "phase.json", _input())
    output = tmp_path / "artifact"

    result = compile_daily_tournament_evaluation(source, output)

    assert result["output_dir"] == str(output)
    before_semifinals = json.loads((output / "states" / "2026-01-03.json").read_text())
    after_semifinals = json.loads((output / "states" / "2026-01-04.json").read_text())
    assert before_semifinals["completed"] == []
    assert [(row["team_a"], row["team_b"]) for row in after_semifinals["completed"]] == [("A", "B"), ("C", "D")]
    assert after_semifinals["temporal_policy"] == "daily_rollforward"
    assert all(row["source_date"] < "2026-01-04" for row in after_semifinals["completed"])
    assert all("id" not in row for row in after_semifinals["completed"])
    ledger = json.loads((output / "readiness_ledger.json").read_text())
    daily_entry = next(entry for entry in ledger["entries"] if entry["origin_id"] == "daily-2026-01-04")
    assert daily_entry["completed_included"] == ["semi-a", "semi-b"]
    assert [(entry["block"], entry["status"]) for entry in ledger["post_round_readiness"]] == [
        ({"stage": "main", "group": "all", "round": 1}, "eligible")
    ]


def test_compiler_distinguishes_unknown_pre_draw_from_published_post_draw(tmp_path):
    source = _write(tmp_path / "phase.json", _input())
    output = tmp_path / "artifact"

    compile_daily_tournament_evaluation(source, output)

    schedule = json.loads((output / "origins.json").read_text())["origins"]
    pre_draw = next(origin for origin in schedule if origin["forecast_mode"] == "pre_draw")
    post_draw = next(origin for origin in schedule if origin["forecast_mode"] == "post_draw")
    assert "state" not in pre_draw
    assert pre_draw["draw_knowledge"]["status"] == "unknown"
    assert pre_draw["draw_knowledge"]["simulation_policy"]["declared_draws"] == [
        {"stage": "main", "draw": {"policy": "uniform"}}
    ]
    post_state = json.loads((output / post_draw["state"]).read_text())
    assert post_state["pairings"][0]["pairs"] == [["A", "B"], ["C", "D"]]
    daily = next(origin for origin in schedule if origin["forecast_mode"] == "daily_rollforward")
    post_round = next(origin for origin in schedule if origin["forecast_mode"] == "post_round")
    assert pre_draw["forecast_scope"] == post_draw["forecast_scope"] == "full_tournament"
    assert daily["forecast_scope"] == post_round["forecast_scope"] == "remaining_tournament"


def test_compiler_excludes_pre_draw_for_deterministic_seeded_graph_without_draw_source(tmp_path):
    payload = _input()
    payload["draws"] = []
    payload["spec"]["stages"][0] = {
        "id": "main", "kind": "graph", "best_of": 1,
        "entrants": ["A", "B", "C", "D"],
        "matches": [
            {"id": "r1", "a": "A", "b": "B"},
            {"id": "upper", "a": "C", "b": "D"},
            {"id": "final", "a": {"winner": "r1"}, "b": {"winner": "upper"}},
        ],
        "ranking": [{"winner": "final"}, {"loser": "final"},
                    {"loser": "r1"}, {"loser": "upper"}],
    }
    for row, node in zip(payload["completed"], ("r1", "upper", "final"), strict=True):
        row["round"] = node
    source = _write(tmp_path / "phase.json", payload)
    output = tmp_path / "artifact"

    compile_daily_tournament_evaluation(source, output)

    schedule = json.loads((output / "origins.json").read_text())["origins"]
    assert "pre_draw" not in {origin["forecast_mode"] for origin in schedule}
    post_draw = next(origin for origin in schedule if origin["forecast_mode"] == "post_draw")
    assert post_draw["draw_knowledge"]["status"] == "deterministic_spec"
    assert "state" not in post_draw
    post_rounds = [origin["checkpoint"]["round"] for origin in schedule if origin["forecast_mode"] == "post_round"]
    assert post_rounds == ["r1", "upper", "final"]
    ledger = json.loads((output / "readiness_ledger.json").read_text())
    assert ledger["pre_draw_readiness"]["status"] == "not_applicable"


def test_compiler_rejects_duplicate_or_unsorted_completed_source_ids(tmp_path):
    payload = _input()
    payload["completed"].append({**payload["completed"][-1], "id": "semi-a"})
    source = _write(tmp_path / "phase.json", payload)

    with pytest.raises(ValueError, match="duplicate|sorted"):
        compile_daily_tournament_evaluation(source, tmp_path / "artifact")


def test_compiler_refuses_to_reuse_immutable_output_directory(tmp_path):
    source = _write(tmp_path / "phase.json", _input())
    output = tmp_path / "artifact"
    compile_daily_tournament_evaluation(source, output)

    with pytest.raises(ValueError, match="exists|new"):
        compile_daily_tournament_evaluation(source, output)


def test_compiler_rejects_target_that_started_at_an_origin_cutoff(tmp_path):
    payload = _input()
    payload["targets"][0]["target_start_at"] = "2026-01-02T00:00:00Z"
    source = _write(tmp_path / "phase.json", payload)

    with pytest.raises(ValueError, match="target.*start|cutoff"):
        compile_daily_tournament_evaluation(source, tmp_path / "artifact")


def _graph_input():
    payload = _input()
    payload["draws"] = []
    payload["spec"]["stages"][0] = {
        "id": "main", "kind": "graph", "best_of": 1,
        "entrants": ["A", "B", "C", "D"],
        "matches": [
            {"id": "r1", "a": "A", "b": "B"},
            {"id": "upper", "a": "C", "b": "D"},
            {"id": "r2", "a": {"winner": "r1"}, "b": {"loser": "upper"}},
            {"id": "final", "a": {"winner": "r2"}, "b": {"winner": "upper"}},
        ],
        "ranking": [{"winner": "final"}, {"loser": "final"},
                    {"loser": "r2"}, {"loser": "r1"}],
    }
    payload["completed"][0]["round"] = "r1"
    payload["completed"][1]["round"] = "upper"
    payload["completed"][2].update(round="r2", team_b="D", loser="D")
    payload["completed"].append({
        **payload["completed"][2], "id": "last", "source_date": "2026-01-05",
        "round": "final", "team_b": "C", "loser": "C",
    })
    payload["targets"][0]["target_start_at"] = "2026-01-05T12:00:00Z"
    return payload


@pytest.mark.parametrize("current_axis", ["axis2", "axis3"])
def test_dynamic_winner_and_loser_checkpoints_reach_runner_with_explicit_final_exclusion(tmp_path, current_axis):
    from scripts.tournament_evaluation import run_evaluation

    payload = _graph_input()
    source = _write(tmp_path / "source.json", payload)
    table = {
        "probability_unit": "series_win",
        "rows": [{"team_a": a, "team_b": b, "best_of": 1, "p": 0.75}
                 for i, a in enumerate(payload["seeds"]) for b in payload["seeds"][i + 1:]],
        "provenance": {
            "model_id": "archived", "model_artifact_sha256": "fixture-artifact",
            "training_cutoff": "2025-12-01T00:00:00Z",
            "calibration_cutoff": "2025-12-01T00:00:00Z",
            "feature_history_max_at": "2025-12-31T00:00:00Z",
        },
    }
    _write(tmp_path / "table.json", table)
    config = {
        "version": "tournament-evaluation-v3", "qualification": "retrospective_reconstruction",
        "models": [{"id": "archived", "kind": "series_table"}],
        "events": [{**payload["event"], "origins": [
            {"origin_id": "start", "axis": "axis1", "forecast_mode": "post_draw",
             "cutoff": "2026-01-02T00:00:00Z", "tables": {"archived": "table.json"}},
            {"origin_id": "current", "axis": current_axis, "forecast_mode": "daily_rollforward",
             "cutoff": "2026-01-02T00:00:00Z",
             "model_exclusions": {"archived": "current features not archived"}},
        ] + [
            {"origin_id": f"daily-{day}", "axis": "axis2", "forecast_mode": "daily_rollforward",
             "cutoff": f"2026-01-{day:02}T00:00:00Z", "tables": {"archived": "table.json"}}
            for day in range(3, 6)
        ]}],
        "outcomes": "must-not-be-read.json",
    }
    config_path = _write(tmp_path / "models.json", config)
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(source, output, model_config=config_path)
    status = run_evaluation(output / "evaluator_manifest.json", tmp_path / "run", simulations=20)
    assert status["status"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "run" / "forecasts.jsonl").read_text().splitlines()]
    daily = {row["model"]: row for row in rows if row["origin_id"] == "daily-2026-01-02"}
    current = daily["archived:refreshed" if current_axis == "axis3" else "archived"]
    assert current["forecast_status"] == "failed"
    assert current["probabilities"] == {}
    if current_axis == "axis3":
        assert set(daily) == {"archived:frozen", "archived:refreshed", "fair_series", "flat"}
        assert daily["archived:frozen"]["probabilities"]["A"] > 0
    else:
        assert set(daily) == {"archived", "fair_series", "flat"}
    dynamic = [row for row in rows if row["origin_id"].startswith("post-round-main-all-r2-")]
    assert {row["model"] for row in dynamic} >= {"archived:frozen", "archived:refreshed"}
    assert all(row["probabilities"]["B"] == row["probabilities"]["D"] == 0 for row in dynamic)
    readiness = json.loads((output / "readiness_ledger.json").read_text())["post_round_readiness"]
    assert {row["block"]["round"] for row in readiness if row["status"] == "eligible"} == {"r1", "upper", "r2"}
    final = next(row for row in readiness if row["block"]["round"] == "final")
    assert final["status"] == "excluded"
    assert final["unresolved_target_ids"] == []
    assert final["cutoff"] == "2026-01-06T00:00:00Z"


@pytest.mark.parametrize("corruption", ["wrong_winner_path", "missing_ancestor", "later_ancestor", "forward_reference"])
def test_compiler_rejects_invalid_completed_ancestor_paths(tmp_path, corruption):
    payload = _graph_input()
    if corruption == "wrong_winner_path":
        payload["completed"][2].update(team_a="B", winner="B")
    elif corruption == "missing_ancestor":
        del payload["completed"][0]
    elif corruption == "later_ancestor":
        payload["completed"][0]["source_date"] = "2026-01-05"
        payload["completed"].sort(key=lambda row: (row["source_date"], row["id"]))
    else:
        payload["spec"]["stages"][0]["matches"][2]["a"] = {"winner": "final"}
    source = _write(tmp_path / "source.json", payload)
    with pytest.raises(ValueError, match="ancestor|Graph|graph"):
        compile_daily_tournament_evaluation(source, tmp_path / "prepared")


def _completed(node, a, b, day, *, stage="main"):
    return {"id": f"{stage}-{node}", "source_date": day, "stage": stage, "group": "all",
            "round": node, "team_a": a, "team_b": b, "best_of": 1, "weight": 1,
            "winner": a, "loser": b, "score_a": 1, "score_b": 0}


@pytest.mark.parametrize("seed_order,lower", [("tournament", "C"), ("stage", "A")])
def test_compiler_resolves_seeded_pool_from_all_completed_ancestors(tmp_path, seed_order, lower):
    from betting_app.tests.test_tournament_completed_extensions import _seeded_graph

    payload = _input()
    payload["spec"] = _seeded_graph()
    payload["spec"]["stages"][0]["seed_order"] = seed_order
    payload["seeds"] = payload["spec"]["teams"]
    payload["draws"] = []
    payload["completed"] = [
        _completed("u1", "D", "A", "2026-01-03", stage="bracket"),
        _completed("u2", "B", "C", "2026-01-03", stage="bracket"),
        _completed("l1", lower, "E", "2026-01-04", stage="bracket"),
    ]
    payload["completed"].sort(key=lambda row: (row["source_date"], row["id"]))
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), output)
    origins = json.loads((output / "origins.json").read_text())["origins"]
    assert any(origin.get("checkpoint", {}).get("round") == "l1" for origin in origins)


@pytest.mark.parametrize("corruption", ["missing", "later"])
def test_compiler_rejects_unproven_seeded_pool_ancestor(tmp_path, corruption):
    from betting_app.tests.test_tournament_completed_extensions import _seeded_graph

    payload = _input()
    payload.update(spec=_seeded_graph(), seeds=list("ABCDEF"), draws=[])
    payload["completed"] = [
        _completed("u1", "D", "A", "2026-01-03", stage="bracket"),
        _completed("u2", "B", "C", "2026-01-03", stage="bracket"),
        _completed("l1", "C", "E", "2026-01-04", stage="bracket"),
    ]
    if corruption == "missing":
        del payload["completed"][0]
    else:
        payload["completed"][0]["source_date"] = "2026-01-05"
    payload["completed"].sort(key=lambda row: (row["source_date"], row["id"]))
    with pytest.raises(ValueError, match="ancestor"):
        compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), tmp_path / "prepared")


def test_compiler_propagates_explicit_bye_without_inventing_completed_match(tmp_path):
    payload = _input()
    payload["seeds"] = payload["spec"]["teams"] = list("ABC")
    payload["draws"] = []
    payload["spec"]["stages"] = [{
        "id": "main", "kind": "graph", "entrants": list("ABC"), "best_of": 1,
        "matches": [{"id": "bye", "a": "A", "b": None},
                    {"id": "semi", "a": "B", "b": "C"},
                    {"id": "final", "a": {"winner": "bye"}, "b": {"winner": "semi"}}],
        "ranking": [{"winner": "final"}, {"loser": "final"}, {"loser": "semi"}],
    }]
    payload["completed"] = [_completed("semi", "B", "C", "2026-01-03"),
                            _completed("final", "A", "B", "2026-01-04")]
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), output)
    origins = json.loads((output / "origins.json").read_text())["origins"]
    assert [origin["checkpoint"]["round"] for origin in origins if "checkpoint" in origin] == ["semi", "final"]


def _linked_input():
    payload = _input()
    qualifiers = [{"stage": "main", "group": "all", "rank": rank} for rank in (1, 2)]
    payload["spec"]["stages"].append({
        "id": "title", "kind": "single_elimination", "entrants": qualifiers,
        "best_of": 1, "draw": {"policy": "uniform"},
    })
    payload["spec"]["champion"]["stage"] = "title"
    payload["draws"].append({"id": "title-round-1", "source_date": "2026-01-05",
                             "stage": "title", "group": "all", "round": 1, "pairs": [["A", "C"]]})
    payload["targets"][0]["target_start_at"] = "2026-01-08T12:00:00Z"
    return payload


def test_compiler_keeps_downstream_draw_future_until_its_source_day_passes(tmp_path):
    payload = _linked_input()
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), output)
    initial = json.loads((output / "states/2026-01-02.json").read_text())
    later = json.loads((output / "states/2026-01-06.json").read_text())
    assert [row["stage"] for row in initial["pairings"]] == ["main"]
    assert [row["stage"] for row in later["pairings"]] == ["main", "title"]


def test_compiler_validates_phase_draw_against_resolved_subset_entrants(tmp_path):
    payload = _linked_input()
    payload["event"].update(forecast_scope="phase_start", phase_stages=["title"],
                            phase_start_at="2026-01-06T12:00:00Z")
    payload.update(pre_draw_cutoff="2026-01-05T00:00:00Z", post_draw_cutoff="2026-01-06T00:00:00Z")
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), output)
    origins = json.loads((output / "origins.json").read_text())["origins"]
    post = next(row for row in origins if row["origin_id"] == "post-draw")
    state = json.loads((output / post["state"]).read_text())
    assert post["phase_stages"] == ["title"]
    assert next(row for row in state["pairings"] if row["stage"] == "title")["pairs"] == [["A", "C"]]


@pytest.mark.parametrize("corruption", ["missing_ancestor", "same_day_draw", "wrong_entrant"])
def test_compiler_rejects_phase_draw_without_available_entrant_evidence(tmp_path, corruption):
    payload = _linked_input()
    payload["event"].update(forecast_scope="phase_start", phase_stages=["title"],
                            phase_start_at="2026-01-06T12:00:00Z")
    payload.update(pre_draw_cutoff="2026-01-05T00:00:00Z", post_draw_cutoff="2026-01-06T00:00:00Z")
    if corruption == "missing_ancestor":
        payload["completed"].pop()
    elif corruption == "same_day_draw":
        payload["draws"][-1]["source_date"] = "2026-01-06"
    else:
        payload["draws"][-1]["pairs"] = [["A", "B"]]
    with pytest.raises(ValueError):
        compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), tmp_path / "prepared")


def test_compiler_resolves_graph_stage_rank_supplies_from_completed_upstream(tmp_path):
    payload = _linked_input()
    refs = payload["spec"]["stages"][-1]["entrants"]
    payload["spec"]["stages"][-1] = {
        "id": "title", "kind": "graph", "entrants": refs, "best_of": 1,
        "matches": [{"id": "final", "a": refs[0], "b": refs[1]}],
        "ranking": [{"winner": "final"}, {"loser": "final"}],
    }
    payload["draws"].pop()
    payload["completed"].append(_completed("final", "A", "C", "2026-01-06", stage="title"))
    output = tmp_path / "prepared"
    compile_daily_tournament_evaluation(_write(tmp_path / "source.json", payload), output)
    origins = json.loads((output / "origins.json").read_text())["origins"]
    assert any(row.get("checkpoint", {}).get("stage") == "title" for row in origins)


def _excluded_model_config(payload):
    return {
        "version": "tournament-evaluation-v3", "qualification": "retrospective_reconstruction",
        "models": [{"id": "archived", "kind": "series_table"}],
        "events": [{**payload["event"], "origins": [
            {"origin_id": "start", "axis": "axis1", "forecast_mode": "post_draw",
             "cutoff": payload["post_draw_cutoff"], "model_exclusions": {"archived": "not archived"}},
            *[{"origin_id": f"daily-{day}", "axis": "axis2", "forecast_mode": "daily_rollforward",
               "cutoff": f"2026-01-{day:02}T00:00:00Z", "model_exclusions": {"archived": "not archived"}}
              for day in range(2, 6)],
        ]}],
    }


@pytest.mark.parametrize("tamper", [None, "changed", "missing", "conflicting_alias"])
def test_compiler_preserves_and_checks_configuration_evidence_pins(tmp_path, tamper):
    import hashlib

    payload = _graph_input()
    source = _write(tmp_path / "source.json", payload)
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    evidence = _write(tmp_path / "rules.json", {"rule": "declared"})
    expected = hashlib.sha256(evidence.read_bytes()).hexdigest()
    config = _excluded_model_config(payload)
    config["evidence_hashes"] = {"../rules.json": expected}
    if tamper == "changed":
        _write(evidence, {"rule": "changed"})
    elif tamper == "missing":
        evidence.unlink()
    elif tamper == "conflicting_alias":
        config["evidence_hashes"][str(evidence)] = "0" * 64
    config_path = _write(config_dir / "models.json", config)
    output = tmp_path / "prepared"
    if tamper:
        with pytest.raises(ValueError, match="evidence|pin"):
            compile_daily_tournament_evaluation(source, output, model_config=config_path)
        assert not output.exists()
    else:
        compile_daily_tournament_evaluation(source, output, model_config=config_path)
        manifest = json.loads((output / "evaluator_manifest.json").read_text())
        assert manifest["evidence_hashes"]["../rules.json"] == expected


@pytest.mark.parametrize("corruption", [None, "missing", "later"])
def test_compiler_resolves_group_rank_graph_supplies_only_from_available_complete_stage(tmp_path, corruption):
    payload = _input()
    refs = [{"stage": "groups", "group_rank": rank, "rank": 1} for rank in (1, 2)]
    payload["spec"]["stages"] = [
        {"id": "groups", "kind": "round_robin", "groups": {"one": ["A", "B"], "two": ["C", "D"]},
         "best_of": 1, "tiebreakers": ["wins", "seed"], "group_tiebreakers": ["wins", "seed"]},
        {"id": "main", "kind": "graph", "entrants": refs, "best_of": 1,
         "matches": [{"id": "final", "a": refs[0], "b": refs[1]}],
         "ranking": [{"winner": "final"}, {"loser": "final"}]},
    ]
    payload["draws"] = []
    payload["completed"] = [
        {**_completed(1, "A", "B", "2026-01-03", stage="groups"), "id": "group-one", "group": "one"},
        {**_completed(1, "C", "D", "2026-01-03", stage="groups"), "id": "group-two", "group": "two"},
        _completed("final", "A", "C", "2026-01-04"),
    ]
    if corruption == "missing":
        del payload["completed"][1]
    elif corruption == "later":
        payload["completed"][1]["source_date"] = "2026-01-05"
        payload["completed"].sort(key=lambda row: (row["source_date"], row["id"]))
    source = _write(tmp_path / "source.json", payload)
    output = tmp_path / "prepared"
    if corruption:
        with pytest.raises(ValueError, match="ancestor"):
            compile_daily_tournament_evaluation(source, output)
    else:
        compile_daily_tournament_evaluation(source, output)
        origins = json.loads((output / "origins.json").read_text())["origins"]
        assert any(row.get("checkpoint", {}).get("round") == "final" for row in origins)
