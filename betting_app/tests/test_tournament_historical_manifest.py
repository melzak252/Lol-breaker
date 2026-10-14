"""Historical admission must not turn outcomes or fuzzy names into inputs."""
from copy import deepcopy

import pytest

from scripts.prepare_tournament_historical_manifest import (
    explicit_aliases, identity_at, prior_seed_rows, replay_elimination, source_boundary,
)


def test_target_standings_cannot_supply_pre_phase_seeds():
    target = {"phase_id": "finals", "title": "Cup Finals", "start_date": "2024-03-01"}
    prior = {"phase_id": "groups", "title": "Cup Groups", "end_date": "2024-02-28", "status": "completed"}
    standings = [{"event_title": "Cup Finals", "placement": str(i), "team": team,
                  "table_index": 1} for i, team in enumerate(["A", "B"], 1)]
    with pytest.raises(ValueError):
        prior_seed_rows(target, [prior], standings, 2)
    standings.extend({"event_title": "Cup Groups", "placement": str(i), "team": team,
                      "table_index": 1} for i, team in enumerate(["B", "A"], 1))
    rows, source = prior_seed_rows(target, [prior], standings, 2)
    assert [row["team"] for row in rows] == ["B", "A"]
    assert source["phase_id"] == "groups"


def test_exact_identity_uses_only_earlier_provider_records():
    records = [{"date": "2024-02-28", "team1": "Alpha", "team1_id": "old",
                "team2": "Beta", "team2_id": "b", "golgg_match_id": "1"},
               {"date": "2024-03-01", "team1": "Alpha", "team1_id": "future",
                "team2": "Beta", "team2_id": "b", "golgg_match_id": "2"}]
    team, evidence = identity_at("Alpha", "2024-03-01", records, {})
    assert team == "old"
    assert evidence["golgg_match_id"] == "1"
    with pytest.raises(ValueError):
        identity_at("Alfa", "2024-03-01", records, {})
    records.append({**records[0], "team1_id": "conflict"})
    with pytest.raises(ValueError):
        identity_at("Alpha", "2024-03-01", records, {})


def test_identity_skips_missing_ids_and_uses_explicit_template_display_aliases():
    aliases = explicit_aliases({"EST": {"resolved": True, "canonical_title": "EStar (Chinese Team)",
                                      "display_name": "eStar Gaming", "raw_path": "archived-template.json"}})
    records = [{"date": "2024-02-27", "team1": "EST", "team1_id": "es",
                "team2": "Beta", "team2_id": "b", "golgg_match_id": "1"},
               {"date": "2024-02-28", "team1": "EST", "team1_id": None,
                "team2": "Beta", "team2_id": "b", "golgg_match_id": "2"}]
    assert identity_at("eStar Gaming", "2024-03-01", records, aliases)[0] == "es"
    with pytest.raises(ValueError):
        identity_at("eStar Gaming", "2024-03-01", records[1:], aliases)


def test_unknown_edition_boundary_is_not_minimum_phase_date():
    event = {"start_date": "2024-03-01", "end_date": "2024-03-03"}
    boundary = source_boundary(event, {"edition_start_at": None})
    assert boundary["edition_start_at"] is None
    assert boundary["edition_start_status"] == "unverified_whole_edition_boundary"
    assert boundary["phase_start_at"] == "2024-03-01T00:00:00Z"
    assert boundary["phase_start_forecast_cutoff"] < boundary["phase_start_at"]


def test_replay_routes_only_observed_ancestors_and_rejects_illegal_final():
    spec = {"teams": ["a", "b", "c"], "stages": [{"id": "ladder", "kind": "graph", "best_of": 3,
            "matches": [{"id": "semi", "a": "a", "b": "b"},
                        {"id": "final", "a": "c", "b": {"winner": "semi"}}],
            "ranking": [{"winner": "final"}, {"loser": "final"}, {"loser": "semi"}]}]}
    records = [{"wiki_series_id": "s1", "team_a": "a", "team_b": "b", "best_of": 3,
                "score_a": 2, "score_b": 1, "source_date": "2024-03-01"},
               {"wiki_series_id": "s2", "team_a": "c", "team_b": "a", "best_of": 3,
                "score_a": 0, "score_b": 2, "source_date": "2024-03-02"}]
    completed, rankings = replay_elimination(spec, records)
    assert rankings["ladder"] == ["a", "c", "b"]
    assert [(row["round"], row["winner"]) for row in completed] == [("semi", "a"), ("final", "a")]
    illegal = deepcopy(records)
    illegal[1]["team_b"] = "b"
    with pytest.raises(ValueError):
        replay_elimination(spec, illegal)
    premature = deepcopy(records)
    premature[1]["source_date"] = "2024-02-28"
    with pytest.raises(ValueError):
        replay_elimination(spec, premature)


@pytest.mark.parametrize("second_day", ["2024-02-29", "2024-03-01"])
def test_replay_serializes_source_chronology_independently_of_graph_order(second_day):
    spec = {"teams": ["a", "b", "c", "d"], "stages": [{
        "id": "playoffs", "kind": "graph", "best_of": 3,
        "matches": [{"id": "ab", "a": "a", "b": "b"},
                    {"id": "cd", "a": "c", "b": "d"},
                    {"id": "final", "a": {"winner": "ab"}, "b": {"winner": "cd"}}],
        "ranking": [{"winner": "final"}, {"loser": "final"},
                    {"loser": "ab"}, {"loser": "cd"}],
    }]}
    records = [
        {"wiki_series_id": "z", "team_a": "a", "team_b": "b", "best_of": 3,
         "score_a": 2, "score_b": 1, "source_date": "2024-03-01"},
        {"wiki_series_id": "a", "team_a": "c", "team_b": "d", "best_of": 3,
         "score_a": 2, "score_b": 0, "source_date": second_day},
        {"wiki_series_id": "final", "team_a": "a", "team_b": "c", "best_of": 3,
         "score_a": 1, "score_b": 2, "source_date": "2024-03-02"},
    ]
    completed, rankings = replay_elimination(spec, records)
    assert [row["id"] for row in completed] == ["a", "z", "final"]
    assert rankings["playoffs"] == ["c", "a", "b", "d"]


def test_explicit_standard_name_format_link_clusters_but_similar_title_does_not():
    from scripts.audit_tournament_replay_readiness import _edition_clusters
    events = [{"phase_id": "groups", "title": "League/2024/Spring Season",
               "family": "lpl", "year": 2024, "start_date": "2024-01-01"},
              {"phase_id": "playoffs", "title": "League/2024/Spring Playoffs",
               "family": "lpl", "year": 2024, "start_date": "2024-03-01"}]
    identity = {"CM_StandardLeague": "League", "CM_Year": "2024", "split": "Spring"}
    archived = {
        "groups": {"verified": True, "identity": {**identity, "CM_StandardName": "League 2024 Spring"},
                   "format_links": []},
        "playoffs": {"verified": True, "identity": identity, "format_links": ["League 2024 Spring"]},
    }
    editions, mapping, _ = _edition_clusters(events, archived)
    assert mapping["groups"] == mapping["playoffs"]
    assert editions[0]["edition_start_at"] is None
    archived["playoffs"]["format_links"] = ["League 2024 Sprng"]
    _, mapping, _ = _edition_clusters(events, archived)
    assert mapping["groups"] != mapping["playoffs"]


def test_historical_labels_are_directly_consumable_by_current_ledger(tmp_path, monkeypatch):
    import json
    from scripts import prepare_tournament_historical_manifest as historical
    from scripts.score_tournament_forecasts import evaluate_ledger
    from betting_app.tests.test_tournament_forecast_scoring import _ledger_row

    monkeypatch.setattr(historical, "ARTIFACT_ROOT", tmp_path)
    cohort, bound = tmp_path / "cohort", tmp_path / "bound"
    cohort.mkdir()
    bound.mkdir()
    historical.write_new(cohort / "outcomes.json", {"labels": {"champion": "A"}, "completed": []})
    historical.write_new(cohort / "coverage_manifest.json", {"phases": [{
        "phase_id": "playoff", "outcomes_path": "outcomes.json",
        "outcomes_sha256": historical.digest(cohort / "outcomes.json"),
    }]})
    row = _ledger_row(origin="post-draw", forecast_mode="post_draw")
    historical.write_new(bound / "evaluator_manifest.json", {"events": [{
        "tournament_id": "edition", "phase_id": "playoff", "origins": [{
            "origin_id": "post-draw", "cutoff": row["cutoff"],
            "targets": [{"target_id": "champion"}],
        }],
    }]})
    output = tmp_path / "labels.json"
    historical.labels(cohort, bound, output)
    scored = evaluate_ledger([row], json.loads(output.read_text()), bootstrap=2)
    assert scored["target_scores"][0]["scores"]["brier"] == pytest.approx(.125)


@pytest.mark.parametrize("clock", ["published_at", "available_at"])
def test_late_roster_evidence_is_rejected_by_pair_consumer_before_inference(clock, monkeypatch):
    from betting_app.tests.test_siamese_research_dataset import match
    from betting_app.tests.test_tournament_prediction import ROSTERS, artifact, predict
    from src.models.tournament_prediction import prepare_pair_bundle

    pairs = prepare_pair_bundle(
        [match(1, "2024-01-01")], ROSTERS,
        cutoff="2024-02-01T00:00:00Z", best_ofs=[1, 3, 5],
        roster_provenance={
            team: {"player_ids": players, "source_sha256": "a" * 64,
                   "published_at": "2024-01-31T00:00:00Z",
                   "available_at": "2024-01-31T00:00:00Z"}
            for team, players in ROSTERS.items()
        },
    )
    # Mutate the delivered bundle, not preparation: a consumer must guard its
    # own boundary even when explicit IDs and all legal pair rows are present.
    pairs["roster_provenance"]["a"][clock] = "2024-02-01T00:00:01Z"
    from src.models.siamese_series import SiameseSeriesModel

    def forbidden_predict(*args, **kwargs):
        pytest.fail("late roster reached prediction callback")

    monkeypatch.setattr(SiameseSeriesModel, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="roster.*(cutoff|origin|availability|publication)"):
        predict(artifact(), pairs, mode="exploratory")


@pytest.mark.parametrize("clock", ["published_at", "available_at"])
@pytest.mark.parametrize("value", [None, "2024-02-01T01:00:00+01:00"])
def test_unknown_or_cutoff_roster_clock_remains_uncertified_scenario(clock, value):
    from betting_app.tests.test_siamese_research_dataset import match
    from betting_app.tests.test_tournament_prediction import ROSTERS, artifact, predict
    from src.models.tournament_prediction import prepare_pair_bundle

    pairs = prepare_pair_bundle(
        [match(1, "2024-01-01")], ROSTERS,
        cutoff="2024-02-01T00:00:00Z", best_ofs=[1, 3, 5],
        roster_provenance={team: {clock: value} for team in ROSTERS},
    )
    probabilities, provenance = predict(artifact(), pairs, mode="exploratory")
    assert probabilities
    assert provenance["availability_certified"] is False
    assert provenance["roster_provenance"] == pairs["roster_provenance"]
    assert provenance["availability_gaps"]


@pytest.mark.parametrize("clock", ["published_at", "available_at"])
def test_naive_delivered_roster_clock_is_rejected_before_inference(clock, monkeypatch):
    from betting_app.tests.test_siamese_research_dataset import match
    from betting_app.tests.test_tournament_prediction import ROSTERS, artifact, predict
    from src.models.siamese_series import SiameseSeriesModel
    from src.models.tournament_prediction import prepare_pair_bundle

    pairs = prepare_pair_bundle(
        [match(1, "2024-01-01")], ROSTERS,
        cutoff="2024-02-01T00:00:00Z", best_ofs=[1, 3, 5],
    )
    pairs["roster_provenance"] = {"a": {clock: "2024-01-31T00:00:00"}}

    def forbidden_predict(*args, **kwargs):
        pytest.fail("naive roster clock reached prediction callback")

    monkeypatch.setattr(SiameseSeriesModel, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="roster.*aware"):
        predict(artifact(), pairs, mode="exploratory")
