"""Tournament probabilities must use direct native C0 series outputs."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from betting_app.services import tournament_cache_service as cache
from betting_app.services import tournament_service
from betting_app.services.tournament_service import TournamentSimulator
from betting_app.services import enc_simulation_service
from betting_app.services.enc_simulation_service import EncSimulator, EncTeam


TEAM_IDS = {"Alpha": "team-source-a", "Beta": "team-source-b"}
TEAM_ROSTERS = {
    "Alpha": ("a1", "a2", "a3", "a4", "a5"),
    "Beta": ("b1", "b2", "b3", "b4", "b5"),
}


def test_tournament_uses_direct_c0_series_probability_and_full_native_request(monkeypatch):
    requests = []

    def predict(features):
        requests.append(features)
        return SimpleNamespace(prob_a=0.72, diagnostics={"model_identity": {"name": "Causal-C0"}})

    monkeypatch.setattr(tournament_service.c0_inference, "predict_c0", predict)
    simulator = TournamentSimulator(
        team_ids=TEAM_IDS,
        team_rosters=TEAM_ROSTERS,
        decision_at=datetime(2026, 10, 5, tzinfo=timezone.utc),
        competition_context="lck_2026_playoffs",
    )

    probability = simulator.estimate_matchup_probability("Alpha", "Beta", best_of=3)

    assert probability == 0.72  # No map-to-series projection or bracket recalibration.
    request = requests[0]
    assert request["canonical"]["id"] is None
    assert request["c0_request"] == {
        "team1_id": "team-source-a",
        "team2_id": "team-source-b",
        "roster_a": list(TEAM_ROSTERS["Alpha"]),
        "roster_b": list(TEAM_ROSTERS["Beta"]),
        "best_of": 3,
        "decision_at": "2026-10-05T00:00:00+00:00",
        "competition_context": "lck_2026_playoffs",
        "start_at": None,
        "mode": "full",
    }
    assert simulator._provenance(["Alpha", "Beta"])["native_prediction_provenance"][0]["diagnostics"] == {
        "model_identity": {"name": "Causal-C0"}
    }


def test_tournament_pair_cache_is_partitioned_by_decision_origin_and_rosters(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tournament_service.c0_inference,
        "predict_c0",
        lambda features: (calls.append(features) or SimpleNamespace(prob_a=0.61)),
    )
    origin = datetime(2026, 10, 5, tzinfo=timezone.utc)
    simulator = TournamentSimulator(team_ids=TEAM_IDS, team_rosters=TEAM_ROSTERS, decision_at=origin)
    assert simulator.estimate_matchup_probability("Alpha", "Beta", 1) == 0.61
    assert simulator.estimate_matchup_probability("Alpha", "Beta", 1) == 0.61
    assert len(calls) == 1

    later = TournamentSimulator(
        team_ids=TEAM_IDS,
        team_rosters={**TEAM_ROSTERS, "Alpha": ("a1", "a2", "a3", "a4", "a6")},
        decision_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )
    assert later.estimate_matchup_probability("Alpha", "Beta", 1) == 0.61
    assert len(calls) == 2
    assert calls[0]["c0_request"]["roster_a"] != calls[1]["c0_request"]["roster_a"]
    assert calls[0]["c0_request"]["decision_at"] != calls[1]["c0_request"]["decision_at"]


def test_tournament_fails_closed_without_exact_c0_team_and_roster_ids(monkeypatch):
    monkeypatch.setattr(
        tournament_service.c0_inference,
        "predict_c0",
        lambda features: pytest.fail("C0 must not be called with invented IDs"),
    )
    simulator = TournamentSimulator(
        team_ids=TEAM_IDS,
        team_rosters={**TEAM_ROSTERS, "Beta": ("b1", "b2", "b3", "b4")},
    )
    with pytest.raises(ValueError, match="Native C0 unavailable"):
        simulator.estimate_matchup_probability("Alpha", "Beta", best_of=5)


def test_tournament_rejects_partial_team_identity_override():
    simulator = TournamentSimulator(team_ids={"Alpha": "team-source-a"})
    with pytest.raises(ValueError, match="requires team ID and roster IDs as a verified pair"):
        simulator._team_identity("Alpha")


def test_legacy_gl_or_a1_probability_cache_cannot_survive_c0_cutover(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.get_cache_file("audit").write_text(json.dumps({
        "probability_model": "gl_logistic_map_iid_series",
        "probability_model_version": "gl-logistic-map-iid-series-v2",
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "input_fingerprint": "current-input",
        "depths": {"10000": {"simulations": 10000, "standings": []}},
    }))
    assert cache.get_cached_simulation("audit", expected_input_fingerprint="current-input") is None


def test_cache_rejects_heuristic_result_even_when_cache_envelope_claims_c0(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.get_cache_file("forged").write_text(json.dumps({
        "probability_model": "Causal-C0",
        "probability_model_version": cache.TOURNAMENT_PROBABILITY_VERSION,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "input_fingerprint": "current-input",
        "depths": {
            "10000": {
                "simulations": 10000,
                "standings": [],
                "provenance": {
                    "probability_model": "gl_logistic_map_iid_series",
                    "probability_model_version": "gl-logistic-map-iid-series-v2",
                },
            }
        },
    }))
    assert cache.get_cached_simulation("forged", expected_input_fingerprint="current-input") is None


def test_c0_cache_requires_matching_inputs_and_freshness(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cached_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "probability_model": "Causal-C0",
        "probability_model_version": cache.TOURNAMENT_PROBABILITY_VERSION,
        "native_prediction_provenance": [],
    }
    cache.get_cache_file("fresh").write_text(json.dumps({
        "probability_model": "Causal-C0",
        "probability_model_version": cache.TOURNAMENT_PROBABILITY_VERSION,
        "cached_at": cached_at,
        "input_fingerprint": "roster-artifact-origin-a",
        "source_identity_fingerprints": {
            "10000": cache._source_identity_fingerprint(provenance),
        },
        "depths": {
            "10000": {
                "simulations": 10000,
                "standings": [],
                "provenance": provenance,
            }
        },
    }))
    assert cache.get_cached_simulation(
        "fresh", expected_input_fingerprint="roster-artifact-origin-a",
    ) is not None
    assert cache.get_cached_simulation(
        "fresh", expected_input_fingerprint="roster-artifact-origin-b",
    ) is None
    stale = json.loads(cache.get_cache_file("fresh").read_text())
    stale["cached_at"] = (datetime.now(timezone.utc) - timedelta(seconds=cache.CACHE_TTL_SECONDS + 1)).isoformat()
    cache.get_cache_file("fresh").write_text(json.dumps(stale))
    assert cache.get_cached_simulation(
        "fresh", expected_input_fingerprint="roster-artifact-origin-a",
    ) is None


def test_enc_uses_synthetic_national_identity_without_organization_features(monkeypatch):
    requests = []
    monkeypatch.setattr(
        enc_simulation_service,
        "predict_c0",
        lambda features: (requests.append(features) or SimpleNamespace(prob_a=0.74)),
    )
    monkeypatch.setattr(enc_simulation_service.random, "random", lambda: 0.5)
    teams = [
        EncTeam(
            nation=f"Nation {index}",
            entry_stage="group_stage" if index < 8 else "play_in",
            organization_id=f"national:nation-{index}",
            roster=tuple(f"player-{index}-{slot}" for slot in range(5)),
        )
        for index in range(32)
    ]

    assert EncSimulator(teams)._winner("Nation 0", "Nation 1", 3) == "Nation 0"
    assert requests[0]["canonical"]["id"] is None
    request = requests[0]["c0_request"]
    assert request["team1_id"] == "national:nation-0"
    assert request["team2_id"] == "national:nation-1"
    assert request["roster_a"] == [f"player-0-{slot}" for slot in range(5)]
    assert request["mode"] == "no_organization"
    assert request["competition_context"] == "enc-2027"


def _mock_current_roster_db(monkeypatch):
    roles = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")

    class Result:
        def __init__(self, team):
            self.team = team

        def mappings(self):
            return self

        def all(self):
            prefix = self.team.lower()
            return [
                {"team_id": "11" if prefix == "alpha" else "12",
                 "player_id": f"{prefix}-{index}", "role": role}
                for index, role in enumerate(roles)
            ]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, parameters):
            return Result(parameters["normalized"])

    monkeypatch.setattr(tournament_service, "get_session", Session)


def test_database_internal_golgg_ids_are_never_used_as_native_organization_ids(monkeypatch):
    _mock_current_roster_db(monkeypatch)
    lookups = []

    def resolve_native_id(internal_team_row_id, *, team_name=None):
        lookups.append((internal_team_row_id, team_name))
        return {"Alpha": "native-org-901", "Beta": "native-org-902"}[team_name]

    monkeypatch.setattr(tournament_service.mapping_service, "native_golgg_team_id", resolve_native_id)
    requests = []
    monkeypatch.setattr(
        tournament_service.c0_inference,
        "predict_c0",
        lambda features: (requests.append(features) or SimpleNamespace(prob_a=0.63)),
    )

    simulator = TournamentSimulator()
    assert simulator.estimate_matchup_probability("Alpha", "Beta", 3) == 0.63
    request = requests[0]["c0_request"]
    assert (request["team1_id"], request["team2_id"]) == ("native-org-901", "native-org-902")
    assert "11" not in (request["team1_id"], request["team2_id"])
    assert request["mode"] == "full"
    assert lookups == [(None, "Alpha"), (None, "Beta")]


def test_missing_database_organization_identity_uses_explicit_no_organization_mode(monkeypatch):
    _mock_current_roster_db(monkeypatch)
    monkeypatch.setattr(
        tournament_service.mapping_service,
        "native_golgg_team_id",
        lambda internal_team_row_id, *, team_name=None: None,
    )
    requests = []
    monkeypatch.setattr(
        tournament_service.c0_inference,
        "predict_c0",
        lambda features: (requests.append(features) or SimpleNamespace(prob_a=0.63)),
    )

    simulator = TournamentSimulator()
    assert simulator.estimate_matchup_probability("Alpha", "Beta", 3) == 0.63
    request = requests[0]["c0_request"]
    assert request["team1_id"] == "tournament-target:alpha"
    assert request["team2_id"] == "tournament-target:beta"
    assert request["mode"] == "no_organization"
    provenance = simulator._provenance(["Alpha", "Beta"])
    assert provenance["organization_identity_available"] == {"Alpha": False, "Beta": False}
    assert provenance["organization_identity_source"] == {"Alpha": "unobserved", "Beta": "unobserved"}


def test_tournament_cache_fingerprint_includes_organization_identity_availability():
    base = {
        "probability_model_version": cache.TOURNAMENT_PROBABILITY_VERSION,
        "team_ids": {"Alpha": "same-target-id"},
        "team_rosters": {"Alpha": ["a1", "a2", "a3", "a4", "a5"]},
        "decision_at": "2026-10-05T00:00:00+00:00",
        "mode": "full",
        "competition_context": "fixture",
        "organization_identity_available": {"Alpha": True},
        "organization_identity_source": {"Alpha": "golgg_native_team_id"},
    }
    absent = {
        **base,
        "organization_identity_available": {"Alpha": False},
        "organization_identity_source": {"Alpha": "unobserved"},
    }
    artifact_identity = {"version": "native-fixture"}
    assert cache._input_fingerprint(base, artifact_identity) != cache._input_fingerprint(absent, artifact_identity)


@pytest.mark.parametrize("explicit", [True, False])
def test_tournament_discloses_actual_roster_basis_without_publication_claim(monkeypatch, explicit):
    _mock_current_roster_db(monkeypatch)
    monkeypatch.setattr(tournament_service.mapping_service, "native_golgg_team_id", lambda *_args, **_kwargs: None)
    diagnostics = {
        "feature_provenance": {
            "source_kind": "canonical_frozen_golgg_history",
            "raw_history_sha256": "actual-source-hash",
            "dataset_freshness": {"raw_history_max_effective_release_day": "2026-09-08"},
        },
        "requested_decision_at": "2026-10-05T00:00:00+00:00",
    }
    monkeypatch.setattr(
        tournament_service.c0_inference, "predict_c0",
        lambda _features: SimpleNamespace(prob_a=0.72, diagnostics=diagnostics),
    )
    simulator = TournamentSimulator(
        team_ids=TEAM_IDS if explicit else None,
        team_rosters=TEAM_ROSTERS if explicit else None,
        decision_at=datetime(2026, 10, 5, tzinfo=timezone.utc),
    )

    assert simulator.estimate_matchup_probability("Alpha", "Beta", 3) == 0.72
    provenance = simulator._provenance(["Alpha", "Beta"])
    pair = provenance["native_prediction_provenance"][0]
    assert pair["diagnostics"] == diagnostics
    assert provenance["roster_basis"]["Alpha"] == (
        "explicit_scenario_roster" if explicit else "current_stored_roster"
    )
    assert pair["roster_provenance"]["basis"]["Alpha"] == provenance["roster_basis"]["Alpha"]
    assert pair["roster_provenance"]["published_at"] is None
    assert pair["roster_provenance"]["authenticated_prior_announcement"] is False
    assert provenance["point_in_time_certified"] is False
