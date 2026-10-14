"""Tests for tournament bracket simulation engine and endpoints across multiple leagues."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from betting_app.api.main import app
from betting_app.api.routers import tournaments as tournament_router
from betting_app.services import c0_inference as c0_core
from betting_app.services import tournament_cache_service as cache_core
from betting_app.services.tournament_service import (
    SUPPORTED_BRACKETS,
    TournamentSimulator,
    BracketMatchNode,
    TournamentBracket,
    get_lck_2026_playoffs_bracket,
    get_lec_2026_summer_playoffs_bracket,
    get_lpl_2026_split3_playoffs_bracket,
    get_lcs_2026_championship_bracket,
)
from betting_app.services.liquipedia_bracket_service import (
    LiquipediaBracketService,
    TOURNAMENT_METADATA,
)
from betting_app.services.enc_simulation_service import (
    EncConfigurationError,
    EncSimulator,
    EncTeam,
    build_enc_configuration,
)
from betting_app.services import enc_simulation_service as enc_core
client = TestClient(app)


def _sim_for(teams, *, seed=82, competition_context=None):
    team_ids = {team: f"test-team-id:{team}" for team in teams}
    rosters = {
        team: [f"test-player-id:{team}:{index}" for index in range(5)]
        for team in teams
    }
    from datetime import UTC, datetime
    return TournamentSimulator(
        seed=seed, team_ids=team_ids, team_rosters=rosters,
        decision_at=datetime(2026, 10, 1, tzinfo=UTC),
        competition_context=competition_context,
    )


@pytest.fixture(autouse=True)
def controlled_native_c0(monkeypatch):
    oracle = lambda _features: SimpleNamespace(prob_a=0.5, diagnostics={})
    monkeypatch.setattr(c0_core, "predict_c0", oracle)
    monkeypatch.setattr(enc_core, "predict_c0", oracle)


@pytest.fixture
def offline_tournament_api(monkeypatch):
    all_teams = {
        team for builder in SUPPORTED_BRACKETS.values() for team in builder().teams
    }

    def simulator_factory(*, seed=None, competition_context=None, **kwargs):
        return _sim_for(
            all_teams, seed=seed or 82, competition_context=competition_context,
        )

    monkeypatch.setattr(tournament_router, "TournamentSimulator", simulator_factory)
    monkeypatch.setattr(cache_core, "TournamentSimulator", simulator_factory)
    monkeypatch.setattr(
        "betting_app.services.tournament_cache_service.get_c0_artifact_identity",
        lambda: {"artifact_sha256": "fixture-c0"},
    )

    def local_bracket(self, tournament_id, **kwargs):
        return {"bracket": SUPPORTED_BRACKETS[tournament_id](), "source": "offline_fixture", "status": "ready"}

    monkeypatch.setattr(LiquipediaBracketService, "sync_bracket", local_bracket)
    monkeypatch.setattr(LiquipediaBracketService, "load_bracket", local_bracket)


def test_supported_brackets_structure() -> None:
    assert "lck_2026_playoffs" in SUPPORTED_BRACKETS
    assert "lec_2026_summer_playoffs" in SUPPORTED_BRACKETS
    assert "lpl_2026_split3_playoffs" in SUPPORTED_BRACKETS
    assert "lcs_2026_championship" in SUPPORTED_BRACKETS

    lck = get_lck_2026_playoffs_bracket()
    assert lck.id == "lck_2026_playoffs"
    assert len(lck.teams) == 6
    assert lck.matches["LB_R1"].winner == "Dplus"

    lec = get_lec_2026_summer_playoffs_bracket()
    assert lec.region == "LEC"
    assert len(lec.teams) == 6
    assert "UB_SF1" in lec.matches

    lpl = get_lpl_2026_split3_playoffs_bracket()
    assert lpl.region == "LPL"
    assert len(lpl.teams) == 8
    assert "UB_R1_M1" in lpl.matches
    assert "LB_R3" in lpl.matches


    lcs = get_lcs_2026_championship_bracket()
    assert lcs.region == "LCS"
    assert len(lcs.teams) == 6
    assert "UB_R1_M1" in lcs.matches
    assert "Grand_Final" in lcs.matches

def test_tournament_simulator_deterministic_manual_override() -> None:
    bracket = get_lck_2026_playoffs_bracket()
    sim = _sim_for(bracket.teams)
    res = sim.simulate(
        bracket, n_simulations=100,
        manual_overrides={"LB_R3": "T1", "LB_Final": "T1", "Grand_Final": "T1"},
    )
    assert res["simulations"] == 100
    standings = {s["team"]: s["champion_prob"] for s in res["standings"]}
    assert standings["T1"] == 1.0


def test_tournaments_api_endpoints(offline_tournament_api) -> None:
    res = client.get("/tournaments")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3
    ids = [t["id"] for t in data]
    assert "lck_2026_playoffs" in ids
    assert "lec_2026_summer_playoffs" in ids
    assert "lpl_2026_split3_playoffs" in ids

    # LCK simulation endpoint
    sim_lck = client.post(
        "/tournaments/lck_2026_playoffs/simulate",
        json={"simulations": 200},
    )
    assert sim_lck.status_code == 200
    standings = sim_lck.json()["standings"]
    assert len(standings) == 6
    total_p = sum(s["champion_prob"] for s in standings)
    assert round(total_p, 1) == 1.0

    # LEC simulation endpoint
    sim_lec = client.post(
        "/tournaments/lec_2026_summer_playoffs/simulate",
        json={"simulations": 200},
    )
    assert sim_lec.status_code == 200
    assert len(sim_lec.json()["standings"]) == 6
    # LPL simulation endpoint
    sim_lpl = client.post(
        "/tournaments/lpl_2026_split3_playoffs/simulate",
        json={"simulations": 200},
    )
    assert sim_lpl.status_code == 200
    lpl_standings = sim_lpl.json()["standings"]
    assert len(lpl_standings) == 8
    assert round(sum(s["champion_prob"] for s in lpl_standings), 1) == 1.0


    # Bracket GET endpoint with sync metadata
    get_lck = client.get("/tournaments/lck_2026_playoffs")
    assert get_lck.status_code == 200
    get_lck_data = get_lck.json()
    assert "source" in get_lck_data
    assert "status" in get_lck_data
    assert "bracket" in get_lck_data
    assert len(get_lck_data["standings"]) == 6

    # Sync POST endpoint
    sync_res = client.post(
        "/tournaments/lck_2026_playoffs/sync",
        json={"source": "auto", "force": False},
    )
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert sync_data["tournament_id"] == "lck_2026_playoffs"
    assert "source" in sync_data
    assert "bracket" in sync_data


def test_liquipedia_parser_wikitext_and_html() -> None:
    service = LiquipediaBracketService()

    # Test Wikitext parsing
    wikitext_sample = (
        "{{Bracket\n"
        "|r1m1team1=KT Rolster |r1m1score1=3 |r1m1win1=1\n"
        "|r1m1team2=Dplus KIA |r1m1score2=0\n"
        "|r1m2team1=T1 |r1m2score1=3 |r1m2win1=1\n"
        "|r1m2team2=BNK FearX |r1m2score2=2\n"
        "}}"
    )
    wiki_matches = service.parse_bracket_wikitext(wikitext_sample)
    assert len(wiki_matches) == 2
    assert wiki_matches[0]["team1"] == "KT Rolster"
    assert wiki_matches[0]["score1"] == 3
    assert wiki_matches[0]["winner"] == "KT Rolster"

    # Test HTML parsing
    html_sample = (
        '<div class="bracket-game">'
        '  <span class="bracket-team">KT Rolster</span><span class="bracket-score">1</span>'
        '  <span class="bracket-team">Dplus KIA</span><span class="bracket-score">3</span>'
        '</div>'
    )
    html_matches = service.parse_bracket_html(html_sample)
    assert len(html_matches) == 1
    assert html_matches[0]["team1"] == "KT Rolster"
    assert html_matches[0]["score2"] == 3
    assert html_matches[0]["winner"] == "Dplus KIA"


def test_bracket_sync_service_chronological_mapping() -> None:
    service = LiquipediaBracketService()
    bracket = get_lck_2026_playoffs_bracket()
    round_order = TOURNAMENT_METADATA["lck_2026_playoffs"]["round_order"]

    parsed_matches = [
        {"team1": "T1", "team2": "BNK FearX", "score1": 3, "score2": 2, "winner": "T1", "date": "2026-08-29 08:00:00"},
        {"team1": "Dplus KIA", "team2": "KT Rolster", "score1": 0, "score2": 3, "winner": "KT Rolster", "date": "2026-08-30 08:00:00"},
        {"team1": "Gen.G", "team2": "KT Rolster", "score1": 3, "score2": 0, "winner": "Gen.G", "date": "2026-09-01 08:00:00"},
        {"team1": "KT Rolster", "team2": "Dplus KIA", "score1": 1, "score2": 3, "winner": "Dplus KIA", "date": "2026-09-04 08:00:00"},
    ]

    updated_bracket, count = service.map_matches_chronologically(bracket, parsed_matches, round_order)
    assert count == 4
    assert updated_bracket.matches["UB_R1_M1"].winner == "KT Rolster"
    assert updated_bracket.matches["UB_R1_M2"].winner == "T1"
    assert updated_bracket.matches["UB_R2_M1"].winner == "Gen.G"
    assert updated_bracket.matches["LB_R2"].winner == "Dplus"



@pytest.mark.parametrize("method,path", [("get", "/tournaments/enc"), ("post", "/tournaments/enc/simulate")])
def test_enc_source_query_failure_is_server_error(client, monkeypatch, method, path):
    from contextlib import nullcontext

    class BrokenSourceSession:
        def execute(self, *_args, **_kwargs):
            raise ConnectionError("private source query diagnostic")

    monkeypatch.setattr(enc_core, "get_session", lambda: nullcontext(BrokenSourceSession()))
    monkeypatch.setattr(enc_core, "_rating_snapshot", lambda _session: (None, []))
    response = getattr(client, method)(path, **({"json": {"simulations": 100}} if method == "post" else {}))
    assert response.status_code == 500
    assert response.json() == {"detail": "Native C0 service unavailable."}
    assert "private source query diagnostic" not in response.text


def test_enc_missing_native_rosters_fail_closed() -> None:
    configuration = build_enc_configuration(rating_rows=[])
    poland = next(team for team in configuration["teams"] if team["nation"] == "Poland")

    assert configuration["simulation_ready"] is False
    assert poland["selection_status"] == "incomplete"
    assert poland["missing_roles"] == ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
    with pytest.raises(EncConfigurationError):
        EncSimulator.from_configuration(configuration)




def test_enc_simulator_uses_published_stage_sizes_and_series_lengths() -> None:
    teams = [
        *(
            EncTeam(
                f"Direct {index}", "group_stage", f"test-org:{index}",
                tuple(f"test-player:{index}:{slot}" for slot in range(5)),
            )
            for index in range(8)
        ),
        *(
            EncTeam(
                f"Play-In {index}", "play_in", f"test-org:{index + 8}",
                tuple(f"test-player:{index + 8}:{slot}" for slot in range(5)),
            )
            for index in range(24)
        ),
    ]
    result = EncSimulator(teams).simulate(100)

    assert result["format"]["participants"] == 32
    assert result["format"]["play_in"].startswith("24 teams, 4 groups of 6")
    assert len(result["standings"]) == 32
    assert sum(team["champion_prob"] for team in result["standings"]) == 1.0
    assert all(team["group_stage_prob"] == 1.0 for team in result["standings"] if team["entry_stage"] == "group_stage")


def test_lec_bracket_round_order_keys_match() -> None:
    """Verify LEC round_order keys in metadata strictly match bracket match IDs."""
    meta = TOURNAMENT_METADATA["lec_2026_summer_playoffs"]
    bracket = get_lec_2026_summer_playoffs_bracket()
    for node_id in meta["round_order"]:
        assert node_id in bracket.matches, f"Node {node_id} from round_order must exist in LEC matches"


def test_score_alignment_reversed_order() -> None:
    """Verify scores are aligned to node.team1 and node.team2 even when parsed match has reversed sides."""
    service = LiquipediaBracketService()
    bracket = get_lck_2026_playoffs_bracket()
    # In bracket: UB_R1_M1 has team1="KT Rolster", team2="Dplus"
    # Parsed match has team1="Dplus KIA" (0) vs team2="KT Rolster" (3)
    parsed = [
        {
            "team1": "Dplus KIA",
            "team2": "KT Rolster",
            "score1": 0,
            "score2": 3,
            "winner": "KT Rolster",
            "date": "2026-08-30 08:00:00",
        }
    ]
    updated, count = service.map_matches_chronologically(bracket, parsed, ["UB_R1_M1"])
    assert count == 1
    node = updated.matches["UB_R1_M1"]
    assert node.team1 == "KT Rolster"
    assert node.team2 == "Dplus"
    assert node.score1 == 3, "KT Rolster (team1) must have score 3, not 0"
    assert node.score2 == 0, "Dplus (team2) must have score 0, not 3"
    assert node.winner == "KT Rolster"


def test_all_supported_brackets_simulate_successfully() -> None:
    """Verify each supported bracket tree simulates without dead ends or missing slots."""
    for tournament_id, builder in SUPPORTED_BRACKETS.items():
        bracket = builder()
        sim = _sim_for(bracket.teams, seed=82)
        res = sim.simulate(bracket, n_simulations=200)
        assert res["tournament_id"] == tournament_id
        assert len(res["standings"]) == len(bracket.teams)
        total_champ = sum(s["champion_prob"] for s in res["standings"])
        assert round(total_champ, 2) == 1.0, f"{tournament_id} champion probabilities must sum to 1"


def test_lpl_bracket_sync_and_upper_finals() -> None:
    """Verify LPL 2026 bracket mapping places AL vs BLG in Upper Finals and does not pre-populate Grand Final."""
    service = LiquipediaBracketService()
    bracket = get_lpl_2026_split3_playoffs_bracket()
    round_order = TOURNAMENT_METADATA["lpl_2026_split3_playoffs"]["round_order"]

    # Simulated real Cargo export matches up to 2026-09-05
    fandom_matches = [
        {"team1": "Top Esports", "team2": "LGD Gaming", "score1": 2, "score2": 3, "winner": "LGD Gaming", "date": "2026-08-29 09:00:00"},
        {"team1": "JD Gaming", "team2": "Team WE", "score1": 1, "score2": 3, "winner": "Team WE", "date": "2026-08-30 09:00:00"},
        {"team1": "Bilibili Gaming", "team2": "Team WE", "score1": 3, "score2": 1, "winner": "Bilibili Gaming", "date": "2026-09-03 09:00:00"},
        {"team1": "Anyone's Legend", "team2": "LGD Gaming", "score1": 3, "score2": 1, "winner": "Anyone's Legend", "date": "2026-09-04 09:00:00"},
        {"team1": "Invictus Gaming", "team2": "Top Esports", "score1": 3, "score2": 2, "winner": "Invictus Gaming", "date": "2026-09-05 06:00:00"},
        {"team1": "Ninjas in Pyjamas.CN", "team2": "JD Gaming", "score1": 1, "score2": 0, "winner": None, "date": "2026-09-05 11:00:00"},
        {"team1": "Invictus Gaming", "team2": "Team WE", "score1": None, "score2": None, "winner": None, "date": "2026-09-06 06:00:00"},
        {"team1": "Anyone's Legend", "team2": "Bilibili Gaming", "score1": None, "score2": None, "winner": None, "date": "2026-09-07 09:00:00"},
    ]

    updated, count = service.map_matches_chronologically(bracket, fandom_matches, round_order)
    assert count >= 6

    # Upper Finals MUST be Anyone's Legend vs Bilibili Gaming
    ub_final = updated.matches["UB_Final"]
    assert {ub_final.team1, ub_final.team2} == {"Anyone's Legend", "Bilibili Gaming"}
    assert ub_final.winner is None

    # Lower Round 2 Match 1 MUST be Invictus Gaming vs Team WE
    lb_r2_m1 = updated.matches["LB_R2_M1"]
    assert {lb_r2_m1.team1, lb_r2_m1.team2} == {"Invictus Gaming", "Team WE"}

    # Lower Round 2 Match 2 MUST await winner of NIP vs JDG and loser of AL vs LGD (LGD Gaming)
    lb_r2_m2 = updated.matches["LB_R2_M2"]
    assert "LGD Gaming" in (lb_r2_m2.team1, lb_r2_m2.team2)

    # Grand Final MUST NOT have predetermined teams or winner
    grand_final = updated.matches["Grand_Final"]
    assert grand_final.team1 is None
    assert grand_final.team2 is None
    assert grand_final.winner is None



def _audit_final(**kwargs) -> TournamentBracket:
    node = BracketMatchNode(
        id="title", name="Title", round_name="Final", bracket_section="final",
        team1="Alpha", team2="Beta", **kwargs,
    )
    return TournamentBracket("audit", "Audit", "test", "single_elimination", {"title": node}, ["Alpha", "Beta"])



def test_audit_rejects_nonparticipant_override() -> None:
    bracket = get_lec_2026_summer_playoffs_bracket()
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(
            bracket, 1, {"UB_SF1": "G2 Esports"},
        )


def test_audit_rejects_override_of_confirmed_result() -> None:
    bracket = get_lck_2026_playoffs_bracket()
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(bracket, 1, {"UB_R1_M1": "Dplus"})


def test_audit_rejects_unknown_override_match() -> None:
    bracket = get_lec_2026_summer_playoffs_bracket()
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(bracket, 1, {"absent": "G2 Esports"})


def test_audit_arbitrary_final_id_preserves_champion_and_runner_up_mass() -> None:
    result = _sim_for(["Alpha", "Beta"]).simulate(
        _audit_final(winner="Alpha", score1=3, score2=0), 1,
    )
    standings = {row["team"]: row for row in result["standings"]}
    assert standings["Alpha"]["champion_prob"] == 1.0
    assert standings["Beta"]["champion_prob"] == 0.0
    assert sum(row["top2_prob"] for row in standings.values()) == 2.0
    assert all(row["top4_prob"] == 1.0 for row in standings.values())


@pytest.mark.parametrize("change", ["cycle", "missing_target", "slot_collision", "missing_opponent", "duplicate_seed"])
def test_audit_rejects_invalid_graph(change) -> None:
    bracket = get_lec_2026_summer_playoffs_bracket()
    if change == "cycle":
        bracket.matches["Grand_Final"].next_match_winner_id = "UB_SF1"
    elif change == "missing_target":
        bracket.matches["UB_SF1"].next_match_winner_id = "absent"
    elif change == "slot_collision":
        bracket.matches["UB_SF2"].next_match_winner_slot = 1
    elif change == "missing_opponent":
        bracket.matches["UB_SF1"].team2 = None
    else:
        bracket.matches["UB_SF1"].team2 = "G2 Esports"
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(bracket, 1)


def test_audit_rejects_contradictory_downstream_live_participants() -> None:
    bracket = get_lck_2026_playoffs_bracket()
    bracket.matches["LB_R2"].team1 = "T1"
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(bracket, 1)


def test_audit_rng_is_local_and_seed_reproducible() -> None:
    state = random.getstate()
    first = _sim_for(["Alpha", "Beta"], seed=82).simulate(_audit_final(), 31)
    second = _sim_for(["Alpha", "Beta"], seed=82).simulate(_audit_final(), 31)
    assert first["standings"] == second["standings"]
    assert random.getstate() == state
    assert sum(row["champion_prob"] for row in first["standings"]) == pytest.approx(1.0)


@pytest.mark.parametrize("count", [0, -1, True])
def test_audit_rejects_invalid_simulation_count(count) -> None:
    with pytest.raises(ValueError):
        _sim_for(["Alpha", "Beta"]).simulate(_audit_final(), count)



def test_audit_api_rejects_impossible_manual_winner(offline_tournament_api) -> None:
    response = client.post(
        "/tournaments/lec_2026_summer_playoffs/simulate",
        json={"simulations": 100, "manual_overrides": {"UB_SF1": "G2 Esports"}},
    )
    assert response.status_code == 422


def test_audit_renamed_double_elimination_routes_and_placements() -> None:
    bracket = get_lec_2026_summer_playoffs_bracket()
    forced = {
        "UB_SF1": "Karmine Corp", "UB_SF2": "G2 Esports",
        "LB_R1_M1": "GIANTX", "LB_R1_M2": "Team Vitality",
        "UB_Final": "Karmine Corp", "LB_SF": "GIANTX",
        "LB_Final": "G2 Esports", "Grand_Final": "G2 Esports",
    }
    renamed = {name: f"node{index}" for index, name in enumerate(bracket.matches)}
    for node in bracket.matches.values():
        node.id = renamed[node.id]
        node.next_match_winner_id = renamed.get(node.next_match_winner_id)
        node.next_match_loser_id = renamed.get(node.next_match_loser_id)
    bracket.matches = {node.id: node for node in bracket.matches.values()}
    result = _sim_for(bracket.teams).simulate(
        bracket, 1, {renamed[name]: winner for name, winner in forced.items()},
    )
    standings = {row["team"]: row for row in result["standings"]}
    assert standings["G2 Esports"]["champion_prob"] == 1.0
    assert standings["Karmine Corp"]["top2_prob"] == 1.0
    assert standings["GIANTX"]["top3_prob"] == 1.0
    assert standings["GIANTX"]["top2_prob"] == 0.0
    assert standings["Team Vitality"]["top4_prob"] == 1.0
    assert standings["Team Vitality"]["top3_prob"] == 0.0
    assert standings["Natus Vincere"]["top4_prob"] == 0.0
    for cutoff, key in ((1, "champion_prob"), (2, "top2_prob"), (3, "top3_prob"), (4, "top4_prob")):
        assert sum(row[key] for row in standings.values()) == cutoff


def test_audit_single_elimination_does_not_invent_third_place() -> None:
    bracket = _audit_final(winner="Alpha", score1=3, score2=0)
    bracket.teams.extend(["Gamma", "Delta"])
    bracket.matches["semi1"] = BracketMatchNode(
        "semi1", "Semi 1", "Semi", "upper", team1="Alpha", team2="Gamma",
        winner="Alpha", next_match_winner_id="title", next_match_winner_slot=1,
    )
    bracket.matches["semi2"] = BracketMatchNode(
        "semi2", "Semi 2", "Semi", "upper", team1="Beta", team2="Delta",
        winner="Beta", next_match_winner_id="title", next_match_winner_slot=2,
    )
    result = _sim_for(bracket.teams).simulate(bracket, 1)
    standings = {row["team"]: row for row in result["standings"]}
    assert standings["Gamma"]["top3_prob"] is None
    assert standings["Delta"]["top3_prob"] is None
    assert all(row["top4_prob"] == 1.0 for row in standings.values())


@pytest.mark.parametrize("side", [1, 2])
def test_audit_missing_opponent_is_not_an_implicit_bye(side) -> None:
    bracket = get_lec_2026_summer_playoffs_bracket()
    setattr(bracket.matches["UB_SF1"], f"team{side}", None)
    with pytest.raises(ValueError):
        _sim_for(bracket.teams).simulate(bracket, 1)


@pytest.mark.parametrize("best_of", [0, 2, 3.5, True])
def test_audit_rejects_invalid_series_length(best_of) -> None:
    sim = _sim_for(["Alpha", "Beta"])
    with pytest.raises(ValueError):
        sim.estimate_matchup_probability("Alpha", "Beta", best_of)




def test_audit_invalid_probability_cannot_fabricate_series_winner(monkeypatch) -> None:
    monkeypatch.setattr(
        c0_core, "predict_c0",
        lambda _features: SimpleNamespace(prob_a=float("nan"), diagnostics={}),
    )
    sim = _sim_for(["Alpha", "Beta"])
    with pytest.raises(ValueError):
        sim.simulate(_audit_final(), 1)




def test_tournament_caching_and_recalculate_endpoints(offline_tournament_api):
    """Verify tournament caching across path depths and instant response."""
    import json
    from betting_app.services.tournament_cache_service import (
        get_cache_file,
        get_cached_simulation,
        recalculate_and_cache,
    )

    t_id = "lck_2026_playoffs"
    recalc_res = recalculate_and_cache(t_id, depths=[5000, 10000])
    assert recalc_res["tournament_id"] == t_id
    assert "standings" in recalc_res

    input_fingerprint = json.loads(
        get_cache_file(t_id).read_text(encoding="utf-8")
    )["input_fingerprint"]
    cached_10k = get_cached_simulation(
        t_id, n_simulations=10000, expected_input_fingerprint=input_fingerprint,
    )
    assert cached_10k is not None
    assert cached_10k["cached"] is True
    assert cached_10k["simulations"] == 10000

    cached_5k = get_cached_simulation(
        t_id, n_simulations=5000, expected_input_fingerprint=input_fingerprint,
    )
    assert cached_5k is not None
    assert cached_5k["cached"] is True
    assert cached_5k["simulations"] == 5000

    # API GET endpoint returns cached
    resp = client.get(f"/tournaments/{t_id}?simulations=10000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cached"] is True
    assert data["simulations"] == 10000

    # API POST recalculate endpoint
    recalc_resp = client.post(f"/tournaments/{t_id}/recalculate", json={"simulations": 5000})
    assert recalc_resp.status_code == 200
    recalc_data = recalc_resp.json()
    assert "standings" in recalc_data

def test_failed_bracket_refresh_preserves_last_success_timestamp(tmp_path, monkeypatch):
    import json
    service = LiquipediaBracketService(cache_dir=tmp_path)
    snapshot = {
        "source": "fandom_cargo", "status": "success",
        "synced_at": "2026-09-01T12:00:00+00:00",
        "matches": {}, "updated_matches": 0,
    }
    service.write_cached_bracket("lck_2026_playoffs", snapshot)
    monkeypatch.setattr(service, "fetch_fandom_cargo_matches", lambda *_args: [])
    result = service.sync_bracket("lck_2026_playoffs", source="fandom", force=True)
    assert result["synced_at"] == snapshot["synced_at"]
    assert result["status"] != "success"
    assert json.loads(service._get_cache_file("lck_2026_playoffs").read_text()) == snapshot

@pytest.mark.parametrize("raw_content", ["", "not a bracket"])
def test_invalid_manual_bracket_import_stays_offline(tmp_path, monkeypatch, raw_content):
    from unittest.mock import Mock
    network = Mock(side_effect=AssertionError("manual import must stay offline"))
    monkeypatch.setattr("urllib.request.urlopen", network)
    service = LiquipediaBracketService(cache_dir=tmp_path)
    result = service.sync_bracket("lck_2026_playoffs", source="auto", raw_content=raw_content)
    assert result["ok"] is False
    assert network.call_count == 0
    assert not service._get_cache_file("lck_2026_playoffs").exists()


@pytest.mark.parametrize("kind", ["regional", "worlds", "enc"])
def test_tournament_response_retains_pair_source_and_unknown_publication(monkeypatch, kind):
    requests = []

    def predict(features):
        request = features["c0_request"]
        requests.append(request)
        return SimpleNamespace(prob_a=0.5, diagnostics={
            "c0_request": request,
            "requested_decision_at": request["decision_at"],
            "feature_provenance": {
                "source_kind": "canonical_frozen_golgg_history",
                "raw_history_sha256": "retained-source-hash",
                "dataset_freshness": {"raw_history_max_effective_release_day": "2026-09-08"},
            },
        })

    monkeypatch.setattr(c0_core, "predict_c0", predict)
    monkeypatch.setattr(enc_core, "predict_c0", predict)
    if kind == "regional":
        bracket = get_lck_2026_playoffs_bracket()
        result = _sim_for(bracket.teams).simulate(bracket, n_simulations=1)
    elif kind == "worlds":
        from betting_app.services.tournament_service import WorldsSimulator, WorldsTeam

        names = [f"Team {index}" for index in range(19)]
        simulator = WorldsSimulator(
            seed=82,
            team_ids={name: f"organization:{index}" for index, name in enumerate(names)},
            team_rosters={name: [f"player:{index}:{slot}" for slot in range(5)] for index, name in enumerate(names)},
        )
        result = simulator.simulate_worlds(
            [WorldsTeam(name, "scenario-region", index % 4 + 1) for index, name in enumerate(names[:15])],
            [WorldsTeam(name, "scenario-region") for name in names[15:]],
            play_in_winner_pool=4, n_simulations=1,
        )
    else:
        simulator = EncSimulator.from_configuration({
            "simulation_ready": True,
            "teams": [
                {"nation": f"Nation {index}", "entry_stage": "group_stage" if index < 8 else "play_in",
                 "organization_id": f"national:{index}", "roster_ids": [f"player:{index}:{slot}" for slot in range(5)]}
                for index in range(32)
            ],
        })
        result = simulator.simulate(1)

    assert result["simulations"] == 1
    provenance = result.get("provenance", result)
    assert provenance["native_prediction_provenance"]
    for pair in provenance["native_prediction_provenance"]:
        diagnostic = pair["diagnostics"]
        assert diagnostic["feature_provenance"]["dataset_freshness"]["raw_history_max_effective_release_day"] == "2026-09-08"
        assert diagnostic["feature_provenance"]["raw_history_sha256"] == "retained-source-hash"
        assert diagnostic["c0_request"] in requests
        assert pair["roster_provenance"]["authenticated_prior_announcement"] is False
        assert pair["roster_provenance"]["published_at"] is None
        assert set(pair["roster_provenance"]["basis"].values()) == {
            "configured_role_selection" if kind == "enc" else "explicit_scenario_roster"
        }
    assert provenance["point_in_time_certified"] is False
