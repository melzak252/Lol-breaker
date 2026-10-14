"""Source admission and edition-specific mechanics for the native research cohort.

Small synthetic identities exercise real tournament rollouts. Historical MSI
pair/winner oracles below are transcribed from the archived stopped series, not
obtained from the cohort builder or used to construct future bracket supplies.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json

import pytest

from src.models.tournament_formats import required_best_ofs, simulate_tournament
from src.models.tournament_research_cohort import (
    build_observed_labels,
    build_research_event_spec,
    validate_research_event,
    _opening_slots,
    _knockout_slots,
    _lookup_roster,
)

from src.models.tournament_research_cohort import _read_raw_cohort, _raw_stopped_scores


@pytest.fixture
def source_contract(tmp_path):
    def make(edition_id, *, known_opening=False):
        year = int(edition_id.rsplit("-", 1)[1])
        contract = {
            "edition_id": edition_id,
            "authenticated_prestart": False,
            "draw_basis": (
                "retrospective_known_opening_draw_scenario" if known_opening
                else "retrospective_randomized_scenario"
            ),
            "opening_seed_basis": "prior_source",
        }
        if edition_id.startswith("worlds-"):
            contract["swiss"] = {
                "no_rematches": year >= 2024,
                "wins_to_advance": 3,
                "losses_to_eliminate": 3,
            }
            if year >= 2024:
                contract["swiss"]["rematch_fallback"] = "waive_only_if_no_feasible_draw"
            contract["knockout"] = {
                "three_zero_opposite_halves": True,
                "three_zero_vs_three_two": True,
            }
        else:
            contract["msi"] = {
                "upper_round_two_loser_routing": "cross_half",
                "final_reset": "single_final",
            }
        # Local rule evidence is a test fixture, not a claimed historical
        # publication certificate. No repository archive or database is loaded.
        evidence = tmp_path / f"{edition_id}-rules.json"
        payload = json.dumps(contract, sort_keys=True).encode()
        evidence.write_bytes(payload)
        contract["source_path"] = str(evidence)
        contract["source_sha256"] = {"rules": hashlib.sha256(payload).hexdigest()}
        return contract

    return make


def _probabilities(spec, p=0.5):
    return {
        (a, b, best_of): p
        for a, b in combinations(spec["teams"], 2)
        for best_of in required_best_ofs(spec)
    }


def _event(spec, contract):
    edition = contract["edition_id"]
    year = int(edition.rsplit("-", 1)[1])
    cutoff = f"{year}-10-02T00:00:00Z"
    teams = spec["teams"]
    contract["roster_context_bindings"] = {
        "Worlds Main Event " + str(year): [{"wiki_title": f"{year} Season World Championship/Main Event"}]
    }
    return {
        "id": edition,
        "edition_id": edition,
        "year": year,
        "cutoff": cutoff,
        "event_title": f"{year} Season World Championship/Main Event",
        "scope": "retrospective_conditional_scenario",
        "teams": teams,
        "rosters": {team: [f"{team}-player-{i}" for i in range(5)] for team in teams},
        "roster_basis": "first_event_actual_roster",
        "roster_evidence": {
            team: {
                "basis": "first_event_actual_roster",
                "source_day": f"{year}-10-03",
                "source_match_id": f"opening-{team}",
                "source_sha256": contract["source_sha256"]["rules"],
                "authenticated_prestart": False,
                "context_basis": "first_event_actual_roster",
                "source_context": f"Worlds Main Event {year}",
            }
            for team in teams
        },
        "spec": spec,
        "labels_path": "separate-observed-labels.json",
        "model_eligibility": {},
        "source_provenance": contract,
    }


def test_missing_edition_rule_source_cannot_become_a_generic_seeded_bracket():
    with pytest.raises(ValueError):
        build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)])


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_contradictory_edition_rematch_rule_is_rejected(source_contract, edition):
    contract = source_contract(edition)
    contract["swiss"]["no_rematches"] = not contract["swiss"]["no_rematches"]
    with pytest.raises(ValueError):
        build_research_event_spec(edition, [f"T{i}" for i in range(16)], source_contract=contract)


def test_future_only_seed_order_is_rejected_even_for_a_retrospective_scenario(source_contract):
    contract = source_contract("worlds-2024")
    contract["opening_seed_basis"] = "future_only"
    with pytest.raises(ValueError):
        build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=contract)


def test_source_fixed_draw_cannot_claim_prestart_authentication_without_publication(source_contract):
    contract = source_contract("worlds-2024")
    teams = [f"T{i}" for i in range(16)]
    spec = build_research_event_spec("worlds-2024", teams, source_contract=contract)
    event = _event(spec, contract)
    event["roster_basis"] = "strictly_prior_observed_same_context"
    for evidence in event["roster_evidence"].values():
        evidence.update({"basis": event["roster_basis"], "context_basis": event["roster_basis"],
                         "source_day": "2024-09-30", "authenticated_prestart": True})
    event["source_provenance"]["draw_basis"] = "source_fixed"
    event["source_provenance"]["authenticated_prestart"] = True
    with pytest.raises(ValueError):
        validate_research_event(event)


def test_source_fixed_draw_publication_at_cutoff_is_not_strictly_prior(source_contract):
    contract = source_contract("worlds-2024")
    spec = build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=contract)
    event = _event(spec, contract)
    event["roster_basis"] = "strictly_prior_observed_same_context"
    for evidence in event["roster_evidence"].values():
        evidence.update({"basis": event["roster_basis"], "context_basis": event["roster_basis"],
                         "source_day": "2024-09-30", "authenticated_prestart": True})
    event["source_provenance"].update({
        "draw_basis": "source_fixed",
        "authenticated_prestart": True,
        "published_at": event["cutoff"],
        "available_at": event["cutoff"],
    })
    with pytest.raises(ValueError):
        validate_research_event(event)


@pytest.mark.parametrize("claim_location", ["roster", "event"])
def test_first_event_actual_roster_cannot_be_authenticated_as_prestart(source_contract, claim_location):
    contract = source_contract("worlds-2024")
    spec = build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=contract)
    event = _event(spec, contract)
    if claim_location == "roster":
        event["roster_evidence"][event["teams"][0]]["authenticated_prestart"] = True
    else:
        event["source_provenance"]["authenticated_prestart"] = True
    with pytest.raises(ValueError):
        validate_research_event(event)


@pytest.mark.parametrize("invalid_roster", [["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3", "p3"]])
def test_native_event_requires_five_distinct_players_per_entrant(source_contract, invalid_roster):
    contract = source_contract("worlds-2024")
    spec = build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=contract)
    event = _event(spec, contract)
    event["rosters"][event["teams"][0]] = invalid_roster
    with pytest.raises(ValueError):
        validate_research_event(event)


def test_declared_actual_roster_scenario_is_admitted_without_false_authentication(source_contract):
    contract = source_contract("worlds-2024")
    spec = build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=contract)
    event = _event(spec, contract)
    validate_research_event(event)
    result = simulate_tournament(spec, _probabilities(spec), simulations=4, seed=17)
    assert sum(result["champion_prob"].values()) == pytest.approx(1)
    assert sum(result["final_prob"].values()) == pytest.approx(2)
    assert sum(result["advance_prob"]["swiss"].values()) == pytest.approx(8)


def test_worlds_2023_allows_record_bracket_rematches_but_2024_avoids_them_when_feasible(source_contract):
    repeats_by_edition = {}
    for edition in ("worlds-2023", "worlds-2024"):
        spec = build_research_event_spec(edition, [f"T{i}" for i in range(16)], source_contract=source_contract(edition))
        repeats = 0
        for seed in range(32):
            result = simulate_tournament(spec, _probabilities(spec), simulations=1, seed=seed)
            played = set()
            for match in result["trace"]:
                if match["stage"] != "swiss":
                    continue
                pair = frozenset((match["team_a"], match["team_b"]))
                repeats += pair in played
                played.add(pair)
        repeats_by_edition[edition] = repeats
    # A stricter shared generic profile would incorrectly make both counts zero.
    # The 2024 source contract retains officials' infeasible-draw waiver; these
    # small seeded rollouts exercise normal feasible draws, not that exception.
    assert repeats_by_edition["worlds-2023"] > 0
    assert repeats_by_edition["worlds-2024"] == 0


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_worlds_quarterfinal_draw_puts_three_zero_teams_in_opposite_halves_against_three_two(source_contract, edition):
    spec = build_research_event_spec(edition, [f"T{i}" for i in range(16)], source_contract=source_contract(edition))
    for seed in range(16):
        result = simulate_tournament(spec, _probabilities(spec), simulations=1, seed=seed)
        records = result["stage_records"]["swiss"]
        unbeaten = {team for team, record in records.items() if (record["wins"], record["losses"]) == (3, 0)}
        quarterfinals = [match for match in result["trace"] if match["stage"] != "swiss" and match["round"] == 1]
        assert len(unbeaten) == 2
        assert len(quarterfinals) == 4
        halves = set()
        for index, match in enumerate(quarterfinals):
            pair = {match["team_a"], match["team_b"]}
            if pair & unbeaten:
                assert len(pair & unbeaten) == 1
                opponent = next(iter(pair - unbeaten))
                assert (records[opponent]["wins"], records[opponent]["losses"]) == (3, 2)
                halves.add(index // 2)
        assert halves == {0, 1}


def test_official_quarterfinal_losers_keep_the_same_fifth_to_eighth_band(source_contract):
    spec = build_research_event_spec("worlds-2024", [f"T{i}" for i in range(16)], source_contract=source_contract("worlds-2024"))
    result = simulate_tournament(spec, _probabilities(spec), simulations=1, seed=9,
                                 score_distributions={3: [1, 0], 5: [1, 0, 0]})
    matches = [{
        "team_a": match["team_a"], "team_b": match["team_b"],
        "best_of": match["best_of"], "wins_a": match["score_a"], "wins_b": match["score_b"],
        "date": f"2024-10-{3 + index // 8:02d}", "match_id": f"fixture-{index}",
        "stage": match["stage"], "round": match["round"],
    } for index, match in enumerate(result["trace"])]
    labels = build_observed_labels("worlds-2024", "worlds-2024", matches, spec=spec)
    qf_losers = {match["loser"] for match in result["trace"] if match["stage"] != "swiss" and match["round"] == 1}
    semifinal_losers = {match["loser"] for match in result["trace"] if match["stage"] != "swiss" and match["round"] == 2}
    assert len(qf_losers) == 4
    assert {tuple(labels["placement_bands"][team]) for team in qf_losers} == {(5, 8)}
    assert {tuple(labels["placement_bands"][team]) for team in semifinal_losers} == {(3, 4)}
    final = next(match for match in result["trace"] if match["stage"] != "swiss" and match["round"] == 3)
    assert labels["placement_bands"][final["winner"]] == [1, 1]
    assert labels["placement_bands"][final["loser"]] == [2, 2]
    band = next(band for band in spec["placement_bands"] if {ref["rank"] for ref in band["placements"]} == {5, 6, 7, 8})
    for team in qf_losers:
        assert result["placement_prob"][team][band["label"]] == 1


# Source: leaguepedia-tournament-rules-20260908/series_results.csv,
# MSI Bracket Stage stopped series. Opening pairs use bracket SLOT order, not
# chronology: 2024's source template swaps R1M1/2 and R1M3/4.
_MSI_ORACLES = {
    "msi-2023": {
        "opening": [("GEN", "G2"), ("MAD", "T1"), ("C9", "BLG"), ("JDG", "GG")],
        "winners": [("GEN", "G2"), ("T1", "MAD"), ("BLG", "C9"), ("JDG", "GG"),
                    ("G2", "MAD"), ("C9", "GG"), ("T1", "GEN"), ("JDG", "BLG"),
                    ("BLG", "G2"), ("GEN", "C9"), ("BLG", "GEN"), ("JDG", "T1"),
                    ("BLG", "T1"), ("JDG", "BLG")],
        "champion": "JDG",
    },
    "msi-2024": {
        "opening": [("GEN", "FNC"), ("TL", "TES"), ("G2", "T1"), ("BLG", "PSG")],
        "winners": [("TES", "TL"), ("GEN", "FNC"), ("BLG", "PSG"), ("T1", "G2"),
                    ("TL", "FNC"), ("G2", "PSG"), ("GEN", "TES"), ("BLG", "T1"),
                    ("G2", "TES"), ("T1", "TL"), ("T1", "G2"), ("GEN", "BLG"),
                    ("BLG", "T1"), ("GEN", "BLG")],
        "champion": "GEN",
    },
}


@pytest.mark.parametrize("edition", ["msi-2023", "msi-2024"])
def test_msi_edition_routes_upper_losers_to_the_source_bound_opposite_lower_half(source_contract, edition):
    oracle = _MSI_ORACLES[edition]
    teams = [team for pair in oracle["opening"] for team in pair]
    spec = build_research_event_spec(edition, teams, initial_pairings=oracle["opening"],
                                     source_contract=source_contract(edition, known_opening=True))
    winners = {frozenset((winner, loser)): winner for winner, loser in oracle["winners"]}
    probabilities = {(a, b, 5): float(winners.get(frozenset((a, b)), a) == a)
                     for a, b in combinations(teams, 2)}
    result = simulate_tournament(spec, probabilities, simulations=1, seed=11)
    actual = [(match["winner"], match["loser"]) for match in result["trace"]]
    assert sorted(actual) == sorted(oracle["winners"])
    assert result["champion_prob"][oracle["champion"]] == 1
    assert result["expected_series"] == 14


@pytest.mark.parametrize("edition", ["msi-2023", "msi-2024"])
def test_msi_single_final_ends_after_one_series_even_when_lower_bracket_winner_wins(source_contract, edition):
    opening = _MSI_ORACLES[edition]["opening"]
    teams = [team for pair in opening for team in pair]
    spec = build_research_event_spec(edition, teams, initial_pairings=opening,
                                     source_contract=source_contract(edition, known_opening=True))
    lower_wins = 0
    for seed in range(24):
        result = simulate_tournament(spec, _probabilities(spec), simulations=1, seed=seed)
        trace = result["trace"]
        assert result["expected_series"] == 14
        final = trace[-1]
        prior_losses = {team: sum(match["loser"] == team for match in trace[:-1]) for team in teams}
        assert sorted(prior_losses[team] for team in (final["team_a"], final["team_b"])) == [0, 1]
        if prior_losses[final["winner"]] == 1:
            lower_wins += 1
            assert prior_losses[final["loser"]] == 0
            assert result["champion_prob"][final["winner"]] == 1
            assert result["final_prob"][final["loser"]] == 1
    assert lower_wins > 0


@pytest.mark.parametrize("override, expected", [
    ("", ["Quarterfinals_1", "Quarterfinals_2", "Quarterfinals_3", "Quarterfinals_4"]),
    ("|R1M1_cargomatch=Quarterfinals_2|R1M2_cargomatch=Quarterfinals_1",
     ["Quarterfinals_2", "Quarterfinals_1", "Quarterfinals_3", "Quarterfinals_4"]),
])
def test_worlds_inline_bracket_uses_round_defaults_and_explicit_cargo_overrides(override, expected):
    text = ("{{Bracket|8SE|cargo=Yes|R1_title=Quarterfinals|R2_title=Semifinals"
            + override + "|R3_title=Finals}}{{MatchDetails|R1M3_cargomatch=unrelated\n}}")
    assert _opening_slots(text, "worlds") == expected
    assert _knockout_slots(text) == {
        **{f"R1M{i}": key for i, key in enumerate(expected, 1)},
        "R2M1": "Semifinals_1", "R2M2": "Semifinals_2", "R3M1": "Finals_1",
    }


def _roster_match(match_id, day, context, prefix):
    return {"match_id": match_id, "date": day, "tournament_name": context, "games": [{
        "t1_id": "team", "t2_id": "other",
        "t1_players": {role: {"player_id": f"{prefix}-{i}"}
                       for i, role in enumerate(("TOP", "JUNGLE", "MID", "ADC", "SUPPORT"))},
    }]}


@pytest.mark.parametrize("playin", [False, True])
def test_roster_lookup_rejects_cup_and_domestic_but_retains_bound_edition_playin(playin):
    import pandas as pd
    columns = ["team1_id", "team2_id", "date", "result_day", "tournament_name",
               "golgg_match_id", "roster_a", "roster_b"]
    reference = pd.DataFrame([
        ["team", "other", "2024-10-01", "2024-10-01", "KeSPA Cup 2024", "90",
         [f"cup-{i}" for i in range(5)], [f"other-{i}" for i in range(5)]],
    ], columns=columns)
    raw = [_roster_match("91", "2024-10-01", "LCK Summer 2024", "domestic"),
           _roster_match("100", "2024-10-03", "Worlds Main Event 2024", "actual")]
    if playin:
        raw.append(_roster_match("80", "2024-09-29", "Worlds Play-In 2024", "playin"))
    roster, evidence = _lookup_roster(
        "team", "2024-10-02T00:00:00Z", reference, raw,
        {"Worlds Main Event 2024", "Worlds Play-In 2024"}, {"100"},
        "reference-hash", "raw-hash")
    prefix = "playin" if playin else "actual"
    assert roster == [f"{prefix}-{i}" for i in range(5)]
    assert evidence["basis"] == ("strictly_prior_observed_same_context" if playin
                                 else "first_event_actual_roster")
    assert evidence["source_context"] == ("Worlds Play-In 2024" if playin
                                          else "Worlds Main Event 2024")
    assert evidence["source_match_id"] == ("80" if playin else "100")


def test_worlds_cargo_override_cannot_duplicate_another_default_node():
    text = ("{{Bracket|8SE|cargo=Yes|R1_title=Quarterfinals|R2_title=Semifinals"
            "|R3_title=Finals|R1M1_cargomatch=Quarterfinals_3}}")
    with pytest.raises(ValueError, match="unique"):
        _knockout_slots(text)


def test_missing_mapped_first_roster_cannot_be_replaced_by_later_event_roster():
    import pandas as pd
    reference = pd.DataFrame(columns=[
        "team1_id", "team2_id", "date", "result_day", "tournament_name",
        "golgg_match_id", "roster_a", "roster_b"])
    raw = [_roster_match("101", "2024-10-04", "Worlds Main Event 2024", "later")]
    with pytest.raises(ValueError, match="NATIVE_FIRST_EVENT_ROSTER_ABSENT: team"):
        _lookup_roster("team", "2024-10-02T00:00:00Z", reference, raw,
                       {"Worlds Main Event 2024"}, {"100"}, "reference-hash", "raw-hash")


@pytest.mark.parametrize("release_day, expected_prefix", [
    ("2024-10-01", "prior"),
    ("2024-10-02", "actual"),
])
def test_streamed_roster_requires_all_maps_released_before_cutoff(tmp_path, release_day, expected_prefix):
    import pandas as pd
    reference = pd.DataFrame(columns=[
        "team1_id", "team2_id", "date", "result_day", "tournament_name",
        "golgg_match_id", "roster_a", "roster_b"])
    prior = _roster_match(99, "2024-10-01", "Worlds Main Event 2024", "prior")
    actual = _roster_match(100, "2024-10-03", "Worlds Main Event 2024", "actual")
    prior["games"][0].update(date=release_day, t1_win=True, t2_win=False)
    actual["games"][0].update(t1_win=False, t2_win=True, t1_name="Team")
    actual["games"].append({"t1_id": "team", "t2_id": "other",
                            "t1_win": True, "t2_win": False})
    actual["games"][0]["t1_players"]["TOP"]["stats"] = {"unused": ["},\\\"", 3.5]}
    path = tmp_path / "matches.json"
    path.write_text(json.dumps([prior, actual]))
    raw = _read_raw_cohort(path)
    roster, evidence = _lookup_roster(
        "team", "2024-10-02T00:00:00Z", reference, raw,
        {"Worlds Main Event 2024"}, {"100"}, "reference-hash", "complete-source-hash")
    assert roster == [f"{expected_prefix}-{i}" for i in range(5)]
    assert evidence["source_match_id"] == ("99" if expected_prefix == "prior" else "100")
    assert evidence["source_sha256"] == "complete-source-hash"
    assert dict(_raw_stopped_scores(raw[1])) == {"team": 1, "other": 1}


@pytest.mark.parametrize("problem", [
    "absent_first", "missing_player", "duplicate_player", "null_players",
    "list_players", "null_player", "list_player",
])
@pytest.mark.parametrize("side", ["t1", "t2"])
def test_streamed_first_actual_roster_does_not_repair_missing_or_malformed_data(tmp_path, problem, side):
    import pandas as pd
    reference = pd.DataFrame(columns=[
        "team1_id", "team2_id", "date", "result_day", "tournament_name",
        "golgg_match_id", "roster_a", "roster_b"])
    first = _roster_match("100", "2024-10-03", "Worlds Main Event 2024", "actual")
    later = _roster_match("101", "2024-10-04", "Worlds Main Event 2024", "later")
    if problem == "missing_player":
        del first["games"][0]["t1_players"]["TOP"]["player_id"]
    elif problem == "duplicate_player":
        first["games"][0]["t1_players"]["TOP"]["player_id"] = "actual-1"
    elif problem == "null_players":
        first["games"][0]["t1_players"] = None
    elif problem == "list_players":
        first["games"][0]["t1_players"] = []
    elif problem == "null_player":
        first["games"][0]["t1_players"]["TOP"] = None
    elif problem == "list_player":
        first["games"][0]["t1_players"]["TOP"] = []
    if side == "t2":
        for match in (first, later):
            game = match["games"][0]
            game["t1_id"], game["t2_id"] = game["t2_id"], game["t1_id"]
            game["t2_players"] = game.pop("t1_players")
    path = tmp_path / "matches.json"
    path.write_text(json.dumps([later] if problem == "absent_first" else [first, later]))
    raw = _read_raw_cohort(path)
    if problem == "absent_first":
        message = "NATIVE_FIRST_EVENT_ROSTER_ABSENT"
    elif problem in {"null_players", "list_players"}:
        message = "raw first-map roster players must be an object"
    elif problem in {"null_player", "list_player"}:
        message = "raw first-map roster player entries must be objects"
    else:
        message = "five unique"
    with pytest.raises(ValueError, match=message):
        _lookup_roster("team", "2024-10-02T00:00:00Z", reference, raw,
                       {"Worlds Main Event 2024"}, {"100"}, "reference-hash", "raw-hash")


@pytest.mark.parametrize("winner", [None, 1, "true"])
def test_streamed_map_winner_requires_actual_boolean(tmp_path, winner):
    match = _roster_match("100", "2024-10-03", "Worlds Main Event 2024", "actual")
    match["games"][0]["t1_win"] = winner
    path = tmp_path / "matches.json"
    path.write_text(json.dumps([match]))
    with pytest.raises(ValueError, match="raw map winner unavailable: 100"):
        _raw_stopped_scores(_read_raw_cohort(path)[0])


def test_streamed_abbreviated_roles_preserve_actual_player_order_and_native_identity(tmp_path):
    import pandas as pd
    from src.models.tournament_research_cohort import _prior_identity
    reference = pd.DataFrame(columns=[
        "team1_id", "team2_id", "date", "result_day", "tournament_name",
        "golgg_match_id", "roster_a", "roster_b"])
    match = _roster_match(123, "2024-10-01", "Worlds Play-In 2024", "prior")
    match["games"][0].update(t1_id=77, t1_name="Équipe", t1_win=True, t2_win=False)
    match["games"][0]["t1_players"] = {
        role: {"player_id": player, "stats": {"ignored": 'x' * 70000 + '\\"},]'}}
        for role, player in zip(("top", "jng", "mid", "bot", "sup"), (11, 12, 13, 14, 15))
    }
    path = tmp_path / "matches.json"
    path.write_text(json.dumps([match], ensure_ascii=False))
    raw = _read_raw_cohort(path)
    native_id, proof = _prior_identity("Équipe", {}, raw, "2024-10-02T00:00:00Z", "raw-hash")
    assert native_id == "77"
    assert proof["source_match_id"] == "123"
    assert proof["source_day"] == "2024-10-01"
    roster, evidence = _lookup_roster(
        native_id, "2024-10-02T00:00:00Z", reference, raw,
        {"Worlds Play-In 2024"}, set(), "reference-hash", "raw-hash")
    assert roster == ["11", "12", "13", "14", "15"]
    assert evidence["basis"] == "strictly_prior_observed_same_context"
    assert dict(_raw_stopped_scores(raw[0])) == {"77": 1, "other": 0}
