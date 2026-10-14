"""Executable recipes remain counterfactual; historical gaps must fail closed."""
from itertools import combinations
import json

from src.models.tournament_catalog import DEFAULT_CATALOG, FAMILIES

import pytest

from src.models.tournament_catalog import audit_catalog, instantiate_profile, list_profiles
from src.models.tournament_formats import simulate_tournament


@pytest.fixture
def source_bundle(tmp_path):
    catalog = json.loads(DEFAULT_CATALOG.read_text())
    profile = next(row for row in catalog["profiles"] if row["id"] == "lck-road-to-msi")
    phase = next(row for row in catalog["phase_rules"] if row["phase_id"] == profile["source_phase_ids"][0])
    (tmp_path / "events.jsonl").write_text(json.dumps(phase) + "\n")
    for name in ("bracket_sources.jsonl", "standings.jsonl"):
        (tmp_path / name).write_text("")
    research = tmp_path / "research"
    research.mkdir()
    for family in FAMILIES:
        (research / f"{family}.json").write_text("{}")
    return tmp_path, phase


def test_changed_source_revision_revokes_the_reviewed_profile_link(source_bundle):
    directory, phase = source_bundle
    before = audit_catalog(directory)["phases"][0]
    assert before["configured"]
    assert "lck-road-to-msi" in before["profile_ids"]
    phase["source"]["revid"] += 1
    (directory / "events.jsonl").write_text(json.dumps(phase) + "\n")
    after = audit_catalog(directory)["phases"][0]
    assert after["status"] == "source_revision_changed"
    assert not after["configured"]
    assert after["profile_ids"] == []
    assert not after["historically_ready"]


def test_unknown_and_duplicate_phases_cannot_inherit_reviewed_rules(source_bundle):
    directory, phase = source_bundle
    phase["phase_id"] = "unreviewed-phase"
    encoded = json.dumps(phase) + "\n"
    (directory / "events.jsonl").write_text(encoded)
    result = audit_catalog(directory)["phases"][0]
    assert result["status"] == "unreviewed_phase"
    assert result["profile_ids"] == []
    (directory / "events.jsonl").write_text(encoded * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        audit_catalog(directory)


def test_profile_instantiation_rejects_wrong_slots_and_does_not_mutate_recipe():
    with pytest.raises(ValueError, match="team"):
        instantiate_profile("lec-2023-season-gsl-playoffs", ["A"])
    with pytest.raises(ValueError, match="unique"):
        instantiate_profile("lec-2023-season-gsl-playoffs", ["A"] * 10)
    first = instantiate_profile("lec-2023-season-gsl-playoffs", [f"t{i}" for i in range(10)])
    first["stages"].clear()
    second = instantiate_profile("lec-2023-season-gsl-playoffs", [f"u{i}" for i in range(10)])
    probabilities = {(a, b, bo): 0.0 for a, b in combinations(second["teams"], 2) for bo in (1, 3, 5)}
    result = simulate_tournament(second, probabilities, simulations=1, seed=81)
    assert result["champion_prob"]["u9"] == 1


@pytest.mark.parametrize("profile", list_profiles(), ids=lambda profile: profile["id"])
def test_every_profile_runs_without_observed_results_or_unresolved_slots(profile):
    teams = [f"synthetic-{i}" for i in range(profile["team_count"])]
    spec = instantiate_profile(profile["id"], teams)
    probabilities = {(a, b, bo): 0.5 for a, b in combinations(teams, 2) for bo in (1, 3, 5)}
    result = simulate_tournament(spec, probabilities, simulations=2, seed=81,
                                 score_distributions={3: [0.5, 0.5], 5: [0.25, 0.5, 0.25]})
    if spec.get("champion"):
        assert sum(result["champion_prob"].values()) == pytest.approx(1)
    for ranks in result["stage_rank_prob"].values():
        for group in ranks.values():
            for position in range(len(next(iter(group.values())))):
                assert sum(row[position] for row in group.values()) == pytest.approx(1)


def test_lec_regular_season_winners_dynamically_fill_gsl_and_partial_double_elimination():
    teams = [f"team-{i}" for i in range(10)]
    spec = instantiate_profile("lec-2023-season-gsl-playoffs", teams)
    probabilities = {(a, b, bo): 0.0 for a, b in combinations(teams, 2) for bo in (1, 3, 5)}
    result = simulate_tournament(spec, probabilities, simulations=3, seed=81)
    assert result["champion_prob"]["team-9"] == 1
    assert result["advance_prob"]["regular"]["team-0"] == 0
    assert result["advance_prob"]["regular"]["team-1"] == 0
    assert result["advance_prob"]["regular"]["team-9"] == 1
