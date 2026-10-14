"""Research protocol guards: chronology, exact identities, outcome-blind scenarios."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_tournament_uncertainty import (
    META, MODELS, _choose_prior, _heldout, _load, _local_pair_clique,
)


def _cohort():
    records = []
    for year in (2024, 2025, 2026):
        for month in (2, 5, 8, 11):
            for number, (first, second) in enumerate(combinations(["A", "B", "C", "D"], 2)):
                records.append({
                    "golgg_match_id": len(records), "date": pd.Timestamp(year, month, number + 1),
                    "team1_id": first, "team2_id": second, "best_of": 1,
                    "y_true": (number + month) % 2, "tournament": f"event-{year}-{month}",
                    **{model: 0.4 + number * 0.03 for model in MODELS},
                })
    return pd.DataFrame(records)


def test_scenario_latent_strength_does_not_reuse_outcome_uniforms(tmp_path):
    from scripts.experiment_tournament_uncertainty import _scenarios

    class SymmetricLogisticStrength:
        """Analytic zero-mean latent fixture, with every pair marginal equal to .5."""

        def predict(self, first, second, probability, integrate=True):
            return np.full(len(first), 0.5)

        def sample(self, teams, simulations, seed):
            probability = np.random.default_rng(seed).uniform(size=simulations)
            draws = {team: np.zeros(simulations) for team in teams}
            draws[teams[0]] = np.log(probability / (1 - probability))
            return draws

    frame = _cohort()
    for model in MODELS:
        frame[model] = 0.5
    fits = {(model, 2025): SymmetricLogisticStrength() for model in MODELS}
    result = _scenarios(frame, {}, fits, tmp_path, simulations=20000, seed=82)
    for model in MODELS:
        forecast = result["scenarios"]["local_declared_single_elimination"][model]
        # A's advancement is .5 after integrating its symmetric strength,
        # regardless of how strengths were sampled. Reusing U to decide U < U
        # instead couples the latent state to its own match result.
        assert forecast["coherent_posterior"]["final_prob"]["A"] == pytest.approx(0.5, abs=0.015)


def test_hyperparameters_do_not_consume_2025_or_2026_outcomes(tmp_path):
    frame = _cohort()
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    before = _choose_prior(frame, MODELS[0], before_dir)
    frame.loc[frame.date >= "2025-01-01", "y_true"] = 1 - frame.loc[frame.date >= "2025-01-01", "y_true"]
    after = _choose_prior(frame, MODELS[0], after_dir)
    assert before == after


def test_annual_posterior_never_consumes_same_year_outcomes(tmp_path):
    frame = _cohort()
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    priors = dict.fromkeys(MODELS, 0.7)
    before, _ = _heldout(frame, priors, before_dir)
    frame.loc[frame.date >= "2025-01-01", "y_true"] = 1
    after, _ = _heldout(frame, priors, after_dir)
    for model in MODELS:
        column = model + "__integrated"
        np.testing.assert_array_equal(before.loc[before.date.dt.year == 2025, column],
                                      after.loc[after.date.dt.year == 2025, column])
        assert not np.allclose(before.loc[before.date.dt.year == 2026, column],
                               after.loc[after.date.dt.year == 2026, column])


def test_monthly_posterior_updates_only_after_outcome_month(tmp_path):
    frame = _cohort()
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    priors = dict.fromkeys(MODELS, 0.7)
    before, _ = _heldout(frame, priors, before_dir, cadence="monthly")
    frame.loc[frame.date >= "2025-05-01", "y_true"] = 1
    after, _ = _heldout(frame, priors, after_dir, cadence="monthly")
    for model in MODELS:
        column = model + "__integrated"
        unaffected = before.date < "2025-06-01"
        np.testing.assert_array_equal(before.loc[unaffected, column], after.loc[unaffected, column])
        later = before.date >= "2025-08-01"
        assert not np.allclose(before.loc[later, column], after.loc[later, column])


def test_local_graph_selection_is_outcome_blind_and_uses_only_earlier_pairs():
    frame = _cohort()
    teams, best_of, rows = _local_pair_clique(frame)
    frame["y_true"] = 1 - frame.y_true
    changed_teams, changed_format, changed_rows = _local_pair_clique(frame)
    assert teams == changed_teams and best_of == changed_format
    assert rows.golgg_match_id.tolist() == changed_rows.golgg_match_id.tolist()
    assert (rows.date < "2025-01-01").all()
    assert {tuple(sorted((row.team1_id, row.team2_id))) for row in rows.itertuples()} == set(combinations(teams, 2))


@pytest.mark.parametrize("fault", ["team", "outcome", "date", "duplicate", "missing"])
def test_loader_rejects_inexact_shared_snapshot_metadata(tmp_path, fault):
    frame = _cohort()
    source = frame[META + ["tournament"]].copy()
    if fault == "team": source.loc[0, "team1_id"] = "wrong"
    elif fault == "outcome": source.loc[0, "y_true"] = 1 - source.loc[0, "y_true"]
    elif fault == "date": source.loc[0, "date"] += pd.Timedelta(days=1)
    elif fault == "duplicate": source = pd.concat([source, source.iloc[:1]])
    elif fault == "missing": source = source.iloc[1:]
    frame[META + list(MODELS)].to_csv(tmp_path / "oof.csv", index=False)
    source.to_csv(tmp_path / "snapshots.csv", index=False)
    with pytest.raises(ValueError):
        _load(tmp_path / "oof.csv", tmp_path / "snapshots.csv")


def test_runner_refuses_to_replace_an_existing_artifact_directory(tmp_path):
    from scripts.experiment_tournament_uncertainty import run

    sentinel = tmp_path / "existing-artifact"
    sentinel.write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert sentinel.read_bytes() == b"immutable"
