"""Dependence, score support and chronological mechanism boundaries."""
from itertools import combinations

import numpy as np
import pytest

from src.models.tournament_formats import simulate_tournament


def cup(bo=1):
    return {'version': 1, 'id': 'cup', 'teams': list('ABCD'), 'stages': [
        {'id': 'event', 'kind': 'single_elimination', 'entrants': list('ABCD'), 'best_of': bo}],
        'champion': {'stage': 'event', 'group': 'all', 'rank': 1}}


def pairs(spec, p=.5):
    return {(a, b, spec['stages'][0]['best_of']): p for a, b in combinations(spec['teams'], 2)}


def test_zero_strength_preserves_random_stream_and_complete_outputs():
    spec = cup(3)
    kwargs = dict(simulations=100, seed=34, score_distributions={3: [.4, .6]})
    fixed = simulate_tournament(spec, pairs(spec, .7), **kwargs)
    assert simulate_tournament(spec, pairs(spec, .7), team_strength_draws={
        team: np.zeros(100) for team in spec['teams']}, **kwargs) == fixed


def test_copula_uses_the_authoritative_rare_tail_when_reverse_p_rounds_to_one():
    from scipy.special import ndtri
    spec = cup()
    spec["teams"] = spec["stages"][0]["entrants"] = ["B", "A"]
    n = 12000
    result = simulate_tournament(
        spec, {("A", "B", 1): 1e-17}, simulations=n, seed=17, copula_rho=.5,
        team_strength_draws={"A": np.full(n, -2 * ndtri(1e-17)), "B": np.zeros(n)},
    )
    assert result["champion_prob"]["B"] == pytest.approx(.5, abs=.015)


def test_copula_preserves_pair_orientation_near_a_machine_certainty():
    from src.models.tournament_bracket_simulator import simulate_tournament_bracket
    probabilities = {("A", "B", 1): 1e-12}
    draws = {"A": [10 * np.sqrt(2)], "B": [0.]}
    values = []
    for teams in (["A", "B"], ["B", "A"]):
        spec = cup()
        spec["teams"] = spec["stages"][0]["entrants"] = teams
        result = simulate_tournament_bracket(
            spec, probabilities, simulations=1, copula_rho=.5,
            team_strength_draws=draws,
        )
        law = result["pair_scoreline_laws"]["rows"][0]
        values.append(law["p"] if law["team_a"] == teams[0] else 1 - law["p"])
    assert sum(values) == pytest.approx(1., abs=2e-13)


def test_copula_zero_team_contrast_retains_conditional_variance_contraction():
    from statistics import NormalDist
    spec = cup()
    spec["teams"] = spec["stages"][0]["entrants"] = ["A", "B"]
    n = 12000
    result = simulate_tournament(
        spec, pairs(spec, .8), simulations=n, seed=17, copula_rho=.5,
        team_strength_draws={team: np.zeros(n) for team in spec["teams"]},
    )
    normal = NormalDist()
    expected = normal.cdf(normal.inv_cdf(.8) / np.sqrt(.5))
    assert result["champion_prob"]["A"] == pytest.approx(expected, abs=.012)


@pytest.mark.parametrize("rho", [None, 0., .5])
@pytest.mark.parametrize("team_order", [["A", "B"], ["B", "A"]])
def test_reverse_fixture_preserves_original_winner_bin_including_zero_copula(rho, team_order):
    from src.models.tournament_bracket_simulator import simulate_tournament_bracket
    spec = cup(3)
    spec["teams"] = team_order
    spec["stages"][0]["entrants"] = ["B", "A"]
    kwargs = {} if rho is None else {
        "copula_rho": rho,
        "team_strength_draws": {"A": np.full(4, 20.), "B": np.full(4, -20.)}}
    result = simulate_tournament_bracket(
        spec, {("A", "B", 3): .2}, simulations=4, seed=0, **kwargs,
        conditional_score_distributions={
            "probability_bins": [0, .2, 1], "distributions": {3: [[0, 1], [1, 0]]}},
    )
    assert [(row["score_a"], row["score_b"]) for row in result["trace"]] == [(0, 2)]
    assert result["pair_scoreline_laws"]["rows"][0]["scoreline_pmf"]["2-0"] == pytest.approx(
        1 if rho == .5 else .2)


def test_copula_score_bins_use_origin_probability_not_latent_conditioned_probability():
    from src.models.tournament_bracket_simulator import simulate_tournament_bracket
    spec = cup(3)
    spec["teams"] = spec["stages"][0]["entrants"] = ["A", "B"]
    result = simulate_tournament_bracket(
        spec, pairs(spec, .4), simulations=4, copula_rho=.5,
        team_strength_draws={"A": np.full(4, 20.), "B": np.full(4, -20.)},
        conditional_score_distributions={
            "probability_bins": [0, .5, 1], "distributions": {3: [[0, 1], [1, 0]]}},
    )
    assert result["champion_prob"] == {"A": 1, "B": 0}
    assert [(row["score_a"], row["score_b"]) for row in result["trace"]] == [(2, 1)]
    law = result["pair_scoreline_laws"]["rows"][0]["scoreline_pmf"]
    assert law["2-1"] == 1
    assert law["2-0"] == 0


def test_zero_copula_rho_preserves_results_despite_nonzero_latent_draws():
    spec = cup(3)
    kwargs = {"simulations": 100, "seed": 34, "score_distributions": {3: [.4, .6]}}
    fixed = simulate_tournament(spec, pairs(spec, .7), **kwargs)
    copula = simulate_tournament(
        spec, pairs(spec, .7), **kwargs, copula_rho=0,
        team_strength_draws={team: np.full(100, 20 * (-1)**i)
                             for i, team in enumerate(spec["teams"])},
    )
    assert copula["hits"] == fixed["hits"]
    assert copula["trace"] == fixed["trace"]


@pytest.mark.parametrize("extra", [
    {"integration": "exact"},
    {"integration": "hybrid"},
    {"state": {"version": 1, "cutoff": "2026-01-02T00:00:00Z", "completed": []}},
])
def test_positive_copula_rejects_unsupported_conditioning_and_integration(extra):
    from src.models.tournament_exact import UnsupportedIntegrationError
    with pytest.raises((ValueError, UnsupportedIntegrationError)):
        simulate_tournament(
            cup(), pairs(cup()), simulations=1, copula_rho=.2, **extra,
            team_strength_draws={team: [0.] for team in cup()["teams"]},
        )


def test_positive_copula_requires_latent_draws_instead_of_using_a_mean_state():
    with pytest.raises(ValueError):
        simulate_tournament(cup(), pairs(cup()), simulations=1, copula_rho=.2)


def test_coherent_strength_changes_repeated_appearance_not_first_match_marginal():
    spec = cup()
    n = 12000
    draws = {team: np.zeros(n) for team in spec['teams']}
    draws['A'] = np.tile([-20., 20.], n // 2)
    fixed = simulate_tournament(spec, pairs(spec), simulations=n, seed=17)
    coherent = simulate_tournament(spec, pairs(spec), simulations=n, seed=17, team_strength_draws=draws)
    # Every A-versus-opponent integrated marginal is exactly .5; surviving the
    # semifinal selects A's positive latent draw, so its final is not independent.
    assert coherent['final_prob']['A'] == pytest.approx(.5, abs=.001)
    assert fixed['champion_prob']['A'] == pytest.approx(.25, abs=.02)
    assert coherent['champion_prob']['A'] == pytest.approx(.5, abs=.001)


def test_extreme_finite_draws_preserve_absorbing_results_and_observed_prefix():
    spec = cup()
    prefix = simulate_tournament(spec, pairs(spec, 1), simulations=1)['trace'][:2]
    state = {'version': 1, 'cutoff': '2026-01-02T00:00:00Z', 'completed': [
        {**row, 'completed_at': '2026-01-01T12:00:00Z', 'available_at': '2026-01-01T12:00:00Z'}
        for row in prefix]}
    result = simulate_tournament(spec, pairs(spec, 0), simulations=4, state=state,
                                 team_strength_draws={team: np.full(4, (-1)**i * 1e308)
                                                      for i, team in enumerate(spec['teams'])})
    assert result['trace'][:2] == prefix
    assert result['final_prob']['C'] == result['final_prob']['D'] == 0
    assert result['future_expected_series'] == 1


def test_conditional_score_preserves_winner_marginal_and_uses_winner_probability():
    spec = cup(3)
    spec['teams'] = spec['stages'][0]['entrants'] = ['A', 'B']
    result = simulate_tournament(spec, pairs(spec, .8), simulations=12000,
        conditional_score_distributions={'probability_bins': [0, .5, 1],
                                         'distributions': {3: [[0, 1], [1, 0]]}},
        observables=[{'id': 'maps', 'kind': 'future_map_count', 'categories': ['2', '3']}])
    assert result['champion_prob']['A'] == pytest.approx(.8, abs=.015)
    assert result['observable_prob']['maps']['2'] == result['champion_prob']['A']


@pytest.mark.parametrize('value', [
    {'probability_bins': [0, .5, 1], 'distributions': {3: [[1, 0]]}},
    {'probability_bins': [.1, 1], 'distributions': {3: [[1, 0]]}},
    {'probability_bins': [0, 1], 'distributions': {3: [[1]]}},
    {'probability_bins': [0, 1], 'distributions': {3.0: [[1, 0]]}},
])
def test_missing_or_illegal_conditional_support_fails(value):
    with pytest.raises(ValueError):
        simulate_tournament(cup(3), pairs(cup(3)), conditional_score_distributions=value)


def test_conditional_fit_excludes_same_day_and_missing_bins_without_fallback():
    from scripts.experiment_tournament_distribution_mechanisms import fit_conditional_scores
    rows = [{'id': 'old', 'date': '2023-01-01', 'best_of': 3, 'score_a': 2, 'score_b': 0, 'p': .8},
            {'id': 'future', 'date': '2024-01-01', 'best_of': 3, 'score_a': 0, 'score_b': 2, 'p': .8}]
    with pytest.raises(ValueError, match='support'):
        fit_conditional_scores(rows, cutoff='2024-01-01', best_ofs=[3], bins=[0, .5, 1])
    model, provenance = fit_conditional_scores(rows, cutoff='2024-01-01', best_ofs=[3], bins=[0, 1])
    assert model['distributions'][3] == [[1., 0.]]
    assert provenance['used_series_ids'] == ['old']


def test_posterior_reconditions_real_daily_prefix_without_future_likelihood():
    from types import SimpleNamespace
    import pandas as pd
    from scripts.experiment_tournament_distribution_mechanisms import _posterior_at_origin

    frame = pd.DataFrame([
        {'golgg_match_id': str(i), 'date': day, 'team1_id': 'A', 'team2_id': 'B', 'y_true': y, 'p': .5}
        for i, (day, y) in enumerate([('2023-01-01', 0), ('2023-01-02', 1), ('2025-01-01', 0)])])
    args = SimpleNamespace(test_start='2024-01-01')
    event = {'tournament_id': 'cup', 'phase_id': 'final', 'edition_start_at': '2024-01-02T00:00:00Z'}
    origin = {'cutoff': '2024-01-04T00:00:00Z'}
    probabilities = {('A', 'B', 1): .5}
    old, _ = _posterior_at_origin(frame, 'p', 1., args, event, origin, None, probabilities)
    state = {'temporal_policy': 'daily_rollforward', 'completed': [
        {'team_a': 'A', 'team_b': 'B', 'best_of': 1, 'winner': 'A', 'source_date': '2024-01-03'}]}
    conditioned, _ = _posterior_at_origin(frame, 'p', 1., args, event, origin, state, probabilities)
    assert old.predict(['A'], ['B'], [.5])[0] == pytest.approx(.5)
    assert conditioned.predict(['A'], ['B'], [.5])[0] > .55
    assert '2' not in conditioned.provenance['fit_match_ids']
    state['completed'][0]['source_date'] = '2024-01-04'
    with pytest.raises(ValueError, match='strictly precede'):
        _posterior_at_origin(frame, 'p', 1., args, event, origin, state, probabilities)


def test_unknown_edition_boundary_cannot_be_substituted_by_phase_start():
    from types import SimpleNamespace
    import pandas as pd
    from scripts.experiment_tournament_distribution_mechanisms import _posterior_at_origin

    frame = pd.DataFrame([{'golgg_match_id': 'old', 'date': '2023-01-01'}])
    event = {'edition_start_at': None, 'phase_start_at': '2024-01-02T00:00:00Z'}
    with pytest.raises(ValueError, match='whole-edition boundary'):
        _posterior_at_origin(frame, 'p', 1., SimpleNamespace(test_start='2024-01-01'),
                             event, {'cutoff': '2024-01-02T00:00:00Z'}, None, {})


def test_source_join_preserves_canonical_orientation_and_exclusive_history_bound(tmp_path):
    import json
    import pandas as pd
    from betting_app.tests.test_siamese_research_dataset import match
    from scripts.experiment_tournament_distribution_mechanisms import _load_evidence

    source = tmp_path / 'matches.json'
    source.write_text(json.dumps([match(1, '2024-01-02')]))
    prediction = tmp_path / 'predictions.csv'
    frame = pd.DataFrame([{'golgg_match_id': '1', 'date': '2024-01-02', 'team1_id': 'b',
                           'team2_id': 'a', 'best_of': 1, 'y_true': 0, 'p': .2,
                           'history_last_source_day': '2024-01-01',
                           'feature_history_max_at': '2024-01-02T00:00:00Z'}])
    frame.to_csv(prediction, index=False)
    admitted, scores, exclusions = _load_evidence(prediction, source, 'p', '2025-01-01')
    assert admitted.golgg_match_id.tolist() == ['1']
    assert (scores[0]['score_a'], scores[0]['score_b']) == (0, 1)
    assert exclusions == []
    frame['history_last_source_day'] = '2024-01-02'
    frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match='strictly prior-day'):
        _load_evidence(prediction, source, 'p', '2025-01-01')


def test_frozen_ablation_survives_missing_current_table(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from scripts import experiment_tournament_distribution_mechanisms as experiment
    from scripts.tournament_evaluation import run_evaluation
    from src.models.tournament_prediction import file_digest
    from betting_app.tests.test_siamese_research_dataset import match
    from betting_app.tests.test_tournament_evaluation_runner import _manifest, _save

    source_root = experiment.ROOT
    for name in ('scripts/experiment_tournament_distribution_mechanisms.py',
                 'scripts/tournament_evaluation.py', 'scripts/score_tournament_forecasts.py',
                 'src/models/tournament_formats.py', 'src/models/tournament_prediction.py',
                 'src/models/tournament_uncertainty.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source_root / name)
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    path, manifest = _manifest(tmp_path)
    manifest['qualification'] = 'retrospective_reconstruction'
    manifest['models'][0]['id'] = 'exp039'
    start = manifest['events'][0]['origins'][0]
    start['tables'] = {'exp039': 'table.json'}
    checkpoint = {
        'origin_id': 'checkpoint', 'axis': 'axis3', 'forecast_scope': 'remaining_tournament',
        'reference_origin_id': 'start', 'forecast_mode': 'daily_rollforward',
        'cutoff': '2026-01-03T00:00:00Z',
        'model_exclusions': {'exp039': 'missing current feature support'},
        'targets': [{**start['targets'][0], 'target_start_at': '2026-01-04T12:00:00Z'}],
        'state': _save(tmp_path / 'state.json', {
            'version': 1, 'cutoff': '2026-01-03T00:00:00Z',
            'temporal_policy': 'daily_rollforward', 'completed': [],
        }),
    }
    manifest['events'][0]['origins'].append(checkpoint)
    _save(path, manifest)
    benchmark = tmp_path / 'benchmark'
    artifact = benchmark / 'models/2026/exp039.json'
    artifact.parent.mkdir(parents=True)
    _save(artifact, {'fixture': 'explicit probability table ancestry'})
    _save(benchmark / 'protocol.json', {'models': {'p_exp039': {}}})
    table = json.loads((tmp_path / 'table.json').read_text())
    table['provenance'].update(
        model_id='exp039', model_artifact_sha256=file_digest(artifact),
        training_cutoff='2024-01-01T00:00:00Z', calibration_cutoff='2025-01-01T00:00:00Z')
    _save(tmp_path / 'table.json', table)
    baseline = tmp_path / 'baseline'
    run_evaluation(path, baseline, simulations=50, seed=81)
    history = tmp_path / 'history.json'
    _save(history, [match(1, '2023-01-02')])
    pd.DataFrame([{
        'golgg_match_id': '1', 'date': '2023-01-02', 'team1_id': 'a', 'team2_id': 'b',
        'best_of': 1, 'y_true': 1, 'p_exp039': .6, 'history_last_source_day': '2023-01-01',
        'feature_history_max_at': '2023-01-02T00:00:00Z',
    }]).to_csv(benchmark / 'predictions.csv', index=False)
    outcomes = tmp_path / 'outcomes.json'
    _save(outcomes, [
        {'tournament_id': 'event', 'phase_id': 'event-final', 'origin_id': origin,
         'target_id': 'champion', 'observed': 'A'} for origin in ('start', 'checkpoint')])
    output = tmp_path / 'data/artifacts/tournament-validation-roadmap-20260910/frozen'
    args = SimpleNamespace(
        manifest=str(path), baseline_run=str(baseline), predictions=str(benchmark),
        source=str(history), outcomes=str(outcomes), output_dir=str(output),
        model='exp039', probability_column='p_exp039', matrix_mode='frozen',
        selection_start='2022-01-01', calibration_start='2023-01-01',
        test_start='2024-01-01', test_end='2027-01-01', simulations=50, seed=81,
        latent_seed=82, bootstrap=2, bootstrap_seed=82, score_window_years=2,
        prior_grid=[1.], probability_bins=[0, 1],
        settings_evidence=str(benchmark / 'protocol.json'), settings_status='report_informed')
    experiment.run(args)
    rows = [json.loads(line) for line in (output / 'forecasts.jsonl').read_text().splitlines()]
    frozen = next(row for row in rows if row['origin_id'] == 'checkpoint' and row['model'] == 'fixed')
    assert frozen['probabilities'] == {'A': 1., 'B': 0.}
