"""Offline EXP081 prediction guards and empirical score boundary contracts."""
from copy import deepcopy
from datetime import date

import pytest

from betting_app.tests.test_siamese_research_dataset import match
from src.models.tournament_prediction import (
    fit_score_distributions, predict_exp081_pairs, prepare_pair_bundle,
)

ROSTERS = {'a': [str(i) for i in range(5)], 'b': [str(i) for i in range(5, 10)]}


def artifact():
    return {
        'model_version': 'exp081-corrected-history-ensemble-calibrated-v1-2024',
        'feature_version': 'ratings-w20-symmetric-series-v1',
        'created_at': '2024-05-01T00:00:00Z', 'status': 'experimental_not_qualified',
        'provenance': {'fold': {'year': 2024, 'fit': {'date_max': '2022-12-31'},
                                'calibration': {'date_max': '2023-12-31'}}},
        'ensemble_calibration': {'calibration_end': '2023-12-31', 'source_sha256': 'a' * 64},
        'scaler': {'feature_names': ['team_elo_logit'], 'means': [0.], 'scales': [1.]},
        'ensemble': [{'w1': [[1., -1.]], 'b1': [0., 0.], 'w2': [[1., 0.], [0., 1.]],
                      'b2': [0., 0.], 'w3': [[1.], [-1.]], 'b3': 0., 'platt_slope': 1.}],
    }


@pytest.fixture
def bundle():
    return prepare_pair_bundle([match(1, '2024-01-01')], ROSTERS,
                               cutoff='2024-02-01T00:00:00Z', best_ofs=[1, 3, 5])


def predict(model, pairs, *, mode='exploratory', teams=('a', 'b')):
    return predict_exp081_pairs(model, pairs, teams=list(teams), best_ofs=[1, 3, 5],
                               cutoff='2024-02-01T00:00:00Z', mode=mode)


def test_unverified_training_feature_semantics_require_explicit_exploratory_mode(bundle):
    with pytest.raises(ValueError, match='verified'):
        predict(artifact(), bundle, mode='verified')
    _, audit = predict(artifact(), bundle)
    assert audit['qualification'] == 'exploratory_retrospective_not_pit_certified'
    assert any('roster' in gap for gap in audit['compatibility_gaps'])


def test_future_training_and_feature_history_are_blocked_even_exploratory(bundle):
    future = artifact()
    future['provenance']['fold']['fit']['date_max'] = '2024-02-01'
    with pytest.raises(ValueError, match='training'):
        predict(future, bundle)
    poisoned = deepcopy(bundle)
    poisoned['rows'][0]['feature_history_max_at'] = '2024-02-01T00:00:01Z'
    with pytest.raises(ValueError, match='history'):
        predict(artifact(), poisoned)



def test_future_calibration_and_nonfinite_weights_are_blocked(bundle):
    future = artifact()
    future['ensemble_calibration']['calibration_end'] = '2024-02-01'
    with pytest.raises(ValueError, match='calibration'):
        predict(future, bundle)
    broken = artifact()
    broken['ensemble'][0]['w1'][0][0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        predict(broken, bundle)

def test_counterfactual_pair_coverage_and_frozen_states_are_required(bundle):
    with pytest.raises(ValueError, match='pair'):
        predict(artifact(), bundle, teams=('a', 'b', 'c'))
    changed = deepcopy(bundle)
    changed['rows'][1]['team_elo'] += .01
    with pytest.raises(ValueError, match='frozen'):
        predict(artifact(), changed)


def test_frozen_bundle_requires_explicit_nonoverlapping_roster_ids(bundle):
    bundle['rosters']['b'][0] = bundle['rosters']['a'][0]
    with pytest.raises(ValueError, match='roster'):
        predict(artifact(), bundle)


def test_calibrated_pair_probabilities_drive_a_bo5_without_duplicate_sides_or_expansion(bundle):
    from src.models.tournament_formats import simulate_tournament

    for row in bundle['rows']:
        row['team_elo'] = 0.7
    probabilities, _ = predict(artifact(), bundle)
    spec = {
        'version': 1, 'id': 'model-final', 'teams': ['a', 'b'],
        'stages': [{'id': 'final', 'kind': 'single_elimination',
                    'entrants': ['a', 'b'], 'best_of': 5}],
        'champion': {'stage': 'final', 'group': 'all', 'rank': 1},
    }
    result = simulate_tournament(spec, probabilities, simulations=20000, seed=81)
    assert result['champion_prob']['a'] == pytest.approx(0.7, abs=0.015)
    assert sum(result['champion_prob'].values()) == 1


def test_previous_roster_proxy_never_uses_target_day_roster():
    future = match(2, '2024-02-01')
    for player in future['games'][0]['t1_players'].values():
        player['player_id'] = 'future-' + str(player['player_id'])
    prepared = prepare_pair_bundle([match(1, '2024-01-01'), future], None,
                                  cutoff='2024-02-01T00:00:00Z', best_ofs=[1],
                                  previous_roster_teams=['a', 'b'])
    assert prepared['rosters'] == ROSTERS
    assert prepared['roster_policy'] == 'previous_observed_roster_proxy'


def test_score_model_is_empirical_and_strictly_prior_without_fallback():
    scores = [{'id': 'old', 'date': '2024-01-01', 'best_of': 3, 'score_a': 2, 'score_b': 1},
              {'id': 'boundary', 'date': '2024-02-01', 'best_of': 3, 'score_a': 2, 'score_b': 0}]
    pmfs, audit = fit_score_distributions(scores, cutoff=date(2024, 2, 1), best_ofs=[3])
    assert pmfs == {3: [0., 1.]}
    assert audit['samples'] == {'3': 1}
    with pytest.raises(ValueError, match='no prior'):
        fit_score_distributions(scores, cutoff=date(2024, 1, 1), best_ofs=[3])
    invalid = [dict(scores[0], score_b=2)]
    with pytest.raises(ValueError, match='score'):
        fit_score_distributions(invalid, cutoff=date(2024, 2, 1), best_ofs=[3])


def chronological_artifact(bundle):
    from scripts.prospective_sports_features import FEATURE_VERSION, FEATURE_CONTRACT, MODEL_VERSION
    from src.models.symmetric_series import build_feature_mapping
    from src.models.tournament_prediction import json_digest
    result = artifact()
    names = list(build_feature_mapping(bundle['rows'][0], best_of=1))
    result.update(model_version=MODEL_VERSION + '-20240101', feature_version=FEATURE_VERSION,
                  feature_contract=dict(FEATURE_CONTRACT), feature_contract_sha256=json_digest(FEATURE_CONTRACT),
                  eligible_from='2024-01-01T00:00:00Z')
    result['scaler'] = {'feature_names': names, 'means': [0.] * len(names), 'scales': [1.] * len(names)}
    result['ensemble'][0]['w1'] = [[1., -1.] if name == 'team_elo_logit' else [0., 0.] for name in names]
    return result


def test_chronological_semantic_parity_does_not_claim_source_availability(bundle):
    model = chronological_artifact(bundle)
    probabilities, audit = predict(model, bundle, mode='chronological')
    assert set(probabilities) == {('a', 'b', bo) for bo in (1, 3, 5)}
    assert audit['feature_semantic_compatibility'] == 'matched_chronological_version'
    assert audit['compatibility_gaps'] == []
    assert audit['availability_gaps']
    assert audit['qualification'] == 'chronologically_reconstructed_research_only'
    with pytest.raises(ValueError, match='availability'):
        predict(model, bundle, mode='verified')


def test_chronological_mode_rejects_old_weights_and_forged_semantic_recipe(bundle):
    with pytest.raises(ValueError, match='chronological'):
        predict(artifact(), bundle, mode='chronological')
    forged = chronological_artifact(bundle)
    forged['feature_contract']['w20'] = 'dummy measurements'
    with pytest.raises(ValueError, match='feature contract'):
        predict(forged, bundle, mode='chronological')


def test_fold_cannot_be_selected_before_its_declared_eligible_origin(bundle):
    model = chronological_artifact(bundle)
    model['eligible_from'] = '2024-02-02T00:00:00Z'
    with pytest.raises(ValueError, match='eligible'):
        predict(model, bundle, mode='chronological')


def test_annual_recipe_uses_semantic_contract_not_version_prefix(bundle):
    model = chronological_artifact(bundle)
    reference, _ = predict(model, bundle, mode='chronological')
    model.update(model_version='exp081-priorday-canonical79-strong-focal-g1-annual-v1-2024',
                 feature_algebra_version='ratings-w20-symmetric-series-v1',
                 prediction_semantics='direct_series_probability',
                 training_end='2022-12-31', calibration_end='2023-12-31',
                 series_projection_count=0)
    actual, audit = predict(model, bundle, mode='chronological')
    assert actual == pytest.approx(reference)
    assert audit['qualification'] == 'chronologically_reconstructed_research_only'


def annual_metadata(bundle):
    model = chronological_artifact(bundle)
    return {key: model[key] for key in (
        'model_version', 'feature_version', 'feature_contract', 'feature_contract_sha256',
        'eligible_from', 'provenance', 'ensemble_calibration')} | {
        'feature_algebra_version': 'ratings-w20-symmetric-series-v1',
        'prediction_semantics': 'direct_series_probability',
        'training_end': '2022-12-31', 'calibration_end': '2023-12-31',
        'series_projection_count': 0,
    }


def reverse_bundle(bundle):
    from scripts.prospective_sports_features import NATIVE_FIELDS
    from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS
    from src.models.team_order import pair_columns
    result = deepcopy(bundle)
    for row in result['rows']:
        row['team_a'], row['team_b'] = row['team_b'], row['team_a']
        for field in _PROBABILITY_FIELDS:
            row[field] = 1-row[field]
        for left, right in pair_columns(sorted(set(_REQUIRED_BASE_FIELDS) | set(NATIVE_FIELDS))):
            row[left], row[right] = row[right], row[left]
    return result


@pytest.fixture
def parity_bundle():
    from betting_app.tests.test_prospective_sports_features import _third_team_match
    history = [match(1, '2024-01-01'), _third_team_match(2, '2024-01-02'),
               match(3, '2024-01-03', winner=False), match(4, '2024-01-04')]
    rosters = ROSTERS | {'c': [str(i) for i in range(10, 15)]}
    return prepare_pair_bundle(history, rosters, cutoff='2024-01-05T00:00:00Z', best_ofs=[1, 3, 5])


@pytest.mark.parametrize('recipe', ['exp039', 'linear79', 'exp081', 'exp081_strong', 'exp081_strong_ensemble'])
def test_fitted_serialized_recipes_match_training_inference_on_all_pairs_and_sides(parity_bundle, tmp_path, recipe):
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression
    from scripts.corrected_history_evaluation import build_exp039_features, exp039_probability
    from scripts.tournament_evaluation import _probabilities
    from scripts.train_and_tune_siamese_series import build_training_features, build_model_artifact, train_single_member
    from src.models.team_order import swap_orientation
    from src.models.tournament_prediction import file_digest, predict_model_pairs
    from src.utils.module_loading import load_module_from_path
    from pathlib import Path

    frame = pd.DataFrame(parity_bundle['rows']).assign(BoN=[r['best_of'] for r in parity_bundle['rows']], y_true=0)
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1])
    metadata = annual_metadata(parity_bundle)
    # Nonuniform train scales and calibrated fitted parameters expose missing
    # scaling, wrong order, averaging probabilities, or double calibration.
    if recipe == 'exp039':
        x, names, rank = build_exp039_features(frame)
        helper = load_module_from_path(
            Path(__file__).resolve().parents[2] / 'scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py',
            'tournament_native_fixture')
        fitted = helper.build_logistic_regression().fit(x[names], labels)
        slope = 1.3
        metadata.update(pipeline=fitted, features=names, rank_probability_features=rank, calibration_slope=slope)
        expected = np.clip(expit(slope * logit(exp039_probability(fitted, x, names, rank))), 1e-12, 1-1e-12)
        flipped = swap_orientation(x, names, rank, np.ones(len(x), dtype=bool))
        expected_reverse = np.clip(expit(slope * logit(exp039_probability(fitted, flipped, names, rank))), 1e-12, 1-1e-12)
    else:
        x, names = build_training_features(frame)
        scales = x.std(axis=0)
        scales[scales == 0] = 1
        x = x / scales
        if recipe == 'linear79':
            fitted = LogisticRegression(C=.1, fit_intercept=False, max_iter=2000).fit(x, labels)
            slope = 1.3
            metadata.update(model=fitted, features=names, scales=scales, calibration_slope=slope)
            expected = np.clip(expit(slope * fitted.decision_function(x)), 1e-12, 1-1e-12)
            expected_reverse = np.clip(expit(slope * fitted.decision_function(-x)), 1e-12, 1-1e-12)
        else:
            strong = recipe != 'exp081'
            members = [train_single_member(x, labels, x, labels, epochs=2, batch_size=4,
                d_h1=4, d_h2=3, seed=seed, weight_decay=.01 if strong else .0001,
                bootstrap_fraction=.8 if strong else 1.) for seed in (81, 98)]
            factor = 1.7 if recipe == 'exp081_strong_ensemble' else 1.
            exported = build_model_artifact(members, names, scales)
            exported.update(metadata)
            for member in exported['ensemble']:
                member['platt_slope'] *= factor
            metadata = exported
            expected = expit(np.stack([m.forward_anti_symmetric(x)[0].ravel() * m.platt_slope for m in members]).mean(axis=0) * factor)
            expected_reverse = expit(np.stack([m.forward_anti_symmetric(-x)[0].ravel() * m.platt_slope for m in members]).mean(axis=0) * factor)
    metadata['model_version'] = 'arbitrary-semantic-version-' + recipe
    kind = 'exp081' if recipe.startswith('exp081') else recipe
    path = tmp_path / (recipe + ('.json' if kind == 'exp081' else '.joblib'))
    if kind == 'exp081':
        path.write_text(json.dumps(metadata))
        loaded = json.loads(path.read_text())
    else:
        joblib.dump(metadata, path)
        loaded = joblib.load(path)
    for pairs, reference in ((parity_bundle, expected), (reverse_bundle(parity_bundle), expected_reverse)):
        actual, audit = predict_model_pairs(loaded, pairs, model_kind=kind, artifact_sha256=file_digest(path),
            teams=list(pairs['rosters']), best_ofs=[1, 3, 5], cutoff=pairs['cutoff'])
        keys = [(r['team_a'], r['team_b'], r['best_of']) for r in pairs['rows']]
        assert set(actual) == set(keys)
        assert [actual[key] for key in keys] == pytest.approx(reference, abs=1e-12)
        assert audit['model_artifact_sha256'] == file_digest(path)
        assert audit['availability_certified'] is False
        table = {'probability_unit': 'series_win', 'rows': [
            {'team_a': a, 'team_b': b, 'best_of': bo, 'p': p} for (a, b, bo), p in actual.items()],
            'provenance': audit | {'model_id': recipe}}
        (tmp_path / 'table.json').write_text(json.dumps(table))
        consumed, _ = _probabilities(
            {'id': recipe, 'kind': 'series_table'},
            {'cutoff': pairs['cutoff'], 'tables': {recipe: 'table.json'}}, {},
            {'teams': list(pairs['rosters'])}, [1, 3, 5], tmp_path, 'retrospective_reconstruction')
        assert [consumed[key] for key in keys] == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize('kind', ['glicko', 'glicko_format'])
def test_glicko_json_matches_training_once_projection_and_reverse(bundle, tmp_path, kind):
    import json
    import pandas as pd
    from scipy.special import expit, logit
    from scripts.run_full_historical_predictions import rating_series_probabilities
    from src.models.tournament_prediction import file_digest, predict_model_pairs
    for row in bundle['rows']:
        row['player_gl'] = .7
    model = annual_metadata(bundle) | {
        'series_projection_count': 1, 'map_probability_field': 'player_gl', 'map_clip': [.001, .999],
        'calibration_slopes': {'1': .8, '3': 1.2, '5': 1.7} if kind == 'glicko_format' else None,
    }
    path = tmp_path / (kind + '.json')
    path.write_text(json.dumps(model))
    loaded = json.loads(path.read_text())
    for pairs in (bundle, reverse_bundle(bundle)):
        expected = rating_series_probabilities(pd.DataFrame(pairs['rows']))
        if kind == 'glicko_format':
            expected = expit(logit(expected) * [.8, 1.2, 1.7])
        actual, audit = predict_model_pairs(loaded, pairs, model_kind=kind, artifact_sha256=file_digest(path),
            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=pairs['cutoff'])
        assert list(actual.values()) == pytest.approx(expected, abs=1e-12)
        assert audit['series_projection_count'] == 1
    if kind == 'glicko':
        assert rating_series_probabilities(pd.DataFrame(bundle['rows'])) == pytest.approx([.7, .784, .83692])


def test_native_supplement_is_identical_state_roster_mean_without_canonical_changes():
    import pandas as pd
    from scripts.prospective_sports_features import ChronologicalFeatureState, NATIVE_FIELDS, _prepare_history
    state = ChronologicalFeatureState()
    series, _ = _prepare_history([match(1, '2024-01-01'), match(2, '2024-01-02', winner=False)], date(2024, 2, 1))
    for item in series:
        state.consume(item)
    canonical = state.pairs(ROSTERS, date(2024, 2, 1), [1, 3, 5])
    native = state.pairs(ROSTERS, date(2024, 2, 1), [1, 3, 5], include_native=True)
    pd.testing.assert_frame_equal(native.drop(columns=list(NATIVE_FIELDS)), canonical)
    for side, team in enumerate(sorted(ROSTERS), 1):
        expected = sum(state.manager.systems['tm'].get_player_rating(p).sigma for p in ROSTERS[team]) / 5
        assert native[f'player_tm_sigma_avg{side}'].tolist() == pytest.approx([expected] * 3)
        assert expected > 0


@pytest.mark.parametrize('mutation', ['missing_field', 'missing_contract', 'zero', 'nonfinite', 'wrong_contract', 'unfrozen'])
def test_malformed_native_supplements_fail_before_inference(bundle, mutation):
    from src.models.tournament_prediction import predict_model_pairs
    if mutation == 'missing_field':
        del bundle['rows'][0]['player_tm_sigma_avg1']
    elif mutation == 'missing_contract':
        del bundle['native_supplement']
    elif mutation == 'wrong_contract':
        bundle['native_supplement']['semantics'] = 'constant imputation'
    else:
        bundle['rows'][0]['player_tm_sigma_avg1'] = {
            'zero': 0, 'nonfinite': float('nan'), 'unfrozen': 123.,
        }[mutation]
    with pytest.raises(ValueError, match='supplement|sports|frozen'):
        predict_model_pairs(annual_metadata(bundle), bundle, model_kind='exp039', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])


def test_generic_siamese_preserves_existing_chronological_artifact_contract(bundle):
    from src.models.tournament_prediction import predict_model_pairs, json_digest
    model = chronological_artifact(bundle)
    expected, _ = predict(model, bundle, mode='chronological')
    actual, _ = predict_model_pairs(model, bundle, model_kind='exp081', artifact_sha256=json_digest(model),
                                    teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('field,value', [
    ('prediction_semantics', 'map_probability'), ('series_projection_count', 1),
    ('feature_algebra_version', 'wrong'), ('training_end', '2024-02-01'),
    ('calibration_end', '2024-02-01'),
])
def test_generic_annual_siamese_rejects_conflicting_semantics_and_completion_bounds(bundle, field, value):
    from src.models.tournament_prediction import predict_model_pairs
    model = chronological_artifact(bundle) | annual_metadata(bundle) | {field: value}
    with pytest.raises(ValueError):
        predict_model_pairs(model, bundle, model_kind='exp081', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])


def test_exp039_cannot_infer_without_native_supplement_even_with_complete_canonical79(bundle):
    from scripts.prospective_sports_features import NATIVE_FIELDS
    from src.models.tournament_prediction import predict_model_pairs
    del bundle['native_supplement']
    for row in bundle['rows']:
        for field in NATIVE_FIELDS:
            del row[field]
    model = annual_metadata(bundle) | {'calibration_slope': 1.}
    with pytest.raises(ValueError, match='native supplement'):
        predict_model_pairs(model, bundle, model_kind='exp039', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])


def test_format_glicko_cannot_substitute_global_or_missing_calibration(bundle):
    from src.models.tournament_prediction import predict_model_pairs
    model = annual_metadata(bundle) | {
        'series_projection_count': 1, 'map_probability_field': 'player_gl', 'map_clip': [.001, .999],
        'calibration_slopes': {'1': 1., '3': 1.},
    }
    with pytest.raises(ValueError, match='all three'):
        predict_model_pairs(model, bundle, model_kind='glicko_format', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])
    model['calibration_slopes']['5'] = -1.
    with pytest.raises(ValueError, match='positive'):
        predict_model_pairs(model, bundle, model_kind='glicko_format', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])


def test_annual_siamese_rejects_feature_reordering_and_member_shape_broadcasting(bundle):
    from src.models.tournament_prediction import predict_model_pairs
    model = chronological_artifact(bundle)
    model['scaler']['feature_names'].reverse()
    with pytest.raises(ValueError, match='feature order'):
        predict_model_pairs(model, bundle, model_kind='exp081', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])
    model = chronological_artifact(bundle)
    model['ensemble'][0]['b1'] = [0.]
    with pytest.raises(ValueError, match='dimensions'):
        predict_model_pairs(model, bundle, model_kind='exp081', artifact_sha256='a'*64,
                            teams=['a', 'b'], best_ofs=[1, 3, 5], cutoff=bundle['cutoff'])


def test_incremental_bundle_matches_standalone_without_mutating_earlier_snapshot():
    from datetime import date
    from scripts.prospective_sports_features import ChronologicalFeatureState, _prepare_history
    from src.models.tournament_prediction import prepare_pair_bundle_from_state

    history = [match(1, '2024-01-01'), match(2, '2024-01-03', winner=False)]
    series, _ = _prepare_history(history, date(2024, 2, 1))
    state = ChronologicalFeatureState()
    state.consume(series[0])
    first = prepare_pair_bundle_from_state(state, ROSTERS, cutoff='2024-01-02T00:00:00Z', best_ofs=[1, 3, 5])
    frozen = deepcopy(first)
    state.consume(series[1])
    later = prepare_pair_bundle_from_state(state, ROSTERS, cutoff='2024-01-04T00:00:00Z', best_ofs=[1, 3, 5])
    for actual in (first, later):
        expected = prepare_pair_bundle(history, ROSTERS, cutoff=actual['cutoff'], best_ofs=[1, 3, 5])
        assert actual['rows'] == expected['rows']
        assert actual['feature_contract_sha256'] == expected['feature_contract_sha256']
        assert actual['audit']['feature_history_max_at'] == expected['audit']['feature_history_max_at']
        assert actual['audit']['used_games'] == expected['audit']['used_games']
    assert first == frozen


def test_incremental_bundle_rejects_state_from_the_cutoff_day():
    from datetime import date
    from scripts.prospective_sports_features import ChronologicalFeatureState, _prepare_history
    from src.models.tournament_prediction import prepare_pair_bundle_from_state

    series, _ = _prepare_history([match(1, '2024-01-02')], date(2024, 2, 1))
    state = ChronologicalFeatureState()
    state.consume(series[0])
    with pytest.raises(ValueError, match='cutoff-day or future outcomes'):
        prepare_pair_bundle_from_state(state, ROSTERS, cutoff='2024-01-02T00:00:00Z', best_ofs=[3])
