"""Input-aligned orchestration regressions; no native inference or fitting."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import c0_input_aligned_study as study


def panel():
    rng = np.random.default_rng(19)
    rows = []
    for phase in ('TRAIN', 'CAL', 'ASSESS'):
        for i in range(30):
            rows.append(dict(row_id=f'{phase}-{i}', phase=phase, month='2026-07',
                p_C0=float(rng.uniform(.05,.95)), q=float(rng.uniform(.05,.95)),
                odds_a=float(rng.uniform(1.5,4)), odds_b=float(rng.uniform(1.5,4)),
                y=i%2, fit_eligible=True, market_eligible=True,
                native_admitted=True, financial_eligible=True))
    return pd.DataFrame(rows)


def test_support_and_membership_never_read_assessment_outcomes():
    frame = panel()
    frame['y'] = frame['y'].astype(object)
    frame.loc[frame.phase == 'ASSESS', 'y'] = 'DO_NOT_READ'
    train, cal, receipt = study.fit_membership(frame)
    assert len(train) == len(cal) == 30
    assert receipt['support_status'] == 'SUPPORTED'
    assert all(key.startswith('TRAIN-') for key in receipt['TRAIN'])
    frame.loc[frame.phase == 'CAL', 'y'] = 1
    assert study.fit_membership(frame)[2]['support_status'] == 'INSUFFICIENT_FIT_SUPPORT'


def test_matched_fits_preserve_market_only_and_mature_zero_action_cash_rows():
    frame = panel()
    frame.loc[0, ['native_admitted','p_C0']] = [False, np.nan]
    frame.loc[1, 'market_eligible'] = False
    before = frame.copy(deep=True)
    train, cal, receipt = study.fit_membership(frame)
    assert len(train) == 28 and len(cal) == 30
    assert receipt['coverage']['market_only'] == 1
    assert receipt['coverage']['financial_eligible'] == 90
    pd.testing.assert_frame_equal(frame, before)


def test_output_preflight_rejects_any_existing_artifact_before_writes(tmp_path):
    outputs = [tmp_path/'models'/'S_selected.json', tmp_path/'report'/'decision.json']
    outputs[1].parent.mkdir()
    outputs[1].write_text('sealed')
    with pytest.raises(FileExistsError):
        study.preflight_outputs(outputs)
    assert not outputs[0].parent.exists()
    assert outputs[1].read_text() == 'sealed'


def test_rank_deficient_train_stops_without_future_support_peek():
    frame = panel()
    frame.loc[frame.phase == 'TRAIN', ['q','p_C0']] = .6
    assert study.fit_membership(frame)[2]['support_status'] == 'INSUFFICIENT_FIT_SUPPORT'


def test_assessment_keeps_unavailable_native_and_no_quote_rows_as_zero_cash():
    from src.models.ev_approach_models import C0MarketHead
    frame = panel().loc[lambda f: f.phase == 'ASSESS'].copy().reset_index(drop=True)
    frame['month'] = ['2026-07']*10 + ['2026-08']*10 + ['2026-09']*10
    frame['q'] = .5
    frame['quotes'] = [[] for _ in range(len(frame))]
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    fixed_market = C0MarketHead('M', (1.,), 0., {'success':True}, {})
    heads = {recipe:fixed_market for recipe in study.RECIPES}
    _, result = study._assessment(frame, heads, {'statistics':{'bootstrap_draws':20,'seed':20261006}})
    for recipe in study.RECIPES:
        ledger = [row for row in result['rows'] if row['recipe'] == recipe]
        assert len(ledger) == 30
        assert all(row['financial_eligible'] and row['profit'] == 0 and not row['selected'] for row in ledger)
    market = [row for row in result['rows'] if row['recipe'] == 'M']
    native = [row for row in result['rows'] if row['recipe'] == 'S']
    assert market[0]['probability'] == .5
    assert native[0]['probability'] is None
    assert result['decision'] == 'NOT_ADOPTED'
    assert not result['gates']['selected_ROI_estimable']


def evaluation_fixture():
    from scipy.special import expit
    from src.models.ev_approach_models import c0_features
    from src.models.c0_betting_price_head import price_features

    class CenterHead:
        def __init__(self, recipe):
            self.recipe = recipe

        def predict_details(self, frame):
            features = (price_features(frame, self.recipe) if self.recipe.endswith('-price')
                        else c0_features(frame, self.recipe))
            centers = {'S': [1], 'S-cal': [1], 'M': [1], 'M-cal': [1],
                       'M-shape': [1, 1], 'Fixed50': [.5, .5], 'H': [0, 1],
                       'M-price': [1, 1, 0], 'H-price': [1, 1, 0, 0]}
            eta = features @ np.asarray(centers[self.recipe])
            raw = expit(eta)
            probability = np.clip(raw, 1e-6, 1-1e-6)
            return {'probability': probability, 'raw_probability': raw,
                    'logit': eta, 'clipped': probability != raw}

    frame = panel().loc[lambda f: f.phase == 'ASSESS'].copy().reset_index(drop=True)
    frame['month'] = ['2026-07']*10 + ['2026-08']*10 + ['2026-09']*10
    frame['forecast_eligible'] = True
    frame['forecast_absence'] = None
    frame['primary_decision'] = True
    frame['quotes'] = [[{'bookmaker':'synthetic', 'quote_id':str(i),
                        'odds_a':4., 'odds_b':4.}] for i in range(len(frame))]
    return frame, {recipe:CenterHead(recipe) for recipe in study.RECIPES}


def test_known_result_forbids_every_recipe_without_erasing_cash_or_quote_inputs():
    frame, heads = evaluation_fixture()
    frame.loc[0, 'forecast_eligible'] = False
    frame.loc[0, 'forecast_absence'] = 'target_already_released_before_cutoff'
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    before = frame.copy(deep=True)
    result = study.evaluate_input_aligned(frame, heads, draws=20, seed=19)
    for recipe in study.RECIPES:
        ledger = [r for r in result['rows'] if r['recipe'] == recipe]
        forbidden = ledger[0]
        assert forbidden['probability'] is None
        assert forbidden['absence'] == 'target_already_released_before_cutoff'
        assert forbidden['forecast_eligible'] is False
        assert forbidden['financial_eligible'] and forbidden['profit'] == 0
        assert not forbidden['selected']
        assert len(ledger) == 30
        assert result['metrics'][recipe]['financial_eligible_count'] == 30
    pd.testing.assert_frame_equal(frame, before)


def test_forecast_membership_requires_explicit_flag_but_keeps_legacy_synthetic_default():
    frame = panel()
    frame['forecast_eligible'] = True
    frame.loc[[0, 30], 'forecast_eligible'] = False
    train, cal, receipt = study.fit_membership(frame)
    assert len(train) == len(cal) == 29
    assert receipt['coverage']['financial_eligible'] == 90
    frame['forecast_eligible'] = frame.forecast_eligible.astype(object)
    frame.loc[1, 'forecast_eligible'] = 1
    with pytest.raises(ValueError, match='forecast_eligible'):
        study.fit_membership(frame)


def test_market_only_extra_coverage_does_not_fail_matched_predictive_coverage_gate():
    frame, heads = evaluation_fixture()
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    result = study.evaluate_input_aligned(frame, heads, draws=20, seed=19)
    assert result['metrics']['M-price']['coverage_count'] == 30
    assert result['metrics']['H-price']['coverage_count'] == 29
    for control in ('M-price', 'M-shape'):
        assert result['gates']['coverage_vs_'+control]
        coverage = result['common_predictive_coverage'][control]
        assert coverage['candidate_count'] == coverage['control_count'] == 29
        assert coverage['row_ids'] == result['primary_contrasts'][control]['row_ids']
    assert result['decision'] == 'NOT_ADOPTED'
    assert result['reasons'] == [key for key, passed in result['gates'].items() if not passed]


def test_input_only_native_contrasts_include_consumer_valid_pairs_without_market():
    prior, heads = evaluation_fixture()
    prior.loc[0, 'q'] = np.nan
    actual = prior.copy(deep=True)
    actual['p_C0'] = 1-prior.p_C0
    protocol = {'statistics':{'bootstrap_draws':20, 'seed':19}}
    result = study.evaluate_input_aligned(prior, heads, draws=20, seed=19)
    diagnostic = study._input_diagnostic(prior, actual, heads, protocol, result)
    for recipe in ('S', 'S-cal'):
        assert diagnostic['paired_input_contrasts'][recipe]['common_count'] == 30
        assert prior.loc[0, 'row_id'] in diagnostic['paired_input_contrasts'][recipe]['row_ids']
    for recipe in set(study.RECIPES)-{'S', 'S-cal'}:
        assert diagnostic['paired_input_contrasts'][recipe]['common_count'] == 29
    assert diagnostic['fixed_prior_S'] == result['own_selections']['S']


def test_evaluation_rejects_nonboolean_forecast_eligibility_before_any_prediction():
    frame, heads = evaluation_fixture()
    frame['forecast_eligible'] = frame.forecast_eligible.astype(object)
    frame.loc[0, 'forecast_eligible'] = 'False'
    with pytest.raises(ValueError, match='forecast_eligible'):
        study.evaluate_input_aligned(frame, heads, draws=20, seed=19)


def test_coverage_gate_counts_pre_prediction_population_not_only_successful_pairs():
    frame, heads = evaluation_fixture()
    frame.loc[0, 'odds_a'] = np.nan
    result = study.evaluate_input_aligned(frame, heads, draws=20, seed=19)
    coverage = result['common_predictive_coverage']['M-shape']
    assert len(coverage['population_row_ids']) == 30
    assert len(coverage['row_ids']) == 29
    assert coverage['candidate_count'] == 29
    assert coverage['control_count'] == 30
    assert not result['gates']['coverage_vs_M-shape']
