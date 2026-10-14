"""Loss-only development regressions; synthetic files, no DB/native generation.

The orchestration fixture is an alternate sealed SOURCE_PROTOCOL, never a way
for the CLI to replace the real run's population or authorization contract.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from src.models.c0_betting_price_head import price_features

from src.analysis import c0_ev_loss_study as study
from src.analysis.c0_ev_loss_evaluation import evaluate_ev_loss
from src.models.c0_ev_loss_head import C0EVLossHead
from src.models.ev_approach_models import C0MarketHead


REFERENCES = {
    'S': (1.,), 'S-cal': (1.,), 'M': (1.,), 'M-cal': (1.,),
    'M-shape': (1., 1.), 'Fixed50': (.5, .5), 'H': (0., 1.),
}
ARM_RECIPES = {'H-likelihood': 'H-price', 'H-return_mse': 'H-price',
               'M-likelihood': 'M-price', 'M-return_mse': 'M-price'}
MONTHS = ['2026-07', '2026-08', '2026-09']


def reference_heads():
    return {name: C0MarketHead(name, theta, 0., {'success': True}, {})
            for name, theta in REFERENCES.items()}


def arms(native_coefficient=0.):
    class ProbabilityRule:
        """Deterministic evaluation fixture, not a claimed fitted artifact."""
        def __init__(self, recipe, objective, theta):
            self.recipe_id, self.objective, self.theta = recipe, objective, theta

        def predict_details(self, frame):
            eta = price_features(frame, self.recipe_id) @ np.asarray(self.theta)
            raw = expit(eta)
            p = np.clip(raw, 1e-6, 1-1e-6)
            return {'logit': eta, 'raw_probability': raw,
                    'probability': p, 'clipped': raw != p}

        def predict(self, frame):
            return self.predict_details(frame)['probability']

    return {name: ProbabilityRule(recipe, name.split('-', 1)[1],
                (1., 1., 0., native_coefficient) if recipe == 'H-price' else (1., 1., 0.))
            for name, recipe in ARM_RECIPES.items()}


def synthetic_panel(n=90):
    rng = np.random.default_rng(29)
    rows = []
    for phase in ('TRAIN', 'CAL', 'ASSESS'):
        for i in range(n):
            month = (MONTHS[min(2, i * 3 // n)] if phase == 'ASSESS'
                     else '2026-02' if phase == 'TRAIN' else '2026-05')
            odds_a, odds_b = rng.uniform(1.5, 4.5, 2)
            rows.append(dict(row_id=f'{phase}-{i}', phase=phase, month=month,
                p_C0=float(rng.uniform(.05, .95)), q=float(rng.uniform(.05, .95)),
                odds_a=float(odds_a), odds_b=float(odds_b), y=i % 2,
                fit_eligible=True, market_eligible=True, native_admitted=True,
                financial_eligible=True, forecast_eligible=True, forecast_absence=None,
                primary_decision=True, quotes=[{'bookmaker': 'synthetic',
                    'quote_id': f'{phase}-{i}', 'odds_a':float(odds_a), 'odds_b':float(odds_b)}]))
    return pd.DataFrame(rows)


def assessment(n=90):
    frame = synthetic_panel(n).loc[lambda f: f.phase == 'ASSESS'].reset_index(drop=True)
    frame.attrs['monthly_universe'] = MONTHS
    return frame


def arm_ledger(result, objective, recipe):
    return {row['row_id']: row for row in result['arms'][objective]['rows']
            if row['recipe'] == recipe}


@pytest.mark.parametrize('winner,sign', [(0, 1), (1, -1)])
def test_selected_return_gate_rejects_both_over_and_underprediction(winner, sign):
    frame = assessment()
    frame[['p_C0', 'q']] = .5
    frame[['odds_a', 'odds_b']] = 2.4
    frame['y'] = winner
    frame['quotes'] = [[{'bookmaker':'synthetic', 'quote_id':str(i),
                        'odds_a':2.4, 'odds_b':2.4}] for i in range(len(frame))]
    result = evaluate_ev_loss(frame, arms(), reference_heads(), draws=100, seed=29)
    candidate = result['arms']['return_mse']['metrics']['H-price']
    expected = sign * .88 * 2.4 * .5
    for name in ('overstatement', 'near_overstatement'):
        assert candidate[name]['point'] == pytest.approx(expected)
        assert candidate[name]['interval'] == pytest.approx([expected, expected])
    assert not result['gates']['selected_overstatement_within_2pp']
    assert not result['gates']['near_band_overstatement_within_2pp']
    assert result['decision'] == 'NOT_ADOPTED'


def test_symmetric_calibration_gate_accepts_zero_error_in_all_three_months():
    frame = assessment()
    frame[['p_C0', 'q']] = .5
    frame[['odds_a', 'odds_b']] = 2.4
    frame['quotes'] = [[{'bookmaker':'synthetic', 'quote_id':str(i),
                        'odds_a':2.4, 'odds_b':2.4}] for i in range(len(frame))]
    result = evaluate_ev_loss(frame, arms(), reference_heads(), draws=100, seed=29)
    for key in ('overstatement', 'near_overstatement'):
        summary = result['arms']['return_mse']['metrics']['H-price'][key]
        assert summary['point'] == pytest.approx(0., abs=1e-12)
        assert summary['interval'] == pytest.approx([0., 0.], abs=1e-12)
    assert result['gates']['selected_overstatement_within_2pp']
    assert result['gates']['near_band_overstatement_within_2pp']
    # Identical heads cannot pass the strict three-improvement requirement.
    assert not any(result['gates']['return_mse_vs_' + control]
                   for control in ('H-likelihood', 'M-return_mse', 'M-likelihood'))
    assert result['decision'] == 'NOT_ADOPTED'


def test_three_ev_risk_contrasts_use_identical_pre_prediction_population_and_paired_draws():
    frame = assessment()
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    frame.loc[1, 'forecast_eligible'] = False
    frame.loc[2, ['market_eligible', 'q']] = [False, np.nan]
    fitted = arms()
    fitted['H-return_mse'].theta = (.9, .7, .1, .3)
    result = evaluate_ev_loss(frame, fitted, reference_heads(), draws=100, seed=29)
    candidate = arm_ledger(result, 'return_mse', 'H-price')
    controls = {'H-likelihood': arm_ledger(result, 'likelihood', 'H-price'),
                'M-return_mse': arm_ledger(result, 'return_mse', 'M-price'),
                'M-likelihood': arm_ledger(result, 'likelihood', 'M-price')}
    expected_ids = frame.loc[3:, 'row_id'].tolist()
    sampled = np.random.default_rng(29).integers(0, 3, (100, 3))
    assert set(result['risk_contrasts']) == set(controls)
    for name, control in controls.items():
        contrast = result['risk_contrasts'][name]
        assert contrast['row_ids'] == expected_ids
        assert contrast['count'] == len(expected_ids)
        assert contrast['confidence_level'] == pytest.approx(1 - .05 / 3)
        matched = frame.loc[3:]
        p = np.array([candidate[key]['probability'] for key in expected_ids])
        q = np.array([control[key]['probability'] for key in expected_ids])
        weight = .5 * .88**2 * (matched.odds_a.to_numpy()**2 + matched.odds_b.to_numpy()**2)
        delta = weight * ((p - matched.y.to_numpy())**2 - (q - matched.y.to_numpy())**2)
        assert contrast['delta_return_mse'] == pytest.approx(delta.mean())
        num = np.array([delta[matched.month.to_numpy() == month].sum() for month in MONTHS])
        den = np.array([(matched.month == month).sum() for month in MONTHS])
        values = num[sampled].sum(axis=1) / den[sampled].sum(axis=1)
        assert contrast['interval'] == pytest.approx(np.quantile(values, [.05/6, 1-.05/6]))
        assert result['gates']['return_mse_vs_' + name] == (contrast['interval'][1] < 0)
    matched = frame.loc[3:]
    p = np.array([candidate[key]['probability'] for key in expected_ids])
    h_ll = np.array([controls['H-likelihood'][key]['probability'] for key in expected_ids])
    y = matched.y.to_numpy()
    delta_ll = np.mean(-y*np.log(p) - (1-y)*np.log1p(-p)
                       + y*np.log(h_ll) + (1-y)*np.log1p(-h_ll))
    delta_brier = np.mean((p-y)**2 - (h_ll-y)**2)
    assert result['gates']['nonworse_LL_vs_H-likelihood'] == (delta_ll <= 0)
    assert result['gates']['nonworse_Brier_vs_H-likelihood'] == (delta_brier <= 0)


def test_full_523_opportunity_cash_denominator_keeps_absent_and_known_result_abstentions():
    frame = assessment(523)
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    frame.loc[1, 'forecast_eligible'] = False
    frame.loc[1, 'forecast_absence'] = 'target_already_released_before_cutoff'
    frame.at[2, 'quotes'] = []
    before = frame.copy(deep=True)
    result = evaluate_ev_loss(frame, arms(), reference_heads(), draws=100, seed=29)
    for objective in ('likelihood', 'return_mse'):
        report = result['arms'][objective]
        for recipe in (*REFERENCES, 'H-price', 'M-price'):
            ledger = arm_ledger(result, objective, recipe)
            metric = report['metrics'][recipe]
            assert len(ledger) == 523
            assert metric['financial_eligible_count'] == 523
            assert metric['profit_per_opportunity']['count'] == 523
            assert metric['profit_per_opportunity']['point'] == pytest.approx(
                sum(row['profit'] for row in ledger.values()) / 523)
            forbidden = ledger['ASSESS-1']
            assert forbidden['probability'] is None and not forbidden['selected']
            assert forbidden['financial_eligible'] and forbidden['profit'] == 0
            assert forbidden['absence'] == 'target_already_released_before_cutoff'
            assert ledger['ASSESS-2']['profit'] == 0 and not ledger['ASSESS-2']['selected']
        market = arm_ledger(result, objective, 'M-price')['ASSESS-0']
        native = arm_ledger(result, objective, 'H-price')['ASSESS-0']
        assert market['probability'] is not None
        assert native['probability'] is None and native['profit'] == 0
        assert report['metrics']['M-price']['coverage_count'] == 522
        assert report['metrics']['H-price']['coverage_count'] == 521
        assert report['gates']['coverage_vs_M-price']
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('selected_months', [[], ['2026-07'], ['2026-07', '2026-08']])
def test_zero_and_sparse_selection_intervals_never_drop_undefined_draws(selected_months):
    frame = assessment()
    frame[['p_C0', 'q']] = .5
    frame[['odds_a', 'odds_b']] = 2.4
    frame['quotes'] = [[{'bookmaker':'synthetic', 'quote_id':str(i),
                        'odds_a':2.4, 'odds_b':2.4}] if month in selected_months else []
                       for i, month in enumerate(frame.month)]
    result = evaluate_ev_loss(frame, arms(), reference_heads(), draws=1000, seed=29)
    metric = result['arms']['return_mse']['metrics']['H-price']
    samples = np.random.default_rng(29).integers(0, 3, (1000, 3))
    monthly_count = np.array([30 if month in selected_months else 0 for month in MONTHS])
    undefined = int((monthly_count[samples].sum(axis=1) == 0).sum())
    for key in ('ROI', 'overstatement', 'near_overstatement'):
        assert metric[key]['undefined_draws'] == undefined
        assert metric[key]['interval'] is None
        assert metric[key]['status'] == 'NOT_ESTIMABLE'
        assert metric[key]['equal_month_point'] is None
    assert not result['gates']['selected_ROI_estimable']
    assert result['decision'] == 'NOT_ADOPTED'
    assert metric['profit_per_opportunity']['count'] == 90


def test_fixed_prior_s_selection_reuses_exact_side_price_quote_and_cash_across_arms():
    frame = assessment()
    fitted = arms(native_coefficient=.7)
    result = evaluate_ev_loss(frame, fitted, reference_heads(), draws=100, seed=29)
    original = result['arms']['likelihood']['own_selections']['S']
    assert original
    expected = {row['row_id']:row for row in original}
    for objective in ('likelihood', 'return_mse'):
        for recipe, fixed in result['arms'][objective]['fixed_S_selection'].items():
            ledger = arm_ledger(result, objective, recipe)
            assert fixed['selection_count'] == len(original)
            assert fixed['fixed_cash_profit'] == pytest.approx(sum(row['profit'] for row in original))
            for row in fixed['rows']:
                source = expected[row['row_id']]
                for key in ('side', 'odds', 'quote_id', 'bookmaker'):
                    assert row[key] == source[key]
                assert row['fixed_profit'] == source['profit']
                p_a = ledger[row['row_id']]['probability']
                y_a = frame.set_index('row_id').loc[row['row_id'], 'y']
                p, y = (p_a, y_a) if row['side'] == 'A' else (1-p_a, 1-y_a)
                assert row['overstatement'] == pytest.approx(.88 * row['odds'] * (p-y))


def test_train_cal_support_is_common_and_does_not_inspect_assessment_labels():
    frame = synthetic_panel(30)
    frame['y'] = frame.y.astype(object)
    frame.loc[frame.phase == 'ASSESS', 'y'] = 'NOT_AVAILABLE_BEFORE_ASSESSMENT'
    frame.loc[0, ['native_admitted', 'p_C0']] = [False, np.nan]
    frame.loc[1, 'forecast_eligible'] = False
    train, cal, receipt = study.fit_membership(frame)
    assert receipt['support_status'] == 'SUPPORTED'
    assert train.row_id.tolist() == [f'TRAIN-{i}' for i in range(2, 30)]
    assert cal.row_id.tolist() == [f'CAL-{i}' for i in range(30)]
    assert receipt['coverage']['financial_eligible'] == 90
    frame.loc[frame.phase == 'CAL', 'y'] = 1
    assert study.fit_membership(frame)[2]['support_status'] == 'INSUFFICIENT_FIT_SUPPORT'


def test_rank_deficient_train_stops_even_when_cal_and_assessment_have_full_support():
    frame = synthetic_panel(30)
    frame.loc[frame.phase == 'TRAIN', ['q', 'p_C0', 'odds_a', 'odds_b']] = [.6, .6, 2., 2.]
    assert study.fit_membership(frame)[2]['support_status'] == 'INSUFFICIENT_FIT_SUPPORT'


def test_preflight_checks_all_destinations_before_creating_anything(tmp_path):
    outputs = [tmp_path/'models', tmp_path/'report'/'decision.json']
    outputs[1].parent.mkdir()
    outputs[1].write_text('sealed')
    with pytest.raises(FileExistsError):
        study.preflight_outputs(outputs)
    assert not outputs[0].exists()
    assert outputs[1].read_text() == 'sealed'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')))
    return path


def file_seal(root, path):
    return {'root':'repository', 'path':str(path.relative_to(root)),
            'sha256':sha(path), 'bytes':path.stat().st_size}


@pytest.fixture
def sealed_source(tmp_path, monkeypatch):
    """Real bytes and heads, not a patched verifier/fitter/evaluator."""
    root = tmp_path
    prior_id = 'fixture_prior'
    stages = {number:root/'data'/directory/'c0_input_aligned'/prior_id
              for number, directory in ((5,'05_model_input'), (6,'06_models'),
                                        (7,'07_model_output'), (8,'08_reporting'))}
    frame = synthetic_panel(30)
    stages[5].mkdir(parents=True)
    outputs = {}
    for name in ('prior_panel', 'actual_lineup_panel', 'fresh300s_prior_panel'):
        path = stages[5]/f'{name}.parquet'
        frame.to_parquet(path, index=False)
        outputs[str(path.relative_to(root))] = file_seal(root, path)
    references = {}
    for recipe, head in reference_heads().items():
        path = save(stages[6]/f'{recipe}_selected.json', head.to_dict())
        references[recipe] = file_seal(root, path)
    source = dict(schema='c0_input_aligned_v1', run_id=prior_id,
        expected_phase_denominators={'TRAIN':30, 'CAL':30, 'ASSESS':30},
        native_identity='Causal-C0:c0-native-2026-w32-e12-v1', mode='full',
        calendar={'TRAIN':['2026-01-01','2026-05-01'], 'CAL':['2026-05-01','2026-07-01'],
                  'ASSESS':['2026-07-01','2026-09-09']},
        tax=.12, epsilon=1e-6, minimum_net_ev_exclusive=.05,
        statistics={'bootstrap_draws':40, 'seed':20261006},
        sources={}, code_sha256={}, immutable_originals={}, preserved_forward_code_hashes={})
    protocol_path = save(stages[5]/'protocol.json', source)
    protocol_sha = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(stages[5]/'freeze_receipt.json', {'protocol_file_sha256':sha(protocol_path),
         'protocol_sha256':protocol_sha, 'formal_head_fits':0, 'base_fits':0})
    save(stages[5]/'preparation_receipt.json', {'protocol_sha256':protocol_sha,
         'evidence_class':'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY', 'outputs':outputs})
    save(stages[8]/'sealed_verification.json', {'status':'PASS', 'artifacts':{
         **outputs, **{ref['path']:ref for ref in references.values()}}})
    monkeypatch.setattr(study, 'PROJECT_ROOT', root)
    monkeypatch.setattr(study, 'SOURCE_PROTOCOL', str(protocol_path.relative_to(root)))
    return root, frame, stages


def new_stages(root, run_id):
    return {number:root/'data'/directory/'c0_ev_loss'/run_id
            for number, directory in ((5,'05_model_input'), (6,'06_models'),
                                      (7,'07_model_output'), (8,'08_reporting'))}


def test_freeze_is_immutable_zero_fit_and_binds_every_consumed_source(sealed_source):
    root, _, source_stages = sealed_source
    path = study.freeze_ev_loss('fixture_ev')
    frozen = json.loads(Path(path).read_text())
    stages = new_stages(root, 'fixture_ev')
    seal = json.loads((stages[5]/'freeze_receipt.json').read_text())
    assert seal['formal_head_fits'] == seal['base_fits'] == 0
    assert frozen['formal_fit_budget'] == 12 and frozen['base_fit_budget'] == 0
    hashes = {ref['sha256'] for ref in frozen['sources'].values()}
    for number, names in ((5, ('protocol.json', 'preparation_receipt.json', 'prior_panel.parquet',
                                'actual_lineup_panel.parquet', 'fresh300s_prior_panel.parquet')),
                          (8, ('sealed_verification.json',)),
                          (6, tuple(f'{recipe}_selected.json' for recipe in REFERENCES))):
        for name in names:
            assert sha(source_stages[number]/name) in hashes
    before = Path(path).read_bytes()
    with pytest.raises(FileExistsError):
        study.freeze_ev_loss('fixture_ev')
    assert Path(path).read_bytes() == before
    assert not stages[6].exists() and not stages[7].exists()


def test_hash_invalid_source_stops_before_any_fit_or_model_write(sealed_source, monkeypatch):
    root, _, source_stages = sealed_source
    protocol = study.freeze_ev_loss('corrupt_ev')
    path = source_stages[5]/'prior_panel.parquet'
    path.write_bytes(path.read_bytes() + b'changed-after-freeze')
    calls = []
    def forbidden_fit(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail('Hash-invalid input reached optimization')
    monkeypatch.setattr(study, 'fit_ev_loss_head', forbidden_fit)
    with pytest.raises(ValueError, match='(?i)(hash|seal|source|changed)'):
        study.run_ev_loss(protocol)
    assert calls == []
    assert not new_stages(root, 'corrupt_ev')[6].exists()


def test_run_preflight_rejects_existing_report_before_fits_or_partial_writes(sealed_source, monkeypatch):
    root, _, _ = sealed_source
    protocol = study.freeze_ev_loss('occupied_ev')
    stages = new_stages(root, 'occupied_ev')
    save(stages[8]/'decision.json', {'user':'preserve'})
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Occupied destination reached optimization')
    monkeypatch.setattr(study, 'fit_ev_loss_head', forbidden_fit)
    with pytest.raises(FileExistsError):
        study.run_ev_loss(protocol)
    assert json.loads((stages[8]/'decision.json').read_text()) == {'user':'preserve'}
    assert not stages[6].exists() and not stages[7].exists()


def test_lightweight_run_uses_twelve_train_fits_same_cal_ev_mse_and_saved_head_replay(sealed_source, monkeypatch):
    root, frame, _ = sealed_source
    protocol = study.freeze_ev_loss('budget_ev')
    real_fit = study.fit_ev_loss_head
    fitted_memberships = []
    def observed_fit(recipe, objective, train, *, regularization):
        fitted_memberships.append((recipe, objective, regularization, train.row_id.tolist()))
        assert set(train.phase) == {'TRAIN'}
        return real_fit(recipe, objective, train, regularization=regularization)
    monkeypatch.setattr(study, 'fit_ev_loss_head', observed_fit)
    decision = study.run_ev_loss(protocol)
    assert decision['formal_head_fits'] == decision['fit_attempt_count'] == 12
    assert decision['base_fits'] == 0
    assert len(fitted_memberships) == 12
    assert {(r, o, lam) for r, o, lam, _ in fitted_memberships} == {
        (r, o, lam) for r in ('H-price','M-price') for o in ('likelihood','return_mse')
        for lam in (.01, .1, 1.)}
    assert all(ids == frame.loc[frame.phase == 'TRAIN','row_id'].tolist()
               for _, _, _, ids in fitted_memberships)
    stages = new_stages(root, 'budget_ev')
    selection = json.loads((stages[6]/'selection.json').read_text())['selection']
    cal = frame.loc[frame.phase == 'CAL']
    for arm in ARM_RECIPES:
        candidates = [C0EVLossHead.from_dict(json.loads(path.read_text()))
                      for path in stages[6].glob(f'{arm}_lambda_*.json')]
        assert len(candidates) == 3
        weight = .5 * .88**2 * (cal.odds_a.to_numpy()**2 + cal.odds_b.to_numpy()**2)
        scores = [(float(np.mean(weight * (
            h.predict_details(cal)['probability'] - cal.y.to_numpy())**2)), h)
                  for h in candidates]
        best = min(score for score, _ in scores)
        chosen = max((h for score, h in scores if score <= best + 1e-12),
                     key=lambda h:h.regularization)
        selected = C0EVLossHead.from_dict(json.loads((stages[6]/f'{arm}_selected.json').read_text()))
        assert selected.regularization == chosen.regularization
        assert selected.coefficients == chosen.coefficients
        assert selection[arm]['criterion'] == 'return_mse'
    ledger = pd.read_parquet(stages[7]/'assessment_predictions_and_ledger.parquet')
    assess = frame.loc[frame.phase == 'ASSESS']
    for arm, recipe in ARM_RECIPES.items():
        selected = C0EVLossHead.from_dict(json.loads((stages[6]/f'{arm}_selected.json').read_text()))
        actual = ledger.loc[(ledger.objective == arm.split('-',1)[1]) & (ledger.recipe == recipe)]
        actual = actual.set_index('row_id').loc[assess.row_id]
        np.testing.assert_allclose(actual.probability, selected.predict_details(assess)['probability'], atol=1e-12)
    assert len(fitted_memberships) == 12  # diagnostics/assessment did not optimize
    before = {path:sha(path) for number in (6,7,8) for path in stages[number].rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        study.run_ev_loss(protocol)
    assert len(fitted_memberships) == 12
    assert all(sha(path) == digest for path, digest in before.items())
