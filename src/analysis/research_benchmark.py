"""Manifest-driven research scoring. No training, source mutation or promotion.

Historical date-only scenarios are intentionally distinct from publication-time
proof. Existing model_benchmark owns core metric definitions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.bootstrap import monthly_block_bootstrap_delta

IDENTITY = ['golgg_match_id', 'team1_id', 'team2_id', 'date', 'best_of', 'y']

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_frame(frame, probabilities, nullable=()):
    required = IDENTITY + ['result_day', 'feature_source_max_day'] + list(probabilities)
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f'missing columns: {missing}')
    if frame.empty or frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError('empty or duplicate match identity')
    if frame[IDENTITY].isna().any().any():
        raise ValueError('null match identity')
    for column in ['date', 'result_day', 'feature_source_max_day']:
        values = frame[column].dropna().astype(str)
        if not values.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError(f'{column}: expected explicit calendar-day scenario')
        pd.to_datetime(values, errors='raise')
    if frame.feature_source_max_day.isna().any() or not (frame.feature_source_max_day < frame.date).all():
        raise ValueError('history must be released strictly before target day')
    if not (frame.result_day >= frame.date).all():
        raise ValueError('result precedes match date')
    if not frame.y.isin([0, 1]).all() or not frame.best_of.isin([1, 3, 5]).all():
        raise ValueError('invalid outcome or BO')
    for column in probabilities:
        values = frame[column]
        if column not in nullable and values.isna().any():
            raise ValueError(f'missing probability: {column}')
        present = values.dropna().to_numpy(float)
        if not np.isfinite(present).all() or ((present <= 0) | (present >= 1)).any():
            raise ValueError(f'invalid probability: {column}; expected 0<p<1')


def attach_candidate(base, candidate, column, *, online_updates=False):
    """Align complete forecasts; online state may evolve after fixed past CAL.

    Online callers must declare the released daily frozen-period update policy.
    Both the latest label used by the state and CAL remain strictly pre-target.
    Batch-model callers retain the closed-TRAIN-before-CAL invariant.
    """
    candidate = candidate.copy()
    validate_frame(candidate, [column])
    candidate.golgg_match_id = candidate.golgg_match_id.astype(str)
    if set(candidate.golgg_match_id) != set(base.golgg_match_id.astype(str)):
        raise ValueError('candidate coverage must exactly match locked cohort; no silent intersection')
    candidate = candidate.set_index('golgg_match_id').loc[base.golgg_match_id.astype(str)]
    for key in IDENTITY[1:] + ['result_day']:
        if not np.array_equal(base[key].astype(str), candidate[key].astype(str)):
            raise ValueError(f'candidate identity mismatch: {key}')
    for key in ['train_end', 'calibration_end']:
        if key not in candidate or candidate[key].isna().any():
            raise ValueError(f'candidate requires {key}')
        values = candidate[key].astype(str)
        if not values.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError(f'{key}: calendar-day release bound required')
        pd.to_datetime(values, errors='raise')
        if not (values.to_numpy() < candidate.date.to_numpy()).all():
            raise ValueError(f'future {key}')
    if online_updates:
        if ('update_policy' not in candidate
                or not candidate.update_policy.eq('released_daily_frozen_map_periods').all()):
            raise ValueError('online candidate requires explicit released update_policy')
    elif not (candidate.train_end <= candidate.calibration_end).all():
        raise ValueError('train_end follows calibration_end')
    out = base.copy()
    out['candidate'] = candidate[column].to_numpy(float)
    return out


def paired_delta(frame, delta, draws):
    delta = np.asarray(delta, float)
    if len(delta) != len(frame) or not np.isfinite(delta).all():
        raise ValueError('invalid paired losses')
    months = frame.date.astype(str).str[:7]
    equal_mean = pd.Series(delta).groupby(months.to_numpy()).mean().mean()
    result = dict(delta=float(delta.mean()), blocks=int(months.nunique()), rows=len(frame),
                  equal_block_delta=float(equal_mean), ci95=None, bootstrap_p_nonnegative=None)
    if result['blocks'] > 1:
        work = pd.DataFrame({'date':frame.date.to_numpy(), 'delta':delta})
        mean, lo, hi, samples = monthly_block_bootstrap_delta(work, 'delta', n_bootstraps=draws, random_seed=20260915)
        result.update(delta=mean, ci95=[lo, hi], bootstrap_p_nonnegative=float(np.mean(samples >= 0)))
    return result


def losses(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    return -np.where(y, np.log(p), np.log1p(-p))


def safe(value):
    if isinstance(value, dict):
        return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_inputs(config, root):
    checks = []
    for item in [config['input']] + config.get('evidence', []):
        path = root / item['path']
        exists = path.is_file()
        actual = sha(path) if exists else None
        checks.append(dict(path=str(path), exists=exists, sha256=actual,
                           matches=exists and actual == item['sha256'], role=item.get('role', 'match_input')))
    return checks


def run_suite(config_path, research_root, output_dir=None, doctor=False,
              candidate_data=None, candidate_col='p', draws=5000):
    config_path, root = Path(config_path), Path(research_root)
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 1:
        raise ValueError('unsupported benchmark manifest')
    if draws < 5000 and not doctor:
        raise ValueError('research suite requires at least5000 monthly resamples')
    checks = inspect_inputs(config, root)
    if doctor:
        return {'status':'READY' if all(c['matches'] for c in checks) else 'MISSING_OR_CHANGED_INPUTS',
                'inputs':checks, 'suite':config['name'], 'does_not_train':True}
    if not all(c['matches'] for c in checks):
        raise ValueError('missing/changed manifest input; run --doctor for details')
    if output_dir is None:
        raise ValueError('--output-dir must name a new run directory')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'run already exists: {output}')
    frame = pd.read_parquet(root / config['input']['path'])
    frame.golgg_match_id = frame.golgg_match_id.astype(str)
    if len(frame) != config['expected_rows']:
        raise ValueError('locked cohort row count mismatch')
    models = dict(config['models'])
    validate_frame(frame, list(models.values()) + ['market_open'], config['nullable'])
    inputs = {'manifest':sha(config_path), **{c['path']:c['sha256'] for c in checks}}
    if candidate_data:
        path = Path(candidate_data)
        candidate = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'golgg_match_id':str})
        frame = attach_candidate(frame, candidate, candidate_col)
        models['candidate'] = 'candidate'
        inputs[str(path)] = sha(path)
    coverage = {name:{'available':int(frame[col].notna().sum()), 'missing':int(frame[col].isna().sum())} for name,col in models.items()}
    masks = {'all':np.ones(len(frame), bool), 'open':frame.market_open.notna().to_numpy(),
             '039_common':frame.p_039.notna().to_numpy(),
             '039_open_common':(frame.p_039.notna() & frame.market_open.notna()).to_numpy()}
    metrics, comparisons, bins, profiles = [], [], [], []
    for cohort, mask in masks.items():
        part = frame.loc[mask]
        if part.empty:
            continue
        active = {n:c for n,c in models.items() if part[c].notna().all()}
        if 'open' in cohort:
            active['market_open'] = 'market_open'
        for name, column in active.items():
            y, p = part.y.to_numpy(), part[column].to_numpy()
            ll = losses(y,p)
            summary = asdict(compute_benchmark_summary(y,p,probability_epsilon=None))
            metrics.append(dict(cohort=cohort, model=name, **summary, tail25=int((ll >= 2.5).sum())))
            for reference in ['A0', 'Glicko', '039', 'market_open']:
                if reference not in active or reference == name:
                    continue
                q = part[active[reference]].to_numpy()
                for measure, delta in [('log_loss',ll-losses(y,q)), ('brier',(p-y)**2-(q-y)**2)]:
                    comparisons.append(dict(cohort=cohort, model=name, reference=reference, measure=measure,
                                            **paired_delta(part,delta,draws)))
            groups = np.clip(np.ceil(p*10).astype(int)-1,0,9)
            for b in range(10):
                selected = groups == b
                bins.append(dict(cohort=cohort,model=name,bin=b,n=int(selected.sum()),
                                 mean_p=float(p[selected].mean()) if selected.any() else None,
                                 observed=float(y[selected].mean()) if selected.any() else None))
            for dimension in ['best_of','competition_tier','experience','coplay','confidence','family','disagreement']:
                if dimension not in part:
                    continue
                for value, sub in part.groupby(dimension,dropna=False):
                    profiles.append(dict(cohort=cohort,model=name,dimension=dimension,value=str(value),n=len(sub),
                                         log_loss=float(losses(sub.y,sub[column]).mean()),
                                         brier=float(np.mean((sub.y-sub[column])**2))))
    report = dict(schema_version=1,status='DEVELOPMENT_ONLY',promotion='NOT_EVALUATED',
                  suite=config['name'],rows=len(frame),coverage=coverage,metrics=metrics,
                  comparisons=comparisons,calibration=bins,profiles=profiles,inputs=inputs,
                  code_sha256=sha(Path(__file__)), metric_policy='strict0<p<1; no clipping; binned Brier decomposition approximate',
                  dependency_hashes={str(path):sha(path) for path in [Path(__import__('src.analysis.model_benchmark',fromlist=['x']).__file__),Path(__import__('src.analysis.bootstrap',fromlist=['x']).__file__),Path(__import__('src.analysis.metrics',fromlist=['x']).__file__)]},bootstrap=dict(draws=draws,unit='month',seed=20260915),
                  scopes={'matches':'RECOMPUTED', 'tournament_phases':'NOT_RUN',
                          'quote_time_EV_CLV_Kelly':'NOT_RUN', 'prospective_confirmation':'NOT_RUN'},
                  evidence=config.get('evidence',[]),limitations=config['limitations'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(safe(report),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for name, records in [('metrics',metrics),('comparisons',comparisons),('calibration',bins),('profiles',profiles)]:
        pd.DataFrame(safe(records)).to_parquet(output/f'{name}.parquet',index=False)
    lines=['# Research benchmark', '', '**DEVELOPMENT_ONLY — brak decyzji produkcyjnej.**', '',
           '| Próba | Model | N | Log loss | Brier | Ogony ≥2,5 |', '|---|---|---:|---:|---:|---:|']
    for row in metrics:
        lines.append(f"| {row['cohort']} | {row['model']} | {row['sample_size']} | {row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['tail25']} |")
    lines += ['', '## Pokrycie', '', json.dumps(coverage,ensure_ascii=False), '',
              '## Zakres', '', 'Przeliczono mecze. Fazy turniejów, quote-time EV/CLV/Kelly i prospektywny test NIE zostały wykonane. Referencje w manifeście opisują wcześniejsze wyniki.', '',
              '## Ograniczenia', ''] + ['- '+x for x in config['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'status':report['status'],'rows':len(frame),'coverage':coverage,'report':str(output/'REPORT.md')}
