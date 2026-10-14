#!/usr/bin/env python3
"""Freeze EXP-089's fixed SERIES models; capture prospective local inputs and score settlements.

Artifacts are exclusive-created, hash-checked local research records, not signed external
publication evidence. Only trusted local joblib artifacts are loaded. No fitting, odds
input, network, database mutation, or promotion occurs in this workflow.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_oddsfree_distillation import predict_artifact
from scripts.evaluate_oddsfree_distillation import _metrics, _paired
from scripts.train_and_tune_siamese_series import build_training_features
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

VERSION = 'exp089-frozen-oddsfree-confirmation-v1'
MODELS = {'outcome': 0.0, 'market_w10': 0.1}
INFERENCE_SOURCES = {
    'scripts/confirm_oddsfree_student.py',
    'scripts/experiment_oddsfree_distillation.py',
    'scripts/train_and_tune_siamese_series.py',
    'src/models/symmetric_series.py',
    'scripts/evaluate_oddsfree_distillation.py',
    'scripts/benchmark_siamese_architectures.py',
    'src/analysis/probability_metrics.py',
}
RAW_FIELDS = _REQUIRED_BASE_FIELDS | {'best_of'}
ID = 'golgg_match_id'
META_FIELDS = {ID, 'match_start_at', 'feature_history_max_at'}
OUTCOME_FIELDS = {ID, 'match_start_at', 'y_true', 'settled_at'}
LIMITATIONS = [
    'Prospective local observation only: external publication and original feature derivation are not certified.',
    'Feature-history timestamps are supplied assertions; point-in-time/source and roster qualification remain unproven.',
    'Hashes detect changed local bytes; they are not signatures or independently trusted timestamps.',
    'Previously inspected EXP-089 holdouts are not new confirmation evidence; later historical rows are not automatically untouched.',
    'Fewer than 10000 settled eligible series cannot meet a 10000-observation confirmation requirement.',
    'Complete SERIES-win probabilities only; never reinterpret or expand them as map probabilities.',
    'Research only: no retuning, qualification, staking, or automatic promotion, regardless of sample size.',
]


def _now():
    return datetime.now(timezone.utc)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _write_bytes(path, data):
    with Path(path).open('xb') as handle:
        handle.write(data)


def _seal(directory, name, value):
    data = _json_bytes(value)
    _write_bytes(directory / f'{name}.json', data)
    _write_bytes(directory / f'{name}.sha256', (_hash(data) + '\n').encode())


def _read_sealed(directory, name, *, expected_version=VERSION):
    data = (directory / f'{name}.json').read_bytes()
    if _hash(data) != (directory / f'{name}.sha256').read_text().strip():
        raise ValueError(f'{name} hash integrity failure')
    value = json.loads(data)
    if not isinstance(value, dict) or value.get('version') != expected_version:
        raise ValueError(f'unsupported {name} version')
    return value, _hash(data)


def _safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('unsafe archive path')
    return path


def _checked_files(directory, hashes):
    result = {}
    for name, expected in hashes.items():
        data = (directory / _safe_relative(name)).read_bytes()
        if _hash(data) != expected:
            raise ValueError(f'hash integrity failure: {name}')
        result[name] = data
    return result


def _timestamp(value, field):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{field} must be a timezone-aware timestamp') from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f'{field} must be a timezone-aware timestamp')
    return stamp.tz_convert('UTC')


def _times(frame, field):
    return pd.Series([_timestamp(value, field) for value in frame[field]], index=frame.index,
                     dtype='datetime64[ns, UTC]')


def _csv(data, required, *, empty=False):
    header = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))), [])
    if len(header) != len(set(header)) or set(header) != set(required):
        raise ValueError(f'CSV columns must be exactly {sorted(required)}; duplicate/unexpected/missing column')
    frame = pd.read_csv(io.BytesIO(data), dtype={ID: 'string'}, keep_default_na=False,
                        float_precision='round_trip')
    if frame.empty and not empty:
        raise ValueError('input rows must not be empty')
    _ids(frame)
    return frame


def _ids(frame):
    ids = frame[ID]
    if ids.isna().any() or ids.astype(str).str.strip().eq('').any() or ids.duplicated().any():
        raise ValueError('match IDs must be nonempty and unique')
    if not ids.astype(str).eq(ids.astype(str).str.strip()).all():
        raise ValueError('match IDs must be exact strings without surrounding whitespace')


def _raw_rows(frame):
    if not frame.columns.is_unique or not RAW_FIELDS.issubset(frame.columns) or set(frame.columns) - RAW_FIELDS - {ID}:
        raise ValueError('raw sports columns must be canonical fields, best_of and optional golgg_match_id; no odds or labels')
    if frame.empty:
        raise ValueError('raw sports rows must not be empty')
    if ID in frame:
        _ids(frame)
    try:
        values = frame[sorted(RAW_FIELDS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('raw sports fields must be finite numeric values') from exc
    if not np.isfinite(values).all():
        raise ValueError('raw sports fields must be finite numeric values')
    return frame


def _environment():
    return {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib')
    }}


def _canonical_names():
    # Only determines the canonical schema; this neutral row is never evidence or training.
    row = {field: 0.5 for field in _REQUIRED_BASE_FIELDS}
    row['best_of'] = 3
    return build_training_features(pd.DataFrame([row]))[1]


def _validate_model(artifact, name, fold, locked_at, canonical):
    if (artifact.get('target') != 'series' or artifact.get('requires_odds_at_inference') is not False
            or artifact.get('model') != name or artifact.get('year') != 2026
            or artifact.get('weight') != MODELS[name]
            or artifact.get('teacher') != (None if name == 'outcome' else 'market')
            or artifact.get('feature_names') != canonical):
        raise ValueError(f'{name}: frozen series model contract mismatch')
    for field in ('coef', 'scale'):
        value = np.asarray(artifact[field], dtype=float)
        if value.shape != (len(canonical),) or not np.isfinite(value).all():
            raise ValueError(f'{name}: invalid {field}')
        if field == 'scale' and (value <= 0).any():
            raise ValueError(f'{name}: scales must be positive')
    slope = artifact['slope']
    if np.ndim(slope) != 0 or not np.isfinite(slope) or slope <= 0:
        raise ValueError(f'{name}: calibration slope must be finite and positive')
    record = fold['models'][name]
    if record['weight'] != MODELS[name] or record['final_calibration_slope'] != slope:
        raise ValueError(f'{name}: final calibration provenance mismatch')
    stages = artifact['stages']
    if stages != fold['stages']:
        raise ValueError(f'{name}: stage provenance mismatch')
    previous = None
    for stage in ('train', 'stop', 'select', 'calibration'):
        record = stages[stage]
        first = pd.Timestamp(record['first'], tz='UTC')
        last = pd.Timestamp(record['last'], tz='UTC')
        if record['n'] <= 0 or pd.isna(first) or pd.isna(last) or first > last or (previous is not None and first <= previous):
            raise ValueError(f'{name}: nonchronological final calibration')
        previous = last
    # Stage provenance has date resolution; require the complete final day before lock.
    if previous + pd.Timedelta(days=1) > locked_at:
        raise ValueError(f'{name}: final calibration must precede lock')


def freeze_protocol(source_run: Path, output_dir: Path) -> dict:
    """Exclusive-create a fixed, hash-locked copy of the approved 2026 EXP-089 arms."""
    source_run, output_dir = Path(source_run), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_protocol_bytes = (source_run / 'protocol.json').read_bytes()
    fold_bytes = (source_run / '2026-fold.json').read_bytes()
    source_protocol, fold = json.loads(source_protocol_bytes), json.loads(fold_bytes)
    if source_protocol.get('experiment') != 'EXP-089' or fold.get('year') != 2026:
        raise ValueError('source must be EXP-089 final 2026 fold')
    archives = {'source/protocol.json': source_protocol_bytes, 'source/2026-fold.json': fold_bytes}
    canonical = _canonical_names()
    artifacts = {}
    for name in MODELS:
        filename = f'2026-{name}.joblib'
        record = fold['models'][name]
        data = (source_run / filename).read_bytes()
        if record['artifact'] != filename or _hash(data) != record['artifact_sha256']:
            raise ValueError(f'{name}: source model hash mismatch')
        archives[filename] = data
        artifacts[name] = joblib.load(io.BytesIO(data))
    dependencies = set(source_protocol.get('source_sha256', {})) | INFERENCE_SOURCES
    source_hashes = {}
    for name in sorted(dependencies):
        relative = _safe_relative(name)
        if relative.suffix != '.py' or relative.parts[0] not in {'scripts', 'src', 'betting_app'}:
            raise ValueError('provenance dependencies must be repository Python source files')
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('provenance source escapes the repository')
        data = path.read_bytes()
        source_hashes[name] = _hash(data)
        expected = source_protocol.get('source_sha256', {}).get(name)
        if expected is not None and source_hashes[name] != expected:
            raise ValueError(f'training-time source contract changed: {name}')
        archives[f'inference_source/{name}'] = data
    locked_at = _now()
    for name, artifact in artifacts.items():
        _validate_model(artifact, name, fold, locked_at, canonical)
    protocol = {
        'version': VERSION, 'locked_at': locked_at.isoformat(), 'target': 'series',
        'requires_odds_at_inference': False, 'models': {
            name: {'artifact': f'2026-{name}.joblib', 'sha256': fold['models'][name]['artifact_sha256'], 'weight': weight}
            for name, weight in MODELS.items()
        }, 'feature_names': canonical, 'source_run': str(source_run.resolve()),
        'source_sha256': source_hashes, 'environment': _environment(),
        'files': {name: _hash(data) for name, data in archives.items()},
        'status': 'frozen_not_qualified', 'promotion': False, 'limitations': LIMITATIONS,
    }
    for name, data in archives.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes(path, data)
    _seal(output_dir, 'protocol', protocol)
    return protocol


def load_protocol(protocol_dir: Path) -> dict:
    """Verify archive, executable source, environment and fixed model contracts."""
    protocol_dir = Path(protocol_dir)
    protocol, _ = _read_sealed(protocol_dir, 'protocol')
    if protocol.get('target') != 'series' or protocol.get('requires_odds_at_inference') is not False or protocol.get('promotion') is not False:
        raise ValueError('frozen protocol contract mismatch')
    lock = _timestamp(protocol['locked_at'], 'locked_at')
    if lock > _now():
        raise ValueError('protocol lock is in the future')
    if protocol['environment'] != _environment():
        raise ValueError('frozen inference environment changed')
    archives = _checked_files(protocol_dir, protocol['files'])
    _checked_files(ROOT, protocol['source_sha256'])
    canonical = _canonical_names()
    if protocol['feature_names'] != canonical or set(protocol['models']) != set(MODELS):
        raise ValueError('canonical feature/model protocol mismatch')
    source_protocol = json.loads(archives['source/protocol.json'])
    expected_sources = set(source_protocol.get('source_sha256', {})) | INFERENCE_SOURCES
    if set(protocol['source_sha256']) != expected_sources:
        raise ValueError('inference source hash coverage mismatch')
    for name, digest in protocol['source_sha256'].items():
        if _hash(archives[f'inference_source/{name}']) != digest:
            raise ValueError('archived inference source hash mismatch')
    fold = json.loads(archives['source/2026-fold.json'])
    for name, weight in MODELS.items():
        model = protocol['models'][name]
        filename = f'2026-{name}.joblib'
        digest = fold['models'][name]['artifact_sha256']
        if model != {'artifact': filename, 'sha256': digest, 'weight': weight} or _hash(archives[filename]) != digest:
            raise ValueError('frozen model hash mismatch')
        _validate_model(joblib.load(io.BytesIO(archives[filename])), name, fold, lock, canonical)
    return protocol


def predict_frozen(protocol_dir: Path, raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Predict complete series wins from canonical raw sports fields only, retaining index."""
    protocol = load_protocol(protocol_dir)
    _raw_rows(raw_rows)
    result = pd.DataFrame(index=raw_rows.index)
    for name in MODELS:
        model = protocol['models'][name]
        data = (Path(protocol_dir) / model['artifact']).read_bytes()
        if _hash(data) != model['sha256']:
            raise ValueError('model hash changed before inference')
        p = np.asarray(predict_artifact(joblib.load(io.BytesIO(data)), raw_rows), dtype=float)
        if p.shape != (len(raw_rows),) or not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
            raise ValueError('predictions must be finite open-interval series probabilities')
        result['p__' + name] = p
    return result


def _capture_metadata(raw, metadata, lock, captured_at, predicted_at):
    if set(raw[ID]) != set(metadata[ID]):
        raise ValueError('metadata must match every raw match ID exactly')
    metadata = metadata.set_index(ID).loc[raw[ID]].reset_index()
    starts, history = _times(metadata, 'match_start_at'), _times(metadata, 'feature_history_max_at')
    if not lock <= captured_at <= predicted_at:
        raise ValueError('capture must follow protocol lock and precede prediction')
    if (history > captured_at).any() or (starts <= predicted_at).any():
        raise ValueError('prospective capture requires feature history <= capture <= prediction < every match start')
    metadata['match_start_at'] = starts.map(lambda value: value.isoformat())
    metadata['feature_history_max_at'] = history.map(lambda value: value.isoformat())
    for name in ('source_available_at', 'data_cutoff_at', 'captured_at'):
        metadata[name] = captured_at.isoformat()
    metadata['predicted_at'] = predicted_at.isoformat()
    return metadata


def capture_forecasts(protocol_dir: Path, raw_rows_csv: Path, metadata_csv: Path, output_dir: Path) -> dict:
    """Observe local input bytes now and archive predictions before every listed event starts."""
    protocol_dir, output_dir = Path(protocol_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(protocol_dir)
    protocol_hash = _hash((protocol_dir / 'protocol.json').read_bytes())
    raw_bytes, metadata_bytes = Path(raw_rows_csv).read_bytes(), Path(metadata_csv).read_bytes()
    captured_at = _now()
    raw = _csv(raw_bytes, RAW_FIELDS | {ID})
    metadata = _csv(metadata_bytes, META_FIELDS)
    lock = _timestamp(protocol['locked_at'], 'locked_at')
    _capture_metadata(raw, metadata, lock, captured_at, captured_at)
    probabilities = predict_frozen(protocol_dir, raw)
    predicted_at = _now()
    predictions = _capture_metadata(raw, metadata, lock, captured_at, predicted_at)
    for name in probabilities:
        predictions[name] = probabilities[name].to_numpy()
    archives = {'raw_rows.csv': raw_bytes, 'metadata.csv': metadata_bytes,
                'predictions.csv': predictions.to_csv(index=False).encode('utf-8')}
    manifest = {
        'version': VERSION, 'kind': 'prospective_local_series_forecasts',
        'protocol_sha256': protocol_hash, 'captured_at': captured_at.isoformat(),
        'predicted_at': predicted_at.isoformat(), 'source_available_at': captured_at.isoformat(),
        'data_cutoff_at': captured_at.isoformat(), 'target': 'series',
        'cohort_ids': raw[ID].tolist(), 'row_count': len(raw),
        'files': {name: _hash(data) for name, data in archives.items()},
        'promotion': False, 'limitations': LIMITATIONS,
    }
    for name, data in archives.items():
        _write_bytes(output_dir / name, data)
    _seal(output_dir, 'manifest', manifest)
    return manifest


def _verified_forecast(protocol_dir, forecast_dir, protocol, now):
    manifest, manifest_hash = _read_sealed(forecast_dir, 'manifest')
    if manifest.get('kind') != 'prospective_local_series_forecasts' or manifest.get('target') != 'series':
        raise ValueError('invalid forecast bundle kind/target')
    if manifest['protocol_sha256'] != _hash((protocol_dir / 'protocol.json').read_bytes()):
        raise ValueError('forecast protocol hash mismatch')
    if set(manifest['files']) != {'raw_rows.csv', 'metadata.csv', 'predictions.csv'}:
        raise ValueError('forecast archive hash coverage mismatch')
    archives = _checked_files(forecast_dir, manifest['files'])
    raw, metadata = _csv(archives['raw_rows.csv'], RAW_FIELDS | {ID}), _csv(archives['metadata.csv'], META_FIELDS)
    capture = _timestamp(manifest['captured_at'], 'captured_at')
    predicted = _timestamp(manifest['predicted_at'], 'predicted_at')
    if predicted > now:
        raise ValueError('forecast prediction is in the future')
    for name in ('source_available_at', 'data_cutoff_at'):
        if _timestamp(manifest[name], name) != capture:
            raise ValueError('source observation timestamp must equal actual capture')
    expected = _capture_metadata(raw, metadata, _timestamp(protocol['locked_at'], 'locked_at'), capture, predicted)
    probabilities = predict_frozen(protocol_dir, raw)
    for name in probabilities:
        expected[name] = probabilities[name].to_numpy()
    saved = _csv(archives['predictions.csv'], set(expected.columns))
    if manifest['cohort_ids'] != raw[ID].tolist() or manifest['row_count'] != len(raw) or len(saved) != len(expected):
        raise ValueError('forecast cohort mismatch')
    for name in expected:
        if name.startswith('p__'):
            try:
                values = saved[name].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError('invalid archived prediction') from exc
            if not np.isfinite(values).all() or not np.array_equal(values, expected[name].to_numpy()):
                raise ValueError('archived prediction differs from recomputed frozen prediction')
        elif not saved[name].astype(str).equals(expected[name].astype(str)):
            raise ValueError(f'archived prediction metadata mismatch: {name}')
    return expected, manifest_hash


def score_confirmation(protocol_dir: Path, forecast_dirs: list[Path], outcomes_csv: Path, output_dir: Path) -> dict:
    """Accumulate disjoint frozen batches; keep every unsettled forecast pending."""
    protocol_dir, output_dir = Path(protocol_dir), Path(output_dir)
    if not forecast_dirs or isinstance(forecast_dirs, (str, Path)):
        raise ValueError('scoring requires a nonempty list of forecast directories')
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(protocol_dir)
    outcome_bytes = Path(outcomes_csv).read_bytes()
    observed_at = _now()
    batches, forecast_hashes = [], []
    for directory in forecast_dirs:
        frame, digest = _verified_forecast(protocol_dir, Path(directory), protocol, observed_at)
        batches.append(frame)
        forecast_hashes.append({'path': str(Path(directory).resolve()), 'sha256': digest})
    predictions = pd.concat(batches, ignore_index=True)
    _ids(predictions)  # Never select a latest forecast or double-count repeated matches.
    outcomes = _csv(outcome_bytes, OUTCOME_FIELDS, empty=True)
    if not set(outcomes[ID]).issubset(set(predictions[ID])):
        raise ValueError('outcomes contain unknown forecast match IDs')
    starts, settled = _times(outcomes, 'match_start_at'), _times(outcomes, 'settled_at')
    if (settled < starts).any() or (settled > observed_at).any():
        raise ValueError('settlement must be after match start and no later than current observation')
    try:
        y = outcomes['y_true'].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('outcomes must be finite binary labels') from exc
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('outcomes must be finite binary labels')
    selected = predictions.set_index(ID).loc[outcomes[ID]].reset_index()
    if not _times(selected, 'match_start_at').equals(starts):
        raise ValueError('outcome match start differs from captured forecast')
    selected['y_true'] = y
    selected['settled_at'] = settled.map(lambda value: value.isoformat())
    metrics = {}
    for name in MODELS:
        # Existing evaluation wrapper uses shared probability_metrics, handling explicitly
        # unavailable diagnostic calibration; diagnostic fits never alter predictions.
        metrics[name] = _metrics(y, selected['p__' + name].to_numpy(float))
        metrics[name]['status'] = 'ok' if len(y) else 'unavailable_empty'
    paired = _paired(y, selected['p__market_w10'].to_numpy(float),
                     selected['p__outcome'].to_numpy(float), starts.dt.tz_localize(None))
    ids = selected[ID].tolist()
    pending = predictions.loc[~predictions[ID].isin(ids), ID].tolist()
    archives = {'outcomes.csv': outcome_bytes, 'settled_predictions.csv': selected.to_csv(index=False).encode()}
    result = {
        'version': VERSION, 'kind': 'frozen_confirmation_score', 'observed_at': observed_at.isoformat(),
        'protocol_sha256': _hash((protocol_dir / 'protocol.json').read_bytes()),
        'forecast_manifests': forecast_hashes, 'forecast_count': len(predictions),
        'settled_count': len(selected), 'pending_count': len(pending), 'cohort_ids': ids,
        'pending_ids': pending, 'metrics': metrics, 'paired': paired,
        'comparison': 'market_w10 minus outcome; negative LogLoss/Brier delta favors challenger',
        'below_10000': len(selected) < 10000, 'status': 'research_only_not_qualified',
        'promotion': False, 'limitations': LIMITATIONS,
        'files': {name: _hash(data) for name, data in archives.items()},
    }
    for name, data in archives.items():
        _write_bytes(output_dir / name, data)
    _seal(output_dir, 'results', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze', help='Lock the two approved local EXP-089 models')
    freeze.add_argument('--source-run', type=Path, required=True)
    freeze.add_argument('--output-dir', type=Path, required=True)
    capture = commands.add_parser('capture', help='Capture local raw sports inputs before series starts')
    capture.add_argument('--protocol-dir', type=Path, required=True)
    capture.add_argument('--raw-rows-csv', type=Path, required=True)
    capture.add_argument('--metadata-csv', type=Path, required=True)
    capture.add_argument('--output-dir', type=Path, required=True)
    score = commands.add_parser('score', help='Score verified forecasts on known settlements only')
    score.add_argument('--protocol-dir', type=Path, required=True)
    score.add_argument('--forecast-dir', dest='forecast_dirs', type=Path, action='append', required=True,
                       help='Immutable batch directory; repeat to accumulate disjoint batches')
    score.add_argument('--outcomes-csv', type=Path, required=True)
    score.add_argument('--output-dir', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    function = {'freeze': freeze_protocol, 'capture': capture_forecasts, 'score': score_confirmation}[command]
    print(json.dumps(function(**args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
