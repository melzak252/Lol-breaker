"""Checksummed recovery of a completed prop study; never fit or activate a model.

The bundle is portable for inference from its prepared historical features. Raw
source reconstruction and the original runner's external guards still require
those separately retained assets. Joblib is executable pickle: trust the bundle
and obtain its SHA-256 from a trusted receipt, not from an untrusted download.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile

SCHEMA = 'prop_model_preservation_v1'
MANIFEST = 'recovery/manifest.json'
REFERENCE = 'recovery/reference.npz'
PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'lightgbm', 'joblib', 'pyarrow')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def _safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError(f'Unsafe archive member: {name}')


def _validate_archive(archive):
    names = [member.filename for member in archive.infolist()]
    if len(names) != len(set(names)):
        raise ValueError('Archive inventory mismatch: duplicate member')
    for member in archive.infolist():
        _safe_member(member.filename)
        kind = stat.S_IFMT(member.external_attr >> 16)
        if member.is_dir() or kind not in (0, stat.S_IFREG):
            raise ValueError(f'Unsafe archive member: {member.filename}')
    manifest = json.loads(archive.read(MANIFEST))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported preservation schema')
    if set(names) != set(manifest['members']) | {MANIFEST}:
        raise ValueError('Archive inventory mismatch')
    for name, expected in manifest['members'].items():
        with archive.open(name) as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if archive.getinfo(name).file_size != expected['bytes'] or digest != expected['sha256']:
            raise ValueError(f'Member integrity mismatch: {name}')
    return manifest


def restore_bundle(bundle, expected_sha256, destination):
    """Validate all members before extracting into a new, task-owned directory."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Check and extract from the same open archive, not a second path lookup.
    with Path(bundle).open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('Bundle SHA-256 mismatch')
        source.seek(0)
        with ZipFile(source) as archive:
            manifest = _validate_archive(archive)
            destination.mkdir(parents=True, exist_ok=False)
            archive.extractall(destination)
    return manifest


def _source_closure(root, seeds):
    """Include local import dependencies and package initializers, not the repo."""
    pending, found = list(seeds), set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(relative)
        parts = Path(relative).parts
        for count in range(1, len(parts)):
            initializer = Path(*parts[:count]) / '__init__.py'
            if (root / initializer).is_file():
                pending.append(initializer.as_posix())
        package = list(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = '.'.join(prefix + ([node.module] if node.module else []))
                modules = [module, *(module + '.' + alias.name for alias in node.names)]
            for module in modules:
                for candidate in (Path(*module.split('.')).with_suffix('.py'), Path(*module.split('.')) / '__init__.py'):
                    if (root / candidate).is_file():
                        pending.append(candidate.as_posix())
    return found


def _distribution_arrays(model, frame, protocol):
    """Complete marginal components plus their consumer probabilities."""
    import numpy as np

    predictions = model.predict(frame)
    result = {'best_of': frame['best_of'].to_numpy(int)}
    parameters = {'kills': ('kills_mu', 'kills_alpha'),
                  'duration': ('duration_log_mu', 'duration_log_sigma'),
                  'length': ('length_probs',)}
    for task, prediction in predictions.items():
        components = prediction.components if hasattr(prediction, 'components') else (prediction,)
        weights = prediction.weights if hasattr(prediction, 'weights') else np.array([1.])
        result[f'{task}/weights'] = np.asarray(weights)
        for index, component in enumerate(components):
            for parameter in parameters[task]:
                result[f'{task}/{index}/{parameter}'] = getattr(component, parameter)
        lines = protocol['fixed_lines']['duration_minutes' if task == 'duration' else task]
        method = getattr(prediction, {'kills': 'kills_over', 'duration': 'duration_over', 'length': 'maps_over'}[task])
        for line in lines:
            result[f'{task}/over/{line}'] = method(line)
        if task == 'length':
            result['length/pmf'] = sum(weight * component.length_probs for weight, component in zip(weights, components))
    return result


def create_bundle(root, run_id, bundle):
    import joblib
    import numpy as np
    import torch
    from src.models import prop_distribution_runner as runner

    root, bundle = Path(root).resolve(), Path(bundle)
    if root != runner.ROOT.resolve():
        raise ValueError('Create must use the original repository root')
    if bundle.exists():
        raise FileExistsError(bundle)
    paths, protocol, binding, frame = runner._load_frozen(run_id)
    decision = json.loads((paths['08_reporting'] / 'decision.json').read_text())
    if decision['status'] != 'COMPLETED':
        raise ValueError('Only completed studies can be preserved')
    model_path = paths['06_models'] / 'selected_model.joblib'
    model_relative = model_path.relative_to(root).as_posix()
    if file_hash(model_path) != decision['output_hashes'][model_relative]:
        raise ValueError('Selected model no longer matches the study decision')
    selection_path = paths['06_models'] / 'selection.json'
    selection_relative = selection_path.relative_to(root).as_posix()
    if file_hash(selection_path) != decision['output_hashes'][selection_relative]:
        raise ValueError('Selection no longer matches the study decision')
    protected = dict(binding['protected_hashes'])
    protected.update({path: file_hash(root / path) for path in (model_relative, selection_relative)})
    model = joblib.load(model_path)
    if model.selected != json.loads(selection_path.read_text())['selected']:
        raise ValueError('Selected model and selection disagree')
    assessment = frame.loc[runner.partition_rows(frame)['ASSESS']].copy()
    assessment['match_id'] = assessment['match_id'].astype(str)
    samples = []
    for best in (1, 3, 5):
        rows = assessment.loc[assessment['best_of'] == best].sort_values(['date', 'match_id']).iloc[:3]
        if len(rows) != 3:
            raise ValueError(f'Insufficient BO{best} recovery examples')
        samples.append(rows)
    import pandas as pd
    sample = pd.concat(samples, ignore_index=True)
    torch.set_num_threads(1)
    reference = io.BytesIO()
    np.savez_compressed(reference, **_distribution_arrays(model, sample, protocol))
    sources = _source_closure(root, (*runner.CODE_PATHS, 'src/analysis/prop_model_preservation.py', 'scripts/preserve_prop_model.py'))
    files = set(sources)
    files.update(('docs/RESEARCH.md', 'scripts/README.md'))
    for directory in paths.values():
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
                if path.is_symlink():
                    raise ValueError(f'Refusing source symlink: {path}')
                files.add(path.relative_to(root).as_posix())
    for relative in files:
        expected = decision['output_hashes'].get(relative)
        if expected is not None and file_hash(root / relative) != expected:
            raise ValueError(f'Study output changed: {relative}')
    manifest = {
        'schema': SCHEMA, 'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'PREPARED_HISTORICAL_INFERENCE_AND_STUDY_EVIDENCE_NOT_DEPLOYMENT',
        'selected': model.selected, 'task_decisions': {task: item['status'] for task, item in decision['tasks'].items()},
        'model_path': model_relative, 'dataset_path': binding['dataset_path'],
        'protocol_path': (paths['05_model_input'] / 'protocol.json').relative_to(root).as_posix(),
        'feature_columns': model.feature_columns, 'dataset_rows': len(frame),
        'replay_match_ids': sample['match_id'].tolist(), 'replay_best_of': sample['best_of'].astype(int).tolist(),
        'python': platform.python_version(), 'packages': _versions(), 'source_files': sorted(sources),
        'protected_hashes_before': protected,
        'external_assets_not_bundled': {path: digest for path, digest in {**binding['pins'], **binding['protected_hashes']}.items() if path not in files},
        'limitations': [
            'Known actual first-map rosters and prior-day releases are historical scenarios, not authenticated forecasts.',
            'Map 1 kills/duration and series length are separate marginals, not winner, later-map, correct-score or parlay forecasts.',
            'Raw-to-feature reconstruction and the original runner guards need the listed external assets; they are not in this bundle.',
            'Requires the recorded Python/package environment. Checksums are integrity checks, not authentication of untrusted pickle.',
            'A second copy on the same disk is not an off-machine backup.',
        ], 'members': {},
    }
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with bundle.open('xb') as output, ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(files):
            path = root / relative
            archive.write(path, relative)
            manifest['members'][relative] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
        content = reference.getvalue()
        archive.writestr(REFERENCE, content)
        manifest['members'][REFERENCE] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    with ZipFile(bundle) as archive:
        _validate_archive(archive)
    if any(file_hash(root / path) != digest for path, digest in protected.items()):
        raise ValueError('Protected artifacts changed during preservation')
    return {'run_id': run_id, 'bundle': str(bundle), 'sha256': file_hash(bundle), 'bytes': bundle.stat().st_size,
            'members': len(manifest['members']) + 1, 'dataset_rows': len(frame), 'replay_cases': len(sample),
            'protected_unchanged': len(protected), 'selected': model.selected, 'fits': 0}


def replay_restored(root, output):
    """Run in a fresh process importing the restored source, not the live repo."""
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if platform.python_version() != manifest['python'] or _versions() != manifest['packages']:
        raise ValueError('Restored inference requires the recorded Python/package versions')
    for relative, expected in manifest['members'].items():
        if file_hash(root / relative) != expected['sha256']:
            raise ValueError(f'Restored member changed: {relative}')
    model = joblib.load(root / manifest['model_path'])
    frame = pd.read_parquet(root / manifest['dataset_path'])
    frame['match_id'] = frame['match_id'].astype(str)
    if frame['match_id'].duplicated().any():
        raise ValueError('Recovery data contains duplicate match identities')
    sample = frame.set_index('match_id').loc[manifest['replay_match_ids']].reset_index()
    torch.set_num_threads(1)
    arrays = _distribution_arrays(model, sample, json.loads((root / manifest['protocol_path']).read_text()))
    local_modules = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name == 'src' or name.startswith(('src.', 'scripts.'))):
            absolute = Path(path).resolve()
            if not absolute.is_relative_to(root):
                raise ValueError(f'Inference imported live repository source: {name}')
            local_modules[name] = absolute.relative_to(root).as_posix()
    with Path(output).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return {'cases': len(sample), 'match_ids': manifest['replay_match_ids'], 'best_of': arrays['best_of'].tolist(),
            'arrays': len(arrays), 'source_modules': local_modules}


def verify_bundle(bundle, expected_sha256, root):
    import numpy as np

    root = Path(root).resolve()
    with TemporaryDirectory(prefix='prop-model-recovery-') as temporary:
        restored = Path(temporary) / 'restored'
        manifest = restore_bundle(bundle, expected_sha256, restored)
        replayed = restored / 'recovery/replayed.npz'
        environment = {**os.environ, 'PYTHONPATH': str(restored), 'PYTHONNOUSERSITE': '1'}
        process = subprocess.run([sys.executable, str(restored / 'scripts/preserve_prop_model.py'), 'replay',
                                  '--root', str(restored), '--output', str(replayed)],
                                 cwd=restored, env=environment, capture_output=True, text=True, check=True)
        replay = json.loads(process.stdout)
        comparisons, scalar_values = 0, 0
        with np.load(restored / REFERENCE, allow_pickle=False) as expected, np.load(replayed, allow_pickle=False) as actual:
            if set(expected.files) != set(actual.files):
                raise ValueError('Recovery distribution inventory differs')
            for name in expected.files:
                if not np.array_equal(expected[name], actual[name]):
                    raise ValueError(f'Recovery distribution differs: {name}')
                comparisons += 1
                scalar_values += expected[name].size
        unchanged = {path: file_hash(root / path) == digest for path, digest in manifest['protected_hashes_before'].items()}
        if not all(unchanged.values()):
            raise ValueError('Original protected artifacts changed since preservation')
    return {'schema': SCHEMA, 'status': 'VERIFIED_EXACT_RESTORED_INFERENCE', 'run_id': manifest['run_id'],
            'verified_at': datetime.now(timezone.utc).isoformat(), 'bundle': str(bundle), 'bundle_sha256': expected_sha256,
            'bundle_bytes': Path(bundle).stat().st_size, 'members': len(manifest['members']) + 1,
            'dataset_rows': manifest['dataset_rows'], 'source_files': manifest['source_files'],
            'replay': replay, 'exact_array_comparisons': comparisons, 'scalar_values_compared': scalar_values,
            'max_absolute_difference': 0.0, 'protected_unchanged': unchanged,
            'packages': manifest['packages'], 'fits': 0, 'original_artifacts_moved_or_deleted': 0,
            'off_machine_backup': False, 'limitations': manifest['limitations']}
