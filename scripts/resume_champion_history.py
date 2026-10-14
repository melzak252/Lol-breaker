#!/usr/bin/env python3
"""One explicitly authorized recovery of the interrupted frozen research run.

The original runner, model, features, calibration and gates remain byte-identical.
Completed seed artifacts are restored, not retrained. The lost D attempt is
repeated from its original seed; its discarded progress is counted separately.
This is not an automatic retry or a new experiment recipe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import champion_history_model as model_module
from src.models import champion_history_runner as runner


def main():
    paths = runner._paths(runner.RUN_ID)
    report = paths['08_reporting']
    authorization_path = paths['05_model_input'] / 'resume_authorization.json'
    authorization = json.loads(authorization_path.read_text())
    interrupted_path = report / 'execution_interrupted.json'
    original = json.loads(interrupted_path.read_text())
    if runner.sha256(interrupted_path) != authorization['interrupted_execution_sha256']:
        raise ValueError('Interrupted execution differs from authorized recovery')
    if runner.sha256(Path(__file__)) != authorization['recovery_code_sha256']:
        raise ValueError('Recovery code differs from authorization binding')
    if original['status'] != 'RUNNING' or original['calibration_fits']:
        raise ValueError('Recovery is only valid before CAL or assessment')
    if authorization['user_decision'] != 'Resume the identical recipe':
        raise ValueError('Explicit lost-attempt repeat authorization is required')
    if (report / 'execution.json').exists() or (report / 'resumption_start.json').exists():
        raise FileExistsError('Recovery already started; no automatic retry')
    completed = { (r['seed'], *runner.VARIANTS[r['variant']]): r
                 for r in original['formal_trajectories'] if r['status'] == 'COMPLETED' }
    interrupted = [r for r in original['formal_trajectories'] if r['status'] != 'COMPLETED']
    if len(completed) != 3 or [(r['variant'], r['seed']) for r in interrupted] != [('D', 20261007)]:
        raise ValueError('Checkpoint population differs from the authorized recovery')
    for record in completed.values():
        if runner.sha256(ROOT / record['model_path']) != record['sha256']:
            raise ValueError('Completed checkpoint changed')
    archive = report / 'interrupted_model_checkpoints'
    archive.mkdir(exist_ok=False)
    for record in completed.values():
        destination = archive / Path(record['model_path']).name
        shutil.copyfile(ROOT / record['model_path'], destination)
        if runner.sha256(destination) != record['sha256']:
            raise ValueError('Original checkpoint byte archive differs')
    real_fit = model_module.fit_history_model
    reused, new_fits = [], []

    def restore_or_fit(arrays, series, maps, indices, *, config, seed, champion_enabled, multitask):
        key = (seed, champion_enabled, multitask)
        if key not in completed:
            new_fits.append({'seed': seed, 'champion_enabled': champion_enabled, 'multitask': multitask})
            print(f'ACTUAL FIT seed={seed} champion={champion_enabled} multitask={multitask}', flush=True)
            return real_fit(arrays, series, maps, indices, config=config, seed=seed,
                            champion_enabled=champion_enabled, multitask=multitask)
        record = completed.pop(key)
        checkpoint = joblib.load(ROOT / record['model_path'])
        receipt = checkpoint.fit_receipt
        index_hash = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
        if (receipt != record['fit_receipt'] or checkpoint.config != config or
                receipt['train_indices_sha256'] != index_hash or receipt['epochs_completed'] != 12 or
                checkpoint.seed != seed or checkpoint.champion_enabled != champion_enabled or
                checkpoint.multitask != multitask):
            raise ValueError('Checkpoint does not belong to this frozen trajectory')
        reused.append(record)
        print(f'RESTORED {record["variant"]} seed={seed}; no fit', flush=True)
        return checkpoint

    runner._json(report / 'resumption_start.json', {
        'started_at': runner._now(), 'authorization_sha256': runner.sha256(authorization_path),
        'interrupted_execution_sha256': runner.sha256(interrupted_path),
        'expected_restored_models': 3, 'expected_new_fit_attempts': 9,
        'expected_completed_trajectories': 12, 'expected_total_formal_fit_attempts': 13,
        'interrupted_epochs_completed': 'UNKNOWN; no epoch checkpoint survived',
        'recipe_or_gate_changes': 0, 'assessment_fits': 0,
    })
    model_module.fit_history_model = restore_or_fit
    try:
        decision = runner.run_experiment(runner.RUN_ID)
    finally:
        model_module.fit_history_model = real_fit
    if completed or len(reused) != 3 or len(new_fits) != 9:
        raise ValueError('Actual recovery accounting differs from authorization')
    # The runner reserializes restored objects. Preserve and explicitly reconcile
    # original-byte checkpoints below; fit receipts identify the actual training.
    execution_path = report / 'execution.json'
    execution = json.loads(execution_path.read_text())
    for old in reused:
        record = next(r for r in execution['formal_trajectories']
                      if r['variant'] == old['variant'] and r['seed'] == old['seed'])
        record['training_started_at'] = old['started_at']
        record['training_completed_at'] = old['completed_at']
        record['restored_checkpoint_sha256'] = old['sha256']
        record['checkpoint_restored_without_fit'] = True
    accounting = {
        'formal_completed_trajectories': 12, 'formal_fit_attempts': 13,
        'interrupted_fit_attempts': 1, 'restored_completed_checkpoints': 3,
        'resumed_actual_fit_attempts': 9, 'interrupted_partial_seed_epochs': 'UNKNOWN',
        'authorization_sha256': runner.sha256(authorization_path),
        'original_execution_sha256': runner.sha256(interrupted_path),
        'no_recipe_or_gate_change': True,
    }
    execution.update(accounting)
    decision.update(accounting)
    runner._json(execution_path, execution)
    runner._json(report / 'decision.json', decision)
    runner._json(report / 'resumption_completed.json', {
        **accounting, 'status': 'COMPLETED', 'completed_at': runner._now(),
        'reused_original_records': reused, 'new_fit_attempts': new_fits,
    })
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    main()
