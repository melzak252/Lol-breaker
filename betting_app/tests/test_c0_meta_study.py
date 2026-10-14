"""Consumer invariants of the bounded meta research recipe."""
import numpy as np
import pandas as pd
import pytest

from src.models.c0_meta_study import MetaNormalizer, combine_seed_logits, fit_amplitude, patch_phases, temporal_partitions


def test_closed_partition_excludes_labels_released_on_fit_boundary():
    frame = pd.DataFrame({
        'date': ['2024-12-30', '2024-12-31', '2025-12-30', '2025-12-31', '2026-01-01'],
        'result_day': ['2024-12-30', '2024-12-31', '2025-12-30', '2025-12-31', '2026-01-01'],
        'effective_label_release_day': ['2024-12-30', '2025-01-01', '2025-12-30', '2026-01-01', '2026-01-01'],
    })
    split = temporal_partitions(frame)
    np.testing.assert_array_equal(split['train'], [0])
    np.testing.assert_array_equal(split['cal'], [2])
    np.testing.assert_array_equal(split['assessment'], [4])
    np.testing.assert_array_equal(split['excluded_fit_labels'], [1, 3])


def _summaries():
    stats = np.zeros((4, 2, 5, 40), dtype=np.float32)
    stats[..., :12] = np.arange(4)[:, None, None, None] + 2
    stats[..., 24:36] = 1
    stats[..., 36] = np.log1p(3)
    return {
        'player_stats': stats.copy(), 'meta_stats': stats.copy(),
        'meta_support': np.ones((4, 2, 5, 2), dtype=np.float32),
        'meta_patch': np.ones((4, 2, 5, 2), dtype=np.float32),
        'player_valid': np.ones((4, 2, 5), dtype=bool),
    }


def test_future_rows_cannot_change_training_scaler_and_missing_statistics_stay_masked():
    arrays = _summaries()
    normalizer = MetaNormalizer.fit(arrays, np.array([0, 1]))
    changed = {k: v.copy() for k, v in arrays.items()}
    changed['player_stats'][2:, ..., :24] = 1e8
    changed['meta_stats'][2:, ..., :24] = -1e8
    other = MetaNormalizer.fit(changed, np.array([0, 1]))
    assert normalizer.to_dict() == other.to_dict()
    arrays['player_stats'][0, 0, 0, :12] = 100
    arrays['player_stats'][0, 0, 0, 24:36] = 0
    arrays['player_valid'][0, 1, 0] = False
    out = normalizer.transform(arrays)
    np.testing.assert_array_equal(out['player_stats'][0, 0, 0, :24], np.zeros(24))
    np.testing.assert_array_equal(out['player_stats'][0, 1, 0], np.zeros(40))
    np.testing.assert_array_equal(arrays['player_stats'][0, 0, 0, :12], np.full(12, 100))


def test_seed_aggregation_is_one_learned_logit_correction_not_probability_blending():
    base = np.array([.2, .7])
    z = np.log(base) - np.log1p(-base)
    seed_logits = np.array([z + .2, z - .1, z + .05])
    combined = combine_seed_logits(base, seed_logits)
    np.testing.assert_allclose(combined, z + .05, atol=1e-15, rtol=0)
    reverse = combine_seed_logits(1 - base, -seed_logits)
    np.testing.assert_allclose(reverse, -combined, atol=1e-15, rtol=0)


def test_calibration_amplitude_finds_convex_interior_optimum():
    base = np.full(3, .6)
    delta = np.full(3, .5)
    fitted = fit_amplitude(base, delta, np.array([1, 1, 0]))
    expected = (np.log(2) - np.log(1.5)) / .5
    assert fitted['amplitude'] == pytest.approx(expected, abs=1e-9)
    assert fitted['cal_log_loss'] < fitted['identity_log_loss']


def test_calibration_can_retain_identity_or_full_correction_without_clipping():
    bad = fit_amplitude(np.full(4, .8), np.full(4, .5), np.array([1, 0, 1, 0]))
    good = fit_amplitude(np.full(4, .5), np.full(4, .5), np.ones(4))
    assert bad['amplitude'] == 0
    assert good['amplitude'] == 1
    with pytest.raises(ValueError):
        fit_amplitude(np.array([0.]), np.array([.1]), np.array([0]))


def test_patch_transition_profiles_use_released_local_evidence_not_future_dates():
    observations = [
        dict(competition='A', patch='14.1', played_day='2024-01-01', release_day='2024-01-02'),
        dict(competition='A', patch='14.2', played_day='2024-01-04', release_day='2024-01-10'),
    ]
    targets = [
        dict(competition='A', patch='14.1', cutoff_day='2024-01-03'),
        dict(competition='A', patch='14.2', cutoff_day='2024-01-05'),
        dict(competition='A', patch='14.2', cutoff_day='2024-01-11'),
        dict(competition='A', patch=None, cutoff_day='2024-01-11'),
        dict(competition='A', patch=np.nan, cutoff_day='2024-01-11'),
        dict(competition='B', patch='14.2', cutoff_day='2024-01-11'),
    ]
    phases = patch_phases(observations, targets)
    assert phases == ['first7days', 'unseen_in_prior_context', 'later', 'unknown', 'unknown', 'no_prior_patch_context']
    changed = observations + [dict(competition='A', patch='future', played_day='2024-01-05', release_day='2024-02-01')]
    assert patch_phases(changed, targets) == phases


def test_saved_head_origin_guard_and_zero_amplitude_identity(tmp_path, monkeypatch):
    import json
    import torch
    from src.models import c0_meta_study as study
    from src.models.c0_meta import C0MetaCorrection

    monkeypatch.setattr(study, 'PROJECT_ROOT', tmp_path)
    arrays = _summaries()
    arrays.update(
        profile=np.zeros((4, 2, 5, 96), dtype=np.float32),
        player_roles=np.zeros((4, 2, 5, 5), dtype=np.float32),
        player_champions=np.zeros((4, 2, 5, 2), dtype=np.float32),
        meta_champions=np.zeros((4, 2, 5, 2), dtype=np.float32),
        base_probability=np.array([.6, .01, .2, .7], dtype=np.float64),
        best_of=np.ones(4, dtype=np.int64),
    )
    arrays['profile'][:, 0, :, 0] = 1
    arrays['profile'][:, 1, :, 0] = -1
    features = tmp_path / 'features.npz'
    np.savez(features, **arrays)
    metadata = tmp_path / 'metadata.parquet'
    pd.DataFrame({'date': ['2025-12-31', '2026-01-02', '2026-01-03', '2026-01-04']}).to_parquet(metadata)
    hashes = {}
    for branch in ('player', 'meta'):
        model = C0MetaCorrection(2, branch == 'meta')
        with torch.no_grad():
            model.readout.weight.fill_(.2)
        path = tmp_path / branch / 'seed_1.pt'
        path.parent.mkdir()
        torch.save(dict(state_dict=model.state_dict(), vocabulary_size=2, use_meta=branch == 'meta',
                        protocol_sha256='fixture'), path)
        hashes[str(path)] = study.sha(path)
    bundle = tmp_path / 'bundle.json'
    bundle.write_text(json.dumps(dict(
        seeds=[1], model_sha256=hashes, protocol_sha256='fixture', origin_day='2026-01-01',
        features_sha256=study.sha(features), frame_sha256=study.sha(metadata),
        normalizer=MetaNormalizer.fit(arrays, np.array([0, 1])).to_dict(),
        inference_source_sha256={str(study.__file__): study.sha(study.__file__)},
        calibrations={'player': {'amplitude': 0.}, 'meta': {'amplitude': 0.}},
    )))
    with pytest.raises(ValueError):
        study.predict_saved(bundle, features, [0, 1])
    result = study.predict_saved(bundle, features, [1, 2, 3])
    for branch in ('player', 'meta'):
        np.testing.assert_array_equal(result[branch], arrays['base_probability'][[1, 2, 3]])
    # A relocated bundle must not load copies outside its verified inventory.
    import shutil
    copied = tmp_path / 'unverified'
    copied.mkdir()
    shutil.copyfile(bundle, copied / 'bundle.json')
    for branch in ('player', 'meta'):
        shutil.copytree(tmp_path / branch, copied / branch)
        checkpoint_path = copied / branch / 'seed_1.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        checkpoint['state_dict']['readout.weight'].fill_(.9)
        torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError):
        study.predict_saved(copied / 'bundle.json', features, [1, 2, 3])
