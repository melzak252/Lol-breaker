"""TRAIN-only scaling and public saved stable-profile inference boundaries."""
import numpy as np
import pytest

from src.models.c0_meta_stable_study import StableNormalizer
from src.models.c0_meta_study import MetaNormalizer


def arrays():
    stats = np.zeros((3, 2, 5, 40), np.float32)
    stats[..., :12] = np.arange(3)[:, None, None, None] + 2
    stats[..., 24:36] = 1
    stats[..., 36] = np.log1p(2)
    activity = np.ones((3, 2, 5, 8), np.float32)
    activity[..., :3] = np.arange(3)[:, None, None, None] + 1
    return dict(player_stats=stats, meta_stats=stats.copy(), player_valid=np.ones((3, 2, 5), bool),
                meta_support=np.ones((3, 2, 5, 2), np.float32),
                meta_patch=np.ones((3, 2, 5, 2), np.float32), player_activity=activity)


def test_scaling_fits_closed_train_only_and_preserves_original_meta_transform():
    raw = arrays()
    original = MetaNormalizer.fit(raw, [0, 1])
    fitted = StableNormalizer.fit(raw, [0, 1], original)
    changed = {key: value.copy() for key, value in raw.items()}
    changed['player_stats'][2, ..., :12] = 99999
    changed['player_activity'][2, ..., :3] = 99999
    assert fitted.to_dict() == StableNormalizer.fit(changed, [0, 1], original).to_dict()
    transformed = fitted.transform(raw)
    prior = original.transform(raw)
    for key in ('meta_stats', 'meta_support', 'meta_patch'):
        np.testing.assert_array_equal(transformed[key], prior[key])
    np.testing.assert_array_equal(transformed['player_activity'][..., 3:], raw['player_activity'][..., 3:])
    np.testing.assert_allclose(transformed['player_stats'][0, ..., :12], -1)
    np.testing.assert_allclose(transformed['player_stats'][1, ..., :12], 1)


def test_missing_performance_and_unknown_inactivity_remain_explicit_not_imputed():
    raw = arrays()
    original = MetaNormalizer.fit(raw, [0, 1])
    fitted = StableNormalizer.fit(raw, [0, 1], original)
    raw['player_stats'][2, ..., 24:36] = 0
    raw['player_activity'][2, ..., 3] = 0
    raw['player_activity'][2, ..., 2] = 999
    transformed = fitted.transform(raw)
    assert not transformed['player_stats'][2, ..., :24].any()
    assert not transformed['player_activity'][2, ..., 2].any()
    replay = StableNormalizer.from_dict(fitted.to_dict()).transform(raw)
    for key in transformed:
        np.testing.assert_array_equal(replay[key], transformed[key])


def test_prior_feature_and_receipt_cannot_be_mutated_and_repinned_together(tmp_path):
    import json
    from src.models import c0_meta_stable_study as study
    from src.models.c0_meta_study import sha
    root = tmp_path
    feature = root / 'data/features.npz'
    feature.parent.mkdir()
    feature.write_bytes(b'authenticated-frozen-player-history')
    prepared = root / 'data/prepared.json'
    prepared.write_text(json.dumps({'features_sha256': sha(feature)}))
    seal = root / 'seal.json'
    seal.write_text(json.dumps(dict(schema='c0_meta_sealed_verification_v1',run_id='prior',
        status='COMPLETED',decision='NOT_ADOPTED',protocol_sha256='a'*64,
        artifact_sha256={'data/features.npz':sha(feature),'data/prepared.json':sha(prepared)})))
    expected = sha(seal)
    study._authenticate_prior_seal(seal,expected,root,'prior','a'*64)
    feature.write_bytes(b'replacement-future-trained-player-history')
    prepared.write_text(json.dumps({'features_sha256':sha(feature)}))
    with pytest.raises(ValueError,match='prior sealed artifact changed'):
        study._authenticate_prior_seal(seal,expected,root,'prior','a'*64)
    with pytest.raises(ValueError,match='prior seal changed'):
        seal.write_text(seal.read_text()+' ')
        study._authenticate_prior_seal(seal,expected,root,'prior','a'*64)
