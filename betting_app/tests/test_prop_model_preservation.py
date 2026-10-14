"""Recovery must reject corrupt or unsafe bundles without damaging existing files."""
import hashlib
import json
from zipfile import ZipFile

import pytest

from src.analysis.prop_model_preservation import restore_bundle


def make_bundle(tmp_path, member='data/model.joblib', content=b'original model', declared_content=None):
    declared = content if declared_content is None else declared_content
    manifest = {
        'schema': 'prop_model_preservation_v1',
        'members': {member: {'sha256': hashlib.sha256(declared).hexdigest(), 'bytes': len(declared)}},
    }
    bundle = tmp_path / 'model.zip'
    with ZipFile(bundle, 'w') as archive:
        archive.writestr(member, content)
        archive.writestr('recovery/manifest.json', json.dumps(manifest))
    return bundle, hashlib.sha256(bundle.read_bytes()).hexdigest()


def test_restore_rejects_wrong_bundle_digest_before_creating_destination(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    destination = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Bundle SHA-256 mismatch'):
        restore_bundle(bundle, '0' * 64, destination)
    assert not destination.exists()


def test_restore_rejects_corrupt_member_even_when_zip_digest_matches(tmp_path):
    bundle, digest = make_bundle(tmp_path, content=b'changed model', declared_content=b'original model')
    destination = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Member integrity mismatch'):
        restore_bundle(bundle, digest, destination)
    assert not destination.exists()


@pytest.mark.parametrize('member', ['../outside.joblib', '/outside.joblib', 'data/../outside.joblib'])
def test_restore_rejects_escaping_paths_without_writing_files(tmp_path, member):
    bundle, digest = make_bundle(tmp_path, member=member)
    destination = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Unsafe archive member'):
        restore_bundle(bundle, digest, destination)
    assert not destination.exists()
    assert not (tmp_path / 'outside.joblib').exists()


def test_restore_never_overwrites_existing_destination(tmp_path):
    bundle, digest = make_bundle(tmp_path)
    destination = tmp_path / 'restored'
    destination.mkdir()
    sentinel = destination / 'user-model.joblib'
    sentinel.write_bytes(b'user work')
    with pytest.raises(FileExistsError):
        restore_bundle(bundle, digest, destination)
    assert sentinel.read_bytes() == b'user work'
    assert not (destination / 'data').exists()


def test_restore_rejects_unmanifested_executable(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    with ZipFile(bundle, 'a') as archive:
        archive.writestr('scripts/unexpected.py', b'raise RuntimeError("unexpected")')
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    destination = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Archive inventory mismatch'):
        restore_bundle(bundle, digest, destination)
    assert not destination.exists()
