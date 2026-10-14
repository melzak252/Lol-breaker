import hashlib

import pytest

from src.models.a0_capacity_runner import _validate_run_id, verify_protocol_hashes


def test_run_id_rejects_path_traversal_before_creating_artifacts():
    with pytest.raises(ValueError, match="safe path component"):
        _validate_run_id("../outside")


def test_protocol_verification_rejects_missing_or_changed_transitive_input(tmp_path):
    source = tmp_path / "archive.py"
    source.write_text("original")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = {"inputs": {str(source): expected}, "source_snapshot": {}}
    verify_protocol_hashes(protocol, tmp_path)

    source.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol_hashes(protocol, tmp_path)




def test_cli_rejects_protocol_and_freeze_modes_together():
    from scripts.train_a0_capacity_experiment import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--freeze-run", "safe", "--protocol", "protocol.json"])
