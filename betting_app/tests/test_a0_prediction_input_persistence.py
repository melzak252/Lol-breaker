import hashlib

import numpy as np

from betting_app.services import a0_shadow_readiness as readiness
from src.analysis.ev_asof_replay import AsOfSnapshot


class _Adapter:
    identity = {"name": "causal_a0_native_annual", "version": "test"}

    def predict_snapshot(self, snapshot, *, mode):
        hashes = {
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for group in (snapshot.raw, snapshot.hist) for key, value in group.items()
        }
        return type("Prediction", (), {
            "prob_a": 0.6, "prob_b": 0.4, "best_of": 3, "model_name": "causal_a0_native_annual",
            "model_version": "test", "feature_version": "native",
            "diagnostics": {"input_hashes": hashes, "information_status": "local_fixture",
                            "generated_at": "2026-01-01T10:30:00+00:00",
                            "simulated_origin_at": "2026-01-01T10:00:00+00:00"},
        })()


def test_offline_cli_refuses_to_replace_prediction_output(tmp_path, monkeypatch):
    import json
    from argparse import Namespace

    import pytest

    from scripts import predict_causal_a0 as cli

    fixture = tmp_path / "fixtures.json"
    metadata = tmp_path / "metadata.json"
    sources = tmp_path / "sources.json"
    releases = tmp_path / "releases.json"
    output = tmp_path / "out.json"
    fixture.write_text("[]")
    metadata.write_text("[]")
    sources.write_text("[]")
    releases.write_text("{}")
    monkeypatch.setattr(cli, "predict_fixture_inputs", lambda *a, **k: [])
    monkeypatch.setattr(cli, "infer_fixture_snapshots", lambda *a, **k: [])
    args = Namespace(research_root=None, fixtures=fixture, metadata=metadata,
                     source_matches=sources, release_days=releases, roster_evidence=None,
                     max_fixtures=1, mode="full", output=output)
    cli.run(args)
    saved = json.loads(output.read_text())
    assert saved["source_authentication"] == "not_assessed_local_declared_inputs"
    with pytest.raises(FileExistsError, match="immutable output"):
        cli.run(args)



def test_record_a0_snapshot_round_trips_evidence_status_in_isolated_postgres():
    import os
    from uuid import uuid4

    import pytest
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from betting_app.services.forecast_evidence import load_forecast_evidence

    url = os.environ.get("TEST_DATABASE_URL")
    if not url or "_test" not in url:
        pytest.skip("requires an explicitly isolated TEST_DATABASE_URL")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                snapshot = AsOfSnapshot({
                    "row_id": "persist-1", "golgg_match_id": "persist-1",
                    "team1_id": "A", "team2_id": "B", "best_of": 3,
                    "roster_status": "unresolved", "competition_context": "League",
                    "roster_source_a": {"source_type": "unresolved", "known_slots": 0},
                    "roster_source_b": {"source_type": "unresolved", "known_slots": 0},
                    "cutoff_day": "2026-01-01",
                }, {"gates": np.zeros((2, 2), dtype=np.float32)},
                    {"mask": np.zeros((2, 5, 16), dtype=np.uint8)}, True, None)
                key = str(uuid4())
                readiness.record_a0_snapshot(
                    session, _Adapter(), snapshot, evidence_key=key, sources=[],
                    input_source_ids=[], schedule={}, quotes=[],
                )
                stored = load_forecast_evidence(session, key)
                assert stored["input"]["evidence_status"]["input_status"] == "LIMITED_EVIDENCE"
                assert stored["input"]["evidence_status"]["predictive_reliability"] == "NOT_ASSESSED"
                assert stored["forecast"]["simulated_origin_at"] != stored["forecast"]["generated_at"]
            transaction.rollback()
    finally:
        engine.dispose()