"""PostgreSQL regressions for persisted model identity and as-of odds selection."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from threading import Event
from time import monotonic, sleep

import pytest
from sqlalchemy import event, select, text

from betting_app.core.db import get_session
from betting_app.core.models import BAYESIAN_SHRUNK_HYBRID, C0_NATIVE
from betting_app.core.models import PredictionEngine, get_active_hybrid
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import OddsSnapshot
from betting_app.models.prediction import (
    CanonicalPrediction,
    ModelArtifact,
    UpcomingMatchFeature,
)
from betting_app.models.golgg import GolggGame, GolggGamePlayer, GolggMatch, GolggTeam
from betting_app.services import mapping_service
from betting_app.services import current_roster_service
from betting_app.services import upcoming_inference_service as service
# Native C0 request fixture; persistence tests do not depend on GL/W20 features.
def _native_features(match_id=0, *, start=None, team_a="Alpha", team_b="Beta"):
    start = start or (datetime.now(UTC) + timedelta(days=1)).isoformat()
    return {
        "canonical": {
            "id": match_id, "team_a_name": team_a, "team_b_name": team_b,
            "start_time_normalized": start, "best_of": 3, "league": "Test",
        },
        "c0_request": {
            "team1_id": "team-a", "team2_id": "team-b",
            "roster_a": [f"a{i}" for i in range(5)],
            "roster_b": [f"b{i}" for i in range(5)],
            "best_of": 3, "decision_at": datetime.now(UTC).isoformat(),
            "competition_context": "Test", "start_at": start, "mode": "full",
        },
    }


def _match(session, start):
    match = CanonicalMatch(
        canonical_key=f"audit-alpha-beta-{start.isoformat()}",
        team_a_name="Alpha", team_b_name="Beta",
        normalized_team_a="alpha", normalized_team_b="beta",
        start_time_normalized=start.isoformat(), best_of=3,
    )
    session.add(match)
    session.flush()
    return match



def test_resolve_exact_canonical_ids_aligns_reversed_linked_offer(monkeypatch):
    import pandas as pd

    match = {
        "id": 812, "team_a_name": "Alpha", "team_b_name": "Beta",
        "league": "LCK", "start_time_normalized": "2026-10-10T12:00:00+00:00",
    }
    offers = pd.DataFrame([{
        "raw_team_a": "Beta", "raw_team_b": "Alpha", "league": "LCK",
        "team_a_golgg_id": 22, "team_b_golgg_id": 11,
    }])
    monkeypatch.setattr(service, "query_df", lambda *args, **kwargs: offers)
    monkeypatch.setattr(
        service,
        "native_golgg_team_id",
        lambda internal_id, **kwargs: f"provider-{internal_id}" if internal_id is not None else None,
    )
    resolved = service.resolve_exact_canonical_team_ids(match)
    assert (resolved["native_team_a_id"], resolved["native_team_b_id"]) == ("provider-11", "provider-22")
    assert (resolved["team_a_golgg_id"], resolved["team_b_golgg_id"]) == (11, 22)

    monkeypatch.setattr(service, "query_df", lambda *args, **kwargs: pd.DataFrame())
    unresolved = service.resolve_exact_canonical_team_ids(match)
    assert unresolved["team_a_golgg_id"] is None
    assert unresolved["team_b_golgg_id"] is None

def test_db_team_row_id_resolves_to_provider_id_without_namespace_leak(monkeypatch):
    from contextlib import nullcontext

    class Result:
        def fetchone(self):
            return {"team_name": "Exact Main", "team_id": 2817}

    class Connection:
        def execute(self, sql, params):
            assert "WHERE id = ?" in sql
            assert params == (1,)
            return Result()

    monkeypatch.setattr(mapping_service, "transaction", lambda: nullcontext(Connection()))
    assert mapping_service.native_golgg_team_id(1, team_name="Exact Main") == "2817"
    assert mapping_service.native_golgg_team_id(1, team_name="Exact Main") != "1"
    with pytest.raises(ValueError, match="does not identify"):
        mapping_service.native_golgg_team_id(1, team_name="Different Main")


@pytest.mark.parametrize("known_side", [None, "a", "b"])
def test_unknown_provider_ids_use_explicit_no_organization_targets(monkeypatch, known_side):
    monkeypatch.setattr(service, "native_golgg_team_id", lambda *args, **kwargs: None)
    roster_a = {"players": [{"player_id": f"a{i}"} for i in range(5)]}
    roster_b = {"players": [{"player_id": f"b{i}"} for i in range(5)]}
    now = datetime.now(UTC)
    request, missing = service._native_c0_request(
        {
            "id": 7, "team_a_name": "Main Team", "team_b_name": "Main Team Academy",
            "team_a_golgg_id": 1, "team_b_golgg_id": 2,
            "native_team_a_id": "2817" if known_side == "a" else None,
            "native_team_b_id": "2818" if known_side == "b" else None,
            "best_of": 3, "_c0_decision_at": now.isoformat(),
            "start_time_normalized": (now + timedelta(hours=1)).isoformat(),
        },
        roster_a,
        roster_b,
    )
    assert request["mode"] == "no_organization"
    assert request["team1_id"] == "no_organization:target:Main%20Team"
    assert request["team2_id"] == "no_organization:target:Main%20Team%20Academy"
    assert "1" not in {request["team1_id"], request["team2_id"]}
    assert missing == []


def test_partial_native_input_never_substitutes_bo1_for_unsupported_format():
    from betting_app.core.models.native_c0 import NativeC0Adapter

    now = datetime.now(UTC)
    request, missing = service._native_c0_request(
        {
            "id": 88, "team_a_name": "Alpha", "team_b_name": "Beta",
            "best_of": 7, "_c0_decision_at": now.isoformat(),
            "start_time_normalized": (now + timedelta(hours=1)).isoformat(),
        },
        {"team_id": "native-a", "players": [{"player_id": f"a{i}"} for i in range(5)]},
        {"team_id": "native-b", "players": [{"player_id": f"b{i}"} for i in range(5)]},
    )
    assert "c0_unsupported_best_of:7" in missing
    with pytest.raises(ValueError, match="best_of"):
        NativeC0Adapter.__new__(NativeC0Adapter)._validate_request(request)



def test_matchup_rejects_database_id_name_mismatch_before_feature_inference(client, monkeypatch):
    with get_session() as session:
        team = GolggTeam(
            team_name="Exact DB Team", team_id=9182738, normalized_name="exact db team",
        )
        session.add(team)
        session.commit()
        row_id = team.id

    def forbidden_inference(*args, **kwargs):
        raise AssertionError("mismatched team identity reached feature inference")

    monkeypatch.setattr(
        "betting_app.services.upcoming_inference_service.build_features_for_match",
        forbidden_inference,
    )
    response = client.post(
        "/matches/matchup",
        json={
            "team_a_name": "Different Team", "team_b_name": "Opponent",
            "team_a_team_row_id": row_id,
        },
    )
    assert response.status_code == 422
    assert "does not identify" in response.json()["detail"]

def test_single_match_inference_resolves_missing_team_ids_before_feature_build(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        service, "resolve_exact_canonical_team_ids",
        lambda match: {**match, "team_a_golgg_id": 11, "team_b_golgg_id": 22},
    )

    def build(match, **kwargs):
        observed.update(team_a_id=match["team_a_golgg_id"], team_b_id=match["team_b_golgg_id"])
        return {
            "status": "ready_player", "missing": [],
            "features": {"c0_request": {}, "canonical": match},
            "data_cutoff_at": None,
        }

    monkeypatch.setattr(service, "build_features_for_match", build)
    monkeypatch.setattr(service, "predict_probability_from_features", lambda *args, **kwargs: (
        0.6, {"data_cutoff_at": "2026-10-05T00:00:00+00:00"},
    ))
    service.predict_operational_match(
        {"id": 42, "team_a_name": "Alpha", "team_b_name": "Beta"},
        model_name=C0_NATIVE.name, model_version=C0_NATIVE.version, persist=False,
    )
    assert observed == {"team_a_id": 11, "team_b_id": 22}


def test_upcoming_batch_skips_closed_fixtures_and_isolates_invalid_rows(client, monkeypatch):
    now = datetime.now(UTC)
    with get_session() as session:
        expired = _match(session, now - timedelta(minutes=1))
        finished = _match(session, now + timedelta(hours=2))
        finished.status = "completed"
        bad = _match(session, now + timedelta(hours=3))
        good = _match(session, now + timedelta(hours=4))
        matches = {"expired": expired, "finished": finished, "bad": bad, "good": good}
        ids = {key: match.id for key, match in matches.items()}
        for match in matches.values():
            feature_data = _native_features(match.id, start=match.start_time_normalized)
            session.add(UpcomingMatchFeature(
                canonical_match_id=match.id, feature_version=C0_NATIVE.feature_version,
                ratings_version=C0_NATIVE.ratings_version, feature_status="ready_player",
                features_json=json.dumps(feature_data),
            ))
        session.commit()
    inferred = []

    def predict(features, **kwargs):
        match_id = features["canonical"]["id"]
        inferred.append(match_id)
        if match_id == ids["bad"]:
            raise ValueError("invalid exact roster")
        return 0.63, {"model_name": C0_NATIVE.name, "model_version": C0_NATIVE.version}

    monkeypatch.setattr(service, "predict_probability_from_features", predict)
    result = service.predict_all_upcoming()
    assert inferred == [ids["bad"], ids["good"]]
    assert [row["canonical_match_id"] for row in result if row.get("prediction_status") == "skipped"] == [
        ids["expired"], ids["finished"], ids["bad"],
    ]
    with get_session() as session:
        predictions = session.scalars(select(CanonicalPrediction).where(
            CanonicalPrediction.canonical_match_id.in_(ids.values()),
        )).all()
        assert [row.canonical_match_id for row in predictions] == [ids["good"]]

def test_native_c0_prediction_records_exact_model_artifact_and_history_cutoff(client, monkeypatch):
    cutoff = "2026-10-05T00:00:00+00:00"
    monkeypatch.setattr(
        service,
        "predict_probability_from_features",
        lambda *args, **kwargs: (0.61, {
            "model_name": C0_NATIVE.name,
            "model_version": C0_NATIVE.version,
            "feature_version": C0_NATIVE.feature_version,
            "data_cutoff_at": cutoff,
            "market_features_used": False,
            "epistemic_sigma_z": None,
            "p_low_a": None,
            "p_low_b": None,
        }),
    )
    with get_session() as session:
        match = _match(session, datetime.now(UTC) + timedelta(days=1))
        c0_features = _native_features(match.id, start=match.start_time_normalized)
        session.add(UpcomingMatchFeature(
            canonical_match_id=match.id, feature_version=C0_NATIVE.feature_version,
            ratings_version=C0_NATIVE.ratings_version, feature_status="ready_player",
            features_json=json.dumps(c0_features),
        ))
        session.commit()
    generated = service.predict_all_upcoming()
    assert len(generated) == 1
    with get_session() as session:
        prediction, artifact = session.execute(select(CanonicalPrediction, ModelArtifact).join(
            ModelArtifact, CanonicalPrediction.model_artifact_id == ModelArtifact.id,
        )).one()
        feature_schema = json.loads(artifact.feature_schema_json)
        model_params = json.loads(artifact.model_params_json)
        diagnostics = json.loads(prediction.diagnostics_json)
        assert (artifact.model_name, artifact.model_version) == (C0_NATIVE.name, C0_NATIVE.version)
        assert feature_schema["input_contract"] == "native-c0-request-v1"
        assert feature_schema["has_uncertainty"] is False
        assert model_params["model_identity"]["name"] == C0_NATIVE.name
        assert model_params["model_identity"]["artifact_hashes"]
        assert prediction.data_cutoff_at == cutoff
        assert diagnostics["market_features_used"] is False
        assert diagnostics["epistemic_sigma_z"] is None


def test_market_consensus_uses_one_available_prematch_quote_per_book(client):
    now = datetime.now(UTC)
    with get_session() as session:
        match = _match(session, now + timedelta(hours=1))
        match_id = match.id
        quotes = [
            (1, now - timedelta(seconds=90), "Alpha", "Beta", 1.5, 3.0),
            (1, now - timedelta(seconds=30), "Beta", "Alpha", 4.0, 4.0 / 3),
            (2, now - timedelta(seconds=30), "Alpha", "Beta", 2.0, 2.0),
            (1, now + timedelta(minutes=30), "Alpha", "Beta", 9.0, 1.125),
            (3, now + timedelta(hours=2), "Alpha", "Beta", 9.0, 1.125),
        ]
        for bookmaker, timestamp, a, b, oa, ob in quotes:
            session.add(OddsSnapshot(
                bookmaker_id=bookmaker, canonical_match_id=match_id, market_type="match_winner",
                raw_team_a=a, raw_team_b=b, odds_a=oa, odds_b=ob, is_live=0, scraped_at=timestamp,
            ))
        session.commit()
    probability = service.fetch_latest_pre_match_market_prob(match_id)
    assert probability == pytest.approx((0.75 + 0.5) / 2)


def test_prediction_history_preserves_recorded_model_identity(client):
    now = datetime.now(UTC)
    with get_session() as session:
        match = _match(session, now + timedelta(days=1))
        match_id = match.id
        session.add(CanonicalPrediction(
            canonical_match_id=match_id, model_name="Hybrid-Operational-Market",
            model_version="exp081-siamese-series-v1-a0.50-t1.00",
            predicted_at=now.isoformat(), prob_a=0.6, prob_b=0.4,
            diagnostics_json=json.dumps({"p_sports": 0.7}),
        ))
        session.commit()
    response = client.get(f"/matches/{match_id}/prediction-history")
    assert response.status_code == 200
    points = response.json()
    assert [(p["model_name"], p["model_version"]) for p in points] == [
        ("Hybrid-Operational-Market", "exp081-siamese-series-v1-a0.50-t1.00")
    ]


def test_batch_hybrid_shrinks_saved_c0_against_available_market_once(client):
    now = datetime.now(UTC)
    base_probability = 0.68
    base_identity = {"name": C0_NATIVE.name, "artifact_hashes": {"checkpoint": "native"}}
    base_cutoff = (now - timedelta(days=1)).isoformat()
    base_diagnostics = {
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "feature_version": C0_NATIVE.feature_version,
        "model_identity": base_identity,
        "feature_provenance": {"history_cutoff": base_cutoff},
        "data_cutoff_at": base_cutoff,
        "market_features_used": False,
        "epistemic_sigma_z": None,
        "p_low_a": None,
        "p_low_b": None,
    }
    with get_session() as session:
        match = _match(session, now + timedelta(hours=1))
        match_id = match.id
        session.add(CanonicalPrediction(
            canonical_match_id=match_id, model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
            predicted_at=(now - timedelta(minutes=5)).isoformat(),
            data_cutoff_at=(now - timedelta(days=1)).isoformat(),
            prob_a=base_probability, prob_b=1.0 - base_probability,
            diagnostics_json=json.dumps(base_diagnostics),
        ))
        for timestamp, oa, ob in [
            (now - timedelta(seconds=30), 4 / 3, 4.0),
            (now + timedelta(minutes=30), 9.0, 1.125),
        ]:
            session.add(OddsSnapshot(
                bookmaker_id=1, canonical_match_id=match_id, market_type="match_winner",
                raw_team_a="Alpha", raw_team_b="Beta", odds_a=oa, odds_b=ob,
                is_live=0, scraped_at=timestamp,
            ))
        session.commit()
    expected_probability = PredictionEngine.blend_with_market(
        base_probability, 0.75, get_active_hybrid(),
    )
    batch = service.generate_hybrid_predictions()
    assert len(batch) == 1
    assert batch[0]["prob_a"] == pytest.approx(expected_probability)
    assert batch[0]["diagnostics"]["model_prob_a"] == pytest.approx(base_probability)
    assert batch[0]["diagnostics"]["market_prob_a_avg_no_vig"] == pytest.approx(0.75)
    with get_session() as session:
        row = session.get(CanonicalPrediction, batch[0]["prediction_id"])
        assert row.model_version == get_active_hybrid().hybrid_model_version
        assert datetime.fromisoformat(row.data_cutoff_at) >= now - timedelta(hours=1)
        assert datetime.fromisoformat(row.data_cutoff_at) <= datetime.fromisoformat(row.predicted_at)
        hybrid_diagnostics = json.loads(row.diagnostics_json)
        assert hybrid_diagnostics["model_identity"] == base_identity
        assert hybrid_diagnostics["base_diagnostics"]["data_cutoff_at"] == base_cutoff
        assert hybrid_diagnostics["base_data_cutoff_at"] == base_cutoff


def test_failed_lineup_refresh_preserves_last_published_forecast(client, monkeypatch):
    now = datetime.now(UTC)
    with get_session() as session:
        match = _match(session, now + timedelta(minutes=30))
        match_id = match.id
        match_data = {
            "id": match_id, "start_time_normalized": match.start_time_normalized,
            "previous_roster_a": [], "previous_roster_b": [],
        }
        session.commit()
    c0_features = _native_features(match_id, start=match_data["start_time_normalized"])
    monkeypatch.setattr(service, "check_confirmed_lineup", lambda **kwargs: {"window_valid": True})
    monkeypatch.setattr(service, "build_features_for_match", lambda *args, **kwargs: {
        "features": c0_features, "data_cutoff_at": (now - timedelta(days=1)).isoformat(),
    })
    cutoff = "2026-10-05T00:00:00+00:00"
    monkeypatch.setattr(service, "predict_probability_from_features", lambda *args, **kwargs: (0.61, {
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "feature_version": C0_NATIVE.feature_version,
        "data_cutoff_at": cutoff,
        "model_identity": {"name": C0_NATIVE.name, "artifact_hashes": {"checkpoint": "observed"}},
        "market_features_used": False,
        "epistemic_sigma_z": None,
        "p_low_a": None,
        "p_low_b": None,
    }))
    arguments = {
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "force_re_infer": True,
    }
    result = service.process_confirmed_lineup_update(match_data, **arguments)
    assert result["prediction_id"] is not None
    assert result["prob_a"] == pytest.approx(0.61)
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, result["prediction_id"])
        artifact = session.get(ModelArtifact, prediction.model_artifact_id)
        assert (artifact.model_name, artifact.model_version) == (prediction.model_name, prediction.model_version)

    def missing_artifact(*args, **kwargs):
        raise FileNotFoundError("Unavailable sports checkpoint")

    monkeypatch.setattr(service, "predict_probability_from_features", missing_artifact)
    with pytest.raises(FileNotFoundError):
        service.process_confirmed_lineup_update(match_data, **arguments)
    with get_session() as session:
        active = session.scalars(select(CanonicalPrediction).where(
            CanonicalPrediction.prediction_status == "active",
        )).all()
        assert [row.id for row in active] == [result["prediction_id"]]


def test_ratings_snapshot_cannot_be_mislabeled_as_active_contract(client):
    with pytest.raises(ValueError):
        service.predict_all_upcoming(ratings_version="another-rating-run")




def test_direct_prediction_keeps_native_identity_cutoff_and_other_matches(client, monkeypatch):
    now = datetime.now(UTC)
    cutoff = "2026-10-05T00:00:00+00:00"
    monkeypatch.setattr(
        service,
        "predict_probability_from_features",
        lambda *args, **kwargs: (0.61, {
            "model_name": C0_NATIVE.name,
            "model_version": C0_NATIVE.version,
            "feature_version": C0_NATIVE.feature_version,
            "data_cutoff_at": cutoff,
            "market_features_used": False,
            "model_identity": {"name": C0_NATIVE.name, "artifact_hashes": {"checkpoint": "observed"}},
        }),
    )
    with get_session() as session:
        match = _match(session, now + timedelta(hours=1))
        match_id = match.id
        other = CanonicalMatch(
            canonical_key="audit-gamma-delta", team_a_name="Gamma", team_b_name="Delta",
            normalized_team_a="gamma", normalized_team_b="delta",
            start_time_normalized=(now + timedelta(days=1)).isoformat(), best_of=3,
        )
        session.add(other)
        session.flush()
        other_prediction = CanonicalPrediction(
            canonical_match_id=other.id, model_name=BAYESIAN_SHRUNK_HYBRID.name,
            model_version=BAYESIAN_SHRUNK_HYBRID.version, predicted_at=now.isoformat(),
            data_cutoff_at=cutoff, prob_a=0.6, prob_b=0.4,
        )
        session.add(other_prediction)
        session.flush()
        other_prediction_id = other_prediction.id
        session.commit()
    c0_features = _native_features(
        match_id, start=(now + timedelta(hours=1)).isoformat(),
    )
    monkeypatch.setattr(service, "build_features_for_match", lambda *args, **kwargs: {
        "status": "ready_player", "features": c0_features, "data_cutoff_at": None,
    })
    response = client.post(f"/matches/{match_id}/predict")
    assert response.status_code == 200
    data = response.json()
    assert (data["model_name"], data["model_version"]) == (C0_NATIVE.name, C0_NATIVE.version)
    assert data["prob_a"] == pytest.approx(0.61)
    with get_session() as session:
        row = session.scalars(select(CanonicalPrediction).where(
            CanonicalPrediction.canonical_match_id == match_id,
            CanonicalPrediction.model_name == C0_NATIVE.name,
        )).one()
        artifact = session.get(ModelArtifact, row.model_artifact_id)
        assert artifact is not None
        assert (artifact.model_name, artifact.model_version) == (row.model_name, row.model_version)
        assert row.data_cutoff_at == cutoff
        assert session.get(CanonicalPrediction, other_prediction_id).prediction_status == "active"


def test_direct_prediction_save_failure_rolls_back_request_session(client, monkeypatch):
    from sqlalchemy import event, text
    from betting_app.api.deps import get_db

    now = datetime.now(UTC)
    with get_session() as session:
        match_id = _match(session, now + timedelta(hours=1)).id
        session.commit()
    c0_features = _native_features(match_id, start=(now + timedelta(hours=1)).isoformat())
    monkeypatch.setattr(service, "build_features_for_match", lambda *args, **kwargs: {
        "status": "ready_player", "features": c0_features,
        "data_cutoff_at": (now - timedelta(days=1)).isoformat(),
    })
    monkeypatch.setattr(service, "predict_probability_from_features", lambda *args, **kwargs: (0.61, {
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "feature_version": C0_NATIVE.feature_version,
        "data_cutoff_at": (now - timedelta(days=1)).isoformat(),
        "model_identity": {"name": C0_NATIVE.name, "artifact_hashes": {"checkpoint": "observed"}},
        "market_features_used": False,
    }))
    with get_session() as failing_session:
        def fail_commit(session):
            session.execute(text("SELECT 1 / 0"))

        def request_session():
            yield failing_session

        event.listen(failing_session, "before_commit", fail_commit)
        client.app.dependency_overrides[get_db] = request_session
        try:
            response = client.post(f"/matches/{match_id}/predict")
            assert response.status_code == 503
            assert failing_session.execute(text("SELECT 1")).scalar_one() == 1
        finally:
            client.app.dependency_overrides.pop(get_db, None)
            event.remove(failing_session, "before_commit", fail_commit)


@pytest.fixture(scope="module")
def native_evidence_replay_client():
    """Retain genuine origin evidence, replaying exact cached arrays in one child."""
    from betting_app.core.models.native_c0 import NativeC0Adapter
    from betting_app.services.c0_inference import C0InferenceClient

    validator = NativeC0Adapter.__new__(NativeC0Adapter)
    snapshots = {}
    worker = C0InferenceClient()

    class EvidenceReplayClient:
        @property
        def identity(self):
            return worker.identity

        def predict_request(self, request, *, snapshot=None, canonical_match_id=None):
            if snapshot is not None:
                # Independent caller arrays keep the worker's unknown source provenance.
                return worker.predict_request(
                    request, snapshot=snapshot, canonical_match_id=canonical_match_id,
                )
            validated = validator._validate_request(request)
            normalized = {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in validated.items()
            }
            key = json.dumps(
                {"request": normalized, "canonical_match_id": canonical_match_id},
                sort_keys=True,
            )
            if key not in snapshots:
                evidence = worker.predict_evidence(normalized, canonical_match_id=canonical_match_id)
                target, raw, hist = validator._decode_snapshot(evidence.encoded_snapshot, validated)
                snapshots[key] = (
                    evidence.prediction,
                    {"target": target, "raw": raw, "hist": hist},
                )
            origin, exact_snapshot = snapshots[key]
            replayed = worker.predict_request(
                normalized, snapshot=exact_snapshot, canonical_match_id=canonical_match_id,
            )
            assert replayed.diagnostics["input_hashes"] == origin.diagnostics["input_hashes"]
            assert validator._array_digest({
                **exact_snapshot["raw"], **exact_snapshot["hist"],
            }) == origin.diagnostics["input_hashes"]
            assert replayed.prob_a == pytest.approx(origin.prob_a, abs=1e-12)
            assert replayed.prob_b == pytest.approx(origin.prob_b, abs=1e-12)
            # These bytes are the retained genuine history-built snapshot for this
            # exact request. Keep its origin diagnostics, never invent source hashes
            # from the replay's independently supplied-array provenance.
            return replace(replayed, diagnostics=deepcopy(origin.diagnostics))

    with pytest.MonkeyPatch.context() as environment:
        environment.delenv("C0_RAW_HISTORY_PATH", raising=False)
        environment.delenv("C0_RELEASE_MAP_PATH", raising=False)
        try:
            yield EvidenceReplayClient()
        finally:
            worker.close()


@pytest.fixture
def native_serving_environment(client, monkeypatch, native_evidence_replay_client):
    """Use owned storage and fixed actual clocks with shared genuine native evidence."""
    from betting_app.api.routers import matches as matches_router
    from betting_app.services import c0_inference

    clock = {"now": datetime(2026, 10, 2, 12, tzinfo=UTC)}

    class ServingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = clock["now"]
            return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(service, "datetime", ServingDatetime)
    monkeypatch.setattr(current_roster_service, "datetime", ServingDatetime)
    monkeypatch.setattr(matches_router, "datetime", ServingDatetime)
    monkeypatch.delenv("C0_RAW_HISTORY_PATH", raising=False)
    monkeypatch.delenv("C0_RELEASE_MAP_PATH", raising=False)
    monkeypatch.setattr(c0_inference, "_default_client", lambda: native_evidence_replay_client)
    yield clock


def _seed_current_native_roster_match(clock):
    roles = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
    player_ids = {"a": ["a5", "a1", "a4", "a2", "a3"], "b": ["b3", "b5", "b1", "b4", "b2"]}
    with get_session() as session:
        match = _match(session, clock["now"] + timedelta(hours=2))
        match.league = "Worlds"
        alpha = GolggTeam(team_name="Alpha", team_id=101, normalized_name="alpha")
        beta = GolggTeam(team_name="Beta", team_id=202, normalized_name="beta")
        session.add_all([alpha, beta])
        session.flush()
        match_data = {
            "id": match.id,
            "team_a_name": "Alpha",
            "team_b_name": "Beta",
            "team_a_golgg_id": alpha.id,
            "team_b_golgg_id": beta.id,
            "start_time_normalized": match.start_time_normalized,
            "best_of": 3,
            "league": "Worlds",
        }
        for side, team_name, provider_id in (("a", "Alpha", "101"), ("b", "Beta", "202")):
            assert current_roster_service.upsert_current_roster(
                session,
                team_name=team_name,
                team_id=provider_id,
                players=[
                    {"player_id": player_id, "player_name": player_id, "role": role}
                    for player_id, role in zip(player_ids[side], roles, strict=True)
                ],
                source="auto",
                source_match_id=f"last-{side}",
                source_game_id=f"last-game-{side}",
                source_match_date=(clock["now"] - timedelta(days=1)).isoformat(),
            )
        session.commit()
    return match_data, player_ids


def _build_current_native_features(match_data, **kwargs):
    return service.build_features_for_match(
        match_data,
        feature_version=C0_NATIVE.feature_version,
        ratings_version=C0_NATIVE.ratings_version,
        w20_version=C0_NATIVE.w20_version,
        native_c0=True,
        **kwargs,
    )


def _change_current_alpha_roster(clock, action):
    with get_session() as session:
        if action == "retract":
            # Retraction uses the same team-state lock as insertion/publication.
            current_roster_service.delete_current_roster(session, "Alpha")
        else:
            assert current_roster_service.upsert_current_roster(
                session,
                team_name="Alpha",
                team_id="101",
                players=[
                    {"player_id": player_id, "player_name": player_id, "role": role}
                    for player_id, role in zip(
                        ["replacement", "a1", "a4", "a2", "a3"],
                        ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"],
                        strict=True,
                    )
                ],
                source="manual",
                source_match_date=clock["now"].isoformat(),
                force=True,
            )
        session.commit()


@pytest.mark.parametrize("endpoint", ["detail", "history"])
def test_native_worker_history_provenance_survives_persistence_and_consumer_get(
    client,
    native_serving_environment,
    endpoint,
):
    clock = native_serving_environment
    match_data, _ = _seed_current_native_roster_match(clock)
    feature_result = _build_current_native_features(match_data)
    assert feature_result["status"] == "ready_player"
    generated = service.predict_all_upcoming()
    assert len(generated) == 1
    assert generated[0]["prediction_id"] is not None
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, generated[0]["prediction_id"])
        saved_diagnostics = json.loads(prediction.diagnostics_json)
        saved_probability = prediction.prob_a
        assert prediction.predicted_at == clock["now"].isoformat()
        assert prediction.data_cutoff_at == "2026-10-02T00:00:00+00:00"
    provenance = saved_diagnostics["feature_provenance"]
    assert provenance["source_kind"] == "canonical_frozen_golgg_history"
    assert provenance["raw_history_sha256"] == "991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca"
    assert provenance["dataset_freshness"]["raw_history_max_effective_release_day"] == "2026-09-08"
    assert saved_diagnostics["requested_decision_at"] == clock["now"].isoformat()
    assert saved_diagnostics["information_status_basis"] == "declared_input_provenance_not_authenticated"
    assert saved_diagnostics["uncertainty_available"] is False
    assert saved_diagnostics["epistemic_sigma_z"] is None
    assert saved_diagnostics["p_low_a"] is None
    assert saved_diagnostics["p_low_b"] is None

    suffix = "" if endpoint == "detail" else "/prediction-history"
    response = client.get(f"/matches/{match_data['id']}{suffix}")
    assert response.status_code == 200
    rows = response.json()["predictions"] if endpoint == "detail" else response.json()
    row = next(row for row in rows if row["model_name"] == C0_NATIVE.name)
    assert row["model_version"] == C0_NATIVE.version
    assert row["prob_a"] == pytest.approx(saved_probability, abs=1e-12)
    assert row["raw_diagnostics"] == saved_diagnostics


@pytest.mark.parametrize("endpoint", ["detail", "history"])
@pytest.mark.parametrize("legacy_diagnostics", [None, {"p_sports": 0.7}], ids=["absent", "partial"])
def test_historical_missing_native_provenance_remains_unknown_in_consumer_get(
    client,
    native_serving_environment,
    endpoint,
    legacy_diagnostics,
):
    clock = native_serving_environment
    match_data, _ = _seed_current_native_roster_match(clock)
    with get_session() as session:
        session.add(CanonicalPrediction(
            canonical_match_id=match_data["id"],
            model_name=C0_NATIVE.name,
            model_version=C0_NATIVE.version,
            predicted_at=clock["now"].isoformat(),
            data_cutoff_at="2026-10-02T00:00:00+00:00",
            prob_a=0.6,
            prob_b=0.4,
            diagnostics_json=json.dumps(legacy_diagnostics) if legacy_diagnostics is not None else None,
        ))
        session.commit()
    suffix = "" if endpoint == "detail" else "/prediction-history"
    response = client.get(f"/matches/{match_data['id']}{suffix}")
    assert response.status_code == 200
    rows = response.json()["predictions"] if endpoint == "detail" else response.json()
    row = next(row for row in rows if row["model_name"] == C0_NATIVE.name)
    assert row["raw_diagnostics"] == legacy_diagnostics
    with get_session() as session:
        prediction = session.scalars(select(CanonicalPrediction)).one()
        assert prediction.diagnostics_json == (
            json.dumps(legacy_diagnostics) if legacy_diagnostics is not None else None
        )


@pytest.mark.parametrize("basis", ["current_stored_roster", "operator_override"])
def test_native_prediction_retains_actual_roster_basis_and_separate_clocks(
    native_serving_environment,
    basis,
):
    clock = native_serving_environment
    match_data, player_ids = _seed_current_native_roster_match(clock)
    kwargs = {}
    if basis == "operator_override":
        kwargs = {
            "team_a_roster_override": {
                "team_name": "Alpha",
                "team_id": "101",
                "source": "manual",
                "source_match_date": clock["now"].isoformat(),
                "players": [
                    {"player_id": player_id, "player_name": player_id, "role": role}
                    for player_id, role in zip(
                        player_ids["a"],
                        ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"],
                        strict=True,
                    )
                ],
            },
            "is_lineup_confirmed": True,
        }
    feature_result = _build_current_native_features(match_data, **kwargs)
    original_features = deepcopy(feature_result["features"])
    probability, diagnostics = service.predict_probability_from_features(feature_result["features"])
    assert 0.0 < probability < 1.0
    assert feature_result["features"] == original_features
    provenance = diagnostics["roster_provenance"]
    assert provenance["decision_at"] == clock["now"].isoformat()
    assert provenance["start_at"] == match_data["start_time_normalized"]
    for side, expected_basis in (("a", basis), ("b", "current_stored_roster")):
        roster = provenance[f"team_{side}"]
        assert roster["basis"] == expected_basis
        assert roster["player_ids"] == player_ids[side]
        assert roster["tensor_player_order"] == sorted(player_ids[side])
        assert roster["source"] == ("manual" if side == "a" and basis == "operator_override" else "auto")
        assert roster["authenticated"] is False
        assert roster["published_at"] is None
        assert roster["observed_at"] is None
        if expected_basis == "current_stored_roster":
            assert roster["updated_at"] == clock["now"].isoformat()
            assert roster["source_match_id"] == f"last-{side}"
            assert roster["source_game_id"] == f"last-game-{side}"
            assert roster["source_match_date"] == (clock["now"] - timedelta(days=1)).isoformat()


@pytest.mark.parametrize("action", ["change", "retract"])
@pytest.mark.parametrize("boundary", ["before_inference", "before_persistence"])
def test_native_batch_withholds_roster_changes_without_staling_published_forecast(
    client,
    native_serving_environment,
    monkeypatch,
    action,
    boundary,
):
    clock = native_serving_environment
    match_data, _ = _seed_current_native_roster_match(clock)
    feature_result = _build_current_native_features(match_data)
    assert feature_result["status"] == "ready_player"
    first = service.predict_all_upcoming()[0]
    assert first["prediction_id"] is not None
    with get_session() as session:
        saved_features = session.scalars(select(UpcomingMatchFeature)).one().features_json
        published_diagnostics = session.get(CanonicalPrediction, first["prediction_id"]).diagnostics_json
    clock["now"] += timedelta(minutes=10)
    if boundary == "before_inference":
        _change_current_alpha_roster(clock, action)
    else:
        real_prediction = service.predict_probability_from_features

        def change_roster_after_real_native_prediction(features, **kwargs):
            prediction = real_prediction(features, **kwargs)
            _change_current_alpha_roster(clock, action)
            return prediction

        monkeypatch.setattr(service, "predict_probability_from_features", change_roster_after_real_native_prediction)
    result = service.predict_all_upcoming()
    assert len(result) == 1
    assert result[0].get("prediction_status") == "skipped"
    assert result[0]["prediction_id"] is None
    assert "roster" in result[0]["error"].lower()
    with get_session() as session:
        predictions = session.scalars(select(CanonicalPrediction)).all()
        assert [prediction.id for prediction in predictions] == [first["prediction_id"]]
        assert predictions[0].prediction_status == "active"
        assert predictions[0].diagnostics_json == published_diagnostics
        assert session.scalars(select(UpcomingMatchFeature)).one().features_json == saved_features
    response = client.get(f"/matches/{match_data['id']}/prediction-history")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["prob_a"] == pytest.approx(first["prob_a"], abs=1e-12)


def test_later_stored_roster_change_cannot_mutate_earlier_explicit_native_request(
    native_serving_environment,
):
    clock = native_serving_environment
    match_data, _ = _seed_current_native_roster_match(clock)
    features = _build_current_native_features(match_data)["features"]
    original = deepcopy(features)
    before_probability, before_diagnostics = service.predict_probability_from_features(features)
    clock["now"] += timedelta(minutes=10)
    _change_current_alpha_roster(clock, "change")
    after_probability, after_diagnostics = service.predict_probability_from_features(features)

    assert features == original
    assert after_probability == pytest.approx(before_probability, abs=1e-12)
    assert after_diagnostics["input_hashes"] == before_diagnostics["input_hashes"]
    assert after_diagnostics["requested_decision_at"] == original["c0_request"]["decision_at"]
    assert after_diagnostics["feature_provenance"] == before_diagnostics["feature_provenance"]


def test_operator_roster_edit_after_feature_construction_cannot_relabel_bound_native_inputs(
    native_serving_environment,
):
    clock = native_serving_environment
    match_data, _ = _seed_current_native_roster_match(clock)
    override = deepcopy(service.load_last_roster("Alpha"))
    feature_result = _build_current_native_features(
        match_data,
        team_a_roster_override=override,
        persist=False,
    )
    features = feature_result["features"]
    original = deepcopy(features)
    before_probability, before_diagnostics = service.predict_probability_from_features(features)
    override["players"][0]["player_id"] = "replacement"
    override["source"] = "manual"
    override["source_match_date"] = (clock["now"] + timedelta(minutes=10)).isoformat()
    after_probability, after_diagnostics = service.predict_probability_from_features(features)

    assert features == original
    assert after_probability == pytest.approx(before_probability, abs=1e-12)
    assert after_diagnostics["input_hashes"] == before_diagnostics["input_hashes"]
    assert after_diagnostics["requested_decision_at"] == original["c0_request"]["decision_at"]


def _seed_native_lock_history(clock, player_ids):
    """Real historical SQL inputs; neural numerics are covered by native replay cases."""
    with get_session() as session:
        session.add(GolggMatch(
            match_id="7001", date=(clock["now"] - timedelta(days=1)).date().isoformat(),
            tournament_name="Worlds", team1_name="Alpha", team2_name="Beta",
        ))
        session.flush()
        session.add(GolggGame(game_id="70010", match_id="7001"))
        session.flush()
        for side, team_name, team_id in (("a", "Alpha", "101"), ("b", "Beta", "202")):
            for player_id, role in zip(player_ids[side], current_roster_service.ROLE_ORDER, strict=True):
                session.add(GolggGamePlayer(
                    game_id="70010", match_id="7001", team_id=team_id,
                    team_name=team_name, side=side, role=role,
                    player_id=player_id, player_name=player_id,
                ))
        session.commit()


def _isolate_native_lock_numerics(monkeypatch):
    """Isolate only inference numerics, keeping actual production PG transactions."""
    monkeypatch.setattr(service, "predict_probability_from_features", lambda *args, **kwargs: (
        0.61, {
            "model_name": C0_NATIVE.name, "model_version": C0_NATIVE.version,
            "data_cutoff_at": "2026-10-01T00:00:00+00:00",
            "market_features_used": False,
        },
    ))
    real_transaction = service.transaction

    @contextmanager
    def bounded_transaction(*args, **kwargs):
        with real_transaction(*args, **kwargs) as connection:
            connection._session.execute(text("SET LOCAL statement_timeout = '15s'"))
            yield connection

    monkeypatch.setattr(service, "transaction", bounded_transaction)


def _wait_for_pg_blocker(session, pid, *, timeout=8):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        blockers = session.execute(
            text("SELECT pg_blocking_pids(:pid)"), {"pid": pid},
        ).scalar_one()
        if blockers:
            return blockers
        sleep(0.01)
    pytest.fail(f"PostgreSQL backend {pid} did not block at the expected production lock")


@pytest.mark.parametrize(
    ("basis", "mutation"),
    [
        ("historical_last_match", "insert"),
        ("current_stored_roster", "replace"),
        ("current_stored_roster", "delete"),
    ],
)
def test_native_publication_serializes_absent_insert_existing_replace_and_delete(
    client, monkeypatch, basis, mutation,
):
    clock = {"now": datetime.now(UTC)}
    match_data, player_ids = _seed_current_native_roster_match(clock)
    _seed_native_lock_history(clock, player_ids)
    if basis == "historical_last_match":
        with get_session() as session:
            current_roster_service.delete_current_roster(session, "Alpha")
            session.commit()
    features = _build_current_native_features(match_data)
    assert features["status"] == "ready_player"
    assert features["features"]["roster_provenance"]["team_a"]["basis"] == basis
    _isolate_native_lock_numerics(monkeypatch)
    publication_locked = Event()
    release_publication = Event()
    writer_started = Event()
    writer_pid = {}
    real_validate = service.validate_native_serving_inputs

    def hold_publication(features, *, session=None, lock=False):
        real_validate(features, session=session, lock=lock)
        if lock:
            publication_locked.set()
            assert release_publication.wait(10), "Publication release was not signaled"

    monkeypatch.setattr(service, "validate_native_serving_inputs", hold_publication)

    def mutate_roster():
        with get_session() as session:
            session.execute(text("SET LOCAL statement_timeout = '15s'"))
            writer_pid["value"] = session.execute(text("SELECT pg_backend_pid()")).scalar_one()
            writer_started.set()
            if mutation == "delete":
                current_roster_service.delete_current_roster(session, " ALPHA ")
            else:
                assert current_roster_service.upsert_current_roster(
                    session, team_name=" ALPHA ", team_id="101",
                    players=[
                        {"player_id": player_id, "player_name": player_id, "role": role}
                        for player_id, role in zip(
                            ["replacement", "a1", "a4", "a2", "a3"],
                            current_roster_service.ROLE_ORDER, strict=True,
                        )
                    ],
                    source="manual", force=True,
                )
            session.commit()

    with ThreadPoolExecutor(max_workers=2) as workers:
        publication = workers.submit(service.predict_all_upcoming)
        writer = None
        try:
            assert publication_locked.wait(10), "Actual batch did not reach publication validation"
            writer = workers.submit(mutate_roster)
            assert writer_started.wait(10)
            with get_session() as observer:
                _wait_for_pg_blocker(observer, writer_pid["value"])
                current = current_roster_service.load_current_roster("Alpha", session=observer)
                if basis == "historical_last_match":
                    assert current is None
                else:
                    assert current["players"][0]["player_id"] == player_ids["a"][0]
                assert observer.scalars(select(CanonicalPrediction)).all() == []
            assert not writer.done()
        finally:
            release_publication.set()
        published = publication.result(timeout=20)
        assert writer is not None
        writer.result(timeout=20)
    assert published[0]["prediction_id"] is not None
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, published[0]["prediction_id"])
        assert prediction.prediction_status == "active"
        current = current_roster_service.load_current_roster("Alpha", session=session)
        if mutation == "delete":
            assert current is None
        else:
            assert current["players"][0]["player_id"] == "replacement"


def test_actual_native_batch_commits_first_fixture_before_concurrent_roster_editor(
    client, monkeypatch,
):
    from betting_app.api.routers import matches as matches_router
    from betting_app.api.schemas import MatchRosterOverrideRequest

    clock = {"now": datetime.now(UTC)}
    first_match, player_ids = _seed_current_native_roster_match(clock)
    _seed_native_lock_history(clock, player_ids)
    with get_session() as session:
        second = _match(session, clock["now"] + timedelta(hours=3))
        second.league = "Worlds"
        second_match = {**first_match, "id": second.id, "start_time_normalized": second.start_time_normalized}
        session.commit()
    for match_data in (first_match, second_match):
        assert _build_current_native_features(match_data)["status"] == "ready_player"
    _isolate_native_lock_numerics(monkeypatch)
    prior = {row["canonical_match_id"]: row for row in service.predict_all_upcoming()}
    first_locked = Event()
    release_first = Event()
    second_lock_attempted = Event()
    editor_started = Event()
    editor_updated = Event()
    release_editor = Event()
    editor_pid = {}
    real_validate = service.validate_native_serving_inputs
    real_upsert = matches_router.upsert_current_roster

    def sequence_publication(features, *, session=None, lock=False):
        match_id = features["canonical_match_id"]
        if lock and match_id == second_match["id"]:
            second_lock_attempted.set()
        real_validate(features, session=session, lock=lock)
        if lock and match_id == first_match["id"]:
            first_locked.set()
            assert release_first.wait(10), "First fixture release was not signaled"

    def hold_editor_after_roster_write(*args, **kwargs):
        result = real_upsert(*args, **kwargs)
        editor_updated.set()
        assert release_editor.wait(10), "Editor commit was not signaled"
        return result

    monkeypatch.setattr(service, "validate_native_serving_inputs", sequence_publication)
    monkeypatch.setattr(matches_router, "upsert_current_roster", hold_editor_after_roster_write)
    body = MatchRosterOverrideRequest(
        team_side="a",
        players=[
            {"player_id": player_id, "player_name": player_id, "role": role}
            for player_id, role in zip(player_ids["a"], current_roster_service.ROLE_ORDER, strict=True)
        ],
    )

    def edit_second_fixture():
        with get_session() as session:
            session.execute(text("SET LOCAL statement_timeout = '15s'"))
            editor_pid["value"] = session.execute(text("SELECT pg_backend_pid()")).scalar_one()
            editor_started.set()
            return matches_router.update_match_roster(second_match["id"], body, db=session)

    with ThreadPoolExecutor(max_workers=2) as workers:
        batch = workers.submit(service.predict_all_upcoming)
        editor = None
        try:
            assert first_locked.wait(10)
            editor = workers.submit(edit_second_fixture)
            assert editor_started.wait(10)
            with get_session() as observer:
                _wait_for_pg_blocker(observer, editor_pid["value"])
            release_first.set()
            if not editor_updated.wait(10):
                # Surface the real PostgreSQL exception/SQLSTATE rather than
                # replacing a deadlock receipt with only an event timeout.
                if editor.done():
                    editor.result()
                if batch.done():
                    batch.result()
                pytest.fail("First fixture retained roster locks into the next fixture")
            assert second_lock_attempted.wait(10), "Actual batch did not reach the second canonical lock"
            with get_session() as observer:
                published_first = observer.scalars(select(CanonicalPrediction).where(
                    CanonicalPrediction.canonical_match_id == first_match["id"],
                    CanonicalPrediction.prediction_status == "active",
                )).one()
                assert published_first.id != prior[first_match["id"]]["prediction_id"]
                assert observer.get(CanonicalPrediction, prior[second_match["id"]]["prediction_id"]).prediction_status == "active"
            release_editor.set()
            assert editor is not None
            editor.result(timeout=20)
            result = batch.result(timeout=20)
        finally:
            release_first.set()
            release_editor.set()
    by_match = {row["canonical_match_id"]: row for row in result}
    assert by_match[first_match["id"]]["prediction_id"] is not None
    assert by_match[second_match["id"]]["prediction_status"] == "skipped"
    assert "roster" in by_match[second_match["id"]]["error"].lower()
    with get_session() as session:
        second_predictions = session.scalars(select(CanonicalPrediction).where(
            CanonicalPrediction.canonical_match_id == second_match["id"],
        )).all()
        assert [row.id for row in second_predictions] == [prior[second_match["id"]]["prediction_id"]]
        assert second_predictions[0].prediction_status == "active"


def test_native_batch_later_commit_failure_rolls_back_only_its_fixture(client, monkeypatch):
    from sqlalchemy.exc import DBAPIError

    clock = {"now": datetime.now(UTC)}
    first_match, _ = _seed_current_native_roster_match(clock)
    with get_session() as session:
        second = _match(session, clock["now"] + timedelta(hours=3))
        second.league = "Worlds"
        second_match = {**first_match, "id": second.id, "start_time_normalized": second.start_time_normalized}
        session.commit()
    for match_data in (first_match, second_match):
        assert _build_current_native_features(match_data)["status"] == "ready_player"
    _isolate_native_lock_numerics(monkeypatch)
    prior = {row["canonical_match_id"]: row for row in service.predict_all_upcoming()}
    with get_session() as session:
        prior_features = {
            row.canonical_match_id: row.features_json
            for row in session.scalars(select(UpcomingMatchFeature)).all()
        }
    real_validate = service.validate_native_serving_inputs

    def fail_second_commit(features, *, session=None, lock=False):
        real_validate(features, session=session, lock=lock)
        if lock and features["canonical_match_id"] == second_match["id"]:
            event.listen(session, "before_commit", lambda db: db.execute(text("SELECT 1 / 0")), once=True)

    monkeypatch.setattr(service, "validate_native_serving_inputs", fail_second_commit)
    with pytest.raises(DBAPIError) as failure:
        service.predict_all_upcoming()
    sqlstate = getattr(failure.value.orig, "sqlstate", None) or getattr(failure.value.orig, "pgcode", None)
    assert sqlstate == "22012"
    with get_session() as session:
        active = {
            row.canonical_match_id: row.id
            for row in session.scalars(select(CanonicalPrediction).where(
                CanonicalPrediction.prediction_status == "active",
            )).all()
        }
        assert active[first_match["id"]] != prior[first_match["id"]]["prediction_id"]
        assert active[second_match["id"]] == prior[second_match["id"]]["prediction_id"]
        assert {
            row.canonical_match_id: row.features_json
            for row in session.scalars(select(UpcomingMatchFeature)).all()
        } == prior_features
