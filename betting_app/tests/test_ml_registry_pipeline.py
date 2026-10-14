from __future__ import annotations

import json

from sqlalchemy import text

from betting_app.core.db import get_session, init_db
from betting_app.ml.config import BacktestConfig, StakingConfig
from betting_app.ml.inference import run_registry_shadow_inference
from betting_app.ml.pipelines.evaluation import EvaluationPipelineConfig, run_evaluation_pipeline
from betting_app.ml.pipelines.weekly_retrain import WeeklyRetrainConfig, run_weekly_retrain_pipeline
from betting_app.ml.registry import (
    ModelVersionRecord,
    get_model_version,
    list_model_versions,
    promote_model_version,
    register_model_version,
)
from betting_app.ml.registry.gates import PromotionGateConfig, evaluate_market_baseline_gate
from betting_app.ml.training.artifacts import train_and_save_model
from betting_app.ml.training.types import ModelCandidateSpec, TrainingDataset, TrainingExample


def test_model_registry_register_and_promote(client):
    del client
    with get_session() as session:
        register_model_version(
            ModelVersionRecord(
                model_name="TestModel",
                model_version="v1",
                status="candidate",
                metrics={"model_log_loss": 0.6},
            ),
            session=session,
        )
        register_model_version(
            ModelVersionRecord(
                model_name="TestModel",
                model_version="v2",
                status="shadow",
                metrics={"model_log_loss": 0.5},
            ),
            session=session,
        )
        promote_model_version("TestModel", "v2", session=session)

        v1 = get_model_version("TestModel", "v1", session=session)
        v2 = get_model_version("TestModel", "v2", session=session)
        production = list_model_versions(status="production", session=session)

    assert v1 is not None
    assert v1["status"] == "candidate"
    assert v2 is not None
    assert v2["status"] == "production"
    assert v2["metrics"] == {"model_log_loss": 0.5}
    assert [row["model_version"] for row in production] == ["v2"]


def test_promotion_gate_compares_model_against_market():
    decision = evaluate_market_baseline_gate(
        {
            "bets": 10,
            "comparison_observations": 10,
            "roi": 0.1,
            "model_log_loss": 0.4,
            "market_log_loss": 0.5,
            "model_brier": 0.15,
            "market_brier": 0.2,
        },
        PromotionGateConfig(min_bets=5, min_comparison_observations=5, min_roi=0.0),
    )
    assert decision.passed
    assert decision.reasons == []

    failed = evaluate_market_baseline_gate(
        {
            "bets": 2,
            "comparison_observations": 2,
            "roi": -0.1,
            "model_log_loss": 0.7,
            "market_log_loss": 0.5,
            "model_brier": 0.3,
            "market_brier": 0.2,
        },
        PromotionGateConfig(min_bets=5, min_comparison_observations=5, min_roi=0.0),
    )
    assert not failed.passed
    assert any("not enough bets" in reason for reason in failed.reasons)
    assert any("logloss worse" in reason for reason in failed.reasons)


def test_evaluation_pipeline_logs_run_and_candidate(client):
    del client
    init_db()
    with get_session() as session:
        session.execute(text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b,
                start_time_normalized, league, status, winner_side
            ) VALUES
            (501, 'm501', 'A', 'B', 'a', 'b', '2026-01-10T12:00:00+00:00', 'LCK', 'finished', 'team_a'),
            (502, 'm502', 'C', 'D', 'c', 'd', '2026-01-11T12:00:00+00:00', 'LCK', 'finished', 'team_b')
        """))
        session.execute(text("""
            INSERT INTO canonical_predictions (
                id, canonical_match_id, model_name, model_version, predicted_at, data_cutoff_at,
                prob_a, prob_b, prediction_status
            ) VALUES
            (1001, 501, 'PipelineModel', 'v1', '2026-01-10T09:00:00+00:00', '2026-01-10T08:00:00+00:00', 0.70, 0.30, 'stale'),
            (1002, 502, 'PipelineModel', 'v1', '2026-01-11T09:00:00+00:00', '2026-01-11T08:00:00+00:00', 0.40, 0.60, 'stale')
        """))
        session.execute(text("""
            INSERT INTO odds_snapshots (
                id, bookmaker_id, canonical_match_id, odds_a, odds_b, scraped_at,
                raw_team_a, raw_team_b
            ) VALUES
            (2001, 2, 501, 2.10, 1.80, '2026-01-10T10:00:00+00:00', 'A', 'B'),
            (2002, 2, 502, 1.90, 2.05, '2026-01-11T10:00:00+00:00', 'C', 'D')
        """))
        session.commit()

        result = run_evaluation_pipeline(
            EvaluationPipelineConfig(
                model_name="PipelineModel",
                model_version="v1",
                days_back=None,
                include_stale=True,
                backtest=BacktestConfig(
                    bankroll_start=100.0,
                    min_ev=0.0,
                    staking=StakingConfig(strategy="fixed", fixed_stake=10.0),
                ),
                promotion_gate=PromotionGateConfig(min_bets=1, min_comparison_observations=1),
            ),
            session=session,
        )
        registered = get_model_version("PipelineModel", "v1", session=session)
        rows = session.execute(text("SELECT COUNT(*) FROM ml_evaluation_runs")).scalar_one()

    assert result.metrics["predictions_loaded"] == 2
    assert result.metrics["labels_loaded"] == 2
    assert result.metrics["odds_quotes_loaded"] == 2
    assert result.metrics["comparison_observations"] == 2
    assert result.metrics["predictions_temporally_eligible"] == 2
    assert result.metrics["prediction_temporal_exclusions"] == {}
    assert result.metrics["cohort_start_at"] == "2026-01-10T12:00:00+00:00"
    assert result.metrics["cohort_end_at"] == "2026-01-11T12:00:00+00:00"
    assert result.metrics["financial_execution_evaluated"] is False
    assert result.evaluation_run_id
    assert registered is not None
    assert registered["metrics"]["predictions_loaded"] == 2
    assert rows == 1


def test_weekly_retrain_pipeline_trains_artifact_and_registers_model(client, tmp_path):
    del client
    init_db()
    with get_session() as session:
        for idx in range(8):
            match_id = 700 + idx
            winner_side = "team_a" if idx % 2 == 0 else "team_b"
            session.execute(
                text("""
                    INSERT INTO canonical_matches (
                        id, canonical_key, team_a_name, team_b_name,
                        normalized_team_a, normalized_team_b,
                        start_time_normalized, league, status, winner_side
                    ) VALUES (
                        :id, :canonical_key, :team_a_name, :team_b_name,
                        :normalized_team_a, :normalized_team_b,
                        :start_time_normalized, 'LCK', 'finished', :winner_side
                    )
                """),
                {
                    "id": match_id,
                    "canonical_key": f"train-{match_id}",
                    "team_a_name": f"A{idx}",
                    "team_b_name": f"B{idx}",
                    "normalized_team_a": f"a{idx}",
                    "normalized_team_b": f"b{idx}",
                    "start_time_normalized": f"2026-02-{idx + 1:02d}T12:00:00+00:00",
                    "winner_side": winner_side,
                },
            )
            target_signal = 1.0 if winner_side == "team_a" else -1.0
            session.execute(
                text("""
                    INSERT INTO upcoming_match_features (
                        canonical_match_id, feature_version, ratings_version,
                        data_cutoff_at, team_a_golgg_name, team_b_golgg_name,
                        feature_status, features_json
                    ) VALUES (
                        :canonical_match_id, 'fv-test', 'rv-test',
                        :data_cutoff_at, :team_a_golgg_name, :team_b_golgg_name,
                        'ready_player', :features_json
                    )
                """),
                {
                    "canonical_match_id": match_id,
                    "data_cutoff_at": f"2026-02-{idx + 1:02d}T09:00:00+00:00",
                    "team_a_golgg_name": f"A{idx}",
                    "team_b_golgg_name": f"B{idx}",
                    "features_json": json.dumps(
                        {
                            "ratings": {
                                "team_a": {"elo": 1500 + target_signal * 20 + idx},
                                "team_b": {"elo": 1500 - target_signal * 20 - idx},
                            },
                            "w20": {
                                "team_a_winrate": 0.6 if winner_side == "team_a" else 0.4,
                                "team_b_winrate": 0.4 if winner_side == "team_a" else 0.6,
                            },
                            "ignored_roster_list": ["x", "y"],
                        }
                    ),
                },
            )
        session.commit()

        result = run_weekly_retrain_pipeline(
            WeeklyRetrainConfig(
                model_name="WeeklyTestModel",
                model_version="weekly-test-v1",
                feature_version="fv-test",
                ratings_version="rv-test",
                min_features=2,
                min_train_size=4,
                test_size=2,
                step_size=2,
                artifact_root=str(tmp_path),
                status_on_success="shadow",
            ),
            candidate_specs=[
                ModelCandidateSpec(
                    name="logreg_test",
                    estimator_type="logistic_regression",
                    params={"C": 1.0, "max_iter": 200},
                )
            ],
            session=session,
        )
        registered = get_model_version("WeeklyTestModel", "weekly-test-v1", session=session)
        run_count = session.execute(
            text("SELECT COUNT(*) FROM ml_evaluation_runs WHERE model_name = 'WeeklyTestModel'")
        ).scalar_one()

    assert result.dataset_size == 8
    assert result.feature_count >= 4
    assert result.best_evaluation.candidate.name == "logreg_test"
    assert result.artifact.artifact_path.endswith("model.joblib")
    assert result.artifact.metadata_path.endswith("metadata.json")
    assert result.artifact.dataset_path.endswith("train_dataset.jsonl")
    assert result.artifact.feature_names_path.endswith("feature_names.json")
    assert result.artifact.dataset_metadata_path.endswith("dataset_metadata.json")
    assert (tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "model.joblib").exists()
    assert (tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "metadata.json").exists()
    dataset_path = tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "train_dataset.jsonl"
    feature_names_path = tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "feature_names.json"
    dataset_metadata_path = tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "dataset_metadata.json"
    assert dataset_path.exists()
    assert feature_names_path.exists()
    assert dataset_metadata_path.exists()
    dataset_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    model_metadata = json.loads((tmp_path / "WeeklyTestModel" / "weekly-test-v1" / "metadata.json").read_text(encoding="utf-8"))
    assert len(dataset_rows) == 8
    assert dataset_rows[0]["canonical_match_id"] == 700
    assert "ratings.team_a.elo" in dataset_rows[0]["features"]
    assert "ignored_roster_list" not in dataset_rows[0]["features"]
    assert len(feature_names) == result.feature_count
    assert dataset_metadata["rows"] == 8
    assert dataset_metadata["feature_count"] == result.feature_count
    assert dataset_metadata["dataset_hash"] == result.artifact.metrics["dataset_hash"]
    assert model_metadata["dataset_hash"] == result.artifact.metrics["dataset_hash"]
    assert model_metadata["dataset_path"] == result.artifact.dataset_path
    assert registered is not None
    assert registered["status"] == "shadow"
    assert registered["metrics"]["dataset_size"] == 8
    assert registered["metrics"]["dataset_hash"] == result.artifact.metrics["dataset_hash"]
    assert registered["metrics"]["best_candidate"]["candidate_name"] == "logreg_test"
    assert run_count == 1


def test_shadow_inference_writes_active_canonical_predictions(client, tmp_path):
    del client
    init_db()
    dataset = TrainingDataset(
        examples=[
            TrainingExample(1, "2026-03-01T12:00:00+00:00", 1, {"rating_diff": 20.0, "winrate_diff": 0.2}),
            TrainingExample(2, "2026-03-02T12:00:00+00:00", 0, {"rating_diff": -20.0, "winrate_diff": -0.2}),
            TrainingExample(3, "2026-03-03T12:00:00+00:00", 1, {"rating_diff": 30.0, "winrate_diff": 0.3}),
            TrainingExample(4, "2026-03-04T12:00:00+00:00", 0, {"rating_diff": -30.0, "winrate_diff": -0.3}),
        ],
        feature_names=["rating_diff", "winrate_diff"],
    )
    artifact = train_and_save_model(
        dataset,
        ModelCandidateSpec(
            name="shadow_logreg",
            estimator_type="logistic_regression",
            params={"C": 1.0, "max_iter": 200},
        ),
        model_name="ShadowInferenceModel",
        model_version="shadow-v1",
        metrics={"mean_log_loss": 0.5},
        artifact_root=tmp_path,
    )

    with get_session() as session:
        register_model_version(
            ModelVersionRecord(
                model_name="ShadowInferenceModel",
                model_version="shadow-v1",
                status="shadow",
                artifact_path=artifact.artifact_path,
                feature_version="fv-shadow",
                metrics={"mean_log_loss": 0.5},
            ),
            session=session,
        )
        session.execute(text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b,
                start_time_normalized, league, status
            ) VALUES (
                900, 'shadow-900', 'A', 'B', 'a', 'b',
                '2026-04-01T12:00:00+00:00', 'LCK', 'upcoming'
            )
        """))
        session.execute(text("""
            INSERT INTO upcoming_match_features (
                canonical_match_id, feature_version, ratings_version,
                data_cutoff_at, team_a_golgg_name, team_b_golgg_name,
                feature_status, features_json
            ) VALUES (
                900, 'fv-shadow', 'rv-shadow', '2026-04-01T09:00:00+00:00',
                'A', 'B', 'ready_player', :features_json
            )
        """), {
            "features_json": json.dumps({"rating_diff": 25.0, "winrate_diff": 0.25, "ignored": [1, 2, 3]})
        })
        session.commit()

        result = run_registry_shadow_inference(model_name="ShadowInferenceModel", session=session)
        row = session.execute(text("""
            SELECT canonical_match_id, model_name, model_version, prob_a, prob_b,
                   prediction_status, features_version, ratings_version, diagnostics_json
            FROM canonical_predictions
            WHERE canonical_match_id = 900
        """)).mappings().one()

    assert result.models_seen == 1
    assert result.models_loaded == 1
    assert result.feature_rows_seen == 1
    assert result.predictions_written == 1
    assert row["model_name"] == "ShadowInferenceModel"
    assert row["model_version"] == "shadow-v1"
    assert row["prediction_status"] == "active"
    assert row["features_version"] == "fv-shadow"
    assert row["ratings_version"] == "rv-shadow"
    assert 0.0 < float(row["prob_a"]) < 1.0
    assert abs(float(row["prob_a"]) + float(row["prob_b"]) - 1.0) < 1e-6
    diagnostics = json.loads(row["diagnostics_json"])
    assert diagnostics["source"] == "ml_registry"
    assert diagnostics["registry_status"] == "shadow"
    assert diagnostics["feature_count"] == 2
