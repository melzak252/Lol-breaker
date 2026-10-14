"""Minimal SQL model registry used by ML jobs.

The app already has many operational tables. This registry is intentionally
small and idempotent so it can be introduced without touching the thesis
scripts or blocking on a full migration refactor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from betting_app.core.db import get_session


MODEL_STATUSES = {"candidate", "shadow", "production", "rejected", "archived"}
RUN_STATUSES = {"running", "completed", "failed"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


@dataclass(frozen=True)
class ModelVersionRecord:
    model_name: str
    model_version: str
    status: str = "candidate"
    artifact_path: str | None = None
    feature_version: str | None = None
    training_start_at: str | None = None
    training_end_at: str | None = None
    dataset_hash: str | None = None
    git_commit: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @property
    def id(self) -> str:
        return f"{self.model_name}:{self.model_version}"


@dataclass(frozen=True)
class EvaluationRunRecord:
    model_name: str
    model_version: str
    run_type: str
    status: str = "completed"
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = field(default_factory=_now_iso)


def ensure_registry_tables(session: Session | None = None) -> None:
    """Create registry tables if they do not exist.

    Uses SQL portable across PostgreSQL and SQLite for tests/local usage.
    """
    own_session = session is None
    sess = session or get_session()
    try:
        sess.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_model_versions (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_path TEXT,
                feature_version TEXT,
                training_start_at TEXT,
                training_end_at TEXT,
                dataset_hash TEXT,
                git_commit TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(model_name, model_version)
            )
        """))
        sess.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_evaluation_runs (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                run_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                config_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT
            )
        """))
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()


def register_model_version(record: ModelVersionRecord, session: Session | None = None) -> ModelVersionRecord:
    if record.status not in MODEL_STATUSES:
        raise ValueError(f"Unsupported model status: {record.status}")
    own_session = session is None
    sess = session or get_session()
    now = _now_iso()
    try:
        ensure_registry_tables(sess)
        existing = get_model_version(record.model_name, record.model_version, sess)
        created_at = existing.get("created_at") if existing else now
        sess.execute(text("""
            INSERT INTO ml_model_versions (
                id, model_name, model_version, status, artifact_path, feature_version,
                training_start_at, training_end_at, dataset_hash, git_commit,
                metrics_json, notes, created_at, updated_at
            ) VALUES (
                :id, :model_name, :model_version, :status, :artifact_path, :feature_version,
                :training_start_at, :training_end_at, :dataset_hash, :git_commit,
                :metrics_json, :notes, :created_at, :updated_at
            )
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                artifact_path = excluded.artifact_path,
                feature_version = excluded.feature_version,
                training_start_at = excluded.training_start_at,
                training_end_at = excluded.training_end_at,
                dataset_hash = excluded.dataset_hash,
                git_commit = excluded.git_commit,
                metrics_json = excluded.metrics_json,
                notes = excluded.notes,
                updated_at = excluded.updated_at
        """), {
            "id": record.id,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "status": record.status,
            "artifact_path": record.artifact_path,
            "feature_version": record.feature_version,
            "training_start_at": record.training_start_at,
            "training_end_at": record.training_end_at,
            "dataset_hash": record.dataset_hash,
            "git_commit": record.git_commit,
            "metrics_json": _json_dumps(record.metrics),
            "notes": record.notes,
            "created_at": created_at,
            "updated_at": now,
        })
        sess.commit()
        return record
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()


def get_model_version(model_name: str, model_version: str, session: Session | None = None) -> dict[str, Any] | None:
    own_session = session is None
    sess = session or get_session()
    try:
        ensure_registry_tables(sess)
        row = sess.execute(text("""
            SELECT * FROM ml_model_versions
            WHERE model_name = :model_name AND model_version = :model_version
        """), {"model_name": model_name, "model_version": model_version}).mappings().first()
        if row is None:
            return None
        data = dict(row)
        data["metrics"] = _json_loads(data.pop("metrics_json", None))
        return data
    finally:
        if own_session:
            sess.close()


def list_model_versions(status: str | None = None, session: Session | None = None) -> list[dict[str, Any]]:
    own_session = session is None
    sess = session or get_session()
    try:
        ensure_registry_tables(sess)
        sql = "SELECT * FROM ml_model_versions"
        params: dict[str, Any] = {}
        if status:
            sql += " WHERE status = :status"
            params["status"] = status
        sql += " ORDER BY updated_at DESC, model_name, model_version"
        rows = sess.execute(text(sql), params).mappings().all()
        out = []
        for row in rows:
            data = dict(row)
            data["metrics"] = _json_loads(data.pop("metrics_json", None))
            out.append(data)
        return out
    finally:
        if own_session:
            sess.close()


def promote_model_version(model_name: str, model_version: str, session: Session | None = None) -> None:
    """Mark one version as production and archive previous production versions."""
    own_session = session is None
    sess = session or get_session()
    now = _now_iso()
    try:
        ensure_registry_tables(sess)
        existing = get_model_version(model_name, model_version, sess)
        if existing is None:
            raise ValueError(f"Model version not registered: {model_name}/{model_version}")
        sess.execute(text("""
            UPDATE ml_model_versions
            SET status = 'archived', updated_at = :updated_at
            WHERE model_name = :model_name AND status = 'production'
        """), {"model_name": model_name, "updated_at": now})
        sess.execute(text("""
            UPDATE ml_model_versions
            SET status = 'production', updated_at = :updated_at
            WHERE model_name = :model_name AND model_version = :model_version
        """), {"model_name": model_name, "model_version": model_version, "updated_at": now})
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()


def record_evaluation_run(record: EvaluationRunRecord, session: Session | None = None) -> EvaluationRunRecord:
    if record.status not in RUN_STATUSES:
        raise ValueError(f"Unsupported evaluation run status: {record.status}")
    own_session = session is None
    sess = session or get_session()
    try:
        ensure_registry_tables(sess)
        sess.execute(text("""
            INSERT INTO ml_evaluation_runs (
                id, model_name, model_version, run_type, status, started_at, finished_at,
                config_json, metrics_json, notes
            ) VALUES (
                :id, :model_name, :model_version, :run_type, :status, :started_at, :finished_at,
                :config_json, :metrics_json, :notes
            )
        """), {
            "id": record.id,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "run_type": record.run_type,
            "status": record.status,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "config_json": _json_dumps(record.config),
            "metrics_json": _json_dumps(record.metrics),
            "notes": record.notes,
        })
        sess.commit()
        return record
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()
