"""Explicit local A0 shadow inference; never writes operational predictions.

This entry point records declared inputs, not admission or source authentication.
A new inference has a new generation time and needs a new evidence key. To retry
persistence of an already-created envelope, use append_forecast_evidence directly.
"""
from __future__ import annotations

import base64
from datetime import date, datetime
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from betting_app.core.models.causal_a0 import CausalA0Adapter
from betting_app.services.forecast_evidence import append_forecast_evidence
from src.analysis.a0_prediction_input import snapshot_evidence_status
from src.analysis.ev_asof_replay import AsOfSnapshot

def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is pd.NaT or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _tensor_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    content = array.tobytes()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_base64": base64.b64encode(content).decode("ascii"),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def record_a0_snapshot(
    session: Session,
    adapter: CausalA0Adapter,
    snapshot: AsOfSnapshot,
    *,
    evidence_key: str,
    sources: list[dict[str, Any]],
    input_source_ids: list[str],
    schedule: dict[str, Any],
    quotes: list[dict[str, Any]],
    mode: str = "full",
) -> ForecastEvidenceSnapshot:
    """Predict from native tensors and append a self-contained evidence envelope.

    All source bytes must be supplied by the caller; this function performs no
    collection. Quotes and schedule are recorded but never model inputs. Session
    commit/rollback belongs to the caller. Unknown source clocks remain null.
    """
    input_status = _json_value(snapshot_evidence_status(snapshot))
    if not input_status["can_predict"]:
        raise ValueError("cannot record a prediction from unavailable input evidence")
    result = adapter.predict_snapshot(snapshot, mode=mode)
    target = _json_value(snapshot.target)
    # Strict JSON rejects ambiguous missing values rather than inventing clocks.
    json.dumps(target, allow_nan=False)
    raw = {key: _tensor_record(value) for key, value in snapshot.raw.items()}
    hist = {key: _tensor_record(value) for key, value in snapshot.hist.items()}
    expected = result.diagnostics["input_hashes"]
    for key, item in {**raw, **hist}.items():
        if item["sha256"] != expected[key]:
            raise ValueError("A0 snapshot changed between prediction and evidence recording")
    roster = {
        key: value for key, value in target.items()
        if key.startswith(("roster_", "tensor_player_order_", "raw_tensor_player_order_", "temporal_tensor_player_order_"))
        or key in {"team1_id", "team2_id", "competition_context"}
    }
    payload = {
        "schema_version": 1,
        "information_status": result.diagnostics["information_status"],
        "model": adapter.identity,
        "input": {
            "target": target, "raw": raw, "hist": hist,
            "source_ids": list(input_source_ids),
            "evidence_status": input_status,
        },
        "forecast": {
            "prob_a": result.prob_a, "prob_b": result.prob_b,
            "best_of": result.best_of, "mode": mode,
            "model_name": result.model_name, "model_version": result.model_version,
            "feature_version": result.feature_version,
            "generated_at": result.diagnostics["generated_at"],
            "simulated_origin_at": result.diagnostics["simulated_origin_at"],
            "availability_evidence": "declared_not_authenticated",
        },
        "roster": roster, "schedule": _json_value(schedule),
        "quotes": _json_value(quotes), "sources": _json_value(sources),
    }
    return append_forecast_evidence(session, evidence_key=evidence_key, payload=payload)
