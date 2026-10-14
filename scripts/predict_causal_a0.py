#!/usr/bin/env python3
"""Offline, bounded native A0 prediction from local fixture/source JSON inputs.

Input files are local JSON arrays (fixtures/metadata) and local JSON objects
(release map/optional roster evidence). The source match input is the existing
raw-history JSON path and is streamed/verified by the replay engine. Source data
is caller-provided; no network, database, or operational write is performed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.models.causal_a0 import CausalA0Adapter, FixedArtifactA0Adapter
from betting_app.core.models.causal_a0 import (
    FIXED_2026_CONTRACT_VERSION, FIXED_2026_CUTOFF_END, FIXED_2026_CUTOFF_START,
)
from src.analysis.a0_prediction_input import infer_fixture_snapshots, predict_fixture_inputs


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON input {path}: {error}") from error


def _records(path: Path, label: str) -> pd.DataFrame:
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} file must contain a JSON array of objects")
    return pd.DataFrame(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> Path:
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    if args.research_root:
        os.environ["ENSEMBLE_RESEARCH_ROOT"] = str(args.research_root.expanduser().resolve())
    fixtures = _records(args.fixtures, "fixtures")
    metadata = _records(args.metadata, "metadata")
    source_matches = args.source_matches.expanduser().resolve()
    release_days = _read_json(args.release_days)
    if not isinstance(release_days, dict):
        raise ValueError("release-days file must contain a JSON object mapping source match IDs to release dates")
    roster_evidence = args.roster_evidence
    if roster_evidence is not None and not isinstance(_read_json(roster_evidence), dict):
        raise ValueError("roster-evidence file must contain a JSON object")
    history_captures_path = getattr(args, "history_captures", None)
    history_captures = []
    if history_captures_path is not None:
        history_captures = _read_json(history_captures_path)
        if not isinstance(history_captures, list) or any(not isinstance(item, dict) for item in history_captures):
            raise ValueError("history-captures file must contain a JSON array of SourceCapture records")
    roster_captures_path = getattr(args, "roster_captures", None)
    roster_captures = []
    if roster_captures_path is not None:
        roster_captures = _read_json(roster_captures_path)
        if not isinstance(roster_captures, list) or any(not isinstance(item, dict) for item in roster_captures):
            raise ValueError("roster-captures file must contain a JSON array of SourceCapture records")
    snapshots = predict_fixture_inputs(
        fixtures, metadata, source_matches, release_days,
        roster_evidence=roster_evidence, history_captures=history_captures,
        roster_captures=roster_captures, max_fixtures=args.max_fixtures,
    )
    serving_contract = getattr(args, "serving_contract", "annual")
    if serving_contract not in {"annual", "fixed-2026"}:
        raise ValueError("serving_contract must be annual or fixed-2026")
    fixed_adapter = None
    if serving_contract == "fixed-2026":
        fixed_adapter = FixedArtifactA0Adapter(research_root=args.research_root)
        fixed_identity = fixed_adapter.identity
        adapter_factory = lambda _year: fixed_adapter
    else:
        fixed_identity = None
        adapter_factory = lambda year: CausalA0Adapter(year, research_root=args.research_root)
    records = infer_fixture_snapshots(snapshots, adapter_factory, mode=args.mode)
    result = {
        "schema_version": 1,
        "model": (
            "causal_a0_frozen_2026_fixed_forward"
            if serving_contract == "fixed-2026" else "causal_a0_native_annual"
        ),
        "serving_contract": (
            {
                **fixed_identity["serving_contract"],
                "name": fixed_identity["name"],
                "version": fixed_identity["version"],
                "artifact_year": fixed_identity["artifact_year"],
                "artifact_hashes": fixed_identity["artifact_hashes"],
                "source_hashes": fixed_identity["source_hashes"],
            }
            if serving_contract == "fixed-2026" else {"version": "annual-year-v1"}
        ),
        "mode": args.mode,
        "probability_semantics": "series_probability; adapter applies BO projection",
        "source_authentication": "not_assessed_local_declared_inputs",
        "predictions": _json_safe(records),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents silently replacing prior prediction evidence.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return destination




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="JSON array; exact IDs, best_of, start_at and UTC decision_at/cutoff_at")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON array of exact match metadata (golgg_match_id, team IDs, best_of, tournament_name)")
    parser.add_argument("--source-matches", type=Path, required=True, help="path to the native raw history JSON; replay streams it and verifies the frozen release map")
    parser.add_argument("--release-days", type=Path, required=True, help="frozen corrected release-day JSON mapping")
    parser.add_argument("--roster-evidence", type=Path, help="optional declared, timestamped roster evidence JSON")
    parser.add_argument(
        "--history-captures", type=Path,
        help="optional JSON array of loaded exact-byte SourceCapture records; bytes may be supplied as content_base64",
    )
    parser.add_argument(
        "--roster-captures", type=Path,
        help="optional JSON array of loaded exact-byte SourceCapture records linked by roster capture_id",
    )
    parser.add_argument("--research-root", type=Path, help="local research archive root; defaults to adapter's configured resolution")
    parser.add_argument("--output", type=Path, required=True, help="new output JSON path; existing paths are rejected")
    parser.add_argument("--mode", choices=("full", "no_w20", "no_organization"), default="full")
    parser.add_argument(
        "--serving-contract", choices=("annual", "fixed-2026"), default="annual",
        help="fixed-2026 explicitly serves the unchanged hash-pinned 2026 artifact on its declared future cutoff range",
    )
    parser.add_argument("--max-fixtures", type=int, default=250)
    args = parser.parse_args()
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
