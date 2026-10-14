"""C0-native tournament simulation cache with versioned probability provenance."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from betting_app.services.c0_inference import get_c0_artifact_identity
from betting_app.services.liquipedia_bracket_service import LiquipediaBracketService
from betting_app.services.tournament_service import (
    SUPPORTED_BRACKETS,
    TournamentSimulator,
    TOURNAMENT_PROBABILITY_VERSION,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "tournament_sim_cache"
DEFAULT_DEPTHS = (10000,)
CACHE_TTL_SECONDS = 60


def _input_fingerprint(provenance: Mapping[str, Any], artifact_identity: Mapping[str, Any]) -> str:
    inputs = {
        "model_inputs": {
            key: provenance.get(key)
            for key in (
                "probability_model_version", "team_ids", "team_rosters", "decision_at",
                "mode", "competition_context", "organization_identity_available",
                "organization_identity_source",
            )
        },
        "c0_artifact_identity": artifact_identity,
    }
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_identity_fingerprint(provenance: Mapping[str, Any]) -> str:
    source_identity = []
    for prediction in provenance.get("native_prediction_provenance", []):
        diagnostics = prediction.get("diagnostics", {}) if isinstance(prediction, Mapping) else {}
        source_identity.append({
            key: diagnostics.get(key)
            for key in (
                "model_identity", "data_cutoff_at", "requested_decision_at",
                "feature_provenance", "input_hashes", "mode", "mode_probabilities",
            )
        })
    encoded = json.dumps(source_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_file(tournament_id: str) -> Path:
    """Return the filesystem path for a tournament simulation cache file."""
    return _ensure_cache_dir() / f"{tournament_id}.json"


def _is_current_c0_result(result: dict[str, Any]) -> bool:
    provenance = result.get("provenance")
    return (
        isinstance(provenance, dict)
        and provenance.get("probability_model") == "Causal-C0"
        and provenance.get("probability_model_version") == TOURNAMENT_PROBABILITY_VERSION
    )
def current_input_fingerprint(tournament_id: str, teams: Sequence[str]) -> str:
    simulator = TournamentSimulator(competition_context=tournament_id)
    return _input_fingerprint(simulator._provenance(teams), get_c0_artifact_identity())




def get_cached_simulation(
    tournament_id: str,
    n_simulations: int = 10000,
    *,
    expected_input_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Load cached tournament simulation from disk.

    Returns the simulation matching the requested path depth if present,
    or the closest available depth. Returns None if cache does not exist or is corrupt.
    """
    cache_path = get_cache_file(tournament_id)
    if not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not expected_input_fingerprint or data.get("input_fingerprint") != expected_input_fingerprint:
            return None
        cached_at = datetime.fromisoformat(str(data.get("cached_at", "")).replace("Z", "+00:00"))
        if cached_at.tzinfo is None:
            return None
        cache_age = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds()
        if cache_age < 0 or cache_age > CACHE_TTL_SECONDS:
            return None
        if (
            data.get("probability_model") != "Causal-C0"
            or data.get("probability_model_version") != TOURNAMENT_PROBABILITY_VERSION
            or data.get("has_manual_overrides")
        ):
            return None
        depths_data = data.get("depths", {})
        str_key = str(n_simulations)

        # Check exact depth match
        if str_key in depths_data:
            result = dict(depths_data[str_key])
            if not _is_current_c0_result(result):
                return None
            if data.get("source_identity_fingerprints", {}).get(str_key) != _source_identity_fingerprint(result["provenance"]):
                return None
            result["cached"] = True
            result["cached_at"] = data.get("cached_at")
            result["available_depths"] = [int(k) for k in depths_data.keys() if k.isdigit()]
            return result

        # Fallback to closest available depth
        if depths_data:
            available = sorted([int(k) for k in depths_data.keys() if k.isdigit()])
            closest = min(available, key=lambda d: abs(d - n_simulations))
            result = dict(depths_data[str(closest)])
            if not _is_current_c0_result(result):
                return None
            if data.get("source_identity_fingerprints", {}).get(str(closest)) != _source_identity_fingerprint(result["provenance"]):
                return None
            result["cached"] = True
            result["cached_at"] = data.get("cached_at")
            result["available_depths"] = available
            result["depth_requested"] = n_simulations
            result["depth_served"] = closest
            return result


    except Exception as exc:
        logger.warning("Failed to read simulation cache for %s: %s", tournament_id, exc)

    return None


def recalculate_and_cache(
    tournament_id: str,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    manual_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run native C0 simulations across path depths and cache truthful provenance."""
    builder = SUPPORTED_BRACKETS.get(tournament_id)
    if not builder:
        raise ValueError(f"Tournament {tournament_id} not supported")

    # Simulation consumes stored state only. External acquisition is /sync.
    service = LiquipediaBracketService()
    sync_res = service.load_bracket(tournament_id)
    bracket = sync_res.get("bracket") or builder()


    simulator = TournamentSimulator(competition_context=tournament_id)

    depths_results: dict[str, Any] = {}
    primary_result: dict[str, Any] | None = None

    for depth in depths:
        clamped_depth = min(max(int(depth), 100), 100000)
        sim_res = simulator.simulate(
            bracket,
            n_simulations=clamped_depth,
            manual_overrides=manual_overrides,
        )
        sim_res["source"] = sync_res.get("source", "curated")
        sim_res["status"] = sync_res.get("status", "ready")
        sim_res["synced_at"] = sync_res.get("synced_at")
        sim_res["updated_matches"] = sync_res.get("updated_matches", 0)

        depths_results[str(clamped_depth)] = sim_res
        if primary_result is None or clamped_depth == 10000:
            primary_result = sim_res

    if primary_result is None:
        raise ValueError("At least one simulation depth is required for caching.")
    c0_artifact_identity = get_c0_artifact_identity()
    input_fingerprint = _input_fingerprint(primary_result["provenance"], c0_artifact_identity)
    now_iso = datetime.now(timezone.utc).isoformat()
    source_identity_fingerprints = {
        depth: _source_identity_fingerprint(result["provenance"])
        for depth, result in depths_results.items()
    }
    cache_payload = {
        "tournament_id": tournament_id,
        "cached_at": now_iso,
        "probability_model": "Causal-C0",
        "probability_model_version": TOURNAMENT_PROBABILITY_VERSION,
        "c0_artifact_identity": c0_artifact_identity,
        "input_fingerprint": input_fingerprint,
        "source_identity_fingerprints": source_identity_fingerprints,
        "has_manual_overrides": bool(manual_overrides),
        "available_depths": list(depths),
        "depths": depths_results,
    }

    cache_file = get_cache_file(tournament_id)
    cache_file.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved multi-depth simulation cache for %s to %s", tournament_id, cache_file)


    out = dict(primary_result)
    out["cached"] = False
    out["cached_at"] = now_iso
    out["available_depths"] = list(depths)
    return out
