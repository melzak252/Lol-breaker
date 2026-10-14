"""Leakage-aware fixture inputs and explicit evidence status for native A0.

This is an input/inference wrapper around the verified as-of replay. It does not
construct features, collect external data, or turn evidence into confidence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.ev_asof_replay import (
    AsOfSnapshot,
    _target_competition_context,
    _timestamp,
    reconstruct_asof_snapshots,
)


MAX_FIXTURES = 250




def snapshot_evidence_status(snapshot: AsOfSnapshot) -> dict[str, Any]:
    """Describe input availability/support; never certify source authenticity."""
    target = snapshot.target
    reasons: list[str] = []
    if not snapshot.available or snapshot.raw is None or snapshot.hist is None:
        reason = snapshot.missing_reason or "snapshot_unavailable"
        source_a, source_b = target.get("roster_source_a"), target.get("roster_source_b")
        source_types = [source.get("source_type") if isinstance(source, Mapping) else None
                        for source in (source_a, source_b)]
        return {
            "can_predict": False, "input_status": "UNAVAILABLE", "reasons": [reason],
            "roster": {"status": target.get("roster_status", "unknown"), "source_types": source_types},
            "history": {"players_without_history": None, "predictive_reliability": "NOT_ASSESSED"},
            "support": {"competition_context": target.get("competition_context"),
                        "feature_source_max_day": target.get("feature_source_max_day"),
                        "observed_history_tokens": None},
            "predictive_reliability": "NOT_ASSESSED",
            "information_status_basis": "declared_not_authenticated",
        }

    mask = np.asarray(snapshot.hist["mask"])
    if mask.ndim != 3 or mask.shape[:2] != (2, 5) or mask.shape[2] == 0 or not np.isin(mask, (0, 1)).all():
        raise ValueError("snapshot mask must be a nonempty binary (2, 5, history_length) array")
    player_support = np.any(mask != 0, axis=2)
    players_without_history = int((~player_support).sum())
    observed_history_tokens = int(np.count_nonzero(mask))
    stat_coverage = csd_xpd_coverage = None
    if "history" in snapshot.hist:
        from src.models.a0_data_focus import _history_coverage
        history = np.asarray(snapshot.hist["history"])
        if history.shape != (*mask.shape, 33):
            raise ValueError("snapshot history does not match the native 33-field token contract")
        coverage, selected = _history_coverage(
            {"history": {"history": history[None], "mask": mask[None]}}, np.array([0]))
        stat_coverage = float(coverage[0]) if np.isfinite(coverage[0]) else None
        csd_xpd_coverage = float(selected[0]) if np.isfinite(selected[0]) else None
    if stat_coverage is None:
        reasons.append("history_statistics_unavailable")
    elif stat_coverage < 1:
        reasons.append("partial_observed_history_statistics")
    roster_sources = [target.get("roster_source_a"), target.get("roster_source_b")]
    source_types = [str(item.get("source_type", "unknown")) if isinstance(item, Mapping) else "unknown"
                    for item in roster_sources]
    roster_status = str(target.get("roster_status", "unknown"))
    context = target.get("competition_context")
    feature_max = target.get("feature_source_max_day")
    cutoff = target.get("cutoff_day")
    stale_days = None
    blocked = False
    if feature_max is not None and cutoff is not None:
        try:
            stale_days = (pd.Timestamp(cutoff).date() - pd.Timestamp(feature_max).date()).days
        except (TypeError, ValueError):
            reasons.append("invalid_feature_source_max_day")
            blocked = True
    if feature_max is None:
        reasons.append("feature_source_max_day_missing")
        blocked = any(np.any(value != 0) for group in (snapshot.raw, snapshot.hist) for value in group.values())
    elif stale_days is None:
        reasons.append("feature_source_max_day_unavailable")
        blocked = True
    elif stale_days <= 0:
        reasons.append("history_not_before_cutoff")
        blocked = True
    if players_without_history:
        reasons.append("players_without_history")
    if observed_history_tokens == 0:
        reasons.append("no_observed_history_tokens")
    if not context:
        reasons.append("competition_context_missing")
    if roster_status == "unresolved":
        reasons.append("roster_support_unresolved")
    if roster_status not in {"prior_observed_scenario", "announced"} or any(
        value not in {"prior_observed_scenario", "announced"} for value in source_types
    ):
        reasons.append("roster_support_unconfirmed")
    gates = _raw_gate_summary(snapshot.raw)
    if gates["available"] < gates["total"]:
        reasons.append("partial_raw_feature_support")
    if any(value == "prior_observed_scenario" for value in source_types):
        reasons.append("roster_is_prior_observed_scenario_not_match_announcement")
    if stale_days is not None and stale_days > 30:
        reasons.append("feature_source_stale_over_30_days")
    limited = bool(reasons)
    return {
        "can_predict": not blocked,
        "input_status": "UNAVAILABLE" if blocked else "LIMITED_EVIDENCE" if limited else "SUPPORTED_INPUT",
        "reasons": reasons,
        "roster": {"status": roster_status, "source_types": source_types,
                   "known_player_slots_by_side": [
                       int(item.get("known_slots", 0)) if isinstance(item, Mapping) else 0
                       for item in roster_sources
                   ],
                   "announced_sides": sum(value == "announced" for value in source_types),
                   "prior_observed_sides": sum(value == "prior_observed_scenario" for value in source_types),
                   "assumption": target.get("scenario_timestamp_assumption")},
        "history": {"players_without_history": players_without_history,
                    "players_with_history": 10 - players_without_history,
                    "observed_history_tokens": observed_history_tokens,
                    "history_slot_capacity": int(mask.size),
                    "history_token_fraction": observed_history_tokens / int(mask.size),
                    "history_stat_coverage": stat_coverage,
                    "csd_xpd_coverage": csd_xpd_coverage,
                    "predictive_reliability": "NOT_ASSESSED"},
        "support": {"competition_context": context,
                    "competition_context_provenance": target.get("roster_context_provenance"),
                    "feature_source_max_day": feature_max, "cutoff_day": cutoff,
                    "feature_stale_days": stale_days,
                    "raw_support_gates": _raw_gate_summary(snapshot.raw)},
        "predictive_reliability": "NOT_ASSESSED",
        "information_status_basis": "declared_not_authenticated",
    }

def _raw_gate_summary(raw: Mapping[str, np.ndarray]) -> dict[str, int]:
    gates = np.asarray(raw.get("gates", np.empty((0,))))
    if gates.shape != (2, 2):
        return {"available": 0, "total": 4}
    return {"available": int(np.count_nonzero(gates)), "total": 4}


def _captured_payload(capture: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Verify captured bytes and clocks without authenticating their publisher."""
    if not isinstance(capture, Mapping):
        raise ValueError("capture must be a source-capture record")
    for field in ("capture_id", "source_id"):
        if not isinstance(capture.get(field), str) or not capture[field]:
            raise ValueError(f"capture requires {field}")
    body = capture.get("body_bytes")
    if body is None:
        try:
            body = base64.b64decode(capture["content_base64"], validate=True)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("capture requires exact body bytes or valid base64") from error
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != capture.get("sha256"):
        raise ValueError("capture SHA-256 differs from exact body bytes")
    if capture.get("attempt_status", "captured") != "captured":
        raise ValueError("failed source capture cannot supply native inputs")
    clocks = {}
    for field in ("observed_at", "response_received_at", "first_observed_at",
                  "request_started_at", "published_at", "source_effective_at"):
        value = capture.get(field)
        if value is None and field not in {"observed_at", "response_received_at", "first_observed_at"}:
            continue
        stamp = _timestamp(value, f"capture {field}")
        if pd.isna(stamp):
            raise ValueError(f"capture requires timezone-aware {field}")
        clocks[field] = stamp
    if (clocks["response_received_at"] > clocks["observed_at"]
            or clocks["first_observed_at"] > clocks["observed_at"]
            or ("request_started_at" in clocks
                and clocks["request_started_at"] > clocks["response_received_at"])):
        raise ValueError("capture request, receipt and observation clocks contradict")
    authentication = capture.get("source_authentication_status")
    if authentication not in {"local_fixture", "unverified", "authenticated"}:
        raise ValueError("capture requires an explicit source authentication declaration")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("captured native input is not valid JSON") from error
    lineage = {
        "capture_id": capture["capture_id"], "source_id": capture["source_id"],
        "sha256": capture["sha256"], "source_authentication_status": authentication,
        "authentication_basis": "external_declaration_not_verified_by_native_builder",
        **{field: stamp.isoformat() for field, stamp in clocks.items()},
    }
    return payload, lineage


def merge_captured_history(
    frozen_matches: Iterable[Mapping[str, Any]],
    release_days: Mapping[str, Any],
    captures: Iterable[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Append actual raw records, with receipt/release floors and immutable warm start."""
    if not isinstance(release_days, Mapping):
        raise ValueError("release_days must be a mapping")
    matches = list(frozen_matches)
    by_id = {}
    for match in matches:
        key = replay._normalize_id(match.get("match_id"), "frozen match ID")
        if key is None or key in by_id:
            raise ValueError("frozen history match IDs must be present and unique")
        by_id[key] = match
    releases = dict(release_days)
    lineage: dict[str, dict[str, Any]] = {}
    seen_captures = {}
    for capture in captures:
        payload, provenance = _captured_payload(capture)
        capture_key = provenance["capture_id"]
        if capture_key in seen_captures:
            if seen_captures[capture_key] != provenance:
                raise ValueError("conflicting metadata for the same history capture ID")
            continue
        seen_captures[capture_key] = provenance
        records = payload if isinstance(payload, list) else [payload]
        if not records or any(not isinstance(record, Mapping) for record in records):
            raise ValueError("captured history must contain raw match objects")
        for match in records:
            key = replay._normalize_id(match.get("match_id"), "captured match ID")
            if key is None:
                raise ValueError("captured history requires exact match IDs")
            if key in by_id:
                if by_id[key] != match:
                    raise ValueError(f"capture changes frozen warm-start match or prior update {key}")
                if key not in lineage:
                    continue
            source_day, _, _ = replay._raw_release_day(match, {})
            available_day = max(source_day, *(
                pd.Timestamp(provenance[field]).date()
                for field in ("observed_at", "response_received_at", "published_at", "source_effective_at")
                if field in provenance
            ))
            if key not in lineage and key in releases:
                if replay._as_day(releases[key], "supplied new release day") < available_day:
                    raise ValueError("release day may not precede capture availability")
                raise ValueError("new history release day must derive from captured clocks")
            record_lineage = {**provenance, "effective_release_day": available_day.isoformat()}
            if key in lineage:
                previous = lineage[key]
                previous["captures"].append(record_lineage)
                # Repeated identical bytes cannot make prior availability depend on input order.
                if (available_day.isoformat(), capture_key) < (releases[key], previous["capture_id"]):
                    lineage[key] = {**record_lineage, "captures": previous["captures"]}
                    releases[key] = available_day.isoformat()
            else:
                by_id[key] = match
                matches.append(match)
                releases[key] = available_day.isoformat()
                lineage[key] = {**record_lineage, "captures": [record_lineage]}
    for record in lineage.values():
        record["captures"].sort(key=lambda item: (item["effective_release_day"], item["capture_id"]))
    return matches, releases, lineage


def _captured_rosters(captures: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive roster/lineage facts from captured bodies, never caller-linked replacement facts."""
    evidence: dict[str, Any] = {"identity_links": [], "announcements": [], "input_hashes": {}}
    lineage = {}
    for capture in captures:
        payload, provenance = _captured_payload(capture)
        if not isinstance(payload, Mapping):
            raise ValueError("captured roster body must be an evidence object")
        key = provenance["capture_id"]
        if key in lineage:
            if lineage[key] != provenance:
                raise ValueError("conflicting roster capture ID")
            continue
        if not capture.get("evidence_uri"):
            raise ValueError("captured roster requires a source evidence URI")
        available = max(pd.Timestamp(provenance[field])
                        for field in ("observed_at", "response_received_at", "published_at")
                        if field in provenance)
        for field in ("identity_links", "announcements"):
            records = payload.get(field, [])
            if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
                raise ValueError(f"captured roster {field} must be a list of source records")
            for record in records:
                evidence[field].append({
                    **record, "capture_id": key,
                    "observed_at": provenance["observed_at"],
                    "evidence_uri": capture["evidence_uri"],
                    "captured_available_at": available.isoformat(),
                })
        evidence["input_hashes"][f"capture:{key}"] = provenance["sha256"]
        lineage[key] = provenance
    return evidence, lineage


def predict_fixture_inputs(
    fixtures: pd.DataFrame,
    metadata: pd.DataFrame,
    source_matches: Iterable[Mapping[str, Any]] | str | Path,
    release_days: Mapping[str, Any],
    *,
    roster_evidence: Any = None,
    history_captures: Iterable[Mapping[str, Any]] = (),
    roster_captures: Iterable[Mapping[str, Any]] = (),
    max_fixtures: int = MAX_FIXTURES,
) -> list[AsOfSnapshot]:
    """Replay bounded local fixtures through the existing release-aware engine.

    Fixture rows require unique IDs, a timezone-aware start_at and a forecast
    origin supplied as decision_at or its cutoff_at alias. If both are supplied,
    they must name the same instant. Actual inference creation time is separate.
    A raw-history Path uses replay's streamed, hash-verified corrected-bank route;
    small historical subsets may be an iterable.
    """
    
    required = {"row_id", "golgg_match_id", "team1_id", "team2_id", "best_of", "start_at"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"fixtures missing required fields: {missing}")
    if not 1 <= max_fixtures <= MAX_FIXTURES:
        raise ValueError(f"max_fixtures must be between 1 and {MAX_FIXTURES}")
    if len(fixtures) > max_fixtures:
        raise ValueError(f"fixture count {len(fixtures)} exceeds bound {max_fixtures}")
    if fixtures["row_id"].isna().any() or fixtures["row_id"].astype(str).duplicated().any():
        raise ValueError("fixture row_id values must be present and unique")
    if fixtures["golgg_match_id"].isna().any() or fixtures["golgg_match_id"].astype(str).duplicated().any():
        raise ValueError("fixture match IDs must be present and unique")
    rows = fixtures.copy(deep=True)
    for field in ("decision_at", "cutoff_at", "start_at"):
        values = rows[field] if field in rows else [pd.NaT] * len(rows)
        rows[field] = pd.Series(
            [_timestamp(value, f"fixture {field}") for value in values],
            index=rows.index, dtype="datetime64[ns, UTC]",
        )
    both = rows["decision_at"].notna() & rows["cutoff_at"].notna()
    if (rows.loc[both, "decision_at"] != rows.loc[both, "cutoff_at"]).any():
        raise ValueError("fixture decision_at and cutoff_at must name the same instant")
    rows["decision_at"] = rows["decision_at"].fillna(rows["cutoff_at"])
    if rows["decision_at"].isna().any():
        raise ValueError("fixtures require explicit decision_at or cutoff_at")
    if rows["start_at"].isna().any():
        raise ValueError("fixtures require explicit start_at")
    if (rows["decision_at"] >= rows["start_at"]).any():
        raise ValueError("every fixture decision_at must be strictly before start_at")
    if rows["best_of"].isna().any() or not rows["best_of"].isin([1, 3, 5]).all():
        raise ValueError("best_of must be 1, 3, or 5")
    # Explicit context may fill a metadata gap, never replace a conflicting unit.
    metadata_rows = metadata.copy(deep=True)
    if "golgg_match_id" not in metadata_rows:
        raise ValueError("metadata requires golgg_match_id")
    if metadata_rows["golgg_match_id"].astype(str).duplicated().any():
        raise ValueError("metadata match IDs must be unique")
    existing = set(metadata_rows["golgg_match_id"].astype(str))
    metadata_index = metadata_rows.set_index(metadata_rows["golgg_match_id"].astype(str), drop=False)
    identity_fields = ("team1_id", "team2_id", "best_of")
    for fixture in rows.to_dict("records"):
        key = str(fixture["golgg_match_id"])
        fixture_context = _target_competition_context(fixture, key)
        if key in existing:
            supplied = metadata_index.loc[key]
            for field in identity_fields:
                if field in supplied and pd.notna(supplied[field]) and str(supplied[field]) != str(fixture[field]):
                    raise ValueError(f"fixture metadata identity conflict for {key}: {field}")
            metadata_context = _target_competition_context(supplied, key)
            if fixture_context and metadata_context and fixture_context != metadata_context:
                raise ValueError(f"fixture metadata competition context conflict for {key}")
            if fixture_context and not metadata_context:
                if "competition_context" not in metadata_rows:
                    metadata_rows["competition_context"] = None
                else:
                    metadata_rows["competition_context"] = metadata_rows["competition_context"].astype(object)
                metadata_rows.loc[
                    metadata_rows["golgg_match_id"].astype(str) == key, "competition_context"
                ] = fixture_context
    supplemental = rows.loc[~rows["golgg_match_id"].astype(str).isin(existing)]
    known = ["golgg_match_id", "team1_id", "team2_id", "best_of", "tournament_name", "competition_context"]
    if not supplemental.empty:
        metadata_rows = pd.concat([metadata_rows, supplemental[[column for column in known if column in supplemental]]], ignore_index=True)
    if metadata_rows.duplicated("golgg_match_id").any():
        raise ValueError("fixture and metadata contain ambiguous match identity")
    # Only explicit identity/cutoff fields reach the replay opportunity table.
    opportunity_fields = ["row_id", "golgg_match_id", "decision_at", "start_at"]
    rows = rows[opportunity_fields].copy()
    if isinstance(source_matches, (str, Path)):
        source_input: Iterable[Mapping[str, Any]] | Path = Path(source_matches).expanduser().resolve()
    else:
        matches = list(source_matches)
        if not matches:
            raise ValueError("source_matches must contain historical source evidence; no empty-history fabrication")
        source_ids = [str(match.get("match_id", "")) for match in matches]
        if any(not value for value in source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("source match identities must be present and unique")
        source_input = matches
    captures = list(history_captures)
    history_lineage = {}
    effective_releases = release_days
    if captures:
        if isinstance(source_input, Path):
            from src.analysis.ev_approach_data import _resolve_research_root
            research_root = _resolve_research_root(None)
            protocol = replay._verify_protocol(
                replay.PROJECT_ROOT / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json"
            )
            _, frozen_releases, _, _ = replay._verify_corrected_bank(research_root, protocol, source_input)
            supplied = {str(key): replay._as_day(value, "warm-start release day").isoformat()
                        for key, value in release_days.items()}
            if supplied != frozen_releases:
                raise ValueError("warm-start release map differs from the verified corrected bank")
            with source_input.open("rb") as stream:
                warm = json.load(stream)
        else:
            warm = source_input
        source_input, effective_releases, history_lineage = merge_captured_history(warm, release_days, captures)
    roster_capture_records = list(roster_captures)
    roster_lineage = {}
    if roster_capture_records:
        if roster_evidence is not None:
            raise ValueError("supply captured roster bodies or declared scenario roster evidence, not both")
        roster_evidence, roster_lineage = _captured_rosters(roster_capture_records)
    snapshots = list(reconstruct_asof_snapshots(
        rows, metadata_rows, source_input, effective_releases, roster_evidence=roster_evidence,
    ))
    for snapshot in snapshots:
        cutoff = replay._as_day(snapshot.target["cutoff_day"], "snapshot cutoff day")
        snapshot.target["history_capture_lineage"] = {
            key: {**value, "captures": [
                item for item in value["captures"]
                if replay._as_day(item["effective_release_day"], "capture release day") < cutoff
            ]}
            for key, value in history_lineage.items()
            if replay._as_day(effective_releases[key], "captured release day") < cutoff
        }
        origin = _timestamp(snapshot.target.get("effective_decision_at"), "snapshot decision cutoff")
        if pd.isna(origin):
            origin = pd.Timestamp(cutoff, tz="UTC")
        available_rosters = {
            key: value for key, value in roster_lineage.items()
            if max(pd.Timestamp(value[field]) for field in
                   ("observed_at", "response_received_at", "published_at") if field in value) <= origin
        }
        snapshot.target["roster_capture_lineage"] = available_rosters
        if roster_lineage:
            snapshot.target["roster_evidence_input_hashes"] = {
                f"capture:{key}": value["sha256"] for key, value in available_rosters.items()
            }
        snapshot.target["capture_information_status"] = "declared_not_authenticated"
    return snapshots


def infer_fixture_snapshots(
    snapshots: Iterable[AsOfSnapshot],
    adapter_factory: Callable[[int], Any],
    *, mode: str = "full",
) -> list[dict[str, Any]]:
    """Infer only available snapshots and retain every unavailable fixture row."""
    output = []
    adapters: dict[int, Any] = {}
    for snapshot in snapshots:
        status = snapshot_evidence_status(snapshot)
        target = snapshot.target
        record: dict[str, Any] = {
            "row_id": target.get("row_id"), "golgg_match_id": target.get("golgg_match_id"),
            "decision_at": target.get("effective_decision_at"), "start_at": target.get("start_at"),
            "best_of": target.get("best_of"), "team1_id": target.get("team1_id"),
            "team2_id": target.get("team2_id"), "prob_team1": None, "prob_team2": None,
            "missing_reason": snapshot.missing_reason or (";".join(status["reasons"]) if not status["can_predict"] else None),
            "evidence_status": status,
            "history_capture_lineage": target.get("history_capture_lineage", {}),
            "roster_capture_lineage": target.get("roster_capture_lineage", {}),
        }
        if status["can_predict"]:
            year = int(target["model_year"])
            if year not in adapters:
                adapters[year] = adapter_factory(year)
            result = adapters[year].predict_snapshot(snapshot, mode=mode)
            record.update(prob_team1=result.prob_a, prob_team2=result.prob_b,
                          model_name=result.model_name, model_version=result.model_version,
                          generated_at=result.diagnostics["generated_at"],
                          simulated_origin_at=result.diagnostics["simulated_origin_at"])
        output.append(record)
    return output
