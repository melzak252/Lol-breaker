"""Causal player-plus-family map replay and annual scalar calibration.

This module is a research runner, not an operational rating loader. R0 and R1
share the same player-only Glicko-2 state; R1 adds the independent diagonal
family-logit bridge from :mod:`src.ratings.family_calibrated_glicko2`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from scipy.optimize import minimize_scalar

from src.models.competition_tiers import (
    CompetitionIdentity,
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.models.prematch_features import _normalize_series, _validate_releases
from src.ratings.family_calibrated_glicko2 import (
    FamilyCalibratedGlicko2,
    FamilyMapBridge,
    RatingEvent,
    series_win_probability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "model_reliability_20261001_055321"
BANK_RELATIVE = Path("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank")
RAW_HISTORY_RELATIVE = Path("data/artifacts/golgg-database-recovery-20260909/matches.json")
EVALUATION_YEARS = (2024, 2025, 2026)
CALIBRATION_YEARS = (2023, 2024, 2025)
EXPECTED_CANONICAL_ROWS = 11_550
TTL_DAYS = 365
TEMPERATURE_BOUNDS = (0.25, 4.0)

PLAYER_CONFIG = {
    "tau": 0.5,
    "convergence_tolerance": 1e-6,
    "rating_period_days": 30.0,
    "initial_family_deviation": 0.0,
    "initial_tier_deviation": 0.0,
    "bridge_process_deviation": 0.0,
    "regional_discount": 1.0,
}
BRIDGE_CONFIG = {"initial_deviation": 150.0, "process_deviation": 1.0}

# Saved in the pre-fit protocol. This is deliberately fixed before any fitting.
FAMILY_PRE_FIT_RECIPE = {
    "schema": "family_map_bridge_recipe_v1",
    "status": "fixed_before_scoring",
    "comparison": "R0 player-only Glicko-2 vs R1 same player state plus family logit bridge",
    "player_state": PLAYER_CONFIG,
    "family_bridge": {
        **BRIDGE_CONFIG,
        "unit": "Glicko rating points",
        "prior_mean": 0.0,
        "prior_variance": "initial_deviation**2",
        "process_variance": "process_deviation**2 * elapsed_calendar_days",
        "link": "logit(p_player_map) + ln(10)/400 * sum_f((w_a[f]-w_b[f])*theta[f])",
        "likelihood": "weighted Bernoulli likelihood per observed map",
        "daily_update_weight": "1 / valid_maps_in_series",
        "score_derivative": "sum_j weight_j * x_jf * ln(10)/400 * (y_j-p_j)",
        "diagonal_information": "sum_j weight_j * (x_jf*ln(10)/400)**2 * p_j*(1-p_j)",
        "posterior_variance": "1 / (1/prior_variance + diagonal_information)",
        "posterior_mean": "prior_mean + posterior_variance * score_derivative",
        "approximation": "one frozen-prior score update; off-diagonal Hessian terms discarded",
        "unknown_affiliation": "zero family weight; neutral",
        "same_family": "equal side weights cancel exactly",
        "tier_or_region_heuristics": False,
        "joint_player_family_posterior": False,
        "numerical_policy": "unclipped interior probabilities; unrepresentable endpoints stop the run, never drop rows",
        "per_match_uncertainty_claim": False,
    },
    "affiliation": {
        "source": "previously released unambiguous domestic map participation",
        "ttl_days": TTL_DAYS,
        "same_release_conflict": "clear affiliation and report unknown until later unambiguous domestic evidence",
        "family_transfer": "change of competition family only; tier reclassification is not a transfer or conflict",
        "cross_league_overwrites": False,
        "historical_map_rosters": "actual participants for each map",
        "target_roster_scenario": "full canonical A0 target rosters; known-lineup scenario only",
    },
    "calibration": {
        "rows": "prior calendar year predictions with result_day before Jan 1 evaluation origin",
        "objective": "binary log loss on projected series probabilities",
        "transform": "sigmoid(logit(p_series)/temperature)",
        "intercept": False,
        "bounds": list(TEMPERATURE_BOUNDS),
        "per_arm": "fit R0 and R1 separately with the same CAL row IDs and rule",
        "series_projection": "once from map probability using exact iid odd-best-of combinatorics",
    },
}

_REQUIRED_METADATA = {
    "golgg_match_id", "team1_id", "team2_id", "date", "result_day",
    "feature_source_max_day", "best_of", "y", "roster_a", "roster_b",
}
_SOURCE_FILES = (
    "src/models/family_reliability.py",
    "src/ratings/family_calibrated_glicko2.py",
    "src/ratings/glicko2_core.py",
    "src/models/competition_tiers.py",
    "src/models/prematch_features.py",
    "scripts/experiment_family_reliability.py",
    "scripts/export_prospective_features.py",
    "src/utils/golgg_schema.py",
)


@dataclass(frozen=True, slots=True)
class ReplayMap:
    game_id: str
    day: date
    roster_a: tuple[str, ...]
    roster_b: tuple[str, ...]
    outcome: int


@dataclass(frozen=True, slots=True)
class ReplaySeries:
    match_id: str
    day: date
    source_header_day: date
    release_day: date
    team_a: str
    team_b: str
    best_of: int
    tournament: str | None
    competition: CompetitionIdentity
    maps: tuple[ReplayMap, ...]


@dataclass(frozen=True, slots=True)
class DomesticAffiliation:
    family: str
    tier: str
    release_day: date
    source_match_id: str
    source_tournament: str | None
    matched_rule: str
    transfer_day: date | None = None
    transferred_from: str | None = None


class ReleasedAffiliations:
    """Player affiliations from prior released, unambiguous domestic evidence."""

    def __init__(self, ttl_days: int = TTL_DAYS) -> None:
        if isinstance(ttl_days, bool) or not isinstance(ttl_days, int) or ttl_days < 0:
            raise ValueError("ttl_days must be a non-negative integer")
        self.ttl_days = ttl_days
        self._states: dict[str, DomesticAffiliation] = {}
        self._conflicts: dict[str, date] = {}
        self._current_date: date | None = None

    @property
    def current_date(self) -> date | None:
        return self._current_date

    def lookup(
        self, player_id: str, query_day: date
    ) -> tuple[DomesticAffiliation | None, str | None, int | None]:
        day = _calendar_day(query_day, "query_day")
        if self._current_date is not None and day < self._current_date:
            raise ValueError("affiliation query cannot precede consumed release history")
        current = self._states.get(str(player_id))
        conflict_day = self._conflicts.get(str(player_id))
        if conflict_day is not None and conflict_day < day and (
            current is None or current.release_day < conflict_day
        ):
            return None, "same_release_conflict", None
        if current is None:
            return None, "no_prior_released_domestic", None
        if current.release_day >= day:
            return None, "not_prior_to_query_day", 0
        age = (day - current.release_day).days
        if age > self.ttl_days:
            return None, "expired_over_365_days", age
        return current, None, age

    def consume_period(
        self, release_day: date, records: Sequence[ReplaySeries]
    ) -> list[dict[str, Any]]:
        """Consume one simultaneous effective-release date, atomically by player."""
        day = _calendar_day(release_day, "release_day")
        if self._current_date is not None and day <= self._current_date:
            raise ValueError("affiliation release periods must be strictly increasing")
        candidates: dict[str, list[DomesticAffiliation]] = defaultdict(list)
        for record in sorted(records, key=lambda row: row.match_id):
            identity = record.competition
            if (
                identity.scope is not CompetitionScope.DOMESTIC
                or identity.family == "unknown"
                or identity.tier is CompetitionTier.UNKNOWN
            ):
                continue
            for game in record.maps:
                for player_id in (*game.roster_a, *game.roster_b):
                    candidates[player_id].append(
                        DomesticAffiliation(
                            family=identity.family,
                            tier=identity.tier.value,
                            release_day=day,
                            source_match_id=record.match_id,
                            source_tournament=record.tournament,
                            matched_rule=identity.matched_rule,
                        )
                    )
        changes: list[dict[str, Any]] = []
        for player_id, evidence in sorted(candidates.items()):
            destinations = {row.family for row in evidence}
            prior = self._states.get(player_id)
            prior_is_active = (
                prior is not None and (day - prior.release_day).days <= self.ttl_days
            )
            if len(destinations) != 1:
                self._states.pop(player_id, None)
                self._conflicts[player_id] = day
                changes.append({"player_id": player_id, "status": "conflict"})
                continue
            selected = min(
                evidence,
                key=lambda row: (row.source_match_id, row.matched_rule),
            )
            transfer_day = None
            transferred_from = None
            if prior_is_active and prior.family != selected.family:
                transfer_day = day
                transferred_from = prior.family
            elif prior_is_active:
                transfer_day = prior.transfer_day
                transferred_from = prior.transferred_from
            current = DomesticAffiliation(
                family=selected.family,
                tier=selected.tier,
                release_day=day,
                source_match_id=selected.source_match_id,
                source_tournament=selected.source_tournament,
                matched_rule=selected.matched_rule,
                transfer_day=transfer_day,
                transferred_from=transferred_from,
            )
            self._states[player_id] = current
            self._conflicts.pop(player_id, None)
            changes.append({
                "player_id": player_id,
                "status": "known",
                "family": current.family,
                "tier": current.tier,
                "transfer_day": _iso(current.transfer_day),
            })
        self._current_date = day
        return changes

    def profile(self, roster: Sequence[str], query_day: date) -> dict[str, Any]:
        day = _calendar_day(query_day, "query_day")
        players = _validated_roster(roster, "target roster")
        weights: dict[str, float] = defaultdict(float)
        known: list[tuple[str, DomesticAffiliation, int]] = []
        unknown_reasons: Counter[str] = Counter()
        for player_id in players:
            affiliation, reason, age = self.lookup(player_id, day)
            if affiliation is None:
                unknown_reasons[str(reason)] += 1
                continue
            known.append((player_id, affiliation, int(age)))
            weights[affiliation.family] += 1.0 / len(players)
        families = sorted(weights)
        transfers = [
            (player_id, item)
            for player_id, item, _ in known
            if item.transfer_day is not None and (day - item.transfer_day).days <= self.ttl_days
        ]
        recent_transfers = [
            (player_id, item)
            for player_id, item in transfers
            if (day - item.transfer_day).days <= 90
        ]
        known_ages = [age for _, _, age in known]
        return {
            "weights": dict(sorted(weights.items())),
            "known_players": len(known),
            "unknown_players": len(players) - len(known),
            "known_fraction": len(known) / len(players),
            "mixed": len(families) > 1,
            "families": families,
            "max_affiliation_age_days": max(known_ages) if known_ages else None,
            "transfer_players": len(transfers),
            "recent_transfer_players": len(recent_transfers),
            "unknown_reasons": dict(sorted(unknown_reasons.items())),
            "receipts": [
                {
                    "player_id": player_id,
                    "family": item.family,
                    "tier": item.tier,
                    "release_day": item.release_day.isoformat(),
                    "source_match_id": item.source_match_id,
                    "source_tournament": item.source_tournament,
                    "matched_rule": item.matched_rule,
                    "age_days": age,
                    "transfer_day": _iso(item.transfer_day),
                    "transferred_from": item.transferred_from,
                }
                for player_id, item, age in sorted(known, key=lambda row: row[0])
            ],
        }

    def state_receipt(self) -> dict[str, Any]:
        return {
            "current_date": _iso(self._current_date),
            "affiliations": {
                player_id: {
                    "family": item.family,
                    "tier": item.tier,
                    "release_day": item.release_day.isoformat(),
                    "source_match_id": item.source_match_id,
                    "source_tournament": item.source_tournament,
                    "matched_rule": item.matched_rule,
                    "transfer_day": _iso(item.transfer_day),
                    "transferred_from": item.transferred_from,
                }
                for player_id, item in sorted(self._states.items())
            },
            "conflicts": {
                player_id: day.isoformat()
                for player_id, day in sorted(self._conflicts.items())
            },
        }


class FamilyBridgeGraph:
    """Inventory empirical family contrasts supporting the diagonal bridge."""

    def __init__(self) -> None:
        self.edges: Counter[tuple[str, str]] = Counter()
        self.nodes: set[str] = set()

    def consume(self, observation: Mapping[str, Any]) -> None:
        weights_a = observation["family_weights_a"]
        weights_b = observation["family_weights_b"]
        active = sorted(
            family
            for family in set(weights_a) | set(weights_b)
            if weights_a.get(family, 0.0) != weights_b.get(family, 0.0)
        )
        self.nodes.update(active)
        for left_index, left in enumerate(active):
            for right in active[left_index + 1 :]:
                self.edges[left, right] += 1

    def support(self, weights_a: Mapping[str, float], weights_b: Mapping[str, float]) -> dict[str, Any]:
        active = sorted(
            family
            for family in set(weights_a) | set(weights_b)
            if weights_a.get(family, 0.0) != weights_b.get(family, 0.0)
        )
        components = self.components()
        edge_support = math.fsum(
            count
            for left_index, left in enumerate(active)
            for right in active[left_index + 1 :]
            for edge, count in self.edges.items()
            if edge == (left, right) or edge == (right, left)
        )
        labels = {components.get(family) for family in active}
        if not active:
            status = "neutral_or_same_family"
        elif len(active) == 1:
            status = "single_family_no_contrast"
        elif edge_support > 0 and len(labels) == 1:
            status = "connected_supported"
        elif edge_support > 0:
            status = "partly_supported_disconnected"
        else:
            status = "unsupported"
        return {
            "edge_support": int(edge_support),
            "status": status,
            "active_families": active,
            "connected_components": len(set(components.values())),
        }

    def components(self) -> dict[str, int]:
        parents: dict[str, str] = {node: node for node in self.nodes}

        def root(node: str) -> str:
            parents.setdefault(node, node)
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        for left, right in sorted(self.edges):
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)
        roots = sorted({root(node) for node in parents})
        root_ids = {value: index for index, value in enumerate(roots)}
        return {node: root_ids[root(node)] for node in sorted(parents)}

    def to_json(self) -> dict[str, Any]:
        components = self.components()
        return {
            "edge_count": int(sum(self.edges.values())),
            "unique_edges": len(self.edges),
            "node_count": len(components),
            "component_count": len(set(components.values())),
            "edges": [
                {"family_a": left, "family_b": right, "map_support": int(count)}
                for (left, right), count in sorted(self.edges.items())
            ],
            "components": components,
        }


def fit_temperature(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[float, dict[str, Any]]:
    """Fit one positive no-intercept scalar temperature by binary log loss."""
    p = [float(value) for value in probabilities]
    y = [int(value) for value in outcomes]
    if not p or len(p) != len(y):
        raise ValueError("temperature calibration requires aligned nonempty rows")
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in p):
        raise ValueError("temperature CAL probabilities must be finite and interior")
    if any(value not in (0, 1) for value in y):
        raise ValueError("temperature CAL outcomes must be binary")
    logits = [math.log(value) - math.log1p(-value) for value in p]

    def objective(temperature: float) -> float:
        total = math.fsum(
            max(z / temperature, 0.0)
            - label * (z / temperature)
            + math.log1p(math.exp(-abs(z / temperature)))
            for z, label in zip(logits, y, strict=True)
        )
        return total / len(y)

    result = minimize_scalar(
        objective,
        bounds=TEMPERATURE_BOUNDS,
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise ArithmeticError("positive scalar temperature fit did not converge")
    candidates = [TEMPERATURE_BOUNDS[0], float(result.x), TEMPERATURE_BOUNDS[1]]
    temperature = min(candidates, key=lambda value: (objective(value), value))
    return temperature, {
        "objective": "mean_binary_log_loss",
        "temperature": temperature,
        "bounds": list(TEMPERATURE_BOUNDS),
        "rows": len(y),
        "intercept": None,
    }


def apply_temperature(probability: float, temperature: float) -> float:
    p = float(probability)
    t = float(temperature)
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("series probability must be finite and interior")
    if not math.isfinite(t) or not TEMPERATURE_BOUNDS[0] <= t <= TEMPERATURE_BOUNDS[1]:
        raise ValueError("temperature is outside the frozen positive bounds")
    logit = math.log(p) - math.log1p(-p)
    return _logistic(logit / t)


def create_protocol(
    path: str | Path,
    *,
    research_root: str | Path | None = None,
    raw_history: str | Path | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Freeze source/input hashes and fixed recipe without running any fit."""
    from src.analysis.ev_approach_data import _resolve_research_root

    root = Path(_resolve_research_root(research_root)).resolve()
    raw_path = Path(raw_history).expanduser().resolve() if raw_history else (PROJECT_ROOT / RAW_HISTORY_RELATIVE).resolve()
    bank = root / BANK_RELATIVE
    receipt_path = bank / "completed.json"
    metadata_path = bank / "metadata.parquet"
    release_path = bank / "release_days.json"
    for input_path in (raw_path, receipt_path, metadata_path, release_path):
        if not input_path.is_file():
            raise FileNotFoundError(f"required family replay input is missing: {input_path}")
    receipt = _read_json(receipt_path)
    if receipt.get("status") not in {"COMPLETE", "PASS"}:
        raise ValueError("corrected-release bank is not complete")
    input_hashes = {
        "raw_history": _sha256(raw_path),
        "corrected_bank_receipt": _sha256(receipt_path),
        "canonical_metadata": _sha256(metadata_path),
        "corrected_release_days": _sha256(release_path),
    }
    if input_hashes["raw_history"] != receipt.get("source_sha256"):
        raise ValueError("raw history is not the source pinned by the corrected bank")
    expected_outputs = receipt.get("outputs_sha256", {})
    for name, actual_key in (
        ("metadata.parquet", "canonical_metadata"),
        ("release_days.json", "corrected_release_days"),
    ):
        expected = expected_outputs.get(name)
        if isinstance(expected, Mapping):
            expected = expected.get("sha256")
        if expected != input_hashes[actual_key]:
            raise ValueError(f"corrected bank receipt hash mismatch for {name}")
    metadata = pd.read_parquet(metadata_path)
    _validate_metadata(metadata)
    canonical_count = int(pd.to_datetime(metadata.date).dt.year.isin(EVALUATION_YEARS).sum())
    if canonical_count != EXPECTED_CANONICAL_ROWS:
        raise ValueError(
            f"corrected bank has {canonical_count} canonical rows; expected {EXPECTED_CANONICAL_ROWS}"
        )
    source_hashes = _source_hashes()
    output_base = PROJECT_ROOT / "data"
    stage_paths = {
        "protocol": str(Path(path).expanduser().resolve()),
        "recipe": str(Path(path).expanduser().resolve().with_name("family_pre_fit_recipe.json")),
        "models": str(output_base / "06_models" / "model_reliability" / run_id / "family"),
        "predictions": str(output_base / "07_model_output" / "model_reliability" / run_id / "family"),
        "receipt": str(output_base / "08_reporting" / "model_reliability" / run_id / "family"),
    }
    protocol = {
        "schema": "family_reliability_protocol_v1",
        "run_id": run_id,
        "research_root": str(root),
        "bank": str(bank),
        "raw_history": str(raw_path),
        "inputs_sha256": input_hashes,
        "sources_sha256": source_hashes,
        "expected_canonical_rows": EXPECTED_CANONICAL_ROWS,
        "evaluation_years": list(EVALUATION_YEARS),
        "calibration_years": list(CALIBRATION_YEARS),
        "stage_paths": stage_paths,
        "recipe": FAMILY_PRE_FIT_RECIPE,
        "pre_fit_only": True,
        "no_metrics": True,
    }
    _validate_protocol(protocol)
    protocol_path = Path(path).expanduser().resolve()
    recipe_path = protocol_path.with_name("family_pre_fit_recipe.json")
    _write_json_exclusive(recipe_path, FAMILY_PRE_FIT_RECIPE)
    protocol["recipe_path"] = str(recipe_path)
    protocol["recipe_sha256"] = _sha256(recipe_path)
    _write_json_exclusive(protocol_path, protocol)
    return protocol


def run_experiment(protocol_path: str | Path) -> dict[str, Any]:
    """Replay released raw history, fit prior-year CAL temperatures, and export rows."""
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = _read_json(protocol_file)
    _verify_protocol(protocol, protocol_file)
    for stage in ("models", "predictions", "receipt"):
        destination = Path(protocol["stage_paths"][stage])
        if destination.exists():
            raise FileExistsError(destination)
    raw_path = Path(protocol["raw_history"])
    bank = Path(protocol["bank"])
    metadata_path = bank / "metadata.parquet"
    release_path = bank / "release_days.json"
    input_hashes = {
        "raw_history": _sha256(raw_path),
        "corrected_bank_receipt": _sha256(bank / "completed.json"),
        "canonical_metadata": _sha256(metadata_path),
        "corrected_release_days": _sha256(release_path),
    }
    if input_hashes != protocol["inputs_sha256"]:
        raise ValueError("family experiment input hashes changed after protocol freeze")
    metadata = pd.read_parquet(metadata_path)
    _validate_metadata(metadata)
    forecast_rows = metadata.loc[
        pd.to_datetime(metadata.date).dt.year.between(2023, 2026)
    ].copy()
    if forecast_rows.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("family forecast membership has duplicate IDs")
    canonical = metadata.loc[
        pd.to_datetime(metadata.date).dt.year.isin(EVALUATION_YEARS)
    ].copy()
    if len(canonical) != EXPECTED_CANONICAL_ROWS:
        raise ValueError("family canonical assessment membership changed")
    with release_path.open("r", encoding="utf-8") as stream:
        release_days = json.load(stream)
    if not isinstance(release_days, Mapping):
        raise ValueError("corrected release-day input must be an object")

    history, history_audit = _load_history(raw_path, release_days)
    predictions, memberships, replay_summary, player_engine, bridge = _replay_forecasts(
        history, forecast_rows
    )
    target_records = _assemble_annual_predictions(
        canonical, forecast_rows, predictions, memberships
    )
    calibration_records = _annual_calibration_rows(
        forecast_rows, predictions, memberships
    )
    outputs = _stage_outputs(protocol, target_records, calibration_records, memberships)
    replay_summary.update({
        "history_audit": history_audit,
        "canonical_rows": len(target_records),
        "calibration_rows": len(calibration_records),
        "canonical_ids_sha256": _ids_hash(target_records.golgg_match_id.astype(str)),
        "source_membership_sha256": _ids_hash(
            sorted(str(row.match_id) for row in history)
        ),
        "final_player_state_sha256": _json_hash(player_engine.to_state()),
        "final_family_state_sha256": _json_hash(bridge.to_state()),
    })
    profile_receipt = _profile_receipt(target_records, replay_summary)
    outputs["family_graph_inventory.json"] = _write_json_exclusive(
        Path(protocol["stage_paths"]["predictions"]) / "family_graph_inventory.json",
        profile_receipt,
    )
    output_hashes = {
        name: _sha256(Path(path)) for name, path in outputs.items()
    }
    receipt = {
        "schema": "family_reliability_execution_receipt_v1",
        "update_policy": "released_daily_frozen_map_periods",
        "calibration_applies_to": "online_algorithm_not_frozen_weights",
        "family_evidence_status": "input_support_only_not_probabilistic_certification",
        "run_id": protocol["run_id"],
        "protocol_sha256": _sha256(protocol_file),
        "inputs_sha256": input_hashes,
        "sources_sha256": protocol["sources_sha256"],
        "input_paths": {
            "raw_history": str(raw_path),
            "corrected_bank_receipt": str(bank / "completed.json"),
            "canonical_metadata": str(metadata_path),
            "corrected_release_days": str(release_path),
        },
        "output_paths": outputs,
        "outputs_sha256": output_hashes,
        "rows": len(target_records),
        "calibration_rows": len(calibration_records),
        "replay": replay_summary,
        "calibration": json.loads(
            (Path(protocol["stage_paths"]["models"]) / "family_calibrators.json").read_text()
        ),
        "probability_metrics": "not computed by this work package",
    }
    receipt_dir = Path(protocol["stage_paths"]["receipt"])
    receipt_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = receipt_dir / "family_reliability_receipt.json"
    _write_json_exclusive(receipt_path, receipt)
    return {"receipt_path": str(receipt_path), **receipt}


def _load_history(
    raw_path: Path, release_days: Mapping[str, str]
) -> tuple[list[ReplaySeries], dict[str, Any]]:
    from scripts.export_prospective_features import iter_matches

    corrections = _validate_releases(release_days)
    match_ids: set[str] = set()
    game_ids: set[str] = set()
    counters: dict[str, Any] = {
        "source_series": 0,
        "source_maps": 0,
        "player_ids": set(),
        "identity_provenance": Counter(),
        "stat_seen": Counter(),
        "stat_missing": Counter(),
        "max_source_header_day": None,
        "max_source_map_day": None,
        "pre_series_date_corrections": [],
    }
    output: list[ReplaySeries] = []
    for source_index, raw in enumerate(iter_matches(raw_path)):
        item = _normalize_series(raw, source_index, match_ids, game_ids, corrections, counters)
        competition = classify_competition(item.tournament, item.day)
        source_games = raw["games"]
        maps = tuple(
            ReplayMap(
                game_id=game.game_id,
                day=_calendar_day(
                    source_game.get("date", raw.get("date")),
                    f"series {item.match_id} map date",
                ),
                roster_a=tuple(game.roster_a),
                roster_b=tuple(game.roster_b),
                outcome=int(game.y_a),
            )
            for game, source_game in zip(item.maps, source_games, strict=True)
        )
        output.append(
            ReplaySeries(
                match_id=item.match_id,
                day=item.day,
                source_header_day=item.source_header_day,
                release_day=item.release_day,
                team_a=item.team_a,
                team_b=item.team_b,
                best_of=item.best_of,
                tournament=item.tournament,
                competition=competition,
                maps=maps,
            )
        )
    output.sort(key=lambda row: (row.day, row.match_id))
    audit = {
        "series": len(output),
        "maps": sum(len(row.maps) for row in output),
        "corrected_release_matches": sum(row.match_id in corrections for row in output),
        "release_correction_entries": len(corrections),
        "unique_match_ids": len(match_ids),
        "unique_map_ids": len(game_ids),
        "unique_players": len(counters["player_ids"]),
        "pre_series_date_corrections": len(counters["pre_series_date_corrections"]),
        "max_source_header_day": _iso(counters["max_source_header_day"]),
        "max_source_map_day": _iso(counters["max_source_map_day"]),
    }
    return output, audit


def _replay_forecasts(
    history: Sequence[ReplaySeries], targets: pd.DataFrame
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    FamilyCalibratedGlicko2,
    FamilyMapBridge,
]:
    player_engine = FamilyCalibratedGlicko2(**PLAYER_CONFIG)
    bridge = FamilyMapBridge(**BRIDGE_CONFIG)
    affiliations = ReleasedAffiliations()
    graph = FamilyBridgeGraph()
    by_start: dict[date, list[tuple[ReplaySeries, ReplayMap]]] = defaultdict(list)
    by_release: dict[date, list[ReplaySeries]] = defaultdict(list)
    by_target: dict[date, list[dict[str, Any]]] = defaultdict(list)
    series_by_id: dict[str, ReplaySeries] = {}
    for item in history:
        by_release[item.release_day].append(item)
        if item.match_id in series_by_id:
            raise ValueError(f"duplicate normalized history series: {item.match_id}")
        series_by_id[item.match_id] = item
        for game in item.maps:
            by_start[game.day].append((item, game))
    for column in _REQUIRED_METADATA:
        if column not in targets:
            raise ValueError(f"family target rows lack {column}")
    for row in targets.to_dict("records"):
        by_target[_calendar_day(row["date"], "target date")].append(row)
    for values in by_start.values():
        values.sort(key=lambda pair: (pair[0].match_id, pair[1].game_id))
    for values in by_release.values():
        values.sort(key=lambda row: row.match_id)
    for values in by_target.values():
        values.sort(key=lambda row: str(row["golgg_match_id"]))

    all_days = sorted(set(by_start) | set(by_target))
    release_calendar = sorted(by_release)
    release_position = 0
    pending_maps: dict[str, dict[str, Any]] = {}
    raw_predictions: dict[str, dict[str, Any]] = {}
    membership_receipts: dict[str, dict[str, Any]] = {}
    state_chain = _json_hash({"player_config": PLAYER_CONFIG, "bridge_config": BRIDGE_CONFIG})
    consumed_release_day: date | None = None
    replay_batches = 0
    released_maps = 0

    def consume_release_day(release_day: date) -> None:
        nonlocal state_chain, consumed_release_day, replay_batches, released_maps
        records = sorted(by_release[release_day], key=lambda row: row.match_id)
        events: list[RatingEvent] = []
        observations: list[dict[str, Any]] = []
        release_players: set[str] = set()
        map_ids: list[str] = []
        outcomes: list[int] = []
        for series in records:
            for game in series.maps:
                frozen = pending_maps.pop(game.game_id, None)
                if frozen is None:
                    raise ValueError(
                        f"map {game.game_id} was released before its pre-result context was frozen"
                    )
                events.append(
                    RatingEvent(
                        event_id=game.game_id,
                        event_date=release_day,
                        team_a_id=series.team_a,
                        team_b_id=series.team_b,
                        players_a=game.roster_a,
                        players_b=game.roster_b,
                        family_a="unknown",
                        family_b="unknown",
                        tier_a="unknown",
                        tier_b="unknown",
                        scores=(game.outcome,),
                        best_of=1,
                    )
                )
                observation = {
                    "player_probability": frozen["player_probability"],
                    "family_weights_a": frozen["family_weights_a"],
                    "family_weights_b": frozen["family_weights_b"],
                    "outcome": game.outcome,
                    "weight": 1.0 / len(series.maps),
                }
                observations.append(observation)
                graph.consume(observation)
                release_players.update((*game.roster_a, *game.roster_b))
                map_ids.append(game.game_id)
                outcomes.append(game.outcome)
                released_maps += 1
        if events:
            player_engine.process_period(events)
        bridge.update_period(release_day, observations)
        affiliation_changes = affiliations.consume_period(release_day, records)
        player_states = {
            player_id: _player_state_json(player_engine.get_player_state(player_id))
            for player_id in sorted(release_players)
        }
        state_chain = _json_hash({
            "previous_state_chain_sha256": state_chain,
            "release_day": release_day.isoformat(),
            "series": [row.match_id for row in records],
            "map_ids": map_ids,
            "outcomes": outcomes,
            "updated_player_states": player_states,
            "family_state": bridge.to_state(),
            "affiliation_changes": affiliation_changes,
        })
        consumed_release_day = release_day
        replay_batches += 1

    for query_day in all_days:
        while release_position < len(release_calendar) and release_calendar[release_position] < query_day:
            consume_release_day(release_calendar[release_position])
            release_position += 1

        # Canonical lineup predictions are queries before all releases on this day.
        for row in by_target.get(query_day, ()):
            match_id = str(row["golgg_match_id"])
            roster_a = _validated_roster(row["roster_a"], f"{match_id} roster_a")
            roster_b = _validated_roster(row["roster_b"], f"{match_id} roster_b")
            team_a, team_b = str(row["team1_id"]), str(row["team2_id"])
            best_of = _positive_odd_integer(row["best_of"], "best_of")
            side_a = affiliations.profile(roster_a, query_day)
            side_b = affiliations.profile(roster_b, query_day)
            event = _player_event(
                match_id, query_day, team_a, team_b, roster_a, roster_b, 0, 1
            )
            player_map_probability = _interior_probability(player_engine.predict_event(event))
            family_map_probability = _interior_probability(bridge.predict_map_probability(
                player_map_probability,
                side_a["weights"],
                side_b["weights"],
                query_day,
            ))
            raw_player_series = _interior_probability(
                series_win_probability(player_map_probability, best_of)
            )
            raw_family_series = _interior_probability(
                series_win_probability(family_map_probability, best_of)
            )
            target_series = series_by_id.get(match_id)
            missing_reason = None
            if target_series is None:
                missing_reason = "raw_source_match_missing"
            else:
                _validate_target_source_row(row, target_series)
            feature_day = _iso_or_none(row.get("feature_source_max_day"))
            if feature_day is not None and feature_day >= query_day.isoformat():
                raise ValueError(f"target feature source reaches prediction day for {match_id}")
            competition = classify_competition(
                row.get("tournament_name"), query_day
            )
            graph_support = graph.support(side_a["weights"], side_b["weights"])
            profile = _family_profile(side_a, side_b)
            state_receipt = {
                "query_day": query_day.isoformat(),
                "query_release_cutoff_exclusive": query_day.isoformat(),
                "state_chain_sha256": state_chain,
                "last_released_day": _iso(consumed_release_day),
                "roster_a": list(roster_a),
                "roster_b": list(roster_b),
                "affiliation_a": side_a["receipts"],
                "affiliation_b": side_b["receipts"],
            }
            receipt_id = _json_hash(state_receipt)
            raw_predictions[match_id] = {
                "p_player_raw": raw_player_series,
                "p_family_raw": raw_family_series,
                "p_player_map_raw": player_map_probability,
                "p_family_map_raw": family_map_probability,
                "competition": competition,
                "profile": profile,
                "side_a": side_a,
                "side_b": side_b,
                "player_support_a": _player_roster_support(player_engine, roster_a, query_day),
                "player_support_b": _player_roster_support(player_engine, roster_b, query_day),
                "graph_support": graph_support,
                "missing_reason": missing_reason,
                "state_receipt_sha256": receipt_id,
                "state_chain_sha256": state_chain,
                "last_released_day": consumed_release_day,
                "player_rosters": (roster_a, roster_b),
                "query_day": query_day,
            }
            membership_receipts[match_id] = {
                "golgg_match_id": match_id,
                "date": query_day.isoformat(),
                "result_day": _iso(_calendar_day(row["result_day"], "result_day")),
                "query_release_cutoff_exclusive": query_day.isoformat(),
                "last_released_day": _iso(consumed_release_day),
                "state_chain_sha256": state_chain,
                "state_receipt_sha256": receipt_id,
                "canonical_roster_a": list(roster_a),
                "canonical_roster_b": list(roster_b),
                "roster_receipt_sha256": _json_hash([list(roster_a), list(roster_b)]),
                "affiliation_a": side_a["receipts"],
                "affiliation_b": side_b["receipts"],
                "source_match_id": match_id if target_series is not None else None,
                "missing_reason": missing_reason,
                "family_evidence_status": "input_support_only_not_probabilistic_certification",
                "lineup_scenario": "full_canonical_roster_known_lineup_scenario",
                "update_policy": "released_daily_frozen_map_periods",
                "calibration_applies_to": "online_algorithm_not_frozen_weights",
            }

        # Freeze each historical map's own actual lineup/context on its map day.
        for series, game in by_start.get(query_day, ()):
            side_a = affiliations.profile(game.roster_a, game.day)
            side_b = affiliations.profile(game.roster_b, game.day)
            event = _player_event(
                game.game_id,
                game.day,
                series.team_a,
                series.team_b,
                game.roster_a,
                game.roster_b,
                game.outcome,
                1,
            )
            player_probability = _interior_probability(player_engine.predict_event(event))
            pending_maps[game.game_id] = {
                "player_probability": player_probability,
                "family_weights_a": side_a["weights"],
                "family_weights_b": side_b["weights"],
            }

        while release_position < len(release_calendar) and release_calendar[release_position] == query_day:
            consume_release_day(release_calendar[release_position])
            release_position += 1

    while release_position < len(release_calendar):
        consume_release_day(release_calendar[release_position])
        release_position += 1
    if pending_maps:
        raise ValueError(f"raw replay left unreleased maps: {sorted(pending_maps)[:5]}")
    summary = {
        "player_state_contract": "identical shared player-only Glicko-2 for R0 and R1",
        "update_policy": "released_daily_frozen_map_periods",
        "calibration_applies_to": "online_algorithm_not_frozen_weights",
        "player_parameters": PLAYER_CONFIG,
        "family_bridge_parameters": BRIDGE_CONFIG,
        "ttl_days": TTL_DAYS,
        "replay_batches": replay_batches,
        "released_maps": released_maps,
        "pending_maps": 0,
        "final_release_day": _iso(consumed_release_day),
        "state_chain_sha256": state_chain,
        "affiliation_player_count": len(affiliations._states),
        "final_affiliation_state_sha256": _json_hash(affiliations.state_receipt()),
        "affiliation_conflict_count": len(affiliations._conflicts),
        "family_graph": graph.to_json(),
        "unknown_affiliation_target_rows": sum(
            row["profile"]["profile"] in {"unknown", "partially_unknown"}
            for row in raw_predictions.values()
        ),
        "mixed_affiliation_target_rows": sum(
            row["profile"]["profile"] == "mixed_affiliations"
            for row in raw_predictions.values()
        ),
    }
    return raw_predictions, membership_receipts, summary, player_engine, bridge


def _player_event(
    event_id: str,
    day: date,
    team_a: str,
    team_b: str,
    roster_a: Sequence[str],
    roster_b: Sequence[str],
    outcome: int,
    best_of: int,
) -> RatingEvent:
    return RatingEvent(
        event_id=event_id,
        event_date=day,
        team_a_id=team_a,
        team_b_id=team_b,
        players_a=tuple(roster_a),
        players_b=tuple(roster_b),
        family_a="unknown",
        family_b="unknown",
        tier_a="unknown",
        tier_b="unknown",
        scores=(outcome,),
        best_of=best_of,
    )


def _assemble_annual_predictions(
    canonical: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    forecasts: Mapping[str, Mapping[str, Any]],
    memberships: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    by_id = {str(row["golgg_match_id"]): row for row in forecast_rows.to_dict("records")}
    canonical_ids = canonical.golgg_match_id.astype(str).tolist()
    if len(set(canonical_ids)) != EXPECTED_CANONICAL_ROWS:
        raise ValueError("canonical 2024-2026 IDs are not unique")
    rows: list[dict[str, Any]] = []
    calibrators: dict[str, dict[str, Any]] = {}
    for year in EVALUATION_YEARS:
        origin = date(year, 1, 1)
        calibration_records = []
        for match_id, row in by_id.items():
            result_day = _calendar_day(row["result_day"], "CAL result_day")
            if (
                _calendar_day(row["date"], "CAL date").year == year - 1
                and result_day < origin
            ):
                prediction = forecasts.get(match_id)
                if prediction is not None:
                    calibration_records.append((match_id, row, prediction))
        calibration_records.sort(key=lambda item: item[0])
        if not calibration_records:
            raise ValueError(f"no released prior-calendar-year CAL rows for {year}")
        temp_player, player_fit = fit_temperature(
            [item[2]["p_player_raw"] for item in calibration_records],
            [_binary_outcome(item[1]["y"], "CAL y") for item in calibration_records],
        )
        temp_family, family_fit = fit_temperature(
            [item[2]["p_family_raw"] for item in calibration_records],
            [_binary_outcome(item[1]["y"], "CAL y") for item in calibration_records],
        )
        cal_end = max(
            _calendar_day(item[1]["result_day"], "CAL result_day")
            for item in calibration_records
        )
        if cal_end >= origin:
            raise ValueError("family CAL result release reached its annual origin")
        calibrators[str(year)] = {
            "origin": origin.isoformat(),
            "calibration_year": year - 1,
            "calibration_rows": len(calibration_records),
            "calibration_ids_sha256": _ids_hash(item[0] for item in calibration_records),
            "calibration_end": cal_end.isoformat(),
            "player": player_fit,
            "family": family_fit,
        }
        annual_rows = canonical.loc[pd.to_datetime(canonical.date).dt.year.eq(year)]
        for row in annual_rows.to_dict("records"):
            match_id = str(row["golgg_match_id"])
            forecast = forecasts.get(match_id)
            membership = memberships.get(match_id)
            if forecast is None or membership is None:
                raise ValueError(f"canonical target forecast or membership receipt missing: {match_id}")
            target_day = _calendar_day(row["date"], "target date")
            if cal_end >= target_day:
                raise ValueError(f"calibration release is not prior to canonical row {match_id}")
            history_end = forecast["last_released_day"]
            if history_end is not None and history_end >= target_day:
                raise ValueError(f"player history release is not prior to target {match_id}")
            profile = forecast["profile"]
            side_a, side_b = forecast["side_a"], forecast["side_b"]
            support_a, support_b = forecast["player_support_a"], forecast["player_support_b"]
            graph_support = forecast["graph_support"]
            status = "complete"
            missing_reason = forecast["missing_reason"]
            if missing_reason is not None:
                status = "forecast_with_raw_source_gap"
            rows.append({
                "golgg_match_id": match_id,
                "team1_id": str(row["team1_id"]),
                "team2_id": str(row["team2_id"]),
                "date": target_day.isoformat(),
                "result_day": _calendar_day(row["result_day"], "result_day").isoformat(),
                "feature_source_max_day": _iso_or_none(row.get("feature_source_max_day")),
                "best_of": int(row["best_of"]),
                "y": int(row["y"]),
                "p_player_raw": forecast["p_player_raw"],
                "p_family_raw": forecast["p_family_raw"],
                "p_player": apply_temperature(forecast["p_player_raw"], temp_player),
                "p_family": apply_temperature(forecast["p_family_raw"], temp_family),
                "forecast_status": status,
                "missing_reason": missing_reason,
                "lineup_scenario": "full_canonical_roster_known_lineup_scenario",
                "family_evidence_status": "input_support_only_not_probabilistic_certification",
                "roster_a": list(forecast["player_rosters"][0]),
                "roster_b": list(forecast["player_rosters"][1]),
                "roster_receipt_sha256": membership["roster_receipt_sha256"],
                "competition_family": forecast["competition"].family,
                "competition_tier": forecast["competition"].tier.value,
                "competition_scope": forecast["competition"].scope.value,
                "competition_rule": forecast["competition"].matched_rule,
                "family_profile": profile["profile"],
                "family_known_players_a": side_a["known_players"],
                "family_known_players_b": side_b["known_players"],
                "family_unknown_players_a": side_a["unknown_players"],
                "family_unknown_players_b": side_b["unknown_players"],
                "family_known_fraction_a": side_a["known_fraction"],
                "family_known_fraction_b": side_b["known_fraction"],
                "family_mixed_a": side_a["mixed"],
                "family_mixed_b": side_b["mixed"],
                "family_weights_a": _json_text(side_a["weights"]),
                "family_weights_b": _json_text(side_b["weights"]),
                "affiliation_receipt_a": _json_text(side_a["receipts"]),
                "affiliation_receipt_b": _json_text(side_b["receipts"]),
                "affiliation_unknown_reasons_a": _json_text(side_a["unknown_reasons"]),
                "affiliation_unknown_reasons_b": _json_text(side_b["unknown_reasons"]),
                "transfer_players_a": side_a["transfer_players"],
                "transfer_players_b": side_b["transfer_players"],
                "recent_transfer_players_a": side_a["recent_transfer_players"],
                "recent_transfer_players_b": side_b["recent_transfer_players"],
                "affiliation_max_age_days_a": side_a["max_affiliation_age_days"],
                "affiliation_max_age_days_b": side_b["max_affiliation_age_days"],
                "player_games_min_a": support_a["min_games"],
                "player_games_min_b": support_b["min_games"],
                "player_unseen_count_a": support_a["unseen_count"],
                "player_unseen_count_b": support_b["unseen_count"],
                "player_inactivity_days_max_a": support_a["max_inactivity_days"],
                "player_inactivity_days_max_b": support_b["max_inactivity_days"],
                "player_raw_rd_max_a": support_a["max_raw_rd"],
                "player_raw_rd_max_b": support_b["max_raw_rd"],
                "family_bridge_edge_support": graph_support["edge_support"],
                "family_bridge_component_status": graph_support["status"],
                "family_bridge_active_families": _json_text(graph_support["active_families"]),
                "family_bridge_component_count": graph_support["connected_components"],
                "state_receipt_sha256": forecast["state_receipt_sha256"],
                "state_chain_sha256": forecast["state_chain_sha256"],
                "query_cutoff_exclusive": target_day.isoformat(),
                "train_end": _iso(history_end),
                "train_state_empty": history_end is None,
                "state_max_release_day": _iso(history_end),
                "calibration_end": cal_end.isoformat(),
                "calibration_source_end": cal_end.isoformat(),
                "calibration_origin": origin.isoformat(),
                "calibration_year": year - 1,
                "calibration_applies_to": "online_algorithm_not_frozen_weights",
                "update_policy": "released_daily_frozen_map_periods",
                "temperature_player": temp_player,
                "temperature_family": temp_family,
                "target_membership_sha256": _json_hash({
                    "id": match_id,
                    "date": target_day.isoformat(),
                    "result_day": _calendar_day(row["result_day"], "result_day").isoformat(),
                    "team1_id": str(row["team1_id"]),
                    "team2_id": str(row["team2_id"]),
                    "best_of": int(row["best_of"]),
                    "y": int(row["y"]),
                    "roster_a": forecast["player_rosters"][0],
                    "roster_b": forecast["player_rosters"][1],
                }),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_CANONICAL_ROWS or frame.golgg_match_id.duplicated().any():
        raise ValueError("family output does not cover the exact canonical 2024-2026 population")
    frame.attrs["calibrators"] = calibrators
    return frame


def _annual_calibration_rows(
    forecast_rows: pd.DataFrame,
    forecasts: Mapping[str, Mapping[str, Any]],
    memberships: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for year in CALIBRATION_YEARS:
        origin = date(year + 1, 1, 1)
        subset = forecast_rows.loc[pd.to_datetime(forecast_rows.date).dt.year.eq(year)]
        for row in subset.to_dict("records"):
            match_id = str(row["golgg_match_id"])
            release_day = _calendar_day(row["result_day"], "CAL result_day")
            if release_day >= origin:
                continue
            prediction = forecasts.get(match_id)
            receipt = memberships.get(match_id)
            if prediction is None or receipt is None:
                raise ValueError(f"annual CAL forecast is missing for {match_id}")
            output.append({
                "golgg_match_id": match_id,
                "team1_id": str(row["team1_id"]),
                "team2_id": str(row["team2_id"]),
                "roster_a": list(prediction["player_rosters"][0]),
                "roster_b": list(prediction["player_rosters"][1]),
                "date": _calendar_day(row["date"], "CAL date").isoformat(),
                "result_day": release_day.isoformat(),
                "best_of": int(row["best_of"]),
                "y": int(row["y"]),
                "p_player_raw": prediction["p_player_raw"],
                "p_family_raw": prediction["p_family_raw"],
                "calibration_for_year": year + 1,
                "calibration_origin": origin.isoformat(),
                "state_receipt_sha256": prediction["state_receipt_sha256"],
                "roster_receipt_sha256": receipt["roster_receipt_sha256"],
            })
    frame = pd.DataFrame(output)
    if frame.empty or frame.golgg_match_id.duplicated().any():
        raise ValueError("annual family calibration rows are empty or duplicated")
    return frame


def _stage_outputs(
    protocol: Mapping[str, Any],
    predictions: pd.DataFrame,
    calibration_rows: pd.DataFrame,
    memberships: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    model_dir = Path(protocol["stage_paths"]["models"])
    prediction_dir = Path(protocol["stage_paths"]["predictions"])
    model_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir.mkdir(parents=True, exist_ok=False)
    calibrators = predictions.attrs.get("calibrators")
    if not isinstance(calibrators, Mapping):
        raise ValueError("family temperature receipts were not retained")
    calibrator_path = model_dir / "family_calibrators.json"
    _write_json_exclusive(calibrator_path, {
        "schema": "family_annual_calibrators_v1",
        "run_id": protocol["run_id"],
        "recipe_sha256": protocol["recipe_sha256"],
        "update_policy": "released_daily_frozen_map_periods",
        "calibration_applies_to": "online_algorithm_not_frozen_weights",
        "family_evidence_status": "input_support_only_not_probabilistic_certification",
        "calibrators": calibrators,
    })
    prediction_path = prediction_dir / "predictions.parquet"
    calibration_path = prediction_dir / "annual_calibration_predictions.parquet"
    receipt_path = prediction_dir / "membership_receipts.jsonl"
    _write_parquet_exclusive(prediction_path, predictions)
    _write_parquet_exclusive(calibration_path, calibration_rows)
    with receipt_path.open("x", encoding="utf-8") as stream:
        for match_id in sorted(memberships):
            stream.write(_json_text(memberships[match_id]) + "\n")
    return {
        "family_calibrators.json": str(calibrator_path),
        "predictions.parquet": str(prediction_path),
        "annual_calibration_predictions.parquet": str(calibration_path),
        "membership_receipts.jsonl": str(receipt_path),
    }


def _profile_receipt(frame: pd.DataFrame, replay: Mapping[str, Any]) -> dict[str, Any]:
    profile_counts = frame.family_profile.value_counts(dropna=False).sort_index()
    status_counts = frame.family_bridge_component_status.value_counts(dropna=False).sort_index()
    return {
        "schema": "family_reliability_profile_inventory_v1",
        "rows": int(len(frame)),
        "family_evidence_status": "input_support_only_not_probabilistic_certification",
        "family_profile_counts": {str(key): int(value) for key, value in profile_counts.items()},
        "bridge_component_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "family_graph": replay["family_graph"],
        "affiliation_player_count": replay["affiliation_player_count"],
        "affiliation_conflict_count": replay["affiliation_conflict_count"],
        "missing_reason_counts": {
            str(key): int(value)
            for key, value in frame.missing_reason.fillna("none").value_counts().sort_index().items()
        },
        "coverage": {
            "canonical_rows": int(len(frame)),
            "required_canonical_rows": EXPECTED_CANONICAL_ROWS,
            "all_rows_retained": int(len(frame)) == EXPECTED_CANONICAL_ROWS,
        },
    }


def _family_profile(side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> dict[str, str]:
    all_known = side_a["known_players"] == 5 and side_b["known_players"] == 5
    if not side_a["known_players"] and not side_b["known_players"]:
        value = "unknown"
    elif side_a["mixed"] or side_b["mixed"]:
        value = "mixed_affiliations"
    elif not all_known:
        value = "partially_unknown"
    elif set(side_a["weights"]) == set(side_b["weights"]):
        value = "known_same_family"
    else:
        value = "known_cross_family"
    return {"profile": value}


def _validate_target_source_row(row: Mapping[str, Any], source: ReplaySeries) -> None:
    match_id = str(row["golgg_match_id"])
    teams = str(row["team1_id"]), str(row["team2_id"])
    if teams == (source.team_a, source.team_b):
        reverse = False
    elif teams == (source.team_b, source.team_a):
        reverse = True
    else:
        raise ValueError(f"canonical/source team identity mismatch for {match_id}")
    checks = (
        ("date", _calendar_day(row["date"], "target date"), source.day),
        ("result_day", _calendar_day(row["result_day"], "target result_day"), source.release_day),
        ("best_of", int(row["best_of"]), source.best_of),
    )
    for field, left, right in checks:
        if left != right:
            raise ValueError(f"canonical/source {field} mismatch for {match_id}: {left!r} != {right!r}")
    first = source.maps[0]
    expected_rosters = (first.roster_b, first.roster_a) if reverse else (first.roster_a, first.roster_b)
    for side, actual in zip(("a", "b"), expected_rosters, strict=True):
        if set(map(str, row[f"roster_{side}"])) != set(actual):
            raise ValueError(f"canonical/source roster_{side} mismatch for {match_id}")
    outcomes = [1-game.outcome if reverse else game.outcome for game in source.maps]
    if source.best_of in (1, 3, 5):
        wins_a, wins_b = sum(outcomes), len(outcomes) - sum(outcomes)
        if int(row["y"]) != int(wins_a > wins_b):
            raise ValueError(f"canonical/source outcome mismatch for {match_id}")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema") != "family_reliability_protocol_v1"
        or not isinstance(protocol.get("run_id"), str)
        or not protocol["run_id"]
        or Path(protocol["run_id"]).name != protocol["run_id"]
        or protocol["run_id"] in {".", ".."}
        or protocol.get("evaluation_years") != list(EVALUATION_YEARS)
        or protocol.get("calibration_years") != list(CALIBRATION_YEARS)
        or protocol.get("expected_canonical_rows") != EXPECTED_CANONICAL_ROWS
        or protocol.get("recipe") != FAMILY_PRE_FIT_RECIPE
        or protocol.get("pre_fit_only") is not True
        or protocol.get("no_metrics") is not True
    ):
        raise ValueError("family reliability protocol differs from the frozen work package")
    for field in ("research_root", "bank", "raw_history", "inputs_sha256", "sources_sha256", "stage_paths"):
        if field not in protocol:
            raise ValueError(f"family protocol missing {field}")


def _verify_protocol(protocol: Mapping[str, Any], protocol_file: Path) -> None:
    _validate_protocol(protocol)
    if _source_hashes() != protocol.get("sources_sha256"):
        raise ValueError("family code/source hashes differ from the frozen protocol")
    recipe_path = Path(protocol.get("recipe_path", ""))
    if not recipe_path.is_file() or _sha256(recipe_path) != protocol.get("recipe_sha256"):
        raise ValueError("machine-readable pre-fit recipe changed after freeze")
    if _read_json(recipe_path) != FAMILY_PRE_FIT_RECIPE:
        raise ValueError("machine-readable pre-fit recipe does not match the implementation")
    if not protocol_file.is_file():
        raise FileNotFoundError(f"family protocol does not exist: {protocol_file}")


def _validate_metadata(frame: pd.DataFrame) -> None:
    missing = _REQUIRED_METADATA.difference(frame.columns)
    if missing:
        raise ValueError(f"corrected metadata missing family target fields: {sorted(missing)}")
    ids = frame.golgg_match_id.astype(str)
    if frame.golgg_match_id.isna().any() or ids.duplicated().any():
        raise ValueError("corrected metadata has null or duplicate match IDs")
    for field in ("team1_id", "team2_id"):
        if frame[field].isna().any():
            raise ValueError(f"corrected metadata has null {field}")
    for field in ("date", "result_day"):
        parsed = pd.to_datetime(frame[field], errors="raise").dt.normalize()
        if parsed.isna().any():
            raise ValueError(f"corrected metadata has invalid {field}")
    if not frame.best_of.map(lambda value: _positive_odd_integer(value, "best_of")).all():
        raise ValueError("canonical metadata best_of is invalid")
    if not frame.y.map(lambda value: _binary_outcome(value, "y") in (0, 1)).all():
        raise ValueError("canonical metadata outcome is invalid")
    for field in ("roster_a", "roster_b"):
        for value in frame[field]:
            _validated_roster(value, field)


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in _SOURCE_FILES
    }


def _player_state_json(state: Any) -> dict[str, float]:
    return {"rating": state.rating, "rd": state.rd, "volatility": state.volatility}


def _player_roster_support(
    player_engine: FamilyCalibratedGlicko2,
    roster: Sequence[str],
    query_day: date,
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for player_id in roster:
        games = player_engine.get_player_games(player_id)
        last_activity = player_engine.get_player_last_activity(player_id)
        inactivity = (
            None if last_activity is None else max(0, (query_day - last_activity).days)
        )
        state = (
            player_engine.get_player_state(player_id)
            if games > 0
            else None
        )
        values.append({
            "games": games,
            "inactivity_days": inactivity,
            "raw_rd": None if state is None else state.rd,
        })
    raw_rds = [value["raw_rd"] for value in values if value["raw_rd"] is not None]
    inactivity = [
        value["inactivity_days"]
        for value in values
        if value["inactivity_days"] is not None
    ]
    return {
        "min_games": min(value["games"] for value in values),
        "unseen_count": sum(value["games"] == 0 for value in values),
        "max_inactivity_days": max(inactivity) if inactivity else None,
        "max_raw_rd": max(raw_rds) if raw_rds else None,
    }


def _validated_roster(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError(f"{field} must contain exactly five player IDs")
    try:
        players = tuple(_identifier(player, field) for player in value)
    except TypeError as error:
        raise ValueError(f"{field} must contain exactly five player IDs") from error
    if len(players) != 5 or len(set(players)) != 5:
        raise ValueError(f"{field} must contain five unique player IDs")
    return players


def _positive_odd_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive odd integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a positive odd integer") from error
    if numeric != value or numeric <= 0 or numeric % 2 == 0:
        raise ValueError(f"{field} must be a positive odd integer")
    return numeric


def _binary_outcome(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be binary") from error
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError(f"{field} must be binary")
    return int(number)


def _identifier(value: Any, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} contains an invalid player ID")
    text = str(value)
    if not text or text != text.strip() or text.lower() in {"none", "nan", "null"}:
        raise ValueError(f"{field} contains an invalid player ID")
    return text


def _calendar_day(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        result = value.date()
    elif isinstance(value, date):
        result = value
    else:
        try:
            result = pd.Timestamp(value).date()
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field} must be a calendar date") from error
    if not isinstance(result, date):
        raise ValueError(f"{field} must be a calendar date")
    return result


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _calendar_day(value, "date").isoformat()


def _interior_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ArithmeticError("forecast probability must be a finite interior float")
    return probability


def _logistic(value: float) -> float:
    if not math.isfinite(value):
        raise ArithmeticError("forecast logit must be finite")
    if value >= 0.0:
        tail = math.exp(-value)
        probability = 1.0 / (1.0 + tail)
    else:
        tail = math.exp(value)
        probability = tail / (1.0 + tail)
    return _interior_probability(probability)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _ids_hash(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_exclusive(path: str | Path, value: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return str(destination)


def _write_parquet_exclusive(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite family output: {destination}")
    frame.to_parquet(destination, index=False)


def _smoke() -> dict[str, Any]:
    """Run deterministic in-memory contract checks without replay, fitting, or scoring."""
    bridge = FamilyMapBridge(**BRIDGE_CONFIG)
    sample = {
        "player_probability": 0.5,
        "family_weights_a": {"LCK": 1.0},
        "family_weights_b": {"LPL": 1.0},
        "outcome": 1,
        "weight": 1.0,
    }
    bridge.update_period(date(2024, 1, 1), [sample])
    before = bridge.to_state()
    forward = bridge.predict_map_probability(
        0.5, {"LCK": 1.0}, {"LPL": 1.0}, date(2024, 1, 2)
    )
    reverse = bridge.predict_map_probability(
        0.5, {"LPL": 1.0}, {"LCK": 1.0}, date(2024, 1, 2)
    )
    if not math.isclose(forward + reverse, 1.0, abs_tol=1e-14):
        raise AssertionError("family bridge side symmetry failed")
    if bridge.to_state() != before:
        raise AssertionError("read-only future query mutated bridge state")
    return {"status": "SMOKE_OK", "fitting": False, "scoring": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--write-protocol", type=Path)
    parser.add_argument("--run-id", default=RUN_ID, help="New immutable run identity when writing a protocol")
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--raw-history", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.write_protocol is not None:
        create_protocol(
            args.write_protocol,
            research_root=args.research_root,
            raw_history=args.raw_history,
            run_id=args.run_id,
        )
        print(f"FROZEN_PROTOCOL {args.write_protocol}")
        return 0
    if args.smoke:
        if args.protocol is not None:
            protocol = _read_json(args.protocol)
            _verify_protocol(protocol, args.protocol.resolve())
        print(_json_text(_smoke()))
        return 0
    if args.protocol is None:
        parser.error("--protocol is required unless --smoke or --write-protocol is supplied")
    result = run_experiment(args.protocol)
    print(f"FAMILY_RELIABILITY_COMPLETE {result['receipt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
