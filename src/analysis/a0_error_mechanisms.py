"""Roster, history and organization diagnostics for the frozen A0 error audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import binary_log_loss_vector, evaluate_binary_probabilities

ROOT = Path(__file__).resolve().parents[2]
STAT_NAMES = ("kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm", "gd@15", "csd@15", "xpd@15", "kp%", "dmg%")
PLAYER_LOG_DAYS_INACTIVE = 13
PLAYER_UNSEEN = 14
HISTORY_STAT_VALUES = slice(0, 12)
HISTORY_STAT_PRESENT = slice(12, 24)
EPSILON = 1e-12
DRAWS = 5000
SEED = 20260930
TAX_RATE = 0.12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any, default: Any = None) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value


def _roster(value: Any) -> list[str]:
    decoded = _decode(value, [])
    if not isinstance(decoded, (list, tuple, np.ndarray)):
        return []
    return [str(item) for item in decoded if item is not None and not pd.isna(item)]


def _float_array(value: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)


def _bool(value: Any) -> bool:
    return False if value is None or pd.isna(value) else bool(value)


def _monthly_interval(values: np.ndarray, months: pd.Series, universe: list[str], weights: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    month_values = months.astype(str).to_numpy()
    weight_values = np.ones(len(values), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(values) != len(month_values) or len(values) != len(weight_values):
        raise ValueError("bootstrap values, months and weights must align")
    if not np.isfinite(values).all() or not np.isfinite(weight_values).all():
        raise ValueError("bootstrap values and weights must be finite")
    grouped = pd.DataFrame({"month": month_values, "numerator": values * weight_values, "denominator": weight_values}).groupby("month", sort=True).sum()
    grouped = grouped.reindex(universe, fill_value=0.0)
    return block_ratio_interval(grouped.numerator, grouped.denominator, grouped.index.astype(str), DRAWS, SEED)


def _proxy_interval(values: np.ndarray, blocks: pd.Series, universe: list[str], weights: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    block_values = blocks.fillna("unknown").astype(str).to_numpy()
    weight_values = np.ones(len(values), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    grouped = pd.DataFrame({"block": block_values, "numerator": values * weight_values, "denominator": weight_values}).groupby("block", sort=True).sum()
    grouped = grouped.reindex(universe, fill_value=0.0)
    return block_ratio_interval(grouped.numerator, grouped.denominator, grouped.index.astype(str), DRAWS, SEED)


def _metric_effect(frame: pd.DataFrame, y: np.ndarray, base: np.ndarray, alt: np.ndarray, label: str,
                   month_universe: list[str], *, weights: np.ndarray | None = None,
                   block_universe: list[str] | None = None, blocks: pd.Series | None = None) -> dict[str, Any]:
    base = np.asarray(base, dtype=float)
    alt = np.asarray(alt, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (len(base) == len(alt) == len(y) == len(frame)):
        raise ValueError(f"unaligned probabilities for {label}")
    ll_base = binary_log_loss_vector(y, base, epsilon=EPSILON)
    ll_alt = binary_log_loss_vector(y, alt, epsilon=EPSILON)
    brier_base = np.square(base - y)
    brier_alt = np.square(alt - y)
    selected_weights = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    out = {
        "contrast": label,
        "n": int(selected_weights.sum()),
        "base_log_loss": float(np.average(ll_base, weights=selected_weights)) if selected_weights.sum() else None,
        "alternative_log_loss": float(np.average(ll_alt, weights=selected_weights)) if selected_weights.sum() else None,
        "log_loss_gain_base_minus_alternative": _monthly_interval(ll_base - ll_alt, frame.month, month_universe, selected_weights),
        "brier_gain_base_minus_alternative": _monthly_interval(brier_base - brier_alt, frame.month, month_universe, selected_weights),
        "mean_abs_probability_change": float(np.average(np.abs(alt - base), weights=selected_weights)) if selected_weights.sum() else None,
        "mean_probability_change_alternative_minus_base": float(np.average(alt - base, weights=selected_weights)) if selected_weights.sum() else None,
    }
    if blocks is not None and block_universe is not None:
        out["tournament_name_year_proxy_log_loss_gain"] = _proxy_interval(ll_base - ll_alt, blocks, block_universe, selected_weights)
    return out


def _origin_probability_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: frame[column].to_numpy(dtype=float) for name, column in {
        "prior_full": "p", "prior_no_w20": "p_no_w20", "prior_no_organization": "p_no_organization",
        "target_roster_full": "p_target", "target_roster_no_w20": "p_target_no_w20",
        "target_roster_no_organization": "p_target_no_organization",
    }.items()}


def _history_summary(raw: dict[str, np.ndarray], history: np.ndarray, mask: np.ndarray,
                     indices: np.ndarray, metadata: pd.DataFrame, origins: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    per_stat = {name: {"present": 0, "missing": 0, "valid_zero": 0, "missing_zero": 0} for name in STAT_NAMES}
    all_tokens = 0
    players_with_history = 0
    stat_cells = 0
    missing_cells = 0
    unseen_players = 0
    known_inactivity_days: list[float] = []
    known_player_slots = 0
    no_history_rows = 0
    gates = raw["gates"]
    for row_number, bank_index in enumerate(indices):
        token_mask = np.asarray(mask[bank_index], dtype=bool)
        token_values = np.asarray(history[bank_index], dtype=np.float32)
        if token_values.shape != (2, 5, 16, 33) or token_mask.shape != (2, 5, 16):
            raise ValueError("corrected-bank history/mask shape mismatch")
        observed = token_values[token_mask]
        row_stat_counts = {}
        if observed.size:
            stat_values = observed[:, HISTORY_STAT_VALUES]
            stat_flags = observed[:, HISTORY_STAT_PRESENT]
            if not np.isfinite(stat_values).all() or not np.isfinite(stat_flags).all() or not np.isin(stat_flags, (0.0, 1.0)).all():
                raise ValueError("observed numeric-stat payload or presence flags are invalid")
            if np.any(stat_values[stat_flags == 0] != 0.0):
                raise ValueError("missing numeric stats were not zero-filled as declared by the extractor")
            flags = stat_flags.astype(bool)
            for stat_index, stat_name in enumerate(STAT_NAMES):
                flag = flags[:, stat_index]
                value = stat_values[:, stat_index]
                record = per_stat[stat_name]
                present = int(flag.sum())
                missing = int((~flag).sum())
                valid_zero = int((flag & (value == 0.0)).sum())
                missing_zero = int(((~flag) & (value == 0.0)).sum())
                record["present"] += present
                record["missing"] += missing
                record["valid_zero"] += valid_zero
                record["missing_zero"] += missing_zero
                row_stat_counts.update({
                    f"stat_{stat_name}_present": present, f"stat_{stat_name}_missing": missing,
                    f"stat_{stat_name}_valid_zero": valid_zero, f"stat_{stat_name}_missing_zero": missing_zero,
                })
            all_tokens += int(token_mask.sum())
            players_with_history += int(token_mask.any(axis=-1).sum())
            stat_cells += int(flags.size)
            missing_cells += int((~flags).sum())
        else:
            no_history_rows += 1
            row_stat_counts = {f"stat_{name}_{kind}": 0 for name in STAT_NAMES for kind in ("present", "missing", "valid_zero", "missing_zero")}
        player_tensor = np.asarray(raw["players"][bank_index], dtype=np.float64)
        unseen = player_tensor[..., PLAYER_UNSEEN]
        inactivity_log = player_tensor[..., PLAYER_LOG_DAYS_INACTIVE]
        if not np.isin(unseen, (0.0, 1.0)).all() or not np.isfinite(inactivity_log).all():
            raise ValueError("raw player unseen/inactivity channels are invalid")
        unseen_players += int(unseen.sum())
        known = unseen == 0.0
        days = np.expm1(inactivity_log[known])
        if len(days):
            known_inactivity_days.extend(days.astype(float).tolist())
            known_player_slots += int(len(days))
        origin = origins.iloc[row_number]
        feature = metadata.iloc[row_number]
        if int(round(unseen.sum())) != int(feature.unknown_players):
            raise ValueError(f"raw player unseen count disagrees with canonical metadata for {origin.golgg_match_id}")
        side_gates = np.asarray(gates[bank_index], dtype=float)
        rows.append({
            "origin_id": str(origin.origin_id), "golgg_match_id": str(origin.golgg_match_id),
            "date": str(origin.date), "canonical_target_roster_scenario": True,
            "history_mask_tokens": int(token_mask.sum()), "players_with_history": int(token_mask.any(axis=-1).sum()),
            "players_without_history": int((~token_mask.any(axis=-1)).sum()),
            "visible_stat_cells": int(observed.shape[0] * 12),
            "missing_stat_cells": int((observed[:, HISTORY_STAT_PRESENT] == 0).sum()) if observed.size else 0,
            "valid_zero_stat_cells": int(((observed[:, HISTORY_STAT_PRESENT] == 1) & (observed[:, HISTORY_STAT_VALUES] == 0)).sum()) if observed.size else 0,
            "missing_zero_stat_cells": int(((observed[:, HISTORY_STAT_PRESENT] == 0) & (observed[:, HISTORY_STAT_VALUES] == 0)).sum()) if observed.size else 0,
            "stat_missing_fraction": float((observed[:, HISTORY_STAT_PRESENT] == 0).mean()) if observed.size else None,
            "unknown_players_tensor": int(unseen.sum()),
            "known_player_inactivity_n": int(len(days)),
            "known_player_inactivity_days_sum": float(days.sum()) if len(days) else 0.0,
            "known_player_inactivity_days_mean": float(np.mean(days)) if len(days) else None,
            "known_player_inactivity_days_max": float(np.max(days)) if len(days) else None,
            "organization_gate_mean": float(side_gates[:, 0].mean()),
            "w20_gate_mean": float(side_gates[:, 1].mean()),
            "organization_gate_min": float(side_gates[:, 0].min()),
            "w20_gate_min": float(side_gates[:, 1].min()),
            **row_stat_counts,
        })
    stats = {}
    for name, values in per_stat.items():
        denominator = values["present"] + values["missing"]
        stats[name] = {
            **values, "cells": denominator,
            "present_fraction": values["present"] / denominator if denominator else None,
            "missing_fraction": values["missing"] / denominator if denominator else None,
        }
    inactivity = np.asarray(known_inactivity_days, dtype=float)
    summary = {
        "scenario": "canonical actual/first-map target rosters, corrected feature bank; not repaired prior inputs except exact same-player/same-cutoff rows",
        "n_origins": len(origins), "n_origins_with_no_visible_history_tokens": no_history_rows,
        "visible_player_history_tokens": all_tokens, "player_slots_with_any_history": players_with_history,
        "visible_numeric_stat_cells": stat_cells, "missing_numeric_stat_cells": missing_cells,
        "missing_numeric_stat_fraction": missing_cells / stat_cells if stat_cells else None,
        "stat_presence_by_channel": stats,
        "valid_zero_vs_missing_zero_are_distinct": True,
        "missing_numeric_values_zero_filled_by_source": True,
        "players_unseen_in_prior_bank_history": unseen_players,
        "known_player_slots_for_inactivity": known_player_slots,
        "known_player_inactivity_days": {
            "mean": float(np.mean(inactivity)) if len(inactivity) else None,
            "median": float(np.median(inactivity)) if len(inactivity) else None,
            "p90": float(np.quantile(inactivity, .9)) if len(inactivity) else None,
            "max": float(np.max(inactivity)) if len(inactivity) else None,
        },
        "inactivity_definition": "exp(log_days_inactive)-1 from the feature extractor, reported only for seen players; unseen players are separately counted and their sentinel is not called inactivity",
        "roster_snapshot_age_days_is_not_player_inactivity": True,
    }
    return pd.DataFrame(rows), summary


def _raw_roster(game: dict[str, Any], team_key: str) -> tuple[list[str], dict[str, Any]]:
    players = game.get(f"{team_key}_players") or {}
    ordered = []
    by_id = {}
    if isinstance(players, dict):
        for role, player in players.items():
            if not isinstance(player, dict):
                continue
            player_id = player.get("player_id", player.get("id"))
            if player_id is None:
                continue
            player_id = str(player_id)
            ordered.append(player_id)
            by_id[player_id] = {"role": str(role), "stats": player.get("stats") or {}}
    return ordered, by_id


def _raw_match_records(path: Path, needed_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        import ijson
    except ImportError:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        iterable = payload if isinstance(payload, list) else payload.get("matches", [])
        for item in iterable:
            match_id = str(item.get("match_id", ""))
            if match_id in needed_ids:
                records[match_id] = item
    else:
        with path.open("rb") as stream:
            for item in ijson.items(stream, "item"):
                match_id = str(item.get("match_id", ""))
                if match_id in needed_ids:
                    records[match_id] = item
    return records


def _extract_raw_case_evidence(cases: pd.DataFrame, origins: pd.DataFrame, raw_history_path: Path) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    origin_by_id = origins.set_index("origin_id", drop=False)
    needed = set(cases.golgg_match_id.astype(str))
    source_refs = {}
    for origin_id in cases.origin_id.astype(str):
        origin = origin_by_id.loc[origin_id]
        for side in ("a", "b"):
            ref = _decode(origin.get(f"roster_source_{side}"), {}) or {}
            if ref.get("match_id") is not None:
                source_id = str(ref["match_id"])
                needed.add(source_id)
                source_refs[(origin_id, side)] = ref
    raw_matches = _raw_match_records(raw_history_path, needed)
    evidence = []
    for case in cases.to_dict("records"):
        origin = origin_by_id.loc[str(case["origin_id"])]
        match = raw_matches.get(str(case["golgg_match_id"]))
        if match is None:
            raise ValueError(f"raw history missing target match {case['golgg_match_id']}")
        games = match.get("games") or []
        if not games:
            raise ValueError(f"raw history target match has no maps: {case['golgg_match_id']}")
        first = games[0]
        t1_id, t2_id = str(first.get("t1_id")), str(first.get("t2_id"))
        t1_roster, t1_players = _raw_roster(first, "t1")
        t2_roster, t2_players = _raw_roster(first, "t2")
        if t1_id == str(origin.team1_id):
            actual_a, actual_b = t1_roster, t2_roster
            players_a, players_b = t1_players, t2_players
        elif t2_id == str(origin.team1_id):
            actual_a, actual_b = t2_roster, t1_roster
            players_a, players_b = t2_players, t1_players
        else:
            actual_a, actual_b = [], []
            players_a, players_b = {}, {}
        target_a, target_b = _roster(origin.canonical_roster_a), _roster(origin.canonical_roster_b)
        current_stat_present = sum(
            1 for player in (*players_a.values(), *players_b.values())
            for stat in STAT_NAMES
            if isinstance(player["stats"].get(stat), (int, float)) and not isinstance(player["stats"].get(stat), bool)
            and math.isfinite(float(player["stats"][stat]))
        )
        row = {
            "case_role": case["case_role"], "case_origin_id": str(case["origin_id"]),
            "golgg_match_id": str(case["golgg_match_id"]), "event_date": str(origin.date),
            "tournament_name": str(origin.tournament_name), "competition_family": str(origin.competition_family),
            "best_of": int(origin.best_of), "team1_id": str(origin.team1_id), "team2_id": str(origin.team2_id),
            "raw_target_first_map_index": 0, "raw_target_first_map_roster_a": actual_a,
            "raw_target_first_map_roster_b": actual_b,
            "raw_first_map_roster_matches_canonical_a": set(actual_a) == set(target_a),
            "raw_first_map_roster_matches_canonical_b": set(actual_b) == set(target_b),
            "prior_scenario_roster_a": _roster(origin.roster_a), "prior_scenario_roster_b": _roster(origin.roster_b),
            "canonical_target_roster_a": target_a, "canonical_target_roster_b": target_b,
            "actual_additions_a_vs_prior": sorted(set(target_a) - set(_roster(origin.roster_a))),
            "actual_additions_b_vs_prior": sorted(set(target_b) - set(_roster(origin.roster_b))),
            "first_map_player_stats_present_cells_post_event": current_stat_present,
            "first_map_player_stats_total_cells_post_event": (len(players_a) + len(players_b)) * len(STAT_NAMES),
            "first_map_player_stats_are_prediction_inputs": False,
            "raw_match_map_count": len(games),
        }
        for side in ("a", "b"):
            ref = source_refs.get((str(case["origin_id"]), side), {})
            source_id = str(ref.get("match_id")) if ref.get("match_id") is not None else None
            source_match = raw_matches.get(source_id) if source_id else None
            map_index = int(ref.get("map_index", 0)) if ref else 0
            source_games = source_match.get("games") or [] if source_match else []
            source_game = source_games[map_index - 1] if source_match and 0 < map_index <= len(source_games) else None
            source_team_id = str(ref.get("team_id")) if ref.get("team_id") is not None else None
            if source_game is not None:
                if str(source_game.get("t1_id")) == source_team_id:
                    source_roster, _ = _raw_roster(source_game, "t1")
                elif str(source_game.get("t2_id")) == source_team_id:
                    source_roster, _ = _raw_roster(source_game, "t2")
                else:
                    source_roster = []
            else:
                source_roster = []
            row[f"prior_roster_source_match_id_{side}"] = source_id
            row[f"prior_roster_source_release_day_{side}"] = ref.get("effective_release_day")
            row[f"prior_roster_source_raw_day_{side}"] = (source_game.get("date", source_game.get("day")) if source_game else None)
            row[f"prior_roster_source_roster_{side}"] = _roster(ref.get("roster"))
            row[f"prior_roster_source_raw_roster_{side}"] = source_roster
            row[f"prior_roster_source_raw_roster_verified_{side}"] = bool(source_game is not None and set(source_roster) == set(_roster(ref.get("roster"))))
        evidence.append(row)
    return pd.DataFrame(evidence)


def _support_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    def minimum(a: str, b: str) -> pd.Series:
        x, y = pd.to_numeric(out[a], errors="coerce"), pd.to_numeric(out[b], errors="coerce")
        return pd.concat([x, y], axis=1).min(axis=1, skipna=False)
    def maximum(a: str, b: str) -> pd.Series:
        x, y = pd.to_numeric(out[a], errors="coerce"), pd.to_numeric(out[b], errors="coerce")
        return pd.concat([x, y], axis=1).max(axis=1, skipna=False)
    out["replacement_total"] = pd.to_numeric(out.hindsight_roster_replacements_a, errors="coerce") + pd.to_numeric(out.hindsight_roster_replacements_b, errors="coerce")
    out["unknown_player_total"] = pd.to_numeric(out.unknown_players_a, errors="coerce") + pd.to_numeric(out.unknown_players_b, errors="coerce")
    out["min_player_series_support"] = minimum("min_player_series_a", "min_player_series_b")
    out["min_coplay_support"] = minimum("co_play_mean_a", "co_play_mean_b")
    out["min_w20_valid_games"] = minimum("w20_valid_games_a", "w20_valid_games_b")
    out["min_w20_window_games"] = minimum("w20_window_games_a", "w20_window_games_b")
    out["w20_valid_fraction_min_side"] = np.divide(
        out.min_w20_valid_games.to_numpy(dtype=float), out.min_w20_window_games.to_numpy(dtype=float),
        out=np.full(len(out), np.nan), where=out.min_w20_window_games.to_numpy(dtype=float) > 0,
    )
    out["max_roster_snapshot_age_days"] = maximum("roster_snapshot_age_days_a", "roster_snapshot_age_days_b")
    overlaps = out.overlapping_prior_player_ids.map(lambda value: bool(_roster(value)))
    target_roster_overlap = []
    for a, b in zip(out.canonical_roster_a, out.canonical_roster_b):
        target_roster_overlap.append(bool(set(_roster(a)) & set(_roster(b))))
    out["prior_lineup_nonoverlap"] = ~overlaps
    out["canonical_lineup_nonoverlap"] = ~np.asarray(target_roster_overlap, dtype=bool)
    out["known_prior_lineup_nonoverlap"] = out.unknown_player_total.eq(0) & out.prior_lineup_nonoverlap & out.canonical_lineup_nonoverlap
    out["known_prior_nonoverlap_same_cutoff"] = out.known_prior_lineup_nonoverlap & out.canonical_same_cutoff_day.astype(bool)
    out["replacement_group"] = pd.cut(out.replacement_total, [-0.1, 0.0, 2.0, 5.0, 10.0, np.inf], labels=["0", "1-2", "3-5", "6-10", "unknown"], include_lowest=True).astype(str)
    out.loc[out.replacement_total.isna(), "replacement_group"] = "unknown"
    out["snapshot_age_group"] = pd.cut(out.max_roster_snapshot_age_days, [-0.1, 7, 30, 90, np.inf], labels=["0-7d", "8-30d", "31-90d", ">90d"], include_lowest=True).astype(str)
    out.loc[out.max_roster_snapshot_age_days.isna(), "snapshot_age_group"] = "unknown"
    out["player_series_group"] = pd.cut(out.min_player_series_support, [-0.1, 0, 2, 5, 10, np.inf], labels=["0", "1-2", "3-5", "6-10", ">10"], include_lowest=True).astype(str)
    out.loc[out.min_player_series_support.isna(), "player_series_group"] = "unknown"
    out["coplay_group"] = pd.cut(out.min_coplay_support, [-0.1, 0, 1, 5, 10, np.inf], labels=["0", ">(0,1]", "(1,5]", "(5,10]", ">10"], include_lowest=True).astype(str)
    out.loc[out.min_coplay_support.isna(), "coplay_group"] = "unknown"
    out["w20_support_group"] = pd.cut(out.min_w20_valid_games, [-0.1, 0, 1, 4, 9, 20, np.inf], labels=["0", "1", "2-4", "5-9", "10-20", ">20"], include_lowest=True).astype(str)
    out.loc[out.min_w20_valid_games.isna(), "w20_support_group"] = "unknown"
    return out


def _probability_side(frame: pd.DataFrame, col_a: str, side_col: str = "side") -> np.ndarray:
    p = pd.to_numeric(frame[col_a], errors="coerce").to_numpy(dtype=float)
    return np.where(frame[side_col].astype(str).to_numpy() == "A", p, 1.0 - p)


def _make_case_sample(bets: pd.DataFrame, primary_origins: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    frame = bets.copy()
    y = frame.won.to_numpy(dtype=float)
    prior = frame.p_model_bet.to_numpy(dtype=float)
    target = _probability_side(frame, "p_target")
    frame["case_prior_log_loss"] = binary_log_loss_vector(y, prior, epsilon=EPSILON)
    frame["case_market_log_loss"] = binary_log_loss_vector(y, frame.q_bet.to_numpy(dtype=float), epsilon=EPSILON)
    frame["case_oracle_gain"] = frame.case_prior_log_loss - binary_log_loss_vector(y, target, epsilon=EPSILON)
    frame["case_a0_minus_market_log_loss"] = frame.case_prior_log_loss - frame.case_market_log_loss
    frame["case_roster_changed"] = frame.hindsight_any_roster_change.astype(bool)
    losses = frame.loc[~frame.won.astype(bool)].sort_values(["case_oracle_gain", "case_prior_log_loss"], ascending=False).head(4).copy()
    if len(losses) < 4:
        extra = frame.loc[~frame.won.astype(bool) & ~frame.index.isin(losses.index)].sort_values("case_prior_log_loss", ascending=False).head(4-len(losses))
        losses = pd.concat([losses, extra], axis=0)
    losses["case_role"] = "high_contribution_selected_loss"
    wins = frame.loc[frame.won.astype(bool)].copy()
    chosen = [*losses.to_dict("records")]
    used_wins: set[Any] = set()
    for loss in losses.to_dict("records"):
        controls = wins.loc[
            wins.competition_family.eq(loss["competition_family"])
            & wins.best_of.eq(loss["best_of"])
            & wins.hindsight_any_roster_change.eq(loss["hindsight_any_roster_change"])
        ].copy()
        odds_band = _odds_band(float(loss["odds"]))
        controls = controls.loc[controls.odds.map(lambda value: _odds_band(float(value)) == odds_band)]
        if controls.empty:
            controls = wins.loc[wins.competition_family.eq(loss["competition_family"]) & wins.best_of.eq(loss["best_of"])].copy()
        if controls.empty:
            controls = wins.loc[wins.best_of.eq(loss["best_of"])].copy()
        if controls.empty:
            continue
        controls["_distance"] = np.abs(controls.p_model_bet.astype(float) - float(loss["p_model_bet"]))
        controls["_already_used"] = controls.bet_id.isin(used_wins)
        controls = controls.sort_values(["_already_used", "_distance"], kind="stable")
        control = controls.iloc[0].to_dict()
        used_wins.add(control["bet_id"])
        control["case_role"] = "matched_selected_win_control"
        control["matched_loss_bet_id"] = loss["bet_id"]
        chosen.append(control)
    result = pd.DataFrame(chosen)
    if "matched_loss_bet_id" not in result:
        result["matched_loss_bet_id"] = None
    return result


def _odds_band(value: float) -> str:
    if value < 1.8:
        return "<1.8"
    if value <= 2.5:
        return "1.8-2.5"
    if value <= 3.5:
        return "2.5-3.5"
    return ">3.5"


def run(tables: dict[str, pd.DataFrame], protocol: dict, output: Path) -> dict[str, Any]:
    origins_all = tables["origins"].copy()
    opportunities = tables["opportunities"].copy()
    bets_all = tables["bets"].copy()
    source_paths = protocol["paths"]
    primary_ops = opportunities.loc[
        opportunities.cohort.eq("oddspapi_common") & opportunities.source.eq("best_price")
        & opportunities.horizon.eq("T-1h") & opportunities.audit_primary_eligible.astype(bool)
    ].copy()
    primary_origins = origins_all.loc[
        origins_all.cohort.eq("oddspapi_common") & origins_all.horizon.eq("T-1h")
        & origins_all.status.eq("available")
    ].merge(
        primary_ops[["origin_id", "fixture_id", "horizon", "p_event", "y_event", "month"]],
        on=["origin_id", "fixture_id", "horizon"], how="inner", validate="one_to_one", suffixes=("", "_opportunity"),
    )
    if len(primary_ops) != 1852 or len(primary_origins) != 1852:
        raise ValueError(f"unexpected primary universe: opportunities={len(primary_ops)}, origins={len(primary_origins)}")
    if not np.allclose(primary_origins.p.to_numpy(dtype=float), primary_origins.p_event.to_numpy(dtype=float), atol=1e-12, rtol=0):
        raise ValueError("repaired opportunity and primary origin probabilities differ")
    if not primary_origins.y.astype(int).eq(primary_origins.y_event.astype(int)).all():
        raise ValueError("repaired opportunity and primary origin outcomes differ")
    month_universe = sorted(primary_ops.month.astype(str).unique().tolist())
    proxy_universe = sorted(primary_origins.edition_proxy.fillna("unknown").astype(str).unique().tolist())
    primary_origins = _support_columns(primary_origins.reset_index(drop=True))
    probs = _origin_probability_columns(primary_origins)
    y = primary_origins.y.to_numpy(dtype=float)
    if not np.isfinite(np.column_stack(list(probs.values()))).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("primary forecast/origin probabilities are not finite binary outcomes")
    origin_mode_effects = []
    for name, alternative in (("prior_full_vs_no_w20", "prior_no_w20"),
                              ("prior_full_vs_no_organization", "prior_no_organization"),
                              ("prior_full_vs_target_roster_oracle", "target_roster_full"),
                              ("target_roster_full_vs_no_w20", "target_roster_no_w20"),
                              ("target_roster_full_vs_no_organization", "target_roster_no_organization")):
        origin_mode_effects.append(_metric_effect(primary_origins, y, probs["prior_full"] if name.startswith("prior_full") else probs["target_roster_full"],
                                                  probs[alternative], name, month_universe,
                                                  block_universe=proxy_universe, blocks=primary_origins.edition_proxy))
    origin_scores = {name: evaluate_binary_probabilities(y, value, epsilon=EPSILON) for name, value in probs.items()}
    origin_mode_effects.extend([
        {"contrast": f"prior_full_vs_{alternative}_within_{truth}",
         **_metric_effect(primary_origins, y, probs["prior_full"], probs[alternative], f"prior_full_vs_{alternative}_within_{truth}",
                          month_universe, weights=primary_origins.y.eq(truth).to_numpy(dtype=float),
                          block_universe=proxy_universe, blocks=primary_origins.edition_proxy)}
        for truth in (0, 1) for alternative in ("target_roster_full", "prior_no_w20", "prior_no_organization")
    ])
    unchanged_exact = primary_origins.hindsight_same_players.astype(bool) & primary_origins.canonical_same_cutoff_day.astype(bool)
    equality = {}
    for name, column in (("prior_full", "p"), ("prior_no_w20", "p_no_w20"), ("prior_no_organization", "p_no_organization")):
        left = primary_origins.loc[unchanged_exact, column].to_numpy(dtype=float)
        right = primary_origins.loc[unchanged_exact, {"prior_full":"p_target", "prior_no_w20":"p_target_no_w20", "prior_no_organization":"p_target_no_organization"}[name]].to_numpy(dtype=float)
        delta = np.abs(left - right)
        equality[name] = {"n": int(unchanged_exact.sum()), "all_equal_atol_1e-12_rtol_0": bool(np.allclose(left, right, atol=1e-12, rtol=0)),
                         "max_abs_probability_difference": float(delta.max()) if len(delta) else None,
                         "mean_abs_probability_difference": float(delta.mean()) if len(delta) else None}
    if not all(value["all_equal_atol_1e-12_rtol_0"] for value in equality.values()):
        raise ValueError(f"unchanged-player/same-cutoff paired probabilities differ: {equality}")

    primary_bets = bets_all.loc[
        bets_all.cohort.eq("oddspapi_common") & bets_all.source.eq("best_price") & bets_all.horizon.eq("T-1h")
        & bets_all.quality.eq("exclude_q25_conflicts") & bets_all.policy.eq("raw_ev_gt_5")
    ].copy()
    if len(primary_bets) != 498:
        raise ValueError(f"unexpected fixed saved selection count: {len(primary_bets)}")
    if primary_bets.bet_id.duplicated().any() or primary_bets.origin_id.isna().any():
        raise ValueError("primary fixed saved bets have duplicate/missing origin identities")
    origin_controls = primary_origins[["origin_id", "p_target_no_w20", "p_target_no_organization", "month", "edition_proxy"]].copy()
    primary_bets = primary_bets.merge(origin_controls, on="origin_id", validate="many_to_one", suffixes=("", "_origin"))
    if not np.allclose(primary_bets.p_model_bet.to_numpy(dtype=float), _probability_side(primary_bets, "p_event"), atol=1e-12, rtol=0):
        raise ValueError("fixed saved-side probability does not agree with repaired origin")
    bet_probabilities = {
        "prior_full": primary_bets.p_model_bet.to_numpy(dtype=float),
        "prior_no_w20": _probability_side(primary_bets, "p_no_w20"),
        "prior_no_organization": _probability_side(primary_bets, "p_no_organization"),
        "target_roster_full": _probability_side(primary_bets, "p_target"),
        "target_roster_no_w20": _probability_side(primary_bets, "p_target_no_w20"),
        "target_roster_no_organization": _probability_side(primary_bets, "p_target_no_organization"),
    }
    bet_y = primary_bets.won.to_numpy(dtype=float)
    bet_odds = pd.to_numeric(primary_bets.odds, errors="raise").to_numpy(dtype=float)
    bet_modes = []
    for name, alternative in (("prior_full_vs_no_w20", "prior_no_w20"),
                              ("prior_full_vs_no_organization", "prior_no_organization"),
                              ("prior_full_vs_target_roster_oracle", "target_roster_full"),
                              ("target_roster_full_vs_no_w20", "target_roster_no_w20"),
                              ("target_roster_full_vs_no_organization", "target_roster_no_organization")):
        base = bet_probabilities["prior_full"] if name.startswith("prior_full") else bet_probabilities["target_roster_full"]
        alt = bet_probabilities[alternative]
        row = _metric_effect(primary_bets, bet_y, base, alt, name, month_universe,
                             block_universe=proxy_universe, blocks=primary_bets.edition_proxy)
        expected_delta = (alt - base) * bet_odds * (1.0 - TAX_RATE)
        row["fixed_selection_expected_net_return_delta_alt_minus_base"] = _monthly_interval(expected_delta, primary_bets.month, month_universe)
        bet_modes.append(row)
    fixed_bet_scores = {name: evaluate_binary_probabilities(bet_y, value, epsilon=EPSILON) for name, value in bet_probabilities.items()}
    bet_modes.extend([
        {"contrast": f"prior_full_vs_{alternative}_within_{truth}",
         **_metric_effect(primary_bets, bet_y, bet_probabilities["prior_full"], bet_probabilities[alternative],
                          f"prior_full_vs_{alternative}_within_{truth}", month_universe,
                          weights=primary_bets.won.eq(truth).to_numpy(dtype=float),
                          block_universe=proxy_universe, blocks=primary_bets.edition_proxy)}
        for truth in (False, True) for alternative in ("target_roster_full", "prior_no_w20", "prior_no_organization")
    ])

    research_root = Path(protocol["resolved_research_root"])
    from src.analysis.ev_asof_replay import FEATURE_SOURCE_SHA256
    extractor_root = research_root / "nonlinear-history-20260912" / "code" / "scripts"
    extractor_hashes = {}
    for filename in ("roster_optional_features.py", "temporal_player_features.py"):
        extractor_path = extractor_root / filename
        actual_hash = _sha256(extractor_path)
        expected_hash = FEATURE_SOURCE_SHA256[f"code/scripts/{filename}"]
        if actual_hash != expected_hash:
            raise ValueError(f"archived feature extractor hash mismatch: {filename}")
        extractor_hashes[filename] = actual_hash
    bank = source_paths["canonical_metadata"].parent
    receipt_path = bank / "completed.json"
    bank_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if bank_receipt.get("status") != "COMPLETE" or int(bank_receipt.get("rows", -1)) != 40636:
        raise ValueError("corrected canonical feature bank is not the expected complete immutable bank")
    metadata_path = source_paths["canonical_metadata"]
    metadata = pd.read_parquet(metadata_path)
    metadata["golgg_match_id"] = metadata.golgg_match_id.astype(str)
    if metadata.golgg_match_id.duplicated().any():
        raise ValueError("corrected bank metadata match identity is not unique")
    metadata_index = pd.Series(np.arange(len(metadata), dtype=np.int64), index=metadata.golgg_match_id)
    missing_ids = sorted(set(primary_origins.golgg_match_id.astype(str)) - set(metadata_index.index.astype(str)))
    if missing_ids:
        raise ValueError(f"canonical feature bank misses primary target IDs: {missing_ids[:5]}")
    bank_indices = primary_origins.golgg_match_id.astype(str).map(metadata_index).to_numpy(dtype=np.int64)
    aligned_meta = metadata.iloc[bank_indices].reset_index(drop=True)
    for key, bank_key in (("canonical_team1_id", "team1_id"), ("canonical_team2_id", "team2_id"), ("date", "date"), ("best_of", "best_of")):
        if not primary_origins[key].astype(str).reset_index(drop=True).equals(aligned_meta[bank_key].astype(str).reset_index(drop=True)):
            raise ValueError(f"feature-bank and primary event identity mismatch for {key}")
    actual_same_sets = []
    target_lineup_equal = []
    for _, origin in primary_origins.iterrows():
        prior_a, prior_b = _roster(origin.roster_a), _roster(origin.roster_b)
        bank_a, bank_b = _roster(aligned_meta.iloc[len(actual_same_sets)].roster_a), _roster(aligned_meta.iloc[len(actual_same_sets)].roster_b)
        actual_same_sets.append(len(prior_a) == len(prior_b) == 5 and set(prior_a) == set(bank_a) and set(prior_b) == set(bank_b))
        target_a, target_b = _roster(origin.canonical_roster_a), _roster(origin.canonical_roster_b)
        target_lineup_equal.append(set(target_a) == set(bank_a) and set(target_b) == set(bank_b))
    if not all(target_lineup_equal):
        raise ValueError("prepared canonical target rosters do not match corrected feature-bank lineups")
    if not np.array_equal(np.asarray(actual_same_sets), primary_origins.hindsight_same_players.to_numpy(dtype=bool)):
        raise ValueError("hindsight same-player flag disagrees with direct roster identity comparison")
    same_day = primary_origins.canonical_same_cutoff_day.astype(bool).to_numpy()
    exact_transfer = np.asarray(actual_same_sets, dtype=bool) & same_day

    outputs = bank_receipt.get("outputs_sha256", {})
    required_bank_files = ("arrays.npz", "history.npy", "mask.npy")
    bank_hashes = {}
    for filename in required_bank_files:
        expected = outputs.get(filename)
        path = bank / filename
        if not expected or _sha256(path) != expected:
            raise ValueError(f"corrected feature bank artifact hash mismatch: {filename}")
        bank_hashes[filename] = str(expected)
    with np.load(bank / "arrays.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    history = np.load(bank / "history.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(bank / "mask.npy", mmap_mode="r", allow_pickle=False)
    expected_raw_shapes = {"players": (len(metadata), 2, 5, 21), "core": (len(metadata), 2, 5), "team": (len(metadata), 2, 11),
                           "w20": (len(metadata), 2, 10), "gates": (len(metadata), 2, 2)}
    for key, shape in expected_raw_shapes.items():
        if key not in raw or raw[key].shape != shape:
            raise ValueError(f"corrected bank {key} tensor shape mismatch")
    if history.shape != (len(metadata), 2, 5, 16, 33) or mask.shape != (len(metadata), 2, 5, 16):
        raise ValueError("canonical temporal feature bank has unexpected history/mask dimensions")
    evidence_frame, bank_evidence = _history_summary(raw, history, mask, bank_indices, aligned_meta, primary_origins)
    evidence_frame["same_players_as_prior"] = actual_same_sets
    evidence_frame["same_cutoff_day_as_prior"] = same_day
    evidence_frame["transferable_to_repaired_prior_input"] = exact_transfer
    bank_evidence["transferable_origins_same_players_and_cutoff"] = int(exact_transfer.sum())
    bank_evidence["nontransferable_actual_roster_scenario_origins"] = int((~exact_transfer).sum())
    transferable = evidence_frame.loc[exact_transfer]
    transferable_stats = {}
    for stat_name in STAT_NAMES:
        present = int(transferable[f"stat_{stat_name}_present"].sum())
        missing = int(transferable[f"stat_{stat_name}_missing"].sum())
        transferable_stats[stat_name] = {
            "present": present, "missing": missing, "cells": present + missing,
            "valid_zero": int(transferable[f"stat_{stat_name}_valid_zero"].sum()),
            "missing_zero": int(transferable[f"stat_{stat_name}_missing_zero"].sum()),
            "missing_fraction": missing / (present + missing) if present + missing else None,
        }
    transferable_visible_cells = int(transferable.visible_stat_cells.sum())
    transferable_missing_cells = int(transferable.missing_stat_cells.sum())
    transferable_inactivity_n = int(transferable.known_player_inactivity_n.sum())
    bank_evidence["transferable_prior_input_evidence"] = {
        "exact_same_player_sets_and_cutoff_origins": int(exact_transfer.sum()),
        "visible_history_tokens": int(transferable.history_mask_tokens.sum()),
        "visible_stat_cells": transferable_visible_cells,
        "missing_stat_cells": transferable_missing_cells,
        "missing_stat_fraction": transferable_missing_cells / transferable_visible_cells if transferable_visible_cells else None,
        "valid_zero_stat_cells": int(transferable.valid_zero_stat_cells.sum()),
        "missing_zero_stat_cells": int(transferable.missing_zero_stat_cells.sum()),
        "stat_presence_by_channel": transferable_stats,
        "known_player_inactivity_count": transferable_inactivity_n,
        "known_player_inactivity_days_mean": (
            float(transferable.known_player_inactivity_days_sum.sum() / transferable_inactivity_n)
            if transferable_inactivity_n else None
        ),
        "known_player_inactivity_days_max": (
            float(transferable.known_player_inactivity_days_max.max())
            if transferable.known_player_inactivity_days_max.notna().any() else None
        ),
        "interpretation": "Only these corrected actual-roster bank tensors are identity/cutoff aligned to the repaired prior-lineup inputs.",
    }
    bank_evidence["raw_tensor_semantics"] = {
        "players_width_21": "11 ratings + log_games + log_series + log_days_inactive + unseen + six role flags, per archived roster_optional_features.py",
        "history_width_33": "12 game-stat values + 12 explicit presence indicators + win + log days since map + opponent strength + six role flags, per archived temporal_player_features.py",
        "mask": "valid player-map history token; false padding is not a measured zero",
        "w20": "ten retrospective rolling summary values, active only under the saved organization/W20 gates",
        "gates": ["organization_continuity_support", "w20_valid_fraction_times_organization_support"],
    }
    # For exact same-player/cutoff rows, canonical-side raw summaries are the only safely transferable tensor evidence.
    evidence_frame.loc[~evidence_frame.transferable_to_repaired_prior_input, "transferable_to_repaired_prior_input"] = False
    primary_origins = primary_origins.merge(evidence_frame, on=["origin_id", "golgg_match_id", "date"], validate="one_to_one")
    same_known = primary_origins.known_prior_nonoverlap_same_cutoff.to_numpy(dtype=bool)
    clean_controls = primary_origins.known_prior_lineup_nonoverlap.to_numpy(dtype=bool)
    oracle_clean_effect = _metric_effect(primary_origins.loc[clean_controls].reset_index(drop=True),
                                         primary_origins.loc[clean_controls, "y"].to_numpy(dtype=float),
                                         primary_origins.loc[clean_controls, "p"].to_numpy(dtype=float),
                                         primary_origins.loc[clean_controls, "p_target"].to_numpy(dtype=float),
                                         "prior_full_vs_target_roster_known_nonoverlap", month_universe,
                                         block_universe=proxy_universe,
                                         blocks=primary_origins.loc[clean_controls, "edition_proxy"].reset_index(drop=True))
    oracle_clean_effect["includes_same_cutoff_requirement"] = False
    oracle_clean_same_day_effect = _metric_effect(primary_origins.loc[same_known].reset_index(drop=True),
                                                   primary_origins.loc[same_known, "y"].to_numpy(dtype=float),
                                                   primary_origins.loc[same_known, "p"].to_numpy(dtype=float),
                                                   primary_origins.loc[same_known, "p_target"].to_numpy(dtype=float),
                                                   "prior_full_vs_target_roster_known_nonoverlap_same_cutoff", month_universe,
                                                   block_universe=proxy_universe,
                                                   blocks=primary_origins.loc[same_known, "edition_proxy"].reset_index(drop=True))
    oracle_clean_same_day_effect["includes_same_cutoff_requirement"] = True

    profile_spec = [
        ("roster_change_type", primary_origins.roster_change_type.fillna("unknown").astype(str)),
        ("replacement_group", primary_origins.replacement_group.astype(str)),
        ("snapshot_age_group", primary_origins.snapshot_age_group.astype(str)),
        ("player_series_group", primary_origins.player_series_group.astype(str)),
        ("coplay_group", primary_origins.coplay_group.astype(str)),
        ("w20_support_group", primary_origins.w20_support_group.astype(str)),
        ("known_prior_nonoverlap", pd.Series(np.where(primary_origins.known_prior_lineup_nonoverlap, "known_nonoverlap", "unknown_or_overlap"))),
        ("league_best_of", primary_origins.competition_family.fillna("unknown").astype(str) + ":BO" + primary_origins.best_of.astype(str)),
    ]
    subgroup_effects = []
    for dimension, labels in profile_spec:
        for level in sorted(pd.Series(labels).unique()):
            group_mask = pd.Series(labels).eq(level).to_numpy(dtype=bool)
            if int(group_mask.sum()) == 0:
                continue
            if dimension == "league_best_of" and int(group_mask.sum()) < 15:
                continue
            for truth, truth_label in ((None, "all_outcomes"), (0, "side_A_lost"), (1, "side_A_won")):
                weights = group_mask.astype(float)
                if truth is not None:
                    weights *= primary_origins.y.eq(truth).to_numpy(dtype=float)
                n = int(weights.sum())
                if n == 0:
                    continue
                for alternative in ("target_roster_full", "prior_no_w20", "prior_no_organization"):
                    row = _metric_effect(primary_origins, y, probs["prior_full"], probs[alternative],
                                         f"prior_full_vs_{alternative}", month_universe, weights=weights,
                                         block_universe=proxy_universe, blocks=primary_origins.edition_proxy)
                    chosen = group_mask & (np.ones(len(group_mask), dtype=bool) if truth is None else primary_origins.y.eq(truth).to_numpy(dtype=bool))
                    row.update({"dimension": dimension, "level": str(level), "outcome_control": truth_label,
                                "eligible_n": int(group_mask.sum()), "n": n,
                                "mean_unknown_players": float(primary_origins.loc[chosen, "unknown_player_total"].mean()),
                                "mean_min_player_series": float(primary_origins.loc[chosen, "min_player_series_support"].mean()),
                                "mean_min_coplay": float(primary_origins.loc[chosen, "min_coplay_support"].mean()),
                                "mean_min_w20_valid_games": float(primary_origins.loc[chosen, "min_w20_valid_games"].mean()),
                                "mean_roster_snapshot_age_days": float(primary_origins.loc[chosen, "max_roster_snapshot_age_days"].mean()),
                                "mean_actual_target_roster_stat_missing_fraction": float(primary_origins.loc[chosen, "stat_missing_fraction"].mean()) if primary_origins.loc[chosen, "stat_missing_fraction"].notna().any() else None})
                    subgroup_effects.append(row)

    bet_origin_fields = ["origin_id", "roster_change_type", "edition_proxy", "hindsight_same_players", "hindsight_any_roster_change",
                         "canonical_same_cutoff_day", "unknown_players_a", "unknown_players_b", "min_player_series_support",
                         "min_coplay_support", "min_w20_valid_games", "max_roster_snapshot_age_days", "replacement_total",
                         "known_prior_lineup_nonoverlap", "stat_missing_fraction", "history_mask_tokens"]
    primary_bets = primary_bets.merge(primary_origins[bet_origin_fields].drop_duplicates("origin_id"), on="origin_id", validate="one_to_one", suffixes=("", "_profile"))
    fixed_bet_profiles = []
    bet_profiles = [
        ("roster_change_type", primary_bets.roster_change_type.fillna("unknown").astype(str)),
        ("changed_vs_same_player_set", np.where(primary_bets.hindsight_any_roster_change.astype(bool), "changed", "same_or_not_changed")),
        ("min_player_series_support", pd.cut(primary_bets.min_player_series_support, [-0.1, 0, 2, 5, 10, np.inf], labels=["0", "1-2", "3-5", "6-10", ">10"], include_lowest=True).astype(str)),
        ("min_w20_valid_games", pd.cut(primary_bets.min_w20_valid_games, [-0.1, 0, 1, 4, 9, 20, np.inf], labels=["0", "1", "2-4", "5-9", "10-20", ">20"], include_lowest=True).astype(str)),
    ]
    for dimension, labels in bet_profiles:
        labels = pd.Series(labels).fillna("unknown").astype(str)
        for level in sorted(labels.unique()):
            group_mask = labels.eq(level).to_numpy(dtype=bool)
            for won, outcome in ((None, "all"), (False, "lost"), (True, "won")):
                weights = group_mask.astype(float)
                if won is not None:
                    weights *= primary_bets.won.eq(won).to_numpy(dtype=float)
                if not weights.sum():
                    continue
                for alternative in ("target_roster_full", "prior_no_w20", "prior_no_organization"):
                    row = _metric_effect(primary_bets, bet_y, bet_probabilities["prior_full"], bet_probabilities[alternative],
                                         f"fixed_saved_side_prior_full_vs_{alternative}", month_universe,
                                         weights=weights, block_universe=proxy_universe, blocks=primary_bets.edition_proxy)
                    selected = group_mask & (np.ones(len(group_mask), dtype=bool) if won is None else primary_bets.won.eq(won).to_numpy(dtype=bool))
                    row.update({"dimension": dimension, "level": level, "outcome_control": outcome,
                                "eligible_n": int(group_mask.sum()), "n": int(weights.sum()),
                                "mean_saved_side_probability": float(primary_bets.loc[selected, "p_model_bet"].mean()),
                                "mean_odds": float(bet_odds[selected].mean()),
                                "realized_wins": int(primary_bets.loc[selected, "won"].sum())})
                    fixed_bet_profiles.append(row)

    support_controls = []
    for dimension, labels in profile_spec:
        if dimension == "league_best_of":
            continue
        labels = pd.Series(labels).fillna("unknown").astype(str)
        for level in sorted(labels.unique()):
            mask_group = labels.eq(level).to_numpy(dtype=bool)
            for truth, truth_name in ((0, "side_A_lost"), (1, "side_A_won")):
                selected = mask_group & primary_origins.y.eq(truth).to_numpy(dtype=bool)
                n = int(selected.sum())
                if n:
                    support_controls.append({"dimension": dimension, "level": level, "outcome_control": truth_name,
                                             "n": n, "fraction_of_whole_cohort": n / len(primary_origins),
                                             "mean_prior_log_loss": float(binary_log_loss_vector(y[selected], probs["prior_full"][selected], epsilon=EPSILON).mean()),
                                             "mean_target_roster_log_loss": float(binary_log_loss_vector(y[selected], probs["target_roster_full"][selected], epsilon=EPSILON).mean()),
                                             "target_roster_oracle_gain": float(np.mean(binary_log_loss_vector(y[selected], probs["prior_full"][selected], epsilon=EPSILON)-binary_log_loss_vector(y[selected], probs["target_roster_full"][selected], epsilon=EPSILON))),
                                             "mean_actual_roster_history_stat_missing_fraction": float(primary_origins.loc[selected, "stat_missing_fraction"].mean()) if primary_origins.loc[selected, "stat_missing_fraction"].notna().any() else None,
                                             "mean_visible_history_tokens": float(primary_origins.loc[selected, "history_mask_tokens"].mean())})

    raw_case_source = _make_case_sample(primary_bets, primary_origins)
    raw_case_evidence = _extract_raw_case_evidence(raw_case_source, primary_origins, source_paths["raw_history"])
    if len(raw_case_evidence):
        raw_case_source = raw_case_source.merge(raw_case_evidence, left_on=["origin_id", "golgg_match_id", "case_role"],
                                                right_on=["case_origin_id", "golgg_match_id", "case_role"], validate="one_to_one")
    output.mkdir(parents=True, exist_ok=True)
    primary_origins.to_parquet(output / "primary_origin_mechanisms.parquet", index=False)
    primary_bets.to_parquet(output / "fixed_saved_bet_mechanisms.parquet", index=False)
    evidence_frame.to_csv(output / "canonical_target_roster_history_evidence.csv", index=False)
    raw_case_source.to_csv(output / "raw_case_evidence.csv", index=False)
    pd.json_normalize(origin_mode_effects).to_csv(output / "origin_mode_effects.csv", index=False)
    pd.json_normalize(bet_modes).to_csv(output / "fixed_bet_mode_effects.csv", index=False)
    pd.json_normalize(subgroup_effects).to_csv(output / "origin_subgroup_effects.csv", index=False)
    pd.json_normalize(fixed_bet_profiles).to_csv(output / "fixed_bet_subgroup_effects.csv", index=False)
    pd.DataFrame(support_controls).to_csv(output / "outcome_support_controls.csv", index=False)

    source_hashes = {name: str(record["sha256"]) for name, record in protocol["sources"].items() if name in (
        "canonical_all", "canonical_metadata", "raw_history", "repair_predictions", "repair_receipt", "previous_paired")}
    source_hashes.update({f"corrected_bank:{key}": value for key, value in bank_hashes.items()})
    source_hashes["corrected_bank:completed.json"] = _sha256(receipt_path)
    source_hashes.update({f"feature_extractor:{name}": value for name, value in extractor_hashes.items()})
    mode_summary = {row["contrast"]: row for row in origin_mode_effects}
    bet_summary = {row["contrast"]: row for row in bet_modes}
    wrong_selected = int((~primary_bets.won.astype(bool)).sum())
    result = {
        "status": "PASS_ROSTER_HISTORY_ORGANIZATION_MECHANISMS",
        "scope": "Frozen repaired predictions and saved flat selections only; all mode contrasts are complete frozen-mode effects, not pure feature coefficients or causal feature effects.",
        "primary_cohort": {"cohort": "oddspapi_common", "source": "best_price", "horizon": "T-1h", "audit_primary_eligible": True,
                           "origins_n": len(primary_origins), "opportunities_n": len(primary_ops), "calendar_months": month_universe,
                           "fixed_saved_bets_n": len(primary_bets), "fixed_saved_bets_wins": int(primary_bets.won.sum()),
                           "fixed_saved_bets_losses": wrong_selected, "selection_quality": "exclude_q25_conflicts", "selection_policy": "raw_ev_gt_5"},
        "identity_checks": {
            "opportunity_origin_probability_equal_atol_1e-12": True,
            "opportunity_origin_outcome_equal": True,
            "same_player_set_same_cutoff_n": int(unchanged_exact.sum()),
            "probability_equality_on_unchanged_player_sets": equality,
            "corrected_bank_primary_ids_matched": int(len(primary_origins)),
            "actual_target_roster_ids_match_bank": True,
            "direct_prior_vs_target_player_set_matches_prepared_flag": True,
            "fixed_saved_selected_side_probability_matches_repaired_origin": True,
        },
        "origin_probability_scores": origin_scores,
        "origin_mode_effects": origin_mode_effects,
        "fixed_saved_bet_probability_scores": fixed_bet_scores,
        "fixed_saved_bet_mode_effects": bet_modes,
        "fixed_saved_bets_are_not_reselected_under_interventions": True,
        "fixed_selection_expected_net_return_uses_tax_12pct_saved_odds_and_is_not_realized_pnl": True,
        "oracle_after_known_nonoverlap_filter": {
            "known_nonoverlap_n": int(clean_controls.sum()), "known_nonoverlap_same_cutoff_n": int(same_known.sum()),
            "known_nonoverlap_effect": oracle_clean_effect, "known_nonoverlap_same_cutoff_effect": oracle_clean_same_day_effect,
        },
        "feature_bank_evidence": bank_evidence,
        "origin_subgroup_effects": subgroup_effects,
        "fixed_bet_subgroup_effects": fixed_bet_profiles,
        "outcome_support_controls": support_controls,
        "case_table": {"path": "raw_case_evidence.csv", "rows": int(len(raw_case_source)),
                       "selection": "Top four fixed selected losses by target-roster oracle log-loss contribution (prioritized), paired to winning saved bets in same competition family, BO and roster-change status when available; odds band preferred.",
                       "raw_first_map_rosters_match_target_bank": bool(raw_case_source.raw_first_map_roster_matches_canonical_a.all() and raw_case_source.raw_first_map_roster_matches_canonical_b.all()) if len(raw_case_source) else None,
                       "verified_prior_roster_source_rows": int(sum(int(raw_case_source[f"prior_roster_source_raw_roster_verified_{side}"].sum()) for side in ("a", "b"))) if len(raw_case_source) else 0,
                       "prior_source_roster_checks": int(2 * len(raw_case_source))},
        "source_hashes": source_hashes,
        "artifacts": ["primary_origin_mechanisms.parquet", "fixed_saved_bet_mechanisms.parquet", "canonical_target_roster_history_evidence.csv",
                      "raw_case_evidence.csv", "origin_mode_effects.csv", "fixed_bet_mode_effects.csv", "origin_subgroup_effects.csv",
                      "fixed_bet_subgroup_effects.csv", "outcome_support_controls.csv"],
        "limitations": [
            "Historical decision times and roster announcements are not certified; this remains an exposed retrospective scenario.",
            "Canonical feature-bank raw tensors use actual first-map target rosters; their history/stat/inactivity diagnostics transfer to repaired prior inputs only when player sets and cutoff day are identical.",
            "Target-roster forecasts are hindsight controls, not actionable prior forecasts. Oracle effects outside identical cutoffs also mix feature-age/cutoff differences.",
            "Roster snapshot age is the age of the prior roster source release, not player inactivity; player inactivity comes separately from the named log_days_inactive channel for seen players only.",
            "Missing numerical stats are measured from the explicit 12-channel presence mask on visible canonical player-map history; zero-valued valid stats and zero-filled missing stats are reported separately.",
            "Different frozen calibrators are used by full/no-W20/no-organization modes; the archived no-organization switch also zeros the W20 gate, so these are complete bundled mode effects, not isolated causal coefficients. No league exception, retraining, fitting, or policy selection is attempted.",
            "Tournament-name-by-year is only a proxy block; tournament editions and provenance timestamps are not certified. Subgroup intervals are nominal exploratory intervals with no multiplicity correction.",
            "Selected-side expected return changes hold saved selection and odds fixed; actual realized returns and wins cannot change under forecast-only interventions.",
        ],
    }
    # Keep an explicit concise top-line dictionary available to the parent without parsing CSVs.
    result["headline_effects"] = {
        "origin_prior_full_vs_no_w20": mode_summary["prior_full_vs_no_w20"],
        "origin_prior_full_vs_no_organization": mode_summary["prior_full_vs_no_organization"],
        "origin_prior_full_vs_target_roster": mode_summary["prior_full_vs_target_roster_oracle"],
        "fixed_bets_prior_full_vs_no_w20": bet_summary["prior_full_vs_no_w20"],
        "fixed_bets_prior_full_vs_no_organization": bet_summary["prior_full_vs_no_organization"],
        "fixed_bets_prior_full_vs_target_roster": bet_summary["prior_full_vs_target_roster_oracle"],
    }
    return result
