"""Outcome-symmetric, saved-input C0 EMEA diagnostics; never fit or select a recipe.

Run via scripts/research/run_c0_emea_diagnostics.py --protocol PATH. Outputs are
fresh report_root/tracks/diagnostics tables and a strictly segregated cause ledger.
Raw continuity is conditional on reconstruction's exact-ID membership trace;
absence is reported, never replaced by name, geography, or nationality inference.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARRAY_SHAPES = {"players": (2, 5, 21), "core": (2, 5), "team": (2, 11),
                "w20": (2, 10), "gates": (2, 2), "history": (2, 5, 16, 33),
                "mask": (2, 5, 16), "champions": (2, 5, 16)}
ROLES = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT", "UNKNOWN"]
STATS = ["kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm", "gd@15", "csd@15", "xpd@15", "kp%", "dmg%"]
RATINGS = ["elo", "gl_mu", "gl_rd", "ts_mu", "ts_sigma", "os_mu", "os_sigma", "pl_mu", "pl_sigma", "tm_mu", "tm_sigma"]
SCHEMA = "c0_emea_diagnostics_v1"


def _path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _ids(frame: pd.DataFrame, column: str = "golgg_match_id") -> pd.DataFrame:
    frame = frame.copy()
    if frame[column].isna().any():
        raise ValueError(f"Null exact ID in {column}")
    frame[column] = frame[column].astype(str)
    return frame


def _unique(frame: pd.DataFrame, label: str) -> None:
    if frame.golgg_match_id.duplicated().any():
        raise ValueError(f"Duplicate IDs in {label}")


def _loss(y: pd.Series, p: pd.Series) -> np.ndarray:
    yv, pv = y.to_numpy(float), p.to_numpy(float)
    if not np.isfinite(pv).all() or ((pv <= 0) | (pv >= 1)).any():
        raise ValueError("Diagnostic loss requires finite probabilities strictly inside (0,1); no clipping")
    if not np.isin(yv, [0, 1]).all():
        raise ValueError("Nonbinary outcome")
    return -(yv * np.log(pv) + (1 - yv) * np.log1p(-pv))


def _constant(path: Path, name: str) -> Any:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source constant {name} in {path}")


def _inventory(research: Path, execution: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    feature = research / "unified-ratings-20260911/code/scripts"
    temporal, roster = feature / "temporal_player_features.py", feature / "roster_optional_features.py"
    if _constant(temporal, "STATS") != STATS or _constant(roster, "ROLES") != ROLES:
        raise ValueError("Archived feature-channel definitions differ from diagnostic contract")
    if _constant(temporal, "HISTORY_WIDTH") != 33 or _constant(temporal, "LENGTH") != 16:
        raise ValueError("Archived history dimensions differ")
    if _constant(roster, "RATING_FEATURES") != RATINGS:
        raise ValueError("Archived rating-channel definitions differ")
    channels = {
        "players": RATINGS + ["log_games", "log_series", "log_days_inactive", "unseen"] + ["role_" + r for r in ROLES],
        "core": _constant(roster, "CORE_FEATURES"), "team": RATINGS,
        "w20": _constant(roster, "W20_FEATURES"),
        "gates": ["organization_continuity_support", "w20_valid_fraction_times_organization_support"],
        "history": STATS + ["valid_" + s for s in STATS] + ["win", "log_age_days", "opponent_prior_mean_glicko"] + ["role_" + r for r in ROLES],
        "mask": ["released_map_token_validity"], "champions": ["frozen_champion_code_not_consumed_by_attention"],
    }
    source_relatives = ["nonlinear-history-20260912/code/src/models/" + n for n in
                        ("temporal_architectures.py", "nonlinear_history.py", "beta_training.py", "auxiliary_player.py", "coherent_series.py", "roster_optional.py")]
    source_hashes = {"research:" + str(p.relative_to(research)): _hash(p) for p in [temporal, roster, *[research / r for r in source_relatives]]}
    # Bind actual archived architecture sources to the original artifact contract,
    # not merely to class defaults or similarly named current local modules.
    original_pins = execution["contract"]["source_sha256"]
    for source_id, digest in source_hashes.items():
        if source_id in original_pins and original_pins[source_id] != digest:
            raise ValueError(f"Original C0 architecture source changed: {source_id}")
    inv = {"schema": SCHEMA, "channels": {k: [{"index": i, "name": n} for i, n in enumerate(names)] for k, names in channels.items()},
           "array_shapes_per_match": ARRAY_SHAPES,
           "wire_dtypes": {k: "float32" for k in ("players", "core", "team", "w20", "gates")},
           "architecture": {"identity": "c0-native-2026-w32-e12-v1", "kind": "attention", "width": 32,
                            "history_tokens_per_player": 16, "history_fields": 33, "player_fields": 21,
                            "player_linear": [21, 32], "history_linear_tanh": [33, 32], "attention_linear": [32, 1],
                            "shared_head_linear": [134, 32, 1], "auxiliary_head_linear": [32, 12],
                            "network_parameter_count_from_bound_source": 6705, "additional_beta_eta_parameters": 1,
                            "parameter_count_scope": "each history member only; excludes fitted baseline/scalers/nonneural experts",
                            "empty_token": "learned; remains valid even with zero prior maps",
                            "pooling": "shared player mean and population std; shared team utility difference",
                            "explicit_champion_embedding": False, "typed_graph_branch": False,
                            "current_target_role_or_draft_used": False,
                            "history_identity_key": "exact player ID globally across source competitions and teams",
                            "organization_identity_key": "exact team ID; no inferred predecessor/name merge",
                            "temporal_memory": "last256 retained maps per exact player; last16 encoded; longer ratings/counts retained",
                            "training": execution["protocol"]["training"], "split": execution["protocol"]["split"],
                            "calibration": execution["protocol"]["calibration"],
                            "blindspot_bounds": "No explicit league/patch/champion token in attention. Omission alone is neither demonstrated defect nor supported new mechanism."},
           "semantics": {"age": "expm1(FP16 log1p(cutoff minus whole-series effective release day)); quantized approximation, not exact raw age",
                         "opponent_rating": "mean first-map opponent Glicko before that series rating update",
                         "stats": "12 values plus independent12 validity masks; zero invalid stats not evidence of true zero",
                         "role": "role from prior released map; current future role not used",
                         "organization_gate": "overlap^2 * min(team_prior_games/20,1) * exp(-age/180)",
                         "w20_gate": "organization_gate * valid_window_games/20",
                         "coplay": "exact player-ID pairs with180-day exponential decay, independent of team/competition IDs",
                         "probability": "saved series winner probability; no second BO projection"}}
    inv["wire_dtypes"].update(history="float16", mask="uint8", champions="uint16")
    return inv, source_hashes


def _feature_tables(selection: pd.DataFrame, bank: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = {k: np.load(bank / (k + ".npy"), mmap_mode="r") for k in ARRAY_SHAPES}
    for key, shape in ARRAY_SHAPES.items():
        if arrays[key].shape[1:] != shape:
            raise ValueError(f"Unexpected bank shape for {key}: {arrays[key].shape}")
    cases, players = [], []
    for row in selection.to_dict("records"):
        idx = int(row["bank_row_index"])
        hist = np.asarray(arrays["history"][idx], dtype=np.float64)
        mask = np.asarray(arrays["mask"][idx], dtype=bool)
        pv = np.asarray(arrays["players"][idx], dtype=np.float64)
        core = np.asarray(arrays["core"][idx], dtype=np.float64)
        gates = np.asarray(arrays["gates"][idx], dtype=np.float64)
        valid = hist[mask]
        age = np.expm1(valid[:, 25]) if len(valid) else np.array([])
        item = {"golgg_match_id": row["golgg_match_id"], "bank_row_index": idx,
                "history_valid_tokens": int(mask.sum()), "history_empty_players": int((mask.sum(2) == 0).sum()),
                "history_age_mean_days_quantized": float(age.mean()) if len(age) else np.nan,
                "history_age_max_days_quantized": float(age.max()) if len(age) else np.nan,
                "history_opponent_rating_mean": float(valid[:, 26].mean()) if len(valid) else np.nan,
                "history_unknown_role_fraction": float(valid[:, 32].mean()) if len(valid) else np.nan,
                "history_statmask_support": float(valid[:, 12:24].mean()) if len(valid) else np.nan,
                "current_prior_unknown_role_players": int(pv[:, :, 20].sum()),
                "organization_gate_min": float(gates[:, 0].min()), "organization_gate_mean": float(gates[:, 0].mean()),
                "w20_gate_min": float(gates[:, 1].min()), "w20_gate_mean": float(gates[:, 1].mean()),
                "coplay_pair_min": float(np.expm1(core[:, 1]).min()),
                "coplay_mean_min": float(np.expm1(core[:, 0]).min()),
                "coplay_unseen_pair_fraction_max": float(core[:, 3].max()),
                "exact_roster_games_min": float(np.expm1(core[:, 4]).min()),
                "organization_gate_zero": bool((gates[:, 0] == 0).any()), "w20_gate_zero": bool((gates[:, 1] == 0).any()),
                "coplay_unseen_pair_any": bool((core[:, 3] > 0).any())}
        for j, role in enumerate(ROLES):
            item["history_role_fraction_" + role] = float(valid[:, 27 + j].mean()) if len(valid) else np.nan
            item["history_prior_current_role_fraction_" + role] = float(pv[:, :, 15 + j].mean())
        item["history_opponent_rating_std"] = float(valid[:, 26].std()) if len(valid) else np.nan
        item["history_age_median_days_quantized"] = float(np.median(age)) if len(age) else np.nan
        w20 = np.asarray(arrays["w20"][idx], dtype=np.float64)
        for j, name in enumerate(("win", "kills", "deaths", "dpm", "vspm", "gd15", "towers", "nashors", "gold", "duration")):
            item["w20_channel_mean_" + name] = float(w20[:, j].mean())
        for j, stat in enumerate(STATS):
            item["stat_support_" + stat] = float(valid[:, 12 + j].mean()) if len(valid) else np.nan
        for side, roster_key in enumerate(("roster_a", "roster_b")):
            roster = row[roster_key]
            if len(roster) != 5:
                raise ValueError("Invalid exact-ID roster")
            for slot, pid in enumerate(roster):
                vh = hist[side, slot][mask[side, slot]]
                ages = np.expm1(vh[:, 25])
                players.append({"golgg_match_id": row["golgg_match_id"], "side": side, "slot": slot, "player_id": str(pid),
                                "target_team_id": str(row["team1_id" if side == 0 else "team2_id"]),
                                "prior_role": ROLES[int(np.argmax(pv[side, slot, 15:21]))],
                                "history_tokens": len(vh), "age_mean_days_quantized": float(ages.mean()) if len(vh) else np.nan,
                                "age_latest_days_quantized": float(ages.min()) if len(vh) else np.nan,
                                "opponent_rating_mean": float(vh[:, 26].mean()) if len(vh) else np.nan,
                                "statmask_support": float(vh[:, 12:24].mean()) if len(vh) else np.nan,
                                "prior_series_quantized": float(np.expm1(pv[side, slot, 12])),
                                "prior_inactive_days_quantized": float(np.expm1(pv[side, slot, 13])),
                                "unseen": bool(pv[side, slot, 14])})
        cases.append(item)
    return pd.DataFrame(cases), pd.DataFrame(players)


def _matching(selection: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Independently reproduce frozen rules, without consulting outcome/market/error."""
    allowed = ["golgg_match_id", "date", "best_of", "min_player_series", "unknown_players", "p_c0", "is_emea_primary", "is_polish_primary"]
    data = selection[allowed].copy()
    data["month"] = data.date.astype(str).str[:7]
    data["day"] = pd.to_datetime(data.date)
    data["support"] = pd.cut(data.min_player_series, [-1, 9, 19, np.inf], labels=["lt10", "10to19", "20plus"]).astype(str)
    data["unknown_class"] = data.unknown_players.gt(0)
    data["confidence"] = np.maximum(data.p_c0, 1 - data.p_c0)
    pool = data[data.is_polish_primary & ~data.is_emea_primary]
    emea = data[data.is_emea_primary]
    if links.emea_golgg_match_id.duplicated().any() or set(links.emea_golgg_match_id) != set(emea.golgg_match_id):
        raise ValueError("Control links must account for each of all89EMEA exactly once")
    records = []
    for target in emea.to_dict("records"):
        eligible = pool[pool.best_of.eq(target["best_of"])]
        month = eligible[eligible.month.eq(target["month"])]
        strict = month[month.support.eq(target["support"]) & month.unknown_class.eq(target["unknown_class"])]
        distance = (eligible.day - target["day"]).dt.days.abs()
        tiers = [("exact_month_format_support_unknown", strict), ("exact_month_format", month),
                 ("same_format_within31days", eligible[distance <= 31])]
        candidates, level = pool.iloc[:0], "unmatched"
        for name, candidates in tiers:
            if len(candidates):
                level = name
                break
        expected = None
        if len(candidates):
            def key(r: Mapping[str, Any]) -> tuple[Any, ...]:
                gid = str(r["golgg_match_id"])
                id_key = (0, int(gid)) if gid.isascii() and gid.isdecimal() else (1, gid)
                return (abs(np.log1p(r["min_player_series"]) - np.log1p(target["min_player_series"])),
                        abs(r["confidence"] - target["confidence"]), abs((r["day"] - target["day"]).days), *id_key)
            expected = min(candidates.to_dict("records"), key=key)["golgg_match_id"]
        supplied = links[links.emea_golgg_match_id.eq(target["golgg_match_id"])].iloc[0]
        actual = None if pd.isna(supplied.control_golgg_match_id) or str(supplied.control_golgg_match_id) == "" else str(supplied.control_golgg_match_id)
        if expected != actual:
            raise ValueError(f"Outcome-blind control mismatch for {target['golgg_match_id']}: frozen={actual}, reproduced={expected}")
        records.append({**supplied.to_dict(), "reproduced_match_level": level, "eligible_candidates_at_level": len(candidates),
                        "outcome_blind_match_reproduced": True, "named_absence": "no_nonEMEA_Polish2plus_same_BO_within31days" if expected is None else None})
    result = pd.DataFrame(records)
    reuse = result.control_golgg_match_id.dropna().value_counts()
    result["control_use_count"] = result.control_golgg_match_id.map(reuse).fillna(0).astype(int)
    return result


def _continuity(report_root: Path, selection: pd.DataFrame, players: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = report_root / "tracks/reconstruction/history_membership.parquet"
    selected = selection[selection.is_emea_primary | selection.is_matched_control]
    skeleton = players[players.golgg_match_id.isin(selected.golgg_match_id)].copy()
    if not path.is_file():
        skeleton["identity_trace_status"] = "ABSENT_RECONSTRUCTION_MEMBERSHIP"
        teams = skeleton[["golgg_match_id", "side", "target_team_id"]].drop_duplicates().rename(columns={"target_team_id": "team_id"})
        teams["team_identity_scope"] = "ABSENT_RECONSTRUCTION_MEMBERSHIP; saved exact target team ID only"
        teams.to_parquet(out / "team_identity_continuity.parquet", index=False)
        return skeleton, {"status": "ABSENT_RECONSTRUCTION_MEMBERSHIP", "named_absence": str(path.relative_to(PROJECT_ROOT)), "exact_raw_continuity_supported": False}
    import pyarrow.dataset as ds
    wanted = ["golgg_match_id", "side", "slot", "player_id", "source_series_id", "source_map_id", "map_index", "played_day", "effective_release_day", "source_team_id", "source_tournament_name", "role", "age_days", "token_index"]
    dataset = ds.dataset(path, format="parquet")
    missing = set(wanted) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Raw membership trace lacks diagnostic contract fields: {sorted(missing)}")
    trace = dataset.to_table(columns=wanted, filter=ds.field("golgg_match_id").isin(selected.golgg_match_id.tolist())).to_pandas()
    for field in ("golgg_match_id", "player_id", "source_series_id", "source_team_id"):
        trace[field] = trace[field].astype("string")
    trace["source_order_numeric"] = trace.source_series_id.map(lambda v: int(v) if str(v).isascii() and str(v).isdecimal() else np.inf)
    trace = trace.sort_values(["golgg_match_id", "side", "slot", "effective_release_day", "source_order_numeric", "source_series_id", "map_index"], kind="stable")
    trace["is_emea_source"] = trace.source_tournament_name.fillna("").str.contains("EMEA", case=False, regex=False)
    trace["source_competition_class"] = np.where(trace.is_emea_source, "EMEA_source_label", "nonEMEA_source_label_not_geography")
    keys = ["golgg_match_id", "side", "slot", "player_id"]
    if trace.duplicated(keys + ["source_map_id"]).any():
        raise ValueError("Duplicate exact player/source-map membership")
    binding = trace[keys].drop_duplicates().merge(skeleton[keys], on=keys, how="left", indicator=True, validate="one_to_one")
    if not binding._merge.eq("both").all():
        raise ValueError("Raw trace player IDs do not match native target slots")
    cutoff = trace.golgg_match_id.map(selected.set_index("golgg_match_id").date)
    raw_age = (pd.to_datetime(cutoff) - pd.to_datetime(trace.effective_release_day)).dt.days
    if raw_age.le(0).any() or not np.array_equal(raw_age.to_numpy(), trace.age_days.to_numpy()):
        raise ValueError("Raw membership violates strict-prior release day or exact age")
    actual_tokens = trace[trace.token_index.notna()].groupby(keys).size()
    expected_tokens = skeleton.set_index(keys).history_tokens
    if not actual_tokens.reindex(expected_tokens.index, fill_value=0).eq(expected_tokens).all():
        raise ValueError("Raw last16 membership differs from saved native valid-token counts")
    grouped = trace.groupby(keys, sort=False, dropna=False)
    for field in ("source_team_id", "role", "source_tournament_name", "source_competition_class"):
        previous = grouped[field].shift()
        trace[field + "_changed"] = previous.notna() & trace[field].ne(previous)
    trace.to_parquet(out / "identity_transitions.parquet", index=False)
    summaries = []
    for key, records in trace.groupby(keys, sort=False, dropna=False):
        summaries.append(dict(zip(keys, key)) | {
            "retained_raw_maps": len(records), "retained_raw_series": records.source_series_id.nunique(),
            "exact_team_ids_seen": records.source_team_id.dropna().astype(str).unique().tolist(),
            "roles_seen": records.role.dropna().astype(str).unique().tolist(),
            "competitions_seen": records.source_tournament_name.dropna().astype(str).unique().tolist(),
            "retained_emea_maps": int(records.is_emea_source.sum()), "retained_nonemea_maps": int((~records.is_emea_source).sum()),
            "cross_competition_exact_id": records.source_tournament_name.nunique() > 1,
            "cross_emea_nonemea_exact_id": records.is_emea_source.any() and (~records.is_emea_source).any(),
            "team_changes": int(records.source_team_id_changed.sum()), "role_changes": int(records.role_changed.sum()),
            "competition_changes": int(records.source_tournament_name_changed.sum()),
            "last_source_team_id": records.iloc[-1].source_team_id, "last_source_role": records.iloc[-1].role,
            "latest_raw_age_days": int(records.age_days.min()),
            "raw_tokens_last16": int(records.token_index.notna().sum()), "identity_trace_status": "EXACT_RAW_ID_TRACE"})
    summary = pd.DataFrame(summaries)
    if len(summary):
        skeleton = skeleton.merge(summary, on=keys, how="left", validate="one_to_one")
        skeleton["identity_trace_status"] = skeleton.identity_trace_status.fillna("NO_RETAINED_RAW_HISTORY")
        skeleton["last_team_differs_target"] = skeleton.last_source_team_id.notna() & skeleton.last_source_team_id.astype(str).ne(skeleton.target_team_id)
    else:
        skeleton["identity_trace_status"] = "NO_RETAINED_RAW_HISTORY"
    team_records = []
    for (gid, side), slots in skeleton.groupby(["golgg_match_id", "side"], sort=False):
        target_team = str(slots.iloc[0].target_team_id)
        retained = trace[trace.golgg_match_id.eq(gid) & trace.side.eq(side)]
        same_team = retained[retained.source_team_id.eq(target_team)]
        team_records.append({"golgg_match_id": gid, "side": side, "team_id": target_team,
                             "target_exact_player_ids": slots.player_id.tolist(),
                             "retained_player_maps_same_team": len(same_team),
                             "retained_player_maps_other_team": len(retained) - len(same_team),
                             "same_team_source_competitions": same_team.source_tournament_name.dropna().unique().tolist(),
                             "same_team_cross_competition": same_team.source_tournament_name.nunique() > 1,
                             "same_team_emea_and_nonemea": bool(same_team.is_emea_source.any() and (~same_team.is_emea_source).any()),
                             "team_identity_scope": "exact target team ID in retained histories of current five players; no inferred organization lineage"})
    pd.DataFrame(team_records).to_parquet(out / "team_identity_continuity.parquet", index=False)
    return skeleton, {"status": "EXACT_RAW_ID_TRACE", "membership_sha256": _hash(path), "rows": len(trace),
                      "target_ids": selected.golgg_match_id.tolist(), "exact_raw_continuity_supported": True,
                      "limits": "last256 released maps/exact player ID; changing ID cannot be linked without authoritative identity evidence; nonEMEA source labels do not imply domestic geography; first-map roster is a conditional historical scenario"}


def _summary(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    d = frame.delta_ll.to_numpy(float)
    return {"group": name, "n": len(frame), "ll_c0": frame.ll_c0.mean(), "ll_market": frame.ll_market.mean(),
            "delta_ll_mean": frame.delta_ll.mean(), "delta_ll_sum": d.sum(),
            "positive_excess_sum": np.maximum(d, 0).sum(), "favorable_compensation_sum": np.minimum(d, 0).sum(),
            "absolute_c0_loss_sum": frame.ll_c0.sum(), "brier_c0": frame.brier_c0.mean(), "brier_market": frame.brier_market.mean(),
            "unfavorable_count": int((d > 0).sum()), "favorable_count": int((d < 0).sum()), "tie_count": int((d == 0).sum())}


def _distributions(cases: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    controls = accounting.control_golgg_match_id.dropna().tolist()
    indexed = cases.set_index("golgg_match_id")
    groups = {"EMEA_all89": cases[cases.is_emea_primary], "controls_unique": indexed.loc[list(dict.fromkeys(controls))].reset_index(),
              "controls_link_weighted_not_independent": indexed.loc[controls].reset_index()}
    groups.update({"EMEA_" + state: cases[cases.is_emea_primary & cases.relative_case.eq(state)] for state in ("favorable", "unfavorable", "tie")})
    groups.update({"controls_unique_" + state: groups["controls_unique"][groups["controls_unique"].relative_case.eq(state)] for state in ("favorable", "unfavorable", "tie")})
    prefixes = ("history_", "stat_support_", "organization_gate_", "w20_", "coplay_", "exact_roster_", "continuity_")
    fields = [c for c in cases if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(cases[c])]
    fields += [c for c in ("min_player_series", "unknown_players", "w20_min_valid_games", "current_prior_unknown_role_players") if c in cases and c not in fields]
    rows = []
    for group, frame in groups.items():
        for field in fields:
            values = pd.to_numeric(frame[field], errors="coerce").dropna().astype(float)
            rows.append({"group": group, "field": field, "n_cases": len(frame), "observed": len(values), "missing": int(frame[field].isna().sum()),
                         "mean": values.mean(), "std_descriptive": values.std(), "min": values.min(),
                         "q25": values.quantile(.25), "median": values.median(), "q75": values.quantile(.75), "max": values.max(),
                         "zero_fraction": values.eq(0).mean(), "comparison_kind": "association_only_not_causal"})
    return pd.DataFrame(rows)


def run_diagnostics(protocol_path: str | Path) -> dict[str, Any]:
    """Write complete fresh diagnostics; parent alone owns candidate freeze/fit."""
    protocol_file = _path(protocol_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    paths = protocol["paths"]
    locator = _path(paths["research_root_locator"])
    research = _path(locator.read_text(encoding="utf-8").strip())
    bank = research / paths["bank_relative"]
    report_root = _path(paths["report_root"])
    out = report_root / "tracks/diagnostics"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Diagnostics output must be fresh: {out}")
    selection = _ids(pd.read_parquet(_path(paths["selection"])))
    native = _ids(pd.read_parquet(_path(paths["native_control"])))
    panel = _ids(pd.read_parquet(_path(paths["previous_audit_panel"])))
    primary = panel[panel.comparison_cohort.eq("Polish2plus")].copy()
    for frame, label in ((selection, "selection"), (native, "native"), (primary, "Polish2plus")):
        _unique(frame, label)
    if len(selection) != 3103 or set(selection.golgg_match_id) != set(native.golgg_match_id):
        raise ValueError("Complete3103nativecohort must be preserved")
    if len(primary) != 1467 or selection.is_polish_primary.sum() != 1467 or set(primary.golgg_match_id) != set(selection.loc[selection.is_polish_primary, "golgg_match_id"]):
        raise ValueError("Complete1467Polishprimarycohort must be preserved")
    if selection.is_emea_primary.sum() != 89:
        raise ValueError("All89primaryEMEA required")
    emea_panel = primary[primary.league.astype(str).str.contains("EMEA", case=False, regex=False)]
    if set(emea_panel.golgg_match_id) != set(selection.loc[selection.is_emea_primary, "golgg_match_id"]):
        raise ValueError("Frozen89EMEA IDs do not match saved primary panel")
    bank_meta = _ids(pd.read_parquet(bank / "metadata.parquet"))
    _unique(bank_meta, "bank")
    indices = selection.bank_row_index.to_numpy(int)
    if not np.array_equal(bank_meta.iloc[indices].golgg_match_id.to_numpy(), selection.golgg_match_id.to_numpy()):
        raise ValueError("Stable bank row/ID alignment differs")
    input_hashes = {str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else "research:" + str(p.relative_to(research)): _hash(p)
                    for p in [protocol_file, locator, _path(paths["selection"]), _path(paths["control_links"]),
                              _path(paths["native_control"]), _path(paths["previous_audit_panel"]), bank / "metadata.parquet"]}
    for name, expected in protocol.get("input_sha256", {}).items():
        if name in input_hashes and input_hashes[name] != expected:
            raise ValueError(f"Frozen diagnostic input changed: {name}")
    links = pd.read_csv(_path(paths["control_links"]), dtype={"emea_golgg_match_id": str, "control_golgg_match_id": str})
    accounting = _matching(selection, links)
    execution_path = _path(paths["native_model_root"]) / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    inventory, source_hashes = _inventory(research, execution)
    input_hashes[str(execution_path.relative_to(PROJECT_ROOT))] = _hash(execution_path)
    for key in ARRAY_SHAPES:
        input_hashes["research:" + str((bank / (key + ".npy")).relative_to(research))] = _hash(bank / (key + ".npy"))
    for name, expected in protocol.get("input_sha256", {}).items():
        if name in input_hashes and input_hashes[name] != expected:
            raise ValueError(f"Frozen diagnostic input changed: {name}")
    features, players = _feature_tables(selection, bank)
    # Native identity/order is independent of panel orientation. Losses use saved
    # primary panel orientation; no positional team-name joins or re-projection.
    native_fields = [c for c in native.columns if c not in selection.columns or c == "golgg_match_id"]
    cases = selection.merge(native[native_fields], on="golgg_match_id", validate="one_to_one").merge(features, on=["golgg_match_id", "bank_row_index"], validate="one_to_one")
    primary["ll_c0"] = _loss(primary.y, primary.p_C0)
    primary["ll_market"] = _loss(primary.y, primary.p_market)
    primary["delta_ll"] = primary.ll_c0 - primary.ll_market
    primary["brier_c0"] = (primary.p_C0 - primary.y) ** 2
    primary["brier_market"] = (primary.p_market - primary.y) ** 2
    primary["relative_case"] = np.select([primary.delta_ll.lt(0), primary.delta_ll.gt(0)], ["favorable", "unfavorable"], default="tie")
    scoring_columns = ["golgg_match_id", "event_id", "p_C0", "p_market", "ll_c0", "ll_market", "delta_ll", "brier_c0", "brier_market", "relative_case"]
    cases = cases.merge(primary[scoring_columns], on="golgg_match_id", how="left", validate="one_to_one")
    out.mkdir(parents=True, exist_ok=True)
    continuity, trace_receipt = _continuity(report_root, selection, players, out)
    trace_receipt["preserved_reconstruction_evidence"] = {}
    for name in ("team_state", "player_state", "array_differences"):
        source = report_root / "tracks/reconstruction" / (name + ".parquet")
        if source.is_file():
            frame = pd.read_parquet(source)
            if "golgg_match_id" in frame:
                frame = _ids(frame)
                if not set(frame.golgg_match_id).issubset(set(selection.golgg_match_id)):
                    raise ValueError(f"Reconstruction {name} contains an unknown native target ID")
            frame.to_parquet(out / (name + "_preserved.parquet"), index=False)
            trace_receipt["preserved_reconstruction_evidence"][name] = {"status": "PRESERVED_ALL_ROWS", "rows": len(frame), "sha256": _hash(source)}
        else:
            trace_receipt["preserved_reconstruction_evidence"][name] = {"status": "NAMED_ABSENCE", "path": str(source.relative_to(PROJECT_ROOT))}
    if "team_changes" in continuity:
        agg = continuity.groupby("golgg_match_id").agg(continuity_team_changes=("team_changes", "sum"),
            continuity_role_changes=("role_changes", "sum"), continuity_competition_changes=("competition_changes", "sum"),
            continuity_cross_competition_players=("cross_competition_exact_id", "sum"),
            continuity_cross_emea_nonemea_players=("cross_emea_nonemea_exact_id", "sum"),
            continuity_last_team_differs_players=("last_team_differs_target", "sum"))
        # Missing trace is not zero support; preserve named missingness.
        counts = continuity.groupby("golgg_match_id").identity_trace_status.apply(lambda s: s.eq("EXACT_RAW_ID_TRACE").sum())
        agg["continuity_players_with_raw_history"] = counts
        cases = cases.merge(agg, on="golgg_match_id", how="left", validate="one_to_one")
    for side_name, link_key in (("emea", "emea_golgg_match_id"), ("control", "control_golgg_match_id")):
        detail = cases.drop(columns=["roster_a", "roster_b"], errors="ignore").rename(columns={c: side_name + "_" + c for c in cases if c not in ("roster_a", "roster_b")})
        accounting = accounting.merge(detail, left_on=link_key, right_on=side_name + "_golgg_match_id", how="left", validate="many_to_one")
    for field in ("delta_ll", "history_statmask_support", "history_age_mean_days_quantized", "history_opponent_rating_mean", "organization_gate_min", "w20_gate_min", "coplay_mean_min"):
        accounting["paired_emea_minus_control_" + field] = accounting["emea_" + field] - accounting["control_" + field]
    distributions = _distributions(cases, accounting)
    scored = cases[cases.is_polish_primary]
    summaries = [_summary(scored, "all1467Polishprimary"), _summary(scored[scored.is_emea_primary], "all89EMEA"),
                 _summary(scored[~scored.is_emea_primary], "all1378nonEMEAprimary")]
    for state in ("favorable", "unfavorable", "tie"):
        summaries.append(_summary(scored[scored.is_emea_primary & scored.relative_case.eq(state)], "EMEA_" + state))
    control_ids = accounting.control_golgg_match_id.dropna().tolist()
    summaries.extend([_summary(scored[scored.golgg_match_id.isin(control_ids)], "controls_unique"),
                      _summary(scored.set_index("golgg_match_id").loc[control_ids].reset_index(), "controls_link_weighted_not_independent")])
    total, emea = summaries[0], summaries[1]
    contribution = {"EMEA_fixture_prevalence": 89 / 1467,
                    "EMEA_signed_net_deficit_share": emea["delta_ll_sum"] / total["delta_ll_sum"] if total["delta_ll_sum"] else None,
                    "EMEA_positive_excess_share": emea["positive_excess_sum"] / total["positive_excess_sum"] if total["positive_excess_sum"] else None,
                    "EMEA_absolute_C0_loss_share": emea["absolute_c0_loss_sum"] / total["absolute_c0_loss_sum"],
                    "interpretation": "signed net share can exceed100% because nonEMEA gains offset losses; not a causal attribution"}
    negative_paths = ["data/08_reporting/a0_capacity/native_capacity_20261004_01/decision.json",
                      "data/08_reporting/c0_meta_stable/c0_meta_stable_20261007_01/evaluation/decision.json",
                      "data/08_reporting/c0_meta/c0_meta_20261007_094552_integrity_v2/evaluation/decision.json",
                      "data/08_reporting/model_bookmaker_audit/model_bookmaker_deep_audit_20261009_213835Z/REPORT.md"]
    negatives = [{"path": p, "sha256": _hash(_path(p)) if _path(p).is_file() else None,
                  "status": "EXISTING_EVIDENCE_NO_REOPEN" if _path(p).is_file() else "NAMED_ABSENCE"} for p in negative_paths]
    reconstruction_receipt = report_root / "tracks/reconstruction/receipt.json"
    reconstruction = json.loads(reconstruction_receipt.read_text()) if reconstruction_receipt.is_file() else None
    ledger = {
        "schema": "c0_emea_cause_ledger_v1", "diagnosis": "NO_IDENTIFIED_CAUSE",
        "confirmedfact": [
            {"claim": "All3103 native inputs/all1467 primary/all89EMEA preserved; no outcome-selected exclusions", "evidence": ["native_case_features.parquet", "primary_panel_preserved.parquet", "emea_cases.parquet"]},
            {"claim": "Per-case control links reproduced from frozen outcome-blind rules using only native date/BO/support/unknown/confidence/ID", "evidence": "case_control_accounting.parquet", "inputs_used": ["date", "best_of", "min_player_series", "unknown_players", "p_c0", "golgg_match_id"], "inputs_never_used": ["y", "market", "prediction_error"]},
            {"claim": "Actual C0 attention already consumes age/opponent-rating/prior-role/statistic masks/player-rating support/coplay/W20/organization gates", "evidence": "feature_channel_inventory.json", "source_sha256": source_hashes},
            {"claim": "Saved-series LL comparisons preserve favorable/unfavorable/tie cases symmetrically", "evidence": ["loss_contributions.json", "feature_distributions.csv"]}],
        "evidence": {"coverage": {"native": len(cases), "Polishprimary": len(primary), "EMEA": int(cases.is_emea_primary.sum())},
                     "contributions": contribution, "raw_continuity": trace_receipt,
                     "reconstruction_receipt": {"path": str(reconstruction_receipt.relative_to(PROJECT_ROOT)), "sha256": _hash(reconstruction_receipt) if reconstruction is not None else None, "receipt": reconstruction}},
        "hypothesis": [
            {"claim": "Cross-competition opponent-quality differences may explain part of EMEA residuals", "status": "UNPROVEN_DIAGNOSTIC_ASSOCIATION_ONLY", "why_not_new_recipe": "Opponent prior rating is already a token input; association cannot isolate an omitted mechanism."},
            {"claim": "Role/team/competition transitions may reduce information transfer", "status": "UNPROVEN", "why_not_defect": "Exact-ID player history is global; continuity observations do not authenticate a changed-ID link or prove missing competition normalization."},
            {"claim": "No explicit patch/competition/champion embedding is an attention representation omission", "status": "OMISSION_NOT_SUPPORTED_BLINDSPOT", "why_not_candidate": "No intervention, source defect, or mechanism-specific symmetric evidence establishes benefit; arbitrary new feature prohibited."}],
        "negativecomparison": [{"proposal": "capacity/depth or generic calibration/meta windows/family connector", "status": "ALREADY_REJECTED_NOT_NEW_RECIPE", "evidence": negatives,
                                 "scope_anchor": "conditional_experiment.no_repeat; family-specific old result is not reconstructed by this diagnostic"},
                                {"proposal": "remove W20/organization or flatten/cap C0", "status": "NOT_SUPPORTED_BY_PRIOR_SAVED_MODE_AUDIT", "evidence": negative_paths[-1]}],
        "named_absences": ([] if trace_receipt["exact_raw_continuity_supported"] else [trace_receipt["named_absence"]])
                          + [v["path"] for v in trace_receipt["preserved_reconstruction_evidence"].values() if v["status"] == "NAMED_ABSENCE"]
                          + ["authenticated_roster_announcement_and_model_issuance_clocks", "authoritative_identity_links_for_changed_player_or_team_IDs"],
        "candidate_gate_recommendation": {"status": "FAILED_NO_SUPPORTED_CAUSE", "fits_recommended": 0, "candidate": None,
                                           "owner": "parent integration/candidate freeze", "reason": "Descriptive prevalence, score concentration, and absence of an explicit channel do not establish an input defect or substantively new supported mechanism. Strict reconstruction differences, if present, require parent/oracle investigation; they are not automatically a diagnosed repair."},
        "limits": ["Retrospective diagnostic associations are not causal effects or confirmatory league tests.",
                   "Controls reuse is reported; paired links are not independent observations.",
                   "All history clocks describe a conditional actual-first-map-roster scenario, not authenticated issuance.",
                   "Raw record identity uses exact source IDs only, no fuzzy merging or nationality inference."]}
    cases.to_parquet(out / "native_case_features.parquet", index=False)
    primary.to_parquet(out / "primary_panel_preserved.parquet", index=False)
    cases[cases.is_emea_primary].to_parquet(out / "emea_cases.parquet", index=False)
    players.to_parquet(out / "player_feature_support.parquet", index=False)
    continuity.to_parquet(out / "player_identity_continuity.parquet", index=False)
    accounting.to_parquet(out / "case_control_accounting.parquet", index=False)
    accounting.drop(columns=[c for c in accounting if accounting[c].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any()]).to_csv(out / "case_control_accounting.csv", index=False)
    distributions.to_csv(out / "feature_distributions.csv", index=False)
    _json(out / "feature_channel_inventory.json", inventory)
    _json(out / "loss_contributions.json", {"groups": summaries, "relative_prevalence_and_contributions": contribution})
    _json(out / "cause_ledger.json", ledger)
    report_lines = [
        "C0 EMEA reconstruction diagnostics",
        "Diagnosis: NO_IDENTIFIED_CAUSE. Recommended fits: 0. Parent owns candidate freeze/fit decision.",
        f"Preserved cohorts: {len(cases)} native inputs; {len(primary)} Polish primary; {int(cases.is_emea_primary.sum())} EMEA.",
        f"Controls: {len(control_ids)} paired links; {len(set(control_ids))} unique controls; repeated links are not independent matches.",
        "Matching independently reproduced without y, market, or prediction error.",
        f"Raw identity trace: {trace_receipt['status']}. IDs exact, no name/nationality/geography inference.",
        f"EMEA prevalence: {contribution['EMEA_fixture_prevalence']:.6f}.",
        f"EMEA signed net deficit share: {contribution['EMEA_signed_net_deficit_share']}; positive-excess share: {contribution['EMEA_positive_excess_share']}.",
        "Outcome-symmetric score groups (same saved series probability orientation, no clipping):",
        *[f"  {r['group']}: n={r['n']}, meanLLdiff={r['delta_ll_mean']}, favorable={r['favorable_count']}, unfavorable={r['unfavorable_count']}, ties={r['tie_count']}" for r in summaries],
        "Age, opponent rating, prior roles, statistic masks, coplay and organization/W20 gates already exist in actual C0.",
        "Attention uses16 maps/player,33 token fields, width32, tanh history projection; neither champion embeddings nor typed graph branches.",
        "Feature_distributions.csv retains both favorable and unfavorable EMEA and control cases, including missingness and reuse-weighted controls.",
        "No observed support association or omitted channel identifies a causal mechanism. Prior capacity/calibration/meta/family negatives are not reopened.",
        "Changed-ID lineage and announcement/issuance clocks remain unavailable; actual first-map rosters are conditional historical scenarios.",
        "Read cause_ledger.json for confirmedfact/evidence/hypothesis/negativecomparison and all named absences.",
    ]
    (out / "REPORT.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    reuse = Counter(control_ids)
    receipt = {"schema": SCHEMA, "status": "COMPLETE_DIAGNOSTIC_NO_IDENTIFIED_CAUSE", "run_id": protocol.get("run_id"),
               "coverage": {"native": len(cases), "Polishprimary": len(primary), "EMEA": int(cases.is_emea_primary.sum()),
                            "control_links": len(accounting), "matched_links": len(control_ids), "unique_controls": len(reuse),
                            "reused_links_beyond_first": len(control_ids) - len(reuse), "controls_reused": sum(v > 1 for v in reuse.values()),
                            "unmatched": int(accounting.control_golgg_match_id.isna().sum())},
               "control_reuse_counts": dict(reuse), "input_sha256": input_hashes, "source_sha256": source_hashes,
               "raw_trace": trace_receipt, "candidate_gate": ledger["candidate_gate_recommendation"],
               "output_schema": {"native_case_features.parquet": "3103 stable IDs plus all input-support diagnostics and nullable primary market comparisons",
                                 "primary_panel_preserved.parquet": "1467 original primary rows/all original columns plus symmetric losses",
                                 "emea_cases.parquet": "all89 EMEA, favorable/unfavorable/tie retained",
                                 "player_feature_support.parquet": "31030 exact native target/player slots",
                                 "player_identity_continuity.parquet": "EMEA plus unique controls,10 exact slots each; named trace absences",
                                 "team_identity_continuity.parquet": "two exact team IDs per EMEA/control case; retained same-team competition support or named trace absence",
                                 "*_state_preserved.parquet": "complete reconstruction player/team state rows when available; source schema retained",
                                 "array_differences_preserved.parquet": "complete strict reconstruction difference ledger when available; never suppress failed comparisons",
                                 "REPORT.txt": "human-readable coverage, outcome-symmetric score contributions, architecture and noncausal diagnosis bounds",
                                 "identity_transitions.parquet": "when trace exists: every retained raw membership for EMEA/controls, exact-ID team/role/competition transitions",
                                 "case_control_accounting.parquet": "89 links, outcome-blind reproduction, candidate counts, reuse, paired case features/losses",
                                 "feature_distributions.csv": "EMEA all and outcome-symmetric states versus unique and link-weighted controls; quantiles/missingness",
                                 "cause_ledger.json": "confirmedfact/evidence/hypothesis/negativecomparison; diagnosis and nonbinding parent-owned gate"},
               "outputs_sha256": {p.name: _hash(p) for p in out.iterdir() if p.is_file()}}
    _json(out / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run_diagnostics(args.protocol)
    print(json.dumps(_safe({"status": receipt["status"], "coverage": receipt["coverage"], "candidate_gate": receipt["candidate_gate"]}), indent=2))
    return 0
