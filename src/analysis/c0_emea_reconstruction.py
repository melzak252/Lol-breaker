"""Fresh, immutable C0 raw-history reconstruction and isolated frozen inference.

This driver deliberately retains the corrected bank's whole-series release rule,
original player slots and original feature arithmetic. First-map identities are a
conditional historical roster scenario, not proof of an authenticated announcement.
The CLI writes strict differences instead of repairing source bytes or tolerances.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SHAPES = {"players": (2, 5, 21), "core": (2, 5), "team": (2, 11),
              "w20": (2, 10), "gates": (2, 2)}
HIST_SHAPES = {"history": (2, 5, 16, 33), "mask": (2, 5, 16), "champions": (2, 5, 16)}
DTYPES = {**{key: np.dtype("float32") for key in RAW_SHAPES},
          "history": np.dtype("float16"), "mask": np.dtype("uint8"), "champions": np.dtype("uint16")}
MODES = ("full", "no_w20", "no_organization")
KINDS = ("full", "neural", "raw_neural")
SERIES_COUNT = 41915
MAP_COUNT = 69370
TARGET_COUNT = 3103


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _source_path(locator: str, research_root: Path) -> Path:
    if locator.startswith("research:"):
        return (research_root / locator.removeprefix("research:")).resolve()
    return _project(locator.removeprefix("project:"))


def _verify(path: Path, expected: str) -> str:
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"Frozen SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def _fresh(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Existing reconstruction namespace is forbidden, including empty directories: {path}")
    for path in paths:
        path.mkdir(parents=True, exist_ok=False)


def _paths(protocol: Mapping[str, Any]) -> dict[str, Path]:
    required = ("input_root", "feature_root", "output_root", "report_root", "raw_history",
                "native_control", "native_model_root", "selection", "control_links")
    values = protocol["paths"]
    result = {key: _project(values[key]) for key in required}
    for key in ("feature_root", "output_root", "report_root"):
        result[key] = result[key] / "tracks/reconstruction"
    return result


def _freeze(protocol_path: Path, protocol: Mapping[str, Any], paths: Mapping[str, Path], root: Path) -> dict[str, Any]:
    """Verify all declared pins and original inventories before any history replay."""
    from src.analysis import ev_asof_replay as replay
    from betting_app.core.models import native_c0

    hashes: dict[str, str] = {"protocol": _sha(protocol_path)}
    inventory = protocol.get("input_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Protocol must freeze nonempty input_sha256 before execution")
    for locator, expected in inventory.items():
        hashes[locator] = _verify(_source_path(locator, root), str(expected))
    binding_path = paths["input_root"] / "execution_binding.json"
    binding = _json(binding_path)
    if binding.get("schema") != "c0_emea_execution_binding_v1" or binding.get("protocol_sha256") != hashes["protocol"]:
        raise ValueError("Final implementation binding does not reference the unchanged frozen protocol")
    source_inventory = binding.get("source_sha256")
    if not isinstance(source_inventory, dict) or not source_inventory:
        raise ValueError("Final implementation binding must freeze source_sha256")
    hashes["execution_binding"] = _sha(binding_path)
    for locator, expected in source_inventory.items():
        hashes["implementation:" + locator] = _verify(_source_path(locator, root), str(expected))
    for relative in ("src/analysis/c0_emea_reconstruction.py", "scripts/research/run_c0_emea_reconstruction.py"):
        declared = source_inventory.get(relative, source_inventory.get("project:" + relative))
        if not declared:
            raise ValueError(f"Final implementation binding does not freeze reconstruction source: {relative}")
        hashes[relative] = _verify(PROJECT_ROOT / relative, declared)
    bank = root / protocol["paths"]["bank_relative"]
    hashes["bank_receipt"] = _verify(bank / "completed.json", replay.BANK_RECEIPT_SHA256)
    receipt = _json(bank / "completed.json")
    if (receipt["rows"], receipt["history_series"], receipt["history_maps"]) != (40636, SERIES_COUNT, MAP_COUNT):
        raise ValueError("Frozen bank receipt has unexpected full-history scope")
    hashes["canonical_raw_history"] = _verify(paths["raw_history"], replay.BANK_SOURCE_SHA256)
    hashes["original_builder"] = _verify(root / "a0-phase-walkforward-20260914/code/rebuild_corrected_bank.py", receipt["code_sha256"])
    for name, expected in receipt["outputs_sha256"].items():
        hashes["bank:" + name] = _verify(bank / name, expected)
    feature_root = root / replay.FEATURE_ARCHIVE_RELATIVE
    hashes["feature_manifest"] = _verify(feature_root / "manifest.json", replay.FEATURE_MANIFEST_SHA256)
    manifest = _json(feature_root / "manifest.json")
    # Rating-engine dependencies are imported lazily by TemporalState. Pin the
    # complete original Python code inventory, not just modules already imported.
    for relative, entry in manifest["files"].items():
        if relative.startswith("code/") and relative.endswith(".py"):
            hashes["research:" + str(replay.FEATURE_ARCHIVE_RELATIVE / relative)] = _verify(feature_root / relative, entry["sha256"])
    # The live checkout contains unrelated, newer rating studies. Verify the
    # actual selected rating dependencies after import; archived files remain
    # fully pinned above. Do not require unused live modules to match the archive.
    hashes["native_execution"] = _verify(paths["native_model_root"] / "execution.json", native_c0._EXECUTION_SHA256)
    hashes["native_completion"] = _verify(_project(native_c0._COMPLETION_RELATIVE), native_c0._COMPLETION_SHA256)
    for locator, expected in {**native_c0._ARTIFACT_SHA256, **native_c0._SERVING_SOURCE_SHA256}.items():
        hashes[locator] = _verify(_source_path(locator, root), expected)
    for locator, expected in native_c0._SOURCE_SHA256.items():
        if locator.startswith("research:"):
            hashes[locator] = _verify(_source_path(locator, root), expected)
    for name in ("selection", "control_links", "native_control"):
        path = paths[name]
        declared = protocol["input_sha256"].get(str(path.relative_to(PROJECT_ROOT)))
        if declared is None:
            declared = protocol["input_sha256"].get("project:" + str(path.relative_to(PROJECT_ROOT)))
        if declared is None:
            raise ValueError(f"Protocol does not pin {name}")
        hashes[name] = _verify(path, declared)
    binding = paths["input_root"] / "oracle_binding.json"
    if not binding.is_file():
        raise FileNotFoundError("Independent oracle binding must be frozen before reconstruction execution")
    oracle_binding = _json(binding)
    if (oracle_binding.get("schema") != "c0_emea_oracle_binding_v1"
            or oracle_binding.get("protocol_sha256") != hashes["protocol"]):
        raise ValueError("Independent oracle binding must reference the frozen protocol")
    hashes["oracle_binding"] = _sha(binding)
    hashes["oracle_source"] = _verify(binding.with_name("verify_raw_inputs.py"),
                                      oracle_binding["source_sha256"])
    hashes["oracle_methods"] = _verify(binding.with_name("oracle_methods.json"),
                                       oracle_binding["methods_sha256"])
    return {"schema": "c0_emea_reconstruction_pins_v1", "run_id": protocol["run_id"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashes,
            "array_rule": "original dtypes; exact bits; signed zero differences retained",
            "probability_absolute_tolerance": float(protocol.get("verification", {}).get("native_probability_absolute", 2e-6)),
            "release_policy": "max(latest original source game day, frozen release override), whole series, strict < bank day",
            "roster_status": "conditional actual first-map identities; no authenticated issuance claim"}


def _verify_feature_imports(root: Path) -> dict[str, str]:
    """Reject late-imported shadow rating/history helpers before any updates."""
    archive = root / "unified-ratings-20260911"
    files = _json(archive / "manifest.json")["files"]
    hashes = {}
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith("scripts.") or name == "src.ratings" or name.startswith("src.ratings.")):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if path.is_relative_to(archive):
            relative = path.relative_to(archive).as_posix()
        elif path.is_relative_to(PROJECT_ROOT):
            relative = "code/" + path.relative_to(PROJECT_ROOT).as_posix()
        else:
            raise ImportError(f"Feature module escaped the verified archive/workspace closure: {name}: {path}")
        record = files.get(relative)
        if not isinstance(record, dict):
            raise ImportError(f"Feature module is not covered by the original manifest: {name}: {path}")
        hashes[f"import:{name}"] = _verify(path, record["sha256"])
    return hashes


def _targets(paths: Mapping[str, Path], bank: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    native = pd.read_parquet(paths["native_control"])
    selection = pd.read_parquet(paths["selection"])
    bank_metadata = pd.read_parquet(bank / "metadata.parquet")
    original = pd.read_json(bank / "original_metadata.jsonl", lines=True, convert_dates=False,
                            dtype={"golgg_match_id": str, "team1_id": str, "team2_id": str})
    for frame in (native, selection, bank_metadata, original):
        frame["golgg_match_id"] = frame.golgg_match_id.astype(str)
        if frame.golgg_match_id.duplicated().any():
            raise ValueError("Duplicate frozen target ID")
    if len(native) != TARGET_COUNT or selection.golgg_match_id.tolist() != native.golgg_match_id.tolist():
        raise ValueError("Selection must contain all3103 native IDs in unchanged native order")
    if not pd.to_datetime(native.date).is_monotonic_increasing:
        raise ValueError("Native target order is not chronological")
    markers = ("is_emea_primary", "is_polish_primary", "is_matched_control")
    if any(key not in selection for key in markers):
        raise ValueError("Selection omits required EMEA/Polish/control group markers")
    if int(selection.is_emea_primary.sum()) != 89 or int(selection.is_polish_primary.sum()) != 1467:
        raise ValueError("Selection reduced the frozen89EMEA/1467Polish scope")
    positions = pd.Series(np.arange(len(bank_metadata)), index=bank_metadata.golgg_match_id)
    indices = positions.loc[native.golgg_match_id].to_numpy(dtype=np.int64)
    if "bank_row_index" not in selection or not np.array_equal(indices, selection.bank_row_index.to_numpy(dtype=np.int64)):
        raise ValueError("Selection bank-row identity differs from the frozen bank")
    frozen_rows = original.iloc[indices].reset_index(drop=True)
    for key in ("date", "team1_id", "team2_id", "best_of", "y", "tournament_name", "roster_a", "roster_b"):
        for i, (old, saved, selected) in enumerate(zip(frozen_rows[key], native[key], selection[key] if key in selection else native[key])):
            values = [list(map(str, v)) if key.startswith("roster_") else str(v) for v in (old, saved, selected)]
            if values[0] != values[1] or values[0] != values[2]:
                raise ValueError(f"Frozen target identity mismatch: row={i}, field={key}")
    targets = native[[key for key in native if not key.startswith("p_")]].copy()
    for key in selection:
        if key not in targets:
            targets[key] = selection[key].to_numpy()
    targets["target_index"] = np.arange(TARGET_COUNT, dtype=np.int64)
    targets["cutoff_day"] = targets.date.astype(str)
    for key in ("roster_a", "roster_b"):
        targets[key] = [list(map(str, roster)) for roster in frozen_rows[key]]
        if any(roster != sorted(roster) or len(roster) != 5 or len(set(roster)) != 5 for roster in targets[key]):
            raise ValueError("Frozen original tensor slots are not canonical distinct sorted IDs")
    links = pd.read_csv(paths["control_links"], dtype=str)
    if len(links) != 89 or set(links.emea_golgg_match_id) != set(targets.loc[targets.is_emea_primary, "golgg_match_id"]):
        raise ValueError("Control links do not cover every frozen EMEA target")
    linked_controls = set(links.control_golgg_match_id.dropna())
    if not linked_controls.issubset(set(targets.loc[targets.is_matched_control, "golgg_match_id"])):
        raise ValueError("Control selection markers omit a frozen matched-control link")
    return targets, native


class _ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.count = 0

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if rows:
            self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
            self.count += len(rows)

    def close(self) -> None:
        self.writer.close()


def _membership_schema() -> pa.Schema:
    strings = ("golgg_match_id", "player_id", "source_series_id", "source_map_id", "played_day",
               "effective_release_day", "source_team_id", "opponent_id", "source_tournament_name",
               "role", "patch", "champion_id", "cutoff_day")
    integers = ("target_index", "side", "slot", "map_index", "history_index", "token_index", "age_days", "champion_code")
    return pa.schema([(key, pa.string()) for key in strings] + [(key, pa.int64()) for key in integers] +
                     [("win", pa.float64()), ("opponent_prior_rating", pa.float64()),
                      ("raw_stats", pa.list_(pa.float64())), ("raw_stat_presence", pa.list_(pa.bool_())),
                      ("opponent_prior_player_ids", pa.list_(pa.string())),
                      ("opponent_prior_player_ratings", pa.list_(pa.float64()))])


def _capture(raw: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(raw["match_id"])
    maps = []
    for index, game in enumerate(raw["games"]):
        champions = {}
        for side in ("t1", "t2"):
            for player in game[side + "_players"].values():
                pid = str(player.get("player_id", player.get("id")))
                champions[pid] = str(player["champion_id"]) if player.get("champion_id") is not None else ""
        maps.append({"source_series_id": sid, "source_map_id": str(game["game_id"]), "map_index": index,
                     "played_day": str(game.get("date", raw["date"])), "player_champions": champions})
    return {"source_series_id": sid, "start_day": str(raw["date"]),
            "source_tournament_name": str(raw.get("tournament_name", "")), "maps": maps}


def _collect(raw_path: Path, releases: Mapping[str, str], backend: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from src.analysis import ev_asof_replay as replay

    sources: dict[str, Any] = {}
    def capture():
        for raw in backend["iter_matches"](raw_path):
            record = _capture(raw)
            sid = record["source_series_id"]
            if sid in sources:
                raise ValueError(f"Duplicate source series ID {sid}")
            sources[sid] = record
            yield raw
    series, metadata, extra, audit = backend["collect_history"](capture())
    if len(series) != SERIES_COUNT or sum(len(item.games) for item in series) != MAP_COUNT:
        raise ValueError("Canonical raw stream did not reconstruct all41915series/69370maps")
    corrected = []
    for item in series:
        source = sources[item.identifier]
        day = replay._corrected_series_day(item, releases)
        source.update(effective_release_day=day.isoformat(), team1_id=item.team1, team2_id=item.team2,
                      best_of=item.best_of, latest_played_day=max(game.day for game in item.games).isoformat())
        for game, record in zip(item.games, source["maps"], strict=True):
            record.update(effective_release_day=day.isoformat(), team1_id=item.team1, team2_id=item.team2,
                          roster_a=list(game.players1), roster_b=list(game.players2),
                          w20_valid_a=game.history1 is not None, w20_valid_b=game.history2 is not None)
        if day != item.games[-1].day:
            item = replace(item, games=[*item.games[:-1], replace(item.games[-1], day=day)])
        corrected.append(item)
    corrected.sort(key=lambda item: (item.games[-1].day, item.order))
    for index, item in enumerate(corrected):
        sources[item.identifier]["source_order"] = index
    return corrected, metadata, extra, sources, audit


def _prior_opponents(state: Any, item: Any) -> dict[str, tuple[list[str], list[float], float]]:
    system = state.base.base.manager.systems["gl"]
    result = {}
    for team, players in ((item.team1, item.games[0].players2), (item.team2, item.games[0].players1)):
        ratings = [float((system.player_ratings[p] if p in system.player_ratings else system.player_ratings.default_factory()).rating) for p in players]
        result[team] = (list(players), ratings, float(np.mean(ratings)))
    return result


def _consume(state: Any, item: Any, metadata: Mapping[str, Any], extra: Sequence[Any], source: Mapping[str, Any],
             histories: dict[str, deque], team_histories: dict[str, deque]) -> None:
    # Read source identities and opponent inputs before the original update. Do
    # not call get_player_rating here: reading a cold ID must not insert it.
    opponents = _prior_opponents(state, item)
    entries = []
    for index, game in enumerate(item.games):
        raw_map = source["maps"][index]
        raw_map["consumed"] = True
        raw_map["opponent_prior_a"] = opponents[item.team1][2]
        raw_map["opponent_prior_b"] = opponents[item.team2][2]
        for team, roster, win, valid in ((item.team1, game.players1, game.score, game.history1 is not None),
                                        (item.team2, game.players2, 1 - game.score, game.history2 is not None)):
            other = item.team2 if team == item.team1 else item.team1
            ids, ratings, mean = opponents[team]
            for pid in roster:
                values, role, _ = extra[index]["players"].get(pid, (np.zeros(24, dtype=np.float32), "UNKNOWN", ""))
                entries.append((pid, {"source_series_id": item.identifier, "source_map_id": raw_map["source_map_id"],
                    "map_index": index, "played_day": raw_map["played_day"],
                    "effective_release_day": source["effective_release_day"], "source_team_id": team,
                    "opponent_id": other, "source_tournament_name": source["source_tournament_name"],
                    "role": role, "patch": extra[index]["patch"], "champion_id": raw_map["player_champions"].get(pid, ""),
                    "raw_stats": values[:12].astype(float).tolist(), "raw_stat_presence": values[12:].astype(bool).tolist(),
                    "opponent_prior_rating": mean, "opponent_prior_player_ids": ids,
                    "opponent_prior_player_ratings": ratings, "win": float(win)}))
            team_histories[team].append({"source_map_id": raw_map["source_map_id"], "valid": valid})
    state.consume(item, metadata, extra)
    for pid, entry in entries:
        entry["champion_code"] = int(state.vocabulary[entry["champion_id"]])
        histories[pid].append(entry)
    source["consumed"] = True


def _extract(state: Any, target: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    cutoff = date.fromisoformat(str(target["date"]))
    pair, details = state.base.pair(str(target["team1_id"]), list(target["roster_a"]),
                                    str(target["team2_id"]), list(target["roster_b"]), cutoff, int(target["best_of"]))
    arrays = {key: np.asarray(pair[key], dtype=DTYPES[key]) for key in RAW_SHAPES}
    player_rows = [[state.player(pid, cutoff) for pid in target[key]] for key in ("roster_a", "roster_b")]
    for key in HIST_SHAPES:
        arrays[key] = np.asarray([[player[key] for player in side] for side in player_rows], dtype=DTYPES[key])
    return arrays, details


def _decayed(entry: Any, cutoff: date) -> float:
    return 0.0 if entry is None else float(entry[0] * math.exp(-(cutoff - entry[1]).days / 180.0))


def _state_rows(state: Any, target: Mapping[str, Any], arrays: Mapping[str, np.ndarray], details: Sequence[Mapping[str, Any]],
                histories: Mapping[str, deque], team_histories: Mapping[str, deque], sources: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = date.fromisoformat(target["date"])
    base = state.base.base
    common = {"golgg_match_id": target["golgg_match_id"], "target_index": int(target["target_index"]), "cutoff_day": target["date"]}
    players, teams, membership = [], [], []
    for side, (team, key) in enumerate(((target["team1_id"], "roster_a"), (target["team2_id"], "roster_b"))):
        roster = target[key]
        pairs = list(combinations(roster, 2))
        counts = [_decayed(state.base.co_play.get(pair), cutoff) for pair in pairs]
        latest = base.latest_rosters.get(team)
        latest_map = sources[str(latest[0][1][1])]["maps"][latest[0][2]] if latest else None
        w20 = list(team_histories.get(team, []))
        teams.append({**common, "side": side, "team_id": team, "roster": roster,
            "team_games": int(base.team_games[team]), "team_series": int(base.team_series[team]),
            "unknown_players": int(details[side]["unknown_players"]), "min_player_series": int(details[side]["min_player_series"]),
            "roster_overlap": float(details[side]["roster_overlap"]), "organization_gate": float(arrays["gates"][side, 0]),
            "w20_gate": float(arrays["gates"][side, 1]), "w20_window_games": len(w20),
            "w20_valid_games": int(details[side]["w20_valid_games"]), "co_play_mean": float(np.mean(counts)),
            "pair_player_ids": [list(pair) for pair in pairs], "pair_counts": counts,
            "pair_last_days": [state.base.co_play[pair][1].isoformat() if pair in state.base.co_play else None for pair in pairs],
            "exact_roster_games": _decayed(state.base.roster_play.get(tuple(roster)), cutoff),
            "latest_roster": list(latest[1]) if latest else [], "latest_roster_day": latest[0][0].isoformat() if latest else None,
            "latest_roster_series_id": latest_map["source_series_id"] if latest_map else None,
            "latest_roster_map_id": latest_map["source_map_id"] if latest_map else None,
            "w20_source_map_ids": [record["source_map_id"] for record in w20], "w20_valid": [record["valid"] for record in w20],
            "team_vector": arrays["team"][side].astype(float).tolist(), "w20_vector": arrays["w20"][side].astype(float).tolist()})
        for slot, pid in enumerate(roster):
            retained = list(histories.get(pid, []))
            actual = list(state.histories.get(pid, []))
            if len(retained) != len(actual):
                raise ValueError("Source provenance and original temporal history lengths diverged")
            vector = arrays["players"][side, slot].astype(float).tolist()
            players.append({**common, "side": side, "slot": slot, "team_id": team, "player_id": pid,
                "player_games": int(base.player_games[pid]), "player_series": int(base.player_series[pid]),
                "last_release_day": state.base.player_last[pid].isoformat() if pid in state.base.player_last else None,
                "role": state.base.roles.get(pid, "UNKNOWN"), "rating_vector": vector[:11], "player_vector": vector,
                "retained_history_maps": len(retained), "selected_history_maps": min(16, len(retained))})
            start = max(0, len(retained) - 16)
            for index, (record, original) in enumerate(zip(retained, actual, strict=True)):
                if original["day"].isoformat() != record["effective_release_day"] or float(original["opponent"]) != record["opponent_prior_rating"]:
                    raise ValueError("Trace differs from the actual pre-update opponent/release inputs")
                membership.append({**common, **record, "side": side, "slot": slot, "player_id": pid,
                    "history_index": index, "token_index": index - start if index >= start else None,
                    "age_days": (cutoff - date.fromisoformat(record["effective_release_day"])).days})
    return players, teams, membership


def _bits(array: np.ndarray) -> np.ndarray:
    return array.view(np.dtype(f"u{array.dtype.itemsize}")) if array.dtype.kind == "f" else array


def compare_arrays(arrays: Mapping[str, np.ndarray], bank: Path, targets: pd.DataFrame, report: Path) -> dict[str, Any]:
    """Retain every changed scalar, including its original-dtype bit representation."""
    schema = pa.schema([("golgg_match_id", pa.string()), ("target_index", pa.int64()), ("bank_row_index", pa.int64()),
                        ("array", pa.string()), ("coordinate", pa.list_(pa.int64())), ("actual", pa.float64()),
                        ("saved", pa.float64()), ("actual_bits", pa.uint64()), ("saved_bits", pa.uint64())])
    sink = _ParquetSink(report / "array_differences.parquet", schema)
    summary = {}
    try:
        for key, value in arrays.items():
            saved = np.load(bank / f"{key}.npy", mmap_mode="r")
            if value.dtype != DTYPES[key] or saved.dtype != value.dtype or value.shape[1:] != saved.shape[1:]:
                raise ValueError(f"Frozen shape/dtype disagreement for {key}")
            changed_rows, changed_values, max_abs = 0, 0, 0.0
            for index, bank_index in enumerate(targets.bank_row_index):
                actual = value[index]
                old = saved[int(bank_index)]
                actual_bits, old_bits = _bits(actual), _bits(old)
                coordinates = np.argwhere(actual_bits != old_bits)
                if len(coordinates):
                    changed_rows += 1
                    changed_values += len(coordinates)
                    max_abs = max(max_abs, float(np.max(np.abs(actual.astype(float) - old.astype(float)))))
                    sink.append([{"golgg_match_id": targets.iloc[index].golgg_match_id, "target_index": index,
                        "bank_row_index": int(bank_index), "array": key, "coordinate": coordinate.tolist(),
                        "actual": float(actual[tuple(coordinate)]), "saved": float(old[tuple(coordinate)]),
                        "actual_bits": int(actual_bits[tuple(coordinate)]), "saved_bits": int(old_bits[tuple(coordinate)])}
                        for coordinate in coordinates])
            summary[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "rows_changed": changed_rows,
                            "values_changed": changed_values, "max_absolute_difference": max_abs, "bit_exact": changed_values == 0}
    finally:
        sink.close()
    return {"bit_exact": all(record["bit_exact"] for record in summary.values()), "arrays": summary,
            "comparison": "by frozen target ID and original participant slot; no tolerance"}


def _input_digests(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key, value in arrays.items()}


def _logical_marker(state: Any) -> str:
    """Hash logical state incrementally; caches are deliberately not model state."""
    base = state.base.base
    digest = hashlib.sha256()
    def add(value: Any) -> None:
        digest.update(repr(value).encode())
        digest.update(b"\0")
    for name, system in sorted(base.manager.systems.items()):
        add(name)
        for kind in ("player_ratings", "team_ratings", "player_last_played", "team_last_played"):
            add(kind)
            for identity, value in sorted(getattr(system, kind, {}).items()):
                add(identity)
                numeric = tuple((field, float(getattr(value, field))) for field in
                                ("rating", "rd", "vol", "mu", "sigma") if hasattr(value, field))
                add(numeric if numeric else value)
    for values in (base.last_order, base.last_day, dict(base.team_games), dict(base.team_series),
                   dict(base.player_games), dict(base.player_series), state.base.co_play,
                   state.base.roster_play, state.base.roles, state.base.player_last,
                   base.latest_rosters, state.vocabulary, list(state.global_champions)):
        add(values)
    for identity, rows in sorted(state.histories.items()):
        add(identity)
        for row in rows:
            add((row["day"], float(row["opponent"]), row["role"], row["champion"], row["win"],
                 row["friends"], row["opponents"], row["patch"]))
            digest.update(row["values"].tobytes())
    for identity, rows in sorted(base.histories.items()):
        add((identity, list(rows)))
    return digest.hexdigest()


def feature_smoke(state: Any, target: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Actual unchanged feature queries for cold identities and partial new rosters."""
    before = _logical_marker(state)
    normal, _ = _extract(state, target)
    cold_target = {**target, "team1_id": "c0-smoke-unseen-team-a", "team2_id": "c0-smoke-unseen-team-b",
                   "roster_a": [f"c0-smoke-unseen-a-{i}" for i in range(5)],
                   "roster_b": [f"c0-smoke-unseen-b-{i}" for i in range(5)]}
    cold, _ = _extract(state, cold_target)
    new_target = {**target, "roster_a": sorted([*target["roster_a"][:4], "c0-smoke-unseen-a-0"]),
                  "roster_b": sorted([*target["roster_b"][:4], "c0-smoke-unseen-b-0"])}
    new, _ = _extract(state, new_target)
    repeated, _ = _extract(state, target)
    unchanged = before == _logical_marker(state)
    repeat = _input_digests(normal) == _input_digests(repeated)
    payload = {f"{case}__{key}": array[None] for case, arrays in (("cold", cold), ("new_roster", new)) for key, array in arrays.items()}
    return payload, {"target_id": target["golgg_match_id"], "query_nonmutation": unchanged,
                     "repeat_bit_exact": repeat, "cold_unknown_players": int(cold["players"][..., 14].sum()),
                     "cold_temporal_maps": int(cold["mask"].sum()),
                     "new_roster_unknown_players": int(new["players"][..., 14].sum()),
                     "passed": unchanged and repeat and int(cold["mask"].sum()) == 0}


def future_target_smoke(backend: Mapping[str, Any], series: Sequence[Any], metadata: Mapping[str, Any], extra: Mapping[str, Any],
                        target: Mapping[str, Any], expected: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Replay a fresh state after perturbing a representative unreleased target.

    The perturbation changes future normalized outcomes, W20 and temporal raw
    stats; exact identities, source/release dates and cohort are left unchanged.
    This is a feature-causality smoke, not a source-validation repair experiment.
    """
    cutoff = date.fromisoformat(target["date"])
    future = next(item for item in series if item.identifier == target["golgg_match_id"])
    if future.games[-1].day < cutoff:
        raise ValueError("Representative target is already released before its own bank cutoff")
    changed = replace(future, games=[replace(game, score=1 - game.score,
        history1={key: value + 123.0 for key, value in game.history1.items()} if game.history1 else None,
        history2={key: value - 123.0 for key, value in game.history2.items()} if game.history2 else None) for game in future.games])
    changed_extra = [{**game, "players": {pid: (values + np.float32(123), role, champion)
                     for pid, (values, role, champion) in game["players"].items()}} for game in extra[future.identifier]]
    state = backend["TemporalState"]()
    state.vocabulary = dict(backend["champion_vocabulary"])
    consumed = 0
    for original in series:
        item = changed if original.identifier == future.identifier else original
        if item.games[-1].day >= cutoff:
            break
        state.consume(item, metadata[item.identifier], changed_extra if item.identifier == future.identifier else extra[item.identifier])
        consumed += 1
    actual, _ = _extract(state, target)
    exact = _input_digests(actual) == _input_digests(expected)
    return {"target_id": target["golgg_match_id"], "target_release_day": future.games[-1].day.isoformat(),
            "cutoff_day": cutoff.isoformat(), "replayed_prefix_series": consumed,
            "perturbation": "future normalized target map outcomes, W20 values and temporal raw stats",
            "target_consumed": False, "eight_arrays_bit_exact": exact, "passed": exact}


def reconstruct(protocol_path: Path) -> dict[str, Any]:
    from src.analysis import ev_asof_replay as replay

    protocol_path = _project(protocol_path)
    protocol = _json(protocol_path)
    paths = _paths(protocol)
    root = replay._resolve_research_root(None)
    # Reject namespaces as a group before making any of them, including empty ones.
    _fresh([paths[key] for key in ("feature_root", "output_root", "report_root")])
    pins = _freeze(protocol_path, protocol, paths, root)
    _save_json(paths["report_root"] / "pins.json", pins)
    bank = root / protocol["paths"]["bank_relative"]
    targets, native = _targets(paths, bank)
    backend, feature_hashes = replay._activate_feature_backend(root)
    releases = _json(bank / "release_days.json")
    series, metadata, extra, sources, history_audit = _collect(paths["raw_history"], releases, backend)
    state = backend["TemporalState"]()
    state.vocabulary = dict(backend["champion_vocabulary"])
    feature_hashes.update(_verify_feature_imports(root))
    arrays = {key: np.zeros((TARGET_COUNT, *shape), dtype=DTYPES[key]) for key, shape in {**RAW_SHAPES, **HIST_SHAPES}.items()}
    histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=256))
    team_histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    membership = _ParquetSink(paths["report_root"] / "history_membership.parquet", _membership_schema())
    player_rows, team_rows, target_rows = [], [], []
    cursor = 0
    representative = int(np.flatnonzero(targets.is_emea_primary.to_numpy())[0])
    smoke_payload = None
    feature_checks = None
    try:
        for target in targets.to_dict("records"):
            cutoff = date.fromisoformat(target["date"])
            while cursor < len(series) and series[cursor].games[-1].day < cutoff:
                item = series[cursor]
                _consume(state, item, metadata[item.identifier], extra[item.identifier], sources[item.identifier], histories, team_histories)
                cursor += 1
            values, details = _extract(state, target)
            index = int(target["target_index"])
            for key, value in values.items():
                arrays[key][index] = value
            players, teams, members = _state_rows(state, target, values, details, histories, team_histories, sources)
            player_rows.extend(players)
            team_rows.extend(teams)
            membership.append(members)
            last_day = state.base.base.last_day.isoformat() if state.base.base.last_day else None
            if last_day is not None and last_day >= target["date"]:
                raise ValueError("History crossed the target-day strict causal boundary")
            target_rows.append({**target, "feature_source_max_day": last_day, "history_series_consumed": cursor,
                "min_player_series": min(row["min_player_series"] for row in details),
                "unknown_players": sum(row["unknown_players"] for row in details),
                "new_organization": any(row["team_prior_games"] == 0 for row in details),
                "co_play_min_mean": min(row["co_play_mean"] for row in details),
                "w20_min_gate": float(values["gates"][:, 1].min()),
                "w20_min_valid_games": min(row["w20_valid_games"] for row in details),
                "tensor_player_order_a": target["roster_a"], "tensor_player_order_b": target["roster_b"],
                "feature_available": True, "roster_authentication": "conditional first-map identity scenario only"})
            if index == representative:
                smoke_payload, feature_checks = feature_smoke(state, target)
    finally:
        membership.close()
    targets = pd.DataFrame(target_rows)
    np.savez_compressed(paths["feature_root"] / "reconstructed_arrays.npz", **arrays)
    targets.to_parquet(paths["feature_root"] / "targets.parquet", index=False)
    pd.DataFrame(player_rows).to_parquet(paths["report_root"] / "player_state.parquet", index=False)
    pd.DataFrame(team_rows).to_parquet(paths["report_root"] / "team_state.parquet", index=False)
    source_rows, map_rows = [], []
    for item in series:
        source = sources[item.identifier]
        source_rows.append({key: value for key, value in source.items() if key != "maps"} | {"consumed": source.get("consumed", False), "map_count": len(item.games)})
        for record in source["maps"]:
            map_rows.append({key: value for key, value in record.items() if key != "player_champions"} |
                            {"consumed": record.get("consumed", False), "source_order": source["source_order"],
                             "source_tournament_name": source["source_tournament_name"]})
    pd.DataFrame(source_rows).to_parquet(paths["report_root"] / "source_series.parquet", index=False)
    pd.DataFrame(map_rows).to_parquet(paths["report_root"] / "source_maps.parquet", index=False)
    array_comparison = compare_arrays(arrays, bank, targets, paths["report_root"])
    _save_json(paths["report_root"] / "array_comparison.json", array_comparison)
    future_checks = future_target_smoke(backend, series, metadata, extra, targets.iloc[representative].to_dict(),
                                        {key: value[representative] for key, value in arrays.items()})
    np.savez_compressed(paths["feature_root"] / "smoke_arrays.npz", **smoke_payload)
    _save_json(paths["report_root"] / "feature_smoke.json", {"query": feature_checks, "future_target": future_checks})
    _save_json(paths["report_root"] / "snapshot_inputs.json",
               {name: _sha(paths["feature_root"] / name) for name in
                ("reconstructed_arrays.npz", "targets.parquet", "smoke_arrays.npz")})
    # Model loading occurs in a fresh subprocess, never in the raw feature import namespace.
    env = os.environ.copy()
    env["BETTING_APP_C0_WORKER"] = "1"
    env["C0_RESEARCH_ROOT"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("C0_MODEL_ROOT", "C0_COMPLETION_REPORT", "C0_SERVING_CONFIG"):
        env.pop(key, None)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/research/run_c0_emea_reconstruction.py"),
                    "--protocol", str(protocol_path), "--native-worker"], cwd=PROJECT_ROOT, env=env, check=True)
    native_receipt = _json(paths["report_root"] / "native_receipt.json")
    receipt = {"schema": "c0_emea_reconstruction_receipt_v1", "run_id": protocol["run_id"],
        "status": "PARITY_CONFIRMED" if array_comparison["bit_exact"] and native_receipt["probability_parity"]
                  and feature_checks["passed"] and future_checks["passed"] and native_receipt["smoke"]["passed"] else "DIFFERENCES_PRESERVED",
        "targets": len(targets), "all_emea": int(targets.is_emea_primary.sum()),
        "all_polish": int(targets.is_polish_primary.sum()), "unique_matched_controls": int(targets.is_matched_control.sum()),
        "raw_history_series": len(series), "raw_history_maps": sum(len(item.games) for item in series),
        "consumed_series": cursor, "history_membership_rows": membership.count,
        "history_membership_semantics": "all retained256 maps per original player slot; last16 token_index; older rating/support sources derive exactly from source_maps and target consumed cursor",
        "array_comparison": array_comparison, "native_replay": native_receipt, "feature_smoke": feature_checks,
        "future_target_smoke": future_checks, "feature_source_sha256": feature_hashes, "history_audit": history_audit,
        "fits": 0, "source_or_model_mutations": 0, "tolerance_changes": 0,
        "diagnosis": "Exact differences retained for independent oracle/outcome-symmetric diagnosis; no repair inferred from negative parity alone.",
        "outputs_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha(path) for key in ("feature_root", "output_root", "report_root")
                           for path in paths[key].iterdir() if path.is_file()}}
    _save_json(paths["report_root"] / "receipt.json", receipt)
    return receipt


def _predict_smoke(predict: Any, arrays: Mapping[str, np.ndarray], best_of: np.ndarray, smoke_arrays: Mapping[str, np.ndarray],
                   tolerance: float) -> dict[str, Any]:
    # Exercise actual BO1/3/5 rows, all nine outputs, rather than a constant stub.
    indices = [int(np.flatnonzero(best_of == value)[0]) for value in (1, 3, 5)]
    sample = {key: value[indices].copy() for key, value in arrays.items()}
    formats = best_of[indices]
    before = _input_digests(sample)
    normal = predict(sample, formats)
    repeat = predict(sample, formats)
    swapped = predict({key: value[:, ::-1].copy() for key, value in sample.items()}, formats)
    zero = predict({key: np.zeros_like(value) for key, value in sample.items()}, formats)
    empty = {key: np.zeros_like(value) if key in HIST_SHAPES else value.copy() for key, value in sample.items()}
    empty_prediction = predict(empty, formats)
    cases = {}
    for case in ("cold", "new_roster"):
        query = {key: np.repeat(smoke_arrays[f"{case}__{key}"], 3, axis=0) for key in arrays}
        prediction = predict(query, np.asarray([1, 3, 5], dtype=np.int64))
        cases[case] = {"finite": all(np.isfinite(value).all() for value in prediction.values()),
                       "probabilities": {key: value.tolist() for key, value in prediction.items()}}
    swap_error = max(float(np.max(np.abs(normal[key] + swapped[key] - 1))) for key in KINDS)
    cold_error = max(float(np.max(np.abs(value - .5))) for value in zero.values())
    repeatable = all(np.array_equal(normal[key], repeat[key]) for key in KINDS)
    nonmutation = _input_digests(sample) == before
    finite_empty = all(np.isfinite(value).all() for value in empty_prediction.values())
    return {"best_of": [1, 3, 5], "modes": list(MODES), "kinds": list(KINDS), "swap_max_error": swap_error,
            "zero_cold_max_error": cold_error, "repeat_bit_exact": repeatable, "input_nonmutation": nonmutation,
            "empty_history_finite": finite_empty, "actual_identity_queries": cases,
            "passed": swap_error <= tolerance and cold_error <= tolerance and repeatable and nonmutation
                      and finite_empty and all(case["finite"] for case in cases.values())}


def native_worker(protocol_path: Path) -> dict[str, Any]:
    """Only the isolated process activates the existing verified C0 import closure."""
    if os.environ.get("BETTING_APP_C0_WORKER") != "1":
        raise RuntimeError("Native replay must run in its isolated worker")
    protocol = _json(_project(protocol_path))
    paths = _paths(protocol)
    pins = _json(paths["report_root"] / "pins.json")
    if _sha(_project(protocol_path)) != pins["sha256"]["protocol"]:
        raise ValueError("Frozen protocol changed between reconstruction and isolated inference")
    for path in (paths["output_root"] / "probabilities.parquet",
                 paths["report_root"] / "probability_comparisons.parquet",
                 paths["report_root"] / "native_receipt.json"):
        if path.exists():
            raise FileExistsError(f"Existing native replay output is forbidden: {path}")
    for name, expected in _json(paths["report_root"] / "snapshot_inputs.json").items():
        _verify(paths["feature_root"] / name, expected)
    for relative in ("src/analysis/c0_emea_reconstruction.py", "scripts/research/run_c0_emea_reconstruction.py"):
        _verify(PROJECT_ROOT / relative, pins["sha256"][relative])
    _verify(paths["native_control"], pins["sha256"]["native_control"])
    from betting_app.core.models.native_c0 import NativeC0Adapter
    from src.models import a0_data_focus

    adapter = NativeC0Adapter()
    identity = adapter.load()
    with np.load(paths["feature_root"] / "reconstructed_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if set(arrays) != set(DTYPES):
        raise ValueError("Native worker requires the exact original eight arrays")
    targets = pd.read_parquet(paths["feature_root"] / "targets.parquet")
    native = pd.read_parquet(paths["native_control"])
    native.golgg_match_id = native.golgg_match_id.astype(str)
    native = native.set_index("golgg_match_id").loc[targets.golgg_match_id].reset_index()
    best_of = targets.best_of.to_numpy(dtype=np.int64)
    def predict(values, formats):
        return a0_data_focus.predict_models(adapter._backend, adapter._shared, adapter._models, adapter._calibration,
            {key: values[key] for key in RAW_SHAPES}, {key: values[key] for key in HIST_SHAPES}, formats)
    digests = _input_digests(arrays)
    predictions = predict(arrays, best_of)
    if _input_digests(arrays) != digests:
        raise ValueError("Frozen C0 mutated reconstructed input arrays")
    output = targets.copy()
    comparisons = []
    tolerance = float(pins["probability_absolute_tolerance"])
    for kind in KINDS:
        values = np.asarray(predictions[kind], dtype=float)
        if values.shape != (TARGET_COUNT, 3) or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError(f"Frozen C0 returned invalid {kind} probabilities")
        for mode_index, mode in enumerate(MODES):
            key = f"p_{kind}_{mode}"
            output[key] = values[:, mode_index]
            saved = native[key].to_numpy(dtype=float)
            error = np.abs(values[:, mode_index] - saved)
            comparisons.extend({"golgg_match_id": str(targets.iloc[index].golgg_match_id), "target_index": index,
                "kind": kind, "mode": mode, "actual_probability": float(values[index, mode_index]),
                "saved_probability": float(saved[index]), "absolute_difference": float(error[index]),
                "within_frozen_tolerance": bool(error[index] <= tolerance)} for index in range(TARGET_COUNT))
    output.to_parquet(paths["output_root"] / "probabilities.parquet", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_parquet(paths["report_root"] / "probability_comparisons.parquet", index=False)
    with np.load(paths["feature_root"] / "smoke_arrays.npz", allow_pickle=False) as archive:
        smoke_arrays = {key: archive[key] for key in archive.files}
    smoke = _predict_smoke(predict, arrays, best_of, smoke_arrays, tolerance)
    receipt = {"schema": "c0_emea_native_replay_v1", "rows": TARGET_COUNT, "comparison_rows": len(comparison),
        "modes": list(MODES), "kinds": list(KINDS), "probability_parity": bool(comparison.within_frozen_tolerance.all()),
        "maximum_absolute_difference": float(comparison.absolute_difference.max()), "absolute_tolerance": tolerance,
        "differences_above_tolerance": int((~comparison.within_frozen_tolerance).sum()),
        "model_identity": identity, "isolated_process": True, "fits": 0, "input_nonmutation": True,
        "runtime_versions": a0_data_focus._versions(), "smoke": smoke}
    _save_json(paths["report_root"] / "native_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    result = native_worker(args.protocol) if args.native_worker else reconstruct(args.protocol)
    print(json.dumps({key: result[key] for key in ("schema", "status", "rows", "targets", "probability_parity") if key in result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
