"""Identity-safe alignment of raw historical player-token banks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FRESH_COLUMNS = {
    "match_id", "date", "release_day", "best_of", "y", "team_a", "team_b",
    "roster_a", "roster_b", "max_history_release_day",
}
_A0_COLUMNS = {
    "golgg_match_id", "date", "result_day", "best_of", "y", "team1_id",
    "team2_id", "roster_a", "roster_b", "feature_source_max_day",
}


def _identifier(value: Any, description: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{description} must be present")
    return str(value)


def _date(value: Any, description: str, *, allow_missing: bool = False) -> pd.Timestamp | None:
    if allow_missing and (value is None or pd.isna(value)):
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    if pd.isna(parsed):
        if allow_missing:
            return None
        raise ValueError(f"{description} must be present")
    return parsed.normalize()


def _roster(value: Any, description: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError(f"{description} must contain exactly five player IDs")
    try:
        players = tuple(_identifier(player, description) for player in value)
    except TypeError as exc:
        raise ValueError(f"{description} must contain exactly five player IDs") from exc
    if len(players) != 5 or len(set(players)) != 5:
        raise ValueError(f"{description} must contain five unique player IDs")
    return players


def _unique_id_index(frame: pd.DataFrame, column: str, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for i, value in enumerate(frame[column].to_numpy()):
        match_id = _identifier(value, f"{label} match ID")
        if match_id in result:
            raise ValueError(f"Duplicate {label} match ID {match_id!r}")
        result[match_id] = i
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_history(
    fresh_metadata: pd.DataFrame,
    a0_metadata: pd.DataFrame,
    history: np.ndarray,
    mask: np.ndarray,
    *,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align an A0 history bank to fresh rows, team sides, and player roles.

    ``history`` is ``[rows, 2, 5, tokens, features]`` and ``mask`` is
    ``[rows, 2, 5, tokens]`` in A0 metadata order. Output follows fresh row
    order and each roster's explicit order. Masked token payloads are copied
    unchanged and remain masked.
    """
    if not isinstance(fresh_metadata, pd.DataFrame) or not isinstance(a0_metadata, pd.DataFrame):
        raise ValueError("Metadata inputs must be pandas DataFrames")
    missing_fresh = _FRESH_COLUMNS.difference(fresh_metadata.columns)
    missing_a0 = _A0_COLUMNS.difference(a0_metadata.columns)
    if missing_fresh or missing_a0:
        raise ValueError(f"Missing metadata columns: fresh={sorted(missing_fresh)}, a0={sorted(missing_a0)}")

    source_history = np.asanyarray(history)
    source_mask = np.asanyarray(mask)
    if source_history.ndim != 5 or source_history.shape[1:3] != (2, 5):
        raise ValueError("history must have shape [rows, 2, 5, tokens, features]")
    if not np.issubdtype(source_history.dtype, np.number):
        raise ValueError("history must use a numeric dtype")
    if source_history.shape[0] != len(fresh_metadata) or source_history.shape[0] != len(a0_metadata):
        raise ValueError("History row count must match both metadata tables")
    if source_mask.shape != source_history.shape[:-1]:
        raise ValueError("mask shape must match history's [rows, 2, 5, tokens] dimensions")
    if source_mask.dtype != np.bool_:
        if not np.issubdtype(source_mask.dtype, np.number) or not np.isin(source_mask, (0, 1)).all():
            raise ValueError("mask values must be boolean or 0/1")
        source_mask = source_mask.astype(bool, copy=False)

    fresh_index = _unique_id_index(fresh_metadata, "match_id", "fresh")
    a0_index = _unique_id_index(a0_metadata, "golgg_match_id", "A0")
    fresh_ids, a0_ids = set(fresh_index), set(a0_index)
    if fresh_ids != a0_ids:
        missing = sorted(fresh_ids - a0_ids)[:5]
        extra = sorted(a0_ids - fresh_ids)[:5]
        raise ValueError(f"Match populations differ; missing from A0={missing}, extra in A0={extra}")

    n, _, _, tokens, features = source_history.shape
    permutation = np.empty((n, 2, 5), dtype=np.int8)
    row_indices = np.empty(n, dtype=np.int64)
    source_sides = np.empty((n, 2), dtype=np.int8)
    side_swaps = 0
    role_changes = 0
    fresh_max: pd.Timestamp | None = None
    a0_max: pd.Timestamp | None = None

    for fresh_row, (_, fresh) in enumerate(fresh_metadata.iterrows()):
        match_id = _identifier(fresh["match_id"], "fresh match ID")
        old_row = a0_index[match_id]
        old = a0_metadata.iloc[old_row]
        row_indices[fresh_row] = old_row
        target_date = _date(fresh["date"], f"fresh date for {match_id}")
        for col, label in (("date", "date"), ("release_day", "result release day")):
            fresh_day = _date(fresh[col], f"fresh {label} for {match_id}")
            old_col = "date" if col == "date" else "result_day"
            old_day = _date(old[old_col], f"A0 {label} for {match_id}")
            if fresh_day != old_day:
                raise ValueError(f"{label} mismatch for match {match_id!r}")
        if fresh["best_of"] != old["best_of"]:
            raise ValueError(f"best_of mismatch for match {match_id!r}")

        fresh_a = _identifier(fresh["team_a"], f"fresh team A for {match_id}")
        fresh_b = _identifier(fresh["team_b"], f"fresh team B for {match_id}")
        old_teams = (
            _identifier(old["team1_id"], f"A0 team 1 for {match_id}"),
            _identifier(old["team2_id"], f"A0 team 2 for {match_id}"),
        )
        if fresh_a == fresh_b or old_teams[0] == old_teams[1]:
            raise ValueError(f"Team IDs must be distinct for match {match_id!r}")
        if set(old_teams) != {fresh_a, fresh_b}:
            raise ValueError(f"Team identity mismatch for match {match_id!r}")
        swap = old_teams != (fresh_a, fresh_b)
        side_swaps += int(swap)
        try:
            old_y, fresh_y = int(old["y"]), int(fresh["y"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Outcomes must be binary for match {match_id!r}") from exc
        if old_y not in (0, 1) or fresh_y not in (0, 1):
            raise ValueError(f"Outcomes must be binary for match {match_id!r}")
        expected_y = 1 - old_y if swap else old_y
        if fresh_y != expected_y:
            raise ValueError(f"Outcome mismatch after side orientation for match {match_id!r}")

        fresh_rosters = (
            _roster(fresh["roster_a"], f"fresh roster A for {match_id}"),
            _roster(fresh["roster_b"], f"fresh roster B for {match_id}"),
        )
        if set(fresh_rosters[0]) & set(fresh_rosters[1]):
            raise ValueError(f"Player IDs overlap between teams for match {match_id!r}")
        old_rosters = (
            _roster(old["roster_a"], f"A0 roster A for {match_id}"),
            _roster(old["roster_b"], f"A0 roster B for {match_id}"),
        )
        if set(old_rosters[0]) & set(old_rosters[1]):
            raise ValueError(f"Player IDs overlap between teams for match {match_id!r}")

        for fresh_side, fresh_team in enumerate((fresh_a, fresh_b)):
            old_side = old_teams.index(fresh_team)
            if set(old_rosters[old_side]) != set(fresh_rosters[fresh_side]):
                raise ValueError(f"Roster membership mismatch for team {fresh_team!r} in match {match_id!r}")
            source_players = old_rosters[old_side]
            slot_map = [source_players.index(player) for player in fresh_rosters[fresh_side]]
            role_changes += int(slot_map != list(range(5)))
            source_sides[fresh_row, fresh_side] = old_side
            permutation[fresh_row, fresh_side] = slot_map

        fresh_history_day = _date(fresh["max_history_release_day"], f"fresh history max day for {match_id}", allow_missing=True)
        old_history_day = _date(old["feature_source_max_day"], f"A0 history max day for {match_id}", allow_missing=True)
        row_has_history = bool(source_mask[old_row].any())
        if row_has_history and (fresh_history_day is None or old_history_day is None):
            raise ValueError(f"Observed history requires release bounds for match {match_id!r}")
        # These are global replay watermarks; new players may have no tokens
        # even though earlier matches contributed to the rating state.
        for history_day, label in ((fresh_history_day, "fresh"), (old_history_day, "A0")):
            if history_day is not None and history_day >= target_date:
                raise ValueError(f"{label} history is not strictly prior to target date for match {match_id!r}")
        if fresh_history_day is not None and (fresh_max is None or fresh_history_day > fresh_max):
            fresh_max = fresh_history_day
        if old_history_day is not None and (a0_max is None or old_history_day > a0_max):
            a0_max = old_history_day

    chunk_size = 128
    for start in range(0, n, chunk_size):
        chunk = source_history[start : start + chunk_size]
        if not np.isfinite(chunk[source_mask[start : start + chunk_size]]).all():
            raise ValueError("Observed history values must be finite")

    if output_dir is None:
        aligned_history = np.empty(source_history.shape, dtype=source_history.dtype)
        aligned_mask = np.empty(source_mask.shape, dtype=bool)
    else:
        destination = Path(output_dir)
        if destination.exists():
            raise ValueError(f"output_dir must not already exist: {destination}")
        destination.mkdir(parents=True)
        aligned_history = np.lib.format.open_memmap(
            destination / "history.npy", mode="w+", dtype=source_history.dtype, shape=source_history.shape
        )
        aligned_mask = np.lib.format.open_memmap(
            destination / "history_mask.npy", mode="w+", dtype=bool, shape=source_mask.shape
        )

    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        for out_row in range(start, stop):
            in_row = row_indices[out_row]
            for side in range(2):
                source_side = source_sides[out_row, side]
                slots = permutation[out_row, side]
                aligned_history[out_row, side] = source_history[in_row, source_side, slots]
                aligned_mask[out_row, side] = source_mask[in_row, source_side, slots]
    if isinstance(aligned_history, np.memmap):
        aligned_history.flush()
        aligned_mask.flush()

    receipt: dict[str, Any] = {
        "row_count": n,
        "side_swaps": side_swaps,
        "role_order_changes": role_changes,
        "max_fresh_history_release_day": fresh_max.date().isoformat() if fresh_max is not None else None,
        "max_a0_feature_source_day": a0_max.date().isoformat() if a0_max is not None else None,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        receipt["output_hashes"] = {
            "history.npy": _sha256(destination / "history.npy"),
            "history_mask.npy": _sha256(destination / "history_mask.npy"),
        }
        (destination / "alignment.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return aligned_history, aligned_mask, receipt
