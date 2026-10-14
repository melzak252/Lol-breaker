"""Read-only, bounded-memory replay of the frozen champion-history source contract.

The legacy parser remains authoritative. Bounded temporary JSON batches permit its
reuse without retaining the raw tree or its full object graph. Only compact event
columns and serialized rating updates survive parsing; temporary files are removed.
"""
from __future__ import annotations

from bisect import insort
from collections import defaultdict
from datetime import date
import json
import math
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.export_prospective_features import iter_matches
from src.features import champion_history_data as old
from src.models import prematch_features as strength

CHANNELS = ("champion", "patch", "competition")
ARRAY_KEYS = {"champion": "champions", "patch": "patches", "competition": "competitions"}


def _flush_source(raw_batch, path, targets, overrides):
    path.write_text(json.dumps(raw_batch, allow_nan=True), encoding="utf-8")
    return old._read_source(path, targets, overrides)[0]


def replay_identity_sources(source_path, series, arrays, schema, *, output, roots):
    """Validate every historical slot, export compact lineage, return safe sidecars.

    Inputs are the original arrays loaded once by the caller. No fitted state,
    target labels, or saved forecast is consulted in this replay.
    """
    output = Path(output)
    if not np.array_equal(series.row_index.to_numpy(), np.arange(len(series))):
        raise ValueError("canonical row index differs from frozen arrays")
    targets = {str(row["match_id"]): row for row in series[
        ["match_id", "date", "release_day", "best_of", "team_a", "team_b", "roster_a", "roster_b"]
    ].to_dict("records")}
    if len(targets) != len(series):
        raise ValueError("duplicate canonical target")
    overrides = old._release_overrides(tuple(Path(root) for root in roots))
    observed_matches, observed_games, observed_events = set(), set(), set()
    source_rows = []
    writer = None
    event_offset = 0
    with TemporaryDirectory(prefix="champion-oov-replay-") as temporary:
        batch_path = Path(temporary) / "batch.json"
        events_path = Path(temporary) / "events.parquet"
        batch = []

        def consume(raw_batch):
            nonlocal writer, event_offset
            records = _flush_source(raw_batch, batch_path, targets, overrides)
            columns = defaultdict(list)
            for record in records:
                if record.match_id in observed_matches:
                    raise ValueError("duplicate source series ID " + record.match_id)
                observed_matches.add(record.match_id)
                vocabulary_events = [
                    (sorted({c for c in event.champions if c is not None}), event.patch, event.competition)
                    for event in record.maps if event.history_eligible]
                source_rows.append((record.match_id, record.day.isoformat(), record.release.isoformat(),
                                    pickle.dumps(record.rating_record, protocol=5), vocabulary_events))
                for event in record.maps:
                    if event.game_id in observed_games:
                        raise ValueError("duplicate source game ID " + event.game_id)
                    observed_games.add(event.game_id)
                    for slot, player in enumerate(event.players):
                        if player is None:
                            continue
                        observation_id = event.game_id + ":" + player
                        if observation_id in observed_events:
                            raise ValueError("duplicate source observation key " + observation_id)
                        observed_events.add(observation_id)
                        values = dict(event_index=event_offset, observation_id=observation_id,
                            match_id=record.match_id, game_id=event.game_id, player_id=player,
                            date=record.day.isoformat(), played_day=date.fromordinal(event.played).isoformat(),
                            release_day=record.release.isoformat(), map_index=event.map_index,
                            source_slot=slot, champion=event.champions[slot], patch=event.patch,
                            competition=event.competition, champion_status=int(event.champion_status[slot]),
                            role=int(event.roles[slot]), history_eligible=event.history_eligible)
                        values.update(zip(old.HISTORY_FEATURE_NAMES[:18], map(float, event.numeric[slot])))
                        for key, value in values.items():
                            columns[key].append(value)
                        event_offset += 1
            table = pa.Table.from_pydict(columns)
            if writer is None:
                # Explicit nullable string types avoid all-null chunk inference.
                for name in ("champion", "patch", "competition"):
                    table = table.set_column(table.schema.get_field_index(name), name, table[name].cast(pa.string()))
                writer = pq.ParquetWriter(events_path, table.schema)
            table = table.cast(writer.schema)
            writer.write_table(table)

        try:
            for raw in iter_matches(Path(source_path)):
                batch.append(raw)
                if len(batch) == 256:
                    consume(batch)
                    batch.clear()
            if batch:
                consume(batch)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("empty original source")
        events = pd.read_parquet(events_path, dtype_backend="pyarrow")
    # String columns remain Arrow-backed; numeric source fields use one compact FP32 matrix.
    sources = sorted(source_rows, key=lambda row: (row[1], old._series_key(row[0])))
    rank_by_id = {row[0]: rank for rank, row in enumerate(sources)}
    events["source_rank"] = events.match_id.map(rank_by_id).astype(np.int64)
    numeric = events[list(old.HISTORY_FEATURE_NAMES[:18])].to_numpy(dtype=np.float32, na_value=np.nan)
    played = np.asarray([date.fromisoformat(str(d)).toordinal() for d in events.played_day], dtype=np.int32)
    released = np.asarray([date.fromisoformat(str(d)).toordinal() for d in events.release_day], dtype=np.int32)
    source_day = np.asarray([date.fromisoformat(str(d)).toordinal() for d in events.date], dtype=np.int32)
    if np.any(released < played) or np.any(released < source_day):
        raise ValueError("source release precedes played/forecast")
    eligible = events.history_eligible.to_numpy(bool)
    source_slots = events.source_slot.to_numpy(np.int8)
    ranks = events.source_rank.to_numpy(np.int64)
    map_indices = events.map_index.to_numpy(np.int32)
    players = events.player_id.to_numpy()
    event_roles = events.role.to_numpy(np.int64)
    status = events.champion_status.to_numpy(np.uint8)
    identity_tables, identity_indices = {}, {}
    for channel in CHANNELS:
        values = events[channel]
        table = [None, *sorted(str(v) for v in values.dropna().unique())]
        lookup = {value: i for i, value in enumerate(table) if value is not None}
        identity_tables[channel] = table
        identity_indices[channel] = values.map(lookup).fillna(0).to_numpy(np.int32)
    # Reproduce vocabulary insertion order, including history-only BO2 observations.
    vocabularies = {channel: {} for channel in CHANNELS}
    for record in sources:
        if record[1] >= old.TRAIN_END or record[2] >= old.TRAIN_END:
            continue
        for champions, patch, competition in record[4]:
            for channel, identities in zip(CHANNELS, (champions, [patch], [competition])):
                for identity in identities:
                    if identity is not None:
                        vocabularies[channel].setdefault(identity, len(vocabularies[channel]) + 2)
    for channel in CHANNELS:
        if vocabularies[channel] != schema[channel + "_vocabulary"]:
            raise ValueError(channel + " original closed TRAIN vocabulary mismatch")
    # Materialize historical opponent strength using the unchanged causal rating helpers.
    source_by_day, releases = defaultdict(list), defaultdict(list)
    for rank, row in enumerate(sources):
        source_by_day[row[1]].append(rank)
        releases[row[2]].append(rank)
    events_by_rank = {int(rank): frame.event_index.to_numpy(np.int64)
                      for rank, frame in events.groupby("source_rank", sort=False)}
    release_days = sorted(releases)
    release_position = 0
    state = strength.ReplayState()
    for day_string in sorted(source_by_day):
        day = date.fromisoformat(day_string)
        while release_position < len(release_days) and release_days[release_position] < day_string:
            release_string = release_days[release_position]
            rr = [pickle.loads(sources[rank][3]) for rank in sorted(
                releases[release_string], key=lambda r: old._series_key(sources[r][0]))]
            rr = [record for record in rr if record.maps]
            if rr:
                strength._consume_release_day(state, rr, date.fromisoformat(release_string))
            release_position += 1
        for rank in source_by_day[day_string]:
            # Rating update blobs are only needed at release, not at source origin.
            indices = events_by_rank.get(rank, np.empty(0, np.int64))
            # Every map may have a different roster; derive it from the compact event ledger.
            for map_index in np.unique(map_indices[indices]):
                ii = indices[map_indices[indices] == map_index]
                for side in range(2):
                    opponent_slots = {int(source_slots[i]) % 5: players[i]
                                      for i in ii if source_slots[i] // 5 == 1 - side}
                    opponent = tuple(opponent_slots.get(slot) for slot in range(5))
                    mean, variance = strength._player_team_strength(state, opponent, day, strength._PROCESS_SLOW)
                    own = ii[source_slots[ii] // 5 == side]
                    numeric[own, 16:18] = mean, math.sqrt(variance)
    del sources, source_rows, state
    # Per-token event pointers retain right-alignment without repeated identity strings.
    shape = arrays["mask"].shape
    pointers = np.full(shape, -1, dtype=np.int64)
    counts = {channel: np.zeros(shape, dtype=np.int64) for channel in CHANNELS}
    token_ids = {channel: np.zeros(shape, dtype=np.int32) for channel in CHANNELS}
    source_order = np.lexsort((source_slots, map_indices, ranks, released))
    target_order = np.argsort(series.date.to_numpy(), kind="stable")
    histories = defaultdict(list)
    totals = {channel: np.zeros(len(identity_tables[channel]), dtype=np.int64) for channel in CHANNELS}
    position = 0
    roles = arrays["roles"]
    mask = arrays["mask"]
    for row_index in target_order:
        target = series.iloc[int(row_index)]
        target_day = date.fromisoformat(str(target.date)).toordinal()
        while position < len(source_order) and released[source_order[position]] < target_day:
            ei = int(source_order[position])
            if eligible[ei]:
                entry = (int(played[ei]), int(ranks[ei]), int(map_indices[ei]), int(source_slots[ei]), ei)
                history = histories[players[ei]]
                insort(history, entry)
                if len(history) > shape[-1]:
                    history.pop(0)
                for channel in CHANNELS:
                    identity_index = identity_indices[channel][ei]
                    if identity_index:
                        totals[channel][identity_index] += 1
            position += 1
        for side, roster in enumerate((target.roster_a, target.roster_b)):
            for slot, player in enumerate(roster):
                history = histories.get(player, ()) if player is not None else ()
                expected_mask = np.zeros(shape[-1], bool)
                if history:
                    expected_mask[-len(history):] = True
                if not np.array_equal(mask[row_index, side, slot], expected_mask):
                    raise ValueError("source/target history mask alignment mismatch")
                for hs, entry in enumerate(history, start=shape[-1] - len(history)):
                    ei = entry[-1]
                    axis = (int(row_index), side, slot, hs)
                    pointers[axis] = ei
                    expected = np.r_[numeric[ei], target_day - played[ei], target_day - released[ei]].astype(np.float32)
                    if not np.array_equal(arrays["history"][axis], expected, equal_nan=True):
                        raise ValueError("source numeric history lineage mismatch at " + str(axis))
                    if roles[axis] != event_roles[ei] or ("champion_status" in arrays and arrays["champion_status"][axis] != status[ei]):
                        raise ValueError("source role/champion reliability lineage mismatch")
                    for channel in CHANNELS:
                        identity_index = identity_indices[channel][ei]
                        identity = identity_tables[channel][int(identity_index)]
                        code = 0 if identity is None else vocabularies[channel].get(identity, 1)
                        if arrays[ARRAY_KEYS[channel]][axis] != code:
                            raise ValueError(channel + " canonical identity/code alignment mismatch")
                        token_ids[channel][axis] = identity_index
                        counts[channel][axis] = totals[channel][identity_index] if identity_index else 0
    for column, values in zip(old.HISTORY_FEATURE_NAMES[:18], numeric.T):
        events[column] = values
    events.to_parquet(output / "source_observations.parquet", index=False)
    axes = np.nonzero(mask)
    pd.DataFrame(dict(row_index=axes[0], side=axes[1], player_slot=axes[2], history_slot=axes[3],
                      event_index=pointers[axes])).to_parquet(output / "identity_tokens.parquet", index=False)
    np.savez_compressed(output / "support_arrays.npz", event_index=pointers,
        **{channel + "_identity_index": token_ids[channel] for channel in CHANNELS},
        **{"n_prior_" + channel: counts[channel] for channel in CHANNELS})
    identity_schema = dict(canonical_id_tables=identity_tables, vocabularies=vocabularies,
        token_axes=["row", "side", "player_slot", "history_slot"], padding_event_index=-1,
        missing_identity_index=0, support_unit="unique game_id:canonical_player_id observation",
        strict_release="source whole-series release_day < target forecast_day", all_slots_validated=True)
    (output / "identity_schema.json").write_text(json.dumps(identity_schema, indent=2) + "\n", encoding="utf-8")
    return {"n_prior": counts, "canonical_identity_indices": token_ids,
            "canonical_id_tables": identity_tables}, events, identity_schema
