"""Trace frozen prior-lineup carry-forward across source competition contexts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.a0_error_mechanisms import _raw_match_records, _raw_roster, _roster
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import binary_log_loss_vector


def run(tables: dict, protocol: dict, output: Path) -> dict:
    primary = tables["opportunities"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.audit_primary_eligible].copy()
    origins = tables["origins"].loc[lambda frame: frame.origin_id.isin(primary.origin_id)].copy()
    selected = tables["bets"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.quality.eq("exclude_q25_conflicts")
        & frame.policy.eq("raw_ev_gt_5")].copy()
    refs = {(row.origin_id, side): json.loads(getattr(row, "roster_source_" + side))
            if isinstance(getattr(row, "roster_source_" + side), str)
            else getattr(row, "roster_source_" + side)
            for row in origins.itertuples() for side in ("a", "b")}
    needed = set(origins.golgg_match_id.astype(str)) | {str(ref["match_id"]) for ref in refs.values()}
    raw = _raw_match_records(protocol["paths"]["raw_history"], needed)
    assert set(raw) == needed
    family = tables["canonical"].set_index("golgg_match_id").competition_family
    family = dict(zip(family.index.astype(str), family.to_numpy()))
    records = []
    for row in origins.itertuples():
        target = raw[str(row.golgg_match_id)]
        for side, team in (("a", str(row.team1_id)), ("b", str(row.team2_id))):
            ref = refs[(row.origin_id, side)]
            source = raw[str(ref["match_id"])]
            source_game = source["games"][int(ref["map_index"]) - 1]
            target_game = target["games"][0]
            source_side = "t1" if str(source_game["t1_id"]) == team else "t2"
            target_side = "t1" if str(target_game["t1_id"]) == team else "t2"
            assert str(source_game[source_side + "_id"]) == team
            assert str(target_game[target_side + "_id"]) == team
            source_players, _ = _raw_roster(source_game, source_side)
            target_players, _ = _raw_roster(target_game, target_side)
            assert set(source_players) == set(_roster(getattr(row, "roster_" + side)))
            assert set(target_players) == set(_roster(getattr(row, "canonical_roster_" + side)))
            source_name, target_name = source["tournament_name"], target["tournament_name"]
            cup_to_lck = source_name.startswith("KeSPA Cup") and target_name.startswith("LCK ") and not target_name.startswith("LCK CL ")
            changed = set(source_players) != set(target_players)
            records.append({"origin_id": row.origin_id, "fixture_id": row.fixture_id,
                "golgg_match_id": str(row.golgg_match_id), "date": row.date, "side": side,
                "team_id": team, "team_name": target_game[target_side + "_name"],
                "source_match_id": str(ref["match_id"]), "source_map_index_1based": int(ref["map_index"]),
                "source_release_day": ref["effective_release_day"], "source_tournament": source_name,
                "target_tournament": target_name, "source_family": family.get(str(ref["match_id"])),
                "target_family": row.competition_family, "source_roster": source_players,
                "target_roster": target_players,
                "source_player_names": [p.get("player_name") for p in source_game[source_side + "_players"].values()],
                "target_player_names": [p.get("player_name") for p in target_game[target_side + "_players"].values()],
                "same_tournament_label": source_name == target_name, "players_changed": changed,
                "replacements": len(set(target_players) - set(source_players)),
                "source_snapshot_age_days": getattr(row, "roster_snapshot_age_days_" + side),
                "cup_to_lck": cup_to_lck, "cup_to_lck_changed": cup_to_lck and changed,
                "source_and_target_rosters_verified": True})
    sides = pd.DataFrame(records)
    flags = sides.groupby("origin_id")[["cup_to_lck", "cup_to_lck_changed", "players_changed"]].any()
    fixtures = primary.merge(flags, on="origin_id", validate="one_to_one")
    fixtures["prior_ll"] = binary_log_loss_vector(fixtures.y_event, fixtures.p_event, epsilon=1e-12)
    fixtures["target_ll"] = binary_log_loss_vector(fixtures.y_event, fixtures.p_target, epsilon=1e-12)
    fixtures["market_ll"] = binary_log_loss_vector(fixtures.y_event, fixtures.q_consensus_a, epsilon=1e-12)
    fixtures["oracle_gain"] = fixtures.prior_ll - fixtures.target_ll
    fixtures["excess_ll"] = fixtures.prior_ll - fixtures.market_ll
    fixtures["selected"] = fixtures.fixture_id.isin(selected.fixture_id)
    subsets = {"all": np.ones(len(fixtures), bool),
        "cup_to_lck": fixtures.cup_to_lck.to_numpy(),
        "cup_to_lck_changed": fixtures.cup_to_lck_changed.to_numpy(),
        "no_cup_to_lck_changed": ~fixtures.cup_to_lck_changed.to_numpy(),
        "unchanged_players": ~fixtures.players_changed.to_numpy()}
    groups = []
    for name, keep in subsets.items():
        group = fixtures.loc[keep]
        ci = block_ratio_interval(np.where(keep, fixtures.oracle_gain, 0.), keep.astype(float), fixtures.month, 5000, 20260930)
        groups.append({"subset": name, "n": len(group), "selected_n": int(group.selected.sum()),
            "prior_ll": float(group.prior_ll.mean()) if len(group) else None,
            "target_ll": float(group.target_ll.mean()) if len(group) else None,
            "market_ll": float(group.market_ll.mean()) if len(group) else None,
            "excess_ll_sum": float(group.excess_ll.sum()), "oracle_gain_sum": float(group.oracle_gain.sum()),
            "oracle_gain_monthly_interval": ci})
    sides.to_parquet(output / "source_roster_transitions.parquet", index=False)
    sides.to_csv(output / "source_roster_transitions.csv", index=False)
    fixtures.to_parquet(output / "fixture_context_effects.parquet", index=False)
    pd.json_normalize(groups).to_csv(output / "context_effects.csv", index=False)
    cup_cases = fixtures.loc[fixtures.cup_to_lck_changed].merge(
        selected[["fixture_id", "side", "p_model_bet", "q_bet", "won", "odds"]],
        on="fixture_id", how="left", validate="one_to_one")
    cup_cases.to_csv(output / "cup_to_lck_changed_cases.csv", index=False)
    return {"status": "COMPLETE_POST_HOC_SOURCE_CONTEXT_AUDIT", "raw_records": len(raw),
        "source_and_target_side_rosters_verified": len(sides), "primary_fixtures": len(fixtures),
        "groups": groups, "cup_to_lck_changed_rows": int(sides.cup_to_lck_changed.sum()),
        "limitation": "A hindsight mismatch is evidence of the prior-lineup scenario limitation, not proof that the actual five had been announced. Cup labels and archived dates are source claims; no external schedule verification or competition exclusion was performed."}
