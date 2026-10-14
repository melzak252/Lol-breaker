"""Diagnose frozen repaired A0 errors without fitting or changing selections."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value) -> None:
    from src.analysis.ev_asof_replay import _json_safe
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def load_protocol(path: Path, research_root: Path | None = None) -> dict:
    from src.models.a0_target_experiment import resolve_research_root
    protocol = json.loads(path.read_text())
    archive = resolve_research_root(research_root)
    paths = {}
    for name, record in protocol["sources"].items():
        source = (archive if record["root"] == "research" else ROOT) / record["path"]
        if sha256(source) != record["sha256"]:
            raise ValueError(f"Audit input changed: {name}")
        paths[name] = source
    protocol["paths"] = paths
    protocol["resolved_research_root"] = archive
    return protocol


def _decoded(value, default):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return json.loads(value) if isinstance(value, str) else value


def prepare(protocol: dict) -> dict:
    """Preserve inventory and mark hindsight diagnostics, never causal features."""
    destination = ROOT / protocol["input_dir"]
    files = {name: destination / f"{name}.parquet" for name in ("origins", "opportunities", "bets", "canonical")}
    if any(path.exists() for path in files.values()):
        raise FileExistsError("Refusing to overwrite prepared diagnostic inputs")
    paths = protocol["paths"]
    origins = pd.read_parquet(paths["repair_predictions"])
    metadata = pd.read_parquet(paths["canonical_2026"])
    paired = pd.read_parquet(paths["previous_paired"])
    opportunities = pd.read_parquet(paths["repair_opportunities"])
    ledger = pd.read_parquet(paths["repair_ledger"])
    for frame in (origins, metadata, paired, opportunities, ledger):
        for key in ("golgg_match_id", "fixture_id", "team1_id", "team2_id"):
            if key in frame:
                frame[key] = frame[key].astype("string")
    if origins.origin_id.duplicated().any() or metadata.golgg_match_id.duplicated().any():
        raise ValueError("Nonunique forecast or canonical identity")
    meta_columns = ["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y", "tournament_name", "competition_family", "competition_tier", "roster_a", "roster_b", "p_full", "p_glicko", "p_elo", "p_039"]
    rename = {key: f"canonical_{key}" for key in ("team1_id", "team2_id", "date", "best_of", "roster_a", "roster_b", "p_full", "p_glicko", "p_elo", "p_039")}
    origins = origins.merge(metadata[meta_columns].rename(columns=rename), on="golgg_match_id", how="left", validate="many_to_one")
    for key in ("team1_id", "team2_id", "best_of"):
        if not origins[key].eq(origins[f"canonical_{key}"]).all():
            raise ValueError(f"Canonical orientation/format mismatch: {key}")
    if origins.y.isna().any():
        raise ValueError("Missing canonical target")
    paired_fields = ["origin_id", "golgg_match_id", "p_golgg_target_roster", "p_no_w20_target_roster", "p_no_organization_target_roster", "roster_change_type"]
    origins = origins.merge(paired[paired_fields], on=["origin_id", "golgg_match_id"], how="left", validate="one_to_one")
    origins = origins.rename(columns={"p_golgg": "p", "p_golgg_target_roster": "p_target", "p_no_w20_target_roster": "p_target_no_w20", "p_no_organization_target_roster": "p_target_no_organization"})
    origins["month"] = pd.to_datetime(origins.date, utc=True).dt.strftime("%Y-%m")
    origins["edition_proxy"] = origins.tournament_name.fillna("unknown").astype(str) + ":" + origins.month.str[:4]
    origins["canonical_same_cutoff_day"] = origins.cutoff_day.eq(origins.canonical_date)
    features = []
    for row in origins.itertuples(index=False):
        evidence = _decoded(row.team_feature_provenance, [])
        if row.status == "available":
            if not isinstance(evidence, list) or len(evidence) != 2:
                raise ValueError("Available origin lacks two-sided feature provenance")
        else:
            evidence = []
        record = {}
        for index, side in enumerate(("a", "b")):
            team = evidence[index] if index < len(evidence) else {}
            expected = getattr(row, f"team{index+1}_id")
            if team and str(team["team_id"]) != expected:
                raise ValueError("Feature provenance team orientation mismatch")
            for field in ("unknown_players", "min_player_series", "team_prior_games", "co_play_mean", "roster_overlap", "w20_window_games", "w20_valid_games"):
                record[f"{field}_{side}"] = team.get(field)
            source = _decoded(getattr(row, f"roster_source_{side}"), {})
            observed = pd.to_datetime(source.get("effective_release_day"), utc=True, errors="coerce")
            cutoff = pd.to_datetime(row.cutoff_day, utc=True)
            record[f"roster_snapshot_age_days_{side}"] = (cutoff-observed).days if pd.notna(observed) else np.nan
            prior = {str(value) for value in _decoded(getattr(row, f"roster_{side}"), []) if value is not None}
            actual = {str(value) for value in _decoded(getattr(row, f"canonical_roster_{side}"), []) if value is not None}
            record[f"hindsight_roster_replacements_{side}"] = len(actual-prior) if len(prior) == len(actual) == 5 else np.nan
        features.append(record)
    origins = pd.concat([origins.reset_index(drop=True), pd.DataFrame(features)], axis=1)
    origins["hindsight_same_players"] = origins.hindsight_roster_replacements_a.eq(0) & origins.hindsight_roster_replacements_b.eq(0)
    origins["hindsight_any_roster_change"] = (origins.hindsight_roster_replacements_a + origins.hindsight_roster_replacements_b).gt(0)
    origin_columns = ["origin_id", "fixture_id", "horizon", "p", "y", "p_no_w20", "p_no_organization", "p_target", "hindsight_same_players", "hindsight_any_roster_change", "canonical_same_cutoff_day", "edition_proxy", "competition_tier", *features[0].keys()]
    native = origins.loc[origins.cohort.eq("oddspapi_common"), origin_columns].rename(columns={"p":"p_event", "y":"y_event"})
    opportunities = opportunities.merge(native, on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    known = opportunities.p_event.notna()
    if not np.allclose(opportunities.loc[known, "p"], opportunities.loc[known, "p_event"], atol=1e-12, rtol=0):
        raise ValueError("Repaired opportunity probability mismatch")
    if not opportunities.loc[known, "y"].eq(opportunities.loc[known, "y_event"]).all():
        raise ValueError("Repaired opportunity outcome mismatch")
    opportunities["audit_primary_eligible"] = opportunities.available & ~opportunities.conflict_q25.astype(bool)
    opportunities["month"] = pd.to_datetime(opportunities.date.combine_first(opportunities.provider_day), utc=True).dt.strftime("%Y-%m")
    if opportunities.month.isna().any():
        raise ValueError("Missing calendar block in opportunity inventory")
    bets = ledger.loc[ledger.arm.eq("repaired") & ledger.strategy.eq("flat")].copy()
    bets = bets.rename(columns={"p":"p_bet", "q":"q_saved", "ev":"ev_bet", "origin_id":"saved_origin_id"})
    native_bets = bets.loc[bets.cohort.eq("oddspapi_common")].merge(native, on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    parent = origins.loc[origins.cohort.eq("parent_origins"), origin_columns].drop(columns=["fixture_id", "horizon"]).rename(columns={"p":"p_event", "y":"y_event"})
    parent_bets = bets.loc[~bets.cohort.eq("oddspapi_common")].merge(parent, left_on="row_id", right_on="origin_id", how="left", validate="many_to_one")
    bets = pd.concat([native_bets, parent_bets], ignore_index=True)
    if bets.origin_id.isna().any():
        raise ValueError("Saved bet has no repaired forecast origin")
    bets["q_bet"] = np.where(bets.side.eq("A"), bets.q_consensus_a, 1-bets.q_consensus_a)
    bets["p_model_bet"] = np.where(bets.side.eq("A"), bets.p_event, 1-bets.p_event)
    expected_win = np.where(bets.side.eq("A"), bets.y_event, 1-bets.y_event)
    if not np.array_equal(expected_win.astype(bool), bets.won.to_numpy(bool)):
        raise ValueError("Selected-side outcome mismatch")
    bets["month"] = pd.to_datetime(bets.date, utc=True).dt.strftime("%Y-%m")
    canonical = pd.read_parquet(paths["canonical_all"])
    if len(canonical) != 11550 or canonical.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("Canonical annual cohort changed")
    data = {"origins":origins,"opportunities":opportunities,"bets":bets,"canonical":canonical}
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_parquet(files[name], index=False)
    receipt = {"status":"PREPARED", "run_id":protocol["run_id"], "created_at_utc":datetime.now(timezone.utc).isoformat(), "rows":{name:len(frame) for name,frame in data.items()}, "sha256":{name:sha256(path) for name,path in files.items()}, "column_policy":"hindsight_* and target/canonical roster forecasts are forensic controls, not prematch covariates; no saved selections changed"}
    save_json(destination / "prepared_receipt.json", receipt)
    return receipt


def read_tables(protocol: dict) -> dict[str, pd.DataFrame]:
    directory = ROOT / protocol["input_dir"]
    receipt = json.loads((directory / "prepared_receipt.json").read_text())
    tables = {}
    for name, expected in receipt["sha256"].items():
        path = directory / f"{name}.parquet"
        if sha256(path) != expected:
            raise ValueError(f"Prepared input changed: {name}")
        tables[name] = pd.read_parquet(path)
    return tables


def global_context(tables: dict, protocol: dict, output: Path) -> dict:
    """Separate population-wide confidence from selection-conditioned error."""
    from src.analysis.bootstrap import block_ratio_interval
    from src.analysis.probability_metrics import binary_log_loss_vector
    canonical = tables["canonical"]
    primary = tables["opportunities"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.audit_primary_eligible].copy()
    selected = tables["bets"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.quality.eq("exclude_q25_conflicts")
        & frame.policy.eq("raw_ev_gt_5")]
    populations = [(f"canonical_{year}", canonical.loc[canonical.year.astype(str).eq(str(year))], "p_full")
                   for year in (2024, 2025, 2026)]
    populations.append(("repaired_T1_primary", primary, "p"))
    confidence = []
    for name, frame, probability in populations:
        p, y = frame[probability].to_numpy(float), frame.y.to_numpy(float)
        favorite = np.maximum(p, 1-p)
        won = np.where(p >= .5, y, 1-y)
        months = pd.to_datetime(frame.date, utc=True).dt.strftime("%Y-%m")
        for low, high in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0001)):
            keep = (favorite >= low) & (favorite < high)
            interval = block_ratio_interval(np.where(keep, favorite-won, 0.), keep.astype(float), months, 5000, 20260930)
            confidence.append({"population":name, "band":f"{low:.1f}-{min(high,1):.1f}", "n":int(keep.sum()),
                "expected_upsets":float((1-favorite[keep]).sum()), "observed_upsets":int((1-won[keep]).sum()),
                "mean_confidence":float(favorite[keep].mean()) if keep.any() else None,
                "accuracy":float(won[keep].mean()) if keep.any() else None, "overconfidence_interval":interval})
    selected_ids = set(selected.fixture_id)
    p, q, y = (primary[column].to_numpy(float) for column in ("p", "q_consensus_a", "y"))
    a0_loss = binary_log_loss_vector(y, p, epsilon=1e-12)
    market_loss = binary_log_loss_vector(y, q, epsilon=1e-12)
    favorite, won = np.maximum(p, 1-p), np.where(p >= .5, y, 1-y)
    membership = primary.fixture_id.isin(selected_ids).to_numpy()
    selection = []
    for name, keep in (("all", np.ones(len(primary), bool)), ("selected", membership), ("unselected", ~membership)):
        delta = a0_loss-market_loss
        selection.append({"population":name, "n":int(keep.sum()), "ll_a0":float(a0_loss[keep].mean()),
            "ll_market":float(market_loss[keep].mean()), "excess_ll_sum":float(delta[keep].sum()),
            "mean_abs_market_gap":float(abs(p[keep]-q[keep]).mean()),
            "delta_ll_interval":block_ratio_interval(np.where(keep, delta, 0.), keep.astype(float), primary.month, 5000, 20260930),
            "favorite_bias_interval":block_ratio_interval(np.where(keep, favorite-won, 0.), keep.astype(float), primary.month, 5000, 20260930)})
    result = {"status":"COMPLETE", "confidence":confidence, "selection":selection,
        "selected_side":{"n":len(selected), "mean_p":float(selected.p_bet.mean()), "mean_q":float(selected.q_bet.mean()), "win_rate":float(selected.won.mean())},
        "limitations":["Canonical annual forecasts assume actual first-map rosters; repaired forecasts use prior lineups. These populations are not interchangeable.",
                       "Positive selected disagreement is not a causal treatment; subgroup metrics are development diagnostics.",
                       "Intervals retain the complete preselection monthly universe."]}
    pd.json_normalize(confidence).to_csv(output / "confidence_by_year.csv", index=False)
    pd.json_normalize(selection).to_csv(output / "selection_context.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    confidence_frame = pd.DataFrame(confidence)
    for name, group in confidence_frame.groupby("population", sort=False):
        axes[0].plot(group.mean_confidence, group.accuracy, marker="o", label=f"{name} (n={group.n.sum()})")
    axes[0].plot([.5, 1], [.5, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="Mean A0 favorite probability", ylabel="Observed favorite win rate", title="Global confidence is not selected-bet calibration")
    axes[0].legend(fontsize=7)
    axes[1].bar(["A0 expectation", "Market expectation", "Observed"], [selected.p_bet.mean(), selected.q_bet.mean(), selected.won.mean()], color=["#2878a0", "#777777", "#a04a3b"])
    axes[1].set(ylim=(0, .6), ylabel="Selected-side win probability / rate", title=f"Fixed raw EV>5% selections: n={len(selected)}")
    fig.suptitle("Frozen A0: annual actual-roster context versus repaired prior-roster T−1h selections")
    fig.tight_layout()
    fig.savefig(output / "global_vs_selected_calibration.png", dpi=150)
    plt.close(fig)
    return result


def verify_evidence(tables: dict, protocol: dict, output: Path) -> dict:
    """Independently reconcile saved selections, orientations and contributions."""
    from src.analysis.bootstrap import block_ratio_interval
    report = ROOT / protocol["report_dir"]
    primary = tables["opportunities"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.audit_primary_eligible].copy()
    selected = tables["bets"].loc[lambda frame:
        frame.cohort.eq("oddspapi_common") & frame.source.eq("best_price")
        & frame.horizon.eq("T-1h") & frame.quality.eq("exclude_q25_conflicts")
        & frame.policy.eq("raw_ev_gt_5")].copy()
    # Float diagnostics leave the persisted Decimal ledger untouched.
    for column in ("odds", "ev_bet"):
        selected[column] = pd.to_numeric(selected[column]).astype(float)
    assert primary.fixture_id.is_unique and selected.fixture_id.is_unique
    tax_factor = 1 - protocol["fixed_contract"]["tax_rate"]
    epsilon = protocol["fixed_contract"]["probability_epsilon"]
    ev_a = tax_factor * primary.odds_a * primary.p_event - 1
    ev_b = tax_factor * primary.odds_b * (1 - primary.p_event) - 1
    reconstructed = primary.loc[np.maximum(ev_a, ev_b) > .05, ["fixture_id"]].copy()
    reconstructed["side"] = np.where(ev_a.loc[reconstructed.index] >= ev_b.loc[reconstructed.index], "A", "B")
    assert set(map(tuple, reconstructed.to_numpy())) == set(map(tuple, selected[["fixture_id", "side"]].to_numpy()))
    predicted_ev = tax_factor * selected.odds * selected.p_model_bet - 1
    realized = tax_factor * selected.odds * selected.won - 1
    error = predicted_ev - realized
    np.testing.assert_allclose(predicted_ev, selected.ev_bet, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        error, tax_factor * selected.odds * (selected.p_model_bet - selected.q_bet)
        + tax_factor * selected.odds * (selected.q_bet - selected.won), atol=1e-12, rtol=0,
    )
    exported = pd.read_parquet(report / "profiles/primary_fixture_profiles.parquet")
    joined = exported.merge(primary[["fixture_id", "p_event", "y_event", "q_consensus_a"]],
                            on="fixture_id", validate="one_to_one", suffixes=("_export", "_input"))
    for name in ("p_event", "y_event", "q_consensus_a"):
        np.testing.assert_array_equal(joined[name + "_export"], joined[name + "_input"])
    p = primary.p_event.to_numpy(float).clip(epsilon, 1-epsilon)
    q = primary.q_consensus_a.to_numpy(float).clip(epsilon, 1-epsilon)
    y = primary.y_event.to_numpy(float)
    # Direct arithmetic is an independent oracle, not another production metric API.
    excess_ll = -y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(q)+(1-y)*np.log1p(-q)
    favorite_bias = np.maximum(p, 1-p) - np.where(p >= .5, y, 1-y)
    profiles = pd.read_csv(report / "profiles/profiles.csv")
    for name, group in profiles.groupby("profile"):
        is_selected = name.startswith("selected_")
        assert group.n.sum() == (len(selected) if is_selected else len(primary)), name
        if is_selected:
            np.testing.assert_allclose(group.selected_ev_error_sum.sum(), -error.sum(), atol=1e-10, rtol=0)
        else:
            np.testing.assert_allclose(group.excess_log_loss_sum.sum(), excess_ll.sum(), atol=1e-10, rtol=0)
            np.testing.assert_allclose(group.signed_error_sum.sum(), favorite_bias.sum(), atol=1e-10, rtol=0)
    expert = pd.read_parquet(report / "experts/expert_predictions.parquet")
    capture = json.loads((report / "experts/capture_receipt.json").read_text())
    np.testing.assert_allclose(expert[["p_neural", "p_rating", "p_mlp"]].to_numpy() @
                               np.array(capture["mixtures"][0]["weights"]), expert.p_reproduced, atol=1e-12, rtol=0)
    assert capture["max_abs_final_probability_difference"] == 0
    canonical = tables["canonical"]
    market_rows = canonical.loc[canonical.market_q.notna()]
    np.testing.assert_allclose(market_rows.market_q,
        np.where(market_rows.p_full >= .5, market_rows.market_open, 1-market_rows.market_open), atol=1e-12, rtol=0)
    months = sorted(primary.month.unique())
    intervals = []
    for population, frame, values in (
        ("all", primary, {"excess_ll": excess_ll, "favorite_bias": favorite_bias}),
        ("selected", selected, {"ev_overstatement": error.to_numpy(float)}),
    ):
        groups = [("all", "all", np.ones(len(frame), bool))]
        groups += [("competition_family", str(key), frame.competition_family.eq(key).to_numpy())
                   for key in sorted(primary.competition_family.dropna().unique())]
        groups += [("best_of", str(key), frame.best_of.eq(key).to_numpy())
                   for key in sorted(primary.best_of.dropna().unique())]
        if population == "selected":
            groups += [("ev_band", label, mask.to_numpy()) for label, mask in (
                ("(5%,10%]", frame.ev_bet.gt(.05) & frame.ev_bet.le(.10)),
                ("(10%,20%]", frame.ev_bet.gt(.10) & frame.ev_bet.le(.20)),
                (">20%", frame.ev_bet.gt(.20)),
            )]
        for dimension, group, keep in groups:
            for metric, vector in values.items():
                stats = pd.DataFrame({"month": frame.month.to_numpy(),
                    "numerator": np.where(keep, vector, 0.), "denominator": keep.astype(float)})
                stats = stats.groupby("month")[["numerator", "denominator"]].sum().reindex(months, fill_value=0)
                interval = block_ratio_interval(stats.numerator, stats.denominator, stats.index, 5000, 20260930)
                intervals.append({"population": population, "dimension": dimension, "group": group,
                                  "metric": metric, "n": int(keep.sum()), **interval})
    pd.DataFrame(intervals).to_csv(output / "profile_intervals.csv", index=False)
    return {"status": "PASS", "primary_n": len(primary), "fixed_selection_n": len(selected),
        "reconstructed_selection_ids_and_sides_match": True,
        "canonical_export_orientation_matches": True, "disjoint_profiles_checked": int(profiles.profile.nunique()),
        "selected_mean_ev": float(predicted_ev.mean()), "selected_realized_return": float(realized.mean()),
        "selected_ev_overstatement": float(error.mean()), "total_excess_log_loss": float(excess_ll.sum()),
        "expert_mixture_and_saved_prediction_parity": True, "canonical_market_q_is_favorite_oriented": True,
        "interval_rows": len(intervals), "interval_universe_months": months,
        "source_hashes": {key: record["sha256"] for key, record in protocol["sources"].items()},
        "limitations": ["Subgroup intervals are exploratory, without multiplicity correction.",
            "Empty block resamples retain undefined intervals; source tournament labels are not certified editions.",
            "A passing numerical audit is not certification of historical quote execution or roster announcements."]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--track", choices=("prepare", "profiles", "mechanisms", "market", "context", "experts", "verify", "roster_context"), required=True)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol, args.research_root)
    if args.track == "prepare":
        result = prepare(protocol)
    else:
        from importlib import import_module
        output = ROOT / protocol["report_dir"] / args.track
        if (output / "results.json").exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True, exist_ok=True)
        tables = read_tables(protocol)
        if args.track == "context":
            result = global_context(tables, protocol, output)
        elif args.track == "verify":
            result = verify_evidence(tables, protocol, output)
        else:
            module = import_module(f"src.analysis.a0_error_{args.track}")
            result = module.run(tables, protocol, output)
        save_json(output / "results.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
