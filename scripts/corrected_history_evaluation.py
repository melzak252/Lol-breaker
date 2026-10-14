#!/usr/bin/env python3
"""Replay NEW corrected GOLGG history and compare explicitly new research replicas.

No database, replay cache, old rating CSV, frozen fitted weights, or market-derived
features enter training. Both CLI stages require a previously nonexistent directory.
Calendar-day snapshots are retrospective diagnostics, never promotion evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from itertools import groupby
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_siamese_research_dataset import build_dataset, sha256, _history_helpers
from scripts.benchmark_siamese_architectures import (
    attach_market, monthly_bootstrap, probability_metrics, temporal_masks,
)
from scripts.train_and_tune_siamese_series import (
    build_model_artifact, build_training_features, fit_platt_scaling,
    train_single_member, write_json_exclusive,
)
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.team_order import swap_orientation, symmetrize_binary_probabilities
from src.utils import golgg_schema as schema
from src.utils.module_loading import load_module_from_path

RATINGS_VERSION = "corrected-history-six-family-daybatch-v1"
EXP039_VERSION = "exp039-corrected-history-research-v1"
EXP081_VERSION = "exp081-corrected-history-research-v1"
SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
FROZEN = tuple(ROOT / "betting_app/models" / name for name in (
    "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib",
    "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib",
))


def frozen_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in FROZEN}


def code_hashes():
    paths = [Path(__file__), ROOT / "scripts/build_siamese_research_dataset.py",
             ROOT / "scripts/prospective_sports_features.py",
             ROOT / "scripts/train_and_tune_siamese_series.py",
             ROOT / "scripts/benchmark_siamese_architectures.py",
             ROOT / "scripts/05_ratingi_baseline/03_generate_ratings.py",
             ROOT / "scripts/06_metamodel/06i_best_metamodel_config_search.py",
             ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py",
             ROOT / "src/models/symmetric_series.py", ROOT / "src/models/team_order.py",
             ROOT / "src/utils/golgg_schema.py"]
    paths.extend(sorted((ROOT / "src/ratings").glob("*.py")))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def game_rosters(game):
    """Require IDs, not ambiguous player-name fallbacks, for both actual map sides."""
    sides = []
    for side in ("t1", "t2"):
        payload = game.get(f"{side}_players")
        if not isinstance(payload, dict) or len(payload) != 5:
            raise ValueError("incomplete map roster")
        roster = []
        for player in payload.values():
            identity = player.get("player_id", player.get("id"))
            if identity is None or str(identity).strip() == "":
                raise ValueError("missing exact player ID")
            roster.append(str(identity))
        sides.append(roster)
    if len(set(sides[0] + sides[1])) != 10:
        raise ValueError("duplicate or overlapping map rosters")
    return sides


def validate_series(match):
    """Reject unresolved labels/identities instead of repairing or guessing them."""
    if not isinstance(match, dict) or match.get("match_id") is None or not str(match["match_id"]).strip():
        raise ValueError("missing match identity")
    date.fromisoformat(match["date"])
    t1, t2 = schema.team1_id(match), schema.team2_id(match)
    explicit1 = schema.first_present(match, ("tid_1", "t1_id"))
    explicit2 = schema.first_present(match, ("tid_2", "t2_id"))
    if explicit1 is None or explicit2 is None or not t1 or not t2 or t1 == t2:
        raise ValueError("missing or duplicate explicit series team IDs")
    if match.get("score_alignment") == "unresolved":
        raise ValueError("unresolved source score alignment")
    bo = schema.best_of(match)
    maps = schema.games(match)
    if bo not in (1, 2, 3, 5) or not maps:
        raise ValueError("unfinished or unsupported series")
    raw_initial = match.get("initial_wins")
    if raw_initial is not None:
        if not isinstance(raw_initial, dict):
            raise ValueError("invalid initial map advantage")
        for k, v in raw_initial.items():
            if not isinstance(v, int) or v < 0:
                raise ValueError("invalid initial map advantage")
    initial = schema.initial_wins(match)
    for tid in initial:
        if tid not in (t1, t2):
            raise ValueError(f"initial_wins team ID {tid} not in series teams ({t1}, {t2})")
    adv1 = initial.get(t1, 0)
    adv2 = initial.get(t2, 0)
    target = bo // 2 + 1 if bo != 2 else bo
    if adv1 < 0 or adv2 < 0 or (adv1 > 0 and adv2 > 0) or (bo != 2 and (adv1 >= target or adv2 >= target)):
        raise ValueError("invalid initial map advantage")
    seen = set()
    scores = []
    for game in maps:
        gid = str(game.get("game_id") or "")
        if not gid or gid in seen:
            raise ValueError("missing or duplicate game IDs")
        seen.add(gid)
        if {str(game.get("t1_id")), str(game.get("t2_id"))} != {t1, t2}:
            raise ValueError("map team identity mismatch")
        if (type(game.get("t1_win")) is not bool or type(game.get("t2_win")) is not bool
                or game["t1_win"] == game["t2_win"] or game.get("draw")):
            raise ValueError("invalid map winner flags")
        game_rosters(game)
        scores.append(schema.game_score_for_match_team1(match, game))
        if bo != 2 and len(scores) < len(maps) and max(sum(scores) + adv1, len(scores) - sum(scores) + adv2) >= target:
            raise ValueError("extra map after completed series")
    wins = sum(scores)
    losses = len(scores) - wins
    total1 = wins + adv1
    total2 = losses + adv2
    if bo == 2:
        if len(maps) != 2:
            raise ValueError("incomplete BO2 state-only series")
    elif max(total1, total2) != target or min(total1, total2) > bo // 2 or match.get("draw"):
        raise ValueError("incomplete series map count")
    for keys, expected in ((("score_1", "t1_score"), wins), (("score_2", "t2_score"), losses)):
        value = schema.first_present(match, keys)
        if value is None or float(value) != expected:
            raise ValueError("series header score does not align with exact map winners")
    for key, expected in (("t1_win", total1 > total2), ("t2_win", total2 > total1)):
        if key in match and (type(match[key]) is not bool or match[key] != expected):
            raise ValueError("series winner flags conflict with maps")
    return scores


def first_rosters(match):
    game = schema.games(match)[0]
    a, b = game_rosters(game)
    return (a, b) if str(game["t1_id"]) == schema.team1_id(match) else (b, a)


def replay_ratings(matches, *, progress=False):
    """Freeze the entire day's decayed, pre-result state before any update.

    Later-day states consume maps in source series/map order, sorting series by
    ID within a date. That ordering is deterministic, not an asserted event time.
    Each Glicko opponent update uses its paired pre-game state. Substitutes update
    only the games in which they actually appeared.
    """
    if len({str(m["match_id"]) for m in matches}) != len(matches):
        raise ValueError("duplicate series IDs")
    for match in matches:
        validate_series(match)
    baseline = load_module_from_path(
        ROOT / "scripts/05_ratingi_baseline/03_generate_ratings.py", "corrected_rating_parameters"
    )
    manager = baseline.create_rating_manager()
    rows = []
    processed_series = processed_maps = 0
    ordered = sorted(matches, key=lambda m: (m["date"], str(m["match_id"])))
    for day, grouped in groupby(ordered, key=lambda m: m["date"]):
        period = list(grouped)
        current = date.fromisoformat(day)
        previous_dates = manager.last_match_date.copy()
        # Decay each participant once, including later-map substitutes, before
        # ANY snapshot. The manager records dates but consumes no result here.
        for match in period:
            for game in schema.games(match):
                p1, p2 = game_rosters(game)
                manager.update_before_match(str(game["t1_id"]), str(game["t2_id"]), p1, p2, current)
        for match in period:
            if schema.best_of(match) == 2:
                continue  # Valid BO2 outcomes update state but are not binary-series targets.
            t1, t2 = schema.team1_id(match), schema.team2_id(match)
            p1, p2 = first_rosters(match)
            prediction = manager.predict_match(t1, t2, p1, p2)
            swapped = manager.predict_match(t2, t1, p2, p1)
            for family in SYSTEMS:
                for entity in ("team", "player"):
                    field = f"{entity}_{family}"
                    # Existing TrueSkill CDF differs from 0.5 by ~1.5e-8 at zero.
                    # Enforce the project's 1e-6 contract without changing ratings.
                    if abs(prediction[field] + swapped[field] - 1) > 1e-6:
                        raise ValueError(f"rating side symmetry failed: {field}")
            days1 = (current - previous_dates[t1]).days if t1 in previous_dates else 30
            days2 = (current - previous_dates[t2]).days if t2 in previous_dates else 30
            prediction.update(
                golgg_match_id=str(match["match_id"]), date=day,
                team1_id=t1, team2_id=t2, team1_name=schema.team1_name(match),
                team2_name=schema.team2_name(match), BoN=schema.best_of(match),
                days_since_last_1=days1, days_since_last_2=days2, days_diff=days1-days2,
                y_true=int(sum(validate_series(match)) + schema.initial_wins(match).get(t1, 0) > len(schema.games(match)) - sum(validate_series(match)) + schema.initial_wins(match).get(t2, 0)),
            )
            rows.append(prediction)
        for match in period:
            for game in schema.games(match):
                t1, t2 = str(game["t1_id"]), str(game["t2_id"])
                p1, p2 = game_rosters(game)
                score = int(game["t1_win"])
                manager.update_after_game(t1, t2, p1, p2, score, 1-score)
                manager.update_after_match(t1, t2, p1, p2, [score])
                processed_maps += 1
            processed_series += 1
        if progress:
            print(json.dumps({"stage": "ratings", "date": day, "state_series": processed_series,
                              "state_maps": processed_maps, "scoring_rows": len(rows), "total_series": len(matches)}), flush=True)
    return pd.DataFrame(rows)


def replay(source, output_dir):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if source == (ROOT / "data/golgg_matches.json").resolve():
        raise ValueError("use a separately named fresh corrected source, not the legacy dataset")
    raw = json.loads(source.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("source must be a nonempty JSON series list")
    frozen = frozen_hashes()
    source_digest = sha256(source)
    collection = {"status": "not_provided", "complete": False, "files": {}}
    manifest_path = source.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        counts = manifest["counts"]
        if manifest["status"] not in ("complete", "incomplete") or counts["pending_matches"]:
            raise ValueError("unfinished collection cannot be used for historical evaluation")
        if counts["clean_matches"] != len(raw):
            raise ValueError("collection clean count does not match source dataset")
        if counts["discovered_matches"] != counts["clean_matches"] + counts["unresolved_matches"]:
            raise ValueError("collection contains unaccounted source identities")
        collection = {
            "status": manifest["status"], "counts": counts,
            "complete": manifest["status"] == "complete"
            and counts["unresolved_matches"] == 0 and counts["discovery_failures"] == 0,
            "files": {str(path): sha256(path) for path in (
                manifest_path, source.with_suffix(".quarantine.json")
            )},
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter(str(m.get("match_id")) for m in raw if isinstance(m, dict))
    game_counts = Counter(str(g.get("game_id")) for m in raw if isinstance(m, dict) for g in schema.games(m))
    valid, rejected = [], []
    for match in raw:
        try:
            validate_series(match)
            if counts[str(match["match_id"])] != 1:
                raise ValueError("duplicate source series ID")
            if any(game_counts[str(g["game_id"])] != 1 for g in schema.games(match)):
                raise ValueError("game ID reused across series")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            rejected.append({"reason": str(exc), "series": match})
        else:
            valid.append(match)
    write_json_exclusive(output_dir / "rejected_series.json", {"n": len(rejected), "rows": rejected})
    if not valid:
        raise ValueError("no valid complete series; see rejected_series.json")
    ratings = replay_ratings(valid, progress=True)
    # First-map roster is the declared retrospective prediction roster; top-level
    # stale legacy roster fields must not override the exact per-map identities.
    for match in valid:
        match["players_1"], match["players_2"] = first_rosters(match)
    snapshots, audit = build_dataset(ratings, valid, day_batched_ratings=True)
    ratings.to_csv(output_dir / "ratings.csv", index=False)
    snapshots.to_csv(output_dir / "snapshots.csv", index=False)
    eligible = set(snapshots.golgg_match_id) if len(snapshots) else set()
    excluded = [m for m in valid if str(m["match_id"]) not in eligible]
    write_json_exclusive(output_dir / "feature_excluded_series.json", {"n": len(excluded), "rows": excluded})
    if frozen_hashes() != frozen or sha256(source) != source_digest:
        raise ValueError("input or frozen artifact changed during replay")
    if any(sha256(Path(path)) != digest for path, digest in collection["files"].items()):
        raise ValueError("collection provenance changed during replay")
    if not collection["complete"]:
        audit["promotion_blockers"].append(
            "collection completeness unverified or quarantined transitions missing; see collection provenance"
        )
    if rejected:
        audit["promotion_blockers"].append(
            "rejected source series create unobserved rating/history transitions; retained separately, not reconstructed"
        )
    audit.update(
        ratings_version=RATINGS_VERSION,
        collection=collection,
        source={"path": str(source), "sha256": source_digest, "raw_series": len(raw)},
        rejected_series=len(rejected), rejected_reasons=dict(Counter(r["reason"] for r in rejected)),
        feature_excluded_series=len(excluded),
        state_transition_series=len(valid),
        state_transition_maps=sum(len(schema.games(m)) for m in valid),
        state_only_series=sum(schema.best_of(m) == 2 for m in valid),
        emitted_binary_rating_rows=len(ratings),
        rating_families=list(SYSTEMS), prediction_entities=["team", "player"],
        rating_policy="same-day pre-result snapshots; actual-date Glicko weekly decay; exact per-map rosters; simultaneous opponent states",
        intraday_update_policy="deterministic series-ID/source-map order after all daily snapshots; event times unknown",
        roster_policy="first map observed roster for series forecast; substitutions updated on actual maps",
        legacy_feature_reuse=False, frozen_artifacts_before=frozen, frozen_artifacts_after=frozen_hashes(),
        code_sha256=code_hashes(),
        outputs={name: sha256(output_dir / name) for name in (
            "ratings.csv", "snapshots.csv", "rejected_series.json", "feature_excluded_series.json"
        )},
    )
    write_json_exclusive(output_dir / "audit.json", audit)
    print(json.dumps({"stage": "replay_complete", "directory": str(output_dir), "ratings": len(ratings), "snapshots": len(snapshots), "rejected": len(rejected)}), flush=True)


def verify_replay(directory):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("ratings_version") != RATINGS_VERSION or audit.get("legacy_feature_reuse") is not False:
        raise ValueError("not a verified corrected replay")
    for name in ("ratings.csv", "snapshots.csv", "rejected_series.json", "feature_excluded_series.json"):
        if sha256(directory / name) != audit["outputs"][name]:
            raise ValueError(f"corrected replay checksum mismatch: {name}")
    if sha256(Path(audit["source"]["path"])) != audit["source"]["sha256"]:
        raise ValueError("corrected replay source changed")
    if audit["frozen_artifacts_before"] != frozen_hashes() or audit["frozen_artifacts_after"] != frozen_hashes():
        raise ValueError("frozen artifact changed since replay")
    if any(sha256(Path(path)) != digest for path, digest in audit["collection"]["files"].items()):
        raise ValueError("collection provenance changed since replay")
    return audit


def build_exp039_features(frame):
    helper = load_module_from_path(
        ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py", "corrected_exp039_helpers"
    )
    history = _history_helpers()
    enriched, binomial = helper.add_binomial_features(frame)
    names = history.OPTUNA_BASE_FEATURES + history.ROLLING_FULL_FEATURES + binomial
    if len(names) != 46 or not np.isfinite(enriched[names].to_numpy(float)).all():
        raise ValueError("EXP039 requires exactly 46 complete finite established features")
    return enriched, names, helper.RANK_PROB_FEATURES


def chronological_masks(dates, year):
    return temporal_masks(dates, year)


def exp039_probability(model, frame, names, rank):
    swapped = swap_orientation(frame, names, rank, np.ones(len(frame), dtype=bool))
    original = np.clip(model.predict_proba(frame[names])[:, 1], 0.001, 0.999)
    reverse = np.clip(model.predict_proba(swapped[names])[:, 1], 0.001, 0.999)
    return symmetrize_binary_probabilities(original, reverse)


def reliability_bins(y, p):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * 10).astype(int), 9)
    return [{"bin": b, "n": int((bins == b).sum()),
             "mean_probability": float(p[bins == b].mean()), "observed_rate": float(y[bins == b].mean())}
            for b in range(10) if np.any(bins == b)]


def paired_ci(result, mask):
    part = result.loc[mask]
    if pd.to_datetime(part.date).dt.to_period("M").nunique() < 2:
        return {"n": len(part), "unavailable": "fewer than two monthly blocks"}
    y, a, b = part.y_true, part[EXP039_VERSION], part[EXP081_VERSION]
    return {"n": len(part), "direction": "EXP081 replica minus EXP039 replica; negative loss difference favors EXP081",
            "log_loss": monthly_bootstrap(binary_log_loss_vector(y, b)-binary_log_loss_vector(y, a), part.date, 5000),
            "brier": monthly_bootstrap((b-y)**2-(a-y)**2, part.date, 5000)}


def evaluate(replay_dir, output_dir, *, epochs=35, odds=None, first_year=2024):
    import joblib
    from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

    if epochs < 1 or first_year < 2024:
        raise ValueError("positive epochs and first test year >=2024 required")
    replay_dir, output_dir = Path(replay_dir).resolve(), Path(output_dir).resolve()
    audit = verify_replay(replay_dir)
    frozen = frozen_hashes()
    frame = pd.read_csv(replay_dir / "snapshots.csv", dtype={"golgg_match_id": str, "team1_id": str, "team2_id": str})
    frame = frame[frame.date >= "2020-01-01"].sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    if frame.empty or frame.golgg_match_id.duplicated().any():
        raise ValueError("empty or duplicate corrected snapshots")
    x39, names39, rank = build_exp039_features(frame)
    x81, names81 = build_training_features(frame)
    reference81 = json.loads(FROZEN[2].read_text())
    architecture = reference81["architecture"]
    if (architecture["d_in"], architecture["d_h1"], architecture["d_h2"], architecture["activation"], architecture["n_members"], architecture["loss"]) != (79, 32, 16, "relu", 5, "focal_loss_gamma_1.0"):
        raise ValueError("checked-in EXP081 architecture differs from supported exact family")
    if names81 != reference81["scaler"]["feature_names"]:
        raise ValueError("EXP081 canonical feature order differs from frozen specification")
    output_dir.mkdir(parents=True, exist_ok=False)
    helper = load_module_from_path(ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py", "corrected_lr_builder")
    y = frame.y_true.to_numpy(float)
    blocks, folds = [], []
    seeds = [20260906 + 101*i for i in range(5)]
    for year in range(first_year, pd.Timestamp(frame.date.max()).year + 1):
        fit, cal, test = chronological_masks(frame.date, year)
        if not test.any():
            continue
        if not fit.any() or not cal.any() or len(np.unique(y[fit])) != 2 or len(np.unique(y[cal])) != 2:
            raise ValueError(f"{year}: insufficient earlier fit/calibration partitions")
        model = helper.build_logistic_regression()
        model.fit(x39.loc[fit, names39], y[fit])
        cal39 = exp039_probability(model, x39.loc[cal], names39, rank)
        slope39 = fit_platt_scaling(logit(cal39), y[cal])
        raw39 = exp039_probability(model, x39.loc[test], names39, rank)
        p39 = np.clip(expit(slope39 * logit(raw39)), 1e-12, 1-1e-12)
        flipped39 = swap_orientation(x39.loc[test], names39, rank, np.ones(test.sum(), dtype=bool))
        reverse39 = np.clip(expit(slope39 * logit(exp039_probability(model, flipped39, names39, rank))), 1e-12, 1-1e-12)
        scales = x81[fit].std(axis=0)
        scales[scales == 0] = 1
        members = []
        for seed in seeds:
            members.append(train_single_member(x81[fit]/scales, y[fit], x81[cal]/scales, y[cal],
                                               epochs=epochs, gamma=1.0, d_h1=32, d_h2=16, seed=seed))
            print(json.dumps({"stage": "fit_exp081", "year": year, "member": len(members), "members": 5}), flush=True)
        logits = np.stack([m.forward_anti_symmetric(x81[test]/scales)[0].ravel()*m.platt_slope for m in members])
        reverse = np.stack([m.forward_anti_symmetric(-x81[test]/scales)[0].ravel()*m.platt_slope for m in members])
        z, sigma = logits.mean(axis=0), logits.std(axis=0, ddof=0)
        p81 = np.clip(expit(z), 1e-12, 1-1e-12)
        reverse81 = np.clip(expit(reverse.mean(axis=0)), 1e-12, 1-1e-12)
        errors = {EXP039_VERSION: float(np.max(abs(p39+reverse39-1))), EXP081_VERSION: float(np.max(abs(p81+reverse81-1)))}
        if max(errors.values()) > 1e-8:
            raise ValueError("model probability side symmetry failed")
        part = frame.loc[test].copy()
        family_slopes = {}
        for family in SYSTEMS:
            for entity in ("team", "player"):
                field = f"{entity}_{family}"
                cal_probability = helper.series_probability(
                    frame.loc[cal, field].to_numpy(float), frame.loc[cal, "BoN"].to_numpy(int)
                )
                raw_probability = helper.series_probability(
                    frame.loc[test, field].to_numpy(float), frame.loc[test, "BoN"].to_numpy(int)
                )
                slope = fit_platt_scaling(logit(cal_probability), y[cal])
                family_slopes[field] = slope
                part[f"{field}_series_raw"] = raw_probability
                part[f"{field}_series_calibrated"] = np.clip(
                    expit(slope * logit(raw_probability)), 1e-12, 1-1e-12
                )
        part[EXP039_VERSION], part[EXP081_VERSION] = p39, p81
        part[f"{EXP039_VERSION}_b"], part[f"{EXP081_VERSION}_b"] = 1-p39, 1-p81
        part["exp081_sigma_z"] = sigma
        part["exp081_low_a"] = expit(z - 0.75*sigma)
        part["exp081_low_b"] = expit(-z - 0.75*sigma)
        part["fold"] = year
        blocks.append(part)
        fold = {"year": year, "symmetry_max_error": errors, "exp039_calibration_slope": slope39,
                "standalone_family_calibration_slopes": family_slopes}
        for name, mask in (("fit", fit), ("calibration", cal), ("test", test)):
            fold[name] = {"n": int(mask.sum()), "date_min": frame.loc[mask, "date"].min(), "date_max": frame.loc[mask, "date"].max()}
        folds.append(fold)
        with (output_dir / f"{EXP039_VERSION}-{year}.joblib").open("xb") as stream:
            joblib.dump({"model_version": f"{EXP039_VERSION}-{year}", "pipeline": model,
                         "calibration_slope": slope39, "features": names39, "fold": fold,
                         "symmetry": "average original and complemented swapped probabilities, then positive slope-only Platt"}, stream)
        artifact = build_model_artifact(members, names81, scales, provenance={"fold": fold, "replay_audit_sha256": sha256(replay_dir / "audit.json")})
        artifact.update(model_version=f"{EXP081_VERSION}-{year}", model_name="EXP081-Corrected-History-Research-Replica")
        artifact["architecture"]["loss"] = "focal_loss_gamma_1.0"
        artifact["training"] = {"epochs": epochs, "batch_size": 64, "lr": 0.003, "seeds": seeds,
                                "selection": "fixed epochs; no test or calibration checkpoint selection"}
        write_json_exclusive(output_dir / f"{EXP081_VERSION}-{year}.json", artifact)
        print(json.dumps({"stage": "fold_complete", "year": year, "test_n": int(test.sum())}), flush=True)
    if not blocks:
        raise ValueError("no 2024+ evaluation rows")
    result = pd.concat(blocks, ignore_index=True)
    if odds is not None:
        result, market_audit = attach_market(result, pd.read_csv(odds, dtype={"golgg_match_id": str}))
        market_audit["source"] = {"path": str(Path(odds).resolve()), "sha256": sha256(Path(odds))}
        market_audit["timing"] = "unverified historical aggregate odds; not a verified closing benchmark"
    else:
        result["market"], result["odds_a"], result["odds_b"] = np.nan, np.nan, np.nan
        market_audit = {"available": False, "blocker": "no separately supplied identity-audited market odds"}
    ref = result[EXP039_VERSION]
    player = result[[f"player_{s}" for s in SYSTEMS]].mean(axis=1)
    team = result[[f"team_{s}" for s in SYSTEMS]].mean(axis=1)
    masks = {"overall": np.ones(len(result), bool),
             "confidence_heavy": (ref >= .75) | (ref <= .25),
             "confidence_moderate": ref.between(.6, .75, inclusive="left") | ref.between(.25, .4, inclusive="right"),
             "confidence_close": ref.between(.45, .55),
             "roster_stable": result.roster_min_prior_series >= 10,
             "roster_rookie": result.roster_min_prior_series < 10,
             "signals_agree": abs(player-team) <= .08, "signals_disagree": abs(player-team) > .15,
             "history_under20": (result.history_games_1 < 20) | (result.history_games_2 < 20),
             "history_full20": (result.history_games_1 == 20) & (result.history_games_2 == 20),
             "market_common": result.market.notna(),
             "odds_underdog_3p5_5": result.odds_a.between(3.5, 5) | result.odds_b.between(3.5, 5)}
    masks.update({f"bo{bo}": result.best_of == bo for bo in (1, 3, 5)})
    masks.update({f"tier_{tier}": result.competition_tier == tier for tier in result.competition_tier.unique()})
    masks.update({f"year_{year}": result.fold == year for year in result.fold.unique()})
    masks.update({f"tournament_{name}": result.tournament == name for name in result.tournament.dropna().unique()})
    metrics = {}
    for name, mask in masks.items():
        metrics[name] = {}
        for version in (EXP039_VERSION, EXP081_VERSION):
            yy, pp = result.loc[mask, "y_true"], result.loc[mask, version]
            values = probability_metrics(yy, pp)
            if len(yy):
                values.update(reliability_bins=reliability_bins(yy, pp),
                              pr_auc=float(average_precision_score(yy, pp)) if yy.nunique() == 2 else None,
                              precision=float(precision_score(yy, pp >= .5, zero_division=0)),
                              recall=float(recall_score(yy, pp >= .5, zero_division=0)),
                              confusion_matrix=confusion_matrix(yy, pp >= .5, labels=[0, 1]).tolist())
            metrics[name][version] = values
    family_metrics = []
    for cohort, mask in masks.items():
        if cohort != "overall" and not cohort.startswith(("year_", "bo", "tier_")):
            continue
        for family in SYSTEMS:
            for entity in ("team", "player"):
                for calibration in ("raw", "calibrated"):
                    column = f"{entity}_{family}_series_{calibration}"
                    family_metrics.append({
                        "cohort": cohort, "rating_version": RATINGS_VERSION,
                        "family": family, "entity": entity, "calibration": calibration,
                        **probability_metrics(result.loc[mask, "y_true"], result.loc[mask, column]),
                    })
    pd.DataFrame(family_metrics).to_csv(output_dir / "rating_family_metrics.csv", index=False)
    if frozen_hashes() != frozen:
        raise ValueError("frozen artifacts changed during evaluation")
    result.to_csv(output_dir / "predictions.csv", index=False)
    summary = {"experiment": "corrected-history-exp039-vs-exp081-research-v1", "scope": "retrospective_diagnostic_only",
               "versions": [EXP039_VERSION, EXP081_VERSION], "folds": folds, "metrics": metrics,
               "standalone_rating_family_metrics": family_metrics,
               "paired_monthly5000": {name: paired_ci(result, mask) for name, mask in masks.items()},
               "market_audit": market_audit,
               "market_metrics": probability_metrics(result.loc[result.market.notna(), "y_true"], result.loc[result.market.notna(), "market"]),
               "feature_names": {EXP039_VERSION: names39, EXP081_VERSION: names81},
               "protocol": {"common_cohort": "identical complete corrected snapshots for both models; fit dates >=2020; expanding annual test >=2024",
                            "split": "fit before preceding calendar year; calibration preceding calendar year; test target calendar year",
                            "calibration": "positive slope-only Platt on disjoint earlier calibration year; never fit-on-test",
                            "standalone_ratings": "all six families x player/team on SAME model holdout; established independent-map BoN transform, both uncalibrated and preceding-year slope-calibrated metrics",
                            "calibration_diagnostics": "test calibration slope/intercept are descriptive metrics only, never used to change probabilities",
                            "classification_threshold": "0.5; confusion/precision/recall ties choose team1; shared accuracy metric gives exact probability ties half credit",
                            "exp039_change": "established ElasticNet46 pipeline; order symmetrization followed by symmetric slope-only calibration, unlike potentially asymmetric historical intercept",
                            "exp081_architecture": architecture, "epochs": epochs, "seeds": seeds,
                            "original_exp081_reproduction_limit": "architecture and frozen feature contract available; original optimizer/epoch/batch/split membership/scaler-fit/calibration provenance absent from frozen artifact, so these are NEW research replicas, not original fitted EXP081",
                            "uncertainty": "exp081_low_a/b are heuristic kappa=.75 population-logit-SD bounds, not confidence guarantees",
                            "slice_reference": "EXP039 replica fixes confidence cohorts for both; no candidate-specific selection"},
               "dataset_audit": audit,
               "promotion": {"approved": False, "blockers": audit["promotion_blockers"] + [
                   "source observation and match start timestamps unavailable; day-only replay is retrospective",
                   "no point-in-time prediction/market quote ledger; ROI and deployability not established",
                   "frozen experiment metrics cannot be relabeled as corrected-history research results"]},
               "provenance": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "code_sha256": code_hashes(), "replay_audit_sha256": sha256(replay_dir / "audit.json"),
                              "legacy_feature_reuse": False, "frozen_weights_used_for_training_or_prediction": False,
                              "frozen_artifacts_before": frozen, "frozen_artifacts_after": frozen_hashes(),
                              "predictions_sha256": sha256(output_dir / "predictions.csv"),
                              "output_artifacts_sha256": {
                                  path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()
                              },
                              "rating_family_metrics_sha256": sha256(output_dir / "rating_family_metrics.csv")}}
    write_json_exclusive(output_dir / "summary.json", summary)
    print(json.dumps({"stage": "evaluation_complete", "directory": str(output_dir), "overall": metrics["overall"]}, allow_nan=False), flush=True)


def chronological_replay(source, output_dir, *, history_provenance=None):
    """Create a new strict source-semantic replay; never adapt or reuse old CSVs."""
    from scripts.export_prospective_features import iter_matches
    from scripts.prospective_sports_features import (
        FEATURE_VERSION, FEATURE_CONTRACT, build_chronological_training_rows,
    )
    from src.models.tournament_prediction import json_digest

    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = frozen_hashes()
    protocol = {
        "version": "chronological-replay-v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION, "feature_contract": dict(FEATURE_CONTRACT),
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "source": {"path": str(source), "sha256": sha256(source)},
        "code_sha256": code_hashes(), "frozen_artifacts": frozen,
        "qualification": "chronologically_reconstructed_research_only",
        "availability_certified": False, "collection_completeness_certified": False,
        "source_policy": "all supplied transitions validated; invalid history is fatal, never dropped",
    }
    if history_provenance is not None:
        provenance_path = Path(history_provenance).resolve()
        evidence = json.loads(provenance_path.read_text())
        if not isinstance(evidence, dict) or evidence.get("projection_sha256") != protocol["source"]["sha256"]:
            raise ValueError("history provenance must identify the exact supplied projection checksum")
        protocol["history_provenance"] = {
            "path": str(provenance_path), "sha256": sha256(provenance_path), "evidence": evidence,
        }
    write_json_exclusive(output_dir / "protocol.json", protocol)
    try:
        snapshots, audit = build_chronological_training_rows(iter_matches(source))
        if snapshots.empty:
            raise ValueError("no eligible chronological snapshots")
    except (ValueError, KeyError, TypeError) as exc:
        write_json_exclusive(output_dir / "failure.json", {
            "status": "failed", "reason": str(exc), "source_sha256": protocol["source"]["sha256"],
            "no_forecasts_or_models_produced": True,
        })
        raise
    snapshots.to_csv(output_dir / "snapshots.csv", index=False)
    write_json_exclusive(output_dir / "target_exclusions.json", {
        "rows": audit.pop("target_exclusions"),
        "policy": "feature-ineligible targets only; their valid observed maps remain in rating/W20 state",
    })
    if frozen_hashes() != frozen or sha256(source) != protocol["source"]["sha256"]:
        raise ValueError("source or frozen artifact changed during chronological replay")
    if history_provenance is not None and sha256(provenance_path) != protocol["history_provenance"]["sha256"]:
        raise ValueError("history provenance changed during chronological replay")
    audit.update(protocol, status="complete", outputs={
        name: sha256(output_dir / name) for name in ("snapshots.csv", "target_exclusions.json", "protocol.json")
    })
    write_json_exclusive(output_dir / "audit.json", audit)
    return audit


def verify_chronological_replay(directory):
    from scripts.prospective_sports_features import FEATURE_VERSION, FEATURE_CONTRACT
    from src.models.tournament_prediction import json_digest

    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    if (audit.get("version") != "chronological-replay-v2" or audit.get("status") != "complete"
            or audit.get("feature_version") != FEATURE_VERSION
            or audit.get("feature_contract") != FEATURE_CONTRACT
            or audit.get("feature_contract_sha256") != json_digest(FEATURE_CONTRACT)):
        raise ValueError("not a compatible complete chronological replay")
    for name in ("snapshots.csv", "target_exclusions.json", "protocol.json"):
        if sha256(directory / name) != audit["outputs"][name]:
            raise ValueError(f"chronological replay checksum mismatch: {name}")
    if sha256(Path(audit["source"]["path"])) != audit["source"]["sha256"]:
        raise ValueError("chronological replay source checksum mismatch")
    provenance = audit.get("history_provenance")
    if provenance and sha256(Path(provenance["path"])) != provenance["sha256"]:
        raise ValueError("history provenance changed since chronological replay")
    if audit["frozen_artifacts"] != frozen_hashes():
        raise ValueError("frozen artifacts changed since chronological replay")
    if audit["code_sha256"] != code_hashes():
        raise ValueError("chronological replay implementation changed; create a fresh replay")
    return audit


def chronological_fold_masks(frame, *, origin, calibration_start, fit_start):
    """Result completion, not just series start, bounds label availability."""
    from src.models.tournament_prediction import utc
    at = utc(origin).date()
    cal = date.fromisoformat(calibration_start)
    start = date.fromisoformat(fit_start)
    if not start < cal < at:
        raise ValueError("require fit_start < calibration_start < origin day")
    starts = pd.to_datetime(frame["date"], errors="raise")
    ends = pd.to_datetime(frame["result_day"], errors="raise")
    if starts.isna().any() or ends.isna().any() or (ends < starts).any():
        raise ValueError("missing or contradictory series completion days")
    fit = (starts >= pd.Timestamp(start)) & (ends < pd.Timestamp(cal))
    calibration = (starts >= pd.Timestamp(cal)) & (ends < pd.Timestamp(at))
    return fit.to_numpy(), calibration.to_numpy()


def train_chronological_fold(replay_dir, output_dir, *, origin, calibration_start, fit_start="2018-01-01", epochs=35):
    """Fit one immutable, explicitly scheduled EXP081 research fold on local CPU.

    Hyperparameters are fixed before fitting. A positive zero-intercept slope is
    fitted to the ensemble's mean raw logit on a disjoint earlier calendar block.
    No held-out tournament result participates in scaling, fitting or calibration.
    """
    from scripts.prospective_sports_features import FEATURE_VERSION, FEATURE_CONTRACT, MODEL_VERSION
    from src.models.tournament_prediction import json_digest, utc

    if epochs < 1:
        raise ValueError("epochs must be positive")
    replay_dir, output_dir = Path(replay_dir).resolve(), Path(output_dir).resolve()
    audit = verify_chronological_replay(replay_dir)
    frame = pd.read_csv(replay_dir / "snapshots.csv", dtype={
        "golgg_match_id": str, "team1_id": str, "team2_id": str,
        "team_a": str, "team_b": str,
    })
    if frame.empty or frame.golgg_match_id.duplicated().any():
        raise ValueError("empty or duplicate chronological snapshots")
    fit, cal = chronological_fold_masks(frame, origin=origin, calibration_start=calibration_start, fit_start=fit_start)
    frame = frame.loc[fit | cal].reset_index(drop=True)
    fit, cal = chronological_fold_masks(frame, origin=origin, calibration_start=calibration_start, fit_start=fit_start)
    x, names = build_training_features(frame)
    y = frame.y_true.to_numpy(float)
    if not fit.any() or not cal.any() or len(np.unique(y[fit])) != 2 or len(np.unique(y[cal])) != 2:
        raise ValueError("insufficient disjoint prior fit/calibration rows with both labels")
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = [20260906 + 101*i for i in range(5)]
    training = {
        "epochs": epochs, "batch_size": 64, "lr": 0.003, "gamma": 1.0,
        "d_h1": 32, "d_h2": 16, "seeds": seeds,
        "selection": "fixed recipe; no early stopping, tuning, tournament/test checkpoint selection",
        "preprocessing": "training-only standard deviation; zero centering; zero-scale columns use scale 1",
        "roster_policy": "previous_observed_roster_proxy",
    }
    fold, memberships = {}, {}
    for label, mask in (("fit", fit), ("calibration", cal)):
        ids = frame.loc[mask, "golgg_match_id"].tolist()
        memberships[label] = ids
        fold[label] = {
            "n": int(mask.sum()), "date_min": frame.loc[mask, "date"].min(),
            "date_max": frame.loc[mask, "result_day"].max(),
            "row_ids_sha256": json_digest(ids),
        }
    recipe = {
        "version": MODEL_VERSION, "eligible_from": utc(origin).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_start": fit_start, "calibration_start": calibration_start,
        "feature_version": FEATURE_VERSION, "feature_contract": dict(FEATURE_CONTRACT),
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "fold": fold, "training": training, "source": audit["source"],
        "replay_audit_sha256": sha256(replay_dir / "audit.json"),
        "code_sha256": code_hashes(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "frozen_artifacts": frozen_hashes(), "promotion": False,
        "history_provenance": audit.get("history_provenance"),
    }
    write_json_exclusive(output_dir / "recipe.json", recipe)
    write_json_exclusive(output_dir / "row_memberships.json", memberships)
    scales = x[fit].std(axis=0)
    scales[scales == 0] = 1
    scaled_fit, scaled_cal = x[fit]/scales, x[cal]/scales
    members = [
        train_single_member(scaled_fit, y[fit], scaled_cal, y[cal], epochs=epochs,
                            gamma=1.0, d_h1=32, d_h2=16, seed=seed)
        for seed in seeds
    ]
    raw_cal = np.stack([member.forward_anti_symmetric(scaled_cal)[0].ravel() for member in members]).mean(axis=0)
    slope = fit_platt_scaling(raw_cal, y[cal])
    # Runtime averages calibrated member logits. A common slope implements the
    # single ensemble calibrator exactly; no second series/binomial transform.
    for member in members:
        member.platt_slope = slope
    artifact = build_model_artifact(members, names, scales, provenance={
        "fold": fold, "replay_audit_sha256": recipe["replay_audit_sha256"],
        "recipe_sha256": sha256(output_dir / "recipe.json"), "source": audit["source"],
        "history_provenance": audit.get("history_provenance"),
    })
    artifact.update(
        model_name="EXP081-Chronological-Prior-Day-Research",
        model_version=MODEL_VERSION + "-" + utc(origin).strftime("%Y%m%d"),
        feature_version=FEATURE_VERSION, feature_algebra_version="ratings-w20-symmetric-series-v1",
        feature_contract=dict(FEATURE_CONTRACT), feature_contract_sha256=json_digest(FEATURE_CONTRACT),
        eligible_from=utc(origin).isoformat(), training=training,
        temporal_limitations=[
            "retrospective reconstruction, not a captured historical forecast",
            "source/roster availability and collection completeness are not certified",
            "previously inspected history is development evidence, not an untouched holdout",
        ],
        ensemble_calibration={
            "method": "positive slope on mean raw ensemble logit, embedded identically in members",
            "slope": slope, "calibration_end": fold["calibration"]["date_max"],
            "source_sha256": audit["outputs"]["snapshots.csv"],
        },
    )
    artifact["architecture"]["loss"] = "focal_loss_gamma_1.0"
    if frozen_hashes() != recipe["frozen_artifacts"] or code_hashes() != recipe["code_sha256"]:
        raise ValueError("frozen artifact or training implementation changed during fit")
    verify_chronological_replay(replay_dir)
    write_json_exclusive(output_dir / "model.json", artifact)
    write_json_exclusive(output_dir / "completed.json", {
        "status": "complete", "model_sha256": sha256(output_dir / "model.json"),
        "recipe_sha256": sha256(output_dir / "recipe.json"),
        "row_memberships_sha256": sha256(output_dir / "row_memberships.json"),
        "qualified_for_production": False,
    })
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    replay_parser = sub.add_parser("replay", help="fresh JSON -> six-family pre-result ratings and W20 snapshots")
    replay_parser.add_argument("--golgg-json", required=True, type=Path)
    replay_parser.add_argument("--output-dir", required=True, type=Path)
    evaluation_parser = sub.add_parser("evaluate", help="verified corrected replay -> new EXP039/081 chronological research replicas")
    evaluation_parser.add_argument("--replay-dir", required=True, type=Path)
    evaluation_parser.add_argument("--output-dir", required=True, type=Path)
    evaluation_parser.add_argument("--epochs", type=int, default=35)
    evaluation_parser.add_argument("--first-test-year", type=int, default=2024)
    evaluation_parser.add_argument("--odds", type=Path, help="optional separate market-only diagnostic; never feature input")
    chrono_replay = sub.add_parser("chronological-replay", help="strict shared v2 ratings/W20 replay; invalid transitions fail")
    chrono_replay.add_argument("--golgg-json", required=True, type=Path)
    chrono_replay.add_argument("--output-dir", required=True, type=Path)
    chrono_replay.add_argument("--history-provenance", type=Path, help="audited partial projection JSON; must pin supplied source SHA256")
    chrono_fit = sub.add_parser("chronological-fit", help="new immutable EXP081 fold; never loads old/final fitted weights")
    chrono_fit.add_argument("--replay-dir", required=True, type=Path)
    chrono_fit.add_argument("--output-dir", required=True, type=Path)
    chrono_fit.add_argument("--origin", required=True, help="declared fold eligibility origin, timezone-aware UTC")
    chrono_fit.add_argument("--calibration-start", required=True)
    chrono_fit.add_argument("--fit-start", default="2018-01-01")
    chrono_fit.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    if args.stage == "replay":
        replay(args.golgg_json, args.output_dir)
    elif args.stage == "chronological-replay":
        chronological_replay(args.golgg_json, args.output_dir, history_provenance=args.history_provenance)
    elif args.stage == "chronological-fit":
        train_chronological_fold(args.replay_dir, args.output_dir, origin=args.origin,
                                 calibration_start=args.calibration_start, fit_start=args.fit_start, epochs=args.epochs)
    else:
        evaluate(args.replay_dir, args.output_dir, epochs=args.epochs, odds=args.odds, first_year=args.first_test_year)


if __name__ == "__main__":
    main()
