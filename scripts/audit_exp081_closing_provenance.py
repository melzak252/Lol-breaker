#!/usr/bin/env python3
"""Read-only provenance audit of the frozen 44 EXP081 close/2025 Benter bets.

No fitting, outcome-driven repair, network access, or production adoption. The
exclusive output directory contains evidence, not an executable betting ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_exp081_market_calibration import (
    MIN_EV, TAX, code_hashes, frozen_hashes, phase_frame, policy,
    selected_uncertainty, sha256, write_json,
)
from src.utils.module_loading import load_module_from_path

MAPPER_PATH = ROOT / "scripts/03_dane_pipeline/00_map_matches.py"
MAPPER = load_module_from_path(MAPPER_PATH, "exp081_provenance_mapper")
SOURCE_ROOT = ROOT.parent / "embedded-rift"
RUN = ROOT / "data/artifacts/exp081-market-calibration-20260908/run01"
BOOK_IDS = {"163": "efortuna", "165": "sts", "502": "lv_bet", "539": "betclic",
            "572": "betfan", "591": "superbet", "819": "fuksiarz"}


def event_id(url):
    """Keep the public event identifier only; never persist capture URLs/bodies."""
    if not isinstance(url, str):
        return None
    if "#" in url:
        candidate = url.rsplit("#", 1)[1]
        return candidate if re.fullmatch(r"[A-Za-z0-9]{8}", candidate) else None
    match = re.search(r"/match-event/\d+-\d+-([A-Za-z0-9]{8})-", url)
    return match.group(1) if match else None


def equal_quote(left, right):
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0))


def valid_pair(pair):
    return bool(np.isfinite(pair).all() and (np.asarray(pair) > 1).all())


def ordered_values(value):
    """Expose source container positions without assuming their team semantics."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value.get("0"), value.get("1")]
    return []


def quote_relation(raw_pair, value):
    values = ordered_values(value)
    if len(values) != 2 or not all(isinstance(v, (int, float)) for v in values):
        return "unavailable_or_nonbinary"
    direct = all(equal_quote(a, b) for a, b in zip(raw_pair, values))
    reverse = all(equal_quote(a, b) for a, b in zip(raw_pair, values[::-1]))
    return "symmetric" if direct and reverse else "positions_0_1" if direct else "positions_1_0" if reverse else "different_quote"


def payload_inventory(path):
    """Whitelist market metadata, identities and quote times; no captured secrets."""
    records = json.loads(path.read_text())
    inventory, by_event = [], defaultdict(list)
    for index, record in enumerate(records):
        if "match-event/" not in record.get("url", ""):
            continue
        decoded = record.get("decrypted")
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                inventory.append({"response_index": index, "status": "invalid_decrypted_json"})
                continue
        if not isinstance(decoded, dict) or not isinstance(decoded.get("d"), dict):
            inventory.append({"response_index": index, "status": "missing_detail_object"})
            continue
        data = decoded["d"]
        endpoint_event = event_id(record.get("url"))
        body_event = data.get("encodeventId", data.get("encodeEventId"))
        entry = {"response_index": index, "endpoint_event_id": endpoint_event,
                 "payload_event_id": body_event, "identity_consistent": endpoint_event == body_event,
                 "time_base": data.get("time-base"), "betting_type_code": data.get("bt"),
                 "scope_code": data.get("sc"), "markets": []}
        for key, market in data.get("oddsdata", {}).get("back", {}).items():
            metadata = {name: market.get(name) for name in (
                "bettingTypeId", "scopeId", "handicapTypeId", "handicapValue",
                "mixedParameterId", "mixedParameterName", "isBack", "outcomeId")}
            books = []
            for bid, book in BOOK_IDS.items():
                for phase, quote_key, time_key in (("open", "openingOdd", "openingChangeTime"),
                                                    ("close", "odds", "changeTime")):
                    value = market.get(quote_key, {}).get(bid)
                    if value is not None:
                        books.append({"book": book, "phase": phase,
                                      "container": type(value).__name__,
                                      "outcome_keys": list(value) if isinstance(value, dict) else list(range(len(value))),
                                      "values_in_positions_0_1": ordered_values(value),
                                      "quote_change_times": market.get(time_key, {}).get(bid)})
            entry["markets"].append({"market_key": key, "metadata": metadata, "books": books})
        inventory.append(entry)
        if endpoint_event == body_event and body_event:
            by_event[body_event].append(entry)
    return {"captured_response_n": len(records), "details": inventory}, by_event


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/artifacts/corrected039081-20260908/replay/snapshots.csv")
    parser.add_argument("--odds", type=Path, default=ROOT / "data/artifacts/exp081-full-audit-20260908/identity-only-v3/odds.csv")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT, help="Read-only local OddsPortal archive directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New exclusive directory; never overwrite")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("run_dir", "snapshots", "odds", "source_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    sources = {
        "raw_csv": args.source_dir / "oddsportal_matches.csv",
        "export_json": args.source_dir / "matches.json",
        "backup_json": args.source_dir / "matches_OLD_backup.json",
        "payloads": args.source_dir / "all_decrypted_responses.json",
        "scraper": args.source_dir / "src/scripts/esport/scrape_oddsportal.py",
        "updater": args.source_dir / "src/scripts/esport/fetch_missing_odds_final.py",
        "prior_audit": ROOT / "data/artifacts/exp081-full-audit-20260908/odds_provenance_audit.json",
    }
    inputs = [args.snapshots, args.odds, *sources.values(), MAPPER_PATH,
              *(args.run_dir / name for name in ("oos_predictions.csv", "selected_diagnostics.csv", "fits.json", "summary.json", "protocol.json"))]
    hashes = {str(path): sha256(path) for path in inputs}
    frozen = frozen_hashes()
    code = {**code_hashes(), str(Path(__file__).resolve()): sha256(Path(__file__)), str(MAPPER_PATH): sha256(MAPPER_PATH)}
    if (TAX, MIN_EV) != (.12, .05):
        raise RuntimeError("shared policy differs from frozen protocol")
    protocol = {
        "experiment": "exp081-close2025-selected-provenance", "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "input_hashes": hashes, "code_hashes": code, "frozen_hashes": frozen,
        "scope": "exact frozen close/2025 Benter selections; all 44 retained regardless of outcomes or odds",
        "no_fitting": True, "expected_selected_n": 44, "expected_selected_wins": 37,
        "policy": {"tax": TAX, "max_ev_strictly_greater_than": MIN_EV, "ties": "abstain", "pair": "finite same-book prices >1"},
        "orientation": "source_row zero-based CSV record; source_sides_swapped then exact mapped/snapshot names; all 32 quote fields reconciled",
        "source_scope": "named local archive/export/backup/captures only; sibling repository read-only",
        "repair_rule": "independent payload event AND outcome/team AND market-period AND quote identity required; no reversal inferred from results or prices",
        "market_caveat": "first back entry and numeric endpoint scope are not independently decoded full-series/pre-match semantics",
        "timing_caveat": "event-start and quote-change timestamps are not capture/availability timestamps or PIT evidence",
        "independent_books_caveat": "distinct bookmaker columns share the same exporter and do not independently validate orientation",
        "bootstrap": {"resamples": 5000, "seed": 82, "unit": "all eligible calendar months including empty selections", "diagnostic_only": True},
        "eligibility_live": 0, "promotion_approved": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    try:
        predictions = pd.read_csv(args.run_dir / "oos_predictions.csv", dtype={k: "string" for k in ("golgg_match_id", "team1_id", "team2_id")})
        if predictions.duplicated(["golgg_match_id", "phase"]).any():
            raise ValueError("duplicate event within prediction phase")
        cohort = predictions.loc[predictions.phase.eq("close") & predictions.year.eq(2025)].copy().reset_index(drop=True)
        selected, side_a = policy(cohort.benter, cohort.odds_a, cohort.odds_b)
        chosen = cohort.loc[selected].copy()
        chosen["selected_side"] = np.where(side_a[selected], "a", "b")
        chosen["selected_probability"] = np.where(side_a, cohort.benter, 1 - cohort.benter)[selected]
        chosen["selected_odds"] = np.where(side_a, cohort.odds_a, cohort.odds_b)[selected]
        chosen["selected_won"] = np.where(side_a, cohort.y_true, 1 - cohort.y_true)[selected]
        if len(chosen) != 44 or int(chosen.selected_won.sum()) != 37:
            raise ValueError("frozen 44-selection/37-win cohort changed; refusing silently different audit")
        ledger = pd.read_csv(args.run_dir / "selected_diagnostics.csv", dtype={"golgg_match_id": "string"})
        ledger = ledger.loc[ledger.phase.eq("close") & ledger.year.eq(2025) & ledger.model.eq("benter")].set_index("golgg_match_id")
        if not ledger.index.is_unique or set(ledger.index) != set(chosen.golgg_match_id):
            raise ValueError("frozen selection ledger IDs do not reconcile")
        for row in chosen.itertuples():
            recorded = ledger.loc[row.golgg_match_id]
            if recorded.side != row.selected_side or any(not equal_quote(recorded[key], getattr(row, value)) for key, value in (
                ("probability", "selected_probability"), ("odds", "selected_odds"), ("won", "selected_won"))):
                raise ValueError(f"selection ledger disagreement: {row.golgg_match_id}")
        mapped = pd.read_csv(args.odds, dtype={"golgg_match_id": "string"}).set_index("golgg_match_id", verify_integrity=True)
        snapshots = pd.read_csv(args.snapshots, dtype={k: "string" for k in ("golgg_match_id", "team1_id", "team2_id")}).set_index("golgg_match_id", verify_integrity=True)
        raw = pd.read_csv(sources["raw_csv"])
        exports = json.loads(sources["export_json"].read_text())
        backup = json.loads(sources["backup_json"].read_text())
        backup_by_event, exports_by_event = defaultdict(list), defaultdict(list)
        for i, row in enumerate(backup):
            backup_by_event[event_id(row.get("url"))].append((i, row))
        for i, row in enumerate(exports):
            exports_by_event[event_id(row.get("url"))].append((i, row))
        inventory, details_by_event = payload_inventory(sources["payloads"])
        write_json(args.output_dir / "payload_inventory.json", inventory)
        # Independently rebuild phase alignment with the existing helper, never fit.
        frame = cohort.merge(snapshots[["team1_name", "team2_name"]], left_on="golgg_match_id", right_index=True, validate="one_to_one")
        reconstructed, alignment = phase_frame(frame, mapped.reset_index(), "close")
        reconstructed = reconstructed.set_index("golgg_match_id")
        rows, book_rows = [], []
        for forecast in chosen.to_dict("records"):
            mid = forecast["golgg_match_id"]
            mapped_row, snapshot = mapped.loc[mid], snapshots.loc[mid]
            source_index = int(mapped_row.source_row)
            if source_index < 0 or source_index >= len(raw):
                raise ValueError(f"invalid raw source row: {mid}")
            source = raw.iloc[source_index]
            event = event_id(source.url)
            errors, missing = [], []
            for key in ("team1_id", "team2_id", "date", "best_of", "y_true"):
                if forecast[key] != snapshot[key]:
                    errors.append(f"snapshot_{key}_mismatch")
            if source.url != mapped_row.oddsportal_url:
                errors.append("raw_source_url_mismatch")
            for source_key, mapped_key in (("home_team", "source_home_team"), ("away_team", "source_away_team"), ("date", "source_date")):
                if source[source_key] != mapped_row[mapped_key]:
                    errors.append(f"raw_{source_key}_mismatch")
            if str(mapped_row.source_sides_swapped).lower() not in ("true", "false"):
                raise ValueError(f"invalid source orientation flag: {mid}")
            swapped = str(mapped_row.source_sides_swapped).lower() == "true"
            raw_names = (source.home_team, source.away_team)
            oriented_names = raw_names[::-1] if swapped else raw_names
            name_identity = all(MAPPER.normalize_team_name(a) == MAPPER.normalize_team_name(b)
                                for a, b in zip(oriented_names, (mapped_row.golgg_team1, mapped_row.golgg_team2)))
            if not name_identity:
                errors.append("independent_normalized_team_orientation_mismatch")
            quote_mismatches = []
            for left, right in MAPPER.QUOTE_PAIRS:
                expected = (source[right], source[left]) if swapped else (source[left], source[right])
                for key, value in zip((left, right), expected):
                    if not equal_quote(mapped_row[key], value):
                        quote_mismatches.append(key)
            if quote_mismatches:
                errors.append("raw_to_mapped_quote_mismatch")
            same = snapshot.team1_name == mapped_row.golgg_team1 and snapshot.team2_name == mapped_row.golgg_team2
            reverse = snapshot.team1_name == mapped_row.golgg_team2 and snapshot.team2_name == mapped_row.golgg_team1
            if not (same ^ reverse):
                errors.append("ambiguous_snapshot_orientation")
            for key in ("odds_a", "odds_b", "sports", "market"):
                if mid not in reconstructed.index or not equal_quote(reconstructed.loc[mid, key], forecast[key]):
                    errors.append(f"reconstructed_{key}_mismatch")
            raw_to_snapshot_swapped = swapped ^ bool(reverse)
            archives = backup_by_event.get(event, [])
            archive = archives[0][1] if len(archives) == 1 else None
            export_matches = exports_by_event.get(event, [])
            export = export_matches[0][1] if len(export_matches) == 1 else None
            export_identity = bool(export and all(export.get(k) == source[k] for k in ("home_team", "away_team", "date")))
            if export is not None and not export_identity:
                errors.append("raw_csv_export_json_identity_difference")
            archive_identity = bool(archive and all(archive.get(k) == source[k] for k in ("home_team", "away_team", "date")))
            exported_quote_mismatches = [key for pair in MAPPER.QUOTE_PAIRS for key in pair if export is not None and not equal_quote(source[key], export.get(key))]
            if export is None:
                missing.append("unique_same_event_json_export")
            elif exported_quote_mismatches:
                errors.append("raw_csv_export_json_quote_difference")
            details = details_by_event.get(event, [])
            if not details:
                missing.append("selected_event_independent_detailed_market_payload")
            missing.extend(["source_outcome_key_to_team_identity", "per_row_scraper_or_missing_price_updater_lineage",
                            "independently_decoded_full_series_market_period", "capture_timestamp_and_predecision_availability",
                            "same_book_executable_stake_limit_and_market_status"])
            if not archive_identity:
                missing.append("unique_backup_event_identity_and_start_timestamp")
            counts = Counter()
            for book in MAPPER.BOOKMAKERS:
                for phase in ("open", "close"):
                    pair = [source[f"odds1_{book}_{phase}"], source[f"odds2_{book}_{phase}"]]
                    oriented = pair[::-1] if raw_to_snapshot_swapped else pair
                    valid = valid_pair(pair)
                    counts[phase] += int(valid)
                    payload_evidence = []
                    for detail in details:
                        for market in detail["markets"]:
                            for quote in market["books"]:
                                if quote["book"] == book and quote["phase"] == phase:
                                    payload_evidence.append({"response_index": detail["response_index"], "market_key": market["market_key"],
                                        "container": quote["container"], "outcome_keys": quote["outcome_keys"],
                                        "relation": quote_relation(pair, quote["values_in_positions_0_1"]),
                                        "quote_change_times": quote["quote_change_times"]})
                    stored_key = "openingsOdds" if phase == "open" else "closingOdds"
                    bid = next(key for key, name in BOOK_IDS.items() if name == book)
                    old_value = archive.get(stored_key, {}).get(bid) if archive_identity else None
                    book_rows.append({"golgg_match_id": mid, "event_id": event, "source_row": source_index,
                        "book": book, "phase": phase, "raw_home_odds": pair[0], "raw_away_odds": pair[1],
                        "snapshot_a_odds": oriented[0], "snapshot_b_odds": oriented[1], "valid_pair": valid,
                        "backup_container": type(old_value).__name__, "backup_relation": quote_relation(pair, old_value),
                        "payload_evidence": json.dumps(payload_evidence, allow_nan=False)})
            opening = [source.odds1_sts_open, source.odds2_sts_open]
            if raw_to_snapshot_swapped:
                opening.reverse()
            open_forecast = predictions.loc[predictions.golgg_match_id.eq(mid) & predictions.phase.eq("open")]
            open_selected, open_side, open_probability = None, None, None
            if len(open_forecast) == 1:
                op = open_forecast.iloc[0]
                mask, side = policy([op.benter], [op.odds_a], [op.odds_b])
                open_selected, open_side, open_probability = bool(mask[0]), "a" if side[0] else "b", float(op.benter)
            rows.append({**forecast, "event_id": event, "source_row": source_index, "source_csv_line": source_index + 2,
                "source_home_team": source.home_team, "source_away_team": source.away_team,
                "source_date": source.date, "tournament": source.tournament,
                "source_to_mapped_swapped": swapped, "mapped_to_snapshot_swapped": bool(reverse),
                "independent_normalized_team_orientation_matches": name_identity,
                "raw_to_snapshot_swapped": raw_to_snapshot_swapped,
                "quote_fields_checked": 2 * len(MAPPER.QUOTE_PAIRS), "quote_mismatches": quote_mismatches,
                "export_json_row": export_matches[0][0] if export is not None else None,
                "export_json_identity_matches": export_identity,
                "export_json_quote_mismatches": exported_quote_mismatches,
                "backup_source_rows": [i for i, _ in archives], "backup_identity_matches": archive_identity,
                "backup_start_unix": archive.get("date-start-base") if archive_identity else None,
                "backup_sts_close_available": bool(archive_identity and archive.get("odds1_sts_close") is not None and archive.get("odds2_sts_close") is not None),
                "details_response_indices": [d["response_index"] for d in details],
                "open_odds_a": opening[0], "open_odds_b": opening[1],
                "close_to_open_ratio_selected_side": forecast["selected_odds"] / opening[0 if forecast["selected_side"] == "a" else 1] if valid_pair(opening) else None,
                "open_benter": open_probability, "open_policy_selected": open_selected, "open_policy_side": open_side,
                "book_pairs_open": counts["open"], "book_pairs_close": counts["close"],
                "independent_other_book_pairs_close": counts["close"] - 1,
                "reconciliation_status": "mismatch" if errors else "export_chain_reconciled",
                "provenance_status": "unresolved_no_independent_quote_identity",
                "reasons": errors, "missing_provenance": missing, "eligibility_live": 0})
        result = pd.DataFrame(rows)
        # JSON roundtrip converts pandas/NumPy missing values to strict JSON null.
        write_json(args.output_dir / "row_audit.json", json.loads(result.to_json(orient="records", double_precision=15)))
        flat = result.copy()
        for key in ("quote_mismatches", "export_json_quote_mismatches", "backup_source_rows", "details_response_indices", "reasons", "missing_provenance"):
            flat[key] = flat[key].map(json.dumps)
        flat.to_csv(args.output_dir / "oos_predictions.csv", index=False, mode="x")
        pd.DataFrame(book_rows).to_csv(args.output_dir / "book_quotes.csv", index=False, mode="x")
        monthly = []
        for month in pd.period_range("2025-01", "2025-12", freq="M"):
            part = result.loc[pd.to_datetime(result.date).dt.to_period("M").eq(month)]
            monthly.append({"month": str(month), "eligible_close_n": int(pd.to_datetime(cohort.date).dt.to_period("M").eq(month).sum()),
                "selected_n": len(part), "wins": int(part.selected_won.sum()),
                "selection_share": len(part) / len(result), "mean_probability": float(part.selected_probability.mean()) if len(part) else None,
                "mean_odds": float(part.selected_odds.mean()) if len(part) else None})
        pd.DataFrame(monthly).to_csv(args.output_dir / "monthly_concentration.csv", index=False, mode="x")
        pp = np.where(side_a, cohort.benter, 1 - cohort.benter)
        won = np.where(side_a, cohort.y_true, 1 - cohort.y_true)
        price = np.where(side_a, cohort.odds_a, cohort.odds_b)
        summary = {
            "status": "research_only_provenance_unresolved", "selected_n": len(result), "wins": int(result.selected_won.sum()),
            "mean_selected_probability": float(result.selected_probability.mean()), "mean_selected_odds": float(result.selected_odds.mean()),
            "reconciliation_counts": dict(Counter(result.reconciliation_status)), "alignment": alignment,
            "evidence_coverage": {"detail_payload_rows": int(result.details_response_indices.map(bool).sum()),
                "backup_identity_rows": int(result.backup_identity_matches.sum()), "backup_sts_close_rows": int(result.backup_sts_close_available.sum()),
                "csv_export_json_rows": int(result.export_json_row.notna().sum()),
                "other_book_close_rows": int(result.independent_other_book_pairs_close.gt(0).sum())},
            "missing_provenance_counts": dict(Counter(reason for row in rows for reason in row["missing_provenance"])),
            "concentration": {"months_with_selections": sum(r["selected_n"] > 0 for r in monthly),
                "largest_month_share": max(r["selection_share"] for r in monthly),
                "month_hhi": sum(r["selection_share"] ** 2 for r in monthly),
                "tournament_counts": {str(k): int(v) for k, v in result.tournament.value_counts().items()},
                "book_close_pair_counts": {str(k): int(v) for k, v in result.book_pairs_close.value_counts().items()}},
            "selected_residual_uncertainty": selected_uncertainty(cohort, selected, pp - won, (1 - TAX) * price * (pp - won)),
            "source_findings": [
                {"status": "confirmed_incompatible_parser_conventions_not_proven_selected_row_reversal",
                 "scraper_evidence": "scrape_oddsportal.py:158-165 writes array positions [0],[1] to home,away",
                 "updater_evidence": "fetch_missing_odds_final.py:116-127 writes string keys ['1'],['0'] to home,away",
                 "lineage_loss": "fetch_missing_odds_final.py:137-139,177-198 updates missing averages in place without per-row parser/payload provenance",
                 "implication": "key reversal is a source risk; dict key/team semantics and updater attribution cannot be inferred from win frequency"},
                {"status": "confirmed_market_timing_metadata_loss",
                 "evidence": "scrape_oddsportal.py:124,141-165 selects fixed numeric endpoint and first back market; CSV drops outcome IDs, scope and change times",
                 "implication": "close is last response odds, not a certified pre-match executable closing quote"}],
            "repair": {"emitted": False, "cohort_changed": False,
                "reason": "No selected quote has independent outcome-key-to-team and market-period identity sufficient for deterministic repair; surprising prices/outcomes and cross-book agreement are not repair evidence."},
            "eligibility_live": 0, "promotion_approved": False,
        }
        write_json(args.output_dir / "summary.json", summary)
        unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
        if not unchanged or frozen_hashes() != frozen or any(sha256(Path(path)) != digest for path, digest in code.items()):
            raise RuntimeError("input, code, or frozen artifacts changed during audit")
        outputs = sorted(path for path in args.output_dir.iterdir() if path.is_file())
        write_json(args.output_dir / "manifest.json", {"input_hashes_unchanged": True, "frozen_hashes_unchanged": True,
            "output_hashes": {path.name: sha256(path) for path in outputs}})
        print(json.dumps({"output_dir": str(args.output_dir), "selected_n": len(result), "eligibility_live": 0,
                          "reconciliation_counts": summary["reconciliation_counts"], "repair_emitted": False}))
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
