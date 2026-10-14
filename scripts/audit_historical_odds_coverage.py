#!/usr/bin/env python3
"""Local historical-odds census; no fitting, timestamp invention, or promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import attach_market
from scripts.build_siamese_research_dataset import sha256

IDENTITIES = {"golgg_match_id": "string", "team1_id": "string", "team2_id": "string"}
TEST_YEARS = tuple(range(2021, 2027))


def years(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.year.astype("Int64")


def year_mask(values: pd.Series, year: int | None) -> pd.Series:
    return values.isna() if year is None else values.eq(year).fillna(False)


def valid_pair(frame: pd.DataFrame, home: str, away: str) -> pd.Series:
    a = pd.to_numeric(frame[home], errors="coerce")
    b = pd.to_numeric(frame[away], errors="coerce")
    return np.isfinite(a) & np.isfinite(b) & a.gt(1) & b.gt(1)


def quote_pairs(columns: pd.Index) -> list[dict]:
    """Pair only exact same-source, same-phase names; never cross bookmakers."""
    pairs = []
    for home, away, source, phase in (
        ("avg_odds_home", "avg_odds_away", "average", "closing_benchmark"),
        ("avg_open_home", "avg_open_away", "average", "open"),
    ):
        if home in columns and away in columns:
            pairs.append({"home": home, "away": away, "source": source, "phase": phase})
    for home in columns:
        match = re.fullmatch(r"odds1_(.+)_(open|close)", home)
        if match:
            source, phase = match.groups()
            away = f"odds2_{source}_{phase}"
            if away in columns:
                pairs.append(
                    {"home": home, "away": away, "source": source, "phase": phase}
                )
    return pairs


def source_schema(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "columns": [
            {
                "name": name,
                "dtype": str(frame[name].dtype),
                "non_null": int(frame[name].notna().sum()),
            }
            for name in frame
        ],
        "missing_id_rows": int(frame.golgg_match_id.isna().sum()),
        "duplicate_id_rows": int(frame.golgg_match_id.duplicated(keep=False).sum()),
        "duplicate_ids": int(
            frame.loc[
                frame.golgg_match_id.duplicated(keep=False), "golgg_match_id"
            ].nunique()
        ),
    }


def save_table(output: Path, name: str, rows: list[dict], audit: dict) -> None:
    pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    audit[name] = rows


def census(
    legacy: pd.DataFrame,
    replay: pd.DataFrame,
    odds: pd.DataFrame,
    output: Path,
    audit: dict,
) -> None:
    audit["source_schema"] = {
        "legacy": source_schema(legacy),
        "replay": source_schema(replay),
        "odds": source_schema(odds),
    }
    # The canonical helper rejects duplicate sporting IDs; do not silently deduplicate.
    for name, frame in (("legacy", legacy), ("replay", replay), ("odds", odds)):
        if frame.golgg_match_id.isna().any():
            raise ValueError(f"{name}: missing canonical IDs; see source_schema")
    full, audit["sporting_join"] = align_legacy_context(legacy, replay)
    normalized_odds = odds.copy()
    for column in ("avg_odds_home", "avg_odds_away", "t1_win"):
        normalized_odds[column] = pd.to_numeric(
            normalized_odds[column], errors="coerce"
        )
    full, audit["market_join"] = attach_market(full, normalized_odds)

    # Probe identity eligibility through the canonical matcher, independent of missing/invalid prices.
    # These unit prices are never reported as quotes or used in any evaluation.
    identity_probe = normalized_odds.copy()
    identity_probe["avg_odds_home"] = 2.0
    identity_probe["avg_odds_away"] = 2.0
    identity, identity_audit = attach_market(
        full.drop(
            columns=[
                "golgg_date",
                "golgg_team1",
                "golgg_team2",
                "t1_win",
                "avg_odds_home",
                "avg_odds_away",
                "odds_a",
                "odds_b",
                "market",
            ]
        ),
        identity_probe,
    )
    audit["identity_only_join"] = {
        **identity_audit,
        "method": "attach_market with neutral finite probe prices, solely to separate identity rejection from quote rejection",
        "not_observed_quotes": True,
    }
    eligible_ids = identity.loc[identity.market.notna(), "golgg_match_id"]
    ly, ry, oy, fy = (
        years(legacy, "date"),
        years(replay, "date"),
        years(odds, "golgg_date"),
        years(full, "date"),
    )
    actual_years = sorted(set(pd.concat([ly, ry, oy, fy]).dropna().astype(int)))
    year_keys = actual_years + (
        [None] if any(v.isna().any() for v in (ly, ry, oy, fy)) else []
    )
    raw_valid = valid_pair(odds, "avg_odds_home", "avg_odds_away")
    raw_missing = odds[["avg_odds_home", "avg_odds_away"]].isna().any(axis=1)
    raw_duplicate = odds.golgg_match_id.duplicated(keep=False)
    market_available = full.market.notna()
    identity_available = identity.market.notna()
    coverage = []
    join_rows = []
    for year in year_keys:
        lm, rm, om, fm = (year_mask(v, year) for v in (ly, ry, oy, fy))
        left = legacy.loc[lm]
        right = replay.loc[replay.golgg_match_id.isin(left.golgg_match_id)]
        _, alignment = align_legacy_context(left, right)
        # Subset source IDs, not source dates: a wrong-date quote must still fail matching.
        frame = full.loc[fm].drop(
            columns=[
                "golgg_date",
                "golgg_team1",
                "golgg_team2",
                "t1_win",
                "avg_odds_home",
                "avg_odds_away",
                "odds_a",
                "odds_b",
                "market",
            ]
        )
        source = normalized_odds.loc[
            normalized_odds.golgg_match_id.isin(frame.golgg_match_id)
        ]
        _, market = attach_market(frame, source)
        join_rows.append(
            {
                "year": year,
                **{f"sporting_{k}": v for k, v in alignment.items()},
                **{f"market_{k}": v for k, v in market.items() if k != "timing"},
            }
        )
        coverage.append(
            {
                "year": year,
                "raw_odds_rows": int(om.sum()),
                "raw_unique_ids": int(odds.loc[om, "golgg_match_id"].nunique()),
                "raw_duplicate_id_rows": int((om & raw_duplicate).sum()),
                "raw_valid_paired_odds_rows": int((om & raw_valid).sum()),
                "raw_missing_quote_rows": int((om & raw_missing).sum()),
                "raw_nonmissing_invalid_pair_rows": int(
                    (om & ~raw_missing & ~raw_valid).sum()
                ),
                "legacy_sporting_rows": int(lm.sum()),
                "replay_sporting_rows": int(rm.sum()),
                "legacy_duplicate_id_rows": int(
                    (lm & legacy.golgg_match_id.duplicated(keep=False)).sum()
                ),
                "replay_duplicate_id_rows": int(
                    (rm & replay.golgg_match_id.duplicated(keep=False)).sum()
                ),
                "aligned_sporting_rows": int(fm.sum()),
                "legacy_without_replay_id": int(
                    (lm & ~legacy.golgg_match_id.isin(replay.golgg_match_id)).sum()
                ),
                "replay_without_legacy_id": int(
                    (rm & ~replay.golgg_match_id.isin(legacy.golgg_match_id)).sum()
                ),
                "legacy_rejected_by_alignment": int(lm.sum() - fm.sum()),
                "valid_joined_paired_odds_rows": int((fm & market_available).sum()),
                "sporting_without_valid_market": int((fm & ~market_available).sum()),
                "sporting_without_source_odds_id": int(
                    (fm & ~full.golgg_match_id.isin(odds.golgg_match_id)).sum()
                ),
                "sporting_unmatched_or_ambiguous_quote_identity": int(
                    (fm & ~identity_available).sum()
                ),
                "identity_matched_missing_quotes": int(
                    (
                        fm
                        & identity_available
                        & full[["avg_odds_home", "avg_odds_away"]].isna().any(axis=1)
                    ).sum()
                ),
                "identity_matched_invalid_or_missing_pair": int(
                    (fm & identity_available & ~market_available).sum()
                ),
                "raw_odds_without_aligned_sporting_id": int(
                    (om & ~odds.golgg_match_id.isin(full.golgg_match_id)).sum()
                ),
                "raw_odds_unmatched_or_ambiguous_identity": int(
                    (om & ~odds.golgg_match_id.isin(eligible_ids)).sum()
                ),
            }
        )
    save_table(output, "coverage_by_year", coverage, audit)
    save_table(output, "join_audit_by_year", join_rows, audit)

    pairs = quote_pairs(odds.columns)
    quote_columns = [
        c for c in odds if c.startswith(("avg_odds_", "avg_open_", "odds1_", "odds2_"))
    ]
    paired_columns = {pair[side] for pair in pairs for side in ("home", "away")}
    audit["quote_pair_definitions"] = pairs
    audit["unpaired_quote_columns"] = [
        c for c in quote_columns if c not in paired_columns
    ]
    column_rows, pair_rows = [], []
    for year in year_keys:
        mask = year_mask(oy, year)
        part = odds.loc[mask]
        eligible = part.golgg_match_id.isin(eligible_ids)
        for column in quote_columns:
            numeric = pd.to_numeric(part[column], errors="coerce")
            column_rows.append(
                {
                    "year": year,
                    "column": column,
                    "raw_rows": len(part),
                    "non_null_rows": int(part[column].notna().sum()),
                    "numeric_rows": int(numeric.notna().sum()),
                    "finite_decimal_above_one_rows": int(
                        (np.isfinite(numeric) & numeric.gt(1)).sum()
                    ),
                }
            )
        for pair in pairs:
            present = part[[pair["home"], pair["away"]]].notna().all(axis=1)
            valid = valid_pair(part, pair["home"], pair["away"])
            pair_rows.append(
                {
                    "year": year,
                    **pair,
                    "raw_rows": len(part),
                    "both_non_null_rows": int(present.sum()),
                    "valid_paired_rows": int(valid.sum()),
                    "missing_either_rows": int((~present).sum()),
                    "nonmissing_invalid_pair_rows": int((present & ~valid).sum()),
                    "canonical_identity_valid_paired_rows": int(
                        (eligible & valid).sum()
                    ),
                }
            )
    save_table(output, "quote_columns_by_year", column_rows, audit)
    save_table(output, "quote_pairs_by_year", pair_rows, audit)
    audit["quote_inventory_unit"] = (
        "Counts are source rows for each exact pair; columns/pairs overlap and MUST NOT be summed as independent matches. Canonical-identity paired counts exclude all duplicate odds IDs."
    )

    stage_rows, folds, protocol_rows = [], [], []
    available = market_available.to_numpy()
    eligible_tests = {}
    extension_by_protocol = {}
    for arm, train_start, test_years in (
        ("original_2020_start", "2020-01-01", TEST_YEARS),
        ("extended_2018_start", "2018-01-01", tuple(range(2020, 2027))),
    ):
        eligible_test = np.zeros(len(full), dtype=bool)
        arm_folds = []
        for year in test_years:
            blocks = temporal_blocks(
                full.date,
                year,
                protocol="semester-calibration",
                train_start=train_start,
            )
            empty_stages = [name for name, mask in blocks.items() if not mask.any()]
            empty_quote_stages = [
                name for name, mask in blocks.items() if not (mask & available).any()
            ]
            eligible = not empty_stages and not any(
                name in empty_quote_stages for name in ("calibration", "test")
            )
            arm_folds.append(
                {
                    "arm": arm,
                    "train_start": train_start,
                    "test_year": year,
                    "all_sporting_stages_nonempty": not empty_stages,
                    "empty_sporting_stages": empty_stages,
                    "empty_quote_stages": empty_quote_stages,
                    "eligible_sporting_and_market_calibration_test": eligible,
                }
            )
            if eligible:
                eligible_test |= blocks["test"]
            for name, mask in blocks.items():
                dates = full.loc[mask, "date"]
                stage_rows.append(
                    {
                        "arm": arm,
                        "train_start": train_start,
                        "test_year": year,
                        "stage": name,
                        "sporting_rows": int(mask.sum()),
                        "paired_quote_rows": int((mask & available).sum()),
                        "missing_or_unaligned_quote_rows": int(
                            (mask & ~available).sum()
                        ),
                        "empty_sporting_stage": not bool(mask.any()),
                        "empty_quote_stage": name in empty_quote_stages,
                        "observed_min_date": None if dates.empty else str(dates.min()),
                        "observed_max_date": None if dates.empty else str(dates.max()),
                        "fold_eligible": eligible,
                    }
                )
        folds.extend(arm_folds)
        eligible_tests[arm] = eligible_test
        eligible_years = [
            f["test_year"]
            for f in arm_folds
            if f["eligible_sporting_and_market_calibration_test"]
        ]
        first_test = min(eligible_years) if eligible_years else None
        protocol_rows.append(
            {
                "arm": arm,
                "name": "semester-calibration",
                "train_start": train_start,
                "test_years_audited": list(test_years),
                "boundaries": f"For test year Y: train [{train_start},Y-1-01-01); stop Q1 of Y-1; select Q2 of Y-1; calibration H2 of Y-1; test calendar Y. All bounds half-open.",
                "earliest_all_sporting_stages_nonempty_year": next(
                    (
                        f["test_year"]
                        for f in arm_folds
                        if f["all_sporting_stages_nonempty"]
                    ),
                    None,
                ),
                "earliest_sporting_and_market_calibration_test_year": first_test,
                "eligibility_note": "All five sporting stages must be nonempty; market follow-up additionally needs calibration/test quotes. Sporting train/stop/select do not require quotes. Counts only: no fitted model or class-support guarantee.",
                "no_arbitrary_row_removal": "All aligned rows retained for census; stage membership comes only from temporal_blocks with explicit training start. Empty stages are reported, not bypassed.",
            }
        )
        start_year = int(train_start[:4])
        extension_by_protocol[arm] = {
            "train_start": train_start,
            "eligible_oos_quote_rows": int((available & eligible_test).sum()),
            "eligible_oos_quote_rows_2024_2026": int(
                (
                    market_available
                    & eligible_test
                    & fy.between(2024, 2026).fillna(False)
                ).sum()
            ),
            "additional_eligible_historical_quote_rows_before_2024": int(
                (market_available & eligible_test & fy.lt(2024).fillna(False)).sum()
            ),
            "joined_quotes_before_train_start": int(
                (market_available & fy.lt(start_year).fillna(False)).sum()
            ),
            "joined_quotes_in_first_training_year": int(
                (market_available & fy.eq(start_year).fillna(False)).sum()
            ),
            "joined_quotes_in_first_stop_select_calibration_year": int(
                (market_available & fy.eq(start_year + 1).fillna(False)).sum()
            ),
            "joined_quotes_not_eligible_oos_in_audited_folds": int(
                (available & ~eligible_test).sum()
            ),
        }
    save_table(output, "temporal_stages", stage_rows, audit)
    audit["folds"] = folds
    audit["protocols"] = protocol_rows
    legacy_2017 = legacy.loc[ly.eq(2017).fillna(False), "date"]
    audit["historical_extension"] = {
        "raw_odds_rows_from_2017": int(oy.ge(2017).fillna(False).sum()),
        "raw_valid_paired_rows_from_2017": int(
            (oy.ge(2017).fillna(False) & raw_valid).sum()
        ),
        "all_valid_joined_quote_rows": int(available.sum()),
        "prior_reported_joined_quote_rows": 11442,
        "matches_prior_reported_count": int(available.sum()) == 11442,
        "protocols": extension_by_protocol,
        "extra_oos_quote_rows_from_2018_start_vs_original": int(
            (
                available
                & eligible_tests["extended_2018_start"]
                & ~eligible_tests["original_2020_start"]
            ).sum()
        ),
        "joined_quotes_with_unparseable_date": int(
            (market_available & fy.isna()).sum()
        ),
        "partial_2017": {
            "legacy_sporting_rows": len(legacy_2017),
            "observed_min_date": None if legacy_2017.empty else str(legacy_2017.min()),
            "observed_max_date": None if legacy_2017.empty else str(legacy_2017.max()),
            "aligned_sporting_rows": int(fy.eq(2017).fillna(False).sum()),
            "raw_odds_rows": int(oy.eq(2017).fillna(False).sum()),
            "valid_joined_quote_rows": int(
                (market_available & fy.eq(2017).fillna(False)).sum()
            ),
            "treatment": "Reported separately, not silently discarded. The 2018 extension starts at the first complete calendar training year chosen from source-date coverage before outcomes; partial 2017 is outside both training protocols.",
        },
        "explanation": "All historically joined quote rows are not an out-of-sample test set. Original 2020-start: pre-2020 precedes training, a 2021 test has an empty train interval, and the first possible 2022 test uses 2020 for training and 2021 for stop/select/calibration. Extended 2018-start: 2018 is training and 2019 is stop/select/calibration for the first possible 2020 test. These are distinct protocols, not arbitrary quote removal. Earlier test years can become training data in later folds. The extended arm requires newly trained chronological models; no modern EXP081 or 2020-start artifacts are backcast or reused.",
    }
    timing_columns = [
        c
        for c in odds
        if any(token in c.lower() for token in ("date", "time", "timestamp"))
    ]
    audit["timing_evidence"] = {
        "available_columns": timing_columns,
        "non_null_counts": {c: int(odds[c].notna().sum()) for c in timing_columns},
        "trustworthy_quote_time_evidence": False,
        "interpretation": "odds_date/golgg_date are source event-date labels, not independently recorded quote capture timestamps. Open/close names identify stored quote types, not verified availability at an execution cutoff. No synthetic timestamps or live/executable-price claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/replay/snapshots.csv",
    )
    parser.add_argument("--odds", type=Path, default=ROOT / "data/odds.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory only; existing artifacts are never overwritten",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        args.legacy,
        args.replay,
        args.odds,
        Path(__file__),
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/build_siamese_research_dataset.py",
    ]
    started = time.monotonic()
    audit = {
        "status": "running",
        "scope": "Local canonical snapshots and odds only; no DB/network/model fitting or production change",
        "paths": {
            "legacy": str(args.legacy),
            "replay": str(args.replay),
            "odds": str(args.odds),
        },
        "sha256_start": {str(path): sha256(path) for path in paths},
    }
    try:
        legacy, replay = (
            pd.read_csv(path, dtype=IDENTITIES, low_memory=False)
            .sort_values(["date", "golgg_match_id"])
            .reset_index(drop=True)
            for path in (args.legacy, args.replay)
        )
        odds = pd.read_csv(
            args.odds, dtype={"golgg_match_id": "string"}, low_memory=False
        )
        census(legacy, replay, odds, args.output_dir, audit)
        audit["status"] = "complete"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["sha256_end"] = {str(path): sha256(path) for path in paths}
        audit["sources_unchanged"] = audit["sha256_start"] == audit["sha256_end"]
        audit["runtime_seconds"] = time.monotonic() - started
        if not audit["sources_unchanged"]:
            audit["status"] = "invalid_source_changed"
        (args.output_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n"
        )
    if not audit["sources_unchanged"]:
        raise RuntimeError("Inputs or shared implementations changed during census")
    print(json.dumps(audit["historical_extension"], indent=2))
    print(f"Coverage census saved to {args.output_dir}")


if __name__ == "__main__":
    main()
