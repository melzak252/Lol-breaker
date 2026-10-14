#!/usr/bin/env python3
"""Build EXP-083 features from a hashed sports-only export; no database access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_siamese_research_dataset import sha256
from src.models.replay_series import build_replay_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source_dir / "manifest.json").read_text())
    frames = {}
    started = time.monotonic()
    for table in ("matches", "games", "players"):
        path = args.source_dir / f"{table}.csv"
        spec = manifest["tables"][table]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {table}")
        identity_columns = {
            name: "string" for name in spec["columns"] if name.endswith("_id")
        }
        frame = pd.read_csv(path, dtype=identity_columns)
        if list(frame.columns) != spec["columns"] or len(frame) != spec["rows"]:
            raise ValueError(f"input manifest mismatch: {table}")
        frames[table] = frame
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame, audit = build_replay_features(
        frames["matches"], frames["games"], frames["players"]
    )
    if frame.golgg_match_id.duplicated().any():
        raise ValueError("replay emitted duplicate series")
    has_history = frame.feature_history_max_date.notna()
    if not (
        pd.to_datetime(frame.loc[has_history, "feature_history_max_date"])
        < pd.to_datetime(frame.loc[has_history, "date"])
    ).all():
        raise ValueError("replay contains current-day or future history")
    predictors = [name for name in frame if name.startswith(("d_", "c_"))]
    audit.update(
        {
            "source_manifest": manifest,
            "predictors": predictors,
            "rows": len(frame),
            "modern_rows": int((frame.date >= "2024-01-01").sum()),
            "first_date": frame.date.min(),
            "last_date": frame.date.max(),
            "rows_by_year": frame.date.str[:4].value_counts().sort_index().to_dict(),
            "runtime_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "code_sha256": {
                str(path): sha256(path)
                for path in (
                    Path(__file__),
                    ROOT / "src/models/replay_series.py",
                    ROOT / "src/ratings/glicko2_core.py",
                    ROOT / "src/models/competition_tiers.py",
                )
            },
            "promotion_blockers": [
                "Retrospective source tables have no historical source_available_at audit",
                "Event dates only: before-day features do not prove exact source/prediction/quote timestamps",
                "Lagged roster is a proxy, not an announced upcoming lineup",
                "No trustworthy locked point-in-time EXP039/operational prediction cohort",
                "2024+ is previously inspected diagnostic data, not an untouched final holdout",
            ],
        }
    )
    frame.to_csv(args.output_dir / "snapshots.csv", index=False)
    audit["snapshots_sha256"] = sha256(args.output_dir / "snapshots.csv")
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in ("source_manifest", "code_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
