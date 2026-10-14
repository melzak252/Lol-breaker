"""Profile project datasets for the whitepaper data-pipeline chapter.

The script inspects raw and derived data files, generates compact CSV/Markdown
artefacts, and prepares the factual material required for point 3 of the
whitepaper: data lineage, dataset structure, date coverage, missingness,
duplicates, and basic matchability between sport and odds datasets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.golgg_schema import (
    games,
    players1,
    players2,
    score1,
    score2,
    team1_id,
    team1_name,
    team2_id,
    team2_name,
    match_tournament,
)


ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "data_profile"
WHITEPAPER_DIR = PROJECT_ROOT / "docs" / "whitepaper"

PRIMARY_DATASET_NAMES = {
    "golgg_matches.json",
    "odds.csv",
    "oddsportal_matches.csv",
}


@dataclass(frozen=True)
class DatasetProfile:
    """Container with high-level dataset metadata."""

    path: Path
    category: str
    file_type: str
    size_mb: float
    rows: int | None
    columns: int | None
    load_status: str
    notes: str


def classify_dataset(path: Path) -> str:
    """Classify a dataset into a methodological category.

    Args:
        path: Dataset path.

    Returns:
        Human-readable category used in the generated inventory.
    """

    name = path.name.lower()
    parts = {part.lower() for part in path.parts}

    if name == "golgg_matches.json":
        return "source_of_truth_sport"
    if name == "odds.csv":
        return "source_of_truth_market_mapped"
    if name == "oddsportal_matches.csv":
        return "mapping_audit_input"
    if "docs" in parts and "assets" in parts:
        return "report_artifact_not_profiled"
    return "not_profiled"


def discover_dataset_files(root: Path) -> list[Path]:
    """Discover only the source-of-truth datasets for point 3.

    Args:
        root: Project root directory.

    Returns:
        Sorted list of discovered dataset paths.
    """

    files = [root / "data" / filename for filename in sorted(PRIMARY_DATASET_NAMES)]
    return [path for path in files if path.exists() and path.is_file()]


def flatten_json_payload(payload: Any) -> pd.DataFrame:
    """Convert a JSON payload into a tabular DataFrame when possible.

    Args:
        payload: Parsed JSON object.

    Returns:
        DataFrame representation of the payload.
    """

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "games" in payload[0]:
            return summarize_golgg_matches_payload(payload)
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return pd.json_normalize(list_values[0])
        return pd.json_normalize(payload, sep=".")
    return pd.DataFrame({"value": [payload]})


def summarize_golgg_matches_payload(payload: list[Any]) -> pd.DataFrame:
    """Create a compact match-level profile for the large GOL.GG JSON.

    The raw file contains deeply nested per-game and per-player statistics. Full
    normalization is unnecessary for the point-3 data-lineage chapter and is
    slow for an almost-900 MB JSON file. This function preserves top-level match
    metadata and adds structural diagnostics required by the whitepaper.

    Args:
        payload: Top-level JSON list loaded from ``golgg_matches.json``.

    Returns:
        Compact match-level DataFrame.
    """

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            rows.append({"raw_value": str(item)})
            continue

        players_1 = players1(item)
        players_2 = players2(item)
        game_payload = games(item)
        row = {
            "match_id": item.get("match_id"),
            "date": item.get("date"),
            "tournament": match_tournament(item),
            "name_1": team1_name(item),
            "name_2": team2_name(item),
            "tid_1": team1_id(item),
            "tid_2": team2_id(item),
            "score_1": score1(item),
            "score_2": score2(item),
        }
        row.update(
            {
                "players_1_count": len(players_1)
                if isinstance(players_1, list)
                else None,
                "players_2_count": len(players_2)
                if isinstance(players_2, list)
                else None,
                "has_full_rosters": (
                    isinstance(players_1, list)
                    and isinstance(players_2, list)
                    and len(players_1) == 5
                    and len(players_2) == 5
                ),
                "games_count": len(game_payload),
                "has_game_payload": bool(game_payload),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset(path: Path) -> tuple[pd.DataFrame | None, str, str]:
    """Load a CSV or JSON file safely.

    Args:
        path: Dataset path.

    Returns:
        Tuple with DataFrame or None, load status, and notes.
    """

    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False), "ok", ""
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return flatten_json_payload(payload), "ok", ""
    except Exception as exc:  # noqa: BLE001 - profiling must continue.
        return None, "error", f"{type(exc).__name__}: {exc}"
    return None, "skipped", "Unsupported extension."


def infer_date_columns(data: pd.DataFrame) -> list[str]:
    """Infer likely date/time columns from names and parseability.

    Args:
        data: Dataset to inspect.

    Returns:
        List of likely date column names.
    """

    likely_names = [
        column
        for column in data.columns
        if any(token in column.lower() for token in ("date", "time", "year"))
    ]
    parseable: list[str] = []
    for column in likely_names[:12]:
        series = pd.to_datetime(data[column], errors="coerce", utc=True)
        if series.notna().mean() >= 0.2:
            parseable.append(column)
    return parseable


def summarize_dates(path: Path, data: pd.DataFrame) -> list[dict[str, Any]]:
    """Create date coverage rows for all detected date columns.

    Args:
        path: Dataset path.
        data: Dataset to inspect.

    Returns:
        List of date coverage dictionaries.
    """

    rows: list[dict[str, Any]] = []
    for column in infer_date_columns(data):
        parsed = pd.to_datetime(data[column], errors="coerce", utc=True)
        valid = parsed.dropna()
        rows.append(
            {
                "dataset": relative_dataset_name(path),
                "column": column,
                "valid_rows": int(valid.shape[0]),
                "valid_share": round(float(parsed.notna().mean()), 4),
                "min_date": valid.min().isoformat() if not valid.empty else "",
                "max_date": valid.max().isoformat() if not valid.empty else "",
            }
        )
    return rows


def summarize_columns(path: Path, data: pd.DataFrame) -> list[dict[str, Any]]:
    """Create a column-level data dictionary.

    Args:
        path: Dataset path.
        data: Dataset to inspect.

    Returns:
        List of column profile dictionaries.
    """

    rows: list[dict[str, Any]] = []
    for column in data.columns:
        series = data[column]
        non_null = int(series.notna().sum())
        unique_count = safe_nunique(series)
        sample_values = [str(value) for value in series.dropna().head(3).tolist()]
        rows.append(
            {
                "dataset": relative_dataset_name(path),
                "column": column,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": int(series.isna().sum()),
                "missing_share": round(float(series.isna().mean()), 4),
                "unique_values": unique_count,
                "sample_values": " | ".join(sample_values),
            }
        )
    return rows


def safe_nunique(series: pd.Series) -> int:
    """Count unique values even when cells contain lists or dictionaries.

    Args:
        series: Series to inspect.

    Returns:
        Number of unique non-null values.
    """

    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().map(str).nunique(dropna=True))


def relative_dataset_name(path: Path) -> str:
    """Return a stable relative dataset name for reports.

    Args:
        path: Absolute or relative dataset path.

    Returns:
        Path relative to the project root when possible, otherwise ``str(path)``.
    """

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def summarize_quality(path: Path, data: pd.DataFrame) -> dict[str, Any]:
    """Create dataset-level quality statistics.

    Args:
        path: Dataset path.
        data: Dataset to inspect.

    Returns:
        Quality summary dictionary.
    """

    missing_cells = int(data.isna().sum().sum())
    total_cells = int(data.shape[0] * data.shape[1])
    duplicate_rows = count_duplicate_rows(data)
    return {
        "dataset": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_cells": missing_cells,
        "missing_cell_share": round(missing_cells / total_cells, 4)
        if total_cells
        else 0.0,
        "duplicate_full_rows": duplicate_rows,
        "duplicate_full_row_share": round(duplicate_rows / data.shape[0], 4)
        if data.shape[0]
        else 0.0,
        "columns_with_missing": int(data.isna().any().sum()),
    }


def count_duplicate_rows(data: pd.DataFrame) -> int:
    """Count duplicate rows, including frames with unhashable cell values.

    Args:
        data: Dataset to inspect.

    Returns:
        Number of fully duplicated rows.
    """

    if data.empty:
        return 0
    try:
        return int(data.duplicated().sum())
    except TypeError:
        hashable_data = data.map(
            lambda value: str(value) if isinstance(value, (list, dict)) else value
        )
        return int(hashable_data.duplicated().sum())


def candidate_key_columns(data: pd.DataFrame) -> list[str]:
    """Return likely identifier columns for duplicate checks.

    Args:
        data: Dataset to inspect.

    Returns:
        Candidate key column names.
    """

    tokens = ("id", "match", "game", "url")
    return [
        column
        for column in data.columns
        if any(token in column.lower() for token in tokens)
        and safe_nunique(data[column]) > 1
    ][:10]


def summarize_key_duplicates(path: Path, data: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize duplicate counts for likely identifier columns.

    Args:
        path: Dataset path.
        data: Dataset to inspect.

    Returns:
        Duplicate summary rows.
    """

    rows: list[dict[str, Any]] = []
    for column in candidate_key_columns(data):
        duplicated = int(data[column].duplicated(keep=False).sum())
        rows.append(
            {
                "dataset": str(path.relative_to(PROJECT_ROOT)),
                "key_column": column,
                "unique_values": int(data[column].nunique(dropna=True)),
                "duplicated_rows_by_key": duplicated,
                "duplicated_share_by_key": round(duplicated / data.shape[0], 4)
                if data.shape[0]
                else 0.0,
            }
        )
    return rows


def profile_file(path: Path) -> tuple[DatasetProfile, pd.DataFrame | None]:
    """Profile a single dataset file.

    Args:
        path: Dataset path.

    Returns:
        Dataset profile and loaded DataFrame if available.
    """

    data, status, notes = load_dataset(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    rows = int(data.shape[0]) if data is not None else None
    columns = int(data.shape[1]) if data is not None else None
    profile = DatasetProfile(
        path=path,
        category=classify_dataset(path),
        file_type=path.suffix.lower().lstrip("."),
        size_mb=round(size_mb, 3),
        rows=rows,
        columns=columns,
        load_status=status,
        notes=notes,
    )
    return profile, data


def choose_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Choose the first existing column from a tuple of candidates.

    Args:
        data: Dataset to inspect.
        candidates: Candidate column names.

    Returns:
        Existing column name or None.
    """

    lowercase_map = {column.lower(): column for column in data.columns}
    for candidate in candidates:
        if candidate.lower() in lowercase_map:
            return lowercase_map[candidate.lower()]
    return None


def summarize_specific_lineage(
    loaded_data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Create whitepaper-oriented lineage rows for known project datasets.

    Args:
        loaded_data: Mapping from relative path to loaded DataFrame.

    Returns:
        Lineage summary rows.
    """

    lineage: list[dict[str, Any]] = []
    stages = [
        ("Raw GOL.GG", "golgg_matches.json", "all parsed GOL.GG records"),
        ("Raw Odds", "odds.csv", "all raw/cleaned bookmaker odds records"),
        (
            "OddsPortal Matches",
            "oddsportal_matches.csv",
            "source-side match metadata for odds mapping",
        ),
    ]
    for stage, filename, definition in stages:
        data = loaded_data.get(filename)
        if data is None:
            lineage.append(
                {
                    "stage": stage,
                    "file": filename,
                    "definition": definition,
                    "rows": "missing",
                    "columns": "missing",
                    "date_range": "missing",
                    "notes": "File not found or not loaded.",
                }
            )
            continue
        date_ranges = summarize_dates(Path(filename), data)
        date_range = ""
        if date_ranges:
            first = date_ranges[0]
            date_range = f"{first['min_date']} - {first['max_date']}"
        lineage.append(
            {
                "stage": stage,
                "file": filename,
                "definition": definition,
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]),
                "date_range": date_range,
                "notes": "Detected and loaded successfully.",
            }
        )
    return lineage


def summarize_matchability(
    loaded_data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Estimate simple overlap diagnostics between known datasets.

    Args:
        loaded_data: Mapping from relative path to loaded DataFrame.

    Returns:
        Rows with overlap diagnostics where comparable keys exist.
    """

    rows: list[dict[str, Any]] = []
    comparisons = [
        ("golgg_matches.json", "odds.csv", "match_id", "golgg_match_id"),
        ("oddsportal_matches.csv", "odds.csv", "url", "oddsportal_url"),
    ]
    for left_name, right_name, preferred_left_key, preferred_right_key in comparisons:
        left = loaded_data.get(left_name)
        right = loaded_data.get(right_name)
        if left is None or right is None:
            continue
        left_key = choose_column(left, (preferred_left_key,))
        right_key = choose_column(right, (preferred_right_key,))
        if left_key is None or right_key is None:
            rows.append(
                {
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "left_key": "not_detected",
                    "right_key": "not_detected",
                    "left_unique": "",
                    "right_unique": "",
                    "intersection": "",
                    "left_coverage": "",
                    "right_coverage": "",
                    "notes": "No obvious common key detected.",
                }
            )
            continue
        left_values = set(left[left_key].dropna().astype(str))
        right_values = set(right[right_key].dropna().astype(str))
        intersection = left_values & right_values
        rows.append(
            {
                "left_dataset": left_name,
                "right_dataset": right_name,
                "left_key": left_key,
                "right_key": right_key,
                "left_unique": len(left_values),
                "right_unique": len(right_values),
                "intersection": len(intersection),
                "left_coverage": round(len(intersection) / len(left_values), 4)
                if left_values
                else 0.0,
                "right_coverage": round(len(intersection) / len(right_values), 4)
                if right_values
                else 0.0,
                "notes": "Naive exact-key overlap; entity resolution may differ.",
            }
        )
    return rows


def summarize_targeted_quality(
    loaded_data: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Create thesis-oriented quality checks for point 3.

    Args:
        loaded_data: Mapping from relative path to loaded DataFrame.

    Returns:
        List of metric rows with values and interpretation notes.
    """

    rows: list[dict[str, Any]] = []
    golgg = loaded_data.get("golgg_matches.json")
    odds = loaded_data.get("odds.csv")
    oddsportal = loaded_data.get("oddsportal_matches.csv")

    if golgg is not None:
        rows.extend(summarize_golgg_quality_checks(golgg))
    if odds is not None:
        rows.extend(summarize_odds_quality_checks(odds))
    if oddsportal is not None and odds is not None:
        rows.extend(summarize_mapping_quality_checks(oddsportal, odds))
    if golgg is not None and odds is not None:
        rows.extend(summarize_golgg_odds_coverage_by_year(golgg, odds))
    return rows


def summarize_golgg_quality_checks(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize roster and BoN-like structural checks for GOL.GG.

    Args:
        data: Compact GOL.GG match-level DataFrame.

    Returns:
        List of quality-check rows.
    """

    rows: list[dict[str, Any]] = []
    total = max(int(data.shape[0]), 1)
    rows.append(
        quality_metric(
            "golgg_matches.json",
            "matches_without_full_rosters",
            int((~data["has_full_rosters"]).sum()),
            int(data.shape[0]),
            "Matches where at least one side does not have five top-level players.",
        )
    )
    for games_count, count in data["games_count"].value_counts().sort_index().items():
        rows.append(
            {
                "dataset": "golgg_matches.json",
                "metric": f"games_count_{games_count}",
                "value": int(count),
                "denominator": total,
                "share": round(float(count) / total, 4),
                "notes": "Observed number of games in a match payload; proxy for BoN/result format.",
            }
        )
    unusual = int((~data["games_count"].isin([1, 2, 3, 4, 5])).sum())
    rows.append(
        quality_metric(
            "golgg_matches.json",
            "unusual_games_count",
            unusual,
            int(data.shape[0]),
            "Matches with games_count outside 1-5.",
        )
    )
    return rows


def summarize_odds_quality_checks(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize missingness and extreme values in mapped odds data.

    Args:
        data: Mapped odds DataFrame.

    Returns:
        List of quality-check rows.
    """

    rows: list[dict[str, Any]] = []
    total = int(data.shape[0])
    avg_cols = ["avg_odds_home", "avg_odds_away", "avg_open_home", "avg_open_away"]
    bookmaker_cols = [column for column in data.columns if column.startswith("odds")]
    missing_avg = int(data[avg_cols].isna().any(axis=1).sum())
    rows.append(
        quality_metric(
            "odds.csv",
            "rows_with_missing_average_odds",
            missing_avg,
            total,
            "Rows where at least one average opening/closing odds field is missing.",
        )
    )
    for match_type, count in data["match_type"].value_counts().items():
        rows.append(
            {
                "dataset": "odds.csv",
                "metric": f"match_type_{match_type}",
                "value": int(count),
                "denominator": total,
                "share": round(float(count) / total, 4),
                "notes": "Entity-resolution match type in mapped odds dataset.",
            }
        )
    numeric_odds = data[bookmaker_cols + avg_cols].select_dtypes(include="number")
    invalid_low = int((numeric_odds < 1.0).sum().sum())
    extreme_high = int((numeric_odds > 50.0).sum().sum())
    rows.append(
        quality_metric(
            "odds.csv",
            "odds_values_below_1_0",
            invalid_low,
            int(numeric_odds.count().sum()),
            "Bookmaker/average odds values below 1.0 are structurally invalid.",
        )
    )
    rows.append(
        quality_metric(
            "odds.csv",
            "odds_values_above_50_0",
            extreme_high,
            int(numeric_odds.count().sum()),
            "Very high odds values requiring manual inspection before financial simulation.",
        )
    )
    for bookmaker in sorted(
        {
            column.removeprefix("odds1_").removesuffix("_open")
            for column in data.columns
            if column.startswith("odds1_") and column.endswith("_open")
        }
    ):
        column = f"odds1_{bookmaker}_open"
        if column in data.columns:
            available = int(data[column].notna().sum())
            rows.append(
                quality_metric(
                    "odds.csv",
                    f"bookmaker_open_coverage_{bookmaker}",
                    available,
                    total,
                    "Share of matches with opening odds for this bookmaker.",
                )
            )
    return rows


def summarize_mapping_quality_checks(
    oddsportal: pd.DataFrame, odds: pd.DataFrame
) -> list[dict[str, Any]]:
    """Summarize URL-level mapping from OddsPortal source to mapped odds.

    Args:
        oddsportal: Source-side OddsPortal match table.
        odds: Final mapped odds table.

    Returns:
        List of mapping quality rows.
    """

    left = set(oddsportal["url"].dropna().astype(str))
    right = set(odds["oddsportal_url"].dropna().astype(str))
    intersection = left & right
    return [
        quality_metric(
            "oddsportal_matches.csv -> odds.csv",
            "oddsportal_url_mapped",
            len(intersection),
            len(left),
            "OddsPortal URLs retained after mapping to GOL.GG matches.",
        ),
        quality_metric(
            "oddsportal_matches.csv -> odds.csv",
            "oddsportal_url_unmapped",
            len(left - right),
            len(left),
            "OddsPortal URLs not present in final mapped odds.csv.",
        ),
    ]


def summarize_golgg_odds_coverage_by_year(
    golgg: pd.DataFrame, odds: pd.DataFrame
) -> list[dict[str, Any]]:
    """Calculate GOL.GG-to-odds coverage by match year.

    Args:
        golgg: Compact GOL.GG match table.
        odds: Final mapped odds table.

    Returns:
        List of yearly coverage rows.
    """

    left = golgg.copy()
    right = odds.copy()
    left["year"] = pd.to_datetime(left["date"], errors="coerce").dt.year
    right["year"] = pd.to_datetime(right["golgg_date"], errors="coerce").dt.year
    mapped_ids = set(right["golgg_match_id"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for year, group in left.groupby("year", dropna=True):
        source_ids = set(group["match_id"].dropna().astype(str))
        mapped = len(source_ids & mapped_ids)
        rows.append(
            quality_metric(
                "golgg_matches.json -> odds.csv",
                f"mapped_coverage_year_{int(year)}",
                mapped,
                len(source_ids),
                "Share of GOL.GG matches in this year with mapped odds.",
            )
        )
    return rows


def quality_metric(
    dataset: str, metric: str, value: int, denominator: int, notes: str
) -> dict[str, Any]:
    """Build a standard quality metric row.

    Args:
        dataset: Dataset or mapping label.
        metric: Metric name.
        value: Numerator.
        denominator: Denominator.
        notes: Interpretation note.

    Returns:
        Standardized metric dictionary.
    """

    return {
        "dataset": dataset,
        "metric": metric,
        "value": int(value),
        "denominator": int(denominator),
        "share": round(float(value) / denominator, 4) if denominator else 0.0,
        "notes": notes,
    }


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Render a DataFrame as a simple GitHub/Obsidian Markdown table.

    This avoids adding ``tabulate`` as a hard dependency only for report output.

    Args:
        data: DataFrame to render.

    Returns:
        Markdown table string.
    """

    if data.empty:
        return "No rows."
    string_data = data.fillna("").astype(str)
    headers = list(string_data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in string_data.iterrows():
        safe_values = [value.replace("|", "\\|") for value in row.tolist()]
        lines.append("| " + " | ".join(safe_values) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    inventory: pd.DataFrame,
    quality: pd.DataFrame,
    lineage: pd.DataFrame,
    matchability: pd.DataFrame,
    targeted_quality: pd.DataFrame,
) -> Path:
    """Write an Obsidian-compatible profiling summary.

    Args:
        inventory: Dataset inventory table.
        quality: Data quality summary table.
        lineage: Known lineage summary table.
        matchability: Matchability diagnostics table.
        targeted_quality: Thesis-oriented quality checks.

    Returns:
        Path to the generated Markdown file.
    """

    WHITEPAPER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WHITEPAPER_DIR / "03_dataset_profile_autogenerated.md"
    primary_inventory = inventory[
        inventory["category"].isin(
            [
                "source_of_truth_sport",
                "source_of_truth_market_mapped",
                "mapping_audit_input",
            ]
        )
    ].copy()
    content = [
        "---",
        "type: generated-data-profile",
        "tags: [whitepaper, data-profile, autogenerated, eda]",
        "project: inzynierka",
        "date: 2026-04-30",
        "source_script: scripts/00_profile_datasets_for_whitepaper.py",
        "---",
        "",
        "# Autogenerated dataset profile for [[00_plan_whitepaper_v2]] point 3",
        "",
        "> [!abstract]",
        "> Ten plik jest automatycznie wygenerowanym profilem danych. Służy jako materiał wejściowy do napisania rozdziału `# 3. Dane, pipeline i kontrola jakości`, a nie jako finalna narracja whitepapera.",
        "",
        "## 1. Primary datasets and pipeline artefacts",
        "",
        dataframe_to_markdown(primary_inventory),
        "",
        "## 2. Known lineage stages",
        "",
        dataframe_to_markdown(lineage),
        "",
        "## 3. Data quality summary",
        "",
        dataframe_to_markdown(quality.head(40)),
        "",
        "## 4. Naive matchability diagnostics",
        "",
        dataframe_to_markdown(matchability)
        if not matchability.empty
        else "No comparable keys detected.",
        "",
        "## 5. Targeted quality checks for chapter 3",
        "",
        dataframe_to_markdown(targeted_quality)
        if not targeted_quality.empty
        else "No targeted quality checks generated.",
        "",
        "## 6. Generated CSV artefacts",
        "",
        "- `docs/assets/data_profile/dataset_inventory.csv`",
        "- `docs/assets/data_profile/column_dictionary.csv`",
        "- `docs/assets/data_profile/data_quality_summary.csv`",
        "- `docs/assets/data_profile/date_coverage.csv`",
        "- `docs/assets/data_profile/key_duplicates.csv`",
        "- `docs/assets/data_profile/lineage_summary.csv`",
        "- `docs/assets/data_profile/matchability_summary.csv`",
        "- `docs/assets/data_profile/targeted_quality_checks.csv`",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")
    return output_path


def main() -> None:
    """Run the dataset profiling workflow."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_dataset_files(PROJECT_ROOT)

    profiles: list[DatasetProfile] = []
    column_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    loaded_data: dict[str, pd.DataFrame] = {}

    for path in files:
        profile, data = profile_file(path)
        profiles.append(profile)
        relative_name = str(path.relative_to(PROJECT_ROOT))
        if data is None:
            continue
        if path.parent == PROJECT_ROOT:
            loaded_data[path.name] = data
        loaded_data[relative_name] = data
        column_rows.extend(summarize_columns(path, data))
        quality_rows.append(summarize_quality(path, data))
        date_rows.extend(summarize_dates(path, data))
        duplicate_rows.extend(summarize_key_duplicates(path, data))

    inventory = pd.DataFrame(
        [
            {
                "dataset": str(profile.path.relative_to(PROJECT_ROOT)),
                "category": profile.category,
                "file_type": profile.file_type,
                "size_mb": profile.size_mb,
                "rows": profile.rows,
                "columns": profile.columns,
                "load_status": profile.load_status,
                "notes": profile.notes,
            }
            for profile in profiles
        ]
    )
    columns = pd.DataFrame(column_rows)
    quality = pd.DataFrame(quality_rows).sort_values("dataset")
    dates = pd.DataFrame(date_rows)
    duplicates = pd.DataFrame(duplicate_rows)
    lineage = pd.DataFrame(summarize_specific_lineage(loaded_data))
    matchability = pd.DataFrame(summarize_matchability(loaded_data))
    targeted_quality = pd.DataFrame(summarize_targeted_quality(loaded_data))

    inventory.to_csv(ASSETS_DIR / "dataset_inventory.csv", index=False)
    columns.to_csv(ASSETS_DIR / "column_dictionary.csv", index=False)
    quality.to_csv(ASSETS_DIR / "data_quality_summary.csv", index=False)
    dates.to_csv(ASSETS_DIR / "date_coverage.csv", index=False)
    duplicates.to_csv(ASSETS_DIR / "key_duplicates.csv", index=False)
    lineage.to_csv(ASSETS_DIR / "lineage_summary.csv", index=False)
    matchability.to_csv(ASSETS_DIR / "matchability_summary.csv", index=False)
    targeted_quality.to_csv(
        ASSETS_DIR / "targeted_quality_checks.csv", index=False
    )

    markdown_path = write_markdown_summary(
        inventory=inventory,
        quality=quality,
        lineage=lineage,
        matchability=matchability,
        targeted_quality=targeted_quality,
    )
    print(f"Profiled {len(files)} dataset files.")
    print(f"Generated artefacts in: {ASSETS_DIR}")
    print(f"Generated Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
