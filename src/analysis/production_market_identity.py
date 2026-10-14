"""Outcome-independent identity alignment for production market inputs."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import pandas as pd


_IDENTITY_FIELDS = ("t1_id", "t2_id", "t1_name", "t2_name")
_ALIAS_CONTEXT_FIELDS = (
    "source_system", "league_pattern", "tournament_pattern", "valid_from", "valid_to"
)


def _has(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _norm(value: Any) -> str:
    if not _has(value):
        return ""
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value)).casefold()
        if char.isalnum()
    )


def _id(value: Any) -> str | None:
    if not _has(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError, OverflowError):
        pass
    return text


def _day(value: Any) -> date | None:
    if not _has(value):
        return None
    return pd.Timestamp(value).date()


def _unique_key(frame: pd.DataFrame, column: str, label: str) -> None:
    if column in frame and frame[column].duplicated().any():
        raise ValueError(f"duplicate {label} primary key: {column}")


def _truth(value: Any) -> bool:
    return value is True or _id(value) == "1"


def _source_identity(record: dict[str, Any]) -> dict[str, Any] | None:
    games = record.get("games")
    first = games[0] if isinstance(games, (list, tuple)) and games else None
    # Preserve the audited rule: a complete first-map identity is atomic; otherwise
    # accept the complete header tuple, never a mixture of the two.
    selected = first if isinstance(first, dict) and all(_has(first.get(k)) for k in _IDENTITY_FIELDS) else record
    if not all(_has(selected.get(k)) for k in _IDENTITY_FIELDS):
        return None
    return {
        "t1_id": _id(selected["t1_id"]),
        "t2_id": _id(selected["t2_id"]),
        "t1_name": str(selected["t1_name"]),
        "t2_name": str(selected["t2_name"]),
    }


def build_identity_tables(
    matches: pd.DataFrame,
    mappings: pd.DataFrame,
    odds: pd.DataFrame,
    aliases: pd.DataFrame,
    sources: dict[str, list[dict[str, Any]]],
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build oriented match identities and quote-name alignments without outcomes."""
    for frame, col, label in (
        (matches, "canonical_match_id", "match"),
        (mappings, "id", "mapping"),
        (odds, "id", "quote"),
        (aliases, "id", "alias"),
        (forecasts, "golgg_match_id", "forecast"),
    ):
        _unique_key(frame, col, label)

    alias_graph: dict[str, set[str]] = defaultdict(set)
    alias_used = 0
    alias_skipped: Counter[str] = Counter()
    for row in aliases.to_dict("records"):
        active = _truth(row.get("is_active"))
        blocked = _truth(row.get("is_blocked"))
        if not active or blocked or str(row.get("review_status", "")).strip().lower() != "approved":
            alias_skipped["not_active_approved_unblocked"] += 1
            continue
        if any(_has(row.get(field)) for field in _ALIAS_CONTEXT_FIELDS):
            alias_skipped["contextual_alias_not_applied"] += 1
            continue
        names = {_norm(row.get(field)) for field in
                 ("normalized_name", "alias", "raw_name", "normalized_alias") if _has(row.get(field))}
        names.discard("")
        for name in names:
            alias_graph[name].update(names)
        alias_used += 1

    def expand(values: list[Any]) -> set[str]:
        seen = {_norm(value) for value in values if _has(value)}
        seen.discard("")
        pending = list(seen)
        while pending:
            for alias in alias_graph[pending.pop()] - seen:
                seen.add(alias)
                pending.append(alias)
        return seen

    mappings_by_match: dict[Any, list[str]] = defaultdict(list)
    for row in mappings.to_dict("records"):
        mappings_by_match[row.get("canonical_match_id")].append(_id(row.get("golgg_match_id")) or "")

    source_by_id: dict[str, dict[str, Any]] = {}
    duplicate_source_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    for source_name, records in sources.items():
        source_counts[source_name] = len(records)
        seen_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            gid = _id(record.get("match_id"))
            if gid is None:
                continue
            if gid in seen_ids:
                duplicate_ids.add(gid)
            seen_ids.add(gid)
            identity = _source_identity(record)
            if identity is None:
                continue
            per_source[gid].append({
                **identity,
                "date": record.get("date"),
                "best_of": record.get("best_of"),
                "sname_t1": record.get("sname_t1"),
                "sname_t2": record.get("sname_t2"),
                "identity_source": source_name,
            })
        duplicate_source_ids.update(duplicate_ids)
        for gid, candidates in per_source.items():
            if gid in duplicate_ids or len(candidates) != 1:
                continue
            if gid not in source_by_id:
                source_by_id[gid] = candidates[0]
    duplicate_source_ids.difference_update(source_by_id)

    forecast_by_id = {
        _id(row.get("golgg_match_id")): row
        for row in forecasts.to_dict("records") if _id(row.get("golgg_match_id")) is not None
    }
    identity_rows: list[dict[str, Any]] = []
    match_side_sets: dict[Any, tuple[set[str], set[str]]] = {}
    for match in matches.to_dict("records"):
        canonical_id = match.get("canonical_match_id")
        candidates = list(dict.fromkeys(gid for gid in mappings_by_match.get(canonical_id, []) if gid))
        row: dict[str, Any] = {
            "canonical_match_id": canonical_id,
            "golgg_match_id": None,
            "identity_status": "no_mapping",
            "canonical_a_is_team1": pd.NA,
            "canonical_team_a": match.get("team_a_name"),
            "canonical_team_b": match.get("team_b_name"),
            "canonical_date": match.get("start_time_normalized"),
            "canonical_best_of": match.get("best_of"),
        }
        if len(candidates) > 1:
            row["identity_status"] = "multiple_golgg_ids"
        elif len(candidates) == 1:
            gid = candidates[0]
            row["golgg_match_id"] = gid
            if gid in duplicate_source_ids:
                row["identity_status"] = "missing_or_duplicate_source_id"
            else:
                source = source_by_id.get(gid)
                if source is None:
                    row["identity_status"] = "missing_or_duplicate_source_id"
                else:
                    row.update({
                        "identity_source": source["identity_source"],
                        "team1_id": source["t1_id"], "team2_id": source["t2_id"],
                        "golgg_team1": source["t1_name"], "golgg_team2": source["t2_name"],
                        "golgg_date": source["date"], "golgg_best_of": source["best_of"],
                        "source_date": source["date"], "source_best_of": source["best_of"],
                        "source_team1_id": source["t1_id"], "source_team2_id": source["t2_id"],
                        "source_golgg_team1": source["t1_name"], "source_golgg_team2": source["t2_name"],
                    })
                    side_a = expand([match.get("team_a_name"), match.get("normalized_team_a")])
                    side_b = expand([match.get("team_b_name"), match.get("normalized_team_b")])
                    side_1 = expand([source.get("t1_name") if _has(source.get("t1_name")) else source.get("sname_t1")])
                    side_2 = expand([source.get("t2_name") if _has(source.get("t2_name")) else source.get("sname_t2")])
                    direct = bool(side_a & side_1) and bool(side_b & side_2)
                    reverse = bool(side_a & side_2) and bool(side_b & side_1)
                    if side_a & side_b or side_1 & side_2 or (direct and reverse):
                        row["identity_status"] = "ambiguous_alias_identity"
                    elif direct or reverse:
                        row["identity_status"] = "verified"
                        row["canonical_a_is_team1"] = direct
                    else:
                        row["identity_status"] = "unresolved_team_names"
                    match_side_sets[canonical_id] = (side_a, side_b)
                    try:
                        gap = abs((_day(match.get("start_time_normalized")) - _day(source.get("date"))).days)
                    except (TypeError, ValueError, OverflowError):
                        gap = None
                    row["date_gap_days"] = gap
                    row["date_within_one_day"] = gap is not None and gap <= 1
                    row["best_of_equal"] = (
                        _has(match.get("best_of")) and _has(source.get("best_of"))
                        and str(match.get("best_of")) == str(source.get("best_of"))
                    )
                    forecast = forecast_by_id.get(gid)
                    row["prediction_available"] = forecast is not None
                    if forecast is not None:
                        source_ids = (_id(row.get("team1_id")), _id(row.get("team2_id")))
                        forecast_ids = (_id(forecast.get("team1_id")), _id(forecast.get("team2_id")))
                        direct_ids = source_ids == forecast_ids
                        reverse_ids = source_ids == forecast_ids[::-1]
                        row["prediction_team_ids_match"] = bool(direct_ids or reverse_ids)
                        row["source_order_reversed_to_prediction"] = bool(reverse_ids and not direct_ids)
                        if reverse_ids and not direct_ids:
                            row["team1_id"], row["team2_id"] = row["team2_id"], row["team1_id"]
                            row["golgg_team1"], row["golgg_team2"] = row["golgg_team2"], row["golgg_team1"]
                            if row["identity_status"] == "verified":
                                row["canonical_a_is_team1"] = not bool(row["canonical_a_is_team1"])
                        elif not direct_ids:
                            row["identity_status"] = "prediction_team_pair_mismatch"
                            row["canonical_a_is_team1"] = pd.NA
                        row["prediction_date_matches_source"] = (
                            _day(source.get("date")) is not None and _day(forecast.get("date")) == _day(source.get("date"))
                        )
                        row["prediction_best_of_matches_source"] = (
                            _has(source.get("best_of")) and _has(forecast.get("best_of"))
                            and str(source.get("best_of")) == str(forecast.get("best_of"))
                        )
        identity_rows.append(row)

    identities = pd.DataFrame(identity_rows)
    if "canonical_a_is_team1" in identities:
        identities["canonical_a_is_team1"] = pd.array(identities["canonical_a_is_team1"], dtype="boolean")

    match_by_id = {row["canonical_match_id"]: row for row in identity_rows}
    quote_rows: list[dict[str, Any]] = []
    for quote in odds.to_dict("records"):
        canonical_id = quote.get("canonical_match_id")
        match = match_by_id.get(canonical_id)
        result: dict[str, Any] = {
            "quote_identity_status": "match_identity_unresolved",
            "raw_a_is_team1": pd.NA,
            "raw_a_is_canonical_a": pd.NA,
        }
        if match is not None and match.get("identity_status") == "verified":
            side_a, side_b = match_side_sets.get(canonical_id, (set(), set()))
            quote_a = expand([quote.get("raw_team_a")])
            quote_b = expand([quote.get("raw_team_b")])
            direct = bool(quote_a & side_a) and bool(quote_b & side_b)
            reverse = bool(quote_a & side_b) and bool(quote_b & side_a)
            if quote_a & quote_b or (direct and reverse):
                result["quote_identity_status"] = "ambiguous_quote_names"
            elif direct or reverse:
                a_is_canonical_a = direct
                canonical_a_team1 = match.get("canonical_a_is_team1")
                result.update({
                    "quote_identity_status": "verified",
                    "raw_a_is_canonical_a": a_is_canonical_a,
                    "raw_a_is_team1": bool(canonical_a_team1) if a_is_canonical_a else not bool(canonical_a_team1),
                })
            else:
                result["quote_identity_status"] = "unresolved_quote_names"
        quote_rows.append({
            "id": quote.get("id"), "canonical_match_id": canonical_id,
            "bookmaker": quote.get("bookmaker"), "raw_team_a": quote.get("raw_team_a"),
            "raw_team_b": quote.get("raw_team_b"), **result,
        })
    quote_alignment = pd.DataFrame(quote_rows)
    for column in ("raw_a_is_team1", "raw_a_is_canonical_a"):
        if column in quote_alignment:
            quote_alignment[column] = pd.array(quote_alignment[column], dtype="boolean")

    receipt = {
        "identity_only": True,
        "version": "v2_coherent_id_name_pairs",
        "team1_orientation": "forecast team1/team2 IDs when forecast exists; original source order preserved in source_* fields",
        "outcomes_used_for_alignment": False,
        "normalization": "Unicode NFKD casefold; retain alphanumeric; preserve academy/challenger words",
        "alias_policy": "approved active nonblocked global unbounded aliases only; contextual entries excluded",
        "source_priority": list(sources),
        "source_record_counts": source_counts,
        "aliases_used": alias_used,
        "aliases_skipped": dict(alias_skipped),
        "match_counts": dict(Counter(identities.get("identity_status", pd.Series(dtype=str)).dropna())),
        "quote_counts": dict(Counter(quote_alignment.get("quote_identity_status", pd.Series(dtype=str)).dropna())),
    }
    return identities, quote_alignment, receipt
