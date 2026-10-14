#!/usr/bin/env python3
"""Replay conservative team-identity and match-link policies on a local snapshot.

The script is read-only. It never connects to the operational database and never
rewrites canonical IDs. Input CSV files are produced by the snapshot export kept
beside the downloaded data under ``data/identity_simulation``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from betting_app.core.matching import ALIASES, normalize_team_name
from betting_app.services.canonical_match_service import TEAM_ALIASES
from betting_app.services.mapping_service import BOOKMAKER_TO_GOLGG_ALIASES
from betting_app.services.team_alias_service import alias_lookup_key, is_short_alias

KNOWN_BAD_CANONICAL_IDS = {144791, 144792, 144795, 144796}
CONTAINMENT_MIN_CONFIDENCE = 0.95
CONTAINMENT_MAX_DATE_DAYS = 1
UPCOMING_AUTO_WINDOW_MINUTES = 90

# Confirmed by the same LPL fixture, kickoff, opponent, and deployed GOL.GG
# targets. This is deliberately competition-scoped rather than a global fuzzy
# abbreviation rule.
PROPOSED_SCOPED_ALIASES = {
    ("nip", "lpl"): "ninjas in pyjamas",
}

COMPETITION_MARKERS: tuple[tuple[str, str], ...] = (
    ("lck challengers", "lck_cl"),
    ("lck cl", "lck_cl"),
    ("lck challenge", "lck_cl"),
    ("lck", "lck"),
    ("lplol", "lpol"),
    ("lpol", "lpol"),
    ("lpl", "lpl"),
    ("lec", "lec"),
    ("na challengers", "nacl"),
    ("nacl", "nacl"),
    ("lcs", "lcs"),
    ("lastlap", "les"),
    ("superliga", "les"),
    ("lvp", "les"),
    ("les", "les"),
    ("lfl", "lfl"),
    ("prime league", "prime_league"),
    ("emea masters", "emea_masters"),
    ("cd", "circuito_desafiante"),
    ("circuito desafiante", "circuito_desafiante"),
    ("cblol", "cblol"),
    ("lcp", "lcp"),
    ("ljl", "ljl"),
    ("lrs", "lrs"),
    ("lrn", "lrn"),
    ("tcl", "tcl"),
    ("nlc", "nlc"),
    ("rift legends", "rift_legends"),
    ("liga regional sur", "lrs"),
    ("japan league", "ljl"),
    ("pg nationals", "pg_nationals"),
    ("kespa cup", "kespa_cup"),
    ("gll", "gll"),
    ("lit", "lit"),
    ("hellenic legends", "hll"),
    ("hll", "hll"),
    ("road of legends", "road_of_legends"),
    ("hitpoint", "hitpoint"),
    ("esports world cup", "ewc"),
    ("world cup", "ewc"),
    ("wolrd cup", "ewc"),
    ("ewc", "ewc"),
    ("msi", "msi"),
)


@dataclass(frozen=True)
class Resolution:
    status: str
    key: str | None
    source: str
    reason: str | None = None


@dataclass(frozen=True)
class ReplaySummary:
    population: int
    auto_accept: int
    review: int
    auto_accept_rate: float
    agrees_with_existing: int
    disagrees_with_existing: int
    known_bad_auto_accepted: int
    known_bad_routed_to_review: int


def _clean_optional(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def competition_family(value: Any) -> str | None:
    text = _clean_optional(value)
    if not text:
        return None
    normalized = alias_lookup_key(text)
    for marker, family in COMPETITION_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return family
    return None


def semantic_key(value: Any) -> str:
    """Apply current static semantic aliases without querying a database."""

    normalized = normalize_team_name(_clean_optional(value) or "")
    compact = normalized.replace(" ", "")
    target = TEAM_ALIASES.get(normalized) or TEAM_ALIASES.get(compact)
    return normalize_team_name(target) if target else normalized


def _legacy_alias_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for aliases in (ALIASES, TEAM_ALIASES, BOOKMAKER_TO_GOLGG_ALIASES):
        for source, target in aliases.items():
            keys = {alias_lookup_key(source), alias_lookup_key(source).replace(" ", "")}
            for key in keys:
                if key:
                    index[key].add(semantic_key(target))
    return dict(index)


LEGACY_ALIAS_INDEX = _legacy_alias_index()


def _date_value(value: Any) -> date | None:
    text = _clean_optional(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, format="mixed", errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _contains_context(value: str | None, pattern: Any) -> bool:
    pattern_text = _clean_optional(pattern)
    if not pattern_text:
        return True
    if not value:
        return False
    return semantic_key(pattern_text) in semantic_key(value)


class SnapshotAliasResolver:
    """Resolve snapshot aliases, returning ambiguity instead of row-order wins."""

    def __init__(self, aliases: pd.DataFrame):
        self._by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in aliases.to_dict("records"):
            normalized = _clean_optional(row.get("normalized_name"))
            if not normalized:
                continue
            self._by_key[normalized].append(row)
            compact = normalized.replace(" ", "")
            if compact != normalized:
                self._by_key[compact].append(row)

    def resolve(
        self,
        raw_name: Any,
        *,
        source_system: str | None,
        competition: str | None,
        tournament: str | None,
        on_date: Any,
    ) -> Resolution:
        name = _clean_optional(raw_name) or ""
        lookup = alias_lookup_key(name)
        family = competition_family(competition or tournament)
        proposed = PROPOSED_SCOPED_ALIASES.get((lookup, family or ""))
        if proposed:
            return Resolution("resolved", semantic_key(proposed), "proposed_scoped")

        lookup_keys = {lookup, lookup.replace(" ", ""), semantic_key(name)}
        rows: dict[int, dict[str, Any]] = {}
        for key in lookup_keys:
            for row in self._by_key.get(key, []):
                rows[int(row["id"])] = row

        applicable: list[tuple[int, dict[str, Any]]] = []
        event_date = _date_value(on_date)
        for row in rows.values():
            if int(row.get("is_active") or 0) != 1:
                continue
            row_source = _clean_optional(row.get("source_system"))
            if row_source and (not source_system or row_source.lower() != source_system.lower()):
                continue
            if not _contains_context(competition, row.get("league_pattern")):
                continue
            if not _contains_context(tournament, row.get("tournament_pattern")):
                continue
            valid_from = _date_value(row.get("valid_from"))
            valid_to = _date_value(row.get("valid_to"))
            if valid_from and (event_date is None or event_date < valid_from):
                continue
            if valid_to and (event_date is None or event_date > valid_to):
                continue
            has_non_source_scope = any(
                _clean_optional(row.get(column))
                for column in ("league_pattern", "tournament_pattern", "valid_from", "valid_to")
            )
            if is_short_alias(name) and not has_non_source_scope:
                continue
            specificity = sum(
                bool(_clean_optional(row.get(column)))
                for column in ("source_system", "league_pattern", "tournament_pattern", "valid_from", "valid_to")
            )
            applicable.append((specificity, row))

        if applicable:
            best_specificity = max(item[0] for item in applicable)
            best_rows = [row for specificity, row in applicable if specificity == best_specificity]
            blocked = [row for row in best_rows if int(row.get("is_blocked") or 0) == 1 or row.get("source") == "blocked"]
            if blocked:
                return Resolution("blocked", None, "database", "applicable_block")
            manual_rows = [row for row in best_rows if str(row.get("source") or "").startswith("manual")]
            selected_rows = manual_rows or best_rows
            targets = {
                semantic_key(row.get("alias"))
                for row in selected_rows
                if _clean_optional(row.get("alias"))
            }
            targets.discard("")
            if len(targets) == 1:
                return Resolution("resolved", targets.pop(), "database")
            if len(targets) > 1:
                return Resolution("ambiguous", None, "database", "conflicting_applicable_targets")

        legacy_targets: set[str] = set()
        for key in lookup_keys:
            legacy_targets.update(LEGACY_ALIAS_INDEX.get(key, set()))
        legacy_targets.discard("")
        if len(legacy_targets) == 1:
            return Resolution("resolved", legacy_targets.pop(), "legacy_static")
        if len(legacy_targets) > 1:
            return Resolution("ambiguous", None, "legacy_static", "conflicting_static_targets")

        key = semantic_key(name)
        if not key:
            return Resolution("unresolved", None, "none", "empty_identity")
        return Resolution("resolved", key, "normalized_raw")


def pair_resolution(
    resolver: SnapshotAliasResolver,
    left: Any,
    right: Any,
    *,
    source_system: str,
    competition: Any,
    tournament: Any,
    on_date: Any,
) -> tuple[Resolution, Resolution, frozenset[str] | None]:
    a = resolver.resolve(
        left,
        source_system=source_system,
        competition=_clean_optional(competition),
        tournament=_clean_optional(tournament),
        on_date=on_date,
    )
    b = resolver.resolve(
        right,
        source_system=source_system,
        competition=_clean_optional(competition),
        tournament=_clean_optional(tournament),
        on_date=on_date,
    )
    if a.status != "resolved" or b.status != "resolved" or not a.key or not b.key or a.key == b.key:
        return a, b, None
    return a, b, frozenset((a.key, b.key))


def load_snapshot(snapshot_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "canonical": "identity_canonical_matches.csv",
        "upcoming": "identity_upcoming_matches.csv",
        "aliases": "identity_team_aliases.csv",
        "golgg_teams": "identity_golgg_teams.csv",
        "golgg": "identity_golgg_matches.csv",
        "mappings": "identity_golgg_match_mappings.csv",
        "odds": "identity_odds_evidence.csv",
    }
    frames = {key: pd.read_csv(snapshot_dir / filename) for key, filename in names.items()}
    expected_unique = {
        "canonical": "id",
        "upcoming": "id",
        "aliases": "id",
        "golgg_teams": "id",
        "golgg": "match_id",
        "mappings": "id",
    }
    for key, column in expected_unique.items():
        if frames[key][column].isna().any() or frames[key][column].duplicated().any():
            raise ValueError(f"{key}.{column} is not a complete unique key")
    return frames


def joined_mapping_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    canonical = frames["canonical"].add_prefix("canonical_")
    golgg = frames["golgg"].add_prefix("golgg_")
    rows = frames["mappings"].merge(
        canonical,
        left_on="canonical_match_id",
        right_on="canonical_id",
        how="left",
        validate="many_to_one",
    ).merge(
        golgg,
        left_on="golgg_match_id",
        right_on="golgg_match_id",
        how="left",
        validate="many_to_one",
    )
    rows["canonical_date"] = pd.to_datetime(
        rows["canonical_start_time_normalized"], format="mixed", utc=True, errors="coerce"
    ).dt.date
    rows["golgg_date_value"] = pd.to_datetime(
        rows["golgg_date"], format="mixed", utc=True, errors="coerce"
    ).dt.date
    rows["date_distance_days"] = rows.apply(
        lambda row: abs((row["canonical_date"] - row["golgg_date_value"]).days)
        if pd.notna(row["canonical_date"]) and pd.notna(row["golgg_date_value"])
        else math.nan,
        axis=1,
    )
    rows["canonical_competition_family"] = rows["canonical_league"].map(competition_family)
    rows["golgg_competition_family"] = rows["golgg_tournament_name"].map(competition_family)
    rows["competition_compatible"] = (
        rows["canonical_competition_family"].notna()
        & rows["golgg_competition_family"].notna()
        & (rows["canonical_competition_family"] == rows["golgg_competition_family"])
    )
    return rows


def simulate_containment(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["containment_one_day_accept"] = (
        (result["confidence"] >= CONTAINMENT_MIN_CONFIDENCE)
        & (result["date_distance_days"] <= CONTAINMENT_MAX_DATE_DAYS)
    )
    result["containment_exact_date_accept"] = (
        (result["confidence"] >= CONTAINMENT_MIN_CONFIDENCE)
        & (result["date_distance_days"] == 0)
    )
    result["competition_accept"] = result["containment_exact_date_accept"] & result["competition_compatible"]
    reasons: list[str] = []
    for row in result.to_dict("records"):
        row_reasons: list[str] = []
        if float(row["confidence"]) < CONTAINMENT_MIN_CONFIDENCE:
            row_reasons.append("confidence_below_0_95")
        if pd.isna(row["date_distance_days"]):
            row_reasons.append("missing_date")
        elif float(row["date_distance_days"]) > CONTAINMENT_MAX_DATE_DAYS:
            row_reasons.append("date_distance_over_1_day")
        if not bool(row["competition_compatible"]):
            if not row["canonical_competition_family"] or not row["golgg_competition_family"]:
                row_reasons.append("unknown_competition_family")
            else:
                row_reasons.append("competition_conflict")
        reasons.append(",".join(row_reasons) or "accepted")
    result["containment_reasons"] = reasons
    return result


def _resolved_canonical_candidates(
    frames: dict[str, pd.DataFrame], resolver: SnapshotAliasResolver
) -> tuple[dict[date, list[dict[str, Any]]], pd.DataFrame]:
    resolved_rows: list[dict[str, Any]] = []
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in frames["canonical"].to_dict("records"):
        event_date = _date_value(row.get("start_time_normalized"))
        a, b, pair = pair_resolution(
            resolver,
            row.get("team_a_name"),
            row.get("team_b_name"),
            source_system="bookmaker",
            competition=row.get("league"),
            tournament=None,
            on_date=event_date,
        )
        item = {
            "canonical_match_id": int(row["id"]),
            "event_date": event_date,
            "competition_family": competition_family(row.get("league")),
            "team_pair": pair,
            "team_a_status": a.status,
            "team_b_status": b.status,
        }
        resolved_rows.append(item)
        if event_date is not None and pair is not None and item["competition_family"] is not None:
            by_date[event_date].append(item)
    return by_date, pd.DataFrame(resolved_rows)


def simulate_identity_replay(
    frames: dict[str, pd.DataFrame], resolver: SnapshotAliasResolver
) -> tuple[pd.DataFrame, ReplaySummary]:
    by_date, _ = _resolved_canonical_candidates(frames, resolver)
    golgg_by_id = frames["golgg"].set_index("match_id", drop=False)
    decisions: list[dict[str, Any]] = []
    for mapping in frames["mappings"].to_dict("records"):
        golgg_id = int(mapping["golgg_match_id"])
        row = golgg_by_id.loc[golgg_id]
        event_date = _date_value(row.get("date"))
        family = competition_family(row.get("tournament_name"))
        a, b, pair = pair_resolution(
            resolver,
            row.get("team1_name"),
            row.get("team2_name"),
            source_system="golgg",
            competition=row.get("tournament_name"),
            tournament=row.get("tournament_name"),
            on_date=event_date,
        )
        candidates: set[int] = set()
        if event_date is not None and family is not None and pair is not None:
            for candidate in by_date.get(event_date, []):
                if candidate["competition_family"] == family and candidate["team_pair"] == pair:
                    candidates.add(int(candidate["canonical_match_id"]))
        if a.status != "resolved" or b.status != "resolved" or pair is None:
            outcome = "review_identity"
        elif family is None:
            outcome = "review_competition"
        elif not candidates:
            outcome = "review_no_candidate"
        elif len(candidates) > 1:
            outcome = "review_ambiguous_candidates"
        else:
            outcome = "auto_accept"
        chosen = next(iter(candidates)) if outcome == "auto_accept" else None
        existing = int(mapping["canonical_match_id"])
        decisions.append(
            {
                "mapping_id": int(mapping["id"]),
                "golgg_match_id": golgg_id,
                "existing_canonical_match_id": existing,
                "outcome": outcome,
                "chosen_canonical_match_id": chosen,
                "candidate_count": len(candidates),
                "candidate_ids": ",".join(str(value) for value in sorted(candidates)),
                "agrees_with_existing": chosen == existing if chosen is not None else False,
                "known_bad_existing": existing in KNOWN_BAD_CANONICAL_IDS,
                "golgg_team_a_status": a.status,
                "golgg_team_b_status": b.status,
                "competition_family": family,
            }
        )
    decision_frame = pd.DataFrame(decisions)
    accepted = decision_frame[decision_frame["outcome"] == "auto_accept"]
    known_bad = decision_frame[decision_frame["known_bad_existing"]]
    summary = ReplaySummary(
        population=len(decision_frame),
        auto_accept=len(accepted),
        review=len(decision_frame) - len(accepted),
        auto_accept_rate=round(len(accepted) / len(decision_frame), 6) if len(decision_frame) else 0.0,
        agrees_with_existing=int(accepted["agrees_with_existing"].sum()),
        disagrees_with_existing=int((~accepted["agrees_with_existing"]).sum()),
        known_bad_auto_accepted=int((known_bad["outcome"] == "auto_accept").sum()),
        known_bad_routed_to_review=int((known_bad["outcome"] != "auto_accept").sum()),
    )
    return decision_frame, summary


def simulate_upcoming_duplicates(
    frames: dict[str, pd.DataFrame], resolver: SnapshotAliasResolver
) -> pd.DataFrame:
    upcoming = frames["canonical"][frames["canonical"]["status"] == "upcoming"].copy()
    candidates: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for row in upcoming.to_dict("records"):
        start = pd.to_datetime(row.get("start_time_normalized"), format="mixed", utc=True, errors="coerce")
        a, b, pair = pair_resolution(
            resolver,
            row.get("team_a_name"),
            row.get("team_b_name"),
            source_system="bookmaker",
            competition=row.get("league"),
            tournament=None,
            on_date=start,
        )
        resolved.append(
            {
                **row,
                "parsed_start": start,
                "competition_family": competition_family(row.get("league")),
                "team_pair": pair,
                "identity_status": "resolved" if pair is not None else f"{a.status}/{b.status}",
            }
        )
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left["team_pair"] is None or left["team_pair"] != right["team_pair"]:
                continue
            if not left["competition_family"] or left["competition_family"] != right["competition_family"]:
                continue
            if pd.isna(left["parsed_start"]) or pd.isna(right["parsed_start"]):
                continue
            diff_minutes = abs((left["parsed_start"] - right["parsed_start"]).total_seconds()) / 60
            if diff_minutes > UPCOMING_AUTO_WINDOW_MINUTES:
                continue
            candidates.append(
                {
                    "left_canonical_match_id": int(left["id"]),
                    "right_canonical_match_id": int(right["id"]),
                    "left_match": f"{left['team_a_name']} vs {left['team_b_name']}",
                    "right_match": f"{right['team_a_name']} vs {right['team_b_name']}",
                    "competition_family": left["competition_family"],
                    "start_difference_minutes": diff_minutes,
                    "decision": "merge_candidate",
                }
            )
    return pd.DataFrame(candidates)


def _reason_counts(values: Iterable[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        for reason in str(value).split(","):
            if reason and reason != "accepted":
                counter[reason] += 1
    return dict(sorted(counter.items()))


def build_summary(
    frames: dict[str, pd.DataFrame],
    containment: pd.DataFrame,
    identity_summary: ReplaySummary,
    identity_decisions: pd.DataFrame,
    upcoming_duplicates: pd.DataFrame,
) -> dict[str, Any]:
    population = len(containment)
    one_day_n = int(containment["containment_one_day_accept"].sum())
    exact_date_n = int(containment["containment_exact_date_accept"].sum())
    competition_n = int(containment["competition_accept"].sum())
    known_bad = containment[containment["canonical_match_id"].isin(KNOWN_BAD_CANONICAL_IDS)]
    return {
        "snapshot_rows": {key: len(frame) for key, frame in frames.items()},
        "historical_existing_links": population,
        "date_distance_days": {
            str(int(key)): int(value)
            for key, value in containment["date_distance_days"].value_counts().sort_index().items()
            if pd.notna(key)
        },
        "policies": {
            "current_auto_fuzzy": {
                "accepted": population,
                "review": 0,
                "accepted_rate": 1.0,
                "known_bad_accepted": int(known_bad.shape[0]),
            },
            "containment_confidence_and_one_day": {
                "accepted": one_day_n,
                "review": population - one_day_n,
                "accepted_rate": round(one_day_n / population, 6),
                "known_bad_accepted": int(known_bad["containment_one_day_accept"].sum()),
                "review_reasons": _reason_counts(
                    containment.loc[~containment["containment_one_day_accept"], "containment_reasons"]
                ),
            },
            "containment_confidence_and_exact_date": {
                "accepted": exact_date_n,
                "review": population - exact_date_n,
                "accepted_rate": round(exact_date_n / population, 6),
                "known_bad_accepted": int(known_bad["containment_exact_date_accept"].sum()),
            },
            "containment_plus_competition": {
                "accepted": competition_n,
                "review": population - competition_n,
                "accepted_rate": round(competition_n / population, 6),
                "known_bad_accepted": int(known_bad["competition_accept"].sum()),
            },
            "identity_first_retrospective_replay": asdict(identity_summary),
        },
        "identity_replay_outcomes": {
            str(key): int(value)
            for key, value in identity_decisions["outcome"].value_counts().sort_index().items()
        },
        "upcoming_duplicate_candidates": upcoming_duplicates.to_dict("records"),
        "known_bad_canonical_ids": sorted(KNOWN_BAD_CANONICAL_IDS),
        "limitations": [
            "Existing mappings are not ground truth; precision cannot be inferred from retention.",
            "Alias rows available on 2026-09-03 may post-date historical fixtures, so identity replay is retrospective rather than live-temporal evaluation.",
            "Competition-family rules are a simulation taxonomy, not a production schema.",
            "The proposed NiP alias is scoped to LPL and derived from the confirmed same-fixture duplicate.",
        ],
    }


def save_plot(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    policy_names = [
        "Current\nauto-fuzzy",
        "Confidence +\n±1 date",
        "Confidence +\nexact date",
        "Exact date +\ncompetition",
        "Identity-first\nreplay",
    ]
    policies = summary["policies"]
    accepted = [
        policies["current_auto_fuzzy"]["accepted"],
        policies["containment_confidence_and_one_day"]["accepted"],
        policies["containment_confidence_and_exact_date"]["accepted"],
        policies["containment_plus_competition"]["accepted"],
        policies["identity_first_retrospective_replay"]["auto_accept"],
    ]
    total = summary["historical_existing_links"]
    review = [total - value for value in accepted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(policy_names, accepted, label="Automatic acceptance", color="#2b6f77")
    ax.bar(policy_names, review, bottom=accepted, label="Manual review", color="#d9a441")
    for index, value in enumerate(accepted):
        ax.text(index, value / 2, f"{value}\n({value / total:.1%})", ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylim(0, total * 1.08)
    ax.set_ylabel("Existing GOL.GG links (rows)")
    ax.set_title(f"Identity policy replay on {total} collected match links\nSnapshot: 2026-09-03; retention is not accuracy")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(snapshot_dir: Path, output_dir: Path, figure_path: Path | None = None) -> dict[str, Any]:
    frames = load_snapshot(snapshot_dir)
    resolver = SnapshotAliasResolver(frames["aliases"])
    joined = joined_mapping_rows(frames)
    containment = simulate_containment(joined)
    identity_decisions, identity_summary = simulate_identity_replay(frames, resolver)
    upcoming_duplicates = simulate_upcoming_duplicates(frames, resolver)
    summary = build_summary(frames, containment, identity_summary, identity_decisions, upcoming_duplicates)

    output_dir.mkdir(parents=True, exist_ok=True)
    containment.to_csv(output_dir / "historical_mapping_policy_decisions.csv", index=False)
    identity_decisions.to_csv(output_dir / "identity_replay_decisions.csv", index=False)
    upcoming_duplicates.to_csv(output_dir / "upcoming_duplicate_candidates.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(summary, figure_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    summary = run(args.snapshot_dir, args.output_dir, args.figure)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
