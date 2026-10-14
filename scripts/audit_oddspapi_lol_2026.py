"""Download an audited OddsPapi LoL cohort and compare it to EXP-039 proxy outputs.

The script deliberately writes only ignored local artifacts.  It never writes to
application tables, never places bets, and keeps the supplied API key in memory.
The model report used by default is a retrospective historical proxy, so its
market comparison is explicitly diagnostic rather than executable performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from betting_app.core.matching import normalize_team_name
from src.analysis.quote_provenance import extract_winner_quote

API_BASE = "https://api.oddspapi.io/v4"
DEFAULT_COHORT = Path("reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv")
DEFAULT_OUTPUT = Path("data/oddspapi_lol_2026_model_audit")

class OddsPapiHTTPError(RuntimeError):
    def __init__(self, path: str, status: int, body: str) -> None:
        self.status = status
        self.retry_after_seconds: float | None = None
        try:
            retry_after = json.loads(body)["error"]["retryAfter"]
            self.retry_after_seconds = float(str(retry_after).split()[0])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        super().__init__(f"OddsPapi {path} returned HTTP {status}: {body[:500]}")


@dataclass(frozen=True)
class CohortMatch:
    canonical_match_id: int
    team_a_name: str
    team_b_name: str
    team_a_norm: str
    team_b_norm: str
    start_at: datetime
    winner_a: int
    model_prob_a: float
    league: str


@dataclass(frozen=True)
class FixtureMatch:
    cohort: CohortMatch
    fixture_id: str
    provider_team_1: str
    provider_team_2: str
    provider_start_at: datetime
    provider_team_1_is_a: bool
    provider_team_1_id: str | None = None
    provider_team_2_id: str | None = None


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def api_get(path: str, params: dict[str, str], *, timeout_seconds: int) -> Any:
    url = f"{API_BASE}{path}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "EnsembleLegends research audit/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 -- fixed HTTPS API base
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise OddsPapiHTTPError(path, error.code, body) from error
    except URLError as error:
        raise RuntimeError(f"OddsPapi {path} request failed: {error.reason}") from error


def load_cohort(path: Path) -> list[CohortMatch]:
    frame = pd.read_csv(path)
    required = {
        "canonical_match_id",
        "team_a_name",
        "team_b_name",
        "winner_side",
        "start_time_normalized",
        "exp039_parity_v2_prob_team_a",
        "league",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cohort is missing required columns: {sorted(missing)}")

    matches: list[CohortMatch] = []
    for row in frame.to_dict("records"):
        start_at = parse_utc(row["start_time_normalized"])
        if start_at.year != 2026:
            continue
        probability = float(row["exp039_parity_v2_prob_team_a"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid EXP-039 probability for canonical match {row['canonical_match_id']}")
        winner_side = str(row["winner_side"])
        if winner_side not in {"team_a", "team_b"}:
            raise ValueError(f"Unknown winner side {winner_side!r} for canonical match {row['canonical_match_id']}")
        team_a_name = str(row["team_a_name"])
        team_b_name = str(row["team_b_name"])
        matches.append(
            CohortMatch(
                canonical_match_id=int(row["canonical_match_id"]),
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                team_a_norm=normalize_team_name(team_a_name),
                team_b_norm=normalize_team_name(team_b_name),
                start_at=start_at,
                winner_a=1 if winner_side == "team_a" else 0,
                model_prob_a=probability,
                league=str(row["league"]),
            )
        )
    return sorted(matches, key=lambda item: (item.start_at, item.canonical_match_id))


def date_windows(start_at: datetime, end_at: datetime) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    cursor = start_at.replace(hour=0, minute=0, second=0, microsecond=0)
    final = end_at.replace(hour=23, minute=59, second=59, microsecond=0)
    while cursor <= final:
        window_end = min(cursor + timedelta(days=6, hours=23, minutes=59, seconds=59), final)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(seconds=1)
    return windows


def fetch_fixtures(
    api_key: str,
    cohort: list[CohortMatch],
    *,
    timeout_seconds: int,
    request_pause_seconds: float,
) -> list[dict[str, Any]]:
    fixtures: dict[str, dict[str, Any]] = {}
    for index, (window_start, window_end) in enumerate(date_windows(cohort[0].start_at, cohort[-1].start_at), start=1):
        if index > 1:
            time.sleep(request_pause_seconds)
        payload = api_get(
            "/fixtures",
            {
                "apiKey": api_key,
                "sportId": "18",
                "from": window_start.isoformat().replace("+00:00", "Z"),
                "to": window_end.isoformat().replace("+00:00", "Z"),
            },
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Expected fixture list for {window_start.date()}, received {type(payload).__name__}")
        for fixture in payload:
            fixture_id = str(fixture.get("fixtureId") or "")
            if fixture_id:
                fixtures[fixture_id] = fixture
        print(f"fixtures window {index}: {len(payload)} rows, {len(fixtures)} unique", file=sys.stderr)
    return sorted(fixtures.values(), key=lambda item: (str(item.get("startTime") or ""), str(item.get("fixtureId") or "")))


def match_fixtures(cohort: list[CohortMatch], fixtures: list[dict[str, Any]]) -> tuple[list[FixtureMatch], list[dict[str, Any]]]:
    by_pair: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        team_1 = normalize_team_name(str(fixture.get("participant1Name") or ""))
        team_2 = normalize_team_name(str(fixture.get("participant2Name") or ""))
        start_raw = fixture.get("startTime")
        if not team_1 or not team_2 or not start_raw:
            continue
        by_pair[frozenset((team_1, team_2))].append(fixture)

    mapped: list[FixtureMatch] = []
    audit: list[dict[str, Any]] = []
    maximum_start_delta = timedelta(hours=12)
    for match in cohort:
        candidates = []
        for fixture in by_pair.get(frozenset((match.team_a_norm, match.team_b_norm)), []):
            provider_start_at = parse_utc(str(fixture["startTime"]))
            delta = abs(provider_start_at - match.start_at)
            if delta <= maximum_start_delta:
                candidates.append((delta, fixture, provider_start_at))
        candidates.sort(key=lambda item: (item[0], str(item[1].get("fixtureId") or "")))
        if len(candidates) != 1:
            audit.append(
                {
                    "canonical_match_id": match.canonical_match_id,
                    "team_a_name": match.team_a_name,
                    "team_b_name": match.team_b_name,
                    "canonical_start_at": match.start_at.isoformat(),
                    "league": match.league,
                    "status": "unmatched" if not candidates else "ambiguous",
                    "candidate_count": len(candidates),
                }
            )
            continue
        _, fixture, provider_start_at = candidates[0]
        provider_team_1 = str(fixture["participant1Name"])
        provider_team_2 = str(fixture["participant2Name"])
        provider_team_1_is_a = normalize_team_name(provider_team_1) == match.team_a_norm
        provider_team_1_id = str(fixture["participant1Id"]) if fixture.get("participant1Id") is not None else None
        provider_team_2_id = str(fixture["participant2Id"]) if fixture.get("participant2Id") is not None else None
        mapped.append(
            FixtureMatch(
                cohort=match,
                fixture_id=str(fixture["fixtureId"]),
                provider_team_1=provider_team_1,
                provider_team_2=provider_team_2,
                provider_start_at=provider_start_at,
                provider_team_1_is_a=provider_team_1_is_a,
                provider_team_1_id=provider_team_1_id,
                provider_team_2_id=provider_team_2_id,
            )
        )
        audit.append(
            {
                "canonical_match_id": match.canonical_match_id,
                "team_a_name": match.team_a_name,
                "team_b_name": match.team_b_name,
                "canonical_start_at": match.start_at.isoformat(),
                "league": match.league,
                "status": "matched_exact_normalized_pair",
                "candidate_count": 1,
                "fixture_id": str(fixture["fixtureId"]),
                "provider_team_1": provider_team_1,
                "provider_team_2": provider_team_2,
                "provider_start_at": provider_start_at.isoformat(),
                "provider_team_1_is_a": provider_team_1_is_a,
                "provider_team_1_id": provider_team_1_id,
                "provider_team_2_id": provider_team_2_id,
            }
        )
    return mapped, audit




def flatten_history(match: FixtureMatch, history: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fixture = {
        "fixtureId": match.fixture_id,
        "startTime": match.provider_start_at,
        "participant1Name": match.provider_team_1,
        "participant1Id": match.provider_team_1_id,
        "participant2Id": match.provider_team_2_id,
        "participant2Name": match.provider_team_2,
    }
    bookmakers = history.get("bookmakers") or {}
    if not isinstance(bookmakers, dict):
        return all_rows, selected_rows

    for bookmaker_slug in bookmakers:
        quote = extract_winner_quote(
            history,
            bookmaker_slug,
            cutoff_at=match.provider_start_at,
            fixture=fixture,
        )
        if quote is None:
            continue

        for outcome_id, state_key, participant_index in (
            (quote["outcome_id_1"], "state_1", 1),
            (quote["outcome_id_2"], "state_2", 2),
        ):
            side_state = quote[state_key]
            for observation in side_state["observations"]:
                source_state = observation["raw_state"]
                observed_at = observation["observed_at"]
                all_rows.append(
                    {
                        "canonical_match_id": match.cohort.canonical_match_id,
                        "fixture_id": match.fixture_id,
                        "bookmaker": bookmaker_slug,
                        "provider_outcome_id": outcome_id,
                        "provider_team_1": match.provider_team_1,
                        "provider_team_2": match.provider_team_2,
                        "provider_team_1_is_a": match.provider_team_1_is_a,
                        "participant_id": quote[f"participant_{participant_index}_id"],
                        "participant_name": quote[f"participant_{participant_index}_name"],
                        "participant_binding_status": quote["participant_binding_status"],
                        "source_created_at": observation["created_at"],
                        "observed_at": observed_at,
                        "match_start_at": match.provider_start_at.isoformat(),
                        "price": source_state.get("price"),
                        "limit": source_state.get("limit"),
                        "active": source_state.get("active"),
                        "state_status": observation["status"],
                        "latest_outcome_status": side_state["status"],
                        "is_pre_match": observation["status"] in {"eligible_for_latest", "latest_source_state"},
                        "source_state_json": json.dumps(source_state, ensure_ascii=False, sort_keys=True),
                    }
                )

        if quote["odds_1"] is None or quote["odds_2"] is None or quote["quote_at"] is None:
            continue
        odds_1 = quote["odds_1"]
        odds_2 = quote["odds_2"]
        implied_1 = 1.0 / odds_1
        implied_2 = 1.0 / odds_2
        margin = implied_1 + implied_2 - 1.0
        provider_prob_1 = implied_1 / (implied_1 + implied_2)
        market_prob_a = provider_prob_1 if match.provider_team_1_is_a else 1.0 - provider_prob_1
        selected_rows.append(
            {
                "canonical_match_id": match.cohort.canonical_match_id,
                "fixture_id": match.fixture_id,
                "source_fixture_id": quote["source_fixture_id"],
                "bookmaker": bookmaker_slug,
                "team_a_name": match.cohort.team_a_name,
                "team_b_name": match.cohort.team_b_name,
                "league": match.cohort.league,
                "match_start_at": match.provider_start_at.isoformat(),
                "quote_at": quote["quote_at"],
                "quote_at_1": quote["quote_at_1"],
                "quote_at_2": quote["quote_at_2"],
                "outcome_id_1": quote["outcome_id_1"],
                "outcome_id_2": quote["outcome_id_2"],
                "odds_provider_team_1": odds_1,
                "odds_provider_team_2": odds_2,
                "odds_a": odds_1 if match.provider_team_1_is_a else odds_2,
                "odds_b": odds_2 if match.provider_team_1_is_a else odds_1,
                "market_prob_a_novig": market_prob_a,
                "market_margin": margin,
                "participant_1_id": quote["participant_1_id"],
                "participant_1_name": quote["participant_1_name"],
                "participant_2_id": quote["participant_2_id"],
                "participant_2_name": quote["participant_2_name"],
                "participant_binding_status": quote["participant_binding_status"],
                "execution_eligibility": quote["execution_eligibility"],
                "provenance_limitations": json.dumps(quote["provenance_limitations"], ensure_ascii=False),
                "source_state_1_json": json.dumps(quote["state_1"], ensure_ascii=False, sort_keys=True),
                "source_state_2_json": json.dumps(quote["state_2"], ensure_ascii=False, sort_keys=True),
                "orientation_evidence_status": "provider_fixture_order_only_unverified_bookmaker_binding",
                "model_prob_a_proxy": match.cohort.model_prob_a,
                "winner_a": match.cohort.winner_a,
                "comparison_classification": "retrospective_market_diagnostic_only",
            }
        )
    return all_rows, selected_rows


def log_loss(y_true: list[int], probabilities: list[float]) -> float:
    epsilon = 1e-15
    clipped = [min(max(probability, epsilon), 1.0 - epsilon) for probability in probabilities]
    return -sum(y * math.log(probability) + (1 - y) * math.log(1 - probability) for y, probability in zip(y_true, clipped, strict=True)) / len(y_true)


def brier(y_true: list[int], probabilities: list[float]) -> float:
    return sum((probability - y) ** 2 for y, probability in zip(y_true, probabilities, strict=True)) / len(y_true)


def accuracy(y_true: list[int], probabilities: list[float]) -> float:
    return sum((probability >= 0.5) == bool(y) for y, probability in zip(y_true, probabilities, strict=True)) / len(y_true)


def weekly_block_bootstrap(rows: list[dict[str, Any]], *, resamples: int = 10_000) -> dict[str, Any]:
    blocks: dict[tuple[int, int], dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "model_log_loss": 0.0, "market_log_loss": 0.0, "model_brier": 0.0, "market_brier": 0.0}
    )
    epsilon = 1e-15
    for row in rows:
        year, week, _ = parse_utc(str(row["match_start_at"])).isocalendar()
        block = blocks[(year, week)]
        outcome = int(row["winner_a"])
        model_probability = min(max(float(row["model_prob_a_proxy"]), epsilon), 1.0 - epsilon)
        market_probability = min(max(float(row["market_prob_a_novig"]), epsilon), 1.0 - epsilon)
        block["count"] += 1
        block["model_log_loss"] += -(outcome * math.log(model_probability) + (1 - outcome) * math.log(1 - model_probability))
        block["market_log_loss"] += -(outcome * math.log(market_probability) + (1 - outcome) * math.log(1 - market_probability))
        block["model_brier"] += (model_probability - outcome) ** 2
        block["market_brier"] += (market_probability - outcome) ** 2

    totals = list(blocks.values())
    total_count = sum(int(block["count"]) for block in totals)
    point_estimates = {
        metric: sum(float(block[f"model_{metric}"]) - float(block[f"market_{metric}"]) for block in totals) / total_count
        for metric in ("log_loss", "brier")
    }
    rng = random.Random(20_260_904)
    sampled = {metric: [] for metric in point_estimates}
    for _ in range(resamples):
        sample = [totals[rng.randrange(len(totals))] for _ in totals]
        sample_count = sum(int(block["count"]) for block in sample)
        for metric in sampled:
            sampled[metric].append(
                sum(float(block[f"model_{metric}"]) - float(block[f"market_{metric}"]) for block in sample) / sample_count
            )
    def quantile(values: list[float], fraction: float) -> float:
        sorted_values = sorted(values)
        return sorted_values[math.floor((len(sorted_values) - 1) * fraction)]

    return {
        "resampling_unit": "ISO week based on match_start_at",
        "resamples": resamples,
        "positive_delta_model_minus_market_favors": "bookmaker",
        "weeks": len(totals),
        "log_loss_model_minus_market": point_estimates["log_loss"],
        "log_loss_model_minus_market_ci95": [quantile(sampled["log_loss"], 0.025), quantile(sampled["log_loss"], 0.975)],
        "brier_model_minus_market": point_estimates["brier"],
        "brier_model_minus_market_ci95": [quantile(sampled["brier"], 0.025), quantile(sampled["brier"], 0.975)],
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "observations": 0,
            "model_log_loss": None,
            "market_log_loss": None,
            "model_brier": None,
            "market_brier": None,
            "model_accuracy": None,
            "market_accuracy": None,
            "average_market_margin": None,
            "weekly_block_bootstrap": None,
        }
    y_true = [int(row["winner_a"]) for row in rows]
    model_probability = [float(row["model_prob_a_proxy"]) for row in rows]
    market_probability = [float(row["market_prob_a_novig"]) for row in rows]
    return {
        "observations": len(rows),
        "model_log_loss": log_loss(y_true, model_probability),
        "market_log_loss": log_loss(y_true, market_probability),
        "model_brier": brier(y_true, model_probability),
        "market_brier": brier(y_true, market_probability),
        "model_accuracy": accuracy(y_true, model_probability),
        "market_accuracy": accuracy(y_true, market_probability),
        "average_market_margin": sum(float(row["market_margin"]) for row in rows) / len(rows),
        "weekly_block_bootstrap": weekly_block_bootstrap(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 2026 OddsPapi LoL history against EXP-039 proxy outputs.")
    parser.add_argument("--api-key-env", default="ODDSPAPI_API_KEY")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bookmakers", default="pinnacle,kalshi")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--fixture-pause-seconds", type=float, default=2.0)
    parser.add_argument("--history-pause-seconds", type=float, default=5.0)
    parser.add_argument("--max-history", type=int, default=None, help="Limit matched fixture histories for a smoke run.")
    parser.add_argument("--resume", action="store_true", help="Reuse raw history JSON already written under output-dir/raw_history.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    bookmakers = [item.strip() for item in args.bookmakers.split(",") if item.strip()]
    if not bookmakers or len(bookmakers) > 3:
        raise SystemExit("--bookmakers must contain between one and three OddsPapi slugs")
    if not args.cohort.is_file():
        raise SystemExit(f"Cohort file does not exist: {args.cohort}")

    output_dir = args.output_dir
    raw_history_dir = output_dir / "raw_history"
    raw_history_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(args.cohort)
    if not cohort:
        raise SystemExit("No 2026 rows in cohort")

    fixtures_path = output_dir / "fixtures.json"
    if args.resume and fixtures_path.is_file():
        fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    else:
        fixtures = fetch_fixtures(
            api_key,
            cohort,
            timeout_seconds=args.timeout_seconds,
            request_pause_seconds=args.fixture_pause_seconds,
        )
        fixtures_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    mapped, mapping_audit = match_fixtures(cohort, fixtures)
    write_csv(output_dir / "mapping_audit.csv", mapping_audit)

    all_history_rows: list[dict[str, Any]] = []
    selected_quote_rows: list[dict[str, Any]] = []
    history_errors: list[dict[str, Any]] = []
    matched_for_history = mapped[: args.max_history] if args.max_history is not None else mapped
    last_history_request_at: float | None = None
    for index, match in enumerate(matched_for_history, start=1):
        raw_path = raw_history_dir / f"{match.fixture_id}.json"
        try:
            if args.resume and raw_path.is_file():
                history = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                while True:
                    if last_history_request_at is not None:
                        elapsed = time.monotonic() - last_history_request_at
                        time.sleep(max(0.0, args.history_pause_seconds - elapsed))
                    last_history_request_at = time.monotonic()
                    try:
                        history = api_get(
                            "/historical-odds",
                            {"apiKey": api_key, "fixtureId": match.fixture_id, "bookmakers": ",".join(bookmakers)},
                            timeout_seconds=args.timeout_seconds,
                        )
                        break
                    except OddsPapiHTTPError as error:
                        if error.status != 429:
                            raise
                        retry_after = error.retry_after_seconds or args.history_pause_seconds
                        print(
                            f"history {index}/{len(matched_for_history)} rate limited; retrying in {retry_after:.2f}s",
                            file=sys.stderr,
                        )
                        time.sleep(retry_after + 0.1)
                        last_history_request_at = None
                raw_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
            history_rows, quote_rows = flatten_history(match, history)
            all_history_rows.extend(history_rows)
            selected_quote_rows.extend(quote_rows)
            print(f"history {index}/{len(matched_for_history)}: {match.fixture_id}, {len(quote_rows)} eligible bookmaker quotes", file=sys.stderr)
        except RuntimeError as error:
            history_errors.append({"canonical_match_id": match.cohort.canonical_match_id, "fixture_id": match.fixture_id, "error": str(error)})
            print(f"history {index}/{len(matched_for_history)} failed: {error}", file=sys.stderr)

    write_csv(output_dir / "winner_history.csv", all_history_rows)
    write_csv(output_dir / "selected_pre_match_quotes.csv", selected_quote_rows)
    write_csv(output_dir / "history_errors.csv", history_errors)
    per_bookmaker = {
        bookmaker: metrics([row for row in selected_quote_rows if row["bookmaker"] == bookmaker])
        for bookmaker in bookmakers
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "OddsPapi /v4",
        "sport": "League of Legends",
        "bookmakers_requested": bookmakers,
        "cohort_source": str(args.cohort),
        "comparison_classification": "retrospective_market_diagnostic_only",
        "comparison_limitation": "EXP-039 values are reconstructed historical proxy probabilities without persisted decision-time predicted_at; do not use this report for ROI, promotion, or executable betting claims.",
        "temporal_quote_policy": "latest active two-sided match-winner quote strictly before OddsPapi fixture start time",
        "cohort_rows": len(cohort),
        "cohort_start_at": cohort[0].start_at.isoformat(),
        "cohort_end_at": cohort[-1].start_at.isoformat(),
        "fixtures_downloaded": len(fixtures),
        "matches_exactly_mapped": len(mapped),
        "matches_unmapped_or_ambiguous": len(cohort) - len(mapped),
        "history_requests_attempted": len(matched_for_history),
        "history_errors": len(history_errors),
        "winner_history_rows": len(all_history_rows),
        "selected_pre_match_quotes": len(selected_quote_rows),
        "metrics_by_bookmaker": per_bookmaker,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
