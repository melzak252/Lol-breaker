"""Incrementally refresh finished GOL.GG matches using the local scraper.

The important betting-app behaviour is incremental: by default it fetches match
lists only for tournaments whose GOL.GG game count is larger than what we have
locally, then downloads nested game details only for *new match IDs*. Existing
match documents are not refetched unless explicitly requested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from betting_app.core.config import PROJECT_ROOT
from betting_app.scrapers.golgg import GolggScraper

try:
    from tqdm.asyncio import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
CONCURRENCY_LIMIT = 60
TOURNAMENT_CONCURRENCY_LIMIT = 8
MAX_RETRIES = 5


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description="Incrementally refresh GOL.GG finished matches")
    parser.add_argument(
        "--output-path",
        "--matches-path",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to golgg_matches.json updated in-place.",
    )
    parser.add_argument(
        "--embedded-rift-esport-dir",
        type=Path,
        default=None,
        help="Deprecated; ignored. The GOL.GG scraper is now vendored into betting_app.",
    )
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--refresh-matches",
        action="store_true",
        help="Fetch match lists for all tournaments instead of only tournaments with missing games.",
    )
    parser.add_argument(
        "--include-incomplete-existing",
        action="store_true",
        help="Also fetch games for existing matches whose nested games are incomplete.",
    )
    parser.add_argument(
        "--refetch-games",
        action="store_true",
        help="Refetch games for selected matches even if they already look complete.",
    )
    parser.add_argument("--match-id", help="Fetch/refetch only one match ID.")
    parser.add_argument("--dry-run", action="store_true", help="Discover missing matches but do not write JSON.")
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [strip_redundant_player_maps(item) for item in data]
    with path.open("w", encoding="utf-8") as file:
        json.dump(cleaned, file, indent=4, ensure_ascii=False)


def strip_redundant_player_maps(match: dict) -> dict:
    cleaned = dict(match)
    redundant_keys = {"t1_player_ids", "t2_player_ids", "t1_player_names", "t2_player_names", "t1_champions", "t2_champions"}
    for key in redundant_keys:
        cleaned.pop(key, None)
    cleaned_games = []
    for game in cleaned.get("games") or []:
        cleaned_game = dict(game)
        for key in redundant_keys:
            cleaned_game.pop(key, None)
        cleaned_games.append(cleaned_game)
    if "games" in cleaned:
        cleaned["games"] = cleaned_games
    return cleaned


def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    result: dict[str, dict] = {}
    without_key = []
    for item in items:
        value = item.get(key)
        if value is None:
            without_key.append(item)
            continue
        result[str(value)] = item
    return without_key + list(result.values())


def count_nested_games_by_tournament(matches: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        tournament = match.get("tournament_name") or match.get("tournament")
        if tournament:
            counts[tournament] = counts.get(tournament, 0) + len(match.get("games") or [])
    return counts


def count_nested_games_by_match(matches: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        match_id = match.get("match_id")
        if match_id:
            counts[str(match_id)] = len(match.get("games") or [])
    return counts


def existing_match_ids(matches: list[dict]) -> set[str]:
    return {str(match["match_id"]) for match in matches if match.get("match_id")}


def normalize_name(name: str | None) -> str:
    return " ".join((name or "").lower().split())


def infer_best_of_from_match(match: dict) -> int | None:
    try:
        t1_score = int(match.get("t1_score", 0))
        t2_score = int(match.get("t2_score", 0))
    except (TypeError, ValueError):
        return None
    if t1_score == t2_score:
        games_played = t1_score + t2_score
        return games_played if games_played > 0 else None
    wins_needed = max(t1_score, t2_score)
    return (wins_needed * 2) - 1 if wins_needed > 0 else None


def enrich_match_result_flags(match: dict) -> None:
    try:
        t1_score = int(match.get("t1_score", 0))
        t2_score = int(match.get("t2_score", 0))
    except (TypeError, ValueError):
        return
    match["games_played"] = t1_score + t2_score
    match["t1_win"] = t1_score > t2_score
    match["t2_win"] = t2_score > t1_score
    match["draw"] = t1_score == t2_score
    match["best_of"] = infer_best_of_from_match(match)


def enrich_match_with_nested_game_metadata(match: dict) -> None:
    enrich_match_result_flags(match)
    games = match.get("games") or []
    if not games:
        return
    first_game = games[0]
    game_teams = {
        normalize_name(first_game.get("t1_name")): {"team_id": first_game.get("t1_id")},
        normalize_name(first_game.get("t2_name")): {"team_id": first_game.get("t2_id")},
    }
    t1_info = game_teams.get(normalize_name(match.get("sname_t1")))
    t2_info = game_teams.get(normalize_name(match.get("sname_t2")))
    if t1_info:
        match["t1_id"] = t1_info.get("team_id")
    if t2_info:
        match["t2_id"] = t2_info.get("team_id")


def expected_games_for_match(match: dict) -> int:
    try:
        return int(match.get("t1_score", 0)) + int(match.get("t2_score", 0))
    except (ValueError, TypeError):
        return 0


def should_fetch_games(
    match: dict,
    known_existing_ids: set[str],
    games_per_match: dict[str, int],
    *,
    include_incomplete_existing: bool,
    refetch_games: bool,
    match_id: str | None,
) -> bool:
    raw_mid = match.get("match_id")
    if not raw_mid:
        return False
    mid = str(raw_mid)
    if match_id and mid != str(match_id):
        return False
    expected_games = expected_games_for_match(match)
    if expected_games <= 0:
        return False
    have_games = games_per_match.get(mid, len(match.get("games") or []))
    is_new = mid not in known_existing_ids
    is_incomplete_existing = include_incomplete_existing and have_games < expected_games
    return refetch_games or is_new or is_incomplete_existing


async def get_tournament_matches(scraper: GolggScraper, tournaments: list[dict]) -> list[dict]:
    all_matches = []
    semaphore = asyncio.Semaphore(TOURNAMENT_CONCURRENCY_LIMIT)

    async def fetch_tournament(tournament: dict) -> list[dict]:
        tournament_name = tournament.get("trname")
        if not tournament_name:
            return []
        for attempt in range(1, MAX_RETRIES + 1):
            async with semaphore:
                try:
                    return await scraper.get_matches_in_tournament(tournament_name)
                except Exception as exc:
                    print(
                        f"[!][Tournament {tournament_name}] attempt "
                        f"{attempt}/{MAX_RETRIES} failed: {exc}"
                    )
            await asyncio.sleep(min(attempt, 5))
        print(f"[!][Tournament {tournament_name}] giving up after {MAX_RETRIES} attempts")
        return []

    tasks = [fetch_tournament(tournament) for tournament in tournaments]
    iterator = asyncio.as_completed(tasks)
    if tqdm:
        iterator = tqdm(iterator, total=len(tasks), desc="Fetching matches for tournaments")
    for coro in iterator:
        all_matches.extend(await coro)
    return all_matches


async def fetch_match_games(scraper: GolggScraper, match: dict, semaphore: asyncio.Semaphore) -> list[dict]:
    match_id = match.get("match_id")
    if not match_id:
        print("Match ID not found, skipping...")
        return []
    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                return await scraper.get_games_in_match(str(match_id))
            except Exception as exc:
                print(f"[!][Match {match_id}] attempt {attempt} failed: {exc}")
        await asyncio.sleep(attempt)
    print(f"[!][Match {match_id}] giving up after {MAX_RETRIES} attempts")
    return []


async def fetch_match_with_games(scraper: GolggScraper, match: dict, semaphore: asyncio.Semaphore) -> dict:
    match_doc = dict(match)
    games = await fetch_match_games(scraper, match, semaphore)
    for game in games:
        game["patch"] = match_doc.get("patch")
        game["date"] = match_doc.get("date")
        game["tournament_name"] = match_doc.get("tournament_name")
    match_doc["games"] = games
    enrich_match_with_nested_game_metadata(match_doc)
    return match_doc


async def get_matches_with_games(matches: list[dict], scraper: GolggScraper, concurrency: int = CONCURRENCY_LIMIT) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [fetch_match_with_games(scraper, match, semaphore) for match in matches]
    result = []
    iterator = asyncio.as_completed(tasks)
    if tqdm:
        iterator = tqdm(iterator, total=len(tasks), desc="Fetching match games")
    for coro in iterator:
        result.append(await coro)
    return result


async def main() -> None:
    args = parse_args()
    if args.embedded_rift_esport_dir:
        print("Note: --embedded-rift-esport-dir is ignored; using betting_app.scrapers.golgg")
    output_path = args.output_path

    print(f"Loading existing GOL.GG matches from {output_path}...")
    matches = deduplicate_by_key(load_json_list(output_path), "match_id")
    known_existing_ids = existing_match_ids(matches)
    games_per_tournament = count_nested_games_by_tournament(matches)
    games_per_match = count_nested_games_by_match(matches)
    print(f"Local cache: {len(matches)} matches, {sum(games_per_match.values())} nested games")

    async with GolggScraper(max_pages=args.max_pages) as scraper:
        print("Fetching GOL.GG tournament index...")
        all_tournaments = await scraper.get_all_tournaments()
        tournaments_to_fetch = select_tournaments_to_fetch(
            all_tournaments,
            games_per_tournament,
            refresh_matches=args.refresh_matches,
        )
        print(
            f"Tournament lists to scan: {len(tournaments_to_fetch)} / {len(all_tournaments)} "
            "(only tournaments with missing/new games unless --refresh-matches)"
        )

        discovered_matches = await get_tournament_matches(scraper, tournaments_to_fetch) if tournaments_to_fetch else []
        new_matches = [
            match
            for match in deduplicate_by_key(discovered_matches, "match_id")
            if match.get("match_id") and str(match["match_id"]) not in known_existing_ids
        ]
        if args.match_id:
            new_matches = [match for match in new_matches if str(match.get("match_id")) == str(args.match_id)]

        print(f"Discovered new match IDs: {len(new_matches)}")
        if new_matches and not args.dry_run:
            matches.extend(new_matches)
            matches = deduplicate_by_key(matches, "match_id")
            save_json(output_path, matches)
            print(f"Saved match metadata for {len(new_matches)} new matches.")
        elif new_matches:
            print("Dry run: not saving newly discovered match metadata.")

        selected_matches = [
            match
            for match in matches
            if should_fetch_games(
                match,
                known_existing_ids,
                games_per_match,
                include_incomplete_existing=args.include_incomplete_existing,
                refetch_games=args.refetch_games,
                match_id=args.match_id,
            )
        ]
        if args.dry_run:
            selected_new_ids = {str(match["match_id"]) for match in new_matches if match.get("match_id")}
            selected_matches = [match for match in new_matches if str(match.get("match_id")) in selected_new_ids]

        if not selected_matches:
            print("No new match game details to fetch.")
            return

        print(
            f"Fetching nested games for {len(selected_matches)} matches "
            "(new only by default; existing incomplete only with --include-incomplete-existing)."
        )
        if args.dry_run:
            print("Dry run: stopping before downloading/saving nested games.")
            return

        await fetch_and_merge_games(
            scraper,
            output_path,
            matches,
            selected_matches,
            batch_size=max(1, args.batch_size),
            concurrency=args.concurrency,
        )


def select_tournaments_to_fetch(all_tournaments: list[dict], games_per_tournament: dict[str, int], *, refresh_matches: bool) -> list[dict]:
    tournaments_to_fetch = []
    for tournament in all_tournaments:
        name = tournament.get("trname")
        if not name:
            continue
        try:
            expected_games = int(tournament.get("nbgames") or 0)
        except (TypeError, ValueError):
            expected_games = 0
        have_games = games_per_tournament.get(name, 0)
        if refresh_matches or have_games < expected_games:
            tournaments_to_fetch.append(tournament)
    return tournaments_to_fetch


async def fetch_and_merge_games(
    scraper: GolggScraper,
    output_path: Path,
    matches: list[dict],
    selected_matches: list[dict],
    *,
    batch_size: int,
    concurrency: int,
) -> None:
    matches_by_id = {str(match["match_id"]): match for match in matches if match.get("match_id")}
    fetched_total = 0
    for start in range(0, len(selected_matches), batch_size):
        batch = selected_matches[start : start + batch_size]
        print(f"Fetching batch {start // batch_size + 1}: {start + 1}-{start + len(batch)} / {len(selected_matches)}")
        fetched_match_docs = await get_matches_with_games(batch, scraper, concurrency=concurrency)
        fetched_match_docs = [match for match in fetched_match_docs if match.get("match_id") and match.get("games")]
        for match_doc in fetched_match_docs:
            matches_by_id[str(match_doc["match_id"])] = match_doc
        if fetched_match_docs:
            fetched_total += len(fetched_match_docs)
            merged = deduplicate_by_key(list(matches_by_id.values()), "match_id")
            total_games = sum(len(match.get("games") or []) for match in merged)
            print(f"Saving progress: {len(merged)} matches, {total_games} nested games to {output_path}")
            save_json(output_path, merged)
    print(f"Finished fetching games for {fetched_total} matches.")


if __name__ == "__main__":
    asyncio.run(main())
