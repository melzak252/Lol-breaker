#!/usr/bin/env python3
"""Audit tournament rulebook certification and verify probability conservation.

Features:
1. Point-in-time publication verification:
   - Compare rulebook/bracket draw release dates against tournament start dates
     for major editions (MSI 2024, Worlds 2024, LCK 2026 Spring/Summer playoffs).
2. Joint probability conservation verification:
   - Run Monte Carlo simulation passes across core tournament formats:
     Swiss stages, GSL groups, double-elimination brackets, and hybrid structures.
   - Rigorously assert that placement probabilities sum to 1.0 for each team
     (sum_{k} P(place=k) = 1.0), placement band mass equals band capacity,
     and champion probabilities sum to 1.0 with zero probability mass leakage.
3. Generation of structured audit artifact:
   - Listing certified tournament editions vs phases remaining in diagnostic mode.
   - Recording SHA256 digests of verified rulebooks and formats.
   - Output path: data/08_reporting/tournament_certification_audit.json.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tournament_catalog import (
    DEFAULT_BUNDLE,
    DEFAULT_CATALOG,
    audit_catalog,
    instantiate_profile,
)
from src.models.tournament_formats import (
    simulate_tournament,
    validate_tournament_spec,
)

REPORT_PATH = ROOT / "data/08_reporting/tournament_certification_audit.json"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_point_in_time_publication(bundle_path: Path) -> dict[str, Any]:
    """Audit point-in-time publication dates vs tournament start dates."""
    events_file = bundle_path / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    target_editions = [
        {
            "edition": "MSI 2024",
            "title": "2024 Mid-Season Invitational",
            "stage": "play_in",
            "official_rulebook_title": "MSI 2024 Rulebook",
            "rulebook_url": "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/b/b0/MSI_2024_Rulebook.pdf/revision/latest?cb=20240422171219",
            "rulebook_published_at": "2024-04-22T17:12:19Z",
        },
        {
            "edition": "MSI 2024",
            "title": "2024 Mid-Season Invitational",
            "stage": "playoffs",
            "official_rulebook_title": "MSI 2024 Rulebook",
            "rulebook_url": "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/b/b0/MSI_2024_Rulebook.pdf/revision/latest?cb=20240422171219",
            "rulebook_published_at": "2024-04-22T17:12:19Z",
        },
        {
            "edition": "Worlds 2024",
            "title": "2024 Season World Championship/Play-In",
            "stage": "play_in",
            "official_rulebook_title": "Worlds 2024 Rulebook",
            "rulebook_url": "https://cdn.sanity.io/files/dsfx7636/news/f69130a70849462c75fbd834f740f170963b173e.pdf",
            "rulebook_published_at": "2024-09-24T15:31:02Z",
        },
        {
            "edition": "Worlds 2024",
            "title": "2024 Season World Championship/Main Event",
            "stage": "swiss",
            "official_rulebook_title": "Worlds 2024 Rulebook & Swiss Primer",
            "rulebook_url": "https://lol.fandom.com/api.php?action=query&format=json&formatversion=2&prop=revisions&rvprop=ids%7Ctimestamp%7Ccontent&rvslots=main&titles=2024%20Season%20World%20Championship%2FMain%20Event&rvstart=2024-09-24T23%3A59%3A59Z&rvdir=older&rvlimit=1",
            "rulebook_published_at": "2024-09-24T15:31:02Z",
        },
        {
            "edition": "Worlds 2024",
            "title": "2024 Season World Championship/Main Event",
            "stage": "playoffs",
            "official_rulebook_title": "Worlds 2024 Knockout Draw Rules",
            "rulebook_url": "https://lolesports.com/news/worlds-2024-primer",
            "rulebook_published_at": "2024-09-24T15:31:02Z",
        },
        {
            "edition": "LCK 2026 Spring (Rounds 1-2)",
            "title": "LCK/2026 Season/Rounds 1-2",
            "stage": "regular_season",
            "official_rulebook_title": "LCK 2026 Official Rulebook (Rounds 1-2)",
            "rulebook_url": "https://lol.fandom.com/wiki/Archive:Official_Rulebooks/Riot/LCK/2026",
            "rulebook_published_at": "2026-04-24T11:57:01Z",
        },
        {
            "edition": "LCK 2026 Road to MSI (Spring Playoffs)",
            "title": "LCK/2026 Season/Road to MSI",
            "stage": "playoffs",
            "official_rulebook_title": "LCK 2026 Road to MSI Tournament Rules",
            "rulebook_url": "https://lol.fandom.com/wiki/Archive:Official_Rulebooks/Riot/LCK/2026",
            "rulebook_published_at": "2026-04-24T11:57:01Z",
        },
        {
            "edition": "LCK 2026 Summer (Rounds 3-4)",
            "title": "LCK/2026 Season/Rounds 3-4",
            "stage": "regular_season",
            "official_rulebook_title": "LCK 2026 Official Rulebook(2) (Rounds 3-4)",
            "rulebook_url": "https://lol.fandom.com/wiki/Archive:Official_Rulebooks/Riot/LCK/2026(2)",
            "rulebook_published_at": "2026-07-06T12:57:46Z",
        },
        {
            "edition": "LCK 2026 Season Play-In",
            "title": "LCK/2026 Season/Season Play-In",
            "stage": "play_in",
            "official_rulebook_title": "LCK 2026 Season Play-In Format & Draw Rules",
            "rulebook_url": "https://lol.fandom.com/wiki/Archive:Official_Rulebooks/Riot/LCK/2026(2)",
            "rulebook_published_at": "2026-07-06T12:57:46Z",
        },
        {
            "edition": "LCK 2026 Season Playoffs (Summer Playoffs)",
            "title": "LCK/2026 Season/Season Playoffs",
            "stage": "playoffs",
            "official_rulebook_title": "LCK 2026 Season Playoffs Format & Bracket Rules",
            "rulebook_url": "https://lol.fandom.com/wiki/Archive:Official_Rulebooks/Riot/LCK/2026(2)",
            "rulebook_published_at": "2026-07-06T12:57:46Z",
        },
    ]

    audited_editions = []
    for item in target_editions:
        matching_events = [
            e
            for e in events
            if e.get("title") == item["title"] and e.get("stage") == item["stage"]
        ]
        if not matching_events:
            continue
        ev = matching_events[0]
        start_date_str = ev.get("start_date")
        source = ev.get("source", {})
        rev_ts_str = source.get("revision_timestamp")

        # Parse start date (UTC midnight) if available
        start_dt = (
            datetime.fromisoformat(f"{start_date_str}T00:00:00+00:00")
            if start_date_str
            else None
        )
        rulebook_pub_dt = (
            datetime.fromisoformat(item["rulebook_published_at"].replace("Z", "+00:00"))
            if item.get("rulebook_published_at")
            else None
        )

        # Verification of rulebook publication prior to start date
        pre_start_rulebook = False
        lead_time_days = None
        if start_dt and rulebook_pub_dt:
            lead_time_seconds = (start_dt - rulebook_pub_dt).total_seconds()
            lead_time_days = round(lead_time_seconds / 86400.0, 2)
            pre_start_rulebook = lead_time_seconds > 0

        # Note about wiki revision timestamp: wiki revision represents current crawl/archive capture time
        audited_editions.append(
            {
                "edition": item["edition"],
                "phase_id": ev.get("phase_id"),
                "title": ev.get("title"),
                "stage": ev.get("stage"),
                "tournament_start_date": start_date_str,
                "rulebook_title": item["official_rulebook_title"],
                "rulebook_url": item["rulebook_url"],
                "rulebook_publication_timestamp": item["rulebook_published_at"],
                "rulebook_prior_to_start": pre_start_rulebook,
                "publication_lead_time_days": lead_time_days,
                "archive_source_revision_timestamp": rev_ts_str,
                "archive_source_url": source.get("url"),
                "pit_certified": pre_start_rulebook,
            }
        )

    return {
        "audited_major_editions": audited_editions,
        "certified_editions_count": sum(
            1 for row in audited_editions if row["pit_certified"]
        ),
        "total_editions_audited": len(audited_editions),
    }


def audit_probability_conservation(simulations: int = 1000) -> dict[str, Any]:
    """Simulate key tournament formats and rigorously assert probability conservation."""
    test_cases = [
        {
            "format_family": "Swiss Stage + Knockout",
            "profile_id": "worlds-2023-swiss-knockout",
            "team_count": 16,
            "description": "Worlds 16-team Swiss Stage advancing 8 teams into single-elimination playoffs",
            "placement_bands": [
                {
                    "label": "1st",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 1}
                    ],
                },
                {
                    "label": "2nd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 2}
                    ],
                },
                {
                    "label": "3-4th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 3},
                        {"stage": "playoffs", "group": "all", "rank": 4},
                    ],
                },
                {
                    "label": "5-8th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": i}
                        for i in range(5, 9)
                    ],
                },
                {
                    "label": "9-16th",
                    "placements": [
                        {"stage": "swiss", "group": "all", "rank": i}
                        for i in range(9, 17)
                    ],
                },
            ],
        },
        {
            "format_family": "GSL Dual Tournament Groups",
            "profile_id": "msi-2024-play-in",
            "team_count": 8,
            "description": "MSI 8-team dual GSL groups (two 4-team double-elimination groups)",
            "placement_bands": [
                {
                    "label": "Group-1st",
                    "placements": [
                        {"stage": "group0", "group": "all", "rank": 1},
                        {"stage": "group1", "group": "all", "rank": 1},
                    ],
                },
                {
                    "label": "Group-2nd",
                    "placements": [
                        {"stage": "group0", "group": "all", "rank": 2},
                        {"stage": "group1", "group": "all", "rank": 2},
                    ],
                },
                {
                    "label": "Group-3rd",
                    "placements": [
                        {"stage": "group0", "group": "all", "rank": 3},
                        {"stage": "group1", "group": "all", "rank": 3},
                    ],
                },
                {
                    "label": "Group-4th",
                    "placements": [
                        {"stage": "group0", "group": "all", "rank": 4},
                        {"stage": "group1", "group": "all", "rank": 4},
                    ],
                },
            ],
        },
        {
            "format_family": "GSL Groups + Knockout",
            "profile_id": "emea-16-gsl-knockout",
            "team_count": 16,
            "description": "16-team tournament with 4 GSL groups advancing 8 teams to single-elimination playoffs",
            "placement_bands": [
                {
                    "label": "1st",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 1}
                    ],
                },
                {
                    "label": "2nd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 2}
                    ],
                },
                {
                    "label": "3-4th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 3},
                        {"stage": "playoffs", "group": "all", "rank": 4},
                    ],
                },
                {
                    "label": "5-8th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": i}
                        for i in range(5, 9)
                    ],
                },
                {
                    "label": "9-16th",
                    "placements": [
                        {"stage": g, "group": "all", "rank": r}
                        for g in ("g0", "g1", "g2", "g3")
                        for r in (3, 4)
                    ],
                },
            ],
        },
        {
            "format_family": "Double-Elimination Bracket (8-team)",
            "profile_id": "msi-2023-2026-double-elimination",
            "team_count": 8,
            "description": "MSI Bracket Stage 8-team full double-elimination with single grand final",
            "placement_bands": [
                {
                    "label": "1st",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 1}
                    ],
                },
                {
                    "label": "2nd",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 2}
                    ],
                },
                {
                    "label": "3rd",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 3}
                    ],
                },
                {
                    "label": "4th",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 4}
                    ],
                },
                {
                    "label": "5-6th",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 5},
                        {"stage": "bracket", "group": "all", "rank": 6},
                    ],
                },
                {
                    "label": "7-8th",
                    "placements": [
                        {"stage": "bracket", "group": "all", "rank": 7},
                        {"stage": "bracket", "group": "all", "rank": 8},
                    ],
                },
            ],
        },
        {
            "format_family": "LCK Partial Double Elimination (6-team)",
            "profile_id": "lck-2023-2024-partial-double-elimination",
            "team_count": 6,
            "description": "LCK 2023-2024 6-team playoffs with single-elimination Round 1 into double-elimination bracket",
            "placement_bands": [
                {
                    "label": "1st",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 1}
                    ],
                },
                {
                    "label": "2nd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 2}
                    ],
                },
                {
                    "label": "3rd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 3}
                    ],
                },
                {
                    "label": "4th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 4}
                    ],
                },
                {
                    "label": "5-6th",
                    "placements": [
                        {"stage": "opening_a", "group": "all", "rank": 2},
                        {"stage": "opening_b", "group": "all", "rank": 2},
                    ],
                },
            ],
        },
        {
            "format_family": "LCK Six-Team Full Double Elimination",
            "profile_id": "six-team-full-double-elimination",
            "team_count": 6,
            "description": "LCK 2025/2026 Cup & Season Playoffs 6-team full double-elimination bracket",
            "placement_bands": [
                {
                    "label": "1st",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 1}
                    ],
                },
                {
                    "label": "2nd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 2}
                    ],
                },
                {
                    "label": "3rd",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 3}
                    ],
                },
                {
                    "label": "4th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 4}
                    ],
                },
                {
                    "label": "5-6th",
                    "placements": [
                        {"stage": "playoffs", "group": "all", "rank": 5},
                        {"stage": "playoffs", "group": "all", "rank": 6},
                    ],
                },
            ],
        },
    ]

    results = []
    max_global_leakage = 0.0

    for tc in test_cases:
        teams = [f"TEAM_{i+1:02d}" for i in range(tc["team_count"])]
        spec = instantiate_profile(tc["profile_id"], teams)
        if "placement_bands" in tc:
            spec["placement_bands"] = tc["placement_bands"]
        validate_tournament_spec(spec)

        # Construct pairwise win probabilities
        probabilities = {}
        for t1, t2 in combinations(teams, 2):
            for bo in (1, 3, 5):
                probabilities[(t1, t2, bo)] = 0.55 if t1 < t2 else 0.45

        sim_output = simulate_tournament(
            spec, probabilities, simulations=simulations, seed=81
        )

        # 1. Champion conservation
        champ_prob = sim_output.get("champion_prob")
        champ_sum = None
        champ_leakage = 0.0
        if champ_prob is not None:
            champ_sum = float(sum(champ_prob.values()))
            champ_leakage = abs(1.0 - champ_sum)
            if champ_leakage > max_global_leakage:
                max_global_leakage = champ_leakage
            if not np.isclose(champ_sum, 1.0, atol=1e-8):
                raise ValueError(
                    f"Champion probability leakage in {tc['profile_id']}: sum={champ_sum}"
                )

        # 2. Team placement probability conservation: sum_{place} P(place) = 1.0
        pl_prob = sim_output.get("placement_prob")
        team_placement_sums = {}
        team_placement_leakages = {}
        if pl_prob is not None:
            for t in teams:
                t_sum = float(sum(pl_prob[t].values()))
                leak = abs(1.0 - t_sum)
                team_placement_sums[t] = t_sum
                team_placement_leakages[t] = leak
                if leak > max_global_leakage:
                    max_global_leakage = leak
                if not np.isclose(t_sum, 1.0, atol=1e-8):
                    raise ValueError(
                        f"Team {t} placement probability leakage in {tc['profile_id']}: sum={t_sum}"
                    )

        # 3. Placement band mass conservation: sum_{teams} P(team in band) = band_size
        band_conservation = {}
        if pl_prob is not None and "placement_bands" in spec:
            for band in spec["placement_bands"]:
                lbl = band["label"]
                expected_mass = float(len(band["placements"]))
                actual_mass = float(sum(pl_prob[t][lbl] for t in teams))
                band_leak = abs(expected_mass - actual_mass)
                band_conservation[lbl] = {
                    "expected_mass": expected_mass,
                    "actual_mass": actual_mass,
                    "mass_leakage": band_leak,
                }
                if band_leak > max_global_leakage:
                    max_global_leakage = band_leak
                if not np.isclose(actual_mass, expected_mass, atol=1e-8):
                    raise ValueError(
                        f"Band {lbl} mass leakage in {tc['profile_id']}: actual={actual_mass}, exp={expected_mass}"
                    )

        results.append(
            {
                "format_family": tc["format_family"],
                "profile_id": tc["profile_id"],
                "team_count": tc["team_count"],
                "description": tc["description"],
                "simulations": simulations,
                "has_champion": champ_prob is not None,
                "champion_prob_sum": champ_sum,
                "champion_prob_leakage": champ_leakage,
                "max_team_placement_leakage": max(
                    team_placement_leakages.values(), default=0.0
                ),
                "band_conservation": band_conservation,
                "probability_conserved": True,
            }
        )

    return {
        "formats_verified": results,
        "max_probability_mass_leakage": max_global_leakage,
        "zero_probability_mass_leakage_verified": max_global_leakage < 1e-7,
    }


def audit_catalog_diagnostics() -> dict[str, Any]:
    """Audit catalog phases and distinguish certified vs diagnostic mode phases."""
    catalog_audit = audit_catalog(DEFAULT_BUNDLE)
    phases = catalog_audit["phases"]

    certified_phases = []
    diagnostic_phases = []

    for phase in phases:
        summary_row = {
            "phase_id": phase["phase_id"],
            "family": phase["family"],
            "year": phase["year"],
            "title": phase["title"],
            "stage": phase["stage"],
            "mechanic": phase["mechanic"],
            "configured": phase["configured"],
            "profile_ids": phase["profile_ids"],
            "status": phase["status"],
            "historically_ready": phase["historically_ready"],
        }
        if phase["configured"]:
            certified_phases.append(summary_row)
        else:
            summary_row["diagnostic_reasons"] = phase["readiness_gaps"]
            diagnostic_phases.append(summary_row)

    return {
        "total_phases": len(phases),
        "certified_format_phases_count": len(certified_phases),
        "diagnostic_mode_phases_count": len(diagnostic_phases),
        "certified_phases_summary": {
            "total": len(certified_phases),
            "families": catalog_audit["families"],
        },
        "diagnostic_mode_phases": diagnostic_phases,
    }


def audit_digests() -> dict[str, str]:
    """Compute SHA256 digests of all primary rulebook, format, and catalog files."""
    files_to_hash = [
        ("conf_base_tournament_formats_json", DEFAULT_CATALOG),
        ("artifact_verification_json", DEFAULT_BUNDLE / "verification.json"),
        ("artifact_source_manifest_json", DEFAULT_BUNDLE / "source_manifest.json"),
        ("artifact_catalog_json", DEFAULT_BUNDLE / "catalog.json"),
        ("artifact_events_jsonl", DEFAULT_BUNDLE / "events.jsonl"),
        ("artifact_bracket_sources_jsonl", DEFAULT_BUNDLE / "bracket_sources.jsonl"),
        ("artifact_standings_jsonl", DEFAULT_BUNDLE / "standings.jsonl"),
        ("research_msi_json", DEFAULT_BUNDLE / "research/msi.json"),
        ("research_worlds_json", DEFAULT_BUNDLE / "research/worlds.json"),
        ("research_lck_json", DEFAULT_BUNDLE / "research/lck.json"),
        ("research_lpl_json", DEFAULT_BUNDLE / "research/lpl.json"),
        ("research_lec_json", DEFAULT_BUNDLE / "research/lec.json"),
        ("research_lcs_json", DEFAULT_BUNDLE / "research/lcs.json"),
        ("research_eu_masters_json", DEFAULT_BUNDLE / "research/eu_masters.json"),
    ]

    digests = {}
    for key, path in files_to_hash:
        if path.exists():
            digests[key] = compute_sha256(path)
        else:
            digests[key] = f"FILE_NOT_FOUND: {path}"
    return digests


def main() -> None:
    print("=== TOURNAMENT PHASE RULEBOOK CERTIFICATION AUDIT ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Catalog: {DEFAULT_CATALOG}")
    print(f"Bundle:  {DEFAULT_BUNDLE}")

    # 1. Point-in-time publication verification
    print("\n[1/3] Auditing point-in-time publication for major tournament editions...")
    pit_results = audit_point_in_time_publication(DEFAULT_BUNDLE)
    print(
        f"  Audited {pit_results['total_editions_audited']} major tournament editions across MSI, Worlds, and LCK."
    )
    for row in pit_results["audited_major_editions"]:
        status = "CERTIFIED" if row["pit_certified"] else "DIAGNOSTIC/POST-START"
        lead = f" (lead time: {row['publication_lead_time_days']} days)" if row['publication_lead_time_days'] is not None else ""
        print(f"  - [{status}] {row['edition']} | {row['stage']} | Start: {row['tournament_start_date']}{lead}")

    # 2. Probability conservation
    print("\n[2/3] Simulating tournament formats and verifying probability conservation...")
    prob_results = audit_probability_conservation(simulations=1000)
    print(f"  Verified {len(prob_results['formats_verified'])} tournament format profiles.")
    for f in prob_results["formats_verified"]:
        champ_str = f"P(champ)={f['champion_prob_sum']:.4f}" if f["has_champion"] else "No champ"
        print(f"  - PASS: {f['format_family']:40} | {champ_str} | Max leak: {f['max_team_placement_leakage']:.2e}")
    print(
        f"  Overall max probability leakage: {prob_results['max_probability_mass_leakage']:.2e}"
    )
    print(
        f"  Zero probability mass leakage verified: {prob_results['zero_probability_mass_leakage_verified']}"
    )

    # 3. Catalog diagnostic mode status
    print("\n[3/3] Inspecting format certification catalog & diagnostic phases...")
    cat_results = audit_catalog_diagnostics()
    print(
        f"  Total catalog phases: {cat_results['total_phases']} "
        f"({cat_results['certified_format_phases_count']} certified format recipes, "
        f"{cat_results['diagnostic_mode_phases_count']} in diagnostic mode)"
    )

    # 4. Compute file digests
    print("\nComputing SHA-256 digests of verified rulebooks, formats, and artifacts...")
    digests = audit_digests()
    for name, d in digests.items():
        print(f"  {name:35}: {d}")

    # 5. Output report
    report = {
        "version": 1,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "Tournament phase rulebook certification and joint probability conservation audit",
        "point_in_time_publication_audit": pit_results,
        "joint_probability_conservation_audit": prob_results,
        "format_catalog_certification_summary": cat_results,
        "sha256_digests": digests,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nAudit complete. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
