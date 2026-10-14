"""Populate team_aliases table from BOOKMAKER_TO_GOLGG_ALIASES + golgg_teams matching.

Usage:
  python scripts/populate_team_aliases.py
  (runs inside Docker container with PYTHONPATH=/app)
"""

import sys
sys.path.insert(0, '/app')

from betting_app.core.db import query_df, transaction
from betting_app.core.matching import normalize_team_name
from betting_app.services.mapping_service import BOOKMAKER_TO_GOLGG_ALIASES


# ── Additional mapping: canonical team name → GOL.GG team name ──────────────
# These are entries NOT already in BOOKMAKER_TO_GOLGG_ALIASES but found via
# direct GOL.GG team name lookup on the server (golgg_teams table).
EXTRA_MAPPINGS: dict[str, str] = {
    # Canonical name (as used in canonical_matches) → GOL.GG team name
    "anyones legend":            "Anyone's Legend",
    "arctic pandas":             "Arctic Pandas",
    "barczaca":                  "Barczaca Esports",
    "bilibili":                  "Bilibili Gaming",
    "bubliki":                   "Bubliki",
    "bulldog":                   "Bulldog Esports",
    "conviction":                "Conviction",
    "dplus challengers":         "Dplus KIA Challengers",
    "dplus kia":                 "Dplus KIA",
    "drx":                       "DRX",
    "eintracht spandau":         "Eintracht Spandau",
    "esuba":                     "eSuba",
    "fluxo w7m":                 "Fluxo",
    "flyquest":                  "FlyQuest",
    "forsaken":                  "Forsaken",
    "g2 nord":                   "G2 NORD",
    "galions":                   "Galions",
    "gam":                       "GAM Esports",
    "gen g global academy":      "Gen.G Global Academy",
    "giantx":                    "GIANTX",
    "gmblers":                   "GMBLERS Esports",
    "hanwha life":               "Hanwha Life Esports",
    "hmble":                     "HMBLE",
    "jd":                        "JD Gaming",
    "karmine corp":              "Karmine Corp",
    "karmine corp blue":         "Karmine Corp Blue",
    "kt rolster":                "KT Rolster",
    "lgd":                       "LGD Gaming",
    "liquid":                    "Team Liquid",
    "los grandes":               "Los Grandes",
    "meavedron":                 "Meavedron",
    "mvk":                       "MVK Esports",
    "natus vincere":             "Natus Vincere",
    "nongshim redforce":         "Nongshim RedForce",
    "nongshim redforce challengers": "Nongshim Esports Academy",
    "ns red force":              "Nongshim RedForce",
    "red canids":                "RED Canids",
    "red canids kalunga":        "RED Canids",
    "ronaldo":                   "Ronaldo Team",
    "secret whales":             "Team Secret Whales",
    "sentinels":                 "Sentinels",
    "shopify rebellion":         "Shopify Rebellion",
    "soopers challengers":       "DN SOOPers Challengers",
    "t1":                        "T1",
    "thundertalk":               "ThunderTalk Gaming",
    "vitality":                  "Team Vitality",
    "we":                        "Team WE",
}


def main() -> None:
    print("=== Populating team_aliases table ===")

    # 1. Insert all entries from BOOKMAKER_TO_GOLGG_ALIASES
    count_builtin = 0
    with transaction() as conn:
        for normalized_name, golgg_name in BOOKMAKER_TO_GOLGG_ALIASES.items():
            conn.execute(
                """
                INSERT INTO team_aliases(normalized_name, alias, source)
                VALUES (?, ?, 'builtin')
                ON CONFLICT(normalized_name, source) DO UPDATE SET
                    alias = excluded.alias
                """,
                (normalized_name, golgg_name),
            )
            count_builtin += 1
    print(f"  Inserted/updated {count_builtin} aliases from BOOKMAKER_TO_GOLGG_ALIASES")

    # 2. Insert EXTRA_MAPPINGS
    count_extra = 0
    with transaction() as conn:
        for normalized_name, golgg_name in EXTRA_MAPPINGS.items():
            conn.execute(
                """
                INSERT INTO team_aliases(normalized_name, alias, source)
                VALUES (?, ?, 'manual')
                ON CONFLICT(normalized_name, source) DO UPDATE SET
                    alias = excluded.alias
                """,
                (normalized_name, golgg_name),
            )
            count_extra += 1
    print(f"  Inserted/updated {count_extra} aliases from EXTRA_MAPPINGS")

    # 3. Try to auto-map remaining unmapped teams via golgg_teams fuzzy match
    unmapped = query_df(
        """
        WITH raw_names AS (
            SELECT normalized_team_a AS name FROM canonical_matches
            UNION
            SELECT normalized_team_b AS name FROM canonical_matches
        )
        SELECT DISTINCT name FROM raw_names
        WHERE LOWER(TRIM(name)) NOT IN (
            SELECT LOWER(TRIM(normalized_name)) FROM team_aliases
        )
        ORDER BY name
        """
    )
    count_fuzzy = 0
    if not unmapped.empty:
        golgg_names = query_df("SELECT team_name FROM golgg_teams")
        golgg_list = golgg_names["team_name"].dropna().str.strip().unique().tolist() if not golgg_names.empty else []

        from betting_app.core.matching import best_match

        for row in unmapped.itertuples():
            name = str(row.name).strip()
            if not name:
                continue
            matched_name, score = best_match(name, golgg_list)
            if matched_name and score >= 0.65:
                with transaction() as conn:
                    conn.execute(
                        """
                        INSERT INTO team_aliases(normalized_name, alias, source)
                        VALUES (?, ?, 'auto')
                        ON CONFLICT(normalized_name, source) DO UPDATE SET
                            alias = excluded.alias
                        """,
                        (name, matched_name),
                    )
                count_fuzzy += 1
                print(f"    Auto-mapped: '{name}' → '{matched_name}' (score={score:.2f})")
            else:
                print(f"    ⚠️  Could not auto-map: '{name}' (best={matched_name}, score={score:.2f})")

    print(f"  Fuzzy-matched {count_fuzzy} additional aliases")

    # 4. Summary
    total = query_df("SELECT COUNT(*) AS cnt FROM team_aliases")
    print(f"\n✅ Done! Total team_aliases: {total.iloc[0]['cnt']}")


if __name__ == "__main__":
    main()
