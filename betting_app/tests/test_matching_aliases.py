from betting_app.core.matching import normalize_team_name, similarity


def test_normalize_esports_suffix_spelled_as_e_sports() -> None:
    assert normalize_team_name("Estral E-Sports") == "estral"
    assert normalize_team_name("Estral Esports") == "estral"
    assert similarity("Estral E-Sports", "Estral Esports") == 1.0


def test_normalize_ub_alma_mater_alias() -> None:
    assert normalize_team_name("UB Alma Mater") == "universitat de barcelona"
    assert normalize_team_name("Universitat de Barcelona") == "universitat de barcelona"
    assert similarity("UB Alma Mater", "Universitat de Barcelona") == 1.0


def test_normalize_kabum_ild_alias() -> None:
    assert normalize_team_name("KaBuM! Ilha das Lendas") == "kabum idl"
    assert normalize_team_name("KaBuM! IDL") == "kabum idl"
    assert similarity("KaBuM! Ilha das Lendas", "KaBuM! IDL") == 1.0


def test_normalize_solary_eclipse_alias() -> None:
    assert normalize_team_name("Solary Eclipse") == "solary"
    assert normalize_team_name("Solary") == "solary"
    assert similarity("Solary Eclipse", "Solary") == 1.0


def test_normalize_les_academy_aliases() -> None:
    assert normalize_team_name("GIANTX Academy") == "giantx itero"
    assert normalize_team_name("GIANTX iTero") == "giantx itero"
    assert normalize_team_name("MKOI") == "movistar koi"
    assert normalize_team_name("Movistar KOI") == "movistar koi"
    assert normalize_team_name("KOI Academy") == "movistar koi fenix"
    assert normalize_team_name("Movistar KOI Academy") == "movistar koi fenix"
    assert normalize_team_name("MKF") == "movistar koi fenix"
    assert normalize_team_name("Movistar KOI Fenix") == "movistar koi fenix"
    assert similarity("GIANTX Academy", "GIANTX iTero") == 1.0
    assert similarity("KOI Academy", "Movistar KOI Fenix") == 1.0
    assert similarity("Movistar KOI", "Movistar KOI Fenix") < 0.72
    assert similarity("MKOI", "MKF") < 0.72


def test_giantx_itero_and_pride_are_not_the_same_squad() -> None:
    assert normalize_team_name("GIANTX iTero") == "giantx itero"
    assert normalize_team_name("GIANTX Pride") == "giantx pride"
    assert similarity("GIANTX iTero", "GIANTX Pride") < 0.68


def test_canonical_koi_context_keeps_main_and_fenix_distinct() -> None:
    from betting_app.services.canonical_match_service import canonical_match_score, canonical_team_key

    assert canonical_team_key("Movistar KOI", league="LEC") == "movistar koi"
    assert canonical_team_key("Movistar KOI", league="LES") == "movistar koi fenix"
    assert canonical_team_key("KOI Academy", league="LES") == "movistar koi fenix"

    score = canonical_match_score(
        "ucam",
        canonical_team_key("Movistar KOI Fenix", league="LES"),
        "2026-08-05T17:30:00+00:00",
        "les",
        {
            "normalized_team_a": "ucam",
            "normalized_team_b": canonical_team_key("Movistar KOI", league="LEC"),
            "start_time_normalized": "2026-08-05T17:30:00+00:00",
            "league": "LEC",
            "status": "upcoming",
        },
    )
    assert score < 0.78


def test_normalize_big_alias_after_esports_suffix_cleanup() -> None:
    assert normalize_team_name("BIG") == "big"
    assert normalize_team_name("Berlin International Gaming") == "big"
    assert normalize_team_name("Berlin International") == "big"
    assert similarity("BIG", "Berlin International Gaming") == 1.0


def test_canonical_team_key_applies_static_aliases_after_scoped_alias(monkeypatch) -> None:
    from betting_app.services import canonical_match_service
    from betting_app.services.team_alias_service import AliasResolution

    monkeypatch.setattr(
        canonical_match_service,
        "resolve_scoped_alias",
        lambda *_args, **_kwargs: AliasResolution(
            target_name="ThunderTalk Gaming",
            normalized_target="thundertalk gaming",
            source="test",
            alias_id=1,
            confidence=1.0,
        ),
    )
    assert canonical_match_service.canonical_team_key("TT") == "thundertalk"


def test_canonical_score_rejects_unrelated_challenger_teams_at_same_time() -> None:
    from betting_app.services.canonical_match_service import canonical_match_score

    score = canonical_match_score(
        "dn soopers challengers",
        "t1 challengers",
        "2026-08-06T05:00:00+00:00",
        "lck challengers",
        {
            "normalized_team_a": "geng challengers",
            "normalized_team_b": "brion challengers",
            "start_time_normalized": "2026-08-06T05:00:00+00:00",
            "league": "lck challengers",
            "status": "upcoming",
        },
    )
    assert score < 0.78


def test_lck_cl_academy_aliases_and_mappings() -> None:
    from betting_app.services.canonical_match_service import canonical_team_key
    from betting_app.services.mapping_service import BOOKMAKER_TO_GOLGG_ALIASES

    # Canonical keys must align variants to the active 2026 academy identity
    assert canonical_team_key("T1 Challengers") == "t1 academy"
    assert canonical_team_key("T1 Esports Academy") == "t1 academy"
    assert canonical_team_key("T1 Academy") == "t1 academy"

    assert canonical_team_key("Drx Challengers") == "krx challengers"
    assert canonical_team_key("Kiwoom DRX Challengers") == "krx challengers"
    assert canonical_team_key("KRX Challengers") == "krx challengers"

    assert canonical_team_key("NS Challengers") == "nongshim academy"
    assert canonical_team_key("Nongshim Redforce Challengers") == "nongshim academy"

    assert canonical_team_key("Gen.G Challengers") == "geng global academy"
    # Main team Gen.G must stay distinct
    assert canonical_team_key("Gen.G") == "gen g"
    assert canonical_team_key("T1") == "t1"
    assert canonical_team_key("DRX") == "drx"

    # Mapping service aliases must point to active 2026 GOL.GG team names
    assert BOOKMAKER_TO_GOLGG_ALIASES["t1 challengers"] == "T1 Esports Academy"
    assert BOOKMAKER_TO_GOLGG_ALIASES["drx challengers"] == "KRX Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["kiwoom drx challengers"] == "KRX Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["kt challengers"] == "KT Rolster Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["kt rolster challengers"] == "KT Rolster Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["ktrolster challengers"] == "KT Rolster Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["ns challengers"] == "Nongshim Esports Academy"
    assert BOOKMAKER_TO_GOLGG_ALIASES["fearx challengers"] == "BNK FEARX Youth"
    assert BOOKMAKER_TO_GOLGG_ALIASES["brion challengers"] == "HANJIN BRION Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["gen g challengers"] == "Gen.G Global Academy"
    assert BOOKMAKER_TO_GOLGG_ALIASES["hanwha life challengers"] == "Hanwha Life Esports Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["dplus challengers"] == "Dplus KIA Challengers"
    assert BOOKMAKER_TO_GOLGG_ALIASES["soopers challengers"] == "DN SOOPers Challengers"
