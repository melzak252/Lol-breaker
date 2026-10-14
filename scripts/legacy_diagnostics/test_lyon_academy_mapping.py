"""Regression tests for LYON / LYON Academy team-name normalization.

Bug: ALIASES mapped both "lyon" and "lyon academy" to "lyonacademy",
     collapsing main LYON and LYON Academy into the same canonical key.
Fix: "lyon" -> "lyon", "lyon gaming" -> "lyon", "lyon academy" -> "lyonacademy".
"""

from betting_app.core.matching import normalize_team_name, similarity


class TestLyonAcademyDistinct:
    """Main LYON and LYON Academy must NOT share the same normalized key."""

    def test_lyon_normalizes_to_lyon(self):
        assert normalize_team_name("LYON") == "lyon"

    def test_lyon_gaming_normalizes_to_lyon(self):
        assert normalize_team_name("Lyon Gaming") == "lyon"

    def test_lyon_academy_normalizes_to_lyonacademy(self):
        assert normalize_team_name("LYON Academy") == "lyonacademy"

    def test_lyon_academy_lowercase_normalizes_to_lyonacademy(self):
        assert normalize_team_name("lyon academy") == "lyonacademy"

    def test_lyon_and_lyon_academy_are_distinct(self):
        """The core regression: these must NOT be equal."""
        norm_main = normalize_team_name("LYON")
        norm_academy = normalize_team_name("LYON Academy")
        assert norm_main != norm_academy

    def test_similarity_lyon_vs_lyon_academy_is_not_perfect(self):
        """similarity() should be < 1.0 so they don't cross-match."""
        sim = similarity("LYON", "LYON Academy")
        assert sim < 1.0

    def test_similarity_lyon_vs_lyon_gaming_is_perfect(self):
        """LYON and Lyon Gaming are the same team — should match perfectly."""
        sim = similarity("LYON", "Lyon Gaming")
        assert sim == 1.0

    def test_lyon_gaming_vs_lyon_academy_is_not_perfect(self):
        """Lyon Gaming (main) vs LYON Academy must not cross-match."""
        sim = similarity("Lyon Gaming", "LYON Academy")
        assert sim < 1.0