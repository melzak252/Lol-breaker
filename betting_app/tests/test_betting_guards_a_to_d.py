"""Unit tests for betting guards A through D and de-vigging algorithms."""
import math
import pytest
from src.analysis.betting_guards_a_to_d import (
    devig_multiplicative,
    devig_additive,
    devig_power,
    devig_shin,
    dynamic_logit_disagreement,
    apply_guard_b_dynamic_disagreement,
    apply_guard_a_plow,
    apply_guard_c_haircut,
    calculate_clv,
    apply_guard_d_clv_drift,
    wasserstein_dro_min_ev_hurdle,
    apply_guard_w_wasserstein_dro,
)

class TestDeviggingMethods:
    def test_multiplicative_symmetry_and_sum(self):
        pa, pb = devig_multiplicative(1.90, 1.90)
        assert pa == pytest.approx(0.50, abs=1e-10)
        assert pb == pytest.approx(0.50, abs=1e-10)
        assert (pa + pb) == pytest.approx(1.0, abs=1e-10)

    def test_additive_favorite_adjustment(self):
        pa, pb = devig_additive(1.30, 3.50)
        pa_mult, pb_mult = devig_multiplicative(1.30, 3.50)
        assert pa > pa_mult
        assert (pa + pb) == pytest.approx(1.0, abs=1e-10)

    def test_power_devig_sum_to_one(self):
        pa, pb = devig_power(1.30, 3.50)
        assert (pa + pb) == pytest.approx(1.0, abs=1e-10)
        assert pa > 0.70

    def test_shin_two_way_does_not_divide_by_zero(self):
        # 2-way market: n = 2. Must not divide by (n-2) = 0!
        pa, pb = devig_shin(1.30, 3.50)
        assert (pa + pb) == pytest.approx(1.0, abs=1e-10)
        assert pa > 0.70
        assert pb < 0.30

    def test_invalid_odds_raise(self):
        for bad in [0.95, -1.0, float('nan'), float('inf')]:
            with pytest.raises(ValueError):
                devig_multiplicative(bad, 2.0)
            with pytest.raises(ValueError):
                devig_additive(bad, 2.0)
            with pytest.raises(ValueError):
                devig_power(bad, 2.0)
            with pytest.raises(ValueError):
                devig_shin(bad, 2.0)


class TestGuardBDynamicDisagreement:
    def test_symmetry_under_role_inversion(self):
        # Model 0.70 vs Market 0.65 must have identical logit gap to Model 0.30 vs Market 0.35
        gap1 = dynamic_logit_disagreement(0.70, 0.65)
        gap2 = dynamic_logit_disagreement(0.30, 0.35)
        assert gap1 == pytest.approx(gap2, abs=1e-12)

    def test_tail_sensitivity_at_50_vs_95(self):
        # At 50%, a 7% gap is a modest logit shift
        passes_50, gap_50 = apply_guard_b_dynamic_disagreement(0.57, 0.50, max_logit_gap=0.35)
        assert passes_50 is True
        assert gap_50 < 0.35

        # At 95%, a 3% gap (98% vs 95%) is a huge logit shift (> 0.7) and must be REJECTED
        passes_95, gap_95 = apply_guard_b_dynamic_disagreement(0.98, 0.95, max_logit_gap=0.35)
        assert passes_95 is False
        assert gap_95 > 0.70

    def test_exact_agreement_zero_gap(self):
        assert dynamic_logit_disagreement(0.65, 0.65) == pytest.approx(0.0, abs=1e-12)


class TestGuardAPlow:
    def test_plow_positive_ev_passes(self):
        # Odds 2.50 with 12% tax -> net multiplier = 2.50 * 0.88 = 2.20
        # If P_low is 0.50 -> EV = 0.50 * 2.20 - 1.0 = +10%
        passes, ev = apply_guard_a_plow(0.50, 2.50, tax_rate=0.12, min_ev=0.05)
        assert passes is True
        assert ev == pytest.approx(0.10, abs=1e-10)

    def test_plow_negative_ev_fails(self):
        # P_low is 0.40 -> EV = 0.40 * 2.20 - 1.0 = -12%
        passes, ev = apply_guard_a_plow(0.40, 2.50, tax_rate=0.12, min_ev=0.0)
        assert passes is False
        assert ev < 0.0


class TestGuardCHaircut:
    def test_haircut_discounts_raw_ev(self):
        # Raw EV = +20%. Haircut multiplier = 0.30, margin = 0.02.
        # Discounted EV = 0.30 * 0.20 - 0.02 = 0.06 - 0.02 = +4.0%
        passes, ev_disc = apply_guard_c_haircut(0.20, haircut_multiplier=0.30, safety_margin=0.02)
        assert passes is True
        assert ev_disc == pytest.approx(0.04, abs=1e-10)

    def test_marginal_ev_fails_after_haircut(self):
        # Raw EV = +5%. Discounted = 0.30 * 0.05 - 0.02 = 0.015 - 0.02 = -0.5%
        passes, ev_disc = apply_guard_c_haircut(0.05, haircut_multiplier=0.30, safety_margin=0.02)
        assert passes is False
        assert ev_disc < 0.0


class TestGuardDCLV:
    def test_clv_beating_closing_line(self):
        # Taken odds 2.50 (after 12% tax = 2.20). Closing no-vig odds = 2.00.
        # CLV = 2.20 / 2.00 - 1.0 = +10.0%
        clv = calculate_clv(2.50, 2.00, tax_rate=0.12)
        assert clv == pytest.approx(0.10, abs=1e-10)

    def test_adverse_market_drift_fails(self):
        # Market prob was 0.55 at bet time, collapsed to 0.50 at close -> drift = +0.05 > 0.03
        passes, drift = apply_guard_d_clv_drift(0.55, 0.50, max_adverse_drift=0.03)
        assert passes is False
        assert drift == pytest.approx(0.05, abs=1e-10)


class TestGuardWWassersteinDRO:
    def test_odds_scaled_hurdle_increases_with_odds(self):
        # At odds=2.00, 12% tax, eps=0.025: 0.88 * 2.00 * 0.025 = 0.044 < 0.05 -> floor 0.05
        h_low = wasserstein_dro_min_ev_hurdle(2.00, tax_rate=0.12, base_min_ev=0.05, epsilon=0.025)
        assert h_low == pytest.approx(0.05)
        # At odds=3.50, 12% tax, eps=0.025: 0.88 * 3.50 * 0.025 = 0.077 > 0.05 -> hurdle 0.077
        h_high = wasserstein_dro_min_ev_hurdle(3.50, tax_rate=0.12, base_min_ev=0.05, epsilon=0.025)
        assert h_high == pytest.approx(0.077)

    def test_apply_guard_w_prunes_fragile_longshot(self):
        # prob=0.345, odds=3.50, tax=0.12 -> raw EV = +6.26%, but worst-case EV_DRO = -1.44%
        passes, ev_dro, hurdle = apply_guard_w_wasserstein_dro(0.345, 3.50, tax_rate=0.12, base_min_ev=0.05, epsilon=0.025)
        assert passes is False
        assert ev_dro < 0.0
        assert hurdle == pytest.approx(0.077)

        # prob=0.360, odds=3.50, tax=0.12 -> worst-case EV_DRO = 0.335 * 3.08 - 1 = +3.18% > 0
        passes_ok, ev_dro_ok, _ = apply_guard_w_wasserstein_dro(0.360, 3.50, tax_rate=0.12, base_min_ev=0.05, epsilon=0.025)
        assert passes_ok is True
        assert ev_dro_ok > 0.0
