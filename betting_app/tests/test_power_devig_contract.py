"""Behavior boundaries for the explicit power margin-removal method."""
import math
import pytest
from src.analysis.devig_comparison import devig_power


def test_extreme_overround_solves_power_not_proportional():
    a, b = devig_power(1.01, 1.02)
    assert a == pytest.approx(.6172218120806231, abs=1e-12)
    assert a + b == pytest.approx(1., abs=1e-10)
    assert -math.log(a) / math.log(1.01) == pytest.approx(48.49354517564157, abs=1e-10)
    assert devig_power(1.02, 1.01) == pytest.approx((b, a), abs=1e-12)


@pytest.mark.parametrize('bad', [None, True, 0, 1, float('nan'), float('inf'), 'invalid'])
def test_invalid_price_is_explicit_failure(bad):
    with pytest.raises(ValueError):
        devig_power(bad, 2.)
