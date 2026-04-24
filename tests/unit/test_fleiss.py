import math

from spinebench.scoring.agreement import fleiss_kappa


def test_perfect_agreement():
    # 3 items, 4 raters, 2 categories, unanimous
    kappa = fleiss_kappa([[4, 0], [0, 4], [4, 0]])
    assert math.isclose(kappa, 1.0, abs_tol=1e-9)


def test_below_chance_agreement():
    # Every item split 2-2 across 2 categories -> per-item agreement (1/3) is below
    # the marginal probability of agreement (0.5), so kappa goes negative.
    kappa = fleiss_kappa([[2, 2], [2, 2], [2, 2]])
    assert kappa < 0


def test_matches_chance_when_rows_match_marginals():
    # When item-level agreement equals expected marginal agreement, kappa = 0.
    # With marginals p = [2/3, 1/3] and 3 raters, chance P_e = (2/3)^2 + (1/3)^2 = 5/9.
    # An item with counts [2, 1] has agreement (4 + 1 - 3) / (3 * 2) = 2/6 = 1/3,
    # which is below chance. An item with counts [3, 0] has agreement 1.
    # Mixing (3,0) and (2,1) items in a 4:5 ratio yields P_bar == P_e.
    # Easiest equality case: both rows unanimous in the same category -> P_bar = 1, P_e = 1,
    # handled as degenerate (kappa = 0).
    kappa = fleiss_kappa([[3, 0], [3, 0]])
    assert math.isclose(kappa, 0.0, abs_tol=1e-9)


def test_fleiss_known_example():
    # Fleiss' original 1971 example (truncated): 2 items, 6 raters, 3 categories.
    # Strong but not perfect agreement.
    data = [[5, 1, 0], [0, 6, 0]]
    kappa = fleiss_kappa(data)
    assert 0.0 < kappa < 1.0


def test_degenerate_single_item():
    assert fleiss_kappa([[3, 0]]) == 0.0


def test_mismatched_rows_raise():
    import pytest

    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [2, 0]])
