"""`choose_threshold`, hand-worked (docs/14 §4 step 4, docs/15 §7 step 3; ticket #81).

Pure integer arithmetic on unit scores: `a = ⌊target × n / 100⌋` flags allowed, `g`
geometry-flagged units, the threshold the `(a − g + 1)`-th highest score among the
geometry-passed units. Every expected number here is worked out by hand.
"""

from __future__ import annotations

import pytest

from qs.eval.threshold import UnitScore, choose_threshold


def units(*scores: float, failed: tuple[int, ...] = ()) -> list[UnitScore]:
    """Units u001… with the given scores; the `failed` indexes are geometry-flagged."""
    return [
        UnitScore(f"u{i + 1:03d}", score, geometry_failed=i in failed)
        for i, score in enumerate(scores)
    ]


def test_fifteen_units_target_five_takes_the_highest_score() -> None:
    """15 units, no geometry flags, target 5: a = 0, the threshold is the highest score,
    k = 0, and the target is below 1/n (one unit of 15 is 6.7%)."""
    choice = choose_threshold(units(*[float(s) for s in range(1, 16)]), 5)
    assert choice.reachable
    assert choice.value == 15.0
    assert (choice.a, choice.g, choice.k, choice.n) == (0, 0, 0, 15)
    assert choice.below_one_over_n
    assert choice.flagged == ()


def test_thirty_units_target_five_takes_the_second_highest() -> None:
    """30 units, target 5: a = 1, the threshold is the second-highest score, and exactly
    one unit lies strictly above it."""
    choice = choose_threshold(units(*[float(s) for s in range(1, 31)]), 5)
    assert choice.reachable
    assert choice.a == 1
    assert choice.value == 29.0
    assert choice.k == 1
    assert not choice.below_one_over_n  # 5 × 30 = 150 ≥ 100
    assert choice.flagged == ("u030",)


def test_one_geometry_flag_leaves_the_highest_of_the_rest() -> None:
    """30 units with 1 geometry-flagged, target 5: a = 1, g = 1, so the threshold is the
    highest score among the other 29 and k = 1 — the flagged unit, combined."""
    choice = choose_threshold(units(*[float(s) for s in range(1, 31)], failed=(29,)), 5)
    assert choice.reachable
    assert (choice.a, choice.g) == (1, 1)
    assert choice.value == 29.0  # u030 failed geometry; the highest of the rest is 29
    assert choice.k == 1
    assert choice.flagged == ("u030",)


def test_target_ten_on_fifteen_units_allows_one_flag() -> None:
    """15 units, target 10: a = ⌊150 / 100⌋ = 1, the second-highest score wins."""
    choice = choose_threshold(units(*[float(s) for s in range(1, 16)]), 10)
    assert choice.reachable
    assert choice.a == 1
    assert choice.value == 14.0
    assert choice.k == 1
    assert not choice.below_one_over_n  # 10 × 15 = 150 ≥ 100


def test_two_geometry_flags_past_a_target_of_one_is_not_reachable() -> None:
    """g = 2 > a = 1: not reachable, naming both flagged units, no threshold."""
    choice = choose_threshold(units(*[float(s) for s in range(1, 16)], failed=(4, 8)), 10)
    assert not choice.reachable
    assert choice.value is None
    assert (choice.a, choice.g, choice.k, choice.n) == (1, 2, 2, 15)
    assert choice.flagged == ("u005", "u009")


def test_a_target_below_one_over_n_is_reported() -> None:
    """The boundary: 10% on 10 units is exactly 1/n, 5% on 10 units is below it."""
    at_one_in_n = choose_threshold(units(*[float(s) for s in range(1, 11)], ), 10)
    assert at_one_in_n.reachable and not at_one_in_n.below_one_over_n
    below = choose_threshold(units(*[float(s) for s in range(1, 11)], ), 5)
    assert below.reachable and below.below_one_over_n


def test_ties_keep_no_more_than_a_minus_g_units_strictly_above() -> None:
    """Tied scores: the (a − g + 1)-th highest value passes every unit at it, so no more
    than a − g units lie strictly above the threshold."""
    # ten units at 5.0, twenty at 4.0, target 5 on 30: a = 1, threshold 5.0, none above.
    tied = units(*([5.0] * 10 + [4.0] * 20))
    choice = choose_threshold(tied, 5)
    assert choice.reachable and choice.value == 5.0
    assert choice.k == 0
    assert all(u.score <= choice.value for u in tied)
    # three tied at the top, a − g = 2 (40 units, target 5): the third-highest value,
    # again none strictly above.
    triple = units(*([7.0] * 3 + [s / 10 for s in range(1, 38)]))
    choice = choose_threshold(triple, 5)
    assert choice.reachable and choice.value == 7.0
    assert choice.k == 0


def test_no_units_is_refused() -> None:
    with pytest.raises(ValueError, match="no units"):
        choose_threshold([], 5)
