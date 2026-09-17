"""The confidence bounds decide the gate criteria, so they get checked against
closed forms and against published Clopper-Pearson values rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bounds import lower_bound, smallest_clean_n, upper_bound  # noqa: E402


@pytest.mark.parametrize("n", [1, 5, 10, 20, 30, 59, 100])
def test_all_successes_matches_the_closed_form(n):
    """For k == n the one-sided lower bound is exactly alpha ** (1/n)."""
    assert lower_bound(n, n) == pytest.approx(0.05 ** (1 / n), abs=1e-9)


@pytest.mark.parametrize("n", [1, 5, 10, 20, 59, 100])
def test_no_occurrences_matches_the_rule_of_three(n):
    """For k == 0 the one-sided upper bound is exactly 1 - alpha ** (1/n)."""
    assert upper_bound(n, 0) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)


def test_degenerate_cases():
    assert lower_bound(10, 0) == 0.0
    assert upper_bound(10, 10) == 1.0


@pytest.mark.parametrize("n,k", [(20, 5), (30, 27), (40, 36), (100, 50)])
def test_the_bounds_bracket_the_point_estimate(n, k):
    assert lower_bound(n, k) < k / n < upper_bound(n, k)


def test_more_evidence_never_weakens_the_bound():
    """Same observed rate, larger n, the lower bound must not get worse."""
    previous = 0.0
    for n in (10, 20, 30, 40, 50, 100):
        b = lower_bound(n, round(0.9 * n))
        assert b >= previous - 1e-9
        previous = b


def test_catching_more_never_weakens_the_bound():
    previous = 0.0
    for k in range(0, 31):
        b = lower_bound(30, k)
        assert b >= previous - 1e-9
        previous = b


# ---------------------------------------------------------------- the gate design

def test_the_retired_gate_could_not_have_meant_anything():
    """5 of 5 dents caught — the 2026-09-02 criterion — is consistent with 55%."""
    assert lower_bound(5, 5) == pytest.approx(0.549, abs=0.001)


def test_thirty_is_the_smallest_defect_count_that_clears_the_gate():
    """Gate A needs an observed 90% to clear a 75% lower bound. n=20 does not."""
    assert lower_bound(20, 18) < 0.75
    assert lower_bound(30, 27) >= 0.75


def test_gate_a_passes_at_27_of_30_and_fails_at_26():
    assert lower_bound(30, 27) >= 0.75
    assert lower_bound(30, 26) < 0.75


def test_demonstrating_a_five_percent_false_alarm_rate_needs_59_clean_units():
    assert smallest_clean_n(0.05) == 59
    assert upper_bound(59, 0) <= 0.05
    assert upper_bound(58, 0) > 0.05


def test_forty_good_units_cannot_demonstrate_five_percent():
    """Why Gate A's A4 is a point estimate and says so, rather than a claim."""
    assert upper_bound(40, 0) > 0.05
    assert upper_bound(40, 2) == pytest.approx(0.149, abs=0.002)
