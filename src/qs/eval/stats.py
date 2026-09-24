"""One-sided 95% Clopper-Pearson bounds (docs/14 §4 step 6, §5, §9; FR-55).

`lower_bound(n, k)` and `upper_bound(n, k)` are the same two functions as `tools/bounds.py`
— the reference implementation, whose docstring names this module — the same exact-binomial
bisection, so a bound in a report and a bound from the tool agree to the last bit. Every
rate in the INDICATIVE report carries its bound: recall its lower bound, the false-alarm
rate its upper bound.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb


def _p_at_least(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _p_at_most(n: int, k: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(predicate: Callable[[float], bool], iterations: int = 200) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def lower_bound(n: int, k: int, alpha: float = 0.05) -> float:
    """One-sided lower confidence bound on the true rate, given k of n caught.

    `k == 0` gives 0.0 — nothing was demonstrated; `k == n` is exactly alpha ** (1/n).
    """
    if k <= 0:
        return 0.0
    return _bisect(lambda p: _p_at_least(n, k, p) < alpha)


def upper_bound(n: int, k: int, alpha: float = 0.05) -> float:
    """One-sided upper confidence bound, given k of n occurred.

    `k >= n` gives 1.0; `k == 0` is exactly 1 - alpha ** (1/n).
    """
    if k >= n:
        return 1.0
    return _bisect(lambda p: _p_at_most(n, k, p) > alpha)
