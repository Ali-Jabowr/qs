#!/usr/bin/env python3
"""Clopper-Pearson bounds — what a given sample size can and cannot demonstrate.

    python3 tools/bounds.py

Prints the tables in docs/14-eval-protocol.md §5. They are generated rather than
asserted so that anyone can check them, including us in three months.

Exact binomial ("Clopper-Pearson") intervals are used rather than the normal
approximation because n here is 20-60, where the normal approximation is wrong in
the direction that flatters us.

This module is the reference implementation. src/qs/eval/stats.py will import the
same two functions; the tests compare them against known values.
"""

from __future__ import annotations

from math import comb


def _p_at_least(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _p_at_most(n: int, k: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(predicate, iterations: int = 200) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def lower_bound(n: int, k: int, alpha: float = 0.05) -> float:
    """One-sided lower confidence bound on the true rate, given k of n succeeded.

    'We caught 27 of 30 dents' -> the true catch rate is at least this, with
    confidence 1 - alpha. For k == n this is exactly alpha ** (1/n).
    """
    if k <= 0:
        return 0.0
    return _bisect(lambda p: _p_at_least(n, k, p) < alpha)


def upper_bound(n: int, k: int, alpha: float = 0.05) -> float:
    """One-sided upper confidence bound, given k of n occurred.

    'We flagged 2 of 40 good units' -> the true false-alarm rate is at most this,
    with confidence 1 - alpha.
    """
    if k >= n:
        return 1.0
    return _bisect(lambda p: _p_at_most(n, k, p) > alpha)


def smallest_clean_n(target: float = 0.05, alpha: float = 0.05, limit: int = 500) -> int:
    """How many good units must pass cleanly to demonstrate FAR <= target."""
    for n in range(1, limit):
        if upper_bound(n, 0, alpha) <= target:
            return n
    raise ValueError("no n below the limit")


def main() -> None:
    print("Recall — what a result on n defective units actually demonstrates")
    print(f"  {'n':>4}  {'all caught':>12}  {'90% caught':>24}")
    for n in (5, 10, 15, 20, 25, 30, 40):
        k90 = round(0.9 * n)
        cell = f"{k90}/{n} -> {lower_bound(n, k90) * 100:.1f}%"
        print(f"  {n:>4}  {lower_bound(n, n) * 100:>11.1f}%  {cell:>24}")

    print()
    print("False alarms — what a result on n good units actually demonstrates")
    print(f"  {'n':>4}  {'none flagged':>14}  {'5% flagged':>24}")
    for n in (10, 20, 30, 40, 50, 60, 80, 100):
        k5 = round(0.05 * n)
        cell = f"{k5}/{n} -> {upper_bound(n, k5) * 100:.1f}%"
        print(f"  {n:>4}  {upper_bound(n, 0) * 100:>13.1f}%  {cell:>24}")

    print()
    n = smallest_clean_n()
    print(f"To demonstrate a false-alarm rate at or below 5% with 95% confidence,")
    print(f"{n} good units must pass with none flagged "
          f"(upper bound {upper_bound(n, 0) * 100:.2f}%).")

    print()
    print("Gate A: 30 dents, lower bound must clear 75%")
    for k in (30, 29, 28, 27, 26, 25):
        print(f"  {k}/30 = {k / 30 * 100:>5.1f}% observed -> "
              f"lower bound {lower_bound(30, k) * 100:.1f}%")


if __name__ == "__main__":
    main()
