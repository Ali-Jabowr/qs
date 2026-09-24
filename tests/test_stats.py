"""`qs.eval.stats` against `tools/bounds.py`, loaded by path (docs/14 §5, §9).

The reference implementation is executed from `tools/bounds.py` exactly as it sits in the
repo; every row of its two tables and the named rows of docs/14 §5 and §9 must come out of
`qs.eval.stats` identical to 1e-9, and the printed percentages to the digit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from qs.eval import stats

_REFERENCE = Path(__file__).resolve().parents[1] / "tools" / "bounds.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("reference_bounds", _REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bounds = _load()

# docs/14 §5: the recall table (n; all caught; 90% caught), §9: the false-alarm table.
RECALL_ROWS = [(n, k) for n in (5, 10, 15, 20, 25, 30, 40)
               for k in (n, round(0.9 * n))]
FAR_ROWS = [(n, k) for n in (10, 20, 30, 40, 50, 60, 80, 100)
            for k in (0, round(0.05 * n))]
GATE_A_ROWS = [(30, k) for k in (30, 29, 28, 27, 26, 25)]

NAMED = [
    ((30, 27), "lower", 76.1),
    ((12, 12), "lower", 77.9),
    ((12, 11), "lower", 66.1),
    ((9, 9), "lower", 71.7),
    ((30, 0), "upper", 9.5),
    ((30, 1), "upper", 14.9),
]


@pytest.mark.parametrize("n,k", [*RECALL_ROWS, *FAR_ROWS, *GATE_A_ROWS])
def test_every_row_matches_the_reference_to_1e_9(n: int, k: int) -> None:
    assert stats.lower_bound(n, k) == pytest.approx(bounds.lower_bound(n, k), abs=1e-9)  # type: ignore[attr-defined]
    assert stats.upper_bound(n, k) == pytest.approx(bounds.upper_bound(n, k), abs=1e-9)  # type: ignore[attr-defined]


@pytest.mark.parametrize(("n_k", "which", "percent"), NAMED)
def test_the_named_rows_of_docs_14(n_k: tuple[int, int], which: str, percent: float) -> None:
    n, k = n_k
    bound = stats.lower_bound(n, k) if which == "lower" else stats.upper_bound(n, k)
    assert round(bound * 100, 1) == percent


def test_the_degenerate_rows_match_the_reference() -> None:
    for n in (1, 10, 30):
        assert stats.lower_bound(n, 0) == 0.0 == bounds.lower_bound(n, 0)  # type: ignore[attr-defined]
        assert stats.upper_bound(n, n) == 1.0 == bounds.upper_bound(n, n)  # type: ignore[attr-defined]
