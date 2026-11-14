"""Limits from block C good units and the locked test table (#177).

The bounds were checked against `tools/bounds.py`: upper_bound(300, 3) = 0.0256 and
lower_bound(10, 9) = 0.6058, as the ticket says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.errors import ConfigError
from qs.lane.defects import Defect
from qs.lane.unit_limits import (
    LimitsFile,
    Scored,
    StateLimits,
    derive_limits,
    from_score_rows,
    judge,
    load_limits,
    read_scores,
    render_report,
    write_limits,
)
from qs.lane.unit_scores import ScoreRow, write_scores

REF = (70.0, 150.0, 120.0)


def good(
    state: str = "full_pink", spread: float = 1.0, colour: tuple[float, float, float] = REF,
    patch: float | None = None, frame: int = 0, video: str = "C-1.mkv",
    shape: tuple[float, float, float] | None = None,
) -> Scored:
    area, ratio, solidity = shape if shape else (None, None, None)
    return Scored(video, frame, state, "", spread, *colour, patch, area, ratio, solidity)


def faulted(state: str = "full_pink", frame: int = 0, video: str = "E-1.mkv") -> Scored:
    return Scored(video, frame, state, "unit size out of range")


def limits(**changes: object) -> StateLimits:
    base: dict[str, object] = dict(
        n_usable=40, n_faulted=0, colour_reference=REF, colour_shift_max=5.0,
        colour_spread_max=5.0)
    return StateLimits(**{**base, **changes})  # type: ignore[arg-type]


def table_cells(text: str, label: str) -> list[str]:
    """The cells after the label of the table row that starts with `| label |`."""
    (line,) = [ln for ln in text.splitlines() if ln.startswith(f"| {label} |")]
    return [cell.strip() for cell in line.strip("|").split("|")][1:]


# --- limits -----------------------------------------------------------------------------


def test_the_spread_limit_lets_one_percent_of_two_hundred_rows_through() -> None:
    rows = [good(spread=float(i)) for i in range(1, 201)]
    assert derive_limits(rows).limits["full_pink"].colour_spread_max == 198.0


def test_under_a_hundred_rows_the_limit_is_the_largest_value() -> None:
    rows = [good(spread=float(i)) for i in range(1, 61)]
    assert derive_limits(rows).limits["full_pink"].colour_spread_max == 60.0


def test_colour_reference_is_the_median_and_the_shift_limit_the_largest_distance() -> None:
    rows = [good() for _ in range(30)] + [good(colour=(70.0, 153.0, 124.0))]
    found = derive_limits(rows).limits["full_pink"]
    assert found.colour_reference == (70.0, 150.0, 120.0)
    assert found.colour_shift_max == 5.0


def test_a_state_with_29_usable_rows_gets_no_limits_and_one_line() -> None:
    rows = [good(state="full_white") for _ in range(29)] + [good() for _ in range(30)]
    derived = derive_limits(rows)
    assert sorted(derived.limits) == ["full_pink"]
    assert derived.notes == ["full_white: 29 usable rows, fewer than 30; no limits"]


def test_faulted_rows_are_left_out_and_counted() -> None:
    rows = [good() for _ in range(31)] + [faulted(), faulted()]
    found = derive_limits(rows).limits["full_pink"]
    assert (found.n_usable, found.n_faulted) == (31, 2)


def test_states_are_derived_apart() -> None:
    rows = [good(spread=1.0) for _ in range(30)]
    rows += [good(state="full", spread=9.0) for _ in range(30)]
    derived = derive_limits(rows).limits
    assert (derived["full_pink"].colour_spread_max, derived["full"].colour_spread_max) == (1.0, 9.0)


def test_the_patch_limit_is_written_only_when_patch_scores_are_present() -> None:
    plain = derive_limits([good() for _ in range(30)]).limits["full_pink"]
    scored = derive_limits([good(patch=float(i)) for i in range(1, 41)]).limits["full_pink"]
    assert (plain.patch_score_max, scored.patch_score_max) == (None, 40.0)


def test_shape_limits_are_the_extremes_seen() -> None:
    rows = [good(shape=(3000.0 + 10 * i, 0.8 + 0.001 * i, 0.95 + 0.001 * i)) for i in range(31)]
    found = derive_limits(rows).limits["full_pink"]
    assert found.area_px == (3000.0, 3300.0)
    assert (found.axis_ratio_min, found.solidity_min) == (0.8, 0.95)


def test_rows_with_empty_shape_cells_give_no_shape_limits() -> None:
    found = derive_limits([good() for _ in range(30)]).limits["full_pink"]
    assert (found.area_px, found.axis_ratio_min, found.solidity_min) == (None, None, None)


# --- files ------------------------------------------------------------------------------


def test_the_limits_file_round_trips_and_is_never_overwritten(tmp_path: Path) -> None:
    given = LimitsFile(
        "a" * 64, "b" * 64, {"full_pink": ("m-1", "c" * 64)},
        {"full_pink": limits(patch_score_max=7.5, area_px=(3000.0, 9000.0),
                             axis_ratio_min=0.8, solidity_min=0.9)})
    path = tmp_path / "unit-thresholds.yml"
    write_limits(path, given)
    assert load_limits(path) == given
    with pytest.raises(ConfigError, match="already exists"):
        write_limits(path, given)


def test_the_scores_csv_is_read_back_without_shape_columns(tmp_path: Path) -> None:
    rows = [ScoreRow("C-1.mkv", 50, 1, "full_pink", 50, "", 1.5, 70.0, 150.0, 120.0, 7.5),
            ScoreRow("C-1.mkv", 146, 4, "full", None, "unit size out of range",
                     None, None, None, None, None)]
    path = tmp_path / "scores.csv"
    write_scores(rows, path)
    assert read_scores(path) == [
        Scored("C-1.mkv", 50, "full_pink", "", 1.5, 70.0, 150.0, 120.0, 7.5),
        Scored("C-1.mkv", 146, "full", "unit size out of range")]
    assert from_score_rows(rows) == read_scores(path)


# --- what flags a pass ------------------------------------------------------------------


def test_a_shifted_unit_is_flagged_by_shift_and_the_combined_row_not_by_spread() -> None:
    verdict = judge(good(colour=(70.0, 156.0, 128.0)), limits())
    assert verdict == {"spread": "ok", "shift": "flag", "colour": "flag"}


def test_a_shift_exactly_at_the_limit_is_not_flagged() -> None:
    verdict = judge(good(colour=(70.0, 153.0, 124.0)), limits())
    assert verdict == {"spread": "ok", "shift": "ok", "colour": "ok"}


def test_a_spread_exactly_at_its_limit_is_not_flagged_and_above_it_is() -> None:
    assert judge(good(spread=5.0), limits())["spread"] == "ok"
    assert judge(good(spread=5.5), limits())["spread"] == "flag"


def test_a_faulted_pass_is_no_answer_for_every_method() -> None:
    assert judge(faulted(), limits(patch_score_max=7.0)) == {
        "spread": "none", "shift": "none", "patch": "none", "colour": "none"}


def test_the_patch_method_needs_its_limit_and_a_score() -> None:
    assert "patch" not in judge(good(patch=9.0), limits())
    with_limit = limits(patch_score_max=7.0)
    assert judge(good(patch=9.0), with_limit)["patch"] == "flag"
    assert judge(good(patch=7.0), with_limit)["patch"] == "ok"
    assert judge(good(), with_limit)["patch"] == "none"


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((3000.0, 0.8, 0.9), "ok"), ((2999.0, 0.8, 0.9), "flag"), ((9001.0, 0.8, 0.9), "flag"),
     ((5000.0, 0.79, 0.9), "flag"), ((5000.0, 0.8, 0.89), "flag")],
)
def test_shape_is_flagged_outside_the_range_or_under_a_minimum(
    shape: tuple[float, float, float], expected: str
) -> None:
    shaped = limits(area_px=(3000.0, 9000.0), axis_ratio_min=0.8, solidity_min=0.9)
    assert judge(good(shape=shape), shaped)["shape"] == expected


# --- the table --------------------------------------------------------------------------


def _file(**changes: object) -> LimitsFile:
    return LimitsFile("a" * 64, "b" * 64, {}, {"full_pink": limits(**changes)})


def test_good_units_flagged_with_the_upper_bound_and_no_answer() -> None:
    rows = ([good(video="E-1.mkv", frame=i, spread=9.0) for i in range(3)]
            + [faulted(frame=i) for i in range(3, 5)]
            + [good(video="E-1.mkv", frame=i) for i in range(5, 300)])
    text = render_report(_file(), rows, ())
    assert table_cells(text, "good units flagged") == ["3 of 300", "0 of 300", "3 of 300"]
    assert table_cells(text, "upper bound")[0] == "0.0256"
    assert table_cells(text, "good units, no answer") == ["2 of 300"] * 3


def test_defective_units_count_only_when_every_pass_is_flagged() -> None:
    rows, defects = [], []
    for unit in range(10):
        for turn in range(3):
            frame = unit * 10 + turn
            missed = unit == 0 and turn < 2
            rows.append(good(video="E-1.mkv", frame=frame, spread=1.0 if missed else 9.0))
            defects.append(Defect("E-1.mkv", frame, "patch", f"D{unit + 1:02d}"))
    text = render_report(_file(), rows, defects)
    assert table_cells(text, "defective passes flagged")[0] == "28 of 30"
    assert table_cells(text, "defective units caught")[0] == "9 of 10"
    assert table_cells(text, "lower bound")[0] == "0.6058"
    assert ("- D01 (patch), 3 passes: colour spread 1 of 3, colour shift 0 of 3, "
            "hand-written colour 1 of 3") in text.splitlines()


def test_defective_units_are_not_counted_among_the_good_ones() -> None:
    rows = [good(video="E-1.mkv", frame=i) for i in range(4)]
    rows += [good(video="E-1.mkv", frame=9, spread=9.0)]
    text = render_report(_file(), rows, [Defect("E-1.mkv", 9, "colour", "D01")])
    assert table_cells(text, "good units flagged")[0] == "0 of 4"


def test_with_no_defects_the_cell_reads_no_real_defective_unit() -> None:
    text = render_report(_file(), [good(video="E-1.mkv")], ())
    assert table_cells(text, "defective passes flagged") == ["no real defective unit"] * 3
    assert table_cells(text, "defective units caught") == ["-"] * 3


def test_without_shape_limits_the_table_has_no_shape_column() -> None:
    text = render_report(_file(), [good(video="E-1.mkv")], ())
    assert "shape" not in text


def test_shape_units_are_judged_by_shape_only() -> None:
    shaped = dict(area_px=(3000.0, 9000.0), axis_ratio_min=0.8, solidity_min=0.9)
    rows = [good(video="E-1.mkv", frame=i, shape=(2000.0, 0.9, 0.95)) for i in range(3)]
    rows += [good(video="E-1.mkv", frame=9, shape=(5000.0, 0.9, 0.95))]
    defects = [Defect("E-1.mkv", i, "shape", "S01") for i in range(3)]
    text = render_report(_file(**shaped), rows, defects)
    assert table_cells(text, "defective passes flagged") == [
        "no real defective unit", "no real defective unit", "no real defective unit", "3 of 3"]
    assert table_cells(text, "defective units caught")[3] == "1 of 1"
    assert table_cells(text, "good units flagged")[3] == "0 of 1"


def test_a_state_without_limits_gets_a_line_and_no_table() -> None:
    rows = [good(state="full_white", video="E-1.mkv", frame=i) for i in range(2)]
    text = render_report(_file(), rows, ())
    assert "full_white: no limits, no result (2 passes)" in text.splitlines()


def test_a_rescored_table_says_so_on_its_first_line() -> None:
    text = render_report(_file(), [good(video="E-1.mkv")], (), rescored_run=2)
    assert text.splitlines()[0] == "RESCORED (run 2)"
    assert not render_report(_file(), [good(video="E-1.mkv")], ()).startswith("RESCORED")
