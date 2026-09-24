"""Unit aggregation on hand-built results (docs/14 §4 step 3, §9; E2, A2/#78).

Every `FrameResult` here is built by hand — no Pipeline, no frames — so each rule of the
aggregation has a hand-countable expected value: one bad rotation catches the unit; a unit
whose frames all came back НЕТ РЕШЕНИЯ is unscored, missed for recall and out of the
false-alarm denominator with the count shown; one scored frame among three carries the
unit's score; a target geometry alone cannot meet is not reachable (A2, #78).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qs.contracts import CheckResult, Label
from qs.eval.scoring import FrameResult
from qs.eval.units import Aggregated, aggregate, early_warning, target_result
from qs.names import parse_name

THRESHOLD = 1.0
GOOD_GEOMETRY = CheckResult("geometry", True, None, None,
                            (("area_px", 2000.0), ("axis_ratio", 1.0), ("solidity", 1.0)))
BAD_GEOMETRY = CheckResult("geometry", False, None, None,
                           (("area_px", 2000.0), ("axis_ratio", 0.5), ("solidity", 1.0)),
                           "axis_ratio<min")


def _result(file: str, *, label: Label = Label.PASS, score: float | None = 0.0,
            geometry: CheckResult | None = GOOD_GEOMETRY,
            reason: str | None = None) -> FrameResult:
    return FrameResult(Path(file), parse_name(file), label, reason, geometry, score,
                       score, 1.0, 2.0)


def one(file: str, **kwargs: str | float | None) -> list[FrameResult]:
    return [_result(file, **kwargs)]  # type: ignore[arg-type]


def _row(file: str, unit: str, role: str, state: str, cls: str = "") -> dict[str, str]:
    name = parse_name(file)
    return {"file": file, "unit": unit, "role": role, "state": state, "class": cls,
            "geometry": name.geometry, "rotation": f"r{name.rotation}",
            "pass": "" if name.pass_ is None else f"pass{name.pass_}", "sha256": "0" * 64}


def _set(*entries: tuple[str, str, str, str, list[FrameResult]]) -> tuple[list[dict[str, str]],
                                                                          list[FrameResult]]:
    """(unit, role, state, class, results) entries → the manifest rows and the results."""
    rows: list[dict[str, str]] = []
    results: list[FrameResult] = []
    for unit, role, state, cls, frames in entries:
        for frame in frames:
            rows.append(_row(frame.file.name, unit, role, state, cls))
            results.append(frame)
    return rows, results


def rotations(unit: str, state: str, **kwargs: str | float | None) -> list[FrameResult]:
    return [_result(f"{unit}_{state}_G2_r{r}.png", **kwargs) for r in range(3)]  # type: ignore[arg-type]


def calib_units(scores: float, count: int, *, failed: int = 0) -> tuple[
        list[dict[str, str]], list[FrameResult]]:
    """`count` calibration units, all scoring `scores`, the last `failed` geometry-failed."""
    return _set(*(
        (f"u{100 + n:03d}", "calib", "good", "",
         rotations(f"u{100 + n:03d}", "good", score=scores,
                   geometry=BAD_GEOMETRY if n > count - failed else GOOD_GEOMETRY))
        for n in range(1, count + 1)
    ))


# --- the hand-built rules --------------------------------------------------------------------


def test_one_bad_rotation_of_three_catches_the_unit() -> None:
    """A unit whose middle rotation scored above the threshold is БРАК at it (E2)."""
    frames = [*one("u001_dmg_G2_r0.png", score=0.5),
              *one("u001_dmg_G2_r1.png", score=5.0),
              *one("u001_dmg_G2_r2.png", score=0.5)]
    rows, results = _set(("u001", "test", "dmg", "dent", frames))
    [unit] = aggregate(rows, results)
    assert unit.score == 5.0 and unit.scored == 3 and unit.geometry_failed is False
    assert unit.flagged(THRESHOLD) is True
    assert unit.caught_by(THRESHOLD) == (False, True)


def test_one_geometry_failure_of_three_catches_the_unit() -> None:
    frames = [*one("u002_dmg_G2_r0.png", score=0.5),
              *one("u002_dmg_G2_r1.png", score=0.5, geometry=BAD_GEOMETRY),
              *one("u002_dmg_G2_r2.png", score=0.5)]
    rows, results = _set(("u002", "test", "dmg", "deformed", frames))
    [unit] = aggregate(rows, results)
    assert unit.score == 0.5 and unit.geometry_failed is True
    assert unit.caught_by(THRESHOLD) == (True, False)


def test_a_unit_with_no_scored_rotation_is_unscored() -> None:
    """All three rotations НЕТ РЕШЕНИЯ: unscored, missed for recall, out of the FAR."""
    rows, results = _set(
        ("u003", "test", "dmg", "dent",
         rotations("u003", "dmg", label=Label.NO_VERDICT, score=None, geometry=None,
                   reason="lighting")),
        ("u004", "test", "dmg", "dent", rotations("u004", "dmg", score=5.0)),
        ("u005", "test", "good", "",
         rotations("u005", "good", label=Label.NO_VERDICT, score=None, geometry=None,
                   reason="capture")),
        ("u006", "test", "good", "", rotations("u006", "good", score=0.0)),
    )
    units = aggregate(rows, results)
    calib_rows, calib_results = calib_units(0.5, 20)
    target = target_result(5, (*units, *aggregate(calib_rows, calib_results)))
    assert target.choice.value == 0.5  # a = 1 of 20: the highest calib score
    [dent] = [r for r in target.recall if r.cls == "dent"]
    assert (dent.caught, dent.total, dent.unscored) == (1, 2, 1)  # u003 missed for recall
    assert target.far_combined.total == 1  # u005 is out of the denominator
    assert target.far_unscored == 1  # the count is shown
    unscored = [u for u in units if u.unscored]
    assert [u.unit for u in unscored] == ["u003", "u005"]
    assert [u.reasons for u in unscored] == [("lighting",), ("capture",)]


def test_two_no_verdict_rotations_take_the_scored_one() -> None:
    frames = [*one("u007_dmg_G2_r0.png", label=Label.NO_VERDICT, score=None,
                       geometry=None, reason="busy"),
              *one("u007_dmg_G2_r1.png", label=Label.NO_VERDICT, score=None,
                       geometry=None, reason="busy"),
              *one("u007_dmg_G2_r2.png", score=3.0)]
    rows, results = _set(("u007", "test", "dmg", "tear", frames))
    [unit] = aggregate(rows, results)
    assert (unit.scored, unit.score) == (1, 3.0)
    assert unit.flagged(THRESHOLD) is True
    assert unit.reasons == ()


def test_the_unscored_reasons_are_the_frames_reasons() -> None:
    frames = [*one("u008_good_G2_r0.png", label=Label.NO_VERDICT, score=None,
                       geometry=None, reason="lighting"),
              *one("u008_good_G2_r1.png", label=Label.NO_VERDICT, score=None,
                       geometry=None, reason="capture"),
              *one("u008_good_G2_r2.png", label=Label.NO_VERDICT, score=None,
                       geometry=None, reason="lighting")]
    rows, results = _set(("u008", "test", "good", "", frames))
    [unit] = aggregate(rows, results)
    assert unit.reasons == ("lighting", "capture")


def test_rows_and_results_must_agree() -> None:
    rows, results = _set(("u001", "test", "dmg", "dent", rotations("u001", "dmg")))
    with pytest.raises(ValueError, match="disagree"):
        aggregate(rows, results[::-1])


# --- the operating row -----------------------------------------------------------------------


def test_a_not_reachable_target_has_no_threshold_and_no_labels() -> None:
    """A2/#78: geometry alone flags more calib units than a 1% target allows on 20."""
    rows, results = _set(
        ("u010", "test", "dmg", "dent", rotations("u010", "dmg", score=5.0)),
        ("u011", "test", "good", "", rotations("u011", "good", score=0.0)),
    )
    calib_rows, calib_results = calib_units(0.5, 20, failed=1)
    target = target_result(1, (*aggregate(rows, results),
                               *aggregate(calib_rows, calib_results)))
    assert target.choice.value is None and not target.reachable
    assert target.threshold is None
    assert target.caught == (("u010", "missed"),)
    assert target.flagged_good == ()
    for unit in aggregate(rows, results):
        assert unit.flagged(None) is None


def test_the_operating_row_counts_recall_far_and_attribution() -> None:
    """One dent caught by appearance, one deformed caught by geometry, one missed, one
    good unit flagged by both — all against 20 clean calibration units."""
    rows, results = _set(
        ("u010", "test", "dmg", "dent", rotations("u010", "dmg", score=5.0)),
        ("u011", "test", "dmg", "dent",
         rotations("u011", "dmg", score=0.5, geometry=BAD_GEOMETRY)),
        ("u012", "test", "dmg", "deformed",
         rotations("u012", "dmg", score=0.5, geometry=BAD_GEOMETRY)),
        ("u013", "test", "dmg", "tear", rotations("u013", "dmg", score=0.5)),
        ("u014", "test", "good", "",
         rotations("u014", "good", score=5.0, geometry=BAD_GEOMETRY)),
        ("u015", "test", "good", "", rotations("u015", "good", score=0.0)),
    )
    calib_rows, calib_results = calib_units(0.5, 20)
    target = target_result(5, (*aggregate(rows, results),
                               *aggregate(calib_rows, calib_results)))
    assert (target.choice.a, target.choice.g, target.choice.k) == (1, 0, 0)
    recall = {r.cls: r for r in target.recall}
    assert (recall["dent"].caught, recall["dent"].total) == (2, 2)
    assert (recall["deformed"].caught, recall["deformed"].total) == (1, 1)
    assert (recall["tear"].caught, recall["tear"].total) == (0, 1)  # u013 missed
    assert target.caught == (("u010", "appearance"), ("u011", "geometry"),
                             ("u012", "geometry"), ("u013", "missed"))
    assert target.far_combined.flagged == 1 and target.far_combined.total == 2
    assert target.far_geometry.flagged == 0 and target.far_appearance.flagged == 0
    assert target.flagged_good == (("u014", "both"),)
    assert target.far_unscored == 0


def test_the_early_warning_fires_below_70_and_not_at_it() -> None:
    def dents(caught: int) -> list[Aggregated]:
        rows, results = _set(*(
            (f"u0{n:02d}", "test", "dmg", "dent",
             rotations(f"u0{n:02d}", "dmg",
                       score=5.0 if n <= caught else 0.0))
            for n in range(1, 11)
        ))
        return list(aggregate(rows, results))

    calib_rows, calib_results = calib_units(0.5, 10)
    calib = list(aggregate(calib_rows, calib_results))
    below = target_result(5, (*dents(6), *calib))
    at = target_result(5, (*dents(7), *calib))
    assert early_warning((below,)) is not None
    assert "6 of 10 (60.0%)" in (early_warning((below,)) or "")
    assert early_warning((at,)) is None  # 7/10 is 70%, and the line is below 70%
