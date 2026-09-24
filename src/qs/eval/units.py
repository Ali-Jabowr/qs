"""Unit aggregation (docs/14 §4 steps 3–7, §9; E2, A2/#78).

The golden set's manifest rows and the `FrameResult`s of `score_frames` are aggregated per
unit and state: a frame that came back НЕТ РЕШЕНИЯ is left out; a unit's appearance score is
the maximum over its scored frames; it is БРАК at a target when any scored frame failed
geometry or scored above that target's threshold. A unit and state with no scored frame is
**unscored**: listed with its reasons, missed for recall, left out of the false-alarm
denominator with its count shown — every choice errs against the flattering number.

`target_result` counts one target's operating row: the threshold from `choose_threshold` on
the golden calib units, recall per damage class with its lower bound, the false-alarm rate
on the test units' good state with its upper bound, and which check caught each defective
unit and flagged each good one. A target whose geometry alone already flags more calib
units than it allows is not reachable (A2, #78): no threshold, no recall row.

The later sections (ticket #83): `control_passes` gives each control unit's score and mean
geometry measures per pass; `pairs` each test unit's damaged − good score (ADR-0014);
`drift` the E9 line by assumption A3 (#79) — shift = the range of a control's unit score
over its scored passes, the largest of the controls; gap = the median paired difference of
the dent units scored in both states; drift when shift ≥ gap, gap ≤ 0, or either cannot be
measured. `latency` gives p50/p95/p99 of the frames that reached the Inspector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from qs.contracts import Label
from qs.eval.scoring import FrameResult
from qs.eval.stats import lower_bound, upper_bound
from qs.eval.threshold import ThresholdChoice, UnitScore, choose_threshold

_STATE_ORDER = {"good": 0, "dmg": 1}


@dataclass(frozen=True, slots=True)
class Aggregated:
    """One unit in one state, aggregated over its frames (docs/14 §4 step 3)."""

    unit: str
    role: str
    state: str
    cls: str
    scored: int
    score: float | None
    geometry_failed: bool | None
    reasons: tuple[str, ...]

    @property
    def unscored(self) -> bool:
        """No scored frame: the unit and state is missed for recall and out of the FAR."""
        return self.score is None

    def flagged(self, threshold: float | None) -> bool | None:
        """БРАК at `threshold`: a scored frame failed geometry or scored above it.

        `None` when the unit is unscored or the target is not reachable — no label.
        """
        if self.score is None or threshold is None:
            return None
        return bool(self.geometry_failed) or self.score > threshold

    def caught_by(self, threshold: float | None) -> tuple[bool, bool]:
        """(geometry, appearance): which of the two checks flagged this unit and state."""
        if self.score is None or threshold is None:
            return (False, False)
        return (bool(self.geometry_failed), self.score > threshold)


def aggregate(
    rows: Sequence[Mapping[str, str]], results: Sequence[FrameResult]
) -> tuple[Aggregated, ...]:
    """Per unit and state, in unit order with `good` before `dmg`; placement frames are
    not aggregated (they are checked, never scored into a rate)."""
    if len(rows) != len(results):
        raise ValueError(
            f"{len(rows)} manifest rows but {len(results)} results — the set and the "
            "scoring disagree"
        )
    per: dict[tuple[str, str, str, str], list[FrameResult]] = {}
    for row, result in zip(rows, results, strict=True):
        if row["file"] != result.file.name:
            raise ValueError(
                f"{row['file']} was scored as {result.file.name} — the rows and the "
                "results disagree"
            )
        if row["role"] == "placement":
            continue  # placement frames are checked, never aggregated into a rate
        per.setdefault((row["unit"], row["role"], row["state"], row["class"]), []).append(
            result
        )
    def key(item: tuple[tuple[str, str, str, str], list[FrameResult]]) -> tuple[str, int]:
        (unit, _role, state, _cls), _ = item
        return (unit, _STATE_ORDER.get(state, len(_STATE_ORDER)))
    return tuple(
        _one(unit, role, state, cls, frames)
        for (unit, role, state, cls), frames in sorted(per.items(), key=key)
    )


def _one(
    unit: str, role: str, state: str, cls: str, frames: list[FrameResult]
) -> Aggregated:
    scored = [r for r in frames if r.label is not Label.NO_VERDICT]
    reasons = tuple(
        dict.fromkeys(r.reason or "" for r in frames if r.label is Label.NO_VERDICT)
    )
    if not scored:
        return Aggregated(unit, role, state, cls, 0, None, None, reasons)
    scores = [r.score for r in scored]
    if any(s is None for s in scores) or any(r.geometry is None for r in scored):
        raise ValueError(
            f"{unit} {state}: a scored frame without a score or a geometry result — "
            "a completed inspection carries both checks (C8)"
        )
    return Aggregated(
        unit,
        role,
        state,
        cls,
        len(scored),
        max(s for s in scores if s is not None),
        any(r.geometry is not None and not r.geometry.passed for r in scored),
        (),
    )


@dataclass(frozen=True, slots=True)
class ClassRecall:
    """One damage class at one target: caught of total, with the lower bound.

    `unscored` units are missed for recall — they sit in `total` and never in `caught`.
    """

    cls: str
    caught: int
    total: int
    unscored: int
    lower: float


@dataclass(frozen=True, slots=True)
class FarRate:
    """One false-alarm row: k flagged of n good test units, with the upper bound."""

    flagged: int
    total: int
    upper: float


@dataclass(frozen=True, slots=True)
class TargetResult:
    """One target's operating row (docs/14 §4 steps 5–7, §9).

    `choice.value` is None when geometry alone already flags more calib units than the
    target allows (A2, #78): the row is not reachable, and nothing is recalled or counted
    at it. `caught` names each defective unit and the check that caught it — or `missed`;
    `flagged_good` names each flagged good unit and the check that flagged it.
    """

    target: int
    choice: ThresholdChoice
    recall: tuple[ClassRecall, ...]
    far_combined: FarRate
    far_geometry: FarRate
    far_appearance: FarRate
    far_unscored: int
    caught: tuple[tuple[str, str], ...]
    flagged_good: tuple[tuple[str, str], ...]

    @property
    def threshold(self) -> float | None:
        return self.choice.value

    @property
    def reachable(self) -> bool:
        return self.choice.reachable


def target_result(target: int, units: Sequence[Aggregated]) -> TargetResult:
    """One target's operating row over the aggregated units (docs/14 §9)."""
    calib = [u for u in units if u.role == "calib"]
    if any(u.unscored for u in calib):
        missed = ", ".join(u.unit for u in calib if u.unscored)
        raise ValueError(f"calibration units without a verdict: {missed}")
    choice = choose_threshold(
        [UnitScore(u.unit, u.score or 0.0, bool(u.geometry_failed)) for u in calib], target
    )
    threshold = choice.value
    classes = sorted({u.cls for u in units if u.role == "test" and u.state == "dmg"})
    recall = tuple(
        _recall(cls, [u for u in units if u.role == "test" and u.state == "dmg"
                      and u.cls == cls], threshold)
        for cls in classes
    )
    good = [u for u in units if u.role == "test" and u.state == "good"]
    scored_good = [u for u in good if not u.unscored]
    n = len(scored_good)
    geometry_only = [u.unit for u in scored_good if u.caught_by(threshold) == (True, False)]
    appearance_only = [u.unit for u in scored_good if u.caught_by(threshold) == (False, True)]
    combined = [u.unit for u in scored_good if u.caught_by(threshold) != (False, False)]
    caught = tuple(
        (u.unit, _which(u, threshold))
        for u in sorted((u for u in units if u.role == "test" and u.state == "dmg"),
                        key=lambda u: u.unit)
    )
    flagged_good = tuple(
        (u.unit, _which(u, threshold)) for u in sorted(scored_good, key=lambda u: u.unit)
        if u.caught_by(threshold) != (False, False)
    )
    return TargetResult(
        target=target,
        choice=choice,
        recall=recall,
        far_combined=FarRate(len(combined), n, upper_bound(n, len(combined))),
        far_geometry=FarRate(len(geometry_only), n, upper_bound(n, len(geometry_only))),
        far_appearance=FarRate(len(appearance_only), n, upper_bound(n, len(appearance_only))),
        far_unscored=len(good) - n,
        caught=caught,
        flagged_good=flagged_good,
    )


def _recall(cls: str, units: list[Aggregated], threshold: float | None) -> ClassRecall:
    caught = sum(1 for u in units if u.flagged(threshold) is True)
    unscored = sum(1 for u in units if u.unscored)
    return ClassRecall(cls, caught, len(units), unscored, lower_bound(len(units), caught))


def _which(unit: Aggregated, threshold: float | None) -> str:
    geometry, appearance = unit.caught_by(threshold)
    if geometry and appearance:
        return "both"
    if geometry:
        return "geometry"
    if appearance:
        return "appearance"
    return "missed"


def early_warning(targets: Sequence[TargetResult]) -> str | None:
    """The MVP's one stop signal: dent recall under 70% at the 5% target (docs/06 §4)."""
    at5 = next((t for t in targets if t.target == 5), None)
    dent = next((r for r in at5.recall if r.cls == "dent"), None) if at5 else None
    if dent is None or dent.total == 0 or dent.caught / dent.total >= 0.70:
        return None
    return (
        f"EARLY WARNING: dent recall at the 5% target is {dent.caught} of {dent.total} "
        f"({dent.caught / dent.total * 100:.1f}%), below 70% — fix the lighting before "
        "building more of the demo (docs/06 §4)"
    )


# --- controls, pairs, drift and latency (ticket #83) ----------------------------------------


@dataclass(frozen=True, slots=True)
class ControlPass:
    """One control unit at one pass, aggregated over its rotations (E9, docs/13 §6).

    `score` is the maximum over the scored rotations and `measures` the mean of each
    geometry measure over them, in the contract's order; a pass whose rotations all came
    back НЕТ РЕШЕНИЯ has neither, and `reasons` says why.
    """

    unit: str
    pass_: int
    scored: int
    score: float | None
    geometry_failed: bool | None
    measures: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]


def control_passes(
    rows: Sequence[Mapping[str, str]], results: Sequence[FrameResult]
) -> tuple[ControlPass, ...]:
    """Each control unit's passes, in unit and pass order."""
    per: dict[tuple[str, int], list[FrameResult]] = {}
    for row, result in zip(rows, results, strict=True):
        if row["role"] != "control":
            continue
        if result.name.pass_ is None:
            raise ValueError(f"{row['file']}: a control frame without its pass")
        per.setdefault((row["unit"], result.name.pass_), []).append(result)
    return tuple(_control(unit, pass_, frames)
                 for (unit, pass_), frames in sorted(per.items()))


def _control(unit: str, pass_: int, frames: list[FrameResult]) -> ControlPass:
    scored = [r for r in frames if r.label is not Label.NO_VERDICT]
    reasons = tuple(
        dict.fromkeys(r.reason or "" for r in frames if r.label is Label.NO_VERDICT)
    )
    if not scored:
        return ControlPass(unit, pass_, 0, None, None, (), reasons)
    if any(r.score is None or r.geometry is None for r in scored):
        raise ValueError(
            f"{unit} pass{pass_}: a scored frame without a score or a geometry result — "
            "a completed inspection carries both checks (C8)"
        )
    sums: dict[str, list[float]] = {}
    for r in scored:
        assert r.geometry is not None
        for name, value in r.geometry.measures:
            sums.setdefault(name, []).append(value)
    return ControlPass(
        unit,
        pass_,
        len(scored),
        max(r.score for r in scored if r.score is not None),
        any(r.geometry is not None and not r.geometry.passed for r in scored),
        tuple((name, sum(values) / len(values)) for name, values in sums.items()),
        reasons,
    )


@dataclass(frozen=True, slots=True)
class Pair:
    """One test unit's two states (ADR-0014); `None` where a state is unscored."""

    unit: str
    cls: str
    good: float | None
    damaged: float | None

    @property
    def difference(self) -> float | None:
        """Damaged unit score − good unit score; `None` when either state is unscored."""
        if self.good is None or self.damaged is None:
            return None
        return self.damaged - self.good


def pairs(units: Sequence[Aggregated]) -> tuple[Pair, ...]:
    """One pair per test unit, in unit order; the class is the damaged state's."""
    good: dict[str, float | None] = {}
    damaged: dict[str, tuple[str, float | None]] = {}
    for u in units:
        if u.role != "test":
            continue
        if u.state == "good":
            good[u.unit] = u.score
        elif u.state == "dmg":
            damaged[u.unit] = (u.cls, u.score)
    return tuple(
        Pair(unit, damaged.get(unit, ("", None))[0], good.get(unit),
             damaged.get(unit, ("", None))[1])
        for unit in sorted(set(good) | set(damaged))
    )


def class_medians(
    paired: Sequence[Pair],
) -> tuple[tuple[str, float | None, int], ...]:
    """Per class: the median paired difference over the units scored in both states, and
    how many there were; `None` when none was."""
    classes = sorted({p.cls for p in paired})
    out: list[tuple[str, float | None, int]] = []
    for cls in classes:
        diffs = [d for p in paired if p.cls == cls and (d := p.difference) is not None]
        out.append((cls, median(diffs) if diffs else None, len(diffs)))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ControlShift:
    """One control's shift: the range of its unit score over its scored passes; `None`
    when fewer than two passes were scored (not measurable)."""

    unit: str
    passes_scored: int
    shift: float | None


@dataclass(frozen=True, slots=True)
class Drift:
    """The E9 line by assumption A3 (#79)."""

    controls: tuple[ControlShift, ...]
    shift: float | None
    shift_unit: str | None
    gap: float | None
    gap_units: int
    drifted: bool
    why: tuple[str, ...]

    def line(self) -> str | None:
        """The stop line, or None when the session did not drift."""
        if not self.drifted:
            return None
        return ("the session drifted: the numbers mean nothing (E9) — "
                + "; ".join(self.why))


DRIFT_CLASS = "dent"


def drift(controls: Sequence[ControlPass], paired: Sequence[Pair]) -> Drift:
    """Shift against gap (A3, #79). Every case that cannot be measured drifts — a session
    whose stability is unknown is never read as stable."""
    per: dict[str, list[float]] = {}
    for c in controls:
        per.setdefault(c.unit, [])
        if c.score is not None:
            per[c.unit].append(c.score)
    shifts = tuple(
        ControlShift(unit, len(scores), max(scores) - min(scores) if len(scores) >= 2 else None)
        for unit, scores in sorted(per.items())
    )
    measured = [s for s in shifts if s.shift is not None]
    largest = max(measured, key=lambda s: s.shift or 0.0) if measured else None
    diffs = [d for p in paired if p.cls == DRIFT_CLASS and (d := p.difference) is not None]
    gap = median(diffs) if diffs else None
    why: list[str] = []
    if not shifts:
        why.append("no control unit was scored — the shift is not measurable")
    for s in shifts:
        if s.shift is None:
            why.append(f"{s.unit}'s shift is not measurable: {s.passes_scored} pass(es) "
                       "scored, two are needed")
    if gap is None:
        why.append(f"the gap is not measurable: no {DRIFT_CLASS} unit scored in both states")
    elif gap <= 0:
        why.append(f"the gap {gap:.6f} is not above 0: damaged {DRIFT_CLASS} units do not "
                   "score above their good state")
    if largest is not None and largest.shift is not None and gap is not None and gap > 0 \
            and largest.shift >= gap:
        why.append(f"the shift {largest.shift:.6f} ({largest.unit}) is not below the gap "
                   f"{gap:.6f}")
    return Drift(
        controls=shifts,
        shift=None if largest is None else largest.shift,
        shift_unit=None if largest is None else largest.unit,
        gap=gap,
        gap_units=len(diffs),
        drifted=bool(why),
        why=tuple(why),
    )


@dataclass(frozen=True, slots=True)
class Latency:
    """p50/p95/p99 over the frames that reached the Inspector; `None` when none did."""

    count: int
    inspect_ms: tuple[float, float, float] | None
    total_ms: tuple[float, float, float] | None


_REACHED = ("placement", "inference")  # the НЕТ РЕШЕНИЯ raised inside the Inspector


def reached_inspector(result: FrameResult) -> bool:
    """Whether the Pipeline called the Inspector on this frame: a verdict, or a
    НЕТ РЕШЕНИЯ raised inside it (placement, inference) — not one decided before it."""
    return result.label is not Label.NO_VERDICT or result.reason in _REACHED


def _percentiles(samples: Sequence[float]) -> tuple[float, float, float]:
    """p50, p95, p99 — numpy's default (linear) method, as `qs latency` computes them."""
    values = np.asarray(samples, dtype=np.float64)
    return (float(np.percentile(values, 50)), float(np.percentile(values, 95)),
            float(np.percentile(values, 99)))


def latency(results: Sequence[FrameResult]) -> Latency:
    """The run's timings — measured, not reproducible; nothing is gated on them."""
    reached = [r for r in results if reached_inspector(r)]
    if not reached:
        return Latency(0, None, None)
    return Latency(len(reached), _percentiles([r.inspect_ms for r in reached]),
                   _percentiles([r.total_ms for r in reached]))
