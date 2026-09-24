"""The INDICATIVE report and its per-unit CSV (docs/14 §7, §9; FR-55, FR-56, ADR-0007).

Pure rendering: everything is computed before this module is called, and it writes nothing —
the command writes the two files, never overwriting one that exists. The report carries no
wall-clock time; two runs on the same set, bundle and config differ in nothing but the file
name (NFR-08). **INDICATIVE** is in the title, the E6 sentence is under it, the stop lines
are at the top, then the set, the operating table at 1%, 5% and 10%, which check caught
what, the unscored units and placement; the later sections (controls, pairs, changed units,
latency) are ticket 4's.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qs.config import LabelsConfig
from qs.eval.units import Aggregated, TargetResult


@dataclass(frozen=True, slots=True)
class RoleCount:
    """One role's units and frames; a test role also carries its units per class."""

    role: str
    units: int
    frames: int
    classes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Report:
    """Everything the report and the CSV render from; the command assembles it."""

    product: str
    geometry: str
    golden_name: str
    golden_hash: str
    model_id: str
    threshold_id: str
    config_hash: str
    invocation: str
    camera_device: str
    camera_is_phone: bool
    counts: tuple[RoleCount, ...]
    targets: tuple[TargetResult, ...]
    units: tuple[Aggregated, ...]
    placement_ok: bool
    placement_bad: tuple[tuple[str, str], ...]
    early_warning: str | None
    labels: LabelsConfig


def counts(rows: Sequence[Mapping[str, str]]) -> tuple[RoleCount, ...]:
    """Units and frames per role; the test role also by damage class (docs/14 §7)."""
    per_role: dict[str, set[str]] = {}
    frames: dict[str, int] = {}
    classes: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        role, unit = row["role"], row["unit"]
        per_role.setdefault(role, set()).add(unit)
        frames[role] = frames.get(role, 0) + 1
        if role == "test" and row["class"]:
            units = classes.setdefault(role, {}).setdefault(row["class"], set())
            units.add(unit)
    order = ("calib", "test", "control", "placement")
    return tuple(
        RoleCount(
            role,
            len(per_role[role]),
            frames[role],
            tuple(sorted((cls, len(units)) for cls, units in classes.get(role, {}).items())),
        )
        for role in order
        if role in per_role
    )


def _pct(k: int, n: int) -> str:
    return "—" if n == 0 else f"{k / n * 100:.1f}%"


def _threshold_line(target: TargetResult) -> str:
    choice = target.choice
    if choice.value is None:
        return (f"- not reachable: geometry alone flags {choice.g} of {choice.n} "
                f"calibration units, the target allows {choice.a} (A2, #78) — no recall "
                "and no false-alarm rate at this target")
    note = (f" (the target is below 1/n on {choice.n} calibration units: one unit is "
            f"{100.0 / choice.n:.1f}%)") if choice.below_one_over_n else ""
    return f"- threshold: {choice.value:.6f}{note}"


def _recall_line(target: TargetResult) -> str:
    parts = [
        f"{r.cls} {r.caught}/{r.total} = {_pct(r.caught, r.total)}, lower bound "
        f"{r.lower * 100:.1f}%"
        + (f" ({r.unscored} unscored)" if r.unscored else "")
        for r in target.recall
    ]
    return "- recall per unit — " + " · ".join(parts)


def _far_line(target: TargetResult) -> str:
    far = target.far_combined
    head = (f"on the {far.total} good test units"
            + (f" ({target.far_unscored} unscored left out)" if target.far_unscored else ""))
    if far.total == 0:
        return f"- false alarms {head} — none scored, the rate is undefined"
    parts = [
        f"combined {far.flagged}/{far.total} = {_pct(far.flagged, far.total)}, upper bound "
        f"{far.upper * 100:.1f}%",
        f"geometry-only {target.far_geometry.flagged}/{far.total}, upper bound "
        f"{target.far_geometry.upper * 100:.1f}%",
        f"appearance-only {target.far_appearance.flagged}/{far.total}, upper bound "
        f"{target.far_appearance.upper * 100:.1f}%",
    ]
    return f"- false alarms {head} — " + " · ".join(parts)


def _attribution(entries: Sequence[tuple[str, str]], which: str) -> str:
    names = [unit for unit, kind in entries if kind == which]
    return f"{which}: " + (", ".join(names) if names else "—")


def _caught_line(target: TargetResult) -> str:
    parts = [_attribution(target.caught, kind)
             for kind in ("appearance", "geometry", "both", "missed")]
    return "- caught — " + " · ".join(parts)


def _flagged_good_line(target: TargetResult) -> str:
    if not target.flagged_good:
        return "- good units flagged — none."
    return "- good units flagged — " + " · ".join(
        _attribution(target.flagged_good, kind)
        for kind in ("appearance", "geometry", "both")
    )


def _target_section(target: TargetResult) -> str:
    head = f"### far {target.target}% target"
    if target.choice.value is None:
        return f"{head}\n{_threshold_line(target)}"
    return "\n".join([
        head,
        _threshold_line(target),
        _recall_line(target),
        _far_line(target),
        _caught_line(target),
        _flagged_good_line(target),
    ])


def _camera_line(r: Report) -> str:
    if r.camera_is_phone:
        return (f"- source camera: the phone (`{r.camera_device}`) — what is learned from "
                "phone frames belongs to the phone; its numbers do not carry over to the "
                "webcam (ADR-0022)")
    return f"- source camera: the webcam (`{r.camera_device}`)"


def _unscored_section(r: Report) -> str:
    unscored = [u for u in r.units if u.unscored]
    if not unscored:
        return "- none. Every unit and state was scored."
    lines = [
        f"- {u.unit} {u.state}: НЕТ РЕШЕНИЯ ({', '.join(u.reasons)}) — missed for recall, "
        "left out of the false-alarm denominator"
        for u in unscored
    ]
    return "\n".join(lines)


def _counts_lines(r: Report) -> list[str]:
    lines: list[str] = []
    for count in r.counts:
        line = f"- {count.role}: {count.units} units, {count.frames} frames"
        if count.classes:
            line += " — " + ", ".join(f"{cls} {n}" for cls, n in count.classes)
        lines.append(line)
    return lines


def _placement_section(r: Report) -> str:
    if r.placement_ok:
        return "- PASS — every placement frame came back НЕТ РЕШЕНИЯ(placement)."
    lines = ["- FAIL — a placement frame must come back НЕТ РЕШЕНИЯ(placement):"]
    lines += [f"  - {file} came back {got}." for file, got in r.placement_bad]
    return "\n".join(lines)


def markdown(r: Report) -> str:
    """The report's text, in docs/14 §7's order, with no wall-clock time in it."""
    stop = [f"- {r.early_warning}"] if r.early_warning else ["- none."]
    sections = [
        f"# INDICATIVE — {r.product}, {r.geometry}, golden {r.golden_name}",
        "",
        "This report is **INDICATIVE**: this measures моти we broke on purpose; it is not "
        "Gate A and not a client claim (ADR-0007, E6).",
        "",
        f"- invocation: `{r.invocation}`",
        f"- golden sha256: `{r.golden_hash}`",
        f"- model_id: `{r.model_id}`",
        f"- threshold_id: `{r.threshold_id}`",
        f"- config_hash: `{r.config_hash}`",
        _camera_line(r),
        "",
        "## Stop lines",
        *stop,
        "",
        "## The set",
        *_counts_lines(r),
        "- the roles are disjoint: every unit is in exactly one role — verified before "
        "scoring (E1)",
        "",
        "## Operating table",
        *[_target_section(t) for t in r.targets],
        "",
        "## Unscored units",
        _unscored_section(r),
        "",
        "## Placement",
        _placement_section(r),
        "",
    ]
    return "\n".join(sections) + "\n"


CSV_COLUMNS = ("golden_sha256", "model_id", "threshold_id", "unit", "role", "class",
               "state", "rotations_scored", "score", "geometry_failed", "label_1",
               "label_5", "label_10")


def csv_text(r: Report) -> str:
    """One row per unit and state, in the aggregation's order; no placement rows (the
    placement frames are checked, never aggregated). The score is the float as written —
    the changed-units comparison of ticket 4 reads it back exactly."""
    targets = {t.target: t for t in r.targets}

    def label(unit: Aggregated, target: int) -> str:
        result = targets[target]
        flagged = unit.flagged(result.threshold)
        if flagged is None:
            return ""
        return r.labels.defect if flagged else r.labels.pass_

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for unit in r.units:
        writer.writerow((
            r.golden_hash,
            r.model_id,
            r.threshold_id,
            unit.unit,
            unit.role,
            unit.cls,
            unit.state,
            unit.scored,
            "" if unit.score is None else repr(unit.score),
            "" if unit.geometry_failed is None else ("true" if unit.geometry_failed
                                                     else "false"),
            label(unit, 1),
            label(unit, 5),
            label(unit, 10),
        ))
    return buffer.getvalue()
