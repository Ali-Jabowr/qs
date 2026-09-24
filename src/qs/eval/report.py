"""The INDICATIVE report and its per-unit CSV (docs/14 §7, §9; FR-55, FR-56, ADR-0007).

Pure rendering: everything is computed before this module is called, and it writes nothing —
the command writes the two files, never overwriting one that exists. The report carries no
wall-clock time; two runs on the same set, bundle and config differ in nothing but the file
name (NFR-08). **INDICATIVE** is in the title, the E6 sentence is under it, the stop lines
(drift, early warning) are at the top, then the set, the operating table at 1%, 5% and 10%,
which check caught what, the unscored units, placement, the controls with the drift line
(E9, A3/#79), the paired differences (ADR-0014), the units changed since the previous run on
the same golden set, and latency last, in a section titled *measured, not reproducible* —
the only section two runs may differ in.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qs.config import LabelsConfig
from qs.eval.units import (
    Aggregated,
    ControlPass,
    Drift,
    Latency,
    Pair,
    TargetResult,
    class_medians,
)

TARGETS = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class Change:
    """One unit and state whose label at one target differs from the previous run's."""

    unit: str
    state: str
    target: int
    old_label: str
    new_label: str
    old_score: str
    new_score: str


@dataclass(frozen=True, slots=True)
class Changes:
    """The regression list (docs/14 §7): `previous` is None when no earlier CSV in the
    report directory carries this golden hash."""

    previous: str | None
    model_id: str
    threshold_id: str
    changes: tuple[Change, ...]


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
    controls: tuple[ControlPass, ...] = ()
    drift: Drift | None = None
    pairs: tuple[Pair, ...] = ()
    changes: Changes | None = None
    latency: Latency | None = None
    backend: str = ""
    latency_report: str | None = None
    latency_command: str = ""


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


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


_MEASURE_FORMAT = (("area_px", ".1f"), ("diameter_px", ".2f"), ("axis_ratio", ".4f"),
                   ("solidity", ".4f"))


def _controls_section(r: Report) -> str:
    if not r.controls:
        return "- no control unit in the set."
    head = ("| control | pass | rotations scored | score | geometry | "
            + " | ".join(name for name, _ in _MEASURE_FORMAT) + " |")
    lines = [head, "|" + "---|" * (5 + len(_MEASURE_FORMAT))]
    for c in r.controls:
        if c.score is None:
            cells = [f"{r.labels.no_verdict} ({', '.join(c.reasons)})", "—"]
            cells += ["—"] * len(_MEASURE_FORMAT)
        else:
            measures = dict(c.measures)
            cells = [_score(c.score), "failed" if c.geometry_failed else "passed"]
            cells += [format(measures[name], spec) if name in measures else "—"
                      for name, spec in _MEASURE_FORMAT]
        lines.append(f"| {c.unit} | pass{c.pass_} | {c.scored} | " + " | ".join(cells) + " |")
    if r.drift is not None:
        lines.append("")
        lines += [
            f"- {s.unit}: shift {_score(s.shift)} over {s.passes_scored} scored passes"
            if s.shift is not None else
            f"- {s.unit}: shift not measurable — {s.passes_scored} scored pass(es)"
            for s in r.drift.controls
        ]
        lines.append(
            f"- gap: the median paired difference of the dent units, {_score(r.drift.gap)} "
            f"over {r.drift.gap_units} units"
        )
        line = r.drift.line()
        lines.append(f"- drift: {line}" if line is not None else
                     f"- drift: none — the largest shift {_score(r.drift.shift)} "
                     f"({r.drift.shift_unit}) is below the gap {_score(r.drift.gap)} (E9)")
    return "\n".join(lines)


def _pairs_section(r: Report) -> str:
    if not r.pairs:
        return "- no test unit in the set."
    lines = ["| unit | class | good | damaged | damaged − good |", "|---|---|---|---|---|"]
    lines += [f"| {p.unit} | {p.cls} | {_score(p.good)} | {_score(p.damaged)} | "
              f"{_score(p.difference)} |" for p in r.pairs]
    lines.append("")
    lines.append("- median per class, over the units scored in both states — " + " · ".join(
        f"{cls} {_score(m)} ({n} units)" for cls, m, n in class_medians(r.pairs)))
    return "\n".join(lines)


def _changes_section(r: Report) -> str:
    c = r.changes
    if c is None or c.previous is None:
        return "- no previous run on this golden set."
    since = f"`{c.previous}` (model_id `{c.model_id}`, threshold_id `{c.threshold_id}`)"
    if not c.changes:
        return f"- no unit changed since {since}."
    lines = [f"- since {since}:"]
    lines += [
        f"  - {x.unit} {x.state} at {x.target}%: {x.old_label or '—'} → "
        f"{x.new_label or '—'}, score {x.old_score or '—'} → {x.new_score or '—'}"
        for x in c.changes
    ]
    return "\n".join(lines)


def _latency_section(r: Report) -> str:
    lat = r.latency
    if lat is None or lat.inspect_ms is None or lat.total_ms is None:
        lines = ["- no frame reached the Inspector: nothing was timed."]
    else:
        lines = [
            f"- backend `{r.backend}`, {lat.count} frames that reached the Inspector",
            "",
            "| ms | p50 | p95 | p99 |",
            "|---|---|---|---|",
            "| inspect_ms | " + " | ".join(f"{v:.1f}" for v in lat.inspect_ms) + " |",
            "| total_ms | " + " | ".join(f"{v:.1f}" for v in lat.total_ms) + " |",
            "",
        ]
    if r.latency_report is not None:
        lines.append(f"- Gate B proxy (ADR-0017): `{r.latency_report}`")
    else:
        lines.append(f"- Gate B proxy (ADR-0017): not measured yet — `{r.latency_command}`")
    lines.append("- nothing is gated here.")
    return "\n".join(lines)


def markdown(r: Report) -> str:
    """The report's text, in docs/14 §7's order, with no wall-clock time in it."""
    drift_line = r.drift.line() if r.drift is not None else None
    stop = [f"- {line}" for line in (drift_line, r.early_warning) if line is not None]
    stop = stop or ["- none."]
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
        "## Controls (E9)",
        _controls_section(r),
        "",
        "## Pairs (ADR-0014)",
        _pairs_section(r),
        "",
        "## Changed units",
        _changes_section(r),
        "",
        "## Latency — measured, not reproducible",
        _latency_section(r),
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


def changed_units(previous: str | None, previous_csv: str, new_csv: str) -> Changes:
    """Every unit and state whose label at 1%, 5% or 10% differs between the previous run's
    CSV and this one's, with both scores as written; `previous` is that CSV's file name, or
    None when there is no previous run on this golden set."""
    if previous is None:
        return Changes(None, "", "", ())
    old_rows = list(csv.DictReader(io.StringIO(previous_csv)))
    old = {(row["unit"], row["state"]): row for row in old_rows}
    changes: list[Change] = []
    for row in csv.DictReader(io.StringIO(new_csv)):
        before = old.get((row["unit"], row["state"]), {})
        for target in TARGETS:
            column = f"label_{target}"
            if before.get(column, "") != row[column]:
                changes.append(Change(
                    row["unit"], row["state"], target, before.get(column, ""), row[column],
                    _as_score(before.get("score", "")), _as_score(row["score"]),
                ))
    first = old_rows[0] if old_rows else {}
    return Changes(previous, first.get("model_id", ""), first.get("threshold_id", ""),
                   tuple(changes))


def _as_score(text: str) -> str:
    return "" if text == "" else f"{float(text):.6f}"
