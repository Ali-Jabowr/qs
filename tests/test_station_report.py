"""The run report and its NFR-01 line (#91), on hand-made timings: pure, no station."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from qs.station_report import EventTiming, RunFacts, nfr01_line, render_report


def counters(**changes: Any) -> dict[str, object]:
    base: dict[str, object] = {
        "events": 0,
        "labels": {"pass": 0, "defect": 0, "no_verdict": 0},
        "no_verdict": {},
        "faults": {},
        "reconnect_attempts": 0,
        "camera_verified": True,
    }
    base.update(changes)
    return base


def facts(**changes: Any) -> RunFacts:
    base = RunFacts(
        run_id="20260927T100000000000Z",
        live=True,
        mode="live, /dev/video0",
        invocation="qs station --config config/station-A-bench.yml --report-dir reports/station",
        config_path="config/station-A-bench.yml",
        config_hash="c" * 64,
        model_id="patchcore-mochi-v1-3f9a1c2b7d4e",
        threshold_id="t-001",
        inspector_type="appearance",
        backend="torch-cuda",
        budget_ms=1000,
        counters=counters(),
        sink={"written": 0, "held": 0, "images_shed": 0, "lost": 0},
        keys_discarded=0,
    )
    return replace(base, **changes)


def passes(*totals: float) -> list[EventTiming]:
    return [EventTiming("pass", None, total, 300.0, 40.0) for total in totals]


def line_of(text: str, start: str) -> str:
    [line] = [line for line in text.splitlines() if line.startswith(start)]
    return line


# AC2 ------------------------------------------------------------------------------------


def test_the_latency_rows_are_numpy_percentiles_and_the_max() -> None:
    timings = passes(500, 100, 300, 200, 400)  # linear p50 300, p95 480, p99 496, max 500
    text = render_report(facts(), timings)
    assert line_of(text, "| total_ms, all events") == (
        "| total_ms, all events | 5 | 300.0 | 480.0 | 496.0 | 500.0 |"
    )
    assert line_of(text, "| total_ms, inspections") == (
        "| total_ms, inspections | 5 | 300.0 | 480.0 | 496.0 | 500.0 |"
    )
    assert float(np.percentile([100, 200, 300, 400, 500], 95)) == 480.0  # the method is linear
    assert line_of(text, "| inspect_ms, inspections") == (
        "| inspect_ms, inspections | 5 | 300.0 | 300.0 |"
    )
    assert line_of(text, "| capture_ms, events with a frame") == (
        "| capture_ms, events with a frame | 5 | 40.0 | 40.0 |"
    )


@pytest.mark.parametrize(
    ("reason", "inspection"),
    [
        ("placement", True),
        ("inference", True),
        ("capture", False),
        ("lighting", False),
        ("camera", False),
        ("camera_settings", False),
        ("busy", False),
    ],
)
def test_which_no_verdicts_are_inspections(reason: str, inspection: bool) -> None:
    timing = EventTiming("no_verdict", reason, 700.0, 0.0, None)
    assert timing.inspection is inspection
    text = render_report(facts(), [*passes(100), timing])
    row = line_of(text, "| total_ms, inspections")
    assert row.split(" | ")[1] == ("2" if inspection else "1")


def test_pass_and_defect_are_inspections() -> None:
    assert EventTiming("pass", None, 1.0, 1.0, 1.0).inspection
    assert EventTiming("defect", None, 1.0, 1.0, 1.0).inspection


def test_a_section_with_no_values_prints_a_dash() -> None:
    text = render_report(facts(), [EventTiming("no_verdict", "camera", 5.0, 0.0, None)])
    assert line_of(text, "| total_ms, inspections") == (
        "| total_ms, inspections | 0 | — | — | — | — |"
    )
    assert line_of(text, "| capture_ms") == "| capture_ms, events with a frame | 0 | — | — |"
    assert "- NO_VERDICT by reason: —" in text and "- FAULT entries by reason: —" in text


def test_breaches_count_events_over_the_budget() -> None:
    text = render_report(facts(budget_ms=450), passes(400, 450, 451, 900))
    assert "Events over the budget (450 ms): 2" in text


def test_the_report_names_the_run_and_its_counts() -> None:
    run = facts(
        counters=counters(
            events=4,
            labels={"pass": 1, "defect": 1, "no_verdict": 2},
            no_verdict={"camera": 1, "placement": 1},
            faults={"camera": 1},
            reconnect_attempts=3,
        ),
        sink={"written": 4, "held": 1, "images_shed": 0, "lost": 0},
        keys_discarded=7,
    )
    text = render_report(run, passes(100))
    assert text.startswith("# Station run 20260927T100000000000Z\n")
    for expected in (
        "- Mode: live, /dev/video0",
        f"config_hash `{'c' * 64}`",
        "- Model: `patchcore-mochi-v1-3f9a1c2b7d4e` · threshold `t-001`",
        "- Inspector: appearance · backend torch-cuda",
        "- Camera verified: yes",
        "- Events: 4",
        "- Labels: pass 1 · defect 1 · no_verdict 2",
        "- NO_VERDICT by reason: camera 1, placement 1",
        "- FAULT entries by reason: camera 1",
        "- Reconnect attempts: 3",
        "- Keys discarded: 7",
        "- Evidence: written 4 · held 1 · images shed 0 · lost 0",
    ):
        assert expected in text
    assert text.rstrip().endswith(nfr01_line(run, passes(100)))


# AC3 ------------------------------------------------------------------------------------


def test_nfr01_passes_a_live_gpu_run_of_200_inspections() -> None:
    assert nfr01_line(facts(), passes(*[400.0] * 200)) == (
        "NFR-01: PASS — p95 400 ms ≤ 1000 ms over 200 inspections"
    )


def test_nfr01_needs_200_inspections() -> None:
    assert nfr01_line(facts(), passes(*[400.0] * 199)) == (
        "NFR-01: not measured — 199 inspections, 200 needed (indicative p95 400 ms)"
    )


def test_nfr01_fails_when_p95_is_over_1000_ms() -> None:
    totals = [400.0] * 189 + [1500.0] * 11
    p95 = float(np.percentile(totals, 95))
    assert p95 == 1500.0
    assert nfr01_line(facts(), passes(*totals)) == (
        "NFR-01: FAIL — p95 1500 ms > 1000 ms over 200 inspections"
    )


def test_nfr01_is_not_measured_on_a_run_that_entered_fault() -> None:
    run = facts(counters=counters(faults={"capture": 1}))
    assert nfr01_line(run, passes(*[400.0] * 200)) == (
        "NFR-01: not measured — the run entered FAULT (indicative p95 400 ms)"
    )


def test_nfr01_is_on_the_gpu() -> None:
    assert nfr01_line(facts(backend="torch-cpu"), passes(*[400.0] * 200)) == (
        "NFR-01: not measured — backend torch-cpu; NFR-01 is on the GPU (indicative p95 400 ms)"
    )


def test_nfr01_is_never_a_stub() -> None:
    assert nfr01_line(facts(inspector_type="stub"), passes(*[400.0] * 200)).startswith(
        "NFR-01: not measured — stub inspector"
    )


def test_nfr01_is_never_a_replay_even_when_all_else_holds() -> None:
    run = facts(live=False, mode="replay of shoot")
    assert nfr01_line(run, passes(*[400.0] * 200)) == (
        "NFR-01: not measured — replay has no capture step (indicative p95 400 ms)"
    )


def test_nfr01_without_an_inspection_has_no_indicative_p95() -> None:
    assert nfr01_line(facts(), [EventTiming("no_verdict", "camera", 5.0, 0.0, None)]) == (
        "NFR-01: not measured — 0 inspections, 200 needed"
    )


def test_the_limit_is_nfr01s_not_the_budget() -> None:
    assert nfr01_line(facts(budget_ms=300), passes(*[400.0] * 200)).startswith("NFR-01: PASS")


# AC6 (pure part) --------------------------------------------------------------------------


def test_the_report_holds_no_image() -> None:
    text = render_report(facts(), passes(100, 200))
    for suffix in (".png", ".jpg", ".jpeg"):
        assert suffix not in text.lower()
