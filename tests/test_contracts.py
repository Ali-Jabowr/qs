import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    REASONS,
    CameraReadback,
    CheckResult,
    Frame,
    Health,
    InspectionRecord,
    Label,
    TriggerEvent,
    Verdict,
    derive_label,
)
from qs.errors import (
    CameraLost,
    CameraSettingsDrift,
    CaptureError,
    ConfigError,
    InferenceError,
    LightingFault,
    PlacementFault,
    QSError,
)

REASONED_ERRORS = (
    CaptureError,
    CameraLost,
    CameraSettingsDrift,
    LightingFault,
    PlacementFault,
    InferenceError,
)


def _event() -> TriggerEvent:
    return TriggerEvent(seq=1, t0=0.0, wall=datetime.now(tz=UTC), source_id="keyboard")


def _readback() -> CameraReadback:
    return CameraReadback(
        exposure_us=10000, gain=0, wb_kelvin=4500, focus=30, verified=True
    )


def _frame() -> Frame:
    return Frame(
        event=_event(),
        pixels=np.zeros((4, 4, 3), dtype=np.uint8),
        grabbed_at=0.012,
        camera=_readback(),
        reference_patch=0.62,
        capture_ms=12.5,
    )


def _check(name: str, passed: bool) -> CheckResult:
    return CheckResult(name=name, passed=passed, value=None, limit=None)


def _verdict(
    label: Label, checks: tuple[CheckResult, ...], reason: str | None = None
) -> Verdict:
    return Verdict(
        label=label,
        checks=checks,
        heatmap=None,
        window_box=None,
        unit_box=None,
        model_id="patchcore-mochi-v1-3f9a1c2b7d4e",
        threshold_id="cal-2026-09-25-far5-a41c07",
        config_hash="0" * 64,
        inspect_ms=3.0,
        reason=reason,
    )


def _record() -> InspectionRecord:
    return InspectionRecord(
        run_id="run-1",
        event=_event(),
        verdict=_verdict(Label.NO_VERDICT, (), reason="capture"),
        camera=None,
        reference_patch=None,
        image_jpeg=None,
        heatmap_png=None,
        capture_ms=None,
        total_ms=4.0,
    )


def _health() -> Health:
    return Health(ok=True, detail="fine")


def test_every_value_record_is_immutable() -> None:
    records: tuple[tuple[object, str], ...] = (
        (_event(), "seq"),
        (_readback(), "gain"),
        (_frame(), "capture_ms"),
        (_check(CHECK_GEOMETRY, True), "passed"),
        (_verdict(Label.PASS, (_check(CHECK_GEOMETRY, True),)), "model_id"),
        (_health(), "detail"),
        (_record(), "total_ms"),
    )
    for record, attribute in records:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, attribute, 1)


def test_label_is_immutable() -> None:
    with pytest.raises((AttributeError, TypeError)):
        Label.PASS = Label.DEFECT  # type: ignore[misc]


def test_verdict_refuses_no_verdict_without_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _verdict(Label.NO_VERDICT, ())


def test_verdict_refuses_pass_without_checks() -> None:
    with pytest.raises(ValueError, match="checks"):
        _verdict(Label.PASS, ())


def test_verdict_refuses_defect_without_checks() -> None:
    with pytest.raises(ValueError, match="checks"):
        _verdict(Label.DEFECT, ())


def test_verdict_accepts_no_verdict_with_a_reason() -> None:
    assert _verdict(Label.NO_VERDICT, (), reason="capture").label is Label.NO_VERDICT


CONFIGURED = frozenset({CHECK_GEOMETRY, CHECK_APPEARANCE})


def test_derive_label_both_checks_pass() -> None:
    checks = (_check(CHECK_GEOMETRY, True), _check(CHECK_APPEARANCE, True))
    assert derive_label(checks, CONFIGURED) is Label.PASS


def test_derive_label_geometry_fails() -> None:
    checks = (_check(CHECK_GEOMETRY, False), _check(CHECK_APPEARANCE, True))
    assert derive_label(checks, CONFIGURED) is Label.DEFECT


def test_derive_label_appearance_fails() -> None:
    checks = (_check(CHECK_GEOMETRY, True), _check(CHECK_APPEARANCE, False))
    assert derive_label(checks, CONFIGURED) is Label.DEFECT


def test_derive_label_both_fail() -> None:
    checks = (_check(CHECK_GEOMETRY, False), _check(CHECK_APPEARANCE, False))
    assert derive_label(checks, CONFIGURED) is Label.DEFECT


def test_derive_label_configured_check_missing() -> None:
    checks = (_check(CHECK_GEOMETRY, True),)
    with pytest.raises(ValueError, match="did not run"):
        derive_label(checks, CONFIGURED)


def test_derive_label_unconfigured_check_name() -> None:
    checks = (_check(CHECK_GEOMETRY, True), _check("marking", True))
    with pytest.raises(ValueError, match="not configured"):
        derive_label(checks, CONFIGURED)


def test_derive_label_duplicate_names() -> None:
    checks = (_check(CHECK_GEOMETRY, True), _check(CHECK_GEOMETRY, True))
    with pytest.raises(ValueError, match="share a name"):
        derive_label(checks, CONFIGURED)


def test_derive_label_empty_configured() -> None:
    with pytest.raises(ValueError, match="no configured checks"):
        derive_label((_check(CHECK_GEOMETRY, True),), frozenset())


def test_every_exception_reason_is_a_reason() -> None:
    for error in REASONED_ERRORS:
        assert error.reason in REASONS, error


def test_exception_reasons_are_reasons_minus_busy() -> None:
    assert {error.reason for error in REASONED_ERRORS} == set(REASONS) - {"busy"}


def test_error_hierarchy() -> None:
    assert issubclass(CameraLost, CaptureError)
    for error in REASONED_ERRORS:
        assert issubclass(error, QSError)
    assert issubclass(ConfigError, QSError)


def test_placement_fault_takes_a_detail() -> None:
    for detail in ("no unit", "more than one unit", "not wholly in view"):
        assert PlacementFault(detail).detail == detail
