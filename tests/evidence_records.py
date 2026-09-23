"""Builders for the evidence tests (P03): valid `InspectionRecord`s made in memory.

A helper module, not a conftest: the P03 test files import it directly. Frames are small, so
a JPEG is a few hundred bytes; the heatmap PNG is any valid PNG. Nothing here writes a file.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_GEOMETRY,
    CameraReadback,
    CheckResult,
    InspectionRecord,
    Label,
    TriggerEvent,
    Verdict,
)
from qs.store import encode_frame_jpeg

RUN = "2026-09-25T10-00-00Z"
STATION = "A-bench"
CONFIG_HASH = "c" * 64
STARTED = datetime(2026, 9, 25, 10, 0, 0, tzinfo=UTC)
CAMERA = CameraReadback(exposure_us=10000, gain=0, wb_kelvin=4500, focus=30, verified=True)


@cache
def frame_jpeg() -> bytes:
    """A small real JPEG q85, from `encode_frame_jpeg`."""
    rng = np.random.default_rng(7)
    return encode_frame_jpeg(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))


@cache
def heatmap_png() -> bytes:
    ok, buf = cv2.imencode(".png", np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 256)
    assert ok
    return bytes(buf.tobytes())


def geometry(passed: bool = True) -> CheckResult:
    return CheckResult(
        name=CHECK_GEOMETRY,
        passed=passed,
        value=None,
        limit=None,
        measures=(("area_px", 104812.0), ("solidity", 0.97), ("axis_ratio", 1.04)),
        failed_on=None if passed else "area_px<min",
    )


def appearance(passed: bool = True) -> CheckResult:
    return CheckResult(
        name=CHECK_APPEARANCE,
        passed=passed,
        value=0.41 if passed else 0.93,
        limit=0.62,
        measures=(("max_score", 0.41 if passed else 0.93),),
        failed_on=None if passed else "score>threshold",
    )


def make_record(
    label: Label | str,
    seq: int,
    *,
    run_id: str = RUN,
    wall: datetime | None = None,
    image: bool | None = None,
    heatmap: bool = False,
    camera: CameraReadback | None | bool = True,
    reason: str | None = None,
    checks: tuple[CheckResult, ...] | None = None,
    heatmap_array: np.ndarray | None = None,
) -> InspectionRecord:
    """A valid record by default: PASS/DEFECT with both checks and a frame, NO_VERDICT
    with reason `lighting` and a frame. `image=False` drops the frame, `camera=False` the
    camera; for NO_VERDICT without a frame the camera defaults to none as well."""
    label = Label(label)
    if image is None:
        image = True
    if camera is True:
        camera_value: CameraReadback | None = CAMERA if image else None
    elif camera is False:
        camera_value = None
    else:
        camera_value = camera
    if checks is None:
        if label is Label.PASS:
            checks = (geometry(True), appearance(True))
        elif label is Label.DEFECT:
            checks = (geometry(True), appearance(False))
        else:
            checks = ()
    if label is Label.NO_VERDICT and reason is None:
        reason = "lighting" if image else "camera"
    verdict = Verdict(
        label=label,
        checks=checks,
        heatmap=heatmap_array,
        window_box=(100, 80, 512, 512) if label is not Label.NO_VERDICT else None,
        unit_box=(140, 120, 400, 380) if label is not Label.NO_VERDICT else None,
        model_id="patchcore-a1b2c3",
        threshold_id="thr-2026-09-25",
        config_hash=CONFIG_HASH,
        inspect_ms=123.5,
        reason=reason,
    )
    when = wall if wall is not None else STARTED + timedelta(seconds=seq, microseconds=250)
    return InspectionRecord(
        run_id=run_id,
        event=TriggerEvent(seq=seq, t0=1000.0 + seq, wall=when, source_id="replay"),
        verdict=verdict,
        camera=camera_value,
        reference_patch=180.25 if image else None,
        image_jpeg=frame_jpeg() if image else None,
        heatmap_png=heatmap_png() if heatmap else None,
        capture_ms=35.5 if image else None,
        total_ms=210.75,
    )


MIX = (
    lambda seq: make_record(Label.PASS, seq),
    lambda seq: make_record(Label.DEFECT, seq, heatmap=True),
    lambda seq: make_record(Label.NO_VERDICT, seq),
    lambda seq: make_record(Label.NO_VERDICT, seq, image=False),
)


def mixed(first: int, count: int, run_id: str = RUN) -> Iterator[InspectionRecord]:
    """`count` records of mixed labels, PASS, DEFECT, NO_VERDICT with and without a frame."""
    for seq in range(first, first + count):
        record = MIX[seq % len(MIX)](seq)
        if run_id != RUN:
            record = InspectionRecord(
                run_id=run_id,
                event=record.event,
                verdict=record.verdict,
                camera=record.camera,
                reference_patch=record.reference_patch,
                image_jpeg=record.image_jpeg,
                heatmap_png=record.heatmap_png,
                capture_ms=record.capture_ms,
                total_ms=record.total_ms,
            )
        yield record


def fast_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PRAGMA synchronous = OFF` on the store's connections, for tests that commit hundreds
    of records: an fsync per commit on the laptop's disk is ~5 ms, and no test may take 2 s.
    It weakens nothing a test here proves — a killed process keeps what it committed either
    way; only power loss differs, which is out of scope (docs/16 §8). The product default
    (SQLite's own journal and `synchronous`) is untouched."""
    import qs.store

    monkeypatch.setattr(qs.store, "PRAGMAS", (*qs.store.PRAGMAS, "PRAGMA synchronous = OFF"))


def files_under(root: Path) -> list[str]:
    """Every file under `images/` and `heatmaps/`, relative to the root, sorted."""
    return sorted(
        path.relative_to(root).as_posix()
        for top in ("images", "heatmaps")
        if (root / top).exists()
        for path in (root / top).rglob("*")
        if path.is_file()
    )
