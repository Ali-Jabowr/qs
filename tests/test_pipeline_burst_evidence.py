"""A warmth station's record comes from the verdict's evidence frame (#132): its image, camera
read-back, reference patch and capture time, not those of the burst's first frame."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from qs.annunciator import NullAnnunciator
from qs.config import StationConfig, load_config
from qs.contracts import CHECK_EMPTY_CUP, CheckResult, Frame, Label, TriggerEvent, Verdict
from qs.pipeline import Pipeline
from station_fixtures import (
    PATCH_GREY,
    VERIFIED,
    FakeClock,
    RecordingSink,
    ScriptedSource,
    event,
    textured,
)
from warmth_fixtures import warmth_config

PATCHES = (PATCH_GREY / 255, PATCH_GREY / 255 + 0.001, PATCH_GREY / 255 + 0.002)


class BurstSource(ScriptedSource):
    """Three frames per event, each with its own capture time (1, 2, 3 ms) and patch."""

    def capture_burst(self, event: TriggerEvent, count: int) -> tuple[Frame, ...]:
        return tuple(
            Frame(event, textured(index), 0.0, VERIFIED, PATCHES[index], float(index + 1))
            for index in range(count)
        )


class MiddleEvidence:
    """Passes a burst's empty-cup check and names the middle frame as the evidence."""

    model_id = "none"

    def __init__(self, config: StationConfig, config_hash: str) -> None:
        self._config = config
        self._hash = config_hash

    def load(self, config: StationConfig) -> None:
        return None

    def inspect(self, frame: Frame) -> Verdict:
        return self.inspect_burst((frame,))

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict:
        checks = (CheckResult(CHECK_EMPTY_CUP, True, 12.0, 5.0),)
        return Verdict(
            label=Label.PASS, checks=checks, heatmap=None, window_box=None, unit_box=None,
            model_id=self.model_id, threshold_id=self._config.threshold.id,
            config_hash=self._hash, inspect_ms=0.0, evidence_index=1,
        )


def test_the_record_carries_the_evidence_frames_capture_time_and_patch(tmp_path: Path) -> None:
    loaded = load_config(warmth_config(tmp_path))
    clock = FakeClock(start=0.0)
    inspector = MiddleEvidence(loaded.config, loaded.config_hash)
    sink = RecordingSink()
    pipeline = Pipeline(
        config=loaded.config, config_hash=loaded.config_hash, run_id="R1",
        source=BurstSource(clock), inspector=inspector, annunciator=NullAnnunciator(),
        sink=sink, clock=clock,
    )
    pipeline.start()

    record = pipeline.handle(event(1))

    assert record.capture_ms == 2.0
    assert record.reference_patch == pytest.approx(PATCH_GREY / 255 + 0.001)
