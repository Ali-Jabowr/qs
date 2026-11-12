"""`LineInspector`: the line station's `Inspector` — one frame, one cup, one verdict (ADR-0028).

`load(config)` keeps the config. Without `appearance` in `inspector.checks` there is no model
to load, so `model_id` is `"none"` and no record of this inspector can be mistaken for a
model's. With it, the bundle `inspector.model` names is loaded, verified and warmed up as on
`AppearanceInspector`, and `model_id` is the bundle's.

`inspect_burst(frames)` judges the empty cup by median warmth over the burst (ADR-0029 §4,
docs/11 "Bursts"), measured and never learned: on each frame the cup wholly in the inspection
zone is found (`find_cup`) and warmth, the mean of red minus blue, is measured over the inner
half of its rim. The check passes when the median is at least `inspector.line.warmth_min`. Too
few frames showing the cup, or too much spread between them, is a `PlacementFault`. `inspect`
judges one frame as a burst of one, so it needs `burst_min_frames` of 1 to give a verdict. The
limits have only been tried on drawn frames. With `appearance` listed, the window is then centred
on the rim's centre of the burst frame whose warmth is closest to the median, as `qs train` cut
the training windows, scored, and its map masked to the cup's inside (ADR-0009); it runs on an
empty cup too, so every record holds both checks. Pure: no I/O, the frame is only ever read, and
nothing changes between frames (C1, C2).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from qs.config import LineConfig, StationConfig
from qs.contracts import (
    CHECK_APPEARANCE,
    CHECK_EMPTY_CUP,
    CheckResult,
    Frame,
    Verdict,
    derive_label,
)
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.appearance import (
    LoadedModel,
    check_appearance,
    load_model,
    real_cuda_available,
    real_scorer,
)
from qs.inspector.cup import Cup, Ellipse, find_cup, inside_of, no_cup_reason
from qs.inspector.model import Scorer

LINE_MODEL_ID = "none"
_WARMTH_SCALE = 0.5  # the "inner half" of the rim: its radius, halved


def warmth(
    pixels: np.ndarray, rim: Ellipse, lens: tuple[int, int, int] | None = None
) -> float | None:
    """The mean of red minus blue over the inner half of the rim (its ellipse at half the axes),
    leaving out the `lens` circle (x, y, radius) in frame coordinates. Measured on the rim's
    bounding box alone, so the masks and the conversion cost a cup, not a frame. None when `lens`
    covers the whole inner half: there is nothing to measure."""
    (cx, cy), axes, angle = rim
    reach = math.ceil(max(axes) / 2) + 2
    height, width = pixels.shape[:2]
    x0, y0 = max(0, round(cx) - reach), max(0, round(cy) - reach)
    crop = pixels[y0 : min(height, round(cy) + reach + 1), x0 : min(width, round(cx) + reach + 1)]
    local = ((round(cx) - x0, round(cy) - y0), axes, angle)
    inside = inside_of(local, crop.shape[:2], _WARMTH_SCALE)
    if lens is not None:
        hole = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(hole, (lens[0] - x0, lens[1] - y0), lens[2], 255, cv2.FILLED)
        inside = inside & ~hole.astype(bool)
    if not inside.any():
        return None
    region = crop[inside].astype(np.int16)
    return float((region[:, 2] - region[:, 0]).mean())


class LineInspector:
    """The `Inspector` of a line station. `config_hash` is the loaded config's hash (C6);
    `models_root`, `make_scorer` and `cuda_available` are used only when `appearance` is listed,
    as on `AppearanceInspector`."""

    def __init__(
        self,
        config_hash: str,
        models_root: str | Path = "models",
        *,
        make_scorer: Callable[[str, np.ndarray], Scorer] = real_scorer,
        cuda_available: Callable[[], bool] = real_cuda_available,
    ) -> None:
        self._config_hash = config_hash
        self._models_root = Path(models_root)
        self._make_scorer = make_scorer
        self._cuda_available = cuda_available
        self._config: StationConfig | None = None
        self._line: LineConfig | None = None
        self._appearance: LoadedModel | None = None

    @property
    def model_id(self) -> str:
        """The loaded bundle's `model_id` when `appearance` is listed, else `"none"`."""
        return LINE_MODEL_ID if self._appearance is None else self._appearance.bundle.model_id

    def load(self, config: StationConfig) -> None:
        if config.inspector.line is None:
            raise ConfigError("inspector.line: a line inspector needs this block")
        self._config = None
        self._line = None
        self._appearance = None
        if CHECK_APPEARANCE in config.inspector.checks:
            self._appearance = load_model(
                self._models_root, config, validation=False,
                make_scorer=self._make_scorer, cuda_available=self._cuda_available,
            )
        self._config = config
        self._line = config.inspector.line

    def inspect_burst(self, frames: Sequence[Frame]) -> Verdict:
        """The median-warmth verdict over the burst, carrying the empty-cup check (C8)."""
        line = self._line
        if line is None:
            raise InferenceError("the line inspector was not loaded")
        return self._judge_by_warmth(frames, line)

    def inspect(self, frame: Frame) -> Verdict:
        """`frame` judged as a burst of one: fewer than `burst_min_frames` is a `PlacementFault`."""
        return self.inspect_burst((frame,))

    def _judge_by_warmth(self, frames: Sequence[Frame], line: LineConfig) -> Verdict:
        """The median warmth of the cup over the burst's frames that show it (ADR-0029 §4). Too
        few frames, or warmth spread over `warmth_spread_max`, is a `PlacementFault`; when a frame
        showed no cup, the fault adds why the first such frame had none (a wrong-size rim after a
        remount, say). The evidence is the frame whose warmth is closest to the median (the
        earliest on a tie)."""
        start = time.perf_counter()
        found = [(i, frame, cup) for i, frame in enumerate(frames)
                 if (cup := find_cup(frame.pixels, line)) is not None]
        measured = [(i, frame, cup, w) for i, frame, cup in found
                    if (w := warmth(frame.pixels, cup.rim, line.lens_mask_px)) is not None]
        masked = len(found) - len(measured)
        if len(measured) < line.burst_min_frames:
            seen = f"the cup was found on {len(found)} of {len(frames)} frames"
            if masked:
                seen += f", the lens mask covered the whole inner half on {masked}"
            fault = f"{seen}, fewer than the {line.burst_min_frames} needed"
            found_at = {i for i, _, _ in found}
            missed = next((f for i, f in enumerate(frames) if i not in found_at), None)
            if missed is not None:
                fault += f"; {no_cup_reason(missed.pixels, line)}"
            raise PlacementFault(fault)
        kept = [(i, frame, cup) for i, frame, cup, _ in measured]
        warmths = np.array([w for *_, w in measured])
        spread = float(warmths.max() - warmths.min())
        if spread > line.warmth_spread_max:
            raise PlacementFault(
                f"the cup's warmth spread {spread:.1f} over the burst is above "
                f"{line.warmth_spread_max:g}"
            )
        median = float(np.median(warmths))
        index, frame, cup = kept[int(np.argmin(np.abs(warmths - median)))]
        passed = median >= line.warmth_min
        check = CheckResult(
            name=CHECK_EMPTY_CUP,
            passed=passed,
            value=median,
            limit=line.warmth_min,
            measures=(("frames", float(len(found))), ("spread", spread)),
            failed_on=None if passed else "warmth<min",
        )
        return self._verdict(frame, cup, check, index, start)

    def _verdict(
        self, frame: Frame, cup: Cup, check: CheckResult, evidence_index: int, start: float
    ) -> Verdict:
        """The verdict of `check` on the cup found in `frame`, with the appearance check beside
        it when it is configured."""
        config = self._config
        if config is None:
            raise InferenceError("the line inspector was not loaded")
        checks: tuple[CheckResult, ...] = (check,)
        heatmap, window_box = None, None
        if self._appearance is not None:
            # Run on an empty cup too: one path, and every record holds both checks.
            appearance, heatmap, window_box = check_appearance(
                frame.pixels, cup.rim[0], cup.inside, self._appearance, config
            )
            checks = (check, appearance)
        return Verdict(
            label=derive_label(checks, config.inspector.checks),
            checks=checks,
            heatmap=heatmap,
            window_box=window_box,
            unit_box=cup.box,
            model_id=self.model_id,
            threshold_id=config.threshold.id,
            config_hash=self._config_hash,
            inspect_ms=(time.perf_counter() - start) * 1000.0,
            evidence_index=evidence_index,
        )
