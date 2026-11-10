"""`LineInspector`: the line station's `Inspector` — one frame, one cup, one verdict (ADR-0028).

`load(config)` keeps the config. Without `appearance` in `inspector.checks` there is no model
to load, so `model_id` is `"none"` and no record of this inspector can be mistaken for a
model's. With it, the bundle `inspector.model` names is loaded, verified and warmed up as on
`AppearanceInspector`, and `model_id` is the bundle's.

`inspect(frame)` finds the cup wholly in the inspection zone (`find_cup`); no such cup is a
`PlacementFault`. Then the empty-cup check, measured from the inside of the found cup and never
learned: `blob_share` is the share of the cup's middle covered by the largest single region that
looks different from the cup's own inner edge. A моти is one large such region. An empty cup
shows the same floor and steel in the middle as at the edge, so only small marks differ. A
saturated pixel is glare and never part of the region, so a large glare patch cannot pass an
empty cup. The check passes when `blob_share` is at least `inspector.line.blob_share_min`. The
limit has only been tried on drawn frames. With `appearance` listed, the window is then centred
on the rim's centre, as `qs train` cut the training windows, scored, and its map masked to the
cup's inside (ADR-0009); it runs on an empty cup too, so every record holds both checks.

A station with `inspector.line.warmth_min` judges by median warmth over a burst instead, in
`inspect_burst` (docs/11 "Bursts"): `blob_share` is not measured, and the appearance check runs on
the burst frame whose warmth is closest to the median. Pure: no I/O, the frame is only ever read,
and nothing changes between frames (C1, C2).
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
    only_frame,
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
_EDGE_BAND_SHARE = 0.2  # the inner edge's width, as a share of the inside's radius
_DIFFERS_BY = 25  # grey levels, on any one channel, for a pixel to differ from the inner edge
_WARMTH_SCALE = 0.5  # the "inner half" of the rim: its radius, halved
_GLARE_MIN = 245  # every channel at or above this is glare: it counts as no blob, however large


def blob_share(pixels: np.ndarray, inside: np.ndarray) -> float:
    """The share of the middle of `inside` covered by its largest region unlike the inner edge."""
    radius = math.sqrt(np.count_nonzero(inside) / math.pi)
    side = 2 * max(1, round(radius * _EDGE_BAND_SHARE)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    middle = cv2.erode(inside.astype(np.uint8), kernel).astype(bool)
    reference = np.median(pixels[inside & ~middle], axis=0)
    differs = np.abs(pixels.astype(np.int16) - reference).max(axis=2) > _DIFFERS_BY
    differs &= pixels.min(axis=2) < _GLARE_MIN
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (differs & middle).astype(np.uint8), connectivity=4
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return largest / int(np.count_nonzero(middle))


def warmth(pixels: np.ndarray, rim: Ellipse) -> float:
    """The mean of red minus blue over the inner half of the rim (its ellipse at half the axes).
    Measured on the rim's bounding box alone, so the mask and the conversion cost a cup, not a
    frame."""
    (cx, cy), axes, angle = rim
    reach = math.ceil(max(axes) / 2) + 2
    height, width = pixels.shape[:2]
    x0, y0 = max(0, round(cx) - reach), max(0, round(cy) - reach)
    crop = pixels[y0 : min(height, round(cy) + reach + 1), x0 : min(width, round(cx) + reach + 1)]
    local = ((round(cx) - x0, round(cy) - y0), axes, angle)
    region = crop[inside_of(local, crop.shape[:2], _WARMTH_SCALE)].astype(np.int16)
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
        """With `warmth_min` set, the median-warmth verdict over the burst; without it, a burst of
        1 is `inspect` and the line inspector judges no longer burst."""
        line = self._line
        if line is None or line.warmth_min is None:
            return self.inspect(only_frame(frames))
        return self._judge_by_warmth(frames, line)

    def inspect(self, frame: Frame) -> Verdict:
        """One `Verdict` carrying the empty-cup check (C8). A warmth station judges bursts, so
        one frame is fewer than its minimum and is a `PlacementFault`."""
        start = time.perf_counter()
        config, line = self._config, self._line
        if config is None or line is None:
            raise InferenceError("the line inspector was not loaded")
        if line.warmth_min is not None:
            return self._judge_by_warmth((frame,), line)
        cup = find_cup(frame.pixels, line)
        if cup is None:
            raise PlacementFault(no_cup_reason(frame.pixels, line))
        share = blob_share(frame.pixels, cup.inside)
        passed = share >= line.blob_share_min
        check = CheckResult(
            name=CHECK_EMPTY_CUP,
            passed=passed,
            value=share,
            limit=line.blob_share_min,
            measures=(("rim_axis_a_px", cup.rim[1][0]), ("rim_axis_b_px", cup.rim[1][1])),
            failed_on=None if passed else "blob_share<min",
        )
        return self._verdict(frame, cup, check, 0, start)

    def _judge_by_warmth(self, frames: Sequence[Frame], line: LineConfig) -> Verdict:
        """The median warmth of the cup over the burst's frames that show it (ADR-0029 §4). Too
        few frames, or warmth spread over `warmth_spread_max`, is a `PlacementFault`. The evidence
        is the frame whose warmth is closest to the median (the earliest on a tie)."""
        start = time.perf_counter()
        if line.warmth_min is None or line.warmth_spread_max is None:
            raise InferenceError("the line inspector has no warmth keys")
        found = [(i, frame, cup) for i, frame in enumerate(frames)
                 if (cup := find_cup(frame.pixels, line)) is not None]
        if len(found) < line.burst_min_frames:
            raise PlacementFault(
                f"the cup was found on {len(found)} of {len(frames)} frames, fewer than the "
                f"{line.burst_min_frames} needed"
            )
        warmths = np.array([warmth(frame.pixels, cup.rim) for _, frame, cup in found])
        spread = float(warmths.max() - warmths.min())
        if spread > line.warmth_spread_max:
            raise PlacementFault(
                f"the cup's warmth spread {spread:.1f} over the burst is above "
                f"{line.warmth_spread_max:g}"
            )
        median = float(np.median(warmths))
        index, frame, cup = found[int(np.argmin(np.abs(warmths - median)))]
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
