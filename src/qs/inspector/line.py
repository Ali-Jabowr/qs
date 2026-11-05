"""`LineInspector`: the line station's `Inspector` — one frame, one cup, one verdict (ADR-0028).

`load(config)` keeps the config; there is no model to load, so `model_id` is `"none"` and no
record of this inspector can be mistaken for a model's.

`inspect(frame)` finds the cup wholly in the inspection zone (`find_cup`); no such cup is a
`PlacementFault`. Then the empty-cup check, measured from the inside of the found cup and never
learned: `blob_share` is the share of the cup's middle covered by the largest single region that
looks different from the cup's own inner edge. A моти is one large such region. An empty cup
shows the same floor and steel in the middle as at the edge, so only small marks differ. A
saturated pixel is glare and never part of the region, so a large glare patch cannot pass an
empty cup. The check passes when `blob_share` is at least `inspector.line.blob_share_min`. The
limit has only been tried on drawn frames. Pure: no I/O, the frame is only ever read, and nothing
changes between frames (C1, C2).
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from qs.config import LineConfig, StationConfig
from qs.contracts import CHECK_EMPTY_CUP, CheckResult, Frame, Verdict, derive_label
from qs.errors import ConfigError, InferenceError, PlacementFault
from qs.inspector.cup import find_cup

LINE_MODEL_ID = "none"
_EDGE_BAND_SHARE = 0.2  # the inner edge's width, as a share of the inside's radius
_DIFFERS_BY = 25  # grey levels, on any one channel, for a pixel to differ from the inner edge
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


class LineInspector:
    """The `Inspector` of a line station. `config_hash` is the loaded config's hash (C6)."""

    def __init__(self, config_hash: str) -> None:
        self._config_hash = config_hash
        self._config: StationConfig | None = None
        self._line: LineConfig | None = None

    @property
    def model_id(self) -> str:
        return LINE_MODEL_ID

    def load(self, config: StationConfig) -> None:
        if config.inspector.line is None:
            raise ConfigError("inspector.line: a line inspector needs this block")
        self._config = config
        self._line = config.inspector.line

    def inspect(self, frame: Frame) -> Verdict:
        """One `Verdict` carrying the empty-cup check (C8)."""
        start = time.perf_counter()
        config, line = self._config, self._line
        if config is None or line is None:
            raise InferenceError("the line inspector was not loaded")
        cup = find_cup(frame.pixels, line)
        if cup is None:
            raise PlacementFault("no cup wholly in the zone")
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
        checks = (check,)
        return Verdict(
            label=derive_label(checks, config.inspector.checks),
            checks=checks,
            heatmap=None,
            window_box=None,
            unit_box=cup.box,
            model_id=LINE_MODEL_ID,
            threshold_id=config.threshold.id,
            config_hash=self._config_hash,
            inspect_ms=(time.perf_counter() - start) * 1000.0,
        )
