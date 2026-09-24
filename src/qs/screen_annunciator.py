"""`ScreenAnnunciator`: the bench `Annunciator` (docs/11 §4, #87).

Every verdict the Pipeline hands it becomes a `Shown` (seq, label, reason, detail line,
image) and is presented as an operator view (`qs.operator_view.render`) in the window.
Between events `refresh()` presents the station's status and the live picture.

- **The image** (docs/15 §6). An appearance DEFECT with a heatmap and a `window_box` shows
  the window cut from the frame (unclipped, black outside it, as `cut_window` cuts it) with
  the heatmap overlaid. Any other DEFECT shows the frame with `unit_box` outlined in red.
  PASS, and НЕТ РЕШЕНИЯ for `placement`, `lighting` or `inference`, show the frame. Every
  other НЕТ РЕШЕНИЯ, and any event without a frame, shows no image, whatever pixels were
  passed (C4).
- **C9.** The live picture is only ever drawn in the live box. It is never stored as a
  `Shown`, or mixed into one.
- **I6.** When building or drawing a verdict fails, `show()` presents a plain amber
  screen and re-raises (the Pipeline counts it, F12). The stored last verdict becomes an
  НЕТ РЕШЕНИЯ with that event's number, so a later `refresh()` can never bring an earlier
  verdict back up for this unit. `refresh()` fails the same way: plain amber, then raise.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qs.config import StationConfig
from qs.contracts import CHECK_APPEARANCE, CHECK_GEOMETRY, CheckResult, Frame, Label, Verdict
from qs.inspector.heatmap import heatmap_overlay
from qs.inspector.window import cut_window
from qs.operator_view import (
    AMBER,
    RED,
    SIZE,
    Shown,
    Status,
    View,
    reason_line,
    render,
)
from qs.screen import ScreenLike

_RECENT = 10
_OUTLINE_PX = 3
_FRAME_REASONS = frozenset({"placement", "lighting", "inference"})


def _rgb(pixels: np.ndarray) -> np.ndarray:
    """A BGR frame as a new RGB array."""
    return np.ascontiguousarray(pixels[:, :, ::-1])


def _cut(pixels: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The window at `box` (x, y, w, h), unclipped and black outside the frame."""
    x, y, w, h = box
    if w != h:
        raise ValueError(f"the window box {box} is not square")
    window, cut_box = cut_window(pixels, (x + w // 2, y + h // 2), w)
    if cut_box != (x, y, w, h):
        raise ValueError(f"the window box {box} could not be cut exactly")
    return window


def _failed(verdict: Verdict, name: str) -> CheckResult | None:
    for check in verdict.checks:
        if check.name == name and check.passed is not True:
            return check
    return None


def _image(frame: Frame | None, verdict: Verdict, threshold: float) -> np.ndarray | None:
    if frame is None:
        return None
    if verdict.label is Label.NO_VERDICT:
        return _rgb(frame.pixels) if verdict.reason in _FRAME_REASONS else None
    if verdict.label is Label.PASS:
        return _rgb(frame.pixels)
    heatmap, window_box = verdict.heatmap, verdict.window_box
    if (_failed(verdict, CHECK_APPEARANCE) is not None and heatmap is not None
            and window_box is not None):
        return heatmap_overlay(_cut(frame.pixels, window_box), heatmap, threshold, heatmap != 0)
    image = _rgb(frame.pixels)
    if verdict.unit_box is not None:
        x, y, w, h = verdict.unit_box
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), RED, _OUTLINE_PX)
    return image


def _geometry_item(item: str, measures: dict[str, float], config: StationConfig) -> str:
    tolerances = config.threshold.geometry
    if item == "area_px<min":
        return f"площадь {measures['area_px']:.0f} px < {tolerances.area_px[0]}"
    if item == "area_px>max":
        return f"площадь {measures['area_px']:.0f} px > {tolerances.area_px[1]}"
    if item == "axis_ratio<min":
        return f"соотношение осей {measures['axis_ratio']:.2f} < {tolerances.axis_ratio_min:.2f}"
    if item == "solidity<min":
        return f"выпуклость {measures['solidity']:.3f} < {tolerances.solidity_min:.3f}"
    return item  # a rule this screen has no words for is shown as the Inspector named it


def _detail(verdict: Verdict, config: StationConfig) -> str:
    if verdict.label is Label.PASS:
        return ""
    if verdict.label is Label.NO_VERDICT:
        return reason_line(verdict.reason)
    parts: list[str] = []
    looks = _failed(verdict, CHECK_APPEARANCE)
    if looks is not None:
        parts.append(f"внешний вид: {looks.value:.2f} > порог {looks.limit:.2f}")
    shape = _failed(verdict, CHECK_GEOMETRY)
    if shape is not None and shape.failed_on:
        measures = dict(shape.measures)
        parts.extend(
            _geometry_item(item.strip(), measures, config)
            for item in shape.failed_on.split(",")
            if item.strip()
        )
    return " · ".join(parts)


class ScreenAnnunciator:
    """Shows every verdict on the operator screen, and the station's state between them."""

    def __init__(self, config: StationConfig, screen: ScreenLike) -> None:
        self._config = config
        self._screen = screen
        self._font = Path(config.display.font)
        self._status = Status("starting", None, frozenset(), None)
        self._last: Shown | None = None
        self._counts: dict[Label, int] = {}
        self._recent: deque[tuple[int | None, Label]] = deque(maxlen=_RECENT)
        self._closed = False

    @property
    def last_shown(self) -> Shown | None:
        """The verdict on the band now, or `None` before the first."""
        return self._last

    @property
    def counts(self) -> dict[Label, int]:
        """Every `show()` call per label, including those whose drawing failed."""
        return dict(self._counts)

    @property
    def recent(self) -> tuple[tuple[int | None, Label], ...]:
        """The newest 10 `(seq, label)` shown, oldest first."""
        return tuple(self._recent)

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        """Present this verdict: its colour, word, detail line and image."""
        seq = None if frame is None else frame.event.seq
        self._counts[verdict.label] = self._counts.get(verdict.label, 0) + 1
        self._recent.append((seq, verdict.label))
        try:
            self._last = Shown(
                seq=seq,
                label=verdict.label,
                reason=verdict.reason,
                detail=_detail(verdict, self._config),
                image=_image(frame, verdict, self._config.threshold.value),
            )
            image = self._render(None)
        except Exception:
            self._last = Shown(seq, Label.NO_VERDICT, None, "", None)
            self._screen.present(Image.new("RGB", SIZE, AMBER))
            raise
        self._screen.present(image)

    def refresh(self, status: Status, live: np.ndarray | None) -> None:
        """Present the station's current state and the live picture between events."""
        self._status = status
        try:
            image = self._render(live)
        except Exception:
            self._screen.present(Image.new("RGB", SIZE, AMBER))
            raise
        self._screen.present(image)

    def close(self) -> None:
        """Close the window; safe to call twice."""
        if not self._closed:
            self._closed = True
            self._screen.close()

    def _render(self, live: np.ndarray | None) -> Image.Image:
        view = View(
            status=self._status,
            live=live,
            last=self._last,
            counts=dict(self._counts),
            recent=tuple(self._recent),
        )
        return render(view, self._config.display.labels, self._font)
