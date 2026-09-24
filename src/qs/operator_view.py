"""The operator view: every screen of docs/16 §5, drawn by one pure function (#86).

`render(view, labels, font_path)` turns a `View` value into the 1280 × 800 RGB image the
operator sees. Everything the operator will ever see is decided here, so every screen is
tested by its pixels and its strings without a display; the Tk window only presents it.
No Tk, no clock, no I/O except reading the font. Inputs are only ever read, never written,
and read-only arrays work. The one piece of state is an optional `FitCache` the caller
owns, which only saves re-fitting the same verdict's image (REVIEW-P09 m1).

Layout (y in px): banner 0–36 · verdict band 36–236 · pictures 236–676 (live left, the
verdict's image right) · session panel 676–740 · status bar 740–800. In mode `fault` the
whole canvas is amber with СТАНЦИЯ НЕ ПРОВЕРЯЕТ and the fault line, and nothing else.

C7 / I6: green is drawn only for a `Shown` whose label is exactly `Label.PASS` (the band and
the panel's squares). НЕТ РЕШЕНИЯ is amber, never green. The wording is the docs' draft
(docs/16 §5): the three verdict words come from the config, the rest are constants here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qs.config import LabelsConfig
from qs.contracts import Label

GREEN = (0, 160, 0)
RED = (200, 0, 0)
AMBER = (255, 170, 0)
GREY = (40, 40, 40)
BANNER = (110, 70, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

SIZE = (1280, 800)
_BANNER_Y = (0, 36)
_BAND_Y = (36, 236)
_WORD_Y = (36, 186)
_LINE_Y = (190, 230)
_PICTURES_Y = (236, 676)
_PANEL_Y = (676, 740)
_STATUS_Y = (740, 800)
_BOX_W = 640
_TEXT_W = 1240
_WORD_MAX_PX = 120
_LINE_PX = 28
_SMALL_PX = 24
_FAULT_WORD_MAX_PX = 120
_FAULT_LINE_PX = 44
_SQUARE_PX = 40
_SQUARE_STEP = 48
_SQUARE_Y = 688
_NEWEST_SQUARE_X = 1232
_RECENT = 10
_PANEL_TEXT_W = _NEWEST_SQUARE_X - (_RECENT - 1) * _SQUARE_STEP - 20 - 10

MODES = frozenset({"starting", "ready", "fault"})
WARNINGS = ("STORAGE", "LIGHTING", "CAMERA_SETTINGS", "LATENCY")
FAULT_TITLE = "СТАНЦИЯ НЕ ПРОВЕРЯЕТ"
BUSY_WORD = "ИЗДЕЛИЕ НЕ ПРОВЕРЕНО"
NO_LIVE = "нет изображения"
NO_FRAME = "нет кадра"

_STATUS = {
    "starting": "ЗАПУСК",
    "ready": "ГОТОВ — положите изделие в рамку и нажмите ПРОБЕЛ",
    "fault": FAULT_TITLE,
}
_REASONS = {
    "placement": "положите изделие в рамку",
    "lighting": "проверьте освещение",
    "capture": "нет кадра",
    "camera_settings": "настройки камеры изменились",
    "inference": "ошибка анализа",
    "camera": "камера не подключена",
    "busy": "изделие не проверено",
}
_UNKNOWN_REASON = "нет решения"
_FAULTS = {
    "camera": "камера не подключена",
    "capture": "нет кадров",
    "inference": "ошибка анализа — перезапустите станцию",
}
_BANNERS = {
    "STORAGE": "архив не записывается",
    "LIGHTING": "освещение вне нормы",
    "CAMERA_SETTINGS": "настройки камеры не подтверждены",
    "LATENCY": "медленная работа",
}


@dataclass(frozen=True, slots=True)
class Status:
    """What the station is doing between events: `starting`, `ready` or `fault`."""

    mode: str
    fault_reason: str | None
    warnings: frozenset[str]
    caption: str | None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown station mode {self.mode!r}")
        unknown = sorted(set(self.warnings) - set(WARNINGS))
        if unknown:
            raise ValueError("unknown warning(s): " + ", ".join(unknown))


@dataclass(frozen=True, slots=True)
class Shown:
    """The last verdict on screen: its event number, label, reason, detail line and image."""

    seq: int | None
    label: Label
    reason: str | None
    detail: str
    image: np.ndarray | None  # RGB uint8


@dataclass(frozen=True, slots=True)
class View:
    """Everything the operator screen shows at one moment."""

    status: Status
    live: np.ndarray | None  # BGR uint8, as frames are
    last: Shown | None
    counts: Mapping[Label, int]
    recent: tuple[tuple[int | None, Label], ...]  # oldest first


def status_line(status: Status, labels: LabelsConfig) -> str:
    """The status bar's text."""
    if status.mode == "ready" and status.caption is not None:
        return status.caption
    return _STATUS[status.mode]


def reason_line(reason: str | None) -> str:
    """The short line under an НЕТ РЕШЕНИЯ: what the operator should fix."""
    if reason is None:
        return _UNKNOWN_REASON
    return _REASONS.get(reason, _UNKNOWN_REASON)


def fault_line(reason: str | None) -> str:
    """The line under СТАНЦИЯ НЕ ПРОВЕРЯЕТ; empty for a reason the docs do not name."""
    if reason is None:
        return ""
    return _FAULTS.get(reason, "")


def banner_line(warnings: frozenset[str]) -> str:
    """Every active warning, in the fixed order, joined by ` · `; `""` when none."""
    return " · ".join(_BANNERS[name] for name in WARNINGS if name in warnings)


def panel_line(counts: Mapping[Label, int], labels: LabelsConfig) -> str:
    """The session count per label, in the config's words; a missing count is 0."""
    return " · ".join(
        f"{label_word(label, None, labels)} {counts.get(label, 0)}"
        for label in (Label.PASS, Label.DEFECT, Label.NO_VERDICT)
    )


def label_word(label: Label, reason: str | None, labels: LabelsConfig) -> str:
    """The verdict word: the config's, except ИЗДЕЛИЕ НЕ ПРОВЕРЕНО for a `busy` NO_VERDICT."""
    if label is Label.PASS:
        return labels.pass_
    if label is Label.DEFECT:
        return labels.defect
    if reason == "busy":
        return BUSY_WORD
    return labels.no_verdict


def label_colour(label: Label) -> tuple[int, int, int]:
    """Green only for exactly PASS, red for DEFECT, amber for anything else (C7)."""
    if label is Label.PASS:
        return GREEN
    if label is Label.DEFECT:
        return RED
    return AMBER


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _fitted_font(path: str, text: str, max_px: int, width: int) -> ImageFont.FreeTypeFont:
    """The largest font of at most `max_px` whose `text` fits `width` pixels wide."""
    size = max_px
    font = _font(path, size)
    while size > 8 and font.getlength(text) > width:
        size -= 2
        font = _font(path, size)
    return font


Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def _fitted_size(width: int, height: int, box: Box) -> tuple[int, int]:
    """The largest (w, h) of this aspect that fits `box`."""
    x0, y0, x1, y1 = box
    scale = min((x1 - x0) / width, (y1 - y0) / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _fit_rgb(rgb: np.ndarray, box: Box) -> Image.Image:
    """An RGB array fitted to `box`, keeping its aspect (Lanczos; the verdict's image)."""
    picture = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    fitted = _fitted_size(picture.width, picture.height, box)
    return picture.resize(fitted, Image.Resampling.LANCZOS)


def _fit_live(bgr: np.ndarray, box: Box) -> Image.Image:
    """A BGR frame fitted to `box`, keeping its aspect, as RGB.

    Shrunk by OpenCV's area filter before Pillow sees it: ~1 ms for 1920 × 1080 instead of
    ~32 ms through Lanczos (REVIEW-P09 m1). It is redrawn on every refresh."""
    height, width = bgr.shape[:2]
    fitted = _fitted_size(width, height, box)
    small = cv2.resize(np.ascontiguousarray(bgr), fitted, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def _paste_centred(canvas: Image.Image, picture: Image.Image, box: Box) -> None:
    x0, y0, x1, y1 = box
    canvas.paste(
        picture, (x0 + (x1 - x0 - picture.width) // 2, y0 + (y1 - y0 - picture.height) // 2)
    )


class FitCache:
    """The last verdict's image, fitted once (REVIEW-P09 m1).

    A verdict's image changes only on `show()`, but the screen is redrawn on every refresh.
    A hit needs the very same `Shown` object (by identity; the cache holds it, so its id
    cannot be reused), the same seq and the same box. Any new `Shown` misses, so an earlier
    verdict's image can never be drawn under a later one."""

    __slots__ = ("_key", "_picture")

    def __init__(self) -> None:
        self._key: tuple[Shown, int | None, Box] | None = None
        self._picture: Image.Image | None = None

    def fitted(self, shown: Shown, box: Box) -> Image.Image:
        key = self._key
        if (key is not None and self._picture is not None and key[0] is shown
                and key[1] == shown.seq and key[2] == box):
            return self._picture
        self._key = self._picture = None
        if shown.image is None:
            raise ValueError("this verdict has no image to fit")
        picture = _fit_rgb(shown.image, box)
        self._key, self._picture = (shown, shown.seq, box), picture
        return picture


def _render_fault(canvas: Image.Image, status: Status, font_path: str) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, SIZE[0] - 1, SIZE[1] - 1), fill=AMBER)
    title = _fitted_font(font_path, FAULT_TITLE, _FAULT_WORD_MAX_PX, _TEXT_W)
    draw.text((SIZE[0] // 2, 340), FAULT_TITLE, fill=BLACK, font=title, anchor="mm")
    line = fault_line(status.fault_reason)
    if line:
        font = _fitted_font(font_path, line, _FAULT_LINE_PX, _TEXT_W)
        draw.text((SIZE[0] // 2, 470), line, fill=BLACK, font=font, anchor="mm")
    return canvas


def render(
    view: View, labels: LabelsConfig, font_path: Path, cache: FitCache | None = None
) -> Image.Image:
    """The operator screen as a 1280 × 800 RGB image (docs/16 §5).

    `cache` keeps the verdict's fitted image between calls; the pixels are the same with or
    without it."""
    path = str(font_path)
    canvas = Image.new("RGB", SIZE, BLACK)
    status = view.status
    if status.mode == "fault":
        return _render_fault(canvas, status, path)
    draw = ImageDraw.Draw(canvas)

    banner = banner_line(status.warnings)
    if banner:
        draw.rectangle((0, _BANNER_Y[0], SIZE[0] - 1, _BANNER_Y[1] - 1), fill=BANNER)
        font = _fitted_font(path, banner, _SMALL_PX, _TEXT_W)
        draw.text((20, sum(_BANNER_Y) // 2), banner, fill=WHITE, font=font, anchor="lm")

    last = view.last
    band = GREY if last is None else label_colour(last.label)
    ink = BLACK if band == AMBER else WHITE
    draw.rectangle((0, _BAND_Y[0], SIZE[0] - 1, _BAND_Y[1] - 1), fill=band)
    if last is not None:
        word = label_word(last.label, last.reason, labels)
        font = _fitted_font(path, word, _WORD_MAX_PX, _TEXT_W)
        draw.text((SIZE[0] // 2, sum(_WORD_Y) // 2), word, fill=ink, font=font, anchor="mm")
        line = "№ —" if last.seq is None else f"№ {last.seq}"
        if last.detail:
            line += f" · {last.detail}"
        font = _fitted_font(path, line, _LINE_PX, _TEXT_W)
        draw.text((SIZE[0] // 2, sum(_LINE_Y) // 2), line, fill=ink, font=font, anchor="mm")

    top, bottom = _PICTURES_Y
    middle = (top + bottom) // 2
    if status.mode == "ready" and view.live is not None:
        box: Box = (0, top, _BOX_W, bottom)
        _paste_centred(canvas, _fit_live(view.live, box), box)
    else:
        text = status.caption if status.caption is not None else NO_LIVE
        font = _fitted_font(path, text, _LINE_PX, _BOX_W - 40)
        draw.text((_BOX_W // 2, middle), text, fill=WHITE, font=font, anchor="mm")
    if last is not None and last.image is not None:
        box = (_BOX_W, top, SIZE[0], bottom)
        picture = _fit_rgb(last.image, box) if cache is None else cache.fitted(last, box)
        _paste_centred(canvas, picture, box)
    else:
        font = _font(path, _LINE_PX)
        draw.text((_BOX_W + _BOX_W // 2, middle), NO_FRAME, fill=WHITE, font=font, anchor="mm")

    panel = panel_line(view.counts, labels)
    font = _fitted_font(path, panel, _SMALL_PX, _PANEL_TEXT_W)
    draw.text((20, sum(_PANEL_Y) // 2), panel, fill=WHITE, font=font, anchor="lm")
    newest = view.recent[-_RECENT:]
    for age, (_, label) in enumerate(reversed(newest)):
        x0 = _NEWEST_SQUARE_X - age * _SQUARE_STEP
        draw.rectangle(
            (x0, _SQUARE_Y, x0 + _SQUARE_PX - 1, _SQUARE_Y + _SQUARE_PX - 1),
            fill=label_colour(label),
        )

    text = status_line(status, labels)
    font = _fitted_font(path, text, _LINE_PX, _TEXT_W)
    draw.text((20, sum(_STATUS_Y) // 2), text, fill=WHITE, font=font, anchor="lm")
    return canvas
