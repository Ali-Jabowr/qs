"""The operator screen: one Tk window, composed by Pillow, driven by the caller's loop.

`compose()` is a pure function, so what the window shows can be tested without a display.
`Screen.tick()` shows one composed image and pumps Tk once; Tk never owns the loop
(ADR-0018). Key presses are queued with Tk's own timestamps for the trigger's 300 ms rule.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from qs.config import DisplayConfig

_PANEL_PX = 80
_TEXT_PX = 22
_LINE_PX = 24
_MARGIN_PX = 10
_WINDOW = (1280, 800)


@dataclass(frozen=True, slots=True)
class KeyPress:
    """One key press: its Tk keysym and the Tk event's own timestamp in ms."""

    keysym: str
    tk_ms: int


def compose(
    image: np.ndarray | None,
    lines: Sequence[str],
    font_path: Path,
    size: tuple[int, int] = _WINDOW,
) -> Image.Image:
    """The screen as an RGB image: the BGR frame fitted above, three lines of text below."""
    width, height = size
    canvas = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), _TEXT_PX)
    area_h = height - _PANEL_PX
    if image is None:
        draw.text((width // 2, area_h // 2), "no camera image", fill="white", font=font,
                  anchor="mm")
    else:
        rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]), "RGB")
        scale = min(width / rgb.width, area_h / rgb.height)
        fitted = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        rgb = rgb.resize(fitted, Image.Resampling.LANCZOS)
        canvas.paste(rgb, ((width - fitted[0]) // 2, (area_h - fitted[1]) // 2))
    for i, line in enumerate(lines[:3]):
        draw.text((_MARGIN_PX, area_h + 4 + i * _LINE_PX), line, fill="white", font=font)
    return canvas


class Screen:
    """The Tk window: shows what `compose()` draws and queues key presses."""

    def __init__(self, display: DisplayConfig, title: str) -> None:
        self._font = Path(display.font)
        self._keys: list[KeyPress] = []
        self._closed = False
        self._root: tk.Tk | None = tk.Tk()
        self._root.title(title)
        self._root.geometry(f"{_WINDOW[0]}x{_WINDOW[1]}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<KeyPress>", self._on_key)
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill="both", expand=True)
        self._photo: ImageTk.PhotoImage | None = None

    def _on_key(self, event: tk.Event[tk.Misc]) -> None:
        self._keys.append(KeyPress(str(event.keysym), int(event.time)))

    def _on_close(self) -> None:
        self._closed = True

    def tick(self, image: np.ndarray | None, lines: Sequence[str]) -> None:
        """Show one composed frame and pump Tk's events once; nothing after `closed`."""
        if self._closed or self._root is None:
            return
        self._photo = ImageTk.PhotoImage(compose(image, lines, self._font, _WINDOW))
        self._label.configure(image=self._photo)
        self._root.update()

    def keys(self) -> list[KeyPress]:
        """Every key pressed since the last call, oldest first."""
        keys, self._keys = self._keys, []
        return keys

    @property
    def closed(self) -> bool:
        """True once the operator closed the window or `close()` was called."""
        return self._closed

    def close(self) -> None:
        """Destroy the window; safe to call twice."""
        self._closed = True
        if self._root is not None:
            root, self._root = self._root, None
            root.destroy()
