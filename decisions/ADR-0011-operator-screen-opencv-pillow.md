# ADR-0011: The operator screen is one OpenCV window with text drawn by Pillow

**Status:** Superseded in part by ADR-0018 (2026-09-21): the window is Tk, not OpenCV, because
Anomalib requires headless OpenCV. The Pillow half of this decision stands.

## Context

FR-31 to FR-34 need a live view, a full-screen verdict readable at 3 m, a heatmap overlay and a
session panel. FR-32 requires Russian words — ГОДЕН / БРАК / НЕТ РЕШЕНИЯ. `10-architecture.md` §8
rules out a web framework. OpenCV's built-in fonts cannot draw Cyrillic.

## Decision

One full-screen OpenCV window. The main thread owns it and reads the keyboard through it, so the
keyboard trigger is fed by the same loop that draws the preview. Every piece of text is rendered
with Pillow using a bundled TTF font that covers Cyrillic (DejaVu Sans, a permissive licence),
then composited onto the frame.

## Alternatives considered

- **Qt (PySide6, LGPL).** Proper widgets, but a second event loop and a large dependency for four
  things on a screen. Rejected for the MVP; revisit if the factory screen grows controls.
- **A browser UI.** Needs a web server — excluded by §8. Rejected.
- **`cv2.freetype`.** Needs an OpenCV contrib build with FreeType, which is fragile to install.
  Rejected.

## Consequences

- No new framework; OpenCV and Pillow are already in the stack.
- One thread owns the screen and the keyboard, so there is no cross-thread UI code.
- Layout is drawn in pixels; restyling means changing code.
- The keyboard trigger is fed by the UI loop rather than being an independent iterator — a
  contract detail settled at Step 4.
