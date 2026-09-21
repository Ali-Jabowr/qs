# ADR-0018: The operator screen is a Tk window; OpenCV stays headless everywhere

**Status:** Accepted · 2026-09-21 · Ali (design Step 10). Supersedes the window half of ADR-0011;
Pillow still draws every piece of text.

## Context

ADR-0011 put the operator screen in an OpenCV window. Anomalib 2.6.2 requires
`opencv-python-headless` unconditionally (its PyPI metadata, checked 2026-09-21), and the headless
build has no window functions: `cv2.imshow` fails with "function not implemented". The two cannot
share an environment cleanly.

## Decision

The screen is a **Tk window** (`tkinter`, Tk 8.6 — part of Python, `python3-tk` already installed on
the laptop). Frames and overlays are composed with Pillow and shown through `ImageTk`. The main loop
calls `update()` once per tick, so Tk never owns the loop — the polled design of `11-interfaces.md`
§3 is unchanged. Key presses arrive as Tk events **with their own timestamps**, which makes the
300 ms rule for key presses exact rather than inferred. OpenCV is the headless build everywhere,
and does image processing only.

## Alternatives considered

- **Install Anomalib, then swap in the GUI build of OpenCV.** Keeps ADR-0011 as written, but `pip
  check` reports a broken requirement and any later install can quietly bring the headless build back.
  Fails NFR-12's fresh-clone requirement in spirit. Rejected.
- **Pygame.** An SDL window that works, and a new LGPL dependency for four things on a screen.
  Rejected.

## Consequences

- No dependency conflict, and no new dependency.
- The Qt LGPL obligation of the non-headless OpenCV wheels disappears (`15-model.md` §11).
- Tk needs a display: the station runs in a desktop session. Replay needs none.
