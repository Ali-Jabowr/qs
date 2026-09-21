# ADR-0010: The bench camera is the USB webcam over V4L2; the phone is the fallback

**Status:** Proposed · 2026-09-21 · confirmed or reversed by the settings-lock test on the first build day

## Context

FR-13 requires exposure, white balance and focus to be set manually **and read back**; a camera
that re-exposes for the defect quietly ruins the dataset (R4). NFR-07 allows no connection except
to the station's own camera. Two cameras are owned and buying one is ruled out (Gate C).

- **USB webcam.** OpenCV over V4L2 exposes auto-exposure, exposure, auto white balance, white
  balance temperature, autofocus and focus as properties that can be set and read back.
- **Android phone.** It reaches the laptop either as HTTP snapshots from an IP-camera app, where
  settings are locked in the app and cannot be read back by us, over a network connection, or as a
  USB video mirror, where the camera runs on automatic settings.

## Decision

The bench `FrameSource` is `WebcamSource` over V4L2.

**The settings-lock test, first build day:** with the rig closed, set every control to manual,
read each back and compare; then capture 50 frames over 5 minutes with the room light switched
on and off. Pass: every read-back equals the set value, and the reference patch reading varies by
less than 1%.

If the webcam fails, the phone becomes the source through snapshots. Its settings are recorded as
`unverified` (FR-13), and the connection to the phone is the one NFR-07 permits.

## Alternatives considered

- **Phone first.** Better optics, but settings we cannot verify. That is exactly the failure R4
  describes. Rejected as primary.
- **Buy an industrial USB camera.** Gate C. Rejected.

## Consequences

- Settings are verifiable from the first frame.
- Webcam optics are modest. At the heights where it focuses, a моти still spans several hundred
  pixels, more than the model input needs.
- A failed lock test costs a morning and moves us to the fallback, not a redesign.
