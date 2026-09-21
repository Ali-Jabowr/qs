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

**The settings-lock test, first build day:** with the rig closed, set every control — exposure, gain, white balance, focus —
to manual, read each back and compare; then capture 50 frames over 5 minutes with the room light switched
on and off. Pass: every read-back equals the set value, and the reference patch reading varies by
less than 1%.

If the webcam fails, the phone becomes the source through snapshots. Its settings are recorded as
`unverified` (FR-13), and the connection to the phone is the one NFR-07 permits.

## Implementation facts — researched 2026-09-21

Sources: OpenCV `cap_v4l.cpp` (4.x), the kernel V4L2 camera-controls documentation, uvcvideo.

- **Values are raw V4L2**, because OpenCV's range normalisation is off by default: auto-exposure
  `1` = manual, `3` = auto. The 0.25 / 0.75 convention seen online is for normalised mode only.
- **Exposure is in 100 µs units.** 10 ms is `100`, 20 ms is `200`. The config holds microseconds;
  the source converts.
- **Order matters.** Switch each control to manual first, then set its value. An absolute value
  written while its auto control is on is ignored or refused.
- **Read-back proves acceptance, not effect.** OpenCV asks the driver every time, but uvcvideo may
  answer from its own cache. The grey-card half of the lock test is what proves the sensor applied it.
- **Set every control on every open.** Another program can reset them between runs.
- **Freshness uses the kernel timestamp.** `CAP_PROP_POS_MSEC` is the buffer's `CLOCK_MONOTONIC`
  stamp, the same clock as Python's `perf_counter`. That Python uses `CLOCK_MONOTONIC` was verified on
  this laptop; that uvcvideo stamps with it is from the driver source, confirmed at the lock test. `capture()` waits
  for a stamp later than `t0` plus one exposure time.
- **Format: uncompressed YUYV**, the highest resolution that gives at least 5 fps. USB 2.0 carries
  1080p YUYV at about 5 fps; 30 fps needs MJPG, whose compression artifacts are exactly the kind of
  texture noise PatchCore scores. One frame per unit does not need 30 fps. MJPG only if YUYV is
  not offered.
- **Flicker:** `power_line_frequency` set to 50 Hz as well as exposure at a multiple of 10 ms, and
  LED lamps at full brightness — dimmed LEDs flicker by PWM at other frequencies.
- **This laptop, today:** no camera enumerated, OpenCV and `v4l-utils` not installed. The first
  build day installs them; the lock test lists the webcam's real controls and formats.

## Alternatives considered

- **Phone first.** Better optics, but settings we cannot verify. That is exactly the failure R4
  describes. Rejected as primary.
- **Buy an industrial USB camera.** Gate C. Rejected.

## Consequences

- Settings are verifiable from the first frame.
- Webcam optics are modest. At the heights where it focuses, a моти still spans several hundred
  pixels, more than the model input needs.
- A failed lock test costs a morning and moves us to the fallback, not a redesign.
