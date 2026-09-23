# ADR-0022: Bench capture by phone until the webcam arrives

**Status:** Accepted · 2026-09-23 · Ali. **Amends [ADR-0010](ADR-0010-bench-camera-usb-webcam.md)**
for the training shoots only; it does not supersede it. The webcam remains the station camera,
and ADR-0010 stays Proposed until its lock test runs (#25).

## Context

The bench webcam was ordered and has not arrived. The training shoot cannot wait for it: the
pilot and the MVP dataset are shot today (23.09) or tomorrow at the latest (ROADMAP), and the
моти are perishable (A-08). Ali's phone is the only camera in the house.

P05b (#27) built `qs split-plan` and `qs shoot` for a live V4L2 webcam: the shoot tool sets every
control, reads each back on every frame (FR-13), measures the grey card (FR-14), and names the
frames as it saves them. None of that reaches a phone. ADR-0010's phone fallback, HTTP snapshots
from an IP-camera app, needs a network connection to the phone and still gives settings we cannot
read back. That is the failure R4 describes, over a connection NFR-07 only barely allows.

A phone does write what it actually used into every photo's EXIF: exposure time, ISO,
white-balance mode, focal length and capture time. That is a per-frame read-back. It comes after
the shoot rather than during it.

## Decision

**The phone shoots offline; the laptop imports afterwards.** The phone is not a `FrameSource`.
There is no app on the laptop side and no network. The photos reach the laptop over a USB cable
after the session, and `qs import-phone` turns them into the same `raw/<session>/` that
`qs shoot` writes.

**Setup — the intent of ADR-0010, on a phone:**

- **Fixed mount.** The phone lies on the rig's lid, main lens straight down through the hole,
  taped at a marked position. It does not move between the pilot's end and the dataset's end
  (`docs/13` §3). Nothing is bought.
- **Manual or pro mode, everything locked:** exposure time a whole multiple of 10 ms (1/100 s or
  1/50 s, for 100 Hz flicker), ISO fixed, white balance a fixed preset or kelvin value, focus
  manual and set on the top of a unit.
- **Off:** HDR, night mode, AI scene or optimiser, beauty, portrait, motion or live photo,
  digital zoom and lens switching (main 1× lens only), and location tagging.
- **Format:** the highest-quality format the phone offers that is not HEIC: JPEG at maximum
  quality. Pillow reads no HEIC without a new dependency. DNG would need a raw decoder, which is
  also a new dependency, and a demosaic of our own choosing, so it stays a question for Ali
  (#51) with JPEG as the default.
- **Constant distance and framing.** The grey card and the placement marks are in view on every
  frame, as with the webcam.

**The lock test becomes an EXIF read-back, and it fails closed.** Every frame of a session must
carry the same exposure time, ISO, white-balance mode and focal length, with the white-balance
mode manual and no digital zoom. The same phone model and the same frame size are also required. A
missing tag counts as a difference. On any difference the import refuses the whole session, names
the photo and the field, and writes nothing into `raw/`. This is the analogue of the V4L2
read-back. The grey card, measured on every frame at import, stays the proof that the settings
held, as it is for the webcam (`docs/13` §4). **The phone's day-zero test:** 10 photos over 2
minutes with the rig closed and the room light switched off and on twice. It passes when the EXIF
lock holds on all 10 and the grey card varies by less than 1%.

**What EXIF cannot show is recorded as unverified.** EXIF gives the white-balance *mode*, not
its kelvin value, and usually no focus distance. `session.yml` therefore says
`camera.verified: false` (FR-13: a value that cannot be read back is `unverified`, never
assumed). The other §8 keys are kept: `device` is `phone:<make> <model>`, `exposure_us` is the
EXIF exposure, `gain` is the ISO, and `wb_kelvin` and `focus` are null.

**No image leaves the building (NFR-07, ADR-0016, T11).** Cloud photo backup — Google Photos,
the vendor's cloud, Yandex Disk auto-upload, any messenger's auto-save — is **off** before the
first test photo. It stays off for as long as any QS photo is on the phone, including when the
phone later holds the backup copy of `raw/`. It may come back on only after the QS photos are
deleted from the gallery and from its trash. The phone is in airplane mode while shooting. The
photos move to the laptop **over a USB cable only**, never through a messenger, e-mail, a share
sheet or a cloud folder.

**Same dataset layout as `qs shoot`.** `qs split-plan` plans the session exactly as before. Ali
follows a numbered, printed shot list: one line per photo, in the plan's order. `qs import-phone`
sorts the photos by EXIF capture time and maps photo *k* to shot *k*. It writes the rule-named
PNGs (`docs/12` §2), `frames.csv`, and a `session.yml` with exactly the `docs/13` §8 keys. It also
keeps the phone's files untouched in `raw/<session>/phone/`, with a manifest mapping each original
to its frame name and both hashes. If the counts do not match, the import refuses. A spoiled photo
is never deleted on the phone: it is retaken at once and marked on the paper list, and the import
moves the spoiled one to `rejected/` (A1, #28).

**One camera per session, and per dataset.** A session is never shot partly by phone and partly by
webcam. When the webcam arrives and passes its lock test, new sessions use `qs shoot`.

## Alternatives considered

- **Wait for the webcam.** Its arrival date is unknown, the week's plan fails with it, and the
  моти are perishable. Rejected.
- **ADR-0010's fallback: IP-camera snapshots.** A network connection to the phone, settings we
  cannot read back, and a new `FrameSource` to build today. Rejected.
- **A USB video mirror of the phone screen.** The camera runs on automatic settings (ADR-0010).
  Rejected.
- **A slate card per unit to anchor the mapping, instead of the photo's position.** It doubles
  the handling and needs someone to read it back. Rejected. The exact count, the EXIF time order
  and the time gaps at damage steps check the mapping instead.
- **Record the phone as `verified: true` because EXIF was read.** The kelvin value and the focus
  were not read. Rejected.

## Consequences

- **No kernel timestamps.** `grabbed_at` and `capture_ms` are empty, and the freshness check of
  ADR-0010 does not apply. Wall times come from the phone's clock. They still show that each test
  unit's good and damaged frames are minutes apart (ADR-0014).
- **The printed order guides the shoot, not the live screen.** There are no on-screen prompts, and
  no grey-card drift or verdict while shooting. An extra or missing photo shifts every name after
  it. The import catches that by count and time gaps, but only after the fact. The mitigation is
  a check mode that Ali runs mid-session, over USB, after the first control pass and again at the
  middle.
- **JPEG compression is in every frame.** It is in train and test alike, so the comparison is fair.
  Storing the frames as PNG keeps the dataset layout and makes every later read identical, but it
  does not remove the compression. That is one more way the phone frames differ from the station's
  uncompressed YUYV (ADR-0010, `docs/12` §2).
- **What is learned from phone frames belongs to the phone.** This covers the model, the
  thresholds, the pixel tolerances (ADR-0008) and the ROIs. None of it carries over to the webcam
  or the station. A webcam dataset needs its own shoot with the same tools. The live demo needs
  the webcam, or it runs in replay (FR-15).
- **The numbers stay INDICATIVE (ADR-0007).** The report names the phone as the source camera.
- ADR-0010, its lock test (#25) and the webcam path of `qs shoot` are unchanged and wait for the
  webcam. P05b gains `qs import-phone` and a printed shot list, and they come first. The recovery
  and pilot tickets (#33, #34) are re-scoped for this path.
- **New work for Ali:** a phone config (`camera.type: phone`, the phone's frame size, both ROIs in
  its pixels), the day-zero test, and the settings listed above. The shooting guide comes later.
