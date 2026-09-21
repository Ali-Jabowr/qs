# CONTEXT — P02

## Why this plan comes second

Every later plan implements one of these protocols or reads this config. If the seams are wrong
here, they are wrong in nine places. `docs/11-interfaces.md` was written before any code, on
purpose: "swap the camera, not the program" is only true if the seams are written down first.

## The rules that matter most, and why

- **C8 — a label is derived, never chosen.** For a completed inspection: DEFECT if any check failed;
  PASS only if **every configured check** ran and passed. A PASS with a check missing behind it is a
  silent pass over open food — the one failure the design forbids (invariant I6). An exception inside
  `inspect()` never reaches `derive_label`; the Pipeline turns it into NO_VERDICT (P04).
- **C6 — every implementation is built from `StationConfig` alone.** That is why the loader has no
  defaults: a default is configuration that lives in code.
- **The config hash is canonical** (`12-data-model.md` §4): YAML parsed, dumped as JSON with sorted
  keys and no whitespace, UTF-8, SHA-256. Two files that differ only in key order, comments or
  spacing must hash the same; any change of value must change the hash. It goes on every record.
- **The validation rules encode decisions made elsewhere**, so a bad config fails at start-up
  instead of quietly ruining a shoot:
  - exposure a multiple of 10 ms under 50 Hz mains — flicker banding (`13-data-collection.md` §3);
  - the reference patch outside the localize ROI — or the Localizer finds the grey card as a unit;
  - `model` a full `model_id` with its hash (`15-model.md` §8);
  - the calibration's `set_on.model` equal to `inspector.model` and `set_on.backend` equal to
    `inspector.backend` — a threshold is never used with a model or backend it was not set on
    (`15-model.md` §9).

## What already exists

`pyproject.toml` (src layout, ruff, mypy strict on `src/`), `.venv` with numpy and PyYAML,
`make check`. `tools/` and its 44 tests are unrelated to `src/qs` and must keep passing.
