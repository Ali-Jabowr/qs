# STATUS — P02

**Status: COMPLETE** — every automated check in ACCEPTANCE.md passes, and every case in
CONTRACTS.md §4 exists as a passing test (60 of them after the round-2 rework; round 1 was 47).
Four deviations are documented below; the forced one is the `display.labels.pass` field named
`pass_`, because `pass` is a Python keyword and cannot be a dataclass field name. Round 1 was
reviewed in `reviews/REVIEW-P02.md` (FAIL, small); the round-2 rework is the last section.

Executed 2026-09-22 on `exec/P02` (from `cabd182`), GLM-5.3 via OpenRouter (ADR-0019).

## What was done, per task

1. **`src/qs/__init__.py`** — `__version__ = "0.1.0"`, nothing else (was the empty file P01 left).
2. **`pyproject.toml`** — `pythonpath = ["src"]` added under `[tool.pytest.ini_options]`; the only
   change to that file. The package is NOT installed into `.venv` (`pip list` untouched).
3. **`src/qs/contracts.py`** — the eight value records of `docs/11-interfaces.md` §2 with the doc's
   fields, types and order (verified mechanically — see Deviation-free checks below), all
   `frozen=True, slots=True`; `Label` a `StrEnum`; the five protocols of §3 with the doc's
   docstrings carried over; `StationConfig` imported under `TYPE_CHECKING` only (made safe at
   runtime by `from __future__ import annotations`); the constants of CONTRACTS §1.3; and
   `derive_label` per CONTRACTS §1.4. No I/O, no clock reads, imports stdlib + numpy only.
4. **`src/qs/errors.py`** — the CONTRACTS §2 hierarchy; every class except `QSError` and
   `ConfigError` carries its `reason`; `PlacementFault` takes `detail: str`.
5. **`src/qs/config.py`** — `StationConfig` as a tree of 14 frozen dataclasses mirroring the
   example YAML, field names identical to the keys (see Deviation 1 for `pass_`); ROIs as
   `tuple[int, int, int, int]`, `[min, max]` as `tuple[int, int]`; `load_config(path)` returning
   `LoadedConfig(config, config_hash, path)`; `canonical_hash(mapping)` = SHA-256 of
   `json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` in UTF-8,
   with YAML dates converted to ISO strings first; `yaml.safe_load` only. The loader is strict:
   unknown key, missing key, wrong type → `ConfigError` naming the full dotted key; no defaults.
   The six enumerations of CONTRACTS §3.1 are enforced at parse, each naming its key.
   The nine validation rules of CONTRACTS §3.3 are checked after parsing, in contract order,
   each raising `ConfigError` that names the offending key.
6. **`config/station-A-bench.example.yml`** — the YAML of `docs/12-data-model.md` §4 verbatim
   except that comments are dropped (including the header that names a different filename);
   key order, spacing and flow mappings are the doc's. It loads and validates.
7. **`tests/test_contracts.py`** (20 tests) and **`tests/test_config.py`** (27 tests) — every case
   in CONTRACTS §4: immutability of all seven records and of `Label`; the two `Verdict`
   refusals; all eight `derive_label` cases; the reason/REASONS equalities for the exception
   hierarchy; the example round-trip with type assertions; hash stability across key order,
   comments and indentation; one value changed → hash changed; one test per §3.3 rule (rule 1
   split into its four conditions), each asserting the key named in the message; missing and
   unknown keys refused. Four small tests beyond the §4 list cover CONTRACTS §3.2 behaviour
   that has no other coverage: `canonical_hash` date conversion, key-order insensitivity and
   64-hex form, the positive case of rule 2 (a multiple of 10 ms loads), and an unreadable
   file raising `ConfigError`.
8. **Runs** — actual output of every ACCEPTANCE check pasted below.

## Test results (actual output)

`make check`:

```
== project state ==
warning 7 client question(s) are open but have never been sent: OQ-1, OQ-2, OQ-3, OQ-4, OQ-5, OQ-6, OQ-9. Every one of them blocks a decision.
project.yml is valid. 1 warning(s).
== tests ==
........................................................................ [ 79%]
...................                                                      [100%]
91 passed, 5 deselected in 0.38s
== diagrams ==
  ok    10-architecture-1      (docs/10-architecture.md)
  ok    10-architecture-2      (docs/10-architecture.md)
  ok    10-architecture-3      (docs/10-architecture.md)
  ok    10-architecture-4      (docs/10-architecture.md)
  ok    10-architecture-5      (docs/10-architecture.md)
  ok    10-architecture-6      (docs/10-architecture.md)
  ok    10-architecture-7      (docs/10-architecture.md)
  ok    10-architecture-8      (docs/10-architecture.md)
  ok    11-interfaces          (docs/11-interfaces.md)
  ok    12-data-model          (docs/12-data-model.md)
  ok    14-eval-protocol       (docs/14-eval-protocol.md)
  ok    15-model-1             (docs/15-model.md)
  ok    15-model-2              (docs/15-model.md)
  ok    16-failure-model-1      (docs/16-failure-model.md)
  ok    16-failure-model-2      (docs/16-failure-model.md)
  ok    17-security            (docs/17-security.md)
  ok    timeline               (docs/diagrams/timeline.mmd)

17 diagram(s) render.
== lint ==
make[1]: Entering directory '/home/marxel/PROJECTS/QS'
.venv/bin/ruff check src tools tests
All checks passed!
.venv/bin/mypy
Success: no issues found in 4 source files
make[1]: Leaving directory '/home/marxel/PROJECTS/QS'
```

The 91 are the 44 pre-existing `tools/` tests plus the 47 new ones; the project-state warning
about the seven open client questions predates P02 and is not P02's to touch.

`.venv/bin/pytest tests/test_contracts.py tests/test_config.py -q`:

```
...............................................                          [100%]
47 passed in 0.23s
```

`PYTHONPATH=src .venv/bin/python -c "from qs.config import load_config; from pathlib import Path; print(load_config(Path('config/station-A-bench.example.yml')).config_hash)"`:

```
cd1b5c35ec61a26fd5c58b64ce941591797ca4d5d9f1aaec362f6dc044b425e7
```

64 hex characters.

`grep -nE '^(import|from) ' src/qs/contracts.py`:

```
1:from __future__ import annotations
3:from dataclasses import dataclass
4:from datetime import datetime
5:from enum import StrEnum
6:from typing import TYPE_CHECKING, Protocol
8:import numpy as np
```

Standard library and numpy only.

## Manual checks

- **Field names and order match `docs/11-interfaces.md` §2 exactly** — verified mechanically with
  `dataclasses.fields` against a table transcribed from the doc; all seven records match, and
  `Label`'s values are `pass`, `defect`, `no_verdict`.
- **`derive_label` never returns NO_VERDICT, and never PASS with a configured check missing** —
  by construction it can only return `Label.DEFECT` or `Label.PASS` or raise; the
  missing-check case is `ValueError` and is tested.
- **Every validation message names the offending key** — each §3.3 test asserts the dotted key
  in the message via `pytest.raises(..., match=re.escape(key))`.
- **No file under `docs/` or `decisions/` was modified** — `git status --porcelain docs decisions`
  is empty.

## Deviations from the plan (with reason)

1. **`display.labels.pass` is the field `pass_`.** `pass` is a Python keyword and cannot be a
   dataclass field name, so CONTRACTS §3.1's "field names identical to the keys" is impossible
   for this one key. Every other field of every other mapping is identical to its YAML key; the
   YAML key itself remains `pass`, and error messages name it as `display.labels.pass`. The
   alternative — storing the labels as a mapping — would have broken the stronger requirement
   of a tree of frozen dataclasses. **Flagged as needing review.**
2. **Rule 4 treats a non-positive ROI extent as outside the frame.** "Any ROI outside
   `camera.width` × `camera.height`" does not say what to do with `[x, y, 0, 0]` or a negative
   width; a degenerate ROI cannot be a region inside the frame, so it is refused under the
   same key. Documented here because the contract is silent on it.
3. **Float fields accept YAML integers and convert.** `axis_ratio_min: 1` parses as int 1 in
   YAML; refusing it would make the inclusive upper bound of (0, 1] unrepresentable. Ints are
   widened to float, per the PEP 484 numeric tower.
4. **Four tests beyond the §4 list** (see task 7) — they cover behaviour CONTRACTS §3.2 specifies
   (date conversion, order insensitivity) and load_config's file-level error, which the §4 list
   has no other way to exercise.

No code was written beyond the plan's scope; no protocol has an implementation.

## Blockers

None.

## Round 2

The round-1 review (`reviews/REVIEW-P02.md`) found the config able to **fail open** — a NaN
or infinite `threshold.value` loaded — and amended CONTRACTS.md (rules 2, 6 and 9 widened,
rules 10 and 11 added, `derive_label` fails closed, the reason test walks the hierarchy).
All six rework items are implemented; nothing from the "Not required" list was touched.

### What changed, per rework item

1. **Validation, `config.py`** — rule 2 also refuses `camera.exposure_us <= 0` (checked
   before the 50 Hz multiple, since a non-positive exposure is the more fundamental fault);
   rule 6 also compares `threshold.set_on.localize_threshold` with
   `inspector.localize.threshold` and `threshold.set_on.mask_dilate_px` with
   `inspector.mask_dilate_px` — a threshold is never used with a different localisation
   (`docs/12-data-model.md` §4); rule 10 refuses a `threshold.value` that is not finite or
   not above 0 (`math.isfinite`), so a NaN or infinite threshold can no longer make every
   appearance check pass; rule 11 refuses a key repeated within one YAML mapping, naming
   the full dotted key.
2. **Rule 9** — a whitespace-only label text is refused: "must not be empty or
   whitespace-only".
3. **`derive_label`, `contracts.py`** — `check.passed is not True` instead of truthiness,
   so a non-bool `passed` (`1`, `"no"`, `None`) is a DEFECT, never a PASS. The docstring was
   updated to say so.
4. **`errors.py`** — every `reason` is now annotated `reason: str`, and the reason tests
   walk `QSError`'s subclasses recursively instead of the hand-written list, so a new error
   class without a `reason` fails them.
5. **Tests** — rule 4 for `inspector.localize.roi`; rule 6 for each of its four
   comparisons; rule 10 parametrised over NaN, infinity, 0 and −1; rule 11 for a nested and
   a top-level duplicate; plus the new arms of rules 2 (`exposure_us: 0`) and 9
   (whitespace-only) that the amended §4's "one test per validation rule" requires. `_refused`
   now takes the match string, and the rule 2, 3 and 4 tests match a distinguishing fragment
   of the message, so each test can tell which rule fired (round-1 test note). `config.py`
   and `errors.py` gained the module docstring and `from __future__ import annotations`.
6. **This section**, and the header above now says four deviations, not one.

The example YAML, `canonical_hash` and the dataclass tree are untouched — the rework adds
validation only, and the example still hashes to `cd1b5c35…`.

### Test results (actual output)

`make check`:

```
== project state ==
warning 7 client question(s) are open but have never been sent: OQ-1, OQ-2, OQ-3, OQ-4, OQ-5, OQ-6, OQ-9. Every one of them blocks a decision.
project.yml is valid. 1 warning(s).
== tests ==
........................................................................ [ 69%]
................................                                         [100%]
104 passed, 5 deselected in 0.46s
== diagrams ==
  ok    10-architecture-1      (docs/10-architecture.md)
  ok    10-architecture-2      (docs/10-architecture.md)
  ok    10-architecture-3      (docs/10-architecture.md)
  ok    10-architecture-4      (docs/10-architecture.md)
  ok    10-architecture-5      (docs/10-architecture.md)
  ok    10-architecture-6      (docs/10-architecture.md)
  ok    10-architecture-7      (docs/10-architecture.md)
  ok    10-architecture-8      (docs/10-architecture.md)
  ok    11-interfaces          (docs/11-interfaces.md)
  ok    12-data-model          (docs/12-data-model.md)
  ok    14-eval-protocol       (docs/14-eval-protocol.md)
  ok    15-model-1             (docs/15-model.md)
  ok    15-model-2              (docs/15-model.md)
  ok    16-failure-model-1      (docs/16-failure-model.md)
  ok    16-failure-model-2      (docs/16-failure-model.md)
  ok    17-security            (docs/17-security.md)
  ok    timeline               (docs/diagrams/timeline.mmd)

17 diagram(s) render.
== lint ==
make[1]: Entering directory '/home/marxel/PROJECTS/QS'
.venv/bin/ruff check src tools tests
All checks passed!
.venv/bin/mypy
Success: no issues found in 4 source files
make[1]: Leaving directory '/home/marxel/PROJECTS/QS'
```

The 104 are the 44 pre-existing `tools/` tests plus 60 new ones (20 contracts, 40 config);
the project-state warning about the seven open client questions predates P02.

`.venv/bin/pytest tests/test_contracts.py tests/test_config.py -q`:

```
............................................................             [100%]
60 passed in 0.39s
```

`PYTHONPATH=src .venv/bin/python -c "from qs.config import load_config; from pathlib import Path; print(load_config(Path('config/station-A-bench.example.yml')).config_hash)"`:

```
cd1b5c35ec61a26fd5c58b64ce941591797ca4d5d9f1aaec362f6dc044b425e7
```

Unchanged from round 1, as required — the rework adds validation only.

`grep -nE '^(import|from) ' src/qs/contracts.py`:

```
1:from __future__ import annotations
3:from dataclasses import dataclass
4:from datetime import datetime
5:from enum import StrEnum
6:from typing import TYPE_CHECKING, Protocol
8:import numpy as np
```

Standard library and numpy only.

### New deviations

1. **Rule 11's mechanism.** CONTRACTS §3.2 says `load_config` uses `yaml.safe_load` only,
   but duplicate keys cannot be detected through `safe_load` — PyYAML silently keeps the
   last value. The loader therefore composes the document once with
   `yaml.compose(text, Loader=yaml.SafeLoader)` and walks the node graph for repeated keys
   before the data itself is built — still by `yaml.safe_load` alone. The same `SafeLoader`
   does the composing, so safety is unchanged, and both rules of the amended contract hold.
2. **Rule 11 compares keys by their YAML text.** Two keys with the same text but different
   tags (`1` and `'1'`) would construct to different Python keys, yet are refused — failing
   closed. The station schema has no such keys.

### Round-2 blockers

None.
