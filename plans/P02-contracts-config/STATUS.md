# STATUS — P02

**Status: COMPLETE** — every automated check in ACCEPTANCE.md passes, and every case in
CONTRACTS.md §4 exists as a passing test (47 of them). One forced deviation is flagged for
the reviewer below: the `display.labels.pass` field is named `pass_`, because `pass` is a
Python keyword and cannot be a dataclass field name.

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
