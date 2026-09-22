# PLAN-P02: The `qs` package — contracts and the station config

## Objective

Turn the two documents every later plan builds on into typed, tested code: the contracts of
`docs/11-interfaces.md` (values, protocols, exceptions, and the rule that derives a label from
checks) and the station configuration of `docs/12-data-model.md` §4 (a strict loader and its
canonical hash). Nothing here touches a camera, a model or a disk beyond reading one YAML file.

## Inputs

- `CONTEXT.md`, `CONTRACTS.md` (this folder).
- `docs/11-interfaces.md` §2, §3, §7; `docs/12-data-model.md` §4; `docs/16-failure-model.md` §3.
- P01 done: `.venv`, `pyproject.toml`, `make check` with ruff and mypy.

## Tasks

1. **`src/qs/__init__.py`** — `__version__ = "0.1.0"`, nothing else. The file already exists, empty (P01 added it so mypy has a file to check; `reviews/REVIEW-P01.md`).
2. **`src/qs/contracts.py`** — every name in CONTRACTS.md §1, exactly as specified. Values are frozen,
   slotted dataclasses; protocols are `typing.Protocol`. Include `derive_label` (CONTRACTS §1.4).
3. **`src/qs/errors.py`** — the exception hierarchy in CONTRACTS.md §2.
4. **`src/qs/config.py`** — `StationConfig` as a tree of frozen dataclasses mirroring the YAML in
   CONTRACTS.md §3, `load_config(path) -> LoadedConfig`, and `canonical_hash(mapping) -> str`. Unknown
   keys, missing keys and wrong types are errors; there are no defaults.
5. **The validation rules** in CONTRACTS.md §3.3, each raising `ConfigError` with a message that names
   the key.
6. **`config/station-A-bench.example.yml`** — the example from `docs/12-data-model.md` §4, verbatim
   except for comments. It must load.
7. **`tests/test_contracts.py`** and **`tests/test_config.py`** — the cases in CONTRACTS.md §4.
8. **Run** `make check`; write `STATUS.md`, with the **actual output** of every ACCEPTANCE check pasted in, not summarised (`reviews/REVIEW-P01.md`).

## Constraints

- **No I/O in `contracts.py`**, no clock reads, no imports beyond the standard library and `numpy`.
- **No defaults in the config.** A missing key is a `ConfigError`, never a silent value (rule C6).
- **No new dependencies.** PyYAML and numpy are already locked.
- **Do not add implementations** of any protocol — no sources, sinks or inspectors. Later plans do.
- **Do not change `docs/` or `decisions/`.** If a contract in the docs is ambiguous, stop and report it.

## Definition of Done

- `make check` green: pytest, ruff, mypy strict on `src/`.
- Every case in CONTRACTS.md §4 exists as a test and passes.
- `STATUS.md` complete; no uncommitted changes.

## Out of Scope

- `InspectionRecord` encoding helpers (P04), the SQLite schema (P03), the Tk `Screen` (P05).
