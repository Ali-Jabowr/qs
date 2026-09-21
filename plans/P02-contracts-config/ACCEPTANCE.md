# ACCEPTANCE-P02

## Automated Checks

- [ ] `make check` — 0 failures; ruff 0 errors; mypy strict 0 errors on `src/`
- [ ] `.venv/bin/pytest tests/test_contracts.py tests/test_config.py -q` — every case listed in
      CONTRACTS.md §4 present and passing
- [ ] `.venv/bin/python -c "from qs.config import load_config; from pathlib import Path; print(load_config(Path('config/station-A-bench.example.yml')).config_hash)"` — prints 64 hex characters
- [ ] `grep -nE '^(import|from) ' src/qs/contracts.py` — standard library and numpy only

## Manual Checks

- [ ] Field names and order in `contracts.py` match `docs/11-interfaces.md` §2 exactly.
- [ ] `derive_label` never returns NO_VERDICT, and never returns PASS with a configured check missing.
- [ ] Every validation message names the offending key.
- [ ] No file under `docs/` or `decisions/` was modified.

## Verdict

PASS only if every box is checked.
