# REVIEW-P03, round 1: evidence store (schema, SqliteSink, outage, verification)

**Verdict: PASS.** 0 blocker · 0 major · 0 minor · 1 open gap (not in any ticket).

Spec #54, tickets #58 (e85912f), #59 (9e9ac3c), #60 (cc0391e) and #61 (7f66f1c). They were built by a
fresh Claude agent in its own worktree, per the ADR-0023 amendment, and merged into `mvp/bench` as
b53f78b. Reviewer: the Claude Code orchestrator, which wrote none of the code.

## What was checked

- `make check` on the merged tree: 585 passed, 11 deselected, ruff and strict mypy clean. The P03
  tests alone: 89 passed, the slowest 1.22 s. The agent re-ran the threaded tests 6 times in a row and
  4 times in parallel under load, with no flakes.
- **Append-only is a property of the file.** `BEFORE UPDATE/DELETE … RAISE(ABORT)` triggers exist on
  `inspection`, `inspection_check` and `override`. A test tries all 6 statements.
- **Files first.** `Store.write_record` writes the image and heatmap, then inserts the row and its
  checks in one transaction. A failure leaves at most orphan files, which `verify` reports. It never
  leaves a row without its files. A duplicate `(run_id, seq)` is refused before any file is written, so
  it can't overwrite the first record's image.
- **Durability.** The product keeps SQLite's default `synchronous`. The `synchronous = OFF` that the
  2 s limit needs is added only by a test helper, through `monkeypatch` on `qs.store.PRAGMAS`, and is
  restored after each test.
- **Never blocks, fails safe.** `write()` never raises and takes under 20 ms with the worker held. Rows
  are held during an outage, and images are shed past 200 queued in the order PASS, then NO_VERDICT,
  then DEFECT. `health()` stays not-ok until the backlog is written (docs/16 F11), so it is never
  optimistic. A stop gives up after `stop_wait_s` and counts what was lost.
- No network import. No `evidence/` folder was written in the repo. `docs/`, `config/` and the P02
  modules are unchanged.

## Conservative choices accepted

`health()` recovers only when the backlog is written. A reused run ID is treated as a storage
failure, never as licence to write into another run. Two measures with the same name in one check
are rejected. The `override` columns are NOT NULL. `verify(root, live_runs=…)`, so an ongoing run
doesn't fail it. The deviations are all test-side: a builder in `tests/evidence_records.py`, a scaled
slow-disk test, and a paced 300-record test.

## Open gap

- **G1:** docs/12 §3 describes a verification query that re-checks C8 across the two tables: the
  label agrees with its checks. No P03 ticket lists it, and `verify` does not do it. It belongs with
  P04, whose pipeline produces the label, or as a small follow-up ticket. It is not a defect in P03.

## Verdict

**PASS.** P03 is done.
