# Planner-Executor-Reviewer workflow

This project layers the Planner-Executor-Reviewer workflow on top of QS's
existing structure. QS's `CLAUDE.md` and `README.md` remain the source of
truth for project rules, architecture, and status — this file only covers
the plan/execute/review mechanics. See `STEPS.md` for the step-by-step
human workflow and `AGENTS.md` for the executor's instructions.

## WORKFLOW

1. Claude Code (Planner) writes plans into `/plans/<plan-id>/`.
2. A human reviews and approves the plan.
3. Claude Code (Executor, ADR-0021) implements the plan on a branch. P01 and P02 were run by OpenCode.
4. It writes `STATUS.md` and opens a PR.
5. A fresh Claude agent reviews the diff; Claude Code fixes the findings and writes
   `/reviews/REVIEW-<plan-id>.md`.
6. Human decides: merge, fix, or re-plan.

## PLAN.md FORMAT

```
# PLAN-<id>: <title>
## Objective
One paragraph.
## Inputs
- Files, schemas, fixtures the executor needs.
## Tasks
1. Numbered, specific, verifiable steps.
2. Each task names the file it creates or modifies.
3. No task requires judgment calls — pin every decision here.
## Constraints
- What the executor must NOT do.
## Definition of Done
- Executable checks (tests, linters, schema validation).
## Out of Scope
- Explicit exclusions.
```

## ACCEPTANCE.md FORMAT

```
# ACCEPTANCE-<id>
## Automated Checks
- [ ] `pytest tests/...` — 0 failures
- [ ] `ruff check src/...` — 0 errors
- [ ] `mypy src/...` — 0 errors
## Manual Checks
- [ ] Specific behavior the reviewer must confirm.
- [ ] Evidence requirements: every claim traceable to a source.
## Verdict
PASS only if every box is checked.
```

## RULES

- Never guess a business rule. Ask.
- Never write code in the plan. Write instructions for code.
- Every task must be verifiable. If it isn't, rewrite it.
- If there isn't enough context, stop and ask.
- Keep plans small. One plan = one module or one feature. Not the whole system.
- When in doubt, produce two alternatives and let the human choose.
- QS-specific standing rules (§6 of `CLAUDE.md`) apply to every plan without
  exception — no hardware purchases before the site visit, no FOD claims in
  a 95% acceptance clause, no face recognition, on-prem only, advisory-only
  in the MVP.
