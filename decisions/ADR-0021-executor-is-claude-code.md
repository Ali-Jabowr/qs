# ADR-0021: Claude Code executes the plans; OpenCode is withdrawn

**Status:** Superseded by ADR-0023 (2026-09-23) · Accepted 2026-09-22 · Ali. Supersedes ADR-0019.

## Context

P01 and P02 were executed by OpenCode on OpenRouter models (ADR-0019). Claude Code wrote each plan
and reviewed the result. On 2026-09-22, after P02 was merged, Ali asked:

> "continue the cycle but without delegating you do it or other claude model do it, don't delegate
> it to opencode"

What P01 and P02 showed:

- **Delegation was not cheap.** P01 cost $2.66, and 147 of its 159 model steps were spent coaxing
  `pip-compile` through a slow link. P02 needed a second round after review.
- **The expensive part was the handoff.** Each fix went through a written brief, a resumed session,
  and a second review of the same diff.
- **The week is tight.** Friday's shoot cannot be redone (ROADMAP). P05, P05b, P06 and P07 all have to
  land before it.

## Decision

- **Claude Code implements each plan itself.** It works on branch `exec/<plan-id>`, follows
  `AGENTS.md`, and writes `STATUS.md` as the executor always has.
- **The review stays independent.** One fresh Claude agent reviews the diff against the plan, with no
  memory of writing it. Claude Code then re-runs the checks, fixes the findings and writes
  `reviews/REVIEW-<plan-id>.md`.
- **Ali still approves every gate and merges every PR.** Plans are still written one ahead.
- `opencode.json` stays in the repo, unused, in case OpenCode is ever brought back. ADR-0019's privacy
  reasoning no longer applies: the code and docs stay inside the Claude Code session.

## Alternatives

- **Keep OpenCode on GLM-5.3 / Flash (ADR-0019).** Rejected by Ali.
- **Claude Code writes the code and reviews it alone.** Rejected: the executor would be grading its
  own work. One independent reviewer costs about one agent run per plan.

## Consequences

- Claude Code's own token use goes up. Plans stay tight, and there is still at most one reviewer
  agent per plan (memory: token economy).
- STATUS.md names the executor, so the record shows which plans OpenCode wrote (P01, P02) and which
  Claude Code wrote (P05 onwards).
- `STEPS.md` §2, `CLAUDE.md` §7 and §9, and `plans/ROADMAP.md` now name Claude Code as the executor.
