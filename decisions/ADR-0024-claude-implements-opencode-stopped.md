# ADR-0024: OpenCode is stopped; Claude agents implement the remaining tickets

**Status:** Accepted · 2026-09-24 · Ali. Supersedes ADR-0023, including its 2026-09-23 amendment.

## Context

Under ADR-0023, OpenCode (`glm-5.3-flash` through OpenRouter) built the bounded tickets, and fresh
Claude agents built the risky ones. On 2026-09-24 OpenCode built #50, #80, #81 and #82, and the
P07 review rework, for about $2.25. Along the way:
- The P07 review found a blocker and a major defect in OpenCode's `qs latency`: Gate B was judged
  on the wrong series, and the warm-up was counted in the sample.
- One run stopped after reading the context and wrote nothing.
- It used `git stash` against its brief.

Mid-morning, Ali wrote:

> "stop opencode and continue yourself"

## Decision

- **No ticket goes to OpenCode.** Every remaining ticket (P08 #83, P09 #86–#91) and every review fix
  is implemented by a fresh Claude agent. Parallel work runs in its own git worktree and is merged
  into `mvp/bench` by the orchestrator.
- **The orchestrator (Claude Code) writes the briefs, reads each diff, re-runs `make check`, commits
  and merges.** It never reviews its own implementation: one independent Claude reviewer per plan
  writes `reviews/REVIEW-<plan>.md`.
- Work OpenCode had already finished and that passed review and the gates (#50, #80–#82) is kept.

## Alternatives

- **Keep the ADR-0023 split.** Rejected by Ali.
- **Rewrite OpenCode's tickets in Claude.** Not done: they passed an independent review (P07, P08)
  after their fixes, so rewriting them would add cost and no safety.

## Consequences

- No OpenRouter spending from here on. Claude session limits become the constraint: on 2026-09-24
  two agents were stopped by the session limit and resumed after it reset.
- The `opencode-delegate` route stays available, but only if Ali asks for it again.
