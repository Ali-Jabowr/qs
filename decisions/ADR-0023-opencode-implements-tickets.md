# ADR-0023: OpenCode implements the tickets again; Claude Code reviews and commits

**Status:** Accepted · 2026-09-23 · Ali. Supersedes ADR-0021. Its model split is ADR-0019's.

## Context

ADR-0021 took OpenCode out on 2026-09-22, and Claude Code wrote P05 and P05b. On 2026-09-23, with
P06, P07, P03, P04, P08 and P09 still to build, Ali asked:

> "pick up from where we left off but use /opencode-delegate And here we'll review the code and
> that's it."

## Decision

- **OpenCode implements** each GitHub ticket, one at a time, on `mvp/bench`. It runs through the
  `opencode-delegate` skill (`relay.mjs`, `build` agent). It does not commit or touch GitHub.
- **Claude Code writes the brief, reviews the diff, re-runs `make check`, and commits**, as
  `<plan> #<N>: …`. Rework goes back to the same OpenCode session with a delta brief. The plan-level
  review in `reviews/REVIEW-<plan>.md` stays Claude's.
- **Models, as ADR-0019 set them** (Ali, 2026-09-23): `openrouter/z-ai/glm-5.3` for P07 (PatchCore),
  and P02/P04 if they are redone. `openrouter/z-ai/glm-5.3-flash` for P03, P05b, P06, P08 and P09.
  `opencode.json`'s privacy settings apply (`data_collection: deny`).
- `AGENTS.md` still describes the `plans/`, STATUS.md and commit flow. Each brief overrides it: no
  STATUS.md, no commit.

## Alternatives

- **Keep ADR-0021 (Claude Code writes the code).** Rejected by Ali.

## Consequences

- OpenRouter bills per token. P01 cost $2.66 under OpenCode (ADR-0021). The cost of each run is in its
  `result.json`.
- The review is now structural: the model that writes the code is never the one that judges it.
