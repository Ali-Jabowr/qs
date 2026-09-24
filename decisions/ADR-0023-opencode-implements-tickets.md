# ADR-0023: OpenCode implements the tickets again; Claude Code reviews and commits

**Status:** Superseded by ADR-0024 (2026-09-24) · was accepted 2026-09-23 · Ali. Supersedes ADR-0021. Its model split is ADR-0019's.

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

## Amendment: the work is split between Claude and GLM (2026-09-23, Ali)

Ali's Claude limits reset, and he wants to keep GLM budget for another project. The split is by risk:

- **A fresh Claude agent implements** the tickets where a quiet bug gives a wrong ГОДЕН or a broken
  record: #46 (PatchCore, bundle, `qs train`), #49 (`AppearanceInspector`), P03 #58–#61 (evidence
  store), and P04 #69–#75 (pipeline). The orchestrator reviews each one, so the author is never the
  reviewer.
- **OpenCode implements** the bounded tickets, on `glm-5.3-flash`: #47, #48, #50, P08 #80–#83, and
  the P09 tickets. #45 had already started on `glm-5.3`. A ticket moves up to `glm-5.3` only if flash
  fails review twice.
- P03 is built in parallel, in its own git worktree, while P07 runs on `mvp/bench`. It is merged into
  `mvp/bench` once it is reviewed.
- Estimated OpenRouter cost for the rest drops from about $18 to about $6.
