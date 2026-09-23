# HANDOFF: resume the GitHub build loop

**Written 2026-09-23, updated the same evening.** This is for whichever model picks this up next. Read CLAUDE.md first, then this file.

## How the work runs

- **Board:** GitHub, repo `Ali-Jabowr/qs`, milestone #1 "Bench MVP (21–27.09)", tracking issue **#26**. `project.yml` is kept in step, but GitHub is primary.
- **Branch:** everything lands on `mvp/bench`. **Never merge to master**; Ali reviews and merges one PR himself.
- **Per plan:** `/to-spec` gives a `spec` issue, and `/to-tickets` gives `ticket` sub-issues with blocked-by links. Each ticket is built by one fresh implementer, in order. When a plan's tickets are done, **one** independent reviewer checks it and a fixer fixes the findings; repeat until PASS. The review goes in `reviews/REVIEW-<plan>.md`.
- **Labels:** `wayfinder` + `question` marks something that needs Ali's decision; the code is built on a cautious default meanwhile. `bench-pending` marks a check that can only be proved with real моти or hardware.
- **Commit style:** `<plan> #<N>: <summary>`, ending with `Co-Authored-By: <model> <noreply@anthropic.com>`. Commit and push in small green steps (`make check` passes each time). Agents have been getting killed by API errors.
- **Crash recovery:** if the tree has uncommitted changes when you start, a previous attempt crashed. Read `git diff`, keep what is sound, and continue.

## Where it stopped (2026-09-23 evening, HEAD a519676)

`make check`: 674 passed. Everything is pushed. Nothing is running. The main checkout must be on
`mvp/bench`: at 22:08 it had been switched to `master`, so check `git branch --show-current` first.

**The split is Ali's standing rule. Keep it (ADR-0023 amendment):**
- **A fresh Claude agent** implements the risky tickets: #49, P04 #74–#75, and anything on the
  safety path. Parallel work goes in its own worktree (`isolation: worktree`, and it must first run
  `git merge --ff-only mvp/bench`). The orchestrator merges it.
- **OpenCode**, through `/opencode-delegate` with `--model openrouter/z-ai/glm-5.3-flash`, implements
  the bounded tickets: #48, #50, P08 #80–#83, P09 #86–#91. Each brief pastes the ticket and its spec,
  overrides AGENTS.md (no STATUS.md, no commit), and says to write in small steps: flash hits its
  output limit otherwise, and `--resume-last` recovers it. A ticket moves up to `glm-5.3` only if
  flash fails review twice.
- **The orchestrator** writes briefs, re-runs `make check`, reads each diff, commits `<plan> #N: …`,
  pushes, comments on the ticket and removes `ready-for-agent`. It writes no product code itself.

| Plan | Spec | State |
|---|---|---|
| P05b shoot tools | #27 | review r2 PASS. Only #33 is left (webcam only, last) |
| P06 geometry | #36 | #39 #40 #41 done, `reviews/REVIEW-P06.md` PASS |
| P07 PatchCore | #42 | #45 #46 #47 done. **Next: #48 (OpenCode) and #49 (Claude, worktree), in parallel**, then #50 (OpenCode). Then the P07 review |
| P03 evidence store | #54 | #58–#61 done, `reviews/REVIEW-P03.md` PASS (gap G1: `verify` doesn't re-check C8 across the tables) |
| P04 pipeline | #68 | #69–#73 merged (a519676), reviewed. #74 needs #49; then #75; then `reviews/REVIEW-P04.md` |
| P08 build/calibrate/evaluate | #76 | #80–#83 not started (OpenCode) |
| P09 station | #84 | tickets #86–#91 written, question #85 (A1: replay until the webcam arrives). Not started (OpenCode) |

**Loose ends:**
- Post the ticket comments for #47 and #69–#73. The code is committed: #47 is f624df0, and P04 #69–#73 are 339f66b, 4705947, 2c06b58, 31d7d83 and e281221, merged in a519676.
- Write the exact commands into docs/18 Parts E and G:
  `bin/qs pilot-report --config <cfg> --session data/raw/<pilot-session>`
  `bin/qs train --config <cfg> --dataset data/datasets/mochi/<G> --window-px <N> --product mochi`

**Cost so far:** OpenRouter $2.07. Claude agents: about 1.05M tokens.

## Next steps, in order

1. The loose ends above.
2. #48 (OpenCode) and #49 (Claude worktree) in parallel. Then #50, then the P07 review.
3. P04 #74 → #75 (Claude), then the P04 review.
4. P08 #80 → #83 (OpenCode), then the review. P09 #86 → #91 (OpenCode), then the review.
5. At the end: `project.yml` statuses, a summary comment on #26, and one PR `mvp/bench → master`. **Do not merge it.**

## Implementer brief (paste for each ticket)

> Implement ticket #N in /home/marxel/PROJECTS/QS, branch `mvp/bench`. Read CLAUDE.md, AGENTS.md, the ticket (`gh issue view N --comments`), its parent spec, and the docs/ADRs they cite. Follow `~/.agents/skills/implement/SKILL.md` and `~/.agents/skills/tdd/SKILL.md`. Skip their /code-review step, because one reviewer checks each plan. Build only this ticket's scope and follow CLAUDE.md §6. On a real design gap: take the conservative option, note it, and open a `wayfinder`+`question` issue. `make check` must be green before every commit; commit and push in small steps. Comment on the ticket with what was built, the test for each acceptance check, anything `bench-pending`, and the sha. Remove `ready-for-agent`, and don't close the ticket. Retry gh up to 5 times.

## Spec brief (for P09)

> Turn plan P09 (ROADMAP row: ScreenAnnunciator, KeyboardTrigger, live `qs station`, NFR-01 measured, non-author test NFR-09; sources docs/11 §3, docs/16 §5, ADR-0018, docs/06) into a spec issue + tickets. Follow `~/.agents/skills/to-spec` and `to-tickets`, and mirror spec #76 and its tickets. It depends on P05 (built) and P08 (#76). The webcam hasn't arrived (ADR-0022), so live-camera checks are `bench-pending`; say whether the station runs on replay or a phone session until then. When editing #26, fetch its body fresh; never write it empty.

## Since the handoff (2026-09-23)

- **Guide:** `docs/18-shoot-and-train-guide.md` is Ali's step-by-step for the phone and the lean size. When #46, #47 and #80–#83 are built, **write their exact commands and flags into its Parts E and G.**
- **OQ-8 decided: lean** (#10 closed). The phone is the Xiaomi, in Pro mode, JPEG.
- **Build order serves the shoot:** P06 #39–#41 and P07 #45–#47 first, because `qs pilot-report` decides the lighting before the dataset shoot.

## Waiting on Ali (none of these block the code)

- **Phone shoot decisions:** #51 (Xiaomi chosen; the rest to confirm). Wayfinder questions: #28–#30, #37, #38, #43, #44, #55–#57, #62, #63, #77–#79.
