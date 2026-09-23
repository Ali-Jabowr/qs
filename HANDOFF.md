# HANDOFF: resume the GitHub build loop

**Written 2026-09-23.** This is for whichever model picks this up next. Read CLAUDE.md first, then this file.

## How the work runs

- **Board:** GitHub, repo `Ali-Jabowr/qs`, milestone #1 "Bench MVP (21–27.09)", tracking issue **#26**. `project.yml` is kept in step, but GitHub is primary.
- **Branch:** everything lands on `mvp/bench`. **Never merge to master**; Ali reviews and merges one PR himself.
- **Per plan:** `/to-spec` gives a `spec` issue, and `/to-tickets` gives `ticket` sub-issues with blocked-by links. Each ticket is built by one fresh implementer, in order. When a plan's tickets are done, **one** independent reviewer checks it and a fixer fixes the findings; repeat until PASS. The review goes in `reviews/REVIEW-<plan>.md`.
- **Labels:** `wayfinder` + `question` marks something that needs Ali's decision; the code is built on a cautious default meanwhile. `bench-pending` marks a check that can only be proved with real моти or hardware.
- **Commit style:** `<plan> #<N>: <summary>`, ending with `Co-Authored-By: <model> <noreply@anthropic.com>`. Commit and push in small green steps (`make check` passes each time). Agents have been getting killed by API errors.
- **Crash recovery:** if the tree has uncommitted changes when you start, a previous attempt crashed. Read `git diff`, keep what is sound, and continue.

## Where it stopped

HEAD is `626d852` plus this commit. `make check` passes (~340 tests).

| Plan | Spec | Tickets (in order) | State |
|---|---|---|---|
| P05b shoot tools | #27 | #31 ✅ #32 ✅ #52 ✅ #53 ✅ #34 ✅ · #33 (webcam-only, last) | Review round 1 FIX, all 6 fixed. **Next: review round 2** |
| P06 geometry | #36 | #39 → #40 → #41 | not started |
| P07 PatchCore | #42 | #45 → #46 → #47 → #48 → #49 → #50 | not started |
| P03 evidence store | #54 | #58 → #59 → #60 → #61 | not started |
| P04 pipeline | #68 | #69 → … → #75 | not started |
| P08 build/calibrate/evaluate | #76 | #80 → #81 → #82 → #83 | not started |
| P09 station | — | — | **spec not written** (an attempt was stopped before it created anything) |

Also done: #35 lock-test prep; ADR-0022 (bench capture by phone, amends ADR-0010); #25 lock test deferred until the webcam arrives.

## Next steps, in order

1. **P05b review round 2.** A fresh reviewer that didn't write the code reads `reviews/REVIEW-P05b.md`, checks F1–F6 are really fixed (commits 2f7aca0…626d852) and looks for regressions. It appends round 2 with a verdict. If FIX: fix it, then re-review. Stop after 3 rounds.
2. **P06:** #39, #40, #41, then review.
3. **P07:** #45 … #50, then review. Write the P09 spec alongside.
4. **P03** → **P04** → **P08** → **P09**, each followed by its review.
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
