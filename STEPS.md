# How to Run This Project

## Setup (once)
Done by the init-project skill; layered onto the existing QS repo on 2026-09-21. The remote is a
private GitHub repository (`origin`). The default branch is `master`.

## For each new plan

### 1. Plan (Claude Code)
Claude Code writes `plans/<plan-id>/PLAN.md`, `CONTEXT.md`, `CONTRACTS.md`, `ACCEPTANCE.md`, links
the plan from its task in `project.yml`, and commits it on `master`. Plans are written one ahead of
execution (`plans/ROADMAP.md`).

```bash
git push origin master
```

### 2. Execute (OpenCode)
```bash
git checkout -b exec/<plan-id> master
```
Open OpenCode. Tell it: "Implement /plans/<plan-id>/."
Watch it work. Intervene if it goes off-track.

When done:
```bash
git push -u origin exec/<plan-id>
gh pr create --base master --head exec/<plan-id> \
  --title "exec: <plan-id>" --body "See plans/<plan-id>/STATUS.md"
```

### 3. Review (Claude Code)
Back in Claude Code:
"OpenCode finished plan <plan-id>. Read STATUS.md, review the diff,
run the ACCEPTANCE checks, write /reviews/REVIEW-<plan-id>.md."

### 4. Decide
- PASS → merge the PR into `master`; mark the plan's tasks done in `project.yml` with the merge
  commit as evidence; `python3 tools/render.py`; move to the next plan
- FAIL small → tell OpenCode to fix on the same branch
- FAIL big → rewrite the plan with Claude Code

## Rules
- One plan at a time.
- Never skip the human review.
- If it feels wrong, stop and re-plan.
- Never push `data/`, `evidence/`, `models/` or any cache — they are git-ignored; keep it that way.
