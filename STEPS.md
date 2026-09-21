# How to Run This Project

## Setup (once)
Done by the init-project skill.

## For each new plan

### 1. Plan (Claude Code)
Open Claude Code in this folder. Give it the task. It writes
/plans/<plan-id>/PLAN.md, CONTEXT.md, CONTRACTS.md, ACCEPTANCE.md.

```bash
git checkout -b plan/<plan-id>
git add plans/<plan-id>/
git commit -m "plan: <plan-id>"
git push origin plan/<plan-id>
```

### 2. Execute (OpenCode)
```bash
git checkout -b exec/<plan-id> plan/<plan-id>
```
Open OpenCode. Tell it: "Implement /plans/<plan-id>/."
Watch it work. Intervene if it goes off-track.

When done:
```bash
git push origin exec/<plan-id>
gh pr create --base plan/<plan-id> --head exec/<plan-id> \
  --title "exec: <plan-id>" --body "See STATUS.md"
```

### 3. Review (Claude Code)
Back in Claude Code:
"OpenCode finished plan <plan-id>. Read STATUS.md, review the diff,
run the ACCEPTANCE checks, write /reviews/REVIEW-<plan-id>.md."

### 4. Decide
- PASS → merge exec/<plan-id> into main, move to next plan
- FAIL small → tell OpenCode to fix on same branch
- FAIL big → rewrite the plan with Claude Code

## Rules
- One plan at a time.
- Never skip the human review.
- If it feels wrong, stop and re-plan.
