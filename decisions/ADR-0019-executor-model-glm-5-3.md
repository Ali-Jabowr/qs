# ADR-0019: The executor runs GLM-5.3 through OpenRouter, only on hosts that do not train on prompts

**Status:** Superseded by [ADR-0021](ADR-0021-executor-is-claude-code.md) · 2026-09-22 (was Accepted the same day; amended after P01 — see **Amendment** below). It governed P01 and P02.

## Context

OpenCode executes the plans (`STEPS.md` §2). Its only credential is OpenRouter, and no model was set.
Ali's limits, given 2026-09-22:

- **Budget:** $5–20 for the executor across P01–P09.
- **Data:** no provider that trains on prompts. The executor sends the repo's code, plans and docs to
  the model host, and the docs name the client and quote them verbatim (`CLAUDE.md` §4). Images never
  go: `data/` is git-ignored and outside every plan's scope so far.

The candidates, checked 2026-09-22. "Index" is the Artificial Analysis Intelligence Index, which they
run themselves; the price is the cheapest OpenRouter host that does not train, per 1M tokens.

| Model | Index | Cost per index task | In / out, $ | Note |
|---|---|---|---|---|
| **GLM-5.3** | 45 | $2.01 at Z.AI's list price | 0.66 / 2.06 (Morph) | 30+ hosts, most do not train |
| GLM-5.3-Flash | 42 | $0.25 | 0.075 / 0.25 (DeepInfra) | |
| MiMo-V2.6-Pro | 46 | $0.13 | 0.435 / 0.87 | One host (Xiaomi), keeps prompts 30 days; listed 2026-09-21 |
| Gemini 3.8 Flash (high) | 41 | $1.24 | 0.75 / 3.75 | |
| DeepSeek V4.1 Flash | 39 | $0.27 | 0.12 / 0.48 | DeepSeek's own API trains on prompts |
| GPT-5.6 Sol (max) | 47 | $1.99 | 2 / 10 | |
| Claude Opus 5 (high) | 48 | $3.61 | 5 / 25 | |

The scores are weak evidence. All 42 Terminal-Bench 2.1 results on llm-stats are self-reported;
Scale's SWE-bench Pro board lists none of these models; the Artificial Analysis index measures general
ability, not coding. Differences of a few points are noise.

Cost depends on caching. An agentic session re-sends its context on every turn — estimated at 5–7M
input tokens per plan, an assumption until P01 is measured. OpenRouter documents automatic caching for
Z.AI and a few first-party providers but says nothing about third-party hosts. For GLM-5.3 the week
comes to roughly $10–15 if caching works and $35–45 if it does not.

## Decision

- **Model:** `openrouter/z-ai/glm-5.3`. **Small model** (titles, summaries):
  `openrouter/z-ai/glm-5.3-flash`.
- **Routing, on every request:** `data_collection: "deny"`, `sort: "price"`; for GLM-5.3, 4-bit hosts
  are excluded (`quantizations: fp8, bf16, unknown`) as a precaution on tool-calling quality.
- **Skills off for the executor** (`permission.skill: deny`). OpenCode discovers Claude Code's skills
  (`~/.claude/skills`, `.claude/skills`); on this laptop that is 271 skills and ~15K tokens of
  descriptions on every request. The executor follows `AGENTS.md` exactly, and a process skill would
  pull it off the plan; some skills write outside the repo.
- All of this is in the repo's `opencode.json`, so it applies only inside QS and is reviewed like code.
- **Ali, once, on openrouter.ai:** turn off endpoints that may train on inputs (paid and free) in the
  privacy settings, and give OpenCode its own API key with a **$20 credit limit**.
- **The rule after P01:** read P01's cost and cached tokens on OpenRouter's Activity page and record
  them in `reviews/REVIEW-P01.md`. If P01 cost more than **$1.50**, or almost nothing was cached, the
  remaining plans run on GLM-5.3-Flash; P04 and P07 may stay on GLM-5.3 if the remaining budget
  covers them.

## Amendment — 2026-09-22, after P01

P01 cost **$2.66**, and **96.5%** of its input tokens came from cache (`reviews/REVIEW-P01.md`). The rule
fired on cost, not on caching: 147 of P01's 159 model steps went on coaxing `pip-compile` through a
link that fell to ~90 KiB/s. Ali chose the per-plan split below instead of the rule's wording, so
that the contracts every later plan builds on get the stronger model:

| Model | Plans |
|---|---|
| `openrouter/z-ai/glm-5.3` (the default in `opencode.json`) | P02, P04, P07 |
| `openrouter/z-ai/glm-5.3-flash` — start OpenCode with `-m openrouter/z-ai/glm-5.3-flash` | P03, P05, P05b, P06, P08, P09 |

Estimated at ~$10 for the week, inside the $20 key limit. Each review records the plan's cost.

## Alternatives considered

- **MiMo-V2.6-Pro.** Best score for the price on paper. Listed on OpenRouter the day before this
  decision, served by one host, no record in OpenCode. The shoot on 25.09 cannot slip, so the
  executor cannot depend on it. Worth a trial after the week.
- **GLM-5.3-Flash as the default.** Under ~$5 for the week either way, three index points lower.
  Kept as the fallback.
- **GPT-5.6 Sol, Claude Opus 5.** Stronger, and $30–75 for the week. Over budget.
- **Gemini 3.8 Flash.** Scores lower than GLM-5.3 and costs more per output token.
- **DeepSeek V4 family.** The first-party API trains on prompts; the other hosts are allowed, but the
  scores are lower.
- **Free models.** Most free endpoints may train on or publish prompts. Excluded by Ali's rule.

## Consequences

- The executor's quality on our plans is unmeasured; `REVIEW-P01` is the first measurement, and the
  review (`STEPS.md` §3) remains the safety net.
- OpenCode reads `AGENTS.md`, not the project `CLAUDE.md` (when both exist, `AGENTS.md` wins). The
  standing rules reach the executor only through each plan's constraints, as `plans/README.md`
  already requires.
- **Plans that touch `data/` (P05, P08, P09) must tell the executor never to open an image file**, so
  no image is sent to the model host.
- Prices and scores move monthly. This is a snapshot of 2026-09-22: re-check before the next build
  week rather than re-opening it during this one.

## Sources (checked 2026-09-22)

- OpenRouter model list, per-model endpoints and provider data policies:
  `https://openrouter.ai/api/v1/models`, `/api/v1/models/<id>/endpoints`,
  `https://openrouter.ai/api/frontend/v1/all-providers`
- OpenRouter docs: provider routing, prompt caching — `https://openrouter.ai/docs/features/`
- Artificial Analysis leaderboard — `https://artificialanalysis.ai/leaderboards/models`
- llm-stats, Terminal-Bench 2.1 — `https://llm-stats.com/benchmarks/terminal-bench-2.1`
- Scale, SWE-bench Pro — `https://labs.scale.com/leaderboard/swe_bench_pro_public`
- OpenCode docs: providers, rules, skills — `https://opencode.ai/docs/`
