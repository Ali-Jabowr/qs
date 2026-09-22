# CLAUDE.md — read this first, every session

**Project:** QS — automated visual quality inspection for a food production line.
**Owner:** Ali (Volgograd). A student friend handles all grant/legal/document work; Ali's job with Claude is **building the MVP**.
**Repo:** `/home/marxel/PROJECTS/QS` · **Planning vault:** `~/Documents/Obsidian Vaults/Second Brain/20 - AREAS/STARTUP/`
**Last updated:** 2026-09-21 — the design day: Gates 0, 1 and 2 passed.

---

## 1. What this is

One factory client in Volgograd, food producer, **verbal intent to pay**. They want every unit checked on the line before packing. There is no signed contract and no money has moved.

**The product is моти (mochi)** — soft rice-dough rounds, dusted with starch, with a filling. The owner said so on a call on 2026-09-20; it is the first product fact that did not come from our own survey text.

There is a parallel, **parked** track: a "QC Vision Box" for Wildberries/Ozon marketplace sellers. It is not being built (ADR-0003).

## 2. Where everything lives

```
QS/
  CLAUDE.md              ← this file. The cold-start anchor.
  project.yml            THE state: stages, tasks, gates, questions, risks. STATUS.md is rendered from it.
  AGENTS.md · STEPS.md   the executor's rules · the plan → execute → review loop
  docs/05-concept.md     Phase 0 — PR/FAQ, the MVP, Gate 0
  docs/06-requirements.md Phase 1 — numbered FR/NFR with checks, Gate 1
  docs/10–17             Phase 2 — architecture, contracts, data model, data collection,
                         evaluation, model, failure model, security
  docs/01–04             history (survey kit, hardware tree, survey answers, old build plan)
  decisions/             ADR-0001 … ADR-0021; README.md is the index
  reviews/GATE-2.md      the design review, with the requirement → design → check matrix
  plans/ROADMAP.md       nine executor plans and the week; plans/Pxx-*/ one folder per plan
  src/ · tests/ · tools/ code (from P02) · tests · state renderer, bounds, diagram checker
  data/                  git-ignored. Raw shoots, datasets, golden sets
```

## 3. Status as of 2026-09-21

| | |
|---|---|
| Design | **Complete.** Gate 0 (concept), Gate 1 (requirements), Gate 2 (independent design review, 26 findings fixed) — all GO |
| Code | Tools and 44 tests only. `src/qs` starts with plan P02 |
| Hardware bought | **None.** Correct. |
| Site visit | Has not happened. Client work is **deferred by decision for the MVP week** (21–27.09) |
| Physical evidence | None from the factory. The bench will use store-bought моти |
| Acceptance number | **95%** (client, verbatim, Q28) |
| Commercial model | hardware + updatable SaaS + annual calibration, **priced by us**, no competing bid (Q26) |
| This week | **The bench MVP** — `plans/ROADMAP.md`. Shoot on Fri 25.09 |

## 4. What the client actually told us

- **Product: моти** (owner, call of 2026-09-20). Chilled or room temperature is Ali's best knowledge, not confirmed (A-01).
- Inspection point: **before the packing zone**, on **open product**. Green light = OK to pack.
- Defects wanted: **dents · shape deviation · dimension deviation · foreign objects**
- **One person** does this inspection today. One more person checks marking.
- Escapes cause **refunds**. Rate quoted as "2%–10%, consider 2% per day". **No ruble figure given.**
- Marking / date code is on a **different conveyor**. **A batch was returned because of marking** (Q21).
- Ethernet and 220 V at the point. Photo archive per batch: yes.

Everything else in the survey sheet — line speed, field of view, sizes, environment, lighting — is **our own text in the customer's form**. The "up to 400 mm" height was ours.

## 5. The three technical facts that drive every decision

1. **Size and shape are measured, not learned** (ADR-0008). A моти is ~3–5 cm tall, so the old 400 mm perspective problem is gone. Two of the four defect types — shape and dimension — are global properties a patch-level anomaly model cannot see, so a deterministic geometry check (area, axis ratio, solidity, against good units, in pixels) runs beside PatchCore. Millimetres with a tolerance stay **OQ-1**.

2. **Open product before packing.** Anything over open food enters their ХАССП plan — food-grade materials, shatter-protected lamps, cleanable surfaces, sign-off by their food-safety officer (OQ-6). Starch dust in the air makes the positive-pressure air purge more important, not less.

3. **"Foreign objects" is the dangerous line item.** A camera sees the top surface only. **Never accept FOD inside a 95% acceptance clause.** Scope it: "surface-visible contrasting objects, best effort, not a substitute for metal detection or X-ray." The anomaly map is masked to the unit (ADR-0009), so a crumb beside it is ignored by design.

## 6. Standing rules — do not violate

- **Nothing gets bought before the site visit produces video and samples** — except the product being inspected. On 2026-09-21 Ali approved no other purchase, not even a backup stick: raw images back up to the phone over USB.
- **Never itemize hardware in a quote.** Price the outcome.
- **Never suggest identifying individuals.** No face recognition, no re-identification. Hard constraint — 152-ФЗ, and a sales line.
- **On-prem only. No image leaves the building.** At runtime the station connects to its own camera and nothing else (NFR-07, ADR-0016). Images, models and evidence are git-ignored and never pushed.
- **Advisory first.** Operator alert, never automatic rejection. No PLC, no actuator.
- **Never build on hardware that cannot be reliably re-ordered.** Rules out Jetson, Basler, IDS. **Hikrobot and Intel are the sourceable choices.**
- **Legal answers are shape, not advice.** The backbone-weight licence and hands-in-frame under 152-ФЗ are **OQ-10**, for the lawyer line.
- **A camera sees surfaces only.**
- **Fail to НЕТ РЕШЕНИЯ, never to ГОДЕН.** A PASS needs every configured check to have run and passed (C8).
- Ali's email is for attribution only; never send it to an unrelated service.

## 7. Decisions

The log is `decisions/README.md` — ADR-0001 to ADR-0021, each with context, alternatives and consequences. Do not re-open one without new facts; supersede it with a new ADR. The ones that shape this week:

| ADR | Decision |
|---|---|
| 0007 | The MVP is demo-grade; its numbers are INDICATIVE. Gate A (13.10) is a separate claim on fresh units |
| 0008 | Size and shape measured in code, beside PatchCore |
| 0009 | The Inspector finds the unit, then judges a fixed window centred on it, with the heatmap masked to it |
| 0010 | Bench camera: USB webcam over V4L2, settings read back — pending the lock test |
| 0014 | Paired shooting: each test unit shot good, damaged, shot again, in random order |
| 0015 | PatchCore from Anomalib 2.6.2, pinned, torch model only |
| 0017 | Gate B judged on a conservative N100 estimate until an N100 exists |
| 0018 | Operator screen is a Tk window; OpenCV headless everywhere |
| 0020 | PatchCore's backbone weights are timm's `wide_resnet50_2.racm_in1k`, named by full tag and hash-checked |
| 0021 | Claude Code executes the plans (from P05); one independent Claude agent reviews each. Supersedes 0019 (OpenCode, which ran P01–P02) |

## 8. Stack

PatchCore from **Anomalib 2.6.2** (torch model only, our own preprocessing) · deterministic geometry and localisation with **headless OpenCV** · **Tk + Pillow** operator screen · SQLite evidence store, append-only by trigger · training and demo on the laptop — **RTX 3050, 4 GB**, i5-11400H, Python 3.12 · later **ONNX → OpenVINO on an Intel N100** · MVTec AD for bring-up only, never shipped.

## 9. How work is done

**Planner → Executor → Reviewer.** Claude Code writes plans (`plans/Pxx-*/`: PLAN, CONTEXT, CONTRACTS, ACCEPTANCE), then implements one plan at a time on `exec/Pxx` and writes STATUS.md (ADR-0021; OpenCode ran P01–P02). A fresh Claude agent that did not write the code reviews it; Claude Code fixes the findings and writes `reviews/REVIEW-Pxx.md`. Ali approves every gate and merges every PR. Docs and ADRs come from Claude directly. Plans are written one ahead of execution.

Research freely for accuracy — web search, context7, agents — and cite sources. `ai-engineer` for model/deployment questions; `web-search-agent` for suppliers and regulation dates (**re-verify dates, they move**). Ali's global `/graphify` skill must be invoked via the Skill tool the moment he types it.

## 10. The next three things

1. **Execute P01 (environment), then P02 (contracts and config)** — `plans/ROADMAP.md`. Review each into `reviews/`.
2. **Ali, at the bench:** `sudo apt install v4l-utils`, build the rig (`docs/13-data-collection.md` §3), plug in the webcam; the lock test comes with P05.
3. **Before the shoot, walk through `docs/13-data-collection.md` together** — Ali asked for this. Settle standard or lean (OQ-8) at the shop.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `Ali-Jabowr/qs`, through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels, each named after its role: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root (not written yet), ADRs in `decisions/`. See `docs/agents/domain.md`.
