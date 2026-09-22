# QS — Food Production Line Visual Inspection

Automated visual quality inspection for a food factory production line. One client, in Volgograd, with verbal intent to pay if it works. **The product is моти.**

**Status (2026-09-21):** the design is complete — Gates 0, 1 and 2 passed. This week builds the **bench MVP** on store-bought моти (`plans/ROADMAP.md`). No hardware bought, no site visit yet; client work is deferred for the week by decision.

👉 **Start every session at [`CLAUDE.md`](CLAUDE.md).** Live state: [`STATUS.md`](STATUS.md), rendered from [`project.yml`](project.yml).

---

## What we're building

An inspection station on the client's production line that photographs every unit before packing, checks it, and signals **ГОДЕН / БРАК / НЕТ РЕШЕНИЯ** — while keeping an image of every unit as a traceability record.

**Two stations, one program** (ADR-0004):

| Station | What it checks | Status |
|---|---|---|
| **M — marking** | Date code present, legible, correct — on the other conveyor | Ships first. Waits on the client's printer and code format |
| **A — appearance** | Dents, tears, deformation and size of each моти, from above | Prototyped first — this week's bench MVP |

Station A runs two checks on every unit: **geometry** in deterministic code (area, axis ratio, solidity, against good units — ADR-0008) and **appearance** by PatchCore anomaly detection, trained on good units only (ADR-0002, 0015).

## Architecture

```
   key press on the bench · photoelectric trigger at the factory
              ↓
   [ camera + shrouded controlled lighting + grey reference card ]
              ↓
   [ laptop on the bench · Intel N100 edge box at the factory — on-prem ]
     capture → lighting check → find the unit → geometry + PatchCore → verdict
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
[ operator screen ]   [ evidence store ]
 ГОДЕН / БРАК /        every image + verdict + provenance,
 НЕТ РЕШЕНИЯ, heatmap  append-only SQLite + files, by batch
```

Full design: `docs/10-architecture.md` to `docs/17-security.md`. Every decision: `decisions/`.

**Invariants** (`docs/10-architecture.md` §7): on-prem, no image leaves the building · every unit writes an immutable record · advisory only, no actuator · one inspected image per trigger · ambient light blocked and drift measured · fail to НЕТ РЕШЕНИЯ, never to ГОДЕН · a camera sees surfaces only · only re-orderable hardware · persistence never blocks a verdict.

## Repo

```
CLAUDE.md · AGENTS.md · STEPS.md    session anchor · executor rules · the plan/execute/review loop
project.yml → STATUS.md             the single source of project state, and its rendering
docs/05–06                          concept (Gate 0) and requirements (Gate 1)
docs/10–17                          architecture, contracts, data model, data collection,
                                    evaluation, model, failure model, security
docs/01–04                          history: survey kit, hardware tree, survey answers, first plan
decisions/                          ADR-0001 … ADR-0020
reviews/                            gate reviews and plan reviews
plans/                              ROADMAP.md and one folder per executor plan
src/ tests/ tools/                  code · tests · renderer, bounds, diagram checker
data/                               git-ignored: raw shoots, datasets, golden sets
```

`make check` validates project state, runs the tests and renders every diagram.

## Constraints carried over

- **Never build on hardware you can't reliably re-order.** Hikrobot and Intel are the sourceable choices.
- **Never itemize hardware in a quote.** Price the outcome.
- **A camera sees surfaces only.** Foreign objects inside opaque product need X-ray or metal detection. Do not promise it.
