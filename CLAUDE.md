# CLAUDE.md — read this first, every session

**Project:** QS — automated visual quality inspection for a food production line.
**Owner:** Ali (Volgograd). A student friend handles all grant/legal/document work; Ali's job with Claude is **building the MVP**.
**Repo:** `/home/marxel/PROJECTS/QS` · **Planning vault:** `~/Documents/Obsidian Vaults/Second Brain/20 - AREAS/STARTUP/`
**Last updated:** 2026-09-02

---

## 1. What this is

One factory client in Volgograd, food producer, **verbal intent to pay**. They want every unit checked on the line before packing. There is no signed contract and no money has moved.

There is a parallel, **parked** track: a "QC Vision Box" for Wildberries/Ozon marketplace sellers. It is not being built. See §7.

## 2. Where everything lives

```
QS/
  CLAUDE.md              ← this file. The cold-start anchor.
  README.md              architecture, invariants, repo layout
  docs/01-site-survey.md the 28-question factory visit kit (sent to client)
  docs/02-hardware.md    conditional decision tree — NOT a shopping list
  docs/03-survey-answers.md   the returned answers + what they actually mean
  docs/04-build-plan.md  ← THE PLAN. Six stages to a demo. Follow it in order.
  data/                  git-ignored. site media, samples, datasets
  src/                   empty. Nothing written until there is real data.
```

Vault: `20 - AREAS/STARTUP/00 QS — Master Plan.md` is the single planning document —
phases, gates, research, agent/skill usage. `Site Survey - Factory Visit Kit.md` there
holds the client's returned answers.

Superseded and closed: **SmartBot Industrial** (PPE/safety CV) under `20 - AREAS/Side projects/`.
Ali closed it 2026-09-01 — "we close it and it didn't work". Do not revive it, do not
cite its gates. What survived is listed in §6.

## 3. Status as of 2026-09-02

| | |
|---|---|
| Hardware bought | **None.** Correct. Do not buy anything yet. |
| Code written | **None.** |
| Site visit | **Has not happened.** The questionnaire came back filled mostly with our own assumptions, not observations. |
| Physical evidence | **None.** No video, no photos, no defective samples. This is the #1 blocker. |
| Acceptance number | **95%** (client, verbatim, Q28) |
| Commercial model | hardware + updatable SaaS + annual calibration, **priced by us**, no competing bid (Q26) |
| Timeline demanded | ASAP (Q27) |

## 4. What the client actually told us

Only ~10 of 28 answers are real customer facts. These are they:

- Inspection point: **before the packing zone**, on **open product**. Green light = OK to pack.
- Defects wanted: **dents · shape deviation · dimension deviation · foreign objects**
- **One person** does this inspection today. One more person checks marking.
- Escapes cause **refunds**. Rate quoted as "2%–10%, consider 2% per day". **No ruble figure given.**
- Marking / date code is on a **different conveyor** — not this station.
- **A batch was returned because of marking** (Q21 = yes). This is the proven pain event.
- Ethernet and 220 V are available at the point. Photo archive per batch: yes.

Everything else in that sheet — line speed, field of view, product sizes, unit separation,
environment, lighting — is **our own specification written into the customer's form**. Treat
it as intent, never as fact.

## 5. The three technical facts that drive every decision

1. **Product height up to 400 mm, and they want "deviation from dimensions".**
   A single overhead camera cannot measure height at all, and its footprint measurement
   scales with object height (at 1 m working distance a 400 mm object reads ~1.67× oversize).
   Telecentric lenses would fix it but do not exist affordably at a 500 mm field.
   → **Unresolved and decisive: do they need millimetres with a tolerance, or pass/fail
   against a golden reference?** Pass/fail = one camera. Millimetres = depth hardware
   (Orbbec-class) or a second side camera. Different product, different price.

2. **Open product before packing.** Anything mounted over open food enters their ХАССП
   plan — food-grade materials, shatter-protected lamps, cleanable surfaces, and sign-off
   by their food-safety officer. That person has not been consulted. Their positive-pressure
   air purge idea for the enclosure is sound and standard; it does not remove the sign-off.

3. **"Foreign objects" is the dangerous line item.** A camera sees the top surface only —
   not inside product, not underneath, not transparent or same-colour items. Industry uses
   metal detectors and X-ray for exactly this. **Never accept FOD inside a 95% acceptance
   clause.** Scope it out in writing: "surface-visible contrasting objects, best effort,
   not a substitute for metal detection or X-ray."

## 6. Standing rules — do not violate

- **Nothing gets bought before the site visit produces video and samples.**
- **Never itemize hardware in a quote.** Price the outcome.
- **Never suggest identifying individuals.** No face recognition, no re-identification.
  Hard constraint — 152-ФЗ biometric exposure, and it is also a sales line.
- **On-prem only. No image leaves the building.** 152-ФЗ plus trade-secret protection.
- **Advisory first.** Operator alert, never automatic rejection in the MVP. No PLC, no
  actuator, no line-safety sign-off.
- **Never build on hardware that cannot be reliably re-ordered.** Rules out NVIDIA Jetson,
  Basler, IDS — parallel import only. **Hikrobot and Intel are the sourceable choices.**
- **Legal answers are shape, not advice.** Point at the lawyer budget line.
- **A camera sees surfaces only.** See §5.3.
- Ali's email is for attribution only; never send it to an unrelated service.

## 7. Decisions already made — do not re-open without new facts

| Decision | Date | Reason |
|---|---|---|
| SmartBot Industrial closed | 2026-09-01 | Zero customers, Gate 1 missed, empty pipeline |
| Dimensioning rejected as V1 | 2026-09-01 | Solved commodity abroad; under the client's own answers it has no user. Activate only if the WB/Ozon seller track becomes active. |
| Anomaly detection, not supervised YOLO | 2026-09-01 | Thousands of good units, almost no defective ones; defect classes open-ended. Anomalib / PatchCore. ⚠️ verify per-model licences before commercial use. |
| Marking station sold alongside, built first | 2026-09-02 | Deterministic, no training data, proven pain (Q21), identical for every future customer. Appearance is the R&D. |

## 8. Stack

Anomalib (PatchCore/PaDiM/FastFlow) for appearance · OpenCV + deterministic checks for
marking · ONNX → **OpenVINO** on an **Intel N100/N150** mini-PC · training on the local
RTX 3060 · MVTec AD for pipeline bring-up before real data exists.

## 9. Useful agents and skills

`ai-engineer` — model/deployment architecture. `web-search-agent` / WebSearch — supplier
availability, Честный ЗНАК deadlines, ФСИ windows; **re-verify regulation dates, they move**.
`context7` — Anomalib / OpenVINO / ultralytics docs. `claude-in-chrome` — supplier pages.
Ali's global `/graphify` skill must be invoked via the Skill tool the moment he types it.

## 10. The next three things

**Follow `docs/04-build-plan.md`.** It fixes the whole sequence; these three are its head.

1. **Bench prototype on supermarket food** — unblocked, start now. 40 identical items,
   a lighting rig, deliberately damage 10, train PatchCore, measure. Answers the central
   technical question and produces a demo without touching the factory.
2. **Get the visit done** — 6 focused follow-up questions (in `docs/03-survey-answers.md`),
   plus video and **defective samples**. Samples are the scarcest resource in the project.
3. **Price and pitch the marking station** on the other conveyor.
