# QS — Food Production Line Visual Inspection

Automated visual quality inspection for a food factory production line. One client, in Volgograd, with verbal intent to pay if it works.

**Status:** pre-survey. No hardware bought, no code written. The site visit is the blocking action.

---

## What we're building

An inspection station on the client's production line that photographs every unit before shipping, checks it, and signals pass/fail — while keeping an image of every unit as a traceability record.

**Two layers, deliberately sequenced:**

| Layer | What it checks | Why this order |
|---|---|---|
| **1 — Packaging & marking** | Date code present, legible, correct · label present and straight · seal integrity | Near-deterministic. Low natural variation, binary ground truth, little training data needed. A missing date code gets a **whole pallet rejected** — the sharpest, most provable pain. |
| **2 — Appearance** | Shape, colour, burn level, breakage | Genuine ML. Huge natural variation, fuzzy ground truth, needs real data from their line. Built once layer 1 is running and generating that data. |

Layer 1 is the product. Layer 2 is the R&D. Shipping them in this order means there's something sellable before the hard problem is solved.

## Architecture

```
   photoelectric trigger (or operator button)
              ↓
   [ camera + shrouded controlled lighting ]
              ↓
   [ edge box — Intel N100, on-prem ]
     capture → preprocess → checks → verdict
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
[ operator signal ]   [ evidence store ]
 green / red light     every image + verdict + timestamp
 buzzer, screen        indexed by batch — local disk
                              ↓
                       [ reports / traceability ]
```

**Invariants — true regardless of what the survey says:**

- **On-prem. No image leaves the building.** Legally cleaner under 152-ФЗ, and in a food factory it's also trade-secret protection. Closes the security objection before it opens.
- **Every image is kept, with its verdict, indexed by batch and timestamp.** Traceability is a legal requirement under ХАССП — the archive has value independent of catching defects, and it's nearly free to build. It also becomes the training data for layer 2.
- **Advisory first.** Operator alert, not automatic rejection. No PLC, no actuator, no line-safety sign-off. Prove detection works before touching their machinery.
- **Hardware trigger, not free-running video.** One image per unit, or you get doubles, partials and misses.
- **Ambient light gets blocked out.** Shift changes and daylight silently degrade models that tested fine.

## Repo

```
docs/    01-site-survey.md    the visit kit — fill this in first
         02-hardware.md       decision tree, not a shopping list
data/    site visit media, samples, datasets
src/     (empty — nothing written until there's data)
notes/   working notes
```

## Next actions

1. **Do the site visit.** `docs/01-site-survey.md` — questions in Russian, plus the list of what to film and physically bring back. Nothing else can be decided without it.
2. **Bring back defective samples.** The scarcest resource in the whole project. Photograph everything, take the physical units if allowed.
3. **Get an acceptance number** (survey Q28) — what detection rate makes them buy. Verbatim.
4. Bench prototype on the recorded video, on the RTX 3060. No purchases.
5. Spec and buy hardware only after that works.

## Open questions

- What food product, how many SKUs — unknown, survey answers it
- Line speed — unknown, and it's the main hardware driver
- Fail behaviour: alert / auto-reject / log only — undecided, ask the client
- Washdown environment? Drives enclosure cost from 10k to 100k ₽

## Constraints carried over

- **Never build on hardware you can't reliably re-order.** Rules out NVIDIA Jetson and Basler/IDS cameras — parallel import only. Hikrobot and Intel are the sourceable choices.
- **Never itemize hardware in a quote.** Price the outcome.
- **A camera sees surfaces only.** Foreign objects inside opaque product need X-ray or metal detection. Do not promise it.
