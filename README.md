# QS — Food Production Line Visual Inspection

Automated visual quality inspection for a food factory production line. One client, in Volgograd, with verbal intent to pay if it works.

**Status:** survey returned 2026-09-02, but **the site visit never happened** — most answers are our own assumptions and no video, photos or defective samples were collected. No hardware bought, no code written.

👉 **Start every session at [`CLAUDE.md`](CLAUDE.md).** Survey analysis: [`docs/03-survey-answers.md`](docs/03-survey-answers.md).

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

1. **Bench prototype — unblocked, start now.** 40 identical supermarket items, an improvised
   rig (phone + cardboard box + desk lamp, spend zero), damage 10 deliberately, train PatchCore,
   test three lighting geometries. Answers the central technical question without the factory.
2. **Send six follow-up questions, not the 28-question sheet again** — see
   `docs/03-survey-answers.md`. The first one, *millimetres with a tolerance or pass/fail against
   a golden reference*, sets the whole hardware budget.
3. **Do the visit and bring back defective samples.** Scarcest resource in the project.
4. **Get one escape priced in rubles.** "2% per day" of an unknown base is not a number.
5. **Price and pitch the marking station** on the other conveyor — deterministic, no training
   data, and a batch has already been returned because of it.
6. Spec and buy hardware only after Gate A. Nothing before.

## Open questions

- **Millimetres or pass/fail?** A top camera cannot see height and its footprint scales ~1.67×
  on a 400 mm object at 1 m working distance. This blocks the camera spec.
- Does product move continuously or index/stop at the point? Blocks the shutter decision.
- Is height itself a checked dimension? Decides one camera or two.
- What is one escape worth in rubles? Blocks pricing.
- Who signs off food safety over open product, and what do they require? Can veto the install.
- Which printer on the marking conveyor, and what goes wrong with it? Blocks station A.

## Constraints carried over

- **Never build on hardware you can't reliably re-order.** Rules out NVIDIA Jetson and Basler/IDS cameras — parallel import only. Hikrobot and Intel are the sourceable choices.
- **Never itemize hardware in a quote.** Price the outcome.
- **A camera sees surfaces only.** Foreign objects inside opaque product need X-ray or metal detection. Do not promise it.
