# 05 · Concept and MVP — Phase 0

**Written 2026-09-21 · Status: Gate 0 GO (Ali, 2026-09-21) · Owner of the "what": Ali**

Phase 0 answers two questions before any requirement is written: is this worth building, and
what exactly is the MVP we build this week. Every statement traces to a client answer (`Qn`,
see `03-survey-answers.md`), the owner call of 2026-09-20, or an existing doc — or is marked
**OPEN**.

> **Scope of this week, decided 2026-09-21:** deliver the MVP. Client-facing work — the six
> follow-up questions, the site visit — is deferred. Where a requirement depends on the client
> it is marked OPEN, and the design proceeds around it.

---

## 0. The product is моти — new fact, 2026-09-20

On a call on 2026-09-20 the factory owner told us the line produces **моти (mochi)**: soft
rice-dough rounds, dusted with starch, with a filling. It is the first product fact that did not
come from our own text in the survey form, and it overrides the product sizes we wrote there —
the "up to 400 mm" height was ours, not theirs.

| Before (our assumption) | Now (моти) | Consequence |
|---|---|---|
| Height up to 400 mm: a top camera cannot measure it, footprint reads ~1.67× oversize | Typically ~3–5 cm tall and ~4–6 cm across (**OPEN — confirm their sizes**) | The height problem of CLAUDE.md §5.1 shrinks to a few percent of perspective error, which calibration absorbs. Diameter from above becomes measurable; height still is not. |
| Stand-in: пряники (flat, hard) and булки (tall) | Store-bought моти, ideally the client's own if sold in retail | Train and demo on the real product type instead of a proxy. |
| Surface unknown | Matte starch dusting over soft, slightly translucent dough | Little glare, which helps. Uneven dusting is natural variation that can look like a defect — R2 gets sharper. Thin dough can show the filling. |
| Defects: dents, shape deviation, dimension deviation, foreign objects | Likely: dents and flattening, tears with filling showing, off-round shape, under- or oversize, units stuck together, debris on the surface (**assumption — to confirm**) | All visible from above. Anomaly detection fits. |
| — | Perishable; chilled or room temperature, not frozen (**assumption** — Ali's best knowledge, 2026-09-21; A-01 in `06-requirements.md`) | Units change during a shoot — drying, softening; frost if they turn out to be frozen after all. Good and damaged units must be shot in the same session, or the model learns "which day" instead of "which defect". |
| — | Starch dust in the air at the line | Lens and window contamination at the factory — the positive-pressure air purge matters. Not a bench issue. |
| — | Several flavours, so several colours (**OPEN**) | The MVP uses one flavour. Each further flavour is a further SKU. |

Documents this ripples into, each fixed at its own design step: `03-survey-answers.md` and
CLAUDE.md §4–§5 (the height problem), `02-hardware.md` and `10-architecture.md` (field of view,
working distance), `14-eval-protocol.md` and `project.yml` S0.5–S0.6 (units to buy).

## 1. Press release — written backwards from the day it works

> *Hypothetical and internal. It fixes what the customer must experience; it is not for publication.*
>
> **Volgograd — [date, after S6].** Every моти on [client]'s line is now photographed and checked
> before it reaches the packing zone. A camera station over the conveyor, under its own shielded
> light, compares each unit with what a good one looks like and shows the packer green or red
> within a second — with the defect outlined on screen when it is red. Every photograph is kept
> with its verdict, indexed by batch, on a computer inside the factory. None leaves the building.
>
> The station advises; it does not stop the line or reject product by itself. It inspects the
> visible top surface only and does not replace metal detection.
>
> *"One person used to look at every unit, and escapes still turned into refunds. Now the station
> points at the unit and at the spot, and when a buyer disputes a batch we open its photographs."*
> — hypothetical quality lead

## 2. FAQ — the customer's questions

| Question | Answer |
|---|---|
| What does it check? | The visible top surface of open product: dents, tears, shape deformation and size seen from above (client's defect list, read for моти in §0). |
| What does it not check? | Height, weight, anything inside or under the product. Foreign objects beyond "surface-visible, contrasting, best effort" (CLAUDE.md §5.3). Marking — a separate station on another conveyor. Millimetre tolerances stay **OPEN — OQ-1**. |
| Does it replace the inspector? | No. It is advisory: an operator signal, no actuator (`10-architecture.md` §7). |
| Does any image leave the factory? | No. On-prem only — 152-ФЗ and trade secret. |
| How accurate is it? | The client's number is 95% (Q28). We quote recall at a stated false-alarm rate, measured on *their* units after training on *their* good product — never before. The bench MVP establishes whether the method works at all. |
| What happens when it cannot decide? | NO_VERDICT — never shown as a pass, never recorded as one (`11-interfaces.md` §7). |

## 3. FAQ — our own questions

| Question | Answer |
|---|---|
| Why anomaly detection and not a trained detector (YOLO)? | Thousands of good units, almost no defective ones, and an open-ended defect list. Train on good units only; flag what differs (CLAUDE.md §7, 2026-09-01). |
| Why build on store-bought моти? | The pipeline is the product and the data is swappable (`04-build-plan.md`). Moving to the client's моти means retraining and swapping the camera, not rewriting. |
| What is the central risk? | Natural variation between good units — dusting, shape, colour — swamps the defect signal and the station cries wolf (R2). The MVP measures exactly this. |
| What does finding out cost? | Product from the shop and one week. No hardware purchase (Gate C). |
| What kills the appearance station? | On one flavour of моти under controlled light, after one re-shoot, the model still cannot separate damaged from good units at a false-alarm rate an operator would tolerate. The pipeline then carries over to the marking station with only the Inspector changed. |

## 4. Problem, customer, benefit

| | |
|---|---|
| **Problem** | Defective units reach packing. Escapes cause refunds — quoted as "2–10%, consider 2% per day". |
| **Today** | One person inspects every unit by eye. The client wants a photo archive per batch. |
| **Customer** | The factory's quality function. User: the operator at the station. Veto holder: whoever signs off food safety over open product (**OPEN — OQ-6**). |
| **Benefit** | (1) Defects caught before packing. (2) An archived image and verdict for every unit — ХАССП traceability, valuable even when nothing is defective. (3) The inspector's attention goes to flagged units instead of all of them. |
| **Measured by** | (1) Recall at a fixed false-alarm rate, per defect type (`14-eval-protocol.md`). (2) Share of units with an archived record — target 100%. (3) Rubles per avoided escape — **OPEN (OQ-5)**. |

## 5. The MVP

**Definition.** The real pipeline, running end to end on the bench, on store-bought моти:
key press → capture in the rig → model → PASS / DEFECT / NO_VERDICT (ГОДЕН / БРАК / НЕТ РЕШЕНИЯ) with a heatmap → append-only
evidence record. Trained on good units only, threshold set on separate calibration units, result
measured on held-out units. Starts with one command and runs offline.

The same code later runs at the factory with a different trigger, camera and training set
(`04-build-plan.md`, the six blocks).

**Scope: A — demo-grade** (Gate 0, 2026-09-21 · `decisions/ADR-0007`). A short lighting pilot on
~10 units picks one geometry; the dataset is shot in that geometry only; numbers are reported as
indicative. Gate A keeps its full protocol, due 13.10, on freshly bought units the MVP never saw.

**In the MVP**

- Product: one flavour of store-bought моти, the client's own if it is sold in retail.
- Trigger: key press. Camera: a USB webcam or an Android phone, both already owned — the primary
  one is chosen in Step 3 by which can hold exposure, white balance and focus locked. Light:
  improvised rig with a reference patch in frame.
- PatchCore, trained on the laptop's RTX 3050 (4 GB). Inference on the same machine, with latency measured on CPU
  as the Intel N100 proxy.
- Screen: live view, a verdict readable across a room, a heatmap on DEFECT.
- Evidence: SQLite plus JPEG, append-only.
- A measured report: recall at 1 / 5 / 10% false alarms, counted per unit, with confidence bounds,
  labelled indicative.

**Not in the MVP**

- Anything at the factory: Intel N100, OpenVINO deployment, Hikrobot camera, photoelectric trigger.
- A second flavour, or any other SKU.
- The marking station — it needs the client's printer and date-code format (M0).
- Millimetre measurement, foreign objects, PLC or actuator, networking, user accounts, cloud.

## 6. Rough business case

| | |
|---|---|
| **Cost to MVP** | Store-bought моти — several times the per-unit price of пряники; scope A needs very roughly 60–80 units, fixed in Step 6 (**OPEN — budget ceiling**). 0 ₽ hardware. One week of Ali's time. |
| **Value of the MVP** | A working thing on the client's own kind of product to put in front of them at the site visit (S5) — the meeting that produces the video, defective samples and answers everything else waits on. |
| **Commercial model** | Hardware + updatable SaaS + annual calibration, priced by us, no competing bid (Q26). |
| **Value per escape** | **OPEN (OQ-5).** "2% per day" of an unknown base is not a number. No price can be quoted until it is. |

## 7. Gate 0 — concept review

**Recommendation: GO** on the bench MVP.

- The problem is real on the client's own word. Finding out costs product from the shop and a
  week. Not doing it sends us to the site visit with nothing to show.
- The client's commitment is verbal — no contract, no money. Acceptable at this cost; not
  acceptable for any hardware spend (Gate C unchanged).

**Kill and pause criteria**

- The appearance station is killed if the condition in §3 is met.
- The engagement is paused if, after the site visit, the client cannot price one escape (OQ-5):
  no business case, no quote.

**Carried open, not blocking the MVP:** OQ-1 … OQ-6, deferred by decision 2026-09-21; the моти
details marked OPEN in §0.

**Decision: GO** · Ali · 2026-09-21
