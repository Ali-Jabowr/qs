# Survey answers — returned 2026-09-02, and what they actually mean

> **Read with `05-concept.md` §0 (2026-09-21).** On a call on 2026-09-20 the owner said the product is
> моти. The "up to 400 mm" height analysed below was our own assumption; the six follow-up
> questions stand, deferred for the MVP week, and are OQ-1…OQ-6 in `project.yml`.

Source: `~/Documents/Obsidian Vaults/Second Brain/20 - AREAS/STARTUP/Site Survey - Factory Visit Kit.md`

---

## The headline: the site visit did not happen

Read the answers' grammar. "**We will** consider separate SKU per line." "**We will** consider
the cheapest version first." "**u can change parameters** due to make the technical part easier."
"**We consider** the environment without moisture."

That is not a factory speaking. That is our own side writing a specification into the customer's
questionnaire. Roughly 10 of 28 answers contain information only the factory could have supplied.
Section G — the physical evidence, which is the entire point of going — produced **nothing**:
no video, no photos of good units, no defective samples. The summary table at the bottom is empty.

This is not a wasted document. As a **draft spec** it is coherent and mostly sensible, and it
contains a few genuinely valuable customer facts. But it cannot be used to buy hardware or to
set a price, because both depend on facts that only exist on the factory floor.

## Split the sheet in two

### Real customer facts (usable)

| Q | Answer | Why it matters |
|---|---|---|
| 6 | Inspection point is **before the packing zone** | Fixes the geometry |
| 7 | **Open product**; green light = OK to pack | The expensive case — hygiene sign-off applies |
| C | Dents · shape deviation · dimension deviation · **foreign objects** | Defect list. Item 4 is a problem — see below |
| 13 | **One person** does this today | The labour baseline we must beat |
| 15 | Escapes cause **refunds** | The ROI mechanism |
| 16 | "2%–10%, consider 2% per day" | A rate, **not a ruble figure**. Still unpriced |
| 20 | One person checks marking | A second labour line, on the other conveyor |
| 21 | **Yes — a batch was returned because of marking** | The proven pain event. The best sales story we have |
| 24–25 | Photo archive yes · Ethernet + 220 V present | Install is easy, traceability is wanted |
| 26 | Hardware + **updatable SaaS** + **annual calibration**, priced by us | Recurring revenue, no competing bid |
| 27 | ASAP | Real urgency |
| 28 | **up to 95%** | The acceptance criterion, verbatim |
| D | Marking is on a **different conveyor** | A second station, not part of this one |

### Our own assumptions, wearing the customer's voice (not facts)

One SKU per line · ≤1 unit/sec · 500 × 500 mm field · units separated · 50–400 mm footprint,
height ≤ 400 mm · "enough space" · no moisture, no dust · own light set · "options" on fail.

Every one of these is a reasonable starting spec. None of them has been checked against the line.

### Still completely blank

Q14 escape rate · Q16 in rubles · Q17–19 date code and seal (deferred to the other conveyor)
· Q23 PLC · **all of section G**.

---

## Three things in here that change the technical plan

### 1. Height ≤ 400 mm plus "deviation from dimensions" — this is the decisive unknown

A single overhead camera has two hard limits against this requirement:

- **It cannot see height at all.** A top view has no information about how tall the object is.
  If "wrong dimensions" includes squashed, under-risen, or short units, a top camera is blind to it.
- **Its footprint measurement scales with height.** Magnification goes as 1/distance, so at a
  1 m working distance a 400 mm-tall object's top surface reads about 1.67× oversize versus the
  belt; even at 2 m it is still ~1.25×. For one SKU of constant height you calibrate that away.
  For varying height you cannot.

A telecentric lens removes the effect entirely, and is the textbook answer — but telecentric
lenses do not exist at a 500 mm field at any price this project can carry.

**So the question that has never been asked is the one that sets the budget:**

> Do they need millimetres with a stated tolerance, or "good / not good" against a golden reference?

- **Pass/fail against a reference** → one camera, one light, cheap, and anomaly detection already
  covers it. This is almost certainly what a food producer actually means.
- **Millimetres with a tolerance** → depth hardware (Orbbec-class stereo/ToF) or a second side
  camera and a calibration rig. Different product, materially higher cost and risk, and it drags
  back in the metrology problem we deliberately parked.

Until this is answered, **do not spec a camera.**

### 2. Open product before packing — there is a person who can veto the install

Anything mounted above open food becomes part of their ХАССП plan: food-grade materials, no
flaking paint, shatter-protected lamp covers so nothing can fall into product, cleanable
surfaces, documented sanitation. Our side answered this with "we consider the environment
without moisture" — which is a statement about *our box*, not about *their rules*.

Their positive-pressure air-purge idea is genuinely good and is standard practice for keeping
dust out of enclosures; factories already have clean dry compressed air. It does not remove
the sign-off requirement.

**Nobody has asked their food-safety officer anything.** That person can stop the installation
after the hardware is bought.

### 3. "Foreign objects" — refuse this scope, in writing

A camera sees the top surface. It cannot see inside opaque product, underneath it, or objects
that are transparent, or the same colour as the product. This is precisely why every food plant
already runs metal detectors and X-ray.

The defect list says foreign objects. The acceptance clause says 95%. Together those promise a
**safety-critical** detection rate that optical inspection cannot deliver, in a category where
the failure mode is not a refund but an injured consumer and a liability claim.

Scope it explicitly: *"surface-visible, contrasting foreign objects, best effort — not a
substitute for metal detection or X-ray."* Get that sentence into whatever is signed.

---

## The strategic finding: the marking station did not go away, it moved

Section D came back with dashes and "different conveyor" — which reads like the marking-first
recommendation is dead. It is not. Read Q20 and Q21 together:

> One person checks marking manually today. **Yes, a batch was returned because of marking.**

The pain is real and already cost them a returned batch. It simply lives on a **different
conveyor, at the same customer, in the same building, on the same purchase order**.

That makes it a **second station, not a competing option** — and it is the easy one:

| | Marking station (other conveyor) | Appearance station (this conveyor) |
|---|---|---|
| Method | Deterministic code — presence, legibility, position, correctness | Machine learning |
| Training data needed | **None** | Hundreds of good units + every defect we can get |
| Reusable at the next customer | **Identical** | Rebuilt per line |
| Proven pain | **Yes — a batch came back** | Refunds, unquantified |
| Time to something working | Weeks | Months, gated on data |

The client asked for the appearance station. Building marking first is a sales move, and it needs
a reason they accept — there is an honest one:

> The marking station can be running in weeks and pays for itself against a single returned batch.
> The appearance station has to learn *your* line, and we need images from it before we can build it.
> So we install marking now — and it starts collecting exactly the data the appearance station needs.

That is true, it is in their interest, and it gets revenue moving months earlier.

## The commercial answer is better than expected

Q26: **hardware + updatable SaaS + calibration service once per year, at our price.**

That is recurring revenue attached to a hardware sale, and it is the direct answer to the
objection that killed this idea in the original roadmap — "bespoke per client, no reusable
asset." An annual calibration visit is also how the appearance model gets retrained on drift,
so the service line is real work, not a fee. And "at our price" plus "ASAP" means nobody has
anchored the number against us.

## ROI arithmetic — the shape, and the hole in it

Two people are being displaced (Q13 line inspection, Q20 marking). Fully loaded — salary plus
roughly 30% payroll contributions — that is the labour ceiling of the pitch, and it should be
computed from **their actual wage figure, which we do not have.**

The refund line is potentially much larger: "2% per day" of output returned. But 2% of *what* —
units, revenue, batches? Nobody asked. **Without a ruble figure there is no price and no
business case**, only a story. Q16 is still open and it is the second-most important number
on the sheet after the metrology question.

---

## The follow-up: six questions, not twenty-eight

Send these, in Russian, and do not send the long sheet again.

1. **Нужны ли вам миллиметры с допуском, или достаточно «годен / не годен» по эталонному
   образцу?** — *sets the entire hardware budget*
2. **Продукт в точке контроля движется непрерывно или останавливается / индексируется?**
   — *continuous motion forces global shutter + strobe + hardware trigger; a stop makes it cheap*
3. **Какая высота продукта на этой линии, меняется ли она, и нужно ли контролировать высоту?**
   — *a top camera cannot see height; this decides whether a second camera is required*
4. **Что вы понимаете под «посторонними предметами»? Что уже стоит на линии — металлодетектор,
   рентген?** — *scopes out the liability, and tells us what they already trust*
5. **Сколько единиц за смену, сколько смен? Во сколько рублей обходится один возврат?**
   — *the ROI arithmetic; "2%" of an unknown base is not a number*
6. **Кто у вас отвечает за пищевую безопасность, и что он должен одобрить, чтобы разрешить
   установку оборудования над открытым продуктом?** — *find the person who can veto the install,
   before the hardware is bought*

Plus, for the marking conveyor: which printer (каплеструйный / лазерный / термотрансферный),
what actually goes wrong with the print, and the story of the returned batch in their own words.

## And the thing that matters more than any answer

**Physical evidence.** None was collected. Nothing in this project can be specified without it:

- 5+ minutes of video of the line running, filmed from where the camera would mount
- slow-motion of the same spot — reveals whether motion blur is a problem
- photos of 30+ good units at the inspection point
- **every defective unit they have**, photographed, and the physical samples if allowed
- the mounting space with a tape measure in frame

Defective samples are the scarcest resource in this entire project. A model that has never seen
a real defect cannot be validated, and a 95% acceptance claim cannot be tested against anything.
