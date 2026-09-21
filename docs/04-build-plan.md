# Build plan — from nothing to a demo the client can hold

> **Superseded — 2026-09-21.** Kept as history. The stages survive in `project.yml`; the product is
> моти, not пряники; the MVP is demo-grade (ADR-0007); the design is docs 05–17 and `decisions/`;
> the work is `plans/ROADMAP.md`. Gate A as written below was restated (ADR-0005, ADR-0017).

**Written 2026-09-02.** Fixed on paper before any code is written. Follow the stages in order.
Do not skip ahead, do not add features between stages.

---

## The one principle everything follows

> **The demo is not a prop. It is the real pipeline, fed easy data first.**

Going from cookies to the client's product means changing the training images and swapping
the camera. It does not mean rewriting the program. That is the whole reason we can start
today without a single item from the factory.

## The six blocks

```
[ trigger ] → [ camera + light ] → [ capture ] → [ model ] → [ verdict ] → [ evidence log ]
```

| Block | Demo (October) | Factory (later) | Rewritten? |
|---|---|---|---|
| Trigger | press spacebar | photoelectric sensor | swap, ~20 lines |
| Camera | webcam / phone | Hikrobot industrial | swap, ~20 lines |
| Light | desk lamp + cardboard box | LED bar + metal shroud | hardware only |
| Capture + preprocess | OpenCV | **same** | **no** |
| Model | PatchCore on cookies | PatchCore on their product | **retrain, same code** |
| Verdict | score → OK / defect | **same**, threshold tuned | **no** |
| Evidence log | SQLite + JPEG folder | same + batch ID | extended |

About 60% of the work is written once and never touched again.

---

## Stage 0 · Set up · 2026-09-02 → 09-07

**Goal:** a working GPU training environment and two boxes of groceries.

### Pick the stand-in product

Two of them, deliberately — one easy, one that tests our biggest unknown.

| | **Easy mode** | **Hard mode** |
|---|---|---|
| What | 40 identical cookies / пряники | 20 bread loaves / булки |
| Why | flat, matte, identical, dents and chips obvious | tall and 3D — tests the height problem |
| Tests | does the method work at all | does it survive natural shape variation |

Selection criteria, in case you substitute something else: **identical units**, **matte or
semi-gloss** (not shiny plastic wrap — that fights the lighting test), **damageable in a way
you can see**, and **cheap enough to buy 40 and destroy 15**. Budget ~1500 ₽ total.

### Environment

- [ ] Python 3.11, virtualenv in the repo
- [ ] PyTorch with CUDA. **Verify:** `torch.cuda.is_available()` is `True` and the device name
      says RTX 3060. If this is wrong, fix it now — not at Stage 3.
- [ ] `anomalib`, `opencv-python`
- [ ] ⚠️ **Check the licence of every model inside Anomalib before it ships.** Anomalib itself
      is Apache-2.0, but per-model terms have caught people out before — same trap as YOLO's AGPL.

### Data layout — decide it now, it never changes

```
data/
  raw/                          untouched originals, per shoot session
  datasets/<product>/
    train/good/                 good units only — this is all the model ever trains on
    test/good/                  held out good units, for measuring false alarms
    test/dent/
    test/broken/
    test/deformed/
```

This is Anomalib's folder format. Using it from day one means the factory dataset drops
straight in with no conversion.

**Done when:** the GPU trains something, and there are cookies on the desk.

---

## Stage 1 · Prove the pipeline on public data · 2 days

**Goal:** know the installation is correct *before* our own data can be blamed.

- [ ] Download **MVTec AD** and run PatchCore on one category
- [ ] Reproduce roughly the published numbers

> [!warning] MVTec AD is licensed CC BY-NC-SA 4.0 — **non-commercial**.
> Use it **only** to validate that our install works. Never train anything that ships on it,
> never distribute weights derived from it. Our own cookie dataset has no such restriction.

**Done when:** PatchCore runs end to end and the score is in the right ballpark. If it isn't,
the problem is the environment, and we found out in two days instead of two weeks.

---

## Stage 2 · Build the rig and shoot · ~1 week · **this is the real work**

**Goal:** a dataset, and an answer to which lighting reveals which defect.

### The rig

- Cardboard box, interior **matte black** — black paper or matte paint. No white, no gloss.
- Camera **fixed and marked** so it returns to the exact same position every session.
- Object placed in the same spot, filling ~80% of the frame.

> [!danger] Lock the camera settings. Manual exposure, manual white balance, manual focus.
> Auto settings are the single most common way to destroy a computer-vision dataset — the
> camera "helpfully" re-exposes for the defect and hides it. On a phone, use a manual camera app.

### The three lighting geometries — this experiment *is* the research

| Setup | How | Reveals |
|---|---|---|
| **Low-angle / darkfield** | light from the side, almost parallel to the surface (~10–30°) | **dents, scratches, surface relief** — the defect casts its own shadow |
| **Diffuse** | light bounced off white paper or through tracing paper | kills glare on shiny surfaces; good for colour and print |
| **Backlight** | light from below, through a translucent sheet | silhouette only — shape, size, missing pieces, deformation |

### The shoot

- [ ] 40 good units × 3 lighting setups × 3 rotations = **~360 good images** per product
- [ ] Damage 15 units: **5 dents · 5 chips/breaks · 5 deformed**
- [ ] Shoot every damaged unit under all 3 lighting setups × 3 rotations
- [ ] Split: ~30 good units into `train/good`, ~10 into `test/good`
- [ ] Photograph each damaged unit's defect close-up for the record

**Done when:** the dataset is in the folder layout above, and you can say from looking at the
images which lighting shows a dent best. That sentence is worth more than the model.

---

## Stage 3 · Train and measure · 3–4 days

- [ ] PatchCore, WideResNet-50 backbone, trained on `train/good` **only** — one model per
      lighting setup, so the setups can be compared fairly
- [ ] Evaluate on the held-out good units and all the damaged ones

### Measure the right thing

AUROC is not the number that sells this. The number that sells it is:

> **At a threshold that raises false alarms on no more than 5% of good units, what fraction
> of real defects do we catch?**

Report recall at **1%, 5% and 10%** false-alarm rates, per lighting setup, per defect type.
A system that flags one good unit in five gets switched off by the operator in a week, no
matter what its AUROC says.

- [ ] Also measure **inference latency**. The client's ceiling is 1 unit/second. PatchCore
      does nearest-neighbour search against a memory bank and can be slow on weak CPUs —
      if it won't fit the budget on an N100, the answers are a smaller coreset, lower input
      resolution, or switching to PaDiM / FastFlow. **Find this out now, not at deployment.**

> [!danger] ⛔ Gate A — 2026-09-22
> **On the flat product, under the best lighting: ≥90% of dents caught at ≤5% false alarms.**
>
> If it fails, **the problem is the lighting, not the model.** Go back to Stage 2 and re-shoot.
> Do not retrain, do not tune hyperparameters, do not add features.
>
> If we cannot hit 90% on cookies in a cardboard box, we will not hit the client's 95% on
> their line with real hardware.

---

## Stage 4 · Wrap it as a demo · ~1 week

**Goal:** something you can carry into a factory and hand to a stranger.

- [ ] Live camera view
- [ ] Spacebar (or auto-detect that an object was placed) → capture → verdict
- [ ] Big **green OK** / **red DEFECT** — readable across a room
- [ ] **Heatmap overlay showing where the defect is.** This is the moment that convinces
      people. A score convinces nobody; a red blob sitting exactly on the dent convinces everybody.
- [ ] Every check logged: photo, verdict, score, timestamp → SQLite + JPEG folder
- [ ] A results list showing the session so far

**Constraints:** starts with one command · runs fully offline · someone who has never seen it
can use it without you touching the keyboard.

**Done when:** you hand the laptop to a friend, say nothing, and they get it.

---

## Stage 5 · Show it to the client

The demo is the reason this meeting goes well. Walk in with a working thing and collect:

- [ ] Answers to the **six questions** (`docs/03-survey-answers.md`) — above all,
      *millimetres with a tolerance, or pass/fail against a golden reference?*
- [ ] 5+ min of video from where the camera would mount, plus slow-motion
- [ ] 30+ good units photographed at the inspection point
- [ ] **Every defective unit they have** — physical samples if allowed
- [ ] Mounting space with a tape measure in frame
- [ ] One escape priced **in rubles**
- [ ] The name of whoever signs off food safety over open product
- [ ] The marking conveyor: printer type, what goes wrong, the returned-batch story in their words

---

## Stage 6 · Retrain on their product

Same code. Their images. Their defects. This is the real MVP, and the point where a price
can be quoted honestly.

---

## Beyond — how this scales

1. **ONNX → OpenVINO on the Intel N100.** On-prem, no image leaves the building.
2. **Industrial camera and real lighting** — Hikrobot; spec follows from the Stage 2 lighting
   result and the client's answer on motion.
3. **Photoelectric trigger** replaces the spacebar. One image per unit.
4. **Operator screen and lamp.** Advisory only — no PLC, no actuator, no rejection mechanism.
5. **Evidence store with batch IDs** — the ХАССП traceability record, and the training data
   for every future retrain.
6. **The marking station** on the other conveyor: deterministic, no training data, weeks not
   months, and a batch has already been returned because of it.
7. **Customer #2** — the marking half ships unchanged; only the appearance model is retrained.

---

## What could go wrong

| Risk | Signal | Response |
|---|---|---|
| Lighting doesn't reveal dents | Gate A fails | Re-shoot. This is exactly what Gate A exists to catch. |
| Camera auto-settings ruin the dataset | Images inconsistent between sessions | Lock exposure/WB/focus. Most common failure in this whole plan. |
| PatchCore too slow for 1 unit/sec | Latency measured at Stage 3 | Smaller coreset, lower resolution, or PaDiM / FastFlow |
| Their real defects are subtler than our deliberate damage | Only visible after Stage 6 | Why defective **samples** matter more than any answer |
| Natural food variation swamps the defect signal | High false alarms on good units | **The central risk of the appearance station.** More training data, tighter lighting, or narrow the scope to one SKU |
| Something trained on MVTec ends up in a deliverable | — | Never happens: MVTec is validation only |

## What we are explicitly not doing yet

- **No hardware purchases** beyond ~1500 ₽ of groceries, until Gate A passes
- **No supervised detection** (YOLO / RF-DETR) — 15 damaged cookies is not a training set
- **No Docker, no web framework, no API, no user accounts, no cloud**
- **No features added between stages.** Finish the stage, pass the gate, then move
- **No promise of foreign-object detection.** A camera sees surfaces. Ever.
