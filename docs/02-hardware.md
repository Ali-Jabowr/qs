# Hardware — Decision Tree, Not a Shopping List

> **Superseded in part — 2026-09-21.** Written before the product was known to be моти (~3–5 cm,
> not 400 mm) and before the design day. The training machine is the laptop's RTX 3050 (4 GB), not an
> RTX 3060. The decision tree still frames the factory hardware; read it with `05-concept.md` §0 and
> `10-architecture.md` before using any number here.

**Rule: buy nothing until the site survey is done.** Camera, lens, lighting and enclosure are all determined by numbers you can only measure standing at the line. The one thing you can do beforehand is prototype on video and photos from the visit, on the RTX 3060 you already own.

---

## Stage 0 — Bench prototype · buy nothing · ~0 ₽

Everything needed comes back from the site visit: video, photos, and physical samples.

- Train and evaluate on the RTX 3060
- Build the full software loop against **recorded video**, not a live camera
- Prove the date-code and label checks on real packaging photos

**Only spend money once this shows the checks are achievable.** If the defect isn't visible in the site photos, no camera purchase fixes that — it's a lighting or positioning problem, and the answer is a second visit.

---

## Stage 1 — Camera: driven entirely by line speed

### If < 60 units/min, or a static operator station

A **rolling-shutter USB3 or GigE industrial camera** is enough, with continuous lighting. Cheapest viable path. A good USB3 machine-vision camera runs ~15–35k ₽.

### If > 60 units/min, or the belt is visibly fast

You need **global shutter + strobed lighting + hardware trigger**. Rolling shutter smears moving product and the model will fail in ways that look like a model problem but aren't. Budget ~40–80k ₽ for the camera.

### Vendor: Hikrobot (machine vision arm of Hikvision)

**This is the right answer for Russia.** Hikvision is already distributed everywhere in the country, the MV-CA / MV-CS machine vision line is properly available, GigE Vision standard, and there's local support — which satisfies the "never build a product on hardware you cannot reliably re-order" rule. Daheng Imaging is a reasonable Chinese alternative.

⚠️ **Avoid Basler and IDS** despite being technically excellent — German, parallel import only, no local support. Same reasoning that ruled out Jetson.

### Lens

Chosen by arithmetic once you know **working distance** and **field of view** from the survey. Fixed focal length C-mount, ~5–15k ₽. Do not buy before those two numbers exist.

**Resolution rule of thumb:** you want roughly **3–5 pixels across the smallest feature you must detect**. For date-code OCR, that means 3–5 px on the *stroke width* of the printed character, not the character height. This calculation drives camera resolution and is the most common place people under-buy.

---

## Stage 2 — Lighting: this is where the project succeeds or fails

More important than the camera and far more important than the model. No model recovers a defect the sensor never captured.

| Setup | Use for | Rough cost |
|---|---|---|
| **Diffuse bar / dome** | General appearance, colour, shape. Even, shadow-free. | 5–15k ₽ |
| **Low-angle (grazing)** | Surface relief — dents, cracks, embossing | 5–15k ₽ |
| **Backlight** | Silhouette — shape, fill level, missing components | 4–10k ₽ |
| **Coaxial / on-axis** | Print and date codes on glossy or reflective film | 10–20k ₽ |
| **Red or IR bar light** | Date-code contrast on busy packaging — often dramatically better than white | 5–12k ₽ |

Chinese OEM lighting via Ozon or AliExpress is completely fine for prototyping. Buy two or three setups and test empirically — the winner is rarely the one you predict.

**Shroud the station.** Block ambient light entirely. Factory lighting changes across shifts and daylight moves through the day; a model trained under one condition silently degrades under another. This is the most common cause of "it worked in testing" failures in food CV.

---

## Stage 3 — Trigger · ~1–4k ₽

A **photoelectric sensor** (Autonics, Omron, or a Chinese equivalent) fires the camera as each unit passes. Do not use free-running video for inline inspection — you'll get partial units, doubles, and misses.

Not needed if the product is static at an operator station.

---

## Stage 4 — Compute · ~20k ₽

**Intel N100/N150 mini-PC, 16GB RAM, 500GB SSD.** Carried over from the previous hardware analysis and still correct:

- **OpenVINO** runs natively and well on Intel — the smoothest deployment path available
- x86 means development and deployment are the same architecture, no cross-compilation pain
- Cheap enough to leave permanently on site
- Fully sourceable in Russia (DNS, Citilink, Ozon)

**Storage:** size for full image retention. At ~500KB/image and 5,000 units/shift that's ~2.5GB/shift, roughly 75GB/month. A 1–2TB SSD covers a year of traceability. Cheap, and the archive is part of what you're selling.

**Revisit only if:** inference can't keep up with the line. Then consider Rockchip RK3588 (NPU, sourceable from China, but the RKNN toolchain is more painful than OpenVINO — right for mass production, wrong for an MVP). ⚠️ NVIDIA Jetson remains ruled out: withdrawn from Russia, parallel import only.

---

## Stage 5 — Enclosure · the cost that surprises people

| Environment | Requirement | Rough cost |
|---|---|---|
| Packaged product, dry area | Basic industrial box | 5–10k ₽ |
| Open product, no washdown | Food-safe materials, sealed | 15–30k ₽ |
| **Washdown zone** | **IP65/IP69K stainless steel** | **40–100k ₽+** |

⚠️ Washdown can cost more than the camera. This is why survey questions 7 and 9 matter, and why "can we mount after the packaging step instead?" is worth asking — moving the inspection point downstream of packaging can remove this cost entirely.

---

## Indicative MVP budget

| Scenario | Total |
|---|---|
| **Static station, packaged product, dry** — the ideal MVP | **~60–90k ₽** |
| Slow conveyor, packaged, dry | ~90–130k ₽ |
| Fast conveyor, open product, washdown | ~250–400k ₽ |

Quote the outcome, never the line items — the same rule that applied before. The client should see one price for a working inspection station, not a parts list they can price-check on Ozon.
