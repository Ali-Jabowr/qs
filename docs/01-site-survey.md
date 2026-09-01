# Site Survey — Factory Visit Kit

**Purpose:** come back from one visit with everything needed to spec hardware, choose the model approach, and quote a price. Nothing gets bought before this is filled in.

**Take:** phone (video + photos), tape measure, notebook, a box or bag for sample units.

> Questions are written in Russian to ask directly. The *italic note* under each says why it matters — that's for you, not them.

---

## A. Product and line

**1. Что именно вы производите? Сколько разных SKU идёт по этой линии?**
*Determines defect classes, model approach, and whether one model covers the line or you need several. If >5 SKUs on one line, per-SKU models don't scale — you need one model plus SKU selection from the line's current job.*

**2. Сколько единиц в минуту идёт по линии? Сколько за смену? Сколько смен в день?**
*The single most important number. >60/min forces global shutter + strobe + hardware trigger. <60/min allows much cheaper hardware. Units/shift drives the ROI arithmetic.*

**3. Ширина конвейера и скорость ленты (м/мин)?**
*Sets field of view, lens focal length, and exposure time needed to freeze motion.*

**4. Продукты идут отдельно друг от друга или вплотную / внахлёст?**
*Separated units = one image per unit, easy. Touching or overlapping = segmentation problem, much harder.*

**5. Размер продукта — минимальный и максимальный (мм)?**
*Field of view and required resolution. Small defects on large products is the expensive combination.*

---

## B. The inspection point

**6. В каком месте линии нужно проверять? Покажите точку.**
*Photograph it. Everything below refers to this exact spot.*

**7. В этой точке продукт открытый или уже упакован?**
*Open product = hygiene rules on anything mounted above it, food-safe materials, possible washdown enclosure. Packaged = far cheaper and simpler install.*

**8. Сколько места есть над линией и сбоку? (измерить рулеткой)**
*Record height above belt, clearance left/right, and what you can physically bolt to. Photograph with the tape measure in frame.*

**9. Есть ли мойка оборудования водой? Как часто, чем?**
*Washdown means IP65/IP69K stainless enclosure — a real cost line. Determines whether a mini-PC on a shelf is acceptable.*

**10. Температура, влажность, пыль (мука, сахар, пар)?**
*Condensation kills electronics and fogs lenses. Flour dust coats them. Both are solvable but must be designed for, not discovered.*

**11. Какое сейчас освещение? Есть ли окна или дневной свет рядом?**
*Ambient light variation across shifts is the #1 silent killer of food CV models. Daylight near the line means the enclosure must block it out entirely.*

---

## C. Defects — what actually goes wrong

**12. Назовите три самых частых дефекта. И три самых дорогих.**
*These are usually different lists. Build for the expensive ones, not the frequent ones.*

**13. Как проверяют сейчас? Сколько человек на этой операции?**
*Establishes the baseline you must beat and the labour cost you're displacing.*

**14. Сколько брака проходит незамеченным? Как вы это узнаёте?**
*If they can't answer, their real problem is that they have no measurement — which is itself something you can sell.*

**15. Что происходит, когда брак уходит клиенту? Возврат, штраф, жалоба?**
**16. Сколько стоит один такой случай — в рублях?**
*This is the ROI number. Without it you cannot price. Push for an actual figure, even a rough one.*

---

## D. Packaging, label and date code — ask these carefully

*This is the highest-value, most reliable part of the system. Spend real time here.*

**17. Что печатается на упаковке — дата, партия, срок годности? Каким принтером (каплеструйный, лазерный, термотрансферный)?**
*Inkjet smears and drifts, laser is crisp, thermal transfer fails as the ribbon runs out. Each has a different failure signature to detect.*

**18. Что именно бывает не так с маркировкой — отсутствует, нечитаемая, смазанная, кривая, неверная дата?**
*Presence, legibility, position, and correctness are four separate checks with four different implementations.*

**19. Бывают ли проблемы со швом или герметичностью упаковки?**
**20. Проверяет ли это кто-то сейчас? Есть ли уже система контроля печати?**
*If a print-verification system exists, find out why it isn't trusted — that's your opening.*

**21. Был ли случай, когда партию вернули из-за маркировки?**
*One retailer-rejected pallet is usually the whole business case, told as a story you can reuse in every future sales call.*

---

## E. What should the system do

**22. Что должно происходить при обнаружении брака — сигнал оператору, автоматический сброс, или просто запись в отчёт?**
*Operator alert = no PLC, no actuator, no safety sign-off. Strongly prefer this for the MVP.*

**23. Если автоматический сброс — какой ПЛК стоит на линии (марка, модель)?**
*Only needed if they insist on inline rejection. Phase 2 regardless.*

**24. Нужен ли отчёт / архив фотографий по каждой партии?**
*Say yes even if they hesitate. Traceability under ХАССП has independent regulatory value and is cheap for you to provide. ⚠️ Verify the exact ХАССП/ТР ТС 021/2011 requirements with your friend on the legal side.*

---

## F. Practical and commercial

**25. Есть ли Ethernet и розетка 220В рядом с этой точкой?**
**26. Кто принимает решение о покупке? Какой бюджет?**
**27. Когда вам это нужно?**

**28. Что для вас значит «работает»? Какой процент брака система должна ловить, чтобы вы её купили?**
*⚠️ The most important question on this sheet. Get a number and write it down verbatim. This is your acceptance criterion, and without it "it works" is whatever they decide later.*

---

## G. What to physically bring back

Hardware selection is impossible without these. Do not leave without them.

- [ ] **Video, 5+ minutes**, of the line running at normal speed, filmed from where the camera would actually mount
- [ ] **Slow-motion video** of the same spot if the phone supports it — reveals whether motion blur will be a problem
- [ ] **Photos of 30+ good units** at the inspection point
- [ ] **Every defective unit they have** — photograph each, and take the physical samples if they'll allow it. Defective samples are the scarcest resource in this entire project.
- [ ] **Photos of packaging with date codes** — good print and bad print, as many bad ones as exist
- [ ] **Photo of the mounting space** with a tape measure in frame
- [ ] **Photo of the nearest power and network point**
- [ ] **Photos of the lighting** at two different times of day if a shift change or daylight is involved

---

## H. Fill in before leaving

| Field | Answer |
|---|---|
| Product / SKUs | |
| Units per minute | |
| Units per shift · shifts/day | |
| Product separated or touching | |
| Open or packaged at check point | |
| Washdown? | |
| Top defect by cost | |
| Cost of one escape (₽) | |
| Date code printer type | |
| On fail: alert / reject / log | |
| Acceptance % (Q28, verbatim) | |
| Decision maker · budget | |
