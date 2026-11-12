# Industry practice: detecting empty or missing items in clear containers, blisters and tray slots

Scope note: about 19 searches and fetches, run on 2026-11-09. Most sources that turned up are vendor or trade-press material. Peer-reviewed numbers for empty-pocket detection were scarce. Two pages (Pepperl+Fuchs, computer-automation.de) returned HTTP 403, so the claims from them below come from search snippets only and are marked that way.

## 1. Blister empty-pocket and cup/tray fill inspection: vision setups and lighting

### Takeaway
For clear blisters, deployed practice is a **backlight under the transparent web**. A missing product lets light through, so the pocket shows as a bright blob. Opaque formats (Alu-Alu) switch to top coaxial or diffuse light and use reflectance differences instead. For cup and tray fill checks, practice splits into 2D vision for presence and fill, 3D or light-section sensors where contrast is low (white product in white trays), and checkweighers as a non-vision alternative.

### Cited Findings
- Jornen (blister machine maker): on PVC blisters, an "illuminant plate" sits directly under the web track. Light "freely penetrates the transparent PVC" when a pocket is empty. A filled pocket shows as a dark pixel cluster and an empty one as high intensity. A downward camera sits in a stainless canopy. For Alu-Alu, the under-light is turned off and top coaxial light is used. No accuracy or speed figures are given. — [Jornen Blister Inspection System](https://jornen.com/?p=2361); [Jornen packaging-line listing](https://trends.directindustry.com/jornen-machinery-co-ltd/project-117985-1101589.html)
- Frain Group whitepaper: backlighting, with the light behind a transparent blister and blocked by the tablets, "may be used to reduce processing". If you only need presence or absence of a complete tablet, that is sufficient. — [Frain vision inspection whitepaper](https://www.fraingroup.com/wp-content/uploads/5-30-17-Vision-inspection-Whitepaper-Whitepaper.pdf)
- Trade press: blister lines use diffuse, direct and backlighting, mainly to reduce shadows and reflections. Rejected packs are diverted automatically. — [Eureka Magazine](https://eurekamagazine.co.uk/content/technology/vision-systems-advance-quality-control-capabilities)
- A tutorial blog of uncertain authority suggests a backlit station for pack integrity, low-angle or coaxial light for the foil side, and diffuse light for print. — [detectdefects guide 2026](https://detectdefectsjekyll.pages.dev/blog/2026/03/20/pharmaceutical-blister-pack-inspection-machine-vision)
- Cup/pudding fill (academic, IRJET): a camera flags a partly filled pudding cup and an arm removes it. A fill-height check only needs the top quarter of the cup in the image (a smaller ROI means less processing). The paper also offers a conveyor scale that compares the cup to a full-cup reference weight as an alternative. — [IRJET v8i1003](https://issuu.com/irjet/docs/irjet-v8i1003)
- Simple cup filling-sealing machines (ZONESUN ZS-FS100, rotary): cup presence comes from an **"electric eye"** (photoelectric), not a camera. The machine "can detect cups accurately and control the machine to work", meaning it skips filling when no cup is there. — [ZONESUN ZS-FS100](https://www.zonesunautopack.com/products/zs-fs100-automatic-rotary-filling-capping-sealing-machine)
- Coffee capsule line (Seidenader + MVTec HALCON): cameras inspect empty capsules before filling, for frayed edges and deformities, and again after sealing. A light barrier triggers the cameras. Throughput is up to 500 capsules/min across two cameras, with pneumatic reject. — [Vision Systems Design](https://www.vision-systems.com/factory/consumer-packaged-goods/article/16748051/automated-inspection-cameras-capture-coffee-capsules); [MVTec success story](https://www.mvtec.com/company/success-stories/cameras-capture-coffee-capsules-seidenader)
- Yogurt (dairy): Industrial Control's ThermoChecker is a 3D smart camera on thermoformed cups. It runs at 600 cups/min, inspects every 3 degrees within 30 ms, and takes up to 130 profiles per part. It checks rim and seal defects, **not** empty or fill status. It is unaffected by the intense curing lamp. 11 systems were deployed across 5 plants, but no false-reject figure is given (2014). — [Dairy Foods](https://dairyfoods.com/articles/90605-yogurt-maker-turns-to-vision-system-to-eliminate-leakers-dented-cups); [Packaging World](https://www.packworld.com/home/article/13366366/3d-vision-inspection-assures-cup-quality)
- Yogurt-pot filling: Opdi-tex sells AI vision QC for yogurt pot filling. They say the teach-in "only takes five rows of yoghurt pots" before the system learns the target. (Search snippet only; the page returned 403.) — [computer-automation.de](https://www.computer-automation.de/images/3/news-about-cameras--amp--co-.htm)
- Food trays: Pepperl+Fuchs uses a **light-section (laser line) sensor** to check whether food-tray compartments are filled. They argue that standard 2D vision sensors need extra lighting and complex setup to see low-contrast white trays reliably. (Search snippet only; the page returned 403.) — [Pepperl+Fuchs](https://pepperl-fuchs.com/usa/en/34805.htm)
- Confectionery trays: LHT and Eagle Vision's "Dirty Tray Vision" checks for leftover jelly on empty trays with an optional extra camera (2016). — [Confectionery News](https://www.confectionerynews.com/Article/2016/02/01/LHT-Consultancy-and-Eagle-Vision-create-Dirty-Tray-Vision-system/)

### Inferences
- The blister backlight trick works because the *carrier is transparent and the product is opaque*. Clear cups on a **polished steel channel** cannot be backlit from below, because steel is opaque. The equivalent move is to make the background behind the cup differ strongly from the product, for example a matte or contrasting insert in the channel, or a fixed diffuse top light. That is an inference, not a sourced recommendation.
- Moti-like dough is light-coloured and matte, and possibly white-on-steel. This is the "low-contrast white" case P+F warns about for 2D. Height (3D or laser line) is the industry fallback when 2D contrast fails. For v1 on one USB camera, a stable diffuse light and a fixed ROI per pocket are the realistic levers.

### Gaps
- I found no source that describes the exact vision and lighting setup on mochi, jelly or pudding cup-fill-seal machines. The machine makers in that segment (mostly Asian rotary and linear fillers) appear to use photoelectric "no cup, no fill" sensing, not cameras.
- I found no vendor documentation from Hamba, Grunwald or Trepko on empty-cup detection.

## 2. Telling "carrier present but empty" apart from "no carrier" without a sensor

### Takeaway
In industry, this is solved by **machine timing, not by the camera**. Inspection is triggered or bucketed per machine index or encoder pitch, so every slot gets a result, including empty ones, and results are tracked in a shift register keyed to the index. A camera that only fires when it sees a part is the classic failure: it skips empty slots, and results slide out of step with the physical positions.

### Cited Findings
- Practitioner failure (SLC-5/04 with two cameras on an indexing conveyor): the BSL shift register only shifted on the camera's "evaluation done" bit. An empty slot produced no trigger and no done bit, so the array **fell out of sync with the belt after a few missed parts**. Recommended fixes:
  1. Shift on every physical index (rising edge of an index-complete signal), whether or not a part is there, and track presence separately.
  2. A timeout override of about 200–500 ms that forces the shift if the camera doesn't answer. Described as a patch.
  3. Most robust: a static array with a read pointer computed from **encoder counts or index count**.
  4. Validation: compare the actual trigger count with the expected count and flag mismatches.

  — [Industrial Monitor Direct KB](https://industrialmonitordirect.com/blogs/knowledgebase/resolving-slc-504-vision-inspection-missing-part-tracking)
- Quality Magazine: encoder (distance-based) input removes inspection timing errors that a photoelectric trigger alone cannot keep aligned while parts move. — [Quality Magazine](https://www.qualitymag.com/articles/91113-encoder-input-improves-part-inspection)
- Indexing machines send a "start inspection" signal after each index step, and the controller flags previously rejected items as they enter. — [US5101609 cigarette package indexing wheel](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5101609)
- Pick-and-place patent: if the vision system doesn't know a cycle was a deliberate non-placement, it reports a false placement error. A separate sensor marks the "nothing attempted" cycles. — [US7559134](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7559134)
- Jornen blister: camera-to-punch distance is a fixed **stroke count** that the operator enters in the HMI. If the entry is wrong, good cards get rejected and bad ones pass. Pitch tracking is a configuration risk, not just a code risk. — [Jornen](https://jornen.com/?p=2361)
- Cognex In-Sight practice: one presence tool per cavity (8 tools for 8 cavities), combined in spreadsheet logic into "all present" and "all clear" outputs. Presence is evaluated per fixed ROI. Third-party blog. — [Industrial Monitor Direct, Cognex multi-sensor logic](https://industrialmonitordirect.com/blogs/knowledgebase/cognex-in-sight-multi-sensor-logic-for-presence-detection)
- Projected-light-line pocket-status detection on IC trays is patented. A camera reads a laser line, which bends or breaks over an occupied pocket. — [US10438338](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10438338)
- Empty-tray recognition with a reference-image approach (patent application). — [EP4083856](https://data.epo.org/publication-server/rest/v1.2/patents/EP4083856NWA1/document.html); [EP4369291A1 "identifying empty trays"](https://patents.google.com/patent/EP4369291A1/de)

### Inferences
- Without an encoder or index signal in v1, the camera has to **derive the machine pitch itself**. The ways to do that:
  - Detect the cup rim or the chain link as the "carrier" feature, which is independent of whether the cup is filled.
  - Detect chain motion and stops from frame differencing, then evaluate on settled frames only.
  - Count pitches by tracking the carrier features.

  "No carrier" (a gap in the chain) means no rim feature at the expected pitch position. "Carrier present but empty" means a rim feature with no product inside the pocket ROI. This mirrors the industry split of carrier tracking versus presence evaluation, done in software.
- The sourced failure modes map directly onto this setup:
  - Event-triggered evaluation skips empty or gap slots, so counts drift.
  - A wrong pitch offset causes systematic misattribution.

  Logging expected count against observed count per stop, as in the SLC fix 4, is a cheap guard.
- Frequent chain stops help a 30 fps camera, because stationary frames avoid motion blur and give multi-frame voting. The tracker must still not double-count a cup that is stopped under the camera.

### Gaps
- I found no published case of a camera-only (sensorless) pitch-tracking system on a cup filler. Everything I found uses a PLC index signal, an encoder or a light barrier.
- No source covers reference marks on chains specifically for vision sync.

## 3. Commercial smart-camera presence/absence tools and transparent-packaging advice

### Takeaway
All the vendors sell the same primitives: per-ROI brightness, contrast, pixel-count or blob, pattern, edge, and now "AI classification trained on a few images". Their advice for transparent items is consistent: **lighting first** (backlight, dark field, diffuse dome, polarized or coaxial, UV for fluorescent materials), then colour or AI tools.

### Cited Findings
- Keyence: "the choice of illumination significantly influences object detection". Cited options are Dark Field and Full Bright Field Diffuse for contrast and uniformity. Transparent or low-contrast parts need "advanced filtering, color processing, and resolution". UV light suits fluorescent coatings, and pattern-projection (3D height) light suits low-contrast parts. IV series is for simple presence checks, XG-X for fill levels in "transparent or amber bottles", and CV-X for multi-camera lines. — [Keyence presence/absence](https://www.keyence.com/products/vision/applications/presence-and-absence-inspection.jsp)
- Cognex: missing-item tools are "trained using only a few images" to classify packages as good (all items present) or NG. Count-based checks pass or fail on a programmed count. The In-Sight 2800 is marketed for presence/absence in totes and trays. — [Cognex missing item inspection](https://www.cognex.com/industries/food-and-beverage/assembly-verification/missing-item-inspection); [Cognex missing product](https://www.cognex.com/industries/consumer-products/packaging/missing-product-detection); [In-Sight 2800 applications](https://www.cognex.com/products/item-detection/in-sight-2800-detector/applications)
- Cognex vision-sensor literature lists brightness, contrast, pattern, pixel count, edge and circle tools for presence/absence. — [Cognex vision sensors product guide (Isotron mirror)](https://cdn.i-pulse.nl/isotron/production/products/939/product-guide-vision-sensors-en.pdf)
- NI: illumination is the particular challenge with transparent materials. A seal inspection uses a backlight with a polarizing plate and a crossed polarizer on the lens. — [NI white paper](https://www.ni.com/en-il/innovations/white-papers/06/inspecting-transparent-materials-for-material-defects.html)
- Patents on transparent containers use a backlit silhouette (also for liquid level in clear containers), a colour-gradient backlight, and polarized transmission. — [US11328380](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11328380); [US10422755](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10422755); [US9080986](https://patents.google.com/patent/US9080986)
- Banner: clear targets let standard sensors "burn through", or reflect light unpredictably and cause false reads. Polarized coaxial optics resist false detection from reflections. — [Banner polarized coaxial](https://www.bannerengineering.com/be/en/company/expert-insights/benefits-polarized-coaxial-optics-clear-object-detection.html); [Banner clear bottles](https://bannerengineering.com/ca/en/solutions/smart-sensors/consistent-detection-of-transparent-plastic-bottles-along-a-conv.html)
- A clear container can reflect enough light to mimic an empty detection zone. — [US5828056](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5828056)
- SICK Inspector and Omron FH: **no relevant app note found** in this pass.

### Inferences
- Over polished steel, the dominant risk is **specular glare**: the steel channel and the clear cup walls create hot spots that look like "bright = empty" or wash out the product. Industry counters it with diffuse (dome or on-axis) light, crossed polarizers, or shifting the camera angle off the specular path. For a USB camera, a cheap linear polarizer on the lens plus a polarized or diffuse ring light is the low-cost analogue. That is inferred, not sourced for this exact case.
- Make the decision on the **pocket interior ROI** (product colour or texture against the empty-cup appearance over steel), not on the cup outline, which is transparent and glare-prone.

### Gaps
- No vendor app note specifically covers clear cups on polished steel.
- No SICK or Omron material was retrieved.

## 4. Case studies with numbers, and reported failure modes

### Takeaway
Hard, independent numbers for empty-pocket detection rate and false-reject rate are **almost absent** from public sources. What exists is vendor marketing (around 99.5–99.6%) and academic accuracy on curated datasets. The failure modes practitioners report are mostly about **sync and tracking, glare on clear materials, and configuration errors**, not classifier accuracy.

### Cited Findings
- iFactory (vendor) claims 99.6% detection for fill-level variance in dairy and about 99.5% for missing tablets and empty blister pockets. These are marketing figures, not independently validated. — [iFactory dairy](https://ifactoryapp.com/industries/food-manufacturing/real-time-ai-vision-inspection-for-dairy-processing-food-manufacturing-operators); [iFactory blister 99.5%](https://ifactoryapp.com/ai-vision-camera/ai-vision-pharma-bliste-pack-inspection)
- IJERT paper (YOLOv8 with attention, for blister defects): YOLOv8 is reliable on "obvious defects such as missing and broken tablets" but less sensitive to subtle surface defects. Year not confirmed. — [IJERT](https://www.ijert.org/?p=52974)
- An improved U2Net study on blister-tablet surface defects reports 99% average accuracy and 96.3% average precision. This is for surface defects, not empty pockets. — [U2Net blister record](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c357t-a22b8c22fc1be4d8b6d9043edf0014e06b8e239768accc4b97d75af5b2852f43/Details)
- Reported failure modes:
  - Results desync when empty slots never trigger the camera. — [IMD KB](https://industrialmonitordirect.com/blogs/knowledgebase/resolving-slc-504-vision-inspection-missing-part-tracking)
  - A wrong stroke-count offset rejects good packs and passes bad ones. — [Jornen](https://jornen.com/?p=2361)
  - Clear objects cause false reads from reflections and burn-through. — [Banner](https://bannerengineering.com/ca/en/solutions/smart-sensors/consistent-detection-of-transparent-plastic-bottles-along-a-conv.html)
  - Glass-bottle false triggering is fixed by response delay or smoothing, plus a PLC debounce requiring N consecutive scans. — [IMD glass bottle](https://industrialmonitordirect.com/blogs/knowledgebase/resolving-glass-bottle-lug-detection-false-triggering-at-7in-distance)
  - Recommended validation practice: count missed detections (present but read FALSE) and false positives (absent but read TRUE) in a PLC over a large controlled test. — [IMD clear film](https://industrialmonitordirect.com/blogs/knowledgebase/sensing-clear-polyethylene-film-sensor-selection-configuration-for-bagging-applications)
- Throughput reference points: 500 capsules/min (two cameras, light-barrier trigger) — [MVTec](https://www.mvtec.com/company/success-stories/cameras-capture-coffee-capsules-seidenader); 600 cups/min, 30 ms per inspection (3D) — [Dairy Foods](https://dairyfoods.com/articles/90605-yogurt-maker-turns-to-vision-system-to-eliminate-leakers-dented-cups). The target line runs at about 56 cups/min, an order of magnitude slower.

### Inferences
- At 56 cups/min (about 1.07 s per cup) and 30 fps, there are about 32 frames per cup pitch. That leaves plenty of room for multi-frame voting and debouncing, the camera analogue of the PLC "N consecutive scans" fix.
- Since there are no trustworthy public benchmarks, the project needs its own acceptance test, built the way practitioners do it: a controlled run with known empty and gap positions, logging misses and false rejects separately.

### Gaps
- No independent, published false-reject rate for empty-cup or empty-pocket detection.
- No forum thread specific to clear cups on steel turned up.

## 5. Food-safety notes on equipment mounted over open food (brief)

### Takeaway
Cameras over open product go in hygienic stainless housings: AISI 316L, Ra below 0.8 µm, IP69K, rounded radii, and EHEDG-guided design. Vendors use "certified" and "designed to guidelines" inconsistently.

### Cited Findings
- autoVimation hygienic kit:
  - AISI 316L, Ra below 0.8 µm, IP69K, built for direct food contact and dripping zones.
  - EHEDG certification is "in progress".
  - Mounting uses stainless rods and hygienic T-pieces.

  — [New Equipment Digest](https://www.newequipment.com/product-directory/facility-operations/product/55117324/68801-autovimation-introduces-the-hygienic-machine-vision-building-kit)
- Baumer VCXG.I housing: 1.4404 stainless, IP69K, designed to EHEDG guidelines, 3 mm radii, electropolished Ra below 0.8 µm. It has separate "non-food zone" configurations. — [Baumer housing quick-start](https://media.baumer.com/Baumer_QuickStart_Housing-Base-SetB-v4_EN_11215493.pdf)

### Inferences
- A bare USB webcam over open moti is fine for a pilot, but production would need at least an enclosed, washdown-rated housing above the food zone. It should have no loose parts and should use a plastic or shatterproof window instead of glass, which avoids foreign-body risk. The glass-avoidance point is common food-plant practice, not sourced here.

### Gaps
- I did not retrieve EHEDG or FDA text on placement rules for equipment directly over open product (drip and drainage requirements, glass policy).
