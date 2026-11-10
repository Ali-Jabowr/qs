# Optical and lighting techniques that make clear plastic cups visible to a camera (presence + empty/filled)

Scope note: about 15 searches and fetches. Sources are mostly vendor and integrator notes, patents, and a few papers. Two primary pages could not be fetched (Vision Systems Design article: HTTP 403; iim-ag dark-field page: HTTP 404), so their content below comes from search snippets only. No source covered the exact setup (clear cup, polished steel channel, overhead USB camera), so everything specific to that setup is under **Inferences**.

## Q1. Lighting geometry: backlight/bright-field, dark-field / low-angle ring, dome/diffuse, coaxial. Which show thin clear plastic, and which fail on polished steel?

### Takeaway
Clear plastic shows up in two classic ways. (a) **Bright-field diffuse backlight** turns refracting edges and walls dark against a bright field. (b) **Dark-field / low-angle light** makes rims, edges and defects scatter light into the camera, so they look bright on a dark field. Both need a background that is either uniformly bright (backlight) or uniformly dark and non-reflective (dark-field). A polished steel floor is neither, because it mirrors the light source and the surroundings.

### Cited Findings
- A flat, diffuse bright-field backlight "works well for transparent objects such as bottles where, due to refraction of the light rays, the edges appear dark". Air bubbles refract light and also turn dark, which makes inclusions visible. — [Microscan lighting tips white paper](https://files.microscan.com/_att/7e54b2b8-570c-4366-a740-5da5b0e5336c/lighting_tips_white_paper.pdf) (via search snippet)
- Backlighting is recommended for transparent bottles because "backlighting highlights fill level or capsules by the outline created". — [NI forum: Transparent bottle detection](https://forums.ni.com/t5/Machine-Vision/Transparent-bottle-detection/m-p/813954) (practitioner forum, low authority)
- Dark-field principle: light aimed at a transparent bottle from outside the camera's view mostly passes through unseen, while edges and defects scatter light into the lens and appear bright on a dark background. A forum reply recommends a dark-field illuminator for the top rim of transparent bottles. — [Vision Systems Design: Lighting exposes bottle defects](https://www.vision-systems.com/boards-software/article/16738638/lighting-exposes-bottle-defects) (fetch returned 403; claims taken from the search snippet); [NI forum](https://forums.ni.com/t5/Machine-Vision/Transparent-bottle-detection/m-p/813954)
- Case study (clear plastic bottle necks): an LED ring light lit **only the bottle neck** as bottles passed close under a vertically mounted camera. Dark-field light is emitted mostly horizontally, so only objects in that horizontal band get lit. — [Vision Systems Design](https://www.vision-systems.com/boards-software/article/16738638/lighting-exposes-bottle-defects) (snippet)
- Low-angle dark field can be built from spot or bar lights at low angles of incidence. **Backlit dark field** is used to highlight defects in semi-transparent or transparent objects. — [iim-ag Lumimax: dark-field lighting](https://www.iim-ag.com/en/lumimax/knowledge/topic-1-lighting-angle-influence-of-reflected-light-and-backlight-on-the-camera-view/reflected-light/dark-field-lighting.html) (snippet; page returned 404 on fetch)
- Transparent-sheet inspection patents pair a **dark stripe source normal to the sheet** with a dark-field source at **about 10–50° from normal**. — [US8072593B2](https://patents.google.com/patent/US8072593B2/Direct); [WO2008100683A1](https://patents.google.com/patent/WO2008100683A1/enExample)
- Keyence lists transparent surfaces as a hard case and says detection on film "is normally difficult or impossible due to the hot spots generated from surface irregularities or glare caused by the film sheet". — [Keyence LED illumination selection guide](https://acomee.com.mx/catalogos/catalogosk/KEYENCE%20LED%20ILLUMINATION%20FOR%20MACHINE%20VISION%20SELECTION%20GUIDE.pdf) (snippet)
- Keyence's IG-series "glass edge mode" relies on the fact that "more light is blocked by the edge of the target than the interior". In other words, the edge of a transparent object is its strongest signal in transmitted light. — [Keyence IG manual p.70](https://www.manualsdir.com/manuals/657976/keyence-ig-series.html?page=70)
- A display-device patent notes that transparent film edges are "sometimes dark and sometimes very transparent", depending on conditions, which makes edge detection unstable. — [US5107131A](https://patents.google.com/patent/US5107131)
- On-axis diffuse (coaxial) light makes flat specular surfaces look bright, and anything that tilts the surface looks dark. One maker uses it to find short-shot (missing material) areas in plastic bottles. — [Vision Systems Design](https://www.vision-systems.com/boards-software/article/16738638/lighting-exposes-bottle-defects) (snippet)

### Inferences
- **Polished steel breaks dark-field.** Dark-field needs a dark background, but a mirror floor reflects the low-angle ring into the camera wherever its surface normal lines up (scratches, weld seams, chain links). That gives bright clutter that looks just like the rim of a cup. **Coaxial or bright-field from above is the worst choice**: flat polished steel reflects it straight back and saturates the image, so a clear cup on top adds little contrast.
- **Backlight through the floor is not possible on an opaque steel channel.** The nearest substitutes are: (1) a horizontal **side backlight** across the channel, with the camera looking sideways or at an angle so the cup walls show dark refraction edges; or (2) using the steel as a **controlled mirror**, i.e. putting a uniform diffuse panel (or a patterned panel, see Q3) where the mirror reflection lands, so the camera sees a "virtual backlight" through the cup. Both are design ideas, not reported practice.
- An overhead camera looks at the cup rim and floor face-on. The rim (a thicker, curved lip) is the best dark-field and refraction target. The wall, seen almost edge-on from above, appears only as a thin ring.

### Gaps
- I found no vendor note on clear plastic over a **polished metal** background. All transparent-object guides assume a backlight or a matte dark background.
- No quantitative contrast figures (for example grey-level difference) for clear PP/PET cups under each type of light.

## Q2. Polarisation: cross-polarised glare suppression, stress birefringence (photoelastic effect) in PET/PP, and polarisation cameras vs a cheap polariser film

### Takeaway
Cross-polarisation (one polariser on the light, one on the lens at 90°) is standard practice for killing specular glare from metal, but it costs about 6–10x in light. Plastics are birefringent, so a clear cup placed between polarisers shows bright, coloured stress patterns while the background stays dark. That makes it the most promising way to tell an **empty clear cup from no cup**, but it works best in **transmission** (light passing through the cup), and visibility depends on the material, thickness and stress. Sony Polarsens cameras compute degree and angle of linear polarisation (DoLP/AoLP) per pixel. Patents use these cues to segment transparent objects.

### Cited Findings
- Cross-polarisation: a polariser on the source and one on the lens at right angles block specular reflections. **All** light sources, including ambient light, must be polarised. — [VA Imaging: Polarization in machine vision (2025)](https://va-imaging.com/en-us/blogs/machine-vision-knowledge-center/polarization-in-your-machine-vision-application)
- Light cost: a polarised ring light showed "a tenfold reduction of light intensity when illuminating a reflecting item", and exposure went from **1 ms to 10 ms**. On diffuse objects, intensity had to rise **6–10x** to match unpolarised light. Most polariser filters work at **400–780 nm**. — [VA Imaging](https://va-imaging.com/en-us/blogs/machine-vision-knowledge-center/polarization-in-your-machine-vision-application)
- Real case: automotive canister interiors (metal) were inspected with all three cameras cross-polarised using Midwest Optical PR032 polarisers plus BP550 UV/NIR-block filters, because standard polarisers do not polarise outside the visible band. — [EPIC Systems: Eliminating glare with polarization](https://epicsysinc.com/blog/eliminating-glare-with-polarization/); [Vision Systems Design case](https://www.vision-systems.com/cameras-accessories/article/16737367/optical-filters-reduce-glare-in-automotive-canister-machine-vision-inspection-application)
- Birefringence: plastics, glass and ice show internal stress when lit with polarised light and viewed through a second (analysing) polariser. — [RIT: Intro to polarized light](https://s3.cad.rit.edu/cadgallery_production/storage/media/uploads/faculty-f-projects/1724/documents/337/intro-to-polarized-light.pdf)
- Polariscope practice: the part sits in front of a **polarised backlight**, and internal stress shows as "bright, colorful strain patterns". Visibility "depends on material transparency, thickness, stress condition, optical setup and inspection environment". — [Mecan MRV-ST polariscope](https://www.mecanusa.com/Strain-Inspection-Polariscope/Polariscope-Strain-Inspection-MRV-ST.htm)
- Injection-moulding inspection patent (2024-era): circularly polarised light is shone through a moulded part and the striped stress pattern is imaged with a polarisation camera. — [US 12259335](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12259335)
- Container patent: stress in a container sidewall changes the polarisation of light passing through it and produces a bright signal on the sensor. — [US 6067155: inspection of transparent containers using IR and polarized visible light](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6067155)
- Polarsens: the Sony IMX250MZR has an on-chip 4-pixel polariser block (0/45/90/135°), so angle and degree of polarisation can be computed per pixel. — [LUCID: Sony's polarized sensor](https://thinklucid.com/?p=9042)
- Transparent materials (glass, plastics, water) partially polarise refracted light. — [LUCID](https://thinklucid.com/?p=9042)
- Vendors market Polarsens cameras for reducing glare on "glass, plastic, and metal" and offer AoLP, DoLP and Stokes outputs. — [Opto Engineering ITALA](https://www.opto-e.com/en/news/Discover-the-new-ITALA-cameras-with-polarized-sensors)
- The Imaging Source says its polarisation cameras can show material stress, reduce glare and improve edge contrast on low-contrast materials. — [The Imaging Source: polarization cameras](https://www.theimagingsource.com/en-us/resource/glossary/polarization-cameras-in-machine-vision/); [5 MP polarization cameras](https://www.theimagingsource.com/en-us/resource/news/2019/09/27)
- Stokes parameters can be mapped to HSV to visualise stress in plastic. — [Stemmer Imaging / computer-automation.de](https://www.computer-automation.de/image-processing/the-third-dimension-of-light.htm)
- A patent family segments transparent objects (glass, acrylic) using polarisation cues, because transparent objects have no texture of their own and RGB cameras miss them. — [US 11302012](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11302012); [US 12380568](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12380568)

### Inferences
- **Crossed polarisers make the background dark and the cup bright.** Polished steel keeps polarisation in a specular reflection, so with crossed polarisers the steel's mirror glare is largely removed. A thermoformed or injection-moulded PP/PET cup has moulding stress and orientation, so it rotates the polarisation and lights up. That gives positive contrast for an empty cup, which is exactly the "empty cup vs no cup" signal. This is a strong hypothesis, not a reported result. It needs a bench test with real cups.
- Geometry: light goes polarised → down through the cup → hits the steel → reflects back up through the cup → passes the crossed analyser on the lens. This is a "double-pass reflection polariscope". The steel acts as the mirror. Scratched or brushed areas of steel depolarise light and will leak through as bright noise.
- Thin, low-stress cups (for example highly crystalline PP) may give weak birefringence. The polariscope vendor itself warns that visibility depends on thickness and stress.
- **Cheap route:** linear polariser film over an LED panel or ring, plus a polariser (filter or film) on the USB camera lens, rotated to extinction. That costs very little compared with a Polarsens camera. The cost is about 6–10x less light, which matters for the short exposures needed at 30 fps on a moving chain. Polariser film is usually PVA/iodine laminate, so it should sit outside the open-food zone or behind a food-safe cover.
- A Polarsens camera gives DoLP/AoLP in one shot and needs no rotation tuning, but none of the sources give a price, and it would replace the cheap USB camera.

### Gaps
- No published birefringence measurements or photos of **thermoformed PP or PET food cups** under crossed polarisers.
- No price data for Polarsens cameras or polariser film in these sources.
- No source tests polarisation over polished vs brushed stainless steel specifically.

## Q3. Background pattern and refraction distortion (background-oriented schlieren style, deflectometry)

### Takeaway
A transparent object distorts whatever textured background sits behind it, and the strength of the distortion depends on its shape and refractive index. Background-oriented schlieren (BOS) measures this as an image displacement field, using cross-correlation or optical flow against a reference image of the empty background. For a fixed station, a reference "no-cup" frame of a patterned background plus a displacement or difference check is a direct, cheap presence test that does not need the cup itself to have visible texture.

### Cited Findings
- Refraction distorts the background, and how much depends on the object's shape and refractive index. — [PMC5030754](https://pmc.ncbi.nlm.nih.gov/articles/PMC5030754)
- Ordinary local features fail on glass and plastic because those objects take on the look of whatever is behind them. Modelling the **distortion** (the light-field distortion feature) enables recognition. — [Maeno et al., CVPR 2013](https://openaccess.thecvf.com/content_cvpr_2013/html/Maeno_Light_Field_Distortion_2013_CVPR_paper.html)
- Single-image reconstruction of thin transparent surfaces is possible from background distortion using coded light-field probes. Diffuse background patterns are a popular way to analyse such distortions, but these methods often need several images or cameras. — [Wetzstein et al., ICCV 2011](https://www.cs.ubc.ca/labs/imager/tr/2011/RefractiveShapeFromLightFieldDistortion/RefractiveShapeFromLightFieldDistortion-Wetzstein2011.pdf)
- BOS measures image displacement of a textured background between a reference image and a test image. It borrows PIV cross-correlation for this, and optical flow with multiscale backgrounds improves it. — [Raffel, Exp Fluids 2015](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s00348-015-1927-5.pdf); [DLR elib 2024](https://elib.dlr.de/206245/1/ELIB-Eintrag-2024-RaffelM-206245-PaperPublished.pdf)
- In BOS, cross-correlation and optical flow gave equivalent results (2024). The properties of the background pattern drive displacement accuracy. — [DLR 2024](https://elib.dlr.de/206245/1/ELIB-Eintrag-2024-RaffelM-206245-PaperPublished.pdf); [ISFV15-086](https://www.itmo.by/pdf/isfv/ISFV15-086.pdf)
- An inspection patent uses reference-free BOS because "change in intensity alone cannot be used" for transparent media. It detects background distortion with a two-stage optical flow. — [US 12602758 (building leakage detector)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12602758)
- A transparent-sheet inspection patent uses a **dark stripe** source together with dark-field, which is effectively a structured background. — [US8072593B2](https://patents.google.com/patent/US8072593B2/Direct)

### Inferences
- A cup is a much stronger refractor than the gas flows BOS is designed for. Its curved wall and rim should shift a fine pattern by many pixels, so even a crude method (normalised cross-correlation of patches, or `cv2.calcOpticalFlowFarneback` against the empty-slot reference) should flag a cup. The cup **floor** is flat and will barely shift the pattern, so the signal comes from the walls, the rim, and the floor-to-wall radius.
- Where the pattern goes: an overhead camera sees the channel floor through the cup. Options are (a) a pattern engraved or laser-etched into the steel floor, or a printed food-safe plate there; or (b) a patterned panel placed where the steel mirror reflects it into the camera (deflectometry: the specular steel itself distorts the pattern, and the cup adds a second distortion). Option (b) also turns glare into signal. Deflectometry on the bare steel will show dents and scratches, so the reference frame must be per-slot or very stable.
- Filled cups block or scatter the pattern (opaque product) or refract it strongly (liquid). That gives a three-way split from one cue: no distortion = no cup; ring distortion with the floor pattern still visible = empty cup; pattern hidden or heavily changed inside the cup outline = filled cup.
- Risks: chain position jitter changes the reference; starch powder or water films on the steel change the pattern texture; motion blur smears fine patterns. Pattern period must be several times larger than the blur length (see Q6).

### Gaps
- I found no industrial (non-academic) report of BOS or pattern distortion used for **conveyor presence of clear containers**. Evidence is from fluids research and CV papers (2011–2013), plus a 2024 BOS method paper.
- No accuracy numbers for this kind of presence detection.

## Q4. Colour / NIR / UV cues (fluorescence, IR absorption, coloured backgrounds)

### Takeaway
None of these is a cheap, proven presence cue for clear PP/PET cups. PET's natural fluorescence is weak (about 325–420 nm emission). PP is essentially non-fluorescent without dyes. Plastic absorption bands near 1200 and 1700 nm need a SWIR camera (not a USB webcam). Water's 1450 nm band is a strong fill cue in SWIR, but again needs SWIR hardware. The cheap colour option is a **saturated coloured background**: the cup and its rim refract or tint it, but this relies on the same refraction cue as Q3.

### Cited Findings
- PET bottle pieces showed "weak fluorescence in the area of 325 to 420 nm". — [Shimadzu application note](https://www.analytik-news.de/en/papers/pdf/shimadzu50e.pdf)
- Intrinsic-fluorescence sorting of six polymers (PP, PS, HDPE, PET, PA6, PMMA) used 245–345 nm excitation, not 365 nm. — [Penso 2025, Bioengineering / PMC12292781](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12292781/)
- With Nile Red dye and 365 nm LED excitation, PET and PS fluoresced and **PP did not take up the stain**. Dye fluorescence (540–640 nm) overlaps with organic matter, which also fluoresces. — [Philippine Journal of Science](https://philjournalsci.dost.gov.ph/?p=201); [Penso 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12292781/)
- Under 370 nm UV, milk fluoresces strongly while HDPE gives a lower signal (fouling-detection study). This suggests food residues may fluoresce more than the plastic itself. — [Woolley & Hill, via core.ac.uk](https://core.ac.uk/download/288353969.pdf) (snippet)
- Common plastic resins absorb near **1200 nm and 1700 nm** (about 1215 and 1732 nm features). — [Sci. Rep. 2023, s41598-023-39754-7](https://www.nature.com/articles/s41598-023-39754-7.pdf)
- Water reflectance is near zero at 1400–1700 nm and masks plastic signals there. — [NERC / ICES J Mar Sci](https://nora.nerc.ac.uk/id/eprint/532152/1/fsac006.pdf)
- Water's absorption peak around 1450 nm appears very dark to SWIR. SWIR is used to check liquid presence and fill level through glass or plastic. — [JAI: SWIR technology](https://www.jai.com/technology/understanding-swir-technology/); [Vision Systems Design SWIR](https://vision-systems.com/14299966)
- An older container patent combines IR with polarised visible light to inspect transparent containers. — [US 6067155](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6067155)

### Inferences
- UV (365 nm) on an open food line also raises eye-safety and food-exposure questions, and the plastic signal is weak. Starch or product residue may fluoresce more than the cup, which works against the goal. Not recommended as the main cue.
- Silicon USB cameras reach only about 1000 nm, so the 1200/1700 nm plastic bands and the 1450 nm water band are out of reach without a SWIR (InGaAs) camera, which is a different price class (no price was found in these sources).
- A coloured or high-contrast background is essentially the pattern idea from Q3 at low spatial frequency.

### Gaps
- No PET or PP transmission data at 1600–1800 nm in the sources found.
- No source validates 365 nm natural fluorescence as a reliable presence cue for undyed PP/PET.
- No SWIR camera prices found.

## Q5. Fill vs empty in a clear container by camera (liquid level, solid product)

### Takeaway
Classic fill checks are backlit side views that threshold or take gradients to find the meniscus or product edge. With an overhead camera, "filled" is better framed as **"inside the cup outline, the floor or background is no longer visible or is strongly changed"**. That is easier than telling empty from absent, because product (especially opaque food) adds texture and colour, while an empty cup adds almost nothing.

### Cited Findings
- Automated fill-volume detection thresholds the image to find the meniscus, which then sets the region of interest and gives the volume. Containers without a transparent base can be lit obliquely at a narrow angle instead. — [US 11803983](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11803983)
- EP3870940B1 finds the air-liquid boundary without contact using image gradients (the point of minimum gradient). — [EP3870940B1](https://patentimages.storage.googleapis.com/79/9d/4b/d713bd873ae877/EP3870940B1.pdf)
- A single camera perpendicular to a transparent neck reads the meniscus against a scale. Parallax is the main error source. The approach was judged suitable for legal metrology. — [Sensors 2021, PMC8069151](https://pmc.ncbi.nlm.nih.gov/articles/PMC8069151)
- Backlight shows fill level by outline. — [NI forum](https://forums.ni.com/t5/Machine-Vision/Transparent-bottle-detection/m-p/813954)
- Vendors claim deep-learning fill-level reading through foam and translucent containers, but give no independent data. — [Roboflow fill-level](https://roboflow.com/ai/fill-level-inspection); [iFactory beverage fill](https://ifactoryapp.com/ai-vision-camera/ai-vision-beverage-fill-level-monitoring) (marketing, unverified)
- General transparent-object segmentation is still hard: Trans10K (5000/1000/4428 train/val/test) authors report TransLab beats 20 segmentation methods, yet call the task "largely unsolved". — [Xie et al., Segmenting Transparent Objects in the Wild (arXiv 2003.13948)](https://arxiv.org/abs/2003.13948v3); [project page](https://xieenze.github.io/projects/TransLAB/TransLAB.html)

### Inferences
- In this fixed, controlled scene, a small classifier (3 classes: none / empty / filled) trained on per-slot crops is much easier than open-world transparent segmentation. But it will only be as good as the optical contrast between "none" and "empty". Fix the optics first (polarisation or pattern), then learn.
- Side-view meniscus methods do not apply to an overhead camera. Fill "level" from above would need either a second, angled camera or a depth cue. Presence of product is the realistic overhead question.

### Gaps
- No published overhead-camera study of fill detection in clear cups.
- No ClearGrasp or Trans10K metric values retrieved (mIoU or depth error).

## Q6. Reported accuracy, cost and failure modes (dust, condensation, starch powder, motion blur)

### Takeaway
I found no peer-reviewed accuracy figure for empty-clear-cup vs no-cup detection. Failure evidence is practical and mostly from vendors. Condensation and fouling on the lens or window cause false rejects and drift. Polarisation costs about 6–10x in light, which pushes exposure up and collides with motion blur. Mitigations are housings, air knives or curtains, cleaning routines, and tracking false-reject rate as a KPI.

### Cited Findings
- Moisture condensing on a sensor lens produced false "defect" readings on good parts. An air curtain across the lens fixed it. — [EXAIR blog](https://blog.exair.com/tag/vision-system/)
- Frozen-food packing case (vendor blog): drifting fog caused 9 false stops. Moving the camera out of the cold air spill, adding lens checks and fitting a housing brought this to 1. — [Fabrico](https://www.fabrico.io/blog/frozen-food-packing-hall-cameras/) (vendor, single anecdote)
- RH above 80% speeds lens condensation, and warm/chilled zone boundaries are a trigger. A rising false-reject rate is an early sign of lens fouling, LED degradation or algorithm drift. — [OxMaint FMCG vision maintenance](https://oxmaint.com/industries/fmcg/vision-inspection-system-maintenance-fmcg-cameras-lighting) (vendor)
- Polarised lighting costs about 10x intensity on reflective items (1 ms → 10 ms exposure) and 6–10x on diffuse items. — [VA Imaging](https://va-imaging.com/en-us/blogs/machine-vision-knowledge-center/polarization-in-your-machine-vision-application)
- Standard polarisers do not work outside the visible band, so broadband sources need UV/IR-block filters. — [EPIC Systems](https://epicsysinc.com/blog/eliminating-glare-with-polarization/)
- Transparent film edges vary between dark and very transparent, which destabilises edge detection. — [US5107131A](https://patents.google.com/patent/US5107131)

### Inferences
- **Motion blur budget:** blur in pixels = chain speed × exposure ÷ (mm per pixel). For example, 0.2 m/s with a 10 ms exposure gives 2 mm of smear. At 0.5 mm/px that is 4 px of blur. Fine BOS patterns and thin rims would wash out. Polarisation's light loss makes this worse, so use a bright strobed or pulsed LED rather than longer exposure. (Arithmetic only; the chain speed is unknown.)
- **Starch powder** on the steel scatters light and depolarises it, so it will look bright under crossed polarisers. It may look like a cup, or hide one. It also changes the background texture for pattern methods. A per-slot reference that updates only on known-empty frames would help, but powder build-up needs a cleaning rule.
- **Condensation on the cup** (cold product) scatters light. That actually helps make a cup visible, but it also makes empty and filled cups look alike.
- **Water films on the steel** change the mirror reflection and the BOS reference.
- Cost (inference, no sourced prices): polariser film plus an LED panel plus a printed or etched pattern adds little cost to a USB camera. A Polarsens or SWIR camera is a different class of hardware.

### Gaps
- No published accuracy, false-accept or false-reject rates for transparent-cup presence on conveyors.
- No sourced component prices (Polarsens camera, polariser film, SWIR camera, BOS pattern plates).
- No source on starch-powder effects on vision specifically.
- No source on food-contact safety of polariser films or printed pattern plates. Materials near open food need separate checking.
