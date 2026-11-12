# Camera-only (sensorless) triggering and conveyor/chain tracking

Setting assumed throughout: ~56 carriers/min, chain stopped >50% of the time, fixed overhead 30 fps USB camera, empty slots occur, deterministic methods preferred.

## 1. Software triggers / "virtual sensors" in commercial smart cameras

### Takeaway
Vendors offer three sensorless modes: (a) a timed internal trigger that captures every N ms and knows nothing about parts, (b) free-run capture, and (c) a real image-based trigger, where cheap vision tools run on a continuous stream and fire the inspection when a part appears (Cognex Checker "internal trigger", the Cognex configurable image trigger patent, Keyence IV4 "AI Trigger"). Vendor docs consistently present hardware triggers as the precise or deterministic option. Software and free-run modes are described as timing-variable.

### Cited Findings
- Cognex In-Sight EasyBuilder has a "Continuous" trigger setting that takes "free running" acquisitions as fast as possible, spaced by a configurable Trigger Interval in ms. The docs give the default as 500 ms in one version and 0 ms in another. — [Cognex Image Settings](https://docs.cognex.com/is_573/web/EN/ezb/Content/EasyBuilder/EasyBuilderLite/ImageSettings.htm); [Cognex Trigger Controls](https://docs.cognex.com/is_573/web/EN/ezb/Content/EasyBuilder/Image_TriggerSettings.htm)
- Cognex documentation reports that Continuous Trigger on In-Sight 2000 sensors can load the CPU enough to break communication with In-Sight Explorer. — [Cognex Connect doc](https://docs.cognex.com/is_573/web/EN/ezb/Content/EasyBuilder/Connect_InSight.htm)
- Cognex says faster trigger rates come from the dedicated acquisition trigger input ("Trigger = Camera") than from a general-purpose input line. — [Cognex Inputs: Triggering Image Acquisition](https://docs.cognex.com/is_573/web/EN/ezb/Content/EasyBuilder/Inputs_ImageAcquisition.htm)
- Cognex patent US9651499B2, "Configurable image trigger" (filed 2011-12-20, granted 2017-05-16):
  - The camera acquires continuously, and user-built "trigger logic" made of vision tools (presence/absence, edge, pattern match, histogram threshold) evaluates every frame. When the logic passes, the frame, or a later one, goes to the main inspection.
  - The trigger can require several consecutive passing images before it fires. It can also detect on a fast, low-resolution stream and then capture a high-resolution inspection image.
  - The example loop runs at about 200 acquisitions/s.
  - The patent's rationale: photoeyes add hardware, have "limited ability to modify the conditions for triggering", and can be hard to mount near the field of view.
  - The patent criticises earlier image-based triggers for "inaccurate acquisition counts", output-gating problems, and loss of synchronisation. — [Google Patents US9651499](https://patents.google.com/patent/US9651499)
- Cognex Checker sensors:
  - Checker 101 (2004) was marketed as a photodetector replacement that "automatically detect[s] when a product is ready to be inspected", with no external trigger.
  - Checker 232 (2008) used "multi-image analysis to determine if a part is present without an external trigger", which also lets it track parts in varying positions.
  - Checker 4G (2014) "internal trigger" "detects parts by locating a feature on the part" and "does not require additional sensors". — [Control Global 2004](https://www.controlglobal.com/industrynews/2004/6); [Control Design 2008](https://www.controldesign.com/vendors/products/2008/060/); [Cognex press release 2014](https://www.cognex.com/company/press-releases/2014/cognex-introduces-easy-to-use-vision-sensor-for-color-applications)
- Keyence IV series:
  - "Internal" trigger means imaging and judgment run "at the timing of set trigger interval". This is timed capture, not part detection. "External" means a photoelectric sensor or PLC. — [Keyence IV manual p.233](https://www.manualsdir.com/manuals/658109/keyence-iv-series.html?page=233)
  - The Keyence IV4 page advertises an "AI Trigger": "Parts passing the sensor are automatically detected", with no fixture, positioning tool or external trigger sensor needed. The page lists no speed limits or caveats. — [Keyence IV4 page](https://www.keyence.com/landing/lpc/iv4_iast.jsp)
- A PLC forum reply on dropping the sensor-triggered Keyence setup suggests running continuous mode after a part is found, and warns that the part will still be in the field of view, so it can be counted more than once. — [MrPLC forum](https://mrplc.com/forums/topic/28862-problem-with-trigger-camera-keynce-with-s7-1200/)
- Basler: a software trigger is a host command sent through pylon. Basler recommends a hardware trigger when a precise trigger or an existing hardware signal is available. — [Basler pypylon triggered acquisition](https://docs.baslerweb.com/pypylon-triggered-image-acquisition); [Basler Triggered Image Acquisition](https://docs.baslerweb.com/triggered-image-acquisition)
- A GenICam guide says software triggers are "not time accurate" because of network latency and command jitter. — [Common Vision Blox GenICam guide](https://help.commonvisionblox.com/GenICam-User-Guide/html_english_genicam_triggerconfigs.htm)
- The Imaging Source puts software-trigger latency in the variable millisecond range and hardware-trigger latency in microseconds. It says free-run timing may not be enough for position-dependent conveyor inspection. — [The Imaging Source glossary](https://www.theimagingsource.com/en-us/resource/glossary/hardware-triggering-and-strobe-synchronization/)
- Matrox Design Assistant has three software trigger types: an operator-view button, a Trigger step, and a PLC trigger over Quick Comm (PROFINET/EtherNet/IP). Without Quick Comm, the app polls a PLC signal. — [Matrox triggering doc](https://matrox.com/apps/imaging_documentation_files/DAXHelpFiles/extfile/z_DA_UG_Part_Image_steps_and_repository/UserGuide/Acquisition/Triggering.htm)

### Inferences
- The production-proven sensorless pattern is the Cognex image-trigger loop: a cheap per-frame detector on a fixed ROI, an N-consecutive-frames debounce, then a separate inspection step. It can run on a 30 fps USB camera in software. Our latency does not matter much because we inspect a stationary or slow carrier, not a moving part that needs a strobe.
- The vendor warnings about software-trigger jitter are about timing an exposure on a moving part. At ~56 carriers/min, with the chain stopped more than half the time, the main risk is count and sync errors (double counts, missed parts), which the Cognex patent names explicitly. Timing jitter matters less.

### Gaps
- I found no Hikrobot MVS, SICK (Inspector/InspectorP) or Teledyne DALSA BOA page describing image-based or part-detecting triggers. Their docs I reached only cover hardware, software or free-run modes.
- No vendor publishes reliability figures for image-based triggers (missed or false trigger rates), or a rule for when to switch to a photoeye. The only vendor-side reliability statements are the Cognex patent's qualitative criticism of earlier image triggers.

## 2. Conveyor speed / displacement estimation from video ("virtual encoder")

### Takeaway
Belt speed from video is a well-studied research topic (optical flow, feature matching, texture periodicity). Reported relative errors run from under 2% to about 5.5%. All of this work targets continuous belt speed, not slot counting. Industrial robot conveyor tracking still pairs cameras with a physical encoder, and "virtual encoder" exists mainly in patents.

### Cited Findings
- Scientific Reports 2025, "Conveyor belt speed detection via the synergistic fusion of optical flow and feature matching":
  - Combines an improved RAFT optical-flow network with Harris-BRIEF-RANSAC feature matching, fused by Bayesian decision.
  - Notes that low-texture belt surfaces leave too few salient features for classic feature matching.
  - Reported relative errors are about 5.25% (RAFT-SEnet) and 5.5% (Harris-BRIEF-RANSAC). Which figure belongs to the fused system was not clear from the search summary. — [Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-30488-2); [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12783289/)
- A point-cloud registration method is cited as reaching under 1.8% average relative error at under 35 ms processing. I saw this only in a search summary, not the primary paper. — [search summary of Sci Rep 2025 related work](https://www.nature.com/articles/s41598-025-30488-2)
- Gao et al., IEEE Access 2019: a contactless belt speed system based on keypoint optical flow, computing running speed in real time. The authors argue that contact (wheel or encoder) measurement suffers from slip and wear. I could not open the full text (HTTP 403). — [ResearchGate](https://www.researchgate.net/publication/335424363_A_Contactless_Measuring_Speed_System_of_Belt_Conveyor_Based_on_Optical_Flow_Techniques); [TU Delft record (ML variant)](https://repository.tudelft.nl/record/uuid:b6e0b30d-bd26-4db3-bbb4-72be31e5f72b)
- MDPI Sensors 2020 compared belt linear-speed measurement methods, including a high-speed camera, by MAE and RMSE. — [MDPI Sensors 20(20):5816](https://mdpi.com/1424-8220/20/20/5816)
- Teledyne patent US11262316B2 (filed 2017, granted 2022-03-01), "Method of synchronizing a line scan camera":
  - Reduces each line of an ROI on a periodic belt texture to a mean.
  - Computes autocorrelation by FFT ("multiplying an FFT by the complex conjugate ... then applying the inverse FFT").
  - Takes the period as the mean distance between adjacent autocorrelation peaks, using about 2,048 lines as an example.
  - Updates line rate = current × (reference period / measured period).
  - Gives no numeric accuracy, only "acceptable tolerances", and mentions "mechanical flutter" in the test drum. — [Google Patents US11262316](https://patents.google.com/patent/US11262316B2/en)
- OpenMV documents per-frame phase-correlation displacement (`find_displacement()`, which returns a confidence "response") for registering frames from a camera moving along a conveyor. No accuracy figures are given. — [OpenMV displacement matching](https://docs.openmv.io/openmvcam/tutorial/image/matching/displacement-and-keypoints.html)
- Phase correlation gives integer peaks. Sub-pixel accuracy needs peak fitting (sinc, Gaussian or paraboloid). Reported accuracy reaches about 0.01 px in video stabilisation. — [Chinese Optics](https://chineseoptics.net.cn/en/article/id/8614); [Argyriou, Kingston](https://eprints.kingston.ac.uk/42634/1/Argyriou-V-42634-AAM.pdf)
- Patent: a "Virtual Encoder", described as non-contact (virtual) position encoding that estimates object pose over time in place of a mechanical encoder. — [ipiq US patent page](https://ipiq.io/patents/1452145)
- Robot conveyor tracking in practice latches a physical encoder at camera capture time. Epson requires matching camera search timing with encoder latch, and Mitsubishi MELFA reads the encoder value when the vision "imaging completion" signal arrives. — [Epson Users Guide](https://download.epson.biz/robots/ww/data/html/UsersGuide/en/170/170-010/170-010-030.html); [Mitsubishi tech news](https://www.mitsubishielectric.com/fa/document/technews/robot/bfp-a6079_0407e/bfp-a6079_0407e.pdf)
- 2024 Bologna MSc thesis: dense optical flow from event cameras on parcel sorting conveyors (Steepest Descent and APG with vectorial TV). The abstract gives no numbers. — [UniBo thesis](https://amslaurea.unibo.it/id/eprint/33355/)
- A USPTO grant, "System and/or method for conveyor motion estimation" (US 12579667), estimates conveyor speed and direction from image displacement. I could not read the text because the PDF is a scanned image. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12579667)

### Inferences
- For a chain whose carriers are the moving objects, displacement between frames is easier than on bare belts. Carriers are high-contrast and periodic, which suits 1-D phase correlation or normalized cross-correlation along the chain axis inside a strip ROI.
- Integrating a per-frame displacement gives a virtual encoder. Its drift is unbounded unless it is re-anchored to carrier positions every pitch. Snapping to pitch (see section 3) bounds the error to under one pitch per anchor.
- A 5% speed error, the order reported, is irrelevant if we count pitches by detecting carrier edges or phase wrap rather than by integrating speed.

### Gaps
- I found no 2022–2026 paper that tracks a discrete slot or carrier count on a chain conveyor from video alone with reported count accuracy. The literature is about continuous belt speed.
- There are no published drift figures for video-integrated displacement over long runs.

## 3. Periodic pitch tracking, phase-locking, stops, occlusion, blur, empty slots

### Takeaway
The most concrete source on locking to a fixed pitch is the Teledyne line-rate patent: find the period by FFT autocorrelation, then servo to a reference period. It shows that period estimation from a 1-D strip signal is industrial practice. For counting, the documented practitioner pattern is detector + tracker + virtual counting line, with a debounce. One vendor case reports correct single counts through brief stops and reversals. I found no source dealing specifically with hand occlusion or empty-slot detection on a chain.

### Cited Findings
- Teledyne US11262316: period comes from autocorrelation peak spacing. The reference period comes either from isotropic texture or from training on objects of known aspect ratio. The rate is corrected multiplicatively, with min and max clamps. — [US11262316B2](https://patents.google.com/patent/US11262316B2/en)
- Chain-wear patent US7540374 measures chain pitch from the spacing of adjacent chain elements in images, using a proximity sensor alongside the camera. — [USPTO US7540374](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7540374)
- Patents on visualising periodic motion use Fourier-domain analysis and search for the best periodic rate and phase by comparing frame offsets. — [US12300273](https://patents.google.com/patent/US12300273); [US11631432](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11631432)
- Roboflow: adding ByteTrack gives consistent IDs so an object isn't counted twice. Place the counting line where each object crosses once and away from where items pile up or stop. — [Roboflow conveyor counting](https://blog.roboflow.com/counting-objects-conveyor-belt/); [Roboflow running total](https://roboflow.com/answers/running-total-for-video-or-shift)
- oee.ai chocolate-bar demo: one overhead camera, a detector, a tracker, a counting zone and a counting line. Each bar is counted "as soon as its centre point crosses the line", "even when the belt stops or reverses briefly". No accuracy figures are given. — [oee.ai video page](https://www.oee.ai/en/video/cookie-counting-computer-vision)
- Cognex image-trigger patent: requiring several consecutive passing frames before firing is an explicit debounce mechanism. — [US9651499](https://patents.google.com/patent/US9651499)

### Inferences (design suggestions for our setting, not sourced practice)
- Frame budget:
  - At 56 carriers/min, the average is 1.07 s per pitch.
  - If the chain moves less than 50% of the time, it must travel at least about 1.9 pitch/s while moving.
  - At 30 fps that is at most about 16 frames per pitch of motion, which is plenty for edge or phase tracking. Blur scales with speed × exposure, so short exposure plus strong light matters more than frame rate.
- Stop-and-go chain: the most deterministic trigger is probably state-based, not position-based:
  - Measure motion energy (frame difference or 1-D phase-correlation shift) in a strip over the chain.
  - Define "stopped" as shift ≈ 0 for k frames.
  - On each moving→stopped transition, check that the accumulated displacement since the last stop is about one pitch (or an integer multiple), then inspect the slot under the fixed ROI.
  - This uses the chain's own dwell as the trigger, with the camera acting as an encoder.
- Empty slots break presence-based triggers, because there is nothing to detect. Track the carrier or chain feature (slot rim, link, pin), not the product, so the count advances whether or not a cup is present. Then decide occupancy separately inside the slot ROI, e.g. with an empty-slot reference template or a median-intensity rule (see section 4).
- Phase lock: keep a predicted phase φ = (accumulated displacement mod pitch). When a carrier edge is observed, correct φ toward the measurement (alpha-beta or PLL-style). Reject corrections larger than about pitch/4 as outliers, for example a hand.
- Occlusion (hands): use a second "chain-only" ROI strip away from where operators reach, if geometry allows. When foreground in the strip exceeds a threshold, freeze tracking (hold the state, don't count), then re-acquire phase from the next clean carrier edges. The pitch prior bounds the ambiguity to an integer number of pitches, so log any re-sync that needs a jump.

### Gaps
- I found no paper or app note on PLL or phase-lock tracking of carriers on an indexing chain with a camera, and none on hand-occlusion handling in sensorless counting.
- No quantitative failure rates exist for the tracker + line-crossing approach under frequent stops; the oee.ai claim is qualitative.

## 4. Line-scan / strip ("virtual light barrier") tricks with an area camera

### Takeaway
A "virtual barrier" is a small ROI on the area-camera image whose statistic (median L in Lab, mean intensity) is thresholded with a moving average and rising-edge counting. A 2022 paper demonstrates it on a chain conveyor with a 640×480 USB webcam. The Teledyne patent shows the related line-statistic trick, reducing each image line to its mean to form a 1-D time signal.

### Cited Findings
- Castillo Alarcón et al., 2022, "Uso de visión artificial con OpenCV como sensor de barrera en una faja transportadora", Rev. Inv. Cient. Ingenierías (UNAJ), DOI 10.47190/nric.v4i1.3:
  - Setup: a chain-type conveyor moving LPG gas cylinders through automated filling, a 640×480 USB camera, and an ROI placed mid-chain as the barrier.
  - Signal: median of the L channel (Lab colour space) in the ROI, about 5 when empty and jumping to about 146–255 when a product crosses.
  - Logic: a moving average over samples suppresses spurious peaks and noise. Only rising edges increment the counter.
  - Output: a serial signal to an Arduino.
  - The GUI exposes ROI size and position, thresholds, and frame-skip to reduce load.
  - The authors claim high selectivity (μ = 34.69, σ = 60.27 over samples) but give no miss or false-count rate. — [UNAJ PDF](https://unaj.edu.pe/revista/index.php/vpin/article/download/223/126/668)
- Teledyne: an ROI near the edge of the field of view, with each line reduced to a statistic (mean), forms a 1-D signal for period estimation. — [US11262316B2](https://patents.google.com/patent/US11262316B2/en)
- OpenMV registers conveyor frames by phase correlation, a strip-stitching application. — [OpenMV docs](https://docs.openmv.io/openmvcam/tutorial/image/matching/displacement-and-keypoints.html)

### Inferences
- Build a pseudo line-scan by taking a 1-px or few-px line across the chain, perpendicular to travel, every frame and stacking it over time into a space-time image (a "kymograph"). Carriers appear as stripes. Stops appear as vertical runs of identical rows, and moving periods as slanted or alternating patterns. This makes stop, move and pitch visible with simple 1-D processing.
- A line along the travel direction gives a 1-D profile per frame. Its shift between frames (1-D cross-correlation) is the displacement, and its autocorrelation gives the pitch in pixels, which can be calibrated once.
- The Castillo virtual barrier counts products, not slots. With empty slots it would undercount, so the barrier ROI should be placed on a carrier feature that is always present.

### Gaps
- I found no published accuracy for area-camera virtual light barriers beyond that one small paper.

## 5. Known failure modes and practitioner reports

### Takeaway
Documented failure modes are double counting while a part sits in the field of view under continuous triggering, inaccurate acquisition counts and sync loss in image-based triggers, CPU overload in continuous mode on small smart cameras, low-texture surfaces defeating feature matching, and items stopping or piling at the counting line. I found no Reddit or r/PLC thread through search.

### Cited Findings
- In continuous mode, a part remains in the field of view after detection, which risks repeat triggers. — [MrPLC forum](https://mrplc.com/forums/topic/28862-problem-with-trigger-camera-keynce-with-s7-1200/)
- Earlier image-based triggers suffered from "inaccurate acquisition counts", gating problems and "loss of synchronization". — [Cognex US9651499](https://patents.google.com/patent/US9651499)
- Continuous trigger can overload the In-Sight 2000 CPU and drop communication. — [Cognex doc](https://docs.cognex.com/is_573/web/EN/ezb/Content/EasyBuilder/Connect_InSight.htm)
- Low-texture belts give insufficient features for classic matching. — [Sci Rep 2025](https://www.nature.com/articles/s41598-025-30488-2)
- Keep the counting line away from where items stop or pile up, to avoid double counts. — [Roboflow](https://roboflow.com/answers/running-total-for-video-or-shift)
- Mechanical flutter in conveyor speed is noted even in the Teledyne lab test. — [US11262316B2](https://patents.google.com/patent/US11262316B2/en)
- Software-trigger timing is not time-accurate (latency and jitter). — [CVB GenICam guide](https://help.commonvisionblox.com/GenICam-User-Guide/html_english_genicam_triggerconfigs.htm)

### Inferences
- For our line, the Roboflow advice ("away from where items stop") conflicts with the physics: the inspection point is where the chain stops. This is the strongest argument for counting by chain displacement and stop events, not by object line-crossing.
- Likely field failures to test for:
  - a hand or tool in the strip read as a carrier
  - a slow creep start read as a stop
  - lighting flicker at 50 Hz aliasing against 30 fps and causing false motion energy, which argues for a fixed exposure that is a multiple of 10 ms
  - vibration on stop causing ±px jitter
  - an empty slot plus a missed edge giving a silent off-by-one
  The pitch prior plus a periodic absolute re-anchor (a distinctive marked carrier, if one exists) bounds the last one.

### Gaps
- Searches turned up no r/computervision, r/PLC or Vision Systems Design threads on sensorless indexing-chain counting. Practitioner evidence is limited to one MrPLC thread and vendor tutorials.
- I found no thesis or app note that quantifies when to give up and fit a photoeye or inductive sensor on the chain. Vendors only say a hardware trigger is preferred for precision.
