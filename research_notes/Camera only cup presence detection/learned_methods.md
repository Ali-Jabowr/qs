# Learned methods for transparent-object and empty-vs-absent container detection

Setting: about 56 items/min, 30 fps overhead USB RGB camera, clear cups in a polished steel channel, food line, later inference on an Intel N100. Deterministic methods preferred; small classifiers acceptable.

## 1. Transparent-object segmentation and detection (Trans10K, Trans2Seg, ClearGrasp, TransCG, ClearPose and later work): reported metrics and transfer to industrial clear cups

### Takeaway
On the academic benchmarks, transparent-object segmentation reaches about 69–75% mIoU (Trans10K-v2). Depth-completion methods reach about 1.5–5 cm RMSE, but they need an RGB-D sensor. These are scene-level models for household or indoor glass. Nothing found shows they transfer to clear cups in a reflective steel channel, and general foundation models (SAM) do badly on glass. For a fixed single-camera line they are the wrong tool. A small classifier trained on the line's own images fits better.

### Cited Findings
- Trans2Seg (transformer) reaches 72.1 mIoU on Trans10K-v2, against 69.0 for the earlier TransLab. The authors trained 20 existing segmentation methods on the dataset and found that "simply applying previous methods to this task is far from sufficient". — [Trans2Seg, arXiv 2101.08461](https://arxiv.org/pdf/2101.08461v1)
- Trans2Seg cost is listed at about 49 GFLOPs for 72.15% mIoU. — [Trans2Seg listing](https://ar5iv.labs.arxiv.org/html/2101.08461)
- Trans10K-v2 has 11 fine-grained glass categories; v1 had only 2. — [Trans2Seg, arXiv 2101.08461](https://arxiv.org/pdf/2101.08461v1)
- Trans4Trans (efficient transformer, 2021) reports 75.14% mIoU on Trans10K-v2 for its medium variant. Trans2Seg-T scored 64.60 and Trans2Seg-S 68.47. The dataset has 10,428 images (5,000 train, 1,000 val, 4,428 test). — [Trans4Trans, arXiv 2108.09174](https://arxiv.org/pdf/2108.09174)
- Trans4Trans by size: (M) 75.14% at about 34.4 GFLOPs, (S) 74.15%, (T) 68.63% at about 10.45 GFLOPs. A separate listing marks these Trans10K numbers as not independently reproduced. — [hyper.ai listing](https://vercel.hyper.ai/en/papers/2107.03172); [sotaverified](https://sotaverified.org/papers/trans4trans-efficient-transformer-for)
- SAM (Segment Anything, 2023) often fails to detect mirrors and glass. It tends to segment what lies behind the glass instead of the glass itself. A summary page lists IoU of about 0.485 on GDD and 0.506 on GSD (secondary page, not verified against the paper). — [SAM Meets Glass, arXiv 2305.00278](https://arxiv.org/abs/2305.00278v1); [sotaverified](https://sotaverified.org/papers/segment-anything-model-sam-meets-glass-mirror)
- Depth completion on the TransCG test set (RGB-D input, RMSE in metres, computed on transparent-object masks only):
  - FDCT: 0.015
  - ClearGrasp baseline: 0.054 per FDCT, 0.052 per TransDiff (the two papers disagree)
  - TransDiff: 0.028
  — [FDCT, arXiv 2307.12274](https://arxiv.org/pdf/2307.12274); [TransDiff, arXiv 2503.12779](https://arxiv.org/pdf/2503.12779); [Transparent Object Depth Completion, arXiv 2405.15299](https://arxiv.org/pdf/2405.15299)
- ClearPose (ECCV 2022) is a real RGB-D dataset: over 350K labelled frames, 63 household objects, various lighting and occlusion, liquids included. Instance-annotation counts differ between versions (5M vs 4M). — [ClearPose, arXiv 2203.03890](https://arxiv.org/abs/2203.03890); [ECCV page](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5905_ECCV_2022_paper.php)
- In an industrial-adjacent case (YOLO on a delta-robot catering line, 2025), clear plastic was a named problem because "transparent-material objects pose a challenge for the model due to potential confusion with the background". The authors fixed it by adding training images that contain only transparent objects. They also labelled standing and lying cups separately, and found standing cups needed more varied-angle images. — [arXiv 2505.11879](https://arxiv.org/pdf/2505.11879)
- No public dataset of transparent cups on a conveyor was found. Roboflow Universe has generic cup sets (for example about 1,100 images in "cups detection"), but none for conveyors or clear cups. — [Roboflow cups detection](https://universe.roboflow.com/touati/cups-detection-cwaiu/dataset/1)

### Inferences
- Trans10K-class models are 10–50 GFLOP segmentation networks trained on cluttered scenes. On a fixed camera with a fixed ROI per slot they solve a harder problem than the line has. They are also too heavy to justify on an N100 at 30 fps when a 3-class crop classifier would do.
- The depth-completion line of work (ClearGrasp, TransCG, ClearPose, FDCT) assumes an RGB-D sensor. It does not apply to an RGB USB camera without new hardware.
- The main cue for a clear cup in a polished steel channel is probably its rim and edge highlights plus the change in reflection pattern, not the "glass texture" these benchmarks learn. That favours training on the line's own images, which is consistent with the delta-robot paper's fix.

### Gaps
- I found no 2023–2026 Trans10K state-of-the-art numbers newer than Trans4Trans (the search turned up nothing), and no ClearPose segmentation or pose scores (they were not in the snippets I got).
- I found no study that tests any transparent-segmentation model on reflective-metal backgrounds or on industrial food lines.

## 2. Small CNNs and tiny classifiers for slot state (full / empty container / no container) on conveyors: accuracy, labelled examples needed, few-shot approaches

### Takeaway
I found no peer-reviewed paper on exactly this 3-state problem for clear cups. Nearby work (fill level, empty bottle, blister empty pocket, empty shelf slot) reports 90–100% accuracy, but mostly in the lab, from vendors, or from student projects, and rarely says how many examples were used. Transfer learning from ImageNet is the consistent advice for small industrial datasets. On an N100, a YOLO-nano detector runs at about 18–19 FPS (INT8, OpenVINO), so a smaller crop classifier should fit within the 30 fps budget. That last point is an inference from the YOLO figure, not a measurement.

### Cited Findings
- A 2026 paper compares Mask R-CNN with YOLOv8x-Seg for empty-bottle and liquid-level inspection on a conveyor. The extracted text contradicts itself: one passage says Mask R-CNN was 100% on liquid level but unreliable on empty bottles; another says it handled all checks at 0.16–0.24 s per bottle. Treat as unverified. — [iasj 2026 paper](https://iasj.rdd.edu.iq/journals/uploads/2026/05/07/a168721472e6d4bff01fe6f833600bb0.pdf)
- A bottling prototype reports 97% level-detection accuracy using Canny edges on a Raspberry Pi. This is a deterministic, non-CNN baseline. — [IJEPE](https://ticket17180.pkpmisc01.publicknowledgeproject.org/index.php/ijepe/article/view/14361)
- A lab-scale beverage-line inspection paper classifies empty, underfilled and overfilled bottles; no accuracy figure in the extracted text. — [IIETA](https://iieta.org/download/file/fid/54261)
- A patent for lab-tube drawer trays classifies each slot crop as empty or occupied with a random forest trained on empty and non-empty patches. This is the per-slot crop-classifier pattern. — [USPTO 10140705](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10140705)
- Cognex markets a deep-learning blister tool, trained on "a small number of good and defective images", that flags missing tablets. This is a vendor claim. — [Cognex](https://www.cognex.com/industries/pharmaceuticals-medical/tablet-capsule-manufacturing/blister-pack-quality-check)
- An Intel developer showcase finds empty and misplaced shelf slots with Fast R-CNN and claims close to 100% accuracy. Not peer-reviewed. — [Intel DevMesh "The MisFit"](https://devmesh.intel.com/projects/the-misfit)
- Small-data transfer-learning results:
  - VGG16 fine-tuned on a small, unbalanced 15-class X-ray weld set: about 90% average accuracy
  - Binary classification of molded-part defects with a very small dataset: 97.7%, with a clear gain from generic-dataset pretraining
  - A few-shot surface-defect study (INDIN 2023) found ImageNet pretraining best overall, and DAGM pretraining helped when close to the target domain
  — [Springer s12008-023-01327-3](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s12008-023-01327-3.pdf); [DFKI publication 14657](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/14657)
- Vendor pages claim 99%+ fill-inspection accuracy with no independent verification. — [Roboflow fill level](https://roboflow.com/ai/fill-level-inspection); [Oxmaint](https://oxmaint.com/sap-integration/ai-defect-detection-bottling-line)
- Compute:
  - YOLOv10n with INT8 on an Intel N100 (LattePanda Mu, OpenVINO) runs at about 18–19 FPS. This is a vendor-style blog with thin methodology.
  - On Intel CPUs, OpenVINO was the fastest runtime and cut YOLOv8 time by 30–60% against the alternatives.
  — [LattePanda N100 blog](https://lattepanda.com/blog-323181.html); [arXiv 2607.11356](https://arxiv.org/pdf/2607.11356)

### Inferences
- At 56 items/min, roughly 32 frames arrive per item. With a fixed slot ROI, a crop classifier (MobileNet- or ResNet18-class, or a pretrained backbone plus logistic regression) only needs to run a few times per slot, so the N100 budget is not the bottleneck. Measure it before relying on this.
- "Empty cup" vs "no cup" is a fine-grained difference in the image: a rim and wall highlight present or absent. Fill level is a bigger visual difference. Labelled empties and no-cup examples, staged on purpose if needed, matter more than model choice.
- No source gives an images-per-class rule. Treat the label budget as an empirical learning curve on the line's own data, not a number taken from the literature.

### Gaps
- No peer-reviewed accuracy figure was found for clear-cup presence or empty classification on a conveyor.
- No source states how many images per class were used.
- No measured N100 latency was found for a small classifier, only for YOLOv10n.

## 3. Anomaly detection (PatchCore, EfficientAD, etc.) for missing or empty items: does it separate "empty container" from "no container", or only flag "unusual"?

### Takeaway
Unsupervised anomaly detection (AD) gives one score: normal vs not normal. It does not label why. If the normal class is "full cup", then both "empty cup" and "no cup" score as anomalies, and the model has no way to tell them apart without a second step (a classifier, or a rule on the anomaly map). A missing object is a "logical anomaly", which is AD's weaker category. Even EfficientAD scores only about 86 AUROC on logical anomalies (MVTec LOCO), far below what a gate catching 1–3% empties needs.

### Cited Findings
- MVTec LOCO AUROC, logical / structural:
  - EfficientAD-S: 85.8 / 94.1
  - EfficientAD-M: 86.8 / 94.7
  - PatchCore: 75.8 / 84.8
  — [EfficientAD, arXiv 2303.14535v3](https://arxiv.org/html/2303.14535v3)
- Mean image AUROC on MVTec AD / VisA / LOCO:
  - EfficientAD-S: 98.8 / 97.5 / 90.0
  - PatchCore: 98.7 / 94.3 / 80.3
  — [EfficientAD, arXiv 2303.14535v3](https://arxiv.org/html/2303.14535v3)
- Latency on an NVIDIA RTX A6000 (batch size 1):
  - EfficientAD-S: 2.2 ms (614 img/s)
  - EfficientAD-M: 4.5 ms
  - PatchCore: 32 ms
  No CPU latency is reported. — [EfficientAD, arXiv 2303.14535v3](https://arxiv.org/html/2303.14535v3)
- The EfficientAD authors say "fine-grained logical anomalies, however, remain a challenge". The method needs training (about 20 minutes in their setup) for its autoencoder to learn the logical constraints. — [EfficientAD, arXiv 2303.14535v3](https://arxiv.org/html/2303.14535v3)
- A benchmark notes PatchCore is better on MVTec AD than on LOCO because it specialises in structural anomalies. It defines logical anomalies as displacement or missing parts. — [IM-IAD, arXiv 2301.13359](https://arxiv.org/pdf/2301.13359)
- Follow-up work reports LA-EAD at 94.2 AUROC on MVTec LOCO by improving EfficientAD's logical branch. — [arXiv 2407.17909](https://arxiv.org/pdf/2407.17909)
- MVTec AD 2 (2025) adds transparent and overlapping objects, back-light and dark-field lighting, and a mixed-lighting test split. State-of-the-art methods stay below 60% average AU-PRO, and lighting changes hurt performance significantly. The lighting claim comes from an AI-generated review; the AU-PRO figure is from the abstract. — [MVTec AD 2, arXiv 2503.21622](https://arxiv.org/abs/2503.21622v1); [Moonlight review](https://www.themoonlight.io/review/the-mvtec-ad-2-dataset-advanced-scenarios-for-unsupervised-anomaly-detection)
- Anomalib documents CPU install and OpenVINO export for PatchCore but publishes no timing numbers. — [Anomalib docs](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)

### Inferences
- For this line, AD fits as a cheap second opinion ("something weird in this slot": a foreign object, a crushed cup, a hand in view), not as the empty / no-cup classifier.
- To get three states from AD, you would need either (a) a separate AD model per normal state, which makes empties "normal" in one model and so needs empty examples anyway, or (b) a supervised head on top. At that point a direct 3-class classifier is simpler.
- Transparent objects plus reflective backgrounds plus lighting shift is exactly where MVTec AD 2 shows current AD is weak, so out-of-the-box AUROC would likely be low.

### Gaps
- No per-defect-type ("missing object only") numbers on LOCO were found.
- No CPU or N100 latency for EfficientAD or PatchCore was found; a forum claim of under 10 ms on an RTX 3060 is unverified and excluded.

## 4. Temporal or video models vs per-frame classification for moving items with motion blur

### Takeaway
The evidence is thin, but it points one way: aggregate predictions per tracked item (voting, best weighted by frame sharpness) rather than run a video network. Quality-weighted temporal ensembling beat single-frame CNNs by about 18–22% F1 when blurry frames were present, but beat plain averaging by only about 2%. So most of the gain comes from using several frames at all.

### Cited Findings
- On packaging defect inspection, a temporal-quality ensemble that weights frames by timing and sharpness gives about 17.6–22.4% higher F1 than single-CNN models when at least one low-quality image is present, and about 1.9–2.1% higher than simple average voting. — [Son et al., Sensors 2024, 24(14):4438](https://mdpi.com/1424-8220/24/14/4438)
- Apple inspection on a dense conveyor: frame-to-frame prediction swings are blamed partly on motion blur. The fix is detection plus tracking, with majority voting over each track. — [arXiv 2602.19278](https://arxiv.org/pdf/2602.19278)
- BeltCrack: high belt speed makes stable temporal cues hard to extract, so the authors add a module that models motion between frames. — [arXiv 2506.17892](https://arxiv.org/pdf/2506.17892)
- Corn-seed sorting at 0.30–0.40 m/s: Total Variation deblurring plus YOLOv8 was more stable than YOLOv11 without deblurring. — [UNHAS record](https://discover.unhas.ac.id/publications/pub_scopus_105034910797)

### Inferences
- At 30 fps and 56 items/min, each slot is visible across many frames. A per-frame crop classifier plus a k-of-n vote, or a sharpness-weighted vote, inside the slot's time window is deterministic, cheap and auditable. It also fits the "deterministic preferred" constraint better than a 3D-CNN or video transformer.
- Short exposure and lighting (hardware) likely matter more than temporal models for blur. This is an inference; no source above tests exposure directly.

### Gaps
- No paper was found that compares per-frame and temporal models on the same backbone at controlled blur levels or conveyor speeds.

## 5. Failure modes: domain shift from lighting, glare and dust; class imbalance (empties are rare, about 1–3%)

### Takeaway
Lighting shift is the documented failure mode most likely to hurt this setup: MVTec AD 2 was built to test it, and methods drop significantly under it. Transparent objects confuse detectors against similar backgrounds, and SAM-class models segment through the glass. I found no source that quantifies class imbalance for empty containers. Rare empties mean the error rate on empties must be measured on a staged, enriched test set, because natural traffic gives too few positives.

### Cited Findings
- MVTec AD 2 includes test splits with lighting changes and a private mixed-lighting set to measure robustness to distribution shift. Methods stay below 60% AU-PRO. — [arXiv 2503.21622](https://arxiv.org/abs/2503.21622v1)
- Transparent objects can blend into the background and skew detector output. Adding transparent-only training images helped. — [arXiv 2505.11879](https://arxiv.org/pdf/2505.11879)
- SAM segments the objects behind transparent surfaces rather than the glass itself. — [arXiv 2305.00278](https://arxiv.org/abs/2305.00278v1)
- The weld study coped with small, unbalanced classes through augmentation and transfer learning, reaching about 90%. — [Springer s12008-023-01327-3](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s12008-023-01327-3.pdf)
- Motion blur causes frame-to-frame label flips on conveyors. — [arXiv 2602.19278](https://arxiv.org/pdf/2602.19278)

### Inferences
- With 1–3% empties, a classifier can be 97–99% "accurate" by never predicting empty. Report recall on empties and no-cup, plus the false-reject rate, not overall accuracy.
- Collect empties and no-cup slots on purpose (staged runs), across shifts and lighting states, including glare angles and wet or dusty channels. Hold out whole sessions or days for testing, not random frames; random frames leak near-duplicates from the same pass.
- Polished steel creates specular hot spots that move with the cup's position. Expect the learned model to latch onto reflection patterns. Fixed diffuse lighting and a fixed ROI reduce that risk.

### Gaps
- No source quantifies how glare or dust affects cup and container classifiers specifically.
- No study was found on class-imbalance handling (reweighting, focal loss, oversampling) for rare empty containers on food lines.
- No case study was found on drift monitoring or retraining frequency for this kind of line.
