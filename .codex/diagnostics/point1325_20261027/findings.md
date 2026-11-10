# Point (1325,291) masked-SIFT diagnosis — 2026-10-27

## Reproduction scope

- Actual input: test_video_Zebra/video_20261006_105420_207407_lossless.avi (not mp4).
- UI-left F37, UI-right F26. The UI candidate label F37 refers to the pair, not the right image frame index.
- Current on-disk MAIN configuration loaded into the isolated lite replay; runtime-only UI parameter changes are unavailable.
- Legacy block-adaptive spatial masks (new specular model default OFF); raw grayscale / CLAHE OFF; no local-plane refinement.
- Same selected RT and plane reproduce recorded OpenCV height 15.528544732684447 mm exactly.
- Python 3.11.9, OpenCV 4.13.0, NumPy 2.4.4.
- No production algorithm, calibration, source video, or original measurement log was modified. Pose analysis used a video copy inside this directory.

## Full pipeline ablation

| Descriptor / mask | Shadow | Active |
|---|---|---|
| OpenCV, RejectSpecPts OFF | Accepted, 15.5285 mm, right (1323.86475,252.31920) | Same |
| Custom, empty masks (diagnostic only) | Accepted, 15.5264 mm, right (1322.91089,252.11169) | Same |
| Custom, actual spatial masks | Rejected: insufficient valid coarse coverage | Rejected: insufficient profile coverage and other confidence gates |

Both actual-mask runs ended with a rejected best candidate at right (1314.35876,294.63327), corresponding to -8.5761 mm if triangulated for diagnosis. This is NOT an accepted measurement. No accepted abnormal custom height was reproduced with the current defaults. Original logs also contain a coverage rejection at this point, not a successful abnormal custom measurement.

## Normal candidate removed before ranking

At along=25 px (OpenCV reference), across=-2..2:

| Anchors | Valid point pairs | Required | Result |
|---|---:|---:|---|
| 28 (10px cells,3/cell) | 4..6 | 21 | All five candidates excluded |
| 64 (40px cells,7/cell) | 32..34 | 48 | All five candidates excluded |

At (along,across)=(-21,-2), away from the reference, 27/28 or 49/64 pairs pass. The search therefore ranks surviving regions, not the excluded reference region. A well-shaped minimum among survivors need not imply correct physical correspondence.

At reference offset (25,0), initial 28 anchors:
- Left descriptor point-valid: 28/28; mean weighted coverage ~1.0.
- Right descriptor point-valid: 19/28 (coverage >=0.6); median weighted coverage 0.6396.
- Common-coverage failures: 24/28; final valid pairs 4/28.
- Only 4..9 of 16 descriptor spatial subcells per point reach common>=0.8; median 5.5 subcells.
- Common fraction zeroes subcells below 0.8 before averaging all 16, then requires average >=0.5. Thus individual point validity does not imply pair validity.
- Left-only mask passes 28/28 at all five reference offsets; right-only mask reproduces the severe loss (4..6/28). Right mask is the dominant cause in this case.

## Mask source and growth

- Original right 30x30 neighborhood around (1324,252): mask ~12.89%; left around (1325,291): 0%.
- Right warped neighborhood at reference: mask ~24.22% after source-mask 1px dilation and projective warp (different coordinate neighborhoods, not an isolated dilation-only measurement).
- Descriptor gradients further erode the valid stencil by 1px; smoothed observation weight must also be >=0.5. Coverage is weighted over each rotated descriptor support, not simply the 30x30 mask ratio.
- In native 131x131 diagnostic crops, selected pixels BEFORE morphology: left63, right967. All selected pixels in both crops come only from the local-hot branch; none meet low-saturation, near-white or strong-highlight branches.
- Legacy local-hot branch has brightness/local-prominence conditions but no low-saturation/whiteness constraint. In these green textured images it can suppress ordinary locally bright texture. Images alone do not establish which pixels are genuinely specular.

## Implications / recommended next work (not implemented)

1. Diagnose/refine the legacy local-hot evidence, rather than loosening ratio or valley gates. Inspect raw-image mask at the correspondence first. A color test should be evaluated against the illumination; not every true highlight must be pure white.
2. Review abrupt 0.8 subcell rejection and subsequent 0.5 whole-point gate. Do not blindly relax until a result appears: masked histograms/angles/scales also change and the discarded physical evidence cannot be recovered by a threshold change.
3. Prefer invalid/insufficient evidence over accepting a distant clear minimum. Distinguish coverage failure from genuine repeated-texture ambiguity in logs/UI.
4. If the UI still returns a successful abnormal custom value, capture its full search/descriptor/config log and whether the new specular backend is ON; current default replay rejects that candidate.

## Artifacts

- diagnose.py: full-pipeline baseline/custom and shadow/active ablation.
- inspect_masks.py: controlled candidate offsets and one-sided mask ablation.
- mask_comparison.png: left and warped-right crops at the OpenCV reference, cyan overlay = excluded pixels.
- *_actual.json / *_empty.json: baseline summaries and stage funnels.
- mask_point_counts.json: per-offset point validity.
- *.pkl: replay session, candidate debug data, and captured descriptor packets for further inspection.
