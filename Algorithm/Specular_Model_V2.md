# Specular model V2

Independent detector: `specular_model_v2.py`. Independent editor:
`specular_model_v2_settings.py`. Main app globals: `SPECULAR_MODEL_V2_CONFIG`
and `SPECULAR_MODEL_V2_ENABLED_DEFAULT`. Every parameter is prefixed `v2_`.
Legacy `SPATIAL_BLOCK_CONFIG` is unchanged and not read by V2.

## UI

In the former measurement-mode area, **新版反光: On/Off** switches backends.
**新版反光參數** edits V2 only and enables V2 on apply. **舊版反光參數**
edits the old detector and selects it on apply. Changes are session-only.
Both current masks are computed before they are committed; errors restore the
old config and uncertainty overlays. Extra-frame masks are invalidated.
The controls are locked during block statistics and drag profiles.

Show Spatial: blue = accepted spatial exclusion mask; orange = uncertain,
including clipped/poorly identifiable pixels. Orange is display-only. Existing
Show Temporal red/purple overlays remain independent. Custom masked SIFT uses
only the spatial exclusion mask; other existing paths may use spatial OR temporal.
Historical measurement results/diagnostics are not recomputed by switching;
click/replay the coordinate again. Logs show backend, mask/uncertain percentage,
and detector duration.

## Model

1. Convert raw BGR to approximate linear RGB using `v2_input_gamma` (default 2.2).
   This is a camera-response approximation, not radiometric calibration.
2. Normalize RGB vectors and subtract the configured illuminant direction to
   obtain an illuminant-orthogonal hue direction.
3. Sample 8 directions at each `v2_reference_radii_px`. Weight neighbours by hue
   affinity and chroma to favour compatible, less white-contaminated tissue.
   Clipped/dark/out-of-image references are excluded. References are streamed,
   not kept as a huge N*H*W stack. No full-image resize is used.
4. Estimate diffuse colour `d` from the weighted mean. Effective sample count,
   hue affinity and colour dispersion describe reference reliability.
5. Compare model A `I=a*d` and model B `I=a*d+s*e`, with nonnegative a/s.
   Solve the two-column least-squares interior and both nonnegative boundary
   cases exactly. Normalize error improvement by absolute+relative noise and
   subtract `v2_model_penalty` for the additional degree of freedom.
6. Evidence combines penalized improvement, specular fraction, fit residual,
   reference reliability and separation of diffuse/illuminant directions.
   This score is NOT a calibrated probability. High-score seeds may grow only
   into adjacent pixels passing `v2_grow_score`, for a bounded number of steps.
   There is no unconditional white exception, percentile quota or closing.
7. Clipped, unsupported or nearly collinear cases are uncertain. Only accepted
   mask pixels are excluded from the existing binary masked-SIFT path.

## Scope and limitations

The local reference estimate is an assumption, not a recovered ground truth.
Neutral tissue and white reflections can be indistinguishable in a single RGB
image. Large highlights can exhaust the reference neighbourhood. Mixed material
boundaries, colour processing and unknown illuminant colour remain failure cases.
No actual reflectance probabilities, exact physical decomposition, temporal
verification or learned segmentation are claimed. The method does not inpaint.
Clipped pixels remain uncertain even when they could be genuine highlights;
they are not silently added to the SIFT mask.

Synthetic tests cover brightness-only changes, yellow stripes, additive glints,
clipping, insufficient support, nonnegative fits and application switching/
rollback. Actual camera images still require evaluation against manually marked
reflective/nonreflective/uncertain regions and downstream depth-error statistics.
