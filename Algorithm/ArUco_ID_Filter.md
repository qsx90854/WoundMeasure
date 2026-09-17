# ArUco ID allowlist

In depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py:

```python
ARUCO_ALLOWED_PATTERN_IDS = [2, 5, 9, 12]
```

Only listed IDs contribute marker observations to temporal frame selection,
the marker map, RT estimation, direct two-image RT, the preview marker poses,
the global/fallback plane and the Shared marker-corner height plane.
This restricts marker observations, not the independent SIFT image ROI.

- `None`: allow all detected IDs (previous behavior).
- `[]`: allow none; do not fall back to excluded markers.
- A list of nonnegative integer IDs: allow exactly those IDs. IDs need not all
  appear in a frame. Existing minimum-marker and pose-quality checks still apply.

Restart the program after changing the list so existing RT, marker maps,
planes and frame-selection caches are rebuilt. At analysis start, the console
prints `[ArUco ID filter] allowed=[2, 5, 9, 12]`.

The temporal analyzer accepts an optional `allowed_marker_ids` argument;
external marker-corner overrides are filtered before entering its caches too.
Other callers that omit this argument retain their previous unrestricted behavior.
