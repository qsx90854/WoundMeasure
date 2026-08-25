fixed_camera_ippe_diag_v5_6 - Raw ArUco corner vs cornerSubPix diagnostics

Run:
  python rt_two_video_validation_ui_v5_6_corner_diag.py

Display legend when "顯示 Raw/SubPix 比較" is enabled:
  yellow x = raw cv2.aruco.detectMarkers corner
  blue +   = cornerSubPix refined corner actually used by the RT estimator

New diagnostics:
  === RAW detectMarkers -> cornerSubPix DIAGNOSTICS ===
  === RAW CORNER vs SubPix IPPE POSE SHIFT ===

Key fields:
  magnitude_px : raw->SubPix displacement magnitude
  inward_px    : signed displacement toward the raw marker centre; positive=inward
  tangent_px   : tangential component
  area_ratio   : refined polygon area / raw polygon area; <1 suggests shrinkage
  perimeter_ratio : refined perimeter / raw perimeter; <1 suggests shrinkage

For follow-up analysis, provide these sections together with:
  === FINAL ORIENTATION DIAGNOSTICS ===
  === RAW IPPE BRANCH DIAGNOSTICS (SELECTED ENDPOINTS) ===
