import cv2
import numpy as np


SPEC_MASK_V_THRESHOLD = 210
SPEC_MASK_S_THRESHOLD = 80
SPEC_MASK_RGB_HIGH_THRESHOLD = 235
SPEC_MASK_WHITENESS_THRESHOLD = 36
SPEC_MASK_DILATE = 5
SPEC_TEMPORAL_OFFSETS = (-6, -3, 3, 6)
SPEC_TEMPORAL_STD_THRESHOLD = 16.0
SPEC_TEMPORAL_RESIDUAL_THRESHOLD = 24.0
SPEC_TEMPORAL_BRIGHT_THRESHOLD = 150

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_CLAHE_CACHE = {}


def _get_clahe(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE):
    key = (clip_limit, tile_size)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return _CLAHE_CACHE[key]


def _preprocess_gray(gray_img, enable_clahe=True):
    if enable_clahe:
        return _get_clahe().apply(gray_img)
    return gray_img


def compute_specular_mask_bgr(
    bgr,
    v_threshold=SPEC_MASK_V_THRESHOLD,
    s_threshold=SPEC_MASK_S_THRESHOLD,
    rgb_high_threshold=SPEC_MASK_RGB_HIGH_THRESHOLD,
    whiteness_threshold=SPEC_MASK_WHITENESS_THRESHOLD,
    dilate=SPEC_MASK_DILATE,
):
    """Detect likely specular highlight pixels without modifying the source image."""
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(bgr)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    whiteness = max_rgb.astype(np.int16) - min_rgb.astype(np.int16)

    bright_low_sat = (v >= v_threshold) & (s <= s_threshold)
    near_white = (max_rgb >= rgb_high_threshold) & (whiteness <= whiteness_threshold)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 9)
    local_hot = gray.astype(np.int16) - blur.astype(np.int16)
    adaptive_hot = (gray >= max(160, v_threshold - 20)) & (local_hot >= 18)

    mask = (bright_low_sat | near_white | adaptive_hot).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if dilate > 0:
        k = 2 * int(dilate) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask


def overlay_specular_mask_rgb(rgb, spatial_mask, temporal_mask=None, alpha=0.55):
    """Overlay spatial specular mask in light blue and temporal instability in light red."""
    if rgb is None or spatial_mask is None:
        return rgb
    out = rgb.copy()
    spatial_bool = spatial_mask > 0
    temporal_bool = (temporal_mask > 0) if temporal_mask is not None else np.zeros(spatial_bool.shape, dtype=bool)
    if not np.any(spatial_bool) and not np.any(temporal_bool):
        return out

    light_blue = np.zeros_like(out)
    light_blue[:, :] = (80, 210, 255)
    light_red = np.zeros_like(out)
    light_red[:, :] = (255, 70, 70)
    light_purple = np.zeros_like(out)
    light_purple[:, :] = (210, 150, 255)

    spatial_only = spatial_bool & ~temporal_bool
    temporal_only = temporal_bool & ~spatial_bool
    both = spatial_bool & temporal_bool
    out[spatial_only] = cv2.addWeighted(out, 1.0 - alpha, light_blue, alpha, 0)[spatial_only]
    out[temporal_only] = cv2.addWeighted(out, 1.0 - alpha, light_red, alpha, 0)[temporal_only]
    out[both] = cv2.addWeighted(out, 1.0 - alpha, light_purple, alpha, 0)[both]
    return out


def detect_aruco_corners_bgr_for_pose(bgr, preprocess_gray_fn=None):
    """Detect ArUco marker corners in an already processed/undistorted UI image."""
    if bgr is None or bgr.size == 0:
        return {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = preprocess_gray_fn(gray, True) if preprocess_gray_fn is not None else _preprocess_gray(gray, True)
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dict_4x4, parameters=params)
    if ids is None or len(ids) == 0:
        return {}
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.0001)
    for c in corners:
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), term)
    return {int(mid[0]): c.reshape(4, 2) for mid, c in zip(ids, corners)}


def estimate_marker_map_pose_from_corners(corners_dict, marker_map, marker_size_mm, K):
    """Estimate camera pose of the marker-map reference frame from detected marker corners."""
    if not corners_dict or not marker_map:
        return None
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    obj_pts = []
    img_pts = []
    for mid, pts in corners_dict.items():
        if mid not in marker_map:
            continue
        R_m2ref, t_m2ref = marker_map[mid]
        pts_w = (R_m2ref @ canon.T).T + t_m2ref.T
        obj_pts.append(pts_w)
        img_pts.append(pts)
    if not obj_pts:
        return None
    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)


def marker_plane_homography_from_pose(R, t, K):
    """Homography from marker-map z=0 plane coordinates to image coordinates."""
    H = K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def compute_rt_aligned_temporal_specular_mask_bgr(
    center_bgr,
    center_frame_idx,
    video_data,
    K,
    marker_size_mm,
    process_frame_fn=None,
    return_parts=False,
    preprocess_gray_fn=None,
):
    """Combine single-frame highlight detection with RT/homography-aligned temporal instability."""
    base_mask = compute_specular_mask_bgr(center_bgr)
    if center_bgr is None or center_frame_idx is None or not video_data:
        temporal_empty = np.zeros_like(base_mask) if base_mask is not None else None
        return (base_mask, base_mask, temporal_empty) if return_parts else base_mask

    frames = video_data.get("all_frames")
    marker_map = video_data.get("marker_map")
    if not frames or marker_map is None:
        temporal_empty = np.zeros_like(base_mask) if base_mask is not None else None
        return (base_mask, base_mask, temporal_empty) if return_parts else base_mask

    center_corners = detect_aruco_corners_bgr_for_pose(center_bgr, preprocess_gray_fn=preprocess_gray_fn)
    center_pose = estimate_marker_map_pose_from_corners(center_corners, marker_map, marker_size_mm, K)
    if center_pose is None:
        temporal_empty = np.zeros_like(base_mask) if base_mask is not None else None
        return (base_mask, base_mask, temporal_empty) if return_parts else base_mask

    R_center, t_center = center_pose
    H_center = marker_plane_homography_from_pose(R_center, t_center, K)
    h, w = center_bgr.shape[:2]
    center_gray = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    aligned = [center_gray]
    valid = [np.ones((h, w), dtype=np.uint8)]
    used_indices = {int(center_frame_idx)}

    for offset in SPEC_TEMPORAL_OFFSETS:
        idx = int(np.clip(center_frame_idx + offset, 0, len(frames) - 1))
        if idx in used_indices:
            continue
        used_indices.add(idx)

        neighbor_bgr = frames[idx]
        if process_frame_fn is not None:
            processed = process_frame_fn(neighbor_bgr)
            neighbor_bgr = processed[0] if isinstance(processed, tuple) else processed

        neighbor_corners = detect_aruco_corners_bgr_for_pose(neighbor_bgr, preprocess_gray_fn=preprocess_gray_fn)
        neighbor_pose = estimate_marker_map_pose_from_corners(neighbor_corners, marker_map, marker_size_mm, K)
        if neighbor_pose is None:
            continue

        R_neighbor, t_neighbor = neighbor_pose
        H_neighbor = marker_plane_homography_from_pose(R_neighbor, t_neighbor, K)
        try:
            H_neighbor_to_center = H_center @ np.linalg.inv(H_neighbor)
        except np.linalg.LinAlgError:
            continue

        neighbor_gray = cv2.cvtColor(neighbor_bgr, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.warpPerspective(
            neighbor_gray,
            H_neighbor_to_center,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)
        warped_valid = cv2.warpPerspective(
            np.ones(neighbor_gray.shape[:2], dtype=np.uint8) * 255,
            H_neighbor_to_center,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned.append(warped_gray)
        valid.append((warped_valid > 0).astype(np.uint8))

    if len(aligned) < 3:
        temporal_empty = np.zeros_like(base_mask) if base_mask is not None else None
        return (base_mask, base_mask, temporal_empty) if return_parts else base_mask

    stack = np.stack(aligned, axis=0)
    valid_stack = np.stack(valid, axis=0).astype(bool)
    enough_valid = np.sum(valid_stack, axis=0) >= 3
    stack_masked = np.where(valid_stack, stack, np.nan)
    temporal_std = np.nanstd(stack_masked, axis=0)
    temporal_median = np.nanmedian(stack_masked, axis=0)
    residual = np.abs(center_gray - temporal_median)

    temporal_mask = (
        enough_valid
        & (center_gray >= SPEC_TEMPORAL_BRIGHT_THRESHOLD)
        & (temporal_std >= SPEC_TEMPORAL_STD_THRESHOLD)
        & ((residual >= SPEC_TEMPORAL_RESIDUAL_THRESHOLD) | (base_mask > 0))
    ).astype(np.uint8) * 255
    temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    temporal_mask = cv2.dilate(temporal_mask, np.ones((7, 7), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(base_mask, temporal_mask)
    return (combined_mask, base_mask, temporal_mask) if return_parts else combined_mask
