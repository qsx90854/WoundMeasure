"""Region-SIFT matching with a shared displacement over a local point group.

The matcher samples a regular grid around a user click in the left image,
warps the right image into the left-image coordinate system with the current
RT/plane homography, and scores a rotated epipolar search band.  Every point in
one candidate uses exactly the same displacement, but independently estimates
its own SIFT scale and dominant orientation in each image.  This module is
deliberately UI-free so it can be tested and tuned independently from the app.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Algorithm import region_sift_frames as dense_frames
from Algorithm.region_sift_scoring import score_candidate_groups


@dataclass(frozen=True)
class RegionSIFTConfig:
    # Left sampling grid.  The default is 3 x 3 cells of 10 x 10 pixels.
    grid_rows: int = 3
    grid_cols: int = 3
    cell_width_px: int = 10
    cell_height_px: int = 10
    points_per_cell: int = 3
    sobel_ksize: int = 3
    min_point_distance_px: float = 1.0
    exclude_click_from_cell_points: bool = True

    # Dense SIFT frame estimation.  Anchor positions stay fixed, while every
    # point independently selects its strongest local DoG scale and dominant
    # gradient orientation.  keypoint_size/angle are fallbacks for flat areas
    # or for explicitly disabling automatic frame estimation.
    auto_scale_orientation: bool = True
    scale_keypoint_sizes_px: Tuple[float, ...] = (
        3.2, 4.0, 5.0, 6.4, 8.0, 10.0, 12.0, 16.0)
    sift_n_octave_layers: int = 3
    sift_sigma: float = 1.6
    descriptor_max_support_radius_px: Optional[float] = 40.0
    flat_keypoint_size_px: float = 3.2
    scale_min_confidence: float = 0.05
    orientation_min_confidence: float = 0.10
    scale_boundary_is_reliable: bool = False
    scale_dog_ratio: float = 2.0 ** (1.0 / 3.0)
    scale_response_pool_sigma_factor: float = 0.75
    scale_response_floor: float = 1e-4
    orientation_bins: int = 36
    orientation_sigma_factor: float = 1.5
    orientation_radius_factor: float = 3.0
    orientation_hist_smooth_passes: int = 2
    frame_coordinate_quantization_px: float = 0.0
    keypoint_size_px: float = 10.0
    keypoint_angle_deg: float = 0.0
    require_all_descriptors: bool = True
    normalize_descriptors: bool = False

    # Rotated epipolar band dimensions.  With unit steps this is 5 x 75
    # candidates: 75 samples along the line and 5 across it.
    search_length_px: float = 75.0
    search_width_px: float = 5.0
    search_along_step_px: float = 1.0
    search_across_step_px: float = 1.0

    # Robust group score.  keep_best_count, when set, overrides the ratio.
    keep_best_ratio: float = 0.75
    keep_best_count: Optional[int] = None
    max_group_score: Optional[float] = None
    epipolar_penalty_weight: float = 0.0
    second_best_exclusion_radius_px: float = 3.0

    # Every row, including low-gradient anchors and P, also contributes to
    # a cell-balanced SIFT score. KEEP/TRIM refers only to the legacy term.
    group_balance_weight: float = 0.35
    frame_consistency_weight: float = 0.05
    frame_scale_tolerance_log2: float = 0.5
    frame_angle_tolerance_deg: float = 30.0
    frame_min_reliable_pairs: int = 3
    reject_uninformative_group: bool = True
    reject_flat_score_surface: bool = True
    flat_score_relative_tolerance: float = 1e-6

    # Warp validity and batching.
    warp_interpolation: int = cv2.INTER_LINEAR
    min_valid_warp_ratio: float = 1.0
    descriptor_border_margin_px: int = 8
    descriptor_batch_size: int = 4096


DEFAULT_CONFIG = RegionSIFTConfig()


class RegionSIFTError(RuntimeError):
    """Expected, user-facing failure in the Region-SIFT matching pipeline."""


def _validate_config(config: RegionSIFTConfig) -> None:
    if config.grid_rows <= 0 or config.grid_cols <= 0:
        raise RegionSIFTError("grid_rows/grid_cols must be positive")
    if config.grid_rows % 2 == 0 or config.grid_cols % 2 == 0:
        raise RegionSIFTError("grid_rows/grid_cols must be odd so there is a center cell")
    if config.cell_width_px <= 0 or config.cell_height_px <= 0:
        raise RegionSIFTError("cell dimensions must be positive")
    if config.points_per_cell <= 0:
        raise RegionSIFTError("points_per_cell must be positive")
    if config.sobel_ksize not in (1, 3, 5, 7):
        raise RegionSIFTError("sobel_ksize must be one of 1, 3, 5, 7")
    if config.keypoint_size_px <= 0:
        raise RegionSIFTError("keypoint_size_px must be positive")
    if (config.sift_n_octave_layers <= 0 or not np.isfinite(config.sift_sigma)
            or config.sift_sigma <= 0.5):
        raise RegionSIFTError("SIFT layers must be positive and sigma must exceed 0.5")
    if (config.descriptor_max_support_radius_px is not None
            and config.descriptor_max_support_radius_px <= 0):
        raise RegionSIFTError("descriptor_max_support_radius_px must be positive or None")
    if config.flat_keypoint_size_px <= 0:
        raise RegionSIFTError("flat_keypoint_size_px must be positive")
    if not 0 <= config.group_balance_weight <= 1:
        raise RegionSIFTError("group_balance_weight must be in [0, 1]")
    if config.frame_consistency_weight < 0 or config.epipolar_penalty_weight < 0:
        raise RegionSIFTError("score penalty weights cannot be negative")
    if config.frame_scale_tolerance_log2 <= 0 or config.frame_angle_tolerance_deg <= 0:
        raise RegionSIFTError("frame consistency tolerances must be positive")
    if config.frame_min_reliable_pairs < 3:
        raise RegionSIFTError("frame_min_reliable_pairs must be at least 3")
    if (not np.isfinite(config.flat_score_relative_tolerance)
            or config.flat_score_relative_tolerance < 0):
        raise RegionSIFTError("flat-score tolerance must be finite and nonnegative")
    if not config.scale_keypoint_sizes_px:
        raise RegionSIFTError("scale_keypoint_sizes_px cannot be empty")
    if any(float(size) <= 0 for size in config.scale_keypoint_sizes_px):
        raise RegionSIFTError("all scale_keypoint_sizes_px values must be positive")
    if config.scale_dog_ratio <= 1.0:
        raise RegionSIFTError("scale_dog_ratio must be greater than 1")
    if config.scale_response_pool_sigma_factor < 0:
        raise RegionSIFTError("scale_response_pool_sigma_factor cannot be negative")
    if config.scale_response_floor < 0:
        raise RegionSIFTError("scale_response_floor cannot be negative")
    if config.orientation_bins < 4:
        raise RegionSIFTError("orientation_bins must be at least 4")
    if config.orientation_sigma_factor <= 0 or config.orientation_radius_factor <= 0:
        raise RegionSIFTError("orientation sigma/radius factors must be positive")
    if config.orientation_hist_smooth_passes < 0:
        raise RegionSIFTError("orientation_hist_smooth_passes cannot be negative")
    if config.frame_coordinate_quantization_px < 0:
        raise RegionSIFTError("frame_coordinate_quantization_px cannot be negative")
    if config.search_length_px <= 0 or config.search_width_px <= 0:
        raise RegionSIFTError("search band dimensions must be positive")
    if config.search_along_step_px <= 0 or config.search_across_step_px <= 0:
        raise RegionSIFTError("search band steps must be positive")
    if not 0.0 < config.keep_best_ratio <= 1.0:
        raise RegionSIFTError("keep_best_ratio must be in (0, 1]")
    if config.second_best_exclusion_radius_px < 0:
        raise RegionSIFTError("second_best_exclusion_radius_px cannot be negative")
    if not 0.0 <= config.min_valid_warp_ratio <= 1.0:
        raise RegionSIFTError("min_valid_warp_ratio must be in [0, 1]")
    if config.descriptor_border_margin_px < 0:
        raise RegionSIFTError("descriptor_border_margin_px cannot be negative")
    if config.descriptor_batch_size <= 0:
        raise RegionSIFTError("descriptor_batch_size must be positive")


def _gradient_label(sample_index: int, sample_count: int) -> str:
    if sample_count == 3:
        return ("high", "mid", "low")[sample_index]
    if sample_count == 1:
        return "mid"
    percentile = 100.0 * (1.0 - sample_index / float(sample_count - 1))
    return f"q{int(round(percentile)):02d}"


def _sample_spaced_ranks(count: int, wanted: int) -> np.ndarray:
    """Return distinct indices from high to low, spread over a sorted array."""
    if wanted > count:
        raise RegionSIFTError(
            f"cell has only {count} usable pixels for {wanted} requested points")
    if wanted == 1:
        return np.array([(count - 1) // 2], dtype=np.int64)
    # linspace is monotonic and distinct whenever wanted <= count.  The array
    # is ascending by gradient, so start at count-1 (High) and end at 0 (Low).
    return np.rint(np.linspace(count - 1, 0, wanted)).astype(np.int64)


def select_region_points(
    image_gray: np.ndarray,
    point_p: Sequence[float],
    config: RegionSIFTConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, Tuple[int, int, int, int]]:
    """Select gradient-ranked cell points plus P.

    Returns ``(points, metadata, gradient_magnitude, roi)``.  P is always the
    last row.  With the default configuration, points has shape (28, 2).
    """
    _validate_config(config)
    gray = np.asarray(image_gray)
    if gray.ndim != 2 or gray.size == 0:
        raise RegionSIFTError("left image must be a non-empty grayscale image")

    px, py = float(point_p[0]), float(point_p[1])
    center_x, center_y = int(round(px)), int(round(py))
    roi_w = config.grid_cols * config.cell_width_px
    roi_h = config.grid_rows * config.cell_height_px
    x0 = center_x - roi_w // 2
    y0 = center_y - roi_h // 2
    x1, y1 = x0 + roi_w, y0 + roi_h
    margin = int(config.descriptor_border_margin_px)
    height, width = gray.shape[:2]
    if x0 < margin or y0 < margin or x1 > width - margin or y1 > height - margin:
        raise RegionSIFTError(
            f"{roi_w}x{roi_h} region plus {margin}px descriptor margin is outside the left image")

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=config.sobel_ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=config.sobel_ksize)
    magnitude = cv2.magnitude(gx, gy)

    click_pixel = (center_x, center_y)
    center_row = config.grid_rows // 2
    center_col = config.grid_cols // 2
    selected: List[Tuple[float, float]] = []
    metadata: List[Dict[str, Any]] = []

    for cell_row in range(config.grid_rows):
        cy0 = y0 + cell_row * config.cell_height_px
        cy1 = cy0 + config.cell_height_px
        for cell_col in range(config.grid_cols):
            cx0 = x0 + cell_col * config.cell_width_px
            cx1 = cx0 + config.cell_width_px
            yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
            coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.int32)
            values = magnitude[cy0:cy1, cx0:cx1].ravel().astype(np.float64)

            is_center_cell = cell_row == center_row and cell_col == center_col
            if is_center_cell and config.exclude_click_from_cell_points:
                keep = np.logical_or(coords[:, 0] != click_pixel[0],
                                     coords[:, 1] != click_pixel[1])
                coords, values = coords[keep], values[keep]

            order = np.argsort(values, kind="mergesort")
            rank_indices = _sample_spaced_ranks(len(order), config.points_per_cell)
            used_flat_indices = set()
            for sample_index, rank_index in enumerate(rank_indices):
                target_rank = int(rank_index)
                rank_candidates = sorted(
                    range(len(order)),
                    key=lambda candidate_rank: (
                        abs(candidate_rank - target_rank), candidate_rank))
                flat_index = None
                for candidate_rank in rank_candidates:
                    candidate_flat_index = int(order[candidate_rank])
                    if candidate_flat_index in used_flat_indices:
                        continue
                    x_try = int(coords[candidate_flat_index, 0])
                    y_try = int(coords[candidate_flat_index, 1])
                    if config.min_point_distance_px > 0 and selected:
                        distances = np.linalg.norm(
                            np.asarray(selected, dtype=np.float32)
                            - np.array([x_try, y_try], dtype=np.float32), axis=1)
                        if np.any(distances < config.min_point_distance_px - 1e-6):
                            continue
                    if (is_center_cell and config.min_point_distance_px > 0
                            and np.linalg.norm(
                                np.array([x_try - px, y_try - py], dtype=np.float32))
                            < config.min_point_distance_px - 1e-6):
                        continue
                    flat_index = candidate_flat_index
                    used_flat_indices.add(candidate_flat_index)
                    break
                if flat_index is None:
                    raise RegionSIFTError(
                        "not enough cell pixels satisfy min_point_distance_px")
                x, y = int(coords[flat_index, 0]), int(coords[flat_index, 1])
                selected.append((float(x), float(y)))
                metadata.append({
                    "index": len(selected) - 1,
                    "is_click": False,
                    "cell_row": cell_row,
                    "cell_col": cell_col,
                    "gradient_label": _gradient_label(
                        sample_index, config.points_per_cell),
                    "gradient_magnitude": float(values[flat_index]),
                    "gradient_rank_fraction": (
                        0.5 if config.points_per_cell == 1
                        else 1.0 - sample_index / float(config.points_per_cell - 1)),
                })

    selected.append((px, py))
    metadata.append({
        "index": len(selected) - 1,
        "is_click": True,
        "cell_row": center_row,
        "cell_col": center_col,
        "gradient_label": "click",
        "gradient_magnitude": float(magnitude[center_y, center_x]),
        "gradient_rank_fraction": None,
    })
    points = np.asarray(selected, dtype=np.float32)
    expected = config.grid_rows * config.grid_cols * config.points_per_cell + 1
    if points.shape != (expected, 2):
        raise RegionSIFTError(
            f"internal point-count mismatch: expected {expected}, got {len(points)}")
    return points, metadata, magnitude, (x0, y0, roi_w, roi_h)


def compute_descriptors_at_points(
    image_gray: np.ndarray,
    points: np.ndarray,
    sift: Optional[Any] = None,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
    sizes_px: Optional[np.ndarray] = None,
    angles_deg: Optional[np.ndarray] = None,
    octaves: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Describe fixed anchors at their selected Gaussian octave/layer."""
    return dense_frames.compute_descriptors_at_points(
        image_gray, points, sift=sift, config=config,
        sizes_px=sizes_px, angles_deg=angles_deg, octaves=octaves)


def estimate_dense_sift_frames(
    image_gray: np.ndarray,
    points: np.ndarray,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
    context: Optional[Any] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Estimate frames without moving anchors or dropping flat descriptors."""
    return dense_frames.estimate_dense_sift_frames(
        image_gray, points, config, context=context, valid_mask=valid_mask)


def plane_homography_left_to_right(
    cand: Dict[str, Any], K_L: np.ndarray
) -> np.ndarray:
    """Build the same RT/plane left-to-right homography used by the UI."""
    plane_n = cand.get("plane_n")
    plane_c = cand.get("plane_c")
    if plane_n is None or plane_c is None:
        raise RegionSIFTError("RT/plane warp requires plane_n and plane_c")
    n = np.asarray(plane_n, dtype=np.float64).reshape(3)
    c = np.asarray(plane_c, dtype=np.float64).reshape(3)
    d_plane = float(np.dot(n, c))
    if not np.isfinite(d_plane) or abs(d_plane) <= 1e-8:
        raise RegionSIFTError("RT/plane warp has an invalid plane distance")
    K_left = np.asarray(K_L, dtype=np.float64).reshape(3, 3)
    K_right = np.asarray(cand["K_R"], dtype=np.float64).reshape(3, 3)
    R = np.asarray(cand["R_rel"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(cand["t_rel"], dtype=np.float64).reshape(3, 1)
    H_lr = K_right @ (R + (t @ n.reshape(1, 3)) / d_plane) @ np.linalg.inv(K_left)
    if not np.all(np.isfinite(H_lr)) or abs(float(np.linalg.det(H_lr))) <= 1e-12:
        raise RegionSIFTError("RT/plane homography is singular")
    return H_lr


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    mapped_h = (np.asarray(homography, dtype=np.float64) @ pts_h.T).T
    valid = np.abs(mapped_h[:, 2]) > 1e-10
    if not np.all(valid):
        raise RegionSIFTError("homography mapped a point to infinity")
    mapped = mapped_h[:, :2] / mapped_h[:, 2:3]
    if not np.all(np.isfinite(mapped)):
        raise RegionSIFTError("homography produced non-finite coordinates")
    return mapped.astype(np.float32)


def _centered_offsets(span_px: float, step_px: float) -> np.ndarray:
    """Symmetric sample offsets whose default 75/1 gives -37..37."""
    half = max(0.0, (float(span_px) - 1.0) / 2.0)
    values = np.arange(-half, half + step_px * 0.25, step_px, dtype=np.float32)
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    if not np.any(np.isclose(values, 0.0, atol=1e-6)):
        values = np.sort(np.append(values, np.float32(0.0)))
    return values.astype(np.float32)


def build_epipolar_band(
    point_p: Sequence[float],
    F: np.ndarray,
    H_lr: np.ndarray,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
) -> Dict[str, np.ndarray]:
    """Create candidate centers in warped-right coordinates."""
    p = np.asarray(point_p, dtype=np.float64).reshape(2)
    line_right = np.asarray(F, dtype=np.float64).reshape(3, 3) @ np.array(
        [p[0], p[1], 1.0], dtype=np.float64)
    line_warp = np.asarray(H_lr, dtype=np.float64).T @ line_right
    line_norm = float(np.hypot(line_warp[0], line_warp[1]))
    if line_norm <= 1e-10:
        raise RegionSIFTError("cannot construct a valid epipolar line")
    line_warp = line_warp / line_norm
    tangent = np.array([-line_warp[1], line_warp[0]], dtype=np.float32)
    normal = np.array([line_warp[0], line_warp[1]], dtype=np.float32)

    seed_right = _transform_points(p.reshape(1, 2), H_lr)[0]
    H_rl = np.linalg.inv(H_lr)
    seed_warp = _transform_points(seed_right.reshape(1, 2), H_rl)[0]
    signed = float(line_warp[0] * seed_warp[0]
                   + line_warp[1] * seed_warp[1] + line_warp[2])
    seed_on_line = seed_warp - normal * signed

    along_values = _centered_offsets(
        config.search_length_px, config.search_along_step_px)
    across_values = _centered_offsets(
        config.search_width_px, config.search_across_step_px)
    centers: List[np.ndarray] = []
    along_out: List[float] = []
    across_out: List[float] = []
    # Along-major ordering makes score maps and deterministic ties intuitive.
    for along in along_values:
        for across in across_values:
            centers.append(seed_on_line + tangent * along + normal * across)
            along_out.append(float(along))
            across_out.append(float(across))
    return {
        "line_right": line_right.astype(np.float64),
        "line_warp": line_warp.astype(np.float64),
        "tangent": tangent,
        "normal": normal,
        "seed_right": seed_right.astype(np.float32),
        "seed_warp": seed_warp.astype(np.float32),
        "seed_on_line": seed_on_line.astype(np.float32),
        "centers_warp": np.asarray(centers, dtype=np.float32),
        "along_offsets": np.asarray(along_out, dtype=np.float32),
        "across_offsets": np.asarray(across_out, dtype=np.float32),
        "along_values": along_values,
        "across_values": across_values,
    }


def _keep_count(point_count: int, config: RegionSIFTConfig) -> int:
    if config.keep_best_count is not None:
        keep = int(config.keep_best_count)
    else:
        keep = int(round(point_count * config.keep_best_ratio))
    return min(point_count, max(1, keep))


def score_descriptor_group(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Return legacy TrimK mean, row distances, keep mask and keep count.

    The full matcher additionally blends in the cell-balanced mean through
    score_candidate_groups; this helper alone is not the final GroupScore.
    """
    left = np.asarray(left_descriptors, dtype=np.float32)
    right = np.asarray(right_descriptors, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 128:
        raise RegionSIFTError(
            f"descriptor matrices must share Nx128 shape, got {left.shape}/{right.shape}")
    distances = np.linalg.norm(right - left, axis=1)
    order = np.argsort(distances, kind="mergesort")
    keep_count = _keep_count(len(distances), config)
    keep_mask = np.zeros(len(distances), dtype=bool)
    keep_mask[order[:keep_count]] = True
    return float(np.mean(distances[order[:keep_count]])), distances, keep_mask, keep_count


def _eroded_valid_mask(valid_mask: np.ndarray, margin: int,
                       min_ratio: float) -> np.ndarray:
    valid = (np.asarray(valid_mask) > 0).astype(np.uint8)
    if margin <= 0:
        return valid
    kernel_size = margin * 2 + 1
    if min_ratio >= 1.0 - 1e-9:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(valid, kernel, iterations=1,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # boxFilter yields the valid fraction in each descriptor support window.
    ratio = cv2.boxFilter(valid.astype(np.float32), -1,
                          (kernel_size, kernel_size), normalize=True,
                          borderType=cv2.BORDER_CONSTANT)
    return (ratio >= float(min_ratio)).astype(np.uint8)


def _search_band_polygon(band: Dict[str, np.ndarray],
                         config: RegionSIFTConfig) -> np.ndarray:
    center = band["seed_on_line"]
    tangent, normal = band["tangent"], band["normal"]
    along_half = max(0.0, (config.search_length_px - 1.0) / 2.0)
    across_half = max(0.0, (config.search_width_px - 1.0) / 2.0)
    return np.asarray([
        center - tangent * along_half - normal * across_half,
        center + tangent * along_half - normal * across_half,
        center + tangent * along_half + normal * across_half,
        center - tangent * along_half + normal * across_half,
    ], dtype=np.float32)


def run_region_sift_matching(
    img_left_gray: np.ndarray,
    img_right_gray: np.ndarray,
    point_p: Sequence[float],
    cand: Dict[str, Any],
    K_L: np.ndarray,
    sift: Optional[Any] = None,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
    left_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Region-SIFT and return an original-right-image match plus debug data."""
    started = time.perf_counter()
    base_result: Dict[str, Any] = {
        "m_pt": None,
        "method": "",
        "reject_reason": None,
        "region_debug": None,
    }
    try:
        _validate_config(config)
        left_gray = np.asarray(img_left_gray)
        right_gray = np.asarray(img_right_gray)
        if left_gray.ndim != 2 or right_gray.ndim != 2:
            raise RegionSIFTError("Region-SIFT requires grayscale left/right images")
        if cand.get("F") is None:
            raise RegionSIFTError("Region-SIFT requires a fundamental matrix")
        if sift is None:
            sift = cv2.SIFT_create(nOctaveLayers=config.sift_n_octave_layers,
                                   sigma=config.sift_sigma)

        # Content identity also detects in-place camera-buffer updates. Shape,
        # object identity, or click coordinates alone cannot identify a frame.
        image_digest = (hashlib.blake2b(
            np.ascontiguousarray(left_gray).tobytes(), digest_size=16).digest()
            if left_cache is not None else None)
        cache_key = (
            tuple(np.asarray(point_p, dtype=np.float64).reshape(2)),
            left_gray.shape,
            left_gray.dtype.str,
            image_digest,
            config,
        )
        cached = left_cache.get("region_sift") if left_cache is not None else None
        if (cached is not None and cached.get("key") == cache_key
                and "frames" in cached):
            left_points = cached["points"]
            point_metadata = [dict(item) for item in cached["metadata"]]
            left_descriptors = cached["descriptors"]
            left_frames = {
                key: np.asarray(value).copy()
                for key, value in cached["frames"].items()
            }
            left_roi = cached["roi"]
        else:
            left_points, point_metadata, _magnitude, left_roi = select_region_points(
                left_gray, point_p, config)
            left_frames = estimate_dense_sift_frames(
                left_gray, left_points, config)
            if not np.all(left_frames['valid']):
                raise RegionSIFTError(
                    'left anchors lack complete SIFT/scale-space support; '
                    'move the click inward or tune the support limit')
            left_descriptors = compute_descriptors_at_points(
                left_gray, left_points, sift, config,
                sizes_px=left_frames["size_px"],
                angles_deg=left_frames["angle_deg"],
                octaves=left_frames['octave'])
            if left_cache is not None:
                left_cache["region_sift"] = {
                    "key": cache_key,
                    "points": left_points,
                    "metadata": [dict(item) for item in point_metadata],
                    "descriptors": left_descriptors,
                    "frames": {
                        key: np.asarray(value).copy()
                        for key, value in left_frames.items()
                    },
                    "roi": left_roi,
                }

        point_count = len(left_points)
        expected_count = config.grid_rows * config.grid_cols * config.points_per_cell + 1
        if left_descriptors.shape != (expected_count, 128):
            raise RegionSIFTError(
                f"left descriptor matrix must be {expected_count}x128, got {left_descriptors.shape}")

        H_lr = plane_homography_left_to_right(cand, K_L)
        H_rl = np.linalg.inv(H_lr)
        height, width = left_gray.shape[:2]
        warped_right = cv2.warpPerspective(
            right_gray, H_rl, (width, height),
            flags=int(config.warp_interpolation),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        source_valid = np.ones(right_gray.shape[:2], dtype=np.float32)
        warped_fraction = cv2.warpPerspective(
            source_valid, H_rl, (width, height),
            flags=int(config.warp_interpolation),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_valid = (warped_fraction >= 1.0 - 1e-6).astype(np.uint8) * 255
        support_valid = _eroded_valid_mask(
            warped_valid, int(config.descriptor_border_margin_px),
            float(config.min_valid_warp_ratio))

        band = build_epipolar_band(point_p, cand["F"], H_lr, config)
        p = np.asarray(point_p, dtype=np.float32).reshape(2)
        point_offsets = left_points - p
        centers = band["centers_warp"]
        all_positions = centers[:, None, :] + point_offsets[None, :, :]
        rounded = np.rint(all_positions).astype(np.int32)
        xs, ys = rounded[:, :, 0], rounded[:, :, 1]
        in_bounds = (
            np.all(np.isfinite(all_positions), axis=2)
            & (all_positions[:, :, 0] >= 0)
            & (all_positions[:, :, 0] <= width - 1)
            & (all_positions[:, :, 1] >= 0)
            & (all_positions[:, :, 1] <= height - 1)
            & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height))
        valid_candidates = np.all(in_bounds, axis=1)
        in_bound_indices = np.flatnonzero(valid_candidates)
        for candidate_index in in_bound_indices:
            valid_candidates[candidate_index] = bool(np.all(
                support_valid[ys[candidate_index], xs[candidate_index]] > 0))
        valid_indices = np.flatnonzero(valid_candidates)
        if len(valid_indices) == 0:
            raise RegionSIFTError("no search candidate has complete valid warp support")

        group_scores = np.full(len(centers), np.inf, dtype=np.float64)
        objective_scores = np.full(len(centers), np.inf, dtype=np.float64)
        trimmed_scores = np.full(len(centers), np.inf, dtype=np.float64)
        balanced_scores = np.full(len(centers), np.inf, dtype=np.float64)
        frame_penalties = np.full(len(centers), np.inf, dtype=np.float64)
        candidate_distances: Dict[int, np.ndarray] = {}
        candidate_descriptors: Dict[int, np.ndarray] = {}
        valid_positions = all_positions[valid_indices]
        flat_valid_positions = valid_positions.reshape(-1, 2)
        right_frames_flat = estimate_dense_sift_frames(
            warped_right, flat_valid_positions, config, valid_mask=warped_valid)
        right_frames = {
            key: np.asarray(value).reshape(len(valid_indices), point_count)
            for key, value in right_frames_flat.items()
        }
        complete_support = np.all(right_frames['valid'], axis=1)
        valid_indices = valid_indices[complete_support]
        right_frames = {key: value[complete_support]
                        for key, value in right_frames.items()}
        if len(valid_indices) == 0:
            raise RegionSIFTError('no search candidate has complete per-scale SIFT support')
        valid_row_by_candidate = {
            int(candidate_index): row_index
            for row_index, candidate_index in enumerate(valid_indices)
        }
        candidates_per_batch = max(
            1, int(config.descriptor_batch_size) // max(1, point_count))

        for start in range(0, len(valid_indices), candidates_per_batch):
            batch_indices = valid_indices[start:start + candidates_per_batch]
            batch_points = all_positions[batch_indices].reshape(-1, 2)
            valid_rows = np.arange(start, start + len(batch_indices))
            batch_descriptors = compute_descriptors_at_points(
                warped_right, batch_points, sift, config,
                sizes_px=right_frames["size_px"][valid_rows].reshape(-1),
                angles_deg=right_frames["angle_deg"][valid_rows].reshape(-1),
                octaves=right_frames['octave'][valid_rows].reshape(-1))
            expected_rows = len(batch_indices) * point_count
            if len(batch_descriptors) != expected_rows:
                raise RegionSIFTError(
                    f"right descriptor batch returned {len(batch_descriptors)}/{expected_rows} rows")
            matrices = batch_descriptors.reshape(len(batch_indices), point_count, 128)
            distances = np.linalg.norm(
                matrices - left_descriptors[None, :, :], axis=2)
            components = score_candidate_groups(
                distances, point_metadata, left_frames,
                {key: value[valid_rows] for key, value in right_frames.items()},
                config)
            scores = components['group_score']
            penalties = (float(config.epipolar_penalty_weight)
                         * np.abs(band["across_offsets"][batch_indices]))
            for local_index, candidate_index in enumerate(batch_indices):
                ci = int(candidate_index)
                group_scores[ci] = float(scores[local_index])
                trimmed_scores[ci] = float(components['trimmed_mean'][local_index])
                balanced_scores[ci] = float(components['balanced_mean'][local_index])
                frame_penalties[ci] = float(components['frame_penalty'][local_index])
                objective_scores[ci] = float(
                    components['objective_without_epi'][local_index] + penalties[local_index])
                candidate_distances[ci] = distances[local_index].astype(np.float32)
                candidate_descriptors[ci] = matrices[local_index].astype(np.float32)

        finite_indices = np.flatnonzero(np.isfinite(objective_scores))
        if len(finite_indices) == 0:
            raise RegionSIFTError("SIFT produced no scoreable search candidate")
        # Deterministic ordering: objective, distance from seed, then input order.
        radial = np.hypot(band["along_offsets"], band["across_offsets"])
        ranked = sorted(
            (int(index) for index in finite_indices),
            key=lambda index: (float(objective_scores[index]),
                               float(radial[index]), index))
        best_index = ranked[0]
        best_group_score = float(group_scores[best_index])
        best_objective_score = float(objective_scores[best_index])
        best_center_warp = centers[best_index].astype(np.float32)
        best_points_warp = all_positions[best_index].astype(np.float32)
        best_center_right = _transform_points(
            best_center_warp.reshape(1, 2), H_lr)[0]
        best_points_right = _transform_points(best_points_warp, H_lr)
        best_distances = candidate_distances[best_index]
        best_descriptors = candidate_descriptors[best_index]
        best_valid_row = valid_row_by_candidate[best_index]
        best_right_frames = {
            key: np.asarray(value[best_valid_row]).copy()
            for key, value in right_frames.items()
        }
        best_components = score_candidate_groups(
            best_distances.reshape(1, -1), point_metadata, left_frames,
            {key: value.reshape(1, -1) for key, value in best_right_frames.items()},
            config)
        distance_order = np.argsort(best_distances, kind="mergesort")
        keep_count = _keep_count(point_count, config)
        keep_mask = np.zeros(point_count, dtype=bool)
        keep_mask[distance_order[:keep_count]] = True

        # The immediately adjacent grid point normally belongs to the same
        # score basin, so use a spatially distinct runner-up when measuring
        # ambiguity.  Offsets are in warped-right pixels.
        best_along = float(band["along_offsets"][best_index])
        best_across = float(band["across_offsets"][best_index])
        exclusion_radius = float(config.second_best_exclusion_radius_px)
        second_index = next((
            index for index in ranked[1:]
            if float(np.hypot(
                float(band["along_offsets"][index]) - best_along,
                float(band["across_offsets"][index]) - best_across))
            > exclusion_radius
        ), None)
        if second_index is None:
            second_group_score = float("nan")
            second_objective_score = float("nan")
            group_score_margin = float("nan")
            objective_score_margin = float("nan")
            objective_score_ratio = float("nan")
            second_along = float("nan")
            second_across = float("nan")
        else:
            second_group_score = float(group_scores[second_index])
            second_objective_score = float(objective_scores[second_index])
            group_score_margin = second_group_score - best_group_score
            objective_score_margin = (
                second_objective_score - best_objective_score)
            objective_score_ratio = (
                best_objective_score / second_objective_score
                if second_objective_score > 1e-12 else float("nan"))
            second_along = float(band["along_offsets"][second_index])
            second_across = float(band["across_offsets"][second_index])

        kept_distances = best_distances[keep_mask]
        descriptor_norm_reference = float(np.median(
            np.linalg.norm(left_descriptors, axis=1)))
        relative_group_score = (
            best_group_score / descriptor_norm_reference
            if descriptor_norm_reference > 1e-12 else float("nan"))
        relative_second_group_score = (
            second_group_score / descriptor_norm_reference
            if descriptor_norm_reference > 1e-12 else float("nan"))
        kept_l2_stats = {
            "mean": float(np.mean(kept_distances)),
            "median": float(np.median(kept_distances)),
            "std": float(np.std(kept_distances)),
            "min": float(np.min(kept_distances)),
            "max": float(np.max(kept_distances)),
        }
        scale_ratios = (
            best_right_frames["size_px"]
            / np.maximum(left_frames["size_px"], 1e-12))
        signed_angle_deltas = (
            best_right_frames["angle_deg"]
            - left_frames["angle_deg"] + 180.0) % 360.0 - 180.0
        kept_scale_ratio_median = float(np.median(scale_ratios[keep_mask]))
        kept_scale_ratio_mad = float(np.median(np.abs(
            scale_ratios[keep_mask] - kept_scale_ratio_median)))
        kept_abs_angle_delta_median = float(np.median(np.abs(
            signed_angle_deltas[keep_mask])))
        kept_abs_angle_delta_mad = float(np.median(np.abs(
            np.abs(signed_angle_deltas[keep_mask])
            - kept_abs_angle_delta_median)))
        kept_cells = {
            (int(record["cell_row"]), int(record["cell_col"]))
            for index, record in enumerate(point_metadata)
            if keep_mask[index] and not record.get("is_click", False)
        }

        for index, record in enumerate(point_metadata):
            record.update({
                "left_point": left_points[index].copy(),
                "right_point_warp": best_points_warp[index].copy(),
                "right_point_original": best_points_right[index].copy(),
                "left_scale_px": float(left_frames["size_px"][index]),
                "left_angle_deg": float(left_frames["angle_deg"][index]),
                "left_scale_response": float(
                    left_frames["scale_response"][index]),
                "left_orientation_strength": float(
                    left_frames["orientation_strength"][index]),
                "right_scale_px": float(best_right_frames["size_px"][index]),
                "right_angle_deg": float(best_right_frames["angle_deg"][index]),
                "right_scale_response": float(
                    best_right_frames["scale_response"][index]),
                "right_orientation_strength": float(
                    best_right_frames["orientation_strength"][index]),
                "scale_ratio": float(
                    scale_ratios[index]),
                "angle_delta_deg": float(signed_angle_deltas[index]),
                "l2_distance": float(best_distances[index]),
                "distance_rank": int(np.where(distance_order == index)[0][0]) + 1,
                "kept": bool(keep_mask[index]),
                'balanced_score_active': config.group_balance_weight > 0,
                'left_octave': int(left_frames['octave'][index]),
                'right_octave': int(best_right_frames['octave'][index]),
                'left_support_radius_px': float(left_frames['support_radius_px'][index]),
                'right_support_radius_px': float(best_right_frames['support_radius_px'][index]),
                'scale_pair_reliable': bool(best_components['frame_scale_pair_mask'][0, index]),
                'angle_pair_reliable': bool(best_components['frame_orientation_pair_mask'][0, index]),
                'scale_residual_log2': float(best_components['frame_scale_residual_log2'][0, index]),
                'angle_residual_deg': float(best_components['frame_angle_residual_deg'][0, index]),
            })

        debug = {
            "config": config,
            "point_count": point_count,
            "auto_point_count": point_count - 1,
            "keep_count": keep_count,
            "trim_count": point_count - keep_count,
            "left_roi": left_roi,
            "left_points": left_points,
            "left_frame_sizes_px": left_frames["size_px"],
            "left_frame_angles_deg": left_frames["angle_deg"],
            "left_scale_responses": left_frames["scale_response"],
            "left_orientation_strengths": left_frames["orientation_strength"],
            "left_descriptors": left_descriptors,
            'left_frames': left_frames,
            'right_frames': best_right_frames,
            "point_metadata": point_metadata,
            "H_lr": H_lr,
            "H_rl": H_rl,
            "warped_right_gray": warped_right,
            "warped_valid_mask": warped_valid,
            "line_right": band["line_right"],
            "line_warp": band["line_warp"],
            "tangent": band["tangent"],
            "normal": band["normal"],
            "seed_right": band["seed_right"],
            "seed_warp": band["seed_warp"],
            "seed_on_line": band["seed_on_line"],
            "search_band_polygon_warp": _search_band_polygon(band, config),
            "candidate_count": len(centers),
            "valid_candidate_count": len(finite_indices),
            "candidate_centers_warp": centers,
            "candidate_group_scores": group_scores,
            "candidate_objective_scores": objective_scores,
            'candidate_trimmed_scores': trimmed_scores,
            'candidate_balanced_scores': balanced_scores,
            'candidate_frame_penalties': frame_penalties,
            "candidate_along_offsets": band["along_offsets"],
            "candidate_across_offsets": band["across_offsets"],
            "best_candidate_index": best_index,
            "best_along_offset_px": best_along,
            "best_across_offset_px": best_across,
            "best_center_warp": best_center_warp,
            "best_center_right": best_center_right,
            "best_displacement_warp": best_center_warp - p,
            "best_points_warp": best_points_warp,
            "best_points_right": best_points_right,
            "right_frame_sizes_px": best_right_frames["size_px"],
            "right_frame_angles_deg": best_right_frames["angle_deg"],
            "right_scale_responses": best_right_frames["scale_response"],
            "right_orientation_strengths": best_right_frames[
                "orientation_strength"],
            "right_descriptors": best_descriptors,
            "distances": best_distances,
            "distance_order": distance_order,
            "keep_mask": keep_mask,
            "group_score": best_group_score,
            "objective_score": best_objective_score,
            'trimmed_score': float(trimmed_scores[best_index]),
            'balanced_score': float(balanced_scores[best_index]),
            'frame_penalty': float(frame_penalties[best_index]),
            'frame_scale_pair_count': int(best_components['frame_scale_pair_count'][0]),
            'frame_orientation_pair_count': int(best_components['frame_orientation_pair_count'][0]),
            'frame_scale_center_log2': float(best_components['frame_scale_center_log2'][0]),
            'frame_angle_center_deg': float(best_components['frame_angle_center_deg'][0]),
            "second_best_exclusion_radius_px": exclusion_radius,
            "second_candidate_index": second_index,
            "second_group_score": second_group_score,
            "second_objective_score": second_objective_score,
            "second_along_offset_px": second_along,
            "second_across_offset_px": second_across,
            "group_score_margin": group_score_margin,
            "objective_score_margin": objective_score_margin,
            "objective_score_ratio": objective_score_ratio,
            "descriptor_norm_reference": descriptor_norm_reference,
            "relative_group_score": relative_group_score,
            "relative_second_group_score": relative_second_group_score,
            "kept_l2_stats": kept_l2_stats,
            "kept_scale_ratio_median": kept_scale_ratio_median,
            "kept_scale_ratio_mad": kept_scale_ratio_mad,
            "kept_abs_angle_delta_median": kept_abs_angle_delta_median,
            "kept_abs_angle_delta_mad": kept_abs_angle_delta_mad,
            "kept_cell_coverage": len(kept_cells),
            "total_cell_count": config.grid_rows * config.grid_cols,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        rejection = None
        if (config.reject_uninformative_group
                and np.all(np.linalg.norm(left_descriptors, axis=1) <= 1e-12)):
            rejection = 'all left SIFT descriptors are zero; the group cannot determine a location'
        score_span = float(np.ptp(group_scores[finite_indices]))
        debug['score_surface_span'] = score_span
        unit = 1.0 if config.normalize_descriptors else 512.0
        if (rejection is None and config.reject_flat_score_surface
                and len(finite_indices) > 1
                and score_span <= config.flat_score_relative_tolerance * unit):
            rejection = 'SIFT group score is flat across the valid search band; location is ambiguous'
        if rejection is not None:
            base_result['reject_reason'] = rejection
            base_result['region_debug'] = debug
            return base_result
        if (config.max_group_score is not None
                and best_group_score > float(config.max_group_score)):
            base_result["reject_reason"] = (
                f"Region-SIFT GroupScore {best_group_score:.3f} exceeds "
                f"{float(config.max_group_score):.3f}")
            base_result["region_debug"] = debug
            return base_result

        base_result.update({
            "m_pt": best_center_right.astype(np.float32),
            "method": f"Region-SIFT({keep_count}/{point_count})",
            "region_debug": debug,
        })
        return base_result
    except (RegionSIFTError, cv2.error, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        base_result["reject_reason"] = str(exc)
        return base_result


def with_config(config: RegionSIFTConfig = DEFAULT_CONFIG, **changes: Any) -> RegionSIFTConfig:
    """Convenience helper for experiments without mutating shared defaults."""
    return replace(config, **changes)


__all__ = [
    "DEFAULT_CONFIG",
    "RegionSIFTConfig",
    "RegionSIFTError",
    "build_epipolar_band",
    "compute_descriptors_at_points",
    "estimate_dense_sift_frames",
    "plane_homography_left_to_right",
    "run_region_sift_matching",
    "score_descriptor_group",
    "select_region_points",
    "with_config",
]
