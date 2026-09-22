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
from Algorithm.region_sift_profiling import begin as begin_detail, end as end_detail


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

    # Custom masked operator only. OpenCV remains the default baseline.
    use_masked_sift: bool = False
    masked_extra_margin_px: int = 0
    masked_min_blur_weight: float = 0.5
    masked_min_valid_fraction: float = 0.6
    masked_min_cell_fraction: float = 0.8
    masked_min_common_fraction: float = 0.5
    masked_descriptor_clip: float = 0.2
    masked_min_group_fraction: float = 0.75
    masked_min_points_per_cell: int = 1
    masked_max_deficient_cells: int = 1
    masked_missing_penalty_weight: float = 0.25

    # Rotated epipolar band dimensions.  With unit steps this is 5 x 75
    # candidates: 75 samples along the line and 5 across it.
    search_length_px: float = 75.0
    search_width_px: float = 5.0
    search_along_step_px: float = 1.0
    search_across_step_px: float = 1.0

    # Opt-in at library level; the main UI enables this explicitly.
    two_stage_search: bool = False
    coarse_along_step_px: float = 2.0
    coarse_side_rescue: bool = True
    fine_half_length_px: float = 10.0
    fine_width_px: float = 5.0
    fine_step_px: float = 1.0
    adaptive_max_expansions: int = 2
    adaptive_cell_increment_px: int = 15
    adaptive_points_increment: int = 2
    valley_min_relative_depth: float = 0.08
    valley_shoulder_distance_px: float = 10.0
    valley_level_fraction: float = 0.5
    valley_min_side_samples: int = 2
    valley_min_valid_fraction: float = 0.7
    valley_max_basin_ratio: float = 0.95

    # Robust group score.  keep_best_count, when set, overrides the ratio.
    keep_best_ratio: float = 0.75
    keep_best_count: Optional[int] = None
    max_group_score: Optional[float] = 500.0
    # Best / spatially distinct second-best Objective; None disables the gate.
    max_objective_score_ratio: Optional[float] = 0.95
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
    specular_check_support: bool = True


DEFAULT_CONFIG = RegionSIFTConfig()


class RegionSIFTError(RuntimeError):
    """Expected, user-facing failure in the Region-SIFT matching pipeline."""


def _validate_config(config: RegionSIFTConfig) -> None:
    from Algorithm.masked_sift_descriptor import validate_config
    try:
        validate_config(config)
    except ValueError as exc:
        raise RegionSIFTError(str(exc)) from exc
    from Algorithm.region_sift_search import validate_search_config
    validate_search_config(config)
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
    if (config.max_objective_score_ratio is not None
            and (not np.isfinite(config.max_objective_score_ratio)
                 or not 0.0 <= config.max_objective_score_ratio <= 1.0)):
        raise RegionSIFTError("max_objective_score_ratio must be in [0, 1] or None")
    if (config.max_group_score is not None
            and (not np.isfinite(config.max_group_score) or config.max_group_score < 0)):
        raise RegionSIFTError("max_group_score must be finite and nonnegative or None")
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
    exclusion_mask: Optional[np.ndarray] = None,
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

    excluded = None
    if exclusion_mask is not None:
        excluded = np.asarray(exclusion_mask) > 0
        if excluded.shape != gray.shape:
            raise RegionSIFTError('left exclusion mask shape differs from image')
        if excluded[center_y, center_x]:
            raise RegionSIFTError('匹配失敗: P overlaps specular pixels or lacks reflection-free support')

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
            if excluded is not None:
                clean = ~excluded[cy0:cy1, cx0:cx1].ravel()
                coords, values = coords[clean], values[clean]

            is_center_cell = cell_row == center_row and cell_col == center_col
            if is_center_cell and config.exclude_click_from_cell_points:
                keep = np.logical_or(coords[:, 0] != click_pixel[0],
                                     coords[:, 1] != click_pixel[1])
                coords, values = coords[keep], values[keep]

            if excluded is not None and len(coords) < config.points_per_cell:
                raise RegionSIFTError(
                    f'匹配失敗: cell ({cell_row + 1},{cell_col + 1}) has only '
                    f'{len(coords)} reflection-free pixels; needs {config.points_per_cell}')

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


def _left_context_roi(
    config: RegionSIFTConfig, left_roi: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Safe pyramid-crop box around the left anchors, or None for the full image.

    None (full image, the pre-optimization behavior) whenever cropping is not
    provably safe: the OpenCV backend (untouched by this crop), or
    frame_coordinate_quantization_px > 0, whose rounding step is not
    generally aligned with the crop's pixel origin -- cropping first could
    then pick a different scale/angle than the full image would.
    """
    if not config.use_masked_sift or config.frame_coordinate_quantization_px > 0:
        return None
    from Algorithm.masked_sift_descriptor import max_support_radius
    margin = max_support_radius(config) + int(config.masked_extra_margin_px) + 2
    rx0, ry0, rw, rh = left_roi
    img_h, img_w = image_shape[:2]
    return (max(0, rx0 - margin), max(0, ry0 - margin),
            min(img_w, rx0 + rw + margin), min(img_h, ry0 + rh + margin))


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


def _right_context_roi(
    config: RegionSIFTConfig, band: Dict[str, np.ndarray], point_offsets: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Safe pyramid-crop box for the masked right context, or None for the
    full warped-right image (see _left_context_roi for the None conditions).

    Must cover every position ANY round of the whole two-stage search could
    ever sample for this anchor layout, not just the current round: along
    and across offsets are always bounded by search_length_px/search_width_px
    regardless of stage or adaptive expansion (run_two_stage clips fine
    offsets back inside the original half-range and caps fine_across by
    search_width_px), so _search_band_polygon's corners -- built from those
    same two config values -- already bound every candidate center. Adding
    the anchors' fixed offsets from the click (Minkowski sum of two boxes)
    then bounds every one of the 28 sampled positions per candidate.
    """
    if not config.use_masked_sift or config.frame_coordinate_quantization_px > 0:
        return None
    from Algorithm.masked_sift_descriptor import max_support_radius
    margin = max_support_radius(config) + int(config.masked_extra_margin_px) + 2
    polygon = _search_band_polygon(band, config)
    offsets = np.asarray(point_offsets, np.float32).reshape(-1, 2)
    lo = polygon.min(axis=0) + offsets.min(axis=0) - margin
    hi = polygon.max(axis=0) + offsets.max(axis=0) + margin
    img_h, img_w = image_shape[:2]
    return (max(0, int(np.floor(lo[0]))), max(0, int(np.floor(lo[1]))),
            min(img_w, int(np.ceil(hi[0])) + 1), min(img_h, int(np.ceil(hi[1])) + 1))


def run_region_sift_matching(
    img_left_gray: np.ndarray,
    img_right_gray: np.ndarray,
    point_p: Sequence[float],
    cand: Dict[str, Any],
    K_L: np.ndarray,
    sift: Optional[Any] = None,
    config: RegionSIFTConfig = DEFAULT_CONFIG,
    left_cache: Optional[Dict[str, Any]] = None,
    *,
    reject_specular: bool = False,
    left_spec_mask: Optional[np.ndarray] = None,
    right_spec_mask: Optional[np.ndarray] = None,
    _session=None,
    _offsets=None,
    _candidate_cache=None,
) -> Dict[str, Any]:
    """Run Region-SIFT and return an original-right-image match plus debug data."""
    from Algorithm.region_sift_search import run_two_stage, frame_context, evaluate_positions
    if config.two_stage_search and _session is None:
        return run_two_stage(
            run_region_sift_matching, img_left_gray, img_right_gray, point_p, cand, K_L,
            sift=sift, config=config, left_cache=left_cache,
            reject_specular=reject_specular, left_spec_mask=left_spec_mask,
            right_spec_mask=right_spec_mask)
    started = time.perf_counter()
    timing_ms: Dict[str, float] = {}
    timing_stage = '初始化與快取檢查'
    timing_started = started

    def timing_next(stage):
        nonlocal timing_stage, timing_started
        now = time.perf_counter()
        if timing_stage is not None:
            timing_ms[timing_stage] = timing_ms.get(timing_stage, 0.0) + (now - timing_started) * 1000.0
        timing_stage, timing_started = stage, now

    base_result: Dict[str, Any] = {
        "m_pt": None,
        "method": "",
        "reject_reason": None,
        "region_debug": None,
        "timing_ms": timing_ms,
        "timing_counts": {'left_cache_hit': False, 'right_descriptor_batches': 0,
                          'right_descriptor_rows': 0},
        "candidate_funnel": {},
    }
    detail_profile = {}
    base_result['detail_profile'] = detail_profile
    detail_token = begin_detail(detail_profile)
    try:
        _validate_config(config)
        if config.use_masked_sift:
            # This backend always requires spatial masks, independently of the
            # legacy Reject SpecPts switch. Never silently run without a mask.
            reject_specular = True
        left_gray = np.asarray(img_left_gray)
        right_gray = np.asarray(img_right_gray)
        if left_gray.ndim != 2 or right_gray.ndim != 2:
            raise RegionSIFTError("Region-SIFT requires grayscale left/right images")
        base_result['timing_counts']['reject_specular'] = bool(reject_specular)
        base_result['timing_counts']['specular_check_support'] = bool(config.specular_check_support and not config.use_masked_sift)
        base_result['descriptor_backend'] = 'custom-masked' if config.use_masked_sift else 'opencv'
        if reject_specular:
            for label, mask, gray in (('left', left_spec_mask, left_gray),
                                      ('right', right_spec_mask, right_gray)):
                if mask is None or np.asarray(mask).shape != gray.shape:
                    raise RegionSIFTError(f'匹配失敗: {label} specular mask missing or shape mismatch')
                if not np.all(np.isfinite(mask)):
                    raise RegionSIFTError(f'匹配失敗: {label} specular mask is non-finite')
            left_spec_mask = (np.asarray(left_spec_mask) > 0).astype(np.uint8)
            right_spec_mask = (np.asarray(right_spec_mask) > 0).astype(np.uint8)
        else:
            left_spec_mask = right_spec_mask = None
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
            hashlib.blake2b(left_spec_mask.tobytes(), digest_size=16).digest()
            if left_spec_mask is not None else None,
        )
        left_masked = None
        cached = left_cache.get("region_sift") if left_cache is not None else None
        if (cached is not None and cached.get("key") == cache_key
                and "frames" in cached):
            timing_next('左圖快取讀取')
            base_result['timing_counts']['left_cache_hit'] = True
            left_points = cached["points"]
            point_metadata = [dict(item) for item in cached["metadata"]]
            left_descriptors = cached["descriptors"]
            left_masked = cached.get('masked_descriptors')
            left_frames = {
                key: np.asarray(value).copy()
                for key, value in cached["frames"].items()
            }
            left_roi = cached["roi"]
        else:
            left_context = None
            left_excluded = left_spec_mask
            left_frame_config = config
            if config.use_masked_sift:
                from Algorithm.masked_sift_descriptor import exclusion_mask
                left_excluded = exclusion_mask(left_spec_mask, config)
            if reject_specular and config.specular_check_support and not config.use_masked_sift:
                timing_next('左圖尺度金字塔與響應圖')
                left_context = frame_context(left_gray, config, _session, 'left')
                # Pre-filter sampling positions at the smallest possible support;
                # scale selection below checks every actual support independently.
                radius = int(np.ceil(min(entry['support'] for entry in left_context['entries'])
                                     + config.frame_coordinate_quantization_px)) + 1
                left_excluded = 1 - _eroded_valid_mask(1 - left_spec_mask, radius, 1.0)
                left_frame_config = replace(config, min_valid_warp_ratio=1.0)
            timing_next('左圖梯度與區域取點')
            left_points, point_metadata, _magnitude, left_roi = select_region_points(
                left_gray, point_p, config, exclusion_mask=left_excluded)
            timing_next('左圖尺度金字塔與響應圖')
            if left_context is None:
                left_context = frame_context(left_gray, config, _session, 'left', left_spec_mask,
                                             roi=_left_context_roi(config, left_roi, left_gray.shape))
            # The context may be a crop; its own outputs carry no coordinates,
            # so only the two calls below (which index into it) need points
            # shifted into its local frame. left_points itself stays absolute.
            left_origin = np.asarray(left_context.get('origin', (0, 0)), np.float32)
            local_left_points = left_points - left_origin
            _oy, _ox = int(left_origin[1]), int(left_origin[0])
            _ch, _cw = left_context['image_shape']
            left_context_image = left_gray[_oy:_oy + _ch, _ox:_ox + _cw]
            timing_next('左圖尺度選擇與角度估計')
            left_frames = estimate_dense_sift_frames(
                left_context_image, local_left_points, left_frame_config, context=left_context,
                valid_mask=(1 - left_spec_mask)
                if reject_specular and config.specular_check_support and not config.use_masked_sift else None)
            if not config.use_masked_sift and not np.all(left_frames['valid']):
                raise RegionSIFTError(
                    'left anchors lack complete reflection-free SIFT/scale-space support; '
                    'move the click inward or tune the support limit')
            timing_next('左圖SIFT descriptor')
            if config.use_masked_sift:
                from Algorithm.masked_sift_descriptor import compute_descriptors, group_validity
                left_masked = compute_descriptors(left_context, local_left_points,
                    left_frames['size_px'], left_frames['angle_deg'], config)
                left_masked['valid_points'] &= left_frames['valid']
                if not group_validity(left_masked['valid_points'][None], point_metadata, config)[0]:
                    raise RegionSIFTError('匹配失敗: 自製 SIFT 左圖有效點比例或每 cell 有效點不足')
                left_descriptors = left_masked['descriptors']
                left_frames['masked_valid_fraction'] = left_masked['valid_fraction']
                left_frames['masked_point_valid'] = left_masked['valid_points']
                left_frames['scale_reliable'] &= left_masked['valid_points']
                left_frames['orientation_reliable'] &= left_masked['valid_points']
            else:
                left_descriptors = compute_descriptors_at_points(
                    left_gray, left_points, sift, config,
                    sizes_px=left_frames["size_px"],
                    angles_deg=left_frames["angle_deg"],
                    octaves=left_frames['octave'])
            del left_context
            timing_next('左圖快取保存')
            if left_cache is not None:
                left_cache["region_sift"] = {
                    "key": cache_key,
                    "points": left_points,
                    "metadata": [dict(item) for item in point_metadata],
                    "descriptors": left_descriptors,
                    "masked_descriptors": left_masked,
                    "frames": {
                        key: np.asarray(value).copy()
                        for key, value in left_frames.items()
                    },
                    "roi": left_roi,
                }

        timing_next('右圖warp與有效遮罩')
        point_count = len(left_points)
        expected_count = config.grid_rows * config.grid_cols * config.points_per_cell + 1
        if left_descriptors.shape != (expected_count, 128):
            raise RegionSIFTError(
                f"left descriptor matrix must be {expected_count}x128, got {left_descriptors.shape}")

        H_lr = plane_homography_left_to_right(cand, K_L)
        H_rl = np.linalg.inv(H_lr)
        height, width = left_gray.shape[:2]
        if _session is not None and 'warp' in _session:
            warped_right, warped_valid, warped_specular, support_valid = _session['warp']
        else:
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
            warped_specular = None
            if reject_specular:
                source_specular = right_spec_mask
                if config.specular_check_support or config.use_masked_sift:
                    # Conservatively include the source interpolation footprint.
                    radius = {cv2.INTER_NEAREST: 0, cv2.INTER_LINEAR: 1,
                              cv2.INTER_CUBIC: 2, cv2.INTER_LANCZOS4: 4}.get(int(config.warp_interpolation), 4)
                    if radius:
                        source_specular = cv2.dilate(source_specular, np.ones((2 * radius + 1,) * 2, np.uint8))
                warped_specular = cv2.warpPerspective(
                    source_specular, H_rl, (width, height), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            support_valid = _eroded_valid_mask(
                warped_valid, int(config.descriptor_border_margin_px),
                float(config.min_valid_warp_ratio))
            if _session is not None:
                _session['warp'] = (warped_right, warped_valid, warped_specular, support_valid)
                _session['warp_builds'] = _session.get('warp_builds', 0) + 1

        timing_next('極線候選建立與初步篩選')
        band = build_epipolar_band(point_p, cand["F"], H_lr, config)
        if _offsets is not None:
            offsets = np.asarray(_offsets, dtype=np.float32).reshape(-1, 2)
            band['along_offsets'], band['across_offsets'] = offsets[:, 0], offsets[:, 1]
            band['centers_warp'] = (band['seed_on_line'][None, :]
                + offsets[:, :1] * band['tangent'][None, :]
                + offsets[:, 1:] * band['normal'][None, :]).astype(np.float32)
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
        funnel = base_result['candidate_funnel']
        funnel['requested_candidates'] = int(len(all_positions))
        funnel['in_bounds_candidates'] = int(np.count_nonzero(valid_candidates))
        in_bound_indices = np.flatnonzero(valid_candidates)
        for candidate_index in in_bound_indices:
            valid_candidates[candidate_index] = bool(np.all(
                support_valid[ys[candidate_index], xs[candidate_index]] > 0))
        funnel['warp_support_candidates'] = int(np.count_nonzero(valid_candidates))
        before_specular = int(np.count_nonzero(valid_candidates))
        if warped_specular is not None and not config.use_masked_sift:
            for candidate_index in np.flatnonzero(valid_candidates):
                valid_candidates[candidate_index] &= not np.any(
                    warped_specular[ys[candidate_index], xs[candidate_index]])
        base_result['timing_counts']['specular_center_rejected_candidates'] = (
            before_specular - int(np.count_nonzero(valid_candidates)))
        funnel['center_mask_candidates'] = int(np.count_nonzero(valid_candidates))
        funnel['initial_eligible_candidates'] = int(np.count_nonzero(valid_candidates))
        valid_indices = np.flatnonzero(valid_candidates)
        if len(valid_indices) == 0:
            raise RegionSIFTError("匹配失敗: no search candidate has valid warp support and reflection-free anchors")

        right_context_roi = _right_context_roi(config, band, point_offsets, warped_right.shape)
        (group_scores, objective_scores, trimmed_scores, balanced_scores,
         frame_penalties, candidate_distances, candidate_descriptors,
         right_frames, valid_row_by_candidate) = evaluate_positions(
            warped_right, all_positions, valid_indices, warped_valid,
            warped_specular, reject_specular, band, left_descriptors,
            point_metadata, left_frames, sift, config, _session, _candidate_cache,
            timing_next, base_result['timing_counts'], left_masked=left_masked,
            funnel=funnel, right_context_roi=right_context_roi)

        timing_next('最佳候選、座標回轉與Debug整理')
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
        distance_order = np.argsort(np.where(best_right_frames.get('masked_point_valid', np.ones(point_count, bool)),
                                             best_distances, np.inf), kind="mergesort")
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
                'masked_pair_valid': bool(best_right_frames.get('masked_point_valid', np.ones(point_count, bool))[index]),
                'distance_includes_missing_penalty': bool(config.use_masked_sift),
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

        second_debug = None
        if second_index is not None:
            second_row = valid_row_by_candidate[second_index]
            second_distances = candidate_distances[second_index].copy()
            second_keep = np.zeros(point_count, dtype=bool)
            second_valid = right_frames.get('masked_point_valid')
            second_sort = second_distances if second_valid is None else np.where(second_valid[second_row], second_distances, np.inf)
            second_keep[np.argsort(second_sort, kind='stable')[:keep_count]] = True
            second_debug = {
                'points_warp': all_positions[second_index].copy(),
                'center_right': _transform_points(centers[second_index].reshape(1, 2), H_lr)[0],
                'distances': second_distances,
                'keep_mask': second_keep,
                'frames': {key: np.asarray(value[second_row]).copy()
                           for key, value in right_frames.items()},
            }

        debug = {
            'second_candidate': second_debug,
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
            "descriptor_backend": base_result['descriptor_backend'],
            "left_masked_coverage": None if left_masked is None else left_masked['cell_valid_fraction'],
            "left_masked_packet": left_masked,
            'left_frames': left_frames,
            'right_frames': best_right_frames,
            "point_metadata": point_metadata,
            "H_lr": H_lr,
            "H_rl": H_rl,
            "warped_right_gray": warped_right,
            "warped_valid_mask": warped_valid,
            "warped_specular_mask": warped_specular,
            "reject_specular": bool(reject_specular),
            "specular_check_support": bool(config.specular_check_support and not config.use_masked_sift),
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
        informative_left = (left_descriptors[left_masked['valid_points']]
                            if left_masked is not None else left_descriptors)
        if (config.reject_uninformative_group
                and np.all(np.linalg.norm(informative_left, axis=1) <= 1e-12)):
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
                f"Region-SIFT 匹配失敗: BestG (GroupScore) {best_group_score:.3f} exceeds "
                f"{float(config.max_group_score):.3f}")
            base_result["region_debug"] = debug
            return base_result

        if (config.max_objective_score_ratio is not None
                and np.isfinite(objective_score_ratio)
                and objective_score_ratio > float(config.max_objective_score_ratio)):
            base_result["reject_reason"] = (
                f"Region-SIFT 匹配失敗: ObjRatio {objective_score_ratio:.6f} exceeds "
                f"{float(config.max_objective_score_ratio):.6f}; ambiguous match")
            base_result["region_debug"] = debug
            return base_result

        base_result.update({
            "m_pt": best_center_right.astype(np.float32),
            "method": f"Region-SIFT({keep_count}/{point_count})" + ('+MaskedSIFT' if config.use_masked_sift else ''),
            "region_debug": debug,
        })
        return base_result
    except (RegionSIFTError, cv2.error, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        base_result["reject_reason"] = str(exc)
        return base_result
    finally:
        # Also report work completed before a rejected/invalid match. Stage
        # durations are disjoint; repeated descriptor/scoring batches add up.
        timing_next(None)
        base_result['elapsed_ms'] = sum(timing_ms.values())
        if base_result.get('region_debug') is not None:
            base_result['region_debug']['timing_ms'] = dict(timing_ms)
            base_result['region_debug']['timing_counts'] = dict(base_result['timing_counts'])
            base_result['region_debug']['elapsed_ms'] = base_result['elapsed_ms']
            base_result['region_debug']['detail_profile'] = detail_profile
        end_detail(detail_token)


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
