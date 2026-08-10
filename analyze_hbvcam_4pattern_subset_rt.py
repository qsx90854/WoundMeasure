"""Analyze every non-empty subset of four ArUco markers in HBVCAM SBS videos.

For every frame, the script detects the four expected marker IDs once and then
estimates stereo left-to-right RT for all 15 non-empty subsets:

    C(4, 1) + C(4, 2) + C(4, 3) + C(4, 4) = 4 + 6 + 4 + 1 = 15

Each marker independently supplies IPPE pose branches in the left and right
camera.  One- and two-marker subsets retain exhaustive branch enumeration.
Three- and four-marker subsets use pattern-level RANSAC plus true held-out
leave-one-marker training hypotheses.  The surviving common RT is refined
directly against all bidirectional corner reprojections with robust,
uncertainty-aware weights.  Marker-to-marker 3D layout is therefore not
required.  An optional coplanar constraint can be enabled for experiments. JSON
extrinsics are never used to select a branch, inlier, or RT; they are used only
after estimation to report answer errors.

The output workbook contains per-frame subset results, per-video/per-distance
summaries, pattern-count summaries, raw corner coordinates, and corner
stability summaries.  One diagnostic MP4 is also produced per source video.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from analyze_hbvcam_aruco_corner_rt_stability import (
    ExcelFormula,
    excel_column,
    write_xlsx,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / "HBVCAM_4M2214HD-2-v11"
DEFAULT_CALIBRATION = ROOT / "calibration_result_HBVCAM_4M2214HD-2-v11.json"
DEFAULT_OUTPUT_NAME = "HBVCAM_4pattern_subset_rt.xlsx"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}

MARKER_SIZE_MM = 8.25
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
ROTATION_PASS_DEG = 2.0
BASELINE_PASS_PERCENT = 5.0
SUBPIX_MIN_HALF_WINDOW = 2
SUBPIX_MAX_HALF_WINDOW = 4
SUBPIX_STABILITY_DELTA = 1
SUBPIX_STABILITY_MAX_RAW_PX = 2.0
ARUCO_INITIAL_CORNER_REFINEMENT = "apriltag"
PATTERN_RANSAC_REPROJECTION_THRESHOLD_PX = 2.0
JOINT_REFINEMENT_ENABLED = True
JOINT_REFINEMENT_MAX_ITERATIONS = 10
JOINT_HUBER_DELTA_PX = 1.5
USE_TEMPORAL_RT_PRIOR = True
TEMPORAL_ROTATION_SCALE_DEG = 1.0
TEMPORAL_TRANSLATION_SCALE_MM = 2.0
ASSUME_COPLANAR_PATTERNS = False
COPLANAR_NORMAL_WEIGHT_PX = 1.0
COPLANAR_CENTER_WEIGHT_PX = 1.0
DIAGNOSTIC_TILE_SIZE = 240

SOLUTION_STRICT = "STRICT_ALL_INLIERS"
SOLUTION_FALLBACK = "ROBUST_FALLBACK"
SOLUTION_FAILED = "FAILED"


FRAME_FIELDS = [
    "distance_cm",
    "angle_deg",
    "video_file",
    "video_path",
    "frame_index",
    "marker_count",
    "marker_ids",
    "subset_key",
    "shared_expected_marker_count",
    "status",
    "failure_reason",
    "solution_class",
    "effective_inlier_count",
    "all_requested_patterns_used",
    "relative_rotation_angle_deg",
    "rotation_error_deg",
    "algorithm_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_delta_mm",
    "baseline_error_percent",
    "absolute_baseline_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "marker_transfer_left_rms_px",
    "marker_transfer_right_rms_px",
    "marker_transfer_combined_rms_px",
    "marker_transfer_max_px",
    "individual_rotation_spread_rms_deg",
    "individual_translation_spread_rms_mm",
    "individual_baseline_std_mm",
    "robust_method",
    "robust_required_inliers",
    "robust_inlier_count",
    "robust_inlier_marker_ids",
    "robust_outlier_count",
    "robust_outlier_marker_ids",
    "pattern_ransac_reprojection_threshold_px",
    "adaptive_gate_mean_px",
    "adaptive_gate_min_px",
    "adaptive_gate_max_px",
    "marker_uncertainty_px",
    "marker_weights",
    "robust_selected_hypothesis_source",
    "robust_hypotheses_evaluated",
    "leave_one_out_models_evaluated",
    "all_marker_transfer_rms_px",
    "pre_rejection_marker_transfer_combined_rms_px",
    "post_rejection_marker_transfer_combined_rms_px",
    "pre_rejection_algorithm_baseline_mm",
    "post_rejection_algorithm_baseline_mm",
    "baseline_change_after_rejection_mm",
    "all_selected_ippe_branches",
    "selected_ippe_branches",
    "evaluated_branch_combinations",
    "joint_refinement_applied",
    "joint_refinement_converged",
    "joint_refinement_iterations",
    "joint_initial_robust_cost",
    "joint_final_robust_cost",
    "joint_initial_transfer_rms_px",
    "joint_final_transfer_rms_px",
    "true_holdout_mean_rms_px",
    "true_holdout_p95_rms_px",
    "true_holdout_max_rms_px",
    "true_holdout_per_marker_rms_px",
    "temporal_prior_used",
    "temporal_prior_rotation_delta_deg",
    "temporal_prior_translation_delta_mm",
    "assume_coplanar_patterns",
    "coplanar_normal_spread_deg",
    "coplanar_center_rms_mm",
    "processing_time_ms",
]


CORNER_FIELDS = [
    "distance_cm",
    "angle_deg",
    "video_file",
    "video_path",
    "frame_index",
    "marker_id",
    "left_initial_detected",
    "left_accepted",
    "left_subpixel_half_window",
    "left_stability_max_raw_px",
    "left_subpixel_shift_rms_px",
    "left_corner_gradient_mean",
    "left_marker_laplacian_variance",
    "left_estimated_corner_uncertainty_px",
    "right_initial_detected",
    "right_accepted",
    "right_subpixel_half_window",
    "right_stability_max_raw_px",
    "right_subpixel_shift_rms_px",
    "right_corner_gradient_mean",
    "right_marker_laplacian_variance",
    "right_estimated_corner_uncertainty_px",
    "shared_accepted",
    *[
        f"{side}_c{corner}_{axis}_px"
        for side in ("left", "right")
        for corner in range(4)
        for axis in ("x", "y")
    ],
    "left_rect_width_px",
    "left_rect_height_px",
    "left_rect_area_px2",
    "left_quad_area_px2",
    "left_side_mean_px",
    "right_rect_width_px",
    "right_rect_height_px",
    "right_rect_area_px2",
    "right_quad_area_px2",
    "right_side_mean_px",
]


VIDEO_SUMMARY_FIELDS = [
    "distance_cm",
    "angle_deg",
    "video_file",
    "marker_count",
    "marker_ids",
    "subset_key",
    "requested_frames",
    "solved_frames",
    "solve_rate_percent",
    "rotation_pass_frames",
    "rotation_pass_rate_percent",
    "baseline_pass_frames",
    "baseline_pass_rate_percent",
    "both_pass_frames",
    "both_pass_rate_percent",
    "rotation_error_mean_deg",
    "rotation_error_median_deg",
    "rotation_error_std_deg",
    "rotation_error_p95_deg",
    "algorithm_baseline_mean_mm",
    "algorithm_baseline_std_mm",
    "baseline_delta_mean_mm",
    "absolute_baseline_delta_mean_mm",
    "absolute_baseline_delta_median_mm",
    "absolute_baseline_delta_std_mm",
    "absolute_baseline_delta_p95_mm",
    "absolute_baseline_error_mean_percent",
    "absolute_baseline_error_median_percent",
    "absolute_baseline_error_std_percent",
    "absolute_baseline_error_p95_percent",
    "marker_transfer_rms_mean_px",
    "marker_transfer_rms_p95_px",
    "individual_rotation_spread_mean_deg",
    "individual_translation_spread_mean_mm",
    "left_corner_coordinate_std_rms_px",
    "right_corner_coordinate_std_rms_px",
    "left_rect_width_std_mean_px",
    "right_rect_width_std_mean_px",
    "left_side_mean_std_mean_px",
    "right_side_mean_std_mean_px",
    "robust_inlier_count_mean",
    "robust_outlier_frame_rate_percent",
    "pre_rejection_marker_transfer_rms_mean_px",
    "post_rejection_marker_transfer_rms_mean_px",
    "joint_initial_transfer_rms_mean_px",
    "joint_final_transfer_rms_mean_px",
    "true_holdout_rms_mean_px",
    "temporal_prior_rotation_delta_mean_deg",
    "temporal_prior_translation_delta_mean_mm",
    "coplanar_normal_spread_mean_deg",
    "coplanar_center_rms_mean_mm",
    "strict_solved_frames",
    "strict_solve_rate_percent",
    "fallback_solved_frames",
    "fallback_solve_rate_percent",
    "strict_rotation_error_mean_deg",
    "strict_absolute_baseline_error_mean_percent",
    "fallback_rotation_error_mean_deg",
    "fallback_absolute_baseline_error_mean_percent",
]


DISTANCE_SUMMARY_FIELDS = [
    "distance_cm",
    "angle_deg",
    "video_count",
    *VIDEO_SUMMARY_FIELDS[3:34],
    "left_corner_coordinate_std_rms_px",
    "right_corner_coordinate_std_rms_px",
    "left_rect_width_std_mean_px",
    "right_rect_width_std_mean_px",
    "left_side_mean_std_mean_px",
    "right_side_mean_std_mean_px",
    "robust_inlier_count_mean",
    "robust_outlier_frame_rate_percent",
    "pre_rejection_marker_transfer_rms_mean_px",
    "post_rejection_marker_transfer_rms_mean_px",
    "joint_initial_transfer_rms_mean_px",
    "joint_final_transfer_rms_mean_px",
    "true_holdout_rms_mean_px",
    "temporal_prior_rotation_delta_mean_deg",
    "temporal_prior_translation_delta_mean_mm",
    "coplanar_normal_spread_mean_deg",
    "coplanar_center_rms_mean_mm",
    "strict_solved_frames",
    "strict_solve_rate_percent",
    "fallback_solved_frames",
    "fallback_solve_rate_percent",
    "strict_rotation_error_mean_deg",
    "strict_absolute_baseline_error_mean_percent",
    "fallback_rotation_error_mean_deg",
    "fallback_absolute_baseline_error_mean_percent",
]


PATTERN_COUNT_SUMMARY_FIELDS = [
    "distance_cm",
    "angle_deg",
    "marker_count",
    "subset_count",
    "requested_rows",
    "solved_rows",
    "solve_rate_percent",
    "rotation_error_mean_deg",
    "rotation_error_median_deg",
    "rotation_error_std_deg",
    "rotation_error_p95_deg",
    "absolute_baseline_delta_mean_mm",
    "absolute_baseline_delta_median_mm",
    "absolute_baseline_delta_std_mm",
    "absolute_baseline_delta_p95_mm",
    "absolute_baseline_error_mean_percent",
    "absolute_baseline_error_median_percent",
    "absolute_baseline_error_std_percent",
    "absolute_baseline_error_p95_percent",
    "rotation_pass_rows",
    "rotation_pass_rate_percent",
    "baseline_pass_rows",
    "baseline_pass_rate_percent",
    "both_pass_rows",
    "both_pass_rate_percent",
    "robust_inlier_count_mean",
    "robust_outlier_row_rate_percent",
    "pre_rejection_marker_transfer_rms_mean_px",
    "post_rejection_marker_transfer_rms_mean_px",
    "joint_initial_transfer_rms_mean_px",
    "joint_final_transfer_rms_mean_px",
    "true_holdout_rms_mean_px",
    "temporal_prior_rotation_delta_mean_deg",
    "temporal_prior_translation_delta_mean_mm",
    "coplanar_normal_spread_mean_deg",
    "coplanar_center_rms_mean_mm",
    "summary_population",
    "effective_inlier_count_mean",
]


CORNER_SUMMARY_FIELDS = [
    "distance_cm",
    "angle_deg",
    "video_file",
    "marker_id",
    "requested_frames",
    "left_detected_frames",
    "right_detected_frames",
    "shared_detected_frames",
    "shared_detection_rate_percent",
    *[
        f"{side}_c{corner}_{axis}_std_px"
        for side in ("left", "right")
        for corner in range(4)
        for axis in ("x", "y")
    ],
    "left_corner_coordinate_std_rms_px",
    "right_corner_coordinate_std_rms_px",
    "left_rect_width_mean_px",
    "left_rect_width_std_px",
    "left_rect_height_mean_px",
    "left_rect_height_std_px",
    "left_rect_area_mean_px2",
    "left_rect_area_std_px2",
    "left_quad_area_mean_px2",
    "left_quad_area_std_px2",
    "left_side_mean_mean_px",
    "left_side_mean_std_px",
    "right_rect_width_mean_px",
    "right_rect_width_std_px",
    "right_rect_height_mean_px",
    "right_rect_height_std_px",
    "right_rect_area_mean_px2",
    "right_rect_area_std_px2",
    "right_quad_area_mean_px2",
    "right_quad_area_std_px2",
    "right_side_mean_mean_px",
    "right_side_mean_std_px",
    *[
        f"{side}_{metric}_{stat}"
        for side in ("left", "right")
        for metric in (
            "subpixel_shift_rms_px",
            "corner_gradient_mean",
            "marker_laplacian_variance",
            "estimated_corner_uncertainty_px",
        )
        for stat in ("mean", "std")
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate HBVCAM stereo RT for all 15 non-empty subsets of four "
            "different ArUco IDs and report RT/baseline/corner stability."
        )
    )
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    parser.add_argument(
        "--marker-ids",
        help="Four comma-separated IDs. If omitted, discover the four most frequent shared IDs.",
    )
    parser.add_argument("--id-discovery-frames", type=int, default=30)
    parser.add_argument(
        "--subpixel-stability-max-raw-px",
        type=float,
        default=SUBPIX_STABILITY_MAX_RAW_PX,
        help=(
            "Reject a marker when neighboring cornerSubPix windows disagree "
            "by more than this many raw pixels. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--aruco-initial-refinement",
        choices=("none", "subpix", "contour", "apriltag"),
        default=ARUCO_INITIAL_CORNER_REFINEMENT,
        help=(
            "OpenCV ArUco initial corner refinement before the custom adaptive "
            "multi-window cornerSubPix stage. Default: apriltag."
        ),
    )
    parser.add_argument("--min-baseline-mm", type=float, default=MIN_BASELINE_MM)
    parser.add_argument("--max-baseline-mm", type=float, default=MAX_BASELINE_MM)
    parser.add_argument(
        "--pattern-ransac-reproj-threshold-px",
        type=float,
        default=PATTERN_RANSAC_REPROJECTION_THRESHOLD_PX,
        help=(
            "A marker is a robust RT inlier when its best bidirectional transfer "
            "RMS is at most an uncertainty-adapted gate based on this nominal "
            "value. Default: 2.0 px."
        ),
    )
    parser.add_argument(
        "--joint-refinement",
        action=argparse.BooleanOptionalAction,
        default=JOINT_REFINEMENT_ENABLED,
        help="Enable robust joint bidirectional RT reprojection refinement.",
    )
    parser.add_argument(
        "--joint-max-iterations",
        type=int,
        default=JOINT_REFINEMENT_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--joint-huber-delta-px",
        type=float,
        default=JOINT_HUBER_DELTA_PX,
    )
    parser.add_argument(
        "--temporal-prior",
        action=argparse.BooleanOptionalAction,
        default=USE_TEMPORAL_RT_PRIOR,
        help=(
            "Use the previous solved frame of the same subset as an additional "
            "branch/RANSAC hypothesis. Appropriate for a rigid stereo camera."
        ),
    )
    parser.add_argument(
        "--assume-coplanar",
        action=argparse.BooleanOptionalAction,
        default=ASSUME_COPLANAR_PATTERNS,
        help=(
            "Experimental: constrain marker normals/centres to one plane. No "
            "known marker-to-marker distances are required."
        ),
    )
    parser.add_argument(
        "--coplanar-normal-weight-px",
        type=float,
        default=COPLANAR_NORMAL_WEIGHT_PX,
    )
    parser.add_argument(
        "--coplanar-center-weight-px",
        type=float,
        default=COPLANAR_CENTER_WEIGHT_PX,
    )
    parser.add_argument("--rotation-pass-deg", type=float, default=ROTATION_PASS_DEG)
    parser.add_argument(
        "--baseline-pass-percent", type=float, default=BASELINE_PASS_PERCENT
    )
    parser.add_argument(
        "--distance-map",
        help=(
            "Optional comma-separated filename/stem=distance overrides, e.g. "
            "video_20260805_160512=35"
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--only-video", help="One exact filename or file stem")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output",
        help=f"Output .xlsx path. Default: <folder>/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--trend-angle-deg",
        type=float,
        help=(
            "Initial angle filter used by the Trend Charts sheet. If omitted, "
            "prefer 0 degrees when available, otherwise use the first angle."
        ),
    )
    parser.add_argument("--no-corner-videos", action="store_true")
    parser.add_argument(
        "--corner-video-dir",
        help="Default: <output stem>_corner_videos",
    )
    parser.add_argument("--corner-padding-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")
    if args.id_discovery_frames <= 0:
        parser.error("--id-discovery-frames must be positive")
    if args.subpixel_stability_max_raw_px <= 0:
        parser.error("--subpixel-stability-max-raw-px must be positive")
    if args.min_baseline_mm <= 0 or args.max_baseline_mm <= args.min_baseline_mm:
        parser.error("baseline limits are invalid")
    if args.pattern_ransac_reproj_threshold_px <= 0:
        parser.error("--pattern-ransac-reproj-threshold-px must be positive")
    if args.joint_max_iterations <= 0:
        parser.error("--joint-max-iterations must be positive")
    if args.joint_huber_delta_px <= 0:
        parser.error("--joint-huber-delta-px must be positive")
    if args.coplanar_normal_weight_px < 0 or args.coplanar_center_weight_px < 0:
        parser.error("coplanar weights cannot be negative")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    if args.corner_padding_ratio < 0:
        parser.error("--corner-padding-ratio cannot be negative")
    return args


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values) -> float | None:
    clean = [finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    return float(np.mean(clean)) if clean else None


def std(values) -> float | None:
    clean = [finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    return float(np.std(clean, ddof=1)) if len(clean) >= 2 else None


def median(values) -> float | None:
    clean = [finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    return float(np.median(clean)) if clean else None


def p95(values) -> float | None:
    clean = [finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    return float(np.percentile(clean, 95)) if clean else None


def parse_marker_ids(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    try:
        marker_ids = tuple(sorted({int(part.strip()) for part in text.split(",")}))
    except ValueError as exc:
        raise ValueError("--marker-ids must contain integers separated by commas") from exc
    if len(marker_ids) != 4:
        raise ValueError("--marker-ids must contain exactly four different IDs")
    return marker_ids


def parse_distance_map(text: str | None) -> dict[str, float]:
    result = {}
    if not text:
        return result
    for entry in text.split(","):
        if "=" not in entry:
            raise ValueError("Each --distance-map entry must be name=distance")
        name, value = entry.split("=", 1)
        result[name.strip().lower()] = float(value)
    return result


def metadata_from_video(path: Path, distance_map: dict[str, float]) -> dict:
    distance = None
    for key in (path.name.lower(), path.stem.lower()):
        if key in distance_map:
            distance = distance_map[key]
            break
    if distance is None:
        match = re.search(r"(?P<distance>\d+(?:\.\d+)?)\s*cm", path.stem, re.I)
        if match:
            distance = float(match.group("distance"))
    angle = 0.0
    angle_match = re.search(r"angle[_-]?(?P<angle>-?\d+(?:\.\d+)?)", path.stem, re.I)
    if angle_match:
        angle = float(angle_match.group("angle"))
    return {
        "path": path.resolve(),
        "distance_cm": distance,
        "angle_deg": angle,
    }


def collect_videos(
    folder: Path,
    recursive: bool,
    only_video: str | None,
    distance_map: dict[str, float],
) -> list[dict]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    requested = str(only_video or "").lower()
    videos = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "_corner_videos" in {part.lower() for part in path.parts}:
            continue
        if path.stem.lower().endswith("_4pattern_corners"):
            continue
        if requested and requested not in {path.name.lower(), path.stem.lower()}:
            continue
        videos.append(metadata_from_video(path, distance_map))
    videos.sort(
        key=lambda item: (
            float("inf") if item["distance_cm"] is None else item["distance_cm"],
            item["path"].name.lower(),
        )
    )
    return videos


def load_calibration(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    left = data["intrinsic_L"]
    right = data["intrinsic_R"]
    extrinsic = data["extrinsic"]
    return (
        np.asarray(left["matrix"], dtype=np.float64).reshape(3, 3),
        np.asarray(left["distortion"], dtype=np.float64).reshape(-1),
        np.asarray(right["matrix"], dtype=np.float64).reshape(3, 3),
        np.asarray(right["distortion"], dtype=np.float64).reshape(-1),
        {
            "R": np.asarray(extrinsic["R"], dtype=np.float64).reshape(3, 3),
            "T": np.asarray(extrinsic["T"], dtype=np.float64).reshape(3, 1),
        },
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def average_rotations_svd(rotations: list[np.ndarray]) -> np.ndarray:
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _singular, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def calculate_answer_errors(R_est, t_est, answer) -> dict:
    R_est = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3, 1)
    R_answer = answer["R"]
    t_answer = answer["T"]
    rotation_error = rotation_angle_deg(R_est @ R_answer.T)
    baseline = float(np.linalg.norm(t_est))
    answer_baseline = float(np.linalg.norm(t_answer))
    baseline_delta = baseline - answer_baseline
    direction_cos = np.clip(
        float((t_est.T @ t_answer)[0, 0]) / max(baseline * answer_baseline, 1e-12),
        -1.0,
        1.0,
    )
    return {
        "relative_rotation_angle_deg": rotation_angle_deg(R_est),
        "rotation_error_deg": rotation_error,
        "algorithm_baseline_mm": baseline,
        "json_baseline_mm": answer_baseline,
        "baseline_delta_mm": baseline_delta,
        "absolute_baseline_delta_mm": abs(baseline_delta),
        "baseline_error_percent": baseline_delta / answer_baseline * 100.0,
        "absolute_baseline_error_percent": abs(baseline_delta) / answer_baseline * 100.0,
        "translation_l2_error_mm": float(np.linalg.norm(t_est - t_answer)),
        "translation_direction_error_deg": float(np.degrees(np.arccos(direction_cos))),
    }


def marker_geometry(corners: np.ndarray) -> dict:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    width = float(np.ptp(points[:, 0]))
    height = float(np.ptp(points[:, 1]))
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return {
        "rect_width_px": width,
        "rect_height_px": height,
        "rect_area_px2": width * height,
        "quad_area_px2": abs(float(cv2.contourArea(points.astype(np.float32)))),
        "side_mean_px": float(np.mean(sides)),
    }


class AdaptiveArucoDetector:
    def __init__(self, stability_max_raw_px: float, initial_refinement="apriltag"):
        self.stability_max_raw_px = float(stability_max_raw_px)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        refinement_methods = {
            "none": cv2.aruco.CORNER_REFINE_NONE,
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
            "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
            "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
        }
        self.initial_refinement = str(initial_refinement).lower()
        self.parameters.cornerRefinementMethod = refinement_methods[
            self.initial_refinement
        ]
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _corner_quality(gray: np.ndarray, points: np.ndarray) -> dict:
        points = np.asarray(points, dtype=np.float64).reshape(4, 2)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        gradients = []
        height, width = gray.shape[:2]
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            x0, x1 = max(0, xi - 2), min(width, xi + 3)
            y0, y1 = max(0, yi - 2), min(height, yi + 3)
            if x1 > x0 and y1 > y0:
                gradients.append(float(np.mean(magnitude[y0:y1, x0:x1])))
        margin = 3
        x0 = max(0, int(math.floor(np.min(points[:, 0]))) - margin)
        x1 = min(width, int(math.ceil(np.max(points[:, 0]))) + margin + 1)
        y0 = max(0, int(math.floor(np.min(points[:, 1]))) - margin)
        y1 = min(height, int(math.ceil(np.max(points[:, 1]))) + margin + 1)
        patch = gray[y0:y1, x0:x1]
        laplacian_variance = (
            float(cv2.Laplacian(patch, cv2.CV_32F).var())
            if patch.size >= 25
            else 0.0
        )
        return {
            "corner_gradient_mean": float(np.mean(gradients)) if gradients else 0.0,
            "marker_laplacian_variance": laplacian_variance,
        }

    def _refine(self, gray: np.ndarray, initial_corners: np.ndarray):
        initial = np.asarray(initial_corners, dtype=np.float32).reshape(4, 1, 2)
        points = initial.reshape(4, 2)
        side_px = float(
            np.mean(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))
        )
        half_window = int(
            np.clip(
                round(side_px / 12.0),
                SUBPIX_MIN_HALF_WINDOW,
                SUBPIX_MAX_HALF_WINDOW,
            )
        )
        windows = sorted(
            {
                max(1, half_window - SUBPIX_STABILITY_DELTA),
                half_window,
                min(
                    SUBPIX_MAX_HALF_WINDOW + SUBPIX_STABILITY_DELTA,
                    half_window + SUBPIX_STABILITY_DELTA,
                ),
            }
        )
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        solutions = {}
        for window in windows:
            refined = initial.copy()
            try:
                cv2.cornerSubPix(gray, refined, (window, window), (-1, -1), term)
            except cv2.error:
                continue
            if np.all(np.isfinite(refined)):
                solutions[window] = refined.reshape(4, 2).astype(np.float64)
        diagnostics = {
            "initial": points.astype(np.float64),
            "display": solutions.get(half_window, points.astype(np.float64)),
            "half_window": half_window,
            "stability_max_raw_px": None,
            "accepted": False,
            "subpixel_shift_rms_px": None,
            "corner_gradient_mean": None,
            "marker_laplacian_variance": None,
            "estimated_corner_uncertainty_px": None,
        }
        if half_window not in solutions or len(solutions) < 2:
            return None, diagnostics
        maximum_disagreement = 0.0
        values = list(solutions.values())
        for first, second in itertools.combinations(values, 2):
            maximum_disagreement = max(
                maximum_disagreement,
                float(np.max(np.linalg.norm(first - second, axis=1))),
            )
        diagnostics["stability_max_raw_px"] = maximum_disagreement
        display = solutions[half_window]
        diagnostics["subpixel_shift_rms_px"] = float(
            np.sqrt(np.mean(np.sum((display - points) ** 2, axis=1)))
        )
        diagnostics.update(self._corner_quality(gray, display))
        side_penalty = float(np.clip(30.0 / max(side_px, 1.0), 0.0, 2.0))
        gradient_penalty = float(
            np.clip(80.0 / max(diagnostics["corner_gradient_mean"], 1.0), 0.0, 2.0)
        )
        diagnostics["estimated_corner_uncertainty_px"] = float(
            np.clip(
                0.15
                + 0.45 * maximum_disagreement
                + 0.15 * diagnostics["subpixel_shift_rms_px"]
                + 0.10 * side_penalty
                + 0.10 * gradient_penalty,
                0.20,
                3.00,
            )
        )
        if maximum_disagreement > self.stability_max_raw_px:
            return None, diagnostics
        diagnostics["accepted"] = True
        return solutions[half_window], diagnostics

    def detect(self, frame: np.ndarray):
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_gray = self.clahe.apply(original_gray)
        if self.detector is not None:
            corners, ids, _rejected = self.detector.detectMarkers(detection_gray)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                detection_gray, self.dictionary, parameters=self.parameters
            )
        accepted = {}
        diagnostics = {}
        if ids is None:
            return accepted, diagnostics
        for initial, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            refined, marker_diagnostics = self._refine(original_gray, initial)
            diagnostics[marker_id] = marker_diagnostics
            if refined is not None:
                accepted[marker_id] = refined
        return accepted, diagnostics


class MultiMarkerRTEstimator:
    def __init__(
        self,
        k_left,
        d_left,
        k_right,
        d_right,
        marker_size_mm,
        min_baseline_mm,
        max_baseline_mm,
        ransac_reprojection_threshold_px,
        joint_refinement=True,
        joint_max_iterations=10,
        joint_huber_delta_px=1.5,
        assume_coplanar=False,
        coplanar_normal_weight_px=1.0,
        coplanar_center_weight_px=1.0,
    ):
        self.k_left = np.asarray(k_left, dtype=np.float64)
        self.d_left = np.asarray(d_left, dtype=np.float64)
        self.k_right = np.asarray(k_right, dtype=np.float64)
        self.d_right = np.asarray(d_right, dtype=np.float64)
        self.min_baseline_mm = float(min_baseline_mm)
        self.max_baseline_mm = float(max_baseline_mm)
        self.ransac_reprojection_threshold_px = float(
            ransac_reprojection_threshold_px
        )
        self.joint_refinement = bool(joint_refinement)
        self.joint_max_iterations = int(joint_max_iterations)
        self.joint_huber_delta_px = float(joint_huber_delta_px)
        self.assume_coplanar = bool(assume_coplanar)
        self.coplanar_normal_weight_px = float(coplanar_normal_weight_px)
        self.coplanar_center_weight_px = float(coplanar_center_weight_px)
        self._frame_candidate_cache = {}
        self._frame_training_cache = {}
        half = float(marker_size_mm) / 2.0
        self.object_points = np.asarray(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    def begin_frame(self):
        """Clear caches whose inputs are the current left/right image pair."""
        self._frame_candidate_cache.clear()
        self._frame_training_cache.clear()

    @staticmethod
    def marker_uncertainty(marker_id, marker_quality) -> float:
        if not marker_quality:
            return 0.5
        record = marker_quality.get(int(marker_id), {})
        value = finite(record.get("uncertainty_px"))
        return float(np.clip(value if value is not None else 0.5, 0.20, 3.00))

    def marker_weight(self, marker_id, marker_quality) -> float:
        sigma = self.marker_uncertainty(marker_id, marker_quality)
        return float(1.0 / max(sigma * sigma, 0.04))

    def adaptive_gate(self, marker_id, marker_quality) -> float:
        sigma = self.marker_uncertainty(marker_id, marker_quality)
        return float(
            np.clip(
                self.ransac_reprojection_threshold_px + 6.0 * sigma,
                self.ransac_reprojection_threshold_px,
                3.0 * self.ransac_reprojection_threshold_px,
            )
        )

    def _coplanar_metrics(self, selected_candidates, rotation, translation):
        if len(selected_candidates) < 2:
            return 0.0, 0.0
        normals = []
        centres = []
        for candidate in selected_candidates:
            left_rotation = candidate["left"]["R"]
            left_translation = candidate["left"]["t"].reshape(3)
            right_rotation = candidate["right"]["R"]
            right_translation = candidate["right"]["t"].reshape(3, 1)
            normals.append(left_rotation[:, 2])
            centres.append(left_translation)
            normals.append(rotation.T @ right_rotation[:, 2])
            centres.append((rotation.T @ (right_translation - translation)).reshape(3))
        reference = np.asarray(normals[0], dtype=np.float64)
        aligned = []
        for normal in normals:
            normal = np.asarray(normal, dtype=np.float64)
            if float(normal @ reference) < 0.0:
                normal = -normal
            aligned.append(normal / max(np.linalg.norm(normal), 1e-12))
        mean_normal = np.sum(aligned, axis=0)
        mean_normal /= max(np.linalg.norm(mean_normal), 1e-12)
        angles = [
            math.degrees(
                math.acos(float(np.clip(normal @ mean_normal, -1.0, 1.0)))
            )
            for normal in aligned
        ]
        centres_array = np.asarray(centres, dtype=np.float64)
        centre_mean = centres_array.mean(axis=0)
        plane_distances = (centres_array - centre_mean) @ mean_normal
        return (
            float(np.sqrt(np.mean(np.square(angles)))),
            float(np.sqrt(np.mean(np.square(plane_distances)))),
        )

    def pose_branches(self, corners, camera_matrix, distortion) -> list[dict]:
        image_points = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2)
        try:
            _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
                self.object_points.astype(np.float32),
                image_points,
                np.asarray(camera_matrix, dtype=np.float64),
                np.asarray(distortion, dtype=np.float64),
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return []
        branches = []
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            camera_points = (
                rotation @ self.object_points.T + translation
            ).T
            if np.any(camera_points[:, 2] <= 0.0):
                continue
            projected, _ = cv2.projectPoints(
                self.object_points,
                rvec,
                translation,
                camera_matrix,
                distortion,
            )
            residuals = np.linalg.norm(
                projected.reshape(4, 2) - image_points.reshape(4, 2), axis=1
            )
            branches.append(
                {
                    "index": int(index),
                    "R": rotation,
                    "t": translation,
                    "self_rms": float(np.sqrt(np.mean(residuals**2))),
                }
            )
        return branches

    def marker_candidates(self, left_corners, right_corners) -> list[dict]:
        left_branches = self.pose_branches(
            left_corners, self.k_left, self.d_left
        )
        right_branches = self.pose_branches(
            right_corners, self.k_right, self.d_right
        )
        candidates = []
        for left in left_branches:
            for right in right_branches:
                rotation = right["R"] @ left["R"].T
                translation = right["t"] - rotation @ left["t"]
                baseline = float(np.linalg.norm(translation))
                if not self.min_baseline_mm <= baseline <= self.max_baseline_mm:
                    continue
                candidates.append(
                    {
                        "left": left,
                        "right": right,
                        "R": rotation,
                        "t": translation,
                        "baseline": baseline,
                        "self_rms": float(
                            math.sqrt(
                                0.5
                                * (left["self_rms"] ** 2 + right["self_rms"] ** 2)
                            )
                        ),
                    }
                )
        return candidates

    @staticmethod
    def _project_camera_points(points_camera, camera_matrix, distortion):
        points_camera = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            points_camera,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            distortion,
        )
        return projected.reshape(-1, 2)

    def evaluate_combination(
        self,
        marker_ids,
        selected_candidates,
        left_markers,
        right_markers,
        rotation_override=None,
        translation_override=None,
        marker_quality=None,
    ) -> dict | None:
        rotations = [candidate["R"] for candidate in selected_candidates]
        translations = [candidate["t"] for candidate in selected_candidates]
        rotation = (
            average_rotations_svd(rotations)
            if rotation_override is None
            else np.asarray(rotation_override, dtype=np.float64).reshape(3, 3)
        )
        translation = (
            np.mean(np.stack(translations, axis=0), axis=0)
            if translation_override is None
            else np.asarray(translation_override, dtype=np.float64).reshape(3, 1)
        )
        baseline = float(np.linalg.norm(translation))
        if not self.min_baseline_mm <= baseline <= self.max_baseline_mm:
            return None

        left_residuals = []
        right_residuals = []
        per_marker_rms = {}
        for marker_id, candidate in zip(marker_ids, selected_candidates):
            points_left = (
                candidate["left"]["R"] @ self.object_points.T
                + candidate["left"]["t"]
            ).T
            predicted_right = (rotation @ points_left.T + translation).T
            if np.any(predicted_right[:, 2] <= 0.0):
                return None
            projected_right = self._project_camera_points(
                predicted_right, self.k_right, self.d_right
            )
            marker_right_residuals = np.linalg.norm(
                projected_right - np.asarray(right_markers[marker_id]).reshape(4, 2),
                axis=1,
            )
            right_residuals.extend(marker_right_residuals.tolist())

            points_right = (
                candidate["right"]["R"] @ self.object_points.T
                + candidate["right"]["t"]
            ).T
            predicted_left = (rotation.T @ (points_right.T - translation)).T
            if np.any(predicted_left[:, 2] <= 0.0):
                return None
            projected_left = self._project_camera_points(
                predicted_left, self.k_left, self.d_left
            )
            marker_left_residuals = np.linalg.norm(
                projected_left - np.asarray(left_markers[marker_id]).reshape(4, 2),
                axis=1,
            )
            left_residuals.extend(marker_left_residuals.tolist())
            per_marker_rms[int(marker_id)] = float(
                np.sqrt(
                    np.mean(
                        np.concatenate(
                            (marker_left_residuals, marker_right_residuals)
                        )
                        ** 2
                    )
                )
            )

        left_values = np.asarray(left_residuals, dtype=np.float64)
        right_values = np.asarray(right_residuals, dtype=np.float64)
        all_values = np.concatenate((left_values, right_values))
        rotation_spread = float(
            math.sqrt(
                np.mean(
                    [rotation_angle_deg(item @ rotation.T) ** 2 for item in rotations]
                )
            )
        )
        translation_spread = float(
            math.sqrt(
                np.mean(
                    [float(np.linalg.norm(item - translation)) ** 2 for item in translations]
                )
            )
        )
        baseline_std = (
            float(np.std([item["baseline"] for item in selected_candidates], ddof=1))
            if len(selected_candidates) >= 2
            else 0.0
        )
        coplanar_normal_spread, coplanar_center_rms = self._coplanar_metrics(
            selected_candidates, rotation, translation
        )
        uncertainties = {
            int(marker_id): self.marker_uncertainty(marker_id, marker_quality)
            for marker_id in marker_ids
        }
        weights = {
            int(marker_id): self.marker_weight(marker_id, marker_quality)
            for marker_id in marker_ids
        }
        gates = {
            int(marker_id): self.adaptive_gate(marker_id, marker_quality)
            for marker_id in marker_ids
        }
        return {
            "R_rel": rotation,
            "t_rel": translation,
            "marker_transfer_left_rms_px": float(
                np.sqrt(np.mean(left_values**2))
            ),
            "marker_transfer_right_rms_px": float(
                np.sqrt(np.mean(right_values**2))
            ),
            "marker_transfer_combined_rms_px": float(
                np.sqrt(np.mean(all_values**2))
            ),
            "marker_transfer_max_px": float(np.max(all_values)),
            "individual_rotation_spread_rms_deg": rotation_spread,
            "individual_translation_spread_rms_mm": translation_spread,
            "individual_baseline_std_mm": baseline_std,
            "per_marker_transfer_rms_px": per_marker_rms,
            "per_marker_uncertainty_px": uncertainties,
            "per_marker_weight": weights,
            "per_marker_adaptive_gate_px": gates,
            "adaptive_gate_mean_px": mean(gates.values()),
            "adaptive_gate_min_px": min(gates.values()),
            "adaptive_gate_max_px": max(gates.values()),
            "marker_uncertainty_px": ";".join(
                f"{marker_id}:{uncertainties[int(marker_id)]:.6f}"
                for marker_id in marker_ids
            ),
            "marker_weights": ";".join(
                f"{marker_id}:{weights[int(marker_id)]:.6f}"
                for marker_id in marker_ids
            ),
            "assume_coplanar_patterns": self.assume_coplanar,
            "coplanar_normal_spread_deg": coplanar_normal_spread,
            "coplanar_center_rms_mm": coplanar_center_rms,
            "selected_ippe_branches": ";".join(
                f"{marker_id}:L{candidate['left']['index']}-R{candidate['right']['index']}"
                for marker_id, candidate in zip(marker_ids, selected_candidates)
            ),
            "self_rms": mean(candidate["self_rms"] for candidate in selected_candidates),
        }

    def _joint_residual_vector(
        self,
        marker_ids,
        selected_candidates,
        rotation,
        translation,
        left_markers,
        right_markers,
        marker_quality,
    ) -> np.ndarray:
        residuals = []
        expected_size = 16 * len(marker_ids) + (
            2 if self.assume_coplanar and len(selected_candidates) >= 2 else 0
        )
        for marker_id, candidate in zip(marker_ids, selected_candidates):
            scale = math.sqrt(self.marker_weight(marker_id, marker_quality))
            points_left = (
                candidate["left"]["R"] @ self.object_points.T
                + candidate["left"]["t"]
            ).T
            predicted_right = (rotation @ points_left.T + translation).T
            if np.any(predicted_right[:, 2] <= 0.0):
                return np.full(expected_size, 1e6, dtype=np.float64)
            projected_right = self._project_camera_points(
                predicted_right, self.k_right, self.d_right
            )
            residuals.extend(
                ((projected_right - np.asarray(right_markers[marker_id]).reshape(4, 2)) * scale)
                .reshape(-1)
                .tolist()
            )
            points_right = (
                candidate["right"]["R"] @ self.object_points.T
                + candidate["right"]["t"]
            ).T
            predicted_left = (rotation.T @ (points_right.T - translation)).T
            if np.any(predicted_left[:, 2] <= 0.0):
                return np.full(expected_size, 1e6, dtype=np.float64)
            projected_left = self._project_camera_points(
                predicted_left, self.k_left, self.d_left
            )
            residuals.extend(
                ((projected_left - np.asarray(left_markers[marker_id]).reshape(4, 2)) * scale)
                .reshape(-1)
                .tolist()
            )
        if self.assume_coplanar and len(selected_candidates) >= 2:
            normal_spread, centre_rms = self._coplanar_metrics(
                selected_candidates, rotation, translation
            )
            residuals.append(
                normal_spread / 5.0 * self.coplanar_normal_weight_px
            )
            residuals.append(
                centre_rms / max(float(np.ptp(self.object_points[:, 0])), 1e-9)
                * self.coplanar_center_weight_px
            )
        return np.asarray(residuals, dtype=np.float64)

    def _huber_cost_and_sqrt_weights(self, residuals: np.ndarray):
        residuals = np.asarray(residuals, dtype=np.float64)
        absolute = np.abs(residuals)
        delta = self.joint_huber_delta_px
        quadratic = absolute <= delta
        cost = np.where(
            quadratic,
            0.5 * residuals * residuals,
            delta * (absolute - 0.5 * delta),
        )
        sqrt_weights = np.ones_like(residuals)
        mask = ~quadratic
        sqrt_weights[mask] = np.sqrt(delta / np.maximum(absolute[mask], 1e-12))
        return float(np.sum(cost)), sqrt_weights

    def joint_refine_rt(
        self,
        marker_ids,
        selected_candidates,
        initial_rotation,
        initial_translation,
        left_markers,
        right_markers,
        marker_quality,
    ) -> dict:
        initial_rotation = np.asarray(initial_rotation, dtype=np.float64).reshape(3, 3)
        initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(3, 1)
        initial_rvec, _ = cv2.Rodrigues(initial_rotation)
        parameters = np.concatenate((initial_rvec.reshape(3), initial_translation.reshape(3)))

        def evaluate_vector(values):
            rotation, _ = cv2.Rodrigues(values[:3].reshape(3, 1))
            translation = values[3:].reshape(3, 1)
            return self._joint_residual_vector(
                marker_ids,
                selected_candidates,
                rotation,
                translation,
                left_markers,
                right_markers,
                marker_quality,
            )

        residuals = evaluate_vector(parameters)
        initial_cost, _ = self._huber_cost_and_sqrt_weights(residuals)
        current_cost = initial_cost
        damping = 1e-3
        converged = False
        iterations = 0
        for iteration in range(self.joint_max_iterations):
            iterations = iteration + 1
            _, sqrt_weights = self._huber_cost_and_sqrt_weights(residuals)
            jacobian = np.empty((residuals.size, 6), dtype=np.float64)
            for column in range(6):
                step = 1e-6 if column < 3 else 1e-3
                shifted = parameters.copy()
                shifted[column] += step
                jacobian[:, column] = (evaluate_vector(shifted) - residuals) / step
            weighted_jacobian = jacobian * sqrt_weights[:, None]
            weighted_residuals = residuals * sqrt_weights
            normal = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_residuals
            diagonal = np.maximum(np.diag(normal), 1e-9)
            try:
                delta = -np.linalg.solve(
                    normal + damping * np.diag(diagonal), gradient
                )
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            trial = parameters + delta
            trial_residuals = evaluate_vector(trial)
            trial_cost, _ = self._huber_cost_and_sqrt_weights(trial_residuals)
            if trial_cost < current_cost:
                parameters = trial
                residuals = trial_residuals
                improvement = current_cost - trial_cost
                current_cost = trial_cost
                damping = max(damping / 3.0, 1e-9)
                if np.linalg.norm(delta[:3]) < 1e-8 and np.linalg.norm(delta[3:]) < 1e-5:
                    converged = True
                    break
                if improvement < 1e-8:
                    converged = True
                    break
            else:
                damping = min(damping * 10.0, 1e9)
        rotation, _ = cv2.Rodrigues(parameters[:3].reshape(3, 1))
        translation = parameters[3:].reshape(3, 1)
        if not self.min_baseline_mm <= float(np.linalg.norm(translation)) <= self.max_baseline_mm:
            rotation = initial_rotation
            translation = initial_translation
            current_cost = initial_cost
            converged = False
        return {
            "R": rotation,
            "t": translation,
            "initial_cost": initial_cost,
            "final_cost": current_cost,
            "iterations": iterations,
            "converged": converged or current_cost < initial_cost,
        }

    def apply_joint_refinement(
        self,
        result,
        marker_ids,
        selected_candidates,
        left_markers,
        right_markers,
        marker_quality,
    ):
        before_rms = result["marker_transfer_combined_rms_px"]
        if not self.joint_refinement:
            result.update(
                {
                    "joint_refinement_applied": False,
                    "joint_refinement_converged": False,
                    "joint_refinement_iterations": 0,
                    "joint_initial_robust_cost": None,
                    "joint_final_robust_cost": None,
                    "joint_initial_transfer_rms_px": before_rms,
                    "joint_final_transfer_rms_px": before_rms,
                }
            )
            return result
        refined = self.joint_refine_rt(
            marker_ids,
            selected_candidates,
            result["R_rel"],
            result["t_rel"],
            left_markers,
            right_markers,
            marker_quality,
        )
        updated = self.evaluate_combination(
            marker_ids,
            selected_candidates,
            left_markers,
            right_markers,
            rotation_override=refined["R"],
            translation_override=refined["t"],
            marker_quality=marker_quality,
        )
        if updated is not None and refined["final_cost"] <= refined["initial_cost"]:
            preserved = {
                key: value
                for key, value in result.items()
                if key.startswith("_") or key not in updated
            }
            result = updated
            result.update(preserved)
        result.update(
            {
                "joint_refinement_applied": True,
                "joint_refinement_converged": bool(refined["converged"]),
                "joint_refinement_iterations": refined["iterations"],
                "joint_initial_robust_cost": refined["initial_cost"],
                "joint_final_robust_cost": refined["final_cost"],
                "joint_initial_transfer_rms_px": before_rms,
                "joint_final_transfer_rms_px": result[
                    "marker_transfer_combined_rms_px"
                ],
            }
        )
        return result

    @staticmethod
    def _branch_label(marker_id, candidate) -> str:
        return (
            f"{int(marker_id)}:L{candidate['left']['index']}"
            f"-R{candidate['right']['index']}"
        )

    def _select_candidates_for_model(
        self,
        marker_ids,
        candidate_lists,
        rotation,
        translation,
        left_markers,
        right_markers,
        marker_quality,
    ):
        selected = {}
        residuals = {}
        evaluations = 0
        for marker_id in marker_ids:
            scored = []
            for candidate in candidate_lists[marker_id]:
                evaluations += 1
                result = self.evaluate_combination(
                    (marker_id,),
                    (candidate,),
                    left_markers,
                    right_markers,
                    rotation_override=rotation,
                    translation_override=translation,
                    marker_quality=marker_quality,
                )
                if result is None:
                    continue
                scored.append(
                    (
                        result["marker_transfer_combined_rms_px"],
                        candidate["self_rms"],
                        self._branch_label(marker_id, candidate),
                        candidate,
                    )
                )
            if not scored:
                continue
            scored.sort(key=lambda item: item[:3])
            residuals[int(marker_id)] = float(scored[0][0])
            selected[int(marker_id)] = scored[0][3]
        return selected, residuals, evaluations

    def _refine_robust_hypothesis(
        self,
        marker_ids,
        candidate_lists,
        seed_rotation,
        seed_translation,
        minimum_inliers,
        left_markers,
        right_markers,
        source,
        marker_quality,
    ) -> dict | None:
        rotation = np.asarray(seed_rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(seed_translation, dtype=np.float64).reshape(3, 1)
        selection_evaluations = 0
        selected_all = {}
        residuals = {}
        inlier_ids = ()
        for _iteration in range(3):
            selected_all, residuals, count = self._select_candidates_for_model(
                marker_ids,
                candidate_lists,
                rotation,
                translation,
                left_markers,
                right_markers,
                marker_quality,
            )
            selection_evaluations += count
            inlier_ids = tuple(
                marker_id
                for marker_id in marker_ids
                if marker_id in selected_all
                and residuals.get(marker_id, math.inf)
                <= self.adaptive_gate(marker_id, marker_quality)
            )
            if len(inlier_ids) < minimum_inliers:
                return None
            selected_inliers = tuple(selected_all[marker_id] for marker_id in inlier_ids)
            refined = self.evaluate_combination(
                inlier_ids,
                selected_inliers,
                left_markers,
                right_markers,
                marker_quality=marker_quality,
            )
            if refined is None:
                return None
            rotation = refined["R_rel"]
            translation = refined["t_rel"]

        selected_all, residuals, count = self._select_candidates_for_model(
            marker_ids,
            candidate_lists,
            rotation,
            translation,
            left_markers,
            right_markers,
            marker_quality,
        )
        selection_evaluations += count
        inlier_ids = tuple(
            marker_id
            for marker_id in marker_ids
            if marker_id in selected_all
            and residuals.get(marker_id, math.inf)
            <= self.adaptive_gate(marker_id, marker_quality)
        )
        if len(inlier_ids) < minimum_inliers:
            return None
        selected_inliers = tuple(selected_all[marker_id] for marker_id in inlier_ids)
        result = self.evaluate_combination(
            inlier_ids,
            selected_inliers,
            left_markers,
            right_markers,
            marker_quality=marker_quality,
        )
        if result is None:
            return None
        result["_selected_all"] = selected_all
        result["_all_marker_residuals"] = residuals
        result["_inlier_ids"] = inlier_ids
        result["_hypothesis_source"] = source
        result["_selection_evaluations"] = selection_evaluations
        return result

    @staticmethod
    def _temporal_deltas(result, temporal_prior):
        if not temporal_prior:
            return None, None
        rotation_delta = rotation_angle_deg(
            np.asarray(result["R_rel"]) @ np.asarray(temporal_prior["R"]).T
        )
        translation_delta = float(
            np.linalg.norm(
                np.asarray(result["t_rel"]).reshape(3, 1)
                - np.asarray(temporal_prior["t"]).reshape(3, 1)
            )
        )
        return rotation_delta, translation_delta

    def _model_penalty(self, result, temporal_prior=None):
        value = float(result["marker_transfer_combined_rms_px"])
        if self.assume_coplanar:
            value += 0.05 * float(result.get("coplanar_normal_spread_deg") or 0.0)
            value += 0.10 * float(result.get("coplanar_center_rms_mm") or 0.0)
        rotation_delta, translation_delta = self._temporal_deltas(
            result, temporal_prior
        )
        if rotation_delta is not None:
            value += 0.03 * (
                rotation_delta / TEMPORAL_ROTATION_SCALE_DEG
                + translation_delta / TEMPORAL_TRANSLATION_SCALE_MM
            )
        return value

    def _robust_result_score(self, result, temporal_prior=None):
        return (
            -len(result["_inlier_ids"]),
            self._model_penalty(result, temporal_prior),
            result["marker_transfer_combined_rms_px"],
            result["individual_rotation_spread_rms_deg"],
            result["individual_translation_spread_rms_mm"],
            result["self_rms"],
            result["selected_ippe_branches"],
        )

    def _estimate_exhaustive_without_rejection(
        self,
        marker_ids,
        candidate_lists,
        left_markers,
        right_markers,
        marker_quality,
        temporal_prior=None,
    ) -> dict:
        evaluated = []
        evaluated_count = 0
        lists = [candidate_lists[marker_id] for marker_id in marker_ids]
        for selected in itertools.product(*lists):
            evaluated_count += 1
            result = self.evaluate_combination(
                marker_ids,
                selected,
                left_markers,
                right_markers,
                marker_quality=marker_quality,
            )
            if result is None:
                continue
            score = (
                self._model_penalty(result, temporal_prior),
                result["marker_transfer_combined_rms_px"],
                result["individual_rotation_spread_rms_deg"],
                result["individual_translation_spread_rms_mm"],
                result["self_rms"],
                result["selected_ippe_branches"],
            )
            evaluated.append((score, result, selected))
        if not evaluated:
            raise RuntimeError("Every branch combination failed the baseline/depth gate")
        evaluated.sort(key=lambda item: item[0])
        best = evaluated[0][1]
        selected_best = tuple(evaluated[0][2])
        best = self.apply_joint_refinement(
            best,
            marker_ids,
            selected_best,
            left_markers,
            right_markers,
            marker_quality,
        )
        baseline = float(np.linalg.norm(best["t_rel"]))
        marker_text = ",".join(map(str, marker_ids))
        best.update(
            {
                "robust_method": "not_applied_for_1_or_2_patterns",
                "robust_required_inliers": len(marker_ids),
                "robust_inlier_count": len(marker_ids),
                "robust_inlier_marker_ids": marker_text,
                "robust_outlier_count": 0,
                "robust_outlier_marker_ids": "",
                "pattern_ransac_reprojection_threshold_px": (
                    self.ransac_reprojection_threshold_px
                ),
                "robust_selected_hypothesis_source": "exhaustive_branch_search",
                "robust_hypotheses_evaluated": 0,
                "leave_one_out_models_evaluated": 0,
                "all_marker_transfer_rms_px": ";".join(
                    f"{marker_id}:{best['per_marker_transfer_rms_px'][marker_id]:.6f}"
                    for marker_id in marker_ids
                ),
                "pre_rejection_marker_transfer_combined_rms_px": best[
                    "marker_transfer_combined_rms_px"
                ],
                "post_rejection_marker_transfer_combined_rms_px": best[
                    "marker_transfer_combined_rms_px"
                ],
                "pre_rejection_algorithm_baseline_mm": baseline,
                "post_rejection_algorithm_baseline_mm": baseline,
                "baseline_change_after_rejection_mm": 0.0,
                "all_selected_ippe_branches": best["selected_ippe_branches"],
                "evaluated_branch_combinations": evaluated_count,
            }
        )
        holdout = self._true_holdout_metrics(
            marker_ids,
            candidate_lists,
            left_markers,
            right_markers,
            marker_quality,
        )
        holdout.pop("_true_holdout_models", None)
        best.update(holdout)
        rotation_delta, translation_delta = self._temporal_deltas(best, temporal_prior)
        best.update(
            {
                "temporal_prior_used": temporal_prior is not None,
                "temporal_prior_rotation_delta_deg": rotation_delta,
                "temporal_prior_translation_delta_mm": translation_delta,
            }
        )
        self._frame_training_cache[tuple(marker_ids)] = (best, selected_best)
        return best

    def _fit_training_subset(
        self,
        marker_ids,
        candidate_lists,
        left_markers,
        right_markers,
        marker_quality,
    ):
        marker_ids = tuple(marker_ids)
        cached = self._frame_training_cache.get(marker_ids)
        if cached is not None:
            return cached
        evaluated = []
        for selected in itertools.product(
            *(candidate_lists[marker_id] for marker_id in marker_ids)
        ):
            result = self.evaluate_combination(
                marker_ids,
                selected,
                left_markers,
                right_markers,
                marker_quality=marker_quality,
            )
            if result is None:
                continue
            evaluated.append(
                (
                    self._model_penalty(result),
                    result["marker_transfer_combined_rms_px"],
                    result["selected_ippe_branches"],
                    result,
                    tuple(selected),
                )
            )
        if not evaluated:
            return None
        evaluated.sort(key=lambda item: item[:3])
        result = evaluated[0][3]
        selected = evaluated[0][4]
        result = self.apply_joint_refinement(
            result,
            marker_ids,
            selected,
            left_markers,
            right_markers,
            marker_quality,
        )
        fitted = (result, selected)
        self._frame_training_cache[marker_ids] = fitted
        return fitted

    def _true_holdout_metrics(
        self,
        marker_ids,
        candidate_lists,
        left_markers,
        right_markers,
        marker_quality,
    ):
        if len(marker_ids) < 2:
            return {
                "true_holdout_mean_rms_px": None,
                "true_holdout_p95_rms_px": None,
                "true_holdout_max_rms_px": None,
                "true_holdout_per_marker_rms_px": "",
                "_true_holdout_models": [],
            }
        values = []
        models = []
        labels = []
        for excluded_id in marker_ids:
            training_ids = tuple(
                marker_id for marker_id in marker_ids if marker_id != excluded_id
            )
            fitted = self._fit_training_subset(
                training_ids,
                candidate_lists,
                left_markers,
                right_markers,
                marker_quality,
            )
            if fitted is None:
                continue
            training_result, _training_selected = fitted
            scored = []
            for candidate in candidate_lists[excluded_id]:
                held_out = self.evaluate_combination(
                    (excluded_id,),
                    (candidate,),
                    left_markers,
                    right_markers,
                    rotation_override=training_result["R_rel"],
                    translation_override=training_result["t_rel"],
                    marker_quality=marker_quality,
                )
                if held_out is not None:
                    scored.append(held_out["marker_transfer_combined_rms_px"])
            if not scored:
                continue
            residual = float(min(scored))
            values.append(residual)
            labels.append(f"{excluded_id}:{residual:.6f}")
            models.append(
                {
                    "excluded_id": int(excluded_id),
                    "R": training_result["R_rel"],
                    "t": training_result["t_rel"],
                    "holdout_rms": residual,
                }
            )
        return {
            "true_holdout_mean_rms_px": mean(values),
            "true_holdout_p95_rms_px": p95(values),
            "true_holdout_max_rms_px": max(values) if values else None,
            "true_holdout_per_marker_rms_px": ";".join(labels),
            "_true_holdout_models": models,
        }

    def estimate_subset(
        self,
        marker_ids,
        left_markers,
        right_markers,
        marker_quality=None,
        temporal_prior=None,
    ) -> dict:
        marker_ids = tuple(int(marker_id) for marker_id in marker_ids)
        candidate_lists = {}
        for marker_id in marker_ids:
            candidates = self._frame_candidate_cache.get(marker_id)
            if candidates is None:
                candidates = self.marker_candidates(
                    left_markers[marker_id], right_markers[marker_id]
                )
                self._frame_candidate_cache[marker_id] = candidates
            if not candidates:
                raise RuntimeError(f"ArUco ID {marker_id} has no valid IPPE branch pair")
            candidate_lists[marker_id] = candidates

        theoretical_branch_combinations = math.prod(
            len(candidate_lists[marker_id]) for marker_id in marker_ids
        )
        if len(marker_ids) <= 2:
            return self._estimate_exhaustive_without_rejection(
                marker_ids,
                candidate_lists,
                left_markers,
                right_markers,
                marker_quality,
                temporal_prior,
            )
        minimum_inliers = 1 if len(marker_ids) == 1 else max(2, len(marker_ids) - 1)
        robust_results = []
        robust_hypotheses = 0

        # Pattern-level RANSAC: every individual marker/IPPE branch pair proposes
        # a common camera-to-camera RT.  Every marker then selects the branch that
        # transfers its corners best under that common RT.
        for marker_id in marker_ids:
            for candidate in candidate_lists[marker_id]:
                robust_hypotheses += 1
                result = self._refine_robust_hypothesis(
                    marker_ids,
                    candidate_lists,
                    candidate["R"],
                    candidate["t"],
                    minimum_inliers,
                    left_markers,
                    right_markers,
                    source=f"ransac_seed_{self._branch_label(marker_id, candidate)}",
                    marker_quality=marker_quality,
                )
                if result is not None:
                    robust_results.append(result)
        if temporal_prior is not None:
            robust_hypotheses += 1
            result = self._refine_robust_hypothesis(
                marker_ids,
                candidate_lists,
                temporal_prior["R"],
                temporal_prior["t"],
                minimum_inliers,
                left_markers,
                right_markers,
                source="previous_frame_temporal_prior",
                marker_quality=marker_quality,
            )
            if result is not None:
                robust_results.append(result)
        if not robust_results:
            raise RuntimeError(
                "No pattern-level RANSAC model reached the required inlier count "
                f"{minimum_inliers}/{len(marker_ids)} at "
                f"{self.ransac_reprojection_threshold_px:g} px"
            )

        robust_results.sort(
            key=lambda item: self._robust_result_score(item, temporal_prior)
        )

        # True holdout: fit every N-1 training subset without reading the omitted
        # marker, measure that marker afterward, then offer the independently
        # trained RT as an additional full-set hypothesis.
        holdout = self._true_holdout_metrics(
            marker_ids,
            candidate_lists,
            left_markers,
            right_markers,
            marker_quality,
        )
        leave_one_out_evaluated = 0
        for held_out_model in holdout.pop("_true_holdout_models", []):
            robust_hypotheses += 1
            leave_one_out_evaluated += 1
            result = self._refine_robust_hypothesis(
                marker_ids,
                candidate_lists,
                held_out_model["R"],
                held_out_model["t"],
                minimum_inliers,
                left_markers,
                right_markers,
                source=(
                    f"true_holdout_ID{held_out_model['excluded_id']}"
                    f"_rms{held_out_model['holdout_rms']:.3f}px"
                ),
                marker_quality=marker_quality,
            )
            if result is not None:
                robust_results.append(result)

        robust_results.sort(
            key=lambda item: self._robust_result_score(item, temporal_prior)
        )
        best = robust_results[0]
        selected_all = best["_selected_all"]
        inlier_ids = tuple(best["_inlier_ids"])
        outlier_ids = tuple(
            marker_id for marker_id in marker_ids if marker_id not in inlier_ids
        )

        pre_marker_ids = tuple(
            marker_id for marker_id in marker_ids if marker_id in selected_all
        )
        all_selected = tuple(selected_all[marker_id] for marker_id in pre_marker_ids)
        pre_rotation = average_rotations_svd(
            [candidate["R"] for candidate in all_selected]
        )
        pre_translation = np.mean(
            np.stack([candidate["t"] for candidate in all_selected], axis=0), axis=0
        )
        pre_baseline = float(np.linalg.norm(pre_translation))
        pre_result = self.evaluate_combination(
            pre_marker_ids,
            all_selected,
            left_markers,
            right_markers,
            rotation_override=pre_rotation,
            translation_override=pre_translation,
            marker_quality=marker_quality,
        )
        selected_inliers = tuple(selected_all[marker_id] for marker_id in inlier_ids)
        best = self.apply_joint_refinement(
            best,
            inlier_ids,
            selected_inliers,
            left_markers,
            right_markers,
            marker_quality,
        )
        post_baseline = float(np.linalg.norm(best["t_rel"]))
        best.update(
            {
                "robust_method": (
                    "adaptive_pattern_ransac+true_holdout+joint_refinement"
                    if self.joint_refinement
                    else "adaptive_pattern_ransac+true_holdout"
                ),
                "robust_required_inliers": minimum_inliers,
                "robust_inlier_count": len(inlier_ids),
                "robust_inlier_marker_ids": ",".join(map(str, inlier_ids)),
                "robust_outlier_count": len(outlier_ids),
                "robust_outlier_marker_ids": ",".join(map(str, outlier_ids)),
                "pattern_ransac_reprojection_threshold_px": (
                    self.ransac_reprojection_threshold_px
                ),
                "robust_selected_hypothesis_source": best["_hypothesis_source"],
                "robust_hypotheses_evaluated": robust_hypotheses,
                "leave_one_out_models_evaluated": leave_one_out_evaluated,
                "all_marker_transfer_rms_px": ";".join(
                    f"{marker_id}:{best['_all_marker_residuals'].get(marker_id, math.inf):.6f}"
                    for marker_id in marker_ids
                ),
                "pre_rejection_marker_transfer_combined_rms_px": (
                    pre_result["marker_transfer_combined_rms_px"]
                    if pre_result is not None
                    else None
                ),
                "post_rejection_marker_transfer_combined_rms_px": best[
                    "marker_transfer_combined_rms_px"
                ],
                "pre_rejection_algorithm_baseline_mm": pre_baseline,
                "post_rejection_algorithm_baseline_mm": post_baseline,
                "baseline_change_after_rejection_mm": post_baseline - pre_baseline,
                "all_selected_ippe_branches": ";".join(
                    (
                        self._branch_label(marker_id, selected_all[marker_id])
                        if marker_id in selected_all
                        else f"{marker_id}:NO_VALID_BRANCH"
                    )
                    for marker_id in marker_ids
                ),
                "evaluated_branch_combinations": theoretical_branch_combinations,
            }
        )
        best.update(holdout)
        rotation_delta, translation_delta = self._temporal_deltas(best, temporal_prior)
        best.update(
            {
                "temporal_prior_used": temporal_prior is not None,
                "temporal_prior_rotation_delta_deg": rotation_delta,
                "temporal_prior_translation_delta_mm": translation_delta,
            }
        )
        self._frame_training_cache[tuple(marker_ids)] = (best, selected_inliers)
        for internal_key in (
            "_selected_all",
            "_all_marker_residuals",
            "_inlier_ids",
            "_hypothesis_source",
            "_selection_evaluations",
        ):
            best.pop(internal_key, None)
        return best


def split_sbs(frame: np.ndarray):
    height, width = frame.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"SBS frame width must be even, got {width}")
    return frame[:, : width // 2], frame[:, width // 2 :]


def discover_marker_ids(videos, detector, max_frames) -> tuple[int, ...]:
    counts = Counter()
    scanned = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video["path"]))
        try:
            while scanned < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                left, right = split_sbs(frame)
                left_markers, _ = detector.detect(left)
                right_markers, _ = detector.detect(right)
                counts.update(set(left_markers) & set(right_markers))
                scanned += 1
                if len(counts) >= 4 and scanned >= min(5, max_frames):
                    break
        finally:
            capture.release()
        if scanned >= max_frames or (len(counts) >= 4 and scanned >= min(5, max_frames)):
            break
    if len(counts) < 4:
        raise RuntimeError(
            f"Only {len(counts)} shared stable marker IDs were discovered: {dict(counts)}"
        )
    selected = tuple(sorted(marker_id for marker_id, _ in counts.most_common(4)))
    print(f"Discovered marker IDs: {selected} | shared counts: {dict(counts)}")
    return selected


def all_subsets(marker_ids) -> list[tuple[int, ...]]:
    return [
        subset
        for count in range(1, len(marker_ids) + 1)
        for subset in itertools.combinations(marker_ids, count)
    ]


def empty_frame_row(video, frame_index, subset, shared_count) -> dict:
    row = {field: None for field in FRAME_FIELDS}
    marker_text = ",".join(str(value) for value in subset)
    row.update(
        {
            "distance_cm": video["distance_cm"],
            "angle_deg": video["angle_deg"],
            "video_file": video["path"].name,
            "video_path": str(video["path"]),
            "frame_index": frame_index,
            "marker_count": len(subset),
            "marker_ids": marker_text,
            "subset_key": f"N{len(subset)}_[{marker_text}]",
            "shared_expected_marker_count": shared_count,
            "status": "FAILED",
            "failure_reason": "",
            "solution_class": SOLUTION_FAILED,
            "effective_inlier_count": 0,
            "all_requested_patterns_used": False,
        }
    )
    return row


def corner_row(video, frame_index, marker_id, left_markers, right_markers, left_diag, right_diag):
    row = {field: None for field in CORNER_FIELDS}
    left_info = left_diag.get(marker_id, {})
    right_info = right_diag.get(marker_id, {})
    row.update(
        {
            "distance_cm": video["distance_cm"],
            "angle_deg": video["angle_deg"],
            "video_file": video["path"].name,
            "video_path": str(video["path"]),
            "frame_index": frame_index,
            "marker_id": marker_id,
            "left_initial_detected": marker_id in left_diag,
            "left_accepted": marker_id in left_markers,
            "left_subpixel_half_window": left_info.get("half_window"),
            "left_stability_max_raw_px": left_info.get("stability_max_raw_px"),
            "left_subpixel_shift_rms_px": left_info.get("subpixel_shift_rms_px"),
            "left_corner_gradient_mean": left_info.get("corner_gradient_mean"),
            "left_marker_laplacian_variance": left_info.get("marker_laplacian_variance"),
            "left_estimated_corner_uncertainty_px": left_info.get(
                "estimated_corner_uncertainty_px"
            ),
            "right_initial_detected": marker_id in right_diag,
            "right_accepted": marker_id in right_markers,
            "right_subpixel_half_window": right_info.get("half_window"),
            "right_stability_max_raw_px": right_info.get("stability_max_raw_px"),
            "right_subpixel_shift_rms_px": right_info.get("subpixel_shift_rms_px"),
            "right_corner_gradient_mean": right_info.get("corner_gradient_mean"),
            "right_marker_laplacian_variance": right_info.get("marker_laplacian_variance"),
            "right_estimated_corner_uncertainty_px": right_info.get(
                "estimated_corner_uncertainty_px"
            ),
            "shared_accepted": marker_id in left_markers and marker_id in right_markers,
        }
    )
    for side, markers in (("left", left_markers), ("right", right_markers)):
        if marker_id not in markers:
            continue
        points = np.asarray(markers[marker_id], dtype=np.float64).reshape(4, 2)
        for corner_index, (x, y) in enumerate(points):
            row[f"{side}_c{corner_index}_x_px"] = float(x)
            row[f"{side}_c{corner_index}_y_px"] = float(y)
        for metric, value in marker_geometry(points).items():
            row[f"{side}_{metric}"] = value
    return row


def analyze_subset(
    video,
    frame_index,
    subset,
    shared_count,
    left_markers,
    right_markers,
    estimator,
    answer,
    left_diag=None,
    right_diag=None,
    temporal_prior=None,
):
    started = time.perf_counter()
    row = empty_frame_row(video, frame_index, subset, shared_count)
    missing = [
        marker_id
        for marker_id in subset
        if marker_id not in left_markers or marker_id not in right_markers
    ]
    try:
        if missing:
            raise RuntimeError(f"Missing shared accepted ArUco IDs: {missing}")
        marker_quality = {}
        left_diag = left_diag or {}
        right_diag = right_diag or {}
        for marker_id in subset:
            values = []
            for diagnostics in (left_diag, right_diag):
                value = finite(
                    diagnostics.get(marker_id, {}).get(
                        "estimated_corner_uncertainty_px"
                    )
                )
                if value is not None:
                    values.append(value)
            marker_quality[int(marker_id)] = {
                "uncertainty_px": (
                    float(np.sqrt(np.mean(np.square(values)))) if values else 0.5
                )
            }
        result = estimator.estimate_subset(
            subset,
            left_markers,
            right_markers,
            marker_quality=marker_quality,
            temporal_prior=temporal_prior,
        )
        for field in (
            "marker_transfer_left_rms_px",
            "marker_transfer_right_rms_px",
            "marker_transfer_combined_rms_px",
            "marker_transfer_max_px",
            "individual_rotation_spread_rms_deg",
            "individual_translation_spread_rms_mm",
            "individual_baseline_std_mm",
            "robust_method",
            "robust_required_inliers",
            "robust_inlier_count",
            "robust_inlier_marker_ids",
            "robust_outlier_count",
            "robust_outlier_marker_ids",
            "pattern_ransac_reprojection_threshold_px",
            "adaptive_gate_mean_px",
            "adaptive_gate_min_px",
            "adaptive_gate_max_px",
            "marker_uncertainty_px",
            "marker_weights",
            "robust_selected_hypothesis_source",
            "robust_hypotheses_evaluated",
            "leave_one_out_models_evaluated",
            "all_marker_transfer_rms_px",
            "pre_rejection_marker_transfer_combined_rms_px",
            "post_rejection_marker_transfer_combined_rms_px",
            "pre_rejection_algorithm_baseline_mm",
            "post_rejection_algorithm_baseline_mm",
            "baseline_change_after_rejection_mm",
            "all_selected_ippe_branches",
            "selected_ippe_branches",
            "evaluated_branch_combinations",
            "joint_refinement_applied",
            "joint_refinement_converged",
            "joint_refinement_iterations",
            "joint_initial_robust_cost",
            "joint_final_robust_cost",
            "joint_initial_transfer_rms_px",
            "joint_final_transfer_rms_px",
            "true_holdout_mean_rms_px",
            "true_holdout_p95_rms_px",
            "true_holdout_max_rms_px",
            "true_holdout_per_marker_rms_px",
            "temporal_prior_used",
            "temporal_prior_rotation_delta_deg",
            "temporal_prior_translation_delta_mm",
            "assume_coplanar_patterns",
            "coplanar_normal_spread_deg",
            "coplanar_center_rms_mm",
        ):
            row[field] = result.get(field)
        row.update(calculate_answer_errors(result["R_rel"], result["t_rel"], answer))
        effective_inliers = int(result.get("robust_inlier_count") or 0)
        all_used = effective_inliers == len(subset)
        row["effective_inlier_count"] = effective_inliers
        row["all_requested_patterns_used"] = all_used
        row["solution_class"] = SOLUTION_STRICT if all_used else SOLUTION_FALLBACK
        row["_R_est"] = np.asarray(result["R_rel"], dtype=np.float64)
        row["_t_est"] = np.asarray(result["t_rel"], dtype=np.float64)
        row["status"] = "OK"
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
    return row


def square_crop(image, points, padding_ratio):
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 8.0)
    side = max(int(math.ceil(span * (1.0 + 2.0 * padding_ratio))), 24)
    x0 = int(math.floor(center[0] - side / 2.0))
    y0 = int(math.floor(center[1] - side / 2.0))
    output = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, image.shape[1]), min(y0 + side, image.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        output[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return output, x0, y0, side


def diagnostic_tile(frame, marker_id, accepted, diagnostics, frame_index, padding_ratio):
    tile_size = DIAGNOSTIC_TILE_SIZE
    info = diagnostics.get(marker_id)
    if marker_id in accepted:
        points = np.asarray(accepted[marker_id], dtype=np.float64)
        status = "OK"
        color = (0, 0, 255)
    elif info is not None:
        points = np.asarray(info["display"], dtype=np.float64)
        status = "UNSTABLE"
        color = (0, 165, 255)
    else:
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(
            tile,
            f"ID:{marker_id} F:{frame_index} MISSING",
            (8, tile_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return tile
    crop, x0, y0, crop_side = square_crop(frame, points, padding_ratio)
    interpolation = cv2.INTER_CUBIC if crop_side < tile_size else cv2.INTER_AREA
    tile = cv2.resize(crop, (tile_size, tile_size), interpolation=interpolation)
    tile_points = (points - np.asarray([x0, y0])) * (tile_size / float(crop_side))
    for corner_index, point in enumerate(tile_points):
        px, py = np.rint(point).astype(int)
        cv2.drawMarker(
            tile,
            (px, py),
            color,
            cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(tile, (px, py), 2, color, -1, cv2.LINE_AA)
        cv2.putText(
            tile,
            str(corner_index),
            (min(px + 5, tile_size - 12), max(py - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(tile, (0, 0), (tile_size - 1, 23), (0, 0, 0), -1)
    stability = finite(info.get("stability_max_raw_px") if info else None)
    stability_text = "" if stability is None else f" d:{stability:.2f}px"
    cv2.putText(
        tile,
        f"ID:{marker_id} F:{frame_index} {status}{stability_text}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def diagnostic_frame(
    left,
    right,
    marker_ids,
    left_markers,
    right_markers,
    left_diag,
    right_diag,
    frame_index,
    padding_ratio,
):
    def panel(frame, accepted, diagnostics, camera_label):
        tiles = [
            diagnostic_tile(
                frame,
                marker_id,
                accepted,
                diagnostics,
                frame_index,
                padding_ratio,
            )
            for marker_id in marker_ids
        ]
        grid = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        cv2.putText(
            grid,
            camera_label,
            (8, grid.shape[0] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return grid

    left_panel = panel(left, left_markers, left_diag, "LEFT")
    right_panel = panel(right, right_markers, right_diag, "RIGHT")
    combined = np.hstack((left_panel, right_panel))
    cv2.line(
        combined,
        (left_panel.shape[1], 0),
        (left_panel.shape[1], combined.shape[0] - 1),
        (255, 255, 255),
        2,
    )
    return combined


def open_diagnostic_writer(path: Path, capture):
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = finite(capture.get(cv2.CAP_PROP_FPS))
    if fps is None or fps <= 0 or fps > 240:
        fps = 30.0
    size = (DIAGNOSTIC_TILE_SIZE * 4, DIAGNOSTIC_TILE_SIZE * 2)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create diagnostic video: {path}")
    return writer


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def add_live_pass_formulas(
    frame_rows,
    video_summaries,
    distance_summaries,
    pattern_count_summaries,
    pattern_fallback_summaries,
    pattern_all_solved_summaries,
    settings,
):
    """Make every pass count/rate depend on Frame Results and Settings."""
    frame_columns = {
        field: excel_column(index + 1) for index, field in enumerate(FRAME_FIELDS)
    }
    last_frame_row = len(frame_rows) + 1

    def frame_range(field):
        column = frame_columns[field]
        return f"'Frame Results'!${column}$2:${column}${last_frame_row}"

    settings_rows = {
        row["parameter"]: index for index, row in enumerate(settings, start=2)
    }
    rotation_threshold_cell = (
        f"'Settings'!$B${settings_rows['rotation_pass_deg']}"
    )
    baseline_threshold_cell = (
        f"'Settings'!$B${settings_rows['absolute_baseline_error_pass_percent']}"
    )

    status_range = frame_range("status")
    rotation_range = frame_range("rotation_error_deg")
    baseline_range = frame_range("absolute_baseline_error_percent")
    all_used_range = frame_range("all_requested_patterns_used")

    def apply_to_sheet(
        records,
        headers,
        group_fields,
        count_suffix="frames",
        population_filter=None,
    ):
        if not records:
            return
        summary_columns = {
            field: excel_column(index + 1) for index, field in enumerate(headers)
        }
        requested_field = "requested_frames" if count_suffix == "frames" else "requested_rows"
        rotation_count_field = f"rotation_pass_{count_suffix}"
        baseline_count_field = f"baseline_pass_{count_suffix}"
        both_count_field = f"both_pass_{count_suffix}"
        rotation_rate_field = "rotation_pass_rate_percent"
        baseline_rate_field = "baseline_pass_rate_percent"
        both_rate_field = "both_pass_rate_percent"

        for row_index, record in enumerate(records, start=2):
            group_criteria = []
            for field in group_fields:
                group_criteria.extend(
                    [
                        frame_range(field),
                        f"${summary_columns[field]}{row_index}",
                    ]
                )

            def count_formula(metric):
                criteria = [*group_criteria, status_range, '"OK"']
                if population_filter == "strict":
                    criteria.extend([all_used_range, "TRUE"])
                elif population_filter == "fallback":
                    criteria.extend([all_used_range, "FALSE"])
                if metric in {"rotation", "both"}:
                    criteria.extend(
                        [rotation_range, f'"<"&{rotation_threshold_cell}']
                    )
                if metric in {"baseline", "both"}:
                    criteria.extend(
                        [baseline_range, f'"<"&{baseline_threshold_cell}']
                    )
                return f"COUNTIFS({','.join(criteria)})"

            requested_cell = f"${summary_columns[requested_field]}{row_index}"
            for metric, count_field, rate_field in (
                ("rotation", rotation_count_field, rotation_rate_field),
                ("baseline", baseline_count_field, baseline_rate_field),
                ("both", both_count_field, both_rate_field),
            ):
                cached_count = int(record.get(count_field) or 0)
                cached_rate = finite(record.get(rate_field)) or 0.0
                record[count_field] = ExcelFormula(
                    count_formula(metric), cached_count
                )
                count_cell = f"{summary_columns[count_field]}{row_index}"
                record[rate_field] = ExcelFormula(
                    f"IFERROR({count_cell}/{requested_cell}*100,0)",
                    cached_rate,
                )

    apply_to_sheet(
        video_summaries,
        VIDEO_SUMMARY_FIELDS,
        ("video_file", "subset_key"),
    )
    apply_to_sheet(
        distance_summaries,
        DISTANCE_SUMMARY_FIELDS,
        ("distance_cm", "angle_deg", "subset_key"),
    )
    apply_to_sheet(
        pattern_count_summaries,
        PATTERN_COUNT_SUMMARY_FIELDS,
        ("distance_cm", "angle_deg", "marker_count"),
        count_suffix="rows",
        population_filter="strict",
    )
    apply_to_sheet(
        pattern_fallback_summaries,
        PATTERN_COUNT_SUMMARY_FIELDS,
        ("distance_cm", "angle_deg", "marker_count"),
        count_suffix="rows",
        population_filter="fallback",
    )
    apply_to_sheet(
        pattern_all_solved_summaries,
        PATTERN_COUNT_SUMMARY_FIELDS,
        ("distance_cm", "angle_deg", "marker_count"),
        count_suffix="rows",
    )


def _trend_metric_value(rows, distance, angle, marker_count, field):
    values = [
        row.get(field)
        for row in rows
        if finite(row.get("distance_cm")) is not None
        and math.isclose(float(row["distance_cm"]), float(distance), abs_tol=1e-9)
        and math.isclose(float(row.get("angle_deg") or 0.0), float(angle), abs_tol=1e-9)
        and int(row.get("marker_count") or 0) == int(marker_count)
    ]
    return mean(values)


def _trend_cell(ref, value=None, formula=None, cached_value=None, style=0):
    style_attribute = f' s="{style}"' if style else ""
    if formula is not None:
        formula_xml = escape(str(formula))
        if cached_value is None:
            return (
                f'<c r="{ref}" t="e"{style_attribute}><f>{formula_xml}</f>'
                '<v>#N/A</v></c>'
            )
        return (
            f'<c r="{ref}"{style_attribute}><f>{formula_xml}</f>'
            f'<v>{float(cached_value):.15g}</v></c>'
        )
    if value is None or value == "":
        return f'<c r="{ref}"{style_attribute}/>'
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f'<c r="{ref}"{style_attribute}><v>{float(value):.15g}</v></c>'
    return (
        f'<c r="{ref}" t="inlineStr"{style_attribute}><is>'
        f'<t xml:space="preserve">{escape(str(value))}</t></is></c>'
    )


def build_trend_worksheet_xml(pattern_count_summaries, angle):
    distances = sorted(
        {
            float(value)
            for row in pattern_count_summaries
            if (value := finite(row.get("distance_cm"))) is not None
        }
    )
    if not distances:
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<dimension ref="A1"/><sheetData><row r="1">'
            + _trend_cell("A1", "No numeric distance data available for trend charts")
            + '</row></sheetData></worksheet>'
        )
        return xml, None

    rotation_header_row = 1
    rotation_first_row = 2
    rotation_last_row = rotation_first_row + len(distances) - 1
    baseline_title_row = rotation_last_row + 3
    baseline_header_row = baseline_title_row + 1
    baseline_first_row = baseline_header_row + 1
    baseline_last_row = baseline_first_row + len(distances) - 1
    rows = defaultdict(list)

    for column, value in zip(("A", "B", "C", "D", "E"), ("distance_cm", 1, 2, 3, 4)):
        rows[rotation_header_row].append(
            _trend_cell(f"{column}{rotation_header_row}", value, style=1)
        )
    rows[1].append(_trend_cell("H1", "Angle filter", style=1))
    rows[2].append(_trend_cell("H2", angle, style=5))
    rows[4].append(_trend_cell("H4", "Population", style=1))
    rows[5].append(_trend_cell("H5", SOLUTION_STRICT))

    source_last_row = 1000
    rotation_values = {count: [] for count in range(1, 5)}
    baseline_values = {count: [] for count in range(1, 5)}
    for offset, distance in enumerate(distances):
        rotation_row = rotation_first_row + offset
        baseline_row = baseline_first_row + offset
        rows[rotation_row].append(_trend_cell(f"A{rotation_row}", distance, style=3))
        rows[baseline_row].append(_trend_cell(f"A{baseline_row}", distance, style=3))
        for count, column in enumerate(("B", "C", "D", "E"), start=1):
            rotation_cached = _trend_metric_value(
                pattern_count_summaries,
                distance,
                angle,
                count,
                "rotation_error_mean_deg",
            )
            baseline_cached = _trend_metric_value(
                pattern_count_summaries,
                distance,
                angle,
                count,
                "absolute_baseline_error_mean_percent",
            )
            rotation_values[count].append(rotation_cached)
            baseline_values[count].append(baseline_cached)
            rotation_formula = (
                "IFERROR(AVERAGEIFS('Pattern Count Summary'!$H$2:$H$"
                f"{source_last_row},'Pattern Count Summary'!$A$2:$A${source_last_row},"
                f"$A{rotation_row},'Pattern Count Summary'!$C$2:$C${source_last_row},"
                f"{column}$1,'Pattern Count Summary'!$B$2:$B${source_last_row},$H$2),NA())"
            )
            baseline_formula = (
                "IFERROR(AVERAGEIFS('Pattern Count Summary'!$P$2:$P$"
                f"{source_last_row},'Pattern Count Summary'!$A$2:$A${source_last_row},"
                f"$A{baseline_row},'Pattern Count Summary'!$C$2:$C${source_last_row},"
                f"{column}${baseline_header_row},'Pattern Count Summary'!$B$2:$B$"
                f"{source_last_row},$H$2),NA())"
            )
            rows[rotation_row].append(
                _trend_cell(
                    f"{column}{rotation_row}",
                    formula=rotation_formula,
                    cached_value=rotation_cached,
                    style=3,
                )
            )
            rows[baseline_row].append(
                _trend_cell(
                    f"{column}{baseline_row}",
                    formula=baseline_formula,
                    cached_value=baseline_cached,
                    style=3,
                )
            )

    rows[baseline_title_row].append(
        _trend_cell(f"B{baseline_title_row}", "使用pattern數量")
    )
    for column, value in zip(("A", "B", "C", "D", "E"), ("distance_cm", 1, 2, 3, 4)):
        rows[baseline_header_row].append(
            _trend_cell(f"{column}{baseline_header_row}", value, style=1)
        )

    row_xml = "".join(
        f'<row r="{row_index}">{"".join(rows[row_index])}</row>'
        for row_index in sorted(rows)
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:H{baseline_last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0" showGridLines="0"/></sheetViews>'
        '<cols>'
        '<col min="1" max="1" width="14" customWidth="1"/>'
        '<col min="2" max="5" width="14" customWidth="1"/>'
        '<col min="6" max="7" width="4" customWidth="1"/>'
        '<col min="8" max="8" width="14" customWidth="1"/>'
        '</cols>'
        f'<sheetData>{row_xml}</sheetData>'
        '<drawing r:id="rId1"/></worksheet>'
    )
    metadata = {
        "distances": distances,
        "rotation_values": rotation_values,
        "baseline_values": baseline_values,
        "rotation_first_row": rotation_first_row,
        "rotation_last_row": rotation_last_row,
        "baseline_first_row": baseline_first_row,
        "baseline_last_row": baseline_last_row,
    }
    return worksheet, metadata


def _chart_title_xml(text):
    return (
        '<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p>'
        '<a:r><a:rPr lang="zh-TW" sz="1400" b="0"/>'
        f'<a:t>{escape(text)}</a:t></a:r>'
        '</a:p></c:rich></c:tx><c:layout/><c:overlay val="0"/></c:title>'
    )


def _chart_axis_title_xml(text):
    return (
        '<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p>'
        '<a:r><a:rPr lang="zh-TW" sz="1000"/>'
        f'<a:t>{escape(text)}</a:t></a:r>'
        '</a:p></c:rich></c:tx><c:layout/><c:overlay val="0"/></c:title>'
    )


def _chart_num_ref_xml(formula, values):
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{float(value):.15g}</c:v></c:pt>'
        for index, value in enumerate(values)
        if value is not None
    )
    return (
        '<c:numRef>'
        f'<c:f>{escape(formula)}</c:f>'
        '<c:numCache><c:formatCode>General</c:formatCode>'
        f'<c:ptCount val="{len(values)}"/>{points}</c:numCache>'
        '</c:numRef>'
    )


def build_scatter_chart_xml(title, y_axis_title, x_formula, y_formulas, x_values, y_values):
    colors = ("4472C4", "ED7D31", "A5A5A5", "FFC000")
    names = ("使用1個pattern", "使用2個", "使用3個", "使用4個")
    series_xml = []
    for index, (name, color, y_formula) in enumerate(
        zip(names, colors, y_formulas)
    ):
        series_xml.append(
            '<c:ser>'
            f'<c:idx val="{index}"/><c:order val="{index}"/>'
            f'<c:tx><c:v>{escape(name)}</c:v></c:tx>'
            '<c:spPr><a:ln w="19050"><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill></a:ln></c:spPr>'
            '<c:marker><c:symbol val="circle"/><c:size val="5"/>'
            '<c:spPr><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill><a:ln><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill></a:ln></c:spPr></c:marker>'
            f'<c:xVal>{_chart_num_ref_xml(x_formula, x_values)}</c:xVal>'
            f'<c:yVal>{_chart_num_ref_xml(y_formula, y_values[index + 1])}</c:yVal>'
            '<c:smooth val="0"/></c:ser>'
        )
    x_axis_id = 1100000000 if "Rotate" in title else 1200000000
    y_axis_id = x_axis_id + 1
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<c:date1904 val="0"/><c:lang val="zh-TW"/><c:roundedCorners val="0"/>'
        '<c:chart>'
        f'{_chart_title_xml(title)}'
        '<c:autoTitleDeleted val="0"/><c:plotArea><c:layout/>'
        '<c:scatterChart><c:scatterStyle val="lineMarker"/><c:varyColors val="0"/>'
        f'{"".join(series_xml)}<c:axId val="{x_axis_id}"/><c:axId val="{y_axis_id}"/>'
        '</c:scatterChart>'
        f'<c:valAx><c:axId val="{x_axis_id}"/><c:scaling><c:orientation val="minMax"/>'
        '</c:scaling><c:delete val="0"/><c:axPos val="b"/>'
        f'{_chart_axis_title_xml("Distance (cm)")}'
        '<c:numFmt formatCode="0.0" sourceLinked="0"/><c:majorTickMark val="out"/>'
        '<c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{y_axis_id}"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="midCat"/></c:valAx>'
        f'<c:valAx><c:axId val="{y_axis_id}"/><c:scaling><c:orientation val="minMax"/>'
        '</c:scaling><c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/>'
        f'{_chart_axis_title_xml(y_axis_title)}'
        '<c:numFmt formatCode="0.00" sourceLinked="0"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        '<c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{x_axis_id}"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="midCat"/></c:valAx>'
        '</c:plotArea><c:legend><c:legendPos val="r"/><c:layout/>'
        '<c:overlay val="0"/></c:legend><c:plotVisOnly val="1"/>'
        '<c:dispBlanksAs val="gap"/><c:showDLblsOverMax val="0"/>'
        '</c:chart><c:printSettings><c:headerFooter/><c:pageMargins b="0.75" l="0.7" '
        'r="0.7" t="0.75" header="0.3" footer="0.3"/><c:pageSetup/></c:printSettings>'
        '</c:chartSpace>'
    )


def _drawing_anchor_xml(chart_id, relationship_id, from_row, from_col, to_row, to_col):
    return (
        '<xdr:twoCellAnchor><xdr:from>'
        f'<xdr:col>{from_col}</xdr:col><xdr:colOff>0</xdr:colOff>'
        f'<xdr:row>{from_row}</xdr:row><xdr:rowOff>0</xdr:rowOff>'
        '</xdr:from><xdr:to>'
        f'<xdr:col>{to_col}</xdr:col><xdr:colOff>0</xdr:colOff>'
        f'<xdr:row>{to_row}</xdr:row><xdr:rowOff>0</xdr:rowOff>'
        '</xdr:to><xdr:graphicFrame macro=""><xdr:nvGraphicFramePr>'
        f'<xdr:cNvPr id="{chart_id + 1}" name="Chart {chart_id}"/>'
        '<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm>'
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
        '<a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'r:id="{relationship_id}"/></a:graphicData></a:graphic>'
        '</xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor>'
    )


def add_trend_charts_to_xlsx(path, pattern_count_summaries, angle, sheet_index=2):
    worksheet, metadata = build_trend_worksheet_xml(pattern_count_summaries, angle)
    if metadata is None:
        chart_parts = {}
    else:
        rotation_first = metadata["rotation_first_row"]
        rotation_last = metadata["rotation_last_row"]
        baseline_first = metadata["baseline_first_row"]
        baseline_last = metadata["baseline_last_row"]
        rotation_x = f"'Trend Charts'!$A${rotation_first}:$A${rotation_last}"
        baseline_x = f"'Trend Charts'!$A${baseline_first}:$A${baseline_last}"
        rotation_y = [
            f"'Trend Charts'!${column}${rotation_first}:${column}${rotation_last}"
            for column in ("B", "C", "D", "E")
        ]
        baseline_y = [
            f"'Trend Charts'!${column}${baseline_first}:${column}${baseline_last}"
            for column in ("B", "C", "D", "E")
        ]
        drawing = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'{_drawing_anchor_xml(1, "rId1", 0, 9, 15, 18)}'
            f'{_drawing_anchor_xml(2, "rId2", 17, 9, 32, 18)}'
            '</xdr:wsDr>'
        )
        drawing_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
            'Target="../charts/chart1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
            'Target="../charts/chart2.xml"/></Relationships>'
        )
        sheet_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
            'Target="../drawings/drawing1.xml"/></Relationships>'
        )
        chart_parts = {
            f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels": sheet_rels,
            "xl/drawings/drawing1.xml": drawing,
            "xl/drawings/_rels/drawing1.xml.rels": drawing_rels,
            "xl/charts/chart1.xml": build_scatter_chart_xml(
                "Rotate Error 與 pattern 使用數量的關聯性",
                "Rotation error (deg)",
                rotation_x,
                rotation_y,
                metadata["distances"],
                metadata["rotation_values"],
            ),
            "xl/charts/chart2.xml": build_scatter_chart_xml(
                "baseline 誤差與 pattern 使用數量的關聯性",
                "Absolute baseline error (%)",
                baseline_x,
                baseline_y,
                metadata["distances"],
                metadata["baseline_values"],
            ),
        }

    path = Path(path)
    temporary = tempfile.NamedTemporaryFile(
        prefix=path.stem + "_trend_",
        suffix=".xlsx",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as destination:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == f"xl/worksheets/sheet{sheet_index}.xml":
                    data = worksheet.encode("utf-8")
                elif info.filename == "[Content_Types].xml" and chart_parts:
                    text = data.decode("utf-8")
                    overrides = (
                        '<Override PartName="/xl/drawings/drawing1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                        '<Override PartName="/xl/charts/chart1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                        '<Override PartName="/xl/charts/chart2.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                    )
                    data = text.replace("</Types>", overrides + "</Types>").encode(
                        "utf-8"
                    )
                destination.writestr(info, data)
            for name, data in chart_parts.items():
                destination.writestr(name, data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def coordinate_std_rms(record: dict, side: str) -> float | None:
    values = [
        finite(record.get(f"{side}_c{corner}_{axis}_std_px"))
        for corner in range(4)
        for axis in ("x", "y")
    ]
    values = [value for value in values if value is not None]
    return float(math.sqrt(np.mean(np.square(values)))) if values else None


def build_corner_summaries(corner_rows) -> list[dict]:
    groups = defaultdict(list)
    for row in corner_rows:
        key = (row["video_file"], row["marker_id"])
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        first = group[0]
        record = {field: None for field in CORNER_SUMMARY_FIELDS}
        record.update(
            {
                "distance_cm": first["distance_cm"],
                "angle_deg": first["angle_deg"],
                "video_file": first["video_file"],
                "marker_id": first["marker_id"],
                "requested_frames": len(group),
                "left_detected_frames": sum(bool(row["left_accepted"]) for row in group),
                "right_detected_frames": sum(bool(row["right_accepted"]) for row in group),
                "shared_detected_frames": sum(bool(row["shared_accepted"]) for row in group),
                "shared_detection_rate_percent": (
                    sum(bool(row["shared_accepted"]) for row in group) / len(group) * 100.0
                ),
            }
        )
        for side in ("left", "right"):
            for corner in range(4):
                for axis in ("x", "y"):
                    field = f"{side}_c{corner}_{axis}_px"
                    record[f"{side}_c{corner}_{axis}_std_px"] = std(
                        row.get(field) for row in group
                    )
            record[f"{side}_corner_coordinate_std_rms_px"] = coordinate_std_rms(
                record, side
            )
            for metric in (
                "rect_width_px",
                "rect_height_px",
                "rect_area_px2",
                "quad_area_px2",
                "side_mean_px",
            ):
                values = [row.get(f"{side}_{metric}") for row in group]
                base = metric.removesuffix("_px").removesuffix("_px2")
                if metric == "side_mean_px":
                    record[f"{side}_side_mean_mean_px"] = mean(values)
                    record[f"{side}_side_mean_std_px"] = std(values)
                elif metric.endswith("_px2"):
                    record[f"{side}_{base}_mean_px2"] = mean(values)
                    record[f"{side}_{base}_std_px2"] = std(values)
                else:
                    record[f"{side}_{base}_mean_px"] = mean(values)
                    record[f"{side}_{base}_std_px"] = std(values)
            for quality_metric in (
                "subpixel_shift_rms_px",
                "corner_gradient_mean",
                "marker_laplacian_variance",
                "estimated_corner_uncertainty_px",
            ):
                quality_values = [
                    row.get(f"{side}_{quality_metric}") for row in group
                ]
                record[f"{side}_{quality_metric}_mean"] = mean(quality_values)
                record[f"{side}_{quality_metric}_std"] = std(quality_values)
        output.append(record)
    return output


def subset_corner_metrics(corner_summary_lookup, video_file, marker_ids) -> dict:
    selected = [
        corner_summary_lookup.get((video_file, marker_id)) for marker_id in marker_ids
    ]
    selected = [record for record in selected if record is not None]
    result = {}
    for side in ("left", "right"):
        coordinate_values = [
            finite(record.get(f"{side}_corner_coordinate_std_rms_px"))
            for record in selected
        ]
        coordinate_values = [value for value in coordinate_values if value is not None]
        result[f"{side}_corner_coordinate_std_rms_px"] = (
            float(math.sqrt(np.mean(np.square(coordinate_values))))
            if coordinate_values
            else None
        )
        result[f"{side}_rect_width_std_mean_px"] = mean(
            record.get(f"{side}_rect_width_std_px") for record in selected
        )
        result[f"{side}_side_mean_std_mean_px"] = mean(
            record.get(f"{side}_side_mean_std_px") for record in selected
        )
    return result


def summary_record(group, metadata, rotation_threshold, baseline_threshold) -> dict:
    solved = [row for row in group if row["status"] == "OK"]
    strict = [row for row in solved if bool(row.get("all_requested_patterns_used"))]
    fallback = [row for row in solved if not bool(row.get("all_requested_patterns_used"))]
    record = dict(metadata)
    record.update(
        {
            "requested_frames": len(group),
            "solved_frames": len(solved),
            "solve_rate_percent": len(solved) / len(group) * 100.0 if group else None,
        }
    )
    rotation_pass = [
        row for row in solved if row["rotation_error_deg"] < rotation_threshold
    ]
    baseline_pass = [
        row
        for row in solved
        if row["absolute_baseline_error_percent"] < baseline_threshold
    ]
    both_pass = [
        row
        for row in solved
        if row["rotation_error_deg"] < rotation_threshold
        and row["absolute_baseline_error_percent"] < baseline_threshold
    ]
    record.update(
        {
            "rotation_pass_frames": len(rotation_pass),
            "rotation_pass_rate_percent": len(rotation_pass) / len(group) * 100.0,
            "baseline_pass_frames": len(baseline_pass),
            "baseline_pass_rate_percent": len(baseline_pass) / len(group) * 100.0,
            "both_pass_frames": len(both_pass),
            "both_pass_rate_percent": len(both_pass) / len(group) * 100.0,
        }
    )
    metric_specs = (
        ("rotation_error_deg", "rotation_error", "deg"),
        ("algorithm_baseline_mm", "algorithm_baseline", "mm"),
        ("absolute_baseline_delta_mm", "absolute_baseline_delta", "mm"),
        ("absolute_baseline_error_percent", "absolute_baseline_error", "percent"),
    )
    record["baseline_delta_mean_mm"] = mean(
        row["baseline_delta_mm"] for row in solved
    )
    for source, name, unit in metric_specs:
        values = [row[source] for row in solved]
        record[f"{name}_mean_{unit}"] = mean(values)
        if name != "algorithm_baseline":
            record[f"{name}_median_{unit}"] = median(values)
            record[f"{name}_p95_{unit}"] = p95(values)
        record[f"{name}_std_{unit}"] = std(values)
    transfer_values = [row["marker_transfer_combined_rms_px"] for row in solved]
    record["marker_transfer_rms_mean_px"] = mean(transfer_values)
    record["marker_transfer_rms_p95_px"] = p95(transfer_values)
    record["individual_rotation_spread_mean_deg"] = mean(
        row["individual_rotation_spread_rms_deg"] for row in solved
    )
    record["individual_translation_spread_mean_mm"] = mean(
        row["individual_translation_spread_rms_mm"] for row in solved
    )
    record["robust_inlier_count_mean"] = mean(
        row.get("robust_inlier_count") for row in solved
    )
    record["robust_outlier_frame_rate_percent"] = (
        sum((finite(row.get("robust_outlier_count")) or 0) > 0 for row in solved)
        / len(group)
        * 100.0
        if group
        else None
    )
    record["pre_rejection_marker_transfer_rms_mean_px"] = mean(
        row.get("pre_rejection_marker_transfer_combined_rms_px") for row in solved
    )
    record["post_rejection_marker_transfer_rms_mean_px"] = mean(
        row.get("post_rejection_marker_transfer_combined_rms_px") for row in solved
    )
    record["joint_initial_transfer_rms_mean_px"] = mean(
        row.get("joint_initial_transfer_rms_px") for row in solved
    )
    record["joint_final_transfer_rms_mean_px"] = mean(
        row.get("joint_final_transfer_rms_px") for row in solved
    )
    record["true_holdout_rms_mean_px"] = mean(
        row.get("true_holdout_mean_rms_px") for row in solved
    )
    record["temporal_prior_rotation_delta_mean_deg"] = mean(
        row.get("temporal_prior_rotation_delta_deg") for row in solved
    )
    record["temporal_prior_translation_delta_mean_mm"] = mean(
        row.get("temporal_prior_translation_delta_mm") for row in solved
    )
    record["coplanar_normal_spread_mean_deg"] = mean(
        row.get("coplanar_normal_spread_deg") for row in solved
    )
    record["coplanar_center_rms_mean_mm"] = mean(
        row.get("coplanar_center_rms_mm") for row in solved
    )
    record.update(
        {
            "strict_solved_frames": len(strict),
            "strict_solve_rate_percent": (
                len(strict) / len(group) * 100.0 if group else None
            ),
            "fallback_solved_frames": len(fallback),
            "fallback_solve_rate_percent": (
                len(fallback) / len(group) * 100.0 if group else None
            ),
            "strict_rotation_error_mean_deg": mean(
                row.get("rotation_error_deg") for row in strict
            ),
            "strict_absolute_baseline_error_mean_percent": mean(
                row.get("absolute_baseline_error_percent") for row in strict
            ),
            "fallback_rotation_error_mean_deg": mean(
                row.get("rotation_error_deg") for row in fallback
            ),
            "fallback_absolute_baseline_error_mean_percent": mean(
                row.get("absolute_baseline_error_percent") for row in fallback
            ),
        }
    )
    return record


def build_video_summaries(frame_rows, corner_summaries, rotation_threshold, baseline_threshold):
    corner_lookup = {
        (row["video_file"], int(row["marker_id"])): row for row in corner_summaries
    }
    groups = defaultdict(list)
    for row in frame_rows:
        groups[(row["video_file"], row["subset_key"])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        first = group[0]
        record = summary_record(
            group,
            {
                "distance_cm": first["distance_cm"],
                "angle_deg": first["angle_deg"],
                "video_file": first["video_file"],
                "marker_count": first["marker_count"],
                "marker_ids": first["marker_ids"],
                "subset_key": first["subset_key"],
            },
            rotation_threshold,
            baseline_threshold,
        )
        marker_ids = tuple(int(value) for value in first["marker_ids"].split(","))
        record.update(subset_corner_metrics(corner_lookup, first["video_file"], marker_ids))
        output.append(record)
    return output


def build_distance_summaries(frame_rows, video_summaries, rotation_threshold, baseline_threshold):
    groups = defaultdict(list)
    for row in frame_rows:
        groups[(row["distance_cm"], row["angle_deg"], row["subset_key"])].append(row)
    video_groups = defaultdict(list)
    for row in video_summaries:
        video_groups[(row["distance_cm"], row["angle_deg"], row["subset_key"])].append(row)
    output = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            float("inf") if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        first = group[0]
        record = summary_record(
            group,
            {
                "distance_cm": first["distance_cm"],
                "angle_deg": first["angle_deg"],
                "video_count": len({row["video_file"] for row in group}),
                "marker_count": first["marker_count"],
                "marker_ids": first["marker_ids"],
                "subset_key": first["subset_key"],
            },
            rotation_threshold,
            baseline_threshold,
        )
        summaries = video_groups[key]
        for field in (
            "left_corner_coordinate_std_rms_px",
            "right_corner_coordinate_std_rms_px",
            "left_rect_width_std_mean_px",
            "right_rect_width_std_mean_px",
            "left_side_mean_std_mean_px",
            "right_side_mean_std_mean_px",
        ):
            record[field] = mean(row.get(field) for row in summaries)
        output.append(record)
    return output


def build_pattern_count_summaries(
    frame_rows,
    rotation_threshold,
    baseline_threshold,
    population="strict",
):
    if population not in {"strict", "fallback", "all"}:
        raise ValueError(f"Unsupported pattern-count population: {population}")
    groups = defaultdict(list)
    for row in frame_rows:
        groups[(row["distance_cm"], row["angle_deg"], row["marker_count"])].append(row)
    output = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            float("inf") if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        all_solved = [row for row in group if row["status"] == "OK"]
        if population == "strict":
            solved = [
                row for row in all_solved if bool(row.get("all_requested_patterns_used"))
            ]
        elif population == "fallback":
            solved = [
                row
                for row in all_solved
                if not bool(row.get("all_requested_patterns_used"))
            ]
        else:
            solved = all_solved
        rotation_pass = [
            row for row in solved if row["rotation_error_deg"] < rotation_threshold
        ]
        baseline_pass = [
            row
            for row in solved
            if row["absolute_baseline_error_percent"] < baseline_threshold
        ]
        both_pass = [
            row
            for row in solved
            if row["rotation_error_deg"] < rotation_threshold
            and row["absolute_baseline_error_percent"] < baseline_threshold
        ]
        record = {
            "distance_cm": key[0],
            "angle_deg": key[1],
            "marker_count": key[2],
            "subset_count": len({row["subset_key"] for row in group}),
            "requested_rows": len(group),
            "solved_rows": len(solved),
            "solve_rate_percent": len(solved) / len(group) * 100.0,
            "rotation_error_mean_deg": mean(row["rotation_error_deg"] for row in solved),
            "rotation_error_median_deg": median(row["rotation_error_deg"] for row in solved),
            "rotation_error_std_deg": std(row["rotation_error_deg"] for row in solved),
            "rotation_error_p95_deg": p95(row["rotation_error_deg"] for row in solved),
            "absolute_baseline_delta_mean_mm": mean(
                row["absolute_baseline_delta_mm"] for row in solved
            ),
            "absolute_baseline_delta_median_mm": median(
                row["absolute_baseline_delta_mm"] for row in solved
            ),
            "absolute_baseline_delta_std_mm": std(
                row["absolute_baseline_delta_mm"] for row in solved
            ),
            "absolute_baseline_delta_p95_mm": p95(
                row["absolute_baseline_delta_mm"] for row in solved
            ),
            "absolute_baseline_error_mean_percent": mean(
                row["absolute_baseline_error_percent"] for row in solved
            ),
            "absolute_baseline_error_median_percent": median(
                row["absolute_baseline_error_percent"] for row in solved
            ),
            "absolute_baseline_error_std_percent": std(
                row["absolute_baseline_error_percent"] for row in solved
            ),
            "absolute_baseline_error_p95_percent": p95(
                row["absolute_baseline_error_percent"] for row in solved
            ),
            "rotation_pass_rows": len(rotation_pass),
            "rotation_pass_rate_percent": len(rotation_pass) / len(group) * 100.0,
            "baseline_pass_rows": len(baseline_pass),
            "baseline_pass_rate_percent": len(baseline_pass) / len(group) * 100.0,
            "both_pass_rows": len(both_pass),
            "both_pass_rate_percent": len(both_pass) / len(group) * 100.0,
            "robust_inlier_count_mean": mean(
                row.get("robust_inlier_count") for row in solved
            ),
            "robust_outlier_row_rate_percent": (
                sum(
                    (finite(row.get("robust_outlier_count")) or 0) > 0
                    for row in solved
                )
                / len(group)
                * 100.0
                if group
                else None
            ),
            "pre_rejection_marker_transfer_rms_mean_px": mean(
                row.get("pre_rejection_marker_transfer_combined_rms_px")
                for row in solved
            ),
            "post_rejection_marker_transfer_rms_mean_px": mean(
                row.get("post_rejection_marker_transfer_combined_rms_px")
                for row in solved
            ),
            "joint_initial_transfer_rms_mean_px": mean(
                row.get("joint_initial_transfer_rms_px") for row in solved
            ),
            "joint_final_transfer_rms_mean_px": mean(
                row.get("joint_final_transfer_rms_px") for row in solved
            ),
            "true_holdout_rms_mean_px": mean(
                row.get("true_holdout_mean_rms_px") for row in solved
            ),
            "temporal_prior_rotation_delta_mean_deg": mean(
                row.get("temporal_prior_rotation_delta_deg") for row in solved
            ),
            "temporal_prior_translation_delta_mean_mm": mean(
                row.get("temporal_prior_translation_delta_mm") for row in solved
            ),
            "coplanar_normal_spread_mean_deg": mean(
                row.get("coplanar_normal_spread_deg") for row in solved
            ),
            "coplanar_center_rms_mean_mm": mean(
                row.get("coplanar_center_rms_mm") for row in solved
            ),
            "summary_population": {
                "strict": SOLUTION_STRICT,
                "fallback": SOLUTION_FALLBACK,
                "all": "ALL_SOLVED_LEGACY",
            }[population],
            "effective_inlier_count_mean": mean(
                row.get("effective_inlier_count") for row in solved
            ),
        }
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    calibration = Path(args.calibration).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Video folder does not exist: {folder}")
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration JSON does not exist: {calibration}")
    distance_map = parse_distance_map(args.distance_map)
    videos = collect_videos(folder, args.recursive, args.only_video, distance_map)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No video files were found in {folder}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else folder
        / (
            Path(DEFAULT_OUTPUT_NAME).stem
            + ("_coplanar" if args.assume_coplanar else "")
            + ".xlsx"
        )
    )
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    corner_video_dir = (
        Path(args.corner_video_dir).expanduser().resolve()
        if args.corner_video_dir
        else output.with_name(output.stem + "_corner_videos")
    )

    k_left, d_left, k_right, d_right, answer = load_calibration(calibration)
    detector = AdaptiveArucoDetector(
        args.subpixel_stability_max_raw_px,
        args.aruco_initial_refinement,
    )
    marker_ids = parse_marker_ids(args.marker_ids)
    if marker_ids is None:
        marker_ids = discover_marker_ids(videos, detector, args.id_discovery_frames)
    subsets = all_subsets(marker_ids)
    if len(subsets) != 15:
        raise RuntimeError(f"Expected 15 non-empty subsets, got {len(subsets)}")
    estimator = MultiMarkerRTEstimator(
        k_left,
        d_left,
        k_right,
        d_right,
        args.marker_size_mm,
        args.min_baseline_mm,
        args.max_baseline_mm,
        args.pattern_ransac_reproj_threshold_px,
        joint_refinement=args.joint_refinement,
        joint_max_iterations=args.joint_max_iterations,
        joint_huber_delta_px=args.joint_huber_delta_px,
        assume_coplanar=args.assume_coplanar,
        coplanar_normal_weight_px=args.coplanar_normal_weight_px,
        coplanar_center_weight_px=args.coplanar_center_weight_px,
    )

    print(f"Videos: {len(videos)}")
    print(f"Marker IDs: {marker_ids}")
    print(f"Subsets per frame: {len(subsets)}")
    print(f"Frames per video: first {args.frames} frames (F0..F{args.frames - 1})")
    print(
        "Pattern robust gate: adaptive bidirectional transfer RMS from nominal "
        f"{args.pattern_ransac_reproj_threshold_px:g} px | "
        "N3 requires 2 inliers, N4 requires 3"
    )
    print(
        f"Joint refinement: {args.joint_refinement} | temporal prior: "
        f"{args.temporal_prior} | assume coplanar: {args.assume_coplanar}"
    )
    print("JSON extrinsics: answer comparison only; never used for branch selection")

    frame_rows = []
    corner_rows = []
    overall_started = time.perf_counter()
    for video_index, video in enumerate(videos, 1):
        print(f"[{video_index}/{len(videos)}] {video['path'].name}")
        capture = cv2.VideoCapture(str(video["path"]))
        if not capture.isOpened():
            print("  WARNING: could not open video; skipped")
            continue
        diagnostic_writer = None
        temporal_priors = {}
        try:
            if not args.no_corner_videos:
                diagnostic_writer = open_diagnostic_writer(
                    corner_video_dir / f"{video['path'].stem}_4pattern_corners.mp4",
                    capture,
                )
            for frame_index in range(args.frames):
                ok, frame = capture.read()
                if not ok:
                    print(f"  Video ended at F{frame_index}")
                    break
                left, right = split_sbs(frame)
                left_markers, left_diag = detector.detect(left)
                right_markers, right_diag = detector.detect(right)
                shared_count = len(
                    set(marker_ids) & set(left_markers) & set(right_markers)
                )
                estimator.begin_frame()
                for marker_id in marker_ids:
                    corner_rows.append(
                        corner_row(
                            video,
                            frame_index,
                            marker_id,
                            left_markers,
                            right_markers,
                            left_diag,
                            right_diag,
                        )
                    )
                for subset in subsets:
                    prior = temporal_priors.get(subset) if args.temporal_prior else None
                    subset_row = analyze_subset(
                        video,
                        frame_index,
                        subset,
                        shared_count,
                        left_markers,
                        right_markers,
                        estimator,
                        answer,
                        left_diag=left_diag,
                        right_diag=right_diag,
                        temporal_prior=prior,
                    )
                    frame_rows.append(subset_row)
                    if subset_row["status"] == "OK" and args.temporal_prior:
                        temporal_priors[subset] = {
                            "R": subset_row["_R_est"],
                            "t": subset_row["_t_est"],
                        }
                if diagnostic_writer is not None:
                    diagnostic_writer.write(
                        diagnostic_frame(
                            left,
                            right,
                            marker_ids,
                            left_markers,
                            right_markers,
                            left_diag,
                            right_diag,
                            frame_index,
                            args.corner_padding_ratio,
                        )
                    )
                if args.progress_every > 0 and (
                    (frame_index + 1) % args.progress_every == 0
                    or frame_index + 1 == args.frames
                ):
                    solved = sum(
                        row["status"] == "OK"
                        for row in frame_rows[-args.progress_every * len(subsets) :]
                    )
                    print(
                        f"  F{frame_index:03d} complete | recent solved subset rows: {solved}"
                    )
        finally:
            capture.release()
            if diagnostic_writer is not None:
                diagnostic_writer.release()

    if not frame_rows:
        raise RuntimeError("No frames were analyzed")
    corner_summaries = build_corner_summaries(corner_rows)
    video_summaries = build_video_summaries(
        frame_rows,
        corner_summaries,
        args.rotation_pass_deg,
        args.baseline_pass_percent,
    )
    distance_summaries = build_distance_summaries(
        frame_rows,
        video_summaries,
        args.rotation_pass_deg,
        args.baseline_pass_percent,
    )
    pattern_count_summaries = build_pattern_count_summaries(
        frame_rows,
        args.rotation_pass_deg,
        args.baseline_pass_percent,
        population="strict",
    )
    pattern_fallback_summaries = build_pattern_count_summaries(
        frame_rows,
        args.rotation_pass_deg,
        args.baseline_pass_percent,
        population="fallback",
    )
    pattern_all_solved_summaries = build_pattern_count_summaries(
        frame_rows,
        args.rotation_pass_deg,
        args.baseline_pass_percent,
        population="all",
    )
    available_trend_angles = sorted(
        {
            float(row.get("angle_deg") or 0.0)
            for row in pattern_count_summaries
            if finite(row.get("angle_deg") or 0.0) is not None
        }
    )
    if not available_trend_angles:
        available_trend_angles = [0.0]
    if args.trend_angle_deg is None:
        trend_angle = next(
            (
                angle
                for angle in available_trend_angles
                if math.isclose(angle, 0.0, abs_tol=1e-9)
            ),
            available_trend_angles[0],
        )
    else:
        matching_angle = next(
            (
                angle
                for angle in available_trend_angles
                if math.isclose(angle, args.trend_angle_deg, abs_tol=1e-9)
            ),
            None,
        )
        if matching_angle is None:
            choices = ", ".join(f"{angle:g}" for angle in available_trend_angles)
            raise ValueError(
                f"--trend-angle-deg {args.trend_angle_deg:g} is not present in "
                f"the analyzed data. Available angles: {choices}"
            )
        trend_angle = matching_angle

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    write_csv(prefix.with_name(prefix.name + "_frame_results.csv"), FRAME_FIELDS, frame_rows)
    write_csv(prefix.with_name(prefix.name + "_corner_frames.csv"), CORNER_FIELDS, corner_rows)
    write_csv(
        prefix.with_name(prefix.name + "_video_summary.csv"),
        VIDEO_SUMMARY_FIELDS,
        video_summaries,
    )
    write_csv(
        prefix.with_name(prefix.name + "_distance_summary.csv"),
        DISTANCE_SUMMARY_FIELDS,
        distance_summaries,
    )
    write_csv(
        prefix.with_name(prefix.name + "_pattern_count_summary.csv"),
        PATTERN_COUNT_SUMMARY_FIELDS,
        pattern_count_summaries,
    )
    write_csv(
        prefix.with_name(prefix.name + "_pattern_fallback_summary.csv"),
        PATTERN_COUNT_SUMMARY_FIELDS,
        pattern_fallback_summaries,
    )
    write_csv(
        prefix.with_name(prefix.name + "_pattern_all_solved_summary.csv"),
        PATTERN_COUNT_SUMMARY_FIELDS,
        pattern_all_solved_summaries,
    )
    write_csv(
        prefix.with_name(prefix.name + "_corner_stability.csv"),
        CORNER_SUMMARY_FIELDS,
        corner_summaries,
    )

    settings = [
        {"parameter": "marker_ids", "value": ",".join(map(str, marker_ids))},
        {"parameter": "marker_size_mm", "value": args.marker_size_mm},
        {"parameter": "frames_per_video", "value": args.frames},
        {"parameter": "rotation_pass_deg", "value": args.rotation_pass_deg},
        {
            "parameter": "absolute_baseline_error_pass_percent",
            "value": args.baseline_pass_percent,
        },
        {"parameter": "min_baseline_mm", "value": args.min_baseline_mm},
        {"parameter": "max_baseline_mm", "value": args.max_baseline_mm},
        {
            "parameter": "pattern_ransac_reprojection_threshold_px",
            "value": args.pattern_ransac_reproj_threshold_px,
        },
        {"parameter": "joint_refinement", "value": args.joint_refinement},
        {"parameter": "joint_max_iterations", "value": args.joint_max_iterations},
        {"parameter": "joint_huber_delta_px", "value": args.joint_huber_delta_px},
        {"parameter": "temporal_prior", "value": args.temporal_prior},
        {"parameter": "assume_coplanar", "value": args.assume_coplanar},
        {
            "parameter": "coplanar_normal_weight_px",
            "value": args.coplanar_normal_weight_px,
        },
        {
            "parameter": "coplanar_center_weight_px",
            "value": args.coplanar_center_weight_px,
        },
        {
            "parameter": "subpixel_stability_max_raw_px",
            "value": args.subpixel_stability_max_raw_px,
        },
        {
            "parameter": "aruco_initial_corner_refinement",
            "value": args.aruco_initial_refinement,
        },
        {
            "parameter": "trend_chart_angle_filter_deg",
            "value": trend_angle,
        },
        {
            "parameter": "pattern_count_primary_population",
            "value": SOLUTION_STRICT,
        },
    ]
    add_live_pass_formulas(
        frame_rows,
        video_summaries,
        distance_summaries,
        pattern_count_summaries,
        pattern_fallback_summaries,
        pattern_all_solved_summaries,
        settings,
    )
    protocol = [
        {
            "parameter": "generated_at",
            "value": datetime.now(timezone.utc).astimezone().isoformat(),
        },
        {"parameter": "source_folder", "value": str(folder)},
        {"parameter": "calibration_file", "value": str(calibration)},
        {"parameter": "selected_videos", "value": len(videos)},
        {"parameter": "marker_subset_count", "value": len(subsets)},
        {
            "parameter": "subset_definition",
            "value": "all non-empty combinations of four marker IDs: 4+6+4+1=15",
        },
        {
            "parameter": "corner_detection",
            "value": (
                f"CLAHE detection with OpenCV {args.aruco_initial_refinement} initial "
                "corner refinement; adaptive multi-window cornerSubPix on original gray; "
                "uncertainty combines window disagreement, subpixel shift, corner gradient, "
                "sharpness and marker pixel size; "
                f"reject above {args.subpixel_stability_max_raw_px:g} raw px disagreement"
            ),
        },
        {
            "parameter": "RT estimation",
            "value": (
                "per-marker left/right IPPE branches; pattern-level RANSAC seeds; "
                "true leave-one-marker-out training/validation; uncertainty-adaptive "
                f"transfer gate from nominal {args.pattern_ransac_reproj_threshold_px:g} px; "
                "robust bidirectional joint RT reprojection refinement"
            ),
        },
        {
            "parameter": "marker_layout_assumption",
            "value": (
                "experimental coplanar normal/centre constraint enabled; no known "
                "marker-to-marker distances are used"
                if args.assume_coplanar
                else "no known 3D transform or coplanarity between markers is used"
            ),
        },
        {
            "parameter": "temporal_prior",
            "value": (
                "previous solved frame of the same subset is an additional hypothesis; "
                "no temporal averaging and no JSON information"
                if args.temporal_prior
                else "disabled; every frame is selected independently"
            ),
        },
        {
            "parameter": "robust_subset_policy",
            "value": (
                "N1/N2 retain exhaustive branch search without rejection; "
                "N3 requires >=2 inlier markers; N4 requires >=3; failed consensus "
                "is reported as FAILED rather than falling back to a contaminated mean"
            ),
        },
        {
            "parameter": "Excel_population_policy",
            "value": (
                "Pattern Count Summary and Trend Charts use STRICT_ALL_INLIERS only; "
                "Pattern Fallback Summary contains successful rows that dropped one or "
                "more requested markers; Pattern All Solved preserves the legacy mixed view"
            ),
        },
        {
            "parameter": "evaluated_branch_combinations",
            "value": (
                "actual exhaustive count for N1/N2; theoretical full Cartesian count "
                "for N3/N4 (actual robust model count is robust_hypotheses_evaluated)"
            ),
        },
        {
            "parameter": "JSON usage",
            "value": "answer errors only after RT; never used for selection",
        },
        {
            "parameter": "pass-rate formulas",
            "value": (
                "Video/Distance/Pattern Count pass counts and rates use COUNTIFS "
                "against Frame Results and reference Settings rotation/baseline thresholds"
            ),
        },
        {
            "parameter": "trend_charts",
            "value": (
                "Trend Charts is generated from strict-all-inlier Pattern Count Summary; "
                "cell H2 is the angle filter and H5 shows the population"
            ),
        },
        {
            "parameter": "corner_video",
            "value": "2x2 crops per eye, LEFT and RIGHT side-by-side; no green polygon lines",
        },
    ]
    write_xlsx(
        output,
        [
            ("Settings", ["parameter", "value"], settings),
            ("Trend Charts", ["placeholder"], []),
            ("Frame Results", FRAME_FIELDS, frame_rows),
            ("Video Summary", VIDEO_SUMMARY_FIELDS, video_summaries),
            ("Distance Summary", DISTANCE_SUMMARY_FIELDS, distance_summaries),
            (
                "Pattern Count Summary",
                PATTERN_COUNT_SUMMARY_FIELDS,
                pattern_count_summaries,
            ),
            (
                "Pattern Fallback Summary",
                PATTERN_COUNT_SUMMARY_FIELDS,
                pattern_fallback_summaries,
            ),
            (
                "Pattern All Solved",
                PATTERN_COUNT_SUMMARY_FIELDS,
                pattern_all_solved_summaries,
            ),
            ("Corner Frames", CORNER_FIELDS, corner_rows),
            ("Corner Stability", CORNER_SUMMARY_FIELDS, corner_summaries),
            ("Protocol", ["parameter", "value"], protocol),
        ],
    )
    add_trend_charts_to_xlsx(
        output,
        pattern_count_summaries,
        trend_angle,
        sheet_index=2,
    )
    print(f"Excel: {output}")
    print(f"Trend Charts: angle filter = {trend_angle:g} deg")
    if not args.no_corner_videos:
        print(f"Corner videos: {corner_video_dir}")
    print(f"Elapsed: {(time.perf_counter() - overall_started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted")
        raise SystemExit(130)
