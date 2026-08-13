"""Evaluate independent monocular temporal pose estimators with ChArUco GT.

The input is one side-by-side (SBS) stereo video.  Left and right streams are
processed as two *independent monocular videos*.  Four test methods are run:

``ID2_TEMPORAL``
    ArUco ID2 IPPE branches + offline temporal branch continuity.
``ID5_TEMPORAL``
    ArUco ID5 IPPE branches + offline temporal branch continuity.
``ID2_ID5_TEMPORAL``
    ID2 and ID5 with an independently learned, fixed 6-DoF marker relation;
    the two markers are not assumed coplanar.
``ID2_SIFT_TEMPORAL``
    ID2 plus same-eye wound SIFT matches and temporal continuity.

The large ChArUco board is used only to estimate an independent pose for each
eye and frame.  The calibrated JSON stereo extrinsic is used only *after* all
poses have been estimated, for stereo-closure scoring.  It is never supplied
to a detector, IPPE branch selector, temporal estimator, SIFT optimizer, or
ChArUco pose solver.

Example (PowerShell)::

    python analyze_independent_monocular_temporal_pose_charuco_gt.py video.mp4 `
      --calibration calibration_result_HBVCAM_4M2214HD-2-v11.json `
      --max-frames 300 --reference-frame 0 `
      --sift-roi-left 500,250,800,600 --sift-roi-right 500,250,800,600 `
      --protect-sift-roi-dark

Pose convention throughout this file is ``X_out = R @ X_in + t``.  A PnP pose
is therefore ``T_camera_from_pattern``.  Temporal relative motion is
``P_current o inverse(P_reference)`` and synchronized stereo closure is
``P_right o inverse(P_left)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

from analyze_hbvcam_aruco_corner_rt_stability import (
    ExcelFormula,
    excel_column,
    write_xlsx,
)
from analyze_monocular_temporal_rt_stereo_grid_gt import (
    Pose,
    build_frame_scatter_chart_xml,
    chart_anchor_xml,
    clip_roi,
    compose,
    inverse,
    load_calibration,
    parse_roi,
    project_points,
    relative_pose,
    robust_lm,
    rotation_angle_deg,
    signed_sampson_residuals,
    split_sbs,
)


# =============================================================================
# User-editable defaults (CLI arguments override these values)
# =============================================================================

VIDEO_PATH = ""
CALIBRATION_PATH = "calibration_result_HBVCAM_4M2214HD-2-v11.json"
CHARUCO_METADATA_PATH = "charuco_a3_12x8_pattern.json"
ID2_METADATA_PATH = "aruco_id2_12_25mm_corner_blocks.json"
ID5_METADATA_PATH = "aruco_id5_12_25mm_corner_blocks.json"

START_FRAME = 0
END_FRAME = -1                   # -1 means the last frame.
MAX_FRAMES = 300                 # 0 means unlimited.
FRAME_STEP = 1
REFERENCE_FRAME = 0

# All ROIs use one-eye coordinates (x, y, width, height).  None = full eye.
LEFT_DETECTION_ROI: tuple[int, int, int, int] | None = None
RIGHT_DETECTION_ROI: tuple[int, int, int, int] | None = None
LEFT_SIFT_ROI: tuple[int, int, int, int] | None = None
RIGHT_SIFT_ROI: tuple[int, int, int, int] | None = None

# ChArUco GT quality gates.  These gate evaluation only; they never change a
# test-method estimate.
GT_MIN_CHARUCO_CORNERS = 12
GT_MIN_PNP_INLIERS = 10
GT_MIN_INLIER_RATE_PERCENT = 65.0
GT_MAX_REPROJECTION_RMS_PX = 1.5
GT_MAX_REPROJECTION_P95_PX = 2.5
GT_MIN_OBJECT_COVERAGE_PERCENT = 15.0
GT_MIN_X_SPAN_PERCENT = 30.0
GT_MIN_Y_SPAN_PERCENT = 30.0
GT_MIN_IMAGE_HULL_PERCENT = 1.0
GT_RANSAC_THRESHOLD_PX = 2.0

# JSON closure is reported separately from independent GT validity.  When this
# gate fails the frame remains in GT sheets, but is excluded from scored rows.
GT_CLOSURE_MAX_ROTATION_ERROR_DEG = 1.5
GT_CLOSURE_MAX_BASELINE_ERROR_PERCENT = 5.0

# Adjustable pass thresholds written to the Settings sheet.  Summary formulas
# reference these cells, so editing them in Excel recalculates pass rates.
STEREO_ROTATION_PASS_DEG = 2.0
STEREO_BASELINE_PASS_PERCENT = 5.0
TEMPORAL_ROTATION_PASS_DEG = 2.0
TEMPORAL_TRANSLATION_L2_PASS_MM = 3.0
MIN_GT_MOTION_FOR_PERCENT_MM = 1.0

# Temporal branch and trajectory settings.
TEMPORAL_BEAM_WIDTH = 6
TEMPORAL_REPROJECTION_WEIGHT = 1.0
TEMPORAL_ROTATION_JUMP_WEIGHT = 0.035
TEMPORAL_TRANSLATION_JUMP_WEIGHT = 0.025
TEMPORAL_ACCEL_ROTATION_WEIGHT = 0.08
TEMPORAL_ACCEL_TRANSLATION_WEIGHT = 0.04
TEMPORAL_SMOOTH_ALPHA = 0.12
TEMPORAL_SMOOTH_ITERATIONS = 2
TEMPORAL_MAX_POST_SMOOTH_MARKER_RMS_PX = 2.0

# ID2-ID5 relation consensus.  Relation is learned separately for each eye.
RELATION_ROTATION_GATE_DEG = 8.0
RELATION_TRANSLATION_GATE_MM = 10.0
RELATION_MIN_SUPPORT_FRAMES = 5
DUAL_MARKER_MAX_GROUP_RMS_PX = 2.5

# SIFT-only image preparation and matching.
SIFT_MAX_FEATURES = 1600
SIFT_RATIO = 0.75
SIFT_RANSAC_THRESHOLD_PX = 1.0
SIFT_MIN_MATCHES = 8
SIFT_MAX_STORED_INLIERS = 100
SIFT_MIN_INLIER_RATE_PERCENT = 25.0
SIFT_MIN_PARALLAX_DEG = 0.25
SIFT_MAX_SAMPSON_P90_PX = 1.5
SIFT_DESCRIPTOR_MIN_CLEARANCE_PX = 8.0
SIFT_DESCRIPTOR_SIZE_FACTOR = 1.5
SIFT_GRID_THRESHOLD_MODE = "otsu"   # otsu or fixed
SIFT_GRID_BLACK_THRESHOLD = 150
SIFT_GRID_THRESHOLD_MIN = 55
SIFT_GRID_THRESHOLD_MAX = 190
SIFT_GRID_DILATION_PX = 2
SIFT_FORBIDDEN_DILATION_PX = 8
SIFT_TARGET_EXTRA_SCALE = 1.15
SIFT_MARKER_WEIGHTS = (12.0, 4.0, 1.0)
SIFT_JOINT_MAX_MARKER_RMS_PX = 2.0
SIFT_JOINT_MAX_HOLDOUT_MEDIAN_PX = 2.0
SIFT_REQUIRE_BOTH_TARGET_MASKS = True

SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_MAX_WIDTH = 1600

METHODS = (
    "ID2_TEMPORAL",
    "ID5_TEMPORAL",
    "ID2_ID5_TEMPORAL",
    "ID2_SIFT_TEMPORAL",
)
EYES = ("L", "R")
METHOD_COLORS = {
    "ID2_TEMPORAL": "4472C4",
    "ID5_TEMPORAL": "ED7D31",
    "ID2_ID5_TEMPORAL": "70AD47",
    "ID2_SIFT_TEMPORAL": "7030A0",
}


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class MarkerPoseCandidate:
    branch: int
    pose: Pose
    reprojection_rms_px: float
    reprojection_max_px: float


@dataclass
class MarkerObservation:
    marker_id: int
    corners: np.ndarray
    side_mean_px: float
    area_px2: float
    candidates: list[MarkerPoseCandidate]


@dataclass
class GtPoseResult:
    pose: Pose | None = None
    pose_available: bool = False
    valid: bool = False
    status: str = "NOT_PROCESSED"
    reject_reason: str = ""
    marker_count: int = 0
    corner_count: int = 0
    inlier_count: int = 0
    inlier_rate_percent: float | None = None
    object_coverage_percent: float | None = None
    x_span_percent: float | None = None
    y_span_percent: float | None = None
    image_hull_percent: float | None = None
    reprojection_rms_px: float | None = None
    reprojection_median_px: float | None = None
    reprojection_p95_px: float | None = None
    reprojection_max_px: float | None = None
    min_depth_mm: float | None = None
    solver: str = ""
    charuco_corners: np.ndarray | None = None
    charuco_ids: np.ndarray | None = None
    inlier_indices: np.ndarray | None = None
    board_polygon: np.ndarray | None = None
    processing_time_ms: float | None = None


@dataclass
class MaskDiagnostics:
    valid: bool = False
    status: str = "NO_GRID_POSE"
    threshold_value: float | None = None
    board_pixels: int = 0
    source_dark_pixels: int = 0
    whitened_pixels: int = 0
    whitened_area_percent: float | None = None
    target_forbidden_pixels: int = 0
    target_mask_complete: bool = False
    missing_target_ids: str = ""
    sift_keypoints: int = 0
    grid_leakage_keypoints: int = 0


@dataclass
class FeatureFrame:
    keypoints: list
    descriptors: np.ndarray | None
    points: np.ndarray
    leakage_flags: np.ndarray


@dataclass
class FeatureEdge:
    previous_frame: int
    current_frame: int
    points_previous: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), np.float64)
    )
    points_current: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), np.float64)
    )
    recovered_pose: Pose | None = None
    previous_keypoints: int = 0
    current_keypoints: int = 0
    mutual_matches: int = 0
    essential_inliers: int = 0
    inlier_rate_percent: float | None = None
    sampson_median_px: float | None = None
    sampson_p90_px: float | None = None
    parallax_median_deg: float | None = None
    grid_leakage_matches: int = 0
    status: str = "NO_MATCHES"


@dataclass
class EyeFrameObservation:
    frame_index: int
    timestamp_s: float
    gt: GtPoseResult
    markers: dict[int, MarkerObservation]
    mask: MaskDiagnostics
    sift_keypoints: int = 0
    sift_grid_leakage_keypoints: int = 0


@dataclass
class MethodPoseResult:
    pose: Pose | None = None
    raw_pose: Pose | None = None
    available: bool = False
    status: str = "UNAVAILABLE"
    reject_reason: str = ""
    source: str = ""
    marker_id: int | None = None
    id2_found: bool = False
    id5_found: bool = False
    selected_id2_branch: int | None = None
    selected_id5_branch: int | None = None
    marker_reprojection_rms_px: float | None = None
    marker_reprojection_max_px: float | None = None
    relation_rotation_residual_deg: float | None = None
    relation_translation_residual_mm: float | None = None
    temporal_prior_used: bool = False
    prediction_rotation_residual_deg: float | None = None
    prediction_translation_residual_mm: float | None = None
    sift_used: bool = False
    sift_inliers: int = 0
    sift_sampson_median_px: float | None = None
    sift_holdout_median_px: float | None = None
    processing_time_ms: float | None = None


@dataclass
class RelationEstimate:
    pose: Pose | None = None
    available: bool = False
    status: str = "UNAVAILABLE"
    candidate_frames: int = 0
    support_frames: int = 0
    rotation_residual_median_deg: float | None = None
    rotation_residual_p95_deg: float | None = None
    translation_residual_median_mm: float | None = None
    translation_residual_p95_mm: float | None = None
    selected_pairs: dict[int, tuple[int, int, Pose]] = field(default_factory=dict)


# =============================================================================
# Rigid transforms and numeric helpers
# =============================================================================


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def percentile(values, q):
    values = [float(value) for value in values if finite(value) is not None]
    return float(np.percentile(values, q)) if values else None


def translation_direction_error_deg(first, second):
    first = np.asarray(first, np.float64).reshape(3)
    second = np.asarray(second, np.float64).reshape(3)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.clip(first @ second / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pose_distance(first: Pose, second: Pose) -> tuple[float, float]:
    return (
        rotation_angle_deg(first.R @ second.R.T),
        float(np.linalg.norm(first.t - second.t)),
    )


def pose_to_columns(pose: Pose | None) -> dict:
    if pose is None:
        return {
            **{f"rvec_{axis}": None for axis in "xyz"},
            **{f"tvec_{axis}_mm": None for axis in "xyz"},
            **{f"R{row}{column}": None for row in range(3) for column in range(3)},
        }
    rvec = cv2.Rodrigues(pose.R)[0].reshape(3)
    tvec = pose.t.reshape(3)
    return {
        "rvec_x": float(rvec[0]),
        "rvec_y": float(rvec[1]),
        "rvec_z": float(rvec[2]),
        "tvec_x_mm": float(tvec[0]),
        "tvec_y_mm": float(tvec[1]),
        "tvec_z_mm": float(tvec[2]),
        **{
            f"R{row}{column}": float(pose.R[row, column])
            for row in range(3)
            for column in range(3)
        },
    }


def answer_errors(estimate: Pose, answer: Pose) -> dict:
    estimated_baseline = float(np.linalg.norm(estimate.t))
    answer_baseline = float(np.linalg.norm(answer.t))
    delta = estimated_baseline - answer_baseline
    return {
        "rotation_error_deg": rotation_angle_deg(estimate.R @ answer.R.T),
        "estimated_baseline_mm": estimated_baseline,
        "json_baseline_mm": answer_baseline,
        "baseline_delta_mm": delta,
        "absolute_baseline_error_percent": (
            abs(delta) / answer_baseline * 100.0
            if answer_baseline > 1e-12
            else None
        ),
        "translation_l2_error_mm": float(np.linalg.norm(estimate.t - answer.t)),
        "translation_direction_error_deg": translation_direction_error_deg(
            estimate.t, answer.t
        ),
    }


def temporal_errors(estimate: Pose, answer: Pose, min_motion_mm: float) -> dict:
    estimated_norm = float(np.linalg.norm(estimate.t))
    answer_norm = float(np.linalg.norm(answer.t))
    delta = estimated_norm - answer_norm
    return {
        "estimated_rotation_angle_deg": rotation_angle_deg(estimate.R),
        "gt_rotation_angle_deg": rotation_angle_deg(answer.R),
        "rotation_error_deg": rotation_angle_deg(estimate.R @ answer.R.T),
        "estimated_translation_norm_mm": estimated_norm,
        "gt_translation_norm_mm": answer_norm,
        "translation_norm_delta_mm": delta,
        "absolute_translation_norm_error_percent": (
            abs(delta) / answer_norm * 100.0
            if answer_norm >= min_motion_mm
            else None
        ),
        "translation_l2_error_mm": float(np.linalg.norm(estimate.t - answer.t)),
        "translation_direction_error_deg": translation_direction_error_deg(
            estimate.t, answer.t
        ),
    }


def square_object_points(size_mm: float) -> np.ndarray:
    half = float(size_mm) * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        np.float64,
    )


def marker_metrics(corners) -> tuple[float, float]:
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    return float(np.mean(sides)), area


def expanded_quad(corners, scale) -> np.ndarray:
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    centre = np.mean(corners, axis=0)
    return centre + (corners - centre) * float(scale)


def project_board_polygon(pose, K, distortion, width_mm, height_mm):
    outer = np.asarray(
        [[0, 0, 0], [width_mm, 0, 0], [width_mm, height_mm, 0], [0, height_mm, 0]],
        np.float64,
    )
    projected = project_points(outer, pose, K, distortion)
    return projected.astype(np.float64)


def rotation_midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = cv2.Rodrigues(second @ first.T)[0].reshape(3) * 0.5
    return cv2.Rodrigues(delta)[0] @ first


def blend_pose(current: Pose, target: Pose, alpha: float) -> Pose:
    delta = cv2.Rodrigues(target.R @ current.R.T)[0].reshape(3) * float(alpha)
    return Pose(
        cv2.Rodrigues(delta)[0] @ current.R,
        current.t + float(alpha) * (target.t - current.t),
    )


def smooth_pose_path(path: dict[int, Pose], alpha, iterations) -> dict[int, Pose]:
    result = {index: Pose(pose.R.copy(), pose.t.copy()) for index, pose in path.items()}
    indexes = sorted(result)
    if len(indexes) < 3 or alpha <= 0 or iterations <= 0:
        return result
    for _ in range(int(iterations)):
        updated = dict(result)
        for position in range(1, len(indexes) - 1):
            previous_index = indexes[position - 1]
            index = indexes[position]
            next_index = indexes[position + 1]
            if index - previous_index != next_index - index:
                continue
            previous = result[previous_index]
            current = result[index]
            following = result[next_index]
            target = Pose(
                rotation_midpoint(previous.R, following.R),
                (previous.t + following.t) * 0.5,
            )
            updated[index] = blend_pose(current, target, alpha)
        result = updated
    return result


def predict_pose(history: list[tuple[int, Pose]], frame_index: int) -> Pose | None:
    if not history:
        return None
    if len(history) == 1:
        return history[-1][1]
    first_index, first = history[-2]
    second_index, second = history[-1]
    gap = second_index - first_index
    if gap <= 0:
        return second
    ratio = (frame_index - second_index) / gap
    # Constant rigid-body increment in SE(3).  Translation cannot be
    # extrapolated independently of rotation: D = T2 o inv(T1), then the
    # fractional increment D**ratio is left-composed onto T2.
    increment = compose(second, inverse(first))
    return compose(se3_power(increment, ratio), second)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], np.float64)


def se3_left_jacobian(rotation_vector: np.ndarray) -> np.ndarray:
    omega = np.asarray(rotation_vector, np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    matrix = skew(omega)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * matrix + (matrix @ matrix) / 6.0
    return (
        np.eye(3)
        + (1.0 - math.cos(theta)) / (theta * theta) * matrix
        + (theta - math.sin(theta)) / (theta * theta * theta) * (matrix @ matrix)
    )


def se3_power(pose: Pose, exponent: float) -> Pose:
    """Raise an SE(3) transform to a real power through log/exp coordinates."""
    omega = cv2.Rodrigues(pose.R)[0].reshape(3)
    jacobian = se3_left_jacobian(omega)
    try:
        velocity = np.linalg.solve(jacobian, pose.t.reshape(3))
    except np.linalg.LinAlgError:
        velocity = np.linalg.pinv(jacobian) @ pose.t.reshape(3)
    scaled_omega = omega * float(exponent)
    scaled_translation = (
        se3_left_jacobian(scaled_omega)
        @ (velocity * float(exponent))
    ).reshape(3, 1)
    return Pose(cv2.Rodrigues(scaled_omega)[0], scaled_translation)


# =============================================================================
# Metadata, calibration and detection
# =============================================================================


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_input(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def validate_metadata(charuco_meta, id2_meta, id5_meta):
    if charuco_meta.get("pattern_type") != "charuco":
        raise ValueError("ChArUco metadata pattern_type must be 'charuco'")
    required = {
        "dictionary",
        "squares_x",
        "squares_y",
        "square_length_mm",
        "marker_length_mm",
        "marker_ids",
        "legacy_pattern",
    }
    missing = required - set(charuco_meta)
    if missing:
        raise ValueError(f"ChArUco metadata is missing: {sorted(missing)}")
    for expected_id, metadata in ((2, id2_meta), (5, id5_meta)):
        if int(metadata.get("aruco_id", -1)) != expected_id:
            raise ValueError(f"Small-pattern metadata must describe ID{expected_id}")
        if metadata.get("dictionary") != charuco_meta.get("dictionary"):
            raise ValueError("Large and small patterns must use the same dictionary")
        size = metadata.get("central_aruco", {}).get("size_mm", [])
        if len(size) != 2 or not np.allclose(size, size[0]):
            raise ValueError(f"ID{expected_id} central ArUco must be square")


def create_charuco_board(metadata):
    dictionary_name = str(metadata["dictionary"])
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )
    marker_ids = np.asarray(metadata["marker_ids"], np.int32).reshape(-1)
    board = cv2.aruco.CharucoBoard(
        (int(metadata["squares_x"]), int(metadata["squares_y"])),
        float(metadata["square_length_mm"]),
        float(metadata["marker_length_mm"]),
        dictionary,
        marker_ids,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(metadata.get("legacy_pattern", False)))
    if not np.array_equal(board.getIds().reshape(-1), marker_ids):
        raise RuntimeError("OpenCV ChArUco board IDs do not match metadata")
    return board, dictionary


class EyeDetector:
    """Detect the board and targets with refinement suited to each scale.

    ChArUco interpolation keeps the conventional SUBPIX detector.  ID2/ID5
    are redetected with APRILTAG corner refinement because the auxiliary black
    corner blocks can bias a plain cornerSubPix window toward the wrong edge.
    """

    def __init__(self, board, dictionary, K, distortion):
        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_parameters.cornerRefinementWinSize = 5
        detector_parameters.cornerRefinementMaxIterations = 50
        detector_parameters.cornerRefinementMinAccuracy = 0.001
        self.aruco = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

        small_parameters = cv2.aruco.DetectorParameters()
        small_parameters.cornerRefinementMethod = getattr(
            cv2.aruco,
            "CORNER_REFINE_APRILTAG",
            cv2.aruco.CORNER_REFINE_SUBPIX,
        )
        small_parameters.cornerRefinementMaxIterations = 60
        small_parameters.cornerRefinementMinAccuracy = 0.0001
        self.small_aruco = cv2.aruco.ArucoDetector(dictionary, small_parameters)

        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_parameters.cameraMatrix = np.asarray(K, np.float64).copy()
        charuco_parameters.distCoeffs = np.asarray(distortion, np.float64).reshape(1, -1)
        charuco_parameters.tryRefineMarkers = True
        if hasattr(charuco_parameters, "checkMarkers"):
            charuco_parameters.checkMarkers = True
        if hasattr(charuco_parameters, "minMarkers"):
            charuco_parameters.minMarkers = 2
        self.charuco = cv2.aruco.CharucoDetector(
            board, charuco_parameters, detector_parameters
        )

    def detect(self, gray, roi):
        x, y, width, height = clip_roi(roi, gray.shape)
        crop = gray[y : y + height, x : x + width]
        marker_corners, marker_ids, rejected = self.aruco.detectMarkers(crop)
        shifted = []
        for corners in marker_corners or []:
            points = np.asarray(corners, np.float32).reshape(4, 1, 2).copy()
            flat = points.reshape(4, 2)
            side = float(
                np.mean(
                    np.linalg.norm(
                        np.roll(flat, -1, axis=0) - flat,
                        axis=1,
                    )
                )
            )
            window = int(np.clip(round(side / 12.0), 2, 9))
            try:
                cv2.cornerSubPix(
                    crop,
                    points,
                    (window, window),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        60,
                        0.0001,
                    ),
                )
            except cv2.error:
                pass
            points[:, 0, 0] += x
            points[:, 0, 1] += y
            shifted.append(points)
        marker_corners = shifted
        if marker_ids is not None and len(marker_corners):
            try:
                charuco_corners, charuco_ids, _, _ = self.charuco.detectBoard(
                    gray, None, None, marker_corners, marker_ids
                )
            except cv2.error:
                charuco_corners, charuco_ids = None, None
        else:
            charuco_corners, charuco_ids = None, None

        if charuco_corners is not None and len(charuco_corners):
            points = np.asarray(charuco_corners, np.float32).reshape(-1, 1, 2)
            height_full, width_full = gray.shape[:2]
            inside = (
                (points[:, 0, 0] >= 6)
                & (points[:, 0, 0] < width_full - 6)
                & (points[:, 0, 1] >= 6)
                & (points[:, 0, 1] < height_full - 6)
            )
            if np.any(inside):
                refined = points[inside].copy()
                try:
                    cv2.cornerSubPix(
                        gray,
                        refined,
                        (5, 5),
                        (-1, -1),
                        (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            50,
                            0.001,
                        ),
                    )
                    points[inside] = refined
                except cv2.error:
                    pass
            charuco_corners = points

        # Retain all board detections from the SUBPIX pass, but replace ID2/ID5
        # with the best APRILTAG-refined detection.  If the second detector
        # misses a target, keep the first pass rather than dropping it.
        merged_corners = []
        merged_ids = []
        target_best = {}
        if marker_ids is not None:
            for corners, marker_id in zip(marker_corners, marker_ids.reshape(-1)):
                marker_id = int(marker_id)
                if marker_id in (2, 5):
                    area = abs(float(cv2.contourArea(np.asarray(corners, np.float32).reshape(4, 2))))
                    if marker_id not in target_best or area > target_best[marker_id][0]:
                        target_best[marker_id] = (area, corners)
                else:
                    merged_corners.append(corners)
                    merged_ids.append(marker_id)

        small_corners, small_ids, _small_rejected = self.small_aruco.detectMarkers(crop)
        if small_ids is not None:
            for corners, marker_id in zip(small_corners, small_ids.reshape(-1)):
                marker_id = int(marker_id)
                if marker_id not in (2, 5):
                    continue
                points = np.asarray(corners, np.float32).reshape(4, 1, 2).copy()
                points[:, 0, 0] += x
                points[:, 0, 1] += y
                area = abs(float(cv2.contourArea(points.reshape(4, 2))))
                if marker_id not in target_best or area >= target_best[marker_id][0] * 0.5:
                    target_best[marker_id] = (area, points)
        for marker_id in (2, 5):
            if marker_id in target_best:
                merged_corners.append(target_best[marker_id][1])
                merged_ids.append(marker_id)
        merged_ids_array = (
            np.asarray(merged_ids, np.int32).reshape(-1, 1) if merged_ids else None
        )
        return (
            charuco_corners,
            charuco_ids,
            merged_corners,
            merged_ids_array,
            rejected,
        )


def marker_observations(marker_corners, marker_ids, sizes, K, distortion):
    result: dict[int, MarkerObservation] = {}
    if marker_ids is None:
        return result
    for raw_corners, marker_id in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id not in sizes:
            continue
        corners = np.asarray(raw_corners, np.float64).reshape(4, 2)
        side, area = marker_metrics(corners)
        candidates = ippe_pose_candidates(corners, sizes[marker_id], K, distortion)
        observation = MarkerObservation(marker_id, corners, side, area, candidates)
        existing = result.get(marker_id)
        if existing is None or observation.area_px2 > existing.area_px2:
            result[marker_id] = observation
    return result


def ippe_pose_candidates(corners, size_mm, K, distortion):
    object_points = square_object_points(size_mm)
    try:
        _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points,
            np.asarray(corners, np.float64).reshape(4, 2),
            K,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    result = []
    for branch, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        pose = Pose(cv2.Rodrigues(rvec)[0], tvec)
        camera_points = (pose.R @ object_points.T + pose.t).T
        if np.min(camera_points[:, 2]) <= 0:
            continue
        projected = project_points(object_points, pose, K, distortion)
        errors = np.linalg.norm(projected - np.asarray(corners).reshape(4, 2), axis=1)
        result.append(
            MarkerPoseCandidate(
                branch,
                pose,
                float(np.sqrt(np.mean(np.square(errors)))),
                float(np.max(errors)),
            )
        )
    return sorted(result, key=lambda item: item.reprojection_rms_px)


# =============================================================================
# Independent per-eye ChArUco ground truth
# =============================================================================


def convex_hull_area(points) -> float:
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(cv2.convexHull(points))))


def solve_charuco_gt(
    board,
    metadata,
    charuco_corners,
    charuco_ids,
    marker_ids,
    K,
    distortion,
    image_shape,
    args,
) -> GtPoseResult:
    started = time.perf_counter()
    result = GtPoseResult()
    board_marker_ids = set(map(int, metadata["marker_ids"]))
    if marker_ids is not None:
        result.marker_count = sum(
            int(marker_id) in board_marker_ids for marker_id in marker_ids.reshape(-1)
        )
    if charuco_corners is None or charuco_ids is None:
        result.status = "NO_CHARUCO"
        result.reject_reason = "No ChArUco corners were interpolated"
        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        return result

    corners = np.asarray(charuco_corners, np.float32).reshape(-1, 1, 2)
    ids = np.asarray(charuco_ids, np.int32).reshape(-1, 1)
    result.corner_count = len(corners)
    result.charuco_corners = corners.reshape(-1, 2).astype(np.float64)
    result.charuco_ids = ids.reshape(-1).copy()
    if len(corners) < 4:
        result.status = "TOO_FEW_CORNERS_FOR_PNP"
        result.reject_reason = f"Only {len(corners)} ChArUco corners"
        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        return result

    try:
        object_points, image_points = board.matchImagePoints(corners, ids)
    except cv2.error as error:
        result.status = "MATCH_IMAGE_POINTS_FAILED"
        result.reject_reason = str(error).splitlines()[0]
        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        return result
    object_points = np.asarray(object_points, np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, np.float64).reshape(-1, 2)

    try:
        ok, rvec_seed, tvec_seed, inlier_indices = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=250,
            reprojectionError=float(args.gt_ransac_threshold_px),
            confidence=0.999,
        )
    except cv2.error as error:
        ok, inlier_indices = False, None
        result.reject_reason = str(error).splitlines()[0]
    if not ok or inlier_indices is None or len(inlier_indices) < 4:
        result.status = "PNP_RANSAC_FAILED"
        if not result.reject_reason:
            result.reject_reason = "solvePnPRansac returned no usable solution"
        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        return result

    inlier_indices = np.asarray(inlier_indices, np.int32).reshape(-1)
    inlier_objects = object_points[inlier_indices]
    inlier_images = image_points[inlier_indices]
    result.inlier_indices = inlier_indices.copy()
    result.inlier_count = len(inlier_indices)
    result.inlier_rate_percent = len(inlier_indices) / len(object_points) * 100.0

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("RANSAC_ITERATIVE", np.asarray(rvec_seed, np.float64), np.asarray(tvec_seed, np.float64))
    ]
    if len(inlier_objects) >= 4:
        try:
            _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
                inlier_objects,
                inlier_images,
                K,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            candidates.extend(
                (f"IPPE_{index}", np.asarray(rvec, np.float64), np.asarray(tvec, np.float64))
                for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs))
            )
        except cv2.error:
            pass

    board_width = float(metadata["squares_x"]) * float(metadata["square_length_mm"])
    board_height = float(metadata["squares_y"]) * float(metadata["square_length_mm"])
    outer_points = np.asarray(
        [[0, 0, 0], [board_width, 0, 0], [board_width, board_height, 0], [0, board_height, 0]],
        np.float64,
    )
    evaluated = []
    for solver, rvec, tvec in candidates:
        try:
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_objects,
                    inlier_images,
                    K,
                    distortion,
                    rvec.copy(),
                    tvec.copy(),
                )
        except cv2.error:
            pass
        pose = Pose(cv2.Rodrigues(rvec)[0], tvec)
        depths = (pose.R @ outer_points.T + pose.t).T[:, 2]
        if np.min(depths) <= 0:
            continue
        projected = project_points(object_points, pose, K, distortion)
        errors = np.linalg.norm(projected - image_points, axis=1)
        inlier_errors = errors[inlier_indices]
        score = float(np.sqrt(np.mean(np.square(inlier_errors))))
        score += 0.05 * float(np.percentile(errors, 95))
        evaluated.append((score, solver, pose, errors, depths))
    if not evaluated:
        result.status = "NO_POSITIVE_DEPTH_SOLUTION"
        result.reject_reason = "All ChArUco PnP candidates place the board behind the camera"
        result.processing_time_ms = (time.perf_counter() - started) * 1000.0
        return result

    _score, solver, pose, errors, depths = min(evaluated, key=lambda item: item[0])
    result.pose = pose
    result.pose_available = True
    result.solver = solver
    result.min_depth_mm = float(np.min(depths))
    result.reprojection_rms_px = float(
        np.sqrt(np.mean(np.square(errors[inlier_indices])))
    )
    result.reprojection_median_px = float(np.median(errors[inlier_indices]))
    result.reprojection_p95_px = float(np.percentile(errors[inlier_indices], 95))
    result.reprojection_max_px = float(np.max(errors[inlier_indices]))

    all_board_corners = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    full_xy = all_board_corners[:, :2]
    used_xy = object_points[inlier_indices, :2]
    full_hull_area = convex_hull_area(full_xy)
    result.object_coverage_percent = (
        convex_hull_area(used_xy) / full_hull_area * 100.0
        if full_hull_area > 0
        else 0.0
    )
    full_x_span = float(np.ptp(full_xy[:, 0]))
    full_y_span = float(np.ptp(full_xy[:, 1]))
    result.x_span_percent = (
        float(np.ptp(used_xy[:, 0])) / full_x_span * 100.0 if full_x_span > 0 else 0.0
    )
    result.y_span_percent = (
        float(np.ptp(used_xy[:, 1])) / full_y_span * 100.0 if full_y_span > 0 else 0.0
    )
    image_area = float(image_shape[0] * image_shape[1])
    result.image_hull_percent = (
        convex_hull_area(image_points[inlier_indices]) / image_area * 100.0
        if image_area > 0
        else 0.0
    )
    result.board_polygon = project_board_polygon(
        pose, K, distortion, board_width, board_height
    )

    rejection_reasons = []
    gates = (
        (result.corner_count >= args.gt_min_charuco_corners,
         f"corners {result.corner_count} < {args.gt_min_charuco_corners}"),
        (result.inlier_count >= args.gt_min_pnp_inliers,
         f"inliers {result.inlier_count} < {args.gt_min_pnp_inliers}"),
        (result.inlier_rate_percent >= args.gt_min_inlier_rate_percent,
         f"inlier rate {result.inlier_rate_percent:.1f}% < {args.gt_min_inlier_rate_percent:.1f}%"),
        (result.reprojection_rms_px <= args.gt_max_reprojection_rms_px,
         f"RMS {result.reprojection_rms_px:.3f}px > {args.gt_max_reprojection_rms_px:.3f}px"),
        (result.reprojection_p95_px <= args.gt_max_reprojection_p95_px,
         f"P95 {result.reprojection_p95_px:.3f}px > {args.gt_max_reprojection_p95_px:.3f}px"),
        (result.object_coverage_percent >= args.gt_min_object_coverage_percent,
         f"coverage {result.object_coverage_percent:.1f}% < {args.gt_min_object_coverage_percent:.1f}%"),
        (result.x_span_percent >= args.gt_min_x_span_percent,
         f"X span {result.x_span_percent:.1f}% < {args.gt_min_x_span_percent:.1f}%"),
        (result.y_span_percent >= args.gt_min_y_span_percent,
         f"Y span {result.y_span_percent:.1f}% < {args.gt_min_y_span_percent:.1f}%"),
        (result.image_hull_percent >= args.gt_min_image_hull_percent,
         f"image hull {result.image_hull_percent:.2f}% < {args.gt_min_image_hull_percent:.2f}%"),
    )
    rejection_reasons.extend(message for passed, message in gates if not passed)
    result.valid = not rejection_reasons
    result.status = "OK" if result.valid else "QUALITY_GATE_FAILED"
    result.reject_reason = "; ".join(rejection_reasons)
    result.processing_time_ms = (time.perf_counter() - started) * 1000.0
    return result


# =============================================================================
# SIFT sanitization and same-eye feature edges
# =============================================================================


def roi_mask(shape, roi):
    mask = np.zeros(shape[:2], np.uint8)
    x, y, width, height = clip_roi(roi, shape)
    mask[y : y + height, x : x + width] = 255
    return mask


def create_sift_image_and_mask(
    gray,
    gt: GtPoseResult,
    markers: dict[int, MarkerObservation],
    sift_roi,
    target_scales,
    args,
):
    sanitized = gray.copy()
    diagnostics = MaskDiagnostics()
    allowed = roi_mask(gray.shape, sift_roi)
    board_mask = np.zeros(gray.shape, np.uint8)
    # Only a GT pose that passed every ChArUco quality gate may define the
    # whitening polygon.  A numerically available but rejected pose could move
    # the mask onto wound texture or leave Grid texture exposed.
    if gt.valid and gt.board_polygon is not None:
        polygon = np.rint(gt.board_polygon).astype(np.int32)
        if len(polygon) == 4:
            cv2.fillConvexPoly(board_mask, polygon, 255)
            diagnostics.valid = True
            diagnostics.status = "OK"
    diagnostics.board_pixels = int(np.count_nonzero(board_mask))

    risk_mask = np.zeros_like(board_mask)
    erase_mask = np.zeros_like(board_mask)
    if diagnostics.valid and diagnostics.board_pixels:
        board_values = gray[board_mask > 0]
        if args.sift_grid_threshold_mode == "otsu":
            threshold, _ = cv2.threshold(
                board_values.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            threshold = float(
                np.clip(
                    threshold,
                    args.sift_grid_threshold_min,
                    args.sift_grid_threshold_max,
                )
            )
        else:
            threshold = float(args.sift_grid_black_threshold)
        diagnostics.threshold_value = threshold
        raw_dark = ((gray <= threshold) & (board_mask > 0)).astype(np.uint8) * 255
        diagnostics.source_dark_pixels = int(np.count_nonzero(raw_dark))
        kernel = np.ones((3, 3), np.uint8)
        if args.sift_grid_dilation_px > 0:
            erase_mask = cv2.dilate(
                raw_dark, kernel, iterations=int(args.sift_grid_dilation_px)
            )
        else:
            erase_mask = raw_dark
        erase_mask[board_mask == 0] = 0
        risk_mask = cv2.dilate(
            raw_dark,
            kernel,
            iterations=max(
                int(args.sift_grid_dilation_px),
                int(args.sift_forbidden_dilation_px),
            ),
        )
        risk_mask[board_mask == 0] = 0
        # Also forbid descriptors centred on the projected board perimeter;
        # otherwise a white board edge against the background could become a
        # GT-derived SIFT feature even after all black cells were whitened.
        cv2.polylines(
            risk_mask,
            [np.rint(gt.board_polygon).astype(np.int32).reshape(-1, 1, 2)],
            True,
            255,
            max(3, 2 * int(args.sift_forbidden_dilation_px) + 1),
            cv2.LINE_8,
        )

        # Optional explicit SIFT ROI protection preserves dark wound texture.
        # It may also preserve visible Grid fragments, hence risk/leakage is
        # still measured against the pre-protection mask.
        protection = None
        if args.protect_sift_roi_dark and sift_roi is not None:
            protection = roi_mask(gray.shape, sift_roi)
            erase_mask[protection > 0] = 0
        sanitized[erase_mask > 0] = 255
        diagnostics.whitened_pixels = int(np.count_nonzero(erase_mask))
        diagnostics.whitened_area_percent = (
            diagnostics.whitened_pixels / diagnostics.board_pixels * 100.0
            if diagnostics.board_pixels
            else 0.0
        )

        # A descriptor centred just outside the whitened pixels can still read
        # a Grid edge.  The wider risk mask is therefore also a forbidden
        # keypoint region.  An explicitly protected wound ROI is the only
        # exception; retained features there remain counted as leakage risk so
        # the diagnostic sheet/video makes that trade-off visible.
        grid_forbidden = risk_mask.copy()
        if protection is not None:
            grid_forbidden[protection > 0] = 0
        allowed[grid_forbidden > 0] = 0

    target_forbidden = np.zeros_like(board_mask)
    for marker_id, observation in markers.items():
        scale = float(target_scales.get(marker_id, 1.0)) * float(
            args.sift_target_extra_scale
        )
        polygon = np.rint(expanded_quad(observation.corners, scale)).astype(np.int32)
        cv2.fillConvexPoly(target_forbidden, polygon, 255)
    if args.sift_forbidden_dilation_px > 0:
        target_forbidden = cv2.dilate(
            target_forbidden,
            np.ones((3, 3), np.uint8),
            iterations=int(args.sift_forbidden_dilation_px),
        )
    diagnostics.target_forbidden_pixels = int(np.count_nonzero(target_forbidden))
    sanitized[target_forbidden > 0] = 255
    allowed[target_forbidden > 0] = 0

    missing_target_ids = sorted({2, 5} - set(markers))
    diagnostics.target_mask_complete = not missing_target_ids
    diagnostics.missing_target_ids = ",".join(map(str, missing_target_ids))
    if args.sift_require_both_target_masks and missing_target_ids:
        # Do not let a temporarily missed ID5 become a wound-SIFT feature.
        # This conservative first version disables SIFT for that frame instead
        # of guessing a stale target position after camera motion.
        allowed[:] = 0
        diagnostics.status = (
            "TARGET_MASK_INCOMPLETE"
            if diagnostics.status == "OK"
            else diagnostics.status + ";TARGET_MASK_INCOMPLETE"
        )

    # If no reliable board polygon exists, SIFT is disabled for this frame to
    # prevent accidental use of ChArUco texture.  Marker-only methods continue.
    if not diagnostics.valid:
        allowed[:] = 0
    return sanitized, allowed, risk_mask, diagnostics


class SameEyeSift:
    def __init__(self, args):
        self.args = args
        self.extractor = cv2.SIFT_create(nfeatures=int(args.sift_max_features))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, gray, allowed_mask, risk_mask) -> FeatureFrame:
        keypoints = self.extractor.detect(gray, allowed_mask)
        if keypoints:
            # A SIFT descriptor reads a neighborhood much larger than its
            # centre.  Require its whole support to stay away from forbidden
            # Grid/target/ROI boundaries, scaled by the detected keypoint size.
            clearance = cv2.distanceTransform(
                (allowed_mask > 0).astype(np.uint8), cv2.DIST_L2, 3
            )
            filtered = []
            for keypoint in keypoints:
                column = int(np.clip(round(keypoint.pt[0]), 0, gray.shape[1] - 1))
                row = int(np.clip(round(keypoint.pt[1]), 0, gray.shape[0] - 1))
                required = max(
                    float(self.args.sift_descriptor_min_clearance_px),
                    float(self.args.sift_descriptor_size_factor) * float(keypoint.size),
                )
                if float(clearance[row, column]) >= required:
                    filtered.append(keypoint)
            keypoints, descriptors = self.extractor.compute(gray, filtered)
            keypoints = keypoints or []
        else:
            keypoints, descriptors = [], None
        points = np.asarray([kp.pt for kp in keypoints], np.float64).reshape(-1, 2)
        leakage = np.zeros(len(points), bool)
        if len(points):
            rounded = np.rint(points).astype(int)
            rounded[:, 0] = np.clip(rounded[:, 0], 0, gray.shape[1] - 1)
            rounded[:, 1] = np.clip(rounded[:, 1], 0, gray.shape[0] - 1)
            leakage = risk_mask[rounded[:, 1], rounded[:, 0]] > 0
        return FeatureFrame(keypoints, descriptors, points, leakage)

    @staticmethod
    def _ratio_pairs(knn, ratio):
        result = {}
        for pair in knn:
            if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
                result[pair[0].queryIdx] = pair[0]
        return result

    def match(self, previous_index, current_index, previous, current, K, distortion):
        result = FeatureEdge(
            previous_index,
            current_index,
            previous_keypoints=len(previous.keypoints),
            current_keypoints=len(current.keypoints),
        )
        if previous.descriptors is None or current.descriptors is None:
            result.status = "NO_DESCRIPTORS"
            return result
        forward = self._ratio_pairs(
            self.matcher.knnMatch(previous.descriptors, current.descriptors, k=2),
            self.args.sift_ratio,
        )
        backward = self._ratio_pairs(
            self.matcher.knnMatch(current.descriptors, previous.descriptors, k=2),
            self.args.sift_ratio,
        )
        matches = [
            match
            for query, match in forward.items()
            if match.trainIdx in backward
            and backward[match.trainIdx].trainIdx == query
        ]
        matches.sort(key=lambda match: match.distance)
        matches = matches[:400]
        result.mutual_matches = len(matches)
        if len(matches) < self.args.sift_min_matches:
            result.status = "TOO_FEW_MUTUAL_MATCHES"
            return result

        raw_previous = np.asarray(
            [previous.keypoints[match.queryIdx].pt for match in matches], np.float64
        )
        raw_current = np.asarray(
            [current.keypoints[match.trainIdx].pt for match in matches], np.float64
        )
        leakage = np.asarray(
            [
                previous.leakage_flags[match.queryIdx]
                or current.leakage_flags[match.trainIdx]
                for match in matches
            ],
            bool,
        )
        points_previous = cv2.undistortPoints(
            raw_previous.reshape(-1, 1, 2), K, distortion, P=K
        ).reshape(-1, 2)
        points_current = cv2.undistortPoints(
            raw_current.reshape(-1, 1, 2), K, distortion, P=K
        ).reshape(-1, 2)
        try:
            essential, mask = cv2.findEssentialMat(
                points_previous,
                points_current,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=float(self.args.sift_ransac_threshold_px),
            )
        except cv2.error:
            essential, mask = None, None
        if essential is None or mask is None:
            result.status = "ESSENTIAL_FAILED"
            return result
        base_mask = mask.reshape(-1).astype(bool)
        best = None
        matrices = [essential] if essential.shape == (3, 3) else [
            essential[index : index + 3]
            for index in range(0, essential.shape[0], 3)
            if essential[index : index + 3].shape == (3, 3)
        ]
        for matrix in matrices:
            try:
                count, rotation, translation, pose_mask = cv2.recoverPose(
                    matrix,
                    points_previous,
                    points_current,
                    K,
                    mask=mask.copy(),
                )
            except cv2.error:
                continue
            combined = base_mask & pose_mask.reshape(-1).astype(bool)
            if best is None or int(np.count_nonzero(combined)) > best[0]:
                best = (
                    int(np.count_nonzero(combined)),
                    Pose(rotation, translation),
                    combined,
                )
        if best is None or best[0] < self.args.sift_min_matches:
            result.status = "RECOVER_POSE_FAILED"
            return result

        count, recovered, inliers = best
        indexes = np.flatnonzero(inliers)
        if len(indexes) > self.args.sift_max_stored_inliers:
            select = np.linspace(
                0, len(indexes) - 1, self.args.sift_max_stored_inliers
            ).round().astype(int)
            indexes = indexes[select]
        result.points_previous = points_previous[indexes]
        result.points_current = points_current[indexes]
        result.recovered_pose = recovered
        result.essential_inliers = count
        result.inlier_rate_percent = count / len(matches) * 100.0
        result.grid_leakage_matches = int(np.count_nonzero(leakage[inliers]))
        sampson = np.abs(
            signed_sampson_residuals(
                recovered,
                points_previous[inliers],
                points_current[inliers],
                K,
            )
        )
        if len(sampson):
            result.sampson_median_px = float(np.median(sampson))
            result.sampson_p90_px = float(np.percentile(sampson, 90))

        inv_K = np.linalg.inv(K)
        one = np.ones((len(indexes), 1), np.float64)
        rays_previous = (inv_K @ np.hstack([result.points_previous, one]).T).T
        rays_current = (inv_K @ np.hstack([result.points_current, one]).T).T
        rays_previous = (recovered.R @ rays_previous.T).T
        rays_previous /= np.maximum(np.linalg.norm(rays_previous, axis=1, keepdims=True), 1e-12)
        rays_current /= np.maximum(np.linalg.norm(rays_current, axis=1, keepdims=True), 1e-12)
        angles = np.degrees(
            np.arccos(np.clip(np.sum(rays_previous * rays_current, axis=1), -1.0, 1.0))
        )
        result.parallax_median_deg = float(np.median(angles)) if len(angles) else None
        if result.inlier_rate_percent < self.args.sift_min_inlier_rate_percent:
            result.status = "LOW_INLIER_RATE"
        elif (
            result.sampson_p90_px is None
            or result.sampson_p90_px > self.args.sift_max_sampson_p90_px
        ):
            result.status = "SAMPSON_P90_GATE_FAILED"
        elif (
            result.parallax_median_deg is None
            or result.parallax_median_deg < self.args.sift_min_parallax_deg
        ):
            result.status = "DEGENERATE_LOW_PARALLAX"
        else:
            result.status = "OK"
        return result


# =============================================================================
# Offline temporal estimators (each call receives one eye only)
# =============================================================================


@dataclass
class BeamHypothesis:
    cost: float
    path: list[tuple[int, MarkerPoseCandidate]]


def temporal_transition_cost(
    path,
    frame_index,
    candidate,
    args,
    feature_edges=None,
):
    if not path:
        return 0.0
    previous_index, previous_candidate = path[-1]
    gap = max(1, frame_index - previous_index)
    relative = relative_pose(previous_candidate.pose, candidate.pose)
    cost = (
        args.temporal_rotation_jump_weight
        * rotation_angle_deg(relative.R)
        / gap
        + args.temporal_translation_jump_weight
        * float(np.linalg.norm(relative.t))
        / gap
    )
    if len(path) >= 2:
        before_index, before_candidate = path[-2]
        before_gap = max(1, previous_index - before_index)
        previous_relative = relative_pose(
            before_candidate.pose, previous_candidate.pose
        )
        rotation_acceleration = rotation_angle_deg(
            relative.R @ previous_relative.R.T
        ) / max(gap, before_gap)
        translation_velocity = relative.t / gap
        previous_velocity = previous_relative.t / before_gap
        translation_acceleration = float(
            np.linalg.norm(translation_velocity - previous_velocity)
        )
        cost += (
            args.temporal_accel_rotation_weight * rotation_acceleration
            + args.temporal_accel_translation_weight * translation_acceleration
        )

    if feature_edges is not None:
        edge = feature_edges.get((previous_index, frame_index))
        if edge is not None and edge.status == "OK" and edge.recovered_pose is not None:
            cost += args.sift_branch_rotation_weight * rotation_angle_deg(
                relative.R @ edge.recovered_pose.R.T
            )
            direction = translation_direction_error_deg(
                relative.t, edge.recovered_pose.t
            )
            if direction is not None:
                cost += args.sift_branch_direction_weight * direction
            if len(edge.points_previous):
                sampson = np.abs(
                    signed_sampson_residuals(
                        relative,
                        edge.points_previous,
                        edge.points_current,
                        args._active_camera_matrix,
                    )
                )
                if len(sampson):
                    cost += args.sift_branch_sampson_weight * float(np.median(sampson))
    return float(cost)


def select_marker_temporal_path(
    observations: dict[int, EyeFrameObservation],
    marker_id: int,
    args,
    feature_edges=None,
):
    available = []
    raw = {}
    for frame_index in sorted(observations):
        marker = observations[frame_index].markers.get(marker_id)
        if marker is None or not marker.candidates:
            continue
        available.append((frame_index, marker.candidates))
        raw[frame_index] = min(
            marker.candidates, key=lambda item: item.reprojection_rms_px
        )
    if not available:
        return {}, raw, None

    beams = [
        BeamHypothesis(
            args.temporal_reprojection_weight * candidate.reprojection_rms_px,
            [(available[0][0], candidate)],
        )
        for candidate in available[0][1]
    ]
    beams.sort(key=lambda item: item.cost)
    beams = beams[: args.temporal_beam_width]
    for frame_index, candidates in available[1:]:
        expanded = []
        for beam in beams:
            for candidate in candidates:
                emission = (
                    args.temporal_reprojection_weight
                    * candidate.reprojection_rms_px
                )
                transition = temporal_transition_cost(
                    beam.path,
                    frame_index,
                    candidate,
                    args,
                    feature_edges,
                )
                expanded.append(
                    BeamHypothesis(
                        beam.cost + emission + transition,
                        beam.path + [(frame_index, candidate)],
                    )
                )
        expanded.sort(key=lambda item: item.cost)
        beams = expanded[: args.temporal_beam_width]
    best = beams[0]
    margin = beams[1].cost - beams[0].cost if len(beams) > 1 else None
    return dict(best.path), raw, margin


def marker_pose_reprojection(pose, observation, size_mm, K, distortion):
    projected = project_points(square_object_points(size_mm), pose, K, distortion)
    errors = np.linalg.norm(projected - observation.corners, axis=1)
    return (
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
    )


def build_single_marker_method(
    observations,
    marker_id,
    size_mm,
    K,
    distortion,
    args,
    feature_edges=None,
    method_name=None,
):
    started = time.perf_counter()
    selected, raw, global_margin = select_marker_temporal_path(
        observations,
        marker_id,
        args,
        feature_edges=feature_edges,
    )
    selected_poses = {index: item.pose for index, item in selected.items()}
    smoothed = smooth_pose_path(
        selected_poses,
        args.temporal_smooth_alpha,
        args.temporal_smooth_iterations,
    )
    results: dict[int, MethodPoseResult] = {}
    history: list[tuple[int, Pose]] = []
    for frame_index in sorted(observations):
        frame = observations[frame_index]
        marker = frame.markers.get(marker_id)
        if frame_index not in selected:
            results[frame_index] = MethodPoseResult(
                available=False,
                status="MARKER_NOT_FOUND" if marker is None else "NO_VALID_IPPE_BRANCH",
                reject_reason=(
                    f"ID{marker_id} not detected"
                    if marker is None
                    else f"ID{marker_id} had no positive-depth IPPE branch"
                ),
                marker_id=marker_id,
                id2_found=2 in frame.markers,
                id5_found=5 in frame.markers,
            )
            continue
        candidate = selected[frame_index]
        pose = smoothed[frame_index]
        prediction = predict_pose(history, frame_index)
        prediction_rotation = None
        prediction_translation = None
        if prediction is not None:
            prediction_rotation, prediction_translation = pose_distance(pose, prediction)
        rms, maximum = marker_pose_reprojection(
            pose, marker, size_mm, K, distortion
        )
        source = (
            "offline_temporal_ippe_sift_branch_path"
            if method_name == "ID2_SIFT_TEMPORAL"
            else "offline_temporal_ippe_path"
        )
        if rms > args.temporal_max_post_smooth_marker_rms_px:
            # Preserve the selected observation-constrained IPPE pose whenever
            # unconstrained trajectory smoothing moves too far from the four
            # measured corners.
            pose = candidate.pose
            rms, maximum = marker_pose_reprojection(
                pose, marker, size_mm, K, distortion
            )
            source += "_SMOOTH_REJECTED"
        result = MethodPoseResult(
            pose=pose,
            raw_pose=raw[frame_index].pose,
            available=True,
            status=(
                "OK_BOOTSTRAP_AMBIGUOUS"
                if len(history) == 0
                and len(marker.candidates) > 1
                and abs(
                    marker.candidates[0].reprojection_rms_px
                    - marker.candidates[1].reprojection_rms_px
                ) < 0.05
                else "OK"
            ),
            source=source,
            marker_id=marker_id,
            id2_found=2 in frame.markers,
            id5_found=5 in frame.markers,
            selected_id2_branch=candidate.branch if marker_id == 2 else None,
            selected_id5_branch=candidate.branch if marker_id == 5 else None,
            marker_reprojection_rms_px=rms,
            marker_reprojection_max_px=maximum,
            temporal_prior_used=bool(history),
            prediction_rotation_residual_deg=prediction_rotation,
            prediction_translation_residual_mm=prediction_translation,
        )
        results[frame_index] = result
        history.append((frame_index, pose))
        history = history[-2:]
    elapsed = (time.perf_counter() - started) * 1000.0
    per_frame = elapsed / max(1, len(results))
    for result in results.values():
        result.processing_time_ms = per_frame
        if global_margin is not None and result.status == "OK_BOOTSTRAP_AMBIGUOUS":
            result.reject_reason = f"global best/second path margin={global_margin:.4f}"
    return results, selected, raw


def robust_pose_mean(poses, initial=None, iterations=10):
    poses = list(poses)
    if not poses:
        return None
    mean = initial or poses[0]
    for _ in range(iterations):
        rotations = np.asarray(
            [cv2.Rodrigues(pose.R @ mean.R.T)[0].reshape(3) for pose in poses]
        )
        translations = np.asarray([(pose.t - mean.t).reshape(3) for pose in poses])
        rotation_norm = np.linalg.norm(rotations, axis=1)
        translation_norm = np.linalg.norm(translations, axis=1)
        rotation_scale = max(float(np.median(rotation_norm)) * 2.5, math.radians(0.25))
        translation_scale = max(float(np.median(translation_norm)) * 2.5, 0.25)
        weights = 1.0 / np.maximum(
            1.0,
            rotation_norm / rotation_scale + translation_norm / translation_scale,
        )
        delta_rotation = np.average(rotations, axis=0, weights=weights)
        delta_translation = np.average(translations, axis=0, weights=weights)
        mean = Pose(
            cv2.Rodrigues(delta_rotation)[0] @ mean.R,
            mean.t + delta_translation.reshape(3, 1),
        )
        if np.linalg.norm(delta_rotation) < 1e-8 and np.linalg.norm(delta_translation) < 1e-5:
            break
    return mean


def marker_relation_candidates(observations):
    by_frame = {}
    for frame_index, frame in observations.items():
        id2 = frame.markers.get(2)
        id5 = frame.markers.get(5)
        if id2 is None or id5 is None or not id2.candidates or not id5.candidates:
            continue
        combinations = []
        for candidate2 in id2.candidates:
            for candidate5 in id5.candidates:
                relation = compose(inverse(candidate2.pose), candidate5.pose)
                combinations.append((candidate2, candidate5, relation))
        by_frame[frame_index] = combinations
    return by_frame


def estimate_marker_relation(observations, args) -> RelationEstimate:
    by_frame = marker_relation_candidates(observations)
    result = RelationEstimate(candidate_frames=len(by_frame))
    if len(by_frame) < args.relation_min_support_frames:
        result.status = (
            f"Need at least {args.relation_min_support_frames} co-visible frames; "
            f"got {len(by_frame)}"
        )
        return result
    seeds = [item[2] for combinations in by_frame.values() for item in combinations]
    if len(seeds) > 400:
        indexes = np.linspace(0, len(seeds) - 1, 400).round().astype(int)
        seeds = [seeds[index] for index in indexes]

    best = None
    for seed in seeds:
        accepted = []
        normalized = []
        for combinations in by_frame.values():
            distances = [pose_distance(item[2], seed) for item in combinations]
            index = int(
                np.argmin(
                    [
                        rotation / args.relation_rotation_gate_deg
                        + translation / args.relation_translation_gate_mm
                        for rotation, translation in distances
                    ]
                )
            )
            rotation, translation = distances[index]
            if (
                rotation <= args.relation_rotation_gate_deg
                and translation <= args.relation_translation_gate_mm
            ):
                accepted.append(combinations[index][2])
                normalized.append(
                    rotation / args.relation_rotation_gate_deg
                    + translation / args.relation_translation_gate_mm
                )
        candidate_score = (-len(accepted), float(np.median(normalized)) if normalized else 1e9)
        if best is None or candidate_score < best[0]:
            best = (candidate_score, seed, accepted)
    if best is None or len(best[2]) < args.relation_min_support_frames:
        result.status = "No stable ID2-ID5 relation consensus"
        return result

    relation = robust_pose_mean(best[2], initial=best[1])
    selected_pairs = {}
    for _ in range(6):
        accepted_relations = []
        selected_pairs = {}
        for frame_index, combinations in by_frame.items():
            distances = [pose_distance(item[2], relation) for item in combinations]
            costs = [
                rotation / args.relation_rotation_gate_deg
                + translation / args.relation_translation_gate_mm
                for rotation, translation in distances
            ]
            index = int(np.argmin(costs))
            rotation, translation = distances[index]
            if (
                rotation <= args.relation_rotation_gate_deg
                and translation <= args.relation_translation_gate_mm
            ):
                candidate2, candidate5, candidate_relation = combinations[index]
                selected_pairs[frame_index] = (
                    candidate2.branch,
                    candidate5.branch,
                    candidate_relation,
                )
                accepted_relations.append(candidate_relation)
        updated = robust_pose_mean(accepted_relations, initial=relation)
        if updated is None:
            break
        rotation_delta, translation_delta = pose_distance(updated, relation)
        relation = updated
        if rotation_delta < 1e-5 and translation_delta < 1e-4:
            break
    if len(selected_pairs) < args.relation_min_support_frames:
        result.status = "Relation consensus fell below minimum support after refinement"
        return result

    rotation_residuals = []
    translation_residuals = []
    for _frame_index, (_branch2, _branch5, candidate_relation) in selected_pairs.items():
        rotation, translation = pose_distance(candidate_relation, relation)
        rotation_residuals.append(rotation)
        translation_residuals.append(translation)
    result.pose = relation
    result.available = True
    result.status = "OK_OFFLINE_RELATION_CONSENSUS"
    result.support_frames = len(selected_pairs)
    result.rotation_residual_median_deg = percentile(rotation_residuals, 50)
    result.rotation_residual_p95_deg = percentile(rotation_residuals, 95)
    result.translation_residual_median_mm = percentile(translation_residuals, 50)
    result.translation_residual_p95_mm = percentile(translation_residuals, 95)
    result.selected_pairs = selected_pairs
    return result


def find_candidate_by_branch(observation, branch):
    if observation is None:
        return None
    for candidate in observation.candidates:
        if candidate.branch == branch:
            return candidate
    return None


def solve_joint_marker_pose(
    id2,
    id5,
    candidate2,
    candidate5,
    relation,
    size2,
    size5,
    K,
    distortion,
    max_group_rms,
):
    object2 = square_object_points(size2)
    object5_local = square_object_points(size5)
    object5 = (relation.R @ object5_local.T + relation.t).T
    inferred5 = compose(candidate5.pose, inverse(relation))
    seed = robust_pose_mean([candidate2.pose, inferred5], initial=candidate2.pose)
    objects = np.vstack([object2, object5])
    images = np.vstack([id2.corners, id5.corners])
    rvec = cv2.Rodrigues(seed.R)[0]
    tvec = seed.t.copy()
    try:
        ok, rvec, tvec = cv2.solvePnP(
            objects,
            images,
            K,
            distortion,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None, None, "JOINT_PNP_FAILED"
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                objects, images, K, distortion, rvec, tvec
            )
    except cv2.error:
        return None, None, None, "JOINT_PNP_FAILED"
    pose = Pose(cv2.Rodrigues(rvec)[0], tvec)
    projected = project_points(objects, pose, K, distortion)
    errors = np.linalg.norm(projected - images, axis=1)
    rms2 = float(np.sqrt(np.mean(np.square(errors[:4]))))
    rms5 = float(np.sqrt(np.mean(np.square(errors[4:]))))
    if rms2 <= max_group_rms and rms5 <= max_group_rms:
        return pose, rms2, rms5, "JOINT_8POINT_PNP"
    if rms2 <= max_group_rms and rms2 < rms5:
        return candidate2.pose, candidate2.reprojection_rms_px, rms5, "FALLBACK_ID2_GROUP_GATE"
    if rms5 <= max_group_rms:
        return inferred5, rms2, candidate5.reprojection_rms_px, "FALLBACK_ID5_GROUP_GATE"
    return None, rms2, rms5, "BOTH_MARKER_GROUPS_REJECTED"


def build_dual_marker_method(
    observations,
    size2,
    size5,
    K,
    distortion,
    args,
    selected2,
    selected5,
):
    started = time.perf_counter()
    relation = estimate_marker_relation(observations, args)
    results = {}
    initial_poses = {}
    for frame_index in sorted(observations):
        frame = observations[frame_index]
        id2 = frame.markers.get(2)
        id5 = frame.markers.get(5)
        common = relation.selected_pairs.get(frame_index)
        candidate2 = None
        candidate5 = None
        if common is not None:
            candidate2 = find_candidate_by_branch(id2, common[0])
            candidate5 = find_candidate_by_branch(id5, common[1])
        if candidate2 is None:
            candidate2 = selected2.get(frame_index)
        if candidate5 is None:
            candidate5 = selected5.get(frame_index)

        result = MethodPoseResult(
            id2_found=id2 is not None,
            id5_found=id5 is not None,
            selected_id2_branch=candidate2.branch if candidate2 else None,
            selected_id5_branch=candidate5.branch if candidate5 else None,
            temporal_prior_used=True,
        )
        if not relation.available:
            result.status = "RELATION_UNAVAILABLE"
            result.reject_reason = relation.status
            results[frame_index] = result
            continue

        pose = None
        rms_values = []
        maximum_values = []
        if id2 is not None and id5 is not None and candidate2 and candidate5:
            pose, rms2, rms5, source = solve_joint_marker_pose(
                id2,
                id5,
                candidate2,
                candidate5,
                relation.pose,
                size2,
                size5,
                K,
                distortion,
                args.dual_marker_max_group_rms_px,
            )
            result.source = source
            rms_values = [value for value in (rms2, rms5) if finite(value) is not None]
            relation_candidate = compose(inverse(candidate2.pose), candidate5.pose)
            (
                result.relation_rotation_residual_deg,
                result.relation_translation_residual_mm,
            ) = pose_distance(relation_candidate, relation.pose)
            maximum_values = [candidate2.reprojection_max_px, candidate5.reprojection_max_px]
        elif candidate2 is not None:
            pose = candidate2.pose
            result.source = "FALLBACK_ID2_ONLY"
            rms_values = [candidate2.reprojection_rms_px]
            maximum_values = [candidate2.reprojection_max_px]
        elif candidate5 is not None:
            pose = compose(candidate5.pose, inverse(relation.pose))
            result.source = "FALLBACK_ID5_ONLY_WITH_LEARNED_RELATION"
            rms_values = [candidate5.reprojection_rms_px]
            maximum_values = [candidate5.reprojection_max_px]

        if pose is None:
            result.status = "NO_USABLE_MARKER_POSE"
            result.reject_reason = result.source or "Neither ID2 nor ID5 yielded a usable pose"
        else:
            result.pose = pose
            result.raw_pose = candidate2.pose if candidate2 is not None else pose
            result.available = True
            result.status = "OK"
            result.marker_reprojection_rms_px = (
                float(np.sqrt(np.mean(np.square(rms_values)))) if rms_values else None
            )
            result.marker_reprojection_max_px = max(maximum_values) if maximum_values else None
            initial_poses[frame_index] = pose
        results[frame_index] = result

    smoothed = smooth_pose_path(
        initial_poses,
        args.temporal_smooth_alpha,
        args.temporal_smooth_iterations,
    )
    history = []
    for frame_index in sorted(smoothed):
        pose = smoothed[frame_index]
        result = results[frame_index]
        frame = observations[frame_index]
        group_rms = []
        group_max = []
        use_id2_group = result.source.startswith("JOINT_8POINT_PNP") or result.source.startswith("FALLBACK_ID2")
        use_id5_group = result.source.startswith("JOINT_8POINT_PNP") or result.source.startswith("FALLBACK_ID5")
        if use_id2_group and frame.markers.get(2) is not None:
            projected = project_points(
                square_object_points(size2), pose, K, distortion
            )
            errors = np.linalg.norm(projected - frame.markers[2].corners, axis=1)
            group_rms.append(float(np.sqrt(np.mean(np.square(errors)))))
            group_max.append(float(np.max(errors)))
        if use_id5_group and frame.markers.get(5) is not None:
            object5 = (
                relation.pose.R @ square_object_points(size5).T + relation.pose.t
            ).T
            projected = project_points(object5, pose, K, distortion)
            errors = np.linalg.norm(projected - frame.markers[5].corners, axis=1)
            group_rms.append(float(np.sqrt(np.mean(np.square(errors)))))
            group_max.append(float(np.max(errors)))
        if group_rms and max(group_rms) > args.dual_marker_max_group_rms_px:
            pose = initial_poses[frame_index]
            result.source += "_SMOOTH_REJECTED"
        elif group_rms:
            result.marker_reprojection_rms_px = float(
                np.sqrt(np.mean(np.square(group_rms)))
            )
            result.marker_reprojection_max_px = max(group_max)
        prediction = predict_pose(history, frame_index)
        result.pose = pose
        if prediction is not None:
            (
                result.prediction_rotation_residual_deg,
                result.prediction_translation_residual_mm,
            ) = pose_distance(pose, prediction)
        history.append((frame_index, pose))
        history = history[-2:]
    elapsed = (time.perf_counter() - started) * 1000.0
    for result in results.values():
        result.processing_time_ms = elapsed / max(1, len(results))
    return results, relation


def refine_id2_with_sift(
    observations,
    selected,
    raw,
    feature_edges,
    size_mm,
    K,
    distortion,
    args,
):
    results = {}
    history: list[tuple[int, Pose]] = []
    object_points = square_object_points(size_mm)
    for frame_index in sorted(observations):
        frame = observations[frame_index]
        marker = frame.markers.get(2)
        candidate = selected.get(frame_index)
        if marker is None or candidate is None:
            results[frame_index] = MethodPoseResult(
                available=False,
                status="ID2_UNAVAILABLE",
                reject_reason="Metric ID2 anchor is required in the first implementation",
                marker_id=2,
                id2_found=marker is not None,
                id5_found=5 in frame.markers,
            )
            continue
        started = time.perf_counter()
        initial = candidate.pose
        prediction = predict_pose(history, frame_index)
        previous_index = history[-1][0] if history else None
        previous_pose = history[-1][1] if history else None
        edge = (
            feature_edges.get((previous_index, frame_index))
            if previous_index is not None
            else None
        )
        best = None
        if edge is not None and edge.status == "OK" and previous_pose is not None:
            points_previous = edge.points_previous
            points_current = edge.points_current
            holdout_mask = np.zeros(len(points_previous), bool)
            holdout_mask[::5] = True
            if np.count_nonzero(~holdout_mask) < args.sift_min_matches:
                holdout_mask[:] = False
            train_previous = points_previous[~holdout_mask]
            train_current = points_current[~holdout_mask]
            hold_previous = points_previous[holdout_mask]
            hold_current = points_current[holdout_mask]
            for marker_weight in args.sift_marker_weights:
                def residual(pose):
                    projected = project_points(object_points, pose, K, distortion)
                    marker_residual = (projected - marker.corners).reshape(-1)
                    parts = [
                        marker_residual
                        * float(marker_weight)
                        / math.sqrt(max(1, len(marker_residual)))
                    ]
                    relative = relative_pose(previous_pose, pose)
                    sampson = signed_sampson_residuals(
                        relative,
                        train_previous,
                        train_current,
                        K,
                    )
                    if len(sampson):
                        parts.append(
                            sampson
                            * args.sift_joint_feature_weight
                            / math.sqrt(max(1, len(sampson)))
                        )
                    if prediction is not None:
                        parts.append(
                            cv2.Rodrigues(pose.R @ prediction.R.T)[0].reshape(3)
                            * args.sift_joint_temporal_rotation_weight
                        )
                        parts.append(
                            (pose.t - prediction.t).reshape(3)
                            * args.sift_joint_temporal_translation_weight
                        )
                    return np.concatenate(parts)

                refined = robust_lm(initial, residual, iterations=12)
                marker_rms, marker_max = marker_pose_reprojection(
                    refined, marker, size_mm, K, distortion
                )
                relative = relative_pose(previous_pose, refined)
                holdout = np.abs(
                    signed_sampson_residuals(
                        relative, hold_previous, hold_current, K
                    )
                )
                holdout_median = float(np.median(holdout)) if len(holdout) else 0.0
                all_sampson = np.abs(
                    signed_sampson_residuals(
                        relative, points_previous, points_current, K
                    )
                )
                score = marker_rms + holdout_median
                if (
                    marker_rms <= args.sift_joint_max_marker_rms_px
                    and holdout_median <= args.sift_joint_max_holdout_median_px
                ):
                    item = (
                        score,
                        refined,
                        marker_rms,
                        marker_max,
                        holdout_median,
                        float(np.median(all_sampson)) if len(all_sampson) else None,
                        marker_weight,
                    )
                    if best is None or item[0] < best[0]:
                        best = item

        if best is None:
            pose = initial
            marker_rms, marker_max = marker_pose_reprojection(
                pose, marker, size_mm, K, distortion
            )
            source = "ID2_TEMPORAL_FALLBACK"
            sift_used = False
            holdout_median = None
            sampson_median = None
        else:
            (
                _score,
                pose,
                marker_rms,
                marker_max,
                holdout_median,
                sampson_median,
                marker_weight,
            ) = best
            source = f"ID2_SIFT_JOINT_W{marker_weight:g}"
            sift_used = True
        prediction_rotation = prediction_translation = None
        if prediction is not None:
            prediction_rotation, prediction_translation = pose_distance(pose, prediction)
        results[frame_index] = MethodPoseResult(
            pose=pose,
            raw_pose=raw[frame_index].pose,
            available=True,
            status="OK",
            source=source,
            marker_id=2,
            id2_found=True,
            id5_found=5 in frame.markers,
            selected_id2_branch=candidate.branch,
            marker_reprojection_rms_px=marker_rms,
            marker_reprojection_max_px=marker_max,
            temporal_prior_used=bool(history),
            prediction_rotation_residual_deg=prediction_rotation,
            prediction_translation_residual_mm=prediction_translation,
            sift_used=sift_used,
            sift_inliers=edge.essential_inliers if edge is not None else 0,
            sift_sampson_median_px=sampson_median,
            sift_holdout_median_px=holdout_median,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        history.append((frame_index, pose))
        history = history[-2:]
    return results


# =============================================================================
# Video preprocessing and diagnostics
# =============================================================================


def build_frame_indexes(frame_count, args):
    end = frame_count - 1 if args.end_frame < 0 else min(args.end_frame, frame_count - 1)
    if args.start_frame < 0 or args.start_frame > end:
        raise ValueError(
            f"Invalid frame range {args.start_frame}..{args.end_frame}; "
            f"video range is 0..{frame_count - 1}"
        )
    indexes = list(range(args.start_frame, end + 1, args.frame_step))
    if args.max_frames > 0:
        indexes = indexes[: args.max_frames]
    if not indexes:
        raise ValueError("No frame selected")
    if args.reference_frame not in indexes:
        if args.reference_frame < 0 or args.reference_frame >= frame_count:
            raise ValueError(
                f"Reference frame {args.reference_frame} is outside 0..{frame_count - 1}"
            )
        indexes.append(args.reference_frame)
        indexes.sort()
    return indexes


def initialize_video_writer(path, source_size, fps, max_width):
    width, height = map(int, source_size)
    scale = min(1.0, float(max_width) / max(width, 1))
    target = (
        max(2, int(round(width * scale)) // 2 * 2),
        max(2, int(round(height * scale)) // 2 * 2),
    )
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), target
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create diagnostic video: {path}")
    return writer, target


def iter_selected_video_frames(capture, indexes):
    """Decode sorted frame indexes without seeking again for every frame.

    Repeated ``CAP_PROP_POS_FRAMES`` calls are particularly slow for MP4/GOP
    video.  We seek once at discontinuities and decode forward through nearby
    selected frames.  This keeps a normal 300-frame, step=1 run sequential.
    """
    cursor = None
    for target in sorted(map(int, indexes)):
        if cursor is None or target < cursor or target - cursor > 120:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            cursor = target
        selected = None
        while cursor <= target:
            ok, frame = capture.read()
            if not ok or frame is None:
                selected = None
                break
            if cursor == target:
                selected = frame
            cursor += 1
        yield target, selected


def draw_small_cross(image, point, color, arm=3):
    x, y = np.rint(point).astype(int)
    cv2.line(image, (x - arm, y), (x + arm, y), color, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - arm), (x, y + arm), color, 1, cv2.LINE_AA)


def draw_eye_detections(image, gt, markers, sift_frame=None):
    canvas = image.copy()
    if gt.charuco_corners is not None:
        inliers = set(
            map(int, gt.inlier_indices.reshape(-1))
            if gt.inlier_indices is not None
            else []
        )
        for index, point in enumerate(gt.charuco_corners):
            draw_small_cross(
                canvas,
                point,
                (0, 255, 0) if index in inliers else (0, 180, 255),
                arm=2,
            )
    if gt.board_polygon is not None:
        cv2.polylines(
            canvas,
            [np.rint(gt.board_polygon).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )
    for marker_id, observation in markers.items():
        color = (255, 0, 255) if marker_id == 2 else (0, 255, 255)
        cv2.polylines(
            canvas,
            [np.rint(observation.corners).astype(np.int32).reshape(-1, 1, 2)],
            True,
            color,
            1,
            cv2.LINE_AA,
        )
        for corner in observation.corners:
            draw_small_cross(canvas, corner, color, arm=3)
        x, y = np.rint(observation.corners[0]).astype(int)
        cv2.putText(
            canvas,
            f"ID{marker_id}",
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    if sift_frame is not None:
        for point, leakage in zip(sift_frame.points, sift_frame.leakage_flags):
            x, y = np.rint(point).astype(int)
            cv2.circle(
                canvas,
                (x, y),
                1,
                (0, 0, 255) if leakage else (255, 80, 0),
                -1,
                cv2.LINE_AA,
            )
    return canvas


def add_text_lines(image, lines, origin=(8, 20), color=(0, 0, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 19


def make_detection_diagnostic(
    left,
    right,
    sanitized_left,
    sanitized_right,
    observation_left,
    observation_right,
    sift_left,
    sift_right,
):
    top_left = draw_eye_detections(
        left, observation_left.gt, observation_left.markers
    )
    top_right = draw_eye_detections(
        right, observation_right.gt, observation_right.markers
    )
    bottom_left = draw_eye_detections(
        cv2.cvtColor(sanitized_left, cv2.COLOR_GRAY2BGR),
        GtPoseResult(),
        {},
        sift_left,
    )
    bottom_right = draw_eye_detections(
        cv2.cvtColor(sanitized_right, cv2.COLOR_GRAY2BGR),
        GtPoseResult(),
        {},
        sift_right,
    )
    add_text_lines(
        top_left,
        [
            f"L GT={observation_left.gt.status} corners={observation_left.gt.corner_count}",
            f"RMS={finite(observation_left.gt.reprojection_rms_px)} px",
        ],
    )
    add_text_lines(
        top_right,
        [
            f"R GT={observation_right.gt.status} corners={observation_right.gt.corner_count}",
            f"RMS={finite(observation_right.gt.reprojection_rms_px)} px",
        ],
    )
    add_text_lines(
        bottom_left,
        [
            f"L SIFT sanitized kp={len(sift_left.keypoints)}",
            f"leakage-risk kp={int(np.count_nonzero(sift_left.leakage_flags))}",
        ],
        color=(255, 0, 0),
    )
    add_text_lines(
        bottom_right,
        [
            f"R SIFT sanitized kp={len(sift_right.keypoints)}",
            f"leakage-risk kp={int(np.count_nonzero(sift_right.leakage_flags))}",
        ],
        color=(255, 0, 0),
    )
    return np.vstack([np.hstack([top_left, top_right]), np.hstack([bottom_left, bottom_right])])


def preprocess_video(
    video_path,
    indexes,
    fps,
    eye_shape,
    board,
    dictionary,
    charuco_metadata,
    marker_sizes,
    target_scales,
    K_by_eye,
    d_by_eye,
    args,
    diagnostic_path=None,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    detectors = {
        eye: EyeDetector(board, dictionary, K_by_eye[eye], d_by_eye[eye])
        for eye in EYES
    }
    sift_extractors = {eye: SameEyeSift(args) for eye in EYES}
    observations = {eye: {} for eye in EYES}
    feature_edges = {eye: {} for eye in EYES}
    previous_features = {eye: None for eye in EYES}
    previous_indexes = {eye: None for eye in EYES}
    diagnostic_writer = None
    diagnostic_size = None
    if diagnostic_path is not None:
        height, width = eye_shape[:2]
        diagnostic_writer, diagnostic_size = initialize_video_writer(
            diagnostic_path,
            (width * 2, height * 2),
            fps / max(1, args.frame_step),
            args.diagnostic_max_width,
        )

    try:
        for sequence_index, (frame_index, sbs) in enumerate(
            iter_selected_video_frames(capture, indexes)
        ):
            if sbs is None:
                print(f"[WARN] Cannot read F{frame_index}; skipped")
                for eye in EYES:
                    observations[eye][frame_index] = EyeFrameObservation(
                        frame_index=frame_index,
                        timestamp_s=frame_index / fps,
                        gt=GtPoseResult(
                            status="FRAME_DECODE_FAILED",
                            reject_reason="OpenCV could not decode the selected frame",
                        ),
                        markers={},
                        mask=MaskDiagnostics(status="FRAME_DECODE_FAILED"),
                    )
                continue
            left, right = split_sbs(sbs)
            images = {"L": left, "R": right}
            sanitized_images = {}
            sift_frames = {}
            for eye in EYES:
                started = time.perf_counter()
                gray = cv2.cvtColor(images[eye], cv2.COLOR_BGR2GRAY)
                roi = args.left_detection_roi if eye == "L" else args.right_detection_roi
                (
                    charuco_corners,
                    charuco_ids,
                    marker_corners,
                    marker_ids,
                    _rejected,
                ) = detectors[eye].detect(gray, roi)
                markers = marker_observations(
                    marker_corners,
                    marker_ids,
                    marker_sizes,
                    K_by_eye[eye],
                    d_by_eye[eye],
                )
                gt = solve_charuco_gt(
                    board,
                    charuco_metadata,
                    charuco_corners,
                    charuco_ids,
                    marker_ids,
                    K_by_eye[eye],
                    d_by_eye[eye],
                    gray.shape,
                    args,
                )
                sift_roi = args.left_sift_roi if eye == "L" else args.right_sift_roi
                sanitized, allowed, risk, mask_diagnostics = create_sift_image_and_mask(
                    gray,
                    gt,
                    markers,
                    sift_roi,
                    target_scales,
                    args,
                )
                feature_frame = sift_extractors[eye].extract(sanitized, allowed, risk)
                mask_diagnostics.sift_keypoints = len(feature_frame.keypoints)
                mask_diagnostics.grid_leakage_keypoints = int(
                    np.count_nonzero(feature_frame.leakage_flags)
                )
                observation = EyeFrameObservation(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    gt=gt,
                    markers=markers,
                    mask=mask_diagnostics,
                    sift_keypoints=len(feature_frame.keypoints),
                    sift_grid_leakage_keypoints=mask_diagnostics.grid_leakage_keypoints,
                )
                observations[eye][frame_index] = observation
                if previous_features[eye] is not None:
                    edge = sift_extractors[eye].match(
                        previous_indexes[eye],
                        frame_index,
                        previous_features[eye],
                        feature_frame,
                        K_by_eye[eye],
                        d_by_eye[eye],
                    )
                    feature_edges[eye][(previous_indexes[eye], frame_index)] = edge
                previous_features[eye] = feature_frame
                previous_indexes[eye] = frame_index
                sanitized_images[eye] = sanitized
                sift_frames[eye] = feature_frame
                observation.gt.processing_time_ms = (
                    time.perf_counter() - started
                ) * 1000.0

            if diagnostic_writer is not None:
                canvas = make_detection_diagnostic(
                    left,
                    right,
                    sanitized_images["L"],
                    sanitized_images["R"],
                    observations["L"][frame_index],
                    observations["R"][frame_index],
                    sift_frames["L"],
                    sift_frames["R"],
                )
                diagnostic_writer.write(
                    cv2.resize(canvas, diagnostic_size, interpolation=cv2.INTER_AREA)
                )

            processed = sequence_index + 1
            if processed % 10 == 0 or processed == len(indexes):
                left_valid = sum(item.gt.valid for item in observations["L"].values())
                right_valid = sum(item.gt.valid for item in observations["R"].values())
                print(
                    f"[Detect] {processed}/{len(indexes)} frames | "
                    f"GT valid L={left_valid}, R={right_valid}"
                )
    finally:
        capture.release()
        if diagnostic_writer is not None:
            diagnostic_writer.release()
    return observations, feature_edges


# =============================================================================
# Argument parsing
# =============================================================================


def parse_marker_weights(text):
    try:
        values = tuple(float(value.strip()) for value in str(text).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Weights must be comma-separated numbers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("All SIFT marker weights must be positive")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--charuco-metadata", default=CHARUCO_METADATA_PATH)
    parser.add_argument("--id2-metadata", default=ID2_METADATA_PATH)
    parser.add_argument("--id5-metadata", default=ID5_METADATA_PATH)
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--end-frame", type=int, default=END_FRAME)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP)
    parser.add_argument("--reference-frame", type=int, default=REFERENCE_FRAME)
    parser.add_argument("--left-detection-roi", type=parse_roi, default=LEFT_DETECTION_ROI)
    parser.add_argument("--right-detection-roi", type=parse_roi, default=RIGHT_DETECTION_ROI)
    parser.add_argument(
        "--sift-roi-left", dest="left_sift_roi", type=parse_roi, default=LEFT_SIFT_ROI
    )
    parser.add_argument(
        "--sift-roi-right", dest="right_sift_roi", type=parse_roi, default=RIGHT_SIFT_ROI
    )

    parser.add_argument("--gt-min-charuco-corners", type=int, default=GT_MIN_CHARUCO_CORNERS)
    parser.add_argument("--gt-min-pnp-inliers", type=int, default=GT_MIN_PNP_INLIERS)
    parser.add_argument("--gt-min-inlier-rate-percent", type=float, default=GT_MIN_INLIER_RATE_PERCENT)
    parser.add_argument("--gt-max-reprojection-rms-px", type=float, default=GT_MAX_REPROJECTION_RMS_PX)
    parser.add_argument("--gt-max-reprojection-p95-px", type=float, default=GT_MAX_REPROJECTION_P95_PX)
    parser.add_argument("--gt-min-object-coverage-percent", type=float, default=GT_MIN_OBJECT_COVERAGE_PERCENT)
    parser.add_argument("--gt-min-x-span-percent", type=float, default=GT_MIN_X_SPAN_PERCENT)
    parser.add_argument("--gt-min-y-span-percent", type=float, default=GT_MIN_Y_SPAN_PERCENT)
    parser.add_argument("--gt-min-image-hull-percent", type=float, default=GT_MIN_IMAGE_HULL_PERCENT)
    parser.add_argument("--gt-ransac-threshold-px", type=float, default=GT_RANSAC_THRESHOLD_PX)
    parser.add_argument("--gt-closure-max-rotation-error-deg", type=float, default=GT_CLOSURE_MAX_ROTATION_ERROR_DEG)
    parser.add_argument("--gt-closure-max-baseline-error-percent", type=float, default=GT_CLOSURE_MAX_BASELINE_ERROR_PERCENT)

    parser.add_argument("--stereo-rotation-pass-deg", type=float, default=STEREO_ROTATION_PASS_DEG)
    parser.add_argument("--stereo-baseline-pass-percent", type=float, default=STEREO_BASELINE_PASS_PERCENT)
    parser.add_argument("--temporal-rotation-pass-deg", type=float, default=TEMPORAL_ROTATION_PASS_DEG)
    parser.add_argument("--temporal-translation-l2-pass-mm", type=float, default=TEMPORAL_TRANSLATION_L2_PASS_MM)
    parser.add_argument("--min-gt-motion-for-percent-mm", type=float, default=MIN_GT_MOTION_FOR_PERCENT_MM)

    parser.add_argument("--temporal-beam-width", type=int, default=TEMPORAL_BEAM_WIDTH)
    parser.add_argument("--temporal-reprojection-weight", type=float, default=TEMPORAL_REPROJECTION_WEIGHT)
    parser.add_argument("--temporal-rotation-jump-weight", type=float, default=TEMPORAL_ROTATION_JUMP_WEIGHT)
    parser.add_argument("--temporal-translation-jump-weight", type=float, default=TEMPORAL_TRANSLATION_JUMP_WEIGHT)
    parser.add_argument("--temporal-accel-rotation-weight", type=float, default=TEMPORAL_ACCEL_ROTATION_WEIGHT)
    parser.add_argument("--temporal-accel-translation-weight", type=float, default=TEMPORAL_ACCEL_TRANSLATION_WEIGHT)
    parser.add_argument("--temporal-smooth-alpha", type=float, default=TEMPORAL_SMOOTH_ALPHA)
    parser.add_argument("--temporal-smooth-iterations", type=int, default=TEMPORAL_SMOOTH_ITERATIONS)
    parser.add_argument("--temporal-max-post-smooth-marker-rms-px", type=float, default=TEMPORAL_MAX_POST_SMOOTH_MARKER_RMS_PX)

    parser.add_argument("--relation-rotation-gate-deg", type=float, default=RELATION_ROTATION_GATE_DEG)
    parser.add_argument("--relation-translation-gate-mm", type=float, default=RELATION_TRANSLATION_GATE_MM)
    parser.add_argument("--relation-min-support-frames", type=int, default=RELATION_MIN_SUPPORT_FRAMES)
    parser.add_argument("--dual-marker-max-group-rms-px", type=float, default=DUAL_MARKER_MAX_GROUP_RMS_PX)

    parser.add_argument("--sift-max-features", type=int, default=SIFT_MAX_FEATURES)
    parser.add_argument("--sift-ratio", type=float, default=SIFT_RATIO)
    parser.add_argument("--sift-ransac-threshold-px", type=float, default=SIFT_RANSAC_THRESHOLD_PX)
    parser.add_argument("--sift-min-matches", type=int, default=SIFT_MIN_MATCHES)
    parser.add_argument("--sift-max-stored-inliers", type=int, default=SIFT_MAX_STORED_INLIERS)
    parser.add_argument("--sift-min-inlier-rate-percent", type=float, default=SIFT_MIN_INLIER_RATE_PERCENT)
    parser.add_argument("--sift-min-parallax-deg", type=float, default=SIFT_MIN_PARALLAX_DEG)
    parser.add_argument("--sift-max-sampson-p90-px", type=float, default=SIFT_MAX_SAMPSON_P90_PX)
    parser.add_argument("--sift-descriptor-min-clearance-px", type=float, default=SIFT_DESCRIPTOR_MIN_CLEARANCE_PX)
    parser.add_argument("--sift-descriptor-size-factor", type=float, default=SIFT_DESCRIPTOR_SIZE_FACTOR)
    parser.add_argument(
        "--sift-grid-threshold-mode",
        choices=("otsu", "fixed"),
        default=SIFT_GRID_THRESHOLD_MODE,
    )
    parser.add_argument("--sift-grid-black-threshold", type=int, default=SIFT_GRID_BLACK_THRESHOLD)
    parser.add_argument("--sift-grid-threshold-min", type=int, default=SIFT_GRID_THRESHOLD_MIN)
    parser.add_argument("--sift-grid-threshold-max", type=int, default=SIFT_GRID_THRESHOLD_MAX)
    parser.add_argument("--sift-grid-dilation-px", type=int, default=SIFT_GRID_DILATION_PX)
    parser.add_argument("--sift-forbidden-dilation-px", type=int, default=SIFT_FORBIDDEN_DILATION_PX)
    parser.add_argument("--sift-target-extra-scale", type=float, default=SIFT_TARGET_EXTRA_SCALE)
    parser.add_argument(
        "--sift-require-both-target-masks",
        action=argparse.BooleanOptionalAction,
        default=SIFT_REQUIRE_BOTH_TARGET_MASKS,
        help=(
            "Disable SIFT for a frame unless both ID2 and ID5 were detected and "
            "masked, preventing a missed target from leaking into wound features."
        ),
    )
    parser.add_argument(
        "--protect-sift-roi-dark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve dark pixels inside the explicit SIFT ROI. This can retain "
            "wound texture, but visible Grid fragments are then reported as leakage risk."
        ),
    )
    parser.add_argument("--sift-marker-weights", type=parse_marker_weights, default=SIFT_MARKER_WEIGHTS)
    parser.add_argument("--sift-joint-max-marker-rms-px", type=float, default=SIFT_JOINT_MAX_MARKER_RMS_PX)
    parser.add_argument("--sift-joint-max-holdout-median-px", type=float, default=SIFT_JOINT_MAX_HOLDOUT_MEDIAN_PX)
    parser.add_argument("--sift-branch-rotation-weight", type=float, default=0.05)
    parser.add_argument("--sift-branch-direction-weight", type=float, default=0.012)
    parser.add_argument("--sift-branch-sampson-weight", type=float, default=0.8)
    parser.add_argument("--sift-joint-feature-weight", type=float, default=5.0)
    parser.add_argument("--sift-joint-temporal-rotation-weight", type=float, default=8.0)
    parser.add_argument("--sift-joint-temporal-translation-weight", type=float, default=0.04)

    parser.add_argument(
        "--diagnostic-video",
        action=argparse.BooleanOptionalAction,
        default=SAVE_DIAGNOSTIC_VIDEO,
    )
    parser.add_argument("--diagnostic-max-width", type=int, default=DIAGNOSTIC_MAX_WIDTH)
    parser.add_argument("--output", help="Output XLSX path")
    args = parser.parse_args()
    if not args.video:
        parser.error("Specify VIDEO or set VIDEO_PATH at the top of this file")
    if args.frame_step <= 0:
        parser.error("--frame-step must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.temporal_beam_width <= 0:
        parser.error("--temporal-beam-width must be positive")
    if not 0 <= args.temporal_smooth_alpha <= 1:
        parser.error("--temporal-smooth-alpha must be between 0 and 1")
    if args.temporal_max_post_smooth_marker_rms_px <= 0:
        parser.error("--temporal-max-post-smooth-marker-rms-px must be positive")
    if args.sift_grid_threshold_min > args.sift_grid_threshold_max:
        parser.error("SIFT Grid threshold min cannot exceed max")
    if not 0 <= args.sift_min_inlier_rate_percent <= 100:
        parser.error("--sift-min-inlier-rate-percent must be within 0..100")
    if args.sift_min_parallax_deg < 0 or args.sift_max_sampson_p90_px <= 0:
        parser.error("SIFT geometry gates must be positive")
    if args.sift_descriptor_min_clearance_px < 0 or args.sift_descriptor_size_factor < 0:
        parser.error("SIFT descriptor support margins cannot be negative")
    return args


# =============================================================================
# Evaluation tables
# =============================================================================


POSE_HEADERS = [
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_mm",
    "tvec_y_mm",
    "tvec_z_mm",
] + [f"R{row}{column}" for row in range(3) for column in range(3)]

GT_FRAME_HEADERS = [
    "frame_index",
    "timestamp_s",
    "eye",
    "pose_available",
    "gt_valid",
    "status",
    "reject_reason",
    "aruco_grid_marker_count",
    "charuco_corner_count",
    "pnp_inlier_count",
    "pnp_inlier_rate_percent",
    "object_hull_coverage_percent",
    "board_x_span_percent",
    "board_y_span_percent",
    "image_hull_area_percent",
    "reprojection_rms_px",
    "reprojection_median_px",
    "reprojection_p95_px",
    "reprojection_max_px",
    "minimum_board_depth_mm",
    "selected_solver",
    *POSE_HEADERS,
    "processing_time_ms",
]

GT_STEREO_HEADERS = [
    "frame_index",
    "timestamp_s",
    "left_pose_available",
    "right_pose_available",
    "left_gt_valid",
    "right_gt_valid",
    "both_eye_gt_valid",
    "gt_stereo_consistent",
    "status",
    "reject_reason",
    "rotation_error_deg",
    "estimated_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "closure_rotation_jump_deg",
    "closure_translation_jump_mm",
    "json_left_to_right_transfer_rms_px",
    "json_right_to_left_transfer_rms_px",
    *POSE_HEADERS,
]

METHOD_FRAME_HEADERS = [
    "frame_index",
    "timestamp_s",
    "eye",
    "method",
    "pose_world_frame",
    "reference_frame_index",
    "estimate_available",
    "status",
    "reject_reason",
    "estimate_source",
    "id2_found",
    "id5_found",
    "selected_id2_branch",
    "selected_id5_branch",
    "marker_reprojection_rms_px",
    "marker_reprojection_max_px",
    "marker_relation_rotation_residual_deg",
    "marker_relation_translation_residual_mm",
    "temporal_prior_used",
    "prediction_rotation_residual_deg",
    "prediction_translation_residual_mm",
    "sift_used",
    "sift_inliers",
    "sift_sampson_median_px",
    "sift_holdout_median_px",
    *POSE_HEADERS,
    *[f"raw_{header}" for header in POSE_HEADERS],
    "processing_time_ms",
]

TEMPORAL_HEADERS = [
    "source_frame_index",
    "target_frame_index",
    "frame_gap",
    "reference_mode",
    "requested_for_scoring",
    "eye",
    "method",
    "sift_used_for_target",
    "gt_pair_valid",
    "estimate_pair_available",
    "comparable",
    "status",
    "reject_reason",
    "estimated_rotation_angle_deg",
    "gt_rotation_angle_deg",
    "rotation_error_deg",
    "estimated_translation_norm_mm",
    "gt_translation_norm_mm",
    "translation_norm_delta_mm",
    "absolute_translation_norm_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "raw_rotation_error_deg",
    "raw_translation_l2_error_mm",
    "rotation_pass",
    "translation_pass",
    "both_pass",
]

STEREO_CLOSURE_HEADERS = [
    "frame_index",
    "timestamp_s",
    "method",
    "left_sift_used",
    "right_sift_used",
    "both_eye_sift_used",
    "left_estimate_available",
    "right_estimate_available",
    "estimate_available",
    "left_gt_valid",
    "right_gt_valid",
    "both_eye_gt_valid",
    "gt_stereo_consistent",
    "comparable",
    "status",
    "reject_reason",
    "rotation_error_deg",
    "estimated_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "raw_rotation_error_deg",
    "raw_absolute_baseline_error_percent",
    "rotation_pass",
    "baseline_pass",
    "both_pass",
]

SIFT_DIAGNOSTIC_HEADERS = [
    "previous_frame_index",
    "frame_index",
    "eye",
    "status",
    "previous_keypoints",
    "current_keypoints",
    "mutual_matches",
    "essential_inliers",
    "inlier_rate_percent",
    "sampson_median_px",
    "sampson_p90_px",
    "parallax_median_deg",
    "grid_leakage_matches",
    "current_grid_leakage_keypoints",
]

MASK_DIAGNOSTIC_HEADERS = [
    "frame_index",
    "eye",
    "mask_valid",
    "status",
    "black_threshold_mode",
    "black_threshold_value",
    "dilation_px",
    "board_pixels",
    "source_dark_pixels",
    "whitened_pixels",
    "whitened_area_percent",
    "target_forbidden_pixels",
    "target_mask_complete",
    "missing_target_ids",
    "sift_keypoints",
    "grid_leakage_keypoints",
]

RELATION_HEADERS = [
    "eye",
    "available",
    "status",
    "candidate_frames",
    "support_frames",
    "rotation_residual_median_deg",
    "rotation_residual_p95_deg",
    "translation_residual_median_mm",
    "translation_residual_p95_mm",
    *POSE_HEADERS,
]


def prefix_pose_columns(pose, prefix):
    return {f"{prefix}{key}": value for key, value in pose_to_columns(pose).items()}


def common_charuco_transfer_rms(
    source_gt,
    target_gt,
    source_to_target,
    board,
    target_K,
    target_distortion,
):
    if (
        source_gt.pose is None
        or source_gt.charuco_ids is None
        or target_gt.charuco_ids is None
    ):
        return None
    source_ids = set(map(int, source_gt.charuco_ids))
    target_lookup = {
        int(marker_id): point
        for marker_id, point in zip(target_gt.charuco_ids, target_gt.charuco_corners)
    }
    common = sorted(source_ids & set(target_lookup))
    if len(common) < 4:
        return None
    object_all = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    objects = object_all[common]
    observed = np.asarray([target_lookup[index] for index in common], np.float64)
    predicted_pose = compose(source_to_target, source_gt.pose)
    projected = project_points(objects, predicted_pose, target_K, target_distortion)
    errors = np.linalg.norm(projected - observed, axis=1)
    return float(np.sqrt(np.mean(np.square(errors))))


def build_gt_tables(observations, indexes, board, K_by_eye, d_by_eye, json_pose, args):
    frame_rows = []
    stereo_rows = []
    consistency = {}
    previous_closure = None
    for frame_index in indexes:
        for eye in EYES:
            observation = observations[eye].get(frame_index)
            if observation is None:
                continue
            gt = observation.gt
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": observation.timestamp_s,
                    "eye": eye,
                    "pose_available": int(gt.pose_available),
                    "gt_valid": int(gt.valid),
                    "status": gt.status,
                    "reject_reason": gt.reject_reason,
                    "aruco_grid_marker_count": gt.marker_count,
                    "charuco_corner_count": gt.corner_count,
                    "pnp_inlier_count": gt.inlier_count,
                    "pnp_inlier_rate_percent": gt.inlier_rate_percent,
                    "object_hull_coverage_percent": gt.object_coverage_percent,
                    "board_x_span_percent": gt.x_span_percent,
                    "board_y_span_percent": gt.y_span_percent,
                    "image_hull_area_percent": gt.image_hull_percent,
                    "reprojection_rms_px": gt.reprojection_rms_px,
                    "reprojection_median_px": gt.reprojection_median_px,
                    "reprojection_p95_px": gt.reprojection_p95_px,
                    "reprojection_max_px": gt.reprojection_max_px,
                    "minimum_board_depth_mm": gt.min_depth_mm,
                    "selected_solver": gt.solver,
                    **pose_to_columns(gt.pose),
                    "processing_time_ms": gt.processing_time_ms,
                }
            )

        left = observations["L"].get(frame_index)
        right = observations["R"].get(frame_index)
        left_gt = left.gt if left else GtPoseResult(status="FRAME_MISSING")
        right_gt = right.gt if right else GtPoseResult(status="FRAME_MISSING")
        closure = None
        errors = {}
        if left_gt.pose is not None and right_gt.pose is not None:
            closure = compose(right_gt.pose, inverse(left_gt.pose))
            errors = answer_errors(closure, json_pose)
        both_valid = left_gt.valid and right_gt.valid
        consistent = bool(
            both_valid
            and errors
            and errors["rotation_error_deg"] <= args.gt_closure_max_rotation_error_deg
            and errors["absolute_baseline_error_percent"]
            <= args.gt_closure_max_baseline_error_percent
        )
        consistency[frame_index] = consistent
        reject = []
        if not left_gt.valid:
            reject.append(f"L:{left_gt.status}")
        if not right_gt.valid:
            reject.append(f"R:{right_gt.status}")
        if both_valid and not consistent and errors:
            reject.append(
                f"JSON closure rot={errors['rotation_error_deg']:.3f}deg, "
                f"baseline={errors['absolute_baseline_error_percent']:.3f}%"
            )
        rotation_jump = translation_jump = None
        if closure is not None and previous_closure is not None:
            rotation_jump, translation_jump = pose_distance(closure, previous_closure)
        if closure is not None:
            previous_closure = closure
        left_to_right_transfer = common_charuco_transfer_rms(
            left_gt,
            right_gt,
            json_pose,
            board,
            K_by_eye["R"],
            d_by_eye["R"],
        )
        right_to_left_transfer = common_charuco_transfer_rms(
            right_gt,
            left_gt,
            inverse(json_pose),
            board,
            K_by_eye["L"],
            d_by_eye["L"],
        )
        stereo_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / max(args._fps, 1e-9),
                "left_pose_available": int(left_gt.pose_available),
                "right_pose_available": int(right_gt.pose_available),
                "left_gt_valid": int(left_gt.valid),
                "right_gt_valid": int(right_gt.valid),
                "both_eye_gt_valid": int(both_valid),
                "gt_stereo_consistent": int(consistent),
                "status": "OK" if consistent else (
                    "JSON_CLOSURE_GATE_FAILED" if both_valid else "EYE_GT_INVALID"
                ),
                "reject_reason": "; ".join(reject),
                **errors,
                "closure_rotation_jump_deg": rotation_jump,
                "closure_translation_jump_mm": translation_jump,
                "json_left_to_right_transfer_rms_px": left_to_right_transfer,
                "json_right_to_left_transfer_rms_px": right_to_left_transfer,
                **pose_to_columns(closure),
            }
        )
    return frame_rows, stereo_rows, consistency


def choose_reference_frames(method_results, observations, consistency, args):
    """Use the user-requested frame N for every method and both eyes.

    No fallback may consult ChArUco validity or JSON closure: those are answers,
    and answer-dependent reference selection would bias and de-align methods.
    """
    references = {}
    modes = {}
    for method in METHODS:
        for eye in EYES:
            if args.reference_frame in observations[eye]:
                references[(method, eye)] = args.reference_frame
                reference_result = method_results[method][eye].get(
                    args.reference_frame, MethodPoseResult()
                )
                modes[(method, eye)] = (
                    "requested_fixed"
                    if reference_result.available
                    else "requested_fixed_estimate_unavailable"
                )
            else:
                references[(method, eye)] = None
                modes[(method, eye)] = "requested_frame_missing"
    return references, modes


def build_method_pose_rows(method_results, observations, references):
    rows = []
    world_frames = {
        "ID2_TEMPORAL": "ID2",
        "ID5_TEMPORAL": "ID5",
        "ID2_ID5_TEMPORAL": "ID2",
        "ID2_SIFT_TEMPORAL": "ID2",
    }
    for method in METHODS:
        for eye in EYES:
            for frame_index in sorted(observations[eye]):
                result = method_results[method][eye].get(
                    frame_index, MethodPoseResult(status="FRAME_NOT_PROCESSED")
                )
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": observations[eye][frame_index].timestamp_s,
                        "eye": eye,
                        "method": method,
                        "pose_world_frame": world_frames[method],
                        "reference_frame_index": references[(method, eye)],
                        "estimate_available": int(result.available),
                        "status": result.status,
                        "reject_reason": result.reject_reason,
                        "estimate_source": result.source,
                        "id2_found": int(result.id2_found),
                        "id5_found": int(result.id5_found),
                        "selected_id2_branch": result.selected_id2_branch,
                        "selected_id5_branch": result.selected_id5_branch,
                        "marker_reprojection_rms_px": result.marker_reprojection_rms_px,
                        "marker_reprojection_max_px": result.marker_reprojection_max_px,
                        "marker_relation_rotation_residual_deg": result.relation_rotation_residual_deg,
                        "marker_relation_translation_residual_mm": result.relation_translation_residual_mm,
                        "temporal_prior_used": int(result.temporal_prior_used),
                        "prediction_rotation_residual_deg": result.prediction_rotation_residual_deg,
                        "prediction_translation_residual_mm": result.prediction_translation_residual_mm,
                        "sift_used": int(result.sift_used),
                        "sift_inliers": result.sift_inliers,
                        "sift_sampson_median_px": result.sift_sampson_median_px,
                        "sift_holdout_median_px": result.sift_holdout_median_px,
                        **pose_to_columns(result.pose),
                        **prefix_pose_columns(result.raw_pose, "raw_"),
                        "processing_time_ms": result.processing_time_ms,
                    }
                )
    return rows


def build_temporal_rows(
    method_results,
    observations,
    consistency,
    references,
    reference_modes,
    args,
):
    rows = []
    for method in METHODS:
        for eye in EYES:
            reference_index = references[(method, eye)]
            reference_result = (
                method_results[method][eye].get(reference_index)
                if reference_index is not None
                else None
            )
            reference_gt = (
                observations[eye][reference_index].gt
                if reference_index is not None and reference_index in observations[eye]
                else None
            )
            for frame_index in sorted(observations[eye]):
                current_result = method_results[method][eye].get(
                    frame_index, MethodPoseResult()
                )
                current_gt = observations[eye][frame_index].gt
                estimate_pair = bool(
                    reference_result
                    and reference_result.pose is not None
                    and current_result.pose is not None
                )
                gt_pair = bool(
                    reference_gt
                    and reference_gt.pose is not None
                    and current_gt.pose is not None
                    and reference_gt.valid
                    and current_gt.valid
                    and consistency.get(reference_index, False)
                    and consistency.get(frame_index, False)
                )
                is_reference = frame_index == reference_index
                comparable = bool(estimate_pair and gt_pair and not is_reference)
                errors = {}
                raw_errors = {}
                if comparable:
                    estimate_relative = relative_pose(
                        reference_result.pose, current_result.pose
                    )
                    gt_relative = relative_pose(reference_gt.pose, current_gt.pose)
                    errors = temporal_errors(
                        estimate_relative,
                        gt_relative,
                        args.min_gt_motion_for_percent_mm,
                    )
                    if (
                        reference_result.raw_pose is not None
                        and current_result.raw_pose is not None
                    ):
                        raw_relative = relative_pose(
                            reference_result.raw_pose, current_result.raw_pose
                        )
                        raw_errors = temporal_errors(
                            raw_relative,
                            gt_relative,
                            args.min_gt_motion_for_percent_mm,
                        )
                reject = []
                if reference_index is None:
                    reject.append("No common valid reference frame")
                elif is_reference:
                    reject.append("Reference frame is not scored")
                else:
                    if not estimate_pair:
                        reject.append("Estimate pair unavailable")
                    if not gt_pair:
                        reject.append("GT pair invalid or JSON closure gate failed")
                rows.append(
                    {
                        "source_frame_index": reference_index,
                        "target_frame_index": frame_index,
                        "frame_gap": (
                            frame_index - reference_index
                            if reference_index is not None
                            else None
                        ),
                        "reference_mode": reference_modes[(method, eye)],
                        "requested_for_scoring": int(not is_reference),
                        "eye": eye,
                        "method": method,
                        "sift_used_for_target": int(current_result.sift_used),
                        "gt_pair_valid": int(gt_pair),
                        "estimate_pair_available": int(estimate_pair),
                        "comparable": int(comparable),
                        "status": "OK" if comparable else (
                            "REFERENCE_NOT_SCORED" if is_reference else "NOT_COMPARABLE"
                        ),
                        "reject_reason": "; ".join(reject),
                        **errors,
                        "raw_rotation_error_deg": raw_errors.get("rotation_error_deg"),
                        "raw_translation_l2_error_mm": raw_errors.get("translation_l2_error_mm"),
                        "rotation_pass": int(
                            comparable
                            and errors["rotation_error_deg"] < args.temporal_rotation_pass_deg
                        ),
                        "translation_pass": int(
                            comparable
                            and errors["translation_l2_error_mm"]
                            < args.temporal_translation_l2_pass_mm
                        ),
                        "both_pass": int(
                            comparable
                            and errors["rotation_error_deg"] < args.temporal_rotation_pass_deg
                            and errors["translation_l2_error_mm"]
                            < args.temporal_translation_l2_pass_mm
                        ),
                    }
                )
    return rows


def build_stereo_closure_rows(
    method_results,
    observations,
    consistency,
    json_pose,
    args,
):
    rows = []
    common_indexes = sorted(set(observations["L"]) | set(observations["R"]))
    for method in METHODS:
        for frame_index in common_indexes:
            left = method_results[method]["L"].get(frame_index, MethodPoseResult())
            right = method_results[method]["R"].get(frame_index, MethodPoseResult())
            estimate_available = left.pose is not None and right.pose is not None
            gt_left_valid = observations["L"].get(
                frame_index, EyeFrameObservation(0, 0, GtPoseResult(), {}, MaskDiagnostics())
            ).gt.valid
            gt_right_valid = observations["R"].get(
                frame_index, EyeFrameObservation(0, 0, GtPoseResult(), {}, MaskDiagnostics())
            ).gt.valid
            gt_consistent = consistency.get(frame_index, False)
            comparable = bool(estimate_available and gt_consistent)
            errors = {}
            raw_errors = {}
            if estimate_available:
                closure = compose(right.pose, inverse(left.pose))
                errors = answer_errors(closure, json_pose)
                if left.raw_pose is not None and right.raw_pose is not None:
                    raw_closure = compose(right.raw_pose, inverse(left.raw_pose))
                    raw_errors = answer_errors(raw_closure, json_pose)
            reject = []
            if not estimate_available:
                reject.append("Independent left/right method poses are not both available")
            if not gt_consistent:
                reject.append("Independent ChArUco GT did not pass JSON closure gate")
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / max(args._fps, 1e-9),
                    "method": method,
                    "left_sift_used": int(left.sift_used),
                    "right_sift_used": int(right.sift_used),
                    "both_eye_sift_used": int(left.sift_used and right.sift_used),
                    "left_estimate_available": int(left.pose is not None),
                    "right_estimate_available": int(right.pose is not None),
                    "estimate_available": int(estimate_available),
                    "left_gt_valid": int(gt_left_valid),
                    "right_gt_valid": int(gt_right_valid),
                    "both_eye_gt_valid": int(gt_left_valid and gt_right_valid),
                    "gt_stereo_consistent": int(gt_consistent),
                    "comparable": int(comparable),
                    "status": "OK" if comparable else "NOT_COMPARABLE",
                    "reject_reason": "; ".join(reject),
                    **errors,
                    "raw_rotation_error_deg": raw_errors.get("rotation_error_deg"),
                    "raw_absolute_baseline_error_percent": raw_errors.get(
                        "absolute_baseline_error_percent"
                    ),
                    "rotation_pass": int(
                        comparable
                        and errors["rotation_error_deg"] < args.stereo_rotation_pass_deg
                    ),
                    "baseline_pass": int(
                        comparable
                        and errors["absolute_baseline_error_percent"]
                        < args.stereo_baseline_pass_percent
                    ),
                    "both_pass": int(
                        comparable
                        and errors["rotation_error_deg"] < args.stereo_rotation_pass_deg
                        and errors["absolute_baseline_error_percent"]
                        < args.stereo_baseline_pass_percent
                    ),
                }
            )
    return rows


def build_sift_and_mask_rows(observations, feature_edges, args):
    sift_rows = []
    mask_rows = []
    for eye in EYES:
        for frame_index, observation in sorted(observations[eye].items()):
            mask = observation.mask
            mask_rows.append(
                {
                    "frame_index": frame_index,
                    "eye": eye,
                    "mask_valid": int(mask.valid),
                    "status": mask.status,
                    "black_threshold_mode": args.sift_grid_threshold_mode,
                    "black_threshold_value": mask.threshold_value,
                    "dilation_px": args.sift_grid_dilation_px,
                    "board_pixels": mask.board_pixels,
                    "source_dark_pixels": mask.source_dark_pixels,
                    "whitened_pixels": mask.whitened_pixels,
                    "whitened_area_percent": mask.whitened_area_percent,
                    "target_forbidden_pixels": mask.target_forbidden_pixels,
                    "target_mask_complete": int(mask.target_mask_complete),
                    "missing_target_ids": mask.missing_target_ids,
                    "sift_keypoints": mask.sift_keypoints,
                    "grid_leakage_keypoints": mask.grid_leakage_keypoints,
                }
            )
        for (_previous, _current), edge in sorted(feature_edges[eye].items()):
            current_observation = observations[eye].get(edge.current_frame)
            sift_rows.append(
                {
                    "previous_frame_index": edge.previous_frame,
                    "frame_index": edge.current_frame,
                    "eye": eye,
                    "status": edge.status,
                    "previous_keypoints": edge.previous_keypoints,
                    "current_keypoints": edge.current_keypoints,
                    "mutual_matches": edge.mutual_matches,
                    "essential_inliers": edge.essential_inliers,
                    "inlier_rate_percent": edge.inlier_rate_percent,
                    "sampson_median_px": edge.sampson_median_px,
                    "sampson_p90_px": edge.sampson_p90_px,
                    "parallax_median_deg": edge.parallax_median_deg,
                    "grid_leakage_matches": edge.grid_leakage_matches,
                    "current_grid_leakage_keypoints": (
                        current_observation.sift_grid_leakage_keypoints
                        if current_observation
                        else None
                    ),
                }
            )
    return sift_rows, mask_rows


def relation_rows(relations):
    rows = []
    for eye in EYES:
        relation = relations[eye]
        rows.append(
            {
                "eye": eye,
                "available": int(relation.available),
                "status": relation.status,
                "candidate_frames": relation.candidate_frames,
                "support_frames": relation.support_frames,
                "rotation_residual_median_deg": relation.rotation_residual_median_deg,
                "rotation_residual_p95_deg": relation.rotation_residual_p95_deg,
                "translation_residual_median_mm": relation.translation_residual_median_mm,
                "translation_residual_p95_mm": relation.translation_residual_p95_mm,
                **pose_to_columns(relation.pose),
            }
        )
    return rows


# =============================================================================
# Formula-driven workbook, charts and CSV files
# =============================================================================


SUMMARY_HEADERS = [
    "scope",
    "method",
    "eye",
    "secondary_metric",
    "requested_rows",
    "gt_valid_rows",
    "estimated_rows",
    "comparable_rows",
    "rotation_error_mean_deg",
    "rotation_error_median_deg",
    "rotation_error_p95_deg",
    "secondary_error_mean",
    "secondary_error_median",
    "secondary_error_p95",
    "rotation_pass_rows",
    "secondary_pass_rows",
    "both_pass_rows",
    "rotation_pass_rate_of_comparable_percent",
    "secondary_pass_rate_of_comparable_percent",
    "both_pass_rate_of_requested_percent",
    "both_pass_rate_of_gt_valid_percent",
    "both_pass_rate_of_estimated_percent",
    "both_pass_rate_of_comparable_percent",
]


def excel_range(sheet, headers, name, row_count):
    column = excel_column(headers.index(name) + 1)
    return f"'{sheet}'!${column}$2:${column}${row_count + 1}"


def quote_excel_text(value):
    return '"' + str(value).replace('"', '""') + '"'


def compute_group_summary(rows, secondary_name):
    requested = [row for row in rows if row.get("requested_for_scoring", 1) == 1]
    comparable = [row for row in rows if row.get("comparable") == 1]
    rotation = [row.get("rotation_error_deg") for row in comparable]
    secondary = [row.get(secondary_name) for row in comparable]
    rotation = [value for value in rotation if finite(value) is not None]
    secondary = [value for value in secondary if finite(value) is not None]
    return {
        "requested": len(requested),
        "gt_valid": sum(
            bool(row.get("gt_pair_valid", row.get("gt_stereo_consistent")))
            for row in requested
        ),
        "estimated": sum(
            bool(row.get("estimate_pair_available", row.get("estimate_available")))
            for row in requested
        ),
        "comparable": len(comparable),
        "rotation_mean": float(np.mean(rotation)) if rotation else None,
        "rotation_median": percentile(rotation, 50),
        "rotation_p95": percentile(rotation, 95),
        "secondary_mean": float(np.mean(secondary)) if secondary else None,
        "secondary_median": percentile(secondary, 50),
        "secondary_p95": percentile(secondary, 95),
        "rotation_pass": sum(bool(row.get("rotation_pass")) for row in comparable),
        "secondary_pass": sum(
            bool(row.get("translation_pass", row.get("baseline_pass")))
            for row in comparable
        ),
        "both_pass": sum(bool(row.get("both_pass")) for row in comparable),
    }


def formula_with_cache(formula, value):
    if value is None or not math.isfinite(float(value)):
        value = 0.0
    return ExcelFormula(formula, value)


def build_summary_rows(temporal_rows, closure_rows, settings_cells):
    summary_rows = []

    def add_scope(
        scope,
        method,
        eye,
        rows,
        sheet,
        headers,
        secondary_name,
        extra_filters=(),
    ):
        is_temporal = sheet == "Temporal RPE"
        stats = compute_group_summary(rows, secondary_name)
        count = len(temporal_rows) if sheet == "Temporal RPE" else len(closure_rows)
        method_range = excel_range(sheet, headers, "method", count)
        comparable_range = excel_range(sheet, headers, "comparable", count)
        rotation_range = excel_range(sheet, headers, "rotation_error_deg", count)
        secondary_range = excel_range(sheet, headers, secondary_name, count)
        gt_name = "gt_pair_valid" if sheet == "Temporal RPE" else "gt_stereo_consistent"
        estimated_name = (
            "estimate_pair_available" if sheet == "Temporal RPE" else "estimate_available"
        )
        gt_range = excel_range(sheet, headers, gt_name, count)
        estimated_range = excel_range(sheet, headers, estimated_name, count)
        criteria = [f"{method_range},{quote_excel_text(method)}"]
        if eye:
            eye_range = excel_range(sheet, headers, "eye", count)
            criteria.append(f"{eye_range},{quote_excel_text(eye)}")
        for column_name, value in extra_filters:
            value_range = excel_range(sheet, headers, column_name, count)
            formatted = quote_excel_text(value) if isinstance(value, str) else value
            criteria.append(f"{value_range},{formatted}")
        if is_temporal:
            scoring_range = excel_range(
                sheet, headers, "requested_for_scoring", count
            )
            criteria.append(f"{scoring_range},1")
        criteria_text = ",".join(criteria)
        requested_formula = f"COUNTIFS({criteria_text})"
        gt_formula = f"COUNTIFS({criteria_text},{gt_range},1)"
        estimated_formula = f"COUNTIFS({criteria_text},{estimated_range},1)"
        comparable_formula = f"COUNTIFS({criteria_text},{comparable_range},1)"
        rotation_pass_formula = (
            f"COUNTIFS({criteria_text},{comparable_range},1,{rotation_range},\"<\"&"
            f"{settings_cells['temporal_rotation_pass_deg' if is_temporal else 'stereo_rotation_pass_deg']})"
        )
        secondary_setting = (
            settings_cells["temporal_translation_l2_pass_mm"]
            if is_temporal
            else settings_cells["stereo_baseline_error_pass_percent"]
        )
        secondary_pass_formula = (
            f"COUNTIFS({criteria_text},{comparable_range},1,{secondary_range},\"<\"&{secondary_setting})"
        )
        both_formula = (
            f"COUNTIFS({criteria_text},{comparable_range},1,{rotation_range},\"<\"&"
            f"{settings_cells['temporal_rotation_pass_deg' if is_temporal else 'stereo_rotation_pass_deg']},"
            f"{secondary_range},\"<\"&{secondary_setting})"
        )
        mean_rotation_formula = (
            f"IFERROR(AVERAGEIFS({rotation_range},{criteria_text},"
            f"{comparable_range},1),0)"
        )
        mean_secondary_formula = (
            f"IFERROR(AVERAGEIFS({secondary_range},{criteria_text},"
            f"{comparable_range},1),0)"
        )
        both = stats["both_pass"]
        summary_rows.append(
            {
                "scope": scope,
                "method": method,
                "eye": eye or "BOTH",
                "secondary_metric": secondary_name,
                "requested_rows": formula_with_cache(requested_formula, stats["requested"]),
                "gt_valid_rows": formula_with_cache(gt_formula, stats["gt_valid"]),
                "estimated_rows": formula_with_cache(estimated_formula, stats["estimated"]),
                "comparable_rows": formula_with_cache(comparable_formula, stats["comparable"]),
                "rotation_error_mean_deg": formula_with_cache(mean_rotation_formula, stats["rotation_mean"]),
                "rotation_error_median_deg": stats["rotation_median"],
                "rotation_error_p95_deg": stats["rotation_p95"],
                "secondary_error_mean": formula_with_cache(mean_secondary_formula, stats["secondary_mean"]),
                "secondary_error_median": stats["secondary_median"],
                "secondary_error_p95": stats["secondary_p95"],
                "rotation_pass_rows": formula_with_cache(rotation_pass_formula, stats["rotation_pass"]),
                "secondary_pass_rows": formula_with_cache(secondary_pass_formula, stats["secondary_pass"]),
                "both_pass_rows": formula_with_cache(both_formula, both),
                "rotation_pass_rate_of_comparable_percent": formula_with_cache(
                    f"IFERROR({rotation_pass_formula}/{comparable_formula}*100,0)",
                    stats["rotation_pass"] / stats["comparable"] * 100.0
                    if stats["comparable"] else 0.0,
                ),
                "secondary_pass_rate_of_comparable_percent": formula_with_cache(
                    f"IFERROR({secondary_pass_formula}/{comparable_formula}*100,0)",
                    stats["secondary_pass"] / stats["comparable"] * 100.0
                    if stats["comparable"] else 0.0,
                ),
                "both_pass_rate_of_requested_percent": formula_with_cache(
                    f"IFERROR({both_formula}/{requested_formula}*100,0)",
                    both / stats["requested"] * 100.0 if stats["requested"] else 0.0,
                ),
                "both_pass_rate_of_gt_valid_percent": formula_with_cache(
                    f"IFERROR({both_formula}/{gt_formula}*100,0)",
                    both / stats["gt_valid"] * 100.0 if stats["gt_valid"] else 0.0,
                ),
                "both_pass_rate_of_estimated_percent": formula_with_cache(
                    f"IFERROR({both_formula}/{estimated_formula}*100,0)",
                    both / stats["estimated"] * 100.0 if stats["estimated"] else 0.0,
                ),
                "both_pass_rate_of_comparable_percent": formula_with_cache(
                    f"IFERROR({both_formula}/{comparable_formula}*100,0)",
                    both / stats["comparable"] * 100.0 if stats["comparable"] else 0.0,
                ),
            }
        )

    for method in METHODS:
        for eye in EYES:
            group = [
                row
                for row in temporal_rows
                if row["method"] == method and row["eye"] == eye
            ]
            add_scope(
                "TEMPORAL_RPE",
                method,
                eye,
                group,
                "Temporal RPE",
                TEMPORAL_HEADERS,
                "translation_l2_error_mm",
            )
        group = [row for row in closure_rows if row["method"] == method]
        add_scope(
            "STEREO_CLOSURE",
            method,
            None,
            group,
            "Stereo Closure",
            STEREO_CLOSURE_HEADERS,
            "absolute_baseline_error_percent",
        )

    # Keep the headline ID2_SIFT_TEMPORAL row as the deployable hybrid
    # pipeline, then expose truly SIFT-assisted and fallback-only subsets so
    # the experiment can quantify SIFT's contribution instead of pooling it.
    for eye in EYES:
        for sift_used, scope in (
            (1, "TEMPORAL_RPE_SIFT_ASSISTED_TARGET"),
            (0, "TEMPORAL_RPE_SIFT_FALLBACK_TARGET"),
        ):
            group = [
                row
                for row in temporal_rows
                if row["method"] == "ID2_SIFT_TEMPORAL"
                and row["eye"] == eye
                and row["sift_used_for_target"] == sift_used
            ]
            add_scope(
                scope,
                "ID2_SIFT_TEMPORAL",
                eye,
                group,
                "Temporal RPE",
                TEMPORAL_HEADERS,
                "translation_l2_error_mm",
                (("sift_used_for_target", sift_used),),
            )
    for sift_used, scope in (
        (1, "STEREO_CLOSURE_SIFT_ASSISTED_BOTH_EYES"),
        (0, "STEREO_CLOSURE_SIFT_PARTIAL_OR_FALLBACK"),
    ):
        group = [
            row
            for row in closure_rows
            if row["method"] == "ID2_SIFT_TEMPORAL"
            and row["both_eye_sift_used"] == sift_used
        ]
        add_scope(
            scope,
            "ID2_SIFT_TEMPORAL",
            None,
            group,
            "Stereo Closure",
            STEREO_CLOSURE_HEADERS,
            "absolute_baseline_error_percent",
            (("both_eye_sift_used", sift_used),),
        )
    return summary_rows


def apply_detail_formulas(temporal_rows, closure_rows, settings_cells):
    temporal_columns = {
        name: excel_column(TEMPORAL_HEADERS.index(name) + 1)
        for name in ("comparable", "rotation_error_deg", "translation_l2_error_mm")
    }
    for excel_row, row in enumerate(temporal_rows, start=2):
        comparable = f"${temporal_columns['comparable']}{excel_row}"
        rotation = f"${temporal_columns['rotation_error_deg']}{excel_row}"
        translation = f"${temporal_columns['translation_l2_error_mm']}{excel_row}"
        row["rotation_pass"] = ExcelFormula(
            f"IF({comparable}=1,--({rotation}<{settings_cells['temporal_rotation_pass_deg']}),0)",
            row["rotation_pass"],
        )
        row["translation_pass"] = ExcelFormula(
            f"IF({comparable}=1,--({translation}<{settings_cells['temporal_translation_l2_pass_mm']}),0)",
            row["translation_pass"],
        )
        row["both_pass"] = ExcelFormula(
            f"IF({comparable}=1,--(AND({rotation}<{settings_cells['temporal_rotation_pass_deg']},"
            f"{translation}<{settings_cells['temporal_translation_l2_pass_mm']})),0)",
            row["both_pass"],
        )

    closure_columns = {
        name: excel_column(STEREO_CLOSURE_HEADERS.index(name) + 1)
        for name in ("comparable", "rotation_error_deg", "absolute_baseline_error_percent")
    }
    for excel_row, row in enumerate(closure_rows, start=2):
        comparable = f"${closure_columns['comparable']}{excel_row}"
        rotation = f"${closure_columns['rotation_error_deg']}{excel_row}"
        baseline = f"${closure_columns['absolute_baseline_error_percent']}{excel_row}"
        row["rotation_pass"] = ExcelFormula(
            f"IF({comparable}=1,--({rotation}<{settings_cells['stereo_rotation_pass_deg']}),0)",
            row["rotation_pass"],
        )
        row["baseline_pass"] = ExcelFormula(
            f"IF({comparable}=1,--({baseline}<{settings_cells['stereo_baseline_error_pass_percent']}),0)",
            row["baseline_pass"],
        )
        row["both_pass"] = ExcelFormula(
            f"IF({comparable}=1,--(AND({rotation}<{settings_cells['stereo_rotation_pass_deg']},"
            f"{baseline}<{settings_cells['stereo_baseline_error_pass_percent']})),0)",
            row["both_pass"],
        )


def write_csv_rows(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for header in headers:
                value = row.get(header)
                clean[header] = value.cached_value if isinstance(value, ExcelFormula) else value
            writer.writerow(clean)


CHART_HEADERS = [
    "frame_index",
    "stereo_rotation_threshold_deg",
    "stereo_baseline_threshold_percent",
    "gt_rotation_gate_deg",
    "gt_baseline_gate_percent",
    *[f"{method}_rotation_error_deg" for method in METHODS],
    *[f"{method}_baseline_error_percent" for method in METHODS],
    "gt_closure_rotation_error_deg",
    "gt_closure_baseline_error_percent",
    "method_index",
    "method_name",
    "pass_rate_requested_percent",
    "pass_rate_gt_valid_percent",
    "pass_rate_estimated_percent",
    "pass_rate_comparable_percent",
]


def cached_value(value):
    return value.cached_value if isinstance(value, ExcelFormula) else value


def build_chart_rows(indexes, closure_rows, gt_stereo_rows, summary_rows, settings_cells, args):
    closure_lookup = {
        (row["frame_index"], row["method"]): row for row in closure_rows
    }
    gt_lookup = {row["frame_index"]: row for row in gt_stereo_rows}
    summary_lookup = {
        row["method"]: (row, excel_row)
        for excel_row, row in enumerate(summary_rows, start=2)
        if row["scope"] == "STEREO_CLOSURE"
    }
    rows = []
    for position, frame_index in enumerate(indexes):
        row = {
            "frame_index": frame_index,
            "gt_rotation_gate_deg": args.gt_closure_max_rotation_error_deg,
            "gt_baseline_gate_percent": args.gt_closure_max_baseline_error_percent,
        }
        # ExcelFormula formulas need the sheet qualifier.  The custom writer
        # accepts formulas without a leading '='.
        row["stereo_rotation_threshold_deg"] = ExcelFormula(
            settings_cells["stereo_rotation_pass_deg"], args.stereo_rotation_pass_deg
        )
        row["stereo_baseline_threshold_percent"] = ExcelFormula(
            settings_cells["stereo_baseline_error_pass_percent"],
            args.stereo_baseline_pass_percent,
        )
        for method in METHODS:
            closure = closure_lookup.get((frame_index, method), {})
            row[f"{method}_rotation_error_deg"] = closure.get("rotation_error_deg")
            row[f"{method}_baseline_error_percent"] = closure.get(
                "absolute_baseline_error_percent"
            )
        gt = gt_lookup.get(frame_index, {})
        row["gt_closure_rotation_error_deg"] = gt.get("rotation_error_deg")
        row["gt_closure_baseline_error_percent"] = gt.get(
            "absolute_baseline_error_percent"
        )
        rows.append(row)

    # The method pass-rate helper table needs four rows even for a very short
    # smoke-test clip.  Blank frame cells are ignored by the frame charts.
    while len(rows) < len(METHODS):
        rows.append(
            {
                "frame_index": None,
                "stereo_rotation_threshold_deg": None,
                "stereo_baseline_threshold_percent": None,
                "gt_rotation_gate_deg": None,
                "gt_baseline_gate_percent": None,
            }
        )
    for position, method in enumerate(METHODS):
        row = rows[position]
        summary, summary_excel_row = summary_lookup.get(method, ({}, None))
        row["method_index"] = position + 1
        row["method_name"] = method
        for chart_name, summary_name in (
            ("pass_rate_requested_percent", "both_pass_rate_of_requested_percent"),
            ("pass_rate_gt_valid_percent", "both_pass_rate_of_gt_valid_percent"),
            ("pass_rate_estimated_percent", "both_pass_rate_of_estimated_percent"),
            ("pass_rate_comparable_percent", "both_pass_rate_of_comparable_percent"),
        ):
            value = cached_value(summary.get(summary_name))
            if summary_excel_row is None:
                row[chart_name] = value
            else:
                summary_column = excel_column(SUMMARY_HEADERS.index(summary_name) + 1)
                row[chart_name] = ExcelFormula(
                    f"'Method Summary'!${summary_column}${summary_excel_row}",
                    value or 0.0,
                )
    return rows


def add_charts_to_workbook(path, chart_rows, charts_sheet_index):
    if not chart_rows:
        return
    last_row = len(chart_rows) + 1
    x_values = [row["frame_index"] for row in chart_rows]
    x_formula = f"'Charts'!$A$2:$A${last_row}"

    def chart_column(name):
        return excel_column(CHART_HEADERS.index(name) + 1)

    charts = []
    rotation_series = []
    baseline_series = []
    for method in METHODS:
        rotation_name = f"{method}_rotation_error_deg"
        baseline_name = f"{method}_baseline_error_percent"
        rotation_series.append(
            {
                "name": method,
                "formula": f"'Charts'!${chart_column(rotation_name)}$2:${chart_column(rotation_name)}${last_row}",
                "values": [row.get(rotation_name) for row in chart_rows],
                "color": METHOD_COLORS[method],
            }
        )
        baseline_series.append(
            {
                "name": method,
                "formula": f"'Charts'!${chart_column(baseline_name)}$2:${chart_column(baseline_name)}${last_row}",
                "values": [row.get(baseline_name) for row in chart_rows],
                "color": METHOD_COLORS[method],
            }
        )
    rotation_series.append(
        {
            "name": "Rotation threshold",
            "formula": f"'Charts'!$B$2:$B${last_row}",
            "values": [cached_value(row["stereo_rotation_threshold_deg"]) for row in chart_rows],
            "color": "C00000",
            "threshold": True,
        }
    )
    baseline_series.append(
        {
            "name": "Baseline threshold",
            "formula": f"'Charts'!$C$2:$C${last_row}",
            "values": [cached_value(row["stereo_baseline_threshold_percent"]) for row in chart_rows],
            "color": "C00000",
            "threshold": True,
        }
    )
    charts.append(
        build_frame_scatter_chart_xml(
            "Independent monocular stereo-closure rotation error",
            "Rotation error (deg)",
            x_formula,
            x_values,
            rotation_series,
            2200000000,
        )
    )
    charts.append(
        build_frame_scatter_chart_xml(
            "Independent monocular stereo-closure baseline error",
            "Absolute baseline error (%)",
            x_formula,
            x_values,
            baseline_series,
            2210000000,
        )
    )
    gt_rotation_name = "gt_closure_rotation_error_deg"
    charts.append(
        build_frame_scatter_chart_xml(
            "Independent ChArUco GT closure rotation check",
            "Rotation error vs JSON (deg)",
            x_formula,
            x_values,
            [
                {
                    "name": "ChArUco closure",
                    "formula": f"'Charts'!${chart_column(gt_rotation_name)}$2:${chart_column(gt_rotation_name)}${last_row}",
                    "values": [row.get(gt_rotation_name) for row in chart_rows],
                    "color": "4472C4",
                },
                {
                    "name": "GT rotation gate",
                    "formula": f"'Charts'!$D$2:$D${last_row}",
                    "values": [row.get("gt_rotation_gate_deg") for row in chart_rows],
                    "color": "C00000",
                    "threshold": True,
                },
            ],
            2220000000,
        )
    )
    gt_baseline_name = "gt_closure_baseline_error_percent"
    charts.append(
        build_frame_scatter_chart_xml(
            "Independent ChArUco GT closure baseline check",
            "Absolute baseline error (%)",
            x_formula,
            x_values,
            [
                {
                    "name": "ChArUco closure",
                    "formula": f"'Charts'!${chart_column(gt_baseline_name)}$2:${chart_column(gt_baseline_name)}${last_row}",
                    "values": [row.get(gt_baseline_name) for row in chart_rows],
                    "color": "70AD47",
                },
                {
                    "name": "GT baseline gate",
                    "formula": f"'Charts'!$E$2:$E${last_row}",
                    "values": [row.get("gt_baseline_gate_percent") for row in chart_rows],
                    "color": "C00000",
                    "threshold": True,
                },
            ],
            2230000000,
        )
    )

    # The fifth scatter chart uses method index 1..4.  The adjacent helper
    # table gives the exact method-name mapping.
    method_count = len(METHODS)
    method_x = list(range(1, method_count + 1))
    method_x_column = chart_column("method_index")
    pass_series = []
    for name, color, label in (
        ("pass_rate_requested_percent", "4472C4", "Requested denominator"),
        ("pass_rate_gt_valid_percent", "ED7D31", "GT-valid denominator"),
        ("pass_rate_estimated_percent", "70AD47", "Estimated denominator"),
        ("pass_rate_comparable_percent", "7030A0", "Comparable denominator"),
    ):
        column = chart_column(name)
        pass_series.append(
            {
                "name": label,
                "formula": f"'Charts'!${column}$2:${column}${method_count + 1}",
                "values": [
                    cached_value(chart_rows[index].get(name))
                    for index in range(method_count)
                ],
                "color": color,
            }
        )
    pass_chart = build_frame_scatter_chart_xml(
            "Stereo both-pass rate by method and denominator",
            "Both-pass rate (%)",
            f"'Charts'!${method_x_column}$2:${method_x_column}${method_count + 1}",
            method_x,
            pass_series,
            2240000000,
        ).replace(">Frame index<", ">Method index<", 1)
    charts.append(pass_chart)

    drawing = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        + "".join(
            chart_anchor_xml(index, f"rId{index}", (index - 1) * 19, index * 19 - 1)
            for index in range(1, len(charts) + 1)
        )
        + "</xdr:wsDr>"
    )
    drawing_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            '<Relationship '
            f'Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
            f'Target="../charts/chart{index}.xml"/>'
            for index in range(1, len(charts) + 1)
        )
        + "</Relationships>"
    )
    sheet_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/></Relationships>'
    )
    parts = {
        f"xl/worksheets/_rels/sheet{charts_sheet_index}.xml.rels": sheet_relationships,
        "xl/drawings/drawing1.xml": drawing,
        "xl/drawings/_rels/drawing1.xml.rels": drawing_relationships,
        **{
            f"xl/charts/chart{index}.xml": chart
            for index, chart in enumerate(charts, start=1)
        },
    }
    path = Path(path)
    temporary = tempfile.NamedTemporaryFile(
        prefix=path.stem + "_charts_",
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
                if info.filename == f"xl/worksheets/sheet{charts_sheet_index}.xml":
                    worksheet = data.decode("utf-8")
                    if "xmlns:r=" not in worksheet.split(">", 1)[0]:
                        worksheet = worksheet.replace(
                            "<worksheet ",
                            '<worksheet xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ',
                            1,
                        )
                    worksheet = worksheet.replace(
                        "</worksheet>", '<drawing r:id="rId1"/></worksheet>'
                    )
                    data = worksheet.encode("utf-8")
                elif info.filename == "[Content_Types].xml":
                    content_types = data.decode("utf-8")
                    overrides = (
                        '<Override PartName="/xl/drawings/drawing1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                        + "".join(
                            '<Override '
                            f'PartName="/xl/charts/chart{index}.xml" '
                            'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                            for index in range(1, len(charts) + 1)
                        )
                    )
                    data = content_types.replace(
                        "</Types>", overrides + "</Types>"
                    ).encode("utf-8")
                destination.writestr(info, data)
            for name, data in parts.items():
                destination.writestr(name, data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_xlsx_package(path):
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"XLSX ZIP CRC failure: {bad}")
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                try:
                    ElementTree.fromstring(archive.read(name))
                except ElementTree.ParseError as error:
                    raise RuntimeError(f"Invalid XML in {name}: {error}") from error


def make_pose_comparison_video(
    video_path,
    indexes,
    closure_rows,
    output_path,
    fps,
    args,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen video for diagnostics: {video_path}")
    lookup = {
        (row["frame_index"], row["method"]): row for row in closure_rows
    }
    capture.set(cv2.CAP_PROP_POS_FRAMES, indexes[0])
    ok, first = capture.read()
    if not ok:
        capture.release()
        return
    panel_height = 125
    writer, target_size = initialize_video_writer(
        output_path,
        (first.shape[1], first.shape[0] + panel_height),
        fps / max(1, args.frame_step),
        args.diagnostic_max_width,
    )
    try:
        for frame_index, frame in iter_selected_video_frames(capture, indexes):
            if frame is None:
                continue
            panel = np.full((panel_height, frame.shape[1], 3), 245, np.uint8)
            lines = [f"F{frame_index} independent L/R pose -> JSON stereo closure"]
            for method in METHODS:
                row = lookup.get((frame_index, method), {})
                rotation = finite(row.get("rotation_error_deg"))
                baseline = finite(row.get("absolute_baseline_error_percent"))
                lines.append(
                    f"{method}: status={row.get('status', 'missing')} "
                    f"rot={rotation if rotation is not None else 'NA'} deg "
                    f"baseline={baseline if baseline is not None else 'NA'} %"
                )
            add_text_lines(panel, lines, origin=(10, 20), color=(0, 0, 180))
            canvas = np.vstack([frame, panel])
            writer.write(cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()


def build_settings(args, paths):
    definitions = [
        (
            "stereo_rotation_pass_deg",
            args.stereo_rotation_pass_deg,
            "deg",
            "Stereo Closure rotation must be strictly below this value.",
        ),
        (
            "stereo_baseline_error_pass_percent",
            args.stereo_baseline_pass_percent,
            "%",
            "Stereo Closure absolute baseline error must be strictly below this value.",
        ),
        (
            "temporal_rotation_pass_deg",
            args.temporal_rotation_pass_deg,
            "deg",
            "Temporal RPE rotation must be strictly below this value.",
        ),
        (
            "temporal_translation_l2_pass_mm",
            args.temporal_translation_l2_pass_mm,
            "mm",
            "Temporal RPE translation-vector L2 error threshold.",
        ),
        (
            "min_gt_motion_for_percent_mm",
            args.min_gt_motion_for_percent_mm,
            "mm",
            "Do not report temporal translation-norm percent below this GT motion.",
        ),
        ("gt_min_charuco_corners", args.gt_min_charuco_corners, "corners", "Independent per-eye GT gate."),
        ("gt_min_pnp_inliers", args.gt_min_pnp_inliers, "corners", "Independent per-eye GT gate."),
        ("gt_min_inlier_rate_percent", args.gt_min_inlier_rate_percent, "%", "Independent per-eye GT gate."),
        ("gt_max_reprojection_rms_px", args.gt_max_reprojection_rms_px, "px", "Independent per-eye GT gate."),
        ("gt_max_reprojection_p95_px", args.gt_max_reprojection_p95_px, "px", "Independent per-eye GT gate."),
        ("gt_min_object_coverage_percent", args.gt_min_object_coverage_percent, "%", "ChArUco internal-corner convex-hull coverage gate."),
        ("gt_min_x_span_percent", args.gt_min_x_span_percent, "%", "ChArUco object-space X span gate."),
        ("gt_min_y_span_percent", args.gt_min_y_span_percent, "%", "ChArUco object-space Y span gate."),
        ("gt_min_image_hull_percent", args.gt_min_image_hull_percent, "%", "Detected inlier image-hull area gate."),
        ("gt_closure_max_rotation_error_deg", args.gt_closure_max_rotation_error_deg, "deg", "Post-estimation ChArUco stereo closure gate against JSON."),
        ("gt_closure_max_baseline_error_percent", args.gt_closure_max_baseline_error_percent, "%", "Post-estimation ChArUco stereo closure gate against JSON."),
        ("temporal_beam_width", args.temporal_beam_width, "hypotheses", "Offline IPPE temporal branch beam width."),
        ("temporal_smooth_alpha", args.temporal_smooth_alpha, "ratio", "Small constant-velocity trajectory smoothing correction."),
        ("temporal_max_post_smooth_marker_rms_px", args.temporal_max_post_smooth_marker_rms_px, "px", "Reject a smoothed pose and retain its observation-constrained pose above this marker RMS."),
        ("relation_rotation_gate_deg", args.relation_rotation_gate_deg, "deg", "ID2-ID5 relation consensus rotation gate."),
        ("relation_translation_gate_mm", args.relation_translation_gate_mm, "mm", "ID2-ID5 relation consensus translation gate."),
        ("relation_min_support_frames", args.relation_min_support_frames, "frames", "Minimum co-visible frames for dual-marker relation."),
        ("sift_grid_threshold_mode", args.sift_grid_threshold_mode, "", "Dark-pixel whitening mode inside projected ChArUco board."),
        ("sift_grid_black_threshold", args.sift_grid_black_threshold, "gray", "Used only in fixed mode."),
        ("sift_grid_dilation_px", args.sift_grid_dilation_px, "px", "Expands whitened Grid dark pixels."),
        ("sift_forbidden_dilation_px", args.sift_forbidden_dilation_px, "px", "Descriptor safety margin around small targets/Grid risk edges."),
        ("sift_min_inlier_rate_percent", args.sift_min_inlier_rate_percent, "%", "Reject weak Essential-matrix edges."),
        ("sift_min_parallax_deg", args.sift_min_parallax_deg, "deg", "Reject translation-direction-degenerate SIFT edges."),
        ("sift_max_sampson_p90_px", args.sift_max_sampson_p90_px, "px", "Reject SIFT edges with poor epipolar P90 residual."),
        ("sift_descriptor_min_clearance_px", args.sift_descriptor_min_clearance_px, "px", "Minimum descriptor support clearance from every forbidden boundary."),
        ("sift_descriptor_size_factor", args.sift_descriptor_size_factor, "x kp.size", "Scale-aware descriptor support clearance."),
        ("sift_require_both_target_masks", int(args.sift_require_both_target_masks), "bool", "Disable SIFT when either ID2 or ID5 target mask is unavailable."),
        ("protect_sift_roi_dark", int(args.protect_sift_roi_dark), "bool", "May preserve wound dark texture but can preserve Grid fragments; inspect leakage columns."),
        ("left_sift_roi", str(args.left_sift_roi), "one-eye px", "Optional wound SIFT ROI."),
        ("right_sift_roi", str(args.right_sift_roi), "one-eye px", "Optional wound SIFT ROI."),
        ("video", str(paths["video"]), "", "Input SBS video."),
        ("calibration", str(paths["calibration"]), "", "Intrinsics; JSON extrinsic is evaluation-only."),
        ("charuco_metadata", str(paths["charuco_metadata"]), "", "Authoritative large-board specification."),
        ("id2_metadata", str(paths["id2_metadata"]), "", "Authoritative ID2 target specification."),
        ("id5_metadata", str(paths["id5_metadata"]), "", "Authoritative ID5 target specification."),
    ]
    rows = [
        {
            "parameter": parameter,
            "value": value,
            "unit": unit,
            "description": description,
        }
        for parameter, value, unit, description in definitions
    ]
    cells = {
        row["parameter"]: f"'Settings'!$B${index}"
        for index, row in enumerate(rows, start=2)
    }
    return rows, cells


def build_protocol_rows(args):
    return [
        {
            "item": "Pose convention",
            "detail": (
                "PnP pose maps pattern/world coordinates to camera coordinates. "
                "Temporal RT = P_current o inverse(P_reference); stereo closure "
                "= P_right o inverse(P_left)."
            ),
        },
        {
            "item": "Strict left/right independence",
            "detail": (
                "Every method estimates left and right trajectories separately. "
                "No stereo match, baseline range, horizontal-rig prior, shared "
                "trajectory, or JSON extrinsic is used by a test estimator."
            ),
        },
        {
            "item": "ChArUco GT",
            "detail": (
                "Each eye uses its own ChArUco RANSAC/IPPE/LM PnP. JSON is used "
                "only after both poses exist, to check independent stereo closure."
            ),
        },
        {
            "item": "ID2/ID5 temporal",
            "detail": (
                "All positive-depth IPPE_SQUARE branches are retained. An offline "
                "beam search uses reprojection, motion continuity and acceleration; "
                "results are explicitly labeled offline temporal."
            ),
        },
        {
            "item": "ID2+ID5",
            "detail": (
                "World gauge is ID2. Each eye independently learns fixed unknown "
                "T_ID2_from_ID5 by per-frame SE(3) consensus. ID5 corners are then "
                "transformed into ID2 coordinates and all 8 corners enter joint PnP. "
                "No common-plane assumption is used."
            ),
        },
        {
            "item": "ID2+SIFT",
            "detail": (
                "Only same-eye adjacent-frame SIFT is used. ChArUco dark pixels and "
                "complete 12.25 mm ID2/ID5 targets are removed in a SIFT-only copy. "
                "Edges must pass inlier-rate, Sampson-P90 and parallax gates. Essential "
                "translation supplies direction only; ID2 supplies metric scale. Hybrid, "
                "SIFT-assisted and fallback-only statistics are reported separately."
            ),
        },
        {
            "item": "SIFT sanitization dependency",
            "detail": (
                "ChArUco detection may locate the erase mask but its pose/corners do "
                "not enter branch scores or pose residuals. If the board polygon is "
                "unavailable, SIFT is disabled to prevent Grid leakage."
            ),
        },
        {
            "item": "Reference selection",
            "detail": (
                f"Reference is fixed to requested F{args.reference_frame} for every "
                "method and both eyes. There is no GT/JSON-dependent fallback. If a "
                "method cannot solve F{args.reference_frame}, its temporal rows are "
                "unavailable. The reference row is excluded from all denominators."
            ),
        },
        {
            "item": "Pass formula",
            "detail": (
                "All comparisons are strict '<'. Summary formulas require comparable=1, "
                "so blank error cells cannot be counted as passing. Rates are stored as "
                "0..100 and show requested, GT-valid, estimated and comparable denominators."
            ),
        },
        {
            "item": "Known limits",
            "detail": (
                "A nearly frontal tiny planar marker can remain ambiguous even with "
                "temporal smoothing. ID2-ID5 must stay rigidly fixed. Wet/specular or "
                "non-rigid wound appearance can invalidate SIFT. The first version uses "
                "the central 8.25 mm ArUco corners; auxiliary 1.6 mm blocks are masked "
                "from SIFT but are not yet pose correspondences."
            ),
        },
    ]


def main():
    args = parse_args()
    if args.left_sift_roi is None or args.right_sift_roi is None:
        print(
            "[WARN] For a clean wound-SIFT experiment, set tight one-eye "
            "--sift-roi-left and --sift-roi-right regions fully covered by the wound model."
        )
    if not args.protect_sift_roi_dark:
        print(
            "[WARN] ChArUco dark-pixel whitening can also remove dark wound texture. "
            "Use tight SIFT ROIs plus --protect-sift-roi-dark, then inspect leakage diagnostics."
        )
    elif args.left_sift_roi is None or args.right_sift_roi is None:
        print(
            "[WARN] --protect-sift-roi-dark only protects an explicit ROI; one or both "
            "SIFT ROIs are missing."
        )
    base = Path.cwd()
    paths = {
        "video": resolve_input(args.video, base),
        "calibration": resolve_input(args.calibration, base),
        "charuco_metadata": resolve_input(args.charuco_metadata, base),
        "id2_metadata": resolve_input(args.id2_metadata, base),
        "id5_metadata": resolve_input(args.id5_metadata, base),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    output_xlsx = (
        resolve_input(args.output, base)
        if args.output
        else paths["video"].with_name(
            paths["video"].stem
            + "_independent_monocular_temporal_charuco_gt.xlsx"
        )
    )
    if output_xlsx.suffix.lower() != ".xlsx":
        output_xlsx = output_xlsx.with_suffix(".xlsx")
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output_xlsx.with_suffix("")
    detection_video_path = output_prefix.with_name(
        output_prefix.name + "_detections_sift.mp4"
    )
    comparison_video_path = output_prefix.with_name(
        output_prefix.name + "_pose_comparison.mp4"
    )

    charuco_metadata = read_json(paths["charuco_metadata"])
    id2_metadata = read_json(paths["id2_metadata"])
    id5_metadata = read_json(paths["id5_metadata"])
    validate_metadata(charuco_metadata, id2_metadata, id5_metadata)
    board, dictionary = create_charuco_board(charuco_metadata)
    marker_sizes = {
        2: float(id2_metadata["central_aruco"]["size_mm"][0]),
        5: float(id5_metadata["central_aruco"]["size_mm"][0]),
    }
    target_scales = {
        2: float(id2_metadata["target_size_mm"][0]) / marker_sizes[2],
        5: float(id5_metadata["target_size_mm"][0]) / marker_sizes[5],
    }

    K_left, d_left, K_right, d_right, json_pose = load_calibration(
        paths["calibration"]
    )
    K_by_eye = {"L": K_left, "R": K_right}
    d_by_eye = {"L": d_left, "R": d_right}

    capture = cv2.VideoCapture(str(paths["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {paths['video']}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    ok, first_sbs = capture.read()
    capture.release()
    if not ok or first_sbs is None:
        raise RuntimeError("Cannot read the first video frame")
    first_left, first_right = split_sbs(first_sbs)
    if first_left.shape != first_right.shape:
        raise ValueError("Left/right SBS halves have different shapes")
    indexes = build_frame_indexes(frame_count, args)
    args._fps = fps

    print("=" * 78)
    print(f"Video: {paths['video']}")
    print(
        f"Frames: {indexes[0]}..{indexes[-1]} | selected={len(indexes)} | "
        f"step={args.frame_step} | fps={fps:.3f}"
    )
    print(
        f"One-eye image: {first_left.shape[1]}x{first_left.shape[0]} | "
        f"JSON baseline={np.linalg.norm(json_pose.t):.4f} mm"
    )
    print(
        "TEST isolation: left/right estimates are independent; JSON is held "
        "outside every estimator and used only below in evaluation."
    )

    observations, feature_edges = preprocess_video(
        paths["video"],
        indexes,
        fps,
        first_left.shape,
        board,
        dictionary,
        charuco_metadata,
        marker_sizes,
        target_scales,
        K_by_eye,
        d_by_eye,
        args,
        detection_video_path if args.diagnostic_video else None,
    )

    # All four estimators run before JSON/GT error evaluation.  They receive
    # only one eye's marker/SIFT observations and intrinsics.
    method_results = {method: {eye: {} for eye in EYES} for method in METHODS}
    relations = {}
    for eye in EYES:
        print(f"[Temporal] Solving independent {eye}-eye trajectories...")
        args._active_camera_matrix = K_by_eye[eye]
        id2_results, selected2, raw2 = build_single_marker_method(
            observations[eye],
            2,
            marker_sizes[2],
            K_by_eye[eye],
            d_by_eye[eye],
            args,
        )
        id5_results, selected5, raw5 = build_single_marker_method(
            observations[eye],
            5,
            marker_sizes[5],
            K_by_eye[eye],
            d_by_eye[eye],
            args,
        )
        method_results["ID2_TEMPORAL"][eye] = id2_results
        method_results["ID5_TEMPORAL"][eye] = id5_results
        dual_results, relation = build_dual_marker_method(
            observations[eye],
            marker_sizes[2],
            marker_sizes[5],
            K_by_eye[eye],
            d_by_eye[eye],
            args,
            selected2,
            selected5,
        )
        method_results["ID2_ID5_TEMPORAL"][eye] = dual_results
        relations[eye] = relation

        sift_selected, sift_raw, _margin = select_marker_temporal_path(
            observations[eye],
            2,
            args,
            feature_edges=feature_edges[eye],
        )
        method_results["ID2_SIFT_TEMPORAL"][eye] = refine_id2_with_sift(
            observations[eye],
            sift_selected,
            sift_raw,
            feature_edges[eye],
            marker_sizes[2],
            K_by_eye[eye],
            d_by_eye[eye],
            args,
        )
        print(
            f"  ID2-ID5 relation {eye}: {relation.status}; "
            f"support={relation.support_frames}/{relation.candidate_frames}"
        )

    # Evaluation begins here.  JSON is first consumed by error functions now,
    # after test outputs are frozen.
    gt_frame_rows, gt_stereo_rows, consistency = build_gt_tables(
        observations,
        indexes,
        board,
        K_by_eye,
        d_by_eye,
        json_pose,
        args,
    )
    references, reference_modes = choose_reference_frames(
        method_results, observations, consistency, args
    )
    method_pose_rows = build_method_pose_rows(
        method_results, observations, references
    )
    temporal_rows = build_temporal_rows(
        method_results,
        observations,
        consistency,
        references,
        reference_modes,
        args,
    )
    closure_rows = build_stereo_closure_rows(
        method_results,
        observations,
        consistency,
        json_pose,
        args,
    )
    sift_rows, mask_rows = build_sift_and_mask_rows(
        observations, feature_edges, args
    )
    marker_relation_rows = relation_rows(relations)

    settings_rows, settings_cells = build_settings(args, paths)
    summary_rows = build_summary_rows(
        temporal_rows, closure_rows, settings_cells
    )
    apply_detail_formulas(temporal_rows, closure_rows, settings_cells)
    chart_rows = build_chart_rows(
        indexes,
        closure_rows,
        gt_stereo_rows,
        summary_rows,
        settings_cells,
        args,
    )

    video_info_rows = [
        {"parameter": "video", "value": str(paths["video"])},
        {"parameter": "frame_count", "value": frame_count},
        {"parameter": "fps", "value": fps},
        {"parameter": "sbs_resolution", "value": f"{first_sbs.shape[1]}x{first_sbs.shape[0]}"},
        {"parameter": "one_eye_resolution", "value": f"{first_left.shape[1]}x{first_left.shape[0]}"},
        {"parameter": "selected_frame_count", "value": len(indexes)},
        {"parameter": "selected_frames", "value": f"{indexes[0]}..{indexes[-1]}, step={args.frame_step}"},
        {"parameter": "requested_reference_frame", "value": args.reference_frame},
        {"parameter": "json_baseline_mm", "value": float(np.linalg.norm(json_pose.t))},
        {"parameter": "opencv_version", "value": cv2.__version__},
        {"parameter": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"parameter": "pose_direction", "value": "pattern/world -> camera"},
        {"parameter": "stereo_json_direction", "value": "left camera -> right camera"},
        {"parameter": "charuco_spec", "value": (
            f"{charuco_metadata['dictionary']}, {charuco_metadata['squares_x']}x"
            f"{charuco_metadata['squares_y']} squares, square="
            f"{charuco_metadata['square_length_mm']}mm, marker="
            f"{charuco_metadata['marker_length_mm']}mm, IDs="
            f"{min(charuco_metadata['marker_ids'])}..{max(charuco_metadata['marker_ids'])}"
        )},
        {"parameter": "small_patterns", "value": (
            f"ID2/ID5 central ArUco={marker_sizes[2]}mm; full target="
            f"{id2_metadata['target_size_mm'][0]}mm"
        )},
    ]
    protocol_rows = build_protocol_rows(args)

    sheets = [
        ("Settings", ["parameter", "value", "unit", "description"], settings_rows),
        ("Video Info", ["parameter", "value"], video_info_rows),
        ("GT Frame Poses", GT_FRAME_HEADERS, gt_frame_rows),
        ("GT Stereo Check", GT_STEREO_HEADERS, gt_stereo_rows),
        ("Method Frame Poses", METHOD_FRAME_HEADERS, method_pose_rows),
        ("Temporal RPE", TEMPORAL_HEADERS, temporal_rows),
        ("Stereo Closure", STEREO_CLOSURE_HEADERS, closure_rows),
        ("Method Summary", SUMMARY_HEADERS, summary_rows),
        ("Marker Relation", RELATION_HEADERS, marker_relation_rows),
        ("SIFT Diagnostics", SIFT_DIAGNOSTIC_HEADERS, sift_rows),
        ("Mask Diagnostics", MASK_DIAGNOSTIC_HEADERS, mask_rows),
        ("Charts", CHART_HEADERS, chart_rows),
        ("Protocol", ["item", "detail"], protocol_rows),
    ]
    write_xlsx(output_xlsx, sheets)
    charts_sheet_index = next(
        index for index, sheet in enumerate(sheets, start=1) if sheet[0] == "Charts"
    )
    add_charts_to_workbook(output_xlsx, chart_rows, charts_sheet_index)
    validate_xlsx_package(output_xlsx)

    csv_outputs = {
        "gt_frame_poses": (GT_FRAME_HEADERS, gt_frame_rows),
        "gt_stereo_check": (GT_STEREO_HEADERS, gt_stereo_rows),
        "method_frame_poses": (METHOD_FRAME_HEADERS, method_pose_rows),
        "temporal_rpe": (TEMPORAL_HEADERS, temporal_rows),
        "stereo_closure": (STEREO_CLOSURE_HEADERS, closure_rows),
        "method_summary": (SUMMARY_HEADERS, summary_rows),
        "sift_diagnostics": (SIFT_DIAGNOSTIC_HEADERS, sift_rows),
        "mask_diagnostics": (MASK_DIAGNOSTIC_HEADERS, mask_rows),
    }
    for suffix, (headers, rows) in csv_outputs.items():
        write_csv_rows(
            output_prefix.with_name(output_prefix.name + f"_{suffix}.csv"),
            headers,
            rows,
        )

    if args.diagnostic_video:
        make_pose_comparison_video(
            paths["video"],
            indexes,
            closure_rows,
            comparison_video_path,
            fps,
            args,
        )

    print("=" * 78)
    gt_pass = sum(row["gt_stereo_consistent"] for row in gt_stereo_rows)
    print(f"GT stereo-consistent frames: {gt_pass}/{len(gt_stereo_rows)}")
    for method in METHODS:
        group = [
            row
            for row in closure_rows
            if row["method"] == method and row["comparable"] == 1
        ]
        both = sum(cached_value(row["both_pass"]) for row in group)
        print(
            f"{method}: comparable={len(group)}, both-pass={both}/"
            f"{len(group)} ({both / len(group) * 100.0 if group else 0.0:.2f}%)"
        )
    print(f"Excel: {output_xlsx}")
    print(f"CSV prefix: {output_prefix}_*.csv")
    if args.diagnostic_video:
        print(f"Detection/SIFT video: {detection_video_path}")
        print(f"Pose comparison video: {comparison_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
