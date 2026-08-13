"""Compare single-frame and temporal stereo RT estimates using only ArUco ID2.

The input is a side-by-side stereo video.  Other ArUco markers may be visible,
but they are deliberately ignored.  Camera intrinsics are fixed to the JSON
calibration.  The JSON stereo extrinsic is used only after estimation as the
answer for error calculation.

Three estimates are reported:

1. ``single_frame``: IPPE square branches are computed independently in both
   eyes, then one branch pair is selected with stereo-rig geometry constraints.
2. ``causal_temporal_window``: the most recent N valid ID2 observations are
   jointly calibrated with one constant left-to-right RT.  Only current and
   past frames are used; the preceding temporal RT can be used as an initial
   guess.
3. ``offline_all_frames``: all valid observations are jointly calibrated once
   to show the best result available from the complete video.

OpenCV ``stereoCalibrateExtended`` estimates an independent marker pose for
each observation while enforcing one shared stereo RT.  This is materially
different from averaging per-frame RT matrices.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

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
    load_calibration,
    parse_roi,
    rotation_angle_deg,
    split_sbs,
)


# =============================================================================
# User defaults (CLI arguments override these values)
# =============================================================================

VIDEO_PATH = ""
CALIBRATION_PATH = "calibration_result_HBVCAM_4M2214HD-2-v11.json"
ARUCO_DICTIONARY = "DICT_4X4_100"
MARKER_ID = 2
MARKER_SIZE_MM = 8.25

START_FRAME = 0
END_FRAME = -1  # -1 means the end of the video.
MAX_FRAMES = 300  # 0 means unlimited.
FRAME_STEP = 1

# ROI coordinates are for one eye.  None means the complete one-eye image.
LEFT_ROI: tuple[int, int, int, int] | None = None
RIGHT_ROI: tuple[int, int, int, int] | None = None

TEMPORAL_WINDOW_SIZE = 30
TEMPORAL_MIN_VIEWS = 5
TEMPORAL_UPDATE_STRIDE = 10
USE_PREVIOUS_TEMPORAL_RT_AS_GUESS = True
MAX_VIEW_REPROJECTION_RMS_PX = 2.5
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
MAX_STEREO_ROTATION_DEG = 30.0
MIN_HORIZONTAL_TRANSLATION_RATIO = 0.50

ROTATION_PASS_THRESHOLD_DEG = 2.0
BASELINE_PASS_THRESHOLD_PERCENT = 5.0
SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_SCALE = 0.5


FRAME_HEADERS = [
    "frame_index",
    "detection_status",
    "left_id2_found",
    "right_id2_found",
    "left_marker_side_mean_px",
    "right_marker_side_mean_px",
    "left_marker_area_px2",
    "right_marker_area_px2",
    "raw_available",
    "raw_ippe_branch_left",
    "raw_ippe_branch_right",
    "raw_selection_score",
    "raw_reprojection_rms_px",
    "raw_rotation_error_deg",
    "raw_baseline_mm",
    "raw_baseline_delta_mm",
    "raw_absolute_baseline_error_percent",
    "raw_translation_l2_error_mm",
    "raw_translation_direction_error_deg",
    "temporal_available",
    "temporal_source",
    "temporal_window_observations",
    "temporal_used_observations",
    "temporal_rejected_observations",
    "temporal_reprojection_rms_px",
    "temporal_per_view_rms_median_px",
    "temporal_centroid_span_px",
    "temporal_marker_size_cv_percent",
    "temporal_pose_tilt_span_deg",
    "temporal_depth_span_mm",
    "temporal_rotation_error_deg",
    "temporal_baseline_mm",
    "temporal_baseline_delta_mm",
    "temporal_absolute_baseline_error_percent",
    "temporal_translation_l2_error_mm",
    "temporal_translation_direction_error_deg",
    "rotation_error_improvement_deg",
    "baseline_error_improvement_percent_points",
    "json_baseline_mm",
    "processing_time_ms",
]

SUMMARY_HEADERS = [
    "method",
    "requested_frames",
    "id2_detected_both_frames",
    "estimated_frames",
    "rotation_error_mean_deg",
    "rotation_error_median_deg",
    "rotation_error_p95_deg",
    "absolute_baseline_error_mean_percent",
    "absolute_baseline_error_median_percent",
    "absolute_baseline_error_p95_percent",
    "rotation_pass_frames",
    "rotation_pass_rate_of_estimated",
    "baseline_pass_frames",
    "baseline_pass_rate_of_estimated",
    "both_pass_frames",
    "both_pass_rate_of_estimated",
    "both_pass_rate_of_requested",
]

FINAL_HEADERS = [
    "method",
    "available",
    "input_observations",
    "used_observations",
    "rejected_observations",
    "reprojection_rms_px",
    "per_view_rms_median_px",
    "centroid_span_px",
    "marker_size_cv_percent",
    "pose_tilt_span_deg",
    "depth_span_mm",
    "rotation_error_deg",
    "baseline_mm",
    "json_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
]

CHART_HEADERS = [
    "frame_index",
    "rotation_threshold_deg",
    "baseline_threshold_percent",
]


@dataclass
class Observation:
    frame_index: int
    left_corners: np.ndarray
    right_corners: np.ndarray
    raw_pose: Pose | None = None

    def __post_init__(self):
        self.left_corners = np.asarray(self.left_corners, np.float32).reshape(4, 2)
        self.right_corners = np.asarray(self.right_corners, np.float32).reshape(4, 2)


@dataclass
class StereoFit:
    pose: Pose
    rms_px: float
    per_view_rms_px: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    input_count: int
    used_count: int
    rejected_count: int
    used_observations: list[Observation]
    branch_left: int | None = None
    branch_right: int | None = None
    selection_score: float | None = None


class FastIdDetector:
    """Detect one requested ID and apply explicit dynamic subpixel refinement."""

    def __init__(self, dictionary_name: str, marker_id: int):
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")
        self.marker_id = int(marker_id)
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, gray: np.ndarray, roi) -> dict[int, np.ndarray]:
        x, y, width, height = clip_roi(roi, gray.shape)
        crop = gray[y : y + height, x : x + width]
        corners, ids, _rejected = self.detector.detectMarkers(crop)
        if ids is None:
            return {}
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.0001,
        )
        best = None
        best_area = -1.0
        for detected, marker_id in zip(corners, ids.reshape(-1)):
            if int(marker_id) != self.marker_id:
                continue
            points = np.asarray(detected, np.float32).reshape(4, 1, 2)
            flat = points.reshape(4, 2)
            side = float(
                np.mean(np.linalg.norm(np.roll(flat, -1, axis=0) - flat, axis=1))
            )
            window = int(np.clip(round(side / 12.0), 2, 9))
            try:
                cv2.cornerSubPix(
                    crop, points, (window, window), (-1, -1), criteria
                )
            except cv2.error:
                pass
            refined = points.reshape(4, 2).astype(np.float64)
            refined[:, 0] += x
            refined[:, 1] += y
            area = abs(float(cv2.contourArea(refined.astype(np.float32))))
            if area > best_area:
                best, best_area = refined, area
        return {self.marker_id: best} if best is not None else {}


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def square_object_points(size_mm: float) -> np.ndarray:
    half = float(size_mm) * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def marker_metrics(corners) -> tuple[float, float]:
    points = np.asarray(corners, np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    area = abs(float(cv2.contourArea(points.astype(np.float32))))
    return float(np.mean(sides)), area


def ippe_pose_branches(corners, object_points, camera_matrix, distortion):
    try:
        _count, rvecs, tvecs, _reported_errors = cv2.solvePnPGeneric(
            np.asarray(object_points, np.float32),
            np.asarray(corners, np.float32).reshape(4, 2),
            np.asarray(camera_matrix, np.float64),
            np.asarray(distortion, np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    result = []
    for branch_index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rotation = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
        translation = np.asarray(tvec, np.float64).reshape(3, 1)
        camera_points = (
            rotation @ np.asarray(object_points, np.float64).T + translation
        ).T
        if np.any(camera_points[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(
            object_points,
            cv2.Rodrigues(rotation)[0],
            translation,
            camera_matrix,
            distortion,
        )[0].reshape(4, 2)
        residual = projected - np.asarray(corners, np.float64).reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))
        result.append((branch_index, Pose(rotation, translation), rms))
    return result


def single_frame_ippe_fit(
    observation,
    object_points,
    k_left,
    d_left,
    k_right,
    d_right,
    min_baseline_mm,
    max_baseline_mm,
    max_stereo_rotation_deg,
    min_horizontal_translation_ratio,
):
    left_branches = ippe_pose_branches(
        observation.left_corners, object_points, k_left, d_left
    )
    right_branches = ippe_pose_branches(
        observation.right_corners, object_points, k_right, d_right
    )
    candidates = []
    for left_index, left_pose, left_rms in left_branches:
        for right_index, right_pose, right_rms in right_branches:
            relative_rotation = right_pose.R @ left_pose.R.T
            relative_translation = right_pose.t - relative_rotation @ left_pose.t
            relative = Pose(relative_rotation, relative_translation)
            baseline = float(np.linalg.norm(relative.t))
            stereo_rotation = rotation_angle_deg(relative.R)
            horizontal_ratio = abs(float(relative.t[0, 0])) / max(baseline, 1e-12)
            score = left_rms + right_rms
            score += 0.025 * stereo_rotation
            score += 2.0 * max(
                0.0, float(min_horizontal_translation_ratio) - horizontal_ratio
            )
            if baseline < min_baseline_mm:
                score += 5.0 + 0.2 * (min_baseline_mm - baseline)
            elif baseline > max_baseline_mm:
                score += 5.0 + 0.05 * (baseline - max_baseline_mm)
            if stereo_rotation > max_stereo_rotation_deg:
                score += 2.0 + 0.1 * (
                    stereo_rotation - max_stereo_rotation_deg
                )
            candidates.append(
                (
                    score,
                    left_index,
                    right_index,
                    relative,
                    left_pose,
                    left_rms,
                    right_rms,
                )
            )
    if not candidates:
        raise RuntimeError("No positive-depth IPPE branch pair")
    best = min(candidates, key=lambda item: item[0])
    score, left_index, right_index, relative, left_pose, left_rms, right_rms = best
    return StereoFit(
        pose=relative,
        rms_px=float(math.sqrt((left_rms * left_rms + right_rms * right_rms) / 2.0)),
        per_view_rms_px=np.asarray([left_rms, right_rms], np.float64),
        rvecs=[cv2.Rodrigues(left_pose.R)[0].reshape(3)],
        tvecs=[left_pose.t.reshape(3)],
        input_count=1,
        used_count=1,
        rejected_count=0,
        used_observations=[observation],
        branch_left=left_index,
        branch_right=right_index,
        selection_score=float(score),
    )


def temporal_seed(observations: list[Observation]) -> Pose | None:
    poses = [item.raw_pose for item in observations if item.raw_pose is not None]
    if not poses:
        return None
    if len(poses) == 1:
        return poses[0]
    costs = []
    for candidate in poses:
        cost = 0.0
        for other in poses:
            cost += rotation_angle_deg(candidate.R @ other.R.T)
            cost += 0.04 * float(np.linalg.norm(candidate.t - other.t))
        costs.append(cost)
    return poses[int(np.argmin(costs))]


def translation_direction_error_deg(first, second) -> float | None:
    first = np.asarray(first, np.float64).reshape(3)
    second = np.asarray(second, np.float64).reshape(3)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.clip(first @ second / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def answer_errors(estimate: Pose, answer: Pose) -> dict:
    estimated_baseline = float(np.linalg.norm(estimate.t))
    answer_baseline = float(np.linalg.norm(answer.t))
    delta = estimated_baseline - answer_baseline
    return {
        "rotation_error_deg": rotation_angle_deg(estimate.R @ answer.R.T),
        "baseline_mm": estimated_baseline,
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


def normalize_per_view_errors(values, count: int) -> np.ndarray:
    errors = np.asarray(values, np.float64)
    if errors.size == 0:
        return np.full(count, np.nan, np.float64)
    if errors.ndim == 1:
        errors = errors.reshape(-1, 1)
    if errors.shape[0] != count and errors.size % count == 0:
        errors = errors.reshape(count, -1)
    if errors.shape[0] != count:
        return np.full(count, np.nan, np.float64)
    return np.sqrt(np.mean(np.square(errors), axis=1))


def run_stereo_calibration(
    observations: list[Observation],
    object_points: np.ndarray,
    k_left,
    d_left,
    k_right,
    d_right,
    image_size,
    initial_pose: Pose | None = None,
) -> StereoFit:
    if not observations:
        raise ValueError("No ID2 observations were supplied")
    objects = [object_points.copy() for _ in observations]
    image_left = [item.left_corners.astype(np.float32) for item in observations]
    image_right = [item.right_corners.astype(np.float32) for item in observations]
    flags = cv2.CALIB_FIX_INTRINSIC
    if initial_pose is None:
        initial_r = np.eye(3, dtype=np.float64)
        initial_t = np.zeros((3, 1), dtype=np.float64)
    else:
        flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
        initial_r = initial_pose.R.copy()
        initial_t = initial_pose.t.copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-7,
    )
    result = cv2.stereoCalibrateExtended(
        objects,
        image_left,
        image_right,
        np.asarray(k_left, np.float64).copy(),
        np.asarray(d_left, np.float64).copy(),
        np.asarray(k_right, np.float64).copy(),
        np.asarray(d_right, np.float64).copy(),
        tuple(map(int, image_size)),
        initial_r,
        initial_t,
        flags=flags,
        criteria=criteria,
    )
    rms, rotation, translation = result[0], result[5], result[6]
    rvecs, tvecs, per_view = result[9], result[10], result[11]
    if not (
        math.isfinite(float(rms))
        and np.all(np.isfinite(rotation))
        and np.all(np.isfinite(translation))
    ):
        raise RuntimeError("OpenCV returned a non-finite stereo solution")
    pose = Pose(rotation, translation)
    return StereoFit(
        pose=pose,
        rms_px=float(rms),
        per_view_rms_px=normalize_per_view_errors(per_view, len(observations)),
        rvecs=[np.asarray(value, np.float64).reshape(3) for value in rvecs],
        tvecs=[np.asarray(value, np.float64).reshape(3) for value in tvecs],
        input_count=len(observations),
        used_count=len(observations),
        rejected_count=0,
        used_observations=list(observations),
    )


def robust_stereo_calibration(
    observations: list[Observation],
    object_points,
    k_left,
    d_left,
    k_right,
    d_right,
    image_size,
    initial_pose,
    minimum_views: int,
    max_view_rms_px: float,
) -> StereoFit:
    first = run_stereo_calibration(
        observations,
        object_points,
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        initial_pose,
    )
    finite_errors = first.per_view_rms_px[np.isfinite(first.per_view_rms_px)]
    if len(observations) <= minimum_views or not len(finite_errors):
        return first
    median = float(np.median(finite_errors))
    mad = float(np.median(np.abs(finite_errors - median)))
    robust_sigma = 1.4826 * mad
    adaptive_gate = max(float(max_view_rms_px), median + 3.0 * robust_sigma)
    keep = np.isfinite(first.per_view_rms_px) & (
        first.per_view_rms_px <= adaptive_gate
    )
    if int(np.count_nonzero(keep)) < minimum_views or bool(np.all(keep)):
        return first
    kept = [item for item, accepted in zip(observations, keep) if accepted]
    refined = run_stereo_calibration(
        kept,
        object_points,
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        first.pose,
    )
    refined.input_count = len(observations)
    refined.used_count = len(kept)
    refined.rejected_count = len(observations) - len(kept)
    refined.used_observations = kept
    return refined


def maximum_pairwise_angle_deg(vectors: list[np.ndarray]) -> float | None:
    if len(vectors) < 2:
        return 0.0 if vectors else None
    unit = []
    for vector in vectors:
        vector = np.asarray(vector, np.float64).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            unit.append(vector / norm)
    maximum = 0.0
    for first_index in range(len(unit)):
        for second_index in range(first_index + 1, len(unit)):
            cosine = float(
                np.clip(unit[first_index] @ unit[second_index], -1.0, 1.0)
            )
            maximum = max(maximum, float(np.degrees(np.arccos(cosine))))
    return maximum


def fit_diversity(fit: StereoFit) -> dict:
    centres = np.asarray(
        [np.mean(item.left_corners, axis=0) for item in fit.used_observations],
        np.float64,
    )
    sizes = np.asarray(
        [marker_metrics(item.left_corners)[0] for item in fit.used_observations],
        np.float64,
    )
    centroid_span = (
        float(np.linalg.norm(np.ptp(centres, axis=0))) if len(centres) else None
    )
    size_cv = (
        float(np.std(sizes, ddof=1) / np.mean(sizes) * 100.0)
        if len(sizes) >= 2 and float(np.mean(sizes)) > 1e-12
        else 0.0 if len(sizes) else None
    )
    normals = [cv2.Rodrigues(rvec)[0][:, 2] for rvec in fit.rvecs]
    depths = np.asarray([value[2] for value in fit.tvecs], np.float64)
    return {
        "centroid_span_px": centroid_span,
        "marker_size_cv_percent": size_cv,
        "pose_tilt_span_deg": maximum_pairwise_angle_deg(normals),
        "depth_span_mm": float(np.ptp(depths)) if len(depths) else None,
        "per_view_rms_median_px": (
            float(np.nanmedian(fit.per_view_rms_px))
            if np.any(np.isfinite(fit.per_view_rms_px))
            else None
        ),
    }


def add_prefixed_errors(row: dict, prefix: str, fit: StereoFit, answer: Pose):
    metrics = answer_errors(fit.pose, answer)
    row[f"{prefix}_reprojection_rms_px"] = fit.rms_px
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def empty_frame_row(frame_index: int, answer: Pose) -> dict:
    row = {header: None for header in FRAME_HEADERS}
    row.update(
        {
            "frame_index": int(frame_index),
            "detection_status": "NOT_PROCESSED",
            "left_id2_found": 0,
            "right_id2_found": 0,
            "raw_available": 0,
            "temporal_available": 0,
            "json_baseline_mm": float(np.linalg.norm(answer.t)),
        }
    )
    return row


def frame_indexes(total_frames, start_frame, end_frame, frame_step, max_frames):
    final = total_frames - 1 if end_frame < 0 else min(end_frame, total_frames - 1)
    indexes = list(range(start_frame, final + 1, frame_step))
    if max_frames > 0:
        indexes = indexes[:max_frames]
    return indexes


def analyze_video(args):
    video_path = Path(args.video).resolve()
    calibration_path = Path(args.calibration).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    k_left, d_left, k_right, d_right, answer = load_calibration(calibration_path)
    object_points = square_object_points(args.marker_size_mm)
    detector = FastIdDetector(args.dictionary, args.marker_id)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested = frame_indexes(
        total_frames,
        args.start_frame,
        args.end_frame,
        args.frame_step,
        args.max_frames,
    )
    if not requested:
        capture.release()
        raise ValueError("The selected frame range is empty")

    rows = []
    valid_observations: list[Observation] = []
    window: deque[Observation] = deque(maxlen=args.temporal_window_size)
    current_temporal: StereoFit | None = None
    previous_temporal_pose: Pose | None = None
    last_update_valid_count = -10**9
    image_size = None
    left_roi = right_roi = None
    next_capture_frame = 0

    for sequence, frame_index in enumerate(requested, start=1):
        started = cv2.getTickCount()
        if int(frame_index) != next_capture_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, sbs = capture.read()
        next_capture_frame = int(frame_index) + 1
        row = empty_frame_row(frame_index, answer)
        if not ok or sbs is None:
            row["detection_status"] = "FRAME_READ_FAILED"
            rows.append(row)
            continue
        left, right = split_sbs(sbs)
        if image_size is None:
            image_size = (left.shape[1], left.shape[0])
            left_roi = clip_roi(args.left_roi, left.shape)
            right_roi = clip_roi(args.right_roi, right.shape)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        markers_left = detector.detect(gray_left, left_roi)
        markers_right = detector.detect(gray_right, right_roi)
        corners_left = markers_left.get(args.marker_id)
        corners_right = markers_right.get(args.marker_id)
        row["left_id2_found"] = int(corners_left is not None)
        row["right_id2_found"] = int(corners_right is not None)

        if corners_left is not None:
            row["left_marker_side_mean_px"], row["left_marker_area_px2"] = (
                marker_metrics(corners_left)
            )
        if corners_right is not None:
            row["right_marker_side_mean_px"], row["right_marker_area_px2"] = (
                marker_metrics(corners_right)
            )

        updated = False
        if corners_left is not None and corners_right is not None:
            row["detection_status"] = "ID2_BOTH"
            observation = Observation(frame_index, corners_left, corners_right)
            valid_observations.append(observation)
            window.append(observation)
            try:
                raw_fit = single_frame_ippe_fit(
                    observation,
                    object_points,
                    k_left,
                    d_left,
                    k_right,
                    d_right,
                    args.min_baseline_mm,
                    args.max_baseline_mm,
                    args.max_stereo_rotation_deg,
                    args.min_horizontal_translation_ratio,
                )
                observation.raw_pose = raw_fit.pose
                row["raw_available"] = 1
                row["raw_ippe_branch_left"] = raw_fit.branch_left
                row["raw_ippe_branch_right"] = raw_fit.branch_right
                row["raw_selection_score"] = raw_fit.selection_score
                add_prefixed_errors(row, "raw", raw_fit, answer)
            except (cv2.error, RuntimeError, ValueError):
                row["raw_available"] = 0

            valid_count = len(valid_observations)
            should_update = (
                len(window) >= args.temporal_min_views
                and valid_count - last_update_valid_count
                >= args.temporal_update_stride
            )
            if should_update:
                seed = None
                if args.use_previous_temporal_rt_as_guess:
                    seed = previous_temporal_pose or temporal_seed(list(window))
                try:
                    current_temporal = robust_stereo_calibration(
                        list(window),
                        object_points,
                        k_left,
                        d_left,
                        k_right,
                        d_right,
                        image_size,
                        seed,
                        args.temporal_min_views,
                        args.max_view_reprojection_rms_px,
                    )
                    previous_temporal_pose = current_temporal.pose
                    last_update_valid_count = valid_count
                    updated = True
                except (cv2.error, RuntimeError, ValueError):
                    pass
        elif corners_left is not None:
            row["detection_status"] = "ID2_LEFT_ONLY"
        elif corners_right is not None:
            row["detection_status"] = "ID2_RIGHT_ONLY"
        else:
            row["detection_status"] = "ID2_NOT_FOUND"

        if current_temporal is not None:
            row["temporal_available"] = 1
            if updated:
                row["temporal_source"] = "UPDATED"
            elif corners_left is None or corners_right is None:
                row["temporal_source"] = "HELD_NO_ID2"
            else:
                row["temporal_source"] = "HELD_UPDATE_STRIDE"
            row["temporal_window_observations"] = len(window)
            row["temporal_used_observations"] = current_temporal.used_count
            row["temporal_rejected_observations"] = current_temporal.rejected_count
            row["temporal_reprojection_rms_px"] = current_temporal.rms_px
            diversity = fit_diversity(current_temporal)
            for name, value in diversity.items():
                row[f"temporal_{name}"] = value
            add_prefixed_errors(row, "temporal", current_temporal, answer)

        if row.get("raw_available") and row.get("temporal_available"):
            row["rotation_error_improvement_deg"] = (
                row["raw_rotation_error_deg"] - row["temporal_rotation_error_deg"]
            )
            row["baseline_error_improvement_percent_points"] = (
                row["raw_absolute_baseline_error_percent"]
                - row["temporal_absolute_baseline_error_percent"]
            )
        row["processing_time_ms"] = (
            (cv2.getTickCount() - started) / cv2.getTickFrequency() * 1000.0
        )
        rows.append(row)
        if sequence == 1 or sequence % 25 == 0 or sequence == len(requested):
            print(
                f"[{sequence:4d}/{len(requested)}] frame={frame_index} "
                f"detected={len(valid_observations)} "
                f"temporal={'yes' if current_temporal else 'no'}"
            )
    capture.release()

    final_fit = None
    if len(valid_observations) >= args.temporal_min_views:
        print(f"Offline joint calibration: {len(valid_observations)} ID2 views...")
        final_fit = robust_stereo_calibration(
            valid_observations,
            object_points,
            k_left,
            d_left,
            k_right,
            d_right,
            image_size,
            (
                previous_temporal_pose or temporal_seed(valid_observations)
                if args.use_previous_temporal_rt_as_guess
                else None
            ),
            args.temporal_min_views,
            args.max_view_reprojection_rms_px,
        )
    return rows, valid_observations, final_fit, answer, image_size, left_roi, right_roi


def numeric(rows, field):
    return [value for row in rows if (value := finite(row.get(field))) is not None]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summary_row(rows, method, prefix, rotation_threshold, baseline_threshold):
    valid = [row for row in rows if int(row.get(f"{prefix}_available") or 0) == 1]
    rotations = numeric(valid, f"{prefix}_rotation_error_deg")
    baselines = numeric(valid, f"{prefix}_absolute_baseline_error_percent")
    detected_both = sum(row.get("detection_status") == "ID2_BOTH" for row in rows)
    record = {
        "method": method,
        "requested_frames": len(rows),
        "id2_detected_both_frames": detected_both,
        "estimated_frames": len(valid),
        "rotation_error_mean_deg": float(np.mean(rotations)) if rotations else None,
        "rotation_error_median_deg": float(np.median(rotations)) if rotations else None,
        "rotation_error_p95_deg": percentile(rotations, 95),
        "absolute_baseline_error_mean_percent": (
            float(np.mean(baselines)) if baselines else None
        ),
        "absolute_baseline_error_median_percent": (
            float(np.median(baselines)) if baselines else None
        ),
        "absolute_baseline_error_p95_percent": percentile(baselines, 95),
    }
    last = len(rows) + 1
    columns = {name: excel_column(FRAME_HEADERS.index(name) + 1) for name in FRAME_HEADERS}
    available_range = (
        f"'Frame Results'!${columns[f'{prefix}_available']}$2:"
        f"${columns[f'{prefix}_available']}${last}"
    )
    rotation_range = (
        f"'Frame Results'!${columns[f'{prefix}_rotation_error_deg']}$2:"
        f"${columns[f'{prefix}_rotation_error_deg']}${last}"
    )
    baseline_range = (
        f"'Frame Results'!${columns[f'{prefix}_absolute_baseline_error_percent']}$2:"
        f"${columns[f'{prefix}_absolute_baseline_error_percent']}${last}"
    )
    rotation_count = sum(value < rotation_threshold for value in rotations)
    baseline_count = sum(value < baseline_threshold for value in baselines)
    both_count = sum(
        finite(row.get(f"{prefix}_rotation_error_deg")) < rotation_threshold
        and finite(row.get(f"{prefix}_absolute_baseline_error_percent"))
        < baseline_threshold
        for row in valid
    )
    record["rotation_pass_frames"] = ExcelFormula(
        f'COUNTIFS({available_range},1,{rotation_range},"<"&\'Settings\'!$B$2)',
        rotation_count,
    )
    record["baseline_pass_frames"] = ExcelFormula(
        f'COUNTIFS({available_range},1,{baseline_range},"<"&\'Settings\'!$B$3)',
        baseline_count,
    )
    record["both_pass_frames"] = ExcelFormula(
        f'COUNTIFS({available_range},1,{rotation_range},"<"&\'Settings\'!$B$2,'
        f'{baseline_range},"<"&\'Settings\'!$B$3)',
        both_count,
    )
    row_number = 2 if prefix == "raw" else 3
    summary_columns = {
        name: excel_column(SUMMARY_HEADERS.index(name) + 1) for name in SUMMARY_HEADERS
    }
    estimated_cell = f"${summary_columns['estimated_frames']}{row_number}"
    requested_cell = f"${summary_columns['requested_frames']}{row_number}"
    for count_name, rate_name, count in (
        ("rotation_pass_frames", "rotation_pass_rate_of_estimated", rotation_count),
        ("baseline_pass_frames", "baseline_pass_rate_of_estimated", baseline_count),
        ("both_pass_frames", "both_pass_rate_of_estimated", both_count),
    ):
        count_cell = f"{summary_columns[count_name]}{row_number}"
        record[rate_name] = ExcelFormula(
            f"IFERROR({count_cell}/{estimated_cell},0)",
            count / len(valid) if valid else 0.0,
        )
    both_cell = f"{summary_columns['both_pass_frames']}{row_number}"
    record["both_pass_rate_of_requested"] = ExcelFormula(
        f"IFERROR({both_cell}/{requested_cell},0)",
        both_count / len(rows) if rows else 0.0,
    )
    return record


def final_record(final_fit, answer):
    if final_fit is None:
        return {"method": "offline_all_frames", "available": 0}
    record = {
        "method": "offline_all_frames",
        "available": 1,
        "input_observations": final_fit.input_count,
        "used_observations": final_fit.used_count,
        "rejected_observations": final_fit.rejected_count,
        "reprojection_rms_px": final_fit.rms_px,
    }
    record.update(fit_diversity(final_fit))
    record.update(answer_errors(final_fit.pose, answer))
    return record


def build_chart_rows(rows, rotation_threshold, baseline_threshold):
    result = []
    for source_row, row in enumerate(rows, start=2):
        result.append(
            {
                "frame_index": ExcelFormula(
                    f"'Frame Results'!$A${source_row}", row["frame_index"]
                ),
                "rotation_threshold_deg": ExcelFormula(
                    "'Settings'!$B$2", rotation_threshold
                ),
                "baseline_threshold_percent": ExcelFormula(
                    "'Settings'!$B$3", baseline_threshold
                ),
            }
        )
    return result


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in FRAME_HEADERS})


def add_comparison_charts(path: Path, rows, rotation_threshold, baseline_threshold):
    if not rows:
        return
    last = len(rows) + 1
    x_values = [row["frame_index"] for row in rows]
    x_formula = f"'Charts'!$A$2:$A${last}"
    columns = {name: excel_column(FRAME_HEADERS.index(name) + 1) for name in FRAME_HEADERS}

    def formula(field):
        return f"'Frame Results'!${columns[field]}$2:${columns[field]}${last}"

    charts = [
        build_frame_scatter_chart_xml(
            "Single-frame vs temporal rotation error",
            "Rotation error (deg)",
            x_formula,
            x_values,
            [
                {"name": "Single frame", "formula": formula("raw_rotation_error_deg"), "values": [row.get("raw_rotation_error_deg") for row in rows], "color": "A5A5A5"},
                {"name": "Causal temporal", "formula": formula("temporal_rotation_error_deg"), "values": [row.get("temporal_rotation_error_deg") for row in rows], "color": "4472C4"},
                {"name": "Pass threshold", "formula": f"'Charts'!$B$2:$B${last}", "values": [rotation_threshold] * len(rows), "color": "C00000", "threshold": True},
            ],
            2000000000,
        ),
        build_frame_scatter_chart_xml(
            "Single-frame vs temporal baseline error",
            "Absolute baseline error (%)",
            x_formula,
            x_values,
            [
                {"name": "Single frame", "formula": formula("raw_absolute_baseline_error_percent"), "values": [row.get("raw_absolute_baseline_error_percent") for row in rows], "color": "A5A5A5"},
                {"name": "Causal temporal", "formula": formula("temporal_absolute_baseline_error_percent"), "values": [row.get("temporal_absolute_baseline_error_percent") for row in rows], "color": "ED7D31"},
                {"name": "Pass threshold", "formula": f"'Charts'!$C$2:$C${last}", "values": [baseline_threshold] * len(rows), "color": "C00000", "threshold": True},
            ],
            2010000000,
        ),
        build_frame_scatter_chart_xml(
            "Estimated stereo baseline convergence",
            "Baseline (mm)",
            x_formula,
            x_values,
            [
                {"name": "Single frame", "formula": formula("raw_baseline_mm"), "values": [row.get("raw_baseline_mm") for row in rows], "color": "A5A5A5"},
                {"name": "Causal temporal", "formula": formula("temporal_baseline_mm"), "values": [row.get("temporal_baseline_mm") for row in rows], "color": "4472C4"},
                {"name": "JSON answer", "formula": formula("json_baseline_mm"), "values": [row.get("json_baseline_mm") for row in rows], "color": "70AD47", "threshold": True},
            ],
            2020000000,
        ),
    ]
    drawing = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'{chart_anchor_xml(1, "rId1", 0, 17)}'
        f'{chart_anchor_xml(2, "rId2", 18, 35)}'
        f'{chart_anchor_xml(3, "rId3", 36, 53)}'
        "</xdr:wsDr>"
    )
    drawing_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart%d.xml"/>' % (index, index)
            for index in range(1, 4)
        )
        + "</Relationships>"
    )
    sheet_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/>'
        "</Relationships>"
    )
    parts = {
        "xl/worksheets/_rels/sheet4.xml.rels": sheet_rels,
        "xl/drawings/drawing1.xml": drawing,
        "xl/drawings/_rels/drawing1.xml.rels": drawing_rels,
        **{f"xl/charts/chart{index}.xml": chart for index, chart in enumerate(charts, 1)},
    }
    temporary = tempfile.NamedTemporaryFile(
        prefix=path.stem + "_charts_", suffix=".xlsx", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as destination:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "xl/worksheets/sheet4.xml":
                    worksheet = data.decode("utf-8").replace(
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
                        1,
                    )
                    data = worksheet.replace(
                        "</worksheet>", '<drawing r:id="rId1"/></worksheet>'
                    ).encode("utf-8")
                elif info.filename == "[Content_Types].xml":
                    content_types = data.decode("utf-8")
                    overrides = (
                        '<Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                        + "".join(
                            '<Override PartName="/xl/charts/chart%d.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>' % index
                            for index in range(1, 4)
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


def draw_id2(image, corners, x_offset, color):
    if corners is None:
        return
    points = np.asarray(corners, np.float64).copy()
    points[:, 0] += x_offset
    polygon = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], True, color, 1, cv2.LINE_AA)
    for index, point in enumerate(np.rint(points).astype(int)):
        x, y = point
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
        cv2.putText(image, str(index), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def write_diagnostic_video(video_path, output_path, rows, args, left_roi, right_roi):
    row_map = {int(row["frame_index"]): row for row in rows}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    detector = FastIdDetector(args.dictionary, args.marker_id)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0) / max(args.frame_step, 1)
    writer = None
    frame_index = 0
    while True:
        ok, sbs = capture.read()
        if not ok:
            break
        row = row_map.get(frame_index)
        frame_index += 1
        if row is None:
            continue
        left, right = split_sbs(sbs)
        markers_left = detector.detect(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), left_roi)
        markers_right = detector.detect(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), right_roi)
        canvas = sbs.copy()
        draw_id2(canvas, markers_left.get(args.marker_id), 0, (0, 255, 0))
        draw_id2(canvas, markers_right.get(args.marker_id), left.shape[1], (255, 255, 0))
        lines = [
            f"F{row['frame_index']} | ID{args.marker_id}: {row['detection_status']}",
            "Raw: rot={} deg | baseline={} %".format(
                "--" if finite(row.get("raw_rotation_error_deg")) is None else f"{row['raw_rotation_error_deg']:.3f}",
                "--" if finite(row.get("raw_absolute_baseline_error_percent")) is None else f"{row['raw_absolute_baseline_error_percent']:.2f}",
            ),
            "Temporal({}): rot={} deg | baseline={} %".format(
                row.get("temporal_used_observations") or 0,
                "--" if finite(row.get("temporal_rotation_error_deg")) is None else f"{row['temporal_rotation_error_deg']:.3f}",
                "--" if finite(row.get("temporal_absolute_baseline_error_percent")) is None else f"{row['temporal_absolute_baseline_error_percent']:.2f}",
            ),
        ]
        for index, text in enumerate(lines):
            y = 28 + index * 26
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        if args.diagnostic_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=args.diagnostic_scale, fy=args.diagnostic_scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (canvas.shape[1], canvas.shape[0]),
            )
        writer.write(canvas)
    capture.release()
    if writer is not None:
        writer.release()
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare per-frame and temporal stereo RT using only ArUco ID2."
    )
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--dictionary", default=ARUCO_DICTIONARY)
    parser.add_argument("--marker-id", type=int, default=MARKER_ID)
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--end-frame", type=int, default=END_FRAME)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP)
    parser.add_argument("--left-roi", type=parse_roi, default=LEFT_ROI)
    parser.add_argument("--right-roi", type=parse_roi, default=RIGHT_ROI)
    parser.add_argument("--temporal-window-size", type=int, default=TEMPORAL_WINDOW_SIZE)
    parser.add_argument("--temporal-min-views", type=int, default=TEMPORAL_MIN_VIEWS)
    parser.add_argument("--temporal-update-stride", type=int, default=TEMPORAL_UPDATE_STRIDE)
    parser.add_argument("--use-previous-temporal-rt-as-guess", action=argparse.BooleanOptionalAction, default=USE_PREVIOUS_TEMPORAL_RT_AS_GUESS)
    parser.add_argument("--max-view-reprojection-rms-px", type=float, default=MAX_VIEW_REPROJECTION_RMS_PX)
    parser.add_argument("--min-baseline-mm", type=float, default=MIN_BASELINE_MM)
    parser.add_argument("--max-baseline-mm", type=float, default=MAX_BASELINE_MM)
    parser.add_argument("--max-stereo-rotation-deg", type=float, default=MAX_STEREO_ROTATION_DEG)
    parser.add_argument("--min-horizontal-translation-ratio", type=float, default=MIN_HORIZONTAL_TRANSLATION_RATIO)
    parser.add_argument("--rotation-threshold-deg", type=float, default=ROTATION_PASS_THRESHOLD_DEG)
    parser.add_argument("--baseline-threshold-percent", type=float, default=BASELINE_PASS_THRESHOLD_PERCENT)
    parser.add_argument("--diagnostic-video", action=argparse.BooleanOptionalAction, default=SAVE_DIAGNOSTIC_VIDEO)
    parser.add_argument("--diagnostic-scale", type=float, default=DIAGNOSTIC_SCALE)
    parser.add_argument("--output", help="Output .xlsx path")
    args = parser.parse_args()
    if not args.video:
        parser.error("Specify a side-by-side MP4 video")
    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")
    if args.start_frame < 0 or args.end_frame < -1:
        parser.error("Invalid frame range")
    if args.end_frame >= 0 and args.end_frame < args.start_frame:
        parser.error("--end-frame must be >= --start-frame")
    if args.frame_step <= 0 or args.temporal_update_stride <= 0:
        parser.error("Frame step and update stride must be positive")
    if args.temporal_min_views < 2:
        parser.error("--temporal-min-views must be at least 2")
    if args.temporal_window_size < args.temporal_min_views:
        parser.error("Temporal window must be >= temporal minimum views")
    if args.min_baseline_mm <= 0 or args.max_baseline_mm <= args.min_baseline_mm:
        parser.error("Invalid single-frame baseline range")
    if not 0.0 <= args.min_horizontal_translation_ratio <= 1.0:
        parser.error("--min-horizontal-translation-ratio must be within 0..1")
    if args.diagnostic_scale <= 0:
        parser.error("--diagnostic-scale must be positive")
    return args


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else video_path.with_name(video_path.stem + "_id2_temporal_rt_comparison.xlsx")
    )
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows, observations, final_fit, answer, image_size, left_roi, right_roi = analyze_video(args)
    raw_summary = summary_row(
        rows,
        "single_frame",
        "raw",
        args.rotation_threshold_deg,
        args.baseline_threshold_percent,
    )
    temporal_summary = summary_row(
        rows,
        "causal_temporal_window",
        "temporal",
        args.rotation_threshold_deg,
        args.baseline_threshold_percent,
    )
    final = final_record(final_fit, answer)
    settings = [
        {"parameter": "rotation_pass_threshold_deg", "value": args.rotation_threshold_deg, "description": "Editable; Summary formulas recalculate."},
        {"parameter": "absolute_baseline_error_pass_percent", "value": args.baseline_threshold_percent, "description": "Editable; Summary formulas recalculate."},
        {"parameter": "video", "value": str(video_path), "description": "SBS stereo input."},
        {"parameter": "calibration", "value": str(Path(args.calibration).resolve()), "description": "Intrinsics fixed; extrinsic used only as answer."},
        {"parameter": "dictionary", "value": args.dictionary, "description": "Only the selected marker ID is retained."},
        {"parameter": "marker_id", "value": args.marker_id, "description": "Other visible markers are ignored."},
        {"parameter": "marker_size_mm", "value": args.marker_size_mm, "description": "Physical black-square side length."},
        {"parameter": "frame_range", "value": f"{args.start_frame}..{args.end_frame}", "description": "-1 end frame means video end."},
        {"parameter": "max_frames", "value": args.max_frames, "description": "0 means unlimited."},
        {"parameter": "frame_step", "value": args.frame_step, "description": "Sampling interval."},
        {"parameter": "left_roi", "value": str(left_roi), "description": "One-eye coordinates."},
        {"parameter": "right_roi", "value": str(right_roi), "description": "One-eye coordinates."},
        {"parameter": "temporal_window_size", "value": args.temporal_window_size, "description": "Most recent valid ID2 observations."},
        {"parameter": "temporal_min_views", "value": args.temporal_min_views, "description": "Warm-up before temporal RT is available."},
        {"parameter": "temporal_update_stride", "value": args.temporal_update_stride, "description": "Valid observations between recalibrations."},
        {"parameter": "use_previous_temporal_rt_as_guess", "value": args.use_previous_temporal_rt_as_guess, "description": "Uses only the previous estimate, never JSON extrinsic."},
        {"parameter": "max_view_reprojection_rms_px", "value": args.max_view_reprojection_rms_px, "description": "Robust temporal outlier floor."},
        {"parameter": "single_frame_baseline_range_mm", "value": f"{args.min_baseline_mm}..{args.max_baseline_mm}", "description": "IPPE branch-pair geometry gate; not from JSON."},
        {"parameter": "single_frame_max_stereo_rotation_deg", "value": args.max_stereo_rotation_deg, "description": "Soft rig-geometry penalty; not from JSON."},
        {"parameter": "single_frame_min_horizontal_translation_ratio", "value": args.min_horizontal_translation_ratio, "description": "Soft rig-geometry penalty; sign is not assumed."},
        {"parameter": "eye_image_size", "value": str(image_size), "description": "Width, height."},
        {"parameter": "valid_id2_observations", "value": len(observations), "description": "ID2 found in both eyes."},
    ]
    protocol = [
        {"item": 1, "description": "Split each SBS frame into left and right images."},
        {"item": 2, "description": "Detect all markers in each eye, then discard every ID except ID2."},
        {"item": 3, "description": "Refine ID2 corners to subpixel coordinates with the existing detector."},
        {"item": 4, "description": "Single-frame: enumerate left/right IPPE square branches and select with non-JSON stereo-rig geometry constraints."},
        {"item": 5, "description": "Temporal: jointly calibrate the most recent valid ID2 views with one shared left-to-right RT."},
        {"item": 6, "description": "Reject temporal views with unusually high per-view reprojection RMS, then refit."},
        {"item": 7, "description": "Use JSON R/T only after estimation to calculate rotation, baseline and translation errors."},
        {"item": 8, "description": "A useful temporal sequence should vary marker position, distance and tilt; identical repeated views mainly average noise."},
    ]
    chart_rows = build_chart_rows(
        rows, args.rotation_threshold_deg, args.baseline_threshold_percent
    )
    write_xlsx(
        output,
        [
            ("Settings", ["parameter", "value", "description"], settings),
            ("Summary", SUMMARY_HEADERS, [raw_summary, temporal_summary]),
            ("Frame Results", FRAME_HEADERS, rows),
            ("Charts", CHART_HEADERS, chart_rows),
            ("Final Estimate", FINAL_HEADERS, [final]),
            ("Protocol", ["item", "description"], protocol),
        ],
    )
    add_comparison_charts(
        output, rows, args.rotation_threshold_deg, args.baseline_threshold_percent
    )
    csv_path = output.with_suffix(".csv")
    write_csv(csv_path, rows)
    diagnostic_path = output.with_name(output.stem + "_diagnostic.mp4")
    diagnostic_written = False
    if args.diagnostic_video:
        diagnostic_written = write_diagnostic_video(
            video_path, diagnostic_path, rows, args, left_roi, right_roi
        )

    print("\n========== ID2 temporal stereo comparison ==========")
    print(f"Requested frames: {len(rows)} | ID2 both eyes: {len(observations)}")
    print(
        "Single-frame mean: rotation={} deg | baseline={} %".format(
            "--" if raw_summary["rotation_error_mean_deg"] is None else f"{raw_summary['rotation_error_mean_deg']:.4f}",
            "--" if raw_summary["absolute_baseline_error_mean_percent"] is None else f"{raw_summary['absolute_baseline_error_mean_percent']:.3f}",
        )
    )
    print(
        "Temporal mean:     rotation={} deg | baseline={} %".format(
            "--" if temporal_summary["rotation_error_mean_deg"] is None else f"{temporal_summary['rotation_error_mean_deg']:.4f}",
            "--" if temporal_summary["absolute_baseline_error_mean_percent"] is None else f"{temporal_summary['absolute_baseline_error_mean_percent']:.3f}",
        )
    )
    if final_fit is not None:
        print(
            "Offline all:       rotation={:.4f} deg | baseline={:.3f} % | used={}/{}".format(
                final["rotation_error_deg"],
                final["absolute_baseline_error_percent"],
                final["used_observations"],
                final["input_observations"],
            )
        )
    print(f"Excel: {output}")
    print(f"CSV:   {csv_path}")
    if diagnostic_written:
        print(f"Video: {diagnostic_path}")
    print("====================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
