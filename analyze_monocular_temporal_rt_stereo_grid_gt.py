"""Evaluate monocular temporal RT with an independent stereo ArUcoGrid GT.

Experiment separation
---------------------
TEST path (the method being evaluated):
    left reference frame N -> left candidate frame M..K
    small ArUco ID 2 + wound-region SIFT + causal temporal prior only.

GT path (used only after the test estimate has been produced):
    synchronized left/right images + a metric ArUco grid board.  The board pose
    is jointly refined against both eyes using the calibrated stereo extrinsic.

Grid defaults are imported directly from ``gen_arucoGrid_SVG_PNG.py``.  The
current generator uses a 7x10 DICT_4X4_100 board with IDs 20..89, so it does not
conflict with the small ID2.  TEST_ROI and GRID_ROI are still required to keep
wound SIFT and GT evidence isolated.  Coordinates are for one eye (not the
complete SBS frame), formatted x,y,w,h.

Example
-------
python analyze_monocular_temporal_rt_stereo_grid_gt.py input.mp4 ^
  --reference-frame 50 --start-frame 60 --end-frame 159 ^
  --test-roi 300,180,850,700 --grid-roi 1180,80,650,850
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import gen_arucoGrid_SVG_PNG as grid_generator

from analyze_hbvcam_aruco_corner_rt_stability import (
    ExcelFormula,
    excel_column,
    write_xlsx,
)


# =============================================================================
# User settings (CLI arguments override these values)
# =============================================================================

VIDEO_PATH = ""
CALIBRATION_PATH = "calibration_result_HBVCAM_4M2214HD-2-v11.json"
REFERENCE_FRAME = 50
START_FRAME = 60
END_FRAME = 159

# Required, in one-eye pixel coordinates: (x, y, width, height).
TEST_ROI: tuple[int, int, int, int] | None = None
GRID_ROI: tuple[int, int, int, int] | None = None

SMALL_MARKER_ID = 2
SMALL_MARKER_SIZE_MM = 8.25
SMALL_MARKER_DICTIONARY = "DICT_4X4_100"

def aruco_dictionary_name(dictionary_value: int | str) -> str:
    """Normalize an OpenCV ArUco constant or an existing CLI dictionary name."""
    if isinstance(dictionary_value, str):
        name = dictionary_value.strip()
        if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
            raise ValueError(f"Unknown ArUco dictionary name: {dictionary_value}")
        return name

    dictionary_value = int(dictionary_value)
    preferred = []
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(cv2.aruco, name)
        if isinstance(value, int) and value == dictionary_value:
            preferred.append(name)
    if not preferred:
        raise ValueError(
            f"Cannot map ArUco dictionary value {dictionary_value} to a name"
        )
    return sorted(preferred, key=lambda item: ("_ORIGINAL" in item, item))[0]


# Read the authoritative board specification directly from the generator.
GRID_SPEC_SOURCE = str(Path(grid_generator.__file__).resolve())
GRID_DICTIONARY = aruco_dictionary_name(grid_generator.ARUCO_DICT)
GRID_ROWS = int(grid_generator.GRID_ROWS)
GRID_COLS = int(grid_generator.GRID_COLS)
GRID_START_ID = int(grid_generator.GRID_START_ID)
GRID_MARKER_SIZE_MM = float(grid_generator.GRID_MARKER_SIZE_MM)
GRID_GAP_MM = float(grid_generator.GRID_GAP_MM)

ROTATION_PASS_THRESHOLD_DEG = 2.0
BASELINE_PASS_THRESHOLD_PERCENT = 5.0
MIN_GT_BASELINE_FOR_PERCENT_MM = 1.0
GT_MIN_MARKERS_PER_EYE = 3
GT_MAX_STEREO_REPROJECTION_RMS_PX = 3.0

SIFT_MAX_FEATURES = 1400
SIFT_RATIO = 0.75
SIFT_RANSAC_THRESHOLD_PX = 1.0
SIFT_MIN_MATCHES = 8
EXCLUDE_SMALL_MARKER_FROM_SIFT = True
TEMPORAL_PRIOR_ENABLED = True
TEMPORAL_PROPAGATION_ENABLED = True
SAVE_DIAGNOSTIC_VIDEO = True


FRAME_HEADERS = [
    "frame_index",
    "status",
    "estimate_source",
    "small_marker_found",
    "sift_keypoints_reference",
    "sift_keypoints_current",
    "sift_mutual_matches",
    "sift_E_inliers",
    "sift_candidate_epi_median_px",
    "temporal_prior_used",
    "test_marker_reprojection_rms_px",
    "gt_left_marker_count",
    "gt_right_marker_count",
    "gt_stereo_reprojection_rms_px",
    "estimated_rotation_angle_deg",
    "gt_rotation_angle_deg",
    "rotation_error_deg",
    "estimated_baseline_mm",
    "gt_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_error_percent",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "rotation_pass",
    "baseline_pass",
    "both_pass",
]


@dataclass
class Pose:
    """Rigid transform X_out = R @ X_in + t."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3, 1)


@dataclass
class MarkerPoseCandidate:
    pose: Pose
    reprojection_rms_px: float


@dataclass
class FeatureMatchResult:
    points_ref: np.ndarray
    points_cur: np.ndarray
    E_inlier_mask: np.ndarray
    recovered_pose: Pose | None
    keypoints_ref: int
    keypoints_cur: int
    mutual_matches: int
    E_inliers: int


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compose(a: Pose, b: Pose) -> Pose:
    """Return a o b."""
    return Pose(a.R @ b.R, a.R @ b.t + a.t)


def inverse(pose: Pose) -> Pose:
    R_inv = pose.R.T
    return Pose(R_inv, -R_inv @ pose.t)


def relative_pose(world_to_ref: Pose, world_to_cur: Pose) -> Pose:
    """Reference-camera coordinates -> current-camera coordinates."""
    return compose(world_to_cur, inverse(world_to_ref))


def rotation_vector_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(a) @ np.asarray(b).T)[0].reshape(3)


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if text is None or not str(text).strip():
        return None
    try:
        values = tuple(int(round(float(v.strip()))) for v in str(text).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height with positive size")
    return values


def clip_roi(roi, image_shape) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if roi is None:
        return 0, 0, width, height
    x, y, w, h = roi
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(x0, min(width, x + w))
    y1 = max(y0, min(height, y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside the {width}x{height} eye image")
    return x0, y0, x1 - x0, y1 - y0


def roi_overlap(a, b) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )


def dictionary_from_name(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


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
        Pose(extrinsic["R"], extrinsic["T"]),
    )


def split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = frame.shape[1]
    if width % 2:
        raise ValueError(f"SBS frame width must be even, got {width}")
    middle = width // 2
    return frame[:, :middle].copy(), frame[:, middle:].copy()


class RoiArucoDetector:
    def __init__(self, dictionary_name: str):
        self.dictionary_name = dictionary_name
        self.dictionary = dictionary_from_name(dictionary_name)
        params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.params = params

    def detect(self, gray: np.ndarray, roi, exclude_roi=None) -> dict[int, np.ndarray]:
        x, y, w, h = clip_roi(roi, gray.shape)
        crop = gray[y : y + h, x : x + w].copy()
        if exclude_roi is not None:
            ex, ey, ew, eh = clip_roi(exclude_roi, gray.shape)
            x0, y0 = max(x, ex), max(y, ey)
            x1, y1 = min(x + w, ex + ew), min(y + h, ey + eh)
            if x1 > x0 and y1 > y0:
                crop[y0 - y:y1 - y, x0 - x:x1 - x] = 255
        if self.detector is not None:
            corners, ids, _rejected = self.detector.detectMarkers(crop)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                crop, self.dictionary, parameters=self.params
            )
        result: dict[int, np.ndarray] = {}
        if ids is None:
            return result
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 1, 2)
            side = float(
                np.mean(
                    np.linalg.norm(
                        np.roll(points.reshape(4, 2), -1, axis=0)
                        - points.reshape(4, 2),
                        axis=1,
                    )
                )
            )
            window = int(np.clip(round(side / 12.0), 2, 9))
            cv2.cornerSubPix(crop, points, (window, window), (-1, -1), criteria)
            points = points.reshape(4, 2).astype(np.float64)
            points[:, 0] += x
            points[:, 1] += y
            marker_id = int(marker_id)
            old = result.get(marker_id)
            if old is None or abs(cv2.contourArea(points.astype(np.float32))) > abs(
                cv2.contourArea(old.astype(np.float32))
            ):
                result[marker_id] = points
        return result


def square_object_points(size_mm: float) -> np.ndarray:
    half = size_mm * 0.5
    return np.asarray(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )


def project_points(points_3d, pose: Pose, K, distortion) -> np.ndarray:
    rvec = cv2.Rodrigues(pose.R)[0]
    projected, _jac = cv2.projectPoints(
        np.asarray(points_3d, dtype=np.float64).reshape(-1, 3),
        rvec,
        pose.t,
        K,
        distortion,
    )
    return projected.reshape(-1, 2)


def small_marker_pose_candidates(corners, size_mm, K, distortion):
    object_points = square_object_points(size_mm)
    try:
        _count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points,
            np.asarray(corners, dtype=np.float64).reshape(4, 2),
            K,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    candidates = []
    errors = np.asarray(errors).reshape(-1) if errors is not None else []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        pose = Pose(cv2.Rodrigues(rvec)[0], tvec)
        camera_points = (pose.R @ object_points.T + pose.t).T
        if np.min(camera_points[:, 2]) <= 0:
            continue
        projected = project_points(object_points, pose, K, distortion)
        rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
        if index < len(errors) and math.isfinite(float(errors[index])):
            rms = float(errors[index])
        candidates.append(MarkerPoseCandidate(pose, rms))
    return candidates


def make_feature_mask(gray, roi, marker_corners=None):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x, y, w, h = clip_roi(roi, gray.shape)
    mask[y : y + h, x : x + w] = 255
    if EXCLUDE_SMALL_MARKER_FROM_SIFT and marker_corners is not None:
        centre = np.mean(marker_corners, axis=0)
        expanded = centre + (np.asarray(marker_corners) - centre) * 1.25
        cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 0)
    return mask


class SiftTemporalMatcher:
    def __init__(self, gray_ref, K, distortion, roi, marker_corners):
        self.K = K
        self.distortion = distortion
        self.extractor = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        mask = make_feature_mask(gray_ref, roi, marker_corners)
        self.kp_ref, self.desc_ref = self.extractor.detectAndCompute(gray_ref, mask)

    @staticmethod
    def _ratio_pairs(knn, ratio):
        result = {}
        for pair in knn:
            if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
                result[pair[0].queryIdx] = pair[0]
        return result

    def match(self, gray_cur, roi, marker_corners) -> FeatureMatchResult:
        mask = make_feature_mask(gray_cur, roi, marker_corners)
        kp_cur, desc_cur = self.extractor.detectAndCompute(gray_cur, mask)
        empty = np.empty((0, 2), dtype=np.float64)
        if self.desc_ref is None or desc_cur is None:
            return FeatureMatchResult(
                empty, empty, np.zeros(0, bool), None,
                len(self.kp_ref), len(kp_cur), 0, 0,
            )
        forward = self._ratio_pairs(
            self.matcher.knnMatch(self.desc_ref, desc_cur, k=2), SIFT_RATIO
        )
        backward = self._ratio_pairs(
            self.matcher.knnMatch(desc_cur, self.desc_ref, k=2), SIFT_RATIO
        )
        matches = [
            match for query, match in forward.items()
            if match.trainIdx in backward and backward[match.trainIdx].trainIdx == query
        ]
        matches.sort(key=lambda item: item.distance)
        matches = matches[:300]
        if not matches:
            return FeatureMatchResult(
                empty, empty, np.zeros(0, bool), None,
                len(self.kp_ref), len(kp_cur), 0, 0,
            )
        raw_ref = np.asarray([self.kp_ref[m.queryIdx].pt for m in matches], np.float64)
        raw_cur = np.asarray([kp_cur[m.trainIdx].pt for m in matches], np.float64)
        points_ref = cv2.undistortPoints(
            raw_ref.reshape(-1, 1, 2), self.K, self.distortion, P=self.K
        ).reshape(-1, 2)
        points_cur = cv2.undistortPoints(
            raw_cur.reshape(-1, 1, 2), self.K, self.distortion, P=self.K
        ).reshape(-1, 2)
        inlier_mask = np.ones(len(matches), dtype=bool)
        recovered = None
        E_inliers = 0
        if len(matches) >= SIFT_MIN_MATCHES:
            E, mask_E = cv2.findEssentialMat(
                points_ref,
                points_cur,
                self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=SIFT_RANSAC_THRESHOLD_PX,
            )
            if E is not None and mask_E is not None:
                inlier_mask = mask_E.reshape(-1).astype(bool)
                best = None
                matrices = [E[i : i + 3] for i in range(0, E.shape[0], 3)]
                for essential in matrices:
                    if essential.shape != (3, 3):
                        continue
                    try:
                        count, R, t, pose_mask = cv2.recoverPose(
                            essential,
                            points_ref,
                            points_cur,
                            self.K,
                            mask=mask_E.copy(),
                        )
                    except cv2.error:
                        continue
                    if best is None or count > best[0]:
                        best = (int(count), Pose(R, t), pose_mask.reshape(-1).astype(bool))
                if best is not None:
                    E_inliers, recovered, pose_inliers = best
                    inlier_mask &= pose_inliers
        return FeatureMatchResult(
            points_ref,
            points_cur,
            inlier_mask,
            recovered,
            len(self.kp_ref),
            len(kp_cur),
            len(matches),
            E_inliers,
        )


def skew(vector) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], np.float64)


def signed_sampson_residuals(pose: Pose, points_ref, points_cur, K) -> np.ndarray:
    if len(points_ref) == 0 or np.linalg.norm(pose.t) < 1e-9:
        return np.zeros(0, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ skew(pose.t.reshape(3)) @ pose.R @ K_inv
    ones = np.ones((len(points_ref), 1), dtype=np.float64)
    x1 = np.hstack([points_ref, ones])
    x2 = np.hstack([points_cur, ones])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numerator = np.sum(x2 * Fx1, axis=1)
    denominator = np.sqrt(
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    )
    return numerator / np.maximum(denominator, 1e-12)


def translation_direction_error_deg(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-9:
        return None
    cosine = np.clip(float(np.dot(a, b) / denominator), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def marker_transfer_residuals(
    pose: Pose,
    ref_marker_pose: Pose | None,
    cur_marker_pose: Pose | None,
    corners_ref,
    corners_cur,
    object_points,
    K,
    distortion,
) -> np.ndarray:
    if ref_marker_pose is None or cur_marker_pose is None:
        return np.zeros(0, np.float64)
    points_ref_camera = (
        ref_marker_pose.R @ object_points.T + ref_marker_pose.t
    ).T
    points_cur_camera = (cur_marker_pose.R @ object_points.T + cur_marker_pose.t).T
    projected_cur = project_points(points_ref_camera, pose, K, distortion)
    projected_ref = project_points(
        points_cur_camera, inverse(pose), K, distortion
    )
    return np.concatenate(
        [(projected_cur - corners_cur).reshape(-1), (projected_ref - corners_ref).reshape(-1)]
    )


def robust_lm(initial: Pose, residual_function, iterations=12) -> Pose:
    params = np.concatenate([cv2.Rodrigues(initial.R)[0].reshape(3), initial.t.reshape(3)])
    damping = 1e-3

    def evaluate(values):
        pose = Pose(cv2.Rodrigues(values[:3])[0], values[3:])
        residual = np.asarray(residual_function(pose), dtype=np.float64).reshape(-1)
        return residual[np.isfinite(residual)]

    residual = evaluate(params)
    if len(residual) < 6:
        return initial
    for _ in range(iterations):
        absolute = np.abs(residual)
        huber = 2.0
        weights = np.ones_like(residual)
        mask = absolute > huber
        weights[mask] = huber / np.maximum(absolute[mask], 1e-12)
        sqrt_w = np.sqrt(weights)
        jacobian = np.empty((len(residual), 6), dtype=np.float64)
        steps = np.asarray([1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
        for column, step in enumerate(steps):
            shifted = params.copy()
            shifted[column] += step
            trial = evaluate(shifted)
            if len(trial) != len(residual):
                return Pose(cv2.Rodrigues(params[:3])[0], params[3:])
            jacobian[:, column] = (trial - residual) / step
        A = jacobian * sqrt_w[:, None]
        b = residual * sqrt_w
        normal = A.T @ A + damping * np.eye(6)
        try:
            delta = np.linalg.solve(normal, -A.T @ b)
        except np.linalg.LinAlgError:
            break
        trial_params = params + delta
        trial_residual = evaluate(trial_params)
        if len(trial_residual) != len(residual):
            break
        if np.mean(trial_residual**2) < np.mean(residual**2):
            params, residual = trial_params, trial_residual
            damping = max(damping * 0.4, 1e-8)
            if np.linalg.norm(delta) < 1e-7:
                break
        else:
            damping = min(damping * 10.0, 1e6)
    return Pose(cv2.Rodrigues(params[:3])[0], params[3:])


def grid_object_map(rows, cols, start_id, marker_mm, gap_mm):
    result, pitch = {}, marker_mm + gap_mm
    for row in range(rows):
        for col in range(cols):
            marker_id = start_id + row * cols + col
            x, y = col * pitch, row * pitch
            result[marker_id] = np.asarray(
                [[x, y, 0], [x + marker_mm, y, 0],
                 [x + marker_mm, y + marker_mm, 0], [x, y + marker_mm, 0]],
                np.float64,
            )
    return result


def board_correspondences(markers, object_map):
    objects, images, ids = [], [], []
    for marker_id in sorted(set(markers) & set(object_map)):
        objects.extend(object_map[marker_id])
        images.extend(markers[marker_id])
        ids.append(marker_id)
    return (
        np.asarray(objects, np.float64).reshape(-1, 3),
        np.asarray(images, np.float64).reshape(-1, 2),
        ids,
    )


def solve_board_pose(objects, images, K, distortion):
    if len(objects) < 8:
        return None
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        objects, images, K, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
        iterationsCount=200,
        reprojectionError=2.0,
        confidence=0.999,
    )
    if not ok:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objects, images, K, distortion, rvec, tvec
        )
    return Pose(cv2.Rodrigues(rvec)[0], tvec)


def average_pose(a, b):
    u, _s, vt = np.linalg.svd(a.R + b.R)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return Pose(rotation, (a.t + b.t) * 0.5)


def estimate_grid_gt(markers_l, markers_r, object_map, K_l, d_l, K_r, d_r,
                     left_to_right, min_markers):
    obj_l, img_l, ids_l = board_correspondences(markers_l, object_map)
    obj_r, img_r, ids_r = board_correspondences(markers_r, object_map)
    diagnostics = {
        "left_markers": len(ids_l),
        "right_markers": len(ids_r),
        "rms_px": None,
    }
    if len(ids_l) < min_markers or len(ids_r) < min_markers:
        return None, diagnostics
    solved_l = solve_board_pose(obj_l, img_l, K_l, d_l)
    solved_r = solve_board_pose(obj_r, img_r, K_r, d_r)
    if solved_l is None or solved_r is None:
        return None, diagnostics
    seed = average_pose(
        solved_l,
        compose(inverse(left_to_right), solved_r),
    )

    def residual(pose):
        error_l = project_points(obj_l, pose, K_l, d_l) - img_l
        pose_r = compose(left_to_right, pose)
        error_r = project_points(obj_r, pose_r, K_r, d_r) - img_r
        return np.concatenate([error_l.reshape(-1), error_r.reshape(-1)])

    pose = robust_lm(seed, residual, 18)
    errors = residual(pose).reshape(-1, 2)
    diagnostics["rms_px"] = float(
        np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
    )
    return pose, diagnostics


def predict_temporal(history, frame_index):
    if not history:
        return None
    if len(history) == 1:
        return history[-1][1]
    frame_a, pose_a = history[-2]
    frame_b, pose_b = history[-1]
    if frame_b == frame_a:
        return pose_b
    ratio = (frame_index - frame_b) / (frame_b - frame_a)
    delta = cv2.Rodrigues(pose_b.R @ pose_a.R.T)[0].reshape(3) * ratio
    return Pose(
        cv2.Rodrigues(delta)[0] @ pose_b.R,
        pose_b.t + (pose_b.t - pose_a.t) * ratio,
    )


def estimate_test_pose(ref_candidates, cur_candidates, corners_ref, corners_cur,
                       features, prediction, object_points, K, distortion):
    mask = features.E_inlier_mask
    if len(mask) and np.count_nonzero(mask) >= SIFT_MIN_MATCHES:
        points_ref = features.points_ref[mask]
        points_cur = features.points_cur[mask]
    else:
        points_ref = features.points_ref
        points_cur = features.points_cur
    best = None
    for ref in ref_candidates:
        for cur in cur_candidates:
            pose = relative_pose(ref.pose, cur.pose)
            marker = marker_transfer_residuals(
                pose, ref.pose, cur.pose, corners_ref, corners_cur,
                object_points, K, distortion,
            )
            marker_rms = float(np.sqrt(np.mean(marker ** 2)))
            epi = np.abs(
                signed_sampson_residuals(pose, points_ref, points_cur, K)
            )
            epi_median = float(np.median(epi)) if len(epi) else 5.0
            score = marker_rms + 0.8 * epi_median
            if features.recovered_pose is not None:
                score += 0.04 * rotation_angle_deg(
                    pose.R @ features.recovered_pose.R.T
                )
                direction = translation_direction_error_deg(
                    pose.t, features.recovered_pose.t
                )
                score += 0.015 * (
                    direction if direction is not None else 180.0
                )
            if prediction is not None:
                score += 0.05 * rotation_angle_deg(
                    pose.R @ prediction.R.T
                )
                score += 0.02 * float(
                    np.linalg.norm(pose.t - prediction.t)
                )
            item = (score, pose, ref.pose, cur.pose)
            if best is None or item[0] < best[0]:
                best = item
    if best is None:
        if not (
            TEMPORAL_PROPAGATION_ENABLED
            and prediction is not None
            and len(points_ref) >= SIFT_MIN_MATCHES
        ):
            return None
        initial, ref_pose, cur_pose = prediction, None, None
        source = "sift_temporal_propagated"
    else:
        _score, initial, ref_pose, cur_pose = best
        source = "marker_sift_temporal"

    def residual(pose):
        parts = []
        marker = marker_transfer_residuals(
            pose, ref_pose, cur_pose, corners_ref, corners_cur,
            object_points, K, distortion,
        )
        if len(marker):
            parts.append(marker * 1.5)
        feature = signed_sampson_residuals(
            pose, points_ref, points_cur, K
        )
        if len(feature):
            parts.append(feature)
        if prediction is not None:
            parts.append(
                rotation_vector_between(pose.R, prediction.R) * 6.0
            )
            parts.append((pose.t - prediction.t).reshape(3) * 0.07)
        return np.concatenate(parts) if parts else np.zeros(0)

    pose = robust_lm(initial, residual, 12)
    marker_rms = None
    if ref_pose is not None:
        marker = marker_transfer_residuals(
            pose, ref_pose, cur_pose, corners_ref, corners_cur,
            object_points, K, distortion,
        )
        marker_rms = float(np.sqrt(np.mean(marker ** 2)))
    epi = np.abs(
        signed_sampson_residuals(pose, points_ref, points_cur, K)
    )
    return {
        "pose": pose,
        "source": source,
        "marker_rms_px": marker_rms,
        "epi_median_px": float(np.median(epi)) if len(epi) else None,
    }


def temporal_errors(estimated, truth):
    estimated_baseline = float(np.linalg.norm(estimated.t))
    truth_baseline = float(np.linalg.norm(truth.t))
    delta = estimated_baseline - truth_baseline
    return {
        "estimated_rotation_angle_deg": rotation_angle_deg(estimated.R),
        "gt_rotation_angle_deg": rotation_angle_deg(truth.R),
        "rotation_error_deg": rotation_angle_deg(
            estimated.R @ truth.R.T
        ),
        "estimated_baseline_mm": estimated_baseline,
        "gt_baseline_mm": truth_baseline,
        "baseline_delta_mm": delta,
        "absolute_baseline_error_percent": (
            abs(delta) / truth_baseline * 100.0
            if truth_baseline >= MIN_GT_BASELINE_FOR_PERCENT_MM
            else None
        ),
        "translation_l2_error_mm": float(
            np.linalg.norm(estimated.t - truth.t)
        ),
        "translation_direction_error_deg": (
            translation_direction_error_deg(estimated.t, truth.t)
        ),
    }


def read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read video frame {frame_index}")
    return frame


def draw_markers(image, markers, color, x_offset=0):
    for marker_id, corners in markers.items():
        shifted = np.asarray(corners, np.float64).copy()
        shifted[:, 0] += x_offset
        polygon = np.rint(shifted).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [polygon], True, color, 1, cv2.LINE_AA)
        x, y = np.rint(shifted[0]).astype(int)
        cv2.putText(
            image,
            str(marker_id),
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_roi(image, roi, color, x_offset=0):
    x, y, width, height = roi
    cv2.rectangle(
        image,
        (x + x_offset, y),
        (x + width + x_offset, y + height),
        color,
        1,
        cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate reference-left F(N) to candidate-left F(M..K) RT using "
            "only small ID2, wound SIFT and a causal temporal prior, then "
            "evaluate it with an independent stereo ArUcoGrid ground truth."
        )
    )
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--reference-frame", type=int, default=REFERENCE_FRAME)
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--end-frame", type=int, default=END_FRAME)
    parser.add_argument("--test-roi", type=parse_roi, default=TEST_ROI)
    parser.add_argument("--grid-roi", type=parse_roi, default=GRID_ROI)
    parser.add_argument("--small-marker-id", type=int, default=SMALL_MARKER_ID)
    parser.add_argument(
        "--small-marker-size-mm", type=float, default=SMALL_MARKER_SIZE_MM
    )
    parser.add_argument(
        "--small-dictionary", default=SMALL_MARKER_DICTIONARY
    )
    parser.add_argument("--grid-dictionary", default=GRID_DICTIONARY)
    parser.add_argument("--grid-rows", type=int, default=GRID_ROWS)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--grid-start-id", type=int, default=GRID_START_ID)
    parser.add_argument(
        "--grid-marker-size-mm", type=float, default=GRID_MARKER_SIZE_MM
    )
    parser.add_argument("--grid-gap-mm", type=float, default=GRID_GAP_MM)
    parser.add_argument(
        "--gt-min-markers-per-eye", type=int, default=GT_MIN_MARKERS_PER_EYE
    )
    parser.add_argument(
        "--gt-max-rms-px",
        type=float,
        default=GT_MAX_STEREO_REPROJECTION_RMS_PX,
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=ROTATION_PASS_THRESHOLD_DEG,
    )
    parser.add_argument(
        "--baseline-threshold-percent",
        type=float,
        default=BASELINE_PASS_THRESHOLD_PERCENT,
    )
    parser.add_argument(
        "--temporal-prior",
        action=argparse.BooleanOptionalAction,
        default=TEMPORAL_PRIOR_ENABLED,
    )
    parser.add_argument(
        "--diagnostic-video",
        action=argparse.BooleanOptionalAction,
        default=SAVE_DIAGNOSTIC_VIDEO,
    )
    parser.add_argument("--output", help="Output .xlsx path")
    args = parser.parse_args()
    if not args.video:
        parser.error("Specify VIDEO or set VIDEO_PATH at the top of this file")
    if args.test_roi is None or args.grid_roi is None:
        parser.error(
            "--test-roi and --grid-roi are required. The default Grid and "
            "the small marker both contain ID2, so spatial isolation is mandatory."
        )
    if args.reference_frame < 0 or args.start_frame < 0:
        parser.error("frame indexes must be non-negative")
    if args.end_frame < args.start_frame:
        parser.error("--end-frame must be >= --start-frame")
    if args.small_marker_size_mm <= 0 or args.grid_marker_size_mm <= 0:
        parser.error("marker sizes must be positive")
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        parser.error("grid rows and columns must be positive")
    return args


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_HEADERS)
        writer.writeheader()
        writer.writerows(
            {header: row.get(header) for header in FRAME_HEADERS}
            for row in rows
        )


def build_summary(rows, rotation_threshold, baseline_threshold):
    comparable = [row for row in rows if row.get("status") == "OK"]
    estimated = [row for row in rows if row.get("estimate_source")]
    direct = [
        row
        for row in rows
        if row.get("estimate_source") == "marker_sift_temporal"
    ]
    propagated = [
        row
        for row in rows
        if row.get("estimate_source") == "sift_temporal_propagated"
    ]
    rotation_pass = sum(
        row["rotation_error_deg"] < rotation_threshold for row in comparable
    )
    baseline_pass = sum(
        row["absolute_baseline_error_percent"] < baseline_threshold
        for row in comparable
    )
    both_pass = sum(
        row["rotation_error_deg"] < rotation_threshold
        and row["absolute_baseline_error_percent"] < baseline_threshold
        for row in comparable
    )
    return {
        "requested_frames": len(rows),
        "estimated_frames": len(estimated),
        "direct_marker_frames": len(direct),
        "temporal_propagated_frames": len(propagated),
        "comparable_frames": len(comparable),
        "rotation_pass_frames": rotation_pass,
        "baseline_pass_frames": baseline_pass,
        "both_pass_frames": both_pass,
        "both_pass_rate": both_pass / len(comparable) if comparable else 0.0,
        "rotation_error_mean_deg": (
            float(np.mean([row["rotation_error_deg"] for row in comparable]))
            if comparable
            else None
        ),
        "baseline_error_mean_percent": (
            float(
                np.mean(
                    [
                        row["absolute_baseline_error_percent"]
                        for row in comparable
                    ]
                )
            )
            if comparable
            else None
        ),
    }


def add_live_summary_formulas(summary, rows):
    if not rows:
        return summary
    last_row = len(rows) + 1
    columns = {
        name: excel_column(FRAME_HEADERS.index(name) + 1)
        for name in (
            "status",
            "estimate_source",
            "rotation_error_deg",
            "absolute_baseline_error_percent",
        )
    }

    def column_range(name):
        letter = columns[name]
        return f"'Frame Results'!${letter}$2:${letter}${last_row}"

    status = column_range("status")
    source = column_range("estimate_source")
    rotation = column_range("rotation_error_deg")
    baseline = column_range("absolute_baseline_error_percent")
    comparable = summary["comparable_frames"]
    summary["estimated_frames"] = ExcelFormula(
        f'COUNTIF({source},"<>")', summary["estimated_frames"]
    )
    summary["comparable_frames"] = ExcelFormula(
        f'COUNTIF({status},"OK")', comparable
    )
    summary["rotation_pass_frames"] = ExcelFormula(
        f'COUNTIFS({status},"OK",{rotation},"<"&\'Settings\'!$B$2)',
        summary["rotation_pass_frames"],
    )
    summary["baseline_pass_frames"] = ExcelFormula(
        f'COUNTIFS({status},"OK",{baseline},"<"&\'Settings\'!$B$3)',
        summary["baseline_pass_frames"],
    )
    both_formula = (
        f'COUNTIFS({status},"OK",{rotation},"<"&\'Settings\'!$B$2,'
        f'{baseline},"<"&\'Settings\'!$B$3)'
    )
    summary["both_pass_frames"] = ExcelFormula(
        both_formula, summary["both_pass_frames"]
    )
    summary["both_pass_rate"] = ExcelFormula(
        f'IFERROR({both_formula}/COUNTIF({status},"OK"),0)',
        summary["both_pass_rate"],
    )
    summary["rotation_error_mean_deg"] = ExcelFormula(
        f'IFERROR(AVERAGEIFS({rotation},{status},"OK"),0)',
        summary["rotation_error_mean_deg"] or 0,
    )
    summary["baseline_error_mean_percent"] = ExcelFormula(
        f'IFERROR(AVERAGEIFS({baseline},{status},"OK"),0)',
        summary["baseline_error_mean_percent"] or 0,
    )
    return summary


def initialize_diagnostic_writer(path, sbs_shape, fps):
    height, width = sbs_shape[:2]
    scale = min(1.0, 1280.0 / width)
    size = (int(round(width * scale)), int(round(height * scale)))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create diagnostic video: {path}")
    return writer, size


CHART_HELPER_HEADERS = [
    "frame_index",
    "rotation_threshold_deg",
    "baseline_threshold_percent",
]


def build_chart_helper_rows(rows, rotation_threshold, baseline_threshold):
    helpers = []
    for source_row_index, row in enumerate(rows, start=2):
        helpers.append(
            {
                "frame_index": ExcelFormula(
                    f"'Frame Results'!$A${source_row_index}",
                    int(row["frame_index"]),
                ),
                "rotation_threshold_deg": ExcelFormula(
                    "'Settings'!$B$2", rotation_threshold
                ),
                "baseline_threshold_percent": ExcelFormula(
                    "'Settings'!$B$3", baseline_threshold
                ),
            }
        )
    return helpers


def chart_number_reference(formula, values):
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{float(value):.15g}</c:v></c:pt>'
        for index, value in enumerate(values)
        if finite(value) is not None
    )
    return (
        "<c:numRef>"
        f"<c:f>{escape(formula)}</c:f>"
        "<c:numCache><c:formatCode>General</c:formatCode>"
        f'<c:ptCount val="{len(values)}"/>{points}</c:numCache>'
        "</c:numRef>"
    )


def chart_title_xml(text, size=1400):
    return (
        "<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p>"
        f'<a:r><a:rPr lang="en-US" sz="{size}"/><a:t>'
        f"{escape(text)}</a:t></a:r></a:p></c:rich></c:tx>"
        '<c:layout/><c:overlay val="0"/></c:title>'
    )


def chart_axis_title_xml(text):
    return chart_title_xml(text, size=1000)


def build_frame_scatter_chart_xml(
    title,
    y_axis_title,
    x_formula,
    x_values,
    series,
    axis_seed,
):
    series_xml = []
    for index, item in enumerate(series):
        color = item["color"]
        marker_xml = (
            '<c:marker><c:symbol val="none"/></c:marker>'
            if item.get("threshold")
            else (
                '<c:marker><c:symbol val="circle"/><c:size val="4"/>'
                "<c:spPr><a:solidFill>"
                f'<a:srgbClr val="{color}"/>'
                "</a:solidFill><a:ln><a:solidFill>"
                f'<a:srgbClr val="{color}"/>'
                "</a:solidFill></a:ln></c:spPr></c:marker>"
            )
        )
        dash_xml = (
            '<a:prstDash val="dash"/>' if item.get("threshold") else ""
        )
        series_xml.append(
            "<c:ser>"
            f'<c:idx val="{index}"/><c:order val="{index}"/>'
            f"<c:tx><c:v>{escape(item['name'])}</c:v></c:tx>"
            f"{marker_xml}"
            '<c:spPr><a:ln w="19050"><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            f"</a:solidFill>{dash_xml}</a:ln></c:spPr>"
            f"<c:xVal>{chart_number_reference(x_formula, x_values)}</c:xVal>"
            f"<c:yVal>{chart_number_reference(item['formula'], item['values'])}</c:yVal>"
            '<c:smooth val="0"/></c:ser>'
        )
    x_axis_id = int(axis_seed)
    y_axis_id = x_axis_id + 1
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<c:date1904 val="0"/><c:lang val="en-US"/><c:roundedCorners val="0"/>'
        f"<c:chart>{chart_title_xml(title)}"
        '<c:autoTitleDeleted val="0"/><c:plotArea><c:layout/>'
        '<c:scatterChart><c:scatterStyle val="lineMarker"/>'
        f'<c:varyColors val="0"/>{"".join(series_xml)}'
        f'<c:axId val="{x_axis_id}"/><c:axId val="{y_axis_id}"/>'
        "</c:scatterChart>"
        f'<c:valAx><c:axId val="{x_axis_id}"/><c:scaling>'
        '<c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
        f'<c:axPos val="b"/>{chart_axis_title_xml("Frame index")}'
        '<c:numFmt formatCode="0" sourceLinked="0"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        '<c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{y_axis_id}"/><c:crosses val="autoZero"/>'
        "</c:valAx>"
        f'<c:valAx><c:axId val="{y_axis_id}"/><c:scaling>'
        '<c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
        f'<c:axPos val="l"/><c:majorGridlines/>{chart_axis_title_xml(y_axis_title)}'
        '<c:numFmt formatCode="0.00" sourceLinked="0"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        '<c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{x_axis_id}"/><c:crosses val="autoZero"/>'
        "</c:valAx></c:plotArea>"
        '<c:legend><c:legendPos val="b"/><c:layout/><c:overlay val="0"/></c:legend>'
        '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/>'
        '<c:showDLblsOverMax val="0"/></c:chart>'
        '<c:printSettings><c:headerFooter/><c:pageMargins b="0.75" l="0.7" '
        'r="0.7" t="0.75" header="0.3" footer="0.3"/><c:pageSetup/>'
        "</c:printSettings></c:chartSpace>"
    )


def chart_anchor_xml(chart_id, relationship_id, top_row, bottom_row):
    return (
        "<xdr:twoCellAnchor><xdr:from>"
        f"<xdr:col>4</xdr:col><xdr:colOff>0</xdr:colOff>"
        f"<xdr:row>{top_row}</xdr:row><xdr:rowOff>0</xdr:rowOff>"
        "</xdr:from><xdr:to>"
        "<xdr:col>14</xdr:col><xdr:colOff>0</xdr:colOff>"
        f"<xdr:row>{bottom_row}</xdr:row><xdr:rowOff>0</xdr:rowOff>"
        "</xdr:to><xdr:graphicFrame macro=\"\"><xdr:nvGraphicFramePr>"
        f'<xdr:cNvPr id="{chart_id + 1}" name="Chart {chart_id}"/>'
        "<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm>"
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
        '<a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'r:id="{relationship_id}"/></a:graphicData></a:graphic>'
        "</xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor>"
    )


def add_frame_charts_to_xlsx(
    path,
    rows,
    rotation_threshold,
    baseline_threshold,
    charts_sheet_index=4,
):
    if not rows:
        return
    last_row = len(rows) + 1
    x_values = [row["frame_index"] for row in rows]
    x_formula = f"'Charts'!$A$2:$A${last_row}"
    frame_columns = {
        name: excel_column(FRAME_HEADERS.index(name) + 1)
        for name in (
            "rotation_error_deg",
            "absolute_baseline_error_percent",
            "estimated_baseline_mm",
            "gt_baseline_mm",
        )
    }

    def frame_formula(name):
        column = frame_columns[name]
        return f"'Frame Results'!${column}$2:${column}${last_row}"

    charts = [
        build_frame_scatter_chart_xml(
            "Rotation error by frame",
            "Rotation error (deg)",
            x_formula,
            x_values,
            [
                {
                    "name": "Rotation error",
                    "formula": frame_formula("rotation_error_deg"),
                    "values": [row.get("rotation_error_deg") for row in rows],
                    "color": "4472C4",
                },
                {
                    "name": "Rotation threshold",
                    "formula": f"'Charts'!$B$2:$B${last_row}",
                    "values": [rotation_threshold] * len(rows),
                    "color": "C00000",
                    "threshold": True,
                },
            ],
            2100000000,
        ),
        build_frame_scatter_chart_xml(
            "Absolute baseline error by frame",
            "Baseline error (%)",
            x_formula,
            x_values,
            [
                {
                    "name": "Absolute baseline error",
                    "formula": frame_formula(
                        "absolute_baseline_error_percent"
                    ),
                    "values": [
                        row.get("absolute_baseline_error_percent")
                        for row in rows
                    ],
                    "color": "ED7D31",
                },
                {
                    "name": "Baseline threshold",
                    "formula": f"'Charts'!$C$2:$C${last_row}",
                    "values": [baseline_threshold] * len(rows),
                    "color": "C00000",
                    "threshold": True,
                },
            ],
            2110000000,
        ),
        build_frame_scatter_chart_xml(
            "Estimated and GT temporal baseline",
            "Baseline (mm)",
            x_formula,
            x_values,
            [
                {
                    "name": "Estimated baseline",
                    "formula": frame_formula("estimated_baseline_mm"),
                    "values": [
                        row.get("estimated_baseline_mm") for row in rows
                    ],
                    "color": "4472C4",
                },
                {
                    "name": "Grid GT baseline",
                    "formula": frame_formula("gt_baseline_mm"),
                    "values": [row.get("gt_baseline_mm") for row in rows],
                    "color": "70AD47",
                },
            ],
            2120000000,
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
            '<Relationship '
            f'Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
            f'Target="../charts/chart{index}.xml"/>'
            for index in range(1, 4)
        )
        + "</Relationships>"
    )
    sheet_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/></Relationships>'
    )
    chart_parts = {
        f"xl/worksheets/_rels/sheet{charts_sheet_index}.xml.rels": sheet_rels,
        "xl/drawings/drawing1.xml": drawing,
        "xl/drawings/_rels/drawing1.xml.rels": drawing_rels,
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
                    worksheet = worksheet.replace(
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
                        1,
                    )
                    worksheet = worksheet.replace(
                        "</worksheet>",
                        '<drawing r:id="rId1"/></worksheet>',
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
                            for index in range(1, 4)
                        )
                    )
                    data = content_types.replace(
                        "</Types>", overrides + "</Types>"
                    ).encode("utf-8")
                destination.writestr(info, data)
            for name, data in chart_parts.items():
                destination.writestr(name, data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    calibration_path = Path(args.calibration).resolve()
    output_xlsx = (
        Path(args.output).resolve()
        if args.output
        else video_path.with_name(
            video_path.stem + "_temporal_rt_grid_gt.xlsx"
        )
    )
    if output_xlsx.suffix.lower() != ".xlsx":
        output_xlsx = output_xlsx.with_suffix(".xlsx")
    output_csv = output_xlsx.with_suffix(".csv")
    diagnostic_path = output_xlsx.with_name(
        output_xlsx.stem + "_diagnostic.mp4"
    )
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    K_l, d_l, K_r, d_r, left_to_right = load_calibration(
        calibration_path
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    requested_indexes = (
        args.reference_frame,
        args.start_frame,
        args.end_frame,
    )
    if min(requested_indexes) < 0 or max(requested_indexes) >= frame_count:
        capture.release()
        raise ValueError(
            f"Requested frame is outside video range 0..{frame_count - 1}"
        )

    reference_sbs = read_frame(capture, args.reference_frame)
    reference_left, reference_right = split_sbs(reference_sbs)
    test_roi = clip_roi(args.test_roi, reference_left.shape)
    grid_roi = clip_roi(args.grid_roi, reference_left.shape)
    overlap = roi_overlap(test_roi, grid_roi)
    print(f"Video: {video_path}")
    print(
        f"Reference F{args.reference_frame}; candidates "
        f"F{args.start_frame}..F{args.end_frame}"
    )
    print(
        f"TEST ROI={test_roi}; GRID ROI={grid_roi}; overlap={overlap}px2"
    )
    print(
        "Grid detection blanks TEST ROI. TEST reads only TEST ROI; keep visible "
        "large-Grid markers outside that ROI."
    )

    small_detector = RoiArucoDetector(args.small_dictionary)
    grid_detector = RoiArucoDetector(args.grid_dictionary)
    grid_end_id = args.grid_start_id + args.grid_rows * args.grid_cols - 1
    grid_dictionary_capacity = int(
        grid_detector.dictionary.bytesList.shape[0]
    )
    if args.grid_start_id < 0 or grid_end_id >= grid_dictionary_capacity:
        capture.release()
        raise ValueError(
            f"Grid IDs {args.grid_start_id}..{grid_end_id} exceed "
            f"{args.grid_dictionary} capacity 0.."
            f"{grid_dictionary_capacity - 1}"
        )
    small_id_conflicts_with_grid = (
        args.small_dictionary == args.grid_dictionary
        and args.grid_start_id <= args.small_marker_id <= grid_end_id
    )
    print(
        f"Grid spec: {args.grid_dictionary}, {args.grid_rows}x{args.grid_cols}, "
        f"IDs {args.grid_start_id}..{grid_end_id}, "
        f"marker={args.grid_marker_size_mm}mm, gap={args.grid_gap_mm}mm"
    )
    if small_id_conflicts_with_grid:
        print(
            f"WARNING: small ID {args.small_marker_id} overlaps the Grid ID "
            "range; ROI isolation is mandatory."
        )
    grid_map = grid_object_map(
        args.grid_rows,
        args.grid_cols,
        args.grid_start_id,
        args.grid_marker_size_mm,
        args.grid_gap_mm,
    )
    small_object = square_object_points(args.small_marker_size_mm)
    gray_reference_left = cv2.cvtColor(
        reference_left, cv2.COLOR_BGR2GRAY
    )
    gray_reference_right = cv2.cvtColor(
        reference_right, cv2.COLOR_BGR2GRAY
    )

    reference_small_markers = small_detector.detect(
        gray_reference_left, test_roi
    )
    reference_small_corners = reference_small_markers.get(
        args.small_marker_id
    )
    if reference_small_corners is None:
        capture.release()
        raise RuntimeError(
            f"Small marker ID {args.small_marker_id} was not found in the "
            "reference frame TEST ROI"
        )
    reference_candidates = small_marker_pose_candidates(
        reference_small_corners,
        args.small_marker_size_mm,
        K_l,
        d_l,
    )
    if not reference_candidates:
        capture.release()
        raise RuntimeError("Reference marker has no valid IPPE solution")
    sift_matcher = SiftTemporalMatcher(
        gray_reference_left,
        K_l,
        d_l,
        test_roi,
        reference_small_corners,
    )

    reference_grid_left = grid_detector.detect(
        gray_reference_left, grid_roi, exclude_roi=test_roi
    )
    reference_grid_right = grid_detector.detect(
        gray_reference_right, grid_roi, exclude_roi=test_roi
    )
    gt_reference, reference_gt_diagnostics = estimate_grid_gt(
        reference_grid_left,
        reference_grid_right,
        grid_map,
        K_l,
        d_l,
        K_r,
        d_r,
        left_to_right,
        args.gt_min_markers_per_eye,
    )
    if gt_reference is None:
        capture.release()
        raise RuntimeError(
            "Reference Grid GT failed. Check GRID ROI, dictionary, rows, "
            "columns, first ID and physical dimensions. "
            f"Diagnostics: {reference_gt_diagnostics}"
        )
    if reference_gt_diagnostics["rms_px"] > args.gt_max_rms_px:
        capture.release()
        raise RuntimeError(
            "Reference Grid GT reprojection RMS is too high: "
            f"{reference_gt_diagnostics['rms_px']:.4f}px > "
            f"{args.gt_max_rms_px:.4f}px"
        )

    diagnostic_writer = None
    diagnostic_size = None
    if args.diagnostic_video:
        diagnostic_writer, diagnostic_size = initialize_diagnostic_writer(
            diagnostic_path, reference_sbs.shape, fps
        )

    rows = []
    history = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    for frame_index in range(args.start_frame, args.end_frame + 1):
        ok, sbs = capture.read()
        row = {header: None for header in FRAME_HEADERS}
        row["frame_index"] = frame_index
        if not ok or sbs is None:
            row["status"] = "FRAME_READ_FAILED"
            rows.append(row)
            continue
        current_left, current_right = split_sbs(sbs)
        gray_left = cv2.cvtColor(current_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(current_right, cv2.COLOR_BGR2GRAY)

        current_small_markers = small_detector.detect(gray_left, test_roi)
        current_small_corners = current_small_markers.get(
            args.small_marker_id
        )
        current_candidates = (
            small_marker_pose_candidates(
                current_small_corners,
                args.small_marker_size_mm,
                K_l,
                d_l,
            )
            if current_small_corners is not None
            else []
        )
        features = sift_matcher.match(
            gray_left, test_roi, current_small_corners
        )
        temporal_prediction = (
            predict_temporal(history, frame_index)
            if args.temporal_prior
            else None
        )
        estimate = estimate_test_pose(
            reference_candidates,
            current_candidates,
            reference_small_corners,
            current_small_corners,
            features,
            temporal_prediction,
            small_object,
            K_l,
            d_l,
        )

        current_grid_left = grid_detector.detect(
            gray_left, grid_roi, exclude_roi=test_roi
        )
        current_grid_right = grid_detector.detect(
            gray_right, grid_roi, exclude_roi=test_roi
        )
        gt_current, gt_diagnostics = estimate_grid_gt(
            current_grid_left,
            current_grid_right,
            grid_map,
            K_l,
            d_l,
            K_r,
            d_r,
            left_to_right,
            args.gt_min_markers_per_eye,
        )
        row.update(
            {
                "small_marker_found": current_small_corners is not None,
                "sift_keypoints_reference": features.keypoints_ref,
                "sift_keypoints_current": features.keypoints_cur,
                "sift_mutual_matches": features.mutual_matches,
                "sift_E_inliers": features.E_inliers,
                "temporal_prior_used": temporal_prediction is not None,
                "gt_left_marker_count": gt_diagnostics["left_markers"],
                "gt_right_marker_count": gt_diagnostics["right_markers"],
                "gt_stereo_reprojection_rms_px": gt_diagnostics["rms_px"],
            }
        )

        if estimate is None:
            row["status"] = "TEST_ESTIMATE_FAILED"
        else:
            estimated_pose = estimate["pose"]
            # Causal TEST history is updated without consulting GT.
            history.append((frame_index, estimated_pose))
            history = history[-3:]
            row.update(
                {
                    "estimate_source": estimate["source"],
                    "test_marker_reprojection_rms_px": estimate[
                        "marker_rms_px"
                    ],
                    "sift_candidate_epi_median_px": estimate[
                        "epi_median_px"
                    ],
                    "estimated_rotation_angle_deg": rotation_angle_deg(
                        estimated_pose.R
                    ),
                    "estimated_baseline_mm": float(
                        np.linalg.norm(estimated_pose.t)
                    ),
                }
            )
            if gt_current is None:
                row["status"] = "GT_FAILED"
            elif gt_diagnostics["rms_px"] > args.gt_max_rms_px:
                row["status"] = "GT_REPROJECTION_TOO_HIGH"
            else:
                ground_truth = relative_pose(gt_reference, gt_current)
                errors = temporal_errors(estimated_pose, ground_truth)
                row.update(errors)
                if errors["absolute_baseline_error_percent"] is None:
                    row["status"] = "GT_BASELINE_TOO_SMALL"
                else:
                    row["status"] = "OK"
                    row["rotation_pass"] = (
                        errors["rotation_error_deg"]
                        < args.rotation_threshold_deg
                    )
                    row["baseline_pass"] = (
                        errors["absolute_baseline_error_percent"]
                        < args.baseline_threshold_percent
                    )
                    row["both_pass"] = bool(
                        row["rotation_pass"] and row["baseline_pass"]
                    )
        rows.append(row)

        if diagnostic_writer is not None:
            canvas = sbs.copy()
            eye_width = current_left.shape[1]
            draw_markers(canvas, current_small_markers, (0, 255, 0))
            draw_markers(canvas, current_grid_left, (255, 255, 0))
            draw_markers(
                canvas,
                current_grid_right,
                (255, 255, 0),
                x_offset=eye_width,
            )
            for offset in (0, eye_width):
                draw_roi(canvas, test_roi, (0, 255, 0), offset)
                draw_roi(canvas, grid_roi, (255, 255, 0), offset)
            rotation_text = finite(row.get("rotation_error_deg"))
            baseline_text = finite(
                row.get("absolute_baseline_error_percent")
            )
            cv2.putText(
                canvas,
                f"F{frame_index} {row['status']} rot={rotation_text}deg "
                f"baseline={baseline_text}%",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            diagnostic_writer.write(
                cv2.resize(canvas, diagnostic_size, interpolation=cv2.INTER_AREA)
            )

        processed = frame_index - args.start_frame + 1
        if processed % 10 == 0 or frame_index == args.end_frame:
            comparable_count = sum(
                item.get("status") == "OK" for item in rows
            )
            pass_count = sum(bool(item.get("both_pass")) for item in rows)
            print(
                f"Processed {processed}/{args.end_frame - args.start_frame + 1}: "
                f"comparable={comparable_count}, both_pass={pass_count}"
            )

    capture.release()
    if diagnostic_writer is not None:
        diagnostic_writer.release()

    summary = build_summary(
        rows,
        args.rotation_threshold_deg,
        args.baseline_threshold_percent,
    )
    summary = add_live_summary_formulas(summary, rows)
    settings = [
        {
            "parameter": "rotation_threshold_deg",
            "value": args.rotation_threshold_deg,
        },
        {
            "parameter": "baseline_threshold_percent",
            "value": args.baseline_threshold_percent,
        },
        {"parameter": "video", "value": str(video_path)},
        {"parameter": "calibration", "value": str(calibration_path)},
        {"parameter": "reference_frame", "value": args.reference_frame},
        {
            "parameter": "candidate_frames",
            "value": f"{args.start_frame}..{args.end_frame}",
        },
        {"parameter": "test_roi", "value": str(test_roi)},
        {"parameter": "grid_roi", "value": str(grid_roi)},
        {"parameter": "grid_spec_source", "value": GRID_SPEC_SOURCE},
        {
            "parameter": "small_marker",
            "value": (
                f"{args.small_dictionary}, ID={args.small_marker_id}, "
                f"size={args.small_marker_size_mm} mm"
            ),
        },
        {
            "parameter": "grid",
            "value": (
                f"{args.grid_dictionary}, {args.grid_rows}x{args.grid_cols}, "
                f"IDs={args.grid_start_id}..{grid_end_id}, "
                f"marker={args.grid_marker_size_mm} mm, "
                f"gap={args.grid_gap_mm} mm"
            ),
        },
        {
            "parameter": "TEST_information",
            "value": (
                "left images only: small ID2 + TEST-ROI SIFT + causal "
                "temporal prior"
            ),
        },
        {
            "parameter": "GT_information",
            "value": (
                "synchronized left/right Grid + calibrated stereo extrinsic; "
                "TEST ROI blanked"
            ),
        },
    ]
    protocol = [
        {
            "item": "Required rigidity",
            "detail": (
                "The Grid board and wound phantom must remain rigidly fixed "
                "relative to each other throughout the video."
            ),
        },
        {
            "item": "No GT leakage",
            "detail": (
                "TEST uses only left TEST ROI. Grid detections and right images "
                "are used only after TEST estimation."
            ),
        },
        {
            "item": "Small/Grid ID isolation",
            "detail": (
                (
                    f"ID conflict exists: small ID {args.small_marker_id} lies "
                    f"inside Grid IDs {args.grid_start_id}..{grid_end_id}. "
                    "TEST ROI is blanked before Grid detection."
                )
                if small_id_conflicts_with_grid
                else (
                    f"No ID conflict: small ID {args.small_marker_id}; Grid "
                    f"IDs {args.grid_start_id}..{grid_end_id}. TEST ROI is "
                    "still blanked before Grid detection for strict separation."
                )
            ),
        },
        {
            "item": "RT direction",
            "detail": (
                "Reference-left camera coordinates to candidate-left camera "
                "coordinates: P_candidate * inverse(P_reference)."
            ),
        },
        {
            "item": "Pass rule",
            "detail": (
                "rotation error < 'Settings'!B2 AND absolute baseline error "
                "percent < 'Settings'!B3. Summary formulas recalculate in Excel."
            ),
        },
        {
            "item": "Near-zero motion",
            "detail": (
                f"Baseline percent is omitted when GT baseline is below "
                f"{MIN_GT_BASELINE_FOR_PERCENT_MM} mm."
            ),
        },
    ]
    chart_helper_rows = build_chart_helper_rows(
        rows,
        args.rotation_threshold_deg,
        args.baseline_threshold_percent,
    )
    write_csv(output_csv, rows)
    write_xlsx(
        output_xlsx,
        [
            ("Settings", ["parameter", "value"], settings),
            ("Summary", list(summary.keys()), [summary]),
            ("Frame Results", FRAME_HEADERS, rows),
            ("Charts", CHART_HELPER_HEADERS, chart_helper_rows),
            ("Protocol", ["item", "detail"], protocol),
        ],
    )
    add_frame_charts_to_xlsx(
        output_xlsx,
        rows,
        args.rotation_threshold_deg,
        args.baseline_threshold_percent,
        charts_sheet_index=4,
    )

    comparable_count = sum(row.get("status") == "OK" for row in rows)
    both_pass_count = sum(bool(row.get("both_pass")) for row in rows)
    pass_percent = (
        both_pass_count / comparable_count * 100.0
        if comparable_count
        else 0.0
    )
    print("=" * 68)
    print(f"Comparable frames: {comparable_count}/{len(rows)}")
    print(
        f"Both pass: {both_pass_count}/{comparable_count} "
        f"({pass_percent:.2f}%)"
    )
    print(f"Excel: {output_xlsx}")
    print(f"CSV:   {output_csv}")
    if args.diagnostic_video:
        print(f"Video: {diagnostic_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
