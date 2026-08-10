"""Stage-1 HBVCAM experiment: compare marker-only RT with SIFT-assisted RT.

The expected file names are::

    HBVCAM_<distance>CM_opposite_<repeat>[_<angle>].mp4

Only repeats 1..6 are included.  Repeats 1..5 are labelled ``object_in_roi``
and repeat 6 is labelled ``pattern_only``.  Every requested frame is kept;
JSON extrinsics are used only after an RT estimate has been produced.

The script has no Excel-package dependency.  It writes checkpoint CSV while it
runs and builds a multi-sheet .xlsx file with Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import cv2
import numpy as np

import depth_measure_multi_aruco_sbs_camera_v7_demo_HBVCAM_v11_ROI as hbvcam
from Algorithm.camera_preprocess import normalized_roi_bounds
from Algorithm.stereo_json_frame_selection import (
    build_common_intrinsic_maps,
    calculate_json_errors,
)


VIDEO_PATTERN = re.compile(
    r"^HBVCAM_(?P<distance>\d+(?:\.\d+)?)CM_opposite_"
    r"(?P<repeat>[1-6])(?:_(?P<angle>-?\d+(?:\.\d+)?))?"
    r"\.(?:mp4|avi|mkv|mov|m4v)$",
    re.IGNORECASE,
)
DEFAULT_FOLDER = Path(__file__).resolve().parent / "HBVCAM_4M2214HD-2-v11"
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent
    / "calibration_result_HBVCAM_4M2214HD-2-v11.json"
)

# Stage-1 recordings were made by moving the camera vertically without keeping
# all scene objects inside the later asymmetric ROIs.  Use every pixel from each
# 1920x1080 eye for both ArUco and SIFT so the experiment matches the captured
# content.  These values are local to this batch script; the interactive HBVCAM
# ROI program keeps its own settings unchanged.
STAGE1_LEFT_ROI_RATIO = (0.0, 0.0, 1.0, 1.0)
STAGE1_RIGHT_ROI_RATIO = (0.0, 0.0, 1.0, 1.0)

RT_PASS_ROTATION_ERROR_DEG = 2.0
RT_PASS_BASELINE_ERROR_PERCENT = 5.0
ARUCO_DIAGNOSTIC_TILE_SIZE = 320
ARUCO_DIAGNOSTIC_PADDING_RATIO = 0.5
ARUCO_SUBPIX_MIN_HALF_WINDOW = 2
ARUCO_SUBPIX_MAX_HALF_WINDOW = 4
ARUCO_SUBPIX_STABILITY_DELTA = 1
ARUCO_SUBPIX_STABILITY_MAX_RAW_PX = 2.0


@dataclass(frozen=True)
class ExcelFormula:
    text: str
    cached_value: float | int = 0

MODES = ("marker_only", "sift_assisted")
RESULT_FIELDS = [
    "distance_cm",
    "repeat",
    "scene_type",
    "video_file",
    "video_path",
    "frame_index",
    "mode",
    "status",
    "failure_reason",
    "rotation_error_deg",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "algorithm_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_mm",
    "absolute_baseline_delta_mm",
    "baseline_error_percent",
    "absolute_baseline_error_percent",
    "shared_marker_count",
    "marker_id",
    "marker_side_left_px",
    "marker_side_right_px",
    "marker_area_left_percent",
    "marker_area_right_percent",
    "ippe_branch_count_left",
    "ippe_branch_count_right",
    "ippe_branch_left",
    "ippe_branch_right",
    "marker_self_reproj_left_px",
    "marker_self_reproj_right_px",
    "marker_bidirectional_rms_px",
    "marker_bidirectional_max_px",
    "feature_match_count",
    "feature_inlier_count",
    "feature_inlier_ratio",
    "feature_median_px",
    "feature_p90_px",
    "feature_grid_coverage",
    "feature_hull_coverage",
    "feature_parallax_deg",
    "rt_sift_applied",
    "rt_sift_role",
    "rt_reliable",
    "processing_time_s",
]

NUMERIC_FIELDS = {
    name
    for name in RESULT_FIELDS
    if name
    not in {
        "scene_type",
        "video_file",
        "video_path",
        "mode",
        "status",
        "failure_reason",
        "rt_sift_role",
        "rt_sift_applied",
        "rt_reliable",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the first N frames of the Stage-1 HBVCAM videos and "
            "compare pure ArUco/IPPE RT against the current SIFT-assisted RT."
        )
    )
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--mode", choices=("both", *MODES), default="both",
        help="Default: run both marker-only and SIFT-assisted estimates.",
    )
    parser.add_argument(
        "--output-prefix",
        help=(
            "Path without extension. Default: "
            "<folder>/HBVCAM_stage1_first300_fullframe_sift_comparison"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an existing *_frame_results.csv checkpoint.",
    )
    parser.add_argument(
        "--only-distance", type=float,
        help="Analyze one distance only; useful for a shorter pilot run.",
    )
    parser.add_argument(
        "--max-videos", type=int,
        help="Limit the number of videos after sorting; intended for testing.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument(
        "--no-aruco-corner-videos",
        action="store_true",
        help="Do not generate the 640x320 left/right ArUco corner diagnostic videos.",
    )
    parser.add_argument(
        "--aruco-corner-video-dir",
        help=(
            "Diagnostic-video output folder. Default: "
            "<output-prefix>_aruco_corner_videos"
        ),
    )
    parser.add_argument(
        "--aruco-crop-padding-ratio",
        type=float,
        default=ARUCO_DIAGNOSTIC_PADDING_RATIO,
        help=(
            "Extra square crop padding on each side relative to the detected "
            "marker size. Default: 0.5."
        ),
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    if args.aruco_crop_padding_ratio < 0:
        parser.error("--aruco-crop-padding-ratio must be non-negative")
    return args


def collect_stage1_videos(folder: Path, only_distance: float | None) -> list[dict]:
    videos = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = VIDEO_PATTERN.match(path.name)
        if match is None:
            continue
        distance = float(match.group("distance"))
        repeat = int(match.group("repeat"))
        angle_text = match.group("angle")
        angle_deg = float(angle_text) if angle_text is not None else 0.0
        if only_distance is not None and not math.isclose(
            distance, only_distance, rel_tol=0.0, abs_tol=1e-6
        ):
            continue
        videos.append(
            {
                "path": path.resolve(),
                "distance_cm": distance,
                "angle_deg": angle_deg,
                "repeat": repeat,
                "scene_type": "pattern_only" if repeat == 6 else "object_in_roi",
            }
        )
    videos.sort(
        key=lambda item: (
            item["distance_cm"],
            item["angle_deg"],
            item["repeat"],
        )
    )
    return videos


def validate_video_set(videos: list[dict], limited: bool) -> None:
    if not videos:
        raise FileNotFoundError("No Stage-1 videos matched the required file name pattern.")
    if limited:
        return
    grouped: dict[float, set[int]] = defaultdict(set)
    for video in videos:
        grouped[video["distance_cm"]].add(video["repeat"])
    warnings = []
    for distance, repeats in sorted(grouped.items()):
        missing = sorted(set(range(1, 7)) - repeats)
        if missing:
            warnings.append(f"{distance:g} cm missing repeats {missing}")
    if len(grouped) != 5:
        warnings.append(f"found {len(grouped)} distances instead of the expected 5")
    if warnings:
        print("WARNING: " + "; ".join(warnings))


def empty_result(video: dict, frame_index: int, mode: str) -> dict:
    row = {field: None for field in RESULT_FIELDS}
    row.update(
        {
            "distance_cm": video["distance_cm"],
            "repeat": video["repeat"],
            "scene_type": video["scene_type"],
            "video_file": video["path"].name,
            "video_path": str(video["path"]),
            "frame_index": frame_index,
            "mode": mode,
            "status": "FAILED",
            "failure_reason": "",
        }
    )
    return row


def finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def marker_geometry(corners: np.ndarray, image_area: float) -> tuple[float, float]:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    side_px = float(np.mean(sides))
    area_percent = abs(float(cv2.contourArea(points.astype(np.float32)))) / image_area * 100.0
    return side_px, area_percent


def square_crop_with_padding(
    image: np.ndarray,
    corners: np.ndarray,
    padding_ratio: float,
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Crop a square around four marker corners, padding outside the image if needed."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    marker_span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 8.0)
    crop_side = max(int(math.ceil(marker_span * (1.0 + 2.0 * padding_ratio))), 24)
    x0 = int(math.floor(center[0] - crop_side / 2.0))
    y0 = int(math.floor(center[1] - crop_side / 2.0))
    x1 = x0 + crop_side
    y1 = y0 + crop_side

    output = np.zeros((crop_side, crop_side, 3), dtype=np.uint8)
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, image.shape[1])
    src_y1 = min(y1, image.shape[0])
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        output[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = image[src_y0:src_y1, src_x0:src_x1]
    return output, (x0, y0), crop_side


def aruco_diagnostic_tile(
    frame: np.ndarray,
    markers: dict[int, np.ndarray],
    preferred_marker_id: int | None,
    camera_label: str,
    frame_index: int,
    padding_ratio: float,
    shared_marker: bool,
    no_marker_reason: str | None = None,
) -> np.ndarray:
    """Draw subpixel corner markers, crop the marker area, and resize to 320x320."""
    tile_size = ARUCO_DIAGNOSTIC_TILE_SIZE
    marker_id = preferred_marker_id if preferred_marker_id in markers else None
    if marker_id is None and markers:
        marker_id = min(markers)
    if marker_id is None:
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(
            tile,
            f"{camera_label} F{frame_index}: {no_marker_reason or 'NO ARUCO'}",
            (12, tile_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return tile

    # Crop and enlarge the untouched image first.  Drawing before resize makes
    # circles and lines grow with the crop and hide the subpixel edge location.
    points = np.asarray(markers[marker_id], dtype=np.float32).reshape(4, 2)
    crop, (crop_x0, crop_y0), crop_side = square_crop_with_padding(
        frame,
        points,
        padding_ratio,
    )
    interpolation = cv2.INTER_CUBIC if crop.shape[0] < tile_size else cv2.INTER_AREA
    tile = cv2.resize(crop, (tile_size, tile_size), interpolation=interpolation)
    scale = tile_size / float(crop_side)
    tile_points = (
        points - np.array([crop_x0, crop_y0], dtype=np.float32)
    ) * scale
    for corner_index, point in enumerate(tile_points):
        px, py = np.rint(point).astype(int)
        cv2.drawMarker(
            tile,
            (px, py),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(tile, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)
        label_x = min(max(px + 5, 1), tile_size - 12)
        label_y = min(max(py - 5, 12), tile_size - 2)
        cv2.putText(
            tile,
            str(corner_index),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    status = "SHARED" if shared_marker else "NO SHARED"
    label = f"{camera_label} ID:{marker_id} F:{frame_index} {status}"
    cv2.rectangle(tile, (0, 0), (tile_size - 1, 24), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def aruco_diagnostic_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_markers: dict[int, np.ndarray],
    right_markers: dict[int, np.ndarray],
    shared_marker_id: int | None,
    frame_index: int,
    padding_ratio: float,
    left_no_marker_reason: str | None = None,
    right_no_marker_reason: str | None = None,
) -> np.ndarray:
    shared = shared_marker_id is not None
    left_tile = aruco_diagnostic_tile(
        left_frame,
        left_markers,
        shared_marker_id,
        "LEFT",
        frame_index,
        padding_ratio,
        shared,
        left_no_marker_reason,
    )
    right_tile = aruco_diagnostic_tile(
        right_frame,
        right_markers,
        shared_marker_id,
        "RIGHT",
        frame_index,
        padding_ratio,
        shared,
        right_no_marker_reason,
    )
    combined = np.hstack((left_tile, right_tile))
    cv2.line(
        combined,
        (ARUCO_DIAGNOSTIC_TILE_SIZE, 0),
        (ARUCO_DIAGNOSTIC_TILE_SIZE, ARUCO_DIAGNOSTIC_TILE_SIZE - 1),
        (255, 255, 255),
        2,
    )
    return combined


def open_aruco_diagnostic_writer(path: Path, source_capture) -> tuple[object, float]:
    fps = finite_or_none(source_capture.get(cv2.CAP_PROP_FPS))
    if fps is None or fps <= 0.0 or fps > 240.0:
        fps = 30.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ARUCO_DIAGNOSTIC_TILE_SIZE * 2, ARUCO_DIAGNOSTIC_TILE_SIZE),
    )
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Unable to create ArUco diagnostic MP4: {path}")
    return writer, fps


class MarkerOnlyEstimator:
    """Single-pattern IPPE estimator with no SIFT and no JSON-based decision."""

    def __init__(self, camera_matrix: np.ndarray, marker_size_mm: float, image_size):
        self.K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.marker_size_mm = float(marker_size_mm)
        self.width, self.height = map(int, image_size)
        half = self.marker_size_mm / 2.0
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = (
            cv2.aruco.ArucoDetector(dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.dictionary = dictionary
        self.parameters = parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_detection_diagnostics = {}
        self.left_roi = normalized_roi_bounds(
            self.width,
            self.height,
            *STAGE1_LEFT_ROI_RATIO,
        )
        self.right_roi = normalized_roi_bounds(
            self.width,
            self.height,
            *STAGE1_RIGHT_ROI_RATIO,
        )

    def _adaptive_subpixel_refine(
        self,
        gray: np.ndarray,
        initial_corners: np.ndarray,
    ) -> tuple[np.ndarray | None, dict]:
        """Refine on non-CLAHE pixels and reject window-sensitive corners."""
        initial = np.asarray(initial_corners, dtype=np.float32).reshape(4, 1, 2)
        points = initial.reshape(4, 2)
        side_px = float(
            np.mean(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))
        )
        half_window = int(
            np.clip(
                round(side_px / 12.0),
                ARUCO_SUBPIX_MIN_HALF_WINDOW,
                ARUCO_SUBPIX_MAX_HALF_WINDOW,
            )
        )
        candidate_windows = sorted(
            {
                max(1, half_window - ARUCO_SUBPIX_STABILITY_DELTA),
                half_window,
                min(
                    ARUCO_SUBPIX_MAX_HALF_WINDOW + ARUCO_SUBPIX_STABILITY_DELTA,
                    half_window + ARUCO_SUBPIX_STABILITY_DELTA,
                ),
            }
        )
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        solutions = {}
        for window in candidate_windows:
            refined = initial.copy()
            try:
                cv2.cornerSubPix(
                    gray,
                    refined,
                    (window, window),
                    (-1, -1),
                    term,
                )
            except cv2.error:
                continue
            if np.all(np.isfinite(refined)):
                solutions[window] = refined.reshape(4, 2)

        diagnostics = {
            "side_px": side_px,
            "half_window": half_window,
            "candidate_windows": candidate_windows,
            "stability_max_raw_px": None,
            "accepted": False,
        }
        if half_window not in solutions or len(solutions) < 2:
            return None, diagnostics

        maximum_disagreement = 0.0
        solution_values = list(solutions.values())
        for first_index in range(len(solution_values)):
            for second_index in range(first_index + 1, len(solution_values)):
                maximum_disagreement = max(
                    maximum_disagreement,
                    float(
                        np.max(
                            np.linalg.norm(
                                solution_values[first_index]
                                - solution_values[second_index],
                                axis=1,
                            )
                        )
                    ),
                )
        diagnostics["stability_max_raw_px"] = maximum_disagreement
        if maximum_disagreement > ARUCO_SUBPIX_STABILITY_MAX_RAW_PX:
            return None, diagnostics
        diagnostics["accepted"] = True
        return solutions[half_window], diagnostics

    def detect(
        self,
        frame: np.ndarray,
        roi_bounds,
        source_camera_matrix: np.ndarray | None = None,
        source_distortion: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = roi_bounds
        original_crop = original_gray[y0:y1, x0:x1]
        # CLAHE is used only to make initial marker detection more robust.  The
        # subpixel optimizer below sees the untouched grayscale gradients.
        detection_crop = self.clahe.apply(original_crop)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(detection_crop)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                detection_crop, self.dictionary, parameters=self.parameters
            )
        diagnostics = {
            "initial_marker_count": 0 if ids is None else int(len(ids)),
            "accepted_marker_count": 0,
            "rejected": [],
        }
        if ids is None or not len(ids):
            self.last_detection_diagnostics = diagnostics
            return {}
        result = {}
        offset = np.array([x0, y0], dtype=np.float32)
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            refined, marker_diagnostics = self._adaptive_subpixel_refine(
                original_crop,
                marker_corners,
            )
            marker_diagnostics["marker_id"] = int(marker_id)
            if refined is None:
                diagnostics["rejected"].append(marker_diagnostics)
                continue
            points_source = refined + offset
            if source_camera_matrix is not None and source_distortion is not None:
                points_common = cv2.undistortPoints(
                    points_source.reshape(-1, 1, 2).astype(np.float32),
                    np.asarray(source_camera_matrix, dtype=np.float64),
                    np.asarray(source_distortion, dtype=np.float64),
                    P=self.K,
                ).reshape(4, 2)
            else:
                points_common = points_source
            result[int(marker_id)] = points_common.astype(np.float32)
            diagnostics["accepted_marker_count"] += 1
        self.last_detection_diagnostics = diagnostics
        return result

    def pose_branches(self, corners: np.ndarray) -> list[dict]:
        image_points = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2)
        try:
            _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
                self.object_points,
                image_points,
                self.K,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return []
        branches = []
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            points_camera = (
                rotation @ self.object_points.astype(np.float64).T + translation
            ).T
            if np.any(points_camera[:, 2] <= 0.0):
                continue
            projected, _ = cv2.projectPoints(
                self.object_points,
                rvec,
                translation,
                self.K,
                None,
            )
            residuals = np.linalg.norm(
                projected.reshape(4, 2) - image_points.reshape(4, 2), axis=1
            )
            branches.append(
                {
                    "index": index,
                    "R": rotation,
                    "t": translation,
                    "rms": float(np.sqrt(np.mean(residuals**2))),
                    "max": float(np.max(residuals)),
                }
            )
        return branches

    def pair_from_markers(
        self,
        left_markers: dict[int, np.ndarray],
        right_markers: dict[int, np.ndarray],
    ) -> dict:
        shared = sorted(set(left_markers) & set(right_markers))
        if not shared:
            raise RuntimeError("No shared ArUco marker was detected inside both ROIs")
        marker_id = min(shared)
        left_corners = left_markers[marker_id]
        right_corners = right_markers[marker_id]
        image_area = float(max(self.width * self.height, 1))
        left_side, left_area = marker_geometry(left_corners, image_area)
        right_side, right_area = marker_geometry(right_corners, image_area)
        return {
            "shared_marker_count": len(shared),
            "marker_id": marker_id,
            "left_corners": left_corners,
            "right_corners": right_corners,
            "marker_side_left_px": left_side,
            "marker_side_right_px": right_side,
            "marker_area_left_percent": left_area,
            "marker_area_right_percent": right_area,
        }

    def detect_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> dict:
        left_markers = self.detect(left_frame, self.left_roi)
        right_markers = self.detect(right_frame, self.right_roi)
        return self.pair_from_markers(left_markers, right_markers)

    def estimate(self, detected: dict) -> dict:
        left_branches = self.pose_branches(detected["left_corners"])
        right_branches = self.pose_branches(detected["right_corners"])
        if not left_branches or not right_branches:
            raise RuntimeError("IPPE did not return a positive-depth pose in both views")

        # A single planar marker has an inherent two-branch IPPE ambiguity.  Without
        # SIFT there is no independent scene constraint, so choose only by the two
        # marker fits plus the same broad physical baseline gate as the main method.
        candidates = []
        for left in left_branches:
            for right in right_branches:
                R_rel = right["R"] @ left["R"].T
                t_rel = right["t"] - R_rel @ left["t"]
                baseline = float(np.linalg.norm(t_rel))
                if not (hbvcam.MIN_BASELINE_MM <= baseline <= hbvcam.MAX_BASELINE_MM):
                    continue
                combined_rms = float(
                    math.sqrt(0.5 * (left["rms"] ** 2 + right["rms"] ** 2))
                )
                combined_max = max(left["max"], right["max"])
                candidates.append(
                    (combined_rms, combined_max, left, right, R_rel, t_rel, baseline)
                )
        if not candidates:
            raise RuntimeError(
                "Every marker-only IPPE branch pair failed the physical baseline gate"
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        rms, maximum, left, right, R_rel, t_rel, baseline = candidates[0]
        result = dict(detected)
        result.update(
            {
                "R_rel": R_rel,
                "t_rel": t_rel,
                "baseline": baseline,
                "ippe_branch_count_left": len(left_branches),
                "ippe_branch_count_right": len(right_branches),
                "ippe_branch_left": left["index"],
                "ippe_branch_right": right["index"],
                "marker_self_reproj_left_px": left["rms"],
                "marker_self_reproj_right_px": right["rms"],
                "marker_bidirectional_rms_px": rms,
                "marker_bidirectional_max_px": maximum,
            }
        )
        return result


def add_answer_errors(row: dict, R_est, t_est, answer_extrinsic: dict) -> None:
    errors = calculate_json_errors(R_est, t_est, answer_extrinsic)
    row.update({key: finite_or_none(value) for key, value in errors.items()})
    json_baseline = row.get("json_baseline_mm")
    delta = row.get("baseline_delta_mm")
    if json_baseline and delta is not None:
        row["baseline_error_percent"] = delta / json_baseline * 100.0
        row["absolute_baseline_error_percent"] = abs(delta) / json_baseline * 100.0


def marker_only_row(
    video: dict,
    frame_index: int,
    estimator: MarkerOnlyEstimator,
    detected: dict,
    answer_extrinsic: dict,
) -> dict:
    started = time.perf_counter()
    row = empty_result(video, frame_index, "marker_only")
    try:
        estimate = estimator.estimate(detected)
        for field in (
            "shared_marker_count",
            "marker_id",
            "marker_side_left_px",
            "marker_side_right_px",
            "marker_area_left_percent",
            "marker_area_right_percent",
            "ippe_branch_count_left",
            "ippe_branch_count_right",
            "ippe_branch_left",
            "ippe_branch_right",
            "marker_self_reproj_left_px",
            "marker_self_reproj_right_px",
            "marker_bidirectional_rms_px",
            "marker_bidirectional_max_px",
        ):
            row[field] = estimate.get(field)
        add_answer_errors(row, estimate["R_rel"], estimate["t_rel"], answer_extrinsic)
        row["status"] = "OK"
        row["rt_reliable"] = None
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    row["processing_time_s"] = time.perf_counter() - started
    return row


def sift_assisted_row(
    video: dict,
    frame_index: int,
    left_common: np.ndarray,
    right_common: np.ndarray,
    common_k: np.ndarray,
    detected: dict,
    left_markers: dict[int, np.ndarray],
    right_markers: dict[int, np.ndarray],
    answer_extrinsic: dict,
) -> dict:
    started = time.perf_counter()
    row = empty_result(video, frame_index, "sift_assisted")
    for field in (
        "shared_marker_count",
        "marker_id",
        "marker_side_left_px",
        "marker_side_right_px",
        "marker_area_left_percent",
        "marker_area_right_percent",
    ):
        row[field] = detected.get(field)
    try:
        video_data = hbvcam.analyze_video_frames(
            str(video["path"]),
            1,
            1,
            common_k,
            np.zeros(5, dtype=np.float64),
            common_k,
            hbvcam.ACTUAL_MARKER_SIZE_MM,
            hbvcam.POSE_SELECT_MODE,
            hbvcam.FRAME_RANGE_MODE,
            frames_override=[right_common, left_common],
            marker_corners_override=[right_markers, left_markers],
            analysis_log_fn=lambda _message: None,
        )
        if not video_data:
            raise RuntimeError("The current SIFT-assisted RT pipeline returned no result")
        add_answer_errors(
            row, video_data["R_rel"], video_data["t_rel"], answer_extrinsic
        )
        quality = video_data.get("rt_quality") or {}
        marker_stats = video_data.get("marker_bidir_stats") or {}
        feature_stats = quality.get("final_feature_stats") or {}
        branch = quality.get("branch")
        if branch is not None and len(branch) == 2:
            # Main analyzer: frame_A is right, frame_B is left.
            row["ippe_branch_right"] = branch[0]
            row["ippe_branch_left"] = branch[1]
        row["marker_self_reproj_left_px"] = finite_or_none(
            video_data.get("marker_pnp_self_reproj_err")
        )
        row["marker_bidirectional_rms_px"] = finite_or_none(
            marker_stats.get("rms_px", video_data.get("marker_reproj_err"))
        )
        row["marker_bidirectional_max_px"] = finite_or_none(
            marker_stats.get("max_px")
        )
        row["feature_match_count"] = video_data.get("rt_sift_match_count")
        row["feature_inlier_count"] = video_data.get("rt_sift_inlier_count")
        row["feature_inlier_ratio"] = finite_or_none(
            feature_stats.get("inlier_ratio")
        )
        if row["feature_inlier_ratio"] is None:
            matches = row["feature_match_count"] or 0
            inliers = row["feature_inlier_count"] or 0
            row["feature_inlier_ratio"] = inliers / matches if matches else None
        row["feature_median_px"] = finite_or_none(
            feature_stats.get("inlier_median_px", quality.get("final_feature_epi_px"))
        )
        row["feature_p90_px"] = finite_or_none(feature_stats.get("inlier_p90_px"))
        row["feature_grid_coverage"] = finite_or_none(
            quality.get("feature_grid_coverage")
        )
        row["feature_hull_coverage"] = finite_or_none(
            quality.get("feature_hull_coverage")
        )
        row["feature_parallax_deg"] = finite_or_none(
            quality.get("feature_parallax_deg")
        )
        row["rt_sift_applied"] = bool(video_data.get("rt_sift_applied", False))
        row["rt_sift_role"] = video_data.get("rt_sift_role")
        row["rt_reliable"] = bool(quality.get("rt_reliable", False))
        row["status"] = "OK" if row["rt_reliable"] else "QUALITY_WARNING"
        if not row["rt_reliable"]:
            row["failure_reason"] = "RT was produced, but the current quality flag is false"
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    row["processing_time_s"] = time.perf_counter() - started
    return row


def normalize_csv_row(row: dict) -> dict:
    converted = dict(row)
    for name in NUMERIC_FIELDS:
        value = converted.get(name)
        if value in (None, ""):
            converted[name] = None
            continue
        number = finite_or_none(value)
        if name in {
            "repeat",
            "frame_index",
            "shared_marker_count",
            "marker_id",
            "ippe_branch_count_left",
            "ippe_branch_count_right",
            "ippe_branch_left",
            "ippe_branch_right",
            "feature_match_count",
            "feature_inlier_count",
        } and number is not None:
            converted[name] = int(round(number))
        else:
            converted[name] = number
    for name in ("rt_sift_applied", "rt_reliable"):
        value = converted.get(name)
        if value in (None, ""):
            converted[name] = None
        elif isinstance(value, bool):
            pass
        else:
            converted[name] = str(value).strip().lower() in {"1", "true", "yes"}
    return converted


def load_checkpoint(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(RESULT_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Checkpoint schema is missing columns: {sorted(missing)}")
        return [normalize_csv_row(row) for row in reader]


def append_csv_row(file_obj, writer: csv.DictWriter, row: dict) -> None:
    writer.writerow({field: row.get(field) for field in RESULT_FIELDS})
    file_obj.flush()


def as_float(row: dict, name: str):
    return finite_or_none(row.get(name))


def percentile(values: Iterable[float], q: float):
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else None


SUMMARY_METRICS = [
    "rotation_error_deg",
    "translation_l2_error_mm",
    "translation_direction_error_deg",
    "absolute_baseline_delta_mm",
    "absolute_baseline_error_percent",
]


def summarize_groups(rows: list[dict], group_fields: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = dict(zip(group_fields, key))
        solved = [row for row in group if as_float(row, "algorithm_baseline_mm") is not None]
        summary.update(
            {
                "total_frames": len(group),
                "solved_frames": len(solved),
                "success_rate_percent": len(solved) / len(group) * 100.0,
                "quality_ok_frames": sum(row.get("status") == "OK" for row in group),
                "quality_warning_frames": sum(
                    row.get("status") == "QUALITY_WARNING" for row in group
                ),
                "failed_frames": sum(row.get("status") == "FAILED" for row in group),
                "feature_supported_frames": sum(
                    (as_float(row, "feature_match_count") or 0) > 0 for row in group
                ),
                "sift_joint_refined_frames": sum(
                    row.get("rt_sift_applied") is True for row in group
                ),
                "rt_reliable_frames": sum(
                    row.get("rt_reliable") is True for row in group
                ),
            }
        )
        for metric in SUMMARY_METRICS:
            values = [as_float(row, metric) for row in solved]
            values = [value for value in values if value is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_median"] = float(np.median(values)) if values else None
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) >= 2 else None
            )
            summary[f"{metric}_p95"] = percentile(values, 95)
        output.append(summary)
    return output


def video_angle_deg(video_file: str | None) -> float:
    match = VIDEO_PATTERN.match(str(video_file or ""))
    if match is None or match.group("angle") is None:
        return 0.0
    return float(match.group("angle"))


def passes_rt_error_threshold(row: dict) -> bool:
    rotation_error = as_float(row, "rotation_error_deg")
    baseline_error_percent = as_float(row, "absolute_baseline_error_percent")
    return (
        rotation_error is not None
        and baseline_error_percent is not None
        and rotation_error < RT_PASS_ROTATION_ERROR_DEG
        and baseline_error_percent < RT_PASS_BASELINE_ERROR_PERCENT
    )


def pass_counts(rows: list[dict]) -> tuple[int, int, int]:
    rotation_passed = 0
    baseline_passed = 0
    both_passed = 0
    for row in rows:
        rotation_error = as_float(row, "rotation_error_deg")
        baseline_error_percent = as_float(row, "absolute_baseline_error_percent")
        rotation_ok = (
            rotation_error is not None
            and rotation_error < RT_PASS_ROTATION_ERROR_DEG
        )
        baseline_ok = (
            baseline_error_percent is not None
            and baseline_error_percent < RT_PASS_BASELINE_ERROR_PERCENT
        )
        rotation_passed += int(rotation_ok)
        baseline_passed += int(baseline_ok)
        both_passed += int(rotation_ok and baseline_ok)
    return rotation_passed, baseline_passed, both_passed


def pass_rate_record(group: list[dict], metadata: dict) -> dict:
    mode_rows: dict[str, dict[tuple, dict]] = defaultdict(dict)
    requested_frames = set()
    for row in group:
        frame_key = (row.get("video_file"), int(row.get("frame_index") or 0))
        requested_frames.add(frame_key)
        mode_rows[str(row.get("mode"))][frame_key] = row

    marker_rows = list(mode_rows.get("marker_only", {}).values())
    sift_rows = list(mode_rows.get("sift_assisted", {}).values())
    marker_rotation_passed, marker_baseline_passed, marker_passed = pass_counts(
        marker_rows
    )
    sift_rotation_passed, sift_baseline_passed, sift_passed = pass_counts(sift_rows)
    sift_joint_refined = sum(row.get("rt_sift_applied") is True for row in sift_rows)
    sift_joint_passed = sum(
        row.get("rt_sift_applied") is True and passes_rt_error_threshold(row)
        for row in sift_rows
    )

    record = dict(metadata)
    record.update(
        {
            "requested_frames": len(requested_frames),
            "marker_total_frames": len(marker_rows),
            "marker_rotation_pass_frames": marker_rotation_passed,
            "marker_rotation_pass_percent": (
                marker_rotation_passed / len(marker_rows) * 100.0
                if marker_rows else None
            ),
            "marker_baseline_pass_frames": marker_baseline_passed,
            "marker_baseline_pass_percent": (
                marker_baseline_passed / len(marker_rows) * 100.0
                if marker_rows else None
            ),
            "marker_qualified_frames": marker_passed,
            "marker_qualified_percent": (
                marker_passed / len(marker_rows) * 100.0 if marker_rows else None
            ),
            "sift_total_frames": len(sift_rows),
            "sift_rotation_pass_frames": sift_rotation_passed,
            "sift_rotation_pass_percent": (
                sift_rotation_passed / len(sift_rows) * 100.0
                if sift_rows else None
            ),
            "sift_baseline_pass_frames": sift_baseline_passed,
            "sift_baseline_pass_percent": (
                sift_baseline_passed / len(sift_rows) * 100.0
                if sift_rows else None
            ),
            "sift_qualified_frames": sift_passed,
            "sift_qualified_percent": (
                sift_passed / len(sift_rows) * 100.0 if sift_rows else None
            ),
            "sift_joint_refined_frames": sift_joint_refined,
            "sift_joint_refined_percent": (
                sift_joint_refined / len(sift_rows) * 100.0 if sift_rows else None
            ),
            "sift_joint_refined_qualified_frames": sift_joint_passed,
            "sift_joint_refined_qualified_percent": (
                sift_joint_passed / len(sift_rows) * 100.0 if sift_rows else None
            ),
            "rotation_error_threshold_deg": RT_PASS_ROTATION_ERROR_DEG,
            "baseline_error_threshold_percent": RT_PASS_BASELINE_ERROR_PERCENT,
        }
    )
    return record


def pass_rate_summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    distance_groups: dict[tuple, list[dict]] = defaultdict(list)
    video_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        distance = as_float(row, "distance_cm")
        angle = video_angle_deg(row.get("video_file"))
        distance_groups[(distance, angle)].append(row)
        video_groups[
            (
                distance,
                angle,
                int(as_float(row, "repeat") or 0),
                row.get("scene_type"),
                row.get("video_file"),
            )
        ].append(row)

    distance_rows = [
        pass_rate_record(
            group,
            {"distance_cm": distance, "angle_deg": angle},
        )
        for (distance, angle), group in sorted(
            distance_groups.items(),
            key=lambda item: (
                float("inf") if item[0][0] is None else item[0][0],
                item[0][1],
            ),
        )
    ]
    video_rows = [
        pass_rate_record(
            group,
            {
                "distance_cm": distance,
                "angle_deg": angle,
                "repeat": repeat,
                "scene_type": scene_type,
                "video_file": video_file,
            },
        )
        for (distance, angle, repeat, scene_type, video_file), group in sorted(
            video_groups.items(),
            key=lambda item: (
                float("inf") if item[0][0] is None else item[0][0],
                item[0][1],
                item[0][2],
                str(item[0][4]),
            ),
        )
    ]
    return distance_rows, video_rows


def sift_effect_rows(rows: list[dict]) -> list[dict]:
    paired = defaultdict(dict)
    for row in rows:
        key = (row.get("distance_cm"), row.get("repeat"), row.get("video_file"), row.get("frame_index"))
        paired[key][row.get("mode")] = row

    records = []
    for (distance, _repeat, _video, _frame), pair in paired.items():
        if "marker_only" not in pair or "sift_assisted" not in pair:
            continue
        records.append(
            {
                "distance_cm": distance,
                "scene_type": pair["marker_only"].get("scene_type"),
                "marker": pair["marker_only"],
                "sift": pair["sift_assisted"],
            }
        )

    group_specs = []
    distances = sorted({record["distance_cm"] for record in records})
    scenes = ("object_in_roi", "pattern_only")
    for distance in distances:
        for scene in scenes:
            group_specs.append((distance, scene))
        group_specs.append((distance, "ALL"))
    for scene in scenes:
        group_specs.append(("ALL", scene))
    group_specs.append(("ALL", "ALL"))

    output = []
    for distance, scene in group_specs:
        group = [
            record
            for record in records
            if (distance == "ALL" or record["distance_cm"] == distance)
            and (scene == "ALL" or record["scene_type"] == scene)
        ]
        if not group:
            continue
        marker_solved = [
            record for record in group
            if as_float(record["marker"], "algorithm_baseline_mm") is not None
        ]
        sift_solved = [
            record for record in group
            if as_float(record["sift"], "algorithm_baseline_mm") is not None
        ]
        both = [
            record for record in group
            if as_float(record["marker"], "algorithm_baseline_mm") is not None
            and as_float(record["sift"], "algorithm_baseline_mm") is not None
        ]
        summary = {
            "distance_cm": distance,
            "scene_type": scene,
            "paired_requested_frames": len(group),
            "marker_solved_frames": len(marker_solved),
            "sift_solved_frames": len(sift_solved),
            "both_solved_frames": len(both),
            "marker_only_solved_frames": sum(
                as_float(record["marker"], "algorithm_baseline_mm") is not None
                and as_float(record["sift"], "algorithm_baseline_mm") is None
                for record in group
            ),
            "sift_only_solved_frames": sum(
                as_float(record["sift"], "algorithm_baseline_mm") is not None
                and as_float(record["marker"], "algorithm_baseline_mm") is None
                for record in group
            ),
            "neither_solved_frames": sum(
                as_float(record["marker"], "algorithm_baseline_mm") is None
                and as_float(record["sift"], "algorithm_baseline_mm") is None
                for record in group
            ),
            "marker_success_rate_percent": len(marker_solved) / len(group) * 100.0,
            "sift_success_rate_percent": len(sift_solved) / len(group) * 100.0,
            "sift_feature_supported_frames": sum(
                (as_float(record["sift"], "feature_match_count") or 0) > 0
                for record in group
            ),
            "sift_joint_refined_frames": sum(
                record["sift"].get("rt_sift_applied") is True for record in group
            ),
            "sift_reliable_frames": sum(
                record["sift"].get("rt_reliable") is True for record in group
            ),
        }
        for metric in (
            "rotation_error_deg",
            "absolute_baseline_delta_mm",
            "absolute_baseline_error_percent",
        ):
            triples = []
            for record in both:
                marker_value = as_float(record["marker"], metric)
                sift_value = as_float(record["sift"], metric)
                if marker_value is not None and sift_value is not None:
                    triples.append((marker_value, sift_value, marker_value - sift_value))
            summary[f"{metric}_paired_count"] = len(triples)
            summary[f"{metric}_marker_median"] = (
                float(np.median([value[0] for value in triples])) if triples else None
            )
            summary[f"{metric}_sift_median"] = (
                float(np.median([value[1] for value in triples])) if triples else None
            )
            summary[f"{metric}_sift_improvement_median"] = (
                float(np.median([value[2] for value in triples])) if triples else None
            )
            summary[f"{metric}_sift_win_rate_percent"] = (
                sum(value[1] < value[0] for value in triples) / len(triples) * 100.0
                if triples
                else None
            )
        output.append(summary)
    return output


def paired_frame_rows(rows: list[dict]) -> list[dict]:
    """Put the two algorithms on one row for direct same-frame comparison."""
    paired = defaultdict(dict)
    metadata = {}
    for row in rows:
        key = (row.get("video_file"), row.get("frame_index"))
        paired[key][row.get("mode")] = row
        metadata[key] = row
    output = []
    for key in sorted(
        paired,
        key=lambda item: (
            finite_or_none(metadata[item].get("distance_cm")) or 0.0,
            int(finite_or_none(metadata[item].get("repeat")) or 0),
            int(finite_or_none(item[1]) or 0),
        ),
    ):
        pair = paired[key]
        if "marker_only" not in pair or "sift_assisted" not in pair:
            continue
        marker = pair["marker_only"]
        sift = pair["sift_assisted"]
        row = {
            "distance_cm": marker.get("distance_cm"),
            "repeat": marker.get("repeat"),
            "scene_type": marker.get("scene_type"),
            "video_file": marker.get("video_file"),
            "frame_index": marker.get("frame_index"),
            "marker_status": marker.get("status"),
            "sift_status": sift.get("status"),
            "marker_rotation_error_deg": as_float(marker, "rotation_error_deg"),
            "sift_rotation_error_deg": as_float(sift, "rotation_error_deg"),
            "rotation_sift_improvement_deg": None,
            "marker_abs_baseline_error_mm": as_float(
                marker, "absolute_baseline_delta_mm"
            ),
            "sift_abs_baseline_error_mm": as_float(
                sift, "absolute_baseline_delta_mm"
            ),
            "baseline_sift_improvement_mm": None,
            "marker_abs_baseline_error_percent": as_float(
                marker, "absolute_baseline_error_percent"
            ),
            "sift_abs_baseline_error_percent": as_float(
                sift, "absolute_baseline_error_percent"
            ),
            "baseline_percent_sift_improvement": None,
            "sift_feature_match_count": sift.get("feature_match_count"),
            "sift_feature_inlier_count": sift.get("feature_inlier_count"),
            "sift_joint_refined": sift.get("rt_sift_applied"),
            "sift_rt_reliable": sift.get("rt_reliable"),
        }
        for marker_name, sift_name, output_name in (
            (
                "marker_rotation_error_deg",
                "sift_rotation_error_deg",
                "rotation_sift_improvement_deg",
            ),
            (
                "marker_abs_baseline_error_mm",
                "sift_abs_baseline_error_mm",
                "baseline_sift_improvement_mm",
            ),
            (
                "marker_abs_baseline_error_percent",
                "sift_abs_baseline_error_percent",
                "baseline_percent_sift_improvement",
            ),
        ):
            if row[marker_name] is not None and row[sift_name] is not None:
                row[output_name] = row[marker_name] - row[sift_name]
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    headers = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if headers:
            writer.writeheader()
            writer.writerows(rows)


def excel_column(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def clean_xml_text(value) -> str:
    text = str(value)
    text = "".join(
        char for char in text
        if char in "\t\n\r" or ord(char) >= 32
    )
    return escape(text)


def add_excel_pass_formulas(
    records: list[dict],
    headers: list[str],
    group_kind: str,
    frame_last_row: int,
    frame_headers: list[str],
) -> None:
    if not records:
        return
    frame_columns = {
        name: excel_column(index + 1) for index, name in enumerate(frame_headers)
    }
    summary_columns = {
        name: excel_column(index + 1) for index, name in enumerate(headers)
    }
    frame_sheet = "'Frame Results'"

    def frame_range(field: str) -> str:
        column = frame_columns[field]
        return f"{frame_sheet}!${column}$2:${column}${frame_last_row}"

    for row_index, record in enumerate(records, 2):
        group_criteria: list[str] = []
        if group_kind == "video":
            group_criteria.extend(
                [
                    frame_range("video_file"),
                    f"${summary_columns['video_file']}{row_index}",
                ]
            )
        else:
            group_criteria.extend(
                [
                    frame_range("distance_cm"),
                    f"${summary_columns['distance_cm']}{row_index}",
                    frame_range("angle_deg"),
                    f"${summary_columns['angle_deg']}{row_index}",
                ]
            )

        rotation_range = frame_range("rotation_error_deg")
        baseline_range = frame_range("absolute_baseline_error_percent")
        mode_range = frame_range("mode")
        sift_applied_range = frame_range("rt_sift_applied")

        def count_formula(mode: str, metric: str, sift_applied: bool = False) -> str:
            criteria = [*group_criteria, mode_range, f'"{mode}"']
            if sift_applied:
                criteria.extend([sift_applied_range, "TRUE"])
            if metric in {"rotation", "both"}:
                criteria.extend(
                    [rotation_range, '"<"&\'Settings\'!$B$2']
                )
            if metric in {"baseline", "both"}:
                criteria.extend(
                    [baseline_range, '"<"&\'Settings\'!$B$3']
                )
            return f"COUNTIFS({','.join(criteria)})"

        formula_fields = [
            ("marker", "marker_only", "marker_total_frames"),
            ("sift", "sift_assisted", "sift_total_frames"),
        ]
        for prefix, mode, total_field in formula_fields:
            total = max(int(record.get(total_field) or 0), 0)
            total_cell = f"${summary_columns[total_field]}{row_index}"
            for metric, count_field, percent_field in (
                (
                    "rotation",
                    f"{prefix}_rotation_pass_frames",
                    f"{prefix}_rotation_pass_percent",
                ),
                (
                    "baseline",
                    f"{prefix}_baseline_pass_frames",
                    f"{prefix}_baseline_pass_percent",
                ),
                (
                    "both",
                    f"{prefix}_qualified_frames",
                    f"{prefix}_qualified_percent",
                ),
            ):
                cached_count = int(record.get(count_field) or 0)
                record[count_field] = ExcelFormula(
                    count_formula(mode, metric), cached_count
                )
                count_cell = f"{summary_columns[count_field]}{row_index}"
                record[percent_field] = ExcelFormula(
                    f"IFERROR({count_cell}/{total_cell},0)",
                    cached_count / total if total else 0.0,
                )

        sift_total = max(int(record.get("sift_total_frames") or 0), 0)
        sift_total_cell = f"${summary_columns['sift_total_frames']}{row_index}"
        refined_count = int(record.get("sift_joint_refined_frames") or 0)
        refined_qualified_count = int(
            record.get("sift_joint_refined_qualified_frames") or 0
        )
        sift_refined_criteria = [
            *group_criteria,
            mode_range,
            '"sift_assisted"',
            sift_applied_range,
            "TRUE",
        ]
        record["sift_joint_refined_frames"] = ExcelFormula(
            f"COUNTIFS({','.join(sift_refined_criteria)})",
            refined_count,
        )
        refined_count_cell = (
            f"{summary_columns['sift_joint_refined_frames']}{row_index}"
        )
        record["sift_joint_refined_percent"] = ExcelFormula(
            f"IFERROR({refined_count_cell}/{sift_total_cell},0)",
            refined_count / sift_total if sift_total else 0.0,
        )
        record["sift_joint_refined_qualified_frames"] = ExcelFormula(
            count_formula("sift_assisted", "both", sift_applied=True),
            refined_qualified_count,
        )
        refined_qualified_cell = (
            f"{summary_columns['sift_joint_refined_qualified_frames']}{row_index}"
        )
        record["sift_joint_refined_qualified_percent"] = ExcelFormula(
            f"IFERROR({refined_qualified_cell}/{sift_total_cell},0)",
            refined_qualified_count / sift_total if sift_total else 0.0,
        )
        record["rotation_error_threshold_deg"] = ExcelFormula(
            "'Settings'!$B$2", RT_PASS_ROTATION_ERROR_DEG
        )
        record["baseline_error_threshold_percent"] = ExcelFormula(
            "'Settings'!$B$3", RT_PASS_BASELINE_ERROR_PERCENT
        )


def cell_style(sheet_name: str, header: str, row_index: int, value) -> int:
    if row_index == 1:
        return 1
    if sheet_name == "Settings" and header == "value":
        return 5
    if (
        isinstance(value, ExcelFormula)
        and header.endswith("_percent")
        and "threshold" not in header
    ):
        return 4
    if header.endswith("_frames") or header in {
        "repeat",
        "frame_index",
        "shared_marker_count",
        "marker_id",
        "ippe_branch_count_left",
        "ippe_branch_count_right",
        "ippe_branch_left",
        "ippe_branch_right",
    }:
        return 2
    if isinstance(value, (int, float, np.integer, np.floating, ExcelFormula)):
        return 3
    return 0


def worksheet_xml(sheet_name: str, headers: list[str], rows: list[dict]) -> str:
    all_rows = [dict(zip(headers, headers)), *rows]
    last_col = excel_column(max(len(headers), 1))
    last_row = max(len(all_rows), 1)
    widths = []
    for index, header in enumerate(headers, 1):
        sample = [str(row.get(header, "") or "") for row in rows[:200]]
        width = min(max([len(str(header)), *(len(value) for value in sample)]) + 2, 45)
        widths.append(
            f'<col min="{index}" max="{index}" width="{max(width, 10)}" customWidth="1"/>'
        )
    xml_rows = []
    for row_index, row in enumerate(all_rows, 1):
        cells = []
        for column_index, header in enumerate(headers, 1):
            ref = f"{excel_column(column_index)}{row_index}"
            value = row.get(header)
            style = cell_style(sheet_name, header, row_index, value)
            style_attr = f' s="{style}"' if style else ""
            if isinstance(value, ExcelFormula):
                cells.append(
                    f'<c r="{ref}"{style_attr}><f>{escape(value.text)}</f>'
                    f'<v>{value.cached_value}</v></c>'
                )
            elif value is None or value == "":
                cells.append(f'<c r="{ref}"{style_attr}/>')
            elif isinstance(value, (bool, np.bool_)):
                cells.append(
                    f'<c r="{ref}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
                )
            elif isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                cells.append(f'<c r="{ref}"{style_attr}><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}" t="inlineStr"{style_attr}><is><t xml:space="preserve">'
                    f'{clean_xml_text(value)}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    auto_filter = f'<autoFilter ref="A1:{last_col}{last_row}"/>' if headers else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_col}{last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        f'<cols>{"".join(widths)}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
        f'{auto_filter}</worksheet>'
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[dict]]]) -> None:
    sheet_entries = []
    workbook_rels = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, rows) in enumerate(sheets, 1):
            headers = list(rows[0]) if rows else ["No data"]
            safe_rows = rows if rows else [{"No data": ""}]
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                worksheet_xml(name, headers, safe_rows),
            )
            sheet_entries.append(
                f'<sheet name="{clean_xml_text(name[:31])}" sheetId="{index}" r:id="rId{index}"/>'
            )
            workbook_rels.append(
                '<Relationship '
                f'Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        style_rel_id = len(sheets) + 1
        workbook_rels.append(
            '<Relationship '
            f'Id="rId{style_rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(sheet_entries)}</sheets>'
            '<calcPr calcId="191029" fullCalcOnLoad="1" forceFullCalc="1"/>'
            '</workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(workbook_rels)}</Relationships>',
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<numFmts count="4">'
            '<numFmt numFmtId="164" formatCode="0.0000"/>'
            '<numFmt numFmtId="165" formatCode="0.00%"/>'
            '<numFmt numFmtId="166" formatCode="0"/>'
            '<numFmt numFmtId="167" formatCode="0.00"/>'
            '</numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="4"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/>'
            '<bgColor indexed="64"/></patternFill></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/>'
            '<bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="6">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
            '<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            '<xf numFmtId="167" fontId="0" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyFill="1"/>'
            '</cellXfs>'
            '</styleSheet>',
        )
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f'{"".join(content_overrides)}</Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )


def finalize_outputs(prefix: Path, rows: list[dict], no_excel: bool, protocol: list[dict]) -> None:
    video_summary = summarize_groups(
        rows, ["distance_cm", "repeat", "scene_type", "video_file", "mode"]
    )
    distance_summary = summarize_groups(
        rows, ["distance_cm", "scene_type", "mode"]
    )
    effects = sift_effect_rows(rows)
    paired_frames = paired_frame_rows(rows)
    distance_pass_rates, video_pass_rates = pass_rate_summaries(rows)
    video_csv = prefix.with_name(prefix.name + "_video_summary.csv")
    distance_csv = prefix.with_name(prefix.name + "_distance_summary.csv")
    effect_csv = prefix.with_name(prefix.name + "_sift_effect.csv")
    paired_csv = prefix.with_name(prefix.name + "_paired_frames.csv")
    distance_pass_csv = prefix.with_name(prefix.name + "_distance_pass_rates.csv")
    video_pass_csv = prefix.with_name(prefix.name + "_video_pass_rates.csv")
    write_csv(video_csv, video_summary)
    write_csv(distance_csv, distance_summary)
    write_csv(effect_csv, effects)
    write_csv(paired_csv, paired_frames)
    write_csv(distance_pass_csv, distance_pass_rates)
    write_csv(video_pass_csv, video_pass_rates)
    if not no_excel:
        excel_frame_headers = ["distance_cm", "angle_deg", *RESULT_FIELDS[1:]]
        excel_frame_rows = []
        for row in rows:
            excel_row = {"distance_cm": row.get("distance_cm")}
            excel_row["angle_deg"] = video_angle_deg(row.get("video_file"))
            excel_row.update(
                {field: row.get(field) for field in RESULT_FIELDS[1:]}
            )
            excel_frame_rows.append(excel_row)
        settings = [
            {
                "parameter": "rotation_error_threshold_deg",
                "value": RT_PASS_ROTATION_ERROR_DEG,
                "unit": "deg",
                "description": (
                    "Edit B2; angle pass counts and percentages update in Excel."
                ),
            },
            {
                "parameter": "absolute_baseline_error_threshold_percent",
                "value": RT_PASS_BASELINE_ERROR_PERCENT,
                "unit": "%",
                "description": (
                    "Edit B3; Frame Results stores percentage points (5 means 5%)."
                ),
            },
        ]
        distance_pass_headers = (
            list(distance_pass_rates[0]) if distance_pass_rates else ["No data"]
        )
        video_pass_headers = (
            list(video_pass_rates[0]) if video_pass_rates else ["No data"]
        )
        frame_last_row = len(rows) + 1
        add_excel_pass_formulas(
            distance_pass_rates,
            distance_pass_headers,
            "distance",
            frame_last_row,
            excel_frame_headers,
        )
        add_excel_pass_formulas(
            video_pass_rates,
            video_pass_headers,
            "video",
            frame_last_row,
            excel_frame_headers,
        )
        xlsx_path = prefix.with_suffix(".xlsx")
        write_xlsx(
            xlsx_path,
            [
                ("Settings", settings),
                ("Frame Results", excel_frame_rows),
                ("Paired Frames", paired_frames),
                ("Video Summary", video_summary),
                ("Distance Summary", distance_summary),
                ("SIFT Effect", effects),
                ("Distance Pass Rate", distance_pass_rates),
                ("Video Pass Rate", video_pass_rates),
                ("Protocol", protocol),
            ],
        )
        print(f"Excel: {xlsx_path}")
    print(f"Frame checkpoint: {prefix.with_name(prefix.name + '_frame_results.csv')}")
    print(f"Video summary: {video_csv}")
    print(f"Distance summary: {distance_csv}")
    print(f"SIFT effect: {effect_csv}")
    print(f"Paired frames: {paired_csv}")
    print(f"Distance pass rates: {distance_pass_csv}")
    print(f"Video pass rates: {video_pass_csv}")


def main() -> int:
    # Windows Traditional-Chinese consoles commonly use cp950, while the existing
    # analyzer logs contain emoji.  Keep a harmless log glyph from aborting a batch.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parse_args()
    folder = Path(args.folder).resolve()
    calibration = Path(args.calibration).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)

    videos = collect_stage1_videos(folder, args.only_distance)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    validate_video_set(
        videos,
        limited=args.only_distance is not None or args.max_videos is not None,
    )
    selected_modes = MODES if args.mode == "both" else (args.mode,)
    prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix
        else folder / "HBVCAM_stage1_first300_fullframe_sift_comparison"
    )
    if prefix.suffix.lower() in {".xlsx", ".csv"}:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    aruco_corner_videos_enabled = not args.no_aruco_corner_videos
    aruco_corner_video_dir = (
        Path(args.aruco_corner_video_dir).resolve()
        if args.aruco_corner_video_dir
        else prefix.with_name(prefix.name + "_aruco_corner_videos")
    )
    if aruco_corner_videos_enabled:
        aruco_corner_video_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = prefix.with_name(prefix.name + "_frame_results.csv")
    existing_rows = []
    completed_keys = set()
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint}\n"
                "Use --resume to continue it, or choose another --output-prefix."
            )
        existing_rows = load_checkpoint(checkpoint)
        completed_keys = {
            (row["video_file"], int(row["frame_index"]), row["mode"])
            for row in existing_rows
        }

    mtx_left, dist_left, mtx_right, dist_right, answer_extrinsic = (
        hbvcam.load_hbvcam_calibration(str(calibration))
    )
    if not answer_extrinsic or "R" not in answer_extrinsic or "T" not in answer_extrinsic:
        raise ValueError("Calibration JSON must contain extrinsic.R and extrinsic.T answers")
    common_k, map_l1, map_l2, map_r1, map_r2 = build_common_intrinsic_maps(
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        (hbvcam.CAMERA_WIDTH // 2, hbvcam.CAMERA_HEIGHT),
    )
    estimator = MarkerOnlyEstimator(
        common_k,
        hbvcam.ACTUAL_MARKER_SIZE_MM,
        (hbvcam.CAMERA_WIDTH // 2, hbvcam.CAMERA_HEIGHT),
    )
    # The HBVCAM wrapper forwards these ratios to the shared RT analyzer.  Set
    # them only in this process so sift_assisted also uses both complete eyes.
    (
        hbvcam.RT_ROI_LEFT_X_RATIO,
        hbvcam.RT_ROI_LEFT_Y_RATIO,
        hbvcam.RT_ROI_LEFT_WIDTH_RATIO,
        hbvcam.RT_ROI_LEFT_HEIGHT_RATIO,
    ) = STAGE1_LEFT_ROI_RATIO
    (
        hbvcam.RT_ROI_RIGHT_X_RATIO,
        hbvcam.RT_ROI_RIGHT_Y_RATIO,
        hbvcam.RT_ROI_RIGHT_WIDTH_RATIO,
        hbvcam.RT_ROI_RIGHT_HEIGHT_RATIO,
    ) = STAGE1_RIGHT_ROI_RATIO
    hbvcam.video_pose_algo.SAVE_DEBUG_PAIR_IMAGES = False
    hbvcam.video_pose_algo.SAVE_RT_SIFT_DIAGNOSTICS = False

    new_file = not checkpoint.exists()
    mode = "a" if checkpoint.exists() else "w"
    all_rows = list(existing_rows)
    overall_started = time.perf_counter()
    print(
        f"Found {len(videos)} videos; frames 0..{args.frames - 1}; "
        f"modes={','.join(selected_modes)}"
    )
    print(
        "Marker-only = full-frame ArUco corners + IPPE branches only; "
        "SIFT-assisted = current analyzer with full-frame ArUco/SIFT."
    )
    with checkpoint.open(mode, encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
            csv_file.flush()

        for video_number, video in enumerate(videos, 1):
            needed = [
                (frame_index, algorithm_mode)
                for frame_index in range(args.frames)
                for algorithm_mode in selected_modes
                if (video["path"].name, frame_index, algorithm_mode)
                not in completed_keys
            ]
            aruco_corner_video_path = aruco_corner_video_dir / (
                video["path"].stem + "_aruco_corners.mp4"
            )
            generate_aruco_corner_video = aruco_corner_videos_enabled and (
                bool(needed) or not aruco_corner_video_path.exists()
            )
            if not needed and not generate_aruco_corner_video:
                print(f"[{video_number}/{len(videos)}] resume skip {video['path'].name}")
                continue
            capture = cv2.VideoCapture(str(video["path"]))
            if not capture.isOpened():
                for frame_index, algorithm_mode in needed:
                    row = empty_result(video, frame_index, algorithm_mode)
                    row["failure_reason"] = "OSError: Unable to open video"
                    append_csv_row(csv_file, writer, row)
                    all_rows.append(row)
                print(f"WARNING: unable to open {video['path']}")
                continue
            reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            aruco_corner_writer = None
            if generate_aruco_corner_video:
                aruco_corner_writer, diagnostic_fps = open_aruco_diagnostic_writer(
                    aruco_corner_video_path,
                    capture,
                )
                print(
                    f"  ArUco corner video: {aruco_corner_video_path} "
                    f"(640x320 @ {diagnostic_fps:.3f} fps)"
                )
            video_started = time.perf_counter()
            print(
                f"[{video_number}/{len(videos)}] {video['path'].name} "
                f"reported_frames={reported}"
            )
            try:
                for frame_index in range(args.frames):
                    ok, sbs = capture.read()
                    if not ok or sbs is None:
                        for remaining_index in range(frame_index, args.frames):
                            for algorithm_mode in selected_modes:
                                key = (video["path"].name, remaining_index, algorithm_mode)
                                if key in completed_keys:
                                    continue
                                row = empty_result(video, remaining_index, algorithm_mode)
                                row["failure_reason"] = (
                                    f"EOF: video ended before frame {remaining_index}"
                                )
                                append_csv_row(csv_file, writer, row)
                                all_rows.append(row)
                                completed_keys.add(key)
                        break
                    frame_modes = [
                        algorithm_mode
                        for algorithm_mode in selected_modes
                        if (video["path"].name, frame_index, algorithm_mode)
                        not in completed_keys
                    ]
                    if not frame_modes and aruco_corner_writer is None:
                        continue
                    height, full_width = sbs.shape[:2]
                    if (full_width, height) != (
                        hbvcam.CAMERA_WIDTH,
                        hbvcam.CAMERA_HEIGHT,
                    ):
                        reason = (
                            f"ValueError: frame is {full_width}x{height}; expected "
                            f"{hbvcam.CAMERA_WIDTH}x{hbvcam.CAMERA_HEIGHT}"
                        )
                        for algorithm_mode in frame_modes:
                            row = empty_result(video, frame_index, algorithm_mode)
                            row["failure_reason"] = reason
                            append_csv_row(csv_file, writer, row)
                            all_rows.append(row)
                            completed_keys.add(
                                (video["path"].name, frame_index, algorithm_mode)
                            )
                        continue
                    half = full_width // 2
                    left_common = cv2.remap(
                        sbs[:, :half], map_l1, map_l2, cv2.INTER_LINEAR
                    )
                    right_common = cv2.remap(
                        sbs[:, half:], map_r1, map_r2, cv2.INTER_LINEAR
                    )
                    left_markers = {}
                    right_markers = {}
                    left_detection_diagnostics = {}
                    right_detection_diagnostics = {}
                    detected = None
                    detection_error = ""
                    try:
                        # Detect/refine on the original distorted pixels.  Only the
                        # accepted subpixel coordinates are mapped into common-K
                        # space, avoiding the blur introduced by image remapping.
                        left_markers = estimator.detect(
                            sbs[:, :half],
                            estimator.left_roi,
                            mtx_left,
                            dist_left,
                        )
                        left_detection_diagnostics = dict(
                            estimator.last_detection_diagnostics
                        )
                        right_markers = estimator.detect(
                            sbs[:, half:],
                            estimator.right_roi,
                            mtx_right,
                            dist_right,
                        )
                        right_detection_diagnostics = dict(
                            estimator.last_detection_diagnostics
                        )
                        if not (set(left_markers) & set(right_markers)):
                            raise RuntimeError(
                                "No shared stable ArUco marker; "
                                f"left initial={left_detection_diagnostics.get('initial_marker_count', 0)} "
                                f"accepted={left_detection_diagnostics.get('accepted_marker_count', 0)}, "
                                f"right initial={right_detection_diagnostics.get('initial_marker_count', 0)} "
                                f"accepted={right_detection_diagnostics.get('accepted_marker_count', 0)}; "
                                "rejected markers exceeded the adaptive-window stability gate"
                            )
                        detected = estimator.pair_from_markers(
                            left_markers,
                            right_markers,
                        )
                    except Exception as exc:
                        detection_error = f"{type(exc).__name__}: {exc}"
                    if aruco_corner_writer is not None:
                        diagnostic_frame = aruco_diagnostic_frame(
                            left_common,
                            right_common,
                            left_markers,
                            right_markers,
                            detected.get("marker_id") if detected is not None else None,
                            frame_index,
                            args.aruco_crop_padding_ratio,
                            (
                                "UNSTABLE ARUCO"
                                if left_detection_diagnostics.get("initial_marker_count", 0) > 0
                                and left_detection_diagnostics.get("accepted_marker_count", 0) == 0
                                else None
                            ),
                            (
                                "UNSTABLE ARUCO"
                                if right_detection_diagnostics.get("initial_marker_count", 0) > 0
                                and right_detection_diagnostics.get("accepted_marker_count", 0) == 0
                                else None
                            ),
                        )
                        aruco_corner_writer.write(diagnostic_frame)
                    if not frame_modes:
                        continue
                    for algorithm_mode in frame_modes:
                        if detected is None:
                            row = empty_result(video, frame_index, algorithm_mode)
                            row["failure_reason"] = detection_error
                        elif algorithm_mode == "marker_only":
                            row = marker_only_row(
                                video,
                                frame_index,
                                estimator,
                                detected,
                                answer_extrinsic,
                            )
                        else:
                            row = sift_assisted_row(
                                video,
                                frame_index,
                                left_common,
                                right_common,
                                common_k,
                                detected,
                                left_markers,
                                right_markers,
                                answer_extrinsic,
                            )
                        append_csv_row(csv_file, writer, row)
                        all_rows.append(row)
                        completed_keys.add(
                            (video["path"].name, frame_index, algorithm_mode)
                        )
                    if args.progress_every > 0 and (
                        (frame_index + 1) % args.progress_every == 0
                        or frame_index + 1 == args.frames
                    ):
                        elapsed = time.perf_counter() - video_started
                        print(
                            f"  F{frame_index:03d} complete | "
                            f"video elapsed {elapsed / 60.0:.1f} min"
                        )
            finally:
                capture.release()
                if aruco_corner_writer is not None:
                    aruco_corner_writer.release()

    protocol = [
        {"parameter": "generated_at", "value": datetime.now(timezone.utc).astimezone().isoformat()},
        {"parameter": "source_folder", "value": str(folder)},
        {"parameter": "calibration_file", "value": str(calibration)},
        {"parameter": "video_name_filter", "value": VIDEO_PATTERN.pattern},
        {"parameter": "frames", "value": f"zero-based 0..{args.frames - 1}; every frame retained"},
        {"parameter": "repeat_1_to_5", "value": "object_in_roi"},
        {"parameter": "repeat_6", "value": "pattern_only"},
        {"parameter": "marker_only", "value": "full-frame ArUco corners + IPPE only; no SIFT and no JSON decision"},
        {"parameter": "sift_assisted", "value": "full-frame ArUco + SIFT branch selection/joint RT refinement"},
        {"parameter": "JSON usage", "value": "error calculation only after RT; never used to select frame or IPPE branch"},
        {"parameter": "ROI mode", "value": "full frame for each 1920x1080 eye (ArUco and SIFT)"},
        {"parameter": "left_roi", "value": str(estimator.left_roi)},
        {"parameter": "right_roi", "value": str(estimator.right_roi)},
        {"parameter": "ArUco detection pixels", "value": "original distorted left/right images; CLAHE for initial detection only"},
        {"parameter": "ArUco subpixel pixels", "value": "original non-CLAHE grayscale; coordinates undistorted to common K after refinement"},
        {"parameter": "ArUco adaptive subpixel half-window", "value": f"round(marker_side_px/12), clamped to {ARUCO_SUBPIX_MIN_HALF_WINDOW}..{ARUCO_SUBPIX_MAX_HALF_WINDOW}"},
        {"parameter": "ArUco stability rejection", "value": f"reject marker when neighboring-window corner disagreement exceeds {ARUCO_SUBPIX_STABILITY_MAX_RAW_PX:g} raw px"},
        {"parameter": "rotation_error", "value": "acos(clamp((trace(R_est*R_json^T)-1)/2))*180/pi"},
        {"parameter": "baseline_error_percent", "value": "(algorithm_baseline-json_baseline)/json_baseline*100"},
        {"parameter": "qualified_rotation_error_deg", "value": f"Excel Settings!B2; default strictly less than {RT_PASS_ROTATION_ERROR_DEG:g}"},
        {"parameter": "qualified_absolute_baseline_error_percent", "value": f"Excel Settings!B3; default strictly less than {RT_PASS_BASELINE_ERROR_PERCENT:g}"},
        {"parameter": "ArUco diagnostic video", "value": "post-undistortion, post-subpixel corners; LEFT 320x320 + RIGHT 320x320"},
        {"parameter": "ArUco diagnostic padding ratio", "value": args.aruco_crop_padding_ratio},
        {"parameter": "ArUco diagnostic output folder", "value": str(aruco_corner_video_dir) if aruco_corner_videos_enabled else "disabled"},
    ]
    all_rows = [normalize_csv_row(row) for row in all_rows]
    finalize_outputs(prefix, all_rows, args.no_excel, protocol)
    if aruco_corner_videos_enabled:
        print(f"ArUco corner videos: {aruco_corner_video_dir}")
    print(f"Total elapsed: {(time.perf_counter() - overall_started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Per-frame checkpoint is preserved; rerun with --resume.")
        raise SystemExit(130)
