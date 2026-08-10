"""HBVCAM ArUco-only RT and corner-stability batch analysis.

This script intentionally uses only the four subpixel-refined ArUco corners in
the simultaneous left/right images.  SIFT, scene features, and JSON extrinsics
do not participate in RT estimation.  JSON extrinsics are used only after an
estimate has been produced, to calculate rotation and baseline errors.

Accepted video names:
    HBVCAM_<distance>CM_opposite_<repeat>.mp4
    HBVCAM_<distance>CM_opposite_<repeat>_<angle>.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np


VIDEO_PATTERN = re.compile(
    r"^HBVCAM_(?P<distance>\d+(?:\.\d+)?)CM_opposite_"
    r"(?P<repeat>[1-6])(?:_(?P<angle>-?\d+(?:\.\d+)?))?"
    r"\.(?:mp4|avi|mkv|mov|m4v)$",
    re.IGNORECASE,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / "HBVCAM_4M2214HD-2-v11"
DEFAULT_CALIBRATION = ROOT / "calibration_result_HBVCAM_4M2214HD-2-v11.json"
DEFAULT_OUTPUT_NAME = "HBVCAM_aruco_corner_rt_stability_20260805.xlsx"

SBS_WIDTH = 3840
SBS_HEIGHT = 1080
MARKER_SIZE_MM = 8.25
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
DEFAULT_ROTATION_THRESHOLD_DEG = 2.0
DEFAULT_BASELINE_THRESHOLD_PERCENT = 5.0


FRAME_FIELDS = [
    "distance_cm",
    "angle_deg",
    "repeat",
    "scene_type",
    "video_file",
    "video_path",
    "frame_index",
    "status",
    "failure_reason",
    "marker_id",
    "shared_marker_count",
    "left_c0_x",
    "left_c0_y",
    "left_c1_x",
    "left_c1_y",
    "left_c2_x",
    "left_c2_y",
    "left_c3_x",
    "left_c3_y",
    "right_c0_x",
    "right_c0_y",
    "right_c1_x",
    "right_c1_y",
    "right_c2_x",
    "right_c2_y",
    "right_c3_x",
    "right_c3_y",
    "left_rect_width_px",
    "left_rect_height_px",
    "left_rect_area_px2",
    "left_quad_area_px2",
    "right_rect_width_px",
    "right_rect_height_px",
    "right_rect_area_px2",
    "right_quad_area_px2",
    "left_side_mean_px",
    "right_side_mean_px",
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
    "marker_reproj_left_px",
    "marker_reproj_right_px",
    "marker_reproj_combined_rms_px",
    "ippe_branch_left",
    "ippe_branch_right",
    "processing_time_ms",
]

CORNER_STD_FIELDS = [
    f"{side}_c{corner}_{axis}_std_px"
    for side in ("left", "right")
    for corner in range(4)
    for axis in ("x", "y")
]

CORNER_METRIC_FIELDS = [
    *CORNER_STD_FIELDS,
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
    "right_rect_width_mean_px",
    "right_rect_width_std_px",
    "right_rect_height_mean_px",
    "right_rect_height_std_px",
    "right_rect_area_mean_px2",
    "right_rect_area_std_px2",
    "right_quad_area_mean_px2",
    "right_quad_area_std_px2",
]

RT_METRIC_FIELDS = [
    "relative_rotation_angle_mean_deg",
    "relative_rotation_angle_std_deg",
    "rotation_error_mean_deg",
    "rotation_error_std_deg",
    "baseline_mean_mm",
    "baseline_std_mm",
    "absolute_baseline_error_mean_percent",
    "absolute_baseline_error_std_percent",
    "rotation_pass_frames",
    "rotation_pass_rate",
    "baseline_pass_frames",
    "baseline_pass_rate",
    "both_pass_frames",
    "both_pass_rate",
]

VIDEO_CORNER_HEADERS = [
    "distance_cm",
    "angle_deg",
    "repeat",
    "scene_type",
    "video_file",
    "total_frames",
    "detected_frames",
    "detection_rate_percent",
    *CORNER_METRIC_FIELDS,
]

DISTANCE_CORNER_HEADERS = [
    "distance_cm",
    "angle_deg",
    "video_count",
    "total_frames",
    "detected_frames",
    "detection_rate_percent",
    *CORNER_METRIC_FIELDS,
]

VIDEO_RT_HEADERS = [
    "distance_cm",
    "angle_deg",
    "repeat",
    "scene_type",
    "video_file",
    "total_frames",
    "solved_frames",
    "solve_rate_percent",
    *RT_METRIC_FIELDS,
]

DISTANCE_RT_HEADERS = [
    "distance_cm",
    "angle_deg",
    "video_count",
    "total_frames",
    "solved_frames",
    "solve_rate_percent",
    *RT_METRIC_FIELDS,
]


@dataclass(frozen=True)
class ExcelFormula:
    text: str
    cached_value: float | int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use only ArUco corners to estimate per-frame stereo RT and report "
            "corner/RT stability by video and distance."
        )
    )
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    parser.add_argument("--min-baseline-mm", type=float, default=MIN_BASELINE_MM)
    parser.add_argument("--max-baseline-mm", type=float, default=MAX_BASELINE_MM)
    parser.add_argument("--marker-id", type=int, help="Require one ArUco ID.")
    parser.add_argument("--only-distance", type=float)
    parser.add_argument(
        "--only-video",
        help="Analyze one exact file name or file stem, e.g. HBVCAM_25CM_opposite_1_15.",
    )
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--output",
        help=f"Output .xlsx path. Default: <folder>/{DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")
    if args.min_baseline_mm <= 0 or args.max_baseline_mm <= args.min_baseline_mm:
        parser.error("baseline limits are invalid")
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    return args


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def collect_videos(
    folder: Path,
    only_distance: float | None,
    only_video: str | None,
) -> list[dict]:
    videos = []
    requested_name = str(only_video or "").lower()
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = VIDEO_PATTERN.match(path.name)
        if match is None:
            continue
        distance = float(match.group("distance"))
        repeat = int(match.group("repeat"))
        angle = float(match.group("angle") or 0.0)
        if only_distance is not None and not math.isclose(
            distance,
            only_distance,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            continue
        if requested_name and requested_name not in {path.name.lower(), path.stem.lower()}:
            continue
        videos.append(
            {
                "path": path.resolve(),
                "distance_cm": distance,
                "angle_deg": angle,
                "repeat": repeat,
                "scene_type": "pattern_only" if repeat == 6 else "object_in_roi",
            }
        )
    videos.sort(
        key=lambda item: (
            item["distance_cm"],
            item["angle_deg"],
            item["repeat"],
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
        np.asarray(left["matrix"], dtype=np.float64),
        np.asarray(left["distortion"], dtype=np.float64),
        np.asarray(right["matrix"], dtype=np.float64),
        np.asarray(right["distortion"], dtype=np.float64),
        {
            "R": np.asarray(extrinsic["R"], dtype=np.float64).reshape(3, 3),
            "T": np.asarray(extrinsic["T"], dtype=np.float64).reshape(3, 1),
        },
    )


def build_common_intrinsic_maps(k_left, d_left, k_right, d_right, image_size):
    common_k, _ = cv2.getOptimalNewCameraMatrix(
        k_left,
        d_left,
        image_size,
        1.0,
        image_size,
    )
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        k_left,
        d_left,
        None,
        common_k,
        image_size,
        cv2.CV_16SC2,
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        k_right,
        d_right,
        None,
        common_k,
        image_size,
        cv2.CV_16SC2,
    )
    return common_k.astype(np.float64), map_l1, map_l2, map_r1, map_r2


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_answer_errors(R_est, t_est, answer):
    R_answer = answer["R"]
    t_answer = answer["T"]
    R_est = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3, 1)
    rotation_error = rotation_angle_deg(R_est @ R_answer.T)
    baseline = float(np.linalg.norm(t_est))
    answer_baseline = float(np.linalg.norm(t_answer))
    baseline_delta = baseline - answer_baseline
    baseline_error_percent = baseline_delta / answer_baseline * 100.0
    translation_l2 = float(np.linalg.norm(t_est - t_answer))
    direction_cos = np.clip(
        float((t_est.T @ t_answer)[0, 0]) / (baseline * answer_baseline),
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
        "baseline_error_percent": baseline_error_percent,
        "absolute_baseline_error_percent": abs(baseline_error_percent),
        "translation_l2_error_mm": translation_l2,
        "translation_direction_error_deg": float(np.degrees(np.arccos(direction_cos))),
    }


def marker_image_metrics(corners: np.ndarray) -> dict:
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


class ArucoCornerRTEstimator:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        marker_size_mm: float,
        min_baseline_mm: float,
        max_baseline_mm: float,
        required_marker_id: int | None,
    ):
        self.K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.min_baseline_mm = float(min_baseline_mm)
        self.max_baseline_mm = float(max_baseline_mm)
        self.required_marker_id = required_marker_id
        half = float(marker_size_mm) / 2.0
        self.object_points = np.asarray(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.parameters,
            )
        if ids is None or not len(ids):
            return {}
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        result = {}
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            cv2.cornerSubPix(gray, marker_corners, (5, 5), (-1, -1), criteria)
            result[int(marker_id)] = marker_corners.reshape(4, 2).astype(np.float64)
        return result

    def select_shared(self, left_markers, right_markers):
        shared = sorted(set(left_markers) & set(right_markers))
        if self.required_marker_id is not None:
            if self.required_marker_id not in shared:
                raise RuntimeError(
                    f"Required ArUco ID {self.required_marker_id} was not detected in both views"
                )
            marker_id = self.required_marker_id
        else:
            if not shared:
                raise RuntimeError("No shared ArUco ID was detected in both views")
            marker_id = shared[0]
        return marker_id, len(shared), left_markers[marker_id], right_markers[marker_id]

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
            camera_points = (
                rotation @ self.object_points.astype(np.float64).T + translation
            ).T
            if np.any(camera_points[:, 2] <= 0.0):
                continue
            projected, _ = cv2.projectPoints(
                self.object_points,
                rvec,
                translation,
                self.K,
                None,
            )
            residuals = np.linalg.norm(
                projected.reshape(4, 2) - image_points.reshape(4, 2),
                axis=1,
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

    def estimate(self, left_corners: np.ndarray, right_corners: np.ndarray) -> dict:
        left_branches = self.pose_branches(left_corners)
        right_branches = self.pose_branches(right_corners)
        if not left_branches or not right_branches:
            raise RuntimeError("IPPE returned no positive-depth pose branch")
        candidates = []
        for left in left_branches:
            for right in right_branches:
                R_rel = right["R"] @ left["R"].T
                t_rel = right["t"] - R_rel @ left["t"]
                baseline = float(np.linalg.norm(t_rel))
                if not self.min_baseline_mm <= baseline <= self.max_baseline_mm:
                    continue
                combined_rms = float(
                    math.sqrt(0.5 * (left["rms"] ** 2 + right["rms"] ** 2))
                )
                candidates.append(
                    (
                        combined_rms,
                        max(left["max"], right["max"]),
                        left,
                        right,
                        R_rel,
                        t_rel,
                    )
                )
        if not candidates:
            raise RuntimeError("Every IPPE branch pair failed the baseline gate")
        candidates.sort(key=lambda item: (item[0], item[1]))
        rms, _maximum, left, right, R_rel, t_rel = candidates[0]
        return {
            "R_rel": R_rel,
            "t_rel": t_rel,
            "marker_reproj_left_px": left["rms"],
            "marker_reproj_right_px": right["rms"],
            "marker_reproj_combined_rms_px": rms,
            "ippe_branch_left": left["index"],
            "ippe_branch_right": right["index"],
        }


def empty_frame_row(video: dict, frame_index: int) -> dict:
    row = {field: None for field in FRAME_FIELDS}
    row.update(
        {
            "distance_cm": video["distance_cm"],
            "angle_deg": video["angle_deg"],
            "repeat": video["repeat"],
            "scene_type": video["scene_type"],
            "video_file": video["path"].name,
            "video_path": str(video["path"]),
            "frame_index": frame_index,
            "status": "FAILED",
            "failure_reason": "",
        }
    )
    return row


def add_corner_values(row: dict, side: str, corners: np.ndarray) -> None:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    for index, (x, y) in enumerate(points):
        row[f"{side}_c{index}_x"] = float(x)
        row[f"{side}_c{index}_y"] = float(y)
    metrics = marker_image_metrics(points)
    for name, value in metrics.items():
        row[f"{side}_{name}"] = value


def analyze_frame(
    video: dict,
    frame_index: int,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    estimator: ArucoCornerRTEstimator,
    answer: dict,
) -> dict:
    started = time.perf_counter()
    row = empty_frame_row(video, frame_index)
    try:
        left_markers = estimator.detect(left_frame)
        right_markers = estimator.detect(right_frame)
        marker_id, shared_count, left_corners, right_corners = estimator.select_shared(
            left_markers,
            right_markers,
        )
        row["marker_id"] = marker_id
        row["shared_marker_count"] = shared_count
        add_corner_values(row, "left", left_corners)
        add_corner_values(row, "right", right_corners)
        row["status"] = "DETECTED"
        estimate = estimator.estimate(left_corners, right_corners)
        row.update(
            {
                name: value
                for name, value in estimate.items()
                if name not in {"R_rel", "t_rel"}
            }
        )
        row.update(
            calculate_answer_errors(
                estimate["R_rel"],
                estimate["t_rel"],
                answer,
            )
        )
        row["status"] = "OK"
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
    return row


def numeric_values(rows: list[dict], field: str) -> list[float]:
    values = [finite(row.get(field)) for row in rows]
    return [value for value in values if value is not None]


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std_or_none(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else None


def corner_summary(group: list[dict], metadata: dict) -> dict:
    detected = [row for row in group if row.get("marker_id") is not None]
    record = dict(metadata)
    record.update(
        {
            "total_frames": len(group),
            "detected_frames": len(detected),
            "detection_rate_percent": len(detected) / len(group) * 100.0 if group else None,
        }
    )
    for side in ("left", "right"):
        coordinate_stds = []
        for corner in range(4):
            for axis in ("x", "y"):
                source = f"{side}_c{corner}_{axis}"
                output = f"{source}_std_px"
                value = std_or_none(numeric_values(detected, source))
                record[output] = value
                if value is not None:
                    coordinate_stds.append(value)
        record[f"{side}_corner_coordinate_std_rms_px"] = (
            float(math.sqrt(np.mean(np.square(coordinate_stds))))
            if coordinate_stds
            else None
        )
        for metric, label in (
            ("rect_width_px", "rect_width"),
            ("rect_height_px", "rect_height"),
            ("rect_area_px2", "rect_area"),
            ("quad_area_px2", "quad_area"),
        ):
            values = numeric_values(detected, f"{side}_{metric}")
            unit = "px2" if metric.endswith("px2") else "px"
            record[f"{side}_{label}_mean_{unit}"] = mean_or_none(values)
            record[f"{side}_{label}_std_{unit}"] = std_or_none(values)
    return record


def rt_static_summary(group: list[dict], metadata: dict) -> dict:
    solved = [row for row in group if row.get("status") == "OK"]
    record = dict(metadata)
    record.update(
        {
            "total_frames": len(group),
            "solved_frames": len(solved),
            "solve_rate_percent": len(solved) / len(group) * 100.0 if group else None,
        }
    )
    for source, mean_name, std_name in (
        (
            "relative_rotation_angle_deg",
            "relative_rotation_angle_mean_deg",
            "relative_rotation_angle_std_deg",
        ),
        ("rotation_error_deg", "rotation_error_mean_deg", "rotation_error_std_deg"),
        ("algorithm_baseline_mm", "baseline_mean_mm", "baseline_std_mm"),
        (
            "absolute_baseline_error_percent",
            "absolute_baseline_error_mean_percent",
            "absolute_baseline_error_std_percent",
        ),
    ):
        values = numeric_values(solved, source)
        record[mean_name] = mean_or_none(values)
        record[std_name] = std_or_none(values)
    return record


def frame_pass_counts(group: list[dict], rotation_threshold: float, baseline_threshold: float):
    solved = [row for row in group if row.get("status") == "OK"]
    rotation_pass = sum(
        finite(row.get("rotation_error_deg")) < rotation_threshold for row in solved
    )
    baseline_pass = sum(
        finite(row.get("absolute_baseline_error_percent")) < baseline_threshold
        for row in solved
    )
    both_pass = sum(
        finite(row.get("rotation_error_deg")) < rotation_threshold
        and finite(row.get("absolute_baseline_error_percent")) < baseline_threshold
        for row in solved
    )
    return rotation_pass, baseline_pass, both_pass


def excel_column(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def add_pass_formulas(
    records: list[dict],
    groups: list[list[dict]],
    headers: list[str],
    group_kind: str,
    frame_last_row: int,
) -> None:
    frame_columns = {name: excel_column(index + 1) for index, name in enumerate(FRAME_FIELDS)}
    summary_columns = {name: excel_column(index + 1) for index, name in enumerate(headers)}
    frame_sheet = "'Frame Results'"
    for row_index, (record, group) in enumerate(zip(records, groups), 2):
        criteria = []
        if group_kind == "video":
            criteria.extend(
                [
                    f"{frame_sheet}!${frame_columns['video_file']}$2:${frame_columns['video_file']}${frame_last_row}",
                    f"${summary_columns['video_file']}{row_index}",
                ]
            )
        else:
            criteria.extend(
                [
                    f"{frame_sheet}!${frame_columns['distance_cm']}$2:${frame_columns['distance_cm']}${frame_last_row}",
                    f"${summary_columns['distance_cm']}{row_index}",
                    f"{frame_sheet}!${frame_columns['angle_deg']}$2:${frame_columns['angle_deg']}${frame_last_row}",
                    f"${summary_columns['angle_deg']}{row_index}",
                ]
            )
        status_range = (
            f"{frame_sheet}!${frame_columns['status']}$2:"
            f"${frame_columns['status']}${frame_last_row}"
        )
        rotation_range = (
            f"{frame_sheet}!${frame_columns['rotation_error_deg']}$2:"
            f"${frame_columns['rotation_error_deg']}${frame_last_row}"
        )
        baseline_range = (
            f"{frame_sheet}!${frame_columns['absolute_baseline_error_percent']}$2:"
            f"${frame_columns['absolute_baseline_error_percent']}${frame_last_row}"
        )
        common = ",".join([*criteria, status_range, '"OK"'])
        rotation_formula = (
            f"COUNTIFS({common},{rotation_range},\"<\"&'Settings'!$B$2)"
        )
        baseline_formula = (
            f"COUNTIFS({common},{baseline_range},\"<\"&'Settings'!$B$3)"
        )
        both_formula = (
            f"COUNTIFS({common},{rotation_range},\"<\"&'Settings'!$B$2,"
            f"{baseline_range},\"<\"&'Settings'!$B$3)"
        )
        rotation_pass, baseline_pass, both_pass = frame_pass_counts(
            group,
            DEFAULT_ROTATION_THRESHOLD_DEG,
            DEFAULT_BASELINE_THRESHOLD_PERCENT,
        )
        total_frames = max(int(record.get("total_frames") or 0), 0)
        record["rotation_pass_frames"] = ExcelFormula(rotation_formula, rotation_pass)
        record["baseline_pass_frames"] = ExcelFormula(baseline_formula, baseline_pass)
        record["both_pass_frames"] = ExcelFormula(both_formula, both_pass)
        total_cell = f"${summary_columns['total_frames']}{row_index}"
        for count_field, rate_field, count in (
            ("rotation_pass_frames", "rotation_pass_rate", rotation_pass),
            ("baseline_pass_frames", "baseline_pass_rate", baseline_pass),
            ("both_pass_frames", "both_pass_rate", both_pass),
        ):
            count_cell = f"{summary_columns[count_field]}{row_index}"
            record[rate_field] = ExcelFormula(
                f"IFERROR({count_cell}/{total_cell},0)",
                count / total_frames if total_frames else 0.0,
            )


def build_summaries(rows: list[dict]):
    video_groups = defaultdict(list)
    distance_groups = defaultdict(list)
    for row in rows:
        video_key = (
            row["distance_cm"],
            row["angle_deg"],
            row["repeat"],
            row["scene_type"],
            row["video_file"],
        )
        distance_key = (row["distance_cm"], row["angle_deg"])
        video_groups[video_key].append(row)
        distance_groups[distance_key].append(row)

    video_items = sorted(video_groups.items(), key=lambda item: item[0])
    distance_items = sorted(distance_groups.items(), key=lambda item: item[0])
    video_corner = []
    video_rt = []
    video_group_rows = []
    for key, group in video_items:
        distance, angle, repeat, scene_type, video_file = key
        metadata = {
            "distance_cm": distance,
            "angle_deg": angle,
            "repeat": repeat,
            "scene_type": scene_type,
            "video_file": video_file,
        }
        video_corner.append(corner_summary(group, metadata))
        video_rt.append(rt_static_summary(group, metadata))
        video_group_rows.append(group)

    distance_corner = []
    distance_rt = []
    distance_group_rows = []
    for (distance, angle), group in distance_items:
        metadata = {
            "distance_cm": distance,
            "angle_deg": angle,
            "video_count": len({row["video_file"] for row in group}),
        }
        distance_corner.append(corner_summary(group, metadata))
        distance_rt.append(rt_static_summary(group, metadata))
        distance_group_rows.append(group)

    frame_last_row = len(rows) + 1
    add_pass_formulas(
        video_rt,
        video_group_rows,
        VIDEO_RT_HEADERS,
        "video",
        frame_last_row,
    )
    add_pass_formulas(
        distance_rt,
        distance_group_rows,
        DISTANCE_RT_HEADERS,
        "distance",
        frame_last_row,
    )
    return video_corner, video_rt, distance_corner, distance_rt


def clean_xml_text(value) -> str:
    text = "".join(
        char
        for char in str(value)
        if char in "\t\n\r" or ord(char) >= 32
    )
    return escape(text)


def cell_style(sheet_name: str, header: str, row_index: int, value) -> int:
    if row_index == 1:
        return 1
    if sheet_name == "Settings" and header == "value":
        return 5
    if isinstance(value, ExcelFormula) and header.endswith("_rate"):
        return 4
    if header.endswith("_rate"):
        return 4
    if header.endswith("_frames") or header in {
        "repeat",
        "frame_index",
        "marker_id",
        "shared_marker_count",
        "video_count",
        "ippe_branch_left",
        "ippe_branch_right",
    }:
        return 2
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 3
    if isinstance(value, ExcelFormula):
        return 3
    return 0


def worksheet_xml(sheet_name: str, headers: list[str], rows: list[dict]) -> str:
    all_rows = [dict(zip(headers, headers)), *rows]
    last_col = excel_column(max(len(headers), 1))
    last_row = max(len(all_rows), 1)
    widths = []
    for index, header in enumerate(headers, 1):
        samples = [str(row.get(header, "") or "") for row in rows[:200]]
        width = min(max([len(header), *(len(value) for value in samples)]) + 2, 42)
        widths.append(
            f'<col min="{index}" max="{index}" width="{max(width, 11)}" customWidth="1"/>'
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


def write_xlsx(path: Path, sheets: list[tuple[str, list[str], list[dict]]]) -> None:
    sheet_entries = []
    workbook_rels = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, headers, rows) in enumerate(sheets, 1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                worksheet_xml(name, headers, rows),
            )
            sheet_entries.append(
                f'<sheet name="{clean_xml_text(name)}" sheetId="{index}" r:id="rId{index}"/>'
            )
            workbook_rels.append(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        style_rel_id = len(sheets) + 1
        workbook_rels.append(
            f'<Relationship Id="rId{style_rel_id}" '
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
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
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


def write_frame_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).resolve()
    calibration_path = Path(args.calibration).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    videos = collect_videos(folder, args.only_distance, args.only_video)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise FileNotFoundError("No video matched the HBVCAM file-name rule")

    output = (
        Path(args.output).resolve()
        if args.output
        else folder / DEFAULT_OUTPUT_NAME
    )
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)

    k_left, d_left, k_right, d_right, answer = load_calibration(calibration_path)
    eye_size = (SBS_WIDTH // 2, SBS_HEIGHT)
    common_k, map_l1, map_l2, map_r1, map_r2 = build_common_intrinsic_maps(
        k_left,
        d_left,
        k_right,
        d_right,
        eye_size,
    )
    estimator = ArucoCornerRTEstimator(
        common_k,
        args.marker_size_mm,
        args.min_baseline_mm,
        args.max_baseline_mm,
        args.marker_id,
    )

    rows = []
    overall_started = time.perf_counter()
    print(
        f"Found {len(videos)} video(s); frames 0..{args.frames - 1}; "
        "algorithm=ArUco corners + IPPE only"
    )
    for video_number, video in enumerate(videos, 1):
        capture = cv2.VideoCapture(str(video["path"]))
        if not capture.isOpened():
            for frame_index in range(args.frames):
                row = empty_frame_row(video, frame_index)
                row["failure_reason"] = "OSError: unable to open video"
                rows.append(row)
            continue
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        started = time.perf_counter()
        print(
            f"[{video_number}/{len(videos)}] {video['path'].name} "
            f"reported_frames={reported}"
        )
        try:
            for frame_index in range(args.frames):
                ok, sbs = capture.read()
                if not ok or sbs is None:
                    for missing_index in range(frame_index, args.frames):
                        row = empty_frame_row(video, missing_index)
                        row["failure_reason"] = f"EOF before frame {missing_index}"
                        rows.append(row)
                    break
                height, width = sbs.shape[:2]
                if (width, height) != (SBS_WIDTH, SBS_HEIGHT):
                    row = empty_frame_row(video, frame_index)
                    row["failure_reason"] = (
                        f"Unexpected SBS size {width}x{height}; expected "
                        f"{SBS_WIDTH}x{SBS_HEIGHT}"
                    )
                    rows.append(row)
                    continue
                half = width // 2
                left = cv2.remap(sbs[:, :half], map_l1, map_l2, cv2.INTER_LINEAR)
                right = cv2.remap(sbs[:, half:], map_r1, map_r2, cv2.INTER_LINEAR)
                rows.append(
                    analyze_frame(
                        video,
                        frame_index,
                        left,
                        right,
                        estimator,
                        answer,
                    )
                )
                if args.progress_every > 0 and (
                    (frame_index + 1) % args.progress_every == 0
                    or frame_index + 1 == args.frames
                ):
                    print(
                        f"  F{frame_index:03d} complete | "
                        f"elapsed={(time.perf_counter() - started):.1f}s"
                    )
        finally:
            capture.release()

    video_corner, video_rt, distance_corner, distance_rt = build_summaries(rows)
    settings = [
        {
            "parameter": "rotation_error_threshold_deg",
            "value": DEFAULT_ROTATION_THRESHOLD_DEG,
            "unit": "deg",
            "description": "Edit B2; RT summary formula columns update in Excel.",
        },
        {
            "parameter": "absolute_baseline_error_threshold_percent",
            "value": DEFAULT_BASELINE_THRESHOLD_PERCENT,
            "unit": "%",
            "description": "Edit B3; values in Frame Results use percentage points (5 means 5%).",
        },
    ]
    protocol = [
        {"parameter": "generated_at", "value": datetime.now(timezone.utc).astimezone().isoformat()},
        {"parameter": "source_folder", "value": str(folder)},
        {"parameter": "calibration", "value": str(calibration_path)},
        {"parameter": "frames", "value": f"0..{args.frames - 1}"},
        {"parameter": "marker_size_mm", "value": args.marker_size_mm},
        {"parameter": "dictionary", "value": "DICT_4X4_100"},
        {"parameter": "corner_refinement", "value": "cornerSubPix window 5x5, max 100, epsilon 0.0001"},
        {"parameter": "image_coordinates", "value": "undistorted common-intrinsic 1920x1080 eye images"},
        {"parameter": "RT inputs", "value": "four left + four right ArUco corners only; no SIFT"},
        {"parameter": "IPPE branch selection", "value": "minimum combined marker reprojection RMS within baseline gate"},
        {"parameter": "baseline_gate_mm", "value": f"{args.min_baseline_mm:g}..{args.max_baseline_mm:g}"},
        {"parameter": "JSON use", "value": "error calculation only; not used for RT or IPPE branch selection"},
        {"parameter": "rotation_error", "value": "acos(clamp((trace(R_est*R_json^T)-1)/2))*180/pi"},
        {"parameter": "baseline_error_percent", "value": "abs(||t_est||-||t_json||)/||t_json||*100"},
        {"parameter": "std definition", "value": "sample standard deviation (ddof=1)"},
        {"parameter": "corner std caution", "value": "includes real camera/pattern motion; pure detection jitter requires a static scene"},
    ]

    write_xlsx(
        output,
        [
            ("Settings", ["parameter", "value", "unit", "description"], settings),
            ("Frame Results", FRAME_FIELDS, rows),
            ("Video Corner Stats", VIDEO_CORNER_HEADERS, video_corner),
            ("Video RT Stats", VIDEO_RT_HEADERS, video_rt),
            ("Distance Corner Stats", DISTANCE_CORNER_HEADERS, distance_corner),
            ("Distance RT Stats", DISTANCE_RT_HEADERS, distance_rt),
            ("Protocol", ["parameter", "value"], protocol),
        ],
    )
    csv_path = output.with_name(output.stem + "_frame_results.csv")
    write_frame_csv(csv_path, rows)
    print(f"Excel: {output}")
    print(f"Frame CSV: {csv_path}")
    print(f"Elapsed: {(time.perf_counter() - overall_started):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
