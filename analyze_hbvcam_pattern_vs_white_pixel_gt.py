"""Compare ArUco subpixel corners against a screen-drawn ground truth.

The program accepts either a folder of automatically paired SBS videos or one
explicit pattern/white pair recorded without moving the stereo camera:

1. A video displaying one ArUco pattern.
2. A video replacing the four ArUco corners with either white blobs or line
   intersections.

The videos do not need to be temporally synchronized.  ArUco corners are first
detected/refined in every requested pattern frame.  Their cross-frame median
positions define the search centers in the white-pixel video.  A white screen
pixel generally covers multiple camera sensor pixels because of optics, focus,
and display sampling; its position is therefore measured as the
background-subtracted intensity-weighted centroid of the selected bright
connected component.  The alternate mode fits the two screen lines and uses
their subpixel intersection.  Cross-frame median detected points form the
fixed GT.

JSON extrinsics are used only after RT estimation to calculate answer errors.
They never participate in ArUco detection, white-blob detection, IPPE branch
selection, or RT estimation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from analyze_hbvcam_4pattern_subset_rt import (
    AdaptiveArucoDetector,
    MultiMarkerRTEstimator,
    calculate_answer_errors,
    finite,
    load_calibration,
    mean,
    median,
    rotation_angle_deg,
    split_sbs,
    std,
)
from analyze_hbvcam_aruco_corner_rt_stability import ExcelFormula, excel_column, write_xlsx


ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = ROOT / "calibration_result_HBVCAM_4M2214HD-2-v11.json"

# Select the default GT point detector here:
#   "white_blob_centroid" -> intensity-weighted center of each white region
#   "cross_intersection"  -> fitted intersection of the two screen lines
# The same choice can be overridden once with --gt-mode on the command line.
GT_DETECTION_MODE = "cross_intersection"

# Cross-intersection detector controls.  The new screen videos use dark lines
# on a bright background; change polarity to "bright" for bright cross lines.
CROSS_LINE_POLARITY = "dark"
CROSS_THRESHOLD_RATIO = 0.35
CROSS_MIN_CONTRAST_GRAY = 12.0
CROSS_LINE_FIT_BAND_PX = 4.0
CROSS_MIN_ARM_SUPPORT_PIXELS = 8
CROSS_PREVIEW_ARM_PX = 5
CROSS_PREVIEW_LINE_WIDTH_PX = 1
CROSS_PREVIEW_JPEG_QUALITY = 95
PATTERN_RANSAC_REPROJECTION_THRESHOLD_PX = 6.0

MARKER_SIZE_MM = 8.25
MIN_BASELINE_MM = 8.0
MAX_BASELINE_MM = 220.0
SUBPIX_STABILITY_MAX_RAW_PX = 2.0
DIAGNOSTIC_TILE_SIZE = 240
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


PATTERN_FRAME_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "frame_index",
    "marker_id",
    "status",
    "failure_reason",
    "left_subpixel_half_window",
    "left_subpixel_stability_max_raw_px",
    "right_subpixel_half_window",
    "right_subpixel_stability_max_raw_px",
    "left_c0_x_px",
    "left_c0_y_px",
    "left_c1_x_px",
    "left_c1_y_px",
    "left_c2_x_px",
    "left_c2_y_px",
    "left_c3_x_px",
    "left_c3_y_px",
    "right_c0_x_px",
    "right_c0_y_px",
    "right_c1_x_px",
    "right_c1_y_px",
    "right_c2_x_px",
    "right_c2_y_px",
    "right_c3_x_px",
    "right_c3_y_px",
    "relative_rotation_angle_deg",
    "rotation_error_json_deg",
    "algorithm_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_json_mm",
    "absolute_baseline_delta_json_mm",
    "baseline_error_json_percent",
    "absolute_baseline_error_json_percent",
    "translation_l2_error_json_mm",
    "translation_direction_error_json_deg",
    "marker_transfer_combined_rms_px",
    "marker_transfer_max_px",
    "selected_ippe_branches",
    "processing_time_ms",
]


WHITE_FRAME_FIELDS = [
    "distance_cm",
    "white_video_file",
    "gt_detection_mode",
    "frame_index",
    "status",
    "failure_reason",
    "detected_corner_count",
    "left_c0_x_px",
    "left_c0_y_px",
    "left_c1_x_px",
    "left_c1_y_px",
    "left_c2_x_px",
    "left_c2_y_px",
    "left_c3_x_px",
    "left_c3_y_px",
    "right_c0_x_px",
    "right_c0_y_px",
    "right_c1_x_px",
    "right_c1_y_px",
    "right_c2_x_px",
    "right_c2_y_px",
    "right_c3_x_px",
    "right_c3_y_px",
    "relative_rotation_angle_deg",
    "rotation_error_json_deg",
    "algorithm_baseline_mm",
    "json_baseline_mm",
    "baseline_delta_json_mm",
    "absolute_baseline_delta_json_mm",
    "baseline_error_json_percent",
    "absolute_baseline_error_json_percent",
    "translation_l2_error_json_mm",
    "translation_direction_error_json_deg",
    "marker_transfer_combined_rms_px",
    "marker_transfer_max_px",
    "selected_ippe_branches",
    "diagnostic_video",
    "cross_preview_jpg",
    "processing_time_ms",
]


FRAME_COMPARISON_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "pattern_frame_index",
    "pattern_status",
    "white_same_index_status",
    "pattern_rotation_error_json_deg",
    "pattern_baseline_mm",
    "pattern_absolute_baseline_error_json_mm",
    "pattern_absolute_baseline_error_json_percent",
    "white_same_index_rotation_error_json_deg",
    "white_same_index_baseline_mm",
    "white_same_index_absolute_baseline_error_json_mm",
    "white_same_index_absolute_baseline_error_json_percent",
    "gt_median_rotation_error_json_deg",
    "gt_median_baseline_mm",
    "gt_median_absolute_baseline_error_json_mm",
    "gt_median_absolute_baseline_error_json_percent",
    "rotation_pattern_vs_gt_median_deg",
    "translation_pattern_vs_gt_median_l2_mm",
    "translation_direction_pattern_vs_gt_median_deg",
    "baseline_pattern_minus_gt_median_mm",
    "absolute_baseline_pattern_vs_gt_median_mm",
    "absolute_baseline_pattern_vs_gt_median_percent",
    "left_corner_error_to_gt_median_mean_px",
    "left_corner_error_to_gt_median_rms_px",
    "left_corner_error_to_gt_median_max_px",
    "right_corner_error_to_gt_median_mean_px",
    "right_corner_error_to_gt_median_rms_px",
    "right_corner_error_to_gt_median_max_px",
    "combined_corner_error_to_gt_median_mean_px",
    "combined_corner_error_to_gt_median_rms_px",
    "combined_corner_error_to_gt_median_max_px",
]


CORNER_COMPARISON_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "gt_detection_mode",
    "frame_index",
    "side",
    "corner_index",
    "pattern_x_px",
    "pattern_y_px",
    "white_same_index_x_px",
    "white_same_index_y_px",
    "gt_median_x_px",
    "gt_median_y_px",
    "pattern_vs_white_same_index_error_px",
    "pattern_vs_gt_median_error_px",
    "pattern_mean_side_length_px",
    "normalized_corner_error_percent",
    "white_same_index_vs_gt_median_error_px",
    "white_peak_gray",
    "white_background_gray",
    "white_contrast_gray",
    "white_blob_area_px",
    "white_distance_from_pattern_search_center_px",
    "cross_line1_support_pixels",
    "cross_line2_support_pixels",
    "cross_line_fit_rms_px",
]


SUMMARY_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "marker_id",
    "gt_detection_mode",
    "requested_pattern_frames",
    "valid_pattern_corner_frames",
    "valid_pattern_rt_frames",
    "requested_white_frames",
    "valid_white_corner_frames",
    "valid_white_rt_frames",
    "pattern_left_corner_coordinate_std_rms_px",
    "pattern_right_corner_coordinate_std_rms_px",
    "white_left_corner_coordinate_std_rms_px",
    "white_right_corner_coordinate_std_rms_px",
    "pattern_vs_gt_corner_error_mean_px",
    "pattern_vs_gt_corner_error_rms_px",
    "pattern_vs_gt_corner_error_std_px",
    "pattern_vs_gt_corner_error_max_px",
    "pattern_vs_gt_normalized_corner_error_mean_percent",
    "pattern_vs_gt_normalized_corner_error_std_percent",
    "pattern_rotation_error_json_mean_deg",
    "pattern_rotation_error_json_std_deg",
    "pattern_absolute_baseline_error_json_mean_mm",
    "pattern_absolute_baseline_error_json_std_mm",
    "pattern_absolute_baseline_error_json_mean_percent",
    "pattern_absolute_baseline_error_json_std_percent",
    "white_rotation_error_json_mean_deg",
    "white_rotation_error_json_std_deg",
    "white_absolute_baseline_error_json_mean_mm",
    "white_absolute_baseline_error_json_std_mm",
    "white_absolute_baseline_error_json_mean_percent",
    "white_absolute_baseline_error_json_std_percent",
    "gt_median_rotation_error_json_deg",
    "gt_median_baseline_mm",
    "gt_median_absolute_baseline_error_json_mm",
    "gt_median_absolute_baseline_error_json_percent",
    "pattern_vs_gt_rotation_mean_deg",
    "pattern_vs_gt_rotation_std_deg",
    "pattern_vs_gt_translation_l2_mean_mm",
    "pattern_vs_gt_translation_l2_std_mm",
    "pattern_vs_gt_absolute_baseline_mean_mm",
    "pattern_vs_gt_absolute_baseline_std_mm",
    "pattern_vs_gt_absolute_baseline_mean_percent",
    "pattern_vs_gt_absolute_baseline_std_percent",
]


DISTANCE_SUMMARY_FIELDS = [
    "distance_cm",
    "pair_count",
    "pattern_frame_count",
    "valid_pattern_corner_frame_count",
    "corner_error_sample_count",
    "left_corner_error_mean_px",
    "right_corner_error_mean_px",
    "combined_corner_error_mean_px",
    "combined_corner_error_rms_px",
    "combined_corner_error_std_px",
    "combined_corner_error_p95_px",
    "combined_corner_error_max_px",
    "normalized_corner_error_sample_count",
    "normalized_corner_error_mean_percent",
    "normalized_corner_error_std_percent",
    "pattern_rotation_error_json_mean_deg",
    "pattern_absolute_baseline_error_json_mean_percent",
    "pattern_vs_gt_rotation_mean_deg",
    "pattern_vs_gt_absolute_baseline_mean_mm",
    "pattern_vs_gt_absolute_baseline_mean_percent",
]


DISTANCE_CHART_FIELDS = [
    "distance_cm",
    "corner_error_mean_px",
    "corner_error_std_px",
    "normalized_corner_error_mean_percent",
]


BATCH_ERROR_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ArUco subpixel corners against a screen-drawn GT. "
            "INPUT may be a folder containing *_pattern/white video pairs, "
            "or a pattern video followed by its white video."
        )
    )
    parser.add_argument("input_path", help="Input folder, or one pattern video")
    parser.add_argument(
        "white_video",
        nargs="?",
        help="White video when INPUT is one explicit pattern video",
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--marker-id", type=int)
    parser.add_argument("--marker-size-mm", type=float, default=MARKER_SIZE_MM)
    parser.add_argument("--min-baseline-mm", type=float, default=MIN_BASELINE_MM)
    parser.add_argument("--max-baseline-mm", type=float, default=MAX_BASELINE_MM)
    parser.add_argument(
        "--subpixel-stability-max-raw-px",
        type=float,
        default=SUBPIX_STABILITY_MAX_RAW_PX,
    )
    parser.add_argument("--white-search-radius-px", type=int, default=20)
    parser.add_argument("--white-threshold-ratio", type=float, default=0.5)
    parser.add_argument("--white-min-contrast-gray", type=float, default=30.0)
    parser.add_argument("--white-min-area-px", type=int, default=2)
    parser.add_argument("--white-max-area-px", type=int, default=300)
    parser.add_argument(
        "--gt-mode",
        choices=("white_blob_centroid", "cross_intersection"),
        default=GT_DETECTION_MODE,
        help=(
            "GT detector. Default comes from GT_DETECTION_MODE near the top of "
            "this file."
        ),
    )
    parser.add_argument("--cross-arm-px", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Process only the first N auto-paired distances (useful for preview)",
    )
    parser.add_argument("--output", help="Output .xlsx path")
    parser.add_argument(
        "--diagnostic-video-dir",
        help="Output folder for one stacked diagnostic MP4 per video pair",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be positive")
    if args.min_baseline_mm <= 0 or args.max_baseline_mm <= args.min_baseline_mm:
        parser.error("baseline limits are invalid")
    if args.subpixel_stability_max_raw_px <= 0:
        parser.error("--subpixel-stability-max-raw-px must be positive")
    if args.white_search_radius_px < 3:
        parser.error("--white-search-radius-px must be at least 3")
    if not 0.05 <= args.white_threshold_ratio <= 0.95:
        parser.error("--white-threshold-ratio must be between 0.05 and 0.95")
    if args.white_min_contrast_gray <= 0:
        parser.error("--white-min-contrast-gray must be positive")
    if args.white_min_area_px <= 0 or args.white_max_area_px < args.white_min_area_px:
        parser.error("white blob area limits are invalid")
    if args.cross_arm_px < 1:
        parser.error("--cross-arm-px must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    return args


def parse_distance(path: Path) -> float | None:
    match = re.search(r"(?P<distance>\d+(?:\.\d+)?)\s*cm", path.stem, re.I)
    return float(match.group("distance")) if match else None


def collect_video_pairs(
    input_path: Path,
    explicit_white: Path | None,
    recursive: bool,
    max_pairs: int | None,
) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        if explicit_white is None:
            raise ValueError("A white video is required when INPUT is a video file")
        if not explicit_white.is_file():
            raise FileNotFoundError(f"White video does not exist: {explicit_white}")
        return [(input_path, explicit_white)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder/video does not exist: {input_path}")
    if explicit_white is not None:
        raise ValueError("Do not provide WHITE_VIDEO when INPUT is a folder")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    videos = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    by_parent_and_name = {
        (str(path.parent).lower(), path.name.lower()): path for path in videos
    }
    pairs = []
    missing = []
    for pattern_video in videos:
        if re.search(r"_pattern$", pattern_video.stem, re.I) is None:
            continue
        white_stem = re.sub(
            r"_pattern$", "_white", pattern_video.stem, flags=re.I
        )
        white_video = None
        for extension in (pattern_video.suffix, *sorted(VIDEO_EXTENSIONS)):
            key = (
                str(pattern_video.parent).lower(),
                (white_stem + extension).lower(),
            )
            if key in by_parent_and_name:
                white_video = by_parent_and_name[key]
                break
        if white_video is None:
            missing.append(pattern_video)
            continue
        pairs.append((pattern_video, white_video))
    pairs.sort(
        key=lambda pair: (
            parse_distance(pair[0]) is None,
            parse_distance(pair[0]) if parse_distance(pair[0]) is not None else math.inf,
            pair[0].name.lower(),
        )
    )
    for path in missing:
        print(f"Warning: no matching *_white video, skipped: {path.name}")
    if not pairs:
        raise RuntimeError(
            "No *_pattern video with a matching *_white video was found"
        )
    return pairs[:max_pairs] if max_pairs is not None else pairs


def blank_row(fields) -> dict:
    return {field: None for field in fields}


def add_points(row, prefix: str, points) -> None:
    if points is None:
        return
    for index, (x, y) in enumerate(np.asarray(points).reshape(4, 2)):
        row[f"{prefix}_c{index}_x_px"] = float(x)
        row[f"{prefix}_c{index}_y_px"] = float(y)


def remap_error_names(errors: dict, prefix: str = "") -> dict:
    mapping = {
        "relative_rotation_angle_deg": "relative_rotation_angle_deg",
        "rotation_error_deg": "rotation_error_json_deg",
        "algorithm_baseline_mm": "algorithm_baseline_mm",
        "json_baseline_mm": "json_baseline_mm",
        "baseline_delta_mm": "baseline_delta_json_mm",
        "absolute_baseline_delta_mm": "absolute_baseline_delta_json_mm",
        "baseline_error_percent": "baseline_error_json_percent",
        "absolute_baseline_error_percent": "absolute_baseline_error_json_percent",
        "translation_l2_error_mm": "translation_l2_error_json_mm",
        "translation_direction_error_deg": "translation_direction_error_json_deg",
    }
    return {prefix + mapping[key]: value for key, value in errors.items() if key in mapping}


def discover_marker_id(pattern_video: Path, detector, requested_frames: int) -> int:
    capture = cv2.VideoCapture(str(pattern_video))
    counts = Counter()
    try:
        for _frame_index in range(min(requested_frames, 30)):
            ok, frame = capture.read()
            if not ok:
                break
            left, right = split_sbs(frame)
            left_markers, _ = detector.detect(left)
            right_markers, _ = detector.detect(right)
            counts.update(set(left_markers) & set(right_markers))
    finally:
        capture.release()
    if not counts:
        raise RuntimeError("No shared stable ArUco ID was found in the pattern video")
    marker_id, count = counts.most_common(1)[0]
    print(f"Automatically selected ArUco ID {marker_id} ({count} discovery frames)")
    return int(marker_id)


class WhiteBlobDetector:
    def __init__(
        self,
        search_radius_px,
        threshold_ratio,
        min_contrast_gray,
        min_area_px,
        max_area_px,
    ):
        self.radius = int(search_radius_px)
        self.threshold_ratio = float(threshold_ratio)
        self.min_contrast = float(min_contrast_gray)
        self.min_area = int(min_area_px)
        self.max_area = int(max_area_px)

    def detect_one(self, gray: np.ndarray, search_center) -> dict:
        x, y = np.asarray(search_center, dtype=np.float64).reshape(2)
        center_x, center_y = int(round(x)), int(round(y))
        x0 = max(0, center_x - self.radius)
        y0 = max(0, center_y - self.radius)
        x1 = min(gray.shape[1], center_x + self.radius + 1)
        y1 = min(gray.shape[0], center_y + self.radius + 1)
        roi = gray[y0:y1, x0:x1].astype(np.float32)
        if roi.size == 0:
            return {"accepted": False, "failure_reason": "empty search ROI"}
        background = float(np.percentile(roi, 35))
        peak = float(np.max(roi))
        contrast = peak - background
        threshold = background + self.threshold_ratio * contrast
        if contrast < self.min_contrast:
            return {
                "accepted": False,
                "failure_reason": f"contrast {contrast:.1f} < {self.min_contrast:.1f}",
                "background_gray": background,
                "peak_gray": peak,
                "contrast_gray": contrast,
            }
        mask = (roi >= threshold).astype(np.uint8)
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates = []
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if not self.min_area <= area <= self.max_area:
                continue
            ys, xs = np.where(labels == component)
            weights = np.maximum(roi[ys, xs] - background, 0.0)
            flux = float(np.sum(weights))
            if flux <= 0:
                continue
            centroid_x = x0 + float(np.sum(xs * weights) / flux)
            centroid_y = y0 + float(np.sum(ys * weights) / flux)
            distance = float(math.hypot(centroid_x - x, centroid_y - y))
            component_peak = float(np.max(roi[ys, xs]))
            score = flux / (1.0 + 0.25 * distance)
            candidates.append(
                (
                    score,
                    -distance,
                    {
                        "accepted": True,
                        "x": centroid_x,
                        "y": centroid_y,
                        "peak_gray": component_peak,
                        "background_gray": background,
                        "contrast_gray": component_peak - background,
                        "blob_area_px": area,
                        "distance_from_search_center_px": distance,
                        "search_bounds": (x0, y0, x1, y1),
                    },
                )
            )
        if not candidates:
            return {
                "accepted": False,
                "failure_reason": "no bright connected component passed area limits",
                "background_gray": background,
                "peak_gray": peak,
                "contrast_gray": contrast,
                "search_bounds": (x0, y0, x1, y1),
            }
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def detect_four(self, frame: np.ndarray, search_centers) -> tuple[np.ndarray | None, list[dict]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diagnostics = [self.detect_one(gray, center) for center in search_centers]
        for item in diagnostics:
            item["detector_mode"] = "white_blob_centroid"
        if not all(item.get("accepted") for item in diagnostics):
            return None, diagnostics
        points = np.asarray(
            [[item["x"], item["y"]] for item in diagnostics], dtype=np.float64
        )
        return points, diagnostics


class CrossIntersectionDetector:
    """Fit two locally known screen-line directions and return their intersection."""

    def __init__(
        self,
        search_radius_px,
        threshold_ratio=CROSS_THRESHOLD_RATIO,
        min_contrast_gray=CROSS_MIN_CONTRAST_GRAY,
        line_fit_band_px=CROSS_LINE_FIT_BAND_PX,
        min_arm_support_pixels=CROSS_MIN_ARM_SUPPORT_PIXELS,
        polarity=CROSS_LINE_POLARITY,
    ):
        self.radius = int(search_radius_px)
        self.threshold_ratio = float(threshold_ratio)
        self.min_contrast = float(min_contrast_gray)
        self.line_fit_band = float(line_fit_band_px)
        self.min_arm_support = int(min_arm_support_pixels)
        self.polarity = str(polarity).strip().lower()
        if self.polarity not in {"dark", "bright"}:
            raise ValueError("CROSS_LINE_POLARITY must be 'dark' or 'bright'")

    @staticmethod
    def _unit(vector) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(2)
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            raise ValueError("degenerate marker edge direction")
        return vector / length

    @classmethod
    def _corner_directions(cls, corners, corner_index) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        center = points[corner_index]
        previous = cls._unit(points[(corner_index - 1) % 4] - center)
        following = cls._unit(points[(corner_index + 1) % 4] - center)
        cross_value = previous[0] * following[1] - previous[1] * following[0]
        if abs(float(cross_value)) < 0.15:
            raise ValueError("screen-line directions are nearly parallel")
        return previous, following

    def _line_signal(self, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        roi = roi.astype(np.float32)
        if self.polarity == "dark":
            background = float(np.percentile(roi, 65))
            extreme = float(np.min(roi))
            contrast = background - extreme
            signal = np.maximum(background - roi, 0.0)
            threshold = background - self.threshold_ratio * contrast
            mask = roi <= threshold
        else:
            background = float(np.percentile(roi, 35))
            extreme = float(np.max(roi))
            contrast = extreme - background
            signal = np.maximum(roi - background, 0.0)
            threshold = background + self.threshold_ratio * contrast
            mask = roi >= threshold
        return signal, mask, background, extreme, contrast

    def detect_one(self, gray: np.ndarray, search_center, directions) -> dict:
        center = np.asarray(search_center, dtype=np.float64).reshape(2)
        center_x, center_y = (int(round(value)) for value in center)
        x0 = max(0, center_x - self.radius)
        y0 = max(0, center_y - self.radius)
        x1 = min(gray.shape[1], center_x + self.radius + 1)
        y1 = min(gray.shape[0], center_y + self.radius + 1)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return {"accepted": False, "failure_reason": "empty search ROI"}
        signal, mask, background, extreme, contrast = self._line_signal(roi)
        common = {
            "detector_mode": "cross_intersection",
            "background_gray": background,
            "peak_gray": extreme,
            "contrast_gray": contrast,
            "search_bounds": (x0, y0, x1, y1),
            "blob_area_px": int(np.count_nonzero(mask)),
        }
        if contrast < self.min_contrast:
            return {
                **common,
                "accepted": False,
                "failure_reason": f"line contrast {contrast:.1f} < {self.min_contrast:.1f}",
            }

        ys, xs = np.where(mask)
        if len(xs) < 2 * self.min_arm_support:
            return {
                **common,
                "accepted": False,
                "failure_reason": "too few thresholded line pixels",
            }
        pixels = np.column_stack((xs + x0, ys + y0)).astype(np.float64)
        weights = signal[ys, xs].astype(np.float64)
        direction1, direction2 = (
            self._unit(directions[0]),
            self._unit(directions[1]),
        )
        normal1 = np.array([-direction1[1], direction1[0]], dtype=np.float64)
        normal2 = np.array([-direction2[1], direction2[0]], dtype=np.float64)
        normal_matrix = np.vstack((normal1, normal2))
        if abs(float(np.linalg.det(normal_matrix))) < 0.15:
            return {
                **common,
                "accepted": False,
                "failure_reason": "screen-line normals are nearly parallel",
            }

        intersection = center.copy()
        arm_masks = None
        for _iteration in range(5):
            relative = pixels - intersection
            distance1 = np.abs(relative @ normal1)
            distance2 = np.abs(relative @ normal2)
            arm1 = (distance1 <= self.line_fit_band) & (distance1 <= distance2)
            arm2 = (distance2 <= self.line_fit_band) & (distance2 < distance1)
            if np.count_nonzero(arm1) < self.min_arm_support or np.count_nonzero(arm2) < self.min_arm_support:
                return {
                    **common,
                    "accepted": False,
                    "failure_reason": (
                        "insufficient pixels on one fitted cross arm: "
                        f"{np.count_nonzero(arm1)}/{np.count_nonzero(arm2)}"
                    ),
                }
            offsets = []
            for arm, normal in ((arm1, normal1), (arm2, normal2)):
                arm_weights = weights[arm]
                signed = (pixels[arm] - intersection) @ normal
                offsets.append(float(np.average(signed, weights=arm_weights)))
            delta = np.linalg.solve(normal_matrix, np.asarray(offsets))
            intersection += delta
            arm_masks = (arm1, arm2)
            if float(np.linalg.norm(delta)) < 1e-4:
                break

        distance_from_center = float(np.linalg.norm(intersection - center))
        if distance_from_center > self.radius * 0.75:
            return {
                **common,
                "accepted": False,
                "failure_reason": (
                    f"intersection shift {distance_from_center:.2f} px exceeds search gate"
                ),
            }
        arm1, arm2 = arm_masks
        residuals = np.concatenate(
            (
                (pixels[arm1] - intersection) @ normal1,
                (pixels[arm2] - intersection) @ normal2,
            )
        )
        residual_weights = np.concatenate((weights[arm1], weights[arm2]))
        line_fit_rms = float(
            np.sqrt(np.average(np.square(residuals), weights=residual_weights))
        )
        return {
            **common,
            "accepted": True,
            "x": float(intersection[0]),
            "y": float(intersection[1]),
            "distance_from_search_center_px": distance_from_center,
            "line1_support_pixels": int(np.count_nonzero(arm1)),
            "line2_support_pixels": int(np.count_nonzero(arm2)),
            "line_fit_rms_px": line_fit_rms,
        }

    def detect_four(self, frame: np.ndarray, search_centers) -> tuple[np.ndarray | None, list[dict]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centers = np.asarray(search_centers, dtype=np.float64).reshape(4, 2)
        diagnostics = []
        for corner_index, center in enumerate(centers):
            try:
                directions = self._corner_directions(centers, corner_index)
                result = self.detect_one(gray, center, directions)
            except Exception as exc:
                result = {
                    "accepted": False,
                    "detector_mode": "cross_intersection",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            diagnostics.append(result)
        if not all(item.get("accepted") for item in diagnostics):
            return None, diagnostics
        points = np.asarray(
            [[item["x"], item["y"]] for item in diagnostics], dtype=np.float64
        )
        return points, diagnostics


def estimate_single_marker_rt(estimator, marker_id, left_points, right_points, answer):
    result = estimator.estimate_subset(
        (marker_id,),
        {marker_id: np.asarray(left_points, dtype=np.float64)},
        {marker_id: np.asarray(right_points, dtype=np.float64)},
    )
    output = {
        "R_rel": result["R_rel"],
        "t_rel": result["t_rel"],
        "marker_transfer_combined_rms_px": result.get(
            "marker_transfer_combined_rms_px"
        ),
        "marker_transfer_max_px": result.get("marker_transfer_max_px"),
        "selected_ippe_branches": result.get("selected_ippe_branches"),
    }
    output.update(calculate_answer_errors(result["R_rel"], result["t_rel"], answer))
    return output


def process_pattern_video(
    video,
    requested_frames,
    marker_id,
    detector,
    estimator,
    answer,
    progress_every,
):
    rows = []
    poses = {}
    corners_by_frame = {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open pattern video: {video}")
    distance = parse_distance(video)
    try:
        for frame_index in range(requested_frames):
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            row = blank_row(PATTERN_FRAME_FIELDS)
            row.update(
                {
                    "distance_cm": distance,
                    "pattern_video_file": video.name,
                    "frame_index": frame_index,
                    "marker_id": marker_id,
                    "status": "FAILED",
                    "failure_reason": "",
                }
            )
            try:
                left, right = split_sbs(frame)
                left_markers, left_diag = detector.detect(left)
                right_markers, right_diag = detector.detect(right)
                left_info = left_diag.get(marker_id, {})
                right_info = right_diag.get(marker_id, {})
                row["left_subpixel_half_window"] = left_info.get("half_window")
                row["left_subpixel_stability_max_raw_px"] = left_info.get(
                    "stability_max_raw_px"
                )
                row["right_subpixel_half_window"] = right_info.get("half_window")
                row["right_subpixel_stability_max_raw_px"] = right_info.get(
                    "stability_max_raw_px"
                )
                if marker_id not in left_markers or marker_id not in right_markers:
                    raise RuntimeError(f"ArUco ID {marker_id} was not accepted in both eyes")
                left_points = left_markers[marker_id]
                right_points = right_markers[marker_id]
                corners_by_frame[frame_index] = {
                    "left": left_points,
                    "right": right_points,
                }
                add_points(row, "left", left_points)
                add_points(row, "right", right_points)
                estimate = estimate_single_marker_rt(
                    estimator, marker_id, left_points, right_points, answer
                )
                poses[frame_index] = {
                    "R": estimate["R_rel"],
                    "t": estimate["t_rel"],
                }
                row.update(remap_error_names(estimate))
                for field in (
                    "marker_transfer_combined_rms_px",
                    "marker_transfer_max_px",
                    "selected_ippe_branches",
                ):
                    row[field] = estimate.get(field)
                row["status"] = "OK"
            except Exception as exc:
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            if progress_every > 0 and (frame_index + 1) % progress_every == 0:
                print(f"  Pattern F{frame_index:03d} complete")
    finally:
        capture.release()
    if not corners_by_frame:
        raise RuntimeError("No valid shared ArUco corners were obtained")
    search_centers = {
        side: np.median(
            np.stack([entry[side] for entry in corners_by_frame.values()]), axis=0
        )
        for side in ("left", "right")
    }
    return rows, corners_by_frame, poses, search_centers


def crop_diagnostic_tile(frame, search_center, result, label, tile_size=DIAGNOSTIC_TILE_SIZE):
    radius = max(3, int(result.get("search_radius_px", 20)))
    x, y = np.asarray(search_center, dtype=np.float64)
    center_x, center_y = int(round(x)), int(round(y))
    x0, y0 = center_x - radius, center_y - radius
    x1, y1 = center_x + radius + 1, center_y + radius + 1
    crop = np.zeros((2 * radius + 1, 2 * radius + 1, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    tile = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
    if result.get("accepted"):
        scale = tile_size / float(2 * radius + 1)
        px = int(round((result["x"] - x0) * scale))
        py = int(round((result["y"] - y0) * scale))
        arm = int(result.get("cross_arm_px", 5))
        color = (0, 0, 255)
        cv2.line(tile, (px - arm, py), (px + arm, py), color, 1, cv2.LINE_8)
        cv2.line(tile, (px, py - arm), (px, py + arm), color, 1, cv2.LINE_8)
        status = f"OK ({result['x']:.2f},{result['y']:.2f})"
    else:
        status = "MISSING"
    cv2.rectangle(tile, (0, 0), (tile_size - 1, 22), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{label} {status}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def white_diagnostic_image(
    left,
    right,
    search_centers,
    left_results,
    right_results,
    frame_index,
    search_radius,
    cross_arm,
):
    def panel(frame, side, results):
        tiles = []
        for corner_index, (center, result) in enumerate(zip(search_centers[side], results)):
            result = dict(result)
            result["search_radius_px"] = search_radius
            result["cross_arm_px"] = cross_arm
            tiles.append(
                crop_diagnostic_tile(
                    frame,
                    center,
                    result,
                    f"{side.upper()} C{corner_index} F{frame_index}",
                )
            )
        return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))

    left_panel = panel(left, "left", left_results)
    right_panel = panel(right, "right", right_results)
    combined = np.hstack((left_panel, right_panel))
    cv2.line(
        combined,
        (left_panel.shape[1], 0),
        (left_panel.shape[1], combined.shape[0] - 1),
        (255, 255, 255),
        1,
    )
    return combined


def pattern_diagnostic_image(
    left,
    right,
    search_centers,
    corners,
    frame_index,
    search_radius,
    cross_arm,
):
    """Build the same 2x2-per-eye view as the white diagnostic panel."""

    def panel(frame, side):
        points = corners.get(side) if corners is not None else None
        tiles = []
        for corner_index, center in enumerate(search_centers[side]):
            result = {
                "accepted": points is not None,
                "search_radius_px": search_radius,
                "cross_arm_px": cross_arm,
            }
            if points is not None:
                result["x"] = float(points[corner_index, 0])
                result["y"] = float(points[corner_index, 1])
            tiles.append(
                crop_diagnostic_tile(
                    frame,
                    center,
                    result,
                    f"{side.upper()} C{corner_index} F{frame_index}",
                )
            )
        return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))

    left_panel = panel(left, "left")
    right_panel = panel(right, "right")
    combined = np.hstack((left_panel, right_panel))
    cv2.line(
        combined,
        (left_panel.shape[1], 0),
        (left_panel.shape[1], combined.shape[0] - 1),
        (255, 255, 255),
        1,
    )
    return combined


def open_diagnostic_writer(path: Path, fps: float, frame_size) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if math.isfinite(fps) and fps > 0 else 25.0,
        tuple(int(value) for value in frame_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open diagnostic MP4 writer: {path}")
    return writer


def save_cross_intersection_preview(
    frame,
    left_points,
    right_points,
    path: Path,
    arm_px=CROSS_PREVIEW_ARM_PX,
    line_width_px=CROSS_PREVIEW_LINE_WIDTH_PX,
):
    """Save the original SBS frame with eight small marks at detected GT points."""
    preview = frame.copy()
    left_width = preview.shape[1] // 2
    point_groups = (
        np.asarray(left_points, dtype=np.float64).reshape(4, 2),
        np.asarray(right_points, dtype=np.float64).reshape(4, 2)
        + np.array([left_width, 0.0], dtype=np.float64),
    )
    for points in point_groups:
        for x, y in points:
            px, py = int(round(float(x))), int(round(float(y)))
            cv2.line(
                preview,
                (px - arm_px, py),
                (px + arm_px, py),
                (0, 0, 255),
                line_width_px,
                cv2.LINE_8,
            )
            cv2.line(
                preview,
                (px, py - arm_px),
                (px, py + arm_px),
                (0, 0, 255),
                line_width_px,
                cv2.LINE_8,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, CROSS_PREVIEW_JPEG_QUALITY],
    ):
        raise RuntimeError(f"Could not save cross-intersection preview JPG: {path}")


def process_white_video(
    pattern_video,
    white_video,
    requested_frames,
    marker_id,
    gt_detector,
    estimator,
    answer,
    search_centers,
    pattern_corners,
    diagnostic_video,
    cross_arm,
    progress_every,
):
    rows = []
    poses = {}
    corners_by_frame = {}
    diagnostics_by_frame = {}
    cross_preview_path = None
    white_capture = cv2.VideoCapture(str(white_video))
    pattern_capture = cv2.VideoCapture(str(pattern_video))
    if not white_capture.isOpened():
        raise RuntimeError(f"Could not open white video: {white_video}")
    if not pattern_capture.isOpened():
        white_capture.release()
        raise RuntimeError(f"Could not reopen pattern video: {pattern_video}")
    distance = parse_distance(white_video)
    fps = float(white_capture.get(cv2.CAP_PROP_FPS))
    writer = open_diagnostic_writer(diagnostic_video, fps, (960, 960))
    try:
        for frame_index in range(requested_frames):
            white_ok, frame = white_capture.read()
            pattern_ok, pattern_frame = pattern_capture.read()
            if not white_ok:
                break
            started = time.perf_counter()
            row = blank_row(WHITE_FRAME_FIELDS)
            row.update(
                {
                    "distance_cm": distance,
                    "white_video_file": white_video.name,
                    "gt_detection_mode": (
                        "cross_intersection"
                        if isinstance(gt_detector, CrossIntersectionDetector)
                        else "white_blob_centroid"
                    ),
                    "frame_index": frame_index,
                    "status": "FAILED",
                    "failure_reason": "",
                    "detected_corner_count": 0,
                }
            )
            left, right = split_sbs(frame)
            left_points, left_results = gt_detector.detect_four(
                left, search_centers["left"]
            )
            right_points, right_results = gt_detector.detect_four(
                right, search_centers["right"]
            )
            diagnostics_by_frame[frame_index] = {
                "left": left_results,
                "right": right_results,
            }
            detected_count = sum(item.get("accepted", False) for item in left_results)
            detected_count += sum(item.get("accepted", False) for item in right_results)
            row["detected_corner_count"] = detected_count
            if left_points is not None:
                add_points(row, "left", left_points)
            if right_points is not None:
                add_points(row, "right", right_points)
            if left_points is not None and right_points is not None:
                corners_by_frame[frame_index] = {
                    "left": left_points,
                    "right": right_points,
                }
                if (
                    cross_preview_path is None
                    and isinstance(gt_detector, CrossIntersectionDetector)
                ):
                    cross_preview_path = diagnostic_video.parent / (
                        f"{white_video.stem}_cross_intersection_F{frame_index:03d}.jpg"
                    )
                    save_cross_intersection_preview(
                        frame,
                        left_points,
                        right_points,
                        cross_preview_path,
                    )
                try:
                    estimate = estimate_single_marker_rt(
                        estimator, marker_id, left_points, right_points, answer
                    )
                    poses[frame_index] = {
                        "R": estimate["R_rel"],
                        "t": estimate["t_rel"],
                    }
                    row.update(remap_error_names(estimate))
                    for field in (
                        "marker_transfer_combined_rms_px",
                        "marker_transfer_max_px",
                        "selected_ippe_branches",
                    ):
                        row[field] = estimate.get(field)
                    row["status"] = "OK"
                except Exception as exc:
                    row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            else:
                failures = [
                    f"{side} C{index}: {item.get('failure_reason', 'missing')}"
                    for side, results in (("left", left_results), ("right", right_results))
                    for index, item in enumerate(results)
                    if not item.get("accepted")
                ]
                row["failure_reason"] = "; ".join(failures)

            diagnostic = white_diagnostic_image(
                left,
                right,
                search_centers,
                left_results,
                right_results,
                frame_index,
                gt_detector.radius,
                cross_arm,
            )
            if pattern_ok:
                pattern_left, pattern_right = split_sbs(pattern_frame)
            else:
                pattern_left = np.zeros_like(left)
                pattern_right = np.zeros_like(right)
            pattern_diagnostic = pattern_diagnostic_image(
                pattern_left,
                pattern_right,
                search_centers,
                pattern_corners.get(frame_index),
                frame_index,
                gt_detector.radius,
                cross_arm,
            )
            stacked = np.vstack((pattern_diagnostic, diagnostic))
            writer.write(stacked)
            row["diagnostic_video"] = str(diagnostic_video)
            if cross_preview_path is not None:
                row["cross_preview_jpg"] = str(cross_preview_path)
            row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            if progress_every > 0 and (frame_index + 1) % progress_every == 0:
                print(f"  White GT F{frame_index:03d} complete")
    finally:
        writer.release()
        pattern_capture.release()
        white_capture.release()
    if not corners_by_frame:
        raise RuntimeError("No GT-video frame had all eight accepted points")
    gt_median = {
        side: np.median(
            np.stack([entry[side] for entry in corners_by_frame.values()]), axis=0
        )
        for side in ("left", "right")
    }
    if cross_preview_path is not None:
        for row in rows:
            row["cross_preview_jpg"] = str(cross_preview_path)
    return (
        rows,
        corners_by_frame,
        poses,
        diagnostics_by_frame,
        gt_median,
        cross_preview_path,
    )


def rt_difference(R_first, t_first, R_reference, t_reference) -> dict:
    R_first = np.asarray(R_first, dtype=np.float64).reshape(3, 3)
    t_first = np.asarray(t_first, dtype=np.float64).reshape(3, 1)
    R_reference = np.asarray(R_reference, dtype=np.float64).reshape(3, 3)
    t_reference = np.asarray(t_reference, dtype=np.float64).reshape(3, 1)
    baseline_first = float(np.linalg.norm(t_first))
    baseline_reference = float(np.linalg.norm(t_reference))
    direction_cos = np.clip(
        float((t_first.T @ t_reference)[0, 0])
        / max(baseline_first * baseline_reference, 1e-12),
        -1.0,
        1.0,
    )
    baseline_delta = baseline_first - baseline_reference
    return {
        "rotation_pattern_vs_gt_median_deg": rotation_angle_deg(
            R_first @ R_reference.T
        ),
        "translation_pattern_vs_gt_median_l2_mm": float(
            np.linalg.norm(t_first - t_reference)
        ),
        "translation_direction_pattern_vs_gt_median_deg": float(
            np.degrees(np.arccos(direction_cos))
        ),
        "baseline_pattern_minus_gt_median_mm": baseline_delta,
        "absolute_baseline_pattern_vs_gt_median_mm": abs(baseline_delta),
        "absolute_baseline_pattern_vs_gt_median_percent": (
            abs(baseline_delta) / baseline_reference * 100.0
        ),
    }


def error_stats(values) -> tuple[float | None, float | None, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None, None
    return (
        float(np.mean(values)),
        float(np.sqrt(np.mean(values**2))),
        float(np.max(values)),
    )


def marker_mean_side_length(points) -> float | None:
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    lengths = [
        float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
        for index in range(4)
    ]
    return mean(lengths)


def build_comparison_rows(
    pattern_video,
    white_video,
    pattern_rows,
    white_rows,
    pattern_corners,
    white_corners,
    pattern_poses,
    white_poses,
    white_diagnostics,
    gt_median,
    gt_median_pose,
):
    distance = parse_distance(pattern_video)
    white_row_lookup = {int(row["frame_index"]): row for row in white_rows}
    comparison_rows = []
    corner_rows = []
    for pattern_row in pattern_rows:
        frame_index = int(pattern_row["frame_index"])
        row = blank_row(FRAME_COMPARISON_FIELDS)
        row.update(
            {
                "distance_cm": distance,
                "pattern_video_file": pattern_video.name,
                "white_video_file": white_video.name,
                "pattern_frame_index": frame_index,
                "pattern_status": pattern_row["status"],
            }
        )
        white_row = white_row_lookup.get(frame_index)
        if white_row is not None:
            row["white_same_index_status"] = white_row["status"]
        if pattern_row["status"] == "OK":
            row.update(
                {
                    "pattern_rotation_error_json_deg": pattern_row[
                        "rotation_error_json_deg"
                    ],
                    "pattern_baseline_mm": pattern_row["algorithm_baseline_mm"],
                    "pattern_absolute_baseline_error_json_mm": pattern_row[
                        "absolute_baseline_delta_json_mm"
                    ],
                    "pattern_absolute_baseline_error_json_percent": pattern_row[
                        "absolute_baseline_error_json_percent"
                    ],
                }
            )
        if white_row is not None and white_row["status"] == "OK":
            row.update(
                {
                    "white_same_index_rotation_error_json_deg": white_row[
                        "rotation_error_json_deg"
                    ],
                    "white_same_index_baseline_mm": white_row["algorithm_baseline_mm"],
                    "white_same_index_absolute_baseline_error_json_mm": white_row[
                        "absolute_baseline_delta_json_mm"
                    ],
                    "white_same_index_absolute_baseline_error_json_percent": white_row[
                        "absolute_baseline_error_json_percent"
                    ],
                }
            )
        row.update(
            {
                "gt_median_rotation_error_json_deg": gt_median_pose[
                    "rotation_error_deg"
                ],
                "gt_median_baseline_mm": gt_median_pose["algorithm_baseline_mm"],
                "gt_median_absolute_baseline_error_json_mm": gt_median_pose[
                    "absolute_baseline_delta_mm"
                ],
                "gt_median_absolute_baseline_error_json_percent": gt_median_pose[
                    "absolute_baseline_error_percent"
                ],
            }
        )
        if frame_index in pattern_poses:
            row.update(
                rt_difference(
                    pattern_poses[frame_index]["R"],
                    pattern_poses[frame_index]["t"],
                    gt_median_pose["R_rel"],
                    gt_median_pose["t_rel"],
                )
            )
        if frame_index in pattern_corners:
            all_errors = []
            for side in ("left", "right"):
                errors = np.linalg.norm(
                    pattern_corners[frame_index][side] - gt_median[side], axis=1
                )
                error_mean, error_rms, error_max = error_stats(errors)
                row[f"{side}_corner_error_to_gt_median_mean_px"] = error_mean
                row[f"{side}_corner_error_to_gt_median_rms_px"] = error_rms
                row[f"{side}_corner_error_to_gt_median_max_px"] = error_max
                all_errors.extend(errors.tolist())
            error_mean, error_rms, error_max = error_stats(all_errors)
            row["combined_corner_error_to_gt_median_mean_px"] = error_mean
            row["combined_corner_error_to_gt_median_rms_px"] = error_rms
            row["combined_corner_error_to_gt_median_max_px"] = error_max
        comparison_rows.append(row)

        for side in ("left", "right"):
            pattern_points = (
                pattern_corners.get(frame_index, {}).get(side)
                if frame_index in pattern_corners
                else None
            )
            white_points = (
                white_corners.get(frame_index, {}).get(side)
                if frame_index in white_corners
                else None
            )
            diagnostics = white_diagnostics.get(frame_index, {}).get(side, [{}] * 4)
            for corner_index in range(4):
                corner_row = blank_row(CORNER_COMPARISON_FIELDS)
                corner_row.update(
                    {
                        "distance_cm": distance,
                        "pattern_video_file": pattern_video.name,
                        "white_video_file": white_video.name,
                        "gt_detection_mode": (
                            white_row.get("gt_detection_mode")
                            if white_row is not None
                            else None
                        ),
                        "frame_index": frame_index,
                        "side": side,
                        "corner_index": corner_index,
                        "gt_median_x_px": float(gt_median[side][corner_index, 0]),
                        "gt_median_y_px": float(gt_median[side][corner_index, 1]),
                    }
                )
                if pattern_points is not None:
                    pattern_point = pattern_points[corner_index]
                    pattern_side_length = marker_mean_side_length(pattern_points)
                    corner_row["pattern_x_px"] = float(pattern_point[0])
                    corner_row["pattern_y_px"] = float(pattern_point[1])
                    corner_error = float(
                        np.linalg.norm(pattern_point - gt_median[side][corner_index])
                    )
                    corner_row["pattern_vs_gt_median_error_px"] = corner_error
                    corner_row["pattern_mean_side_length_px"] = pattern_side_length
                    if pattern_side_length is not None and pattern_side_length > 0:
                        corner_row["normalized_corner_error_percent"] = (
                            corner_error / pattern_side_length * 100.0
                        )
                if white_points is not None:
                    white_point = white_points[corner_index]
                    corner_row["white_same_index_x_px"] = float(white_point[0])
                    corner_row["white_same_index_y_px"] = float(white_point[1])
                    corner_row["white_same_index_vs_gt_median_error_px"] = float(
                        np.linalg.norm(white_point - gt_median[side][corner_index])
                    )
                    if pattern_points is not None:
                        corner_row["pattern_vs_white_same_index_error_px"] = float(
                            np.linalg.norm(pattern_points[corner_index] - white_point)
                        )
                diagnostic = diagnostics[corner_index] if corner_index < len(diagnostics) else {}
                for source, target in (
                    ("peak_gray", "white_peak_gray"),
                    ("background_gray", "white_background_gray"),
                    ("contrast_gray", "white_contrast_gray"),
                    ("blob_area_px", "white_blob_area_px"),
                    (
                        "distance_from_search_center_px",
                        "white_distance_from_pattern_search_center_px",
                    ),
                    ("line1_support_pixels", "cross_line1_support_pixels"),
                    ("line2_support_pixels", "cross_line2_support_pixels"),
                    ("line_fit_rms_px", "cross_line_fit_rms_px"),
                ):
                    corner_row[target] = diagnostic.get(source)
                if corner_row.get("gt_detection_mode") is None:
                    corner_row["gt_detection_mode"] = diagnostic.get("detector_mode")
                corner_rows.append(corner_row)
    return comparison_rows, corner_rows


def coordinate_std_rms(corners_by_frame, side) -> float | None:
    if len(corners_by_frame) < 2:
        return None
    values = np.stack([entry[side] for entry in corners_by_frame.values()])
    coordinate_stds = np.std(values, axis=0, ddof=1).reshape(-1)
    return float(np.sqrt(np.mean(coordinate_stds**2)))


def build_summary(
    pattern_video,
    white_video,
    marker_id,
    requested_frames,
    pattern_rows,
    white_rows,
    pattern_corners,
    white_corners,
    comparison_rows,
    corner_rows,
    gt_median_pose,
    gt_detection_mode,
):
    record = blank_row(SUMMARY_FIELDS)
    record.update(
        {
            "distance_cm": parse_distance(pattern_video),
            "pattern_video_file": pattern_video.name,
            "white_video_file": white_video.name,
            "marker_id": marker_id,
            "gt_detection_mode": gt_detection_mode,
            "requested_pattern_frames": min(requested_frames, len(pattern_rows)),
            "valid_pattern_corner_frames": len(pattern_corners),
            "valid_pattern_rt_frames": sum(row["status"] == "OK" for row in pattern_rows),
            "requested_white_frames": min(requested_frames, len(white_rows)),
            "valid_white_corner_frames": len(white_corners),
            "valid_white_rt_frames": sum(row["status"] == "OK" for row in white_rows),
            "pattern_left_corner_coordinate_std_rms_px": coordinate_std_rms(
                pattern_corners, "left"
            ),
            "pattern_right_corner_coordinate_std_rms_px": coordinate_std_rms(
                pattern_corners, "right"
            ),
            "white_left_corner_coordinate_std_rms_px": coordinate_std_rms(
                white_corners, "left"
            ),
            "white_right_corner_coordinate_std_rms_px": coordinate_std_rms(
                white_corners, "right"
            ),
            "gt_median_rotation_error_json_deg": gt_median_pose["rotation_error_deg"],
            "gt_median_baseline_mm": gt_median_pose["algorithm_baseline_mm"],
            "gt_median_absolute_baseline_error_json_mm": gt_median_pose[
                "absolute_baseline_delta_mm"
            ],
            "gt_median_absolute_baseline_error_json_percent": gt_median_pose[
                "absolute_baseline_error_percent"
            ],
        }
    )
    valid_pattern_rows = [row for row in pattern_rows if row["status"] == "OK"]
    valid_white_rows = [row for row in white_rows if row["status"] == "OK"]
    metric_specs = (
        (
            valid_pattern_rows,
            "rotation_error_json_deg",
            "pattern_rotation_error_json_mean_deg",
            "pattern_rotation_error_json_std_deg",
        ),
        (
            valid_pattern_rows,
            "absolute_baseline_delta_json_mm",
            "pattern_absolute_baseline_error_json_mean_mm",
            "pattern_absolute_baseline_error_json_std_mm",
        ),
        (
            valid_pattern_rows,
            "absolute_baseline_error_json_percent",
            "pattern_absolute_baseline_error_json_mean_percent",
            "pattern_absolute_baseline_error_json_std_percent",
        ),
        (
            valid_white_rows,
            "rotation_error_json_deg",
            "white_rotation_error_json_mean_deg",
            "white_rotation_error_json_std_deg",
        ),
        (
            valid_white_rows,
            "absolute_baseline_delta_json_mm",
            "white_absolute_baseline_error_json_mean_mm",
            "white_absolute_baseline_error_json_std_mm",
        ),
        (
            valid_white_rows,
            "absolute_baseline_error_json_percent",
            "white_absolute_baseline_error_json_mean_percent",
            "white_absolute_baseline_error_json_std_percent",
        ),
    )
    for rows, source, mean_field, std_field in metric_specs:
        values = [row[source] for row in rows]
        record[mean_field] = mean(values)
        record[std_field] = std(values)

    corner_errors = [
        row["pattern_vs_gt_median_error_px"]
        for row in corner_rows
        if finite(row.get("pattern_vs_gt_median_error_px")) is not None
    ]
    record["pattern_vs_gt_corner_error_mean_px"] = mean(corner_errors)
    record["pattern_vs_gt_corner_error_rms_px"] = (
        float(np.sqrt(np.mean(np.square(corner_errors)))) if corner_errors else None
    )
    record["pattern_vs_gt_corner_error_std_px"] = std(corner_errors)
    record["pattern_vs_gt_corner_error_max_px"] = max(corner_errors) if corner_errors else None
    normalized_corner_errors = numeric_values(
        corner_rows, "normalized_corner_error_percent"
    )
    record["pattern_vs_gt_normalized_corner_error_mean_percent"] = mean(
        normalized_corner_errors
    )
    record["pattern_vs_gt_normalized_corner_error_std_percent"] = std(
        normalized_corner_errors
    )

    comparison_specs = (
        (
            "rotation_pattern_vs_gt_median_deg",
            "pattern_vs_gt_rotation_mean_deg",
            "pattern_vs_gt_rotation_std_deg",
        ),
        (
            "translation_pattern_vs_gt_median_l2_mm",
            "pattern_vs_gt_translation_l2_mean_mm",
            "pattern_vs_gt_translation_l2_std_mm",
        ),
        (
            "absolute_baseline_pattern_vs_gt_median_mm",
            "pattern_vs_gt_absolute_baseline_mean_mm",
            "pattern_vs_gt_absolute_baseline_std_mm",
        ),
        (
            "absolute_baseline_pattern_vs_gt_median_percent",
            "pattern_vs_gt_absolute_baseline_mean_percent",
            "pattern_vs_gt_absolute_baseline_std_percent",
        ),
    )
    for source, mean_field, std_field in comparison_specs:
        values = [row[source] for row in comparison_rows]
        record[mean_field] = mean(values)
        record[std_field] = std(values)
    return record


def numeric_values(rows, field) -> list[float]:
    values = []
    for row in rows:
        value = finite(row.get(field))
        if value is not None:
            values.append(float(value))
    return values


def build_distance_summary(
    pair_summaries,
    frame_rows,
    corner_rows,
) -> list[dict]:
    distances = sorted(
        {
            float(value)
            for row in pair_summaries
            if (value := finite(row.get("distance_cm"))) is not None
        }
    )
    output = []
    for distance in distances:
        pair_group = [
            row
            for row in pair_summaries
            if finite(row.get("distance_cm")) == distance
        ]
        frame_group = [
            row for row in frame_rows if finite(row.get("distance_cm")) == distance
        ]
        corner_group = [
            row for row in corner_rows if finite(row.get("distance_cm")) == distance
        ]
        left_errors = numeric_values(
            [row for row in corner_group if row.get("side") == "left"],
            "pattern_vs_gt_median_error_px",
        )
        right_errors = numeric_values(
            [row for row in corner_group if row.get("side") == "right"],
            "pattern_vs_gt_median_error_px",
        )
        errors = left_errors + right_errors
        normalized_errors = numeric_values(
            corner_group, "normalized_corner_error_percent"
        )
        record = blank_row(DISTANCE_SUMMARY_FIELDS)
        record.update(
            {
                "distance_cm": distance,
                "pair_count": len(pair_group),
                "pattern_frame_count": len(frame_group),
                "valid_pattern_corner_frame_count": sum(
                    finite(row.get("combined_corner_error_to_gt_median_mean_px"))
                    is not None
                    for row in frame_group
                ),
                "corner_error_sample_count": len(errors),
                "left_corner_error_mean_px": mean(left_errors),
                "right_corner_error_mean_px": mean(right_errors),
                "combined_corner_error_mean_px": mean(errors),
                "combined_corner_error_rms_px": (
                    float(np.sqrt(np.mean(np.square(errors)))) if errors else None
                ),
                "combined_corner_error_std_px": std(errors),
                "combined_corner_error_p95_px": (
                    float(np.percentile(errors, 95)) if errors else None
                ),
                "combined_corner_error_max_px": max(errors) if errors else None,
                "normalized_corner_error_sample_count": len(normalized_errors),
                "normalized_corner_error_mean_percent": mean(normalized_errors),
                "normalized_corner_error_std_percent": std(normalized_errors),
                "pattern_rotation_error_json_mean_deg": mean(
                    numeric_values(frame_group, "pattern_rotation_error_json_deg")
                ),
                "pattern_absolute_baseline_error_json_mean_percent": mean(
                    numeric_values(
                        frame_group,
                        "pattern_absolute_baseline_error_json_percent",
                    )
                ),
                "pattern_vs_gt_rotation_mean_deg": mean(
                    numeric_values(frame_group, "rotation_pattern_vs_gt_median_deg")
                ),
                "pattern_vs_gt_absolute_baseline_mean_mm": mean(
                    numeric_values(
                        frame_group,
                        "absolute_baseline_pattern_vs_gt_median_mm",
                    )
                ),
                "pattern_vs_gt_absolute_baseline_mean_percent": mean(
                    numeric_values(
                        frame_group,
                        "absolute_baseline_pattern_vs_gt_median_percent",
                    )
                ),
            }
        )
        output.append(record)
    return output


def build_distance_chart_rows(distance_summary) -> list[dict]:
    source_fields = {
        "distance_cm": "distance_cm",
        "corner_error_mean_px": "combined_corner_error_mean_px",
        "corner_error_std_px": "combined_corner_error_std_px",
        "normalized_corner_error_mean_percent": (
            "normalized_corner_error_mean_percent"
        ),
    }
    rows = []
    for source_row_index, source_row in enumerate(distance_summary, start=2):
        chart_row = {}
        for chart_field, source_field in source_fields.items():
            value = finite(source_row.get(source_field))
            if value is None:
                chart_row[chart_field] = None
                continue
            source_column = excel_column(DISTANCE_SUMMARY_FIELDS.index(source_field) + 1)
            chart_row[chart_field] = ExcelFormula(
                f"'Distance Summary'!${source_column}${source_row_index}",
                float(value),
            )
        rows.append(chart_row)
    return rows


def _chart_title_xml(text, font_size=1400):
    return (
        '<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p>'
        f'<a:r><a:rPr lang="zh-TW" sz="{font_size}"/><a:t>{escape(text)}</a:t>'
        '</a:r></a:p></c:rich></c:tx><c:layout/><c:overlay val="0"/></c:title>'
    )


def _chart_num_ref_xml(formula, values):
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{float(value):.15g}</c:v></c:pt>'
        for index, value in enumerate(values)
        if finite(value) is not None
    )
    return (
        '<c:numRef>'
        f'<c:f>{escape(formula)}</c:f>'
        '<c:numCache><c:formatCode>General</c:formatCode>'
        f'<c:ptCount val="{len(values)}"/>{points}</c:numCache>'
        '</c:numRef>'
    )


def build_distance_scatter_chart_xml(
    title,
    y_axis_title,
    x_formula,
    x_values,
    series,
    axis_seed,
):
    series_xml = []
    for index, entry in enumerate(series):
        color = entry["color"]
        series_xml.append(
            '<c:ser>'
            f'<c:idx val="{index}"/><c:order val="{index}"/>'
            f'<c:tx><c:v>{escape(entry["name"])}</c:v></c:tx>'
            '<c:spPr><a:ln w="19050"><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill></a:ln></c:spPr>'
            '<c:marker><c:symbol val="circle"/><c:size val="6"/>'
            '<c:spPr><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill><a:ln><a:solidFill>'
            f'<a:srgbClr val="{color}"/>'
            '</a:solidFill></a:ln></c:spPr></c:marker>'
            f'<c:xVal>{_chart_num_ref_xml(x_formula, x_values)}</c:xVal>'
            f'<c:yVal>{_chart_num_ref_xml(entry["formula"], entry["values"])}</c:yVal>'
            '<c:smooth val="0"/></c:ser>'
        )
    x_axis_id = int(axis_seed)
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
        f'{_chart_title_xml("Distance (cm)", 1000)}'
        '<c:numFmt formatCode="0.0" sourceLinked="0"/><c:majorTickMark val="out"/>'
        '<c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{y_axis_id}"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="midCat"/></c:valAx>'
        f'<c:valAx><c:axId val="{y_axis_id}"/><c:scaling><c:orientation val="minMax"/>'
        '</c:scaling><c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/>'
        f'{_chart_title_xml(y_axis_title, 1000)}'
        '<c:numFmt formatCode="0.000" sourceLinked="0"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        '<c:tickLblPos val="nextTo"/>'
        f'<c:crossAx val="{x_axis_id}"/><c:crosses val="autoZero"/>'
        '<c:crossBetween val="midCat"/></c:valAx>'
        '</c:plotArea><c:legend><c:legendPos val="r"/><c:layout/>'
        '<c:overlay val="0"/></c:legend><c:plotVisOnly val="1"/>'
        '<c:dispBlanksAs val="gap"/><c:showDLblsOverMax val="0"/>'
        '</c:chart></c:chartSpace>'
    )


def _drawing_anchor_xml(chart_number, relationship_id, from_row, to_row):
    return (
        '<xdr:twoCellAnchor><xdr:from><xdr:col>5</xdr:col><xdr:colOff>0</xdr:colOff>'
        f'<xdr:row>{from_row}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
        '<xdr:to><xdr:col>14</xdr:col><xdr:colOff>0</xdr:colOff>'
        f'<xdr:row>{to_row}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>'
        '<xdr:graphicFrame macro=""><xdr:nvGraphicFramePr>'
        f'<xdr:cNvPr id="{chart_number + 1}" name="Chart {chart_number}"/>'
        '<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm>'
        '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
        '<a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'r:id="{relationship_id}"/></a:graphicData></a:graphic>'
        '</xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor>'
    )


def add_distance_charts_to_xlsx(path, distance_summary, sheet_index=2):
    if not distance_summary:
        return
    first_row = 2
    last_row = first_row + len(distance_summary) - 1
    distances = [row.get("distance_cm") for row in distance_summary]
    means = [row.get("combined_corner_error_mean_px") for row in distance_summary]
    stds = [row.get("combined_corner_error_std_px") for row in distance_summary]
    normalized = [
        row.get("normalized_corner_error_mean_percent") for row in distance_summary
    ]
    x_formula = f"'Distance Charts'!$A${first_row}:$A${last_row}"
    chart1 = build_distance_scatter_chart_xml(
        "Corner Error Mean and Standard Deviation by Distance",
        "Corner error (px)",
        x_formula,
        distances,
        [
            {
                "name": "Mean error",
                "formula": f"'Distance Charts'!$B${first_row}:$B${last_row}",
                "values": means,
                "color": "4472C4",
            },
            {
                "name": "Standard deviation",
                "formula": f"'Distance Charts'!$C${first_row}:$C${last_row}",
                "values": stds,
                "color": "ED7D31",
            },
        ],
        1100000000,
    )
    chart2 = build_distance_scatter_chart_xml(
        "Normalized Corner Error Mean by Distance",
        "Normalized corner error mean (%)",
        x_formula,
        distances,
        [
            {
                "name": "Normalized mean",
                "formula": f"'Distance Charts'!$D${first_row}:$D${last_row}",
                "values": normalized,
                "color": "70AD47",
            }
        ],
        1200000000,
    )
    drawing = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'{_drawing_anchor_xml(1, "rId1", 0, 15)}'
        f'{_drawing_anchor_xml(2, "rId2", 17, 32)}'
        '</xdr:wsDr>'
    )
    sheet_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/></Relationships>'
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
    new_parts = {
        f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels": sheet_rels,
        "xl/drawings/drawing1.xml": drawing,
        "xl/drawings/_rels/drawing1.xml.rels": drawing_rels,
        "xl/charts/chart1.xml": chart1,
        "xl/charts/chart2.xml": chart2,
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
                if info.filename == f"xl/worksheets/sheet{sheet_index}.xml":
                    worksheet = data.decode("utf-8")
                    worksheet = worksheet.replace(
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
                        1,
                    )
                    data = worksheet.replace(
                        "</worksheet>", '<drawing r:id="rId1"/></worksheet>', 1
                    ).encode("utf-8")
                    data = data.replace(
                        b'<sheetView workbookViewId="0">',
                        b'<sheetView workbookViewId="0" tabSelected="1">',
                        1,
                    )
                elif info.filename == "xl/workbook.xml":
                    workbook_xml = data.decode("utf-8")
                    if "<bookViews>" not in workbook_xml:
                        workbook_xml = workbook_xml.replace(
                            "<sheets>",
                            '<bookViews><workbookView activeTab="1"/></bookViews><sheets>',
                            1,
                        )
                    data = workbook_xml.encode("utf-8")
                elif info.filename == "[Content_Types].xml":
                    content_types = data.decode("utf-8")
                    overrides = (
                        '<Override PartName="/xl/drawings/drawing1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                        '<Override PartName="/xl/charts/chart1.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                        '<Override PartName="/xl/charts/chart2.xml" '
                        'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                    )
                    data = content_types.replace(
                        "</Types>", overrides + "</Types>"
                    ).encode("utf-8")
                destination.writestr(info, data)
            for name, data in new_parts.items():
                destination.writestr(name, data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    explicit_white = (
        Path(args.white_video).expanduser().resolve() if args.white_video else None
    )
    calibration = Path(args.calibration).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration JSON does not exist: {calibration}")
    pairs = collect_video_pairs(
        input_path,
        explicit_white,
        args.recursive,
        args.max_pairs,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            input_path / "HBVCAM_pattern_vs_white_pixel_gt_batch.xlsx"
            if input_path.is_dir()
            else input_path.with_name(input_path.stem + "_vs_white_pixel_gt.xlsx")
        )
    )
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    diagnostic_video_dir = (
        Path(args.diagnostic_video_dir).expanduser().resolve()
        if args.diagnostic_video_dir
        else output.with_name(output.stem + "_diagnostic_videos")
    )

    k_left, d_left, k_right, d_right, answer = load_calibration(calibration)
    detector = AdaptiveArucoDetector(args.subpixel_stability_max_raw_px)
    estimator = MultiMarkerRTEstimator(
        k_left,
        d_left,
        k_right,
        d_right,
        args.marker_size_mm,
        args.min_baseline_mm,
        args.max_baseline_mm,
        PATTERN_RANSAC_REPROJECTION_THRESHOLD_PX,
    )
    if args.gt_mode == "cross_intersection":
        gt_detector = CrossIntersectionDetector(args.white_search_radius_px)
    else:
        gt_detector = WhiteBlobDetector(
            args.white_search_radius_px,
            args.white_threshold_ratio,
            args.white_min_contrast_gray,
            args.white_min_area_px,
            args.white_max_area_px,
        )

    print(f"Input: {input_path}")
    print(f"Matched video pairs: {len(pairs)}")
    print(f"Frames requested from each video: {args.frames}")
    print(f"GT detection mode: {args.gt_mode}")
    started = time.perf_counter()
    all_pattern_rows = []
    all_white_rows = []
    all_comparison_rows = []
    all_corner_rows = []
    pair_summaries = []
    batch_errors = []
    diagnostic_paths = []
    cross_preview_paths = []
    for pair_index, (pattern_video, white_video) in enumerate(pairs, start=1):
        print("")
        print(f"[{pair_index}/{len(pairs)}] Pattern: {pattern_video.name}")
        print(f"[{pair_index}/{len(pairs)}] White GT: {white_video.name}")
        try:
            marker_id = (
                int(args.marker_id)
                if args.marker_id is not None
                else discover_marker_id(pattern_video, detector, args.frames)
            )
            print(f"ArUco ID: {marker_id}")
            print("Phase 1/2: ArUco subpixel corners and pattern RT...")
            (
                pattern_rows,
                pattern_corners,
                pattern_poses,
                search_centers,
            ) = process_pattern_video(
                pattern_video,
                args.frames,
                marker_id,
                detector,
                estimator,
                answer,
                args.progress_every,
            )
            diagnostic_video = diagnostic_video_dir / (
                pattern_video.stem + "_vs_white_pixel_gt_diagnostic.mp4"
            )
            print(f"Phase 2/2: {args.gt_mode} GT and stacked diagnostic MP4...")
            (
                white_rows,
                white_corners,
                white_poses,
                white_diagnostics,
                gt_median,
                cross_preview_path,
            ) = process_white_video(
                pattern_video,
                white_video,
                args.frames,
                marker_id,
                gt_detector,
                estimator,
                answer,
                search_centers,
                pattern_corners,
                diagnostic_video,
                args.cross_arm_px,
                args.progress_every,
            )
            gt_estimate = estimate_single_marker_rt(
                estimator,
                marker_id,
                gt_median["left"],
                gt_median["right"],
                answer,
            )
            gt_median_pose = {
                "R_rel": gt_estimate["R_rel"],
                "t_rel": gt_estimate["t_rel"],
                **{
                    key: value
                    for key, value in gt_estimate.items()
                    if key
                    in {
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
                    }
                },
            }
            comparison_rows, corner_rows = build_comparison_rows(
                pattern_video,
                white_video,
                pattern_rows,
                white_rows,
                pattern_corners,
                white_corners,
                pattern_poses,
                white_poses,
                white_diagnostics,
                gt_median,
                gt_median_pose,
            )
            pair_summary = build_summary(
                pattern_video,
                white_video,
                marker_id,
                args.frames,
                pattern_rows,
                white_rows,
                pattern_corners,
                white_corners,
                comparison_rows,
                corner_rows,
                gt_median_pose,
                args.gt_mode,
            )
            all_pattern_rows.extend(pattern_rows)
            all_white_rows.extend(white_rows)
            all_comparison_rows.extend(comparison_rows)
            all_corner_rows.extend(corner_rows)
            pair_summaries.append(pair_summary)
            diagnostic_paths.append(diagnostic_video)
            if cross_preview_path is not None:
                cross_preview_paths.append(cross_preview_path)
                print(f"Cross preview JPG: {cross_preview_path}")
            print(
                f"Valid pattern corners: {len(pattern_corners)}/{len(pattern_rows)} | "
                f"valid white GT corners: {len(white_corners)}/{len(white_rows)}"
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"ERROR, pair skipped: {error_text}")
            batch_errors.append(
                {
                    "distance_cm": parse_distance(pattern_video),
                    "pattern_video_file": pattern_video.name,
                    "white_video_file": white_video.name,
                    "error": error_text,
                }
            )

    if not pair_summaries:
        raise RuntimeError("All video pairs failed; no Excel report was written")
    distance_summary = build_distance_summary(
        pair_summaries,
        all_comparison_rows,
        all_corner_rows,
    )
    distance_chart_rows = build_distance_chart_rows(distance_summary)
    batch_error_sheet_rows = (
        batch_errors
        if batch_errors
        else [
            {
                "distance_cm": None,
                "pattern_video_file": "(none)",
                "white_video_file": "(none)",
                "error": "No batch errors. All matched video pairs completed successfully.",
            }
        ]
    )

    settings = [
        {"parameter": "input_path", "value": str(input_path)},
        {"parameter": "matched_video_pair_count", "value": len(pairs)},
        {"parameter": "successful_video_pair_count", "value": len(pair_summaries)},
        {"parameter": "frames_requested_per_video", "value": args.frames},
        {"parameter": "gt_detection_mode", "value": args.gt_mode},
        {
            "parameter": "marker_id",
            "value": args.marker_id if args.marker_id is not None else "auto per pair",
        },
        {"parameter": "marker_size_mm", "value": args.marker_size_mm},
        {
            "parameter": "subpixel_stability_max_raw_px",
            "value": args.subpixel_stability_max_raw_px,
        },
        {"parameter": "white_search_radius_px", "value": args.white_search_radius_px},
        {"parameter": "white_threshold_ratio", "value": args.white_threshold_ratio},
        {
            "parameter": "white_min_contrast_gray",
            "value": args.white_min_contrast_gray,
        },
        {"parameter": "white_blob_area_min_px", "value": args.white_min_area_px},
        {"parameter": "white_blob_area_max_px", "value": args.white_max_area_px},
        {"parameter": "cross_line_polarity", "value": CROSS_LINE_POLARITY},
        {"parameter": "cross_threshold_ratio", "value": CROSS_THRESHOLD_RATIO},
        {
            "parameter": "cross_min_contrast_gray",
            "value": CROSS_MIN_CONTRAST_GRAY,
        },
        {"parameter": "cross_line_fit_band_px", "value": CROSS_LINE_FIT_BAND_PX},
        {
            "parameter": "cross_min_arm_support_pixels",
            "value": CROSS_MIN_ARM_SUPPORT_PIXELS,
        },
        {"parameter": "cross_preview_arm_px", "value": CROSS_PREVIEW_ARM_PX},
        {
            "parameter": "cross_preview_line_width_px",
            "value": CROSS_PREVIEW_LINE_WIDTH_PX,
        },
        {
            "parameter": "cross_preview_jpeg_quality",
            "value": CROSS_PREVIEW_JPEG_QUALITY,
        },
        {"parameter": "diagnostic_cross_arm_px", "value": args.cross_arm_px},
        {"parameter": "diagnostic_cross_line_width_px", "value": 1},
        {"parameter": "diagnostic_video_width_px", "value": 960},
        {"parameter": "diagnostic_video_height_px", "value": 960},
        {
            "parameter": "pattern_ransac_reprojection_threshold_px",
            "value": PATTERN_RANSAC_REPROJECTION_THRESHOLD_PX,
        },
    ]
    protocol = [
        {
            "parameter": "generated_at",
            "value": datetime.now(timezone.utc).astimezone().isoformat(),
        },
        {"parameter": "input_path", "value": str(input_path)},
        {"parameter": "calibration_file", "value": str(calibration)},
        {
            "parameter": "batch pairing",
            "value": "*_pattern video paired with *_white in the same folder",
        },
        {
            "parameter": "temporal pairing",
            "value": (
                "videos are not assumed synchronized; fixed GT is the cross-frame "
                "median of accepted GT points"
            ),
        },
        {
            "parameter": "ArUco corner",
            "value": (
                "CLAHE initial detection on raw distorted image; adaptive cornerSubPix "
                "on original gray"
            ),
        },
        {
            "parameter": "GT point detector",
            "value": (
                "white_blob_centroid uses the background-subtracted intensity-weighted "
                "center of a bright component; cross_intersection fits the two local "
                "screen-line centerlines and uses their subpixel intersection"
            ),
        },
        {
            "parameter": "corner comparison coordinates",
            "value": "raw distorted camera pixels; pattern and GT use the same image domain",
        },
        {
            "parameter": "RT",
            "value": "single-marker left/right IPPE branch pairs; physical baseline gate",
        },
        {
            "parameter": "JSON usage",
            "value": "answer errors only after RT; never used for detection or selection",
        },
        {
            "parameter": "diagnostic MP4",
            "value": (
                f"top=ArUco subpixel corners, bottom={args.gt_mode} points; "
                "2x2 enlarged corner ROIs per eye; red cross line width=1 px"
            ),
        },
        {
            "parameter": "cross preview JPG",
            "value": (
                "cross_intersection mode only; one original-resolution SBS frame per "
                "GT video, using the first frame with all eight accepted points; "
                "detected positions are marked by red crosses with 1 px line width"
            ),
        },
        {
            "parameter": "Distance Summary",
            "value": (
                "one row per parsed CM distance; corner error compares each ArUco "
                "subpixel corner with that pair's cross-frame median GT point; "
                "normalized corner error = corner error / per-frame mean marker side length * 100"
            ),
        },
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    write_csv(
        prefix.with_name(prefix.name + "_pattern_frames.csv"),
        PATTERN_FRAME_FIELDS,
        all_pattern_rows,
    )
    write_csv(
        prefix.with_name(prefix.name + "_white_frames.csv"),
        WHITE_FRAME_FIELDS,
        all_white_rows,
    )
    write_csv(
        prefix.with_name(prefix.name + "_frame_comparison.csv"),
        FRAME_COMPARISON_FIELDS,
        all_comparison_rows,
    )
    write_csv(
        prefix.with_name(prefix.name + "_corner_comparison.csv"),
        CORNER_COMPARISON_FIELDS,
        all_corner_rows,
    )
    write_csv(
        prefix.with_name(prefix.name + "_distance_summary.csv"),
        DISTANCE_SUMMARY_FIELDS,
        distance_summary,
    )
    write_csv(
        prefix.with_name(prefix.name + "_pair_summary.csv"),
        SUMMARY_FIELDS,
        pair_summaries,
    )
    write_xlsx(
        output,
        [
            ("Settings", ["parameter", "value"], settings),
            ("Distance Charts", DISTANCE_CHART_FIELDS, distance_chart_rows),
            ("Distance Summary", DISTANCE_SUMMARY_FIELDS, distance_summary),
            ("Pair Summary", SUMMARY_FIELDS, pair_summaries),
            ("Frame Comparison", FRAME_COMPARISON_FIELDS, all_comparison_rows),
            ("Corner Comparison", CORNER_COMPARISON_FIELDS, all_corner_rows),
            ("Pattern Frames", PATTERN_FRAME_FIELDS, all_pattern_rows),
            ("White GT Frames", WHITE_FRAME_FIELDS, all_white_rows),
            ("Batch Errors", BATCH_ERROR_FIELDS, batch_error_sheet_rows),
            ("Protocol", ["parameter", "value"], protocol),
        ],
    )
    add_distance_charts_to_xlsx(output, distance_summary, sheet_index=2)
    print(f"Excel: {output}")
    print("Charts sheet: Distance Charts (set as the active sheet when Excel opens)")
    print(f"Diagnostic MP4 folder: {diagnostic_video_dir}")
    print(f"Diagnostic MP4 files written: {len(diagnostic_paths)}")
    if args.gt_mode == "cross_intersection":
        print(f"Cross preview JPG files written: {len(cross_preview_paths)}")
    if batch_errors:
        print(f"Pairs skipped after errors: {len(batch_errors)} (see Batch Errors sheet)")
    else:
        print("Batch Errors: none; the sheet contains a success-status message")
    print(f"Elapsed: {(time.perf_counter() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted")
        raise SystemExit(130)
