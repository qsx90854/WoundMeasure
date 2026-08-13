"""Estimate monocular camera intrinsics from a checkerboard video.

The default target has 11 x 17 squares, so OpenCV must detect 10 x 16
*inner corners*.  Each square is 15 mm wide.

Example:
    python calibrate_camera_from_video.py calibration.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class View:
    frame_index: int
    time_seconds: float
    corners: np.ndarray
    sharpness: float
    board_area_ratio: float
    feature: np.ndarray
    reprojection_rms: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use checkerboard frames in a video to calibrate one camera."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera_calibration_distance_12cm.json"),
        help="Output JSON path (default: camera_calibration.json)",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=10,
        help="Number of INNER corners horizontally (default: 10 for 11 squares)",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=16,
        help="Number of INNER corners vertically (default: 16 for 17 squares)",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=15.0,
        help="Square edge length in millimetres (default: 15)",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=0.5,
        help="Time between examined frames (default: 0.5 seconds)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Maximum number of frames to examine (default: 300)",
    )
    parser.add_argument(
        "--detection-max-width",
        type=int,
        default=1920,
        help="Downscale wider frames for detection; 0 keeps full resolution (default: 1920)",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=60,
        help="Maximum detected views used for calibration (default: 60)",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=12,
        help="Minimum detected views required (default: 12)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=40.0,
        help="Minimum Laplacian variance; use 0 to disable (default: 40)",
    )
    parser.add_argument(
        "--min-board-area",
        type=float,
        default=0.02,
        help="Minimum board/image area ratio; use 0 to disable (default: 0.02)",
    )
    parser.add_argument(
        "--max-view-error",
        type=float,
        default=0.8,
        help="Reject views above this reprojection RMS in pixels (default: 0.8)",
    )
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.5,
        help="Maximum fraction of selected views that may be removed (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Undistortion crop balance, 0=crop and 1=keep all pixels (default: 0)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for accepted-frame previews (default: beside output)",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Do not save checkerboard/undistortion preview images",
    )
    args = parser.parse_args()

    if args.board_cols < 2 or args.board_rows < 2:
        parser.error("--board-cols and --board-rows must be at least 2")
    if args.square_size_mm <= 0:
        parser.error("--square-size-mm must be positive")
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")
    if args.max_samples < 1 or args.max_views < 3 or args.min_views < 3:
        parser.error("invalid sample/view count")
    if args.detection_max_width < 0:
        parser.error("--detection-max-width cannot be negative")
    if args.max_views < args.min_views:
        parser.error("--max-views must be at least --min-views")
    if args.min_sharpness < 0 or not 0 <= args.min_board_area < 1:
        parser.error("invalid quality threshold")
    if args.max_view_error <= 0 or not 0 <= args.alpha <= 1:
        parser.error("invalid reprojection threshold or alpha")
    if not 0 <= args.max_outlier_fraction < 1:
        parser.error("--max-outlier-fraction must be at least 0 and less than 1")
    return args


def make_object_points(board_size: tuple[int, int], square_mm: float) -> np.ndarray:
    cols, rows = board_size
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= square_mm
    return points


def detect_corners(
    gray: np.ndarray, board_size: tuple[int, int], max_width: int
) -> tuple[bool, np.ndarray | None]:
    scale = 1.0
    detection_image = gray
    if max_width > 0 and gray.shape[1] > max_width:
        scale = max_width / gray.shape[1]
        detection_image = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(detection_image, board_size, flags)
        if found:
            corners = np.asarray(corners, dtype=np.float32) / scale
            if scale < 1.0:
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(detection_image, board_size, flags)
    if not found:
        return False, None
    corners = np.asarray(corners, dtype=np.float32) / scale
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, np.asarray(corners, dtype=np.float32)


def view_geometry(
    corners: np.ndarray, board_size: tuple[int, int], image_size: tuple[int, int]
) -> tuple[float, np.ndarray]:
    cols, rows = board_size
    pts = corners.reshape(-1, 2)
    quad = np.float32([pts[0], pts[cols - 1], pts[-1], pts[(rows - 1) * cols]])
    area = abs(float(cv2.contourArea(quad)))
    width, height = image_size
    area_ratio = area / float(width * height)

    center = pts.mean(axis=0) / np.float32([width, height])
    horizontal = pts[cols - 1] - pts[0]
    vertical = pts[(rows - 1) * cols] - pts[0]
    h_len = max(float(np.linalg.norm(horizontal)), 1e-9)
    v_len = max(float(np.linalg.norm(vertical)), 1e-9)
    horizontal /= h_len
    vertical /= v_len
    # Weighted pose descriptor used only to avoid selecting many near-identical frames.
    feature = np.float32(
        [
            center[0] * 2.0,
            center[1] * 2.0,
            math.sqrt(max(area_ratio, 0.0)) * 3.0,
            horizontal[0] * 0.5,
            horizontal[1] * 0.5,
            vertical[0] * 0.5,
            vertical[1] * 0.5,
        ]
    )
    return area_ratio, feature


def collect_views(
    capture: cv2.VideoCapture,
    board_size: tuple[int, int],
    image_size: tuple[int, int],
    fps: float,
    frame_count: int,
    args: argparse.Namespace,
) -> tuple[list[View], dict[str, int]]:
    step = max(1, round(fps * args.sample_seconds))
    if frame_count > 0:
        indices = list(range(0, frame_count, step))
        if len(indices) > args.max_samples:
            indices = np.linspace(0, frame_count - 1, args.max_samples, dtype=int).tolist()
    else:
        indices = [i * step for i in range(args.max_samples)]

    views: list[View] = []
    stats = {"examined": 0, "read_failed": 0, "not_found": 0, "blurred": 0, "too_small": 0}
    print(f"Video: {image_size[0]}x{image_size[1]}, {fps:.3f} fps")
    print(f"Examining up to {len(indices)} frames; checkerboard inner corners: {board_size[0]}x{board_size[1]}")

    for number, frame_index in enumerate(indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            stats["read_failed"] += 1
            if frame_count <= 0:
                break
            continue
        stats["examined"] += 1
        if (frame.shape[1], frame.shape[0]) != image_size:
            stats["read_failed"] += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, board_size, args.detection_max_width)
        if not found or corners is None:
            stats["not_found"] += 1
            continue

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < args.min_sharpness:
            stats["blurred"] += 1
            continue
        area_ratio, feature = view_geometry(corners, board_size, image_size)
        if area_ratio < args.min_board_area:
            stats["too_small"] += 1
            continue
        views.append(
            View(
                frame_index=int(frame_index),
                time_seconds=float(frame_index / fps),
                corners=corners,
                sharpness=sharpness,
                board_area_ratio=area_ratio,
                feature=feature,
            )
        )
        print(
            f"  detected {len(views):3d}: frame {frame_index:7d}, "
            f"t={frame_index / fps:8.2f}s, sharpness={sharpness:7.1f}, area={area_ratio:6.2%}"
        )
        if number % 25 == 0:
            print(f"  progress: {number}/{len(indices)} sampled frames")
    stats["usable"] = len(views)
    return views, stats


def select_diverse_views(views: list[View], limit: int) -> list[View]:
    if len(views) <= limit:
        return views
    features = np.vstack([view.feature for view in views])
    selected = [int(np.argmax([view.board_area_ratio for view in views]))]
    min_distance = np.full(len(views), np.inf, dtype=np.float64)
    while len(selected) < limit:
        last = selected[-1]
        distances = np.linalg.norm(features - features[last], axis=1)
        min_distance = np.minimum(min_distance, distances)
        min_distance[selected] = -1.0
        selected.append(int(np.argmax(min_distance)))
    return [views[index] for index in sorted(selected)]


def calibrate(
    views: list[View], object_template: np.ndarray, image_size: tuple[int, int]
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[float]]:
    object_points = [object_template.copy() for _ in views]
    image_points = [view.corners for view in views]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors: list[float] = []
    for obj, image, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        delta = image.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return float(rms), matrix, distortion, list(rvecs), list(tvecs), errors


def reject_outliers(
    views: list[View],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    min_views: int,
    absolute_limit: float,
    max_outlier_fraction: float,
) -> tuple[list[View], list[View], tuple, dict[str, float | int]]:
    kept = list(views)
    rejected: list[View] = []
    result: tuple | None = None
    minimum_keep = max(min_views, math.ceil(len(views) * (1.0 - max_outlier_fraction)))
    iterations = 0
    for _ in range(12):
        iterations += 1
        result = calibrate(kept, object_template, image_size)
        errors = np.asarray(result[-1], dtype=np.float64)
        for view, error in zip(kept, errors):
            view.reprojection_rms = float(error)
        median = float(np.median(errors))
        mad_sigma = 1.4826 * float(np.median(np.abs(errors - median)))
        robust_limit = max(0.3, median + 2.5 * mad_sigma)
        limit = min(absolute_limit, robust_limit)
        bad = np.flatnonzero(errors > limit).tolist()
        if not bad:
            break
        can_remove = len(kept) - minimum_keep
        if can_remove <= 0:
            break
        # Remove only a small worst-error batch, then recalibrate.  Reprojection
        # errors can move after each fit, so deleting every current outlier in
        # one pass is unnecessarily aggressive.
        batch_limit = max(1, math.ceil(len(kept) * 0.1))
        bad = sorted(bad, key=lambda i: errors[i], reverse=True)[
            : min(can_remove, batch_limit)
        ]
        rejected.extend(kept[i] for i in bad)
        bad_set = set(bad)
        kept = [view for i, view in enumerate(kept) if i not in bad_set]
    if result is None:
        raise RuntimeError("calibration did not run")
    # Ensure returned parameters correspond exactly to the final kept set.
    result = calibrate(kept, object_template, image_size)
    for view, error in zip(kept, result[-1]):
        view.reprojection_rms = float(error)
    final_errors = np.asarray(result[-1], dtype=np.float64)
    final_median = float(np.median(final_errors))
    final_mad_sigma = 1.4826 * float(np.median(np.abs(final_errors - final_median)))
    final_limit = min(absolute_limit, max(0.3, final_median + 2.5 * final_mad_sigma))
    diagnostics: dict[str, float | int] = {
        "iterations": iterations,
        "minimum_views_allowed": minimum_keep,
        "absolute_view_error_limit_px": absolute_limit,
        "final_effective_view_error_limit_px": final_limit,
        "remaining_above_effective_limit": int(np.count_nonzero(final_errors > final_limit)),
        "remaining_above_absolute_limit": int(
            np.count_nonzero(final_errors > absolute_limit)
        ),
    }
    return kept, rejected, result, diagnostics


def save_debug_images(
    video_path: Path,
    debug_dir: Path,
    views: list[View],
    board_size: tuple[int, int],
    matrix: np.ndarray,
    distortion: np.ndarray,
    new_matrix: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    for view in views:
        capture.set(cv2.CAP_PROP_POS_FRAMES, view.frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        drawn = frame.copy()
        cv2.drawChessboardCorners(drawn, board_size, view.corners, True)
        label = f"frame={view.frame_index}  RMS={view.reprojection_rms:.3f}px"
        cv2.putText(drawn, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / f"frame_{view.frame_index:08d}_corners.jpg"), drawn)
    # Save one side-by-side undistortion check.
    if views:
        middle = views[len(views) // 2]
        capture.set(cv2.CAP_PROP_POS_FRAMES, middle.frame_index)
        ok, frame = capture.read()
        if ok:
            corrected = cv2.undistort(frame, matrix, distortion, None, new_matrix)
            comparison = np.hstack([frame, corrected])
            cv2.putText(comparison, "original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(
                comparison,
                "undistorted",
                (frame.shape[1] + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(debug_dir / "undistortion_comparison.jpg"), comparison)
    capture.release()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not video_path.is_file():
        print(f"ERROR: video does not exist: {video_path}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"ERROR: OpenCV cannot open video: {video_path}", file=sys.stderr)
        return 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        print("ERROR: invalid video size or frame rate", file=sys.stderr)
        capture.release()
        return 1

    board_size = (args.board_cols, args.board_rows)
    image_size = (width, height)
    views, stats = collect_views(capture, board_size, image_size, fps, frame_count, args)
    capture.release()
    print(f"Detection summary: {stats}; usable={len(views)}")
    if len(views) < args.min_views:
        print(
            f"ERROR: only {len(views)} usable views; at least {args.min_views} are required.\n"
            "Check whether --board-cols/--board-rows are INNER-corner counts, and try "
            "--min-sharpness 0 or --min-board-area 0 if needed.",
            file=sys.stderr,
        )
        return 2

    views = select_diverse_views(views, args.max_views)
    print(f"Using {len(views)} pose-diverse views before reprojection filtering")
    object_template = make_object_points(board_size, args.square_size_mm)
    try:
        kept, outliers, result, filtering = reject_outliers(
            views,
            object_template,
            image_size,
            args.min_views,
            args.max_view_error,
            args.max_outlier_fraction,
        )
    except cv2.error as error:
        print(f"ERROR: OpenCV calibration failed: {error}", file=sys.stderr)
        return 3

    rms, matrix, distortion, rvecs, tvecs, per_view_errors = result
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        matrix, distortion, image_size, args.alpha, image_size
    )
    fov_x, fov_y, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(
        matrix, image_size, width, height
    )

    for view, error in zip(kept, per_view_errors):
        view.reprojection_rms = error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_json = {
        "schema_version": 1,
        "calibration_type": "monocular_pinhole",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_video": str(video_path),
        "image_size": {"width": width, "height": height},
        "video": {"fps": fps, "frame_count": frame_count},
        "checkerboard": {
            "inner_corners": {"columns": args.board_cols, "rows": args.board_rows},
            "squares": {"columns": args.board_cols + 1, "rows": args.board_rows + 1},
            "square_size_mm": args.square_size_mm,
        },
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "optimal_camera_matrix": new_matrix.tolist(),
        "valid_roi_xywh": [int(value) for value in roi],
        "rms_reprojection_error_px": rms,
        "mean_view_rms_px": float(np.mean(per_view_errors)),
        "max_view_rms_px": float(np.max(per_view_errors)),
        "field_of_view_degrees": {"horizontal": float(fov_x), "vertical": float(fov_y)},
        "calibration_matrix_values": {
            "focal_length": float(focal_length),
            "principal_point": [float(principal_point[0]), float(principal_point[1])],
            "aspect_ratio_fy_over_fx": float(aspect_ratio),
            "note": "Physical sensor size was not supplied; focal_length/principal_point use pixel-equivalent units.",
        },
        "sampling": {
            "sample_seconds": args.sample_seconds,
            "detection_statistics": stats,
            "detected_before_diversity_selection": stats["usable"],
            "selected_before_outlier_filter": len(views),
            "used_view_count": len(kept),
            "reprojection_outlier_count": len(outliers),
            "outlier_filter": filtering,
        },
        "used_views": [
            {
                "frame_index": view.frame_index,
                "time_seconds": view.time_seconds,
                "sharpness": view.sharpness,
                "board_area_ratio": view.board_area_ratio,
                "reprojection_rms_px": view.reprojection_rms,
                "rvec": np.asarray(rvec).reshape(-1).tolist(),
                "tvec_mm": np.asarray(tvec).reshape(-1).tolist(),
            }
            for view, rvec, tvec in zip(kept, rvecs, tvecs)
        ],
        "rejected_reprojection_outliers": [
            {
                "frame_index": view.frame_index,
                "time_seconds": view.time_seconds,
                "reprojection_rms_px": view.reprojection_rms,
            }
            for view in outliers
        ],
    }
    output_path.write_text(json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_debug_images:
        debug_dir = (
            args.debug_dir.expanduser().resolve()
            if args.debug_dir is not None
            else output_path.parent / f"{output_path.stem}_debug"
        )
        save_debug_images(video_path, debug_dir, kept, board_size, matrix, distortion, new_matrix)
        print(f"Debug images: {debug_dir}")

    print("\nCalibration completed")
    print(f"  used views: {len(kept)} (reprojection outliers removed: {len(outliers)})")
    print(f"  RMS reprojection error: {rms:.4f} px")
    print(
        f"  final per-view limit: {filtering['final_effective_view_error_limit_px']:.4f} px"
    )
    if filtering["remaining_above_effective_limit"]:
        print(
            "  WARNING: "
            f"{filtering['remaining_above_effective_limit']} view(s) still exceed the "
            "effective limit because the minimum-view/outlier-fraction safeguard was reached."
        )
    print(f"  fx={matrix[0, 0]:.4f}, fy={matrix[1, 1]:.4f}")
    print(f"  cx={matrix[0, 2]:.4f}, cy={matrix[1, 2]:.4f}")
    print(f"  distortion={distortion.reshape(-1).tolist()}")
    print(f"  result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
