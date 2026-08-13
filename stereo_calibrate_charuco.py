"""Calibrate an SBS stereo camera using gen_charuco_a3_png.py's board.

The script detects ChArUco chessboard intersections, estimates left and right
camera intrinsics, then uses ChArUco corner IDs visible in both cameras to
estimate the left-to-right stereo extrinsics. Partial board occlusion is allowed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import gen_charuco_a3_png as pattern_config


DEFAULT_IMAGE_DIR = "HBVCAM_Charuco_12x8"
DEFAULT_OUTPUT_JSON = "calibration_result_HBVCAM_charuco.json"
DEFAULT_LAYOUT = "side-by-side"
DEFAULT_MIN_VIEWS = 8
DEFAULT_MIN_CORNERS = 12
DEFAULT_ALPHA = 0.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(DEFAULT_IMAGE_DIR),
        help=f"Directory containing packed stereo images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_JSON),
        help=f"Output calibration JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--layout",
        choices=("side-by-side", "top-bottom"),
        default=DEFAULT_LAYOUT,
        help="Stereo image packing (default: side-by-side)",
    )
    parser.add_argument(
        "--swap-cameras",
        action="store_true",
        help="Swap the first and second image halves",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional output directory for detection visualization images",
    )
    parser.add_argument("--min-views", type=int, default=DEFAULT_MIN_VIEWS)
    parser.add_argument(
        "--min-corners",
        type=int,
        default=DEFAULT_MIN_CORNERS,
        help=(
            "Minimum ChArUco corners per camera and common to both cameras "
            f"(default: {DEFAULT_MIN_CORNERS})"
        ),
    )
    parser.add_argument("--squares-x", type=int, default=pattern_config.SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=pattern_config.SQUARES_Y)
    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=pattern_config.SQUARE_LENGTH_MM,
    )
    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=pattern_config.MARKER_LENGTH_MM,
    )
    parser.add_argument(
        "--marker-start-id",
        type=int,
        default=pattern_config.MARKER_START_ID,
    )
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        default=pattern_config.LEGACY_PATTERN,
        help="Use the pre-OpenCV-4.6 ChArUco pattern layout",
    )
    parser.add_argument(
        "--rational-model",
        action="store_true",
        help="Use OpenCV's 8-coefficient rational radial distortion model",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Stereo rectification free scaling in [0, 1] (default: 0)",
    )
    args = parser.parse_args()

    total_corners = (args.squares_x - 1) * (args.squares_y - 1)
    if args.squares_x < 2 or args.squares_y < 2:
        parser.error("--squares-x and --squares-y must both be at least 2")
    if args.square_length_mm <= 0:
        parser.error("--square-length-mm must be positive")
    if not 0 < args.marker_length_mm < args.square_length_mm:
        parser.error("--marker-length-mm must be positive and smaller than a square")
    if args.marker_start_id < 0:
        parser.error("--marker-start-id cannot be negative")
    if args.min_views < 3:
        parser.error("--min-views must be at least 3")
    if args.min_corners < 4:
        parser.error("--min-corners must be at least 4")
    if args.min_corners > total_corners:
        parser.error(f"--min-corners cannot exceed board total {total_corners}")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    return args


def find_image_paths(image_dir: Path, recursive: bool) -> list[Path]:
    candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_stereo_image(
    image: np.ndarray,
    layout: str,
    swap_cameras: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if layout == "side-by-side":
        if width % 2:
            raise ValueError(f"side-by-side image width must be even, got {width}")
        split = width // 2
        left, right = image[:, :split], image[:, split:]
    else:
        if height % 2:
            raise ValueError(f"top-bottom image height must be even, got {height}")
        split = height // 2
        left, right = image[:split, :], image[split:, :]

    if swap_cameras:
        left, right = right, left
    return left, right


def create_board_and_detector(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    legacy_pattern: bool,
    marker_start_id: int,
):
    board, dictionary = pattern_config.create_board(
        squares_x,
        squares_y,
        square_length_mm,
        marker_length_mm,
        legacy_pattern,
        marker_start_id,
    )
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_parameters.tryRefineMarkers = True
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_parameters.cornerRefinementWinSize = 5
    detector_parameters.cornerRefinementMaxIterations = 50
    detector_parameters.cornerRefinementMinAccuracy = 0.001
    detector = cv2.aruco.CharucoDetector(
        board,
        charuco_parameters,
        detector_parameters,
    )
    return board, dictionary, detector


def detect_charuco(gray: np.ndarray, detector):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray
    )
    detections: dict[int, np.ndarray] = {}
    if charuco_ids is not None:
        for point, corner_id in zip(charuco_corners, charuco_ids.flatten()):
            detections[int(corner_id)] = np.asarray(point, dtype=np.float32).reshape(2)
    marker_count = 0 if marker_ids is None else int(marker_ids.size)
    return detections, charuco_corners, charuco_ids, marker_corners, marker_ids, marker_count


def collect_points(
    detections: dict[int, np.ndarray],
    corner_ids: list[int],
    board_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(
        [board_corners[corner_id] for corner_id in corner_ids],
        dtype=np.float32,
    ).reshape(-1, 3)
    image_points = np.asarray(
        [detections[corner_id] for corner_id in corner_ids],
        dtype=np.float32,
    ).reshape(-1, 2)
    return object_points, image_points


def draw_detection(
    image: np.ndarray,
    charuco_corners,
    charuco_ids,
    marker_corners,
    marker_ids,
    label: str,
) -> np.ndarray:
    display = image.copy()
    if marker_ids is not None and marker_corners:
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
    if charuco_ids is not None and charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            display,
            charuco_corners,
            charuco_ids,
            (0, 0, 255),
        )
    marker_count = 0 if marker_ids is None else int(marker_ids.size)
    corner_count = 0 if charuco_ids is None else int(charuco_ids.size)
    cv2.putText(
        display,
        f"{label}: markers={marker_count}, ChArUco corners={corner_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def save_debug_image(
    debug_dir: Path,
    relative_name: str,
    left: np.ndarray,
    right: np.ndarray,
    left_raw,
    right_raw,
) -> None:
    left_display = draw_detection(left, *left_raw, "L")
    right_display = draw_detection(right, *right_raw, "R")
    combined = cv2.hconcat([left_display, right_display])
    safe_name = relative_name.replace("/", "__").replace("\\", "__")
    output_path = debug_dir / f"{Path(safe_name).stem}_charuco.jpg"
    cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])


def per_view_reprojection_rms(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs,
    tvecs,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for object_view, image_view, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        residual = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return errors


def dictionary_name(dictionary_id: int) -> str:
    return pattern_config.dictionary_name(dictionary_id)


def int_roi(roi) -> list[int]:
    return [int(value) for value in roi]


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    debug_dir = args.debug_dir.expanduser().resolve() if args.debug_dir else None

    if not image_dir.is_dir():
        print(f"[ERROR] Image directory does not exist: {image_dir}")
        return 1
    image_paths = find_image_paths(image_dir, args.recursive)
    if not image_paths:
        print(f"[ERROR] No supported images found in: {image_dir}")
        return 1

    try:
        board, dictionary, detector = create_board_and_detector(
            args.squares_x,
            args.squares_y,
            args.square_length_mm,
            args.marker_length_mm,
            args.legacy_pattern,
            args.marker_start_id,
        )
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(
        -1, 3
    )
    total_charuco_corners = int(board_corners.shape[0])
    marker_count = int(board.getIds().size)
    marker_end_id = args.marker_start_id + marker_count - 1
    if marker_count > int(dictionary.bytesList.shape[0]):
        print("[ERROR] Selected dictionary does not contain enough marker IDs.")
        return 1
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    left_object_views: list[np.ndarray] = []
    left_image_views: list[np.ndarray] = []
    left_view_names: list[str] = []
    right_object_views: list[np.ndarray] = []
    right_image_views: list[np.ndarray] = []
    right_view_names: list[str] = []
    stereo_object_views: list[np.ndarray] = []
    stereo_left_views: list[np.ndarray] = []
    stereo_right_views: list[np.ndarray] = []
    stereo_view_names: list[str] = []
    stereo_common_counts: list[int] = []
    image_records: list[dict[str, object]] = []
    image_size: tuple[int, int] | None = None

    print(
        f"Pattern: {dictionary_name(pattern_config.ARUCO_DICT)}, "
        f"{args.squares_x}x{args.squares_y} squares, "
        f"square={args.square_length_mm:g} mm, marker={args.marker_length_mm:g} mm, "
        f"marker IDs={args.marker_start_id}-{marker_end_id}, "
        f"ChArUco corners={total_charuco_corners}"
    )
    print(f"Found {len(image_paths)} image(s) in {image_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        relative_name = str(image_path.relative_to(image_dir))
        record: dict[str, object] = {"image": relative_name}
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            record["status"] = "read_failed"
            image_records.append(record)
            print(f"[{index}/{len(image_paths)}] read failed: {relative_name}")
            continue

        try:
            left, right = split_stereo_image(image, args.layout, args.swap_cameras)
        except ValueError as error:
            record.update({"status": "invalid_stereo_shape", "reason": str(error)})
            image_records.append(record)
            print(f"[{index}/{len(image_paths)}] rejected: {relative_name} ({error})")
            continue

        current_size = (left.shape[1], left.shape[0])
        if right.shape[:2] != left.shape[:2]:
            record["status"] = "left_right_size_mismatch"
            image_records.append(record)
            print(f"[{index}/{len(image_paths)}] unequal halves: {relative_name}")
            continue
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            record.update(
                {"status": "image_size_mismatch", "size": list(current_size)}
            )
            image_records.append(record)
            print(f"[{index}/{len(image_paths)}] size mismatch: {relative_name}")
            continue

        left_result = detect_charuco(
            cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), detector
        )
        right_result = detect_charuco(
            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), detector
        )
        left_detections = left_result[0]
        right_detections = right_result[0]
        left_ids = sorted(left_detections)
        right_ids = sorted(right_detections)
        common_ids = sorted(set(left_ids).intersection(right_ids))

        if len(left_ids) >= args.min_corners:
            object_view, image_view = collect_points(
                left_detections, left_ids, board_corners
            )
            left_object_views.append(object_view)
            left_image_views.append(image_view)
            left_view_names.append(relative_name)
        if len(right_ids) >= args.min_corners:
            object_view, image_view = collect_points(
                right_detections, right_ids, board_corners
            )
            right_object_views.append(object_view)
            right_image_views.append(image_view)
            right_view_names.append(relative_name)
        if len(common_ids) >= args.min_corners:
            object_view, left_view = collect_points(
                left_detections, common_ids, board_corners
            )
            _, right_view = collect_points(
                right_detections, common_ids, board_corners
            )
            stereo_object_views.append(object_view)
            stereo_left_views.append(left_view)
            stereo_right_views.append(right_view)
            stereo_view_names.append(relative_name)
            stereo_common_counts.append(len(common_ids))

        stereo_usable = len(common_ids) >= args.min_corners
        record.update(
            {
                "status": "stereo_usable" if stereo_usable else "insufficient_common_corners",
                "left_aruco_markers": left_result[5],
                "right_aruco_markers": right_result[5],
                "left_charuco_corners": len(left_ids),
                "right_charuco_corners": len(right_ids),
                "common_charuco_corners": len(common_ids),
            }
        )
        image_records.append(record)
        print(
            f"[{index}/{len(image_paths)}] {relative_name}: "
            f"L={len(left_ids)}, R={len(right_ids)}, common={len(common_ids)} "
            f"({'use' if stereo_usable else 'skip'})"
        )

        if debug_dir is not None:
            left_raw = left_result[1:5]
            right_raw = right_result[1:5]
            save_debug_image(
                debug_dir,
                relative_name,
                left,
                right,
                left_raw,
                right_raw,
            )

    counts = {
        "left": len(left_object_views),
        "right": len(right_object_views),
        "stereo": len(stereo_object_views),
    }
    if image_size is None or any(count < args.min_views for count in counts.values()):
        print(
            f"[ERROR] Not enough usable views (required {args.min_views}): "
            f"left={counts['left']}, right={counts['right']}, stereo={counts['stereo']}"
        )
        print(
            "Capture the flat board at varied positions, distances, and tilts, "
            "with enough common ChArUco corners in both cameras."
        )
        return 1

    calibration_flags = cv2.CALIB_RATIONAL_MODEL if args.rational_model else 0
    try:
        print("Calibrating left camera intrinsics...")
        (
            left_rms,
            left_matrix,
            left_distortion,
            left_rvecs,
            left_tvecs,
        ) = cv2.calibrateCamera(
            left_object_views,
            left_image_views,
            image_size,
            None,
            None,
            flags=calibration_flags,
        )
        print("Calibrating right camera intrinsics...")
        (
            right_rms,
            right_matrix,
            right_distortion,
            right_rvecs,
            right_tvecs,
        ) = cv2.calibrateCamera(
            right_object_views,
            right_image_views,
            image_size,
            None,
            None,
            flags=calibration_flags,
        )

        print("Calibrating left-to-right stereo extrinsics...")
        stereo_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-7,
        )
        (
            stereo_rms,
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            rotation,
            translation,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            stereo_object_views,
            stereo_left_views,
            stereo_right_views,
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            image_size,
            criteria=stereo_criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        (
            rectify_left,
            rectify_right,
            projection_left,
            projection_right,
            disparity_to_depth,
            valid_roi_left,
            valid_roi_right,
        ) = cv2.stereoRectify(
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            image_size,
            rotation,
            translation,
            alpha=args.alpha,
        )
    except cv2.error as error:
        print(f"[ERROR] OpenCV calibration failed:\n{error}")
        return 1

    left_view_rms = per_view_reprojection_rms(
        left_object_views,
        left_image_views,
        left_rvecs,
        left_tvecs,
        left_matrix,
        left_distortion,
    )
    right_view_rms = per_view_reprojection_rms(
        right_object_views,
        right_image_views,
        right_rvecs,
        right_tvecs,
        right_matrix,
        right_distortion,
    )
    baseline_mm = float(np.linalg.norm(translation))

    result = {
        "schema_version": 1,
        "calibration_type": "stereo_pinhole_charuco",
        "calibration_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_size_per_camera": {"width": image_size[0], "height": image_size[1]},
        "reprojection_error": float(stereo_rms),
        "intrinsic_L": {
            "rms_reprojection_error_px": float(left_rms),
            "matrix": left_matrix.tolist(),
            "distortion": left_distortion.reshape(-1).tolist(),
            "per_view_rms_px": [
                {"image": name, "rms": error}
                for name, error in zip(left_view_names, left_view_rms)
            ],
        },
        "intrinsic_R": {
            "rms_reprojection_error_px": float(right_rms),
            "matrix": right_matrix.tolist(),
            "distortion": right_distortion.reshape(-1).tolist(),
            "per_view_rms_px": [
                {"image": name, "rms": error}
                for name, error in zip(right_view_names, right_view_rms)
            ],
        },
        "extrinsic": {
            "convention": "X_right = R * X_left + T",
            "units": "millimetres",
            "stereo_rms_reprojection_error_px": float(stereo_rms),
            "baseline_mm": baseline_mm,
            "R": rotation.tolist(),
            "T": translation.reshape(-1).tolist(),
            "E": essential.tolist(),
            "F": fundamental.tolist(),
        },
        "rectification": {
            "alpha": args.alpha,
            "R1": rectify_left.tolist(),
            "R2": rectify_right.tolist(),
            "P1": projection_left.tolist(),
            "P2": projection_right.tolist(),
            "Q": disparity_to_depth.tolist(),
            "valid_roi_L": int_roi(valid_roi_left),
            "valid_roi_R": int_roi(valid_roi_right),
        },
        "pattern": {
            "source": "gen_charuco_a3_png.py",
            "dictionary": dictionary_name(pattern_config.ARUCO_DICT),
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "charuco_corner_count": total_charuco_corners,
            "marker_count": marker_count,
            "marker_start_id": args.marker_start_id,
            "marker_end_id": marker_end_id,
            "marker_ids": board.getIds().reshape(-1).astype(int).tolist(),
            "square_length_mm": args.square_length_mm,
            "marker_length_mm": args.marker_length_mm,
            "legacy_pattern": args.legacy_pattern,
            "coordinate_system": (
                "board top-left outer corner is (0,0,0); +X right, +Y down, "
                "all ChArUco corners have Z=0"
            ),
        },
        "dataset": {
            "image_directory": str(image_dir),
            "layout": args.layout,
            "swap_cameras": args.swap_cameras,
            "minimum_corners": args.min_corners,
            "minimum_views": args.min_views,
            "rational_model": args.rational_model,
            "usable_left_views": len(left_object_views),
            "usable_right_views": len(right_object_views),
            "usable_stereo_views": len(stereo_object_views),
            "stereo_views": [
                {"image": name, "common_charuco_corners": count}
                for name, count in zip(stereo_view_names, stereo_common_counts)
            ],
            "images": image_records,
        },
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, ensure_ascii=False)
    except OSError as error:
        print(f"[ERROR] Could not write calibration JSON: {output_path}\n{error}")
        return 1

    print("Calibration complete.")
    print(f"Left intrinsic RMS : {left_rms:.4f} px")
    print(f"Right intrinsic RMS: {right_rms:.4f} px")
    print(f"Stereo RMS         : {stereo_rms:.4f} px")
    print(f"Baseline           : {baseline_mm:.4f} mm")
    print(f"Saved JSON         : {output_path}")
    if debug_dir is not None:
        print(f"Detection images   : {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
