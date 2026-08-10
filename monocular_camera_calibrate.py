"""Calibrate one pinhole camera from checkerboard photographs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_DIR = "monocular_calibration_images"
DEFAULT_OUTPUT_JSON = "monocular_calibration_result.json"
DEFAULT_BOARD_COLS = 10
DEFAULT_BOARD_ROWS = 16
DEFAULT_SQUARE_SIZE_MM = 15.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate monocular camera intrinsics from checkerboard images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(DEFAULT_IMAGE_DIR),
        help=f"Directory containing calibration images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_JSON),
        help=f"Output calibration JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help=f"Checkerboard inner corners across columns (default: {DEFAULT_BOARD_COLS})",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help=f"Checkerboard inner corners across rows (default: {DEFAULT_BOARD_ROWS})",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=DEFAULT_SQUARE_SIZE_MM,
        help=f"Checkerboard square edge length in millimetres (default: {DEFAULT_SQUARE_SIZE_MM})",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Minimum number of usable images required (default: 10)",
    )
    parser.add_argument(
        "--detector",
        choices=("sb", "classic"),
        default="sb",
        help="Checkerboard detector; sb is usually more robust (default: sb)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search subdirectories for images",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Free scaling for optimal camera matrix, from 0 to 1 (default: 0)",
    )
    args = parser.parse_args()

    if args.board_cols < 2 or args.board_rows < 2:
        parser.error("--board-cols and --board-rows must both be at least 2")
    if args.square_size_mm <= 0:
        parser.error("--square-size-mm must be greater than 0")
    if args.min_images < 3:
        parser.error("--min-images must be at least 3")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    return args


def find_image_paths(image_dir: Path, recursive: bool) -> list[Path]:
    candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def detect_checkerboard(gray, board_size: tuple[int, int], detector: str):
    if detector == "sb" and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        return cv2.findChessboardCornersSB(gray, board_size, flags)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def make_object_points(
    board_size: tuple[int, int], square_size_mm: float
) -> np.ndarray:
    points = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    points[:, :2] *= square_size_mm
    return points


def reprojection_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs,
) -> tuple[list[float], list[float]]:
    rms_errors = []
    mean_errors = []
    for object_view, image_view, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            object_view, rvec, tvec, camera_matrix, dist_coeffs
        )
        distances = np.linalg.norm(
            image_view.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
        )
        rms_errors.append(float(np.sqrt(np.mean(np.square(distances)))))
        mean_errors.append(float(np.mean(distances)))
    return rms_errors, mean_errors


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not image_dir.is_dir():
        print(f"[錯誤] 找不到照片資料夾：{image_dir}")
        return 1

    image_paths = find_image_paths(image_dir, args.recursive)
    if not image_paths:
        print(f"[錯誤] 資料夾內沒有支援的影像：{image_dir}")
        return 1

    board_size = (args.board_cols, args.board_rows)
    detector = args.detector
    if detector == "sb" and not hasattr(cv2, "findChessboardCornersSB"):
        print("[警告] 此 OpenCV 沒有 findChessboardCornersSB，改用 classic detector。")
        detector = "classic"
    object_template = make_object_points(board_size, args.square_size_mm)
    object_points = []
    image_points = []
    used_paths: list[Path] = []
    rejected_images: list[dict[str, str]] = []
    image_size: tuple[int, int] | None = None

    print(f"找到 {len(image_paths)} 張照片，開始偵測 {board_size[0]}x{board_size[1]} 內角點…")
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        relative_name = str(image_path.relative_to(image_dir))
        if image is None:
            rejected_images.append({"image": relative_name, "reason": "read_failed"})
            print(f"[{index}/{len(image_paths)}] 讀取失敗：{relative_name}")
            continue

        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected_images.append(
                {
                    "image": relative_name,
                    "reason": f"size_mismatch_{current_size[0]}x{current_size[1]}",
                }
            )
            print(
                f"[{index}/{len(image_paths)}] 略過（尺寸 {current_size[0]}x{current_size[1]} "
                f"不等於 {image_size[0]}x{image_size[1]}）：{relative_name}"
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_checkerboard(gray, board_size, detector)
        if not found:
            rejected_images.append({"image": relative_name, "reason": "board_not_found"})
            print(f"[{index}/{len(image_paths)}] 找不到棋盤格：{relative_name}")
            continue

        object_points.append(object_template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        used_paths.append(image_path)
        print(f"[{index}/{len(image_paths)}] 有效：{relative_name}")

    if len(used_paths) < args.min_images:
        print(
            f"[錯誤] 僅有 {len(used_paths)} 張有效照片，至少需要 {args.min_images} 張。"
        )
        print("請從不同距離、角度與畫面位置拍攝清晰棋盤格照片後再試。")
        return 1
    if image_size is None:
        print("[錯誤] 無法取得影像尺寸。")
        return 1

    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    except cv2.error as error:
        print(f"[錯誤] OpenCV 無法完成標定：{error}")
        return 1
    per_image_rms, per_image_mean = reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        args.alpha,
        image_size,
    )

    used_names = [str(path.relative_to(image_dir)) for path in used_paths]
    result = {
        "schema_version": 1,
        "calibration_type": "monocular_pinhole",
        "calibration_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "checkerboard": {
            "inner_corners": {"columns": board_size[0], "rows": board_size[1]},
            "square_size_mm": float(args.square_size_mm),
        },
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(np.mean(per_image_mean)),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "optimal_camera_matrix": {
            "alpha": float(args.alpha),
            "matrix": optimal_matrix.tolist(),
            "valid_roi": [int(value) for value in roi],
        },
        "images": {
            "source_directory": str(image_dir),
            "found_count": len(image_paths),
            "used_count": len(used_paths),
            "rejected_count": len(rejected_images),
            "used": [
                {
                    "image": name,
                    "rms_reprojection_error_px": error_rms,
                    "mean_reprojection_error_px": error_mean,
                }
                for name, error_rms, error_mean in zip(
                    used_names, per_image_rms, per_image_mean
                )
            ],
            "rejected": rejected_images,
        },
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"[錯誤] 無法寫入標定結果 {output_path}：{error}")
        return 1

    print("\n標定完成")
    print(f"有效照片：{len(used_paths)} / {len(image_paths)}")
    print(f"OpenCV RMS 重投影誤差：{rms:.4f} px")
    print(f"平均重投影誤差：{np.mean(per_image_mean):.4f} px")
    print("相機內參矩陣：")
    print(camera_matrix)
    print(f"畸變係數：{dist_coeffs.reshape(-1)}")
    print(f"結果已儲存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
