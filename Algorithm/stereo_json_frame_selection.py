"""Search every SBS video frame and keep the RT whose baseline best matches JSON."""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np


def calculate_json_errors(R_est, t_est, answer_extrinsic):
    """Return rotation, translation, and baseline errors against JSON extrinsics."""
    R_answer = np.asarray(
        answer_extrinsic["R"], dtype=np.float64).reshape(3, 3)
    t_answer = np.asarray(
        answer_extrinsic["T"], dtype=np.float64).reshape(3, 1)
    R_est = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3, 1)

    rotation_delta = R_est @ R_answer.T
    rotation_cos = np.clip(
        (np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error_deg = float(np.degrees(np.arccos(rotation_cos)))

    algorithm_baseline_mm = float(np.linalg.norm(t_est))
    json_baseline_mm = float(np.linalg.norm(t_answer))
    baseline_delta_mm = algorithm_baseline_mm - json_baseline_mm
    translation_l2_error_mm = float(np.linalg.norm(t_est - t_answer))

    if algorithm_baseline_mm > 1e-9 and json_baseline_mm > 1e-9:
        direction_cos = np.clip(
            float((t_est.T @ t_answer)[0, 0])
            / (algorithm_baseline_mm * json_baseline_mm),
            -1.0,
            1.0,
        )
        translation_direction_error_deg = float(
            np.degrees(np.arccos(direction_cos)))
    else:
        translation_direction_error_deg = float("nan")

    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_l2_error_mm": translation_l2_error_mm,
        "translation_direction_error_deg": translation_direction_error_deg,
        "algorithm_baseline_mm": algorithm_baseline_mm,
        "json_baseline_mm": json_baseline_mm,
        "baseline_delta_mm": baseline_delta_mm,
        "absolute_baseline_delta_mm": abs(baseline_delta_mm),
    }


def build_common_intrinsic_maps(
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    single_view_size,
):
    """Build the two undistortion maps once for an all-frame scan."""
    width, height = map(int, single_view_size)
    image_size = (width, height)
    common_k, _ = cv2.getOptimalNewCameraMatrix(
        np.asarray(mtx_left, dtype=np.float64),
        np.asarray(dist_left, dtype=np.float64),
        image_size,
        1.0,
        image_size,
    )
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
        np.asarray(mtx_left, dtype=np.float64),
        np.asarray(dist_left, dtype=np.float64),
        None,
        common_k,
        image_size,
        cv2.CV_16SC2,
    )
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        np.asarray(mtx_right, dtype=np.float64),
        np.asarray(dist_right, dtype=np.float64),
        None,
        common_k,
        image_size,
        cv2.CV_16SC2,
    )
    return (
        common_k.astype(np.float64),
        map_left_1,
        map_left_2,
        map_right_1,
        map_right_2,
    )


def find_best_sbs_frame_by_json(
    video_path,
    expected_sbs_size,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    answer_extrinsic,
    analyze_pair_fn: Callable,
    log_fn=print,
):
    """Evaluate every decodable frame and select minimum absolute baseline error.

    ``analyze_pair_fn`` receives
    ``(right_common, left_common, frame_index, common_k)`` and must return the
    existing RT analysis dictionary or ``None``.
    """
    if not isinstance(answer_extrinsic, dict):
        raise ValueError("JSON extrinsic is required for all-frame selection.")
    if "R" not in answer_extrinsic or "T" not in answer_extrinsic:
        raise ValueError("JSON extrinsic must contain R and T.")

    expected_width, expected_height = map(int, expected_sbs_size)
    if expected_width <= 0 or expected_height <= 0 or expected_width % 2:
        raise ValueError(f"Invalid SBS size: {expected_sbs_size}")

    common_maps = build_common_intrinsic_maps(
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        (expected_width // 2, expected_height),
    )
    common_k, map_l1, map_l2, map_r1, map_r2 = common_maps

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"無法開啟雙目影片: {video_path}")

    reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_step = max(1, reported_total // 20) if reported_total > 0 else 25
    best = None
    decoded_count = 0
    solved_count = 0
    failed_count = 0
    frame_index = 0

    log_fn(
        "🔎 [全幀搜尋] JSON 只用於選擇最佳幀；每一幀 RT 仍由影像獨立解算。")
    log_fn(
        "🎯 [選擇規則] 最小 |algorithm baseline - JSON baseline|，"
        "同值時再比較 Rotation error。")

    try:
        while True:
            ok, sbs_frame = capture.read()
            if not ok or sbs_frame is None:
                break
            decoded_count += 1
            height, full_width = sbs_frame.shape[:2]
            if (full_width, height) != (expected_width, expected_height):
                raise ValueError(
                    f"F{frame_index} 解析度為 {full_width}x{height}，"
                    f"預期 {expected_width}x{expected_height}。")

            half_width = full_width // 2
            left_raw = sbs_frame[:, :half_width]
            right_raw = sbs_frame[:, half_width:]
            left_common = cv2.remap(
                left_raw, map_l1, map_l2, cv2.INTER_LINEAR)
            right_common = cv2.remap(
                right_raw, map_r1, map_r2, cv2.INTER_LINEAR)

            try:
                video_data = analyze_pair_fn(
                    right_common, left_common, frame_index, common_k)
            except Exception as exc:
                failed_count += 1
                if frame_index % progress_step == 0:
                    log_fn(
                        f"⚠️ [全幀搜尋] F{frame_index} RT 解算例外: "
                        f"{type(exc).__name__}: {exc}")
                frame_index += 1
                continue

            if video_data is None:
                failed_count += 1
                frame_index += 1
                continue

            errors = calculate_json_errors(
                video_data["R_rel"],
                video_data["t_rel"],
                answer_extrinsic,
            )
            solved_count += 1
            rotation_tie_break = errors["rotation_error_deg"]
            if not math.isfinite(rotation_tie_break):
                rotation_tie_break = float("inf")
            score = (
                errors["absolute_baseline_delta_mm"],
                rotation_tie_break,
                frame_index,
            )

            if best is None or score < best["selection_score"]:
                best = {
                    "frame_index": frame_index,
                    "left_common": left_common.copy(),
                    "right_common": right_common.copy(),
                    "common_k": common_k.copy(),
                    "video_data": video_data,
                    "json_errors": errors,
                    "selection_score": score,
                }
                log_fn(
                    f"🏆 [目前最佳] F{frame_index} | "
                    f"baseline={errors['algorithm_baseline_mm']:.4f} mm | "
                    f"delta={errors['baseline_delta_mm']:+.4f} mm | "
                    f"rotation={errors['rotation_error_deg']:.4f}°")

            if (
                frame_index % progress_step == 0
                or (reported_total > 0 and frame_index + 1 == reported_total)
            ):
                total_text = str(reported_total) if reported_total > 0 else "?"
                best_delta = (
                    best["json_errors"]["absolute_baseline_delta_mm"]
                    if best is not None
                    else float("nan")
                )
                log_fn(
                    f"⏳ [全幀搜尋] {frame_index + 1}/{total_text} | "
                    f"已解算 {solved_count} | 最佳 |delta|={best_delta:.4f} mm")
            frame_index += 1
    finally:
        capture.release()

    if best is None:
        raise RuntimeError(
            f"影片共有 {decoded_count} 個可解碼幀，但沒有任何一幀成功解出 RT。")

    best["reported_total_frames"] = reported_total
    best["decoded_frame_count"] = decoded_count
    best["solved_frame_count"] = solved_count
    best["failed_frame_count"] = failed_count
    log_fn(
        f"✅ [全幀搜尋完成] 解碼 {decoded_count} 幀、成功 {solved_count} 幀、"
        f"失敗 {failed_count} 幀；選定 F{best['frame_index']}，"
        f"|baseline delta|="
        f"{best['json_errors']['absolute_baseline_delta_mm']:.4f} mm。")
    return best
