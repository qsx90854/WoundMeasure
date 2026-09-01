#!/usr/bin/env python3
"""Two-video, marker-only RT validation UI.

This tool deliberately lives outside the production Zebra application.  Video A
and Video B are treated as two independent temporal segments: the validation
backend selects one endpoint from each segment and estimates the B-to-A relative
pose without using SIFT.

The backend is imported lazily so the UI can still start and report a useful
error if its optional validation module is not available.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "雙影片 Marker-only RT 驗證工具"
UI_BUILD = "2026-08-26 fixed-camera-v5.8-outer-edge-refine"
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm"),
    ("All files", "*.*"),
]
JSON_FILE_TYPES = [("JSON files", "*.json"), ("All files", "*.*")]

CORNER_MODE_LABEL_TO_VALUE = {
    "Raw detectMarkers": "RAW",
    "SubPix 3x3": "SUBPIX_3",
    "SubPix 5x5（原本）": "SUBPIX_5",
    "Contour refine": "CONTOUR",
    "AprilTag refine": "APRILTAG",
}
DETECTOR_PRESET_LABEL_TO_VALUE = {
    "OpenCV Default": "DEFAULT",
    "LCD Robust": "LCD_ROBUST",
    "LCD Aggressive": "LCD_AGGRESSIVE",
}
DISPLAY_VIEW_MODES = ("原始彩色", "偵測灰階", "Adaptive binary（診斷）")


def load_monocular_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the common calibration JSON layouts used in this workspace.

    Supported layouts include both::

        {"intrinsic_L": {"matrix": ..., "distortion": ...}}

    and::

        {"camera_matrix": ..., "dist_coeffs": ...}
    """

    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    matrix: Any = None
    distortion: Any = None
    source = ""

    intrinsic_l = data.get("intrinsic_L")
    if isinstance(intrinsic_l, dict):
        matrix = intrinsic_l.get("matrix") or intrinsic_l.get("camera_matrix")
        distortion = (
            intrinsic_l.get("distortion")
            if intrinsic_l.get("distortion") is not None
            else intrinsic_l.get("dist_coeffs")
        )
        source = "intrinsic_L"

    if matrix is None:
        matrix = data.get("camera_matrix")
        distortion = (
            data.get("dist_coeffs")
            if data.get("dist_coeffs") is not None
            else data.get("distortion")
        )
        source = "camera_matrix"

    if matrix is None and isinstance(data.get("intrinsic"), dict):
        intrinsic = data["intrinsic"]
        matrix = intrinsic.get("matrix") or intrinsic.get("camera_matrix")
        distortion = (
            intrinsic.get("distortion")
            if intrinsic.get("distortion") is not None
            else intrinsic.get("dist_coeffs")
        )
        source = "intrinsic"

    if matrix is None:
        raise ValueError(
            "標定 JSON 找不到相機內參；需要 intrinsic_L.matrix 或 camera_matrix。"
        )
    if distortion is None:
        raise ValueError(
            "標定 JSON 找不到畸變係數；需要 intrinsic_L.distortion 或 dist_coeffs。"
        )

    camera_matrix = np.asarray(matrix, dtype=np.float64)
    dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"相機內參矩陣必須是 3x3，目前為 {camera_matrix.shape}。")
    if dist_coeffs.size < 4:
        raise ValueError(f"畸變係數至少需要 4 個，目前只有 {dist_coeffs.size} 個。")
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(dist_coeffs)):
        raise ValueError("標定 JSON 含有 NaN 或 Infinity。")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("fx 與 fy 必須大於 0。")

    metadata = {
        "path": str(calibration_path),
        "source": source,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "distortion_count": int(dist_coeffs.size),
    }
    return camera_matrix, dist_coeffs, metadata


def _marker_colour(marker_id: Any) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR colour for one marker ID."""

    try:
        seed = int(marker_id)
    except (TypeError, ValueError):
        seed = sum(ord(ch) for ch in str(marker_id))
    palette = (
        (0, 255, 255),
        (60, 220, 60),
        (255, 180, 0),
        (255, 80, 210),
        (80, 180, 255),
        (230, 160, 70),
    )
    return palette[seed % len(palette)]


def _iter_marker_corners(corners: Any) -> Iterable[tuple[Any, np.ndarray]]:
    """Yield ``(marker_id, 4x2 corners)`` from several practical layouts."""

    if corners is None:
        return

    if isinstance(corners, dict):
        for marker_id, value in corners.items():
            if isinstance(value, dict):
                value = value.get("corners", value.get("points"))
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32).reshape(-1, 2)
            if len(array) >= 4 and np.all(np.isfinite(array[:4])):
                yield marker_id, array[:4]
        return

    # Also accept a list of {id, corners} records for exported/reloaded results.
    if isinstance(corners, (list, tuple)):
        for index, value in enumerate(corners):
            marker_id: Any = index
            points: Any = value
            if isinstance(value, dict):
                marker_id = value.get("id", value.get("marker_id", index))
                points = value.get("corners", value.get("points"))
            if points is None:
                continue
            array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if len(array) >= 4 and np.all(np.isfinite(array[:4])):
                yield marker_id, array[:4]


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Return an 8-bit BGR display copy without drawing any overlays."""
    if frame is None:
        raise ValueError("Cannot display a null frame")
    output = np.asarray(frame).copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif output.ndim == 3 and output.shape[2] == 4:
        output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
    return output


def draw_marker_overlay_post_scale(
        shown: np.ndarray, refined_corners: Any, raw_corners: Any, caption: str, *,
        source_x0: float, source_y0: float,
        source_crop_w: float, source_crop_h: float,
        show_raw_vs_subpix: bool = True) -> np.ndarray:
    """Draw raw/selected corner diagnostics after crop/resize.

    Final corner coordinates are blue ``+`` crosshairs.  Raw CORNER_REFINE_NONE
    detectMarkers coordinates are yellow ``x`` crosshairs when diagnostic
    comparison is enabled.  The source->display transform follows OpenCV
    resize pixel-centre geometry: (x+0.5)*scale-0.5.
    """
    output = _ensure_bgr(shown)
    shown_h, shown_w = output.shape[:2]
    sx = shown_w / max(float(source_crop_w), 1e-9)
    sy = shown_h / max(float(source_crop_h), 1e-9)

    def map_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 2)
        mapped = np.empty_like(points, dtype=np.float64)
        mapped[:, 0] = (points[:, 0] - float(source_x0) + 0.5) * sx - 0.5
        mapped[:, 1] = (points[:, 1] - float(source_y0) + 0.5) * sy - 0.5
        return mapped

    refined_by_id = {mid: pts for mid, pts in _iter_marker_corners(refined_corners)}
    raw_by_id = {mid: pts for mid, pts in _iter_marker_corners(raw_corners)}
    marker_ids = sorted(set(refined_by_id) | set(raw_by_id), key=lambda value: str(value))

    raw_colour = (0, 255, 255)      # yellow in BGR
    refined_colour = (255, 80, 40)  # bright blue-ish in BGR
    text_colour = (245, 245, 245)
    arm = 8
    gap = 2

    for marker_id in marker_ids:
        visible_points = []

        if show_raw_vs_subpix and marker_id in raw_by_id:
            for xf, yf in map_points(raw_by_id[marker_id]):
                x = int(round(float(xf)))
                y = int(round(float(yf)))
                if x < -arm or x >= shown_w + arm or y < -arm or y >= shown_h + arm:
                    continue
                # Thin yellow 'x', centre left clear.
                cv2.line(output, (x - arm, y - arm), (x - gap, y - gap), raw_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x + gap, y + gap), (x + arm, y + arm), raw_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x - arm, y + arm), (x - gap, y + gap), raw_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x + gap, y - gap), (x + arm, y - arm), raw_colour, 1, cv2.LINE_AA)

        if marker_id in refined_by_id:
            for corner_index, (xf, yf) in enumerate(map_points(refined_by_id[marker_id])):
                x = int(round(float(xf)))
                y = int(round(float(yf)))
                if x < -arm or x >= shown_w + arm or y < -arm or y >= shown_h + arm:
                    continue
                visible_points.append((x, y))
                # Thin blue '+', centre left clear.
                cv2.line(output, (x - arm, y), (x - gap, y), refined_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x + gap, y), (x + arm, y), refined_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x, y - arm), (x, y - gap), refined_colour, 1, cv2.LINE_AA)
                cv2.line(output, (x, y + gap), (x, y + arm), refined_colour, 1, cv2.LINE_AA)
                cv2.putText(
                    output, str(corner_index),
                    (x + arm + 3, y - arm - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, refined_colour, 1, cv2.LINE_AA)

        if visible_points:
            cx = int(round(sum(x for x, _ in visible_points) / len(visible_points)))
            cy = int(round(sum(y for _, y in visible_points) / len(visible_points)))
            cv2.putText(
                output, f"ID {marker_id}", (cx + 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_colour, 1, cv2.LINE_AA)

    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (20, 20, 20), -1)
    legend = " | RAW=yellow x, FINAL=blue +" if show_raw_vs_subpix else " | FINAL=blue +"
    cv2.putText(
        output, f"{caption} | markers: {len(marker_ids)}{legend}",
        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (245, 245, 245), 1, cv2.LINE_AA)
    return output

def _read_video_frame(video_path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"無法重新開啟影片以顯示 frame：{video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise OSError(f"無法讀取 frame {frame_index}：{video_path}")
        return frame
    finally:
        capture.release()


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rotation_angle_deg(rotation: Any) -> float | None:
    try:
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rotation_to_euler_zyx_deg(rotation: Any) -> np.ndarray | None:
    """Return [roll_X, pitch_Y, yaw_Z] for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    try:
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(R)):
        return None
    sy = math.hypot(float(R[0, 0]), float(R[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(-float(R[2, 0]), sy)
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def _rotation_axis_angle_deg(rotation: Any) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return (unit axis, angle_deg, Rodrigues vector in degree units)."""
    try:
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
    except (TypeError, ValueError, cv2.error):
        return None
    rv = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle_rad = float(np.linalg.norm(rv))
    if angle_rad <= 1e-12:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = rv / angle_rad
    return axis, math.degrees(angle_rad), np.degrees(rv)


def _plane_normal_camera(rotation: Any) -> np.ndarray | None:
    """Pattern +Z plane normal expressed in camera coordinates."""
    try:
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    normal = R[:, 2].astype(np.float64, copy=True)
    norm = float(np.linalg.norm(normal))
    return None if norm <= 1e-12 else normal / norm


def _vector_angle_deg(first: Any, second: Any) -> float | None:
    try:
        a = np.asarray(first, dtype=np.float64).reshape(3)
        b = np.asarray(second, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _pose_pair_orientation_diagnostics(R_A: Any, R_B: Any) -> dict[str, Any]:
    try:
        RA = np.asarray(R_A, dtype=np.float64).reshape(3, 3)
        RB = np.asarray(R_B, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return {}
    R_rel = RA @ RB.T
    normal_a = _plane_normal_camera(RA)
    normal_b = _plane_normal_camera(RB)
    return {
        "euler_A_zyx_deg": _rotation_to_euler_zyx_deg(RA),
        "euler_B_zyx_deg": _rotation_to_euler_zyx_deg(RB),
        "euler_rel_zyx_deg": _rotation_to_euler_zyx_deg(R_rel),
        "axis_angle_rel": _rotation_axis_angle_deg(R_rel),
        "normal_A_camera": normal_a,
        "normal_B_camera": normal_b,
        "normal_angle_deg": (
            _vector_angle_deg(normal_a, normal_b)
            if normal_a is not None and normal_b is not None else None
        ),
    }


def _fixed_camera_pattern_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Derive diagnostics for the fixed-camera / translated-pattern experiment.

    R_A/t_A and R_B/t_B are pattern-to-camera poses.  With the camera fixed and
    the same pattern coordinate frame used at A and B, the pattern origin motion
    is directly ``t_B - t_A`` in camera coordinates.  Its norm is therefore the
    primary metric-scale baseline for a pure pattern translation experiment.

    ``||t_rel||`` is retained as a secondary relative-RT diagnostic because it
    also contains the effect of any estimated A/B rotation mismatch.
    """
    t_a_raw = result.get("t_A")
    t_b_raw = result.get("t_B")
    delta = None
    pattern_baseline = None
    if t_a_raw is not None and t_b_raw is not None:
        try:
            t_a = np.asarray(t_a_raw, dtype=np.float64).reshape(3)
            t_b = np.asarray(t_b_raw, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(t_a)) and np.all(np.isfinite(t_b)):
                delta = t_b - t_a
                pattern_baseline = float(np.linalg.norm(delta))
        except (TypeError, ValueError):
            pass

    relative_baseline = _float_or_none(result.get("baseline_mm"))
    if relative_baseline is None and result.get("t_rel") is not None:
        try:
            relative_baseline = float(
                np.linalg.norm(np.asarray(result["t_rel"], dtype=np.float64).reshape(3))
            )
        except (TypeError, ValueError):
            pass

    rotation_change = _rotation_angle_deg(result.get("R_rel"))
    baseline_gap = (
        abs(relative_baseline - pattern_baseline)
        if relative_baseline is not None and pattern_baseline is not None
        else None
    )
    return {
        "pattern_translation_vector_camera_mm": delta,
        "pattern_translation_baseline_mm": pattern_baseline,
        "relative_rt_baseline_mm": relative_baseline,
        "relative_vs_pattern_baseline_gap_mm": baseline_gap,
        "pattern_rotation_change_deg": rotation_change,
    }



def _append_ippe_endpoint_lines(lines: list[str], label: str, diag: Any) -> None:
    if not isinstance(diag, dict):
        lines.append(f"{label}: IPPE diagnostics unavailable")
        return
    lines.append(f"{label}: status={diag.get('status')}  marker_id={diag.get('marker_id')}")
    selected = diag.get("selected")
    if isinstance(selected, dict):
        lines.append(
            f"  temporal selected: label={selected.get('label')}  "
            f"source={selected.get('source')}  seed_branch={selected.get('seed_branch')}  "
            f"candidate_index={selected.get('temporal_candidate_index')}"
        )
        rms = _float_or_none(selected.get("reprojection_rms_px"))
        emission = _float_or_none(selected.get("emission_cost"))
        lines.append(
            "  selected temporal RMS/emission: "
            + (f"{rms:.9f} px / " if rms is not None else "— / ")
            + (f"{emission:.9f}" if emission is not None else "—")
        )
    branches = diag.get("branches")
    if not isinstance(branches, list) or not branches:
        reason = diag.get("reason")
        if reason:
            lines.append(f"  reason: {reason}")
        return
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        b = branch.get("branch")
        rms = _float_or_none(branch.get("raw_marker_reprojection_rms_px"))
        lines.append(f"  Branch {b}:")
        lines.append(f"    raw IPPE reprojection RMS: {rms:.9f} px" if rms is not None
                     else "    raw IPPE reprojection RMS: —")
        R = branch.get("R_reference")
        t = branch.get("t_reference")
        if R is not None:
            euler = _rotation_to_euler_zyx_deg(R)
            normal = _plane_normal_camera(R)
            if euler is not None:
                lines.append("    Euler [roll X, pitch Y, yaw Z] deg: "
                             + np.array2string(euler, precision=6, suppress_small=True))
            if normal is not None:
                lines.append("    plane normal Camera XYZ:            "
                             + np.array2string(normal, precision=6, suppress_small=True))
        if t is not None:
            lines.append("    t_reference Camera XYZ (mm):         "
                         + np.array2string(np.asarray(t, dtype=np.float64).reshape(3),
                                           precision=6, suppress_small=True))
        rd = _float_or_none(branch.get("rotation_distance_to_selected_deg"))
        td = _float_or_none(branch.get("translation_distance_to_selected_mm"))
        lines.append(f"    distance to selected pose: rotation={rd:.6f} deg"
                     if rd is not None else "    distance to selected pose: rotation=—")
        if td is not None:
            lines[-1] += f", translation={td:.6f} mm"


def _append_corner_refinement_lines(lines: list[str], label: str, diag: Any) -> None:
    if not isinstance(diag, dict):
        lines.append(f"{label}: corner refinement diagnostics unavailable")
        return
    mean_shift = _float_or_none(diag.get("overall_mean_shift_px"))
    max_shift = _float_or_none(diag.get("overall_max_shift_px"))
    mean_inward = _float_or_none(diag.get("overall_mean_inward_px"))
    lines.append(
        f"{label}: mean_shift={mean_shift:.6f}px, max_shift={max_shift:.6f}px, "
        f"mean_inward={mean_inward:+.6f}px, all_corners_inward={diag.get('all_corners_inward')}"
        if None not in (mean_shift, max_shift, mean_inward)
        else f"{label}: {diag}"
    )
    markers = diag.get("markers")
    if not isinstance(markers, dict):
        return
    for marker_id, marker in sorted(markers.items(), key=lambda item: str(item[0])):
        if not isinstance(marker, dict):
            continue
        area_ratio = _float_or_none(marker.get("area_ratio_refined_over_raw"))
        perimeter_ratio = _float_or_none(marker.get("perimeter_ratio_refined_over_raw"))
        lines.append(
            f"  ID{marker_id}: mean_shift={float(marker.get('mean_shift_px', 0.0)):.6f}px, "
            f"mean_inward={float(marker.get('mean_inward_px', 0.0)):+.6f}px, "
            f"all4_inward={marker.get('all_four_inward')}, "
            f"area_ratio={area_ratio:.9f}, perimeter_ratio={perimeter_ratio:.9f}"
            if area_ratio is not None and perimeter_ratio is not None
            else f"  ID{marker_id}: {marker}"
        )
        per_corner = marker.get("per_corner")
        if isinstance(per_corner, list):
            for item in per_corner:
                if not isinstance(item, dict):
                    continue
                raw = np.asarray(item.get("raw_xy"), dtype=np.float64).reshape(2)
                refined = np.asarray(item.get("refined_xy"), dtype=np.float64).reshape(2)
                delta = np.asarray(item.get("delta_xy"), dtype=np.float64).reshape(2)
                lines.append(
                    f"    C{item.get('corner')}: raw=({raw[0]:.6f},{raw[1]:.6f}) "
                    f"selected=({refined[0]:.6f},{refined[1]:.6f}) "
                    f"d=({delta[0]:+.6f},{delta[1]:+.6f}) "
                    f"mag={float(item.get('magnitude_px', 0.0)):.6f}px "
                    f"inward={float(item.get('inward_px', 0.0)):+.6f}px "
                    f"tangent={float(item.get('tangential_px', 0.0)):+.6f}px"
                )


def _append_raw_vs_subpix_ippe_lines(lines: list[str], label: str, diag: Any) -> None:
    if not isinstance(diag, dict):
        return
    comparisons = diag.get("comparisons")
    if not isinstance(comparisons, list):
        return
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        rot = _float_or_none(item.get("rotation_raw_to_refined_deg"))
        trans = _float_or_none(item.get("translation_raw_to_refined_mm"))
        raw_rms = _float_or_none(item.get("raw_reprojection_rms_px"))
        subpix_rms = _float_or_none(item.get("refined_reprojection_rms_px"))
        lines.append(
            f"{label} branch {item.get('branch')}: raw→selected pose shift "
            f"rotation={rot:.6f} deg, translation={trans:.6f} mm, "
            f"raw_RMS={raw_rms!r}px, selected_RMS={subpix_rms!r}px"
            if rot is not None and trans is not None
            else f"{label} branch {item.get('branch')}: {item}"
        )


def _append_ippe_pair_lines(lines: list[str], diag: Any) -> None:
    if not isinstance(diag, dict):
        return
    lines.extend(("", "=== A/B IPPE BRANCH COMBINATIONS (DIAGNOSTIC ONLY) ==="))
    lines.append(
        f"Temporal selection: A branch {diag.get('selected_branch_A')} + "
        f"B branch {diag.get('selected_branch_B')}"
    )
    combinations = diag.get("combinations")
    if isinstance(combinations, list):
        for combo in combinations:
            if not isinstance(combo, dict):
                continue
            rot = _float_or_none(combo.get("rotation_difference_deg"))
            norm = _float_or_none(combo.get("plane_normal_difference_deg"))
            pb = _float_or_none(combo.get("pattern_translation_baseline_mm"))
            rb = _float_or_none(combo.get("relative_rt_baseline_mm"))
            reproj = _float_or_none(combo.get("sum_raw_reprojection_rms_px"))
            lines.append(
                f"A{combo.get('branch_A')} + B{combo.get('branch_B')}: "
                f"rotation={rot:.6f} deg, normal={norm:.6f} deg, "
                f"||tB-tA||={pb:.6f} mm, ||t_rel||={rb:.6f} mm, "
                f"sum raw RMS={reproj:.9f} px"
                if None not in (rot, norm, pb, rb, reproj)
                else f"A{combo.get('branch_A')} + B{combo.get('branch_B')}: {combo}"
            )
    selected = diag.get("selected_combination")
    if isinstance(selected, dict):
        rot = _float_or_none(selected.get("rotation_difference_deg"))
        if rot is not None:
            lines.append(f"CURRENT selected branch-pair rotation difference: {rot:.6f} deg")
    best = diag.get("minimum_rotation_difference_combination")
    if isinstance(best, dict):
        rot = _float_or_none(best.get("rotation_difference_deg"))
        lines.append(
            f"Minimum-rotation branch pair (not applied): A{best.get('branch_A')} + "
            f"B{best.get('branch_B')}"
            + (f" -> {rot:.6f} deg" if rot is not None else "")
        )
    lines.append("NOTE: 上述 branch 組合只做診斷，程式沒有用 fixed-camera ground truth 改選 branch。")

def _json_safe(value: Any, *, omit_images: bool = True) -> Any:
    """Convert backend output to a compact JSON-safe representation."""

    if isinstance(value, np.ndarray):
        if omit_images and value.ndim >= 2 and value.size > 100_000:
            return {"omitted": "image array", "shape": list(value.shape), "dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, omit_images=omit_images) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, omit_images=omit_images) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten_for_csv(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten mappings while storing arrays/lists as JSON in one CSV cell."""

    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_for_csv(item, child))
    elif isinstance(value, (list, tuple)):
        flattened[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        flattened[prefix] = value
    return flattened


def _flatten_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_scalars(item, child))
    elif isinstance(value, np.ndarray):
        if value.size <= 16:
            rows.append((prefix, np.array2string(value, precision=6, suppress_small=True)))
    elif isinstance(value, (list, tuple)):
        if len(value) <= 16 and all(not isinstance(item, (dict, list, tuple)) for item in value):
            rows.append((prefix, value))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


class TwoVideoRTValidationUI(tk.Tk):
    """Tkinter front-end for the isolated marker-only validation backend."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} | {UI_BUILD}")
        self.geometry("1460x980")
        self.minsize(1050, 760)

        self.video_a_var = tk.StringVar()
        self.video_b_var = tk.StringVar()
        self.calibration_var = tk.StringVar()
        self.marker_size_var = tk.StringVar(value="8.25")
        self.known_baseline_var = tk.StringVar(value="100.0")
        self.dx_var = tk.StringVar()
        self.dy_var = tk.StringVar()
        self.dz_var = tk.StringVar()
        self.min_baseline_var = tk.StringVar(value="0")
        self.max_baseline_var = tk.StringVar(value="220")
        self.local_window_var = tk.BooleanVar(value=True)
        self.local_window_radius_var = tk.IntVar(value=2)
        self.klt_var = tk.BooleanVar(value=False)
        self.aruco_clahe_var = tk.BooleanVar(value=True)
        self.aruco_corner_mode_var = tk.StringVar(value="Contour refine")
        self.aruco_detector_preset_var = tk.StringVar(value="LCD Robust")
        self.aruco_outer_edge_refine_var = tk.BooleanVar(value=True)
        self.display_view_mode_var = tk.StringVar(value="原始彩色")
        self.display_raw_subpix_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="請載入兩段影片與單目相機標定 JSON。")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._run_inputs: dict[str, Any] | None = None
        self._display_frames: tuple[np.ndarray, np.ndarray] | None = None
        self._display_overlays: tuple[tuple[Any, Any, str], tuple[Any, Any, str]] | None = None
        self._photos: list[tk.PhotoImage | None] = [None, None]
        self._canvas_items: list[int | None] = [None, None]
        self._render_job: str | None = None

        # Per-canvas image navigation state.  zoom=1.0 means fit the whole frame.
        self._zoom_factors: list[float] = [1.0, 1.0]
        self._view_centers: list[tuple[float, float] | None] = [None, None]
        self._drag_last: list[tuple[int, int] | None] = [None, None]
        self._min_zoom = 1.0
        self._max_zoom = 12.0
        self._zoom_step = 1.20

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=4)
        self.rowconfigure(4, weight=3)

        sources = ttk.LabelFrame(self, text="輸入資料", padding=8)
        sources.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        sources.columnconfigure(1, weight=1)
        self._path_row(sources, 0, "Video A（起始位置／前段）", self.video_a_var, self._choose_video_a)
        self._path_row(sources, 1, "Video B（結束位置／後段）", self.video_b_var, self._choose_video_b)
        self._path_row(sources, 2, "單目相機標定 JSON", self.calibration_var, self._choose_calibration)

        options = ttk.LabelFrame(self, text="驗證參數", padding=8)
        options.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for column in range(15):
            options.columnconfigure(column, weight=0)

        ttk.Label(options, text="Marker 邊長 (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.marker_size_var, width=9).grid(row=0, column=1, padx=(4, 14))
        ttk.Label(options, text="已知 baseline (mm)").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.known_baseline_var, width=9).grid(row=0, column=3, padx=(4, 14))
        ttk.Label(options, text="baseline gate (mm)").grid(row=0, column=4, sticky="w")
        ttk.Entry(options, textvariable=self.min_baseline_var, width=7).grid(row=0, column=5, padx=(4, 2))
        ttk.Label(options, text="～").grid(row=0, column=6)
        ttk.Entry(options, textvariable=self.max_baseline_var, width=7).grid(row=0, column=7, padx=(2, 14))
        ttk.Checkbutton(options, text="Local window", variable=self.local_window_var).grid(row=0, column=8, padx=5)
        ttk.Label(options, text="半徑").grid(row=0, column=9)
        ttk.Spinbox(options, from_=0, to=10, textvariable=self.local_window_radius_var, width=4).grid(
            row=0, column=10, padx=(3, 10)
        )
        ttk.Checkbutton(options, text="KLT", variable=self.klt_var).grid(row=0, column=11, padx=5)
        ttk.Checkbutton(
            options, text="ArUco CLAHE", variable=self.aruco_clahe_var
        ).grid(row=0, column=12, padx=(10, 5))
        ttk.Checkbutton(
            options, text="顯示 Raw/Selected 比較",
            variable=self.display_raw_subpix_var,
            command=self._schedule_render,
        ).grid(row=0, column=13, columnspan=2, padx=(10, 5), sticky="w")

        ttk.Label(options, text="Corner mode").grid(row=2, column=0, sticky="w", pady=(9, 0))
        corner_box = ttk.Combobox(
            options, textvariable=self.aruco_corner_mode_var, state="readonly", width=18,
            values=tuple(CORNER_MODE_LABEL_TO_VALUE),
        )
        corner_box.grid(row=2, column=1, columnspan=2, sticky="w", padx=(4, 14), pady=(9, 0))
        ttk.Label(options, text="Detector preset").grid(row=2, column=3, sticky="e", pady=(9, 0))
        preset_box = ttk.Combobox(
            options, textvariable=self.aruco_detector_preset_var, state="readonly", width=18,
            values=tuple(DETECTOR_PRESET_LABEL_TO_VALUE),
        )
        preset_box.grid(row=2, column=4, columnspan=2, sticky="w", padx=(4, 14), pady=(9, 0))
        ttk.Label(options, text="預覽影像").grid(row=2, column=6, sticky="e", pady=(9, 0))
        display_box = ttk.Combobox(
            options, textvariable=self.display_view_mode_var, state="readonly", width=22,
            values=DISPLAY_VIEW_MODES,
        )
        display_box.grid(row=2, column=7, columnspan=3, sticky="w", padx=(4, 14), pady=(9, 0))
        display_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_display_mode())
        ttk.Label(
            options,
            text="LCD preset 會擴大 adaptive-threshold 視窗；binary 僅為代表性診斷圖，ArUco 內部會測多個 window。",
            foreground="#555555",
        ).grid(row=2, column=10, columnspan=5, sticky="w", pady=(9, 0))

        ttk.Checkbutton(
            options,
            text="Outer-edge refine（建議開啟）",
            variable=self.aruco_outer_edge_refine_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(7, 0))
        ttk.Label(
            options,
            text="ArUco 只做粗定位，再從原始高解析灰階沿四邊往外找黑→白實體邊界；最終角點會真正進入 PnP / RT。",
            foreground="#555555",
        ).grid(row=3, column=3, columnspan=12, sticky="w", pady=(7, 0))

        ttk.Label(options, text="Pattern 已知物理位移 A→B (mm，可留空)").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(9, 0)
        )
        for column, label, variable in (
            (3, "dx", self.dx_var),
            (5, "dy", self.dy_var),
            (7, "dz", self.dz_var),
        ):
            ttk.Label(options, text=label).grid(row=1, column=column, sticky="e", pady=(9, 0))
            ttk.Entry(options, textvariable=variable, width=9).grid(
                row=1, column=column + 1, padx=(3, 9), pady=(9, 0)
            )
        ttk.Label(
            options,
            text=(
                "固定相機驗證：主要比較 ||t_B-t_A|| 與 known baseline。"
                "dx/dy/dz 僅在已知 Camera XYZ 位移向量時填；只知道螢幕平移距離時可全留空。"
            ),
            foreground="#555555",
        ).grid(row=1, column=9, columnspan=5, sticky="w", pady=(9, 0))

        actions = ttk.Frame(self, padding=(8, 4))
        actions.grid(row=2, column=0, sticky="ew")
        self.run_button = ttk.Button(actions, text="開始 Marker-only 分析", command=self._start_analysis)
        self.run_button.pack(side="left")
        ttk.Label(actions, text=f"UI build: {UI_BUILD}", foreground="#8a2be2").pack(
            side="left", padx=(10, 4)
        )
        self.export_json_button = ttk.Button(
            actions, text="匯出 JSON", command=self._export_json, state="disabled"
        )
        self.export_json_button.pack(side="left", padx=(8, 3))
        self.export_csv_button = ttk.Button(
            actions, text="匯出 CSV", command=self._export_csv, state="disabled"
        )
        self.export_csv_button.pack(side="left", padx=3)
        ttk.Button(actions, text="清除紀錄", command=self._clear_log).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(
            actions, maximum=100.0, variable=self.progress_var, length=320, mode="determinate"
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(15, 8))
        ttk.Label(actions, textvariable=self.status_var, anchor="e").pack(side="right")

        image_frame = ttk.Frame(self)
        image_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(2, weight=1)
        ttk.Label(
            image_frame,
            text="圖片操作：滑鼠滾輪縮放｜左鍵拖曳平移｜雙擊左鍵恢復全圖",
            anchor="center",
            foreground="#555555",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        ttk.Label(image_frame, text="Video A 選定 frame", anchor="center").grid(row=1, column=0, sticky="ew")
        ttk.Label(image_frame, text="Video B 選定 frame", anchor="center").grid(row=1, column=1, sticky="ew")
        self.canvas_a = tk.Canvas(image_frame, background="#181818", highlightthickness=1)
        self.canvas_b = tk.Canvas(image_frame, background="#181818", highlightthickness=1)
        self.canvas_a.grid(row=2, column=0, sticky="nsew", padx=(0, 3))
        self.canvas_b.grid(row=2, column=1, sticky="nsew", padx=(3, 0))
        self.canvas_a.bind("<Configure>", self._schedule_render)
        self.canvas_b.bind("<Configure>", self._schedule_render)
        for slot, canvas in enumerate((self.canvas_a, self.canvas_b)):
            canvas.bind("<MouseWheel>", lambda event, s=slot: self._on_canvas_mousewheel(event, s))
            canvas.bind("<Button-4>", lambda event, s=slot: self._on_canvas_mousewheel(event, s))
            canvas.bind("<Button-5>", lambda event, s=slot: self._on_canvas_mousewheel(event, s))
            canvas.bind("<ButtonPress-1>", lambda event, s=slot: self._on_canvas_drag_start(event, s))
            canvas.bind("<B1-Motion>", lambda event, s=slot: self._on_canvas_drag(event, s))
            canvas.bind("<ButtonRelease-1>", lambda event, s=slot: self._on_canvas_drag_end(event, s))
            canvas.bind("<Double-Button-1>", lambda event, s=slot: self._reset_canvas_view(s))
        self._set_canvas_placeholder(self.canvas_a, "尚未分析")
        self._set_canvas_placeholder(self.canvas_b, "尚未分析")

        notebook = ttk.Notebook(self)
        notebook.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        metrics_page = ttk.Frame(notebook)
        log_page = ttk.Frame(notebook)
        notebook.add(metrics_page, text="RT 與誤差")
        notebook.add(log_page, text="執行紀錄")
        metrics_page.columnconfigure(0, weight=1)
        metrics_page.rowconfigure(0, weight=1)
        log_page.columnconfigure(0, weight=1)
        log_page.rowconfigure(0, weight=1)
        self.metrics_text = tk.Text(metrics_page, wrap="none", height=12, font=("Consolas", 10))
        self.log_text = tk.Text(log_page, wrap="word", height=12, font=("Consolas", 9))
        self._add_scrollbars(metrics_page, self.metrics_text)
        self._add_scrollbars(log_page, self.log_text)
        self.metrics_text.insert("1.0", "分析完成後會顯示 R、t、baseline、誤差與 marker diagnostics。\n")
        self.metrics_text.configure(state="disabled")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _add_scrollbars(parent: ttk.Frame, widget: tk.Text) -> None:
        vertical = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=widget.xview)
        widget.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        widget.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

    @staticmethod
    def _path_row(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label, width=28).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(parent, text="瀏覽…", command=command).grid(row=row, column=2, pady=2)

    def _choose_video_a(self) -> None:
        self._choose_file(self.video_a_var, VIDEO_FILE_TYPES, "選擇 Video A")

    def _choose_video_b(self) -> None:
        self._choose_file(self.video_b_var, VIDEO_FILE_TYPES, "選擇 Video B")

    def _choose_calibration(self) -> None:
        chosen = self._choose_file(self.calibration_var, JSON_FILE_TYPES, "選擇單目相機標定 JSON")
        if not chosen:
            return
        try:
            _matrix, _distortion, metadata = load_monocular_calibration(chosen)
            self.status_var.set(
                f"標定載入成功：fx={metadata['fx']:.2f}, fy={metadata['fy']:.2f}, {metadata['source']}"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("標定檔錯誤", str(exc), parent=self)

    @staticmethod
    def _choose_file(variable: tk.StringVar, filetypes: list[tuple[str, str]], title: str) -> str:
        initial = Path(variable.get()).parent if variable.get().strip() else Path.cwd()
        chosen = filedialog.askopenfilename(title=title, initialdir=str(initial), filetypes=filetypes)
        if chosen:
            variable.set(str(Path(chosen).resolve()))
        return chosen

    @staticmethod
    def _required_float(text: str, label: str, *, minimum: float | None = None) -> float:
        try:
            value = float(text.strip())
        except ValueError as exc:
            raise ValueError(f"{label} 必須是數字。") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} 必須是有限數值。")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} 必須大於或等於 {minimum}。")
        return value

    def _collect_inputs(self) -> dict[str, Any]:
        video_a = Path(self.video_a_var.get().strip()).expanduser()
        video_b = Path(self.video_b_var.get().strip()).expanduser()
        calibration = Path(self.calibration_var.get().strip()).expanduser()
        for label, path in (("Video A", video_a), ("Video B", video_b), ("標定 JSON", calibration)):
            if not str(path).strip() or not path.is_file():
                raise ValueError(f"{label} 檔案不存在：{path}")

        marker_size = self._required_float(self.marker_size_var.get(), "Marker 邊長", minimum=1e-9)
        known_text = self.known_baseline_var.get().strip()
        known_baseline = (
            self._required_float(known_text, "已知 baseline", minimum=0.0) if known_text else None
        )
        min_baseline = self._required_float(self.min_baseline_var.get(), "最小 baseline", minimum=0.0)
        max_baseline = self._required_float(self.max_baseline_var.get(), "最大 baseline", minimum=0.0)
        if max_baseline <= min_baseline:
            raise ValueError("最大 baseline 必須大於最小 baseline。")

        vector_text = (self.dx_var.get().strip(), self.dy_var.get().strip(), self.dz_var.get().strip())
        known_vector = None
        if any(vector_text):
            known_vector = tuple(
                self._required_float(component, label) if component else 0.0
                for component, label in zip(vector_text, ("dx", "dy", "dz"))
            )
            if np.linalg.norm(known_vector) <= 1e-12:
                raise ValueError("已知位移向量不能是零向量；若不比較方向，請將三軸全部留空。")
            vector_norm = float(np.linalg.norm(known_vector))
            if known_baseline is not None and not math.isclose(
                known_baseline, vector_norm, rel_tol=0.005, abs_tol=0.1
            ):
                raise ValueError(
                    "已知 baseline 與位移向量長度不一致："
                    f"{known_baseline:.4f} mm vs {vector_norm:.4f} mm。"
                )

        radius = int(self.local_window_radius_var.get())
        if radius < 0:
            raise ValueError("Local window 半徑不能小於 0。")

        camera_matrix, distortion, calibration_metadata = load_monocular_calibration(calibration)
        return {
            "video_a_path": str(video_a.resolve()),
            "video_b_path": str(video_b.resolve()),
            "calibration_path": str(calibration.resolve()),
            "camera_matrix": camera_matrix,
            "distortion": distortion,
            "calibration_metadata": calibration_metadata,
            "marker_size_mm": marker_size,
            "known_translation_mm": known_baseline,
            "known_translation_vector": known_vector,
            "min_baseline_mm": min_baseline,
            "max_baseline_mm": max_baseline,
            "local_window": bool(self.local_window_var.get()),
            "local_window_radius": radius,
            "klt_enabled": bool(self.klt_var.get()),
            "aruco_use_clahe": bool(self.aruco_clahe_var.get()),
            "aruco_corner_mode": CORNER_MODE_LABEL_TO_VALUE[self.aruco_corner_mode_var.get()],
            "aruco_detector_preset": DETECTOR_PRESET_LABEL_TO_VALUE[self.aruco_detector_preset_var.get()],
            "aruco_outer_edge_refine": bool(self.aruco_outer_edge_refine_var.get()),
        }

    def _start_analysis(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            inputs = self._collect_inputs()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("輸入資料錯誤", str(exc), parent=self)
            return

        self._result = None
        self._run_inputs = inputs
        self._display_frames = None
        self._display_overlays = None
        self._photos = [None, None]
        self._canvas_items = [None, None]
        self._zoom_factors = [1.0, 1.0]
        self._view_centers = [None, None]
        self._drag_last = [None, None]
        self._set_canvas_placeholder(self.canvas_a, "分析中…")
        self._set_canvas_placeholder(self.canvas_b, "分析中…")
        self.progress_var.set(0.0)
        self.status_var.set("準備分析…")
        self.run_button.configure(state="disabled")
        self.export_json_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self._replace_metrics("分析進行中…\n")
        self._append_log("=" * 72)
        self._append_log(f"Video A: {inputs['video_a_path']}")
        self._append_log(f"Video B: {inputs['video_b_path']}")
        self._append_log(
            f"Marker-only | local_window={inputs['local_window']} "
            f"radius={inputs['local_window_radius']} | KLT={inputs['klt_enabled']} | "
            f"ArUco_CLAHE={inputs['aruco_use_clahe']} | "
            f"corner={inputs['aruco_corner_mode']} | preset={inputs['aruco_detector_preset']} | "
            f"outer_edge={'ON' if inputs['aruco_outer_edge_refine'] else 'OFF'} | "
            f"SIFT=disabled"
        )

        self._worker = threading.Thread(
            target=self._analysis_worker,
            args=(inputs,),
            name="marker-only-rt-validation",
            daemon=True,
        )
        self._worker.start()
        self.after(80, self._poll_events)

    def _analysis_worker(self, inputs: dict[str, Any]) -> None:
        started = time.perf_counter()

        def progress_callback(percent: float, message: str = "") -> None:
            self._events.put(("progress", (float(percent), str(message))))

        def log_callback(message: Any) -> None:
            self._events.put(("log", str(message)))

        try:
            from Algorithm import (
                video_pose_analysis_temporal_unified_pattern_guided_local_window_validation_v5_8
                as validation_backend,
            )

            result = validation_backend.analyze_two_video_segments(
                inputs["video_a_path"],
                inputs["video_b_path"],
                inputs["camera_matrix"],
                inputs["distortion"],
                marker_size_mm=inputs["marker_size_mm"],
                known_translation_mm=inputs["known_translation_mm"],
                known_translation_vector=inputs["known_translation_vector"],
                local_window=inputs["local_window"],
                local_window_radius=inputs["local_window_radius"],
                klt_enabled=inputs["klt_enabled"],
                aruco_use_clahe=inputs["aruco_use_clahe"],
                aruco_corner_mode=inputs["aruco_corner_mode"],
                aruco_detector_preset=inputs["aruco_detector_preset"],
                aruco_outer_edge_refine=inputs["aruco_outer_edge_refine"],
                progress_callback=progress_callback,
                log_callback=log_callback,
                min_baseline_mm=inputs["min_baseline_mm"],
                max_baseline_mm=inputs["max_baseline_mm"],
            )
            if not isinstance(result, dict):
                raise RuntimeError("Validation backend 必須回傳 dict。")
            result.setdefault("runtime_s", time.perf_counter() - started)
            self._events.put(("done", result))
        except Exception as exc:  # Worker must deliver all failures to Tk's main thread.
            self._events.put(("error", (str(exc), traceback.format_exc())))

    def _poll_events(self) -> None:
        processed_terminal_event = False
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                percent, message = payload
                # The validation backend contract defines percent on a 0..100 scale.
                percent = float(np.clip(percent, 0.0, 100.0))
                self.progress_var.set(percent)
                if message:
                    self.status_var.set(message)
            elif kind == "log":
                self._append_log(payload)
            elif kind == "done":
                processed_terminal_event = True
                self._handle_result(payload)
            elif kind == "error":
                processed_terminal_event = True
                message, trace = payload
                self._handle_error(message, trace)

        if not processed_terminal_event and self._worker is not None and self._worker.is_alive():
            self.after(80, self._poll_events)

    @staticmethod
    def _aruco_detection_input_frame(frame: np.ndarray, use_clahe: bool) -> np.ndarray:
        """Reproduce the grayscale image actually fed to ArUco detection."""
        image = np.asarray(frame)
        if image.ndim == 2:
            gray = image.astype(np.uint8, copy=False)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    @staticmethod
    def _aruco_representative_binary_frame(
            frame: np.ndarray, use_clahe: bool, parameters: dict[str, Any]) -> np.ndarray:
        gray = RTValidationApp._aruco_detection_input_frame(frame, use_clahe)
        lo = int(parameters.get("adaptiveThreshWinSizeMin", 3))
        hi = int(parameters.get("adaptiveThreshWinSizeMax", 23))
        window = max(3, int(round((lo + hi) * 0.5)))
        if window % 2 == 0:
            window += 1
        constant = float(parameters.get("adaptiveThreshConstant", 7.0))
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, window, constant)

    def _refresh_display_mode(self) -> None:
        """Switch between original selected frames and the ArUco detector input."""
        if not isinstance(self._result, dict):
            return
        try:
            result = self._result
            frame_a = result.get("selected_frame_a_bgr")
            frame_b = result.get("selected_frame_b_bgr")
            index_a = int(result.get("selected_frame_a_index", -1))
            index_b = int(result.get("selected_frame_b_index", -1))
            if frame_a is None:
                if self._run_inputs is None:
                    raise RuntimeError("Missing Video A input path")
                frame_a = _read_video_frame(self._run_inputs["video_a_path"], index_a)
            if frame_b is None:
                if self._run_inputs is None:
                    raise RuntimeError("Missing Video B input path")
                frame_b = _read_video_frame(self._run_inputs["video_b_path"], index_b)

            use_clahe = bool(result.get(
                "aruco_use_clahe",
                self._run_inputs.get("aruco_use_clahe", True) if self._run_inputs else True,
            ))
            view_mode = self.display_view_mode_var.get()
            params = result.get("aruco_detector_parameters") or {}
            if view_mode == "偵測灰階":
                frame_a_show = self._aruco_detection_input_frame(frame_a, use_clahe)
                frame_b_show = self._aruco_detection_input_frame(frame_b, use_clahe)
                mode = "CLAHE gray" if use_clahe else "raw gray"
                caption_a = f"Video A | frame {index_a} | detector gray: {mode}"
                caption_b = f"Video B | frame {index_b} | detector gray: {mode}"
            elif view_mode == "Adaptive binary（診斷）":
                frame_a_show = self._aruco_representative_binary_frame(frame_a, use_clahe, params)
                frame_b_show = self._aruco_representative_binary_frame(frame_b, use_clahe, params)
                caption_a = f"Video A | frame {index_a} | representative adaptive binary"
                caption_b = f"Video B | frame {index_b} | representative adaptive binary"
            else:
                frame_a_show = frame_a
                frame_b_show = frame_b
                caption_a = f"Video A | frame {index_a} | original"
                caption_b = f"Video B | frame {index_b} | original"

            # Keep the source image clean.  Marker corner diagnostics are drawn
            # only after the current crop has been resized for the canvas.
            self._display_frames = (_ensure_bgr(frame_a_show), _ensure_bgr(frame_b_show))
            self._display_overlays = (
                (result.get("corners_a"), result.get("raw_corners_a"), caption_a),
                (result.get("corners_b"), result.get("raw_corners_b"), caption_b),
            )
            self._zoom_factors = [1.0, 1.0]
            self._view_centers = [None, None]
            self._drag_last = [None, None]
            self._schedule_render()
        except Exception as exc:
            self._append_log(f"切換 ArUco 偵測影像顯示失敗：{exc}")

    def _handle_result(self, result: dict[str, Any]) -> None:
        self._result = result
        self.progress_var.set(100.0)
        self.run_button.configure(state="normal")
        self.export_json_button.configure(state="normal")
        self.export_csv_button.configure(state="normal")

        sift_calls = int(result.get("sift_calls", 0) or 0)
        if sift_calls != 0:
            self.status_var.set(f"分析完成，但警告：SIFT calls = {sift_calls}")
            self._append_log(f"警告：marker-only validation 回報 SIFT calls = {sift_calls}")
        else:
            self.status_var.set("分析完成（SIFT calls = 0）")
        self._append_log("分析完成。")

        fixed = _fixed_camera_pattern_metrics(result)
        pattern_baseline = fixed.get("pattern_translation_baseline_mm")
        relative_baseline = fixed.get("relative_rt_baseline_mm")
        rotation_change = fixed.get("pattern_rotation_change_deg")
        pattern_vector = fixed.get("pattern_translation_vector_camera_mm")
        diag_line = (
            f"[FIXED-CAMERA DIAG] ||tB-tA||={pattern_baseline!r} mm | "
            f"||t_rel||={relative_baseline!r} mm | rotation={rotation_change!r} deg"
        )
        self._append_log(diag_line)
        if pattern_vector is not None:
            self._append_log(
                "[FIXED-CAMERA DIAG] tB-tA camera XYZ = "
                + np.array2string(np.asarray(pattern_vector), precision=6, suppress_small=True)
                + " mm"
            )
        print(f"[UI BUILD] {UI_BUILD}")
        print(diag_line)
        if pattern_vector is not None:
            print(
                "[FIXED-CAMERA DIAG] tB-tA camera XYZ = "
                + np.array2string(np.asarray(pattern_vector), precision=6, suppress_small=True)
                + " mm"
            )

        R_A = result.get("R_A")
        R_B = result.get("R_B")
        if R_A is not None and R_B is not None:
            orient = _pose_pair_orientation_diagnostics(R_A, R_B)
            e_a = orient.get("euler_A_zyx_deg")
            e_b = orient.get("euler_B_zyx_deg")
            e_rel = orient.get("euler_rel_zyx_deg")
            axis_angle = orient.get("axis_angle_rel")
            normal_angle = orient.get("normal_angle_deg")
            if e_a is not None:
                self._append_log("[POSE-DIAG] A Euler roll/pitch/yaw = " + np.array2string(e_a, precision=6))
                print("[POSE-DIAG] A Euler roll/pitch/yaw =", np.array2string(e_a, precision=6))
            if e_b is not None:
                self._append_log("[POSE-DIAG] B Euler roll/pitch/yaw = " + np.array2string(e_b, precision=6))
                print("[POSE-DIAG] B Euler roll/pitch/yaw =", np.array2string(e_b, precision=6))
            if e_rel is not None:
                self._append_log("[POSE-DIAG] Relative Euler roll/pitch/yaw = " + np.array2string(e_rel, precision=6))
                print("[POSE-DIAG] Relative Euler roll/pitch/yaw =", np.array2string(e_rel, precision=6))
            if axis_angle is not None:
                axis, angle_deg, rvec_deg = axis_angle
                msg = ("[POSE-DIAG] Relative axis-angle: angle="
                       f"{angle_deg:.6f} deg axis="
                       + np.array2string(axis, precision=6)
                       + " rvec_deg=" + np.array2string(rvec_deg, precision=6))
                self._append_log(msg)
                print(msg)
            if normal_angle is not None:
                msg = f"[POSE-DIAG] Pattern plane-normal A/B angle = {normal_angle:.6f} deg"
                self._append_log(msg)
                print(msg)

        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict):
            pre = diagnostics.get("pre_refinement_pose")
            delta = diagnostics.get("refinement_delta")
            if isinstance(pre, dict):
                msg = ("[POSE-DIAG PRE-REFINE] pattern="
                       f"{_float_or_none(pre.get('pattern_translation_baseline_mm'))!r} mm | "
                       f"relative={_float_or_none(pre.get('relative_rt_baseline_mm'))!r} mm | "
                       f"rotation={_float_or_none(pre.get('rotation_change_deg'))!r} deg")
                self._append_log(msg)
                print(msg)
            if isinstance(delta, dict):
                msg = ("[POSE-DIAG REFINE-SHIFT] "
                       f"A_rot={_float_or_none(delta.get('A_rotation_shift_deg'))!r} deg | "
                       f"B_rot={_float_or_none(delta.get('B_rotation_shift_deg'))!r} deg | "
                       f"A_t={_float_or_none(delta.get('A_translation_shift_mm'))!r} mm | "
                       f"B_t={_float_or_none(delta.get('B_translation_shift_mm'))!r} mm")
                self._append_log(msg)
                print(msg)

        self._refresh_display_mode()

        self._replace_metrics(self._format_metrics(result))

    def _handle_error(self, message: str, trace: str) -> None:
        self.progress_var.set(0.0)
        self.status_var.set("分析失敗")
        self.run_button.configure(state="normal")
        self.export_json_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self._replace_metrics(f"分析失敗\n\n{message}\n")
        self._append_log(trace.rstrip())
        messagebox.showerror("RT 驗證失敗", message, parent=self)

    def _format_metrics(self, result: dict[str, Any]) -> str:
        lines: list[str] = []
        index_a = result.get("selected_frame_a_index", "—")
        index_b = result.get("selected_frame_b_index", "—")
        lines.append(f"*** UI BUILD: {UI_BUILD} ***")
        lines.append("Fixed-camera / translated-pattern pose validation")
        lines.append(f"Selected frame: Video A = {index_a}, Video B = {index_b}")
        lines.append(
            f"Feature mode: {result.get('feature_mode', 'marker_only')} | "
            f"SIFT calls: {result.get('sift_calls', 0)} | "
            f"ArUco CLAHE: {'ON' if result.get('aruco_use_clahe', True) else 'OFF'}"
        )
        lines.append(
            f"Corner mode: {result.get('aruco_corner_mode', '—')} | "
            f"Detector preset: {result.get('aruco_detector_preset', '—')} | "
            f"Outer-edge: {'ON' if result.get('aruco_outer_edge_refine', False) else 'OFF'}"
        )
        if result.get("validation_build"):
            lines.append(f"Backend build: {result.get('validation_build')}")
        lines.append(
            f"Pose measurement: A={result.get('selected_measurement_mode_A', '—')} "
            f"IDs={result.get('selected_inlier_marker_ids_A', [])} | "
            f"B={result.get('selected_measurement_mode_B', '—')} "
            f"IDs={result.get('selected_inlier_marker_ids_B', [])}"
        )
        diagnostics_head = result.get("diagnostics")
        if isinstance(diagnostics_head, dict):
            marker_map_head = diagnostics_head.get("marker_map")
            if isinstance(marker_map_head, dict):
                map_diag = marker_map_head.get("diagnostics")
                validation_diag = map_diag.get("_validation") if isinstance(map_diag, dict) else None
                strategy = validation_diag.get("strategy") if isinstance(validation_diag, dict) else "—"
                lines.append(
                    f"Marker map: IDs={marker_map_head.get('marker_ids', [])} | strategy={strategy}"
                )

        rotation = result.get("R_rel")
        translation = result.get("t_rel")
        t_a = result.get("t_A")
        t_b = result.get("t_B")
        fixed = _fixed_camera_pattern_metrics(result)
        pattern_vector = fixed["pattern_translation_vector_camera_mm"]
        pattern_baseline = fixed["pattern_translation_baseline_mm"]
        relative_baseline = fixed["relative_rt_baseline_mm"]
        baseline_gap = fixed["relative_vs_pattern_baseline_gap_mm"]
        rotation_change = fixed["pattern_rotation_change_deg"]
        known = self._known_baseline_from_inputs()

        pattern_abs_error = (
            abs(pattern_baseline - known)
            if pattern_baseline is not None and known is not None
            else None
        )
        pattern_pct_error = (
            pattern_abs_error / known * 100.0
            if pattern_abs_error is not None and known is not None and known > 0
            else None
        )
        relative_abs_error = (
            abs(relative_baseline - known)
            if relative_baseline is not None and known is not None
            else None
        )
        relative_pct_error = (
            relative_abs_error / known * 100.0
            if relative_abs_error is not None and known is not None and known > 0
            else None
        )

        lines.extend((
            "",
            "=== PRIMARY: fixed-camera pattern translation ===",
            (f"Pattern translation baseline ||t_B - t_A||: {pattern_baseline:.6f} mm"
             if pattern_baseline is not None else
             "Pattern translation baseline ||t_B - t_A||: —"),
            (f"Known physical translation:                   {known:.6f} mm"
             if known is not None else
             "Known physical translation:                   —"),
            (f"Pattern baseline abs error:                  {pattern_abs_error:.6f} mm"
             if pattern_abs_error is not None else
             "Pattern baseline abs error:                  —"),
            (f"Pattern baseline error:                      {pattern_pct_error:.4f} %"
             if pattern_pct_error is not None else
             "Pattern baseline error:                      —"),
        ))
        if pattern_vector is not None:
            lines.append(
                "Pattern A→B vector in Camera XYZ (mm):      "
                + np.array2string(np.asarray(pattern_vector), precision=6, suppress_small=True)
            )

        lines.extend((
            "",
            "=== SECONDARY: relative RT diagnostic ===",
            (f"Relative RT baseline ||t_rel||:               {relative_baseline:.6f} mm"
             if relative_baseline is not None else
             "Relative RT baseline ||t_rel||:               —"),
            (f"Relative RT error vs known:                   {relative_pct_error:.4f} %"
             if relative_pct_error is not None else
             "Relative RT error vs known:                   —"),
            (f"Relative-vs-pattern baseline gap:             {baseline_gap:.6f} mm"
             if baseline_gap is not None else
             "Relative-vs-pattern baseline gap:             —"),
            (f"Estimated pattern rotation change:            {rotation_change:.6f} deg (expected 0)"
             if rotation_change is not None else
             "Estimated pattern rotation change:            —"),
        ))

        if (
            baseline_gap is not None
            and pattern_baseline is not None
            and baseline_gap > max(2.0, 0.05 * max(pattern_baseline, 1.0))
        ):
            if rotation_change is not None and rotation_change > 1.0:
                lines.append(
                    "DIAGNOSIS: relative RT 與 direct pattern translation 差異明顯；"
                    "A/B rotation estimate 可能正在污染 ||t_rel||。"
                )
            else:
                lines.append(
                    "DIAGNOSIS: relative RT 與 direct pattern translation 差異明顯，"
                    "但 rotation change 不大；請檢查 pose convention / refinement。"
                )

        if t_a is not None:
            lines.extend(("", "t_A (Pattern origin in Camera XYZ, mm):",
                          np.array2string(np.asarray(t_a, dtype=np.float64).reshape(-1),
                                          precision=9, suppress_small=True)))
        if t_b is not None:
            lines.extend(("", "t_B (Pattern origin in Camera XYZ, mm):",
                          np.array2string(np.asarray(t_b, dtype=np.float64).reshape(-1),
                                          precision=9, suppress_small=True)))

        R_A = result.get("R_A")
        R_B = result.get("R_B")
        if R_A is not None and R_B is not None:
            orient = _pose_pair_orientation_diagnostics(R_A, R_B)
            lines.extend(("", "=== FINAL ORIENTATION DIAGNOSTICS ==="))
            e_a = orient.get("euler_A_zyx_deg")
            e_b = orient.get("euler_B_zyx_deg")
            e_rel = orient.get("euler_rel_zyx_deg")
            if e_a is not None:
                lines.append("A Euler [roll X, pitch Y, yaw Z] deg:       "
                             + np.array2string(e_a, precision=6, suppress_small=True))
            if e_b is not None:
                lines.append("B Euler [roll X, pitch Y, yaw Z] deg:       "
                             + np.array2string(e_b, precision=6, suppress_small=True))
            if e_rel is not None:
                lines.append("Relative Euler [roll X, pitch Y, yaw Z] deg: "
                             + np.array2string(e_rel, precision=6, suppress_small=True))
            axis_angle = orient.get("axis_angle_rel")
            if axis_angle is not None:
                axis, angle_deg, rvec_deg = axis_angle
                lines.append(f"Relative rotation angle:                    {angle_deg:.6f} deg")
                lines.append("Relative rotation axis:                     "
                             + np.array2string(axis, precision=6, suppress_small=True))
                lines.append("Relative Rodrigues vector (deg units):       "
                             + np.array2string(rvec_deg, precision=6, suppress_small=True))
            normal_a = orient.get("normal_A_camera")
            normal_b = orient.get("normal_B_camera")
            if normal_a is not None:
                lines.append("Pattern plane normal A in Camera XYZ:        "
                             + np.array2string(normal_a, precision=6, suppress_small=True))
            if normal_b is not None:
                lines.append("Pattern plane normal B in Camera XYZ:        "
                             + np.array2string(normal_b, precision=6, suppress_small=True))
            normal_angle = orient.get("normal_angle_deg")
            if normal_angle is not None:
                lines.append(f"Plane-normal A/B angle:                      {normal_angle:.6f} deg (expected 0)")

        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict):
            pre = diagnostics.get("pre_refinement_pose")
            refine_delta = diagnostics.get("refinement_delta")
            refinement = diagnostics.get("refinement")
            ippe = diagnostics.get("ippe_branch_diagnostics")
            corner_refine = diagnostics.get("corner_refinement_diagnostics")
            if isinstance(corner_refine, dict):
                lines.extend(("", "=== RAW detectMarkers -> FINAL CORNER DIAGNOSTICS ==="))
                lines.append(f"corner mode: {corner_refine.get('corner_mode')} | subpix window: {corner_refine.get('subpix_window')}")
                _append_corner_refinement_lines(lines, "Endpoint A", corner_refine.get("A"))
                _append_corner_refinement_lines(lines, "Endpoint B", corner_refine.get("B"))
            if isinstance(ippe, dict):
                if isinstance(ippe.get("raw_vs_cornerSubPix_A"), dict) or isinstance(ippe.get("raw_vs_cornerSubPix_B"), dict):
                    lines.extend(("", "=== RAW CORNER vs SELECTED-CORNER IPPE POSE SHIFT ==="))
                    _append_raw_vs_subpix_ippe_lines(lines, "Endpoint A", ippe.get("raw_vs_cornerSubPix_A"))
                    _append_raw_vs_subpix_ippe_lines(lines, "Endpoint B", ippe.get("raw_vs_cornerSubPix_B"))
                lines.extend(("", "=== RAW IPPE BRANCH DIAGNOSTICS (SELECTED ENDPOINTS) ==="))
                _append_ippe_endpoint_lines(lines, "Endpoint A", ippe.get("A"))
                _append_ippe_endpoint_lines(lines, "Endpoint B", ippe.get("B"))
                _append_ippe_pair_lines(lines, ippe.get("pair_combinations"))
            if isinstance(pre, dict):
                lines.extend(("", "=== BEFORE ENDPOINT REFINEMENT ==="))
                pre_pb = _float_or_none(pre.get("pattern_translation_baseline_mm"))
                pre_rb = _float_or_none(pre.get("relative_rt_baseline_mm"))
                pre_rc = _float_or_none(pre.get("rotation_change_deg"))
                lines.append(f"Pattern baseline before refinement:          {pre_pb:.6f} mm" if pre_pb is not None else "Pattern baseline before refinement:          —")
                lines.append(f"Relative RT baseline before refinement:      {pre_rb:.6f} mm" if pre_rb is not None else "Relative RT baseline before refinement:      —")
                lines.append(f"Rotation change before refinement:           {pre_rc:.6f} deg" if pre_rc is not None else "Rotation change before refinement:           —")
                pre_RA = pre.get("R_A")
                pre_RB = pre.get("R_B")
                if pre_RA is not None and pre_RB is not None:
                    pre_orient = _pose_pair_orientation_diagnostics(pre_RA, pre_RB)
                    pre_e_rel = pre_orient.get("euler_rel_zyx_deg")
                    if pre_e_rel is not None:
                        lines.append("Relative Euler before refinement [X,Y,Z]:    "
                                     + np.array2string(pre_e_rel, precision=6, suppress_small=True))
                    pre_norm_ang = pre_orient.get("normal_angle_deg")
                    if pre_norm_ang is not None:
                        lines.append(f"Plane-normal angle before refinement:         {pre_norm_ang:.6f} deg")
                pre_reproj = pre.get("marker_reprojection")
                if isinstance(pre_reproj, dict):
                    lines.append("Marker reprojection before refinement:")
                    for key, value in _flatten_scalars(pre_reproj):
                        lines.append(f"  {key}: {value}")

            lines.extend(("", "=== ENDPOINT REFINEMENT IMPACT ==="))
            if isinstance(refinement, dict):
                lines.append(f"Refinement applied:                          {refinement.get('applied')}")
                lines.append(f"Refinement role:                             {refinement.get('role')}")
                lines.append(f"Refinement marker_ok:                        {refinement.get('marker_ok')}")
            if isinstance(refine_delta, dict):
                for key in ("A_rotation_shift_deg", "B_rotation_shift_deg",
                            "A_translation_shift_mm", "B_translation_shift_mm"):
                    value = _float_or_none(refine_delta.get(key))
                    lines.append(f"{key}: {value:.9f}" if value is not None else f"{key}: —")
            final_reproj = diagnostics.get("final_marker_reprojection")
            if isinstance(final_reproj, dict):
                lines.append("Final marker reprojection after refinement:")
                for key, value in _flatten_scalars(final_reproj):
                    lines.append(f"  {key}: {value}")

        if rotation is not None:
            matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            lines.extend(("", "R_rel (B→A):",
                          np.array2string(matrix, precision=9, suppress_small=True)))
        if translation is not None:
            vector = np.asarray(translation, dtype=np.float64).reshape(-1)
            lines.extend(("", "t_rel (B→A, mm):",
                          np.array2string(vector, precision=9, suppress_small=True)))

        runtime = _float_or_none(result.get("runtime_s"))
        lines.append("")
        lines.append(f"Runtime: {runtime:.3f} s" if runtime is not None else "Runtime: —")

        ground_truth = result.get("ground_truth")
        if isinstance(ground_truth, dict) and ground_truth:
            lines.extend((
                "",
                "Backend relative-RT ground-truth diagnostics (secondary; uses t_rel):"
            ))
            for key, value in _flatten_scalars(ground_truth):
                lines.append(f"  {key}: {value}")

        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            lines.extend(("", "Marker / temporal diagnostics:"))
            for key, value in _flatten_scalars(diagnostics):
                lines.append(f"  {key}: {value}")

        return "\n".join(lines) + "\n"

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self._render_job is None:
            self._render_job = self.after_idle(self._render_selected_frames)

    def _view_geometry(
        self, slot: int, *, zoom: float | None = None
    ) -> dict[str, float]:
        """Return the current source-image viewport geometry for one canvas."""
        if self._display_frames is None:
            raise RuntimeError("No display frames")
        canvas = (self.canvas_a, self.canvas_b)[slot]
        frame = self._display_frames[slot]
        width = max(2, canvas.winfo_width())
        height = max(2, canvas.winfo_height())
        frame_h, frame_w = frame.shape[:2]

        fit_scale = min(width / frame_w, height / frame_h)
        zoom_value = float(self._zoom_factors[slot] if zoom is None else zoom)
        zoom_value = min(self._max_zoom, max(self._min_zoom, zoom_value))
        scale = fit_scale * zoom_value

        visible_w = min(float(frame_w), width / scale)
        visible_h = min(float(frame_h), height / scale)
        half_w = visible_w * 0.5
        half_h = visible_h * 0.5

        center = self._view_centers[slot]
        if center is None:
            center_x = frame_w * 0.5
            center_y = frame_h * 0.5
        else:
            center_x, center_y = center

        if visible_w >= frame_w - 1e-9:
            center_x = frame_w * 0.5
        else:
            center_x = min(frame_w - half_w, max(half_w, center_x))
        if visible_h >= frame_h - 1e-9:
            center_y = frame_h * 0.5
        else:
            center_y = min(frame_h - half_h, max(half_h, center_y))

        return {
            "canvas_w": float(width),
            "canvas_h": float(height),
            "frame_w": float(frame_w),
            "frame_h": float(frame_h),
            "fit_scale": float(fit_scale),
            "zoom": zoom_value,
            "scale": float(scale),
            "visible_w": float(visible_w),
            "visible_h": float(visible_h),
            "center_x": float(center_x),
            "center_y": float(center_y),
        }

    def _clamp_view_center(self, slot: int) -> None:
        geometry = self._view_geometry(slot)
        self._view_centers[slot] = (geometry["center_x"], geometry["center_y"])

    def _on_canvas_mousewheel(self, event: tk.Event, slot: int) -> str:
        if self._display_frames is None:
            return "break"

        if getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "num", None) == 5:
            direction = -1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return "break"
            direction = 1 if delta > 0 else -1

        old_geometry = self._view_geometry(slot)
        old_zoom = old_geometry["zoom"]
        new_zoom = old_zoom * (self._zoom_step if direction > 0 else 1.0 / self._zoom_step)
        new_zoom = min(self._max_zoom, max(self._min_zoom, new_zoom))
        if abs(new_zoom - old_zoom) < 1e-12:
            return "break"

        # Keep the source point under the mouse at approximately the same screen
        # position while zooming.  This makes it easy to inspect a marker corner.
        mouse_x = float(getattr(event, "x", old_geometry["canvas_w"] * 0.5))
        mouse_y = float(getattr(event, "y", old_geometry["canvas_h"] * 0.5))
        source_x = old_geometry["center_x"] + (mouse_x - old_geometry["canvas_w"] * 0.5) / old_geometry["scale"]
        source_y = old_geometry["center_y"] + (mouse_y - old_geometry["canvas_h"] * 0.5) / old_geometry["scale"]

        self._zoom_factors[slot] = new_zoom
        new_geometry = self._view_geometry(slot, zoom=new_zoom)
        desired_center_x = source_x - (mouse_x - new_geometry["canvas_w"] * 0.5) / new_geometry["scale"]
        desired_center_y = source_y - (mouse_y - new_geometry["canvas_h"] * 0.5) / new_geometry["scale"]
        self._view_centers[slot] = (desired_center_x, desired_center_y)
        self._clamp_view_center(slot)
        self._schedule_render()
        return "break"

    def _on_canvas_drag_start(self, event: tk.Event, slot: int) -> str:
        if self._display_frames is not None:
            self._drag_last[slot] = (int(event.x), int(event.y))
            (self.canvas_a, self.canvas_b)[slot].configure(cursor="fleur")
        return "break"

    def _on_canvas_drag(self, event: tk.Event, slot: int) -> str:
        if self._display_frames is None or self._drag_last[slot] is None:
            return "break"
        last_x, last_y = self._drag_last[slot]
        dx = int(event.x) - last_x
        dy = int(event.y) - last_y
        self._drag_last[slot] = (int(event.x), int(event.y))

        geometry = self._view_geometry(slot)
        center_x = geometry["center_x"] - dx / geometry["scale"]
        center_y = geometry["center_y"] - dy / geometry["scale"]
        self._view_centers[slot] = (center_x, center_y)
        self._clamp_view_center(slot)
        self._schedule_render()
        return "break"

    def _on_canvas_drag_end(self, _event: tk.Event, slot: int) -> str:
        self._drag_last[slot] = None
        (self.canvas_a, self.canvas_b)[slot].configure(cursor="")
        return "break"

    def _reset_canvas_view(self, slot: int, _event: tk.Event | None = None) -> str:
        self._zoom_factors[slot] = 1.0
        self._view_centers[slot] = None
        self._drag_last[slot] = None
        self._schedule_render()
        return "break"

    def _render_selected_frames(self) -> None:
        self._render_job = None
        if self._display_frames is None:
            return
        for slot, (canvas, frame) in enumerate(
            ((self.canvas_a, self._display_frames[0]), (self.canvas_b, self._display_frames[1]))
        ):
            geometry = self._view_geometry(slot)
            width = int(geometry["canvas_w"])
            height = int(geometry["canvas_h"])
            frame_h, frame_w = frame.shape[:2]
            scale = geometry["scale"]
            visible_w = geometry["visible_w"]
            visible_h = geometry["visible_h"]
            center_x = geometry["center_x"]
            center_y = geometry["center_y"]
            self._view_centers[slot] = (center_x, center_y)

            crop_w = max(1, min(frame_w, int(round(visible_w))))
            crop_h = max(1, min(frame_h, int(round(visible_h))))
            x0 = int(round(center_x - crop_w * 0.5))
            y0 = int(round(center_y - crop_h * 0.5))
            x0 = min(max(0, x0), frame_w - crop_w)
            y0 = min(max(0, y0), frame_h - crop_h)
            crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]

            shown_w = max(1, min(width, int(round(crop_w * scale))))
            shown_h = max(1, min(height, int(round(crop_h * scale))))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            shown = cv2.resize(crop, (shown_w, shown_h), interpolation=interpolation)

            # Draw marker corner diagnostics AFTER resize.  This keeps the
            # crosshair at one screen pixel regardless of zoom and prevents a
            # pre-rendered symbol from growing until it hides the true corner.
            if self._display_overlays is not None:
                overlay_corners, overlay_raw_corners, overlay_caption = self._display_overlays[slot]
                shown = draw_marker_overlay_post_scale(
                    shown, overlay_corners, overlay_raw_corners, overlay_caption,
                    source_x0=x0, source_y0=y0,
                    source_crop_w=crop_w, source_crop_h=crop_h,
                    show_raw_vs_subpix=bool(self.display_raw_subpix_var.get()))

            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            ppm = f"P6\n{shown_w} {shown_h}\n255\n".encode("ascii") + rgb.tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM")
            self._photos[slot] = photo
            if self._canvas_items[slot] is None:
                canvas.delete("all")
                self._canvas_items[slot] = canvas.create_image(
                    width // 2, height // 2, anchor="center", image=photo
                )
            else:
                canvas.coords(self._canvas_items[slot], width // 2, height // 2)
                canvas.itemconfigure(self._canvas_items[slot], image=photo)

            # Small unobtrusive zoom indicator in the upper-left corner.
            canvas.delete("zoom_indicator")
            canvas.create_text(
                10, 10,
                anchor="nw",
                text=f"{geometry['zoom'] * 100:.0f}%",
                fill="#eeeeee",
                font=("TkDefaultFont", 10, "bold"),
                tags="zoom_indicator",
            )

    @staticmethod
    def _set_canvas_placeholder(canvas: tk.Canvas, message: str) -> None:
        canvas.delete("all")
        canvas.create_text(12, 12, anchor="nw", text=message, fill="#bbbbbb", font=("TkDefaultFont", 11))

    def _replace_metrics(self, text: str) -> None:
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _export_payload(self) -> dict[str, Any]:
        if self._result is None or self._run_inputs is None:
            raise RuntimeError("目前沒有可匯出的分析結果。")
        configuration = {
            key: value
            for key, value in self._run_inputs.items()
            if key not in {"camera_matrix", "distortion"}
        }
        configuration["camera_matrix"] = self._run_inputs["camera_matrix"]
        configuration["distortion"] = self._run_inputs["distortion"]

        fixed = _fixed_camera_pattern_metrics(self._result)
        known = self._known_baseline_from_inputs()
        pattern_baseline = fixed["pattern_translation_baseline_mm"]
        relative_baseline = fixed["relative_rt_baseline_mm"]
        pattern_absolute_error = (
            abs(pattern_baseline - known)
            if pattern_baseline is not None and known is not None
            else None
        )
        relative_absolute_error = (
            abs(relative_baseline - known)
            if relative_baseline is not None and known is not None
            else None
        )
        derived_metrics = {
            "validation_mode": "fixed_camera_translated_pattern",
            "primary_pattern_translation_vector_camera_mm": fixed[
                "pattern_translation_vector_camera_mm"
            ],
            "primary_pattern_translation_baseline_mm": pattern_baseline,
            "known_baseline_mm": known,
            "primary_pattern_baseline_absolute_error_mm": pattern_absolute_error,
            "primary_pattern_baseline_error_percent": (
                pattern_absolute_error / known * 100.0
                if pattern_absolute_error is not None and known is not None and known > 0
                else None
            ),
            "secondary_relative_rt_baseline_mm": relative_baseline,
            "secondary_relative_rt_absolute_error_mm": relative_absolute_error,
            "secondary_relative_rt_error_percent": (
                relative_absolute_error / known * 100.0
                if relative_absolute_error is not None and known is not None and known > 0
                else None
            ),
            "relative_vs_pattern_baseline_gap_mm": fixed[
                "relative_vs_pattern_baseline_gap_mm"
            ],
            "pattern_rotation_change_deg": fixed["pattern_rotation_change_deg"],
        }
        return {
            "schema": "two_video_marker_only_rt_validation/v2_fixed_camera_pattern",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "configuration": _json_safe(configuration),
            "ui_derived_metrics": _json_safe(derived_metrics),
            "result": _json_safe(self._result),
        }

    def _known_baseline_from_inputs(self) -> float | None:
        if self._run_inputs is None:
            return None
        known = self._run_inputs.get("known_translation_mm")
        if known is not None:
            return float(known)
        vector = self._run_inputs.get("known_translation_vector")
        if vector is None:
            return None
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64).reshape(3)))

    def _export_json(self) -> None:
        try:
            payload = self._export_payload()
        except RuntimeError as exc:
            messagebox.showinfo("無結果", str(exc), parent=self)
            return
        target = filedialog.asksaveasfilename(
            title="匯出 RT 驗證 JSON",
            defaultextension=".json",
            filetypes=JSON_FILE_TYPES,
            initialfile="rt_two_video_validation_result.json",
        )
        if not target:
            return
        try:
            with Path(target).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self.status_var.set(f"已匯出 JSON：{Path(target).name}")
        except OSError as exc:
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    def _export_csv(self) -> None:
        try:
            payload = self._export_payload()
        except RuntimeError as exc:
            messagebox.showinfo("無結果", str(exc), parent=self)
            return
        target = filedialog.asksaveasfilename(
            title="匯出 RT 驗證 CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="rt_two_video_validation_result.csv",
        )
        if not target:
            return
        row = _flatten_for_csv(payload)
        try:
            with Path(target).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            self.status_var.set(f"已匯出 CSV：{Path(target).name}")
        except OSError as exc:
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    def _on_close(self) -> None:
        # The analysis thread is daemonized and never touches Tk directly.
        self.destroy()


def main() -> None:
    app = TwoVideoRTValidationUI()
    app.mainloop()


if __name__ == "__main__":
    main()
