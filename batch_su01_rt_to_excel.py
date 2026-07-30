"""Batch SU01 stereo RT estimation and export the results to Excel.

Only the fixed-frame stereo RT stage is executed. The measurement UI, wound
model, interactive matching, and depth measurement stages are not started.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import depth_measure_multi_aruco_sbs_camera_v7_demo_SU01 as su01


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
DEFAULT_EXPORTER = Path(__file__).resolve().parent / "hbvcam_rt_excel_export.mjs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only SU01 F50 stereo RT estimation for every video in a "
            "folder and export the errors to Excel."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing stereo videos. A folder picker opens if omitted.",
    )
    parser.add_argument(
        "--output",
        help="Output .xlsx path. Default: <folder>/SU01_RT_batch_<timestamp>.xlsx",
    )
    parser.add_argument(
        "--calibration",
        default=su01.PARAMS_JSON_PATH,
        help=f"Calibration JSON (default: {su01.PARAMS_JSON_PATH})",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=su01.STEREO_REFERENCE_FRAME_INDEX,
        help=(
            "Zero-based SBS reference frame index "
            f"(default: {su01.STEREO_REFERENCE_FRAME_INDEX})"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include videos in subfolders.",
    )
    parser.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep the intermediate result JSON beside the workbook.",
    )
    return parser.parse_args()


def choose_folder() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Select SU01 stereo video folder")
    root.destroy()
    return Path(selected) if selected else None


def collect_videos(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def resolve_calibration_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def calculate_answer_errors(
    R_est: np.ndarray,
    t_est: np.ndarray,
    answer_extrinsic: dict,
) -> dict[str, float]:
    R_answer = np.asarray(answer_extrinsic["R"], dtype=np.float64).reshape(3, 3)
    t_answer = np.asarray(answer_extrinsic["T"], dtype=np.float64).reshape(3, 1)
    R_est = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3, 1)

    rotation_delta = R_est @ R_answer.T
    rotation_cos = np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
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
            np.degrees(np.arccos(direction_cos))
        )
    else:
        translation_direction_error_deg = float("nan")

    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_l2_error_mm": translation_l2_error_mm,
        "translation_direction_error_deg": translation_direction_error_deg,
        "algorithm_baseline_mm": algorithm_baseline_mm,
        "json_baseline_mm": json_baseline_mm,
        "baseline_delta_mm": baseline_delta_mm,
    }


def empty_result(index: int, video_path: Path, frame_index: int) -> dict:
    return {
        "run_number": index,
        "video_file": video_path.name,
        "video_path": str(video_path.resolve()),
        "status": "FAILED",
        "failure_reason": "",
        "frame_index": frame_index,
        "total_frames": None,
        "rotation_error_deg": None,
        "translation_l2_error_mm": None,
        "translation_direction_error_deg": None,
        "algorithm_baseline_mm": None,
        "json_baseline_mm": None,
        "baseline_delta_mm": None,
        "rt_internal_error_px": None,
        "marker_bidirectional_rms_px": None,
        "feature_median_px": None,
        "rt_reliable": None,
        "processing_time_s": None,
    }


def process_video(
    index: int,
    video_path: Path,
    frame_index: int,
    mtx_left: np.ndarray,
    dist_left: np.ndarray,
    mtx_right: np.ndarray,
    dist_right: np.ndarray,
    answer_extrinsic: dict,
) -> dict:
    started = time.perf_counter()
    result = empty_result(index, video_path, frame_index)
    video_data = None
    try:
        left_raw, right_raw, total_frames = su01.read_sbs_reference_frame(
            str(video_path), frame_index
        )
        result["total_frames"] = total_frames
        left_common, right_common, common_k = (
            su01.map_stereo_pair_to_common_intrinsic(
                left_raw,
                right_raw,
                mtx_left,
                dist_left,
                mtx_right,
                dist_right,
            )
        )

        zero_distortion = np.zeros(5, dtype=np.float64)
        # The existing analyzer defines start/frame_A as the UI right image and
        # end/frame_B as the UI left image.
        fixed_stereo_frames = [right_common, left_common]
        video_data = su01.analyze_video_frames(
            str(video_path),
            1,
            1,
            common_k,
            zero_distortion,
            common_k,
            su01.ACTUAL_MARKER_SIZE_MM,
            su01.POSE_SELECT_MODE,
            "half_half",
            frames_override=fixed_stereo_frames,
        )
        if video_data is None:
            raise RuntimeError(
                "RT analysis returned no result; check F50 ArUco visibility and image quality."
            )

        comparison = calculate_answer_errors(
            video_data["R_rel"],
            video_data["t_rel"],
            answer_extrinsic,
        )
        rt_quality = video_data.get("rt_quality") or {}
        result.update(comparison)
        result["rt_internal_error_px"] = finite_or_none(
            video_data.get("min_reproj_err")
        )
        result["marker_bidirectional_rms_px"] = finite_or_none(
            video_data.get("marker_reproj_err")
        )
        result["feature_median_px"] = finite_or_none(
            rt_quality.get("final_feature_epi_px")
        )
        result["rt_reliable"] = bool(rt_quality.get("rt_reliable", False))
        result["status"] = "OK" if result["rt_reliable"] else "QUALITY_WARNING"
        if not result["rt_reliable"]:
            result["failure_reason"] = (
                "RT result produced, but the analyzer quality flag is false."
            )
    except Exception as exc:
        result["status"] = "FAILED"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["processing_time_s"] = time.perf_counter() - started
        del video_data
        gc.collect()
    return result


def finite_or_none(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def locate_artifact_runtime() -> tuple[Path, Path]:
    dependencies = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
    )
    node_candidates = [
        dependencies / "node" / "bin" / "node.exe",
        dependencies / "node" / "bin" / "node",
    ]
    node_path = next((path for path in node_candidates if path.exists()), None)
    node_modules = dependencies / "node" / "node_modules"
    if node_path is None or not node_modules.is_dir():
        raise FileNotFoundError(
            "Codex spreadsheet runtime was not found. "
            f"Expected it below: {dependencies}"
        )
    return node_path, node_modules


def create_windows_junction(junction: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Unable to prepare the spreadsheet runtime junction: "
            f"{completed.stdout}{completed.stderr}"
        )


def export_excel(
    payload: dict,
    output_path: Path,
    keep_json: bool,
    preview_path: Path | None = None,
) -> Path:
    node_path, node_modules = locate_artifact_runtime()
    if not DEFAULT_EXPORTER.exists():
        raise FileNotFoundError(f"Excel exporter not found: {DEFAULT_EXPORTER}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    persistent_json = output_path.with_suffix(".json")
    if keep_json:
        persistent_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with tempfile.TemporaryDirectory(prefix="hbvcam_rt_excel_") as temp_name:
        temp_dir = Path(temp_name)
        junction = temp_dir / "node_modules"
        try:
            if os.name == "nt":
                create_windows_junction(junction, node_modules)
            else:
                junction.symlink_to(node_modules, target_is_directory=True)

            input_json = temp_dir / "results.json"
            exporter_copy = temp_dir / DEFAULT_EXPORTER.name
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shutil.copy2(DEFAULT_EXPORTER, exporter_copy)

            command = [
                str(node_path),
                str(exporter_copy),
                str(input_json),
                str(output_path),
            ]
            if preview_path is not None:
                command.append(str(preview_path))

            completed = subprocess.run(
                command,
                cwd=temp_dir,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            if completed.stdout:
                print(completed.stdout.rstrip())
            if completed.returncode != 0:
                raise RuntimeError(
                    "Excel exporter failed: "
                    f"{completed.stderr or completed.stdout}"
                )
        finally:
            # Remove only the junction itself before TemporaryDirectory cleanup.
            if junction.exists():
                os.rmdir(junction)
    return output_path


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser() if args.folder else choose_folder()
    if folder is None:
        print("No folder selected.")
        return 0
    folder = folder.resolve()
    if not folder.is_dir():
        print(f"Folder does not exist: {folder}")
        return 2

    videos = collect_videos(folder, args.recursive)
    if not videos:
        print(f"No supported video files found in: {folder}")
        return 1

    calibration_path = resolve_calibration_path(args.calibration)
    try:
        mtx_left, dist_left, mtx_right, dist_right, answer_extrinsic = (
            su01.load_su01_calibration(calibration_path)
        )
        if "R" not in answer_extrinsic or "T" not in answer_extrinsic:
            raise ValueError("Calibration JSON extrinsic must contain R and T.")
    except Exception as exc:
        print(f"Unable to load calibration: {exc}")
        return 2

    su01.RECORD_SAVE_DIR = str(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else folder / f"SU01_RT_batch_{timestamp}.xlsx"
    )
    if output_path.suffix.lower() != ".xlsx":
        output_path = output_path.with_suffix(".xlsx")
    output_path = output_path.resolve()

    rows = []
    print(
        f"Found {len(videos)} videos. RT frame index={args.frame_index}. "
        "The measurement UI and wound model will not be started."
    )
    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] {video_path.name}")
        row = process_video(
            index,
            video_path,
            args.frame_index,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            answer_extrinsic,
        )
        rows.append(row)
        if row["status"] == "FAILED":
            print(f"  FAILED: {row['failure_reason']}")
        else:
            baseline_percent = (
                abs(row["baseline_delta_mm"]) / row["json_baseline_mm"] * 100.0
            )
            print(
                f"  {row['status']}: rotation={row['rotation_error_deg']:.4f} deg, "
                f"baseline delta={row['baseline_delta_mm']:+.4f} mm "
                f"({baseline_percent:.2f}%)"
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_folder": str(folder),
        "calibration_file": str(calibration_path),
        "reference_frame_index": args.frame_index,
        "recursive": bool(args.recursive),
        "rows": rows,
    }

    try:
        export_excel(payload, output_path, args.keep_json)
    except Exception as exc:
        fallback_json = output_path.with_suffix(".json")
        fallback_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Excel export failed: {exc}")
        print(f"Raw batch results were retained at: {fallback_json}")
        return 3

    ok_count = sum(row["status"] != "FAILED" for row in rows)
    print(f"\nCompleted: {ok_count}/{len(rows)} videos produced RT results.")
    print(f"Excel: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
