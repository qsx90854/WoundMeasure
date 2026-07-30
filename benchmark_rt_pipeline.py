"""Headless benchmark for the Zebra video RT analysis pipeline."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

from Algorithm import camera_preprocess
from Algorithm import video_pose_analysis


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "test_video_Zebra" / "test_rt"
DEFAULT_CAMERA_JSON = ROOT / "calibration_result_Zebra_1_no_dis.json"
DEFAULT_BASELINE_GIT_REF = "e0f9f5752e4026239094918ff1146b199898f70d"
MAX_TIME_RATIO = 0.20
MAX_RESULT_DELTA_PCT = 3.0


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_percent(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a64 - b64) / max(np.linalg.norm(b64), 1e-12) * 100.0)


def load_camera(video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mtx_l, dist_l, *_ = camera_preprocess.load_json_camera_params(str(DEFAULT_CAMERA_JSON))
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    h, w = frame.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (w, h), 1.0, (w, h))
    return mtx_l, dist_l, np.asarray(new_k, dtype=np.float64)


def extract_free_selection_quality(result: dict) -> dict:
    quality = result.get("rt_quality") or {}
    feature = quality.get("final_feature_stats") or {}
    marker = quality.get("marker_bidir") or {}
    return {
        "rt_reliable": bool(quality.get("rt_reliable", False)),
        "feature_geometry_ok": bool(quality.get("feature_quality_ok", False)),
        "feature_final_ok": bool(quality.get("feature_final_ok", False)),
        "feature_match_count": int(quality.get("feature_matches", 0) or 0),
        "feature_inlier_count": int(quality.get("feature_inliers", 0) or 0),
        "feature_inlier_ratio": quality.get("feature_inlier_ratio"),
        "feature_grid_coverage": quality.get("feature_grid_coverage"),
        "feature_hull_coverage": quality.get("feature_hull_coverage"),
        "feature_parallax_deg": quality.get("feature_parallax_deg"),
        "feature_model_epi_px": quality.get("feature_model_epi_px"),
        "feature_rotation_agreement_deg": quality.get("feature_rot_agreement_deg"),
        "marker_parallax_deg": quality.get("marker_parallax_deg"),
        "effective_baseline_mm": quality.get("effective_baseline_mm"),
        "predicted_depth_sigma_mm": quality.get("predicted_depth_sigma_mm"),
        "final_feature_inlier_count": int(feature.get("inlier_count", 0) or 0),
        "final_feature_inlier_ratio": feature.get("inlier_ratio"),
        "final_feature_median_px": feature.get("inlier_median_px"),
        "final_feature_p90_px": feature.get("inlier_p90_px"),
        "holdout_count": int(feature.get("holdout_count", 0) or 0),
        "holdout_median_px": feature.get("holdout_median_px"),
        "holdout_p90_px": feature.get("holdout_p90_px"),
        "marker_rms_px": marker.get("rms_px"),
        "marker_max_px": marker.get("max_px"),
        "marker_left_to_right_rms_px": marker.get("left_to_right_rms_px"),
        "marker_right_to_left_rms_px": marker.get("right_to_left_rms_px"),
    }


def load_analysis_from_git(git_ref: str):
    source_path = "Algorithm/video_pose_analysis.py"
    command = ["git", "show", f"{git_ref}:{source_path}"]
    try:
        source = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(
            f"Cannot load baseline algorithm from Git ref {git_ref}: {detail.strip()}") from error

    module_name = "Algorithm._benchmark_video_pose_analysis_baseline"
    module = types.ModuleType(module_name)
    module.__file__ = f"{git_ref}:{source_path}"
    module.__package__ = "Algorithm"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def run_video(
    video_path: Path,
    range_mode: str,
    full_log_dir: Path | None = None,
    analysis_module=video_pose_analysis,
) -> dict:
    mtx_l, dist_l, new_k = load_camera(video_path)
    captured = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        result = analysis_module.analyze_video_frames(
            str(video_path),
            30,
            30,
            new_k,
            dist_l,
            mtx_l,
            8.25,
            "reproj_min",
            range_mode,
        )
    elapsed = time.perf_counter() - started
    if full_log_dir is not None:
        full_log_dir.mkdir(parents=True, exist_ok=True)
        (full_log_dir / f"{video_path.stem}.log").write_text(captured.getvalue(), encoding="utf-8")
    if result is None:
        return {
            "video": video_path.name,
            "success": False,
            "elapsed_s": elapsed,
            "log_tail": captured.getvalue().splitlines()[-20:],
        }
    return {
        "video": video_path.name,
        "success": True,
        "elapsed_s": elapsed,
        "idx_A": int(result["idx_A"]),
        "idx_B": int(result["idx_B"]),
        "R": np.asarray(result["R_rel"], dtype=float).tolist(),
        "t": np.asarray(result["t_rel"], dtype=float).reshape(3).tolist(),
        "baseline": float(result["baseline"]),
        "rt_reliable": bool(result.get("rt_quality", {}).get("rt_reliable", False)),
        "marker_bidir_rms_px": result.get("rt_quality", {}).get("marker_bidir_rms_px"),
        "feature_epi_px": result.get("rt_quality", {}).get("final_feature_epi_px"),
        "free_quality": extract_free_selection_quality(result),
        "stage_lines": [
            line.strip()
            for line in captured.getvalue().splitlines()
            if "ms (" in line or "s" in line and "總耗時" in line
        ],
    }


def add_comparison_rows(results: list[dict], baseline_results: list[dict]) -> None:
    baseline = {row["video"]: row for row in baseline_results}
    for row in results:
        ref = baseline.get(row["video"])
        if not row.get("success") or not ref or not ref.get("success"):
            continue
        comparison = {
            "time_ratio": float(row["elapsed_s"] / ref["elapsed_s"]),
            "R_frobenius_pct": relative_percent(row["R"], ref["R"]),
            "R_angle_deg": rotation_delta_deg(np.asarray(row["R"]), np.asarray(ref["R"])),
            "t_pct": relative_percent(row["t"], ref["t"]),
            "baseline_pct": abs(row["baseline"] - ref["baseline"]) / max(abs(ref["baseline"]), 1e-12) * 100.0,
            "same_frames": row["idx_A"] == ref["idx_A"] and row["idx_B"] == ref["idx_B"],
        }
        comparison["new_time_pct_of_old"] = comparison["time_ratio"] * 100.0
        comparison["time_reduction_pct"] = (1.0 - comparison["time_ratio"]) * 100.0
        comparison["speedup_x"] = 1.0 / max(comparison["time_ratio"], 1e-12)
        row["comparison"] = comparison
        row["acceptance"] = {
            "time_ok": comparison["time_ratio"] <= MAX_TIME_RATIO,
            "R_ok": comparison["R_frobenius_pct"] <= MAX_RESULT_DELTA_PCT,
            "t_ok": comparison["t_pct"] <= MAX_RESULT_DELTA_PCT,
            "baseline_ok": comparison["baseline_pct"] <= MAX_RESULT_DELTA_PCT,
        }
        row["acceptance"]["all_ok"] = all(row["acceptance"].values())


def add_comparison(results: list[dict], baseline_path: Path) -> None:
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_results = payload.get("baseline_results") or payload["results"]
    add_comparison_rows(results, baseline_results)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_csv(output_path: Path, results: list[dict]) -> Path:
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    fieldnames = [
        "video",
        "old_elapsed_s",
        "new_elapsed_s",
        "new_time_pct_of_old",
        "time_reduction_pct",
        "speedup_x",
        "R_frobenius_pct",
        "t_pct",
        "baseline_pct",
        "same_frames",
        "all_ok",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            comparison = row.get("comparison") or {}
            time_ratio = comparison.get("time_ratio")
            old_elapsed = (
                row["elapsed_s"] / time_ratio
                if time_ratio is not None and time_ratio > 0
                else None
            )
            writer.writerow({
                "video": row["video"],
                "old_elapsed_s": old_elapsed,
                "new_elapsed_s": row.get("elapsed_s"),
                "new_time_pct_of_old": (
                    comparison.get("new_time_pct_of_old")
                    if "new_time_pct_of_old" in comparison
                    else time_ratio * 100.0 if time_ratio is not None else None
                ),
                "time_reduction_pct": (
                    comparison.get("time_reduction_pct")
                    if "time_reduction_pct" in comparison
                    else (1.0 - time_ratio) * 100.0 if time_ratio is not None else None
                ),
                "speedup_x": (
                    comparison.get("speedup_x")
                    if "speedup_x" in comparison
                    else 1.0 / time_ratio if time_ratio else None
                ),
                "R_frobenius_pct": comparison.get("R_frobenius_pct"),
                "t_pct": comparison.get("t_pct"),
                "baseline_pct": comparison.get("baseline_pct"),
                "same_frames": comparison.get("same_frames"),
                "all_ok": (row.get("acceptance") or {}).get("all_ok"),
            })
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, required=True)
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline", type=Path)
    baseline_group.add_argument(
        "--baseline-git-ref",
        nargs="?",
        const=DEFAULT_BASELINE_GIT_REF,
        metavar="REF",
        help=(
            "run the old algorithm from a Git ref before the current algorithm; "
            f"omit REF to use {DEFAULT_BASELINE_GIT_REF[:12]}"
        ),
    )
    parser.add_argument("--range-mode", choices=("half_half", "fixed"), default="half_half")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-log-dir", type=Path)
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    baseline_results = None
    if args.baseline_git_ref:
        print(f"Loading baseline algorithm from {args.baseline_git_ref}", flush=True)
        baseline_module = load_analysis_from_git(args.baseline_git_ref)
        baseline_results = []
        baseline_log_dir = args.full_log_dir / "baseline" if args.full_log_dir else None
        for index, video in enumerate(videos, 1):
            print(f"[baseline {index}/{len(videos)}] {video.name}", flush=True)
            row = run_video(video, args.range_mode, baseline_log_dir, baseline_module)
            baseline_results.append(row)
            print(f"  success={row['success']} elapsed={row['elapsed_s']:.3f}s", flush=True)

    results = []
    current_log_dir = (
        args.full_log_dir / "current"
        if args.full_log_dir and args.baseline_git_ref
        else args.full_log_dir
    )
    for index, video in enumerate(videos, 1):
        label = "current " if args.baseline_git_ref else ""
        print(f"[{label}{index}/{len(videos)}] {video.name}", flush=True)
        row = run_video(video, args.range_mode, current_log_dir)
        results.append(row)
        print(f"  success={row['success']} elapsed={row['elapsed_s']:.3f}s", flush=True)

    if args.baseline:
        add_comparison(results, args.baseline)
    elif baseline_results is not None:
        add_comparison_rows(results, baseline_results)
    payload = {
        "range_mode": args.range_mode,
        "baseline_git_ref": args.baseline_git_ref,
        "baseline_results": baseline_results,
        "results": results,
    }
    args.output.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    if args.baseline or args.baseline_git_ref:
        summary_path = write_summary_csv(args.output, results)
        print(f"Wrote {summary_path}")
        failed = [
            row["video"] for row in results
            if not row.get("acceptance", {}).get("all_ok", False)
        ]
        if failed:
            raise SystemExit(f"Acceptance failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
