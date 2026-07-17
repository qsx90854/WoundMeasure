"""Headless benchmark for the Zebra video RT analysis pipeline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np

from Algorithm import camera_preprocess
from Algorithm import video_pose_analysis


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "test_video_Zebra" / "test_rt"
DEFAULT_CAMERA_JSON = ROOT / "calibration_result_Zebra_1_no_dis.json"
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


def run_video(video_path: Path, range_mode: str, full_log_dir: Path | None = None) -> dict:
    mtx_l, dist_l, new_k = load_camera(video_path)
    captured = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        result = video_pose_analysis.analyze_video_frames(
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
        "stage_lines": [
            line.strip()
            for line in captured.getvalue().splitlines()
            if "ms (" in line or "s" in line and "總耗時" in line
        ],
    }


def add_comparison(results: list[dict], baseline_path: Path) -> None:
    baseline = {row["video"]: row for row in json.loads(baseline_path.read_text(encoding="utf-8"))["results"]}
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
        row["comparison"] = comparison
        row["acceptance"] = {
            "time_ok": comparison["time_ratio"] <= MAX_TIME_RATIO,
            "R_ok": comparison["R_frobenius_pct"] <= MAX_RESULT_DELTA_PCT,
            "t_ok": comparison["t_pct"] <= MAX_RESULT_DELTA_PCT,
            "baseline_ok": comparison["baseline_pct"] <= MAX_RESULT_DELTA_PCT,
        }
        row["acceptance"]["all_ok"] = all(row["acceptance"].values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--range-mode", choices=("half_half", "fixed"), default="half_half")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-log-dir", type=Path)
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    results = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        row = run_video(video, args.range_mode, args.full_log_dir)
        results.append(row)
        print(f"  success={row['success']} elapsed={row['elapsed_s']:.3f}s", flush=True)

    if args.baseline:
        add_comparison(results, args.baseline)
    payload = {"range_mode": args.range_mode, "results": results}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.baseline:
        failed = [
            row["video"] for row in results
            if not row.get("acceptance", {}).get("all_ok", False)
        ]
        if failed:
            raise SystemExit(f"Acceptance failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
