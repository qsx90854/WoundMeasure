"""Run the production temporal RT analyzer and export its timing hierarchy.

This is a non-UI profiling entry point.  It uses the same calibration handling
and Zebra overrides as ``zebra_0825v2.py`` so timing runs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from Algorithm import camera_preprocess as camera_algo
from Algorithm import video_pose_analysis_temporal_unified_pattern_guided_local_window as pose_algo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("calibration_result_Zebra_1_monocular.json"))
    parser.add_argument("--marker-size-mm", type=float, default=8.25)
    parser.add_argument("--start-frames", type=int, default=30)
    parser.add_argument("--end-frames", type=int, default=30)
    parser.add_argument(
        "--range-mode", choices=("fixed", "half_half"), default="half_half")
    parser.add_argument(
        "--mode", choices=("original", "angle_guided"), default="original")
    parser.add_argument(
        "--feature-roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
        default=None,
        help="Normalized RT-SIFT-only crop; ArUco remains full-frame")
    parser.add_argument("--feature-scale", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    mtx_l, dist_l, _mtx_r, _dist_r, _extrinsic, _fundamental = (
        camera_algo.load_json_camera_params(str(args.calibration)))
    if mtx_l is None or dist_l is None:
        raise RuntimeError(f"Cannot load calibration: {args.calibration}")

    cap = cv2.VideoCapture(str(args.video))
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read video: {args.video}")
    height, width = first_frame.shape[:2]
    calibrated_k, _map1, _map2, _processor = camera_algo.build_undistort_processor(
        mtx_l, dist_l, (width, height), alpha=1.0)

    # Match zebra_0825v2.py runtime overrides.
    pose_algo.MIN_BASELINE_MM = 35.0
    pose_algo.MAX_BASELINE_MM = 220.0
    pose_algo.IDEAL_BASELINE_MM = 45.0
    pose_algo.PAIR_SCORE_REPROJ_W = 1.00
    pose_algo.PAIR_SCORE_BASELINE_W = 0.18
    pose_algo.PAIR_SCORE_BLUR_W = 0.18
    pose_algo.PAIR_SCORE_COVER_W = 0.12
    pose_algo.PAIR_SCORE_MARKER_W = 0.08
    pose_algo.FEATURE_IMAGE_SCALE = float(args.feature_scale)

    angle_config = {
        "enabled": args.mode == "angle_guided",
        "direction_mode": "auto",
        "normalize_output_roles": True,
        "target_frame_A_deg": 15.0,
        "target_frame_B_deg": 35.0,
        "target_tolerance_deg": 6.0,
        "coarse_samples_per_segment": 12,
        "max_scan_frames_per_segment": 18,
        "candidates_per_side": 3,
        "pair_score_weight": 0.35,
    }
    result = pose_algo.analyze_video_frames(
        str(args.video), args.start_frames, args.end_frames,
        calibrated_k, dist_l, mtx_l, args.marker_size_mm,
        "reproj_min", args.range_mode,
        feature_roi_ratio=(
            tuple(args.feature_roi) if args.feature_roi is not None else None),
        angle_guided_config=angle_config)
    if result is None:
        raise RuntimeError("Pose analysis failed")

    report = {
        "video": str(args.video.resolve()),
        "frame_count": int(len(result["all_frames"])),
        "mode": args.mode,
        "feature_roi_ratio": args.feature_roi,
        "feature_image_scale": float(args.feature_scale),
        "idx_A": int(result["idx_A"]),
        "idx_B": int(result["idx_B"]),
        "baseline_mm": float(result["baseline"]),
        "analysis_total_elapsed_s": float(result["analysis_total_elapsed_s"]),
        "analysis_stage_timing_s": result["analysis_stage_timing_s"],
        "pair_search_timing_s": result["pair_search_timing_s"],
        "pair_search_detail_timing_s": result["pair_search_detail_timing_s"],
    }
    output_path = args.output or args.video.with_name(
        f"{args.video.stem}_{args.mode}_timing.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Timing JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()
