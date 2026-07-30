"""Benchmark free frame selection using independent geometric quality checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_rt_pipeline import DEFAULT_VIDEO_DIR, json_safe, run_video


MAX_TIME_S = 3.0
MIN_FEATURE_INLIERS = 11
MIN_HOLDOUT_COUNT = 3
MAX_HOLDOUT_MEDIAN_PX = 1.5
MAX_FINAL_FEATURE_P90_PX = 1.25
MAX_MARKER_RMS_PX = 1.5
MAX_MARKER_POINT_PX = 2.0
MIN_BASELINE_MM = 20.0
MAX_BASELINE_MM = 220.0


def finite_at_most(value, limit: float) -> bool:
    return value is not None and float(value) <= limit


def evaluate(row: dict) -> dict:
    quality = row.get("free_quality") or {}
    baseline = row.get("baseline")
    checks = {
        "time_ok": bool(row.get("success") and row.get("elapsed_s", float("inf")) < MAX_TIME_S),
        "baseline_range_ok": bool(
            baseline is not None and MIN_BASELINE_MM <= baseline <= MAX_BASELINE_MM),
        "marker_ok": bool(
            finite_at_most(quality.get("marker_rms_px"), MAX_MARKER_RMS_PX)
            and finite_at_most(quality.get("marker_max_px"), MAX_MARKER_POINT_PX)),
        "feature_geometry_ok": bool(quality.get("feature_geometry_ok", False)),
        "feature_final_ok": bool(
            quality.get("feature_final_ok", False)
            and quality.get("final_feature_inlier_count", 0) >= MIN_FEATURE_INLIERS
            and finite_at_most(
                quality.get("final_feature_p90_px"), MAX_FINAL_FEATURE_P90_PX)),
        "feature_holdout_ok": bool(
            quality.get("holdout_count", 0) >= MIN_HOLDOUT_COUNT
            and finite_at_most(
                quality.get("holdout_median_px"), MAX_HOLDOUT_MEDIAN_PX)),
    }
    checks["all_ok"] = all(checks.values())
    return checks


def write_summary(output_path: Path, results: list[dict]) -> Path:
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    fields = [
        "video", "elapsed_s", "idx_A", "idx_B", "baseline_mm", "all_ok",
        "time_ok", "baseline_range_ok", "rt_reliable", "marker_ok",
        "feature_geometry_ok", "feature_final_ok",
        "feature_holdout_ok", "marker_rms_px", "marker_max_px",
        "feature_inlier_count", "feature_inlier_ratio", "feature_grid_coverage",
        "feature_hull_coverage", "feature_parallax_deg", "final_feature_inlier_count",
        "final_feature_p90_px", "holdout_count", "holdout_median_px",
        "marker_parallax_deg", "effective_baseline_mm", "predicted_depth_sigma_mm",
    ]
    chinese_labels = {
        "video": "影片名稱",
        "elapsed_s": "RT計算耗時(秒)",
        "idx_A": "影格A索引",
        "idx_B": "影格B索引",
        "baseline_mm": "基線距離(mm)",
        "all_ok": "全部通過",
        "time_ok": "耗時通過",
        "baseline_range_ok": "基線範圍通過",
        "rt_reliable": "RT可信",
        "marker_ok": "Marker通過",
        "feature_geometry_ok": "初始特徵幾何通過",
        "feature_final_ok": "精修特徵通過",
        "feature_holdout_ok": "Holdout通過",
        "marker_rms_px": "Marker RMS誤差(px)",
        "marker_max_px": "Marker最大誤差(px)",
        "feature_inlier_count": "初始特徵內點數",
        "feature_inlier_ratio": "初始特徵內點率",
        "feature_grid_coverage": "特徵網格覆蓋率",
        "feature_hull_coverage": "特徵凸包覆蓋率",
        "feature_parallax_deg": "特徵視差角(度)",
        "final_feature_inlier_count": "精修後特徵內點數",
        "final_feature_p90_px": "精修後特徵P90誤差(px)",
        "holdout_count": "Holdout點數",
        "holdout_median_px": "Holdout中位誤差(px)",
        "marker_parallax_deg": "Marker視差角(度)",
        "effective_baseline_mm": "有效基線(mm)",
        "predicted_depth_sigma_mm": "預估深度誤差(mm)",
    }
    with summary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(chinese_labels)
        for row in results:
            quality = row.get("free_quality") or {}
            checks = row.get("free_acceptance") or {}
            writer.writerow({
                "video": row.get("video"),
                "elapsed_s": row.get("elapsed_s"),
                "idx_A": row.get("idx_A"),
                "idx_B": row.get("idx_B"),
                "baseline_mm": row.get("baseline"),
                "all_ok": checks.get("all_ok"),
                "time_ok": checks.get("time_ok"),
                "baseline_range_ok": checks.get("baseline_range_ok"),
                "rt_reliable": quality.get("rt_reliable"),
                "marker_ok": checks.get("marker_ok"),
                "feature_geometry_ok": checks.get("feature_geometry_ok"),
                "feature_final_ok": checks.get("feature_final_ok"),
                "feature_holdout_ok": checks.get("feature_holdout_ok"),
                "marker_rms_px": quality.get("marker_rms_px"),
                "marker_max_px": quality.get("marker_max_px"),
                "feature_inlier_count": quality.get("feature_inlier_count"),
                "feature_inlier_ratio": quality.get("feature_inlier_ratio"),
                "feature_grid_coverage": quality.get("feature_grid_coverage"),
                "feature_hull_coverage": quality.get("feature_hull_coverage"),
                "feature_parallax_deg": quality.get("feature_parallax_deg"),
                "final_feature_inlier_count": quality.get("final_feature_inlier_count"),
                "final_feature_p90_px": quality.get("final_feature_p90_px"),
                "holdout_count": quality.get("holdout_count"),
                "holdout_median_px": quality.get("holdout_median_px"),
                "marker_parallax_deg": quality.get("marker_parallax_deg"),
                "effective_baseline_mm": quality.get("effective_baseline_mm"),
                "predicted_depth_sigma_mm": quality.get("predicted_depth_sigma_mm"),
            })
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="append", help="video filename or stem; repeat as needed")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-log-dir", type=Path)
    parser.add_argument("--range-mode", choices=("half_half", "fixed"), default="half_half")
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if args.video:
        selected = {Path(name).stem for name in args.video}
        videos = [video for video in videos if video.stem in selected]
        missing = selected - {video.stem for video in videos}
        if missing:
            raise SystemExit(f"Videos not found: {', '.join(sorted(missing))}")
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"No MP4 videos found in {args.video_dir}")

    results = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        row = run_video(video, args.range_mode, args.full_log_dir)
        row["free_acceptance"] = evaluate(row)
        results.append(row)
        print(
            f"  elapsed={row['elapsed_s']:.3f}s frames={row.get('idx_A')}/{row.get('idx_B')} "
            f"pass={row['free_acceptance']['all_ok']}",
            flush=True,
        )

    thresholds = {
        "max_time_s": MAX_TIME_S,
        "min_feature_inliers": MIN_FEATURE_INLIERS,
        "min_holdout_count": MIN_HOLDOUT_COUNT,
        "max_holdout_median_px": MAX_HOLDOUT_MEDIAN_PX,
        "max_final_feature_p90_px": MAX_FINAL_FEATURE_P90_PX,
        "max_marker_rms_px": MAX_MARKER_RMS_PX,
        "max_marker_point_px": MAX_MARKER_POINT_PX,
        "baseline_range_mm": [MIN_BASELINE_MM, MAX_BASELINE_MM],
    }
    payload = {"thresholds": thresholds, "results": results}
    args.output.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_path = write_summary(args.output, results)
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")

    failed = [row["video"] for row in results if not row["free_acceptance"]["all_ok"]]
    if failed:
        raise SystemExit(f"Free-selection acceptance failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
