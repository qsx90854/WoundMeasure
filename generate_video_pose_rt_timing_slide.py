"""Generate a 16:9 Traditional-Chinese timing infographic from profiler JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
FONT_REGULAR = r"C:\Windows\Fonts\msjh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msjhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, fill, bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def fit_text(draw, xy, value, max_width, start_size, fill, bold=False, min_size=18):
    size = start_size
    while size > min_size:
        bbox = draw.textbbox((0, 0), value, font=font(size, bold))
        if bbox[2] - bbox[0] <= max_width:
            break
        size -= 1
    text(draw, xy, value, size, fill, bold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("rt_timing_profile_original.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("flowcharts/video_pose_rt_timing_overview_original.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    parent = data["pair_search_timing_s"]
    detail = data["pair_search_detail_timing_s"]
    total_ms = data["analysis_total_elapsed_s"] * 1000.0
    search_ms = sum(
        parent[key] for key in (
            "endpoint_proposal", "marker_map_temporal",
            "pattern_guided_expansion", "local_window_klt",
            "candidate_enumeration", "sift_essential_rerank",
            "other_overhead")) * 1000.0

    stages = [
        ("endpoint_proposal", "端點提案", "固定fraction；或ArUco PnP搜尋15°/35°", "原始模式不掃角度", "#94A3B8"),
        ("marker_map_temporal", "Marker map＋時序DP", "共視圖建圖、IPPE多解、二階DP選連續姿態", "probe解碼/ArUco", "#2563EB"),
        ("pattern_guided_expansion", "Pattern-guided擴充", "檢查baseline、重疊、視差、depth σ；必要才補probe", "Core幾何評估", "#14B8A6"),
        ("local_window_klt", "Local window＋KLT", "端點鄰幀、KLT連續性、清晰度與Local/Global DP重排", "Local重排＋DP", "#F59E0B"),
        ("candidate_enumeration", "Pair / IPPE枚舉", "Pair×branch計算RT、重投影、baseline與Top-K預算", "組合幾何評分", "#8B5CF6"),
        ("sift_essential_rerank", "SIFT＋Essential重排", "ratio/mutual匹配、E-RANSAC、recoverPose與極線驗證", "SIFT特徵提取", "#EF4444"),
    ]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#F4F7FB")
    draw = ImageDraw.Draw(image)

    # Header
    draw.rectangle((0, 0, WIDTH, 118), fill="#112A5C")
    text(draw, (62, 31), "Video Frame Pair 與 RT 估計｜算法耗時分解", 46, "white", True)
    text(draw, (1856, 49), "ORIGINAL MODE", 22, "#BFD7FF", True, "ra")
    text(draw, (62, 130), "257幀・1920×1080・8.25 mm ArUco Pattern｜數值為單次實測，重點是定位瓶頸與理解資料流", 23, "#53657D")

    # Summary chips
    chips = [
        ("完整分析", f"{total_ms / 1000.0:.3f} s", "#112A5C"),
        ("Frame pair搜尋", f"{search_ms / 1000.0:.3f} s", "#2563EB"),
        ("搜尋占完整流程", f"{search_ms / max(total_ms, 1e-9) * 100.0:.1f}%", "#F59E0B"),
        ("最終Pair / baseline", f"F{data['idx_A']}  |  F{data['idx_B']}  /  {data['baseline_mm']:.2f} mm", "#14B8A6"),
    ]
    chip_x = [62, 388, 714, 1040]
    chip_w = [296, 296, 296, 818]
    for (label, value, color), x, width in zip(chips, chip_x, chip_w):
        rounded(draw, (x, 174, x + width, 248), 18, "white", "#D7E0EC", 2)
        text(draw, (x + 18, 188), label, 18, "#64748B")
        fit_text(draw, (x + 18, 215), value, width - 36, 27, color, True, 20)

    # Left timing panel
    left = (48, 274, 718, 898)
    rounded(draw, left, 24, "white", "#D8E2EF", 2)
    text(draw, (78, 301), "搜尋階段耗時占比", 31, "#15233D", True)
    text(draw, (78, 340), "父階段（依執行順序，不依大小排序）", 19, "#718096")

    # Donut chart
    cx, cy, radius = 245, 535, 148
    start_angle = -90.0
    stage_total = max(search_ms, 1e-9)
    for key, _label, _desc, _hotspot, color in stages:
        sweep = parent[key] * 1000.0 / stage_total * 360.0
        draw.pieslice(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            start=start_angle, end=start_angle + sweep, fill=color)
        start_angle += sweep
    draw.ellipse(
        (cx - 88, cy - 88, cx + 88, cy + 88), fill="white")
    text(draw, (cx, cy - 20), f"{search_ms / 1000.0:.3f}s", 38, "#13233F", True, "mm")
    text(draw, (cx, cy + 29), "PAIR SEARCH", 17, "#64748B", True, "mm")

    legend_y = 386
    for index, (key, label, _desc, _hotspot, color) in enumerate(stages, start=1):
        elapsed_ms = parent[key] * 1000.0
        pct = elapsed_ms / stage_total * 100.0
        y = legend_y + (index - 1) * 52
        rounded(draw, (430, y + 5, 450, y + 25), 5, color)
        text(draw, (463, y), f"{index}. {label}", 19, "#26364F", True)
        text(draw, (463, y + 26), f"{elapsed_ms:,.1f} ms  ·  {pct:.1f}%", 17, "#64748B")

    rounded(draw, (78, 724, 688, 864), 20, "#EEF5FF", "#C8DCF8", 2)
    text(draw, (100, 744), "效能重點", 23, "#1D4ED8", True)
    text(draw, (100, 780), "Local window、Marker時序、SIFT", 25, "#14233D", True)
    top_three = (
        parent["local_window_klt"] + parent["marker_map_temporal"]
        + parent["sift_essential_rerank"])
    text(draw, (100, 817), f"前三區合計 {top_three * 1000.0 / stage_total * 100.0:.1f}%；優化時先看解碼/ArUco與Local DP。", 19, "#50647F")

    # Right algorithm panel
    rounded(draw, (744, 274, 1872, 898), 24, "white", "#D8E2EF", 2)
    text(draw, (774, 301), "六區算法：做什麼、花多久、瓶頸在哪", 31, "#15233D", True)
    text(draw, (774, 340), "每一列對應終端Log的一個父階段；縮排子項可再定位內部成本", 19, "#718096")

    row_y = 377
    row_h = 78
    row_gap = 10
    for index, (key, label, desc, hotspot, color) in enumerate(stages, start=1):
        y = row_y + (index - 1) * (row_h + row_gap)
        rounded(draw, (772, y, 1844, y + row_h), 17, "#F9FBFE", "#E1E8F2", 1)
        rounded(draw, (790, y + 14, 840, y + 64), 15, color)
        text(draw, (815, y + 39), index, 25, "white", True, "mm")
        text(draw, (860, y + 11), label, 24, "#1A2A44", True)
        text(draw, (860, y + 43), desc, 18, "#52647D")
        elapsed_ms = parent[key] * 1000.0
        pct = elapsed_ms / stage_total * 100.0
        text(draw, (1818, y + 13), f"{elapsed_ms:,.1f} ms", 24, color, True, "ra")
        text(draw, (1818, y + 45), f"{pct:.1f}%  ·  熱點：{hotspot}", 17, "#66758A", False, "ra")

    # Bottom full-pipeline strip
    text(draw, (58, 920), "Pair選出後（完整流程其餘階段）", 23, "#26364F", True)
    stage_map = {entry["stage"]: entry["elapsed_s"] * 1000.0
                 for entry in data.get("analysis_stage_timing_s", [])}
    post = [
        ("影片索引", stage_map.get(f"影片索引建立({data['frame_count']} 幀)", 0.0), "建立lazy frame索引"),
        ("聯合RT精修", stage_map.get("混合RT精修+次佳打包", 0.0), "Marker公制＋SIFT robust residual"),
        ("品質與診斷", stage_map.get("最終RT閉環+品質驗證", 0.0) + stage_map.get("RT SIFT診斷檔輸出", 0.0), "閉環、重投影、極線統計"),
        ("影像校正", stage_map.get("影像去畸變+輸出幀準備", 0.0), "remap最終左右影格"),
        ("基準平面", stage_map.get("基準平面建立", 0.0), "三角化reference corners"),
    ]
    post_colors = ["#64748B", "#2563EB", "#8B5CF6", "#14B8A6", "#F59E0B"]
    x = 58
    box_w = 352
    for (label, elapsed, desc), color in zip(post, post_colors):
        rounded(draw, (x, 953, x + box_w, 1033), 16, "white", "#D9E2ED", 2)
        draw.rectangle((x, 953, x + 8, 1033), fill=color)
        text(draw, (x + 22, 966), label, 20, "#26364F", True)
        text(draw, (x + box_w - 18, 966), f"{elapsed:.1f} ms", 20, color, True, "ra")
        fit_text(draw, (x + 22, 1002), desc, box_w - 44, 16, "#66758A", False, 14)
        x += box_w + 20

    text(draw, (1862, 1054), "時間受硬體、影片內容、快取與OS排程影響｜來源：rt_timing_profile_original.json", 15, "#8492A6", anchor="ra")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
