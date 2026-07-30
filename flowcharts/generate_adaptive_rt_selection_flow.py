from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 3400
OUTPUT = Path(__file__).with_name("adaptive_rt_selection_flow.png")


def load_font(size, bold=False):
    names = (
        ("msjhbd.ttc", "msjh.ttc", "seguisb.ttf")
        if bold else
        ("msjh.ttc", "msjhbd.ttc", "segoeui.ttf")
    )
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(52, bold=True)
FONT_SUBTITLE = load_font(25)
FONT_NODE_TITLE = load_font(28, bold=True)
FONT_NODE_BODY = load_font(22)
FONT_BRANCH = load_font(20, bold=True)
FONT_FOOTER = load_font(19)


def wrap_lines(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            trial = current + char
            if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
                lines.append(current)
                current = char
            else:
                current = trial
        lines.append(current)
    return lines


def draw_centered_lines(draw, box, text, font, fill, spacing=8):
    x, y, w, h = box
    lines = wrap_lines(draw, text, font, w)
    heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total = sum(heights) + spacing * max(len(lines) - 1, 0)
    cursor_y = y + (h - total) / 2
    for line, line_h in zip(lines, heights):
        bounds = draw.textbbox((0, 0), line, font=font)
        line_w = bounds[2] - bounds[0]
        draw.text((x + (w - line_w) / 2, cursor_y), line, font=font, fill=fill)
        cursor_y += line_h + spacing


def process_node(draw, box, title, body, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=18, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 15, y + h - 1), radius=6, fill=accent)
    draw_centered_lines(draw, (x + 45, y + 18, w - 80, 45), title,
                        FONT_NODE_TITLE, "#162033", spacing=2)
    draw_centered_lines(draw, (x + 55, y + 68, w - 100, h - 82), body,
                        FONT_NODE_BODY, "#465166", spacing=7)


def decision_node(draw, center, size, title, body):
    cx, cy = center
    w, h = size
    points = [(cx, cy - h / 2), (cx + w / 2, cy),
              (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(points, fill="#fff7dc", outline="#c49324")
    draw.line(points + [points[0]], fill="#c49324", width=4, joint="curve")
    draw_centered_lines(draw, (cx - w * 0.31, cy - 53, w * 0.62, 44), title,
                        FONT_NODE_TITLE, "#4b3811", spacing=2)
    draw_centered_lines(draw, (cx - w * 0.28, cy - 2, w * 0.56, 90), body,
                        FONT_NODE_BODY, "#66511d", spacing=5)


def arrow(draw, points, color="#657184", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 17
    wing_a = (x2 + size * math.cos(angle + 2.55),
              y2 + size * math.sin(angle + 2.55))
    wing_b = (x2 + size * math.cos(angle - 2.55),
              y2 + size * math.sin(angle - 2.55))
    draw.polygon([(x2, y2), wing_a, wing_b], fill=color)


def branch_label(draw, x, y, text, fill="#ffffff"):
    bounds = draw.textbbox((0, 0), text, font=FONT_BRANCH)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    draw.rounded_rectangle(
        (x - 12, y - 7, x + w + 12, y + h + 8), radius=8,
        fill=fill, outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_BRANCH, fill="#374151")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fa")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "自適應 RT 選幀與品質驗證流程", font=FONT_TITLE, fill="#111827")
draw.text(
    (92, 125),
    "Adaptive frame selection | Baseline >= 20 mm | Target runtime < 3 s",
    font=FONT_SUBTITLE,
    fill="#526071",
)
draw.line((90, 180, WIDTH - 90, 180), fill="#c8d0db", width=3)

boxes = {
    "input": (350, 230, 1100, 150),
    "sample": (250, 465, 1300, 185),
    "aruco": (250, 740, 1300, 190),
    "candidate": (250, 1025, 1300, 220),
    "rank": (250, 1505, 1300, 220),
    "sift1": (250, 1795, 1300, 180),
    "sift2": (1050, 2235, 650, 190),
    "refine": (250, 2460, 1300, 205),
    "reject": (70, 2890, 580, 185),
    "output": (810, 2890, 900, 215),
    "csv": (400, 3200, 1000, 145),
}

process_node(draw, boxes["input"], "輸入影片與相機參數",
             "讀取影片索引、內參 K、畸變參數與 ArUco 尺寸",
             "#e8f1ff", "#6d9ee8", "#2563eb")
process_node(draw, boxes["sample"], "跨段候選抽幀",
             "取影片約 40%、50%、55%、61%、70% 位置\n避免只固定使用影片中段",
             "#e7f7f3", "#58a999", "#0f766e")
process_node(draw, boxes["aruco"], "平行 ArUco 快速分析",
             "灰階 → CLAHE → ArUco 偵測並行執行\n建立共享 marker 地圖、初估 pose、清晰度與覆蓋率",
             "#eef7e9", "#7cac67", "#3c8c37")
process_node(draw, boxes["candidate"], "產生並初篩幀對",
             "ArUco 初估 baseline：22–220 mm（含 2 mm 安全裕度）\n計算 marker 雙向重投影、視差角、有效 baseline\n估算 400 mm 距離、1 px 匹配誤差下的深度 σ",
             "#f2eefe", "#9478d3", "#6d45b8")

decision_node(draw, (900, 1370), (900, 210), "存在合格候選？",
              "至少一組共享 marker 且初估 baseline >= 22 mm")

process_node(draw, boxes["rank"], "ArUco 候選排序",
             "綜合重投影、baseline 接近 30 mm、預估深度 σ、\n清晰度、marker 覆蓋率；保留第一名與接近的第二名",
             "#e8f6fb", "#62a9c4", "#147b9d")
process_node(draw, boxes["sift1"], "第一候選 SIFT 幾何驗證",
             "Essential RANSAC、inlier 分布、feature parallax、\nmarker-feature 極線殘差與 rotation agreement",
             "#fff0e5", "#d9955c", "#c05a18")

decision_node(draw, (900, 2110), (900, 230), "需要第二候選？",
              "geometry 失敗，或 parallax < 5°，\n或 marker_epi > 6 px，或 rotation 差 > 5°")

process_node(draw, boxes["sift2"], "第二候選 SIFT",
             "只在第一候選不穩時才執行\n以 >= 0.05 的改善門檻避免微小波動換幀",
             "#fdebec", "#d98287", "#b4232f")
process_node(draw, boxes["refine"], "RT 聯合精修與獨立 holdout",
             "IPPE 分支消歧 + marker 約束 + feature robust residual\n保留部分 Essential inliers 不參與最佳化，作為獨立驗證",
             "#f0ecfa", "#8f7abc", "#67469b")

decision_node(draw, (900, 2785), (900, 220), "最終品質全部通過？",
              "baseline >= 20 mm、marker RMS <= 1.5 px、max <= 2 px\nfeature p90 <= 1.25 px、holdout median <= 1.5 px")

process_node(draw, boxes["reject"], "拒絕／標記不可靠",
             "不輸出可信 RT\n記錄失敗品質欄位供追查",
             "#edf1f6", "#8491a3", "#536174")
process_node(draw, boxes["output"], "輸出可靠結果",
             "目標 Frame A / B、R、t、baseline\nRT reliable = True；進入後續去畸變、平面與深度流程",
             "#e8f5ea", "#70aa77", "#2f7d3c")
process_node(draw, boxes["csv"], "CSV / JSON 品質紀錄",
             "baseline_mm、marker_parallax_deg、effective_baseline_mm、predicted_depth_sigma_mm",
             "#e9f2ff", "#6f98cf", "#315f9b")

arrow(draw, [(900, 380), (900, 465)])
arrow(draw, [(900, 650), (900, 740)])
arrow(draw, [(900, 930), (900, 1025)])
arrow(draw, [(900, 1245), (900, 1265)])
arrow(draw, [(900, 1475), (900, 1505)])
branch_label(draw, 930, 1478, "是")
arrow(draw, [(450, 1370), (185, 1370), (185, 2890)])
branch_label(draw, 205, 1315, "否")
arrow(draw, [(900, 1725), (900, 1795)])
arrow(draw, [(900, 1975), (900, 1995)])

arrow(draw, [(1350, 2110), (1675, 2110), (1675, 2235)])
branch_label(draw, 1400, 2075, "是")
arrow(draw, [(1375, 2425), (1375, 2440), (900, 2440), (900, 2460)])
arrow(draw, [(900, 2225), (900, 2460)])
branch_label(draw, 930, 2310, "否：略過第二組 SIFT")

arrow(draw, [(900, 2665), (900, 2675)])
arrow(draw, [(450, 2785), (360, 2785), (360, 2890)])
branch_label(draw, 380, 2730, "否")
arrow(draw, [(1350, 2785), (1260, 2785), (1260, 2890)])
branch_label(draw, 1280, 2730, "是")
arrow(draw, [(1260, 3105), (1260, 3140), (900, 3140), (900, 3200)])

draw.text(
    (90, HEIGHT - 38),
    "驗證基準：test_video_Zebra/test_rt，共 19 支影片；19/19 通過，最慢 2.56 s。",
    font=FONT_FOOTER,
    fill="#687386",
)

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
