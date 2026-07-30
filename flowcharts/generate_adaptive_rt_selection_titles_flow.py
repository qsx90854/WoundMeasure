from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 2850
OUTPUT = Path(__file__).with_name("adaptive_rt_selection_titles_flow.png")


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
FONT_NODE = load_font(30, bold=True)
FONT_BRANCH = load_font(21, bold=True)


def centered_text(draw, box, text, font, fill):
    x, y, w, h = box
    bounds = draw.textbbox((0, 0), text, font=font)
    tw = bounds[2] - bounds[0]
    th = bounds[3] - bounds[1]
    draw.text(
        (x + (w - tw) / 2, y + (h - th) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def process_node(draw, box, title, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=18, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 15, y + h - 1), radius=6, fill=accent)
    centered_text(draw, (x + 40, y, w - 75, h), title, FONT_NODE, "#162033")


def decision_node(draw, center, size, title):
    cx, cy = center
    w, h = size
    points = [(cx, cy - h / 2), (cx + w / 2, cy),
              (cx, cy + h / 2), (cx - w / 2, cy)]
    draw.polygon(points, fill="#fff7dc", outline="#c49324")
    draw.line(points + [points[0]], fill="#c49324", width=4, joint="curve")
    centered_text(draw, (cx - w * 0.31, cy - h * 0.25, w * 0.62, h * 0.50),
                  title, FONT_NODE, "#4b3811")


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


def branch_label(draw, x, y, text):
    bounds = draw.textbbox((0, 0), text, font=FONT_BRANCH)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    draw.rounded_rectangle(
        (x - 12, y - 7, x + w + 12, y + h + 8), radius=8,
        fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_BRANCH, fill="#374151")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fa")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "自適應 RT 選幀與品質驗證流程", font=FONT_TITLE, fill="#111827")
draw.text((92, 125), "Title-only flow", font=FONT_SUBTITLE, fill="#526071")
draw.line((90, 180, WIDTH - 90, 180), fill="#c8d0db", width=3)

boxes = {
    "input": (350, 225, 1100, 125),
    "sample": (250, 425, 1300, 135),
    "aruco": (250, 635, 1300, 135),
    "candidate": (250, 845, 1300, 135),
    "rank": (250, 1230, 1300, 135),
    "sift1": (250, 1440, 1300, 135),
    "sift2": (1080, 1780, 620, 135),
    "refine": (250, 2025, 1300, 135),
    "reject": (70, 2390, 580, 145),
    "output": (810, 2390, 900, 145),
    "csv": (400, 2645, 1000, 130),
}

process_node(draw, boxes["input"], "輸入影片與相機參數",
             "#e8f1ff", "#6d9ee8", "#2563eb")
process_node(draw, boxes["sample"], "跨段候選抽幀",
             "#e7f7f3", "#58a999", "#0f766e")
process_node(draw, boxes["aruco"], "平行 ArUco 快速分析",
             "#eef7e9", "#7cac67", "#3c8c37")
process_node(draw, boxes["candidate"], "幀對產生與 Baseline 初篩",
             "#f2eefe", "#9478d3", "#6d45b8")
decision_node(draw, (900, 1085), (850, 150), "存在合格候選？")
process_node(draw, boxes["rank"], "ArUco 候選排序",
             "#e8f6fb", "#62a9c4", "#147b9d")
process_node(draw, boxes["sift1"], "第一候選 SIFT 幾何驗證",
             "#fff0e5", "#d9955c", "#c05a18")
decision_node(draw, (900, 1680), (850, 160), "需要第二候選？")
process_node(draw, boxes["sift2"], "第二候選 SIFT",
             "#fdebec", "#d98287", "#b4232f")
process_node(draw, boxes["refine"], "RT 聯合精修與獨立 Holdout",
             "#f0ecfa", "#8f7abc", "#67469b")
decision_node(draw, (900, 2280), (850, 160), "最終品質全部通過？")
process_node(draw, boxes["reject"], "拒絕／標記不可靠",
             "#edf1f6", "#8491a3", "#536174")
process_node(draw, boxes["output"], "輸出可靠 RT 與 Baseline",
             "#e8f5ea", "#70aa77", "#2f7d3c")
process_node(draw, boxes["csv"], "CSV／JSON 品質紀錄",
             "#e9f2ff", "#6f98cf", "#315f9b")

arrow(draw, [(900, 350), (900, 425)])
arrow(draw, [(900, 560), (900, 635)])
arrow(draw, [(900, 770), (900, 845)])
arrow(draw, [(900, 980), (900, 1010)])
arrow(draw, [(900, 1160), (900, 1230)])
branch_label(draw, 930, 1170, "是")
arrow(draw, [(475, 1085), (190, 1085), (190, 2390)])
branch_label(draw, 205, 1035, "否")
arrow(draw, [(900, 1365), (900, 1440)])
arrow(draw, [(900, 1575), (900, 1600)])
arrow(draw, [(1325, 1680), (1390, 1680), (1390, 1780)])
branch_label(draw, 1410, 1630, "是")
arrow(draw, [(1390, 1915), (1390, 1965), (900, 1965), (900, 2025)])
arrow(draw, [(900, 1760), (900, 2025)])
branch_label(draw, 930, 1825, "否")
arrow(draw, [(900, 2160), (900, 2200)])
arrow(draw, [(475, 2280), (360, 2280), (360, 2390)])
branch_label(draw, 380, 2230, "否")
arrow(draw, [(1325, 2280), (1260, 2280), (1260, 2390)])
branch_label(draw, 1280, 2230, "是")
arrow(draw, [(1260, 2535), (1260, 2585), (900, 2585), (900, 2645)])

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
