from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 3350
OUTPUT = Path(__file__).with_name("zebra_point_measure_titles_flow.png")


def load_font(names, size):
    font_dir = Path("C:/Windows/Fonts")
    for name in names:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 46)
FONT_SUBTITLE = load_font(("msjh.ttc", "segoeui.ttf", "arial.ttf"), 23)
FONT_NODE = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 30)
FONT_LABEL = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 20)


def text_width(draw, text, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0]


def centered_text(draw, box, text, font, fill):
    x, y, w, h = box
    bounds = draw.textbbox((0, 0), text, font=font)
    tw = bounds[2] - bounds[0]
    th = bounds[3] - bounds[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - bounds[1]), text, font=font, fill=fill)


def node(draw, box, title, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=18, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 13, y + h - 1), radius=6, fill=accent)
    centered_text(draw, (x + 30, y, w - 60, h), title, FONT_NODE, "#162033")


def arrow(draw, points, color="#5b6575", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    wing_a = (x2 + size * math.cos(angle + 2.55), y2 + size * math.sin(angle + 2.55))
    wing_b = (x2 + size * math.cos(angle - 2.55), y2 + size * math.sin(angle - 2.55))
    draw.polygon([(x2, y2), wing_a, wing_b], fill=color)


def branch_label(draw, x, y, text):
    bounds = draw.textbbox((0, 0), text, font=FONT_LABEL)
    draw.rounded_rectangle(
        (x - 10, y - 5, x + bounds[2] + 10, y + bounds[3] + 5),
        radius=7, fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_LABEL, fill="#384152")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "Zebra 點選量測流程", font=FONT_TITLE, fill="#111827")
draw.text((92, 115), "僅顯示主要流程標題", font=FONT_SUBTITLE, fill="#4b5563")
draw.line((90, 165, WIDTH - 90, 165), fill="#c7cfdb", width=3)

boxes = {
    "click": (300, 220, 1200, 150),
    "prepare": (300, 460, 1200, 150),
    "candidate": (300, 700, 1200, 150),
    "pose": (550, 940, 700, 140),
    "skip": (80, 1190, 650, 160),
    "aruco": (80, 1480, 760, 170),
    "feature": (960, 1480, 760, 170),
    "refine": (220, 1760, 1360, 170),
    "match": (550, 2040, 700, 140),
    "triangulate": (300, 2290, 1200, 170),
    "depth": (550, 2570, 700, 140),
    "quality": (220, 2820, 1360, 170),
    "fuse": (300, 3100, 1200, 170),
}

arrow(draw, [(900, 370), (900, 460)])
arrow(draw, [(900, 610), (900, 700)])
arrow(draw, [(900, 850), (900, 940)])

arrow(draw, [(550, 1010), (405, 1010), (405, 1190)])
branch_label(draw, 300, 1035, "無效")

arrow(draw, [(900, 1080), (900, 1400), (460, 1400), (460, 1480)])
arrow(draw, [(900, 1080), (900, 1400), (1340, 1400), (1340, 1480)])
branch_label(draw, 590, 1415, "ArUco 角點")
branch_label(draw, 1240, 1415, "一般候選點")

arrow(draw, [(460, 1650), (460, 1700), (900, 1700), (900, 1760)])
arrow(draw, [(1340, 1650), (1340, 1700), (900, 1700), (900, 1760)])
arrow(draw, [(900, 1930), (900, 2040)])

arrow(draw, [(550, 2110), (405, 2110), (405, 1350)])
branch_label(draw, 300, 2060, "失敗")
arrow(draw, [(900, 2180), (900, 2290)])
branch_label(draw, 930, 2215, "成功")

arrow(draw, [(900, 2460), (900, 2570)])
arrow(draw, [(550, 2640), (405, 2640), (405, 1350)])
branch_label(draw, 285, 2590, "Z 無效")
arrow(draw, [(900, 2710), (900, 2820)])
branch_label(draw, 930, 2740, "Z 合理")

arrow(draw, [(900, 2990), (900, 3100)])

node(draw, boxes["click"], "左圖點選候選點", "#e8f1ff", "#6d9ee8", "#2563eb")
node(draw, boxes["prepare"], "點擊與影像前處理", "#e7f7f3", "#58a999", "#0f766e")
node(draw, boxes["candidate"], "逐一處理右圖候選影格", "#fff0e5", "#d9955c", "#c05a18")
node(draw, boxes["pose"], "Pose 與 baseline 驗證", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["skip"], "略過無效候選", "#edf1f6", "#8491a3", "#536174")
node(draw, boxes["aruco"], "ArUco 直接對應", "#f2eefe", "#9478d3", "#6d45b8")
node(draw, boxes["feature"], "一般特徵匹配", "#e8f6fb", "#62a9c4", "#147b9d")
node(draw, boxes["refine"], "匹配點精修與幾何檢查", "#fdebec", "#d98287", "#b4232f")
node(draw, boxes["match"], "右圖匹配點驗證", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["triangulate"], "雙視圖三角化（dual_direct）", "#f0ecfa", "#8f7abc", "#67469b")
node(draw, boxes["depth"], "3D 深度 Z 驗證", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["quality"], "單候選量測與品質指標", "#e8f5ea", "#70aa77", "#2f7d3c")
node(draw, boxes["fuse"], "多候選深度融合與結果輸出", "#e9f2ff", "#6f98cf", "#315f9b")

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
