from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2920
HEIGHT = 1900
OUTPUT = Path(__file__).with_name("adaptive_rt_selection_compact_flow.png")


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
FONT_NODE = load_font(31, bold=True)
FONT_NOTE = load_font(22)
FONT_BRANCH = load_font(21, bold=True)
FONT_CALLOUT_TITLE = load_font(23, bold=True)
FONT_CALLOUT_BODY = load_font(20)


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
        (x, y, x + w, y + h), radius=16, fill=fill, outline=border, width=3
    )
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 15, y + h - 1), radius=6, fill=accent
    )
    centered_text(draw, (x + 40, y, w - 75, h), title, FONT_NODE, "#162033")


def decision_node(draw, center, size, title):
    cx, cy = center
    w, h = size
    points = [
        (cx, cy - h / 2),
        (cx + w / 2, cy),
        (cx, cy + h / 2),
        (cx - w / 2, cy),
    ]
    draw.polygon(points, fill="#fff7dc", outline="#c49324")
    draw.line(points + [points[0]], fill="#c49324", width=4, joint="curve")
    centered_text(
        draw,
        (cx - w * 0.31, cy - h * 0.25, w * 0.62, h * 0.50),
        title,
        FONT_NODE,
        "#4b3811",
    )


def arrow(draw, points, color="#657184", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 17
    wing_a = (
        x2 + size * math.cos(angle + 2.55),
        y2 + size * math.sin(angle + 2.55),
    )
    wing_b = (
        x2 + size * math.cos(angle - 2.55),
        y2 + size * math.sin(angle - 2.55),
    )
    draw.polygon([(x2, y2), wing_a, wing_b], fill=color)


def branch_label(draw, x, y, text):
    bounds = draw.textbbox((0, 0), text, font=FONT_BRANCH)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    draw.rounded_rectangle(
        (x - 12, y - 7, x + w + 12, y + h + 8),
        radius=8,
        fill="#ffffff",
        outline="#cbd3df",
        width=2,
    )
    draw.text((x, y), text, font=FONT_BRANCH, fill="#374151")


def wrapped_lines(draw, text, font, max_width):
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


def callout(draw, box, title, body, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=10,
        fill="#ffffff", outline="#cbd3df", width=2
    )
    draw.rectangle((x, y, x + 8, y + h), fill=accent)
    draw.text((x + 28, y + 15), title, font=FONT_CALLOUT_TITLE, fill="#172033")
    cursor_y = y + 51
    for line in wrapped_lines(draw, body, FONT_CALLOUT_BODY, w - 55):
        draw.text((x + 28, cursor_y), line, font=FONT_CALLOUT_BODY, fill="#4b5565")
        cursor_y += 28


def annotation_link(draw, start, end):
    mid_x = 1815
    draw.line([start, (mid_x, start[1]), (mid_x, end[1]), end],
              fill="#aab4c2", width=3)
    draw.ellipse((start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5),
                 fill="#778397")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fa")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "自適應 RT 選幀與品質驗證流程", font=FONT_TITLE, fill="#111827")
draw.text((92, 125), "精簡版｜Baseline >= 20 mm", font=FONT_SUBTITLE, fill="#526071")
draw.line((90, 180, WIDTH - 90, 180), fill="#c8d0db", width=3)
draw.text((1920, 92), "與原算法差異／加速原因", font=FONT_NODE, fill="#263247")
draw.text((1922, 137), "原：原算法　新：目前流程", font=FONT_NOTE, fill="#6b7688")
draw.line((1845, 205, 1845, HEIGHT - 80), fill="#d5dbe4", width=3)

boxes = {
    "input": (300, 225, 1200, 125),
    "aruco": (300, 425, 1200, 125),
    "candidate": (300, 625, 1200, 125),
    "sift": (300, 825, 1200, 125),
    "refine": (300, 1025, 1200, 125),
    "reject": (80, 1425, 580, 135),
    "output": (820, 1425, 900, 135),
    "record": (400, 1660, 1000, 125),
}

process_node(
    draw, boxes["input"], "輸入影片、相機參數與跨段抽幀",
    "#e8f1ff", "#6d9ee8", "#2563eb"
)
process_node(
    draw, boxes["aruco"], "平行 ArUco 偵測與姿態初估",
    "#eef7e9", "#7cac67", "#3c8c37"
)
process_node(
    draw, boxes["candidate"], "幀對初篩、Baseline 檢查與排序",
    "#f2eefe", "#9478d3", "#6d45b8"
)
process_node(
    draw, boxes["sift"], "最佳候選 SIFT 驗證（必要時改用次佳）",
    "#fff0e5", "#d9955c", "#c05a18"
)
process_node(
    draw, boxes["refine"], "RT 聯合精修與 Holdout 驗證",
    "#f0ecfa", "#8f7abc", "#67469b"
)

decision_node(draw, (900, 1285), (900, 165), "最終品質通過？")

process_node(
    draw, boxes["reject"], "拒絕／標記不可靠",
    "#edf1f6", "#8491a3", "#536174"
)
process_node(
    draw, boxes["output"], "輸出 Frame、RT 與 Baseline",
    "#e8f5ea", "#70aa77", "#2f7d3c"
)
process_node(
    draw, boxes["record"], "CSV／JSON 品質與耗時紀錄",
    "#e9f2ff", "#6f98cf", "#315f9b"
)

callouts = {
    "input": (1900, 220, 920, 142),
    "aruco": (1900, 410, 920, 142),
    "candidate": (1900, 600, 920, 142),
    "sift": (1900, 790, 920, 142),
    "refine": (1900, 980, 920, 142),
    "decision": (1900, 1165, 920, 142),
    "reject": (1900, 1335, 920, 132),
    "output": (1900, 1480, 920, 132),
    "record": (1900, 1650, 920, 142),
}

callout(draw, callouts["input"], "抽幀範圍",
        "原：10→50 幀逐級擴張｜新：固定跨段 5 幀\n加速：大幅減少影片解碼與後續分析量", "#2563eb")
callout(draw, callouts["aruco"], "ArUco 分析",
        "原：前處理後再平行偵測｜新：解碼、CLAHE、偵測整段並行\n加速：CPU 工作重疊，且只處理 5 張候選幀", "#3c8c37")
callout(draw, callouts["candidate"], "候選幀對",
        "原：候選隨抽樣階段快速增加｜新：22 mm 先篩，再以廉價指標排序\n加速：昂貴步驟只保留最多 2 組候選", "#6d45b8")
callout(draw, callouts["sift"], "SIFT 驗證",
        "原：Top 3 與額外候選皆可能執行｜新：先算第一名，必要才算第二名\n加速：多數影片可少做一次以上 SIFT", "#c05a18")
callout(draw, callouts["refine"], "RT 精修",
        "原：最佳與多個次佳候選分別精修｜新：只精修通過前篩的少量候選\n加速：降低 least_squares 最佳化次數", "#67469b")
callout(draw, callouts["decision"], "品質判定",
        "原：失敗後擴大抽樣並重新搜尋｜新：單輪後以硬門檻驗證\n加速：不再重複 ArUco、配對與 SIFT", "#c49324")
callout(draw, callouts["reject"], "不可靠結果",
        "原：搜尋到上限後仍可能降級採用｜新：未通過即停止\n加速：避免不合格結果繼續進入後續處理", "#536174")
callout(draw, callouts["output"], "結果輸出",
        "新：增加 Baseline >= 20 mm 與 reliable 旗標\n速度：不直接加速，作用是確保輸出可用", "#2f7d3c")
callout(draw, callouts["record"], "品質紀錄",
        "新：增加 baseline、視差與預估深度誤差\n速度：沿用既有中間量，額外成本極低", "#315f9b")

annotation_link(draw, (1500, 287), (1900, 291))
annotation_link(draw, (1500, 487), (1900, 481))
annotation_link(draw, (1500, 687), (1900, 671))
annotation_link(draw, (1500, 887), (1900, 861))
annotation_link(draw, (1500, 1087), (1900, 1051))
annotation_link(draw, (1350, 1285), (1900, 1236))
draw.line([(660, 1425), (660, 1385), (1815, 1385), (1815, 1401), (1900, 1401)],
          fill="#aab4c2", width=3)
draw.ellipse((655, 1420, 665, 1430), fill="#778397")
annotation_link(draw, (1720, 1492), (1900, 1546))
annotation_link(draw, (1400, 1722), (1900, 1721))

arrow(draw, [(900, 350), (900, 425)])
arrow(draw, [(900, 550), (900, 625)])
arrow(draw, [(900, 750), (900, 825)])
arrow(draw, [(900, 950), (900, 1025)])
arrow(draw, [(900, 1150), (900, 1202)])

arrow(draw, [(450, 1285), (370, 1285), (370, 1425)])
branch_label(draw, 390, 1235, "否")
arrow(draw, [(1350, 1285), (1270, 1285), (1270, 1425)])
branch_label(draw, 1290, 1235, "是")
arrow(draw, [(1270, 1560), (1270, 1605), (900, 1605), (900, 1660)])

draw.text(
    (90, HEIGHT - 55),
    "關鍵門檻：Baseline >= 20 mm；品質未通過時不輸出可信 RT。",
    font=FONT_NOTE,
    fill="#687386",
)

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
