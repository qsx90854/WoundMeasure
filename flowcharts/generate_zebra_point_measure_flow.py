from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 4400
OUTPUT = Path(__file__).with_name("zebra_point_measure_flow.png")


def load_font(names, size):
    font_dir = Path("C:/Windows/Fonts")
    for name in names:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 46)
FONT_SUBTITLE = load_font(("msjh.ttc", "segoeui.ttf", "arial.ttf"), 23)
FONT_NODE_TITLE = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 29)
FONT_BODY = load_font(("msjh.ttc", "segoeui.ttf", "arial.ttf"), 22)
FONT_LABEL = load_font(("msjhbd.ttc", "seguisb.ttf", "arialbd.ttf"), 20)
FONT_FOOTER = load_font(("msjh.ttc", "segoeui.ttf", "arial.ttf"), 19)


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            trial = current + char
            if current and text_width(draw, trial, font) > max_width:
                lines.append(current.rstrip())
                current = char.lstrip()
            else:
                current = trial
        if current:
            lines.append(current.rstrip())
    return lines


def centered_lines(draw, lines, box, font, fill, spacing=8):
    x, y, w, h = box
    heights = []
    for line in lines:
        bounds = draw.textbbox((0, 0), line or " ", font=font)
        heights.append(bounds[3] - bounds[1])
    total_height = sum(heights) + spacing * max(0, len(lines) - 1)
    cursor_y = y + max(0, (h - total_height) / 2)
    for line, line_height in zip(lines, heights):
        cursor_x = x + (w - text_width(draw, line, font)) / 2
        draw.text((cursor_x, cursor_y), line, font=font, fill=fill)
        cursor_y += line_height + spacing


def node(draw, box, title, body, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=18, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 13, y + h - 1), radius=6, fill=accent)
    title_lines = wrap_text(draw, title, FONT_NODE_TITLE, w - 80)
    title_height = max(38, len(title_lines) * 37)
    centered_lines(
        draw, title_lines, (x + 35, y + 18, w - 70, title_height),
        FONT_NODE_TITLE, "#162033", spacing=3)
    if body:
        body_lines = wrap_text(draw, body, FONT_BODY, w - 92)
        centered_lines(
            draw, body_lines,
            (x + 46, y + 28 + title_height, w - 92, h - title_height - 48),
            FONT_BODY, "#354052", spacing=7)


def arrow(draw, points, color="#5b6575", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    if len(points) < 2:
        return
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
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
    bounds = draw.textbbox((0, 0), text, font=FONT_LABEL)
    pad_x, pad_y = 10, 5
    draw.rounded_rectangle(
        (x - pad_x, y - pad_y, x + bounds[2] + pad_x, y + bounds[3] + pad_y),
        radius=7, fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_LABEL, fill="#384152")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "Zebra 點選量測流程", font=FONT_TITLE, fill="#111827")
draw.text(
    (92, 115),
    "左圖候選點 → 右圖特徵匹配 → 3D 三角化 → 多候選深度融合（目前預設 dual_direct）",
    font=FONT_SUBTITLE,
    fill="#4b5563",
)
draw.line((90, 165, WIDTH - 90, 165), fill="#c7cfdb", width=3)

boxes = {
    "input": (300, 220, 1200, 190),
    "prepare": (260, 500, 1280, 250),
    "candidates": (300, 840, 1200, 220),
    "gate": (550, 1150, 700, 170),
    "skip": (60, 1430, 620, 190),
    "aruco": (130, 1740, 700, 260),
    "feature": (970, 1740, 700, 300),
    "refine": (260, 2150, 1280, 280),
    "match_gate": (550, 2520, 700, 160),
    "triangulate": (260, 2790, 1280, 260),
    "depth_gate": (550, 3140, 700, 160),
    "quality": (260, 3410, 1280, 280),
    "collect": (300, 3780, 1200, 190),
    "fuse": (220, 4060, 1360, 230),
}

# Main vertical flow.
arrow(draw, [(900, 410), (900, 500)])
arrow(draw, [(900, 750), (900, 840)])
arrow(draw, [(900, 1060), (900, 1150)])

# Candidate validation branches.
arrow(draw, [(550, 1235), (370, 1235), (370, 1430)])
arrow(draw, [(900, 1320), (900, 1660), (480, 1660), (480, 1740)])
arrow(draw, [(900, 1320), (900, 1660), (1320, 1660), (1320, 1740)])
branch_label(draw, 245, 1260, "無效：記錄原因")
branch_label(draw, 760, 1587, "附近有 ArUco 角點")
branch_label(draw, 1260, 1587, "一般候選點")

# Matching branches rejoin refinement.
arrow(draw, [(480, 2000), (480, 2090), (900, 2090), (900, 2150)])
arrow(draw, [(1320, 2040), (1320, 2090), (900, 2090), (900, 2150)])
arrow(draw, [(900, 2430), (900, 2520)])

# Match failure goes to skip; success continues.
arrow(draw, [(550, 2600), (370, 2600), (370, 1620)])
arrow(draw, [(900, 2680), (900, 2790)])
branch_label(draw, 275, 2550, "失敗")
branch_label(draw, 930, 2710, "成功")

# Triangulation and depth validation.
arrow(draw, [(900, 3050), (900, 3140)])
arrow(draw, [(550, 3220), (370, 3220), (370, 1620)])
arrow(draw, [(900, 3300), (900, 3410)])
branch_label(draw, 275, 3170, "Z 無效")
branch_label(draw, 930, 3330, "0 < Z ≤ 上限")

# Per-candidate result, fusion and output footer.
arrow(draw, [(900, 3690), (900, 3780)])
arrow(draw, [(900, 3970), (900, 4060)])

node(
    draw, boxes["input"], "輸入",
    "左圖點選座標 (u, v)、最佳與次佳右圖候選影格、相機內參 K、候選影格 R_rel / t_rel / F、平面與高光遮罩。",
    "#e8f1ff", "#6d9ee8", "#2563eb")
node(
    draw, boxes["prepare"], "點擊與影像前處理",
    "on_release() 接收左圖點擊，先吸附附近 ArUco 角點，再進入 do_measure()。清除舊標記、鎖定左右影像、轉灰階、依設定套用 CLAHE，並建立可跨候選影格重用的左圖特徵快取。",
    "#e7f7f3", "#58a999", "#0f766e")
node(
    draw, boxes["candidates"], "逐一處理右圖候選影格",
    "依序處理目前最佳影格與 extra candidates。預設為了速度，次佳影格停用 ECC 與 Precise；每個候選都呼叫 compute_measure()。",
    "#fff0e5", "#d9955c", "#c05a18")
node(
    draw, boxes["gate"], "Pose 與 baseline 合格？",
    "候選 pose 必須有效，且 baseline 必須位於 8–220 mm。",
    "#fff6d9", "#d6aa42", "#b7791f")
node(
    draw, boxes["skip"], "略過無效候選",
    "保留 fail_reason，這個候選不進入後續融合；流程繼續處理下一個候選影格。",
    "#edf1f6", "#8491a3", "#536174")
node(
    draw, boxes["aruco"], "ArUco 直接對應",
    "若點擊位置距離左圖 ArUco 角點小於 10 px，將左點校正到精確角點，並以相同 marker ID、相同角點序號取得右圖座標。",
    "#f2eefe", "#9478d3", "#6d45b8")
node(
    draw, boxes["feature"], "一般特徵匹配",
    "依開關執行 Grad-SIFT 或 run_improved_matching_flow()；使用梯度／局部特徵、SIFT 描述子、高光遮罩及幾何限制。若仍無匹配且 Precise 啟用，再以 find_precise_match() 搜尋。",
    "#e8f6fb", "#62a9c4", "#147b9d")
node(
    draw, boxes["refine"], "匹配點精修與幾何檢查",
    "依匹配方法與 UI 開關，選擇性執行 Epi-band 搜尋、強制極線對齊、RT／平面預測偏移限制與 ECC 亞像素精修；精修後再次檢查極線與 RT 邊界。",
    "#fdebec", "#d98287", "#b4232f")
node(
    draw, boxes["match_gate"], "取得有效右圖匹配點？",
    "若找不到匹配點或超出 RT 邊界，該候選量測失敗。",
    "#fff6d9", "#d6aa42", "#b7791f")
node(
    draw, boxes["triangulate"], "雙視圖三角化（dual_direct）",
    "triangulate_point_3d() 由 K_L、K_R、R_rel、t_rel 建立 P0 / P1，將左右像素正規化後呼叫 cv2.triangulatePoints()，取得左相機座標系中的 3D 點。",
    "#f0ecfa", "#8f7abc", "#67469b")
node(
    draw, boxes["depth_gate"], "3D 深度 Z 合理？",
    "要求 Z > 0 且不超過 MAX_DEPTH_MM。",
    "#fff6d9", "#d6aa42", "#b7791f")
node(
    draw, boxes["quality"], "單候選量測與品質指標",
    "計算相機距離 ||p3d||、相對平面距離、ArUco 世界座標與右圖重投影誤差；再由原始極線誤差、ZNCC、Masked Score 計算 confidence，輸出此候選的完整結果。",
    "#e8f5ea", "#70aa77", "#2f7d3c")
node(
    draw, boxes["collect"], "收集所有有效候選結果",
    "全部候選處理完成後，保留具有有效 d 與 p3d 的結果；最佳影格原始 3D 點另外保存在 p3d_best。",
    "#e9f2ff", "#6f98cf", "#315f9b")
node(
    draw, boxes["fuse"], "多候選深度融合與輸出",
    "先依最佳影格深度做一致性閘門；候選數 ≥ 3 時用 Median + MAD 剔除離群；最後依 (baseline / Z²)² 加權平均 3D 點。寫入量測 TXT，並更新匹配點、連線、深度、融合數與剔除原因。",
    "#e8f5ea", "#70aa77", "#2f7d3c")

draw.text(
    (90, 4330),
    "來源：depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py、Algorithm/stereo_matching.py",
    font=FONT_FOOTER,
    fill="#667085",
)

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
