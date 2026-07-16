from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 3600
OUTPUT = Path(__file__).with_name("rt_calculation_flow.png")


def load_font(name, size):
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font("seguisb.ttf", 46)
FONT_SUBTITLE = load_font("segoeui.ttf", 23)
FONT_NODE_TITLE = load_font("seguisb.ttf", 28)
FONT_BODY = load_font("segoeui.ttf", 22)
FONT_LABEL = load_font("seguisb.ttf", 20)
FONT_FOOTER = load_font("segoeui.ttf", 19)


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if text_width(draw, trial, font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_centered_lines(draw, lines, box, font, fill, spacing=8, top_offset=0):
    x, y, w, h = box
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        heights.append(bbox[3] - bbox[1])
    total = sum(heights) + spacing * max(0, len(lines) - 1)
    cursor_y = y + top_offset + max(0, (h - top_offset - total) / 2)
    for line, line_h in zip(lines, heights):
        line_w = text_width(draw, line, font)
        draw.text((x + (w - line_w) / 2, cursor_y), line, font=font, fill=fill)
        cursor_y += line_h + spacing


def draw_node(draw, box, title, body, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=18, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 13, y + h - 1), radius=6, fill=accent)
    title_lines = wrap_text(draw, title, FONT_NODE_TITLE, w - 70)
    title_height = len(title_lines) * 34
    title_box = (x + 30, y + 20, w - 60, title_height)
    draw_centered_lines(draw, title_lines, title_box, FONT_NODE_TITLE, "#162033", spacing=3)
    if body:
        body_lines = wrap_text(draw, body, FONT_BODY, w - 80)
        body_box = (x + 40, y + 28 + title_height, w - 80, h - 48 - title_height)
        draw_centered_lines(draw, body_lines, body_box, FONT_BODY, "#354052", spacing=7)


def draw_arrow(draw, points, color="#5b6575", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    if len(points) < 2:
        return
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    wings = []
    for delta in (2.55, -2.55):
        wings.append((x2 + size * math.cos(angle + delta), y2 + size * math.sin(angle + delta)))
    draw.polygon([(x2, y2), wings[0], wings[1]], fill=color)


def label(draw, xy, text):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
    pad_x, pad_y = 10, 5
    draw.rounded_rectangle(
        (x - pad_x, y - pad_y, x + bbox[2] + pad_x, y + bbox[3] + pad_y),
        radius=7, fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_LABEL, fill="#384152")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "Current RT Estimation Flow", font=FONT_TITLE, fill="#111827")
draw.text(
    (92, 115),
    "Shared by Zebra and Circle | final transform: UI left (Frame B) -> UI right (Frame A)",
    font=FONT_SUBTITLE,
    fill="#4b5563",
)
draw.line((90, 165, WIDTH - 90, 165), fill="#c7cfdb", width=3)

boxes = {
    "input": (300, 220, 1200, 170),
    "sample": (300, 470, 1200, 220),
    "mode": (600, 780, 600, 130),
    "single": (80, 1020, 770, 250),
    "multi": (950, 1020, 770, 250),
    "candidates": (300, 1370, 1200, 250),
    "marker": (80, 1740, 770, 330),
    "feature": (950, 1740, 770, 330),
    "rerank": (300, 2180, 1200, 220),
    "shortlist": (300, 2490, 1200, 170),
    "joint": (220, 2760, 1360, 300),
    "validate": (220, 3150, 1360, 250),
}

draw_arrow(draw, [(900, 390), (900, 470)])
draw_arrow(draw, [(900, 690), (900, 780)])
draw_arrow(draw, [(900, 910), (900, 960), (465, 960), (465, 1020)])
draw_arrow(draw, [(900, 910), (900, 960), (1335, 960), (1335, 1020)])
draw_arrow(draw, [(465, 1270), (465, 1320), (900, 1320), (900, 1370)])
draw_arrow(draw, [(1335, 1270), (1335, 1320), (900, 1320), (900, 1370)])
draw_arrow(draw, [(900, 1620), (900, 1680), (465, 1680), (465, 1740)])
draw_arrow(draw, [(900, 1620), (900, 1680), (1335, 1680), (1335, 1740)])
draw_arrow(draw, [(465, 2070), (465, 2130), (900, 2130), (900, 2180)])
draw_arrow(draw, [(1335, 2070), (1335, 2130), (900, 2130), (900, 2180)])
draw_arrow(draw, [(900, 2400), (900, 2490)])
draw_arrow(draw, [(900, 2660), (900, 2760)])
draw_arrow(draw, [(900, 3060), (900, 3150)])

label(draw, (265, 934), "1 shared marker")
label(draw, (1270, 934), ">= 2 shared markers")

draw_node(
    draw, boxes["input"], "Inputs",
    "Video frames + camera intrinsics/distortion + known ArUco marker size",
    "#e8f1ff", "#6d9ee8", "#2563eb")
draw_node(
    draw, boxes["sample"], "Progressive frame sampling and detection",
    "Uniformly sample up to 10 -> 20 -> 30 -> 40 -> 50 frames from each segment. "
    "Detect DICT_4X4_100 markers and refine corners to subpixel precision.",
    "#e7f7f3", "#58a999", "#0f766e")
draw_node(
    draw, boxes["mode"], "Shared-marker mode",
    "Select the smallest shared ID as world origin.",
    "#fff6d9", "#d6aa42", "#b7791f")
draw_node(
    draw, boxes["single"], "Single-marker pose branches",
    "IPPE_SQUARE returns both planar pose branches for every frame. Keep all start/end branch combinations; Feature geometry later resolves ambiguity.",
    "#f2eefe", "#9478d3", "#6d45b8")
draw_node(
    draw, boxes["multi"], "Multi-marker joint pose",
    "Build a rigid marker map relative to the reference marker. Combine mapped 3D corners in each frame and solve one iterative joint PnP pose.",
    "#eaf7e9", "#6eaf69", "#347a36")
draw_node(
    draw, boxes["candidates"], "Enumerate metric RT candidates",
    "R = R_start R_end^T; t = t_start - R t_end; baseline = ||t||. Keep 8-220 mm. "
    "Pre-rank by self-PnP reprojection, baseline near 45 mm, sharpness, marker coverage and marker count.",
    "#fff0e5", "#d9955c", "#c05a18")
draw_node(
    draw, boxes["marker"], "ArUco cross-view constraint",
    "Choose per-view IPPE metric marker models. Measure left->right and right->left corner transfer. Candidate gate: each-direction RMS <= 3 px and max <= 6 px.",
    "#fdebec", "#d98287", "#b4232f")
draw_node(
    draw, boxes["feature"], "Independent SIFT / Essential geometry",
    "Mask markers; SIFT <= 2000; ratio 0.75; mutual match; 6x4 spatial balance. E-RANSAC 0.75 px + recoverPose. Check support, coverage, parallax and planar degeneracy.",
    "#e8f6fb", "#62a9c4", "#147b9d")
draw_node(
    draw, boxes["rerank"], "Top-12 diverse pair reranking",
    "Prefer marker-gate pass, then valid Feature geometry, then combined epipolar/rotation/quality score. "
    "Stop early only for strong Feature geometry with E-model median < 0.8 px; otherwise expand sampling.",
    "#edf1f6", "#8491a3", "#536174")
draw_node(
    draw, boxes["shortlist"], "Shortlist primary + up to 5 secondary pairs",
    "Secondary pairs share the UI-left/end frame and use different UI-right/start frames; each must pass marker and Feature prechecks.",
    "#fff7df", "#cfaa54", "#9a6b13")
draw_node(
    draw, boxes["joint"], "6-DoF marker + Feature nonlinear refinement",
    "Start from marker RT and, when valid, Feature R with t direction scaled by known marker edges. "
    "Least-squares residual = bidirectional marker pixels + robust pseudo-Huber Sampson Feature residuals + baseline bounds. "
    "Search marker weights 12, 4, 1 and 0.25; reserve every fifth inlier as holdout when support >= 35.",
    "#f0ecfa", "#8f7abc", "#67469b")
draw_node(
    draw, boxes["validate"], "Hard validation, final pair selection and output",
    "Marker: each direction RMS <= 1.5 px, point max <= 2 px, also close to marker-only floor. "
    "Feature: >= 25 inliers, >= 30%, p90 <= 1.25 px, holdout median <= 1.5 px. "
    "Choose near-best marker solutions by Feature p90/median/support; output R, metric t, baseline and reliability diagnostics.",
    "#e8f5ea", "#70aa77", "#2f7d3c")

draw.text(
    (90, 3480),
    "Metric scale is always anchored by the known ArUco size; Essential geometry alone never supplies baseline magnitude.",
    font=FONT_FOOTER,
    fill="#4b5563",
)
draw.text(
    (90, 3520),
    "Source: Algorithm/video_pose_analysis.py (current workspace)",
    font=FONT_FOOTER,
    fill="#667085",
)

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)

