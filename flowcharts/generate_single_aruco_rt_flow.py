from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 4200
OUTPUT = Path(__file__).with_name("single_aruco_rt_flow.png")


def load_font(name, size):
    for path in (
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font("seguisb.ttf", 46)
FONT_SUBTITLE = load_font("segoeui.ttf", 23)
FONT_NODE_TITLE = load_font("seguisb.ttf", 29)
FONT_BODY = load_font("segoeui.ttf", 22)
FONT_LABEL = load_font("seguisb.ttf", 20)
FONT_FOOTER = load_font("segoeui.ttf", 19)


def text_width(draw, text, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0]


def wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if text_width(draw, trial, font) <= max_width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
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
    title_height = max(38, len(title_lines) * 35)
    centered_lines(
        draw, title_lines, (x + 35, y + 19, w - 70, title_height),
        FONT_NODE_TITLE, "#162033", spacing=3)
    body_lines = wrap_text(draw, body, FONT_BODY, w - 90)
    centered_lines(
        draw, body_lines,
        (x + 45, y + 28 + title_height, w - 90, h - title_height - 48),
        FONT_BODY, "#354052", spacing=7)


def arrow(draw, points, color="#5b6575", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
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
    draw.rounded_rectangle(
        (x - 10, y - 5, x + bounds[2] + 10, y + bounds[3] + 5),
        radius=7, fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_LABEL, fill="#384152")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
draw = ImageDraw.Draw(image)

draw.text((90, 55), "Single-ArUco RT Estimation Flow", font=FONT_TITLE, fill="#111827")
draw.text(
    (92, 115),
    "Current Zebra / Circle path when exactly one marker ID is shared across both video segments",
    font=FONT_SUBTITLE,
    fill="#4b5563",
)
draw.line((90, 165, WIDTH - 90, 165), fill="#c7cfdb", width=3)

boxes = {
    "input": (300, 220, 1200, 170),
    "sample": (300, 480, 1200, 220),
    "ippe": (300, 800, 1200, 240),
    "pairs": (300, 1140, 1200, 260),
    "preselect": (300, 1500, 1200, 230),
    "feature": (220, 1830, 1360, 340),
    "geometry": (220, 2270, 1360, 270),
    "decision": (550, 2640, 700, 150),
    "fallback": (80, 2910, 760, 250),
    "joint": (960, 2910, 760, 330),
    "validation": (220, 3340, 1360, 300),
    "selection": (300, 3740, 1200, 230),
}

arrow(draw, [(900, 390), (900, 480)])
arrow(draw, [(900, 700), (900, 800)])
arrow(draw, [(900, 1040), (900, 1140)])
arrow(draw, [(900, 1400), (900, 1500)])
arrow(draw, [(900, 1730), (900, 1830)])
arrow(draw, [(900, 2170), (900, 2270)])
arrow(draw, [(900, 2540), (900, 2640)])
arrow(draw, [(900, 2790), (900, 2850), (460, 2850), (460, 2910)])
arrow(draw, [(900, 2790), (900, 2850), (1340, 2850), (1340, 2910)])
arrow(draw, [(460, 3160), (460, 3290), (900, 3290), (900, 3340)])
arrow(draw, [(1340, 3240), (1340, 3290), (900, 3290), (900, 3340)])
arrow(draw, [(900, 3640), (900, 3740)])

branch_label(draw, 285, 2822, "Feature geometry rejected")
branch_label(draw, 1220, 2822, "Feature geometry accepted")

node(
    draw, boxes["input"], "Inputs and coordinate convention",
    "Video + intrinsics/distortion + known marker size. Final transform maps UI-left Frame B to UI-right Frame A: X_right = R X_left + t.",
    "#e8f1ff", "#6d9ee8", "#2563eb")
node(
    draw, boxes["sample"], "Progressive sampling and marker detection",
    "Uniformly sample up to 10 -> 20 -> 30 -> 40 -> 50 frames from each segment. Detect the shared marker and refine its four corners with cornerSubPix.",
    "#e7f7f3", "#58a999", "#0f766e")
node(
    draw, boxes["ippe"], "Keep both IPPE pose branches per frame",
    "Run SOLVEPNP_IPPE_SQUARE on the single shared marker. A planar square can produce two plausible poses, so both branches are retained instead of trusting the first PnP solution.",
    "#f2eefe", "#9478d3", "#6d45b8")
node(
    draw, boxes["pairs"], "Enumerate every frame-pair and branch combination",
    "Each start/end frame pair can yield up to four IPPE combinations. For each: R = R_start R_end^T; t = t_start - R t_end; baseline = ||t||. Reject baselines outside 8-220 mm.",
    "#fff0e5", "#d9955c", "#c05a18")
node(
    draw, boxes["preselect"], "Pre-rank and admit up to 12 diverse frame pairs",
    "Score self-PnP reprojection, baseline near 45 mm, sharpness and marker coverage. Keep all IPPE branches of an admitted pair; limit repeated start/end frames for diversity.",
    "#fff7df", "#cfaa54", "#9a6b13")
node(
    draw, boxes["feature"], "Independent wound/background Feature geometry",
    "Mask the marker; SIFT <= 2000; KNN ratio 0.75; mutual best; 6x4 spatial balance. Use Essential RANSAC 0.75 px and recoverPose. Require >= 25 inliers, >= 30%, spatial coverage, parallax and no planar degeneracy.",
    "#e8f6fb", "#62a9c4", "#147b9d")
node(
    draw, boxes["geometry"], "Use Feature geometry to judge the IPPE combinations",
    "For every ArUco branch RT, evaluate bidirectional marker transfer, Feature epipolar residual and rotation agreement with R_E. Re-rank pairs and branches; expand sampling unless a strong solution is found.",
    "#fdebec", "#d98287", "#b4232f")
node(
    draw, boxes["decision"], "Is Feature geometry reliable?",
    "Quality gates decide whether Feature may influence RT.",
    "#fff6d9", "#d6aa42", "#b7791f")
node(
    draw, boxes["fallback"], "Marker-only refinement fallback",
    "Optimize the bidirectional marker transfer residual only. The RT can still be returned, but without final Feature validation it is not considered reliable.",
    "#edf1f6", "#8491a3", "#536174")
node(
    draw, boxes["joint"], "Metric Marker + Feature joint refinement",
    "Try marker RT and Feature R/t-direction initializations. Scale Feature t from the known marker edge. Optimize all 6 DoF using bidirectional marker pixels + pseudo-Huber Sampson residuals, with marker weights 12, 4, 1 and 0.25.",
    "#f0ecfa", "#8f7abc", "#67469b")
node(
    draw, boxes["validation"], "Final hard validation",
    "Marker: left->right and right->left RMS <= 1.5 px; point max <= 2 px; also stay near the marker-only floor. Feature: >= 25 final inliers, >= 30%, p90 <= 1.25 px, holdout median <= 1.5 px.",
    "#e8f5ea", "#70aa77", "#2f7d3c")
node(
    draw, boxes["selection"], "Select final primary/secondary pair and output RT",
    "Keep pairs passing both hard gates, stay within 0.25 px of best marker RMS, then rank by Feature p90, median and inlier support. Output R, metric t, baseline, reliability and SIFT diagnostics.",
    "#e9f2ff", "#6f98cf", "#315f9b")

draw.text(
    (90, 4070),
    "Key point: single-marker self-PnP reprojection cannot reliably resolve the IPPE mirror ambiguity; independent Feature geometry is the main discriminator.",
    font=FONT_FOOTER,
    fill="#4b5563",
)
draw.text(
    (90, 4110),
    "Metric scale remains anchored by the known ArUco size; SIFT / Essential alone supplies no absolute baseline magnitude.",
    font=FONT_FOOTER,
    fill="#667085",
)

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)

