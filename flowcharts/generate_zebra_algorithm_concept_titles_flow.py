from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1800
HEIGHT = 3550
OUTPUT = Path(__file__).with_name("zebra_algorithm_concept_titles_flow.png")


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
FONT_NODE = load_font("seguisb.ttf", 29)
FONT_LABEL = load_font("seguisb.ttf", 20)


def text_width(draw, text, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0]


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

draw.text((90, 55), "Zebra Point-Depth Estimation", font=FONT_TITLE, fill="#111827")
draw.text((92, 115), "Algorithm-concept flow | title-only view", font=FONT_SUBTITLE, fill="#4b5563")
draw.line((90, 165, WIDTH - 90, 165), fill="#c7cfdb", width=3)

boxes = {
    "target": (300, 220, 1200, 150),
    "observations": (300, 460, 1200, 150),
    "geometry": (430, 700, 940, 150),
    "marker": (80, 990, 760, 170),
    "feature": (960, 990, 760, 170),
    "consistency": (220, 1270, 1360, 170),
    "refinement": (300, 1550, 1200, 170),
    "validity": (550, 1830, 700, 140),
    "reject": (80, 2070, 650, 160),
    "triangulation": (300, 2290, 1200, 170),
    "plausibility": (500, 2570, 800, 140),
    "confidence": (220, 2810, 1360, 170),
    "outliers": (220, 3080, 1360, 170),
    "fusion": (300, 3350, 1200, 150),
}

arrow(draw, [(900, 370), (900, 460)])
arrow(draw, [(900, 610), (900, 700)])

arrow(draw, [(900, 850), (900, 920), (460, 920), (460, 990)])
arrow(draw, [(900, 850), (900, 920), (1340, 920), (1340, 990)])
branch_label(draw, 610, 875, "Known landmark available")
branch_label(draw, 1185, 875, "General scene point")

arrow(draw, [(460, 1160), (460, 1210), (900, 1210), (900, 1270)])
arrow(draw, [(1340, 1160), (1340, 1210), (900, 1210), (900, 1270)])
arrow(draw, [(900, 1440), (900, 1550)])
arrow(draw, [(900, 1720), (900, 1830)])

arrow(draw, [(550, 1900), (405, 1900), (405, 2070)])
branch_label(draw, 300, 1925, "Rejected")
arrow(draw, [(900, 1970), (900, 2290)])
branch_label(draw, 930, 2030, "Accepted")

arrow(draw, [(900, 2460), (900, 2570)])
arrow(draw, [(500, 2640), (405, 2640), (405, 2230)])
branch_label(draw, 285, 2590, "Implausible")
arrow(draw, [(900, 2710), (900, 2810)])
branch_label(draw, 930, 2740, "Plausible")

arrow(draw, [(900, 2980), (900, 3080)])
arrow(draw, [(900, 3250), (900, 3350)])

node(draw, boxes["target"], "Target Point Selection in the Reference View", "#e8f1ff", "#6d9ee8", "#2563eb")
node(draw, boxes["observations"], "Multi-Frame Stereo Observation Set", "#e7f7f3", "#58a999", "#0f766e")
node(draw, boxes["geometry"], "Stereo Geometry Validation", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["marker"], "Marker-Constrained Correspondence", "#f2eefe", "#9478d3", "#6d45b8")
node(draw, boxes["feature"], "Feature-Based Local Correspondence", "#e8f6fb", "#62a9c4", "#147b9d")
node(draw, boxes["consistency"], "Epipolar and Pose Consistency Filtering", "#fdebec", "#d98287", "#b4232f")
node(draw, boxes["refinement"], "Subpixel Correspondence Refinement", "#fff0e5", "#d9955c", "#c05a18")
node(draw, boxes["validity"], "Correspondence Validity Test", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["reject"], "Reject Invalid Observation", "#edf1f6", "#8491a3", "#536174")
node(draw, boxes["triangulation"], "Metric Stereo Triangulation", "#f0ecfa", "#8f7abc", "#67469b")
node(draw, boxes["plausibility"], "Physical Depth Plausibility Test", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["confidence"], "Per-Observation Confidence Estimation", "#e8f5ea", "#70aa77", "#2f7d3c")
node(draw, boxes["outliers"], "Robust Multi-View Outlier Rejection", "#e9f2ff", "#6f98cf", "#315f9b")
node(draw, boxes["fusion"], "Uncertainty-Aware 3D Fusion and Final Depth", "#e8f5ea", "#70aa77", "#2f7d3c")

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
