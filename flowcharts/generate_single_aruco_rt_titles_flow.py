from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 1740
OUTPUT = Path(__file__).with_name("single_aruco_rt_titles_flow.png")


def load_font(name, size):
    for path in (
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font("seguisb.ttf", 44)
FONT_SUBTITLE = load_font("segoeui.ttf", 22)
FONT_NODE = load_font("seguisb.ttf", 28)
FONT_LABEL = load_font("seguisb.ttf", 18)


def text_size(draw, text, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def node(draw, box, title, fill, border, accent):
    x, y, w, h = box
    draw.rounded_rectangle(
        (x, y, x + w, y + h), radius=17, fill=fill, outline=border, width=3)
    draw.rounded_rectangle(
        (x + 1, y + 1, x + 13, y + h - 1), radius=6, fill=accent)
    text_w, text_h = text_size(draw, title, FONT_NODE)
    draw.text(
        (x + (w - text_w) / 2, y + (h - text_h) / 2 - 3),
        title,
        font=FONT_NODE,
        fill="#162033",
    )


def arrow(draw, points, color="#5b6575", width=5):
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 15
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
    text_w, text_h = text_size(draw, text, FONT_LABEL)
    draw.rounded_rectangle(
        (x - 9, y - 4, x + text_w + 9, y + text_h + 5),
        radius=6, fill="#ffffff", outline="#cbd3df", width=2)
    draw.text((x, y), text, font=FONT_LABEL, fill="#384152")


image = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
draw = ImageDraw.Draw(image)

draw.text((80, 45), "Single-ArUco RT Flow", font=FONT_TITLE, fill="#111827")
draw.text(
    (82, 103),
    "Title-only view of the current Zebra / Circle pipeline",
    font=FONT_SUBTITLE,
    fill="#4b5563",
)
draw.line((80, 145, WIDTH - 80, 145), fill="#c7cfdb", width=3)

boxes = {
    "frame_select": (210, 190, 1180, 120),
    "feature": (260, 400, 1080, 110),
    "judge": (170, 600, 1260, 120),
    "decision": (460, 810, 680, 110),
    "fallback": (60, 1030, 690, 120),
    "joint": (850, 1030, 690, 120),
    "validation": (260, 1280, 1080, 110),
    "selection": (260, 1490, 1080, 110),
}

arrow(draw, [(800, 310), (800, 400)])
arrow(draw, [(800, 510), (800, 600)])
arrow(draw, [(800, 720), (800, 810)])
arrow(draw, [(800, 920), (800, 970), (405, 970), (405, 1030)])
arrow(draw, [(800, 920), (800, 970), (1195, 970), (1195, 1030)])
arrow(draw, [(405, 1150), (405, 1220), (800, 1220), (800, 1280)])
arrow(draw, [(1195, 1150), (1195, 1220), (800, 1220), (800, 1280)])
arrow(draw, [(800, 1390), (800, 1490)])

branch_label(draw, 230, 940, "Rejected")
branch_label(draw, 1235, 940, "Accepted")

node(draw, boxes["frame_select"], "Select candidate frame pairs with single-ArUco IPPE", "#fff0e5", "#d9955c", "#c05a18")
node(draw, boxes["feature"], "Independent wound/background Feature geometry", "#e8f6fb", "#62a9c4", "#147b9d")
node(draw, boxes["judge"], "Rank frame pairs and IPPE branches with Marker + Feature geometry", "#fdebec", "#d98287", "#b4232f")
node(draw, boxes["decision"], "Is Feature geometry reliable?", "#fff6d9", "#d6aa42", "#b7791f")
node(draw, boxes["fallback"], "Marker-only refinement fallback", "#edf1f6", "#8491a3", "#536174")
node(draw, boxes["joint"], "Metric Marker + Feature joint refinement", "#f0ecfa", "#8f7abc", "#67469b")
node(draw, boxes["validation"], "Final hard validation", "#e8f5ea", "#70aa77", "#2f7d3c")
node(draw, boxes["selection"], "Select final pair and output RT", "#e9f2ff", "#6f98cf", "#315f9b")

image.save(OUTPUT, format="PNG", optimize=True, dpi=(144, 144))
print(OUTPUT)
