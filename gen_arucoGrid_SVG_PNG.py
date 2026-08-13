import cv2
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================
# 使用者設定
# ============================================================

# A3 橫式尺寸
PAGE_WIDTH_MM = 420.0
PAGE_HEIGHT_MM = 297.0

# 外圍白邊
MARGIN_MM = 25.0

# ArUco dictionary
ARUCO_DICT = cv2.aruco.DICT_4X4_100

# Grid 設定
GRID_ROWS = 7
GRID_COLS = 10
GRID_START_ID = 20              # 20 ~ 89
GRID_MARKER_SIZE_MM = 31.0
GRID_GAP_MM = 5.0

# ArUco black border thickness
BORDER_BITS = 1

# PNG 解析度
DPI = 600

# 輸出檔名
OUTPUT_SVG = "aruco_grid_a3_7x10.svg"
OUTPUT_PNG = "aruco_grid_a3_7x10.png"


# ============================================================
# 工具函式
# ============================================================

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def generate_marker_cells(dictionary, marker_id, border_bits=1):
    """
    產生 ArUco marker 的 cell 級黑白圖。
    """
    marker_size = dictionary.markerSize
    total_cells = marker_size + 2 * border_bits

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            total_cells,
            borderBits=border_bits
        )
    else:
        marker = np.zeros((total_cells, total_cells), dtype=np.uint8)
        cv2.aruco.drawMarker(
            dictionary,
            marker_id,
            total_cells,
            marker,
            border_bits
        )

    return marker


def compute_grid_size_mm():
    grid_w = GRID_COLS * GRID_MARKER_SIZE_MM + (GRID_COLS - 1) * GRID_GAP_MM
    grid_h = GRID_ROWS * GRID_MARKER_SIZE_MM + (GRID_ROWS - 1) * GRID_GAP_MM
    return grid_w, grid_h


def compute_layout():
    usable_w = PAGE_WIDTH_MM - 2 * MARGIN_MM
    usable_h = PAGE_HEIGHT_MM - 2 * MARGIN_MM

    grid_w, grid_h = compute_grid_size_mm()

    if grid_w > usable_w:
        raise ValueError(f"Grid 寬度 {grid_w:.2f} mm 超過可用寬度 {usable_w:.2f} mm")
    if grid_h > usable_h:
        raise ValueError(f"Grid 高度 {grid_h:.2f} mm 超過可用高度 {usable_h:.2f} mm")

    # 置中
    grid_x = MARGIN_MM + (usable_w - grid_w) / 2.0
    grid_y = MARGIN_MM + (usable_h - grid_h) / 2.0

    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "grid_x": grid_x,
        "grid_y": grid_y
    }


def draw_marker_svg(svg_lines, dictionary, marker_id, x_mm, y_mm, size_mm, border_bits=1):
    marker = generate_marker_cells(dictionary, marker_id, border_bits)
    num_cells = marker.shape[0]
    cell_size_mm = size_mm / num_cells

    # 白底
    svg_lines.append(
        f'<rect x="{x_mm:.8f}" y="{y_mm:.8f}" '
        f'width="{size_mm:.8f}" height="{size_mm:.8f}" fill="white"/>\n'
    )

    # 畫黑格
    for r in range(num_cells):
        for c in range(num_cells):
            if marker[r, c] == 0:
                rect_x = x_mm + c * cell_size_mm
                rect_y = y_mm + r * cell_size_mm
                svg_lines.append(
                    f'<rect x="{rect_x:.8f}" y="{rect_y:.8f}" '
                    f'width="{cell_size_mm:.8f}" height="{cell_size_mm:.8f}" '
                    f'fill="black"/>\n'
                )


def paste_marker_png(canvas, dictionary, marker_id, x_px, y_px, size_px, border_bits=1):
    marker_cells = generate_marker_cells(dictionary, marker_id, border_bits)
    marker_img = cv2.resize(
        marker_cells,
        (size_px, size_px),
        interpolation=cv2.INTER_NEAREST
    )
    canvas[y_px:y_px + size_px, x_px:x_px + size_px] = marker_img


# ============================================================
# SVG
# ============================================================

def generate_svg(output_path):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    total_markers = GRID_ROWS * GRID_COLS
    end_id = GRID_START_ID + total_markers - 1

    dictionary_size = dictionary.bytesList.shape[0]
    if end_id >= dictionary_size:
        raise ValueError(
            f"需要 ID {GRID_START_ID} ~ {end_id}，但 dictionary 只有 0 ~ {dictionary_size - 1}"
        )

    layout = compute_layout()

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAGE_WIDTH_MM}mm" height="{PAGE_HEIGHT_MM}mm" '
        f'viewBox="0 0 {PAGE_WIDTH_MM} {PAGE_HEIGHT_MM}">\n'
    )

    # 整張白底
    svg.append(
        f'<rect x="0" y="0" width="{PAGE_WIDTH_MM}" height="{PAGE_HEIGHT_MM}" fill="white"/>\n'
    )

    # 畫 7x10 grid
    marker_id = GRID_START_ID
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = layout["grid_x"] + c * (GRID_MARKER_SIZE_MM + GRID_GAP_MM)
            y = layout["grid_y"] + r * (GRID_MARKER_SIZE_MM + GRID_GAP_MM)

            draw_marker_svg(
                svg,
                dictionary,
                marker_id,
                x,
                y,
                GRID_MARKER_SIZE_MM,
                BORDER_BITS
            )
            marker_id += 1

    svg.append("</svg>\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(svg)

    print(f"SVG saved: {output_path}")


# ============================================================
# PNG
# ============================================================

def generate_png(output_path, dpi=600):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    layout = compute_layout()

    page_w_px = mm_to_px(PAGE_WIDTH_MM, dpi)
    page_h_px = mm_to_px(PAGE_HEIGHT_MM, dpi)

    canvas = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)

    marker_px = mm_to_px(GRID_MARKER_SIZE_MM, dpi)
    gap_px = mm_to_px(GRID_GAP_MM, dpi)
    grid_x_px = mm_to_px(layout["grid_x"], dpi)
    grid_y_px = mm_to_px(layout["grid_y"], dpi)

    marker_id = GRID_START_ID
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x_px = grid_x_px + c * (marker_px + gap_px)
            y_px = grid_y_px + r * (marker_px + gap_px)

            paste_marker_png(
                canvas,
                dictionary,
                marker_id,
                x_px,
                y_px,
                marker_px,
                BORDER_BITS
            )
            marker_id += 1

    if PIL_AVAILABLE:
        img = Image.fromarray(canvas)
        img.save(output_path, dpi=(dpi, dpi))
    else:
        cv2.imwrite(output_path, canvas)

    print(f"PNG saved: {output_path}")
    print(f"PNG size : {page_w_px} x {page_h_px} px")
    print(f"DPI      : {dpi}")


# ============================================================
# 主程式
# ============================================================

def main():
    grid_w, grid_h = compute_grid_size_mm()
    layout = compute_layout()
    total_markers = GRID_ROWS * GRID_COLS
    end_id = GRID_START_ID + total_markers - 1

    print("================================================")
    print("A3 ArUco 7x10 Grid Generator")
    print("================================================")
    print(f"Page size         : {PAGE_WIDTH_MM} x {PAGE_HEIGHT_MM} mm (A3 landscape)")
    print(f"Margin            : {MARGIN_MM} mm")
    print(f"Dictionary        : DICT_4X4_100")
    print(f"Grid              : {GRID_ROWS} x {GRID_COLS}")
    print(f"Marker size       : {GRID_MARKER_SIZE_MM} mm")
    print(f"Marker gap        : {GRID_GAP_MM} mm")
    print(f"Grid size         : {grid_w} x {grid_h} mm")
    print(f"Grid start/end ID : {GRID_START_ID} ~ {end_id}")
    print(f"Grid origin       : ({layout['grid_x']:.2f}, {layout['grid_y']:.2f}) mm")
    print(f"Output SVG        : {OUTPUT_SVG}")
    print(f"Output PNG        : {OUTPUT_PNG}")
    print(f"PNG DPI           : {DPI}")
    print("================================================")

    generate_svg(OUTPUT_SVG)
    generate_png(OUTPUT_PNG, DPI)

    print("Done.")


if __name__ == "__main__":
    main()