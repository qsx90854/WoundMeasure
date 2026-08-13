"""Generate a print-ready A3 ChArUco calibration board PNG.

The default board is designed for A3 landscape paper and OpenCV 4.6 or newer:
12 x 8 squares, 29 mm square length, 22 mm ArUco marker length, using
DICT_4X4_100 marker IDs 20-67. Print the PNG at 100% / actual size without
fit-to-page scaling.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# A3 landscape page.
PAGE_WIDTH_MM = 420.0
PAGE_HEIGHT_MM = 297.0
MARGIN_MM = 25.0

# ChArUco board geometry. SQUARES_X/Y count squares, not internal corners.
ARUCO_DICT = cv2.aruco.DICT_4X4_100
SQUARES_X = 12
SQUARES_Y = 8
SQUARE_LENGTH_MM = 29.0
MARKER_LENGTH_MM = 22.0
MARKER_START_ID = 20
BORDER_BITS = 1
LEGACY_PATTERN = False

DPI = 600
OUTPUT_PNG = "charuco_a3_12x8.png"
OUTPUT_METADATA = "charuco_a3_12x8_pattern.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PNG))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(OUTPUT_METADATA),
        help=f"Pattern metadata JSON (default: {OUTPUT_METADATA})",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--squares-x", type=int, default=SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=SQUARES_Y)
    parser.add_argument("--square-length-mm", type=float, default=SQUARE_LENGTH_MM)
    parser.add_argument("--marker-length-mm", type=float, default=MARKER_LENGTH_MM)
    parser.add_argument("--marker-start-id", type=int, default=MARKER_START_ID)
    parser.add_argument("--margin-mm", type=float, default=MARGIN_MM)
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        default=LEGACY_PATTERN,
        help="Generate the pre-OpenCV-4.6 pattern layout",
    )
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.squares_x < 2 or args.squares_y < 2:
        parser.error("--squares-x and --squares-y must both be at least 2")
    if args.square_length_mm <= 0:
        parser.error("--square-length-mm must be positive")
    if not 0 < args.marker_length_mm < args.square_length_mm:
        parser.error("--marker-length-mm must be positive and smaller than a square")
    if args.marker_start_id < 0:
        parser.error("--marker-start-id cannot be negative")
    if args.margin_mm < 0:
        parser.error("--margin-mm cannot be negative")
    return args


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / 25.4 * dpi))


def dictionary_name(dictionary_id: int) -> str:
    for name in dir(cv2.aruco):
        if name.startswith("DICT_") and getattr(cv2.aruco, name) == dictionary_id:
            return name
    return str(dictionary_id)


def create_board(
    squares_x: int = SQUARES_X,
    squares_y: int = SQUARES_Y,
    square_length_mm: float = SQUARE_LENGTH_MM,
    marker_length_mm: float = MARKER_LENGTH_MM,
    legacy_pattern: bool = LEGACY_PATTERN,
    marker_start_id: int = MARKER_START_ID,
):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    default_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        dictionary,
    )
    marker_count = int(default_board.getIds().size)
    marker_end_id = marker_start_id + marker_count - 1
    dictionary_capacity = int(dictionary.bytesList.shape[0])
    if marker_start_id < 0 or marker_end_id >= dictionary_capacity:
        raise ValueError(
            f"Marker IDs {marker_start_id}-{marker_end_id} exceed dictionary "
            f"range 0-{dictionary_capacity - 1}."
        )
    marker_ids = np.arange(
        marker_start_id,
        marker_end_id + 1,
        dtype=np.int32,
    )
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm,
        marker_length_mm,
        dictionary,
        marker_ids,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(legacy_pattern)
    return board, dictionary


def compute_layout(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    margin_mm: float,
) -> dict[str, float]:
    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm
    usable_width_mm = PAGE_WIDTH_MM - 2.0 * margin_mm
    usable_height_mm = PAGE_HEIGHT_MM - 2.0 * margin_mm

    if board_width_mm > usable_width_mm or board_height_mm > usable_height_mm:
        raise ValueError(
            f"Board {board_width_mm:g} x {board_height_mm:g} mm does not fit "
            f"inside usable A3 area {usable_width_mm:g} x {usable_height_mm:g} mm."
        )

    return {
        "board_width_mm": board_width_mm,
        "board_height_mm": board_height_mm,
        "board_x_mm": (PAGE_WIDTH_MM - board_width_mm) / 2.0,
        "board_y_mm": (PAGE_HEIGHT_MM - board_height_mm) / 2.0,
    }


def save_png(path: Path, canvas: np.ndarray, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if PIL_AVAILABLE:
        Image.fromarray(canvas).save(path, dpi=(dpi, dpi))
        return
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"cv2.imwrite failed for {path}")

    # OpenCV does not write PNG resolution metadata. Insert a standards-based
    # pHYs chunk so print software still knows the intended physical scale.
    png_data = path.read_bytes()
    png_signature = b"\x89PNG\r\n\x1a\n"
    if not png_data.startswith(png_signature) or png_data[12:16] != b"IHDR":
        raise OSError(f"Unexpected PNG structure in {path}")
    ihdr_length = struct.unpack(">I", png_data[8:12])[0]
    ihdr_end = 8 + 12 + ihdr_length
    pixels_per_metre = int(round(dpi / 0.0254))
    chunk_type = b"pHYs"
    chunk_data = struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1)
    chunk_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    phys_chunk = (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", chunk_crc)
    )
    path.write_bytes(png_data[:ihdr_end] + phys_chunk + png_data[ihdr_end:])


def main() -> int:
    args = parse_args()
    try:
        layout = compute_layout(
            args.squares_x,
            args.squares_y,
            args.square_length_mm,
            args.margin_mm,
        )
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1

    try:
        board, dictionary = create_board(
            args.squares_x,
            args.squares_y,
            args.square_length_mm,
            args.marker_length_mm,
            args.legacy_pattern,
            args.marker_start_id,
        )
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    marker_count = int(board.getIds().size)
    marker_end_id = args.marker_start_id + marker_count - 1

    page_width_px = mm_to_px(PAGE_WIDTH_MM, args.dpi)
    page_height_px = mm_to_px(PAGE_HEIGHT_MM, args.dpi)
    board_width_px = mm_to_px(layout["board_width_mm"], args.dpi)
    board_height_px = mm_to_px(layout["board_height_mm"], args.dpi)
    board_x_px = round((page_width_px - board_width_px) / 2)
    board_y_px = round((page_height_px - board_height_px) / 2)

    board_image = board.generateImage(
        (board_width_px, board_height_px),
        marginSize=0,
        borderBits=BORDER_BITS,
    )
    canvas = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)
    canvas[
        board_y_px : board_y_px + board_height_px,
        board_x_px : board_x_px + board_width_px,
    ] = board_image

    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    metadata = {
        "schema_version": 1,
        "pattern_type": "charuco",
        "opencv_version": cv2.__version__,
        "dictionary": dictionary_name(ARUCO_DICT),
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "internal_charuco_corners": (args.squares_x - 1) * (args.squares_y - 1),
        "marker_count": marker_count,
        "marker_start_id": args.marker_start_id,
        "marker_end_id": marker_end_id,
        "marker_ids": board.getIds().reshape(-1).astype(int).tolist(),
        "square_length_mm": args.square_length_mm,
        "marker_length_mm": args.marker_length_mm,
        "border_bits": BORDER_BITS,
        "legacy_pattern": args.legacy_pattern,
        "page": {
            "width_mm": PAGE_WIDTH_MM,
            "height_mm": PAGE_HEIGHT_MM,
            "margin_mm": args.margin_mm,
            "dpi": args.dpi,
            "width_px": page_width_px,
            "height_px": page_height_px,
        },
        "board": {
            **layout,
            "width_px": board_width_px,
            "height_px": board_height_px,
            "x_px": board_x_px,
            "y_px": board_y_px,
        },
        "print_instruction": "Print at 100% / actual size; disable fit-to-page scaling.",
    }

    try:
        save_png(output_path, canvas, args.dpi)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    except OSError as error:
        print(f"[ERROR] Could not save output:\n{error}")
        return 1

    print("ChArUco A3 pattern generated.")
    print(f"Dictionary       : {dictionary_name(ARUCO_DICT)}")
    print(f"Squares          : {args.squares_x} x {args.squares_y}")
    print(f"ChArUco corners  : {(args.squares_x - 1) * (args.squares_y - 1)}")
    print(f"ArUco markers    : {marker_count}")
    print(f"Marker ID range  : {args.marker_start_id}-{marker_end_id}")
    print(f"Square length    : {args.square_length_mm:g} mm")
    print(f"Marker length    : {args.marker_length_mm:g} mm")
    print(
        f"Board size       : {layout['board_width_mm']:g} x "
        f"{layout['board_height_mm']:g} mm"
    )
    print(f"PNG size         : {page_width_px} x {page_height_px} px @ {args.dpi} DPI")
    print(f"Legacy pattern   : {args.legacy_pattern}")
    print(f"Saved PNG        : {output_path}")
    print(f"Saved metadata   : {metadata_path}")
    print("IMPORTANT: print at 100% / actual size; do not use fit-to-page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
