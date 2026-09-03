"""Generate a 10 mm ArUco ID 2 target with four corner-touching black blocks.

The four black blocks touch the central marker at its outer corners and provide
additional high-contrast geometry for a downstream custom subpixel detector.
They are not part of the ArUco code itself: OpenCV's detectMarkers() returns
only the central ArUco marker corners.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np


# User-adjustable physical geometry, in millimetres.
TARGET_SIZE_MM = 12.25
ARUCO_MARKER_SIZE_MM = 8.25
CORNER_BLOCK_SIZE_MM = 1.6
CORNER_BLOCK_MARGIN_MM = 0.4

ARUCO_DICT = cv2.aruco.DICT_4X4_100
ARUCO_ID = 12
BORDER_BITS = 1
DPI = 1200

OUTPUT_PNG = "aruco_id12_12_25mm_corner_blocks.png"
OUTPUT_METADATA = "aruco_id5_12_25mm_corner_blocks.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PNG))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(OUTPUT_METADATA),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--target-size-mm", type=float, default=TARGET_SIZE_MM)
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=ARUCO_MARKER_SIZE_MM,
    )
    parser.add_argument(
        "--corner-block-size-mm",
        type=float,
        default=CORNER_BLOCK_SIZE_MM,
    )
    parser.add_argument(
        "--corner-block-margin-mm",
        type=float,
        default=CORNER_BLOCK_MARGIN_MM,
    )
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.target_size_mm <= 0:
        parser.error("--target-size-mm must be positive")
    if not 0 < args.marker_size_mm < args.target_size_mm:
        parser.error("--marker-size-mm must be smaller than the full target")
    if args.corner_block_size_mm <= 0:
        parser.error("--corner-block-size-mm must be positive")
    if args.corner_block_margin_mm < 0:
        parser.error("--corner-block-margin-mm cannot be negative")
    corner_inner_edge = (
        args.corner_block_margin_mm + args.corner_block_size_mm
    )
    marker_margin = (args.target_size_mm - args.marker_size_mm) / 2.0
    if corner_inner_edge > marker_margin:
        parser.error(
            "Corner blocks cannot overlap the central marker; make their inner "
            "corners touch or leave a gap"
        )
    return args


def mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / 25.4 * dpi))


def dictionary_name(dictionary_id: int) -> str:
    for name in dir(cv2.aruco):
        if name.startswith("DICT_") and getattr(cv2.aruco, name) == dictionary_id:
            return name
    return str(dictionary_id)


def generate_marker(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            size_px,
            borderBits=BORDER_BITS,
        )
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(
        dictionary,
        marker_id,
        size_px,
        marker,
        BORDER_BITS,
    )
    return marker


def corner_block_rectangles(
    target_size_mm: float,
    block_size_mm: float,
    margin_mm: float,
) -> dict[str, tuple[float, float, float, float]]:
    far_start = target_size_mm - margin_mm - block_size_mm
    return {
        "top_left": (margin_mm, margin_mm, block_size_mm, block_size_mm),
        "top_right": (far_start, margin_mm, block_size_mm, block_size_mm),
        "bottom_right": (far_start, far_start, block_size_mm, block_size_mm),
        "bottom_left": (margin_mm, far_start, block_size_mm, block_size_mm),
    }


def insert_png_dpi(path: Path, dpi: int) -> None:
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


def block_metadata(
    rectangles: dict[str, tuple[float, float, float, float]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, (x, y, width, height) in rectangles.items():
        result[name] = {
            "top_left_mm": [x, y],
            "size_mm": [width, height],
            "center_mm": [x + width / 2.0, y + height / 2.0],
            "corners_mm": [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
            ],
        }
    return result


def main() -> int:
    args = parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    dictionary_capacity = int(dictionary.bytesList.shape[0])
    if not 0 <= ARUCO_ID < dictionary_capacity:
        print(
            f"[ERROR] Marker ID {ARUCO_ID} is outside dictionary range "
            f"0-{dictionary_capacity - 1}."
        )
        return 1

    target_px = mm_to_px(args.target_size_mm, args.dpi)
    marker_px = mm_to_px(args.marker_size_mm, args.dpi)
    marker_start = round((target_px - marker_px) / 2)
    marker_end = marker_start + marker_px
    marker_start_mm = (args.target_size_mm - args.marker_size_mm) / 2.0

    canvas = np.full((target_px, target_px), 255, dtype=np.uint8)
    marker = generate_marker(dictionary, ARUCO_ID, marker_px)
    canvas[marker_start:marker_end, marker_start:marker_end] = marker

    rectangles = corner_block_rectangles(
        args.target_size_mm,
        args.corner_block_size_mm,
        args.corner_block_margin_mm,
    )
    for x_mm, y_mm, width_mm, height_mm in rectangles.values():
        x0 = mm_to_px(x_mm, args.dpi)
        y0 = mm_to_px(y_mm, args.dpi)
        x1 = mm_to_px(x_mm + width_mm, args.dpi)
        y1 = mm_to_px(y_mm + height_mm, args.dpi)
        canvas[y0:y1, x0:x1] = 0

    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    metadata = {
        "schema_version": 1,
        "pattern_type": "aruco_with_corner_touching_blocks",
        "dictionary": dictionary_name(ARUCO_DICT),
        "aruco_id": ARUCO_ID,
        "border_bits": BORDER_BITS,
        "dpi": args.dpi,
        "image_size_px": [target_px, target_px],
        "target_size_mm": [args.target_size_mm, args.target_size_mm],
        "coordinate_system": "origin at target top-left; +X right, +Y down",
        "central_aruco": {
            "top_left_mm": [marker_start_mm, marker_start_mm],
            "size_mm": [args.marker_size_mm, args.marker_size_mm],
            "corners_mm": [
                [marker_start_mm, marker_start_mm],
                [marker_start_mm + args.marker_size_mm, marker_start_mm],
                [
                    marker_start_mm + args.marker_size_mm,
                    marker_start_mm + args.marker_size_mm,
                ],
                [marker_start_mm, marker_start_mm + args.marker_size_mm],
            ],
        },
        "auxiliary_corner_blocks": block_metadata(rectangles),
        "important": (
            "The corner-touching blocks are not part of the ArUco code. Detect "
            "and refine their extra geometry separately if it is to improve pose "
            "precision."
        ),
        "print_instruction": "Print at 100% / actual size; disable scaling.",
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"cv2.imwrite failed for {output_path}")
        insert_png_dpi(output_path, args.dpi)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    except OSError as error:
        print(f"[ERROR] Could not save output:\n{error}")
        return 1

    white_gap_mm = marker_start_mm - (
        args.corner_block_margin_mm + args.corner_block_size_mm
    )
    print("ArUco ID 2 composite target generated.")
    print(f"Dictionary         : {dictionary_name(ARUCO_DICT)}")
    print(f"ArUco ID           : {ARUCO_ID}")
    print(f"Full target size   : {args.target_size_mm:g} x {args.target_size_mm:g} mm")
    print(f"Central marker     : {args.marker_size_mm:g} x {args.marker_size_mm:g} mm")
    print(
        f"Corner blocks      : 4 x ({args.corner_block_size_mm:g} x "
        f"{args.corner_block_size_mm:g} mm)"
    )
    connection = "corner-touching" if abs(white_gap_mm) < 1e-9 else "separated"
    print(f"Corner connection  : {connection} (gap={white_gap_mm:g} mm)")
    print(f"PNG                : {target_px} x {target_px} px @ {args.dpi} DPI")
    print(f"Saved PNG          : {output_path}")
    print(f"Saved metadata     : {metadata_path}")
    print("IMPORTANT: print at 100% / actual size; do not use fit-to-page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
