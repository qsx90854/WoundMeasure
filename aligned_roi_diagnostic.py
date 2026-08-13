"""Shared pixel-aligned diagnostic ROI video helpers.

The crop is fixed from the temporally stable Pattern corners, then reused for
the Pattern and GT videos.  The image is filtered and enlarged before the
subpixel marks are drawn, so the red one-pixel lines remain crisp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ALIGNED_ROI_OUTPUT_SIZE_PX = 320
BILATERAL_DIAMETER_PX = 5
BILATERAL_SIGMA_COLOR = 25.0
BILATERAL_SIGMA_SPACE = 25.0


@dataclass(frozen=True)
class AlignedRoiSpec:
    """One square crop in original camera-pixel coordinates."""

    x0: int
    y0: int
    size_px: int

    @property
    def x1(self) -> int:
        return self.x0 + self.size_px

    @property
    def y1(self) -> int:
        return self.y0 + self.size_px


def build_aligned_roi_spec(
    stable_pattern_corners,
    margin_each_side_px: int = 10,
) -> AlignedRoiSpec:
    """Build a square ROI around all four stable Pattern corners.

    A marker spanning roughly 50 px with the default 10 px margin on each side
    produces a roughly 70x70 source crop.
    """
    points = np.asarray(stable_pattern_corners, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("Pattern ROI corners contain non-finite coordinates")
    margin = max(0, int(margin_each_side_px))
    span_x = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    span_y = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    size = max(3, int(math.ceil(max(span_x, span_y) + 2 * margin)))
    center = np.mean(points, axis=0)
    x0 = int(round(float(center[0]) - size / 2.0))
    y0 = int(round(float(center[1]) - size / 2.0))
    return AlignedRoiSpec(x0=x0, y0=y0, size_px=size)


def crop_with_padding(frame: np.ndarray, roi: AlignedRoiSpec) -> np.ndarray:
    """Crop without shifting coordinates; pad black if an ROI reaches an edge."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected one BGR frame")
    output = np.zeros((roi.size_px, roi.size_px, 3), dtype=frame.dtype)
    source_x0 = max(0, roi.x0)
    source_y0 = max(0, roi.y0)
    source_x1 = min(frame.shape[1], roi.x1)
    source_y1 = min(frame.shape[0], roi.y1)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return output
    destination_x0 = source_x0 - roi.x0
    destination_y0 = source_y0 - roi.y0
    destination_x1 = destination_x0 + (source_x1 - source_x0)
    destination_y1 = destination_y0 + (source_y1 - source_y0)
    output[destination_y0:destination_y1, destination_x0:destination_x1] = frame[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return output


def bilateral_lanczos_enlarge(
    crop: np.ndarray,
    output_size_px: int = ALIGNED_ROI_OUTPUT_SIZE_PX,
) -> np.ndarray:
    """Edge-preserving denoise followed by high-quality Lanczos enlargement."""
    output_size = max(16, int(output_size_px))
    filtered = cv2.bilateralFilter(
        crop,
        BILATERAL_DIAMETER_PX,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE,
    )
    return cv2.resize(
        filtered,
        (output_size, output_size),
        interpolation=cv2.INTER_LANCZOS4,
    )


def _point_to_output(point, roi: AlignedRoiSpec, output_size_px: int) -> tuple[int, int]:
    x, y = (float(value) for value in point)
    scale = output_size_px / float(roi.size_px)
    # Pixel-center mapping consistent with OpenCV resize.
    output_x = int(round((x - roi.x0 + 0.5) * scale - 0.5))
    output_y = int(round((y - roi.y0 + 0.5) * scale - 0.5))
    return output_x, output_y


def draw_subpixel_crosses(
    image: np.ndarray,
    points,
    roi: AlignedRoiSpec,
    cross_arm_px: int = 5,
) -> np.ndarray:
    """Draw thin red crosses after filtering/resizing, preserving line quality."""
    if points is None:
        return image
    arm = max(1, int(cross_arm_px))
    for point in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        if not np.all(np.isfinite(point)):
            continue
        x, y = _point_to_output(point, roi, image.shape[1])
        cv2.line(image, (x - arm, y), (x + arm, y), (0, 0, 255), 1, cv2.LINE_8)
        cv2.line(image, (x, y - arm), (x, y + arm), (0, 0, 255), 1, cv2.LINE_8)
    return image


def make_aligned_roi_frame(
    frame: np.ndarray,
    roi: AlignedRoiSpec,
    points=None,
    output_size_px: int = ALIGNED_ROI_OUTPUT_SIZE_PX,
    cross_arm_px: int = 5,
) -> np.ndarray:
    """Crop, bilateral-filter, Lanczos-enlarge, then draw subpixel points."""
    enlarged = bilateral_lanczos_enlarge(
        crop_with_padding(frame, roi),
        output_size_px,
    )
    return draw_subpixel_crosses(enlarged, points, roi, cross_arm_px)


def open_aligned_roi_writer(
    path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Prefer lossless FFV1 AVI, with MJPG AVI as a widely supported fallback."""
    path = Path(path)
    if path.suffix.lower() != ".avi":
        raise ValueError(f"Aligned ROI diagnostics must use .avi: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 25.0
    size = tuple(int(value) for value in frame_size)
    attempted = []
    for codec in ("FFV1", "MJPG"):
        attempted.append(codec)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open aligned ROI AVI writer ({'/'.join(attempted)}): {path}"
    )
