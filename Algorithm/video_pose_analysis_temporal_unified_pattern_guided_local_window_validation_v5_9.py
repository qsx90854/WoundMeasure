"""Two-video, marker-only RT validation front end.

This module is deliberately isolated from the production Zebra analyser.  It
reuses the production module's ArUco/IPPE, marker-map, exact temporal-DP and
marker-only endpoint optimisation helpers, but it never constructs SIFT and it
never calls ``analyze_video_frames``.

The experiment represented here has two *discontinuous* video segments:

``video A``
    Pattern at the first physical position.

``video B``
    Pattern at the second physical position.

Each segment gets its own temporal path.  There is intentionally no temporal
transition, KLT track, or velocity prior across the A/B boundary.  The final
relative-pose convention is the same one used by Zebra::

    X_A = R_rel @ X_B + t_rel

For a fixed camera and a pure Pattern displacement ``delta = B - A`` expressed
in the camera coordinate system, the expected translation is therefore
``t_rel = -delta`` and the expected rotation is identity.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import video_pose_analysis_temporal_unified_pattern_guided_local_window as _base


FEATURE_MODE = "marker_only"
VALIDATION_BUILD = "2026-08-26-v5.9-4k-halfres-fullres-subpix"
CORE_A_FRACTIONS = (0.80, 1.00)
CORE_B_FRACTIONS = (0.10, 0.22, 0.40)
CORE_PROBE_FALLBACK_RADIUS = 3

# _base._unified_optimize_endpoint_world_poses reads these production module
# globals.  Serialise the short override so simultaneous validation calls cannot
# observe each other's gates, and always restore the production values.
_BASELINE_GATE_LOCK = threading.Lock()


def _normalise_segment_name(segment: str) -> str:
    value = str(segment).strip().upper()
    if value not in ("A", "B"):
        raise ValueError(f"segment must be 'A' or 'B', got {segment!r}")
    return value


class LazyVideoSegment:
    """A small, thread-safe lazy reader for one video or an in-memory sequence."""

    def __init__(self, video_path: Optional[str], source_override: Any = None):
        self.video_path = None if video_path is None else str(video_path)
        self._source_override = source_override
        self._cache: Dict[int, np.ndarray] = {}
        self._capture = None
        self._lock = threading.Lock()

        if source_override is not None:
            if not hasattr(source_override, "__len__") or not hasattr(source_override, "__getitem__"):
                raise TypeError("Each frame-source override must be a sized indexable object")
            self._frame_count = int(len(source_override))
        else:
            if not self.video_path:
                raise ValueError("A video path is required when no frame-source override is supplied")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                cap.release()
                raise OSError(f"Unable to open video: {self.video_path}")
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if self._frame_count <= 0:
            raise ValueError(f"Video segment has no frames: {self.video_path or '<override>'}")

    def __len__(self) -> int:
        return self._frame_count

    def __bool__(self) -> bool:
        return self._frame_count > 0

    def __getitem__(self, index: int) -> np.ndarray:
        index = int(index)
        if index < 0:
            index += self._frame_count
        if not 0 <= index < self._frame_count:
            raise IndexError(index)

        cached = self._cache.get(index)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                return cached
            if self._source_override is not None:
                frame = np.asarray(self._source_override[index])
                if frame.ndim not in (2, 3):
                    raise ValueError(f"Frame {index} has invalid shape {frame.shape}")
                frame = frame.copy()
            else:
                cap = self._get_capture()
                current = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if current != index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise IndexError(f"Unable to decode frame {index} from {self.video_path}")
            self._cache[index] = frame
            return frame

    def _get_capture(self):
        if self._capture is None:
            self._capture = cv2.VideoCapture(self.video_path)
            if not self._capture.isOpened():
                self._capture.release()
                self._capture = None
                raise OSError(f"Unable to open video: {self.video_path}")
        return self._capture

    def preload(self, indices: Iterable[int]) -> None:
        # Only a handful of probes are requested.  Sorted access preserves
        # forward decoding while keeping the implementation mock-friendly.
        for index in sorted(set(int(value) for value in indices)):
            if 0 <= index < self._frame_count:
                _ = self[index]

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def __del__(self):  # pragma: no cover - best-effort native resource cleanup
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()


class TwoSegmentVideoFrames:
    """Virtual A+B index space backed by two independent lazy readers.

    Global indices ``0 .. NA-1`` map to A.  Indices ``NA .. NA+NB-1`` map to
    B.  The mapping is for storage/UI only; temporal inference remains separate.
    """

    def __init__(
        self,
        video_a_path: Optional[str],
        video_b_path: Optional[str],
        frame_source_override: Any = None,
    ):
        override_a = override_b = None
        if frame_source_override is not None:
            if isinstance(frame_source_override, Mapping):
                override_a = frame_source_override.get("A", frame_source_override.get("a"))
                override_b = frame_source_override.get("B", frame_source_override.get("b"))
            elif isinstance(frame_source_override, (tuple, list)) and len(frame_source_override) == 2:
                override_a, override_b = frame_source_override
            else:
                raise TypeError(
                    "frame_source_override must be {'A': frames, 'B': frames} or a two-item tuple")
            if override_a is None or override_b is None:
                raise ValueError("frame_source_override must contain both A and B")

        self.A = LazyVideoSegment(video_a_path, override_a)
        self.B = LazyVideoSegment(video_b_path, override_b)
        self._a_count = len(self.A)
        self._b_count = len(self.B)

    def __len__(self) -> int:
        return self._a_count + self._b_count

    @property
    def segment_lengths(self) -> Dict[str, int]:
        return {"A": self._a_count, "B": self._b_count}

    def global_to_segment(self, index: int) -> Tuple[str, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self._a_count:
            return "A", index
        return "B", index - self._a_count

    # Alias useful to callers that use the term "virtual index".
    global_to_local = global_to_segment

    def segment_to_global(self, segment: str, local_index: int) -> int:
        segment = _normalise_segment_name(segment)
        local_index = int(local_index)
        length = self._a_count if segment == "A" else self._b_count
        if local_index < 0:
            local_index += length
        if not 0 <= local_index < length:
            raise IndexError(local_index)
        return local_index if segment == "A" else self._a_count + local_index

    local_to_global = segment_to_global

    def get_segment_frame(self, segment: str, local_index: int) -> np.ndarray:
        segment = _normalise_segment_name(segment)
        return (self.A if segment == "A" else self.B)[int(local_index)]

    def __getitem__(self, global_index: int) -> np.ndarray:
        segment, local_index = self.global_to_segment(global_index)
        return self.get_segment_frame(segment, local_index)

    def preload_segment(self, segment: str, local_indices: Iterable[int]) -> None:
        segment = _normalise_segment_name(segment)
        (self.A if segment == "A" else self.B).preload(local_indices)

    def preload(self, global_indices: Iterable[int]) -> None:
        grouped = {"A": [], "B": []}
        for index in global_indices:
            segment, local = self.global_to_segment(int(index))
            grouped[segment].append(local)
        self.A.preload(grouped["A"])
        self.B.preload(grouped["B"])

    def close(self) -> None:
        self.A.close()
        self.B.close()


def compose_relative_pose(
    R_A: np.ndarray,
    t_A: np.ndarray,
    R_B: np.ndarray,
    t_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``T_A<-B`` from two marker-world endpoint poses."""
    R_A = np.asarray(R_A, np.float64).reshape(3, 3)
    R_B = np.asarray(R_B, np.float64).reshape(3, 3)
    t_A = np.asarray(t_A, np.float64).reshape(3, 1)
    t_B = np.asarray(t_B, np.float64).reshape(3, 1)
    R_rel = R_A @ R_B.T
    t_rel = t_A - R_rel @ t_B
    return R_rel, t_rel


def _float_or_none(value: Any) -> Optional[float]:
    """Return a finite float, otherwise None (diagnostic formatting helper)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rotation_error_deg(rotation: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    reference = np.eye(3, dtype=np.float64) if reference is None else np.asarray(
        reference, np.float64).reshape(3, 3)
    delta = rotation @ reference.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _direction_error_deg(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    first = np.asarray(first, np.float64).reshape(3)
    second = np.asarray(second, np.float64).reshape(3)
    n_first = float(np.linalg.norm(first))
    n_second = float(np.linalg.norm(second))
    if n_first <= 1e-12 or n_second <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(first, second) / (n_first * n_second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_ground_truth_metrics(
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    known_translation_mm: Optional[float] = None,
    known_translation_vector: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compare an estimate with the fixed-camera, moved-Pattern experiment.

    ``known_translation_vector`` is the physical Pattern displacement A -> B.
    Since the analyser returns ``T_A<-B``, the expected relative translation is
    its negative.  The known values are metrics only and are never used by pair
    ranking or optimisation.
    """
    R_rel = np.asarray(R_rel, np.float64).reshape(3, 3)
    t_rel = np.asarray(t_rel, np.float64).reshape(3, 1)
    estimated_baseline = float(np.linalg.norm(t_rel))

    expected_t_rel = None
    vector_error = None
    direction_error = None
    vector_norm = None
    if known_translation_vector is not None:
        delta = np.asarray(known_translation_vector, np.float64).reshape(3)
        if not np.all(np.isfinite(delta)):
            raise ValueError("known_translation_vector must contain finite values")
        expected_t_rel = -delta
        vector_norm = float(np.linalg.norm(delta))
        vector_error = float(np.linalg.norm(t_rel.reshape(3) - expected_t_rel))
        direction_error = _direction_error_deg(t_rel, expected_t_rel)

    if known_translation_mm is None and vector_norm is not None:
        known_baseline = vector_norm
    elif known_translation_mm is None:
        known_baseline = None
    else:
        known_baseline = float(known_translation_mm)
        if not np.isfinite(known_baseline) or known_baseline < 0.0:
            raise ValueError("known_translation_mm must be a finite non-negative value")

    absolute_error = (
        None if known_baseline is None
        else abs(estimated_baseline - known_baseline))
    percent_error = (
        None if known_baseline is None or known_baseline <= 1e-12
        else 100.0 * absolute_error / known_baseline)
    scalar_vector_mismatch = None
    scalar_vector_consistent = None
    if known_translation_mm is not None and vector_norm is not None:
        scalar_vector_mismatch = abs(float(known_translation_mm) - vector_norm)
        consistency_tolerance = max(0.10, 0.01 * max(float(known_translation_mm), vector_norm))
        scalar_vector_consistent = bool(scalar_vector_mismatch <= consistency_tolerance)
    return {
        "estimated_baseline_mm": estimated_baseline,
        "known_baseline_mm": known_baseline,
        "baseline_absolute_error_mm": absolute_error,
        "baseline_error_percent": percent_error,
        "rotation_error_deg": _rotation_error_deg(R_rel),
        "translation_direction_error_deg": direction_error,
        "translation_vector_error_mm": vector_error,
        "expected_t_rel": expected_t_rel,
        "known_scalar_vector_mismatch_mm": scalar_vector_mismatch,
        "known_scalar_vector_consistent": scalar_vector_consistent,
        "known_vector_convention": (
            None if known_translation_vector is None
            else "input is Pattern A->B; expected analyser t_rel is its negative"),
    }


def _sample_fraction_indices(frame_count: int, fractions: Sequence[float]) -> list[int]:
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    return list(dict.fromkeys(
        int(round((frame_count - 1) * float(fraction))) for fraction in fractions))


def _core_probe_search_indices(anchor: int, frame_count: int, radius: int = CORE_PROBE_FALLBACK_RADIUS) -> list[int]:
    """Return nominal, -1, +1, -2, +2, ... probe order within segment bounds."""
    anchor = int(anchor)
    frame_count = int(frame_count)
    radius = max(0, int(radius))
    if frame_count <= 0 or not (0 <= anchor < frame_count):
        return []
    ordered = [anchor]
    for delta in range(1, radius + 1):
        for candidate in (anchor - delta, anchor + delta):
            if 0 <= candidate < frame_count and candidate not in ordered:
                ordered.append(candidate)
    return ordered


def _normalise_corner_dict(value: Any) -> Dict[int, np.ndarray]:
    if not value:
        return {}
    return {
        int(marker_id): np.asarray(points, np.float32).reshape(4, 2).copy()
        for marker_id, points in dict(value).items()
    }


def _segment_corner_override(overrides: Any, segment: str) -> Any:
    if overrides is None:
        return None
    if not isinstance(overrides, Mapping):
        raise TypeError("marker_corners_override must be a mapping with A and B segments")
    return overrides.get(segment, overrides.get(segment.lower()))


CORNER_MODE_RAW = "RAW"
CORNER_MODE_SUBPIX_3 = "SUBPIX_3"
CORNER_MODE_SUBPIX_5 = "SUBPIX_5"
CORNER_MODE_CONTOUR = "CONTOUR"
CORNER_MODE_APRILTAG = "APRILTAG"
CORNER_MODES = {
    CORNER_MODE_RAW,
    CORNER_MODE_SUBPIX_3,
    CORNER_MODE_SUBPIX_5,
    CORNER_MODE_CONTOUR,
    CORNER_MODE_APRILTAG,
}

ARUCO_PRESET_DEFAULT = "DEFAULT"
ARUCO_PRESET_LCD_ROBUST = "LCD_ROBUST"
ARUCO_PRESET_LCD_AGGRESSIVE = "LCD_AGGRESSIVE"
ARUCO_PRESETS = {
    ARUCO_PRESET_DEFAULT,
    ARUCO_PRESET_LCD_ROBUST,
    ARUCO_PRESET_LCD_AGGRESSIVE,
}

MARKER_MAP_RESCUE_RADIUS = 3
MARKER_MAP_RELAXED_MIN_SUPPORT = 2
_MARKER_MAP_LOCK = threading.Lock()


def _new_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _configure_detector_parameters(preset: str, refinement_method: int):
    preset = str(preset or ARUCO_PRESET_DEFAULT).strip().upper()
    if preset not in ARUCO_PRESETS:
        raise ValueError(f"Unknown ArUco detector preset: {preset}")
    params = _new_detector_parameters()
    params.cornerRefinementMethod = int(refinement_method)

    # DEFAULT intentionally preserves OpenCV's defaults.  The LCD presets are
    # validation-only alternatives for a high-resolution monitor where the
    # panel sub-pixel grid / moire can disturb the small default threshold
    # windows.  Exact values are exported so runs remain reproducible.
    if preset == ARUCO_PRESET_LCD_ROBUST:
        params.adaptiveThreshWinSizeMin = 15
        params.adaptiveThreshWinSizeMax = 63
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 7.0
        params.polygonalApproxAccuracyRate = 0.025
        params.minCornerDistanceRate = 0.03
        params.minMarkerPerimeterRate = 0.015
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.10
    elif preset == ARUCO_PRESET_LCD_AGGRESSIVE:
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 83
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 5.0
        params.polygonalApproxAccuracyRate = 0.04
        params.minCornerDistanceRate = 0.02
        params.minMarkerPerimeterRate = 0.008
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.08
    return params


def _detector_parameters_dict(params) -> Dict[str, Any]:
    names = (
        "adaptiveThreshWinSizeMin", "adaptiveThreshWinSizeMax",
        "adaptiveThreshWinSizeStep", "adaptiveThreshConstant",
        "polygonalApproxAccuracyRate", "minCornerDistanceRate",
        "minMarkerPerimeterRate", "maxMarkerPerimeterRate",
        "perspectiveRemovePixelPerCell", "perspectiveRemoveIgnoredMarginPerCell",
        "cornerRefinementMethod", "cornerRefinementWinSize",
        "cornerRefinementMaxIterations", "cornerRefinementMinAccuracy",
    )
    result = {}
    for name in names:
        if hasattr(params, name):
            value = getattr(params, name)
            if isinstance(value, (np.integer, int)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            result[name] = value
    return result


CORNER_MODE_RAW = "RAW"
CORNER_MODE_SUBPIX_3 = "SUBPIX_3"
CORNER_MODE_SUBPIX_5 = "SUBPIX_5"
CORNER_MODE_CONTOUR = "CONTOUR"
CORNER_MODE_APRILTAG = "APRILTAG"
CORNER_MODES = {
    CORNER_MODE_RAW,
    CORNER_MODE_SUBPIX_3,
    CORNER_MODE_SUBPIX_5,
    CORNER_MODE_CONTOUR,
    CORNER_MODE_APRILTAG,
}

ARUCO_PRESET_DEFAULT = "DEFAULT"
ARUCO_PRESET_LCD_ROBUST = "LCD_ROBUST"
ARUCO_PRESET_LCD_AGGRESSIVE = "LCD_AGGRESSIVE"
ARUCO_PRESETS = {
    ARUCO_PRESET_DEFAULT,
    ARUCO_PRESET_LCD_ROBUST,
    ARUCO_PRESET_LCD_AGGRESSIVE,
}

MARKER_MAP_RESCUE_RADIUS = 3
MARKER_MAP_RELAXED_MIN_SUPPORT = 2
_MARKER_MAP_LOCK = threading.Lock()


def _new_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _configure_detector_parameters(preset: str, refinement_method: int):
    preset = str(preset or ARUCO_PRESET_DEFAULT).strip().upper()
    if preset not in ARUCO_PRESETS:
        raise ValueError(f"Unknown ArUco detector preset: {preset}")
    params = _new_detector_parameters()
    params.cornerRefinementMethod = int(refinement_method)

    # DEFAULT intentionally preserves OpenCV's defaults.  The LCD presets are
    # validation-only alternatives for a high-resolution monitor where the
    # panel sub-pixel grid / moire can disturb the small default threshold
    # windows.  Exact values are exported so runs remain reproducible.
    if preset == ARUCO_PRESET_LCD_ROBUST:
        params.adaptiveThreshWinSizeMin = 15
        params.adaptiveThreshWinSizeMax = 63
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 7.0
        params.polygonalApproxAccuracyRate = 0.025
        params.minCornerDistanceRate = 0.03
        params.minMarkerPerimeterRate = 0.015
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.10
    elif preset == ARUCO_PRESET_LCD_AGGRESSIVE:
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 83
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 5.0
        params.polygonalApproxAccuracyRate = 0.04
        params.minCornerDistanceRate = 0.02
        params.minMarkerPerimeterRate = 0.008
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.08
    return params


def _detector_parameters_dict(params) -> Dict[str, Any]:
    names = (
        "adaptiveThreshWinSizeMin", "adaptiveThreshWinSizeMax",
        "adaptiveThreshWinSizeStep", "adaptiveThreshConstant",
        "polygonalApproxAccuracyRate", "minCornerDistanceRate",
        "minMarkerPerimeterRate", "maxMarkerPerimeterRate",
        "perspectiveRemovePixelPerCell", "perspectiveRemoveIgnoredMarginPerCell",
        "cornerRefinementMethod", "cornerRefinementWinSize",
        "cornerRefinementMaxIterations", "cornerRefinementMinAccuracy",
    )
    result = {}
    for name in names:
        if hasattr(params, name):
            value = getattr(params, name)
            if isinstance(value, (np.integer, int)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            result[name] = value
    return result


class _MarkerDetector:
    def __init__(
            self, use_clahe: bool = True,
            corner_mode: str = CORNER_MODE_SUBPIX_5,
            detector_preset: str = ARUCO_PRESET_DEFAULT,
            outer_edge_refine: bool = False,
            four_k_halfres_seed: bool = True):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco module is unavailable")
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self._use_clahe = bool(use_clahe)
        self._clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if self._use_clahe else None
        )
        self._corner_mode = str(corner_mode or CORNER_MODE_SUBPIX_5).strip().upper()
        if self._corner_mode not in CORNER_MODES:
            raise ValueError(f"Unknown corner mode: {self._corner_mode}")
        self._detector_preset = str(detector_preset or ARUCO_PRESET_DEFAULT).strip().upper()
        if self._detector_preset not in ARUCO_PRESETS:
            raise ValueError(f"Unknown ArUco detector preset: {self._detector_preset}")
        self._outer_edge_refine = bool(outer_edge_refine)
        self._four_k_halfres_seed = bool(four_k_halfres_seed)

        # Dedicated 4K coarse detector.  On 3840x2160 input we first denoise and
        # downsample to 1920x1080, detect a stable coarse ArUco quad there, scale
        # the coordinates back x2, then run cornerSubPix on the ORIGINAL 4K gray.
        # The aggressive half-res preset is intentional: on the supplied 4K clip
        # it keeps both markers detectable on every frame; the large full-res
        # SubPix window then pulls the coarse seed onto the physical black/white corner.
        self._half4k_parameters = _configure_detector_parameters(
            ARUCO_PRESET_LCD_AGGRESSIVE, cv2.aruco.CORNER_REFINE_NONE)
        self._half4k_detector = (
            cv2.aruco.ArucoDetector(self._dictionary, self._half4k_parameters)
            if hasattr(cv2.aruco, "ArucoDetector") else None
        )

        # Raw detector always disables built-in corner refinement so we can
        # distinguish polygon-quad placement from the selected refinement mode.
        self._raw_parameters = _configure_detector_parameters(
            self._detector_preset, cv2.aruco.CORNER_REFINE_NONE)
        self._raw_detector = (
            cv2.aruco.ArucoDetector(self._dictionary, self._raw_parameters)
            if hasattr(cv2.aruco, "ArucoDetector") else None
        )

        self._refined_detector = None
        self._refined_parameters = None
        if self._corner_mode in (CORNER_MODE_CONTOUR, CORNER_MODE_APRILTAG):
            attr = (
                "CORNER_REFINE_CONTOUR"
                if self._corner_mode == CORNER_MODE_CONTOUR
                else "CORNER_REFINE_APRILTAG"
            )
            if not hasattr(cv2.aruco, attr):
                raise RuntimeError(
                    f"This OpenCV build does not support {attr}; choose another Corner mode")
            method = getattr(cv2.aruco, attr)
            self._refined_parameters = _configure_detector_parameters(
                self._detector_preset, method)
            if hasattr(cv2.aruco, "ArucoDetector"):
                self._refined_detector = cv2.aruco.ArucoDetector(
                    self._dictionary, self._refined_parameters)

    @property
    def corner_mode(self) -> str:
        return self._corner_mode

    @property
    def detector_preset(self) -> str:
        return self._detector_preset

    @property
    def parameters_dict(self) -> Dict[str, Any]:
        params = self._refined_parameters or self._raw_parameters
        return _detector_parameters_dict(params)

    def prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            gray = frame.astype(np.uint8, copy=False)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gray) if self._use_clahe else gray

    def representative_binary(self, frame: np.ndarray) -> np.ndarray:
        """Diagnostic threshold image; ArUco internally tests multiple windows."""
        gray = self.prepare_gray(frame)
        params = self._raw_parameters
        lo = int(params.adaptiveThreshWinSizeMin)
        hi = int(params.adaptiveThreshWinSizeMax)
        win = max(3, int(round((lo + hi) * 0.5)))
        if win % 2 == 0:
            win += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, win, float(params.adaptiveThreshConstant))

    def _detect_with_params(self, gray: np.ndarray, detector, params):
        if detector is not None:
            return detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self._dictionary, parameters=params)

    @staticmethod
    def _to_corner_dict(corners, ids) -> Dict[int, np.ndarray]:
        if ids is None or not len(ids):
            return {}
        return {
            int(marker_id): np.asarray(marker_corners, np.float32).reshape(4, 2).copy()
            for marker_id, marker_corners in zip(ids.reshape(-1), corners)
        }

    def _selected_mode_corners(
            self, detect_gray: np.ndarray, raw_result: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Apply the user-selected OpenCV corner mode, before outer-edge recovery."""
        if self._corner_mode == CORNER_MODE_RAW:
            return {mid: pts.copy() for mid, pts in raw_result.items()}

        if self._corner_mode in (CORNER_MODE_SUBPIX_3, CORNER_MODE_SUBPIX_5):
            win = 3 if self._corner_mode == CORNER_MODE_SUBPIX_3 else 5
            term = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.0001,
            )
            selected = {}
            for marker_id, raw in raw_result.items():
                refined = raw.copy()
                try:
                    cv2.cornerSubPix(
                        detect_gray, refined.reshape(-1, 1, 2),
                        (win, win), (-1, -1), term)
                except cv2.error:
                    pass
                selected[int(marker_id)] = refined
            return selected

        # CONTOUR/APRILTAG: run the same detector preset with OpenCV's built-in
        # refinement enabled.  If a refined pass misses an ID that raw detection
        # found, keep the raw corners for that ID instead of silently dropping it.
        refined_corners, refined_ids, _refined_rejected = self._detect_with_params(
            detect_gray, self._refined_detector, self._refined_parameters)
        refined_result = self._to_corner_dict(refined_corners, refined_ids)
        return {
            mid: np.asarray(refined_result.get(mid, raw), np.float32).reshape(4, 2).copy()
            for mid, raw in raw_result.items()
        }

    def _detect_4k_halfres_then_fullres_subpix(
            self, frame: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
        """4K-only coarse-to-fine ArUco corner localization.

        1) Original 3840x2160 BGR -> grayscale.
        2) Bilateral filter on the full-resolution gray.
        3) Resize to 1920x1080 with INTER_AREA.
        4) Optional CLAHE, then ArUco detection on the half-resolution image.
        5) Multiply coarse corner coordinates by exactly 2.
        6) Run cornerSubPix on the ORIGINAL, unfiltered 3840x2160 gray image.

        A 13x13 SubPix half-window is deliberate: an inward 126 px quad versus
        a true ~143 px quad is about 8 px per side, so a normal 3x3/5x5 window
        cannot reach the physical corner from the coarse seed.
        """
        image = np.asarray(frame)
        if image.ndim == 2:
            full_gray = image.astype(np.uint8, copy=False)
        else:
            full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        filtered = cv2.bilateralFilter(full_gray, 5, 25.0, 25.0)
        half_gray = cv2.resize(filtered, (1920, 1080), interpolation=cv2.INTER_AREA)
        if self._use_clahe:
            half_gray = self._clahe.apply(half_gray)

        half_corners, half_ids, _ = self._detect_with_params(
            half_gray, self._half4k_detector, self._half4k_parameters)
        half_result = self._to_corner_dict(half_corners, half_ids)
        if not half_result:
            return {}, {}, {
                "enabled": True, "used": True, "status": "HALF_RES_NO_MARKER",
                "half_resolution": [1920, 1080], "full_resolution": [3840, 2160],
                "subpix_window": [13, 13],
            }

        coarse_full = {
            int(mid): (np.asarray(pts, np.float32).reshape(4, 2) * 2.0).copy()
            for mid, pts in half_result.items()
        }
        refined = {}
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.0001,
        )
        per_marker = {}
        for marker_id, seed in coarse_full.items():
            points = seed.copy()
            before = points.copy()
            try:
                cv2.cornerSubPix(
                    full_gray, points.reshape(-1, 1, 2),
                    (13, 13), (-1, -1), term)
                status = "OK"
            except cv2.error as exc:
                status = f"SUBPIX_ERROR: {exc}"
            refined[int(marker_id)] = points
            delta = points - before
            per_marker[int(marker_id)] = {
                "status": status,
                "coarse_half_xy": (before / 2.0).tolist(),
                "coarse_full_xy": before.tolist(),
                "refined_full_xy": points.tolist(),
                "mean_shift_px": float(np.mean(np.linalg.norm(delta, axis=1))),
                "max_shift_px": float(np.max(np.linalg.norm(delta, axis=1))),
            }

        return refined, coarse_full, {
            "enabled": True, "used": True, "status": "OK",
            "half_resolution": [1920, 1080],
            "full_resolution": [3840, 2160],
            "bilateral": {"d": 5, "sigmaColor": 25.0, "sigmaSpace": 25.0},
            "resize_interpolation": "INTER_AREA",
            "coarse_detector_preset": ARUCO_PRESET_LCD_AGGRESSIVE,
            "subpix_window": [13, 13],
            "markers": per_marker,
        }

    def detect_with_details(
            self, frame: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray],
               Dict[int, np.ndarray], Dict[str, Any]]:
        """Return (final, raw, mode-selected, outer-edge diagnostics).

        OpenCV ArUco occasionally returns a smaller valid quadrilateral on these
        high-resolution LCD recordings.  The optional outer-edge stage uses the
        ArUco result only as a coarse seed, searches the original grayscale image
        outward from the central 56% of each side, robustly fits four physical
        black/white edges, and intersects those lines to recover the final corners.
        """
        image = np.asarray(frame)
        is_exact_4k = image.ndim >= 2 and image.shape[0] == 2160 and image.shape[1] == 3840
        if self._four_k_halfres_seed and is_exact_4k:
            mode_result, raw_result, coarse_diag = self._detect_4k_halfres_then_fullres_subpix(frame)
            if mode_result:
                if not self._outer_edge_refine:
                    return (
                        {mid: pts.copy() for mid, pts in mode_result.items()},
                        raw_result,
                        {mid: pts.copy() for mid, pts in mode_result.items()},
                        {"enabled": False, "markers": {}, "four_k_coarse_to_fine": coarse_diag},
                    )
                edge_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8, copy=False)
                final_result, outer_diag = _outer_edge_refine_corner_dict(edge_gray, mode_result)
                outer_diag["four_k_coarse_to_fine"] = coarse_diag
                return final_result, raw_result, mode_result, outer_diag
            # If half-resolution ArUco unexpectedly finds nothing, preserve v5.9
            # behavior as a fallback rather than dropping the frame completely.

        detect_gray = self.prepare_gray(frame)
        raw_corners, raw_ids, _raw_rejected = self._detect_with_params(
            detect_gray, self._raw_detector, self._raw_parameters)
        raw_result = self._to_corner_dict(raw_corners, raw_ids)
        if not raw_result:
            return {}, {}, {}, {"enabled": self._outer_edge_refine, "markers": {}}

        mode_result = self._selected_mode_corners(detect_gray, raw_result)
        if not self._outer_edge_refine:
            return (
                {mid: pts.copy() for mid, pts in mode_result.items()},
                raw_result,
                {mid: pts.copy() for mid, pts in mode_result.items()},
                {"enabled": False, "markers": {}},
            )

        image = np.asarray(frame)
        if image.ndim == 2:
            edge_gray = image.astype(np.uint8, copy=False)
        else:
            edge_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        final_result, outer_diag = _outer_edge_refine_corner_dict(edge_gray, mode_result)
        return final_result, raw_result, mode_result, outer_diag

    def detect_with_raw(
            self, frame: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        final_result, raw_result, _mode_result, _outer_diag = self.detect_with_details(frame)
        return final_result, raw_result

    def detect(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        selected, _raw = self.detect_with_raw(frame)
        return selected

def _polygon_area_abs(points: np.ndarray) -> float:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))



def _sample_gray_bilinear(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(-1, 2)
    map_x = points[:, 0].reshape(-1, 1)
    map_y = points[:, 1].reshape(-1, 1)
    sampled = cv2.remap(
        gray, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    return sampled.reshape(-1).astype(np.float32)


def _fit_line_robust(points: Sequence[np.ndarray]):
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None, float("inf"), np.zeros(len(pts), dtype=bool)
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(3):
        if int(np.count_nonzero(keep)) < 4:
            return None, float("inf"), keep
        vx, vy, x0, y0 = cv2.fitLine(
            pts[keep], cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
        direction = np.asarray([vx, vy], np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        point = np.asarray([x0, y0], np.float64)
        normal = np.asarray([-direction[1], direction[0]], np.float64)
        distances = np.abs((pts.astype(np.float64) - point) @ normal)
        active = distances[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = max(0.75, median + 2.5 * max(mad, 0.25))
        new_keep = distances <= threshold
        if int(np.count_nonzero(new_keep)) < 4 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    vx, vy, x0, y0 = cv2.fitLine(
        pts[keep], cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
    direction = np.asarray([vx, vy], np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    point = np.asarray([x0, y0], np.float64)
    normal = np.asarray([-direction[1], direction[0]], np.float64)
    distances = np.abs((pts[keep].astype(np.float64) - point) @ normal)
    rms = float(np.sqrt(np.mean(distances * distances))) if len(distances) else float("inf")
    return (point, direction), rms, keep


def _intersect_lines(first, second):
    if first is None or second is None:
        return None
    p1, d1 = first
    p2, d2 = second
    matrix = np.column_stack((d1, -d2))
    det = float(np.linalg.det(matrix))
    if abs(det) <= 1e-8:
        return None
    try:
        scale = np.linalg.solve(matrix, p2 - p1)[0]
    except np.linalg.LinAlgError:
        return None
    return p1 + float(scale) * d1


def _quad_is_convex(points: np.ndarray) -> bool:
    pts = np.asarray(points, np.float64).reshape(4, 2)
    crosses = []
    for index in range(4):
        a = pts[(index + 1) % 4] - pts[index]
        b = pts[(index + 2) % 4] - pts[(index + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    positive = all(value > 1e-6 for value in crosses)
    negative = all(value < -1e-6 for value in crosses)
    return positive or negative


def _outer_edge_refine_marker(gray: np.ndarray, seed_corners: np.ndarray):
    seed = np.asarray(seed_corners, np.float64).reshape(4, 2)
    height, width = gray.shape[:2]
    side_lengths = np.asarray([
        np.linalg.norm(seed[(index + 1) % 4] - seed[index])
        for index in range(4)
    ], np.float64)
    nominal_side = float(np.median(side_lengths))
    diagnostics: Dict[str, Any] = {
        "accepted": False,
        "reason": "UNINITIALIZED",
        "seed_side_mean_px": float(np.mean(side_lengths)),
        "search_outward_px": None,
        "search_inward_px": None,
        "edges": [],
    }
    if not np.isfinite(nominal_side) or nominal_side < 12.0:
        diagnostics["reason"] = "SEED_TOO_SMALL"
        return seed.astype(np.float32), diagnostics

    # The bad ArUco solution in the supplied 4K video is about 7-11 px inward
    # on a ~126 px side.  Scale the search with marker size while keeping sane
    # bounds for other resolutions.
    search_outward = float(np.clip(0.16 * nominal_side, 6.0, 24.0))
    search_inward = float(np.clip(0.04 * nominal_side, 2.5, 5.0))
    diagnostics["search_outward_px"] = search_outward
    diagnostics["search_inward_px"] = search_inward

    work = cv2.GaussianBlur(np.asarray(gray, np.uint8), (3, 3), 0)
    offsets = np.arange(-search_inward, search_outward + 0.001, 0.5, dtype=np.float64)
    fractions = np.linspace(0.22, 0.78, 13, dtype=np.float64)
    center = np.mean(seed, axis=0)
    fitted_lines = []

    for edge_index in range(4):
        p0 = seed[edge_index]
        p1 = seed[(edge_index + 1) % 4]
        tangent = p1 - p0
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-9:
            diagnostics["reason"] = "DEGENERATE_SEED_EDGE"
            return seed.astype(np.float32), diagnostics
        tangent /= tangent_norm
        outward = np.asarray([-tangent[1], tangent[0]], np.float64)
        midpoint = 0.5 * (p0 + p1)
        if float(np.dot(outward, midpoint - center)) < 0.0:
            outward = -outward

        edge_points = []
        peak_records = []
        for fraction in fractions:
            base = (1.0 - fraction) * p0 + fraction * p1
            positions = base[None, :] + offsets[:, None] * outward[None, :]
            values = _sample_gray_bilinear(work, positions)
            gradient = np.gradient(values, offsets)
            if len(gradient) < 5:
                continue
            peak_index = int(np.argmax(gradient[2:-2])) + 2
            peak_strength = float(gradient[peak_index])
            before = max(0, peak_index - 4)  # ~2 px on each side
            after = min(len(values) - 1, peak_index + 4)
            contrast = float(values[after] - values[before])
            if peak_strength < 8.0 or contrast < 18.0:
                continue

            peak_offset = float(offsets[peak_index])
            # Quadratic interpolation of the gradient maximum gives a stable
            # sub-pixel offset without relying on cornerSubPix's seed basin.
            if 1 <= peak_index < len(gradient) - 1:
                y1 = float(gradient[peak_index - 1])
                y2 = float(gradient[peak_index])
                y3 = float(gradient[peak_index + 1])
                denominator = y1 - 2.0 * y2 + y3
                if abs(denominator) > 1e-9:
                    delta = 0.5 * (y1 - y3) / denominator
                    if abs(delta) <= 1.0:
                        peak_offset += 0.5 * float(delta)

            candidate = base + peak_offset * outward
            if not (-1.0 <= candidate[0] <= width and -1.0 <= candidate[1] <= height):
                continue
            edge_points.append(candidate)
            peak_records.append({
                "fraction": float(fraction),
                "offset_px": peak_offset,
                "gradient": peak_strength,
                "contrast": contrast,
            })

        fitted, rms, keep = _fit_line_robust(edge_points)
        edge_diag = {
            "edge": int(edge_index),
            "candidate_count": int(len(edge_points)),
            "inlier_count": int(np.count_nonzero(keep)) if len(keep) else 0,
            "fit_rms_px": rms,
            "median_offset_px": (
                float(np.median([item["offset_px"] for item in peak_records]))
                if peak_records else None),
            "median_gradient": (
                float(np.median([item["gradient"] for item in peak_records]))
                if peak_records else None),
        }
        diagnostics["edges"].append(edge_diag)
        if fitted is None or edge_diag["inlier_count"] < 6 or not np.isfinite(rms) or rms > 1.5:
            diagnostics["reason"] = f"EDGE_{edge_index}_FIT_FAILED"
            return seed.astype(np.float32), diagnostics
        fitted_lines.append(fitted)

    refined = []
    for corner_index in range(4):
        point = _intersect_lines(fitted_lines[(corner_index - 1) % 4], fitted_lines[corner_index])
        if point is None or not np.all(np.isfinite(point)):
            diagnostics["reason"] = "LINE_INTERSECTION_FAILED"
            return seed.astype(np.float32), diagnostics
        refined.append(point)
    refined = np.asarray(refined, np.float64).reshape(4, 2)

    if not _quad_is_convex(refined):
        diagnostics["reason"] = "NON_CONVEX_RESULT"
        return seed.astype(np.float32), diagnostics
    if np.any(refined[:, 0] < -1.0) or np.any(refined[:, 0] > width) or \
       np.any(refined[:, 1] < -1.0) or np.any(refined[:, 1] > height):
        diagnostics["reason"] = "RESULT_OUTSIDE_IMAGE"
        return seed.astype(np.float32), diagnostics

    seed_area = _polygon_area_abs(seed)
    refined_area = _polygon_area_abs(refined)
    area_ratio = refined_area / max(seed_area, 1e-9)
    refined_lengths = np.asarray([
        np.linalg.norm(refined[(index + 1) % 4] - refined[index])
        for index in range(4)
    ], np.float64)
    length_ratios = refined_lengths / np.maximum(side_lengths, 1e-9)
    corner_shifts = np.linalg.norm(refined - seed, axis=1)

    diagnostics.update({
        "area_ratio_final_over_seed": float(area_ratio),
        "final_side_mean_px": float(np.mean(refined_lengths)),
        "side_length_ratios": length_ratios.tolist(),
        "corner_shift_px": corner_shifts.tolist(),
        "mean_corner_shift_px": float(np.mean(corner_shifts)),
        "max_corner_shift_px": float(np.max(corner_shifts)),
    })

    if not (0.90 <= area_ratio <= 1.45):
        diagnostics["reason"] = "AREA_RATIO_REJECTED"
        return seed.astype(np.float32), diagnostics
    if np.any(length_ratios < 0.85) or np.any(length_ratios > 1.35):
        diagnostics["reason"] = "SIDE_RATIO_REJECTED"
        return seed.astype(np.float32), diagnostics
    if float(np.max(corner_shifts)) > max(8.0, 0.25 * nominal_side):
        diagnostics["reason"] = "CORNER_SHIFT_TOO_LARGE"
        return seed.astype(np.float32), diagnostics

    diagnostics["accepted"] = True
    diagnostics["reason"] = "OK"
    return refined.astype(np.float32), diagnostics


def _outer_edge_refine_corner_dict(
        gray: np.ndarray, selected_corners: Mapping[int, np.ndarray]
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    output: Dict[int, np.ndarray] = {}
    markers: Dict[int, Any] = {}
    for marker_id, points in selected_corners.items():
        refined, diag = _outer_edge_refine_marker(gray, points)
        output[int(marker_id)] = refined
        markers[int(marker_id)] = diag
    accepted = sum(bool(item.get("accepted")) for item in markers.values())
    return output, {
        "enabled": True,
        "accepted_markers": int(accepted),
        "total_markers": int(len(markers)),
        "markers": markers,
    }


def _corner_refinement_diagnostics(
        raw_corners: Mapping[int, np.ndarray],
        refined_corners: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    """Quantify raw detectMarkers -> cornerSubPix movement for each marker.

    ``inward_px`` is the signed projection of the refinement displacement onto
    the direction from the raw corner toward the raw marker centre.  Positive
    means cornerSubPix moved the corner inward.
    """
    common_ids = sorted(int(mid) for mid in set(raw_corners) & set(refined_corners))
    markers: Dict[int, Any] = {}
    all_magnitudes = []
    all_inward = []
    for marker_id in common_ids:
        raw = np.asarray(raw_corners[marker_id], np.float64).reshape(4, 2)
        refined = np.asarray(refined_corners[marker_id], np.float64).reshape(4, 2)
        center = np.mean(raw, axis=0)
        per_corner = []
        magnitudes = []
        inward_values = []
        tangential_values = []
        for corner_index, (p_raw, p_refined) in enumerate(zip(raw, refined)):
            delta = p_refined - p_raw
            magnitude = float(np.linalg.norm(delta))
            inward_axis = center - p_raw
            axis_norm = float(np.linalg.norm(inward_axis))
            if axis_norm > 1e-12:
                inward_unit = inward_axis / axis_norm
                tangent_unit = np.asarray([-inward_unit[1], inward_unit[0]], np.float64)
                inward_px = float(np.dot(delta, inward_unit))
                tangential_px = float(np.dot(delta, tangent_unit))
            else:
                inward_px = 0.0
                tangential_px = 0.0
            per_corner.append({
                'corner': int(corner_index),
                'raw_xy': p_raw.copy(),
                'refined_xy': p_refined.copy(),
                'delta_xy': delta.copy(),
                'magnitude_px': magnitude,
                'inward_px': inward_px,
                'tangential_px': tangential_px,
            })
            magnitudes.append(magnitude)
            inward_values.append(inward_px)
            tangential_values.append(tangential_px)
            all_magnitudes.append(magnitude)
            all_inward.append(inward_px)

        raw_edges = np.linalg.norm(np.roll(raw, -1, axis=0) - raw, axis=1)
        refined_edges = np.linalg.norm(np.roll(refined, -1, axis=0) - refined, axis=1)
        raw_area = _polygon_area_abs(raw)
        refined_area = _polygon_area_abs(refined)
        raw_perimeter = float(np.sum(raw_edges))
        refined_perimeter = float(np.sum(refined_edges))
        markers[int(marker_id)] = {
            'raw_center_xy': center.copy(),
            'per_corner': per_corner,
            'mean_shift_px': float(np.mean(magnitudes)),
            'max_shift_px': float(np.max(magnitudes)),
            'mean_inward_px': float(np.mean(inward_values)),
            'min_inward_px': float(np.min(inward_values)),
            'max_inward_px': float(np.max(inward_values)),
            'all_four_inward': bool(all(value > 0.0 for value in inward_values)),
            'mean_abs_tangential_px': float(np.mean(np.abs(tangential_values))),
            'raw_edge_lengths_px': raw_edges.copy(),
            'refined_edge_lengths_px': refined_edges.copy(),
            'raw_area_px2': raw_area,
            'refined_area_px2': refined_area,
            'area_ratio_refined_over_raw': (
                None if raw_area <= 1e-12 else float(refined_area / raw_area)),
            'raw_perimeter_px': raw_perimeter,
            'refined_perimeter_px': refined_perimeter,
            'perimeter_ratio_refined_over_raw': (
                None if raw_perimeter <= 1e-12 else float(refined_perimeter / raw_perimeter)),
        }

    return {
        'marker_ids': common_ids,
        'markers': markers,
        'overall_mean_shift_px': (None if not all_magnitudes else float(np.mean(all_magnitudes))),
        'overall_max_shift_px': (None if not all_magnitudes else float(np.max(all_magnitudes))),
        'overall_mean_inward_px': (None if not all_inward else float(np.mean(all_inward))),
        'all_corners_inward': bool(all_inward and all(value > 0.0 for value in all_inward)),
    }


def _compare_ippe_raw_vs_refined(
        raw_diag: Mapping[str, Any], refined_diag: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare same-index IPPE branches computed from raw/refined corners."""
    raw_by_branch = {int(v['branch']): v for v in (raw_diag.get('branches') or [])}
    refined_by_branch = {int(v['branch']): v for v in (refined_diag.get('branches') or [])}
    comparisons = []
    for branch in sorted(set(raw_by_branch) & set(refined_by_branch)):
        raw = raw_by_branch[branch]
        refined = refined_by_branch[branch]
        R_raw = np.asarray(raw['R_reference'], np.float64).reshape(3, 3)
        R_refined = np.asarray(refined['R_reference'], np.float64).reshape(3, 3)
        t_raw = np.asarray(raw['t_reference'], np.float64).reshape(3)
        t_refined = np.asarray(refined['t_reference'], np.float64).reshape(3)
        comparisons.append({
            'branch': int(branch),
            'rotation_raw_to_refined_deg': _rotation_error_deg(R_refined, R_raw),
            'translation_raw_to_refined_mm': float(np.linalg.norm(t_refined - t_raw)),
            'raw_reprojection_rms_px': _float_or_none(raw.get('raw_marker_reprojection_rms_px')),
            'refined_reprojection_rms_px': _float_or_none(refined.get('raw_marker_reprojection_rms_px')),
            'raw_plane_normal_camera': np.asarray(raw['plane_normal_camera'], np.float64).reshape(3),
            'refined_plane_normal_camera': np.asarray(refined['plane_normal_camera'], np.float64).reshape(3),
        })
    return {
        'status': 'OK' if comparisons else 'UNAVAILABLE',
        'comparisons': comparisons,
    }


def _frame_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(_frame_gray(frame), cv2.CV_64F).var())


def _marker_coverage(corners: Mapping[int, np.ndarray], width: int, height: int) -> float:
    if not corners:
        return 0.0
    points = np.vstack([
        np.asarray(value, np.float32).reshape(-1, 2) for value in corners.values()
    ])
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points)
    return float(np.clip(cv2.contourArea(hull) / max(float(width * height), 1.0), 0.0, 1.0))


def _build_segment_path(
    infos: Sequence[Dict[str, Any]],
    marker_map: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    marker_map_diagnostics: Mapping[Any, Any],
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray],
    marker_size_mm: float,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], Dict[int, list]]:
    frame_candidates: Dict[int, list] = {}
    for item in infos:
        candidates = _base._build_temporal_frame_candidates(
            item,
            marker_map,
            camera_matrix,
            distortion,
            marker_size_mm,
            marker_map_diagnostics=marker_map_diagnostics,
        )
        if candidates:
            frame_candidates[int(item["idx"])] = candidates
    selected, diagnostics = _base._select_temporal_pose_path(
        frame_candidates,
        nominal_probe_indices=[int(item["idx"]) for item in infos],
    )
    return selected, diagnostics, frame_candidates


def _pose_path_stability(path: Mapping[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise within-segment pose spread without crossing the A/B cut."""
    items = [(int(index), candidate) for index, candidate in sorted(path.items())]
    if not items:
        return {
            "count": 0,
            "frame_indices": [],
            "rotation_deviation_median_deg": None,
            "rotation_deviation_p95_deg": None,
            "camera_center_deviation_median_mm": None,
            "camera_center_deviation_p95_mm": None,
        }
    mean_pose = _base._temporal_robust_pose_mean(
        [{
            "R": np.asarray(candidate["R"], np.float64),
            "t": np.asarray(candidate["t"], np.float64).reshape(3, 1),
            "emission": float(candidate.get("emission_cost", 0.0)),
        } for _index, candidate in items])
    mean_R = np.asarray(mean_pose["R"], np.float64).reshape(3, 3)
    centers = np.asarray([
        np.asarray(candidate.get(
            "camera_center",
            _base._temporal_camera_center(candidate["R"], candidate["t"])),
            np.float64).reshape(3)
        for _index, candidate in items
    ])
    robust_center = np.median(centers, axis=0)
    rotation_deviation = np.asarray([
        _base._temporal_rotation_distance_deg(candidate["R"], mean_R)
        for _index, candidate in items
    ], np.float64)
    center_deviation = np.linalg.norm(centers - robust_center, axis=1)
    return {
        "count": int(len(items)),
        "frame_indices": [index for index, _candidate in items],
        "rotation_deviation_median_deg": float(np.median(rotation_deviation)),
        "rotation_deviation_p95_deg": float(np.percentile(rotation_deviation, 95)),
        "rotation_deviation_max_deg": float(np.max(rotation_deviation)),
        "camera_center_deviation_median_mm": float(np.median(center_deviation)),
        "camera_center_deviation_p95_mm": float(np.percentile(center_deviation, 95)),
        "camera_center_deviation_max_mm": float(np.max(center_deviation)),
    }



def _selected_ippe_branch_diagnostics(
    corners_dict: Mapping[int, np.ndarray],
    selected_candidate: Mapping[str, Any],
    marker_map: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray],
    marker_size_mm: float,
) -> Dict[str, Any]:
    """Expose raw single-marker IPPE branches for one selected endpoint.

    This is diagnostic-only: it does not alter temporal-DP selection or refinement.
    Branch poses are converted to the same T_camera<-reference frame used by the
    selected temporal candidate so their rotations/translations are directly
    comparable.
    """
    available_ids = sorted(int(mid) for mid in set(corners_dict) & set(marker_map))
    selected_info = {
        "label": selected_candidate.get("label"),
        "source": selected_candidate.get("source"),
        "seed_marker_id": selected_candidate.get("seed_marker_id"),
        "seed_branch": selected_candidate.get("seed_branch"),
        "basin": selected_candidate.get("basin"),
        "temporal_candidate_index": selected_candidate.get("temporal_candidate_index"),
        "reprojection_rms_px": selected_candidate.get("reprojection_rms_px"),
        "emission_cost": selected_candidate.get("emission_cost"),
        "measurement_mode": selected_candidate.get("measurement_mode"),
    }
    result: Dict[str, Any] = {
        "status": "UNAVAILABLE",
        "available_marker_ids": available_ids,
        "selected": selected_info,
        "branches": [],
    }
    if not available_ids:
        result["reason"] = "No mapped marker is visible at the selected endpoint"
        return result

    marker_id = selected_candidate.get("seed_marker_id")
    try:
        marker_id = int(marker_id) if marker_id is not None else None
    except (TypeError, ValueError):
        marker_id = None
    if marker_id not in available_ids:
        if len(available_ids) == 1:
            marker_id = available_ids[0]
        else:
            result["reason"] = (
                "Selected temporal candidate is not a single-marker IPPE seed; "
                "raw branch comparison is only emitted when a unique seed marker is known"
            )
            return result

    raw_branches = _base._temporal_marker_pose_branches(
        corners_dict[marker_id], camera_matrix, distortion, marker_size_mm)
    selected_R = np.asarray(selected_candidate["R"], np.float64).reshape(3, 3)
    selected_t = np.asarray(selected_candidate["t"], np.float64).reshape(3, 1)
    branches = []
    for raw in raw_branches:
        anchor_R, anchor_t = _base._temporal_anchor_pose(raw, marker_map[marker_id])
        anchor_R = np.asarray(anchor_R, np.float64).reshape(3, 3)
        anchor_t = np.asarray(anchor_t, np.float64).reshape(3, 1)
        branches.append({
            "branch": int(raw.get("branch", len(branches))),
            "marker_id": int(marker_id),
            "raw_marker_reprojection_rms_px": _float_or_none(
                raw.get("reprojection_rms_px")),
            "R_reference": anchor_R,
            "t_reference": anchor_t,
            "plane_normal_camera": anchor_R[:, 2].copy(),
            "rotation_distance_to_selected_deg": _rotation_error_deg(anchor_R, selected_R),
            "translation_distance_to_selected_mm": float(np.linalg.norm(
                anchor_t.reshape(3) - selected_t.reshape(3))),
        })

    result.update({
        "status": "OK" if branches else "NO_VALID_IPPE_BRANCHES",
        "marker_id": int(marker_id),
        "branches": branches,
    })
    return result


def _ippe_pair_combination_diagnostics(
    diag_a: Mapping[str, Any], diag_b: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare every raw IPPE branch pair without influencing selection."""
    branches_a = list(diag_a.get("branches") or [])
    branches_b = list(diag_b.get("branches") or [])
    combinations = []
    for branch_a in branches_a:
        R_A = np.asarray(branch_a["R_reference"], np.float64).reshape(3, 3)
        t_A = np.asarray(branch_a["t_reference"], np.float64).reshape(3, 1)
        for branch_b in branches_b:
            R_B = np.asarray(branch_b["R_reference"], np.float64).reshape(3, 3)
            t_B = np.asarray(branch_b["t_reference"], np.float64).reshape(3, 1)
            R_rel, t_rel = compose_relative_pose(R_A, t_A, R_B, t_B)
            n_a = R_A[:, 2]
            n_b = R_B[:, 2]
            normal_cos = float(np.clip(
                np.dot(n_a, n_b) /
                max(float(np.linalg.norm(n_a) * np.linalg.norm(n_b)), 1e-12),
                -1.0, 1.0))
            combinations.append({
                "branch_A": int(branch_a["branch"]),
                "branch_B": int(branch_b["branch"]),
                "rotation_difference_deg": _rotation_error_deg(R_rel),
                "plane_normal_difference_deg": float(np.degrees(np.arccos(normal_cos))),
                "pattern_translation_baseline_mm": float(np.linalg.norm(
                    t_B.reshape(3) - t_A.reshape(3))),
                "relative_rt_baseline_mm": float(np.linalg.norm(t_rel)),
                "R_rel": R_rel,
                "t_rel": t_rel,
                "sum_raw_reprojection_rms_px": float(
                    (branch_a.get("raw_marker_reprojection_rms_px") or 0.0)
                    + (branch_b.get("raw_marker_reprojection_rms_px") or 0.0)),
            })
    selected_a = diag_a.get("selected", {}).get("seed_branch")
    selected_b = diag_b.get("selected", {}).get("seed_branch")
    selected_combo = None
    for combo in combinations:
        if combo["branch_A"] == selected_a and combo["branch_B"] == selected_b:
            selected_combo = combo
            break
    best_rotation_combo = min(
        combinations, key=lambda item: item["rotation_difference_deg"], default=None)
    return {
        "status": "OK" if combinations else "UNAVAILABLE",
        "selected_branch_A": selected_a,
        "selected_branch_B": selected_b,
        "selected_combination": selected_combo,
        "minimum_rotation_difference_combination": best_rotation_combo,
        "combinations": combinations,
        "diagnostic_only": True,
        "selection_was_not_changed": True,
    }


def _estimate_temporal_marker_relation_relaxed(
        frame_infos, reference_id, marker_id, camera_matrix, distortion,
        marker_size_mm):
    """Validation-only copy of production relation fitting with 2-frame fallback.

    Production intentionally requires >=3 supporting co-visible frames.  For this
    controlled validation UI we first try the unchanged production rule; only if
    the map is still incomplete do we permit two mutually consistent frames.
    Residual gates and branch-consensus logic remain unchanged.
    """
    candidates_by_frame = {}
    for item in frame_infos:
        corners = item.get('corners', {})
        if reference_id not in corners or marker_id not in corners:
            continue
        reference_branches = _base._temporal_marker_pose_branches(
            corners[reference_id], camera_matrix, distortion, marker_size_mm)
        marker_branches = _base._temporal_marker_pose_branches(
            corners[marker_id], camera_matrix, distortion, marker_size_mm)
        combinations = []
        for reference in reference_branches:
            for marker in marker_branches:
                relation_R = reference['R'].T @ marker['R']
                relation_t = reference['R'].T @ (marker['t'] - reference['t'])
                combinations.append({
                    'R': relation_R,
                    't': relation_t,
                    'emission': (
                        reference['reprojection_rms_px']
                        + marker['reprojection_rms_px']),
                    'reference_branch': reference['branch'],
                    'marker_branch': marker['branch'],
                    'frame_index': int(item['idx']),
                })
        if combinations:
            candidates_by_frame[int(item['idx'])] = combinations

    candidate_frames = len(candidates_by_frame)
    diagnostics = {
        'status': 'NO_COVISIBLE_FRAMES',
        'candidate_frames': candidate_frames,
        'minimum_support_frames': MARKER_MAP_RELAXED_MIN_SUPPORT,
        'support_frames': 0,
        'support_frame_indices': [],
        'outlier_frame_indices': sorted(candidates_by_frame),
        'selected_branches_by_frame': {},
        'rotation_residual_median_deg': None,
        'rotation_residual_p95_deg': None,
        'translation_residual_median_mm': None,
        'translation_residual_p95_mm': None,
        'validation_relaxed_min_support': True,
    }
    if not candidates_by_frame:
        return None, diagnostics
    if candidate_frames < MARKER_MAP_RELAXED_MIN_SUPPORT:
        diagnostics['status'] = 'INSUFFICIENT_COVISIBILITY'
        return None, diagnostics

    seeds = [candidate for values in candidates_by_frame.values() for candidate in values]
    best = None
    for seed in seeds:
        accepted = []
        normalized = []
        for combinations in candidates_by_frame.values():
            ranked = []
            for candidate in combinations:
                rotation, translation = _base._temporal_relation_distance(candidate, seed)
                score = (
                    rotation / _base.TEMPORAL_RELATION_ROTATION_GATE_DEG
                    + translation / _base.TEMPORAL_RELATION_TRANSLATION_GATE_MM
                    + 0.02 * candidate['emission'])
                ranked.append((score, rotation, translation, candidate))
            score, rotation, translation, candidate = min(ranked, key=lambda entry: entry[0])
            if (rotation <= _base.TEMPORAL_RELATION_ROTATION_GATE_DEG
                    and translation <= _base.TEMPORAL_RELATION_TRANSLATION_GATE_MM):
                accepted.append(candidate)
                normalized.append(score)
        seed_score = (
            -len(accepted),
            float(np.median(normalized)) if normalized else float('inf'),
            float(seed['emission']),
            int(seed['frame_index']),
            int(seed['reference_branch']),
            int(seed['marker_branch']),
        )
        if best is None or seed_score < best[0]:
            best = (seed_score, seed, accepted)

    relation = _base._temporal_robust_pose_mean(best[2], initial=best[1])
    selected = {}
    for _ in range(6):
        accepted = []
        selected = {}
        for frame_index, combinations in candidates_by_frame.items():
            ranked = []
            for candidate in combinations:
                rotation, translation = _base._temporal_relation_distance(candidate, relation)
                score = (
                    rotation / _base.TEMPORAL_RELATION_ROTATION_GATE_DEG
                    + translation / _base.TEMPORAL_RELATION_TRANSLATION_GATE_MM
                    + 0.02 * candidate['emission'])
                ranked.append((score, rotation, translation, candidate))
            _score, rotation, translation, candidate = min(
                ranked, key=lambda entry: entry[0])
            if (rotation <= _base.TEMPORAL_RELATION_ROTATION_GATE_DEG
                    and translation <= _base.TEMPORAL_RELATION_TRANSLATION_GATE_MM):
                selected[frame_index] = candidate
                accepted.append(candidate)
        updated = _base._temporal_robust_pose_mean(accepted, initial=relation)
        if updated is None:
            break
        rotation_delta, translation_delta = _base._temporal_relation_distance(updated, relation)
        relation = updated
        if rotation_delta < 1e-5 and translation_delta < 1e-4:
            break

    support = len(selected)
    minimum_support = max(
        MARKER_MAP_RELAXED_MIN_SUPPORT,
        math.ceil(0.60 * candidate_frames))
    # With exactly 3 candidates this validation fallback intentionally accepts 2/3.
    if candidate_frames == 3:
        minimum_support = MARKER_MAP_RELAXED_MIN_SUPPORT
    rotation_residuals = []
    translation_residuals = []
    for candidate in selected.values():
        rotation, translation = _base._temporal_relation_distance(candidate, relation)
        rotation_residuals.append(rotation)
        translation_residuals.append(translation)
    diagnostics.update({
        'status': 'OK_RELAXED_2FRAME_CONSENSUS' if support >= minimum_support else 'LOW_SUPPORT_FALLBACK',
        'minimum_support_frames': int(minimum_support),
        'support_frames': support,
        'support_frame_indices': sorted(selected),
        'outlier_frame_indices': sorted(set(candidates_by_frame) - set(selected)),
        'selected_branches_by_frame': {
            int(frame_index): {
                'reference_branch': int(candidate['reference_branch']),
                'marker_branch': int(candidate['marker_branch']),
            }
            for frame_index, candidate in selected.items()
        },
        'rotation_residual_median_deg': (
            float(np.median(rotation_residuals)) if rotation_residuals else None),
        'rotation_residual_p95_deg': (
            float(np.percentile(rotation_residuals, 95)) if rotation_residuals else None),
        'translation_residual_median_mm': (
            float(np.median(translation_residuals)) if translation_residuals else None),
        'translation_residual_p95_mm': (
            float(np.percentile(translation_residuals, 95)) if translation_residuals else None),
    })
    if support < minimum_support:
        return None, diagnostics
    return (relation['R'], relation['t']), diagnostics


def _build_marker_map(
        frame_infos, marker_ids, camera_matrix, distortion, marker_size_mm,
        start_marker_ids, end_marker_ids, relaxed=False):
    if not relaxed:
        return _base._build_temporal_marker_map_graph(
            frame_infos, marker_ids, camera_matrix, distortion, marker_size_mm,
            start_marker_ids=start_marker_ids, end_marker_ids=end_marker_ids)
    with _MARKER_MAP_LOCK:
        original = _base._estimate_temporal_marker_relation
        _base._estimate_temporal_marker_relation = _estimate_temporal_marker_relation_relaxed
        try:
            return _base._build_temporal_marker_map_graph(
                frame_infos, marker_ids, camera_matrix, distortion, marker_size_mm,
                start_marker_ids=start_marker_ids, end_marker_ids=end_marker_ids)
        finally:
            _base._estimate_temporal_marker_relation = original

def _run_marker_only_endpoint_refinement(
    candidate: Dict[str, Any],
    corners_a: Mapping[int, np.ndarray],
    corners_b: Mapping[int, np.ndarray],
    marker_map: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    marker_map_diagnostics: Mapping[Any, Any],
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray],
    marker_size_mm: float,
    min_baseline_mm: float,
    max_baseline_mm: float,
) -> Optional[Dict[str, Any]]:
    with _BASELINE_GATE_LOCK:
        old_min = _base.MIN_BASELINE_MM
        old_max = _base.MAX_BASELINE_MM
        try:
            _base.MIN_BASELINE_MM = float(min_baseline_mm)
            _base.MAX_BASELINE_MM = float(max_baseline_mm)
            return _base._unified_optimize_endpoint_world_poses(
                candidate["R_A"],
                candidate["t_A"],
                candidate["R_B"],
                candidate["t_B"],
                corners_a,
                corners_b,
                marker_map,
                camera_matrix,
                distortion,
                camera_matrix,
                marker_size_mm,
                feature_points_B=None,
                feature_points_A=None,
                marker_map_diagnostics=marker_map_diagnostics,
                marker_group_weight=_base.UNIFIED_MARKER_GROUP_WEIGHT,
                max_nfev=min(int(_base.JOINT_RT_MAX_NFEV), 80),
            )
        finally:
            _base.MIN_BASELINE_MM = old_min
            _base.MAX_BASELINE_MM = old_max


def analyze_two_video_segments(
    video_a_path,
    video_b_path,
    camera_matrix,
    distortion,
    marker_size_mm=8.25,
    known_translation_mm=None,
    known_translation_vector=None,
    local_window=True,
    local_window_radius=2,
    klt_enabled=False,
    aruco_use_clahe=True,
    aruco_corner_mode=CORNER_MODE_SUBPIX_5,
    aruco_detector_preset=ARUCO_PRESET_DEFAULT,
    aruco_outer_edge_refine=False,
    aruco_four_k_halfres_seed=True,
    progress_callback=None,
    log_callback=None,
    min_baseline_mm=0,
    max_baseline_mm=220,
    *,
    frame_source_override=None,
    marker_corners_override=None,
    marker_corners_a_override=None,
    marker_corners_b_override=None,
):
    """Estimate marker-only RT between two independent video segments.

    The optional override arguments are intentionally keyword-only and exist for
    deterministic validation/tests.  They follow exactly the same marker-only
    geometry path and never enable feature extraction.
    """
    started = time.perf_counter()
    K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
    dist = None if distortion is None else np.asarray(distortion, np.float64).reshape(-1, 1)
    marker_size_mm = float(marker_size_mm)
    min_baseline_mm = float(min_baseline_mm)
    max_baseline_mm = float(max_baseline_mm)
    local_window_radius = max(0, int(local_window_radius))
    aruco_use_clahe = bool(aruco_use_clahe)
    aruco_outer_edge_refine = bool(aruco_outer_edge_refine)
    aruco_four_k_halfres_seed = bool(aruco_four_k_halfres_seed)
    aruco_corner_mode = str(aruco_corner_mode or CORNER_MODE_SUBPIX_5).strip().upper()
    aruco_detector_preset = str(aruco_detector_preset or ARUCO_PRESET_DEFAULT).strip().upper()
    if aruco_corner_mode not in CORNER_MODES:
        raise ValueError(f"Unknown aruco_corner_mode: {aruco_corner_mode}")
    if aruco_detector_preset not in ARUCO_PRESETS:
        raise ValueError(f"Unknown aruco_detector_preset: {aruco_detector_preset}")
    if marker_size_mm <= 0.0 or not np.isfinite(marker_size_mm):
        raise ValueError("marker_size_mm must be positive and finite")
    if min_baseline_mm < 0.0 or max_baseline_mm <= min_baseline_mm:
        raise ValueError("Require 0 <= min_baseline_mm < max_baseline_mm")
    if not np.all(np.isfinite(K)) or abs(float(np.linalg.det(K))) <= 1e-12:
        raise ValueError("camera_matrix must be finite and invertible")

    def emit_progress(percent: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(float(percent), str(message))

    def emit_log(message: str) -> None:
        if log_callback is not None:
            log_callback(str(message))

    # Support the more explicit per-segment aliases without making the UI use
    # test-only parameters.
    if marker_corners_a_override is not None or marker_corners_b_override is not None:
        if marker_corners_override is not None:
            raise ValueError("Use marker_corners_override or the A/B aliases, not both")
        marker_corners_override = {
            "A": marker_corners_a_override or {},
            "B": marker_corners_b_override or {},
        }

    frames = TwoSegmentVideoFrames(
        video_a_path, video_b_path, frame_source_override=frame_source_override)
    detector = None
    detected: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {"A": {}, "B": {}}
    detected_raw: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {"A": {}, "B": {}}
    detected_mode: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {"A": {}, "B": {}}
    detected_outer_diag: Dict[str, Dict[int, Dict[str, Any]]] = {"A": {}, "B": {}}
    override_by_segment = {
        "A": _segment_corner_override(marker_corners_override, "A"),
        "B": _segment_corner_override(marker_corners_override, "B"),
    }

    def detect(segment: str, local_index: int) -> Dict[int, np.ndarray]:
        segment = _normalise_segment_name(segment)
        local_index = int(local_index)
        if local_index in detected[segment]:
            return detected[segment][local_index]
        override = override_by_segment[segment]
        if override is not None:
            if isinstance(override, Mapping):
                value = override.get(local_index, {})
            else:
                value = override[local_index] if 0 <= local_index < len(override) else {}
            corners = _normalise_corner_dict(value)
            raw_corners = {mid: pts.copy() for mid, pts in corners.items()}
        else:
            corners, raw_corners, mode_corners, outer_diag = detector.detect_with_details(
                frames.get_segment_frame(segment, local_index))
            detected_mode[segment][local_index] = mode_corners
            detected_outer_diag[segment][local_index] = outer_diag
        detected[segment][local_index] = corners
        detected_raw[segment][local_index] = raw_corners
        if local_index not in detected_mode[segment]:
            detected_mode[segment][local_index] = {mid: pts.copy() for mid, pts in corners.items()}
        if local_index not in detected_outer_diag[segment]:
            detected_outer_diag[segment][local_index] = {"enabled": False, "markers": {}}
        return corners

    try:
        emit_progress(2, "Opening the two independent video segments")
        detector = _MarkerDetector(
            use_clahe=aruco_use_clahe,
            corner_mode=aruco_corner_mode,
            detector_preset=aruco_detector_preset,
            outer_edge_refine=aruco_outer_edge_refine,
            four_k_halfres_seed=aruco_four_k_halfres_seed)
        emit_log(
            f"ArUco preprocessing: CLAHE={'ON' if aruco_use_clahe else 'OFF'} | "
            f"corner_mode={aruco_corner_mode} | preset={aruco_detector_preset} | "
            f"outer_edge={'ON' if aruco_outer_edge_refine else 'OFF'} | "
            f"4k_halfres_seed={'ON' if aruco_four_k_halfres_seed else 'OFF'}")
        emit_log(f"ArUco parameters: {detector.parameters_dict}")
        counts = frames.segment_lengths
        shape_a = frames.get_segment_frame("A", 0).shape[:2]
        shape_b = frames.get_segment_frame("B", 0).shape[:2]
        if shape_a != shape_b:
            raise ValueError(
                "Video A and B must have the same resolution for one camera calibration; "
                f"got A={shape_a[::-1]} and B={shape_b[::-1]}")
        core_indices = {
            "A": _sample_fraction_indices(counts["A"], CORE_A_FRACTIONS),
            "B": _sample_fraction_indices(counts["B"], CORE_B_FRACTIONS),
        }
        frames.preload_segment("A", core_indices["A"])
        frames.preload_segment("B", core_indices["B"])
        emit_progress(12, "Detecting core ArUco probes (SIFT disabled)")

        core_infos: Dict[str, list] = {"A": [], "B": []}
        core_probe_diagnostics: Dict[str, list] = {"A": [], "B": []}
        for segment in ("A", "B"):
            used_indices = set()
            for nominal_index in core_indices[segment]:
                tried = []
                selected_index = None
                selected_corners = None
                search_indices = _core_probe_search_indices(
                    nominal_index, counts[segment], CORE_PROBE_FALLBACK_RADIUS)
                for local_index in search_indices:
                    tried.append(int(local_index))
                    corners = detect(segment, local_index)
                    if corners:
                        selected_index = int(local_index)
                        selected_corners = corners
                        break
                core_probe_diagnostics[segment].append({
                    "nominal_index": int(nominal_index),
                    "tried_indices": tried,
                    "selected_index": selected_index,
                    "success": bool(selected_corners),
                })
                if selected_corners is not None:
                    if selected_index not in used_indices:
                        core_infos[segment].append({
                            "idx": selected_index,
                            "corners": selected_corners,
                            "nominal_idx": int(nominal_index),
                        })
                        used_indices.add(selected_index)
                    if selected_index == int(nominal_index):
                        emit_log(
                            f"Core probe {segment} nominal={nominal_index}: ArUco OK")
                    else:
                        emit_log(
                            f"Core probe {segment} nominal={nominal_index}: fallback -> "
                            f"frame {selected_index} after tried={tried}")
                else:
                    emit_log(
                        f"Core probe {segment} nominal={nominal_index}: ArUco FAIL "
                        f"after tried={tried}")

        failed_segments = [segment for segment in ("A", "B") if not core_infos[segment]]
        if failed_segments:
            details = []
            for segment in failed_segments:
                nominal = [int(value) for value in core_indices[segment]]
                tried = sorted({
                    int(value)
                    for item in core_probe_diagnostics[segment]
                    for value in item["tried_indices"]
                })
                selected_ok = [
                    int(item["selected_index"])
                    for item in core_probe_diagnostics[segment]
                    if item["selected_index"] is not None
                ]
                details.append(
                    f"Video {segment}: nominal={nominal}, tried={tried}, detected={selected_ok}")
            raise RuntimeError(
                "ArUco core detection failed even after +/-"
                f"{CORE_PROBE_FALLBACK_RADIUS} frame fallback. " + " | ".join(details))

        # Build the rigid multi-marker map separately from the temporal core path.
        # First preserve the production rule.  If a co-visible marker (e.g. ID5)
        # is rejected only because the five nominal core probes provide too little
        # support, scan nearby frames for map construction and finally permit a
        # validation-only 2-frame consensus while retaining all geometric gates.
        def build_map_infos(per_segment_infos):
            return [
                {"idx": int(item["idx"]), "corners": item["corners"]}
                for item in per_segment_infos["A"]
            ] + [
                {
                    "idx": frames.segment_to_global("B", int(item["idx"])),
                    "corners": item["corners"],
                }
                for item in per_segment_infos["B"]
            ]

        map_infos_by_segment = {
            "A": [dict(item) for item in core_infos["A"]],
            "B": [dict(item) for item in core_infos["B"]],
        }

        def marker_ids_for(segment):
            values = set()
            for item in map_infos_by_segment[segment]:
                values.update(int(mid) for mid in item.get("corners", {}))
            return values

        marker_ids_a = marker_ids_for("A")
        marker_ids_b = marker_ids_for("B")
        combined_map_infos = build_map_infos(map_infos_by_segment)
        marker_map, marker_map_diagnostics, reference_id = _build_marker_map(
            combined_map_infos, marker_ids_a | marker_ids_b, K, dist, marker_size_mm,
            marker_ids_a, marker_ids_b, relaxed=False)
        marker_map_strategy = "core_production"
        initial_marker_map_ids = sorted(int(mid) for mid in marker_map)

        visible_union = marker_ids_a | marker_ids_b
        if len(visible_union) >= 2 and len(marker_map) < len(visible_union):
            emit_log(
                f"Marker-map rescue: production core map only kept {sorted(marker_map)} "
                f"from visible IDs {sorted(visible_union)}; scanning +/-"
                f"{MARKER_MAP_RESCUE_RADIUS} nearby frames")
            for segment in ("A", "B"):
                rescue_indices = set(int(idx) for idx in detected[segment])
                for item in core_infos[segment]:
                    anchor = int(item["idx"])
                    for offset in range(-MARKER_MAP_RESCUE_RADIUS, MARKER_MAP_RESCUE_RADIUS + 1):
                        idx = anchor + offset
                        if 0 <= idx < counts[segment]:
                            rescue_indices.add(int(idx))
                frames.preload_segment(segment, sorted(rescue_indices))
                by_index = {int(item["idx"]): item for item in map_infos_by_segment[segment]}
                for idx in sorted(rescue_indices):
                    corners = detect(segment, idx)
                    if corners:
                        by_index[int(idx)] = {"idx": int(idx), "corners": corners}
                map_infos_by_segment[segment] = [by_index[idx] for idx in sorted(by_index)]

            marker_ids_a = marker_ids_for("A")
            marker_ids_b = marker_ids_for("B")
            visible_union = marker_ids_a | marker_ids_b
            combined_map_infos = build_map_infos(map_infos_by_segment)
            rescued_map, rescued_diag, rescued_ref = _build_marker_map(
                combined_map_infos, visible_union, K, dist, marker_size_mm,
                marker_ids_a, marker_ids_b, relaxed=False)
            if len(rescued_map) >= len(marker_map):
                marker_map, marker_map_diagnostics, reference_id = (
                    rescued_map, rescued_diag, rescued_ref)
                marker_map_strategy = "neighbor_rescue_production"

            if len(marker_map) < len(visible_union):
                relaxed_map, relaxed_diag, relaxed_ref = _build_marker_map(
                    combined_map_infos, visible_union, K, dist, marker_size_mm,
                    marker_ids_a, marker_ids_b, relaxed=True)
                if len(relaxed_map) > len(marker_map):
                    marker_map, marker_map_diagnostics, reference_id = (
                        relaxed_map, relaxed_diag, relaxed_ref)
                    marker_map_strategy = "neighbor_rescue_relaxed_2frame"

        if not marker_map or reference_id is None:
            raise RuntimeError(
                "Unable to build one rigid marker map connecting video A and video B")
        marker_map_diagnostics = dict(marker_map_diagnostics)
        marker_map_diagnostics["_validation"] = {
            "strategy": marker_map_strategy,
            "initial_marker_map_ids": initial_marker_map_ids,
            "final_marker_map_ids": sorted(int(mid) for mid in marker_map),
            "map_frame_count_A": len(map_infos_by_segment["A"]),
            "map_frame_count_B": len(map_infos_by_segment["B"]),
            "rescue_radius": int(MARKER_MAP_RESCUE_RADIUS),
            "relaxed_min_support_frames": int(MARKER_MAP_RELAXED_MIN_SUPPORT),
        }
        emit_log(
            f"Marker-only map: strategy={marker_map_strategy}, "
            f"reference ID {reference_id}, IDs={sorted(marker_map)}")
        emit_progress(30, "Solving one temporal DP path per video segment")

        path_a, temporal_a, candidates_a = _build_segment_path(
            core_infos["A"], marker_map, marker_map_diagnostics,
            K, dist, marker_size_mm)
        path_b, temporal_b, candidates_b = _build_segment_path(
            core_infos["B"], marker_map, marker_map_diagnostics,
            K, dist, marker_size_mm)
        if not path_a or not path_b:
            raise RuntimeError("No valid marker pose path in one or both video segments")

        all_info_by_segment = {
            "A": {int(item["idx"]): item for item in core_infos["A"]},
            "B": {int(item["idx"]): item for item in core_infos["B"]},
        }
        sharpness_cache: Dict[Tuple[str, int], float] = {}

        def frame_sharpness(segment: str, index: int) -> float:
            key = (segment, int(index))
            if key not in sharpness_cache:
                sharpness_cache[key] = _sharpness(frames.get_segment_frame(*key))
            return sharpness_cache[key]

        def make_pairs(
            segment_path_a: Mapping[int, Dict[str, Any]],
            segment_path_b: Mapping[int, Dict[str, Any]],
        ) -> list[Dict[str, Any]]:
            if not segment_path_a or not segment_path_b:
                return []
            indices_a = sorted(segment_path_a)
            indices_b = sorted(segment_path_b)
            max_sharp_a = max([frame_sharpness("A", i) for i in indices_a] + [1.0])
            max_sharp_b = max([frame_sharpness("B", i) for i in indices_b] + [1.0])
            first_frame = frames.get_segment_frame("A", indices_a[0])
            height, width = first_frame.shape[:2]
            output = []
            for index_a in indices_a:
                pose_a = segment_path_a[index_a]
                corners_a = all_info_by_segment["A"][index_a]["corners"]
                for index_b in indices_b:
                    pose_b = segment_path_b[index_b]
                    corners_b = all_info_by_segment["B"][index_b]["corners"]
                    R_rel, t_rel = compose_relative_pose(
                        pose_a["R"], pose_a["t"], pose_b["R"], pose_b["t"])
                    baseline = float(np.linalg.norm(t_rel))
                    if not (min_baseline_mm <= baseline <= max_baseline_mm):
                        continue
                    marker_stats = _base._unified_pair_reprojection_stats(
                        pose_a["R"], pose_a["t"], corners_a,
                        pose_b["R"], pose_b["t"], corners_b,
                        marker_map, K, dist, marker_size_mm)
                    if marker_stats is None:
                        continue
                    obs_a = _base._pattern_guided_marker_observability(
                        pose_a, corners_a, marker_map, marker_size_mm,
                        marker_map_diagnostics=marker_map_diagnostics)
                    obs_b = _base._pattern_guided_marker_observability(
                        pose_b, corners_b, marker_map, marker_size_mm,
                        marker_map_diagnostics=marker_map_diagnostics)
                    ranges = [
                        float(value) for value in (obs_a.get("range_mm"), obs_b.get("range_mm"))
                        if value is not None and np.isfinite(value)
                    ]
                    nominal_depth = float(np.median(ranges)) if ranges else 200.0
                    geometry = _base._unified_nominal_pair_geometry(
                        R_rel, t_rel, K, width, height,
                        nominal_depth_mm=nominal_depth)

                    blur_penalty = 0.5 * (
                        1.0 - min(frame_sharpness("A", index_a) / max_sharp_a, 1.0)
                        + 1.0 - min(frame_sharpness("B", index_b) / max_sharp_b, 1.0))
                    coverage = 0.5 * (
                        _marker_coverage(corners_a, width, height)
                        + _marker_coverage(corners_b, width, height))
                    temporal_penalty = 0.5 * (
                        float(pose_a.get("emission_cost", 0.0))
                        + float(pose_b.get("emission_cost", 0.0)))
                    depth_sigma = float(geometry.get("predicted_depth_sigma_mm", float("inf")))
                    depth_penalty = 4.0 if not np.isfinite(depth_sigma) else min(depth_sigma / 5.0, 4.0)
                    angle_p10 = float(geometry.get("triangulation_p10_deg", 0.0))
                    angle_penalty = max(0.0, 3.0 - angle_p10) / 3.0
                    overlap_penalty = 1.0 - float(np.clip(geometry.get("overlap_ratio", 0.0), 0.0, 1.0))
                    confidence_penalty = 1.0 - 0.5 * (
                        float(pose_a.get("measurement_confidence", 0.75))
                        + float(pose_b.get("measurement_confidence", 0.75)))

                    # Intentionally contains no known-baseline or ideal-baseline
                    # term.  Ground truth cannot leak into endpoint selection.
                    score = (
                        1.00 * float(marker_stats["rms_px"])
                        + 0.08 * min(float(marker_stats["max_px"]) / 3.0, 3.0)
                        + 0.10 * temporal_penalty
                        + 0.12 * blur_penalty
                        + 0.10 * (1.0 - min(coverage / 0.01, 1.0))
                        + 0.12 * depth_penalty
                        + 0.12 * angle_penalty
                        + 0.15 * overlap_penalty
                        + 0.08 * confidence_penalty)
                    output.append({
                        "score": float(score),
                        "index_A": int(index_a),
                        "index_B": int(index_b),
                        "R_A": np.asarray(pose_a["R"], np.float64),
                        "t_A": np.asarray(pose_a["t"], np.float64).reshape(3, 1),
                        "R_B": np.asarray(pose_b["R"], np.float64),
                        "t_B": np.asarray(pose_b["t"], np.float64).reshape(3, 1),
                        "R_rel": R_rel,
                        "t_rel": t_rel,
                        "baseline_mm": baseline,
                        "marker_stats": marker_stats,
                        "geometry": geometry,
                        "observability_A": obs_a,
                        "observability_B": obs_b,
                        "score_terms": {
                            "marker_rms_px": float(marker_stats["rms_px"]),
                            "temporal_penalty": float(temporal_penalty),
                            "blur_penalty": float(blur_penalty),
                            "marker_coverage": float(coverage),
                            "depth_penalty": float(depth_penalty),
                            "angle_penalty": float(angle_penalty),
                            "overlap_penalty": float(overlap_penalty),
                            "confidence_penalty": float(confidence_penalty),
                        },
                    })
            return sorted(output, key=lambda item: (
                item["score"], item["marker_stats"]["rms_px"],
                item["index_A"], item["index_B"]))

        provisional_pairs = make_pairs(path_a, path_b)
        if not provisional_pairs:
            raise RuntimeError(
                f"No core frame pair passed baseline gate {min_baseline_mm:g}..{max_baseline_mm:g} mm")
        provisional = provisional_pairs[0]
        emit_log(
            f"Core pair A[{provisional['index_A']}] / B[{provisional['index_B']}], "
            f"baseline={provisional['baseline_mm']:.3f} mm")

        local_diagnostics: Dict[str, Any] = {
            "enabled": bool(local_window),
            "radius": int(local_window_radius),
            "klt_enabled": bool(klt_enabled),
            "provisional_pair": {
                "A": int(provisional["index_A"]),
                "B": int(provisional["index_B"]),
            },
            "window_indices": {"A": [], "B": []},
            "klt": {"A": {}, "B": {}},
            "status": "DISABLED",
        }

        final_path_a, final_path_b = path_a, path_b
        if local_window:
            emit_progress(52, "Evaluating marker-only local windows")
            windows = {
                "A": _base._local_window_indices(
                    provisional["index_A"], range(counts["A"]), counts["A"],
                    radius=local_window_radius, stride=1),
                "B": _base._local_window_indices(
                    provisional["index_B"], range(counts["B"]), counts["B"],
                    radius=local_window_radius, stride=1),
            }
            local_diagnostics["window_indices"] = {
                segment: [int(value) for value in indices]
                for segment, indices in windows.items()
            }
            local_paths = {}
            for segment, anchor, base_path in (
                ("A", provisional["index_A"], path_a),
                ("B", provisional["index_B"], path_b),
            ):
                frames.preload_segment(segment, windows[segment])
                infos = []
                for index in windows[segment]:
                    corners = detect(segment, index)
                    if not corners:
                        continue
                    item = {"idx": int(index), "corners": corners}
                    all_info_by_segment[segment][int(index)] = item
                    infos.append(item)
                selected, temporal_diag, candidate_map = _build_segment_path(
                    infos, marker_map, marker_map_diagnostics,
                    K, dist, marker_size_mm)
                klt_cache = {}
                if klt_enabled and candidate_map:
                    available = sorted(candidate_map)
                    for previous, current in zip(available, available[1:]):
                        if current - previous != 1:
                            continue
                        klt_cache[(previous, current)] = _base._local_klt_track_pair(
                            _frame_gray(frames.get_segment_frame(segment, previous)),
                            _frame_gray(frames.get_segment_frame(segment, current)),
                            K,
                            marker_corners=detected[segment].get(previous),
                            config={"radius": local_window_radius},
                        )
                    observations = [
                        (index, candidate_map[index]) for index in sorted(candidate_map)
                    ]
                    anchor_candidate = base_path.get(anchor)
                    if anchor_candidate is not None and observations:
                        selected, temporal_diag = _base._local_window_path(
                            observations,
                            anchor,
                            anchor_candidate,
                            {index: frame_sharpness(segment, index) for index in candidate_map},
                            klt_cache,
                            {"radius": local_window_radius},
                        )
                local_diagnostics["klt"][segment] = {
                    f"{a}->{b}": {
                        key: value for key, value in diagnostic.items()
                        if key != "R_curr_from_prev"
                    }
                    for (a, b), diagnostic in klt_cache.items()
                }
                local_diagnostics[f"temporal_{segment}"] = temporal_diag
                local_paths[segment] = selected or {anchor: base_path[anchor]}
            final_path_a = local_paths.get("A", path_a)
            final_path_b = local_paths.get("B", path_b)
            local_diagnostics["status"] = (
                "OK" if final_path_a and final_path_b else "FALLBACK_CORE")

        emit_progress(72, "Ranking frame pairs with marker-only geometry")
        pairs = make_pairs(final_path_a, final_path_b)
        if not pairs:
            pairs = provisional_pairs
            local_diagnostics["status"] = "FALLBACK_CORE_NO_LOCAL_PAIR"
        selected = pairs[0]
        corners_a = all_info_by_segment["A"][selected["index_A"]]["corners"]
        corners_b = all_info_by_segment["B"][selected["index_B"]]["corners"]
        raw_corners_a = detected_raw["A"].get(
            int(selected["index_A"]), {mid: pts.copy() for mid, pts in corners_a.items()})
        raw_corners_b = detected_raw["B"].get(
            int(selected["index_B"]), {mid: pts.copy() for mid, pts in corners_b.items()})
        mode_corners_a = detected_mode["A"].get(
            int(selected["index_A"]), {mid: pts.copy() for mid, pts in corners_a.items()})
        mode_corners_b = detected_mode["B"].get(
            int(selected["index_B"]), {mid: pts.copy() for mid, pts in corners_b.items()})
        outer_edge_diag_a = detected_outer_diag["A"].get(
            int(selected["index_A"]), {"enabled": False, "markers": {}})
        outer_edge_diag_b = detected_outer_diag["B"].get(
            int(selected["index_B"]), {"enabled": False, "markers": {}})
        corner_refine_diag_a = _corner_refinement_diagnostics(raw_corners_a, corners_a)
        corner_refine_diag_b = _corner_refinement_diagnostics(raw_corners_b, corners_b)

        # Preserve the exact selected pair before endpoint refinement.  The
        # fixed-camera validation UI uses this to determine whether an A/B
        # rotation mismatch already exists in PnP/temporal selection or is
        # introduced later by endpoint refinement.
        pre_R_A = np.asarray(selected["R_A"], np.float64).reshape(3, 3).copy()
        pre_t_A = np.asarray(selected["t_A"], np.float64).reshape(3, 1).copy()
        pre_R_B = np.asarray(selected["R_B"], np.float64).reshape(3, 3).copy()
        pre_t_B = np.asarray(selected["t_B"], np.float64).reshape(3, 1).copy()
        pre_R_rel = np.asarray(selected["R_rel"], np.float64).reshape(3, 3).copy()
        pre_t_rel = np.asarray(selected["t_rel"], np.float64).reshape(3, 1).copy()
        pre_baseline = float(selected["baseline_mm"])
        pre_pattern_delta = (pre_t_B - pre_t_A).reshape(3)
        pre_pattern_baseline = float(np.linalg.norm(pre_pattern_delta))
        pre_rotation_change_deg = _rotation_error_deg(pre_R_rel)

        # Diagnostic-only raw IPPE branch inspection at the two selected endpoints.
        # The currently selected temporal candidates are intentionally left untouched.
        selected_candidate_a = final_path_a[selected["index_A"]]
        selected_candidate_b = final_path_b[selected["index_B"]]
        ippe_diag_a = _selected_ippe_branch_diagnostics(
            corners_a, selected_candidate_a, marker_map, K, dist, marker_size_mm)
        ippe_diag_b = _selected_ippe_branch_diagnostics(
            corners_b, selected_candidate_b, marker_map, K, dist, marker_size_mm)
        ippe_pair_diag = _ippe_pair_combination_diagnostics(ippe_diag_a, ippe_diag_b)
        raw_ippe_diag_a = _selected_ippe_branch_diagnostics(
            raw_corners_a, selected_candidate_a, marker_map, K, dist, marker_size_mm)
        raw_ippe_diag_b = _selected_ippe_branch_diagnostics(
            raw_corners_b, selected_candidate_b, marker_map, K, dist, marker_size_mm)
        raw_vs_refined_ippe_a = _compare_ippe_raw_vs_refined(raw_ippe_diag_a, ippe_diag_a)
        raw_vs_refined_ippe_b = _compare_ippe_raw_vs_refined(raw_ippe_diag_b, ippe_diag_b)

        # Emit compact, copy/paste-friendly diagnostics for the selected frames.
        emit_log(
            f"[CORNER-DIAG] selected resolution={shape_a[1]}x{shape_a[0]} "
            f"A[{selected['index_A']}] B[{selected['index_B']}] "
            f"CLAHE={'ON' if aruco_use_clahe else 'OFF'} corner_mode={aruco_corner_mode} preset={aruco_detector_preset}")
        for segment_name, frame_index, outer_diag in (
                ('A', int(selected['index_A']), outer_edge_diag_a),
                ('B', int(selected['index_B']), outer_edge_diag_b)):
            if isinstance(outer_diag, dict) and outer_diag.get('enabled'):
                emit_log(
                    f"[OUTER-EDGE] {segment_name}[{frame_index}] "
                    f"accepted={outer_diag.get('accepted_markers')}/{outer_diag.get('total_markers')}")
                for marker_id, marker_diag in sorted((outer_diag.get('markers') or {}).items()):
                    emit_log(
                        f"[OUTER-EDGE] {segment_name}[{frame_index}] ID{marker_id}: "
                        f"accepted={marker_diag.get('accepted')} reason={marker_diag.get('reason')} "
                        f"seed_side={marker_diag.get('seed_side_mean_px')} "
                        f"final_side={marker_diag.get('final_side_mean_px')} "
                        f"area_ratio={marker_diag.get('area_ratio_final_over_seed')} "
                        f"mean_shift={marker_diag.get('mean_corner_shift_px')}")
        for segment_name, frame_index, diag in (
                ('A', int(selected['index_A']), corner_refine_diag_a),
                ('B', int(selected['index_B']), corner_refine_diag_b)):
            emit_log(
                f"[CORNER-DIAG] {segment_name}[{frame_index}] overall: "
                f"mean_shift={diag.get('overall_mean_shift_px')!r}px "
                f"max_shift={diag.get('overall_max_shift_px')!r}px "
                f"mean_inward={diag.get('overall_mean_inward_px')!r}px "
                f"all_inward={diag.get('all_corners_inward')}")
            for marker_id, marker_diag in sorted((diag.get('markers') or {}).items()):
                emit_log(
                    f"[CORNER-DIAG] {segment_name}[{frame_index}] ID{marker_id}: "
                    f"mean_shift={marker_diag['mean_shift_px']:.6f}px "
                    f"max_shift={marker_diag['max_shift_px']:.6f}px "
                    f"mean_inward={marker_diag['mean_inward_px']:.6f}px "
                    f"all4_inward={marker_diag['all_four_inward']} "
                    f"area_ratio={marker_diag['area_ratio_refined_over_raw']!r} "
                    f"perimeter_ratio={marker_diag['perimeter_ratio_refined_over_raw']!r}")
                for item in marker_diag['per_corner']:
                    raw_xy = np.asarray(item['raw_xy'], np.float64).reshape(2)
                    refined_xy = np.asarray(item['refined_xy'], np.float64).reshape(2)
                    delta_xy = np.asarray(item['delta_xy'], np.float64).reshape(2)
                    emit_log(
                        f"[CORNER-DIAG] {segment_name}[{frame_index}] ID{marker_id} C{item['corner']}: "
                        f"raw=({raw_xy[0]:.6f},{raw_xy[1]:.6f}) "
                        f"subpix=({refined_xy[0]:.6f},{refined_xy[1]:.6f}) "
                        f"d=({delta_xy[0]:+.6f},{delta_xy[1]:+.6f}) "
                        f"mag={item['magnitude_px']:.6f}px "
                        f"inward={item['inward_px']:+.6f}px "
                        f"tangent={item['tangential_px']:+.6f}px")
        for segment_name, comparison in (
                ('A', raw_vs_refined_ippe_a), ('B', raw_vs_refined_ippe_b)):
            for item in comparison.get('comparisons') or []:
                emit_log(
                    f"[CORNER->IPPE] {segment_name} branch {item['branch']}: "
                    f"raw_to_subpix_rotation={item['rotation_raw_to_refined_deg']:.6f}deg "
                    f"translation={item['translation_raw_to_refined_mm']:.6f}mm "
                    f"raw_rms={item['raw_reprojection_rms_px']!r}px "
                    f"subpix_rms={item['refined_reprojection_rms_px']!r}px")

        emit_progress(82, "Refining endpoints using mapped ArUco corners only")
        refined = _run_marker_only_endpoint_refinement(
            selected, corners_a, corners_b,
            marker_map, marker_map_diagnostics,
            K, dist, marker_size_mm,
            min_baseline_mm, max_baseline_mm)
        refinement_applied = refined is not None
        if refined is not None:
            R_A = refined["R_A"]
            t_A = refined["t_A"]
            R_B = refined["R_B"]
            t_B = refined["t_B"]
            R_rel = refined["R_rel"]
            t_rel = refined["t_rel"]
            baseline = float(refined["baseline"])
            final_marker_stats = refined["marker"]
        else:
            R_A, t_A = selected["R_A"], selected["t_A"]
            R_B, t_B = selected["R_B"], selected["t_B"]
            R_rel, t_rel = selected["R_rel"], selected["t_rel"]
            baseline = float(selected["baseline_mm"])
            final_marker_stats = selected["marker_stats"]

        ground_truth = compute_ground_truth_metrics(
            R_rel, t_rel,
            known_translation_mm=known_translation_mm,
            known_translation_vector=known_translation_vector)
        runtime_s = float(time.perf_counter() - started)
        emit_progress(100, "Marker-only RT validation complete")
        emit_log(
            f"Selected A[{selected['index_A']}] / B[{selected['index_B']}], "
            f"baseline={baseline:.3f} mm, SIFT calls=0, runtime={runtime_s:.3f} s")

        frame_a = np.asarray(frames.get_segment_frame("A", selected["index_A"])).copy()
        frame_b = np.asarray(frames.get_segment_frame("B", selected["index_B"])).copy()
        marker_gate_ok = bool(
            final_marker_stats is not None
            and float(final_marker_stats.get("rms_px", float("inf")))
                <= float(_base.UNIFIED_MARKER_DIRECT_RMS_MAX_PX)
            and float(final_marker_stats.get("max_px", float("inf")))
                <= float(_base.UNIFIED_MARKER_DIRECT_MAX_PX))
        finite_rt = bool(
            np.all(np.isfinite(R_rel))
            and np.all(np.isfinite(t_rel))
            and min_baseline_mm <= baseline <= max_baseline_mm)
        rt_reliable = bool(marker_gate_ok and finite_rt)
        diagnostics = {
            "feature_mode": FEATURE_MODE,
            "validation_build": VALIDATION_BUILD,
            "sift_calls": 0,
            "aruco_use_clahe": bool(aruco_use_clahe),
            "aruco_corner_mode": aruco_corner_mode,
            "aruco_detector_preset": aruco_detector_preset,
            "aruco_outer_edge_refine": bool(aruco_outer_edge_refine),
            "aruco_four_k_halfres_seed": bool(aruco_four_k_halfres_seed),
            "aruco_detector_parameters": detector.parameters_dict,
            "selected_measurement_mode_A": selected_candidate_a.get("measurement_mode"),
            "selected_measurement_mode_B": selected_candidate_b.get("measurement_mode"),
            "selected_inlier_marker_ids_A": list(selected_candidate_a.get("inlier_marker_ids") or []),
            "selected_inlier_marker_ids_B": list(selected_candidate_b.get("inlier_marker_ids") or []),
            "quality_mode": "marker_temporal_only",
            "rt_reliable": rt_reliable,
            "known_ground_truth_used_for_ranking": False,
            "relative_pose_convention": "X_A = R_rel @ X_B + t_rel",
            "source_videos": {
                "A": None if video_a_path is None else str(video_a_path),
                "B": None if video_b_path is None else str(video_b_path),
                "using_frame_source_override": bool(frame_source_override is not None),
            },
            "core_fractions": {"A": list(CORE_A_FRACTIONS), "B": list(CORE_B_FRACTIONS)},
            "segments": {
                "A": {
                    "frame_count": int(counts["A"]),
                    "core_indices": [int(value) for value in core_indices["A"]],
                    "core_probe_fallback_radius": int(CORE_PROBE_FALLBACK_RADIUS),
                    "core_probe_diagnostics": core_probe_diagnostics["A"],
                    "detected_indices": sorted(int(value) for value in detected["A"]),
                    "temporal": temporal_a,
                    "stability": _pose_path_stability(final_path_a),
                },
                "B": {
                    "frame_count": int(counts["B"]),
                    "core_indices": [int(value) for value in core_indices["B"]],
                    "core_probe_fallback_radius": int(CORE_PROBE_FALLBACK_RADIUS),
                    "core_probe_diagnostics": core_probe_diagnostics["B"],
                    "detected_indices": sorted(int(value) for value in detected["B"]),
                    "temporal": temporal_b,
                    "stability": _pose_path_stability(final_path_b),
                },
            },
            "marker_map": {
                "reference_marker_id": int(reference_id),
                "marker_ids": sorted(int(value) for value in marker_map),
                "diagnostics": marker_map_diagnostics,
            },
            "local_window": local_diagnostics,
            "pair_candidate_count": int(len(pairs)),
            "selected_pair_score": float(selected["score"]),
            "selected_pair_score_terms": selected["score_terms"],
            "selected_geometry": selected["geometry"],
            "ippe_branch_diagnostics": {
                "A": ippe_diag_a,
                "B": ippe_diag_b,
                "pair_combinations": ippe_pair_diag,
                "raw_detectMarkers_A": raw_ippe_diag_a,
                "raw_detectMarkers_B": raw_ippe_diag_b,
                "raw_vs_cornerSubPix_A": raw_vs_refined_ippe_a,
                "raw_vs_cornerSubPix_B": raw_vs_refined_ippe_b,
            },
            "corner_refinement_diagnostics": {
                "corner_mode": aruco_corner_mode,
                "outer_edge_refine": bool(aruco_outer_edge_refine),
                "four_k_halfres_seed": bool(aruco_four_k_halfres_seed),
                "subpix_window": (
                    [3, 3] if aruco_corner_mode == CORNER_MODE_SUBPIX_3 else
                    [5, 5] if aruco_corner_mode == CORNER_MODE_SUBPIX_5 else None),
                "A": corner_refine_diag_a,
                "B": corner_refine_diag_b,
            },
            "outer_edge_refinement": {
                "enabled": bool(aruco_outer_edge_refine),
                "A": outer_edge_diag_a,
                "B": outer_edge_diag_b,
            },
            "pre_refinement_pose": {
                "R_A": pre_R_A,
                "t_A": pre_t_A,
                "R_B": pre_R_B,
                "t_B": pre_t_B,
                "R_rel": pre_R_rel,
                "t_rel": pre_t_rel,
                "relative_rt_baseline_mm": pre_baseline,
                "pattern_translation_vector_camera_mm": pre_pattern_delta,
                "pattern_translation_baseline_mm": pre_pattern_baseline,
                "rotation_change_deg": pre_rotation_change_deg,
                "marker_reprojection": selected.get("marker_stats"),
            },
            "post_refinement_pose": {
                "R_A": np.asarray(R_A, np.float64).reshape(3, 3),
                "t_A": np.asarray(t_A, np.float64).reshape(3, 1),
                "R_B": np.asarray(R_B, np.float64).reshape(3, 3),
                "t_B": np.asarray(t_B, np.float64).reshape(3, 1),
                "R_rel": np.asarray(R_rel, np.float64).reshape(3, 3),
                "t_rel": np.asarray(t_rel, np.float64).reshape(3, 1),
                "relative_rt_baseline_mm": float(baseline),
                "pattern_translation_vector_camera_mm":
                    (np.asarray(t_B, np.float64).reshape(3)
                     - np.asarray(t_A, np.float64).reshape(3)),
                "pattern_translation_baseline_mm": float(np.linalg.norm(
                    np.asarray(t_B, np.float64).reshape(3)
                    - np.asarray(t_A, np.float64).reshape(3))),
                "rotation_change_deg": _rotation_error_deg(R_rel),
                "marker_reprojection": final_marker_stats,
            },
            "refinement_delta": {
                "A_rotation_shift_deg": _rotation_error_deg(
                    np.asarray(R_A, np.float64).reshape(3, 3), pre_R_A),
                "B_rotation_shift_deg": _rotation_error_deg(
                    np.asarray(R_B, np.float64).reshape(3, 3), pre_R_B),
                "A_translation_shift_mm": float(np.linalg.norm(
                    np.asarray(t_A, np.float64).reshape(3) - pre_t_A.reshape(3))),
                "B_translation_shift_mm": float(np.linalg.norm(
                    np.asarray(t_B, np.float64).reshape(3) - pre_t_B.reshape(3))),
            },
            "refinement": {
                "attempted": True,
                "applied": bool(refinement_applied),
                "role": None if refined is None else refined.get("role"),
                "marker_ok": None if refined is None else bool(refined.get("marker_ok", False)),
                "feature_ok": False,
                "applied_feature": False,
            },
            "final_marker_reprojection": final_marker_stats,
            "quality_gate": {
                "marker_rms_threshold_px": float(_base.UNIFIED_MARKER_DIRECT_RMS_MAX_PX),
                "marker_max_threshold_px": float(_base.UNIFIED_MARKER_DIRECT_MAX_PX),
                "marker_gate_ok": marker_gate_ok,
                "finite_rt_and_baseline_gate_ok": finite_rt,
            },
            "ground_truth_consistency": {
                "scalar_vector_mismatch_mm": ground_truth[
                    "known_scalar_vector_mismatch_mm"],
                "scalar_vector_consistent": ground_truth[
                    "known_scalar_vector_consistent"],
            },
        }
        return {
            "feature_mode": FEATURE_MODE,
            "sift_calls": 0,
            "aruco_use_clahe": bool(aruco_use_clahe),
            "aruco_corner_mode": aruco_corner_mode,
            "aruco_detector_preset": aruco_detector_preset,
            "aruco_detector_parameters": detector.parameters_dict,
            "selected_measurement_mode_A": selected_candidate_a.get("measurement_mode"),
            "selected_measurement_mode_B": selected_candidate_b.get("measurement_mode"),
            "selected_inlier_marker_ids_A": list(selected_candidate_a.get("inlier_marker_ids") or []),
            "selected_inlier_marker_ids_B": list(selected_candidate_b.get("inlier_marker_ids") or []),
            "quality_mode": "marker_temporal_only",
            "rt_reliable": rt_reliable,
            "selected_frame_a_index": int(selected["index_A"]),
            "selected_frame_b_index": int(selected["index_B"]),
            "selected_frame_a_global_index": frames.segment_to_global(
                "A", selected["index_A"]),
            "selected_frame_b_global_index": frames.segment_to_global(
                "B", selected["index_B"]),
            "selected_frame_a_bgr": frame_a,
            "selected_frame_b_bgr": frame_b,
            "selected_frames": {"A": frame_a, "B": frame_b},
            "corners_a": {key: value.copy() for key, value in corners_a.items()},
            "corners_b": {key: value.copy() for key, value in corners_b.items()},
            "mode_corners_a": {key: value.copy() for key, value in mode_corners_a.items()},
            "mode_corners_b": {key: value.copy() for key, value in mode_corners_b.items()},
            "raw_corners_a": {key: value.copy() for key, value in raw_corners_a.items()},
            "raw_corners_b": {key: value.copy() for key, value in raw_corners_b.items()},
            "outer_edge_refinement_a": outer_edge_diag_a,
            "outer_edge_refinement_b": outer_edge_diag_b,
            "corner_refinement_diagnostics": {
                "corner_mode": aruco_corner_mode,
                "outer_edge_refine": bool(aruco_outer_edge_refine),
                "four_k_halfres_seed": bool(aruco_four_k_halfres_seed),
                "subpix_window": (
                    [3, 3] if aruco_corner_mode == CORNER_MODE_SUBPIX_3 else
                    [5, 5] if aruco_corner_mode == CORNER_MODE_SUBPIX_5 else None),
                "A": corner_refine_diag_a,
                "B": corner_refine_diag_b,
            },
            "R_A": np.asarray(R_A, np.float64),
            "t_A": np.asarray(t_A, np.float64).reshape(3, 1),
            "R_B": np.asarray(R_B, np.float64),
            "t_B": np.asarray(t_B, np.float64).reshape(3, 1),
            "R_rel": np.asarray(R_rel, np.float64),
            "t_rel": np.asarray(t_rel, np.float64).reshape(3, 1),
            "baseline_mm": float(baseline),
            "ground_truth": ground_truth,
            "diagnostics": diagnostics,
            "runtime_s": runtime_s,
        }
    finally:
        frames.close()


__all__ = [
    "FEATURE_MODE",
    "LazyVideoSegment",
    "TwoSegmentVideoFrames",
    "compose_relative_pose",
    "compute_ground_truth_metrics",
    "analyze_two_video_segments",
]
