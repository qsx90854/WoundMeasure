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


class _MarkerDetector:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco module is unavailable")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self._dictionary = dictionary
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters())
            self._parameters = None
        else:  # OpenCV 4.6 compatibility
            self._detector = None
            self._parameters = cv2.aruco.DetectorParameters_create()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            gray = frame.astype(np.uint8, copy=False)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_gray = self._clahe.apply(gray) if _base.ARUCO_USE_CLAHE else gray
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(detect_gray)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                detect_gray, self._dictionary, parameters=self._parameters)
        if ids is None or not len(ids):
            return {}
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        result: Dict[int, np.ndarray] = {}
        for marker_id, marker_corners in zip(ids.reshape(-1), corners):
            refined = np.asarray(marker_corners, np.float32).reshape(4, 2).copy()
            try:
                cv2.cornerSubPix(
                    detect_gray, refined.reshape(-1, 1, 2), (5, 5), (-1, -1), term)
            except cv2.error:
                pass
            result[int(marker_id)] = refined
        return result


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
        else:
            corners = detector.detect(frames.get_segment_frame(segment, local_index))
        detected[segment][local_index] = corners
        return corners

    try:
        emit_progress(2, "Opening the two independent video segments")
        detector = _MarkerDetector()
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

        # Marker-map relation estimation only needs unique frame IDs.  B uses the
        # virtual offset here, but its later temporal DP returns to local indices.
        combined_map_infos = [
            {"idx": int(item["idx"]), "corners": item["corners"]}
            for item in core_infos["A"]
        ] + [
            {"idx": frames.segment_to_global("B", int(item["idx"])), "corners": item["corners"]}
            for item in core_infos["B"]
        ]
        marker_ids_a = set().union(*(set(item["corners"]) for item in core_infos["A"]))
        marker_ids_b = set().union(*(set(item["corners"]) for item in core_infos["B"]))
        marker_map, marker_map_diagnostics, reference_id = (
            _base._build_temporal_marker_map_graph(
                combined_map_infos,
                marker_ids_a | marker_ids_b,
                K,
                dist,
                marker_size_mm,
                start_marker_ids=marker_ids_a,
                end_marker_ids=marker_ids_b,
            )
        )
        if not marker_map or reference_id is None:
            raise RuntimeError(
                "Unable to build one rigid marker map connecting video A and video B")
        emit_log(
            f"Marker-only map: reference ID {reference_id}, IDs={sorted(marker_map)}")
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
            "sift_calls": 0,
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
