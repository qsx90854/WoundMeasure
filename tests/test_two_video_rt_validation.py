"""Synthetic tests for the two-video marker-only RT validation backend.

The validation utility is intentionally tested without physical videos, ArUco
detection, a display server, or SIFT.  In particular, these tests lock down the
important experiment convention: a known pattern displacement from A to B is
reported by the production pipeline as the inverse (B-to-A) relative camera
translation.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock

import cv2
import numpy as np


MODULE_NAME = (
    "Algorithm."
    "video_pose_analysis_temporal_unified_pattern_guided_local_window_validation"
)

CAMERA_MATRIX = np.asarray(
    [
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DISTORTION = np.zeros(5, dtype=np.float64)
MARKER_SIZE_MM = 40.0


def import_validation_module_fresh():
    """Import the non-UI backend while making import-time work fail loudly."""
    sys.modules.pop(MODULE_NAME, None)
    with mock.patch.object(
        cv2,
        "SIFT_create",
        side_effect=AssertionError("SIFT must not be created while importing"),
    ), mock.patch(
        "tkinter.Tk",
        side_effect=AssertionError("the validation backend must not create a UI"),
    ):
        return importlib.import_module(MODULE_NAME)


class FakeVideoCapture:
    """Small OpenCV ``VideoCapture`` double with observable lazy decoding."""

    videos = {}
    instances = []

    @classmethod
    def install_videos(cls, videos):
        cls.videos = {
            str(path): [np.asarray(frame).copy() for frame in frames]
            for path, frames in videos.items()
        }
        cls.instances = []

    def __init__(self, path, *_args):
        self.path = str(path)
        self.frames = self.videos.get(self.path)
        self.position = 0
        self.last_grabbed = None
        self.read_indices = []
        self.grab_indices = []
        self.released = False
        type(self).instances.append(self)

    def isOpened(self):
        return self.frames is not None and not self.released

    def get(self, prop):
        if self.frames is None:
            return 0.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1]) if self.frames else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0]) if self.frames else 0.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        self.last_grabbed = None
        return True

    def read(self):
        if not self.isOpened() or not 0 <= self.position < len(self.frames):
            return False, None
        index = self.position
        self.position += 1
        self.read_indices.append(index)
        return True, self.frames[index].copy()

    def grab(self):
        if not self.isOpened() or not 0 <= self.position < len(self.frames):
            return False
        index = self.position
        self.position += 1
        self.last_grabbed = index
        self.grab_indices.append(index)
        return True

    def retrieve(self):
        if self.last_grabbed is None:
            return False, None
        return True, self.frames[self.last_grabbed].copy()

    def release(self):
        self.released = True


def rotation_x(degrees):
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def project_marker(translation_camera_from_marker, rotation_camera_from_marker=None):
    half = MARKER_SIZE_MM / 2.0
    object_points = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        if rotation_camera_from_marker is None
        else np.asarray(rotation_camera_from_marker, dtype=np.float64).reshape(3, 3)
    )
    rvec, _ = cv2.Rodrigues(rotation)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        np.asarray(translation_camera_from_marker, dtype=np.float64),
        CAMERA_MATRIX,
        DISTORTION,
    )
    return image_points.reshape(4, 2).astype(np.float32)


class TwoVideoRTValidationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.validation = import_validation_module_fresh()

    def test_backend_import_does_not_construct_sift_or_ui(self):
        module = import_validation_module_fresh()
        self.assertTrue(callable(module.analyze_two_video_segments))
        self.assertTrue(callable(module.compute_ground_truth_metrics))

    def test_ground_truth_metrics_use_b_to_a_translation_convention(self):
        displacement_a_to_b = np.asarray([50.0, 0.0, 0.0])
        estimated_t_b_to_a = -displacement_a_to_b

        metrics = self.validation.compute_ground_truth_metrics(
            np.eye(3, dtype=np.float64),
            estimated_t_b_to_a,
            known_translation_mm=50.0,
            known_translation_vector=displacement_a_to_b,
        )

        self.assertAlmostEqual(metrics["estimated_baseline_mm"], 50.0, places=9)
        self.assertAlmostEqual(metrics["known_baseline_mm"], 50.0, places=9)
        self.assertAlmostEqual(metrics["baseline_absolute_error_mm"], 0.0, places=9)
        self.assertAlmostEqual(metrics["baseline_error_percent"], 0.0, places=9)
        self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0, places=9)
        self.assertAlmostEqual(
            metrics["translation_direction_error_deg"], 0.0, places=9)
        np.testing.assert_allclose(
            np.asarray(metrics["expected_t_rel"]).reshape(3),
            estimated_t_b_to_a,
            atol=1e-12,
        )

    def test_ground_truth_metrics_allow_length_only_experiment(self):
        angle = np.radians(2.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        metrics = self.validation.compute_ground_truth_metrics(
            rotation,
            np.asarray([-49.0, 0.0, 0.0]),
            known_translation_mm=50.0,
        )

        self.assertAlmostEqual(metrics["baseline_absolute_error_mm"], 1.0, places=9)
        self.assertAlmostEqual(metrics["baseline_error_percent"], 2.0, places=9)
        self.assertAlmostEqual(metrics["rotation_error_deg"], 2.0, places=7)
        self.assertIsNone(metrics["translation_direction_error_deg"])
        self.assertIsNone(metrics["expected_t_rel"])

    def test_two_segment_reader_maps_indices_and_decodes_lazily(self):
        frames_a = [
            np.full((4, 6, 3), value, dtype=np.uint8)
            for value in (10, 11)
        ]
        frames_b = [
            np.full((4, 6, 3), value, dtype=np.uint8)
            for value in (20, 21, 22)
        ]
        FakeVideoCapture.install_videos({"video-a": frames_a, "video-b": frames_b})

        with mock.patch.object(
            self.validation.cv2, "VideoCapture", FakeVideoCapture
        ):
            frames = self.validation.TwoSegmentVideoFrames("video-a", "video-b")

            self.assertEqual(len(frames), 5)
            self.assertEqual(frames.segment_lengths, {"A": 2, "B": 3})
            self.assertEqual(frames.global_to_segment(0), ("A", 0))
            self.assertEqual(frames.global_to_segment(1), ("A", 1))
            self.assertEqual(frames.global_to_segment(2), ("B", 0))
            self.assertEqual(frames.global_to_segment(-1), ("B", 2))
            self.assertEqual(frames.segment_to_global("A", 1), 1)
            self.assertEqual(frames.segment_to_global("B", 0), 2)
            self.assertEqual(frames.segment_to_global("b", -1), 4)

            # Construction reads metadata only.  No image is decoded until a
            # virtual frame is actually requested.
            self.assertFalse(any(
                instance.read_indices or instance.grab_indices
                for instance in FakeVideoCapture.instances
            ))

            first_b = frames[3]
            self.assertEqual(int(first_b[0, 0, 0]), 21)
            decoded_after_first_access = sum(
                len(instance.read_indices) + len(instance.grab_indices)
                for instance in FakeVideoCapture.instances
            )
            self.assertEqual(decoded_after_first_access, 1)

            # The second access must come from the segment cache.
            self.assertIs(frames[3], first_b)
            decoded_after_cached_access = sum(
                len(instance.read_indices) + len(instance.grab_indices)
                for instance in FakeVideoCapture.instances
            )
            self.assertEqual(decoded_after_cached_access, 1)

            self.assertEqual(int(frames[1][0, 0, 0]), 11)
            decoded_by_path = {
                path: [
                    index
                    for instance in FakeVideoCapture.instances
                    if instance.path == path
                    for index in instance.read_indices
                ]
                for path in ("video-a", "video-b")
            }
            self.assertEqual(decoded_by_path, {"video-a": [1], "video-b": [1]})
            frames.close()

        self.assertTrue(all(
            instance.released for instance in FakeVideoCapture.instances
        ))

    def test_marker_only_analysis_completes_without_constructing_sift(self):
        frame_count = 8
        blank_frames_a = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        blank_frames_b = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        marker_rotation = rotation_x(8.0)
        corners_a = project_marker(
            [0.0, 0.0, 400.0], marker_rotation)
        corners_b = project_marker(
            [50.0, 0.0, 400.0], marker_rotation)
        corner_overrides = {
            "A": [{0: corners_a.copy()} for _ in range(frame_count)],
            "B": [{0: corners_b.copy()} for _ in range(frame_count)],
        }

        with mock.patch.object(
            self.validation.cv2,
            "SIFT_create",
            side_effect=AssertionError("marker-only analysis attempted SIFT"),
        ):
            result = self.validation.analyze_two_video_segments(
                None,
                None,
                CAMERA_MATRIX,
                DISTORTION,
                marker_size_mm=MARKER_SIZE_MM,
                known_translation_mm=50.0,
                known_translation_vector=[50.0, 0.0, 0.0],
                local_window=True,
                klt_enabled=False,
                min_baseline_mm=0.0,
                max_baseline_mm=100.0,
                frame_source_override={
                    "A": blank_frames_a,
                    "B": blank_frames_b,
                },
                marker_corners_override=corner_overrides,
            )

        self.assertEqual(result["feature_mode"], "marker_only")
        self.assertEqual(result["sift_calls"], 0)
        self.assertEqual(result["diagnostics"]["feature_mode"], "marker_only")
        self.assertEqual(result["diagnostics"]["sift_calls"], 0)
        self.assertFalse(
            result["diagnostics"]["known_ground_truth_used_for_ranking"])
        self.assertTrue(result["diagnostics"]["local_window"]["enabled"])
        self.assertEqual(result["diagnostics"]["local_window"]["status"], "OK")
        self.assertAlmostEqual(result["baseline_mm"], 50.0, delta=0.25)
        self.assertAlmostEqual(
            result["ground_truth"]["baseline_error_percent"], 0.0, delta=0.5)
        self.assertAlmostEqual(
            result["ground_truth"]["rotation_error_deg"], 0.0, delta=0.1)
        self.assertAlmostEqual(
            result["ground_truth"]["translation_direction_error_deg"],
            0.0,
            delta=0.1,
        )
        np.testing.assert_allclose(
            np.asarray(result["t_rel"]).reshape(3),
            np.asarray([-50.0, 0.0, 0.0]),
            atol=0.25,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
