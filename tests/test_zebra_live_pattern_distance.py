import unittest

import cv2
import numpy as np

import depth_measure_multi_aruco_sbs_camera_v7_demo_zebra as zebra


class _FixedCornerDetector:
    def __init__(self, corners, marker_id=7):
        self._corners = np.asarray(corners, dtype=np.float32).reshape(1, 4, 2)
        self._ids = np.array([[marker_id]], dtype=np.int32)

    def detectMarkers(self, _gray):
        return [self._corners.copy()], self._ids.copy(), []


class LivePatternDistanceTests(unittest.TestCase):
    def test_tilted_marker_recovers_camera_center_distance(self):
        K = np.array([
            [1570.0, 0.0, 960.0],
            [0.0, 1570.0, 540.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        distortion = np.zeros(5, dtype=np.float64)
        marker_size = 8.25
        half = marker_size * 0.5
        object_points = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        rvec = np.array([[0.12], [-0.08], [0.02]], dtype=np.float64)
        tvec = np.array([[12.0], [-5.0], [200.0]], dtype=np.float64)
        corners, _ = cv2.projectPoints(
            object_points, rvec, tvec, K, distortion)
        detector = _FixedCornerDetector(corners, marker_id=7)

        estimates = zebra.estimate_aruco_pattern_distances(
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            K,
            distortion,
            marker_size,
            detector=detector,
            calibration_image_size=(1920, 1080),
        )

        self.assertIn(7, estimates)
        self.assertAlmostEqual(
            estimates[7]["distance_mm"], float(np.linalg.norm(tvec)), places=3)
        self.assertTrue(np.isfinite(estimates[7]["view_angle_deg"]))
        self.assertGreaterEqual(estimates[7]["view_angle_deg"], 0.0)
        self.assertLessEqual(estimates[7]["view_angle_deg"], 90.0)
        self.assertLess(estimates[7]["reprojection_rms_px"], 1e-3)


if __name__ == "__main__":
    unittest.main()
