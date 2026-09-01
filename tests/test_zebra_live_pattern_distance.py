import unittest

import cv2
import numpy as np

import depth_measure_multi_aruco_sbs_camera_v7_demo_zebra as zebra
import zebra_0825v2 as zebra_v2


class _FixedCornerDetector:
    def __init__(self, corners, marker_id=7):
        self._corners = np.asarray(corners, dtype=np.float32).reshape(1, 4, 2)
        self._ids = np.array([[marker_id]], dtype=np.int32)

    def detectMarkers(self, _gray):
        return [self._corners.copy()], self._ids.copy(), []


class LivePatternDistanceTests(unittest.TestCase):
    def test_rt_sift_roi_preview_obeys_each_zebra_switch(self):
        for module in (zebra, zebra_v2):
            old_enabled = module.ENABLE_RT_SIFT_ROI
            old_ratio = module.RT_SIFT_ROI_RATIO
            old_scale = module.RT_SIFT_IMAGE_SCALE
            try:
                module.ENABLE_RT_SIFT_ROI = True
                module.RT_SIFT_ROI_RATIO = (0.10, 0.10, 0.80, 0.80)
                module.RT_SIFT_IMAGE_SCALE = 0.5
                preview = np.zeros((100, 200, 3), dtype=np.uint8)
                module.draw_rt_sift_roi_preview(preview)
                self.assertTrue(np.any(preview), module.__name__)
                self.assertTrue(
                    np.any(preview[8:13, 18:23]), module.__name__)

                module.ENABLE_RT_SIFT_ROI = False
                disabled_preview = np.zeros((100, 200, 3), dtype=np.uint8)
                module.draw_rt_sift_roi_preview(disabled_preview)
                self.assertFalse(np.any(disabled_preview), module.__name__)
            finally:
                module.ENABLE_RT_SIFT_ROI = old_enabled
                module.RT_SIFT_ROI_RATIO = old_ratio
                module.RT_SIFT_IMAGE_SCALE = old_scale

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
