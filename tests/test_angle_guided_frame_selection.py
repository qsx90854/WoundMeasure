import contextlib
import unittest
from unittest import mock

import cv2
import numpy as np

from Algorithm import (
    video_pose_analysis_temporal_unified_pattern_guided_local_window as angle_pose,
)


K = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)
MARKER_SIZE_MM = 40.0


def rotation_y(degrees):
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def marker_corners(angle_degrees):
    half = MARKER_SIZE_MM * 0.5
    points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    rvec = cv2.Rodrigues(rotation_y(angle_degrees))[0]
    image, _ = cv2.projectPoints(
        points, rvec, np.array([[0.0], [0.0], [400.0]]), K, DIST)
    return image.reshape(4, 2).astype(np.float32)


class AngleGuidedSelectionTests(unittest.TestCase):
    def test_single_marker_measurement_uses_preview_incidence_definition(self):
        measurement = angle_pose._angle_guided_marker_measurement(
            {2: marker_corners(15.0)}, 2, K, DIST, MARKER_SIZE_MM)
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement['incidence_deg'], 15.0, delta=0.05)

    def test_full_analysis_targets_right_15_and_left_35(self):
        angles = np.linspace(5.0, 45.0, 20)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in angles]
        corner_overrides = [{2: marker_corners(angle)} for angle in angles]
        config = {
            'enabled': True,
            'target_frame_A_deg': 15.0,
            'target_frame_B_deg': 35.0,
            'target_tolerance_deg': 4.0,
            'coarse_samples_per_segment': 10,
            'max_scan_frames_per_segment': 10,
            'candidates_per_side': 3,
            'pair_score_weight': 1.0,
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                angle_pose, 'SAVE_RT_SIFT_DIAGNOSTICS', False))
            stack.enter_context(mock.patch.object(
                angle_pose, 'SAVE_DEBUG_PAIR_IMAGES', False))
            stack.enter_context(mock.patch.object(
                angle_pose, 'log_and_print', lambda _message: None))
            result = angle_pose.analyze_video_frames(
                'synthetic-angle-guided',
                10,
                10,
                K,
                DIST,
                K,
                MARKER_SIZE_MM,
                range_mode='half_half',
                frames_override=frames,
                marker_corners_override=corner_overrides,
                local_window=False,
                angle_guided_config=config,
            )

        self.assertIsNotNone(result)
        diagnostics = result['angle_guided_diagnostics']
        self.assertEqual(diagnostics['status'], 'OK_ANGLE_GUIDED')
        self.assertEqual(diagnostics['reference_marker_id'], 2)
        pair = diagnostics['final_selected_pair']['pair_measurement']
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair['target_A_deg'], 15.0)
        self.assertAlmostEqual(pair['target_B_deg'], 35.0)
        self.assertLessEqual(pair['error_A_deg'], 4.0)
        self.assertLessEqual(pair['error_B_deg'], 4.0)
        self.assertEqual(
            result['temporal_diagnostics']['path']['observation_count'], 20)
        self.assertGreaterEqual(result['idx_A'], 0)
        self.assertLess(result['idx_A'], 10)
        self.assertGreaterEqual(result['idx_B'], 10)
        self.assertLess(result['idx_B'], 20)

    def test_auto_direction_keeps_right_15_left_35_for_reverse_video(self):
        angles = np.linspace(45.0, 5.0, 20)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in angles]
        corner_overrides = [{2: marker_corners(angle)} for angle in angles]
        config = {
            'enabled': True,
            'direction_mode': 'auto',
            'normalize_output_roles': True,
            'target_frame_A_deg': 15.0,
            'target_frame_B_deg': 35.0,
            'target_tolerance_deg': 4.0,
            'coarse_samples_per_segment': 10,
            'max_scan_frames_per_segment': 10,
            'candidates_per_side': 3,
            'pair_score_weight': 1.0,
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                angle_pose, 'SAVE_RT_SIFT_DIAGNOSTICS', False))
            stack.enter_context(mock.patch.object(
                angle_pose, 'SAVE_DEBUG_PAIR_IMAGES', False))
            stack.enter_context(mock.patch.object(
                angle_pose, 'log_and_print', lambda _message: None))
            result = angle_pose.analyze_video_frames(
                'synthetic-angle-guided-reverse',
                10,
                10,
                K,
                DIST,
                K,
                MARKER_SIZE_MM,
                range_mode='half_half',
                frames_override=frames,
                marker_corners_override=corner_overrides,
                local_window=False,
                angle_guided_config=config,
            )

        self.assertIsNotNone(result)
        diagnostics = result['angle_guided_diagnostics']
        self.assertEqual(diagnostics['status'], 'OK_ANGLE_GUIDED')
        self.assertEqual(diagnostics['direction'], 'reverse')
        self.assertTrue(diagnostics['output_role_normalized'])
        pair = diagnostics['final_selected_pair']['pair_measurement']
        self.assertAlmostEqual(pair['target_A_deg'], 15.0)
        self.assertAlmostEqual(pair['target_B_deg'], 35.0)
        self.assertLessEqual(pair['error_A_deg'], 4.0)
        self.assertLessEqual(pair['error_B_deg'], 4.0)
        # Output A/right is the later 15-degree frame; output B/left is the
        # earlier 35-degree frame, even though optimization stayed early->late.
        self.assertGreaterEqual(result['idx_A'], 10)
        self.assertLess(result['idx_B'], 10)
        np.testing.assert_allclose(
            np.linalg.inv(result['R_rel']), result['R_rel'].T, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
