import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
import cv2
import numpy as np
from Algorithm.aruco_id_filter import (
    normalize_allowed_ids, filter_marker_detections, filter_marker_mapping)
from Algorithm import aruco_pose
from Algorithm import video_pose_analysis_temporal_unified_pattern_guided_local_window as temporal

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'


def main_functions(*names):
    tree = ast.parse(MAIN.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = dict(np=np, cv2=cv2, ARUCO_ALLOWED_PATTERN_IDS=[2, 5, 9, 12],
               filter_marker_mapping=filter_marker_mapping,
               filter_marker_detections=filter_marker_detections)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN), 'exec'), env)
    return env


class ArucoIdFilterTests(unittest.TestCase):
    def test_pairing_empty_and_unrestricted(self):
        corners = [np.full((1, 4, 2), mid, np.float32) for mid in [99, 2, 12, 1]]
        ids = np.array([[99], [2], [12], [1]])
        kept, values = filter_marker_detections(corners, ids, [2, 5, 9, 12])
        self.assertEqual(values.ravel().tolist(), [2, 12])
        self.assertIs(kept[0], corners[1])
        self.assertIs(kept[1], corners[2])
        self.assertEqual(filter_marker_detections(corners, ids, []), ([], None))
        self.assertIs(filter_marker_detections(corners, ids, None)[0], corners)
        self.assertEqual(filter_marker_mapping({2: 1, 99: 2}, []), {})
        with self.assertRaises(ValueError):
            normalize_allowed_ids([2.5])

    def test_global_plane_filters_before_refinement_and_pnp(self):
        image = np.zeros((100, 100), np.uint8)
        corners = [np.zeros((1, 4, 2), np.float32) for _ in range(3)]
        detector = Mock()
        detector.detectMarkers.return_value = corners, np.array([[2], [99], [5]]), []
        with patch.object(cv2.aruco, 'ArucoDetector', return_value=detector), \
             patch.object(cv2, 'cornerSubPix') as refine, \
             patch.object(cv2, 'solvePnP', return_value=(True, np.zeros((3, 1)), np.array([[0.], [0.], [100.]]))) as pnp:
            n, c = aruco_pose.compute_global_plane(image, np.eye(3), 10.,
                                                   allowed_marker_ids=[2, 5], log_fn=lambda _: None)
            self.assertIsNotNone(n)
            self.assertEqual(pnp.call_count, 2)
            self.assertEqual(refine.call_count, 2)
            pnp.reset_mock()
            refine.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(aruco_pose.compute_global_plane(image, np.eye(3), 10.,
                                                                 allowed_marker_ids=[]), (None, None))
            pnp.assert_not_called()
            refine.assert_not_called()

    def test_actual_marker_detection_applies_allowlist(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        image = np.full((140, 250, 3), 255, np.uint8)
        for mid, x in [(2, 20), (99, 145)]:
            marker = cv2.aruco.generateImageMarker(dictionary, mid, 80)
            image[30:110, x:x+80] = marker[:, :, None]
        detected = aruco_pose.detect_aruco_corners_bgr_for_pose(image, lambda gray, _: gray)
        self.assertEqual(set(detected), {2, 99})
        filtered = aruco_pose.detect_aruco_corners_bgr_for_pose(
            image, lambda gray, _: gray, allowed_marker_ids=[2, 5, 9, 12])
        self.assertEqual(set(filtered), {2})

    def test_direct_rt_and_shared_plane_do_not_use_excluded_ids(self):
        env = main_functions('get_joint_relative_pose', 'compute_shared_marker_corner_plane')
        detector = Mock()
        corners = [np.zeros((1, 4, 2), np.float32)]
        detector.detectMarkers.return_value = corners, np.array([[99]]), []
        with patch.object(cv2.aruco, 'ArucoDetector', return_value=detector), \
             patch.object(cv2, 'solvePnP') as pnp, patch.object(cv2, 'cornerSubPix') as refine:
            result = env['get_joint_relative_pose'](np.zeros((100, 100), np.uint8),
                np.zeros((100, 100), np.uint8), np.eye(3), np.eye(3), 10.)
            self.assertEqual(result, (None, False))
            pnp.assert_not_called()
            refine.assert_not_called()
        n, c, diag = env['compute_shared_marker_corner_plane'](
            {2: corners[0], 99: corners[0]}, {2: corners[0], 99: corners[0]},
            np.eye(3), np.eye(3), np.eye(3), np.zeros(3))
        self.assertIsNone(n)
        self.assertEqual(diag['shared_marker_ids'], [2])

    def test_temporal_override_cannot_bypass_filter(self):
        K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
        canon = np.array([[-20., 20, 0], [20, 20, 0], [20, -20, 0], [-20, -20, 0]])
        overrides = []
        for degrees in np.linspace(5, 45, 20):
            points, _ = cv2.projectPoints(canon, np.array([0., np.radians(degrees), 0.]),
                                         np.array([0., 0., 400.]), K, np.zeros(5))
            # Deliberately invalid excluded data proves exclusion precedes reshape/PnP.
            overrides.append({2: points.reshape(4, 2).astype(np.float32), 99: 'excluded'})
        config = dict(enabled=True, target_frame_A_deg=15., target_frame_B_deg=35.,
                      target_tolerance_deg=4., coarse_samples_per_segment=10,
                      max_scan_frames_per_segment=10, candidates_per_side=3, pair_score_weight=1.)
        with contextlib.ExitStack() as stack:
            for flag in ['SAVE_RT_SIFT_DIAGNOSTICS', 'SAVE_DEBUG_PAIR_IMAGES']:
                stack.enter_context(patch.object(temporal, flag, False))
            stack.enter_context(patch.object(temporal, 'log_and_print', lambda _: None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = temporal.analyze_video_frames('synthetic-allowlist', 10, 10,
                K, np.zeros(5), K, 40., range_mode='half_half',
                frames_override=[np.zeros((480, 640, 3), np.uint8) for _ in overrides],
                marker_corners_override=overrides, local_window=False,
                angle_guided_config=config, allowed_marker_ids=[2, 5, 9, 12])
            empty = temporal.analyze_video_frames('synthetic-allowlist-empty', 10, 10,
                K, np.zeros(5), K, 40., range_mode='half_half',
                frames_override=[np.zeros((480, 640, 3), np.uint8) for _ in overrides],
                marker_corners_override=overrides, local_window=False,
                angle_guided_config=config, allowed_marker_ids=[])
            self.assertIsNone(empty)
        self.assertIsNotNone(result)
        self.assertEqual(set(result['cornersA']), {2})
        self.assertEqual(set(result['cornersB']), {2})
        self.assertEqual(set(result['marker_map']), {2})
        self.assertEqual(result['allowed_marker_ids'], [2, 5, 9, 12])


if __name__ == '__main__':
    unittest.main()
