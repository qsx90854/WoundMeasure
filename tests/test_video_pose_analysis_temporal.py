"""Synthetic tests for the lightweight temporal marker-pose front end.

These tests deliberately avoid real videos and ArUco detection.  Marker image
corners are generated with ``cv2.projectPoints`` so branch ambiguity, marker-map
outliers, coordinate conventions, and probe selection can be tested exactly.
"""

import contextlib
import unittest
from unittest import mock

import cv2
import numpy as np

from Algorithm import video_pose_analysis_temporal as temporal


CAMERA_MATRIX = np.asarray([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DISTORTION = np.zeros(5, dtype=np.float64)
MARKER_SIZE_MM = 40.0
PROBE_INDICES = (3, 4, 5, 6, 7)


def rotation_x(degrees):
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ], dtype=np.float64)


def rotation_y(degrees):
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def rotation_z(degrees):
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotation_error_deg(estimated, expected):
    return temporal._temporal_rotation_distance_deg(estimated, expected)


def project_marker_pose(rotation_camera_from_marker, translation_camera_from_marker):
    object_points = temporal._temporal_marker_object_points(MARKER_SIZE_MM)
    rvec, _ = cv2.Rodrigues(
        np.asarray(rotation_camera_from_marker, dtype=np.float64).reshape(3, 3))
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        np.asarray(translation_camera_from_marker, dtype=np.float64).reshape(3, 1),
        CAMERA_MATRIX,
        DISTORTION,
    )
    return image_points.reshape(4, 2).astype(np.float32)


def project_mapped_marker(frame_index, marker_rotation, marker_translation):
    """Project T_reference<-marker through a simple moving camera trajectory."""
    rotation_camera_from_reference = np.eye(3, dtype=np.float64)
    translation_camera_from_reference = np.asarray(
        [-8.0 * frame_index, 0.0, 400.0], dtype=np.float64)
    rotation_camera_from_marker = (
        rotation_camera_from_reference @ np.asarray(marker_rotation, np.float64))
    translation_camera_from_marker = (
        rotation_camera_from_reference
        @ np.asarray(marker_translation, np.float64).reshape(3)
        + translation_camera_from_reference)
    return project_marker_pose(
        rotation_camera_from_marker, translation_camera_from_marker)


def marker_relation_outlier_frames(frame_indices=PROBE_INDICES):
    """Five co-visible frames with exactly one rigid-relation outlier."""
    frame_infos = []
    for frame_index in frame_indices:
        marker_two_rotation = np.eye(3, dtype=np.float64)
        marker_two_translation = np.asarray([80.0, 0.0, 0.0])
        if frame_index == 5:
            marker_two_rotation = rotation_z(10.0)
            marker_two_translation = np.asarray([100.0, -10.0, 0.0])
        frame_infos.append({
            'idx': int(frame_index),
            'corners': {
                1: project_mapped_marker(
                    frame_index, np.eye(3), np.zeros(3)),
                2: project_mapped_marker(
                    frame_index, marker_two_rotation, marker_two_translation),
            },
        })
    return frame_infos


def ippe_branch_flip_candidates():
    """Generate a deterministic frame where local IPPE ordering is wrong."""
    random = np.random.default_rng(369)
    marker_map = {
        1: (
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        ),
    }
    ground_truth = {}
    frame_candidates = {}
    for frame_index in PROBE_INDICES:
        rotation = rotation_y(1.0 + 0.15 * (frame_index - 5))
        translation = np.asarray(
            [-8.0 * frame_index, 0.0, 400.0], dtype=np.float64)
        corners = project_marker_pose(rotation, translation).astype(np.float64)
        corners += random.normal(0.0, 0.1, size=(4, 2))
        candidates = temporal._build_temporal_frame_candidates(
            {
                'idx': frame_index,
                'corners': {1: corners.astype(np.float32)},
            },
            marker_map,
            CAMERA_MATRIX,
            DISTORTION,
            MARKER_SIZE_MM,
        )
        ground_truth[frame_index] = rotation
        frame_candidates[frame_index] = candidates
    return frame_candidates, ground_truth


class TemporalMarkerPoseTests(unittest.TestCase):

    def test_robust_marker_map_rejects_one_of_five_relation_outliers(self):
        marker_map, diagnostics = temporal._build_temporal_marker_map(
            marker_relation_outlier_frames(),
            marker_ids={1, 2},
            reference_id=1,
            camera_matrix=CAMERA_MATRIX,
            distortion=DISTORTION,
            marker_size_mm=MARKER_SIZE_MM,
        )

        self.assertIn(2, marker_map)
        estimated_rotation, estimated_translation = marker_map[2]
        self.assertLess(
            rotation_error_deg(estimated_rotation, np.eye(3)), 0.5)
        self.assertLess(
            np.linalg.norm(
                estimated_translation.reshape(3)
                - np.asarray([80.0, 0.0, 0.0])),
            1.0,
        )

        marker_diagnostics = diagnostics[2]
        self.assertEqual(marker_diagnostics['status'], 'OK_ROBUST_CONSENSUS')
        self.assertEqual(marker_diagnostics['candidate_frames'], 5)
        self.assertEqual(marker_diagnostics['support_frames'], 4)
        self.assertEqual(marker_diagnostics['support_frame_indices'], [3, 4, 6, 7])
        self.assertEqual(marker_diagnostics['outlier_frame_indices'], [5])
        self.assertLessEqual(marker_diagnostics['minimum_support_frames'], 4)

    def test_marker_map_rejects_relation_without_three_covisible_frames(self):
        marker_map, diagnostics = temporal._build_temporal_marker_map(
            marker_relation_outlier_frames(frame_indices=(3, 4)),
            marker_ids={1, 2},
            reference_id=1,
            camera_matrix=CAMERA_MATRIX,
            distortion=DISTORTION,
            marker_size_mm=MARKER_SIZE_MM,
        )

        self.assertNotIn(2, marker_map)
        self.assertEqual(
            diagnostics[2]['status'], 'INSUFFICIENT_COVISIBILITY')
        self.assertEqual(diagnostics[2]['candidate_frames'], 2)
        self.assertEqual(diagnostics[2]['minimum_support_frames'], 3)

    def test_marker_map_rejects_three_inconsistent_relation_frames(self):
        translations = (
            np.asarray([80.0, 0.0, 0.0]),
            np.asarray([105.0, -15.0, 0.0]),
            np.asarray([55.0, 18.0, 5.0]),
        )
        rotations = (np.eye(3), rotation_z(20.0), rotation_x(-18.0))
        frame_infos = []
        for frame_index, relation_rotation, relation_translation in zip(
                (3, 4, 5), rotations, translations):
            frame_infos.append({
                'idx': frame_index,
                'corners': {
                    1: project_mapped_marker(
                        frame_index, np.eye(3), np.zeros(3)),
                    2: project_mapped_marker(
                        frame_index, relation_rotation, relation_translation),
                },
            })

        marker_map, diagnostics = temporal._build_temporal_marker_map(
            frame_infos,
            marker_ids={1, 2},
            reference_id=1,
            camera_matrix=CAMERA_MATRIX,
            distortion=DISTORTION,
            marker_size_mm=MARKER_SIZE_MM,
        )

        self.assertNotIn(2, marker_map)
        self.assertEqual(diagnostics[2]['status'], 'LOW_SUPPORT_FALLBACK')
        self.assertLess(diagnostics[2]['support_frames'], 3)

    def test_temporal_beam_selects_smooth_correct_ippe_branch(self):
        frame_candidates, ground_truth = ippe_branch_flip_candidates()

        # Frame 4 is deliberately ambiguous: its slightly lower reprojection
        # branch is geometrically wrong by about ten degrees.
        greedy = min(
            frame_candidates[4], key=lambda candidate: candidate['emission_cost'])
        self.assertEqual(greedy['seed_branch'], 0)
        self.assertGreater(
            rotation_error_deg(greedy['R'], ground_truth[4]), 8.0)

        selected, diagnostics = temporal._select_temporal_pose_path(
            frame_candidates)

        self.assertEqual(diagnostics['status'], 'OK_TEMPORAL_BEAM_PATH')
        self.assertEqual(diagnostics['observation_count'], 5)
        self.assertGreater(diagnostics['margin'], 0.0)
        self.assertEqual(selected[4]['seed_branch'], 1)
        self.assertEqual(diagnostics['frames'][4]['seed_branch'], 1)
        self.assertLess(
            rotation_error_deg(selected[4]['R'], ground_truth[4]), 3.0)

        selected_rotations = [selected[index]['R'] for index in PROBE_INDICES]
        maximum_neighbor_jump = max(
            rotation_error_deg(current, previous)
            for previous, current in zip(
                selected_rotations, selected_rotations[1:]))
        self.assertLess(maximum_neighbor_jump, 3.0)

    def test_camera_center_convention_drives_translation_transition(self):
        camera_center = np.asarray([25.0, -12.0, -400.0], dtype=np.float64)
        first_rotation = np.eye(3, dtype=np.float64)
        second_rotation = rotation_y(30.0)
        first_translation = (-first_rotation @ camera_center).reshape(3, 1)
        second_translation = (-second_rotation @ camera_center).reshape(3, 1)

        first_center = temporal._temporal_camera_center(
            first_rotation, first_translation)
        second_center = temporal._temporal_camera_center(
            second_rotation, second_translation)
        np.testing.assert_allclose(first_center, camera_center, atol=1e-10)
        np.testing.assert_allclose(second_center, camera_center, atol=1e-10)
        self.assertGreater(
            np.linalg.norm(second_translation - first_translation), 100.0)

        first_candidate = {
            'R': first_rotation,
            't': first_translation,
            'camera_center': first_center,
        }
        second_candidate = {
            'R': second_rotation,
            't': second_translation,
            'camera_center': second_center,
        }
        cost, details = temporal._temporal_transition_cost(
            [(2, first_candidate)], 7, second_candidate)

        self.assertAlmostEqual(details['camera_center_jump_mm'], 0.0, places=9)
        self.assertAlmostEqual(
            details['rotation_rate_deg_per_frame'], 30.0 / 5.0, places=9)
        expected_cost = temporal.TEMPORAL_ROTATION_JUMP_WEIGHT * (30.0 / 5.0)
        self.assertAlmostEqual(cost, expected_cost, places=9)

    def test_rotation_acceleration_uses_fixed_world_frame(self):
        candidates = []
        for degrees in (0.0, 10.0, 20.0):
            # Q maps camera -> world; the production pose stores R=Q.T.
            rotation_world_from_camera = rotation_y(degrees)
            rotation_camera_from_world = rotation_world_from_camera.T
            camera_center = np.asarray([degrees, 0.0, -400.0])
            candidates.append({
                'R': rotation_camera_from_world,
                't': (-rotation_camera_from_world @ camera_center).reshape(3, 1),
                'camera_center': camera_center,
            })

        _cost, details = temporal._temporal_transition_cost(
            [(0, candidates[0]), (1, candidates[1])], 2, candidates[2])

        self.assertAlmostEqual(details['rotation_acceleration'], 0.0, places=8)
        self.assertAlmostEqual(details['camera_center_acceleration'], 0.0, places=8)

    def test_anchor_conversion_recovers_common_reference_pose(self):
        anchor_rotation = rotation_y(12.0) @ rotation_x(-4.0)
        anchor_translation = np.asarray([14.0, -7.0, 410.0]).reshape(3, 1)
        marker_rotation = rotation_z(9.0)
        marker_translation = np.asarray([80.0, 5.0, -2.0]).reshape(3, 1)

        camera_from_marker = {
            'R': anchor_rotation @ marker_rotation,
            't': anchor_rotation @ marker_translation + anchor_translation,
        }
        recovered_rotation, recovered_translation = temporal._temporal_anchor_pose(
            camera_from_marker, (marker_rotation, marker_translation))

        self.assertLess(
            rotation_error_deg(recovered_rotation, anchor_rotation), 1e-6)
        np.testing.assert_allclose(
            recovered_translation, anchor_translation, atol=1e-9)
        np.testing.assert_allclose(
            temporal._temporal_camera_center(
                recovered_rotation, recovered_translation),
            temporal._temporal_camera_center(
                anchor_rotation, anchor_translation),
            atol=1e-9,
        )

    def test_neighbor_expansion_is_bounded_by_each_segment(self):
        self.assertEqual(
            temporal._expand_temporal_probe_indices(
                [3, 4], range(0, 5), radius=1),
            [2, 3, 4],
        )
        self.assertEqual(
            temporal._expand_temporal_probe_indices(
                [5, 6, 7], range(5, 10), radius=1),
            [5, 6, 7, 8],
        )
        self.assertEqual(
            temporal._expand_temporal_probe_indices(
                [7, 3, 4], range(0, 10), radius=0),
            [3, 4, 7],
        )

    def test_frames_override_wires_neighbor_probes_and_diagnostics(self):
        frame_infos = marker_relation_outlier_frames(range(10))
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        marker_corners = [item['corners'] for item in frame_infos]

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                temporal, 'ENABLE_TEMPORAL_NEIGHBOR_PROBES', True))
            stack.enter_context(mock.patch.object(
                temporal, 'SAVE_RT_SIFT_DIAGNOSTICS', False))
            stack.enter_context(mock.patch.object(
                temporal, 'SAVE_DEBUG_PAIR_IMAGES', False))
            stack.enter_context(mock.patch.object(
                temporal, 'log_and_print', lambda _message: None))
            result = temporal.analyze_video_frames(
                'synthetic-no-video-io',
                5,
                5,
                CAMERA_MATRIX,
                DISTORTION,
                CAMERA_MATRIX,
                MARKER_SIZE_MM,
                frames_override=frames,
                marker_corners_override=marker_corners,
            )

        self.assertIsNotNone(result)
        diagnostics = result['temporal_diagnostics']
        self.assertTrue(diagnostics['neighbor_probes_enabled'])
        self.assertEqual(diagnostics['core_start_indices'], [3, 4])
        self.assertEqual(diagnostics['core_end_indices'], [5, 6, 7])
        self.assertEqual(diagnostics['temporal_start_indices'], [2, 3, 4])
        self.assertEqual(diagnostics['temporal_end_indices'], [5, 6, 7, 8])
        self.assertEqual(diagnostics['path']['observation_count'], 7)

        estimated_rotation, estimated_translation = result['marker_map'][2]
        self.assertLess(
            rotation_error_deg(estimated_rotation, np.eye(3)), 0.5)
        self.assertLess(
            np.linalg.norm(
                estimated_translation.reshape(3)
                - np.asarray([80.0, 0.0, 0.0])),
            1.0,
        )
        self.assertEqual((result['idx_A'], result['idx_B']), (3, 7))
        self.assertAlmostEqual(result['baseline'], 32.0, delta=0.5)
        self.assertAlmostEqual(
            result['baseline'], float(np.linalg.norm(result['t_rel'])), places=8)
        rotation_start, translation_start = result['valid_poses'][result['idx_A']]
        rotation_end, translation_end = result['valid_poses'][result['idx_B']]
        recomposed_rotation = rotation_start @ rotation_end.T
        recomposed_translation = (
            translation_start - recomposed_rotation @ translation_end)
        np.testing.assert_allclose(
            recomposed_rotation, result['R_rel'], atol=1e-9)
        np.testing.assert_allclose(
            recomposed_translation, result['t_rel'], atol=1e-9)

    def test_per_frame_single_marker_is_ambiguous_even_with_two_shared_ids(self):
        frame_infos = marker_relation_outlier_frames(range(10))
        frame_infos[5]['corners'][2] = project_mapped_marker(
            5, np.eye(3), np.asarray([80.0, 0.0, 0.0]))
        # The segment-wide ID unions both contain {1, 2}, while two core
        # endpoints individually contain only one marker.  This used to bypass
        # IPPE branch handling because mode selection looked only at the union.
        frame_infos[3]['corners'].pop(2)
        frame_infos[7]['corners'].pop(1)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                temporal, 'ENABLE_TEMPORAL_NEIGHBOR_PROBES', False))
            stack.enter_context(mock.patch.object(
                temporal, 'SAVE_RT_SIFT_DIAGNOSTICS', False))
            stack.enter_context(mock.patch.object(
                temporal, 'SAVE_DEBUG_PAIR_IMAGES', False))
            stack.enter_context(mock.patch.object(
                temporal, 'log_and_print', lambda _message: None))
            stack.enter_context(mock.patch.object(
                temporal, 'compute_global_plane',
                lambda *_args, **_kwargs: (
                    np.asarray([0.0, 0.0, 1.0]), 400.0)))
            result = temporal.analyze_video_frames(
                'synthetic-per-frame-single-marker',
                5,
                5,
                CAMERA_MATRIX,
                DISTORTION,
                CAMERA_MATRIX,
                MARKER_SIZE_MM,
                frames_override=frames,
                marker_corners_override=[item['corners'] for item in frame_infos],
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result['temporal_diagnostics']['ambiguous_core_indices'], [3, 7])
        self.assertEqual(
            result['temporal_diagnostics']['path']['frames'][3]
            ['inlier_marker_ids'], [1])
        self.assertEqual(
            result['temporal_diagnostics']['path']['frames'][7]
            ['inlier_marker_ids'], [2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
